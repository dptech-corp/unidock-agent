//
// Created by Congcong Liu on 24-10-23.
//

#ifndef BFGS_CUH
#define BFGS_CUH

#include <cooperative_groups/reduce.h>
#include "common.cuh"
#include "maps.cuh"
#include "mymath_warp.cuh"
#include "geometry/quaternion.h"
#include "geometry/rotation.h"
#include "myutils/matrix.h"

namespace cg = cooperative_groups;

// Receptor walk in cal_e_f_warp: atoms are tested in groups of this many steps, their
// coordinates loaded one group ahead.
#ifndef UD2_WALK_DEPTH
#define UD2_WALK_DEPTH 2
#endif


// ---------------- Flat BFGS state of one pose ----------------
// A DOF vector (gradient, search direction, ...) has dim_g = dof_g + ntorsion components: 0-2 centre, 3-5 orientation
// (-torque), dof_g + i torsion i (the index order of grad_index_read / grad_index_write). The warp BFGS keeps STRIDE_G of
// them (slots below) interleaved in one array V: component i of slot k is V[i * STRIDE_G + k], so a vector is the
// pointer V + k with stride STRIDE_G, and one base register with immediate slot offsets addresses all six. Next to it
// the Hessian (upper triangle, uptri_mat_index). Their global home is the per-pose block
// aux_h.matrix = [H: dim_x (dim_x + 1) / 2 | V: STRIDE_G * dim_g] (task.cu); a launch can place V and / or H in dynamic
// shared memory behind the force accumulator instead (bfgs_state_ptrs, flags chosen per launch on the host by
// bfgs_state_flags).
#define BV_G 0      // gradient at the current pose
#define BV_G_NEW 1  // gradient at the kept line-search trial
#define BV_G_ORI 2  // gradient at the start of bfgs_warp
#define BV_P 3      // search direction
#define BV_Y 4      // g_new - g
#define BV_MHY 5    // -H y

/**
 * @brief Per-pose BFGS state pointers: H and V (STRIDE_G vectors of dim_g) in the pose's global block, or in dynamic
 * shared memory after the force accumulator (natom * 3 Reals when natom <= KERNEL_SHARED_FORCE_MAX_ATOMS, none
 * otherwise), V first, as state_flags says. Zeroes V: every component BFGS reads is written first, except the centre
 * and orientation ones under constraint docking, which cal_grad_warp leaves alone and the search expects at 0.
 */
template<class Tile>
__device__ __forceinline__ void bfgs_state_ptrs(const Tile& tile, Real* global_block,
                                                int natom, int ntorsion, int state_flags, Real** out_h, Real** out_v){
    extern __shared__ Real s_force_dyn[];
    const int dim_g = dof_g + ntorsion;
    const int dim_x = dof_x + ntorsion;
    Real* s = s_force_dyn + (natom <= KERNEL_SHARED_FORCE_MAX_ATOMS ? natom * 3 : 0);
    Real* v = global_block + dim_x * (dim_x + 1) / 2;
    if (state_flags & BFGS_STATE_V_SHARED){
        v = s;
        s += STRIDE_G * dim_g;
    }
    *out_v = v;
    *out_h = (state_flags & BFGS_STATE_H_SHARED) ? s : global_block;
    for (int i = tile.thread_rank(); i < STRIDE_G * dim_g; i += tile.num_threads()){
        v[i] = 0;
    }
    tile.sync();
}


template<class Tile>
__device__ __forceinline__ void duplicate_grad_warp(const Tile& tile,
                                                    Real* out_g_new, const Real* g_old, int dim){
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        out_g_new[i * STRIDE_G] = g_old[i * STRIDE_G];
    }
    tile.sync();
}


template<class Tile>
__device__ __forceinline__ void duplicate_pose_warp(const Tile& tile,
                                                    FlexPose* out_pose_new, const FlexPose* pose_old, int dim,
                                                    int natom){
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        if (i < 3){
            out_pose_new->center[i] = pose_old->center[i];
        }
        else if (i < dof_x){
            out_pose_new->rot_vec[i - 3] = pose_old->rot_vec[i - 3];
        }
        else{
            out_pose_new->dihedrals[i - dof_x] = pose_old->dihedrals[i - dof_x];
        }
    }
    tile.sync();

    // copy cartesian coordinates
    for (int i = tile.thread_rank(); i < natom * 3; i += tile.num_threads()){
        out_pose_new->coords[i] = pose_old->coords[i];
    }
    tile.sync();

    // set energy
    if (tile.thread_rank() == 0){
        // assign e but not coords
        out_pose_new->energy = pose_old->energy;
    }
    tile.sync();
}


/**
 * @brief Compute gradient on dihedral, center, and orientation
 */
template<class Tile>
__device__ __forceinline__ void cal_grad_warp(const Tile& tile,
                                               const Real* aux_f, const FlexTopo& flex_topo,
                                               const FlexPose* pose, Real* out_g
){
    Real tmp1[3] = {0}, tmp2[3] = {0}, tmp3[3] = {0}, tmp4[3] = {0}, tmp5[3] = {0};
    Real mat_tmp[9] = {0.};
    Real mat[9] = {0.};
    int i_at = 0;

    // each thread for a torsion. Cal gradient on dihedral
    for (int i_tor = tile.thread_rank(); i_tor < flex_topo.ntorsion; i_tor += tile.num_threads()){
        int begin = flex_topo.rotated_inds[i_tor * 2];
        int end = flex_topo.rotated_inds[i_tor * 2 + 1];
        int iat_from = flex_topo.axis_atoms[i_tor * 2];
        int iat_to = flex_topo.axis_atoms[i_tor * 2 + 1];

        tmp5[0] = tmp5[1] = tmp5[2] = 0.;

        for (int j = begin; j < end + 1; j++){
            i_at = flex_topo.rotated_atoms[j];
            // H atoms are NOT skipped: the angular HBond term (and intra pairs) put forces
            // on H atoms that must propagate through the torsion gradient (matches cal_grad_cpu).
            // compute the lever arm vector, "to" atom is the origin of this fragment
            tmp1[0] = pose->coords[i_at * 3] - pose->coords[iat_to * 3];
            tmp1[1] = pose->coords[i_at * 3 + 1] - pose->coords[iat_to * 3 + 1];
            tmp1[2] = pose->coords[i_at * 3 + 2] - pose->coords[iat_to * 3 + 2];


            // take gradient from aux_f
            tmp2[0] = aux_f[i_at * 3];
            tmp2[1] = aux_f[i_at * 3 + 1];
            tmp2[2] = aux_f[i_at * 3 + 2];

            // compute the torque on the atom for the torsion
            cross_product(tmp1, tmp2, tmp3);
            // accumulate the torque for the torsion
            tmp5[0] += tmp3[0];
            tmp5[1] += tmp3[1];
            tmp5[2] += tmp3[2];
        }

        // take the axis
        tmp3[0] = pose->coords[iat_to * 3] - pose->coords[iat_from * 3];
        tmp3[1] = pose->coords[iat_to * 3 + 1] - pose->coords[iat_from * 3 + 1];
        tmp3[2] = pose->coords[iat_to * 3 + 2] - pose->coords[iat_from * 3 + 2];

        // projection on axis as the gradient on dihedral
        out_g[(dof_g + i_tor) * STRIDE_G] = dot_product(tmp5, tmp3) / cal_norm(tmp3);
        DPrint("dihedrals_g: %f\n", out_g[(dof_g + i_tor) * STRIDE_G]);

    }
    tile.sync();

    // Derivative on center
    tmp5[0] = tmp5[1] = tmp5[2] = 0.;

    // cal inverse of this rotation (namely last orientation_g;
    Real q[4] = {0.};
    rotvec_to_quaternion(q, pose->rot_vec);
    quaternion_conjugate(q);

    for (i_at = tile.thread_rank(); i_at < flex_topo.natom; i_at += tile.num_threads()){
        // H atoms included so their forces reach the translation / rotation gradients (matches cal_grad_cpu)
        // compute the coord of atom relative to rotation center
        tmp1[0] = pose->coords[i_at * 3] - pose->center[0];
        tmp1[1] = pose->coords[i_at * 3 + 1] - pose->center[1];
        tmp1[2] = pose->coords[i_at * 3 + 2] - pose->center[2];


        // rotate tmp1 to get real rotation coord
        rotate_vec_by_quaternion(tmp1, q);
        // DPrint("Last coord %d: %f, %f, %f\n", i_at, tmp1[0], tmp1[1], tmp1[2]);


        // take grad over pos of each atom
        tmp2[0] = aux_f[i_at * 3];
        tmp2[1] = aux_f[i_at * 3 + 1];
        tmp2[2] = aux_f[i_at * 3 + 2];

        tmp4[0] += tmp2[0]; // sum the grad on position
        tmp4[1] += tmp2[1];
        tmp4[2] += tmp2[2];

        // compute (\bar f) \outer (\bar x)
        outer_product(tmp2, tmp1, mat_tmp);
        for (int j = 0; j < 9; ++j){
            mat[j] += mat_tmp[j];
        }
    }
    tile.sync();
    // gradient over position
    tmp4[0] = cg::reduce(cg_tile(tile), tmp4[0], cg::plus<Real>());
    tmp4[1] = cg::reduce(cg_tile(tile), tmp4[1], cg::plus<Real>());
    tmp4[2] = cg::reduce(cg_tile(tile), tmp4[2], cg::plus<Real>());
    // (\bar f) \outer (\bar x)
    mat[0] = cg::reduce(cg_tile(tile), mat[0], cg::plus<Real>());
    mat[1] = cg::reduce(cg_tile(tile), mat[1], cg::plus<Real>());
    mat[2] = cg::reduce(cg_tile(tile), mat[2], cg::plus<Real>());
    mat[3] = cg::reduce(cg_tile(tile), mat[3], cg::plus<Real>());
    mat[4] = cg::reduce(cg_tile(tile), mat[4], cg::plus<Real>());
    mat[5] = cg::reduce(cg_tile(tile), mat[5], cg::plus<Real>());
    mat[6] = cg::reduce(cg_tile(tile), mat[6], cg::plus<Real>());
    mat[7] = cg::reduce(cg_tile(tile), mat[7], cg::plus<Real>());
    mat[8] = cg::reduce(cg_tile(tile), mat[8], cg::plus<Real>());
    tile.sync();


    // write gradient of center
    if ((!FLAG_CONSTRAINT_DOCK) and (tile.thread_rank() == 0)){
        out_g[0] = tmp4[0];
        out_g[STRIDE_G] = tmp4[1];
        out_g[2 * STRIDE_G] = tmp4[2];
        DPrint("Total force: %f, %f, %f\n", tmp4[0], tmp4[1], tmp4[2]);
        // DPrint("mat: \n%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n",
            // mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]);

        // gradient of rotation over vector
        // DPrint("pose.orientation (rotvec in real): %f, %f, %f\n", pose->rot_vec[0], pose->rot_vec[1], pose->rot_vec[2]);
        for (int i = 0; i < 3; ++i){
            Real dR_dv[9] = {0.};
            cal_grad_of_rot_over_vec(dR_dv, pose->rot_vec, i);
            // DPrint("dR/dv_%d: \n%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n", i,
            //     dR_dv[0], dR_dv[1], dR_dv[2], dR_dv[3], dR_dv[4], dR_dv[5], dR_dv[6], dR_dv[7], dR_dv[8]);

            // Mirzaei, H., Beglov, D., Paschalidis, I. C., Vajda, S., Vakili, P., & Kozakov, D. (2012).
            // Rigid body energy minimization on manifolds for molecular docking. Journal of Chemical
            // Theory and Computation, 8(11), 4374–4380. https://doi.org/10.1021/ct300272j
            out_g[(3 + i) * STRIDE_G] = frobenius_product(mat, dR_dv);
        }
        DPrint("orientation_g: %f, %f, %f\n", out_g[3 * STRIDE_G], out_g[4 * STRIDE_G], out_g[5 * STRIDE_G]);

    }
    tile.sync();
}


SCOPE_INLINE Real cal_box_penalty_clamp_coord(Real* out_x, Real* out_y, Real* out_z, Real* out_f){
    Real penalty = 0.;
    if (*out_x < BOX_X_LO){
        penalty += BOX_X_LO - *out_x;
        *out_x = BOX_X_LO;
        out_f[0] += -PENALTY_SLOPE;
    }
    else if (*out_x > BOX_X_HI){
        penalty += *out_x - BOX_X_HI;
        *out_x = BOX_X_HI;
        out_f[0] -= -PENALTY_SLOPE;
    }

    if (*out_y < BOX_Y_LO){
        penalty += BOX_Y_LO - *out_y;
        *out_y = BOX_Y_LO;
        out_f[1] += -PENALTY_SLOPE;
    }
    else if (*out_y > BOX_Y_HI){
        penalty += *out_y - BOX_Y_HI;
        *out_y = BOX_Y_HI;
        out_f[1] -= -PENALTY_SLOPE;
    }

    if (*out_z < BOX_Z_LO){
        penalty += BOX_Z_LO - *out_z;
        *out_z = BOX_Z_LO;
        out_f[2] += -PENALTY_SLOPE;
    }
    else if (*out_z > BOX_Z_HI){
        penalty += *out_z - BOX_Z_HI;
        *out_z = BOX_Z_HI;
        out_f[2] -= -PENALTY_SLOPE;
    }

    return penalty * PENALTY_SLOPE;
}


// Vina-style energy capping (curl), identical to curl_cpu in the CPU backend:
// caps positive pair energy at ~v and squashes its gradient by (v/(v+e))^2.
#ifndef UD2_CURL_V
#define UD2_CURL_V 10.0
#endif
#define CURL_V_CU Real(UD2_CURL_V)   // -DUD2_CURL_V=1e30 disables the cap (experiments)
#ifndef UD2_CURL_V_AUTHENTIC
#define UD2_CURL_V_AUTHENTIC 1000.0
#endif
#define CURL_V_AUTHENTIC_CU Real(UD2_CURL_V_AUTHENTIC)  // near-uncapped scoring for rescoring and refinement after the search (Vina authentic_v)
__device__ __forceinline__ void curl_warp(Real& e, Real& f, Real v){
    if (e > 0){
        Real tmp = v / (v + e);
        e *= tmp;
        f *= tmp * tmp;
    }
}


/**
 * @brief Angle-weighted HBond adjustment for one ligand-receptor pair (mirrors the CPU backend).
 * Direction 1: ligand donor (i1, with bonded H from h_topo) -> receptor acceptor.
 * Direction 2: receptor donor (bonded H coords from rh) -> ligand acceptor (i1).
 * For DA-DA pairs the direction with the larger angular weight wins.
 * On return e_pair / f_div_r carry (w-1)*E_hb and the angular gradient terms have been
 * added: the ligand donor/acceptor contribution into (fx, fy, fz), the ligand H
 * contribution (direction 1 only) into aux_f via atomicAdd_block (the H belongs to a
 * lane other than this one).
 */
__device__ __forceinline__ void hbond_angular_pair_warp(
        int i1, Real lx, Real ly, Real lz, bool lig_is_donor, bool lig_is_acceptor,
        int i2, Real rx, Real ry, Real rz, int t1, int t2, Real d,
        const FlexPose* pose, const Real* fix_coords,
        const BondedHTopo& h_topo, const RecDonorHTopo& rh, const AtomTypeProps& atp,
        Real& e_pair, Real& f_div_r, Real& fx, Real& fy, Real& fz, Real* aux_f){
    Real f_hb = 0;
    Real e_hb = Score.weight_hbond_val() * Score.vina_hbond(d, &f_hb);
    f_hb *= Score.weight_hbond_val();
    if (fabsf(e_hb) <= Real(1e-12)) return;

    // Same rules as cal_e_f_cpu: rotatable donors (-OH, -NH2, -NH3+) use the best hydrogen
    // on the bond cone (hbond_cone_cos), donors without hydrogens or heavy neighbour (water,
    // metals) keep full weight, fixed donors use their bonded hydrogens.
    // mode: 0 none, 1 explicit ligand H (best_idx = H atom), 2 ligand cone (best_idx =
    // ligand antecedent), 3 explicit receptor H (best_idx = table row), 4 receptor cone
    // (best_idx = receptor antecedent)
    Real best_w = 0, best_dw = 0, best_cos = 1;
    int best_mode = 0, best_idx = -1;
    const Real D1[3] = {lx, ly, lz};
    const Real A1[3] = {rx, ry, rz};

    auto consider_cos = [&](Real cos_dha, int mode, int idx){
        Real dw = 0;
        Real w = angular_hbond_weight(cos_dha, &dw);
        if (w > best_w){ best_w = w; best_dw = dw; best_cos = cos_dha; best_mode = mode; best_idx = idx; }
    };
    auto consider_h = [&](Real hx, Real hy, Real hz, const Real* Dp, const Real* Ap, int mode, int idx){
        Real hd_x = Dp[0] - hx, hd_y = Dp[1] - hy, hd_z = Dp[2] - hz;
        Real ha_x = Ap[0] - hx, ha_y = Ap[1] - hy, ha_z = Ap[2] - hz;
        Real hd_sq = hd_x * hd_x + hd_y * hd_y + hd_z * hd_z;
        Real ha_sq = ha_x * ha_x + ha_y * ha_y + ha_z * ha_z;
        if (hd_sq < Real(1e-12) || ha_sq < Real(1e-12)) return;
        consider_cos((hd_x * ha_x + hd_y * ha_y + hd_z * ha_z) / (sqrtf(hd_sq) * sqrtf(ha_sq)), mode, idx);
    };
    auto full_credit = [&](){
        if (best_w < Real(1)){ best_w = Real(1); best_dw = 0; best_mode = 0; }
    };

    // Direction 1: ligand donor -> receptor acceptor
    if (lig_is_donor && atp.is_acceptor[t2]){
        const int hb = h_topo.h_start[i1];
        const int he = h_topo.h_start[i1 + 1];
        const int n_h = he - hb;
        const bool rot = h_topo.donor_rotatable ? h_topo.donor_rotatable[i1] != 0 : hbond_donor_rotatable(t1, n_h);
        if (rot){
            const int x = h_topo.antecedent ? h_topo.antecedent[i1] : -1;
            if (x >= 0) consider_cos(hbond_cone_cos(D1, pose->coords + x * 3, A1, nullptr, nullptr, nullptr), 2, x);
            else full_credit();
        } else if (n_h == 0){
            full_credit();
        } else {
            for (int ih = hb; ih < he; ih++){
                const int hi = h_topo.h_indices[ih];
                consider_h(pose->coords[hi * 3], pose->coords[hi * 3 + 1], pose->coords[hi * 3 + 2], D1, A1, 1, hi);
            }
        }
    }

    // Direction 2: receptor donor -> ligand acceptor
    if (lig_is_acceptor && atp.is_donor[t2] && rh.natom > 0){
        const int rs = rh.h_start[i2];
        const int re = rh.h_start[i2 + 1];
        const int n_h = re - rs;
        const bool rot = rh.donor_rotatable ? rh.donor_rotatable[i2] != 0 : hbond_donor_rotatable(t2, n_h);
        if (rot){
            const int x = rh.antecedent ? rh.antecedent[i2] : -1;
            if (x >= 0) consider_cos(hbond_cone_cos(A1, fix_coords + x * 3, D1, nullptr, nullptr, nullptr), 4, x);
            else full_credit();
        } else if (n_h == 0){
            full_credit();
        } else {
            for (int ih = rs; ih < re; ih++){
                consider_h(rh.h_coords[ih * 3], rh.h_coords[ih * 3 + 1], rh.h_coords[ih * 3 + 2], A1, D1, 3, ih);
            }
        }
    }

    // E = E_non_hb + w*E_hb = E_total + (w-1)*E_hb
    e_pair += (best_w - Real(1)) * e_hb;
    f_div_r += (best_w - Real(1)) * f_hb;

    if (best_mode == 0 || fabsf(best_dw) <= Real(1e-12)) return;
    const Real scale = e_hb * best_dw;

    if (best_mode == 1){
        // Gradient on ligand D (this lane's atom) and ligand H (another lane's atom)
        const int h3 = best_idx * 3;
        const Real hx = pose->coords[h3], hy = pose->coords[h3 + 1], hz = pose->coords[h3 + 2];
        Real hd_x = lx - hx, hd_y = ly - hy, hd_z = lz - hz;
        Real ha_x = rx - hx, ha_y = ry - hy, ha_z = rz - hz;
        Real inv_hd = rsqrtf(hd_x * hd_x + hd_y * hd_y + hd_z * hd_z);
        Real inv_ha = rsqrtf(ha_x * ha_x + ha_y * ha_y + ha_z * ha_z);
        Real inv_hd_ha = inv_hd * inv_ha;
        Real hd_hat_x = hd_x * inv_hd, hd_hat_y = hd_y * inv_hd, hd_hat_z = hd_z * inv_hd;
        Real ha_hat_x = ha_x * inv_ha, ha_hat_y = ha_y * inv_ha, ha_hat_z = ha_z * inv_ha;

        Real dcos_dH_x = -inv_hd_ha * (lx + rx - 2 * hx) + best_cos * (hd_hat_x * inv_hd + ha_hat_x * inv_ha);
        Real dcos_dH_y = -inv_hd_ha * (ly + ry - 2 * hy) + best_cos * (hd_hat_y * inv_hd + ha_hat_y * inv_ha);
        Real dcos_dH_z = -inv_hd_ha * (lz + rz - 2 * hz) + best_cos * (hd_hat_z * inv_hd + ha_hat_z * inv_ha);

        atomicAdd_block(aux_f + h3,     scale * dcos_dH_x);
        atomicAdd_block(aux_f + h3 + 1, scale * dcos_dH_y);
        atomicAdd_block(aux_f + h3 + 2, scale * dcos_dH_z);

        fx += scale * (inv_hd_ha * ha_x - best_cos * hd_hat_x * inv_hd);
        fy += scale * (inv_hd_ha * ha_y - best_cos * hd_hat_y * inv_hd);
        fz += scale * (inv_hd_ha * ha_z - best_cos * hd_hat_z * inv_hd);
    }
    else if (best_mode == 2){
        // Ligand cone: gradient on the donor (this lane) and its antecedent (another lane)
        Real gD[3], gX[3], gA[3];
        hbond_cone_cos(D1, pose->coords + best_idx * 3, A1, gD, gX, gA);
        const int x3 = best_idx * 3;
        atomicAdd_block(aux_f + x3,     scale * gX[0]);
        atomicAdd_block(aux_f + x3 + 1, scale * gX[1]);
        atomicAdd_block(aux_f + x3 + 2, scale * gX[2]);
        fx += scale * gD[0];
        fy += scale * gD[1];
        fz += scale * gD[2];
    }
    else if (best_mode == 3){
        // Explicit receptor H: gradient only on the ligand acceptor (this lane's atom)
        const Real hx = rh.h_coords[best_idx * 3], hy = rh.h_coords[best_idx * 3 + 1], hz = rh.h_coords[best_idx * 3 + 2];
        Real hd_x = rx - hx, hd_y = ry - hy, hd_z = rz - hz;
        Real ha_x = lx - hx, ha_y = ly - hy, ha_z = lz - hz;
        Real inv_hd = rsqrtf(hd_x * hd_x + hd_y * hd_y + hd_z * hd_z);
        Real inv_ha = rsqrtf(ha_x * ha_x + ha_y * ha_y + ha_z * ha_z);
        Real inv_hd_ha = inv_hd * inv_ha;
        Real ha_hat_x = ha_x * inv_ha, ha_hat_y = ha_y * inv_ha, ha_hat_z = ha_z * inv_ha;

        fx += scale * (inv_hd_ha * hd_x - best_cos * ha_hat_x * inv_ha);
        fy += scale * (inv_hd_ha * hd_y - best_cos * ha_hat_y * inv_ha);
        fz += scale * (inv_hd_ha * hd_z - best_cos * ha_hat_z * inv_ha);
    }
    else {
        // Receptor cone: gradient only on the ligand acceptor (this lane's atom)
        Real gD[3], gX[3], gA[3];
        hbond_cone_cos(A1, fix_coords + best_idx * 3, D1, gD, gX, gA);
        fx += scale * gA[0];
        fy += scale * gA[1];
        fz += scale * gA[2];
    }
}


/**
 * @brief Energy and Cartesian forces of one pose, computed by one tile (warp).
 * Layout mirrors cal_e_f_cpu: one lane per ligand atom, register accumulation of the
 * atom's own force, so no atomics are needed except for the rare ligand-H angular term.
 *   1. inter (ligand heavy atom x receptor heavy atom) with angular HBond and curl
 *   2. box penalty per ligand heavy atom
 *   3. intra over the symmetric CSR neighbor list with curl (each pair seen from both sides,
 *      energy counted once); pair terms use Score (the search set) scaled by `intra`, whereas the
 *      host score() uses Vina's default pair weights (SF_INTRA); kept deliberately, see score.h
 * GRID = true replaces step 1 by the receptor maps (g_maps) plus an exact hydrogen-bond pass over
 * g_pgrid (see maps.cuh); every other step is shared. GRID = false is the analytic path.
 * PREFETCH = true loads one step ahead in the polar candidate walk (GRID only) and in the intra CSR
 * walk; the result is bit-identical. mc_kernel sets it only in its large-launch (screening)
 * instantiation, so every other kernel compiles to the PREFETCH = false code.
 */
template<bool GRID = false, bool PREFETCH = false, class Tile>
__device__ __forceinline__ Real cal_e_f_warp(const Tile& tile,
                                             const FlexPose* pose, const FlexTopo& flex_topo,
                                             const FixMol& fix_mol, const FlexParamVina& flex_param,
                                             const FixParamVina& fix_param,
                                             Real* aux_f, Real v_cap = CURL_V_CU){
    Real energy = 0.;
    const int natom = flex_topo.natom;
    const Real r2_cutoff = Score.r2_cutoff;
    const Real w_intra = Score.weight_intra_val();
    const Real w_elec = Score.weight_elec_val();
    const Real* lig_q = (w_elec != Real(0)) ? flex_param.charges : nullptr;
    const bool use_angular = USE_ANGULAR_HBOND;
    const BondedHTopo& h_topo = flex_param.bonded_h_topo;
    const RecDonorHTopo& rh = fix_param.rec_donor_h;
    PROF_DECL; PROF_COUNT(PROF_N_EVAL);

    // Per-type tables are read with lane-divergent indices inside the pair loops. In
    // constant memory such reads serialize on the constant cache; shared memory serves
    // them in parallel. Stage them once per evaluation (one warp per block).
    __shared__ AtomTypeProps s_atp;
    __shared__ Real s_vdw[MAX_VN_TYPES];
    {
        const int* src = reinterpret_cast<const int*>(&g_atp);
        int* dst = reinterpret_cast<int*>(&s_atp);
        for (int i = tile.thread_rank(); i < (int)(sizeof(AtomTypeProps) / sizeof(int)); i += tile.num_threads()) dst[i] = src[i];
        for (int i = tile.thread_rank(); i < MAX_VN_TYPES; i += tile.num_threads()) s_vdw[i] = g_vdw_radii[i];
    }
    tile.sync();
    const AtomTypeProps& atp = s_atp;
#ifdef UD2_PROFILE
    unsigned int prof_visited = 0, prof_scored = 0;
#endif

    // 0. initialize aux_f as zero
    for (int i = tile.thread_rank(); i < natom * 3; i += tile.num_threads()){
        aux_f[i] = 0.;
    }
    tile.sync();

    // 1. inter-molecular: flex heavy atoms x receptor heavy atoms.
    //
    //    Grid path (cell list uploaded): lanes walk the 27 cells (as 9 z-spans) around their
    //    own ligand atom in lockstep, but only the cheap distance test runs there. Pairs
    //    inside the cutoff are pushed into a shared queue with a warp ballot, and scored 32
    //    at a time with every lane busy (score_batch). Without this, every warp iteration
    //    executed the full scoring path for the few lanes that had a hit (~11 % of visits).
    //    Forces accumulate in dynamic shared memory (s_force, natom * 3 Reals) because a
    //    batch mixes ligand atoms across lanes.
    //
    //    Fallback (no grid): the original per-lane loop over every receptor atom.
    extern __shared__ Real s_force_dyn[];   // sized at launch: min(max natom, KERNEL_SHARED_FORCE_MAX_ATOMS) * 3
    __shared__ unsigned int s_queue[32 * UD2_WALK_DEPTH + 32];  // (i1 << 20) | grid atom index; D pushes of <= 32 before a flush
    // force accumulator for the compacted paths: shared memory when it fits, else aux_f itself
    const bool force_in_shared = natom <= KERNEL_SHARED_FORCE_MAX_ATOMS;
    Real* s_force = force_in_shared ? s_force_dyn : aux_f;

    // score one queued pair on this lane; forces via shared atomics, energy into `energy`
    auto score_pair = [&](int i1, int k){
        const int t1 = flex_param.atom_types[i1];
        const Real lx = pose->coords[i1 * 3], ly = pose->coords[i1 * 3 + 1], lz = pose->coords[i1 * 3 + 2];
        const GridAtom rec = g_grid.atoms[k];
        const Real r_vec_x = rec.x - lx, r_vec_y = rec.y - ly, r_vec_z = rec.z - lz;
        Real r = sqrtf(r_vec_x * r_vec_x + r_vec_y * r_vec_y + r_vec_z * r_vec_z);
        Real d = r - (s_vdw[t1] + rec.radius);
        Real f_div_r = 0;
        Real e_pair = Score.eval_ef(d, t1, rec.type, &f_div_r, atp);
        if (lig_q != nullptr){
            Real f_el = 0;
            e_pair += w_elec * Score.elec_pair(lig_q[i1] * rec.q, r, &f_el);
            f_div_r += w_elec * f_el;
        }
        Real fx = 0., fy = 0., fz = 0.;
        const bool lig_is_donor = use_angular && atp.is_donor[t1];
        const bool lig_is_acceptor = use_angular && atp.is_acceptor[t1];
        if ((lig_is_donor || lig_is_acceptor) && d < Vina::hbond_high_val() && xs_h_bond_possible(t1, rec.type, atp)){
            hbond_angular_pair_warp(i1, lx, ly, lz, lig_is_donor, lig_is_acceptor,
                                    rec.index, rec.x, rec.y, rec.z, t1, rec.type, d, pose, fix_mol.coords, h_topo, rh, atp,
                                    e_pair, f_div_r, fx, fy, fz, s_force);
        }
        curl_warp(e_pair, f_div_r, v_cap);
        energy += e_pair;
        if (r < EPSILON) r = EPSILON;
        f_div_r /= r;
        fx -= f_div_r * r_vec_x;
        fy -= f_div_r * r_vec_y;
        fz -= f_div_r * r_vec_z;
        atomicAdd_block(s_force + i1 * 3,     fx);
        atomicAdd_block(s_force + i1 * 3 + 1, fy);
        atomicAdd_block(s_force + i1 * 3 + 2, fz);
    };

    if (force_in_shared){
        for (int i = tile.thread_rank(); i < natom * 3; i += tile.num_threads()) s_force[i] = 0.;
    }
    tile.sync();

    if constexpr (GRID){
        // GRID path. Per heavy lane the receptor maps replace the cell walk and the pair scoring:
        // trilinear value and gradient of the per-pair-curled sum of eval_ef_nohb for this lane's
        // atom type, cap set chosen by v_cap. Hydrogen bonds are not in the maps: a donor / acceptor
        // lane walks g_pgrid (receptor donors / acceptors, 4 A cells) and, for every pair with d < 0,
        // adds what score_pair computes for it minus what the map holds for it (eval_ef_nohb curled
        // the same way), so the total is the analytic sum up to the interpolation error of the nohb
        // terms. Forces go to s_force as in score_pair; the lane's own atom needs atomics too because
        // the angular routine adds to other lanes' atoms (ligand H / antecedent).
        __shared__ int s_slot[MAX_VN_TYPES];
        for (int i = tile.thread_rank(); i < MAX_VN_TYPES; i += tile.num_threads()) s_slot[i] = g_maps.slot_of_type[i];
        tile.sync();
        const int cap = (v_cap == CURL_V_AUTHENTIC_CU) ? 1 : 0;
        const int mnx = g_maps.nx, mny = g_maps.ny, mnz = g_maps.nz;
        const int m_ny_nz = mny * mnz;
        const int m_pts = mnx * m_ny_nz;
        const Real inv_h = g_maps.inv_h;
        const float* __restrict__ m_cap = g_maps.data + cap * g_maps.nslot * m_pts;
        const bool polar_pass = g_pgrid.built;
        const int p_ny_nz = g_pgrid.ny * g_pgrid.nz;
        const int* __restrict__ pcs = g_pgrid.cell_start;
        // candidate lists (maps.cu upload_polar_cands) instead of the walk. A ligand whose free-rotor donors share
        // an antecedent keeps the walk: hbond_angular_pair_warp mode 2 adds to the antecedent's s_force slot from
        // two lanes inside the loop, and the lists change at which iteration each lane reaches a pair, hence the
        // order of those atomics (the per-lane ordering argument in maps.cu does not cover this).
        const bool cand_pass = g_pcand.built && h_topo.n_shared_cone_antecedent == 0;
        const int* __restrict__ ccs = g_pcand.cell_start;
        const GridAtom* __restrict__ catoms = g_pcand.atoms;
        const int n_walk = flex_param.heavy_ids ? flex_param.nheavy : natom;
        for (int slot = tile.thread_rank(); slot < n_walk; slot += tile.num_threads()){
            const int i1 = flex_param.heavy_ids ? flex_param.heavy_ids[slot] : slot;
            const int t1 = flex_param.atom_types[i1];
            if (atp.is_hydrogen[t1]) continue;
            const int s = s_slot[t1];
            if (s < 0) continue;
            const Real lx = pose->coords[i1 * 3], ly = pose->coords[i1 * 3 + 1], lz = pose->coords[i1 * 3 + 2];

            // map lookup: the position is clamped into [origin, origin + (n - 1) h] per axis; beyond
            // the map the energy is the edge value and its derivative along that axis is zero
            Real gx = (lx - g_maps.origin[0]) * inv_h;
            Real gy = (ly - g_maps.origin[1]) * inv_h;
            Real gz = (lz - g_maps.origin[2]) * inv_h;
            const Real sx = (gx < Real(0) || gx > Real(mnx - 1)) ? Real(0) : inv_h;
            const Real sy = (gy < Real(0) || gy > Real(mny - 1)) ? Real(0) : inv_h;
            const Real sz = (gz < Real(0) || gz > Real(mnz - 1)) ? Real(0) : inv_h;
            gx = fminf(fmaxf(gx, Real(0)), Real(mnx - 1));
            gy = fminf(fmaxf(gy, Real(0)), Real(mny - 1));
            gz = fminf(fmaxf(gz, Real(0)), Real(mnz - 1));
            const int ix = min((int)gx, mnx - 2), iy = min((int)gy, mny - 2), iz = min((int)gz, mnz - 2);
            const Real tx = gx - ix, ty = gy - iy, tz = gz - iz;
            const float* m = m_cap + s * m_pts + ix * m_ny_nz + iy * mnz + iz;
            const Real c000 = __ldg(m),                 c001 = __ldg(m + 1);
            const Real c010 = __ldg(m + mnz),           c011 = __ldg(m + mnz + 1);
            const Real c100 = __ldg(m + m_ny_nz),       c101 = __ldg(m + m_ny_nz + 1);
            const Real c110 = __ldg(m + m_ny_nz + mnz), c111 = __ldg(m + m_ny_nz + mnz + 1);
            // lerp along z, then y, then x; the gradient is the exact derivative of the interpolant
            // (aux_f convention: +dE/dx, as the pair loops' fx -= f_div_r * r_vec_x)
            const Real z00 = c000 + tz * (c001 - c000), z01 = c010 + tz * (c011 - c010);
            const Real z10 = c100 + tz * (c101 - c100), z11 = c110 + tz * (c111 - c110);
            const Real y0 = z00 + ty * (z01 - z00), y1 = z10 + ty * (z11 - z10);
            energy += y0 + tx * (y1 - y0);
            Real fx = (y1 - y0) * sx;
            Real fy = ((Real(1) - tx) * (z01 - z00) + tx * (z11 - z10)) * sy;
            Real fz = ((Real(1) - tx) * ((Real(1) - ty) * (c001 - c000) + ty * (c011 - c010))
                     + tx * ((Real(1) - ty) * (c101 - c100) + ty * (c111 - c110))) * sz;

            // polar pass: pairs with d < hbond_high (= 0) only; beyond it the hbond ramp is zero and
            // eval_ef equals eval_ef_nohb, so the difference vanishes
            if (polar_pass && (atp.is_donor[t1] || atp.is_acceptor[t1])){
                const Real r1 = s_vdw[t1];
                const bool lig_is_donor = use_angular && atp.is_donor[t1];
                const bool lig_is_acceptor = use_angular && atp.is_acceptor[t1];
                // Receptor polar atoms to test, as z-spans of one CSR cell list: the candidate list of the
                // lane's g_pcand cell when built (one span of one cell), else the 27 cells of g_pgrid around
                // the lane's cell (9 spans of zspan cells). Both yield the contributing atoms in the same
                // order (maps.cu upload_polar_cands), and the loop body below is the per-pair code of both.
                const int* __restrict__ cs;
                const GridAtom* __restrict__ src;
                int wx_lo, wx_hi, wy_lo, wy_hi, zlo, zspan, stride_x, stride_y;
                if (cand_pass){
                    int cx = (int)((lx - g_pcand.origin[0]) * g_pcand.inv_cell);
                    int cy = (int)((ly - g_pcand.origin[1]) * g_pcand.inv_cell);
                    int cz = (int)((lz - g_pcand.origin[2]) * g_pcand.inv_cell);
                    cx = min(max(cx, 0), g_pcand.nx - 1); cy = min(max(cy, 0), g_pcand.ny - 1); cz = min(max(cz, 0), g_pcand.nz - 1);
                    cs = ccs; src = catoms;
                    wx_lo = wx_hi = cx; wy_lo = wy_hi = cy; zlo = cz; zspan = 1;
                    stride_x = g_pcand.ny * g_pcand.nz; stride_y = g_pcand.nz;
                } else {
                    int cx = (int)((lx - g_pgrid.origin[0]) * g_pgrid.inv_cell_size);
                    int cy = (int)((ly - g_pgrid.origin[1]) * g_pgrid.inv_cell_size);
                    int cz = (int)((lz - g_pgrid.origin[2]) * g_pgrid.inv_cell_size);
                    cx = min(max(cx, 0), g_pgrid.nx - 1); cy = min(max(cy, 0), g_pgrid.ny - 1); cz = min(max(cz, 0), g_pgrid.nz - 1);
                    cs = pcs; src = g_pgrid.atoms;
                    wx_lo = max(0, cx - g_pgrid.half); wx_hi = min(g_pgrid.nx - 1, cx + g_pgrid.half);
                    wy_lo = max(0, cy - g_pgrid.half); wy_hi = min(g_pgrid.ny - 1, cy + g_pgrid.half);
                    zlo = max(0, cz - g_pgrid.half); zspan = min(g_pgrid.nz - 1, cz + g_pgrid.half) - zlo + 1;
                    stride_x = p_ny_nz; stride_y = g_pgrid.nz;
                }
                for (int wx = wx_lo; wx <= wx_hi; wx++){
                    for (int wy = wy_lo; wy <= wy_hi; wy++){
                        const int c = wx * stride_x + wy * stride_y + zlo;
                        const int k_end = __ldg(cs + c + zspan);
                        if constexpr (PREFETCH){
                            const int k_begin = __ldg(cs + c);
                            // The receptor type is loaded one step ahead (never past the span's end), so the type test
                            // of step k waits on no global load; the atom itself is loaded at the top of step k and
                            // arrives during that test. Still one atom per iteration and one copy of the per-pair code,
                            // so each lane meets its pairs, and issues its atomics, at the same iteration as before.
                            int t_next = (k_begin < k_end) ? __ldg(&src[k_begin].type) : 0;
                            for (int k = k_begin; k < k_end; k++){
                                const int t2 = t_next;
                                if (k + 1 < k_end) t_next = __ldg(&src[k + 1].type);
                                const GridAtom rec = src[k];
#ifdef UD2_PROFILE
                                prof_visited++;
#endif
                                if (!xs_h_bond_possible(t1, t2, atp)) continue;
                                const Real r_vec_x = rec.x - lx, r_vec_y = rec.y - ly, r_vec_z = rec.z - lz;
                                Real r = sqrtf(r_vec_x * r_vec_x + r_vec_y * r_vec_y + r_vec_z * r_vec_z);
                                const Real d = r - (r1 + rec.radius);
                                if (!(d < Vina::hbond_high_val())) continue;
#ifdef UD2_PROFILE
                                prof_scored++;
#endif
                                // the pair as score_pair scores it (rec.index is the original receptor index)
                                Real f_div_r = 0;
                                Real e_pair = Score.eval_ef(d, t1, t2, &f_div_r, atp);
                                Real hx = 0., hy = 0., hz = 0.;
                                if (lig_is_donor || lig_is_acceptor){
                                    hbond_angular_pair_warp(i1, lx, ly, lz, lig_is_donor, lig_is_acceptor,
                                                            rec.index, rec.x, rec.y, rec.z, t1, t2, d, pose, fix_mol.coords, h_topo, rh, atp,
                                                            e_pair, f_div_r, hx, hy, hz, s_force);
                                }
                                curl_warp(e_pair, f_div_r, v_cap);
                                // minus what the map holds for it
                                Real f_m = 0;
                                Real e_m = Score.eval_ef_nohb(d, t1, t2, &f_m, atp);
                                curl_warp(e_m, f_m, v_cap);
                                energy += e_pair - e_m;
                                if (r < EPSILON) r = EPSILON;
                                const Real fd = (f_div_r - f_m) / r;
                                fx += hx - fd * r_vec_x;
                                fy += hy - fd * r_vec_y;
                                fz += hz - fd * r_vec_z;
                            }
                        } else {
                            for (int k = __ldg(cs + c); k < k_end; k++){
                                const GridAtom rec = src[k];
#ifdef UD2_PROFILE
                                prof_visited++;
#endif
                                const int t2 = rec.type;
                                if (!xs_h_bond_possible(t1, t2, atp)) continue;
                                const Real r_vec_x = rec.x - lx, r_vec_y = rec.y - ly, r_vec_z = rec.z - lz;
                                Real r = sqrtf(r_vec_x * r_vec_x + r_vec_y * r_vec_y + r_vec_z * r_vec_z);
                                const Real d = r - (r1 + rec.radius);
                                if (!(d < Vina::hbond_high_val())) continue;
#ifdef UD2_PROFILE
                                prof_scored++;
#endif
                                // the pair as score_pair scores it (rec.index is the original receptor index)
                                Real f_div_r = 0;
                                Real e_pair = Score.eval_ef(d, t1, t2, &f_div_r, atp);
                                Real hx = 0., hy = 0., hz = 0.;
                                if (lig_is_donor || lig_is_acceptor){
                                    hbond_angular_pair_warp(i1, lx, ly, lz, lig_is_donor, lig_is_acceptor,
                                                            rec.index, rec.x, rec.y, rec.z, t1, t2, d, pose, fix_mol.coords, h_topo, rh, atp,
                                                            e_pair, f_div_r, hx, hy, hz, s_force);
                                }
                                curl_warp(e_pair, f_div_r, v_cap);
                                // minus what the map holds for it
                                Real f_m = 0;
                                Real e_m = Score.eval_ef_nohb(d, t1, t2, &f_m, atp);
                                curl_warp(e_m, f_m, v_cap);
                                energy += e_pair - e_m;
                                if (r < EPSILON) r = EPSILON;
                                const Real fd = (f_div_r - f_m) / r;
                                fx += hx - fd * r_vec_x;
                                fy += hy - fd * r_vec_y;
                                fz += hz - fd * r_vec_z;
                            }
                        }
                    }
                }
            }
            atomicAdd_block(s_force + i1 * 3,     fx);
            atomicAdd_block(s_force + i1 * 3 + 1, fy);
            atomicAdd_block(s_force + i1 * 3 + 2, fz);
        }
    } else {
    if (g_grid.built){
        const int ny_nz = g_grid.ny * g_grid.nz;
        unsigned int qn = 0;  // queue fill, warp-uniform

        // flush `count` entries from the queue head: lane j scores entry j, then compact the rest
        auto flush = [&](unsigned int count){
            unsigned int e = 0;
            if (tile.thread_rank() < count) e = s_queue[tile.thread_rank()];
            tile.sync();
            if (tile.thread_rank() < count) score_pair((int)(e >> 20), (int)(e & 0xFFFFFu));
            unsigned int rest = qn - count;
            for (unsigned int off = 0; off < rest; off += tile.num_threads()){
                unsigned int moved = 0;
                if (off + tile.thread_rank() < rest) moved = s_queue[count + off + tile.thread_rank()];
                tile.sync();
                if (off + tile.thread_rank() < rest) s_queue[off + tile.thread_rank()] = moved;
            }
            tile.sync();
            qn = rest;
        };

        // Walk only the heavy atoms: hydrogens score nothing and would idle a lane each.
        //  - Cells (ix, iy, cz_lo..cz_hi) have consecutive indices, so their atoms form one
        //    contiguous range of the cell-sorted array: a lane walks (2*half+1)^2 z-spans
        //    instead of (2*half+1)^3 cells, in the same order. The bounds of the lane's next
        //    span are loaded when a span is entered.
        //  - The distance test reads the first 16 B of the GridAtom (x, y, z, radius); the
        //    atom's index is kept in a register. The scoring of a hit then reads the same
        //    GridAtom, which the test has just brought into L1.
        //  - Software pipeline, D steps per group: p[s] holds the atom of step s. The D tests
        //    run first, then p[] is refilled with the next D atoms (loads in flight during the
        //    pushes and the next group's tests), then the hits of the D steps are pushed and
        //    the queue is flushed 32 at a time. One flush site per group keeps a single inlined
        //    copy of score_pair in the loop: the hot code of this loop decides throughput when
        //    several processes share the GPU under MPS (their code competes for the SM's
        //    instruction cache), so it must not grow with D.
        // Every step tests the same atom on every lane as a one-atom-per-iteration walk, and
        // hits are pushed and flushed in the same order, so the scoring batches are unchanged.
        // Steps past a lane's end test nothing; steps past the warp's end push nothing.
        constexpr int D = UD2_WALK_DEPTH;
        const int n_walk = flex_param.heavy_ids ? flex_param.nheavy : natom;
        const int* __restrict__ cs = g_grid.cell_start;
        static_assert(sizeof(Real) == sizeof(float) && sizeof(GridAtom) == 2 * sizeof(float4),
                      "z-span walk reads GridAtom as two float4 with (x, y, z, radius) first: needs Real == float "
                      "(USE_DOUBLE is not supported here) and a 32-byte GridAtom");
        const float4* __restrict__ xyz = reinterpret_cast<const float4*>(g_grid.atoms);  // (x, y, z, radius), 32 B stride
        const unsigned int lanemask_lt = (1u << tile.thread_rank()) - 1u;
        for (int base = 0; base < n_walk; base += tile.num_threads()){
            const int slot = base + tile.thread_rank();
            const int i1 = (slot < n_walk)
                         ? (flex_param.heavy_ids ? flex_param.heavy_ids[slot] : slot)
                         : natom;
            // per-lane cursor over the z-spans around this lane's atom; the next atom to fetch is k
            bool active = false;       // the cursor has an atom at k
            bool nvalid = false;       // (sx, sy) is a span still to visit; nk / nkend are its bounds
            int cx_hi = -1, cy_lo = 0, cy_hi = -1, zlo = 0, zspan = 0;
            int sx = 0, sy = 0, k = 0, kend = 0, nk = 0, nkend = 0;
            Real lx = 0, ly = 0, lz = 0;
            // move (sx, sy) to the following span and load its bounds
            auto next_span = [&](){
                if (++sy > cy_hi){ sy = cy_lo; if (++sx > cx_hi) nvalid = false; }
                if (nvalid){
                    const int c = sx * ny_nz + sy * g_grid.nz + zlo;
                    nk = __ldg(cs + c); nkend = __ldg(cs + c + zspan);
                }
            };
            if (i1 < natom && !atp.is_hydrogen[flex_param.atom_types[i1]]){
                lx = pose->coords[i1 * 3]; ly = pose->coords[i1 * 3 + 1]; lz = pose->coords[i1 * 3 + 2];
                int cx = (int)((lx - g_grid.origin[0]) * g_grid.inv_cell_size);
                int cy = (int)((ly - g_grid.origin[1]) * g_grid.inv_cell_size);
                int cz = (int)((lz - g_grid.origin[2]) * g_grid.inv_cell_size);
                cx = min(max(cx, 0), g_grid.nx - 1); cy = min(max(cy, 0), g_grid.ny - 1); cz = min(max(cz, 0), g_grid.nz - 1);
                sx = max(0, cx - g_grid.half); cx_hi = min(g_grid.nx - 1, cx + g_grid.half);
                cy_lo = max(0, cy - g_grid.half); cy_hi = min(g_grid.ny - 1, cy + g_grid.half);
                zlo = max(0, cz - g_grid.half); zspan = min(g_grid.nz - 1, cz + g_grid.half) - zlo + 1;
                sy = cy_lo;
                const int c = sx * ny_nz + sy * g_grid.nz + zlo;
                k = __ldg(cs + c); kend = __ldg(cs + c + zspan);
                nvalid = true;
                next_span();
                active = true;
            }
            // advance to the next non-empty span if the current one is exhausted
            auto settle = [&](){
                while (active && k >= kend){
                    if (!nvalid){ active = false; break; }
                    k = nk; kend = nkend;
                    next_span();
                }
            };
            settle();

            float4 p[D]; bool v[D]; int kp[D];
#pragma unroll
            for (int s = 0; s < D; s++){
                p[s] = make_float4(0.f, 0.f, 0.f, 0.f);
                v[s] = active;
                kp[s] = 0;
                if (active){ kp[s] = k; p[s] = __ldg(xyz + 2 * k); k++; settle(); }
            }
            while (tile.any(v[0])){
                bool hit[D]; unsigned int entry[D];
#pragma unroll
                for (int s = 0; s < D; s++){
                    const Real dx = p[s].x - lx, dy = p[s].y - ly, dz = p[s].z - lz;
                    hit[s] = v[s] && (dx * dx + dy * dy + dz * dz < r2_cutoff);
                    entry[s] = ((unsigned int)i1 << 20) | (unsigned int)kp[s];
#ifdef UD2_PROFILE
                    if (v[s]) prof_visited++;
                    if (hit[s]) prof_scored++;
#endif
                }
#pragma unroll
                for (int s = 0; s < D; s++){
                    v[s] = active;
                    if (active){ kp[s] = k; p[s] = __ldg(xyz + 2 * k); k++; settle(); }
                }
#pragma unroll
                for (int s = 0; s < D; s++){
                    const unsigned int mask = tile.ballot(hit[s]);
                    if (hit[s]) s_queue[qn + __popc(mask & lanemask_lt)] = entry[s];
                    qn += __popc(mask);
                }
                while (qn >= tile.num_threads()){ tile.sync(); flush(tile.num_threads()); }
            }
        }
        if (qn > 0){ tile.sync(); flush(qn); }
    } else {
        const int n_walk = flex_param.heavy_ids ? flex_param.nheavy : natom;
        for (int slot = tile.thread_rank(); slot < n_walk; slot += tile.num_threads()){
            const int i1 = flex_param.heavy_ids ? flex_param.heavy_ids[slot] : slot;
            const int t1 = flex_param.atom_types[i1];
            if (atp.is_hydrogen[t1]) continue;
            const Real lx = pose->coords[i1 * 3], ly = pose->coords[i1 * 3 + 1], lz = pose->coords[i1 * 3 + 2];
            const Real r1 = s_vdw[t1];
            const bool lig_is_donor = use_angular && atp.is_donor[t1];
            const bool lig_is_acceptor = use_angular && atp.is_acceptor[t1];
            const bool lig_hb_atom = lig_is_donor || lig_is_acceptor;
            Real fx = 0., fy = 0., fz = 0.;
            for (int i2 = 0; i2 < fix_mol.natom; i2++){
                const int t2 = fix_param.atom_types[i2];
                if (atp.is_hydrogen[t2]) continue;
                const Real rx = fix_mol.coords[i2 * 3], ry = fix_mol.coords[i2 * 3 + 1], rz = fix_mol.coords[i2 * 3 + 2];
                const Real r_vec_x = rx - lx, r_vec_y = ry - ly, r_vec_z = rz - lz;
                Real rr = r_vec_x * r_vec_x + r_vec_y * r_vec_y + r_vec_z * r_vec_z;
#ifdef UD2_PROFILE
                prof_visited++;
#endif
                if (rr >= r2_cutoff) continue;
#ifdef UD2_PROFILE
                prof_scored++;
#endif
                Real r = sqrtf(rr);
                Real d = r - (r1 + s_vdw[t2]);
                Real f_div_r = 0;
                Real e_pair = Score.eval_ef(d, t1, t2, &f_div_r, atp);
                if (lig_q != nullptr && fix_param.charges != nullptr){
                    Real f_el = 0;
                    e_pair += w_elec * Score.elec_pair(lig_q[i1] * fix_param.charges[i2], r, &f_el);
                    f_div_r += w_elec * f_el;
                }
                if (lig_hb_atom && xs_h_bond_possible(t1, t2, atp)){
                    hbond_angular_pair_warp(i1, lx, ly, lz, lig_is_donor, lig_is_acceptor,
                                            i2, rx, ry, rz, t1, t2, d, pose, fix_mol.coords, h_topo, rh, atp,
                                            e_pair, f_div_r, fx, fy, fz, aux_f);
                }
                curl_warp(e_pair, f_div_r, v_cap);
                energy += e_pair;
                if (r < EPSILON) r = EPSILON;
                f_div_r /= r;
                fx -= f_div_r * r_vec_x;
                fy -= f_div_r * r_vec_y;
                fz -= f_div_r * r_vec_z;
            }
            aux_f[i1 * 3]     += fx;
            aux_f[i1 * 3 + 1] += fy;
            aux_f[i1 * 3 + 2] += fz;
        }
    }
    }
    tile.sync();
    PROF_MARK(PROF_E_INTER);
#ifdef UD2_PROFILE
    {
        unsigned int vis_sum = cg::reduce(cg_tile(tile), prof_visited, cg::plus<unsigned int>());
        unsigned int vis_max = cg::reduce(cg_tile(tile), prof_visited, cg::greater<unsigned int>());
        unsigned int sc_sum = cg::reduce(cg_tile(tile), prof_scored, cg::plus<unsigned int>());
        if (g_prof_ptr && tile.thread_rank() == 0){
            g_prof_ptr[blockIdx.x * PROF_SLOTS + 8] += vis_sum;   // receptor atoms visited, summed over lanes
            g_prof_ptr[blockIdx.x * PROF_SLOTS + 9] += vis_max;   // max over lanes (warp-serial cost driver)
            g_prof_ptr[blockIdx.x * PROF_SLOTS + 10] += sc_sum;   // pairs within cutoff (scored)
        }
    }
    PROF_RESET();
#endif

    // 2. box penalty for every heavy atom (this lane's atom only)
    for (int i = tile.thread_rank(); i < natom; i += tile.num_threads()){
        if (atp.is_hydrogen[flex_param.atom_types[i]]) continue;
        Real cx = pose->coords[i * 3], cy = pose->coords[i * 3 + 1], cz = pose->coords[i * 3 + 2];
        Real f_tmp[3] = {0., 0., 0.};
        energy += cal_box_penalty_clamp_coord(&cx, &cy, &cz, f_tmp);
        if (fabsf(f_tmp[0]) > EPSILON) aux_f[i * 3]     += f_tmp[0];
        if (fabsf(f_tmp[1]) > EPSILON) aux_f[i * 3 + 1] += f_tmp[1];
        if (fabsf(f_tmp[2]) > EPSILON) aux_f[i * 3 + 2] += f_tmp[2];
    }
    tile.sync();

    // 2b. Inner box: wall penalty on the heavy-atom centroid (same slope as the box walls),
    //     its force shared equally by the heavy atoms (dE/dx_i = dE/dc / nheavy). aux_f holds +dE/dx.
    if (USE_INNER_BOX){
        Real sx = 0., sy = 0., sz = 0.; int nh = 0;
        for (int i = tile.thread_rank(); i < natom; i += tile.num_threads()){
            if (atp.is_hydrogen[flex_param.atom_types[i]]) continue;
            sx += pose->coords[i * 3]; sy += pose->coords[i * 3 + 1]; sz += pose->coords[i * 3 + 2]; nh += 1;
        }
        sx = cg::reduce(cg_tile(tile), sx, cg::plus<Real>());
        sy = cg::reduce(cg_tile(tile), sy, cg::plus<Real>());
        sz = cg::reduce(cg_tile(tile), sz, cg::plus<Real>());
        nh = cg::reduce(cg_tile(tile), nh, cg::plus<int>());
        if (nh > 0){
            const Real inv = Real(1) / Real(nh);
            const Real cx = sx * inv, cy = sy * inv, cz = sz * inv;
            Real pen = 0., gx = 0., gy = 0., gz = 0.;
            if (cx < INNER_X_LO){ pen += INNER_X_LO - cx; gx = -PENALTY_SLOPE; } else if (cx > INNER_X_HI){ pen += cx - INNER_X_HI; gx = PENALTY_SLOPE; }
            if (cy < INNER_Y_LO){ pen += INNER_Y_LO - cy; gy = -PENALTY_SLOPE; } else if (cy > INNER_Y_HI){ pen += cy - INNER_Y_HI; gy = PENALTY_SLOPE; }
            if (cz < INNER_Z_LO){ pen += INNER_Z_LO - cz; gz = -PENALTY_SLOPE; } else if (cz > INNER_Z_HI){ pen += cz - INNER_Z_HI; gz = PENALTY_SLOPE; }
            if (pen > 0.){
                if (tile.thread_rank() == 0) energy += pen * PENALTY_SLOPE;
                for (int i = tile.thread_rank(); i < natom; i += tile.num_threads()){
                    if (atp.is_hydrogen[flex_param.atom_types[i]]) continue;
                    aux_f[i * 3] += gx * inv; aux_f[i * 3 + 1] += gy * inv; aux_f[i * 3 + 2] += gz * inv;
                }
            }
        }
        tile.sync();
    }

    // 3. intra-molecular over the pair list (each pair once), same lockstep-walk + queue
    //    scheme as the receptor loop: the plain per-atom neighbor walk left lanes holding
    //    well-connected atoms (e.g. PROTAC atoms next to a rigid E3 domain) iterating
    //    hundreds of times while their neighbors idled. Forces of both atoms go to s_force.
    // Small ligands: the per-lane neighbor walk over the symmetric CSR list is cheaper
    // (dense enough, no atomics). Large / imbalanced ones use the compacted pair list.
    constexpr int INTRA_COMPACT_MIN_PAIRS = 4096;
    if (flex_param.npair_intra <= INTRA_COMPACT_MIN_PAIRS){
        for (int i1 = tile.thread_rank(); i1 < natom; i1 += tile.num_threads()){
            const int t1 = flex_param.atom_types[i1];
            const Real x1 = pose->coords[i1 * 3], y1 = pose->coords[i1 * 3 + 1], z1 = pose->coords[i1 * 3 + 2];
            const Real r1 = s_vdw[t1];
            Real fx = 0., fy = 0., fz = 0.;
            if constexpr (PREFETCH){
                // Software-pipelined walk: the coordinates and type of neighbour k+1 and the index of k+2
                // are loaded before pair k is scored, so their latency overlaps the pair math instead of
                // heading the next pair's dependent chain. Loads beyond the CSR row are predicated off
                // (the fallback index is a valid atom whose data is never scored); the body exists once.
                // The list pointer is read unconditionally here: read inside the predicated loads, it was
                // re-fetched from flex_param on every iteration, a dependent load ahead of the index load.
                const int* const nbr = flex_param.intra_nbr;
                const int k_end = flex_param.intra_start[i1 + 1];
                int k = flex_param.intra_start[i1];
                int i2 = k < k_end ? nbr[k] : i1;
                int i2_next = k + 1 < k_end ? nbr[k + 1] : i2;
                Real x2 = pose->coords[i2 * 3], y2 = pose->coords[i2 * 3 + 1], z2 = pose->coords[i2 * 3 + 2];
                int t2 = flex_param.atom_types[i2];
#pragma unroll 1
                for (; k < k_end; k++){
                    const int i2_after = k + 2 < k_end ? nbr[k + 2] : i2_next;
                    const Real x2_next = pose->coords[i2_next * 3];
                    const Real y2_next = pose->coords[i2_next * 3 + 1];
                    const Real z2_next = pose->coords[i2_next * 3 + 2];
                    const int t2_next = flex_param.atom_types[i2_next];
                    const Real r_vec_x = x2 - x1;
                    const Real r_vec_y = y2 - y1;
                    const Real r_vec_z = z2 - z1;
                    Real rr = r_vec_x * r_vec_x + r_vec_y * r_vec_y + r_vec_z * r_vec_z;
                    if (!(rr >= r2_cutoff)){  // same NaN behaviour as the former `if (rr >= r2_cutoff) continue;`
                        Real r = sqrtf(rr);
                        Real f_div_r = 0;
                        Real e_tmp = Score.eval_ef(r - (r1 + s_vdw[t2]), t1, t2, &f_div_r, atp);
                        e_tmp *= w_intra; f_div_r *= w_intra;
                        curl_warp(e_tmp, f_div_r, v_cap);
                        if (i1 < i2) energy += e_tmp;  // each pair is visited from both ends; count once
                        if (r < EPSILON) r = EPSILON;
                        f_div_r /= r;
                        fx -= f_div_r * r_vec_x;
                        fy -= f_div_r * r_vec_y;
                        fz -= f_div_r * r_vec_z;
                    }
                    i2 = i2_next; i2_next = i2_after; t2 = t2_next;
                    x2 = x2_next; y2 = y2_next; z2 = z2_next;
                }
            } else {
                for (int k = flex_param.intra_start[i1]; k < flex_param.intra_start[i1 + 1]; k++){
                    const int i2 = flex_param.intra_nbr[k];
                    const Real r_vec_x = pose->coords[i2 * 3] - x1;
                    const Real r_vec_y = pose->coords[i2 * 3 + 1] - y1;
                    const Real r_vec_z = pose->coords[i2 * 3 + 2] - z1;
                    Real rr = r_vec_x * r_vec_x + r_vec_y * r_vec_y + r_vec_z * r_vec_z;
                    if (rr >= r2_cutoff) continue;
                    const int t2 = flex_param.atom_types[i2];
                    Real r = sqrtf(rr);
                    Real f_div_r = 0;
                    Real e_tmp = Score.eval_ef(r - (r1 + s_vdw[t2]), t1, t2, &f_div_r, atp);
                    e_tmp *= w_intra; f_div_r *= w_intra;
                    curl_warp(e_tmp, f_div_r, v_cap);
                    if (i1 < i2) energy += e_tmp;  // each pair is visited from both ends; count once
                    if (r < EPSILON) r = EPSILON;
                    f_div_r /= r;
                    fx -= f_div_r * r_vec_x;
                    fy -= f_div_r * r_vec_y;
                    fz -= f_div_r * r_vec_z;
                }
            }
            s_force[i1 * 3]     += fx;
            s_force[i1 * 3 + 1] += fy;
            s_force[i1 * 3 + 2] += fz;
        }
    } else {
        const int npair = flex_param.npair_intra;
        const int* pairs = flex_param.pairs_intra;
        unsigned int qn = 0;

        auto score_intra = [&](int pidx){
            const int i1 = pairs[pidx * 2], i2 = pairs[pidx * 2 + 1];
            const int t1 = flex_param.atom_types[i1], t2 = flex_param.atom_types[i2];
            const Real r_vec_x = pose->coords[i2 * 3]     - pose->coords[i1 * 3];
            const Real r_vec_y = pose->coords[i2 * 3 + 1] - pose->coords[i1 * 3 + 1];
            const Real r_vec_z = pose->coords[i2 * 3 + 2] - pose->coords[i1 * 3 + 2];
            Real r = sqrtf(r_vec_x * r_vec_x + r_vec_y * r_vec_y + r_vec_z * r_vec_z);
            Real f_div_r = 0;
            Real e_tmp = Score.eval_ef(r - (s_vdw[t1] + s_vdw[t2]), t1, t2, &f_div_r, atp);
            e_tmp *= w_intra; f_div_r *= w_intra;
            curl_warp(e_tmp, f_div_r, v_cap);
            energy += e_tmp;
            if (r < EPSILON) r = EPSILON;
            f_div_r /= r;
            atomicAdd_block(s_force + i1 * 3,     -f_div_r * r_vec_x);
            atomicAdd_block(s_force + i1 * 3 + 1, -f_div_r * r_vec_y);
            atomicAdd_block(s_force + i1 * 3 + 2, -f_div_r * r_vec_z);
            atomicAdd_block(s_force + i2 * 3,      f_div_r * r_vec_x);
            atomicAdd_block(s_force + i2 * 3 + 1,  f_div_r * r_vec_y);
            atomicAdd_block(s_force + i2 * 3 + 2,  f_div_r * r_vec_z);
        };
        auto flush_intra = [&](unsigned int count){
            unsigned int e = 0;
            if (tile.thread_rank() < count) e = s_queue[tile.thread_rank()];
            tile.sync();
            if (tile.thread_rank() < count) score_intra((int)e);
            unsigned int rest = qn - count, moved = 0;
            if (tile.thread_rank() < rest) moved = s_queue[count + tile.thread_rank()];
            tile.sync();
            if (tile.thread_rank() < rest) s_queue[tile.thread_rank()] = moved;
            tile.sync();
            qn = rest;
        };

        for (int base = 0; base < npair; base += tile.num_threads()){
            const int pidx = base + tile.thread_rank();
            bool hit = false;
            if (pidx < npair){
                const int i1 = pairs[pidx * 2], i2 = pairs[pidx * 2 + 1];
                const Real dx = pose->coords[i2 * 3]     - pose->coords[i1 * 3];
                const Real dy = pose->coords[i2 * 3 + 1] - pose->coords[i1 * 3 + 1];
                const Real dz = pose->coords[i2 * 3 + 2] - pose->coords[i1 * 3 + 2];
                hit = (dx * dx + dy * dy + dz * dz) < r2_cutoff;
            }
            const unsigned int mask = tile.ballot(hit);
            if (hit){
                const unsigned int lanemask_lt = (1u << tile.thread_rank()) - 1u;
                s_queue[qn + __popc(mask & lanemask_lt)] = (unsigned int)pidx;
            }
            qn += __popc(mask);
            if (qn >= tile.num_threads()){ tile.sync(); flush_intra(tile.num_threads()); }
        }
        if (qn > 0){ tile.sync(); flush_intra(qn); }
    }
    tile.sync();
    if (force_in_shared){
        for (int i = tile.thread_rank(); i < natom * 3; i += tile.num_threads()) aux_f[i] += s_force[i];
    }
    tile.sync();

#ifndef UD2_GPU_BIAS_OFF
    // 3b. extension terms (score/terms.h): TermSet::accumulate_as runs every term's per-atom loop, the position
    //     bias being the former loop of this block, the same body score_cpu.cpp "4." runs (ext/bias/bias_term.h).
    //     Each lane owns atom i, so aux_f (+dE/dx) is written without atomics. The settings come from the
    //     __constant__ mirror g_term_settings (common.cu init_constants), not from flex_param.terms.settings, so
    //     the guard stays a constant-memory read.
    if (flex_param.terms.active_as(g_term_settings)){
        flex_param.terms.accumulate_as(g_term_settings, tile.thread_rank(), tile.num_threads(), natom,
                                       pose->coords, flex_param.atom_types, energy, aux_f);
        tile.sync();
    }
#endif

    // Sum the total energy of all threads in this warp
    energy = cg::reduce(cg_tile(tile), energy, cg::plus<Real>());
    PROF_MARK(PROF_E_INTRA);


    // 4. ligand torsional strain. Same expression as torsion_strain_energy in score/score.h, so the
    //    search and the host rescoring agree. aux_f holds +dE/dx (see the pair loop: r_vec is
    //    rec - lig and fx -= f_div_r * r_vec_x yields dE/dx_lig), so the derivative goes in as is.
    if (flex_param.n_tors_penalty > 0){
        const Real w_ts = Score.weight_torsion_strain_val();
        if (w_ts != Real(0)){
            Real e_ts = 0;
            for (int r = tile.thread_rank(); r < flex_param.n_tors_penalty; r += tile.num_threads()){
                const TorsionPenaltyRow row = flex_param.tors_penalty[r];
                Real d[12];
                const Real pen = torsion_penalty_ef(pose->coords + row.atoms[0] * 3,
                                                    pose->coords + row.atoms[1] * 3,
                                                    pose->coords + row.atoms[2] * 3,
                                                    pose->coords + row.atoms[3] * 3,
                                                    row.kind, row.tolerance, d);
                e_ts += row.weight * pen;
                const Real sc = w_ts * row.weight;
                for (int a = 0; a < 4; a++){
                    const int base = row.atoms[a] * 3;
                    atomicAdd_block(aux_f + base,     sc * d[a * 3]);
                    atomicAdd_block(aux_f + base + 1, sc * d[a * 3 + 1]);
                    atomicAdd_block(aux_f + base + 2, sc * d[a * 3 + 2]);
                }
            }
            energy += w_ts * cg::reduce(cg_tile(tile), e_ts, cg::plus<Real>());
        }
    }
    tile.sync();

    return energy;
}



template<bool GRID = false, bool PREFETCH = false, class Tile>
__device__ __forceinline__ Real cal_e_grad_warp(const Tile& tile,
                                                const FlexPose* pose, Real* out_g,
                                                const FlexTopo& flex_topo,
                                                const FixMol& fix_mol, const FlexParamVina& flex_param,
                                                const FixParamVina& fix_param,
                                                Real* aux_f, Real v_cap = CURL_V_CU){
    // Total energy
    Real energy = cal_e_f_warp<GRID, PREFETCH>(tile, pose, flex_topo, fix_mol, flex_param, fix_param, aux_f, v_cap);

    // Compute gradients on -force
    PROF_DECL;
    cal_grad_warp(tile, aux_f, flex_topo, pose, out_g);
    PROF_MARK(PROF_GRAD);
    return energy;
}


template<class Tile>
__device__ __forceinline__ void minus_mat_vec_product_warp(const Tile& tile,
                                                           Real* out_p, const Real* h, const Real* g, int dim){
    // return -Hg
    // here uses square matrix dim * dim
    Real sum = 0;
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        sum = 0;
        // same j order (one fma chain); unrolled so the loads of four steps issue together
#pragma unroll 4
        for (int j = 0; j < dim; j++){
            sum += h[tri_mat_index(i, j)] * g[j * STRIDE_G];
        }
        out_p[i * STRIDE_G] = -sum;
    }
    tile.sync();
}

template<class Tile>
__device__ __forceinline__ Real g_dot_product_warp(const Tile& tile,
                                                   const Real* a, const Real* b, int dim){
    Real tmp = 0;
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        tmp += a[i * STRIDE_G] * b[i * STRIDE_G];
    }
    tile.sync();

    return cg::reduce(cg_tile(tile), tmp, cg::plus<Real>());
}


// TORLIB (here and below): the kernel variant for launches with torsion-library ranges (TORLIB = true: wrap-aware range
// clamp, active set and bent line-search step); false compiles to the code without them.
template<bool TORLIB = false, class Tile>
__device__ __forceinline__ void apply_grad_update_dihe_warp(const Tile& tile,
                                                            FlexPose* out_x, const FlexTopo* flex_topo,
                                                            int i_tor, Real dihe_incre_raw, bool update_center = true){
    int begin = 0;
    int end = 0;
    Real tmp1[3] = {0};
    Real coord_to[3] = {0};
    Real axis_unit[3] = {0};
    Real dihe_incre = 0;
    int iat_from = -1;
    int iat_to = -1;

    if (tile.thread_rank() == 0){
        dihe_incre = dihe_incre_raw;
        dihe_incre = normalize_angle(dihe_incre_raw);
        DPrint1("\ni_tor: %d, dihe_incre_raw: %f, dihe_incre: %f\n", i_tor, dihe_incre_raw, dihe_incre);

        // apply constraint by Torsion Library
        int i_lo = flex_topo->range_inds[i_tor * 2];
        tmp1[0] = normalize_angle(out_x->dihedrals[i_tor] + dihe_incre);

        Real dihe_new = clamp_by_ranges<TORLIB>(tmp1[0],
                                                flex_topo->range_list + i_lo, flex_topo->range_inds[i_tor * 2 + 1]);
        DPrint("Ranges: %f, %f, %f, %f, %f, %f\n",
            flex_topo->range_list[0], flex_topo->range_list[1], flex_topo->range_list[2], flex_topo->range_list[3],
            flex_topo->range_list[4], flex_topo->range_list[5]);

        // update dihedral value
        dihe_incre = dihe_new - out_x->dihedrals[i_tor];
        DPrint("dihe_new[%d] after clamping: %f, dihe_now: %f, dihe_incre: %f\n", i_tor, dihe_new, out_x->dihedrals[i_tor], dihe_incre);
        out_x->dihedrals[i_tor] = dihe_new;

        // influenced atoms
        begin = flex_topo->rotated_inds[i_tor * 2];
        end = flex_topo->rotated_inds[i_tor * 2 + 1];

        // define rotation axis
        iat_from = flex_topo->axis_atoms[i_tor * 2];
        iat_to = flex_topo->axis_atoms[i_tor * 2 + 1];
        coord_to[0] = out_x->coords[iat_to * 3];
        coord_to[1] = out_x->coords[iat_to * 3 + 1];
        coord_to[2] = out_x->coords[iat_to * 3 + 2];

        axis_unit[0] = coord_to[0] - out_x->coords[iat_from * 3];
        axis_unit[1] = coord_to[1] - out_x->coords[iat_from * 3 + 1];
        axis_unit[2] = coord_to[2] - out_x->coords[iat_from * 3 + 2];

        Real l = cal_norm(axis_unit);

        if (l <= EPSILON_cu){ //todo: not allowed
            // CUDA_ERROR("l <= EPSILON_cu");
            axis_unit[0] = 1;
            axis_unit[1] = 0;
            axis_unit[2] = 0;
        }
        else{
            axis_unit[0] /= l;
            axis_unit[1] /= l;
            axis_unit[2] /= l;
        }
    }
    tile.sync();
    begin = tile.shfl(begin, 0);
    end = tile.shfl(end, 0);
    coord_to[0] = tile.shfl(coord_to[0], 0);
    coord_to[1] = tile.shfl(coord_to[1], 0);
    coord_to[2] = tile.shfl(coord_to[2], 0);
    axis_unit[0] = tile.shfl(axis_unit[0], 0);
    axis_unit[1] = tile.shfl(axis_unit[1], 0);
    axis_unit[2] = tile.shfl(axis_unit[2], 0);
    dihe_incre = tile.shfl(dihe_incre, 0);


    // parallel on updating atomic coords
    for (int j = tile.thread_rank(); j < end - begin + 1; j += tile.num_threads()){
        int i_at = flex_topo->rotated_atoms[j + begin];
        // move the rotation origin to the axis
        tmp1[0] = out_x->coords[i_at * 3] - coord_to[0];
        tmp1[1] = out_x->coords[i_at * 3 + 1] - coord_to[1];
        tmp1[2] = out_x->coords[i_at * 3 + 2] - coord_to[2];
        // Rotate atom by Rodrigues' formula
        rotate_vec_by_rodrigues(tmp1, axis_unit, dihe_incre);
        // move back to the original position
        out_x->coords[i_at * 3] = tmp1[0] + coord_to[0];
        out_x->coords[i_at * 3 + 1] = tmp1[1] + coord_to[1];
        out_x->coords[i_at * 3 + 2] = tmp1[2] + coord_to[2];
    }
    tile.sync();

    // update center
    if (!update_center) return;
    Real center_new[3] = {0};
    for (int j = tile.thread_rank(); j < flex_topo->natom; j += tile.num_threads()){
        center_new[0] += out_x->coords[j * 3];
        center_new[1] += out_x->coords[j * 3 + 1];
        center_new[2] += out_x->coords[j * 3 + 2];
    }
    tile.sync();

    center_new[0] = cg::reduce(cg_tile(tile), center_new[0], cg::plus<Real>());
    center_new[1] = cg::reduce(cg_tile(tile), center_new[1], cg::plus<Real>());
    center_new[2] = cg::reduce(cg_tile(tile), center_new[2], cg::plus<Real>());

    if (tile.thread_rank() == 0){
        out_x->center[0] = center_new[0] / flex_topo->natom;
        out_x->center[1] = center_new[1] / flex_topo->natom;
        out_x->center[2] = center_new[2] / flex_topo->natom;
    }
    tile.sync();
}


/**
 * @brief Rotate the atoms of torsion i_tor by dihe_incre about its current axis: the part of apply_grad_update_dihe_warp
 * after its lane-0 block, except that every lane reads the axis from the coordinates itself (same expressions) instead
 * of receiving it from lane 0.
 */
template<class Tile>
__device__ __forceinline__ void rotate_torsion_warp(const Tile& tile, FlexPose* out_x,
                                                    const FlexTopo& flex_topo, int i_tor, Real dihe_incre){
    const int begin = flex_topo.rotated_inds[i_tor * 2];
    const int end = flex_topo.rotated_inds[i_tor * 2 + 1];
    const int iat_from = flex_topo.axis_atoms[i_tor * 2];
    const int iat_to = flex_topo.axis_atoms[i_tor * 2 + 1];
    Real tmp1[3] = {0};
    Real coord_to[3] = {0};
    Real axis_unit[3] = {0};
    coord_to[0] = out_x->coords[iat_to * 3];
    coord_to[1] = out_x->coords[iat_to * 3 + 1];
    coord_to[2] = out_x->coords[iat_to * 3 + 2];

    axis_unit[0] = coord_to[0] - out_x->coords[iat_from * 3];
    axis_unit[1] = coord_to[1] - out_x->coords[iat_from * 3 + 1];
    axis_unit[2] = coord_to[2] - out_x->coords[iat_from * 3 + 2];

    Real l = cal_norm(axis_unit);

    if (l <= EPSILON_cu){ //todo: not allowed
        axis_unit[0] = 1;
        axis_unit[1] = 0;
        axis_unit[2] = 0;
    }
    else{
        axis_unit[0] /= l;
        axis_unit[1] /= l;
        axis_unit[2] /= l;
    }

    // parallel on updating atomic coords
    for (int j = tile.thread_rank(); j < end - begin + 1; j += tile.num_threads()){
        int i_at = flex_topo.rotated_atoms[j + begin];
        // move the rotation origin to the axis
        tmp1[0] = out_x->coords[i_at * 3] - coord_to[0];
        tmp1[1] = out_x->coords[i_at * 3 + 1] - coord_to[1];
        tmp1[2] = out_x->coords[i_at * 3 + 2] - coord_to[2];
        // Rotate atom by Rodrigues' formula
        rotate_vec_by_rodrigues(tmp1, axis_unit, dihe_incre);
        // move back to the original position
        out_x->coords[i_at * 3] = tmp1[0] + coord_to[0];
        out_x->coords[i_at * 3 + 1] = tmp1[1] + coord_to[1];
        out_x->coords[i_at * 3 + 2] = tmp1[2] + coord_to[2];
    }
    tile.sync();
}


/**
 * @brief A torsion increment shortened to at most lim in magnitude, sign kept; lim = TOR_ROOM_FREE leaves it alone
 */
__device__ __forceinline__ Real clip_torsion_step(Real step, Real lim){
    return lim != TOR_ROOM_FREE && fabs(step) > lim ? copysign(lim, step) : step;
}


/**
 * @brief Apply gradient to update the pose, including center, orientation, and dihedrals
 * @param tile Thread block tile
 * @param out_x Pose to be updated
 * @param g Gradient
 * @param flex_topo Flex topology
 * @param alpha Step length
 * @param scratch ntorsion Reals, STRIDE_G apart, free for this call (a DOF vector slot of the BFGS state)
 * @param lim TORLIB only: nullptr, or per-torsion limits of the increment (a DOF vector, torsion components read),
 * applied by clip_torsion_step
 */
template<bool TORLIB = false, class Tile>
__device__ __forceinline__ void apply_grad_update_pose_warp(const Tile& tile,
                                                            FlexPose* out_x, const Real* g,
                                                            const FlexTopo& flex_topo,
                                                            Real alpha, Real* scratch, const Real* lim = nullptr){
    // -------------- torsion increment --------------
    // apply_grad_update_dihe_warp for torsions 0 .. ntorsion - 1 in turn, then the center update it does after the last
    // one, regrouped: the clamped new dihedral of a torsion depends only on its own value and increment, so the lanes
    // compute it for all torsions at once (lane-0 expressions, increments parked in scratch); the rotations stay
    // serial, since each one moves the axes of the torsions after it.
    const int ntor = flex_topo.ntorsion;
    for (int i_tor = tile.thread_rank(); i_tor < ntor; i_tor += tile.num_threads()){
        DPrint1("dihedral_g[%d] = %f\n", i_tor, g[(dof_g + i_tor) * STRIDE_G]);
        const Real step = g[(dof_g + i_tor) * STRIDE_G] * alpha;
        const Real dihe_incre = normalize_angle(TORLIB && lim != nullptr
                                                ? clip_torsion_step(step, lim[(dof_g + i_tor) * STRIDE_G]) : step);
        // apply constraint by Torsion Library
        const int i_lo = flex_topo.range_inds[i_tor * 2];
        const Real dihe_old = out_x->dihedrals[i_tor];
        const Real dihe_new = clamp_by_ranges<TORLIB>(normalize_angle(dihe_old + dihe_incre),
                                                      flex_topo.range_list + i_lo, flex_topo.range_inds[i_tor * 2 + 1]);
        scratch[i_tor * STRIDE_G] = dihe_new - dihe_old;
        out_x->dihedrals[i_tor] = dihe_new;
    }
    tile.sync();
    for (int i_tor = 0; i_tor < ntor; i_tor++){
        rotate_torsion_warp(tile, out_x, flex_topo, i_tor, scratch[i_tor * STRIDE_G]);
    }
    if (ntor > 0){
        // update center
        Real center_new[3] = {0};
        for (int j = tile.thread_rank(); j < flex_topo.natom; j += tile.num_threads()){
            center_new[0] += out_x->coords[j * 3];
            center_new[1] += out_x->coords[j * 3 + 1];
            center_new[2] += out_x->coords[j * 3 + 2];
        }
        tile.sync();

        center_new[0] = cg::reduce(cg_tile(tile), center_new[0], cg::plus<Real>());
        center_new[1] = cg::reduce(cg_tile(tile), center_new[1], cg::plus<Real>());
        center_new[2] = cg::reduce(cg_tile(tile), center_new[2], cg::plus<Real>());

        if (tile.thread_rank() == 0){
            out_x->center[0] = center_new[0] / flex_topo.natom;
            out_x->center[1] = center_new[1] / flex_topo.natom;
            out_x->center[2] = center_new[2] / flex_topo.natom;
        }
        tile.sync();
    }
    tile.sync();


    if (!FLAG_CONSTRAINT_DOCK){
        Real q[4] = {0}, tmp1[3] = {0}, tmp2[3] = {0};

        // -------------- orientation & center increment --------------
        // first, save the original center
        tmp1[0] = out_x->center[0];
        tmp1[1] = out_x->center[1];
        tmp1[2] = out_x->center[2];

        if (tile.thread_rank() == 0){
            // then update center
            out_x->center[0] = clamp_to_center(tmp1[0] + alpha * g[0], BOX_X_HI, BOX_X_LO);
            out_x->center[1] = clamp_to_center(tmp1[1] + alpha * g[STRIDE_G], BOX_Y_HI, BOX_Y_LO);
            out_x->center[2] = clamp_to_center(tmp1[2] + alpha * g[2 * STRIDE_G], BOX_Z_HI, BOX_Z_LO);

            // and update orientation
            tmp2[0] = g[3 * STRIDE_G] * alpha;
            tmp2[1] = g[4 * STRIDE_G] * alpha;
            tmp2[2] = g[5 * STRIDE_G] * alpha;
            rotvec_to_quaternion(q, tmp2);
            DPrint1("RotVec: %f, %f, %f, q: %f, %f, %f, %f\n", tmp2[0], tmp2[1], tmp2[2], q[0], q[1], q[2], q[3]);
            out_x->rot_vec[0] = tmp2[0]; // record rotvec fixme
            out_x->rot_vec[1] = tmp2[1];
            out_x->rot_vec[2] = tmp2[2];
        }
        tile.sync();
        q[0] = tile.shfl(q[0], 0);
        q[1] = tile.shfl(q[1], 0);
        q[2] = tile.shfl(q[2], 0);
        q[3] = tile.shfl(q[3], 0);

        // Finally, add center_g to and rotate by orientation_g coords of all atoms
        for (int i_at = tile.thread_rank(); i_at < flex_topo.natom; i_at += tile.num_threads()){
            // locate rotation origin on the original center
            tmp2[0] = out_x->coords[i_at * 3] - tmp1[0];
            tmp2[1] = out_x->coords[i_at * 3 + 1] - tmp1[1];
            tmp2[2] = out_x->coords[i_at * 3 + 2] - tmp1[2];

            rotate_vec_by_quaternion(tmp2, q);

            // move to the new center
            out_x->coords[i_at * 3] = tmp2[0] + out_x->center[0];
            out_x->coords[i_at * 3 + 1] = tmp2[1] + out_x->center[1];
            out_x->coords[i_at * 3 + 2] = tmp2[2] + out_x->center[2];
        }
        tile.sync();
    }
}




constexpr Real TOR_PINNED_ROOM = 1e-3; // radian; a torsion with less room than this towards -g is held at its boundary


#ifdef UD2_PROFILE
/**
 * @brief Profile only: count the torsions that sit at a boundary of their allowed set with the step pointing out
 * (PROF_PINNED_TORSIONS) and those outside every range (PROF_OUTSIDE_TORSIONS), at the start of a line search
 * @return The pinned count of the warp (same on all lanes)
 */
template<class Tile>
__device__ __forceinline__ int prof_torlib_ls_start(const Tile& tile, const FlexPose* x, const Real* p,
                                                    const FlexTopo& flex_topo){
    int n_pinned = 0, n_outside = 0;
    for (int i_tor = tile.thread_rank(); i_tor < flex_topo.ntorsion; i_tor += tile.num_threads()){
        const Real p_i = p[(dof_g + i_tor) * STRIDE_G];
        const Real room = torsion_room(x->dihedrals[i_tor], flex_topo.range_list + flex_topo.range_inds[i_tor * 2],
                                       flex_topo.range_inds[i_tor * 2 + 1], p_i >= 0 ? 1 : -1);
        n_outside += room < 0;
        n_pinned += room >= 0 && room < TOR_PINNED_ROOM && p_i != 0;
    }
    n_pinned = cg::reduce(cg_tile(tile), n_pinned, cg::plus<int>());
    n_outside = cg::reduce(cg_tile(tile), n_outside, cg::plus<int>());
    PROF_ADD(PROF_PINNED_TORSIONS, n_pinned);
    PROF_ADD(PROF_OUTSIDE_TORSIONS, n_outside);
    return n_pinned;
}

/**
 * @brief Profile only: count the torsions the range clamp moved in one line-search trial (PROF_SNAPS), and those
 * snapped to a boundary farther round the circle than the nearest real one (PROF_SNAPS_FAR)
 * @param x Pose before the trial
 * @param x_new Pose after apply_grad_update_pose_warp(x, p, alpha, lim)
 * @return The snap count of the warp (same on all lanes)
 */
template<class Tile>
__device__ __forceinline__ int prof_torlib_ls_trial(const Tile& tile, const FlexPose* x, const FlexPose* x_new,
                                                    const Real* p, const FlexTopo& flex_topo, Real alpha,
                                                    const Real* lim){
    int n_snap = 0, n_far = 0;
    for (int i_tor = tile.thread_rank(); i_tor < flex_topo.ntorsion; i_tor += tile.num_threads()){
        const Real* ranges = flex_topo.range_list + flex_topo.range_inds[i_tor * 2];
        const int nrange = flex_topo.range_inds[i_tor * 2 + 1];
        // the unclamped result, as apply_grad_update_pose_warp computes it
        const Real step = p[(dof_g + i_tor) * STRIDE_G] * alpha;
        const Real target = normalize_angle(x->dihedrals[i_tor] + normalize_angle(
            lim == nullptr ? step : clip_torsion_step(step, lim[(dof_g + i_tor) * STRIDE_G])));
        const Real d_got = circular_distance(target, x_new->dihedrals[i_tor]);
        if (d_got <= TOR_TOL){
            continue;
        }
        n_snap++;
        // A real boundary is an endpoint the allowed set does not continue past: not a +-pi seam join, not a touch.
        Real d_nearest = TOR_ROOM_FREE;
        for (int j = 0; j < 2 * nrange; j++){
            if (torsion_room(ranges[j], ranges, nrange, j % 2 ? 1 : -1) > TOR_TOL){
                continue;
            }
            const Real d = circular_distance(target, ranges[j]);
            if (d < d_nearest){
                d_nearest = d;
            }
        }
        n_far += d_got > d_nearest + TOR_TOL;
    }
    n_snap = cg::reduce(cg_tile(tile), n_snap, cg::plus<int>());
    n_far = cg::reduce(cg_tile(tile), n_far, cg::plus<int>());
    PROF_ADD(PROF_SNAPS, n_snap);
    PROF_ADD(PROF_SNAPS_FAR, n_far);
    return n_snap;
}
#endif


/**
 * @brief Whether any torsion is limited by the torsion library, i.e. its allowed set is not the full circle
 * @return Same on all lanes; false with the torsion library off
 */
template<class Tile>
__device__ __forceinline__ bool has_limited_torsion_warp(const Tile& tile, const FlexPose* x, const FlexTopo& flex_topo){
    int n_limited = 0;
    for (int i_tor = tile.thread_rank(); i_tor < flex_topo.ntorsion; i_tor += tile.num_threads()){
        n_limited += torsion_room(x->dihedrals[i_tor], flex_topo.range_list + flex_topo.range_inds[i_tor * 2],
                                  flex_topo.range_inds[i_tor * 2 + 1], 1) != TOR_ROOM_FREE;
    }
    return cg::reduce(cg_tile(tile), n_limited, cg::plus<int>()) > 0;
}

/**
 * @brief Active set A of the torsion ranges: the limited torsions at a boundary of their allowed set with -g pointing
 * out of it (room towards -g below TOR_PINNED_ROOM)
 * @param out_g_free g with the components of A zeroed
 * @return |A| (same on all lanes)
 */
template<class Tile>
__device__ __forceinline__ int active_torsions_warp(const Tile& tile, const FlexPose* x, const Real* g,
                                                    Real* out_g_free, const FlexTopo& flex_topo, int dim_g){
    int n_active = 0;
    for (int i = tile.thread_rank(); i < dim_g; i += tile.num_threads()){
        Real g_i = g[i * STRIDE_G];
        if (i >= dof_g && g_i != 0){
            const int i_tor = i - dof_g;
            const Real room = torsion_room(x->dihedrals[i_tor], flex_topo.range_list + flex_topo.range_inds[i_tor * 2],
                                           flex_topo.range_inds[i_tor * 2 + 1], g_i < 0 ? 1 : -1);
            if (room >= 0 && room < TOR_PINNED_ROOM){
                g_i = 0;
                n_active++;
            }
        }
        out_g_free[i * STRIDE_G] = g_i;
    }
    tile.sync();
    return cg::reduce(cg_tile(tile), n_active, cg::plus<int>());
}

/**
 * @brief Projected quasi-Newton direction for a non-empty active set A (Bertsekas): p = -H g_free with p_A = 0, which
 * descends while H is positive definite (p.g = -g_F H_FF g_F), else the projected steepest descent p = -g_free
 * @param g_free g with g_A zeroed (active_torsions_warp)
 * @return false when no descent direction is left: g_free is zero, the pose is at a minimum within the torsion ranges
 * (same on all lanes)
 */
template<class Tile>
__device__ __forceinline__ bool projected_direction_warp(const Tile& tile, Real* out_p, const Real* h, const Real* g,
                                                         const Real* g_free, int dim_g){
    minus_mat_vec_product_warp(tile, out_p, h, g_free, dim_g);
    for (int i = tile.thread_rank(); i < dim_g; i += tile.num_threads()){
        if (g_free[i * STRIDE_G] == 0 && g[i * STRIDE_G] != 0){ // i in A
            out_p[i * STRIDE_G] = 0;
        }
    }
    tile.sync();
    if (g_dot_product_warp(tile, out_p, g, dim_g) < 0){
        return true;
    }
    for (int i = tile.thread_rank(); i < dim_g; i += tile.num_threads()){
        out_p[i * STRIDE_G] = -g_free[i * STRIDE_G];
    }
    tile.sync();
    const bool descent = g_dot_product_warp(tile, out_p, g, dim_g) < 0;
    PROF_ADD(PROF_STEPS_STEEPEST_GUARD, descent);
    PROF_ADD(PROF_STEPS_CONSTRAINED_STOP, !descent);
    return descent;
}

/**
 * @brief Line-search step limits of the torsions (bent path): a limited torsion turns by at most its room in the
 * direction of p, and by at most π, past which normalize_angle would turn it the other way round
 * @param out_lim Torsion components written: the limit, TOR_ROOM_FREE (not clipped) for FREE and OUTSIDE torsions
 */
template<class Tile>
__device__ __forceinline__ void torsion_step_limits_warp(const Tile& tile, const FlexPose* x, const Real* p,
                                                         const FlexTopo& flex_topo, Real* out_lim){
    for (int i_tor = tile.thread_rank(); i_tor < flex_topo.ntorsion; i_tor += tile.num_threads()){
        const Real room = torsion_room(x->dihedrals[i_tor], flex_topo.range_list + flex_topo.range_inds[i_tor * 2],
                                       flex_topo.range_inds[i_tor * 2 + 1], p[(dof_g + i_tor) * STRIDE_G] > 0 ? 1 : -1);
        out_lim[(dof_g + i_tor) * STRIDE_G] = room == TOR_ROOM_FREE || room == TOR_ROOM_OUTSIDE ? TOR_ROOM_FREE
                                                                                               : fmin(room, PI);
    }
    tile.sync();
}

/**
 * @brief The torsions clip_torsion_step shortens in the trial alpha * p, and the predicted decrease they do not take,
 * sum over them of g_i (s_i - alpha p_i) with s_i the clipped increment, so that alpha p.g + that = g.s
 * @param out_n_clip Number of clipped torsions (same on all lanes)
 * @return The sum (same on all lanes)
 */
template<class Tile>
__device__ __forceinline__ Real clipped_decrease_warp(const Tile& tile, const Real* p, const Real* g, const Real* lim,
                                                      int ntorsion, Real alpha, int* out_n_clip){
    Real d = 0;
    int n_clip = 0;
    for (int i_tor = tile.thread_rank(); i_tor < ntorsion; i_tor += tile.num_threads()){
        const int k = (dof_g + i_tor) * STRIDE_G;
        const Real step = p[k] * alpha;
        const Real s = clip_torsion_step(step, lim[k]);
        if (s != step){
            n_clip++;
            d += g[k] * (s - step);
        }
    }
    *out_n_clip = cg::reduce(cg_tile(tile), n_clip, cg::plus<int>());
    return cg::reduce(cg_tile(tile), d, cg::plus<Real>());
}

/**
 * @brief Backtracking line search along p from x
 * @param p Search direction; with aux_lim (TORLIB), the components of the torsions clipped in the kept trial are set to
 * the increment taken / the alpha of that trial, so that the BFGS update sees the step taken
 * @param aux_lim TORLIB only: nullptr, or a DOF vector free for this call: the limited torsions then stop at the
 * boundary of their allowed set (torsion_step_limits_warp) while the other DOFs take the full alpha * p
 */
template<bool GRID = false, bool PREFETCH = false, bool TORLIB = false, class Tile>
__device__ __forceinline__ void line_search_warp(const Tile& tile,
                                                 const FixMol& fix_mol, const FixParamVina& fix_param,
                                                 const FlexParamVina& flex_param, const FlexPose* x,
                                                 const Real* g, const FlexTopo& flex_topo,
                                                 Real* p,
                                                 Real* aux_f,
                                                 FlexPose* out_x_new, Real* out_g_new, Real* aux_scratch,
                                                 const Real e0, int dim_g, int dim_x, Real* out_e, Real* out_alpha,
                                                 Real v_cap = CURL_V_CU, Real* aux_lim = nullptr){
    Real alpha = 1.0; // step length, which is the target of BLS
    Real e_new = 0.; // energy of tried step
    Real alpha_kept = alpha; // TORLIB: alpha of the kept trial (the returned alpha is halved once more when every trial fails)
    int n_clip = 0; // TORLIB: torsions clipped in the latest trial

    const Real pg = g_dot_product_warp(tile, p, g, dim_g);
#ifdef UD2_PROFILE
    int prof_pinned = 0, prof_clamped = 0, prof_clamped_fail = 0, prof_free_fail = 0;
    int prof_clipped_fail = 0, prof_unclipped_fail = 0, prof_nondescent = 0;
    bool prof_had_nondescent = false;
#endif
    if constexpr (TORLIB){
        if (aux_lim != nullptr){
            torsion_step_limits_warp(tile, x, p, flex_topo, aux_lim);
        }
#ifdef UD2_PROFILE
        prof_pinned = prof_torlib_ls_start(tile, x, p, flex_topo);
#endif
    }

    int trial = 0;
    for (; trial < LINE_SEARCH_STEPS; trial++){
        PROF_DECL; PROF_COUNT(PROF_N_LS_TRIAL);

        duplicate_pose_warp(tile, out_x_new, x, dim_x, flex_topo.natom); // x_new = x
        // DPrint1("coord_0: %f, %f, %f\n\n", out_x_new->coords[0], out_x_new->coords[1], out_x_new->coords[2]);

        // apply alpha * gradient, get new x
        apply_grad_update_pose_warp<TORLIB>(tile, out_x_new, p, flex_topo, alpha, aux_scratch, aux_lim); // apply gradient increment
        Real d_clip = 0;
        if constexpr (TORLIB){
            if (aux_lim != nullptr){
                d_clip = clipped_decrease_warp(tile, p, g, aux_lim, flex_topo.ntorsion, alpha, &n_clip);
            }
        }
        PROF_MARK(PROF_LS_UPDATE);
#ifdef UD2_PROFILE
        int prof_snaps = 0;
        if constexpr (TORLIB){
            prof_snaps = prof_torlib_ls_trial(tile, x, out_x_new, p, flex_topo, alpha, aux_lim);
            PROF_ADD(PROF_CLIPPED_TRIALS, n_clip > 0);
            PROF_ADD(PROF_CLIPPED_TORSIONS, n_clip);
        }
#endif

        // Energy and forces only: the force-to-torque tree runs once, after the loop, for the trial that is kept.
        e_new = cal_e_f_warp<GRID, PREFETCH>(tile, out_x_new, flex_topo, fix_mol, flex_param, fix_param, aux_f, v_cap);
        bool accept;
        if constexpr (TORLIB){
            // Armijo on the step taken: its predicted decrease is g.s = alpha p.g + d_clip
            accept = n_clip == 0 ? e_new - e0 < LINE_SEARCH_C0 * alpha * pg
                                 : e_new - e0 < LINE_SEARCH_C0 * (alpha * pg + d_clip);
            alpha_kept = alpha;
#ifdef UD2_PROFILE
            prof_clamped += prof_snaps > 0;
            prof_clamped_fail += prof_snaps > 0 && !accept;
            prof_free_fail += prof_snaps == 0 && !accept;
            prof_clipped_fail += n_clip > 0 && !accept;
            prof_unclipped_fail += n_clip == 0 && !accept;
            const bool nondescent = n_clip > 0 && alpha * pg + d_clip >= 0; // the step taken predicts no decrease
            prof_nondescent += nondescent;
            prof_had_nondescent = prof_had_nondescent || nondescent;
#endif
        }
        else{
            accept = e_new - e0 < LINE_SEARCH_C0 * alpha * pg;
        }
        // DPrint1("\n[LINE SEARCH] coord0_new: %f, %f, %f coord10_new: %f, %f, %f e_new: %f, alpha: %f, p: %f, %f, %f, %f, %f, %f, pg: %f \n",
        //     out_x_new->coords[0], out_x_new->coords[1], out_x_new->coords[2],
        //     out_x_new->coords[30], out_x_new->coords[31], out_x_new->coords[32],
        //     e_new, alpha,
        //     p->center_g[0], p->center_g[1], p->center_g[2],
        //     p->orientation_g[0], p->orientation_g[1], p->orientation_g[2], pg);
        DPrint1("Alpha = %f, e_new = %f\n", alpha, e_new);

        if (accept){
            // DPrint1("\nLine search SUCCEED!\n", 1);
            break;
        }
        //todo: Wolfe condition

        alpha *= LINE_SEARCH_MULTIPLIER; // lower step length
    }
    // Gradient of the kept trial (the accepted one, or the last one when every trial was rejected). cal_grad_warp
    // reads only out_x_new and aux_f, which still hold that trial, and overwrites every component of out_g_new,
    // so the result is the same as computing it inside every trial.
    {
        PROF_DECL;
        cal_grad_warp(tile, aux_f, flex_topo, out_x_new, out_g_new);
        PROF_MARK(PROF_GRAD);
    }
    tile.sync();
    if constexpr (TORLIB){
        if (n_clip > 0){
            for (int i_tor = tile.thread_rank(); i_tor < flex_topo.ntorsion; i_tor += tile.num_threads()){
                const int k = (dof_g + i_tor) * STRIDE_G;
                const Real step = p[k] * alpha_kept;
                const Real s = clip_torsion_step(step, aux_lim[k]);
                if (s != step){
                    p[k] = s / alpha_kept;
                }
            }
            tile.sync();
        }
#ifdef UD2_PROFILE
        const int n_trials = trial < LINE_SEARCH_STEPS ? trial + 1 : LINE_SEARCH_STEPS;
        PROF_ADD(PROF_LS_CLAMPED_TRIALS, prof_clamped);
        PROF_ADD(PROF_LS_CLAMPED_TRIALS_FAIL, prof_clamped_fail);
        PROF_ADD(PROF_LS_FREE_TRIALS_FAIL, prof_free_fail);
        PROF_ADD(PROF_LS_EXHAUSTED, trial == LINE_SEARCH_STEPS);
        PROF_ADD(PROF_BFGS_STEPS_ANY_CLAMP, prof_clamped > 0);
        PROF_ADD(PROF_TRIALS_IN_ANY_CLAMP_STEPS, prof_clamped > 0 ? n_trials : 0);
        PROF_ADD(PROF_BFGS_STEPS_PINNED, prof_pinned > 0);
        PROF_ADD(PROF_TRIALS_IN_PINNED_STEPS, prof_pinned > 0 ? n_trials : 0);
        PROF_ADD(PROF_LS_CLIPPED_TRIALS_FAIL, prof_clipped_fail);
        PROF_ADD(PROF_LS_UNCLIPPED_TRIALS_FAIL, prof_unclipped_fail);
        PROF_ADD(PROF_CLIPPED_NONDESCENT, prof_nondescent);
        PROF_ADD(PROF_LS_EXHAUSTED_WITH_NONDESCENT, trial == LINE_SEARCH_STEPS && prof_had_nondescent);
#endif
    }

    // fixme: WHY? this version will lead to active lock (100% occupancy of GPU, never stop)
    // fixme: requiring fact-checking @25-03-20
    // if (tile.thread_rank() == 0){ // check why the active lock appears
    //     *out_e = e_new;
    //     *out_alpha = alpha;
    // }


    // this version works, no lock
    *out_e = e_new; //
    *out_alpha = alpha;

    tile.sync();
}


/**
 * @brief Update Hessian matrix for BFGS
 * @param out_h Hessian matrix, triangular matrix
 * @param dim Dimension of Hessian & gradient
 * @param p vector p
 * @param y vector y
 * @param aux_minus_hy Auxiliary vector -Hy
 * @param alpha Step length
 */
template<class Tile>
__forceinline__ __device__ void bfgs_update_hessian_warp(const Tile& tile,
                                                         Real* out_h, int dim, const Real* p, const Real* y,
                                                         Real* aux_minus_hy, const Real alpha){
    Real yp = 0, yhy = 0;
    yp = g_dot_product_warp(tile, y, p, dim);
    DPrint1("yp is %f\n", yp);
    if (alpha * yp < EPSILON_cu){
        return;
    }

    // minus_hy = -hy (dim*1 vector)
    minus_mat_vec_product_warp(tile, aux_minus_hy, out_h, y, dim);

    // y^THy
    yhy = -g_dot_product_warp(tile, y, aux_minus_hy, dim);
    Real r = 1 / (alpha * yp); //1 / (s^T * y) , where s = alpha * p

    // tmp = alpha r (mhy_i p_j + mhy_j p_i) + alpha^2 (r^2 yhy + r) p_i p_j   (s * s == alpha * alpha * p * p)
    // The roundings are pinned to what the FlexPoseGradient accessors of 660c102 compiled to (NVVM fused the second
    // product of the inner sum there, but fuses the first one when the operands are plain loads):
    // fma(alpha r, fma(mhy_j, p_i, mhy_i p_j), ((alpha^2 (r^2 yhy + r)) p_i) p_j).
    // Every element is updated once and independently, so the lanes take the stored elements idx = i + j (j + 1) / 2
    // (i <= j) round-robin instead of one row each (row i used to run dim - i serial iterations on lane i).
    const Real c1 = alpha * r;
    const Real c2 = alpha * alpha * (r * r * yhy + r);
    const int n = dim * (dim + 1) / 2;
    int idx = tile.thread_rank(), i = idx, j = 0;
    while (i > j){ i -= j + 1; j++; }   // (i, j) of element idx
    for (; idx < n; idx += tile.num_threads()){
        const Real mhy_i = aux_minus_hy[i * STRIDE_G], mhy_j = aux_minus_hy[j * STRIDE_G];
        const Real p_i = p[i * STRIDE_G], p_j = p[j * STRIDE_G];
        const Real tmp = __fmaf_rn(c1, __fmaf_rn(mhy_j, p_i, __fmul_rn(mhy_i, p_j)), __fmul_rn(__fmul_rn(c2, p_i), p_j));
        out_h[idx] += tmp;
        i += tile.num_threads();
        while (i > j){ i -= j + 1; j++; }
    }
    tile.sync();
}

SCOPE_INLINE void print_uptri_mat(Real* mat, int dim){
    for (int i = 0; i < dim; i++){
        for (int j = 0; j < dim; j++){
            if (j < i){
                printf("      ");
            }
            else{
                printf("%.3f ", mat[uptri_mat_index(i, j)]);
            }
        }
        printf("\n");
    }
}

SCOPE_INLINE void print_g(const Real* g, int dim){
    DPrint("\nG: Center: ", 1);
    for (int i = 0; i < 3; i ++){
        DPrint("%f ", g[i * STRIDE_G]);
    }
    DPrint("Orientation: ", 1);
    for (int i = 3; i < dof_g; i ++){
        DPrint("%f ", g[i * STRIDE_G]);
    }
    DPrint("Dihedrals: ", 1);
    for (int i = dof_g; i < dim; i ++){
        DPrint("%f ", g[i * STRIDE_G]);
    }
    DPrint("\n", 1);

}


template<bool GRID = false, bool PREFETCH = false, bool TORLIB = false, class Tile>
__forceinline__ __device__ void bfgs_warp(const Tile& tile,
                                          FlexPose* out_x, const FlexTopo& flex_topo, const FixMol& fix_mol,
                                          const FlexParamVina& flex_param, const FixParamVina& fix_param,
                                          FlexPose* aux_x_new, FlexPose* aux_x_ori,
                                          Real* aux_v, Real* aux_h, FlexForce* aux_f, int max_steps,
                                          Real v_cap = CURL_V_CU){
    int dim_g = dof_g + flex_topo.ntorsion;
    int dim_x = dof_x + flex_topo.ntorsion; // center, orientation, torsion
    // the six DOF vectors of the flat state (BV_* slots, stride STRIDE_G)
    Real* aux_g = aux_v + BV_G;
    Real* aux_g_new = aux_v + BV_G_NEW;
    Real* aux_g_ori = aux_v + BV_G_ORI;
    Real* aux_p = aux_v + BV_P;
    Real* aux_y = aux_v + BV_Y;
    Real* aux_minus_hy = aux_v + BV_MHY;

    Real E_ori = cal_e_grad_warp<GRID, PREFETCH>(tile, out_x, aux_g, flex_topo, fix_mol, flex_param, fix_param, aux_f->f, v_cap);
    PROF_DECL;

    // Record initial energy and gradient. E_ori is energy, aux_g is set as current gradient
    // alpha is step length
    Real E = E_ori, E1 = 0, alpha = 0;
    DPrint1("\nBefore BFGS, Energy is %f\n", E_ori);
    // print_g(aux_g, dim);


    // Initialize hessian as Unit Matrix
    init_tri_mat_warp(tile, aux_h, dim_g, 0); // set zero
    set_tri_mat_diagonal_warp(tile, aux_h, dim_g, 1); // set diagonal to 1

    // Initialize aux_g_new
    duplicate_grad_warp(tile, aux_g_new, aux_g, dim_g);
    // Initialize aux_pose_new
    duplicate_pose_warp(tile, aux_x_new, out_x, dim_x, flex_topo.natom);

    // Record gradient before optimization: aux_g_ori = aux_g
    duplicate_grad_warp(tile, aux_g_ori, aux_g, dim_g);
    // aux_pose_ori := copy of out_pose
    duplicate_pose_warp(tile, aux_x_ori, out_x, dim_x, flex_topo.natom);

    // aux_p := copy of aux_g, just for initialization
    duplicate_grad_warp(tile, aux_p, aux_g, dim_g);

    // With the torsion library (TORLIB), the direction is projected on the active set of the torsion ranges and the
    // line search bends the step at the range boundaries; aux_minus_hy, free until bfgs_update_hessian_warp, holds g_free
    // and then the step limits. Without limited torsions (a ligand of a TORLIB launch without any) the search is the
    // plain BFGS one.
    const bool limited = TORLIB && has_limited_torsion_warp(tile, out_x, flex_topo);

    for (int step = 0; step < max_steps; step++){
        PROF_COUNT(PROF_N_BFGS_STEP);
        bool stop = false;
        if constexpr (TORLIB){
            const int n_active = limited ? active_torsions_warp(tile, out_x, aux_g, aux_minus_hy, flex_topo, dim_g) : 0;
            if (n_active == 0){
                minus_mat_vec_product_warp(tile, aux_p, aux_h, aux_g, dim_g);
            }
            else{
                stop = !projected_direction_warp(tile, aux_p, aux_h, aux_g, aux_minus_hy, dim_g);
            }
            PROF_ADD(PROF_ACTIVE_TORSIONS, n_active);
            PROF_ADD(PROF_STEPS_WITH_ACTIVE, n_active > 0);
        }
        else{
            // compute line search direction aux_p = -Hg (dim*1 vector)
            minus_mat_vec_product_warp(tile, aux_p, aux_h, aux_g, dim_g);
        }
        PROF_MARK(PROF_BFGS_ALGEBRA);
        if (stop){
            break; // at a minimum within the torsion ranges: keep the current pose, energy and gradient
        }

        // find the best alpha, and updates x_new & aux_g_new. f1 is the new energy
        line_search_warp<GRID, PREFETCH, TORLIB>(tile, fix_mol, fix_param, flex_param, out_x, aux_g, flex_topo,
                                                 aux_p, aux_f->f, aux_x_new, aux_g_new, aux_y, E, dim_g, dim_x, &E1, &alpha,
                                                 v_cap, limited ? aux_minus_hy : nullptr);
        PROF_RESET();
        DPrint1("Alpha is %f, New Energy is %f\n", alpha, E1);

        // aux_y = aux_g_new - aux_g
        for (int i = tile.thread_rank(); i < dim_g; i += tile.num_threads()){
            aux_y[i * STRIDE_G] = aux_g_new[i * STRIDE_G] - aux_g[i * STRIDE_G];
        }
        tile.sync();
        if(tile.thread_rank() == 0){
            DPrint1("p is \n", 1);
            print_g(aux_p, dim_g);
            DPrint1("g_new is \n", 1);
            print_g(aux_g_new, dim_g);
            DPrint1("g_old is \n", 1);
            print_g(aux_g, dim_g);
            DPrint1("y is \n", 1);
            print_g(aux_y, dim_g);
        }
        // Update energy as the new one
        E = E1;

        // out_x := aux_x_new
        duplicate_pose_warp(tile, out_x, aux_x_new, dim_x, flex_topo.natom);

        // Stop criterion: check convergence todo: why not checking g_new?
        Real gg = g_dot_product_warp(tile, aux_g, aux_g, dim_g);
        if (sqrtf(gg) < 1e-5f){
            break;
        }

        // Update aux_g:= aux_g_new
        duplicate_grad_warp(tile, aux_g, aux_g_new, dim_g);

        // Choose a better initial hessian
        if (step == 0){
            Real yy = g_dot_product_warp(tile, aux_y, aux_y, dim_g);
            if (fabs(yy) > EPSILON_cu){
                // yp = aux_y * -Hg
                Real yp = g_dot_product_warp(tile, aux_y, aux_p, dim_g);
                set_tri_mat_diagonal_warp(tile, aux_h, dim_g, alpha * yp / yy); // heuristic value

                DExec(
                    printf("yy is %f, yp is %f\n", yy, yp);
                    printf("modified Hessian of Step 1: \n");
                    print_uptri_mat(aux_h, dim_g);
                )
            }

        }
        tile.sync();



        // aux_minus_hy serves a container rather than a given parameter
        bfgs_update_hessian_warp(tile, aux_h, dim_g, aux_p, aux_y, aux_minus_hy, alpha);
        PROF_MARK(PROF_BFGS_ALGEBRA);


        DExec(
            printf("Updated Hessian: \n");
            print_uptri_mat(aux_h, dim_g);
        )
    }

    // If this optimization fails (a higher energy is found), resume to the original state
    if (E > E_ori){
        E = E_ori;
        duplicate_pose_warp(tile, out_x, aux_x_ori, dim_x, flex_topo.natom);
        duplicate_grad_warp(tile, aux_g, aux_g_ori, dim_g);
    }

    // write output_type_cuda energy
    if (tile.thread_rank() == 0){
        out_x->energy = E;
    }
    tile.sync();
}


#endif //BFGS_CUH
