//
// CPU Full Force Field Scoring Implementation
//
// Convention: aux_f stores the energy GRADIENT ∂E/∂x (same as Vina scoring).
// For a pair (i,j): aux_f[i] += -(dE/dr / r) * (x_j - x_i) = ∂E/∂x_i
// This is the NEGATIVE of force; cal_grad_ff_cpu then projects this into
// internal coordinates for BFGS, which uses p = -H*g for descent.
//

#include "score_ff_cpu.h"
#include "score/gaff2.h"
#include "myutils/matrix.h"
#include <cstring>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const Real EPSILON_FF = 1e-10;


Real compute_dihedral_angle(const Real* p0, const Real* p1, const Real* p2, const Real* p3) {
    // Vectors along bonds
    Real b1[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
    Real b2[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
    Real b3[3] = {p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]};

    // Normal vectors to planes
    Real n1[3], n2[3];
    cross_cpu(b1, b2, n1);  // b1 x b2
    cross_cpu(b2, b3, n2);  // b2 x b3

    // Cross product of normals
    Real m[3];
    cross_cpu(n1, n2, m);

    Real b2_len = norm_cpu(b2);
    if (b2_len < EPSILON_FF) return 0;

    Real x = dot_cpu(n1, n2);
    Real y = dot_cpu(m, b2) / b2_len;

    return std::atan2(y, x);
}


void distribute_dihedral_forces(const Real* p0, const Real* p1, const Real* p2, const Real* p3,
                                Real dE_dphi,
                                Real* f0, Real* f1, Real* f2, Real* f3) {
    // Bond vectors
    Real b1[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
    Real b2[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
    Real b3[3] = {p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]};

    // Normal vectors
    Real m[3], n[3];
    cross_cpu(b1, b2, m);  // m = b1 x b2
    cross_cpu(b2, b3, n);  // n = b2 x b3

    Real mm = dot_cpu(m, m);
    Real nn = dot_cpu(n, n);
    Real b2sq = dot_cpu(b2, b2);
    Real b2len = std::sqrt(b2sq);

    if (mm < EPSILON_FF || nn < EPSILON_FF || b2len < EPSILON_FF) return;

    Real b1_dot_b2 = dot_cpu(b1, b2);
    Real b3_dot_b2 = dot_cpu(b3, b2);

    // Force on atom 0: F0 = -dE/dphi * |b2| / |m|^2 * m
    Real scale0 = -dE_dphi * b2len / mm;
    // Force on atom 3: F3 = dE/dphi * |b2| / |n|^2 * n
    Real scale3 = dE_dphi * b2len / nn;

    Real F0[3] = {scale0 * m[0], scale0 * m[1], scale0 * m[2]};
    Real F3[3] = {scale3 * n[0], scale3 * n[1], scale3 * n[2]};

    // Force on atom 1 and 2 (from chain rule via bond vectors)
    Real fga = b1_dot_b2 / b2sq;
    Real hgb = b3_dot_b2 / b2sq;

    // F1 = -F0 + fga*F0 - hgb*F3 = (fga-1)*F0 - hgb*F3
    // F2 = -F3 - fga*F0 + hgb*F3 = -fga*F0 + (hgb-1)*F3
    Real c1a = fga - 1.0f;
    Real c1b = -hgb;
    Real c2a = -fga;
    Real c2b = hgb - 1.0f;

    // Accumulate forces
    for (int k = 0; k < 3; k++) {
        f0[k] += F0[k];
        f1[k] += c1a * F0[k] + c1b * F3[k];
        f2[k] += c2a * F0[k] + c2b * F3[k];
        f3[k] += F3[k];
    }
}


void distribute_angle_forces(const Real* pi, const Real* pj, const Real* pk,
                             Real dE_dtheta,
                             Real* fi, Real* fj, Real* fk) {
    Real a[3] = {pi[0]-pj[0], pi[1]-pj[1], pi[2]-pj[2]};
    Real c[3] = {pk[0]-pj[0], pk[1]-pj[1], pk[2]-pj[2]};

    Real a_len = norm_cpu(a);
    Real c_len = norm_cpu(c);
    if (a_len < EPSILON_FF || c_len < EPSILON_FF) return;

    Real ac_inv = 1.0f / (a_len * c_len);
    Real a2_inv = 1.0f / (a_len * a_len);
    Real c2_inv = 1.0f / (c_len * c_len);

    Real cos_t = dot_cpu(a, c) * ac_inv;
    if (cos_t > 1.0f) cos_t = 1.0f;
    if (cos_t < -1.0f) cos_t = -1.0f;

    Real sin_t = std::sqrt(1.0f - cos_t * cos_t);
    if (sin_t < 1e-7f) sin_t = 1e-7f;

    Real inv_neg_sin = -1.0f / sin_t;

    Real d_a[3], d_c[3];
    for (int k = 0; k < 3; k++) {
        d_a[k] = (c[k] * ac_inv - cos_t * a[k] * a2_inv) * inv_neg_sin;
        d_c[k] = (a[k] * ac_inv - cos_t * c[k] * c2_inv) * inv_neg_sin;
    }

    for (int k = 0; k < 3; k++) {
        Real gi = dE_dtheta * d_a[k];
        Real gk = dE_dtheta * d_c[k];
        fi[k] += gi;
        fk[k] += gk;
        fj[k] -= (gi + gk);
    }
}


Real cal_e_f_ff_cpu(const FlexPose* pose, const FlexTopo& flex_topo,
                    const FixMol& /* fix_mol */, const FFMinData& ff,
                    Real* aux_f) {
    Real energy = 0;
    const int nlig = ff.lig_atoms.natom;
    const int nrec = ff.rec_atoms.natom;
    const Real* __restrict__ lig_coords = pose->coords;
    // Use receptor coordinates from DMS (ff.rec_atoms), NOT from fix_mol.
    // fix_mol only contains atoms within the docking box (subset),
    // while ff.rec_atoms has ALL protein atoms from DMS.

    // Initialize forces to zero
    std::memset(aux_f, 0, nlig * 3 * sizeof(Real));

    // Use precomputed pair_type lookup (built once at DMS load time)
    const uint8_t* __restrict__ pt = ff.pair_type.data();

    // ================================================================
    // 1. Intra-molecular nonbonded (non-excluded, non-1-4)
    // ================================================================
    for (int i = 0; i < nlig; i++) {
        for (int j = i + 1; j < nlig; j++) {
            // O(1) lookup: skip excluded (1-2/1-3) and 1-4 pairs
            uint8_t ptype = pt[FFMinData::tri_index(j, i)];
            if (ptype != FF_PAIR_NORMAL) continue;

            Real dx = lig_coords[j*3]   - lig_coords[i*3];
            Real dy = lig_coords[j*3+1] - lig_coords[i*3+1];
            Real dz = lig_coords[j*3+2] - lig_coords[i*3+2];
            Real r2 = dx*dx + dy*dy + dz*dz;
            Real r = std::sqrt(r2);
            if (r < FF_MIN_DIST) r = FF_MIN_DIST;

            // Lorentz-Berthelot combining rules
            Real sig_ij = (ff.lig_atoms.sigma[i] + ff.lig_atoms.sigma[j]) * 0.5f;
            Real eps_ij = std::sqrt(ff.lig_atoms.epsilon[i] * ff.lig_atoms.epsilon[j]);

            Real dE_dr_lj = 0, dE_dr_coul = 0;
            energy += ff_lj(r, sig_ij, eps_ij, &dE_dr_lj);
            energy += ff_coulomb(r, ff.lig_atoms.charge[i], ff.lig_atoms.charge[j], &dE_dr_coul);

            Real dE_dr = dE_dr_lj + dE_dr_coul;
            Real f_scale = -dE_dr / r;  // = -(dE/dr)/r for gradient accumulation

            aux_f[i*3]   += f_scale * dx;  // ∂E/∂x_i contribution
            aux_f[i*3+1] += f_scale * dy;
            aux_f[i*3+2] += f_scale * dz;
            aux_f[j*3]   -= f_scale * dx;  // ∂E/∂x_j contribution (opposite sign)
            aux_f[j*3+1] -= f_scale * dy;
            aux_f[j*3+2] -= f_scale * dz;
        }
    }

    // ================================================================
    // 2. 1-4 pair interactions (pre-scaled from DMS)
    // ================================================================
    for (const auto& p : ff.lig_pairs_14) {
        int i = p.i, j = p.j;
        Real dx = lig_coords[j*3]   - lig_coords[i*3];
        Real dy = lig_coords[j*3+1] - lig_coords[i*3+1];
        Real dz = lig_coords[j*3+2] - lig_coords[i*3+2];
        Real r2 = dx*dx + dy*dy + dz*dz;
        Real r = std::sqrt(r2);
        if (r < FF_MIN_DIST) r = FF_MIN_DIST;

        Real dE_dr = 0;
        energy += ff_pair14(r, p.aij, p.bij, p.qij, &dE_dr);

        Real f_scale = -dE_dr / r;
        aux_f[i*3]   += f_scale * dx;
        aux_f[i*3+1] += f_scale * dy;
        aux_f[i*3+2] += f_scale * dz;
        aux_f[j*3]   -= f_scale * dx;
        aux_f[j*3+1] -= f_scale * dy;
        aux_f[j*3+2] -= f_scale * dz;
    }

    // ================================================================
    // 2b. Bond stretch (harmonic, fc absorbs the 1/2 factor)
    // ================================================================
    for (const auto& b : ff.lig_bonds) {
        int i = b.i, j = b.j;
        Real dx = lig_coords[j*3]   - lig_coords[i*3];
        Real dy = lig_coords[j*3+1] - lig_coords[i*3+1];
        Real dz = lig_coords[j*3+2] - lig_coords[i*3+2];
        Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < EPSILON_FF) continue;

        Real dE_dr = 0;
        energy += ff_bond(r, b.r0, b.fc, &dE_dr);

        Real f_scale = -dE_dr / r;
        aux_f[i*3]   += f_scale * dx;
        aux_f[i*3+1] += f_scale * dy;
        aux_f[i*3+2] += f_scale * dz;
        aux_f[j*3]   -= f_scale * dx;
        aux_f[j*3+1] -= f_scale * dy;
        aux_f[j*3+2] -= f_scale * dz;
    }

    // ================================================================
    // 2c. Angle bend (harmonic, fc absorbs the 1/2 factor)
    // ================================================================
    for (const auto& a : ff.lig_angles) {
        const Real* pi = lig_coords + a.i * 3;
        const Real* pj = lig_coords + a.j * 3;
        const Real* pk = lig_coords + a.k * 3;

        Real va[3] = {pi[0]-pj[0], pi[1]-pj[1], pi[2]-pj[2]};
        Real vc[3] = {pk[0]-pj[0], pk[1]-pj[1], pk[2]-pj[2]};
        Real a_len = norm_cpu(va);
        Real c_len = norm_cpu(vc);
        if (a_len < EPSILON_FF || c_len < EPSILON_FF) continue;

        Real cos_t = dot_cpu(va, vc) / (a_len * c_len);
        if (cos_t > 1.0f) cos_t = 1.0f;
        if (cos_t < -1.0f) cos_t = -1.0f;
        Real theta = std::acos(cos_t);

        Real dE_dtheta = 0;
        energy += ff_angle(theta, a.theta0, a.fc, &dE_dtheta);

        distribute_angle_forces(pi, pj, pk, dE_dtheta,
                                aux_f + a.i*3,
                                aux_f + a.j*3,
                                aux_f + a.k*3);
    }

    // ================================================================
    // 3. Torsion energy (all dihedrals from DMS)
    // ================================================================
    for (const auto& tor : ff.lig_torsions) {
        const Real* pp0 = lig_coords + tor.p0 * 3;
        const Real* pp1 = lig_coords + tor.p1 * 3;
        const Real* pp2 = lig_coords + tor.p2 * 3;
        const Real* pp3 = lig_coords + tor.p3 * 3;

        Real phi = compute_dihedral_angle(pp0, pp1, pp2, pp3);

        Real dE_dphi = 0;
        energy += ff_torsion(phi, tor.phi0, tor.fc, &dE_dphi);

        // Distribute torsion force to 4 atoms
        distribute_dihedral_forces(pp0, pp1, pp2, pp3, dE_dphi,
                                   aux_f + tor.p0*3,
                                   aux_f + tor.p1*3,
                                   aux_f + tor.p2*3,
                                   aux_f + tor.p3*3);
    }

    // ================================================================
    // 4. Inter-molecular LJ + Coulomb (ligand-receptor, within cutoff)
    //    Receptor coords from ff.rec_atoms (DMS), stored as separate x/y/z vectors
    // ================================================================
    const Real* __restrict__ rec_x = ff.rec_atoms.x.data();
    const Real* __restrict__ rec_y = ff.rec_atoms.y.data();
    const Real* __restrict__ rec_z = ff.rec_atoms.z.data();

    for (int i = 0; i < nlig; i++) {
        Real lx = lig_coords[i*3];
        Real ly = lig_coords[i*3+1];
        Real lz = lig_coords[i*3+2];

        for (int j = 0; j < nrec; j++) {
            Real dx = rec_x[j] - lx;
            Real dy = rec_y[j] - ly;
            Real dz = rec_z[j] - lz;
            Real r2 = dx*dx + dy*dy + dz*dz;

            if (r2 >= FF_CUTOFF_SQ) continue;

            Real r = std::sqrt(r2);
            if (r < FF_MIN_DIST) r = FF_MIN_DIST;

            // Cross-type combining rules
            Real sig_ij = (ff.lig_atoms.sigma[i] + ff.rec_atoms.sigma[j]) * 0.5f;
            Real eps_ij = std::sqrt(ff.lig_atoms.epsilon[i] * ff.rec_atoms.epsilon[j]);

            Real dE_dr_lj = 0, dE_dr_coul = 0;
            energy += ff_lj(r, sig_ij, eps_ij, &dE_dr_lj);
            energy += ff_coulomb(r, ff.lig_atoms.charge[i], ff.rec_atoms.charge[j], &dE_dr_coul);

            Real dE_dr = dE_dr_lj + dE_dr_coul;
            Real f_scale = -dE_dr / r;

            // Only accumulate force on ligand atom (receptor is fixed)
            aux_f[i*3]   += f_scale * dx;
            aux_f[i*3+1] += f_scale * dy;
            aux_f[i*3+2] += f_scale * dz;
        }
    }

    // ================================================================
    // 5. Box penalty (reuse existing implementation)
    // ================================================================
    for (int i = 0; i < nlig; i++) {
        Real coord_x = lig_coords[i*3];
        Real coord_y = lig_coords[i*3+1];
        Real coord_z = lig_coords[i*3+2];
        Real f_tmp[3] = {0, 0, 0};
        energy += cal_box_penalty_clamp_coord_cpu(&coord_x, &coord_y, &coord_z, f_tmp);
        aux_f[i*3]   += f_tmp[0];
        aux_f[i*3+1] += f_tmp[1];
        aux_f[i*3+2] += f_tmp[2];
    }

    return energy;
}


std::vector<FFAtomTerms> score_decomp_ff(const FlexPose* pose, const FFMinData& ff) {
    const int nlig = ff.lig_atoms.natom;
    const int nrec = ff.rec_atoms.natom;
    const Real* __restrict__ lig_coords = pose->coords;

    std::vector<FFAtomTerms> atom_terms(nlig);

    // Use precomputed pair_type lookup
    const uint8_t* __restrict__ pt = ff.pair_type.data();

    // 1. Intra-molecular nonbonded (non-excluded, non-1-4)
    for (int i = 0; i < nlig; i++) {
        for (int j = i + 1; j < nlig; j++) {
            if (pt[FFMinData::tri_index(j, i)] != FF_PAIR_NORMAL) continue;

            Real dx = lig_coords[j*3]   - lig_coords[i*3];
            Real dy = lig_coords[j*3+1] - lig_coords[i*3+1];
            Real dz = lig_coords[j*3+2] - lig_coords[i*3+2];
            Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (r < FF_MIN_DIST) r = FF_MIN_DIST;

            Real sig_ij = (ff.lig_atoms.sigma[i] + ff.lig_atoms.sigma[j]) * 0.5f;
            Real eps_ij = std::sqrt(ff.lig_atoms.epsilon[i] * ff.lig_atoms.epsilon[j]);

            Real dE_dummy = 0;
            Real e_lj   = ff_lj(r, sig_ij, eps_ij, &dE_dummy);
            Real e_coul = ff_coulomb(r, ff.lig_atoms.charge[i], ff.lig_atoms.charge[j], &dE_dummy);

            // Split evenly between the two atoms
            atom_terms[i].intra_lj      += e_lj * 0.5f;
            atom_terms[j].intra_lj      += e_lj * 0.5f;
            atom_terms[i].intra_coulomb  += e_coul * 0.5f;
            atom_terms[j].intra_coulomb  += e_coul * 0.5f;
        }
    }

    // 2. 1-4 pair interactions (pre-scaled from DMS) — count as intra
    for (const auto& p : ff.lig_pairs_14) {
        int i = p.i, j = p.j;
        Real dx = lig_coords[j*3]   - lig_coords[i*3];
        Real dy = lig_coords[j*3+1] - lig_coords[i*3+1];
        Real dz = lig_coords[j*3+2] - lig_coords[i*3+2];
        Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < FF_MIN_DIST) r = FF_MIN_DIST;

        // Decompose pair14 into LJ and Coulomb parts separately
        Real inv_r = 1.0f / r;
        Real inv_r2 = inv_r * inv_r;
        Real inv_r6 = inv_r2 * inv_r2 * inv_r2;
        Real inv_r12 = inv_r6 * inv_r6;

        Real e_lj   = p.aij * inv_r12 - p.bij * inv_r6;
        Real e_coul = COULOMB_CONST * p.qij / DDD_FACTOR * inv_r2;

        atom_terms[i].intra_lj      += e_lj * 0.5f;
        atom_terms[j].intra_lj      += e_lj * 0.5f;
        atom_terms[i].intra_coulomb  += e_coul * 0.5f;
        atom_terms[j].intra_coulomb  += e_coul * 0.5f;
    }

    // 3. Torsion energy — distribute half to p1, half to p2 (central bond atoms)
    for (const auto& tor : ff.lig_torsions) {
        const Real* pp0 = lig_coords + tor.p0 * 3;
        const Real* pp1 = lig_coords + tor.p1 * 3;
        const Real* pp2 = lig_coords + tor.p2 * 3;
        const Real* pp3 = lig_coords + tor.p3 * 3;

        Real phi = compute_dihedral_angle(pp0, pp1, pp2, pp3);
        Real dE_dummy = 0;
        Real e_tor = ff_torsion(phi, tor.phi0, tor.fc, &dE_dummy);

        atom_terms[tor.p1].torsion += e_tor * 0.5f;
        atom_terms[tor.p2].torsion += e_tor * 0.5f;
    }

    // 3b. Bond stretch (harmonic) — split evenly between the two atoms.
    //     Per-atom attribution of a 2-body term is arbitrary; only the sum is
    //     physical, so an even split keeps decomp_sum == cal_e_f_ff_cpu total.
    for (const auto& b : ff.lig_bonds) {
        int i = b.i, j = b.j;
        Real dx = lig_coords[j*3]   - lig_coords[i*3];
        Real dy = lig_coords[j*3+1] - lig_coords[i*3+1];
        Real dz = lig_coords[j*3+2] - lig_coords[i*3+2];
        Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < EPSILON_FF) continue;

        Real dE_dummy = 0;
        Real e_bond = ff_bond(r, b.r0, b.fc, &dE_dummy);
        atom_terms[i].bond += e_bond * 0.5f;
        atom_terms[j].bond += e_bond * 0.5f;
    }

    // 3c. Angle bend (harmonic) — split evenly among the three atoms.
    for (const auto& a : ff.lig_angles) {
        const Real* pi = lig_coords + a.i * 3;
        const Real* pj = lig_coords + a.j * 3;
        const Real* pk = lig_coords + a.k * 3;

        Real va[3] = {pi[0]-pj[0], pi[1]-pj[1], pi[2]-pj[2]};
        Real vc[3] = {pk[0]-pj[0], pk[1]-pj[1], pk[2]-pj[2]};
        Real a_len = norm_cpu(va);
        Real c_len = norm_cpu(vc);
        if (a_len < EPSILON_FF || c_len < EPSILON_FF) continue;

        Real cos_t = dot_cpu(va, vc) / (a_len * c_len);
        if (cos_t > 1.0f) cos_t = 1.0f;
        if (cos_t < -1.0f) cos_t = -1.0f;
        Real theta = std::acos(cos_t);

        Real dE_dummy = 0;
        Real e_angle = ff_angle(theta, a.theta0, a.fc, &dE_dummy);
        Real third = e_angle / 3.0f;
        atom_terms[a.i].angle += third;
        atom_terms[a.j].angle += third;
        atom_terms[a.k].angle += third;
    }

    // 4. Inter-molecular LJ + Coulomb (ligand-receptor, within cutoff)
    //    Receptor coordinates come from ff.rec_atoms (DMS source), not from FixMol
    for (int i = 0; i < nlig; i++) {
        Real lx = lig_coords[i*3];
        Real ly = lig_coords[i*3+1];
        Real lz = lig_coords[i*3+2];

        for (int j = 0; j < nrec; j++) {
            Real dx = ff.rec_atoms.x[j] - lx;
            Real dy = ff.rec_atoms.y[j] - ly;
            Real dz = ff.rec_atoms.z[j] - lz;
            Real r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= FF_CUTOFF_SQ) continue;

            Real r = std::sqrt(r2);
            if (r < FF_MIN_DIST) r = FF_MIN_DIST;

            Real sig_ij = (ff.lig_atoms.sigma[i] + ff.rec_atoms.sigma[j]) * 0.5f;
            Real eps_ij = std::sqrt(ff.lig_atoms.epsilon[i] * ff.rec_atoms.epsilon[j]);

            Real dE_dummy = 0;
            Real e_lj   = ff_lj(r, sig_ij, eps_ij, &dE_dummy);
            Real e_coul = ff_coulomb(r, ff.lig_atoms.charge[i], ff.rec_atoms.charge[j], &dE_dummy);

            atom_terms[i].inter_lj      += e_lj;
            atom_terms[i].inter_coulomb  += e_coul;
        }
    }

    // Compute totals
    for (int i = 0; i < nlig; i++) {
        atom_terms[i].total = atom_terms[i].inter_lj + atom_terms[i].inter_coulomb
                            + atom_terms[i].intra_lj + atom_terms[i].intra_coulomb
                            + atom_terms[i].torsion
                            + atom_terms[i].bond + atom_terms[i].angle;
    }

    return atom_terms;
}


void cal_grad_ff_cpu(const Real* aux_f, const FlexTopo& flex_topo,
                     const FlexPose* pose, FlexPoseGradient* out_g) {
    Real tmp1[3], tmp2[3], tmp3[3], tmp5[3];
    Real mat[9] = {0};

    // Gradient on dihedrals (torque projection)
    for (int i_tor = 0; i_tor < flex_topo.ntorsion; i_tor++) {
        int begin = flex_topo.rotated_inds[i_tor * 2];
        int end = flex_topo.rotated_inds[i_tor * 2 + 1];
        int iat_from = flex_topo.axis_atoms[i_tor * 2];
        int iat_to = flex_topo.axis_atoms[i_tor * 2 + 1];

        tmp5[0] = tmp5[1] = tmp5[2] = 0.0;

        for (int j = begin; j <= end; j++) {
            int i_at = flex_topo.rotated_atoms[j];
            // NOTE: Unlike Vina version, do NOT skip hydrogens for FF scoring

            // Lever arm: atom position relative to rotation axis point
            tmp1[0] = pose->coords[i_at*3]   - pose->coords[iat_to*3];
            tmp1[1] = pose->coords[i_at*3+1] - pose->coords[iat_to*3+1];
            tmp1[2] = pose->coords[i_at*3+2] - pose->coords[iat_to*3+2];

            // Force on this atom
            tmp2[0] = aux_f[i_at*3];
            tmp2[1] = aux_f[i_at*3+1];
            tmp2[2] = aux_f[i_at*3+2];

            // Torque = lever x force
            cross_cpu(tmp1, tmp2, tmp3);
            tmp5[0] += tmp3[0];
            tmp5[1] += tmp3[1];
            tmp5[2] += tmp3[2];
        }

        // Axis direction
        tmp3[0] = pose->coords[iat_to*3]   - pose->coords[iat_from*3];
        tmp3[1] = pose->coords[iat_to*3+1] - pose->coords[iat_from*3+1];
        tmp3[2] = pose->coords[iat_to*3+2] - pose->coords[iat_from*3+2];

        // Project torque onto axis
        Real axis_norm = norm_cpu(tmp3);
        out_g->dihedrals_g[i_tor] = dot_cpu(tmp5, tmp3) / axis_norm;
    }

    // Center and orientation gradients
    Real tmp4[3] = {0, 0, 0};
    std::memset(mat, 0, sizeof(mat));

    // Inverse rotation quaternion
    Real q[4];
    rotvec_to_quaternion(q, pose->rot_vec);
    quaternion_conjugate(q);

    for (int i_at = 0; i_at < flex_topo.natom; i_at++) {
        // NOTE: Include ALL atoms (including H) for FF

        // Relative position to center
        tmp1[0] = pose->coords[i_at*3]   - pose->center[0];
        tmp1[1] = pose->coords[i_at*3+1] - pose->center[1];
        tmp1[2] = pose->coords[i_at*3+2] - pose->center[2];

        // Rotate to reference frame
        rotate_vec_by_quaternion(tmp1, q);

        // Force
        tmp2[0] = aux_f[i_at*3];
        tmp2[1] = aux_f[i_at*3+1];
        tmp2[2] = aux_f[i_at*3+2];

        // Sum center gradient (= sum of forces)
        tmp4[0] += tmp2[0];
        tmp4[1] += tmp2[1];
        tmp4[2] += tmp2[2];

        // Accumulate outer product for orientation gradient
        Real mat_tmp[9];
        outer_product(tmp2, tmp1, mat_tmp);
        for (int k = 0; k < 9; k++) mat[k] += mat_tmp[k];
    }

    // Write center gradient
    if (!g_cpu_constants.FLAG_CONSTRAINT_DOCK) {
        out_g->center_g[0] = tmp4[0];
        out_g->center_g[1] = tmp4[1];
        out_g->center_g[2] = tmp4[2];

        // Orientation gradient
        for (int i = 0; i < 3; i++) {
            Real dR_dv[9];
            cal_grad_of_rot_over_vec(dR_dv, pose->rot_vec, i);
            out_g->orientation_g[i] = frobenius_product(mat, dR_dv);
        }
    }
}


Real cal_e_grad_ff_cpu(const FlexPose* pose, FlexPoseGradient* out_g,
                       const FlexTopo& flex_topo,
                       const FixMol& fix_mol, const FFMinData& ff,
                       Real* aux_f) {
    Real energy = cal_e_f_ff_cpu(pose, flex_topo, fix_mol, ff, aux_f);
    cal_grad_ff_cpu(aux_f, flex_topo, pose, out_g);
    return energy;
}


Real cal_e_grad_cartesian_cpu(const FlexPose* pose, const FlexTopo& flex_topo,
                              const FixMol& fix_mol, const FFMinData& ff,
                              Real* grad, Real* aux_f) {
    // aux_f already stores +dE/dx (see header convention). Gradient = aux_f directly.
    Real energy = cal_e_f_ff_cpu(pose, flex_topo, fix_mol, ff, aux_f);
    std::memcpy(grad, aux_f, flex_topo.natom * 3 * sizeof(Real));
    return energy;
}
