//
// Position-bias term (plan step 5a): the ud2align Gaussian (BT_POS) and Z-align (BT_ALIGN) potentials that pull
// ligand atoms onto reference positions, with the per-ligand row data the search kernels read.
//
// The two row potentials are the former Vina::eval_ef_pos / Vina::eval_ef_zalign of score/vina.h, moved here
// unchanged, and accumulate_as is the former per-atom loop of cpu/score_cpu.cpp ("4. Compute position-bias") and
// cuda/bfgs.cuh ("3b. position bias"), which were the same expression in the same order. Both call sites now run
// this one body, so the plain path (type BT_NO) and the biased path compute what they computed before, operation
// for operation.
//

#ifndef UD2_EXT_BIAS_TERM_H
#define UD2_EXT_BIAS_TERM_H

#include <cmath>
#include "myutils/common.h"
#include "myutils/mymath.h"
#include "constants/constants.h"
#include "model/model.h"

/**
 * @brief One ligand's position-bias rows, as the search kernels read them.
 *
 * Row layout (UDFlexMol::bias, read by ext/bias/bias_reader.cpp, flattened by flatten_bias_rows below for
 * cpu/task_cpu.cpp and cuda/task.cu): inds[a * 2], inds[a * 2 + 1] is the half-open range of atom a's rows in
 * params, in Real slots, [0, 0] when the atom has none; each row is five Reals [ref_x, ref_y, ref_z, V_set, r], the
 * rows of one atom contiguous.
 *
 * The run-wide settings (DockParam::bias_type / bias_k) are not stored here but in TermSettings (score/terms.h):
 * the CPU path's copy in TermSet::settings, the CUDA kernel's __constant__ mirror g_term_settings. TermSet is the
 * one dispatch the kernels call; it passes the settings into accumulate_as.
 */
struct PositionBiasTerm {
    int*  inds = nullptr;    // size: natom * 2. each is [start, end] index into params, [0,0] for no bias
    Real* params = nullptr;  // size: nbias * 5. each 5 is [x, y, z, V_set, r^2]

    /**
     * @brief Compute Gaussian position-bias energy and force.
     * @param r_ Real[3], vector from biased atom to the bias position (bias_pos - atom_pos)
     * @param V_set Bias depth (scaled by bias_k)
     * @param r2 Gaussian width parameter (sigma^2)
     * @param out_f Output force (accumulated)
     * @return energy
     */
    static SCOPE_INLINE Real eval_pos(Real* r_, Real V_set, Real r2, Real* out_f){
        Real rr = r_[0] * r_[0] + r_[1] * r_[1] + r_[2] * r_[2];
        Real e_bias = V_set * std::exp(- rr / r2);

        out_f[0] += e_bias * 2 * r_[0] / r2;
        out_f[1] += e_bias * 2 * r_[1] / r2;
        out_f[2] += e_bias * 2 * r_[2] / r2;

        return e_bias;
    }

    /**
     * @brief Compute Z-align attractive potential.
     * Wang, Z., Zhou, F., Wang, Z., Hu, Q., Li, Y. Q., Wang, S., Wei, Y., Zheng, L., Li, W., &
     * Peng, X. (2024). Fully Flexible Molecular Alignment Enables Accurate Ligand Structure
     * Modeling. Journal of Chemical Information and Modeling, 64(15), 6205-6215.
     *
     * @param r_ Real[3], vector from biased atom to the bias position
     * @param a_qt Affinity score between biased atom and the template atom at the bias position
     * @param vn_type Atom type of the biased atom
     * @param out_f Output derivative (-force)
     * @return energy
     */
    static SCOPE_INLINE Real eval_zalign(Real* r_, Real a_qt, int vn_type, Real* out_f){
        const Real s_a = 3.;
        const Real c_a = 1.;
        Real k_a = (vn_type == VN_TYPE_H ? Real(0.1) : Real(30.));

        Real r = cal_norm(r_);

        Real e_item = std::exp(s_a * (c_a - r));
        Real e_bias = a_qt * k_a / (1 + e_item);
        Real f = - e_bias * e_item / (1 + e_item) * s_a;

        out_f[0] += f * r_[0] / r;
        out_f[1] += f * r_[1] / r;
        out_f[2] += f * r_[2] / r;

        return e_bias;
    }

    /**
     * @brief Energy and +dE/dx of the bias on atoms lane, lane + stride, ... below natom (one lane per atom, so
     * aux_f is written without atomics), with the bias settings given (TermSet::accumulate_as passes them: the
     * CUDA kernel's __constant__ mirror, the CPU path's copy). Every row's energy is added to `energy` in row
     * order and the atom's force to aux_f[i * 3 .. i * 3 + 2], as the two former loops did.
     * @param coords ligand coordinates, natom * 3
     * @param types ligand atom types (the hydrogen switch of the Z-align potential)
     */
    SCOPE_INLINE void accumulate_as(int bias_type, Real bias_k, int lane, int stride, int natom,
                                    const Real* coords, const int* types, Real& energy, Real* aux_f) const {
        for (int i = lane; i < natom; i += stride){
            const int j1 = inds[i * 2], j2 = inds[i * 2 + 1];
            if (j1 >= j2) continue;
            const Real ax = coords[i * 3], ay = coords[i * 3 + 1], az = coords[i * 3 + 2];
            Real f_bias[3] = {0., 0., 0.};
            for (int j = j1; j < j2; j += 5){
                const Real* pb = params + j;
                Real r_[3] = {pb[0] - ax, pb[1] - ay, pb[2] - az};            // ref - atom
                if (bias_type == BT_POS)
                    energy += eval_pos(r_, pb[3] * bias_k, pb[4], f_bias);
                else if (bias_type == BT_ALIGN)
                    energy += eval_zalign(r_, pb[3] * bias_k, types[i], f_bias);
            }
            aux_f[i * 3] += f_bias[0]; aux_f[i * 3 + 1] += f_bias[1]; aux_f[i * 3 + 2] += f_bias[2];
        }
    }
};

/**
 * @brief One ligand's position-bias rows (UDFlexMol::bias, rowid order) in the PositionBiasTerm layout, on the
 * host: inds[0, natom * 2) zeroed, then each atom's half-open Real-slot range; the five values of each row into
 * params, in row order. The one flattening the CPU fill (cpu/task_cpu.cpp, into its arena) and the CUDA fill
 * (cuda/task.cu, before the upload) share, and from which a host-side TermSet (a pose pass) points its bias term.
 * @param inds natom * 2 ints
 * @param params bias.rows.size() * 5 Reals
 * @return the Real slots written (5 per row)
 */
inline int flatten_bias_rows(const UDFlexMol& m, int* inds, Real* params){
    for (int k = 0; k < m.natom * 2; k++) inds[k] = 0;
    int last_i = -1;
    int offset = 0;
    for (const auto& b : m.bias.rows){
        if (b.i != last_i) inds[b.i * 2] = offset;
        for (int k = 0; k < 5; k++) params[offset + k] = static_cast<Real>(b.param[k]);   // Bias::param is float
        offset += 5;
        inds[b.i * 2 + 1] = offset;
        last_i = b.i;
    }
    return offset;
}

#endif // UD2_EXT_BIAS_TERM_H
