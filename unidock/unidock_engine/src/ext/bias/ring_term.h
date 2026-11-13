//
// Aromatic-ring bias term (plan step 6): pulls a query ring of the ligand onto a template ring (centroid, normal),
// the ud2align ring rows (tables ud2_ring_bias + ud2_ring_bias_atoms, ext/bias/ring_reader.cpp), scored next to the
// position bias when bias_type is BT_ALIGN.
//
// RingBiasFrame, ring_bias_frame and ring_bias_atom_grad are the October functions of score/vina.h
// (feat/ud2align-ring-bias), same arithmetic in the same order, with the three compile-time constants RING_BIAS_K /
// _S / _D0 read from the run's RingBiasParams instead (model/model.h; the macros are their defaults).
// RingBiasTerm::accumulate_as is the per-row loop of the October blocks of cpu/score_cpu.cpp and cuda/bfgs.cuh "3c":
// one body, the force added with atomicAdd_block on the device (rows share atoms: fused rings, one query ring paired
// with several template rings) and with += on the host, as the two blocks did.
//
// RingBiasRow / RingBiasRows / RingBiasParams sit in model/model.h next to Bias / PositionBiasRows, since UDFlexMol
// and DockParam hold them.
//

#ifndef UD2_EXT_RING_TERM_H
#define UD2_EXT_RING_TERM_H

#include <vector>
#include "model/model.h"
#include "myutils/common.h"
#include "myutils/mymath.h"
#include "constants/constants.h"


/**
 * @brief The factors of one aromatic-ring bias row that every ring atom's derivative shares
 * (ring_bias_frame fills them, ring_bias_atom_grad uses them).
 */
struct RingBiasFrame {
    Real gc[3]; // centroid part of dE/dp_k, the same for every ring atom
    Real w[3];  // n_t - (n . n_t) n
    Real so;    // -2 a F (n . n_t) / |N|
};

/**
 * @brief Aromatic-ring bias energy of one row, and the factors of its Cartesian derivative.
 *
 *     E = -a F(d) u,  a = v_set * bias_k * p.k,  F(d) = 1 / (1 + exp(p.s (d - p.d0))),
 *     d = |c - c_t|,  u = (n . n_t)^2,
 * c the centroid of the ring atoms and n the unit Newell normal (N = sum p_i x p_{i+1}, cyclic), so
 * a flipped ring counts as parallel. Per ring atom k:
 *     dE/dp_k = -a [ F'(d) u (c - c_t) / (d m) + 2 F (n . n_t) (w x (p_{k-1} - p_{k+1})) / |N| ],
 *     w = n_t - (n . n_t) n.
 * F is evaluated as e / (1 + e), e = exp(s (d0 - d)) <= exp(s d0), which cannot overflow.
 * The derivative comes atom by atom from ring_bias_atom_grad, so no per-row array is needed (on the
 * GPU such an array would sit in local memory of every kernel that scores).
 *
 * @param p the run's ring parameters (k 30, s 3, d0 1 by default: RING_BIAS_K / _S / _D0)
 * @param out_frame overwritten. The centroid part is zero for d < 1e-6; E and every factor are zero
 *                  for |N| < 1e-6.
 * @return E
 */
SCOPE_INLINE Real ring_bias_frame(const Real* coords, const RingBiasRow& row, Real bias_k, const RingBiasParams& p,
                                  RingBiasFrame& out_frame){
    for (int i = 0; i < 3; i++){ out_frame.gc[i] = 0; out_frame.w[i] = 0; }
    out_frame.so = 0;
    const int m = row.n;
    const Real inv_m = Real(1) / Real(m);
    Real c[3] = {0, 0, 0};
    for (int k = 0; k < m; k++){
        const Real* p_k = coords + row.atoms[k] * 3;
        c[0] += p_k[0]; c[1] += p_k[1]; c[2] += p_k[2];
    }
    c[0] *= inv_m; c[1] *= inv_m; c[2] *= inv_m;

    // the Newell sum on centred coordinates: the same vector, without cancellation far from the origin
    Real N[3] = {0, 0, 0};
    for (int k = 0; k < m; k++){
        const Real* p_k = coords + row.atoms[k] * 3;
        const Real* q = coords + row.atoms[k + 1 < m ? k + 1 : 0] * 3;
        const Real a1[3] = {p_k[0] - c[0], p_k[1] - c[1], p_k[2] - c[2]};
        const Real a2[3] = {q[0] - c[0], q[1] - c[1], q[2] - c[2]};
        Real t[3];
        cross_product(a1, a2, t);
        N[0] += t[0]; N[1] += t[1]; N[2] += t[2];
    }
    const Real nN = cal_norm(N);
    if (nN < Real(1e-6)) return 0;
    const Real n[3] = {N[0] / nN, N[1] / nN, N[2] / nN};

    const Real dc[3] = {c[0] - row.c[0], c[1] - row.c[1], c[2] - row.c[2]};
    const Real d = cal_norm(dc);
    const Real cos_t = dot_product(n, row.nt);
    const Real u = cos_t * cos_t;
    const Real ex = exp(p.s * (p.d0 - d));
    const Real F = ex / (1 + ex);
    const Real a = row.v_set * bias_k * p.k;

    if (d >= Real(1e-6)){
        const Real dF = -p.s * F / (1 + ex);  // F'(d) = -s F (1 - F)
        const Real sc = -a * u * dF / d * inv_m;
        for (int i = 0; i < 3; i++) out_frame.gc[i] = sc * dc[i];
    }
    for (int i = 0; i < 3; i++) out_frame.w[i] = row.nt[i] - cos_t * n[i];
    out_frame.so = -a * F * 2 * cos_t / nN;
    return -a * F * u;
}

/**
 * @brief dE/dx of the atom at ring position k (0 <= k < row.n) of a row whose frame ring_bias_frame
 * filled at the same coordinates.
 * @param out_g 3 Reals, overwritten
 */
SCOPE_INLINE void ring_bias_atom_grad(const Real* coords, const RingBiasRow& row, const RingBiasFrame& frame,
                                      int k, Real* out_g){
    const int m = row.n;
    const Real* pp = coords + row.atoms[k > 0 ? k - 1 : m - 1] * 3;
    const Real* pn = coords + row.atoms[k + 1 < m ? k + 1 : 0] * 3;
    const Real v[3] = {pp[0] - pn[0], pp[1] - pn[1], pp[2] - pn[2]};
    Real t[3];
    cross_product(frame.w, v, t);
    for (int i = 0; i < 3; i++) out_g[i] = frame.gc[i] + frame.so * t[i];
}

/**
 * @brief One ligand's aromatic-ring bias rows, as the search kernels read them: the host rows of UDFlexMol::ring
 * (CPU, cpu/task_cpu.cpp) or their device copy (cuda/task.cu, cuda/vina.cu); nullptr / 0 without rows.
 *
 * The run-wide settings (bias_type, bias_k, the RingBiasParams) are in TermSettings (score/terms.h); TermSet is the
 * one dispatch the kernels call and passes them into accumulate_as.
 */
struct RingBiasTerm {
    const RingBiasRow* rows = nullptr;  // size: n
    int n = 0;

    /// +dE/dx into one coordinate of aux_f: atomically on the device (several rows, on several lanes, can share an
    /// atom), a plain += on the host.
    static SCOPE_INLINE void add_force(Real* f, Real v){
#ifdef __CUDA_ARCH__
        atomicAdd_block(f, v);
#else
        *f += v;
#endif
    }

    /**
     * @brief Energy and +dE/dx of rows lane, lane + stride, ... (the CPU path: lane 0, stride 1), with the bias
     * scale and ring parameters given. Every row's energy is added to `energy` and its ring atoms' derivative to
     * aux_f, in row order and ring order, as the two former blocks did.
     */
    SCOPE_INLINE void accumulate_as(Real bias_k, const RingBiasParams& p, int lane, int stride, const Real* coords,
                                    Real& energy, Real* aux_f) const {
        for (int r = lane; r < n; r += stride){
            const RingBiasRow& row = rows[r];
            RingBiasFrame fr;
            energy += ring_bias_frame(coords, row, bias_k, p, fr);
            for (int k = 0; k < row.n; k++){
                Real g[3];
                ring_bias_atom_grad(coords, row, fr, k, g);
                const int base = row.atoms[k] * 3;
                add_force(aux_f + base,     g[0]);
                add_force(aux_f + base + 1, g[1]);
                add_force(aux_f + base + 2, g[2]);
            }
        }
    }
};

/**
 * @brief Aromatic-ring bias energy of a pose, the sum of ring_bias_frame over the rows: the search term, reported for
 * diagnostics (DockTask::run_score -> ScoreDiag::ring_bias -> SDF tag ud2_ring_bias_raw) and part of no score.
 */
inline Real ring_bias_energy(const std::vector<RingBiasRow>& rows, const Real* coords, Real bias_k,
                             const RingBiasParams& p){
    Real total = 0;
    RingBiasFrame fr;
    for (const auto& r : rows) total += ring_bias_frame(coords, r, bias_k, p, fr);
    return total;
}

#endif // UD2_EXT_RING_TERM_H
