//
// The list of extension terms a ligand's search energy carries besides the Vina pair terms (plan step 5b), and the
// one dispatch both kernels call. A compile-time list of plain structs, one per term, read by the CPU and CUDA
// kernels through FlexParamVina::terms; no virtual calls, so the kernel keeps its register budget.
//
// Adding a term (step 6 did it for the ring bias): its struct and row data under ext/bias, a member of TermSet, its
// run-wide settings in TermSettings (filled from DockParam by term_settings_of), one line each in active_as and
// accumulate_as. The kernels (cpu/score_cpu.cpp "4.", cuda/bfgs.cuh "3b.") and the constant upload (cuda/common.cu,
// one struct) do not change; the new term's rows need their own per-ligand upload next to the bias rows'
// (cpu/task_cpu.cpp, cuda/task.cu, cuda/vina.cu).
//

#ifndef UD2_SCORE_TERMS_H
#define UD2_SCORE_TERMS_H

#include <algorithm>
#include <vector>
#include "ext/bias/bias_term.h"
#include "ext/bias/ring_term.h"

/**
 * @brief The run-wide settings of every term (DockParam::bias_type / bias_k / ring_bias), one struct. The CPU path reads
 * the copy every FlexParamVina carries (TermSet::settings, filled per ligand by DockTask::alloc_device); the CUDA
 * kernel reads the one __constant__ mirror g_term_settings (cuda/common.cuh) that init_constants uploads from the
 * same DockParam, so its guard stays a constant-memory operand and not a global load through FlexParamVina.
 */
struct TermSettings {
    int  bias_type = BT_NO;   // BiasType: BT_NO / BT_POS / BT_ALIGN
    Real bias_k = 0.3;        // bias_k: scale of every bias row's V_set (atom rows and ring rows)
    RingBiasParams ring;      // the ring term's k, s, d0 (DockParam::ring_bias)
};

/// The term settings of a run: the DockParam fields the config parser sets (config/parse.h set_bias).
inline TermSettings term_settings_of(const DockParam& dock_param){
    TermSettings s;
    s.bias_type = static_cast<int>(dock_param.bias_type);   // BiasType has no fixed underlying type
    s.bias_k = dock_param.bias_k;
    s.ring = dock_param.ring_bias;
    return s;
}

struct TermSet {
    PositionBiasTerm bias;          // ext/bias: ud2align position bias (ud2_bias table)
    RingBiasTerm ring;              // ext/bias: ud2align aromatic-ring bias (ud2_ring_bias tables), scored under BT_ALIGN
    TermSettings settings;          // the CPU path's copy of the run's settings

    /// True when some term contributes under the settings s: the CUDA kernel's guard around accumulate_as and
    /// its tile.sync (a term with no rows for this ligand contributes nothing).
    SCOPE_INLINE bool active_as(const TermSettings& s) const {
        return bias_active_as(s) || ring_active_as(s);
    }
    /// The position bias contributes: a bias type and the ligand's row table (all atoms, possibly without rows).
    SCOPE_INLINE bool bias_active_as(const TermSettings& s) const { return s.bias_type != BT_NO && bias.inds != nullptr; }
    /// The ring bias contributes: Z-align bias and ring rows for this ligand (the October gate).
    SCOPE_INLINE bool ring_active_as(const TermSettings& s) const { return s.bias_type == BT_ALIGN && ring.n > 0; }

    /// Energy and +dE/dx of every term, in order, under the settings s: the position bias on atoms lane,
    /// lane + stride, ... below natom (one lane per atom, so aux_f is written without atomics), then the ring bias
    /// on rows lane, lane + stride, ... (atomics on the device). Each term guards itself, so the plain run (every
    /// term off) adds nothing. On the device the warp is synchronised between the two (lane i's plain += on its
    /// atom must land before another lane's atomic on the same atom), as the October kernel did with its tile.sync
    /// between blocks 3b and 3c; the caller's tile.sync follows the last term.
    SCOPE_INLINE void accumulate_as(const TermSettings& s, int lane, int stride, int natom, const Real* coords,
                                    const int* types, Real& energy, Real* aux_f) const {
        const bool bias_on = bias_active_as(s);
        if (bias_on){
            bias.accumulate_as(s.bias_type, s.bias_k, lane, stride, natom, coords, types, energy, aux_f);
        }
        if (ring_active_as(s)){
#ifdef __CUDA_ARCH__
            if (bias_on) __syncwarp();
#endif
            ring.accumulate_as(s.bias_k, s.ring, lane, stride, coords, energy, aux_f);
        }
    }

    /// active_as / accumulate_as under this struct's own settings (the CPU path; serial: lane 0, stride 1).
    SCOPE_INLINE bool active() const { return active_as(settings); }
    SCOPE_INLINE void accumulate(int natom, const Real* coords, const int* types, Real& energy, Real* aux_f) const {
        accumulate_as(settings, 0, 1, natom, coords, types, energy, aux_f);
    }
};

/**
 * @brief One ligand's TermSet on the host, for the pose pass (DockTask::run_score, plan step 7): the position-bias
 * rows flattened once by flatten_bias_rows, the layout the CPU fill (cpu/task_cpu.cpp) and the CUDA fill
 * (cuda/task.cu) share, so both builds evaluate the same rows; the ring term on the ligand's own rows
 * (UDFlexMol::ring, as the CPU fill points it); the run's settings (term_settings_of). energy() is TermSet::accumulate,
 * the code both kernels run, at one pose's coordinates, its force going to a scratch buffer that is dropped.
 * Points into its own vectors and into the ligand, so it is not copied and the ligand outlives it. One per ligand
 * and thread: energy() writes the scratch buffer.
 */
struct HostTermSet {
    std::vector<int> bias_inds;     // natom * 2
    std::vector<Real> bias_params;  // bias.rows.size() * 5
    std::vector<Real> scratch_f;    // natom * 3, the dropped +dE/dx
    const int* types;               // the ligand's vina types (the hydrogen switch of the Z-align potential)
    int natom;
    TermSet terms;

    HostTermSet(const UDFlexMol& m, const TermSettings& s)
        : bias_inds(m.natom * 2), bias_params(m.bias.rows.size() * 5), scratch_f(m.natom * 3),
          types(m.vina_types.data()), natom(m.natom){
        flatten_bias_rows(m, bias_inds.data(), bias_params.data());
        terms.bias.inds = bias_inds.data();
        terms.bias.params = bias_params.data();   // read only through an atom's row range, so never when empty
        terms.ring.n = static_cast<int>(m.ring.rows.size());
        terms.ring.rows = m.ring.rows.empty() ? nullptr : m.ring.rows.data();
        terms.settings = s;
    }
    HostTermSet(const HostTermSet&) = delete;
    HostTermSet& operator=(const HostTermSet&) = delete;

    /// Energy of every term at coords (natom * 3), in the order the search adds them (atom bias, then ring bias);
    /// 0 when every term is off under the settings.
    Real energy(const Real* coords){
        std::fill(scratch_f.begin(), scratch_f.end(), Real(0));
        Real e = 0;
        terms.accumulate(natom, coords, types, e, scratch_f.data());
        return e;
    }
};

#endif // UD2_SCORE_TERMS_H
