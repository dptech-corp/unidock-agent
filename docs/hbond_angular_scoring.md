# Bidirectional Angular HBond Scoring

## Overview

Standard Vina scoring treats hydrogen bonds as distance-only interactions between
donor (D) and acceptor (A) heavy atoms. This ignores the D-H...A angle, allowing
geometrically poor H-bonds (e.g. 90° bends) to receive full energy credit.

The angular HBond scoring system applies a smooth angle-dependent weight **w** ∈ [0, 1]
to the Vina HBond energy term **during** MC/BFGS optimization, enabling the optimizer
to steer poses toward proper H-bond geometry. It handles both directions:

- **Ligand donor → Receptor acceptor** (ligand H coordinates from pose)
- **Receptor donor → Ligand acceptor** (receptor H coordinates from DMS bond table)

## Architecture

Angular HBond scoring runs on **both backends** and is applied inside MC/BFGS, so the
optimizer steers poses toward good H-bond geometry rather than penalising them afterwards.

| Stage | Where | Angular HBond |
|-------|-------|---------------|
| `run_search()` (MC), `run_refine()` (BFGS) | GPU (`cuda/bfgs.cuh`) or CPU (`cpu/mc_cpu.cpp`) | yes, inside `cal_e_f_warp` / `cal_e_f_cpu` |
| `run_score()` (final rescoring) | CPU (`score/score.h`) | yes, via `angular_hbond_pair_weight` |
| Per-atom decomposition (`--decomp` only) | CPU | `rescore_hbond()` binary 100 deg check, see below |

Inputs, built once per receptor and per ligand:

```
DMS receptor (bond + particle tables) --> RecDonorHTopo   (donor -> H coords, antecedent, rotatable flag)
ligand topology (bonds < 1.3 A)       --> BondedHTopo     (heavy atom -> H indices)
```

Both are uploaded to the device: `common.cu` copies `USE_ANGULAR_HBOND` to a device constant
and `vina.cu` uploads the receptor donor-H side table, so the GPU kernel evaluates the same
weight and the same gradients as the CPU.

## C++ Engine

### Data Structures

**RecDonorHTopo** — receptor donor → bonded-H coordinate lookup (CSR format):

```cpp
struct RecDonorHTopo {
    int*  h_start;   // size: natom + 1  (row pointers)
    Real* h_coords;  // size: total_h * 3  (x, y, z)
    int   natom;     // heavy atoms in receptor box
    int   total_h;   // total bonded-H entries
};
```

Built once from the DMS bond table during `read_receptor_dms_full()`. Receptor H atoms
are stripped from `fix_mol`, so their coordinates are stored explicitly.

**BondedHTopo** — ligand heavy atom → bonded-H index lookup (CSR format):

```cpp
struct BondedHTopo {
    int* h_start;    // size: natom + 1  (row pointers)
    int* h_indices;  // size: total_h  (indices into pose_coords)
    int  natom;
    int  total_h;
};
```

Built per-ligand during task init via distance check (< 1.3 Å). Stores indices
rather than coordinates because ligand H positions update every pose.

### Angular Weight Function

```
angular_hbond_weight(cos_dha) -> w in [0, 1]      (score/vina.h)
```

Linear in cos(D-H...A) between 90 deg (no credit) and 140 deg (full credit), clamped outside:

```
cos_zero = ANGULAR_HBOND_COS_ZERO = cos(90 deg)  =  0.0
cos_full = ANGULAR_HBOND_COS_FULL = cos(140 deg) = -0.76604444
w        = clamp((cos_zero - cos_dha) / (cos_zero - cos_full), 0, 1)
dw/d(cos) = -1 / (cos_zero - cos_full) = -1.3054 inside the ramp, 0 in both clamped regions
```

| D-H...A angle | cos(angle) | Weight w | Meaning                    |
|---------------|------------|----------|----------------------------|
| <= 90 deg     | >= 0.000   | 0        | no HBond credit            |
| 100 deg       | -0.174     | 0.23     | mostly removed             |
| 120 deg       | -0.500     | 0.65     | partial credit             |
| 140 deg       | -0.766     | 1        | full credit                |
| 140 - 180 deg | <= -0.766  | 1        | full credit (plateau)      |

Calibrated on the Uni-Dock-Benchmarks crystal poses: this ramp leaves fixed-donor hydrogen
bonds ~88 % of their Vina energy, where the earlier 100 -> 180 deg quadratic ramp kept 63 %
and gave a 150 deg hydrogen bond only 70 % credit.

**Free-rotor donors.** For -OH, -SH, pyramidal -NH2 and -NH3+ (and water), the prepared
hydrogen position carries no information, so the angle is evaluated with the best hydrogen on
the X-D rotation cone (`hbond_cone_cos`). Planar -NH2 (amide, guanidinium, aniline-like) keeps
its prepared hydrogens: the two cases are told apart geometrically, by the X-N-H1 + X-N-H2 +
H1-N-H2 angle sum, below 345 deg counting as pyramidal. A donor with no hydrogen and no heavy
antecedent (water oxygen, metal) has no directional information and keeps w = 1.

### Scoring Loop Integration

In `cal_e_f_cpu()`, `cal_e_f_warp()` (GPU) and `angular_hbond_pair_weight()` (rescoring), for
each ligand-receptor pair whose Vina types can hydrogen bond:

1. Compute the base HBond energy: `e_hb = weight_hbond * vina_hbond(d)`
2. **Direction 1** (lig donor -> rec acceptor): ligand bonded H from `BondedHTopo`, or the cone
   hydrogen for a free rotor, giving w1
3. **Direction 2** (rec donor -> lig acceptor): receptor bonded H from `RecDonorHTopo`, same
   cone rule, giving w2
4. `best_w = max(w1, w2)` — for donor-acceptor pairs on both sides the better geometry wins
5. Adjust: `e_pair += (best_w - 1) * e_hb`. Since `e_hb <= 0`, w = 0 removes the credit entirely
   and w = 1 leaves it untouched; the base term was already added, so this is not a second term
6. Accumulate the angular gradient on the movable atoms (H and D for direction 1, A for
   direction 2). Ligand hydrogens are therefore not skipped in force accumulation

### Configuration

```cpp
// cpu/common_cpu.cpp — set from DockParam
g_cpu_constants.USE_ANGULAR_HBOND          =  dock_param.angular_hbond;   // default true
g_cpu_constants.USE_POST_BFGS_HBOND_CORR   = !dock_param.angular_hbond;   // legacy, mutually exclusive
```

Controlled via `DockParam::angular_hbond` (default `true`, CLI `--no-angular-hbond`). The two
mechanisms are mutually exclusive by construction: the legacy binary post-BFGS correction
(`hbond_angle_correction_cpu()`, ligand-donor direction only, angle < 100 deg) switches on only
when the smooth angular weight is off, so the angle is never counted twice.

### rescore_hbond() — decomposition only

`rescore_hbond()` is **not** part of `vina_pose_score`. It is called from `DockTask.cpp` only
inside the `--decomp` branch, where it masks geometrically invalid hydrogen bonds in the
per-atom and per-residue decomposition output:

- **Lig donor direction**: find bonded H by distance, require D-H...A angle > 100 deg and
  H...A distance < 3.0 A
- **Rec donor direction**: use `RecDonorHTopo` for the angle check; fall back to D...A < 3.5 A
  when the donor-H table is unavailable
- Invalid HBonds: zero the hbond term and recompute the total

Its 100 deg binary threshold is independent of the deployed 90 -> 140 deg ramp, so a pose score
and its decomposition can disagree for marginal geometry.

### Key Constants

| Parameter                        | Value      | Usage                                          |
|----------------------------------|------------|------------------------------------------------|
| `ANGULAR_HBOND_COS_ZERO`         | cos(90 deg)  | w = 0 at or below this angle (deployed ramp)   |
| `ANGULAR_HBOND_COS_FULL`         | cos(140 deg) | w = 1 at or above this angle (deployed ramp)   |
| Pyramidal / planar NH2 threshold | 345 deg    | angle sum below this = free rotor, use the cone |
| Bonded-H distance cutoff         | 1.3 A      | detect covalently bonded H                      |
| Legacy angle threshold           | 100 deg    | `hbond_angle_correction_cpu()` and `rescore_hbond()` only |
| H...A distance cutoff            | 3.0 A      | `rescore_hbond()` geometry check                |
| D...A fallback distance          | 3.5 A      | `rescore_hbond()` when receptor H unavailable   |
| Vina HBond weight                | -0.587439  | affinity weight set; search -0.767, rescore -0.791096 |

## GPU Status

Implemented. The GPU evaluates the same angular weight and gradients as the CPU during
sampling, so poses are steered toward good H-bond geometry rather than corrected afterwards.

- **Kernel**: `cuda/bfgs.cuh` — `hbond_angular_pair_warp()` is called from `cal_e_f_warp()` for
  both directions; hydrogens are kept in the force accumulation so the angular gradient reaches
  the ligand H, D and A atoms
- **Device uploads**: `cuda/common.cu` copies `USE_ANGULAR_HBOND` to a device constant;
  `cuda/vina.cu` uploads the receptor donor-H side table; `cuda/task.cu` carries the per-ligand
  bonded-H topology
- **Tests**: `test/unit/cuda/test_score_gpu_vs_cpu.cpp` checks GPU against CPU for both
  directions with the term on and off, and requires a bent geometry to score worse;
  `test/unit/cpu/test_angular_hbond.cpp` adds finite-difference gradient checks for both
  directions and for cone hydrogens

## Python (ud2align)

### New Aligner Presets

**HBondAligner** — H-bond-only alignment bias:

- Pairs template acceptors with query acceptors, template donors with query donors
- Produces `BiasEntry(query_idx, template_idx, v_set=w_hbond)` for each match
- Sparse: avoids full N×M similarity matrix, directly iterates donor/acceptor sets
- Default weight: `w_hbond=4.0`

**HBondChargeAligner** — H-bond + formal charge:

- Two-phase: H-bond pairing (same as above) + formal charge matching
- Atoms with non-zero formal charge on both query and template are paired
- Weights accumulate per pair via dictionary merge
- Default weights: `w_hbond=1.3`, `w_formal_charge=4.0`

### Bias Injection

`inject_bias_entries_into_dms(bias_entries, dms_path, template_coords)`:

- Writes directly to SQLite `ud2_bias` table
- Schema: `(atom_id, ref_x, ref_y, ref_z, v_set, r)`
- Each entry maps query atom → template 3D position with bias strength

`build_hbond_bias_pairs(mol_q, mol_t, ref_atom_indices)`:

- Helper that instantiates HBondAligner internally
- Returns `[[template_idx, query_idx], ...]` list for legacy compatibility

### Bias Priority in perform_align()

Three injection paths, evaluated in order of precedence:

1. **Pre-computed `bias_entries`** — caller provides `List[BiasEntry]` directly
2. **Legacy `bias_pairs`** — caller provides `[[t_idx, q_idx], ...]` with similarity matrix
3. **Aligner-based** — calls `aligner.compute_bias_entries()` automatically

### Facade Functions

| Function                   | Receptor | Optimization | Use case                   |
|----------------------------|----------|--------------|----------------------------|
| `perform_mol_align()`      | No       | Yes          | Ligand-only alignment      |
| `perform_mol_align_score()`| No       | Score only   | Similarity scoring         |
| `perform_dock_align()`     | Yes      | Yes          | Docking with alignment     |
| `perform_dock_align_score()`| Yes     | Score only   | Docking score evaluation   |

All accept `aligner`, `bias_entries`, `bias_pairs`, and `ref_atom_indices` parameters.

### Public API (__init__.py)

```python
from ud2align import (
    HBondAligner,
    HBondChargeAligner,
    get_aligner_by_name,       # Factory: "hbond" | "hbond_charge" | "zalign"
    perform_mol_align,
    perform_dock_align,
    inject_bias_entries_into_dms,
    BiasEntry,
)
```
