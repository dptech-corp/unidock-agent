# Energy Decomposition

## Overview

Uni-Dock2 supports per-atom and per-residue energy decomposition of docking poses.
When enabled, each output pose in the SDF file contains detailed breakdowns of
the scoring function terms, allowing users to identify key protein-ligand
interactions at both atomic and residue resolution.

Three levels of decomposition are available:

| Level | Scope | Terms |
|-------|-------|-------|
| Per-atom Vina | Each ligand atom in SDF order (hydrogens 0) | gauss1, gauss2, repulsion, hydrophobic, hbond; plus elec, arom_contact, polar_env when their pose weight is nonzero |
| Per-atom FF | Each ligand atom in SDF order | inter_lj, inter_coulomb, intra_lj, intra_coulomb, torsion; plus bond, angle when any atom of the pose has a nonzero bond or angle term |
| Per-residue Vina | Each receptor residue with a heavy atom within box + 8 A | same terms as per-atom Vina |

## Enabling Decomposition

### Via YAML config (ud2 binary)

```yaml
Settings:
  decomp: true
```

### Via Python API

```python
runner = UnidockProtocolRunner(
    ...,
    decomp=True,
)
```

### Via CLI

```bash
unidock2 docking --decomp ...
```

The `ud2` engine binary has no flag; it reads `Settings.decomp` from its YAML.

## Which Energy Is Decomposed

The Vina decomposition splits the **inter part of `vina_pose_score`**, the score that
orders the poses of a ligand. It uses:

- **The same weight set as that pass.** The compiled rescore set (`default_rescore_weights()`
  in `src/model/model.h`, the set fitted for ranking poses), with any key `Settings.rescore_weights`
  overrides. The search runs on the compiled search set and `Settings.score_weights` overrides
  keys of that one; a key left out of either block keeps its own default, not the other set's
  value. The Python API and the `unidock2` CLI pass no weight blocks, so they run the compiled
  sets, which are the ones behind the benchmark numbers.
- **The same inter terms as `score()`**, pair by pair:
  - the five Vina terms;
  - the screened Coulomb term (`elec`);
  - the aromatic-contact term (`arom_contact`);
  - the saturated apolar-environment credit (`polar_env`);
  - the per-pair angular HBond weight (see `atom_hbond` below).
- **The same hydrogen topology** as the search: bonded ligand H and the receptor donor-H
  table.

Intra-ligand energy, torsional strain and the box penalty are not split per atom. They are
reported at the pose level as `vina_pose_intra` and `vina_pose_penalty`.

The final scoring pass uses the affinity set, Vina's own weights by default, with any key
`Settings.affinity_weights` overrides. `vina_inter`, `vina_intra`, `vina_box_penalty` and
`vina_affinity_score` come from that pass and differ from `vina_pose_*`. The decomposition
always follows the pose pass.

## Output SDF Properties

When `decomp=true`, each pose record in the output SDF gains the following
data items.

### Pose-level parts of vina_pose_score (decomp only)

| Property | Description |
|----------|-------------|
| `vina_pose_inter` | Inter energy of the pose-score pass (what the atoms and residues decompose) |
| `vina_pose_intra` | Intra-ligand energy plus weighted torsional strain of that pass |
| `vina_pose_penalty` | Box penalty of that pass |

`vina_pose_inter + vina_pose_intra + vina_pose_penalty == vina_pose_score`.

### Per-atom Vina decomposition

One value per ligand atom in SDF order, space-separated. Hydrogens have no inter pairs and
are 0 in every column. All values are weighted with the pose weight set.

| Property | Description |
|----------|-------------|
| `atom_inter_energy` | Total inter-molecular energy per atom (sum of the columns below) |
| `atom_gauss1` | Attractive Gaussian (offset=0, width=0.5) |
| `atom_gauss2` | Attractive Gaussian (offset=3, width=2.0) |
| `atom_repulsion` | Steric repulsion (d < 0) |
| `atom_hydrophobic` | Hydrophobic contact (both atoms hydrophobic) |
| `atom_hbond` | Hydrogen bond, angular-weighted per pair (see below) |
| `atom_elec` | Screened Coulomb, united-atom charges (only when the `elec` weight is nonzero) |
| `atom_arom_contact` | Ligand aromatic C x receptor ring C contact (only when the `arom_contact` weight is nonzero) |
| `atom_polar_env` | Saturated apolar-environment credit of the atom (only when the `polar_env` weight is nonzero) |

**HBond.** Each donor-acceptor pair carries the angular weight that `score()` applies:

- a linear ramp in cos(D-H...A) from 90 deg (weight 0) to 140 deg (weight 1);
- the better of the two directions (ligand donor, receptor donor);
- for rotatable donors (-OH, -NH2, -NH3+), the best hydrogen on the bond cone, or weight 1
  when the donor has no heavy neighbour to define the cone (water O);
- weight 1 for donors that have no hydrogen and are not rotatable.

A direction without hydrogen geometry has weight 0, so a pair that only it could form loses
its HBond, as in the search:

- receptor without a donor-H table (`rh_topo->natom == 0`): the receptor-donor direction is
  skipped, so ligand acceptors get no HBond from receptor donors;
- no bonded-H topology for the ligand (`h_topo` null or without `h_start`; `DockTask` builds
  one whenever `angular_hbond` is on): the ligand-donor direction is skipped for every ligand
  donor, including H-less donors that would otherwise get weight 1.

With `Settings.angular_hbond: false` the HBond term is distance-only.

**Optional columns.** Whether `atom_elec`, `atom_arom_contact` and `atom_polar_env` are written
depends only on the weight set, so every pose of a run has the same columns. With the default
weights the columns are exactly the five Vina terms. A column whose weight is nonzero but whose
input annotation is missing (for example a DMS without charges) is written as zeros, because
`score()` adds nothing for it either.

### Per-atom FF decomposition

Requires FF-parameterized DMS inputs. One value per ligand atom:

| Property | Description |
|----------|-------------|
| `atom_ff_total` | Sum of all FF components |
| `atom_ff_inter_lj` | Intermolecular Lennard-Jones |
| `atom_ff_inter_coulomb` | Intermolecular electrostatics |
| `atom_ff_intra_lj` | Intramolecular Lennard-Jones |
| `atom_ff_intra_coulomb` | Intramolecular electrostatics |
| `atom_ff_torsion` | Torsional energy |
| `atom_ff_bond` | Harmonic bond stretch, split 1/2 per atom (see below for when it is written) |
| `atom_ff_angle` | Harmonic angle bend, split 1/3 per atom (written together with `atom_ff_bond`) |

`atom_ff_bond` and `atom_ff_angle` are written for a pose when any of its atoms has a nonzero
bond or angle term. In practice that means the ligand DMS has `stretch_harm` / `angle_harm`
tables. `atom_ff_total` is the sum of the written parts.

The pose-level `ff_intra_ligand_energy` includes the bond and angle energy, so
`ff_total_energy == ff_interaction_energy + ff_intra_ligand_energy`.

Where the FF parameters come from:

- Single-ligand DMS: `Advanced.ligand_dms` / `Advanced.protein_dms` when those are set, else
  `Inputs.ligand_dms_full` / `Inputs.receptor_dms_full`.
- Merged multi-ligand DMS: always `Inputs.ligand_dms_full` / `Inputs.receptor_dms_full`, per
  ligand. `Advanced.ligand_dms` / `protein_dms` are not read for FF parameters.

FF scoring needs nonbonded parameters for **every** atom of the ligand. A ligand DMS with no
`nonbonded_param` table, or an empty one (Vina-only DMS files keep an empty one), loads no FF
and logs at info level only (a single-ligand run reads the ligand DMS twice, so the line can
appear twice). A partial table also loads no FF and adds a warning. In a merged DMS
this is decided per ligand, and one warning counts the ligands whose FF tables do not cover
every atom.

The receptor also needs every atom parameterised when its `nonbonded_param` table exists. An
empty or partial table switches the FF off for every ligand, with a warning. A receptor DMS
with no `nonbonded_param` table leaves a loaded ligand FF on without receptor terms:
`ff_interaction_energy` is 0 and FF minimisation sees the ligand alone.

Without a loaded FF, a pose gets no `ff_*` or `atom_ff_*` properties.

### Per-residue Vina decomposition

One value per receptor residue with a heavy atom within the docking box plus 8 A:

| Property | Description |
|----------|-------------|
| `residue_labels` | Space-separated residue IDs (`chain:resname:resid`) |
| `residue_inter_energy` | Total inter-molecular energy per residue |
| `residue_gauss1` | Gaussian-1 contribution per residue |
| `residue_gauss2` | Gaussian-2 contribution per residue |
| `residue_repulsion` | Steric repulsion per residue |
| `residue_hydrophobic` | Hydrophobic contact per residue |
| `residue_hbond` | Hydrogen bond per residue (angular-weighted per pair) |
| `residue_elec` | Screened Coulomb per residue (only when the `elec` weight is nonzero) |
| `residue_arom_contact` | Aromatic contact per residue (only when the `arom_contact` weight is nonzero) |
| `residue_polar_env` | Apolar-environment credit per residue (only when the `polar_env` weight is nonzero) |

The residue labels use the format `chain:resname:resid` (e.g., `A:LEU:14`).
Empty chains in the DMS default to chain `A`.

All pairwise terms go to exactly one residue. `polar_env` is a per-atom term: it saturates
over all receptor hydrophobic carbons around the atom. Each atom's credit is therefore split
over residues in proportion to that residue's share of the atom's raw soft count. The split is
proportional, not physical, but the shares sum exactly to the atom value.

### PROTAC inputs (sdf_natom)

For a PROTAC DMS with `sdf_natom` metadata, the flexible system contains the degrader plus the
E3 ligase atoms, while the SDF writes only the degrader.

- Every `atom_*` and `atom_ff_*` array covers the written molecule: the first `sdf_natom`
  flex atoms, in SDF order.
- The sum over the unwritten flex atoms of each array `atom_<X>` is written as
  `decomp_unwritten_<X>`, for example `decomp_unwritten_inter_energy`,
  `decomp_unwritten_hbond` and `decomp_unwritten_ff_total`.
- Residue arrays keep the contacts of all flex atoms, including E3-to-target contacts.
- `ud2_heavy_*` arrays cover the written heavy atoms.
- `ud2_n_heavy` and the other `ud2_*` scalars stay whole-system values, because they are what
  the score used.

For normal ligands no `decomp_unwritten_*` property is written.

## Consistency

The following identities hold per pose. They are exact in memory up to float rounding. The
absolute tolerances below cover the printed precision (`%.3f` for pose-level values, `%.4f`
for array entries). Each bound also gets a relative term, 1e-6 times the magnitude M in the
last column, for float rounding at large energies:

| Identity | File tolerance | M (sum of absolute values) |
|----------|----------------|----------------------------|
| `vina_pose_inter + vina_pose_intra + vina_pose_penalty == vina_pose_score` | 2e-3 | the three parts |
| `sum(atom_inter_energy) (+ decomp_unwritten_inter_energy) == vina_pose_inter` | 5e-4 + 5e-5 per written atom + 1e-3 | the atom values (and the unwritten sum) |
| per atom: `atom_inter_energy == sum of that atom's written term columns` | 5e-5 x (1 + number of columns) | that atom's columns |
| per term X: `sum(residue_X) == sum(atom_X) (+ decomp_unwritten_X)` | 5e-5 x (nonzero residues + atoms) | the residue and atom values of X |
| `ff_total_energy == ff_interaction_energy + ff_intra_ligand_energy` | 2e-3 | `ff_interaction_energy`, `ff_intra_ligand_energy` |
| per atom: `atom_ff_total == sum of the written atom_ff_* parts` | 5e-5 x 8 | that atom's parts |

The relative term matters only for clashing poses, where FF energies are large. Measured on
tyk2 ejm42 moved 2-3 A into its receptor and scored as given:

- `ff_total_energy` reached 8.6e8 kcal/mol, where one float step is tens of kcal/mol.
- The pose-level FF identity was off by up to 68 kcal/mol (at 5.5e8) and by up to 1.9e-7
  relative (at 3.2e8).
- The per-atom FF identity was off by up to 16 (1.3e-7 relative).
- The Vina energies stayed below 200 kcal/mol, because the repulsion is quadratic, and within
  the absolute bounds.

`vina_pose_inter` equals, bit for bit, the inter energy of the pose-score pass. The unit tests
in `test/unit/score/test_score_decomp.cpp` check the in-memory identities: the decomposition's
running sum against `score()` to a few ulp, because the compiler may contract `e += w * x`
into a fused multiply-add differently at each call site, and the regrouped atom and residue
sums to float rounding.

Decomposition outputs written before this change are not comparable:

- HBond used a binary per-atom geometry gate (`rescore_hbond`) instead of the per-pair angular
  weight.
- The Vina default weights were used whatever weight set scored the poses.

## Parsing Output in Python

```python
from rdkit import Chem

poses = [m for m in Chem.SDMolSupplier("docked.sdf", removeHs=False) if m]
mol = poses[0]

# Per-residue decomposition
labels = mol.GetProp("residue_labels").split()
energies = [float(x) for x in mol.GetProp("residue_inter_energy").split()]
hbonds = [float(x) for x in mol.GetProp("residue_hbond").split()]

# Top contributing residues
paired = sorted(zip(labels, energies), key=lambda x: x[1])
for res, e in paired[:10]:
    print(f"{res:>15s}  {e:8.3f} kcal/mol")

# Residues forming hydrogen bonds
for res, hb in zip(labels, hbonds):
    if abs(hb) > 0.001:
        print(f"{res}: HBond = {hb:.4f}")
```

Reconciling the decomposition with the pose score:

```python
def floats(mol, key):
    return [float(x) for x in mol.GetProp(key).split()]

def unwritten(mol, name):
    key = "decomp_unwritten_" + name
    return float(mol.GetProp(key)) if mol.HasProp(key) else 0.0

for mol in poses:
    inter = float(mol.GetProp("vina_pose_inter"))
    intra = float(mol.GetProp("vina_pose_intra"))
    penalty = float(mol.GetProp("vina_pose_penalty"))
    assert abs(inter + intra + penalty - float(mol.GetProp("vina_pose_score"))) < 2e-3

    atoms = floats(mol, "atom_inter_energy")
    print(f"pose inter {inter:.3f}  atom sum "
          f"{sum(atoms) + unwritten(mol, 'inter_energy'):.3f}  "
          f"residue sum {sum(floats(mol, 'residue_inter_energy')):.3f}")
```

## Architecture

```
DMS receptor (particle table)
  ├── coords, vina_types, charges  →  UDFixMol
  └── chain, resid, resname        →  UDFixMol.residues + atom_to_residue
                                       (ResidueKey: chain/resid/resname)

Scoring pipeline (DockTask::run_score), per ligand:
  1. SF.set_weights(pose_weights()); sf_pose = copy of SF
  2. score() on every pose       →  vina_pose_score = intra + inter + penalty;
                                     center[0..2] stashed per pose (--decomp)
  3. stable sort by vina_pose_score; a biased run (bias type set) sorts by
     vina_pose_score + the pose's bias energy (score/terms.h HostTermSet,
     written as ud2_bias_energy), vina_pose_score itself stays unbiased
  4. optional affinity pass       →  SF re-weighted with affinity_weights; its reference
                                     intra is the written top pose's (the first of the
                                     order of step 3), also in a biased run
  5. per output pose:
       score() with diagnostics   →  vina_inter / vina_intra / vina_affinity_score, ud2_*
       score_decomp_ff()          →  per-atom FFAtomTerms (when FF data loaded)
       score_decomp(sf_pose)      →  per-atom and per-residue VinaTerms (--decomp)

SDF writer (sdf_writer.cpp):
  Writes the pose parts and all decomposition arrays as SDF data items,
  sized to the written molecule
```

## Key Files

| File | Content |
|------|---------|
| `src/score/score.h` | `score()`, `score_decomp()` (atoms and residues) |
| `src/score/vina.h` | `VinaTerms`, `eval_ef_decomp()` |
| `src/score/rescore_hbond.h` | Binary geometric HBond check; not used by the decomposition, kept with its unit tests |
| `src/model/model.h` | `ResidueKey`, `UDFixMol` with residue mapping, `DockParam::pose_weights()` |
| `src/format/dms_reader.cpp` | Reads residue info and FF parameters from DMS |
| `src/format/sdf_writer.cpp` | Writes decomposition to SDF output |
| `src/task/DockTask.cpp` | Orchestrates decomposition in `run_score()` (`sf_pose`, pose-pass stash) |
| `src/cpu/score_ff_cpu.cpp` | `score_decomp_ff()` |
