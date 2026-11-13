---
name: ud2-bias-align
description: Align or dock a ligand onto a template ligand's pose with a position bias, using the ud2align plugin of Uni-Dock2 — molecular alignment without a receptor (mol-align), template-biased docking in a pocket (dock-align), score-only checks, aromatic-ring (pi-stacking) bias, custom atom pairings, and the `bias:` block / --bias_* options of `unidock2 docking`. Use whenever the user wants a pose "like the crystal ligand", alignment to a template, ud2align, ud2align-cli, perform_mol_align / perform_dock_align, a bias potential, ring stacking onto template rings, or keeping a pharmacophore in place. For a hard constraint on a mapped core use ud2-template-docking.
---

# Template alignment with the bias plugin (ud2align)

## What it is

The engine carries the bias terms; ud2align decides what to pull where. It pairs query atoms with template atoms,
writes the pairs as bias rows into the ligand DMS, and switches the engine's bias on. The search stays free: the bias
is an energy term that rises when a paired atom moves away from its template position, so a pose can still leave the
template if the receptor demands it.

- **Install:** ud2align is an external package, not included in this repository. Install it separately in the
  unidock2 environment. Without it a `bias:` block or a `--bias_*` option stops with an error naming ud2align.
- **Engine:** needs `ud2` 0.4.5 or newer (0.4.6 for ring bias). ud2align refuses an older binary rather than dock
  unbiased silently.

## Choose the route

| situation | route |
|---|---|
| one or more queries, one template, no receptor (ligand-based alignment) | `ud2align-cli -q queries.sdf -t template.sdf -o out.sdf` or `perform_mol_align` |
| one or more queries in a pocket, template from a co-crystal | add `-r receptor.dms` / `perform_dock_align` |
| score a given pose against the template, no search | `--score-only` / `perform_mol_align_score`, `perform_dock_align_score` |
| many ligands in a pocket, inside a normal docking run | `bias:` block or `--bias_*` options of `unidock2 docking`, or `UnidockProtocolRunner(..., plugins=[BiasPlugin(...)])` |

A query SDF with several records is aligned in one engine run (one merged ligand DMS, bias rows per query) and the
output holds `--topn` poses per query in input order. Pass the whole set in one file: on the GPU one call per ligand
costs about 0.34 s (mostly engine start-up), a 34-ligand file 0.04 s per ligand (box, p38).

```bash
ud2align-cli -q query.sdf -t template.sdf -r receptor.dms -o out.sdf --aligner hbond --bias-weight 0.05
```

```yaml
bias:
  template_sdf: template.sdf
  aligner: zalign            # zalign | hbond | hbond_charge
  bias_k: 0.05
  bias_type: align           # align (Z-align well) | pos (Gaussian)
  ring_bias: false           # true adds the aromatic-ring term
  translate: true            # move each ligand to the template centroid first
  # ref_atom_indices: [...]  restrict the template atoms;  ref_ring_indices: [...] restrict the template rings
```

## Choose the aligner and strength

- **`zalign`** (default): element, charge, bond and ring environment. The general choice.
- **`hbond`**: pairs donors with donors and acceptors with acceptors. Use when the template's hydrogen-bond pattern
  is what must be kept.
- **`hbond_charge`**: hbond plus formal charge.
- **`bias_k` 0.05** is the tested default. Larger values hold the template harder at the cost of interaction score.
- **Ring bias** (`ring_bias: true`): stacks query aromatic rings parallel onto template rings. On the CPT / topo I-DNA
  case it lifted top-1 under 1 Å from 63 to 71 of 71 parents. It only applies with `bias_type: align`.
- Custom pairings: `bias_entries` (explicit query atom, template atom, weight) override everything;
  `ref_atom_indices` keeps only some template atoms.
- **Ligands the template cannot bias.** `hbond` pairs only donors with donors and acceptors with acceptors, and
  `hbond_charge` also charged atoms of the same formal charge, so a query without a matching atom gets no bias entry.
  `unidock2 docking` (SDF ligands) and `ud2align-cli` skip such a ligand with a warning, list it in
  `skipped_ligands.csv` next to the output (source file, record, title, reason) and process the others. The run stops
  with a warning and no output when every ligand is skipped or when the template has no usable atom among
  `ref_atom_indices` (`perform_align` and `ud2align-cli` report that as an error). An `hbond_charge` template with
  only one of its two terms present warns and continues.

## Check the result

Measure the pose against the template; do not trust the score alone:

```bash
python <skill>/scripts/check_alignment.py --template template.sdf --poses out.sdf
```

It prints the MCS RMSD to the template and the polar match (mean distance from each template donor, acceptor and
charged atom to the nearest query atom of the same kind). The shipped references pass at MCS RMSD < 2.5 Å and
polar match < 2.0 Å (< 2.5 Å for the receptor-free 1E9 case). The script prints how many atoms the MCS covers:
when it is only a few (the 7SJ3 query and template share 2), the MCS RMSD says little and the polar match carries
the judgement.

## What to expect in the output

- Poses are ordered by `vina_pose_score` + bias energy; `vina_pose_score` and the affinities stay unbiased, and the
  bias is in `ud2_bias_energy` (`ud2_ring_bias_raw` for the ring part).
- Affinities are relative to the written top pose.
- The CNN rescoring stage is skipped for biased runs, so the bias-aware order is kept.
- CPU and CUDA can land in different basins (one pose of the 7SJ3 hbond_charge case differed); judge both by the
  check above, not by identity to a stored pose.
