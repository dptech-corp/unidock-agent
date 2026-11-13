---
name: ud2-covalent-docking
description: Covalent docking with Uni-Dock2 — a ligand bonded to a receptor residue (cysteine warheads, acrylamides, nitriles, boronic acids, serine/lysine adducts). Covers the ligand SDF annotations the covalent mode needs, the `covalent:` YAML block, the --covalent_* options and the runner API, and what the mode does and does not do. Use whenever the user mentions covalent docking, a warhead, a covalent inhibitor, or docking a ligand attached to a residue.
---

# Covalent docking

## What it does

The ligand SDF carries copies of the receptor residue atoms it is bonded to. The covalent mode removes those copies
from the ligand, roots the ligand at the anchor atom, and docks with `constraint_docking` on, so the attachment stays
put and only the ligand's torsions move. The covalent plugin ships with the core package.

## Input contract

The ligand SDF holds the ligand plus the residue atoms (for a cysteine: CA, CB, SG and their hydrogens) in their
crystal or modelled positions, and five properties listing those atoms, comma-separated and in the same order:

| property | example (1EWL) |
|---|---|
| `covalent_atom_indices` | `6,5,4,34,31,32` (atom indices in this SDF) |
| `covalent_atom_names` | `CA,CB,SG,HA,HB3,HB2` |
| `covalent_residue_names` | `CYS,CYS,CYS,CYS,CYS,CYS` |
| `covalent_residue_indices` | `25,25,25,25,25,25` |
| `covalent_chain_indices` | `A,A,A,A,A,A` |

The first entry is the anchor. See `examples/covalent_docking/1EWL/`.

## Run

```yaml
covalent:
  covalent_ligand: true
  covalent_residue_atom_info_list: [["", "CYX", 25, "CA"], ["", "CYX", 25, "CB"], ["", "CYX", 25, "SG"]]
  # anchor, bond start, bond end: [chain, residue name, residue index, atom name]
```

```bash
unidock2 docking -r receptor.dms -l covalent_mol.sdf -c X Y Z -cf config.yaml --covalent_ligand \
    --covalent_residue_atom_info_list '[["", "CYX", 25, "CA"], ["", "CYX", 25, "CB"], ["", "CYX", 25, "SG"]]'
```

The old `Preprocessing:` keys still work, with a deprecation warning. No CNN stage: covalent runs keep the engine order.

## Known limits

- **The receptor residue is not edited in a runner run.** The receptor-side step that would strip the bonded
  residue's atoms is reached only by the receptor builder's full analysis, which the runner never calls. The
  receptor keeps its own copy of the residue (for 1EWL: CYS 25 with HG). Inspect the pose around the anchor.
- **From a PDB the receptor preparation is not deterministic.** Prepare the receptor DMS once and dock against it
  when comparing runs.
- The shipped covalent test only checks that a pose is written. Judge quality against a crystal structure yourself.
