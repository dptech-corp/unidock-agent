---
name: ud2-template-docking
description: Core-constrained (template) docking with Uni-Dock2 — place a ligand's core on a reference ligand's crystal pose and dock only the rest, for congeneric series, FEP / ABFE pose generation and "dock this analogue like the parent". Covers the `template:` YAML block, the --template_docking options, automatic MCS or manual core maps, and the atom_mapper tree workflow (tree_align + scheduled_docking). Use whenever the user mentions template docking, constraint docking, MCS-guided docking, a reference/parent pose to keep, atom_mapper, tree_align or scheduled_docking. For a soft pull toward a template without fixing a core use ud2-bias-align.
---

# Template (core-constrained) docking

## What it does

The core that maps onto the reference ligand is placed on the reference coordinates and held there; only the
non-core torsions are searched, and the ligand root does not translate or rotate (`constraint_docking`). Use it when
the binding mode of the shared core is known. When only "stay close to the template" is wanted, use the bias plugin
(ud2-bias-align): the search stays free and the result can leave the template.

The template plugin ships with the core package; nothing extra to install.

## One ligand or a set with known maps

```yaml
template:
  template_docking: true
  reference_sdf_file_name: reference.sdf          # the reference pose
  core_atom_mapping_dict_list: null               # null = automatic RDKit MCS against the reference
# or one map per ligand: [{reference_atom_idx: query_atom_idx, ...}, ...]
```

```bash
unidock2 docking -r receptor.dms -l ligands.sdf -cf config.yaml --template_docking \
    --reference_sdf_file_name reference.sdf [--core_atom_mapping_dict_list '[{"0": 14, "1": 15}]']
```

- The box centre defaults to the reference's centre when the run gives (0, 0, 0).
- Old YAMLs put these keys under `Preprocessing:`; they still work, with a deprecation warning.
- No CNN stage for template runs: they keep the engine order.

## A congeneric series: the atom_mapper workflow

atom_mapper is a separate package (repo `smallmolecule/mol_align/atom_mapper`, use `main`). It builds an MCS tree over
the series and docks it parent by parent, each child templated on an already docked parent.

```bash
python -m atom_mapper.main tree_align -i lig*.sdf -r reference.sdf -o tree.json -n 6
python -m atom_mapper.main scheduled_docking -p receptor.dms -r reference.sdf -l lig*.sdf -j tree.json -o out -c 0.7
```

- **Dependencies:** numba and rustworkx (not in a plain unidock2 environment); `atom_mapper.main` turns a missing
  import into a misleading "unrecognized arguments" error.
- **c-MCES must be built on each machine** (`atom_mapper/modules/mcs/cmces_engine/build.sh`; it is the default seed).
  Without it the RASCAL fallback finds no MCS for most analogues and they fail.
- **The reference needs a title** (`_Name`). An untitled reference enters the tree as `''` and every ligand fails.
  Write a titled copy; never edit the input.
- **`-c` is the shape-overlap cutoff:** 0.7 by default, 0.35 for structurally diverse sets.
- **Prepare the receptor DMS once** (`unidock2 protein_prep`) and pass it with `-p`.
- **Output** is `out/<ligand>.sdf`, one pose per ligand.

## Check the result

- In the scheduled_docking log, `not promoted as ref` and `greater than 10` mark ligands whose templated pose
  clashes. Such a pose is still written to the output, with no flag in the SDF. A scaffold hop that only maps a few
  atoms usually lands here: dock it free instead and compare.
- Core RMSD to the parent pose should be a few tenths of an Å; larger means the map or the parent was wrong.
