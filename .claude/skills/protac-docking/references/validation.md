# Diagnosing a run: is it sampling or scoring?

When docking fails to produce a pose you believe, there are three distinct
causes and they need different fixes. Establish which one you have before
changing anything.

1. **The scoring function does not prefer the right answer** — no amount of
   sampling will help.
2. **The torsion tree cannot represent the right answer** — the pose is outside
   the search space entirely.
3. **The search never reached it** — scoring and representation are fine; you
   need more or better sampling.

Each has its own experiment. Run them in this order: they get progressively more
expensive, and an early failure makes the later ones moot.

## 1. Does scoring prefer the reference pose?

Score the reference geometry as a single point and compare against your docked
poses. If the reference does not win, the scoring function is the ceiling.

`task="score"` sets `randomize=false`, `mc_steps=0`, and returns the input pose
untouched — a true single-point evaluation. Build a ligand DMS whose coordinates
are the reference pose (same prep path, but with both warheads pinned to their
reference positions and the E3 left untransformed), then:

```python
run_docking_pipeline(
    ligand_dms_path=..., receptor_dms_path=...,
    output_dir=...,
    center_x=..., center_y=..., center_z=...,
    size_x=100.0, size_y=100.0, size_z=100.0,
    task="score", search_mode="free", constraint_docking=True,
)
```

Run it in a **clean subprocess** that imports `pipeline` before rdkit or scipy,
otherwise it segfaults on a duplicate OpenMP runtime.

Comparing scores across runs is not straightforward: the reported affinity is
`(inter + box_penalty) / (1 + 0.05846 * n_torsions)` referenced to the intra
energy of that run's rank-1 pose. Because each parallel run picks its own
reference pose, scores from different runs are offset relative to each other —
on the benchmark by up to 1.7 kcal/mol, a third of the whole dynamic range. For
an apples-to-apples comparison recompute every pose as
`(vina_inter + vina_box_penalty) / (1 + 0.05846 * n_tors)`.

On MZ1 the crystal pose scored −13.64 against a best docked −12.61, i.e. the
scoring function did have its global minimum in the right place. That result is
what shifted the diagnosis to sampling.

## 2. Can the torsion tree reach it?

Comparing torsion values between your prep structure and the reference is
**not** a valid test — the linker can compensate for frozen bonds that differ,
so disagreeing dihedrals do not imply unreachability. Doing that comparison
naively suggests the pose is out of reach when it is not.

The valid test is an optimisation: vary the active torsions to minimise
deviation from the reference (multi-start, since the objective is rugged), and
see how close you can get. Superpose on the frozen root fragment first, then
measure the E3 warhead.

On MZ1 the floor was 0.29 Å with the shipped 11 torsions — the crystal pose was
comfortably representable, and freeing additional frozen bonds bought only
0.07 Å. Also check the resulting geometry is not strained (compare MMFF energy
against the reference) or you may be measuring an artifact.

## 3. Did the search get there?

Now compare what docking actually produced:

```bash
python <skill>/scripts/compare_to_reference.py \
    --reference crystal_ligand.sdf --poses out/poses/representatives_protac.sdf
```

If the floor is ~0.3 Å, scoring prefers the reference, and docking returns 3 Å,
the gap is sampling. Raise `--exhaustiveness` before anything else.

## Convergence radius: how close is close enough?

Worth measuring once for a new system. Perturb the reference pose's linker
torsions by increasing amounts, re-dock from each with `randomize=False`, and
see from how far away the search still falls back into the right basin. When
perturbing, make sure the rotation moves the E3 side rather than the frozen root
— RDKit's `SetDihedralDeg` moves the side holding the fourth atom, so order the
quadruple accordingly, or you will swing the root warhead out of its pocket and
measure nonsense.

On MZ1 the radius was under ~1 Å of E3-warhead deviation. A basin that tight
means the sampler essentially has to land on the answer; it will not be refined
into from several Å away.

## Reporting honestly

Two framings that are easy to slip into and are not supported by the method:

- *"Docking reproduced the crystal ternary complex."* Both warheads were given.
  The prediction is the linker conformation and the E3 placement.
- *"The top-ranked pose is the answer."* On MZ1 the crystal-like pose ranked first
  by `vina_pose_score` in 5 of 5 seed sets, but that is one system. Report the rank
  distribution, or at least look at the top few. Do not rank merged runs by
  `vina_binding_free_energy`: it is relative to each run's top pose.

Useful things to quote alongside RMSD: the E3 heavy-atom deviation without
superposition, and whether the native protein–protein interface is rebuilt
(contact count under 4 Å versus the reference, plus clash count under 2 Å). A
pose can look good by ligand RMSD while the protein interface is wrong.
