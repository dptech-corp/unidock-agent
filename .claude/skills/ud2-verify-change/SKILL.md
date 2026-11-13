---
name: ud2-verify-change
description: Verify a code change to Uni-Dock2 (engine C++/CUDA, the unidock_processing package and bundled plugins) before calling it done — which unit and pytest suites to run, how to prove the change left results byte-identical on the frozen benchmark baselines, how to compare CPU and GPU runs fairly, and what counts as a regression. Use whenever you edit Uni-Dock2 code, refactor it, port a feature, or are asked whether a change "breaks anything", "changes results", or is "safe to merge".
---

# Verifying a Uni-Dock2 change

The standard: a change that is not meant to change results must leave them **byte-identical**; a change that is meant
to change them must say where, by how much, and pass its own criterion. Measure; never assume.

## 1. Build and unit suites

```bash
cmake -S unidock/unidock_engine -B build_cuda_on -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BIN=ON \
      -DUSE_CUDA=ON          # -DUSE_CUDA=OFF for the CPU build; keep every other flag identical between arms
cmake --build build_cuda_on -j
(cd scratch && <build>/test/unit/unit_test)        # it writes unit_test.log into its working directory
```

On Linux a conda env's `nvcc` can break CUDA compiler detection; point CMake at the system one. Compare the case
count before and after: new tests add to it, and a failing case is a stop.

## 2. Python suites

```bash
pytest unidock/unidock_processing/test/core                       # must pass with no plugin installed
pytest unidock/unidock_processing/test/scripts                    # receptor/ligand/protocol suites
pytest unidock/unidock_processing/plugins/template/tests unidock/unidock_processing/plugins/covalent/tests
pytest tests/test_protac_docking.py                                # needs the pybind module in api/python
```

Some failures are known and pre-existing, their count depending on the environment (FF tests that need `utop`,
two receptor-topology tests, one torsion-rotation test, `test_yaml_parsing`); `test_postprocess_full` in the PROTAC file expects an opt-in
dump. A failure outside that list is yours.

## 3. Identity against the frozen baselines

Baselines are not shipped with this repository. Keep them in a directory of your own (`<baselines>`): for each
benchmark route (re-docking, screening, template and PROTAC), the exact commands and an md5 list of the outputs
of a trusted build. Re-run a route with the new build and check it:

```bash
(cd <new run dir> && md5sum -c <baselines>/<route>.md5 | grep -c ': OK$')
```

Without baselines, use sections 2 and 4: the test suites, and old and new binaries run side by side on the same inputs.

Paths come from `UD2_SCRATCH`; never hard-code a home directory in a committed script.

## 4. Rules for fair comparisons

- **Same binary, or compare both.** A kernel edit can move GPU results without any intended change; re-run the old
  binary in the same session rather than trusting an old run.
- **CPU runs only at `ncpu: 1`.** At higher thread counts the CPU engine gives slightly different poses on a re-run.
- **One receptor DMS.** Preparing from PDB twice gives different receptors (fepfixer).
- **Timing:** one old and one new run back to back, then swap the order once; a single pair can mislead
  (a PoseX +7 % did not reproduce in the swapped pair).
- **When identity fails,** compare the engine logs' resolved parameters first, then the inputs (DMS tables), then
  the kernel SASS (`cuobjdump --dump-sass`) before calling it a real change.

## 5. Report

State the identity counts, the hit rates of both arms, the speed ratio, every test failure with its cause, and
anything not run. A change meant to alter results needs its own measured criterion (for example the bias-aware pose
pass: hbond_charge aligned at 5 of 5 seeds on CPU and CUDA).
