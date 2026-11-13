#!/usr/bin/env python3
"""
PROTAC Ternary Complex Docking Tutorial — MZ1-BRD4-VHL
=======================================================

This tutorial demonstrates the full Uni-Dock2 PROTAC docking pipeline
using the MZ1 degrader (BRD4–MZ1–VHL ternary complex, PDB: 5T35).

Workflow
--------
1. **Prepare input**: Constrained-embed PROTAC to both pharmacophores,
   transform VHL, build enriched DMS files for ligand + receptor.
2. **Run docking**: Uni-Dock2 engine samples PROTAC linker conformations
   with frozen pharmacophore–protein contacts.
3. **Post-process**: Reconstruct PROTAC SDF, VHL PDB, full complex PDB.
4. **Cluster**: Group poses by heavy-atom RMSD, extract representatives.

Required input files (all in MZ1-BRD4-VHL/):
  - brd_lig.sdf      Target pharmacophore (TP) — BRD4 warhead pose
  - vhl_lig.sdf      E3 pharmacophore (EP) — VHL warhead pose
  - protac_random.sdf PROTAC molecule (MZ1, 131 atoms with H)
  - BRD4.pdb         Target receptor (bromodomain-4)
  - VHL.pdb          E3 ligase (VHL)

Usage
-----
    cd examples/protac_docking
    python run_tutorial.py                     # quick test (1 run, 8 exhaustiveness)
    python run_tutorial.py --full              # production (8 runs, 128 exhaustiveness)
    python run_tutorial.py --full --n-runs 4   # customised production run
"""
import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="PROTAC docking tutorial — MZ1-BRD4-VHL"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run production settings (8 runs, 128 exhaustiveness). "
             "Default is quick mode (1 run, 8 exhaustiveness).",
    )
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--exhaustiveness", type=int, default=None)
    args = parser.parse_args()

    input_dir = os.path.join(SCRIPT_DIR, "MZ1-BRD4-VHL")
    output_dir = os.path.join(SCRIPT_DIR, "output")

    # Validate input files exist
    required = ["brd_lig.sdf", "vhl_lig.sdf", "protac_random.sdf",
                "BRD4.pdb", "VHL.pdb"]
    for fname in required:
        path = os.path.join(input_dir, fname)
        if not os.path.isfile(path):
            logger.error("Missing input file: %s", path)
            sys.exit(1)

    # Select parameters
    if args.full:
        n_runs = args.n_runs or 8
        exhaustiveness = args.exhaustiveness or 128
    else:
        n_runs = args.n_runs or 1
        exhaustiveness = args.exhaustiveness or 8

    logger.info("=" * 60)
    logger.info("PROTAC Docking Tutorial — MZ1-BRD4-VHL")
    logger.info("  Input:           %s", input_dir)
    logger.info("  Output:          %s", output_dir)
    logger.info("  Runs:            %d", n_runs)
    logger.info("  Exhaustiveness:  %d", exhaustiveness)
    logger.info("=" * 60)

    # Ensure this directory is on sys.path so run_protac_docking is importable
    sys.path.insert(0, SCRIPT_DIR)
    from run_protac_docking import main as run_main

    sys.argv = [
        "run_protac_docking.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--ep-sdf", "vhl_lig.sdf",
        "--tp-sdf", "brd_lig.sdf",
        "--protac-sdf", "protac_random.sdf",
        "--receptor-pdb", "BRD4.pdb",
        "--e3-pdb", "VHL.pdb",
        "--n-runs", str(n_runs),
        "--exhaustiveness", str(exhaustiveness),
        "--mc-steps", "20",
        "--opt-steps", "15",
        "--refine-steps", "5",
        "--num-pose", "1000",
        "--energy-range", "100.0",
        "--seed", "42",
    ]
    run_main()


if __name__ == "__main__":
    main()
