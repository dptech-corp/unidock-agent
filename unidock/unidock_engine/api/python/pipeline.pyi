from typing import Literal

__version__: str  # the engine version (VERSION_NUMBER), since ud2 0.4.6

def run_docking_pipeline(
    ligand_dms_path: str,
    receptor_dms_path: str,
    output_dir: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float = 30.0,
    size_y: float = 30.0,
    size_z: float = 30.0,
    task: Literal['screen', 'score', 'benchmark_one', 'mc'] = 'screen',
    search_mode: Literal['fast', 'balance', 'detail', 'free'] = 'balance',
    exhaustiveness: int = 512,
    randomize: bool = True,
    mc_steps: int = 40,
    opt_steps: int = -1,
    refine_steps: int = 5,
    num_pose: int = 10,
    rmsd_limit: float = 1.0,
    energy_range: float = 5.0,
    seed: int = 1234567,
    constraint_docking: bool = False,
    use_tor_lib: bool = False,
    cluster_atom_indices: list[list[int]] = [],
    decomp: bool = False,
    ff_minimize_steps: int = 0,
    bias: str = "no",
    bias_k: float = 0.0,
    gpu_device_id: int = 0,
) -> None:
    """
    Run docking pipeline using the Uni-Dock2 engine.

    Args:
        ligand_dms_path: Path to the enriched ligand DMS file
        receptor_dms_path: Path to the enriched receptor DMS file
        output_dir: Directory to store output files
        center_x: X-coordinate of box center
        center_y: Y-coordinate of box center
        center_z: Z-coordinate of box center
        size_x: Box size in X dimension
        size_y: Box size in Y dimension
        size_z: Box size in Z dimension
        task: Task type ("screen", "score", "benchmark_one", or "mc")
        search_mode: Search mode ("fast", "balance", "detail", or "free")
        exhaustiveness: Number of independent runs
        randomize: Whether to randomize starting positions
        mc_steps: Number of Monte Carlo steps
        opt_steps: Number of optimization steps
        refine_steps: Number of refinement steps
        num_pose: Number of poses to generate
        rmsd_limit: RMSD limit for clustering
        energy_range: Maximum energy difference from best pose
        seed: Random seed
        constraint_docking: Whether to use constrained docking
        use_tor_lib: Whether to use torsion library
        cluster_atom_indices: Per-ligand atom indices for clustering RMSD
        decomp: Whether to decompose scoring
        ff_minimize_steps: Number of FF minimization steps (0 = off)
        bias: Bias type ("no", "pos", or "align")
        bias_k: Bias force constant
        gpu_device_id: GPU device ID to use
    """
    ...
