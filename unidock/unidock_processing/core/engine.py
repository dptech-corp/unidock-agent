"""The ud2 engine run: the YAML configuration the runner writes and the subprocess / pybind11 call.

``render_yaml`` is the template that ``UnidockProtocolRunner._run_engine_subprocess`` carried; the core blocks are
rendered from an ``EngineConfig`` and the plugins' blocks follow them verbatim, except that a plugin block named
after a core block (``CORE_BLOCKS``) is merged into it (``{"Advanced": {"seed": 7}}`` puts the key where the ud2
binary reads it, and the pybind11 call passes the merged keys as keyword arguments). The engine's own top-level
``bias`` block (``{"bias": {"type": "align", "k": 0.05}}``, the bias plugin's) is rendered as such for the binary and
mapped onto the pipeline's ``bias`` / ``bias_k`` arguments by ``run_engine_pybind11``. A ud2 binary ignores a
top-level block it does not know, so the subprocess route checks the binary's version against the one that
introduced each such block (``ENGINE_BLOCK_MIN_VERSION``, ``require_engine_blocks``) instead of running the plain
docking; the same holds for the aromatic-ring bias (the bias block's ``ring:`` sub-block and the ring-bias tables of
the ligand DMS, ``RING_BIAS_MIN_VERSION``), which an older binary ignores the same way. The pybind11 route refuses
the ring tables against a pipeline module older than that version (its ``__version__``, absent before 0.4.6).
"""
from contextlib import closing
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple
import importlib
import json
import os
import re
import sqlite3
import subprocess
from shutil import which

# The blocks of the core YAML, in the order render_yaml writes them; a plugin block of one of these names is merged
# into the core block (its keys after the core keys), any other name is a new top-level block after them.
CORE_BLOCKS = ('Advanced', 'Hardware', 'Settings', 'Outputs', 'Inputs')


@dataclass
class EngineConfig:
    """The values one engine run is rendered from (the keys of the core YAML blocks)."""
    seed: int
    exhaustiveness: int
    randomize: bool
    mc_steps: int
    opt_steps: int
    refine_steps: int
    rmsd_limit: float
    num_pose: int
    rescore_pool: int
    energy_range: float
    use_tor_lib: bool
    ff_minimize_steps: int
    ligand_dms: Optional[str]
    protein_dms: Optional[str]
    cluster_atom_indices_list: Optional[List[Optional[List[int]]]]
    ncpu: int
    task: str
    search_mode: str
    constraint_docking: bool
    decomp: bool
    center: Tuple[float, float, float]
    box_size: Tuple[float, float, float]
    output_dir: str
    inputs: Dict[str, str] = field(default_factory=dict)
    working_dir: str = '.'
    gpu_device_id: int = 0

    def update(self, policy: Dict[str, Any]) -> None:
        """Apply a plugin search policy; every key must be a field of this configuration."""
        known = {f.name for f in fields(self)}
        unknown = sorted(set(policy) - known)
        if unknown:
            raise KeyError(f"search policy keys that are not engine settings: {unknown}")
        for key, value in policy.items():
            setattr(self, key, value)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, dict):
        return json.dumps(value)            # a sub-block (bias: ring: {k, s, d0}) as a YAML flow mapping
    return str(value)


def _block_lines(block: Dict[str, Any]) -> str:
    return "".join(f"  {key}: {_yaml_scalar(value)}\n" for key, value in block.items())


def split_extra_blocks(
    extra_blocks: Optional[Dict[str, Dict[str, Any]]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """The plugin blocks merged into a core block (name -> keys, one entry per CORE_BLOCKS name) and the ones that
    become top-level blocks, in their order."""
    merged: Dict[str, Dict[str, Any]] = {name: {} for name in CORE_BLOCKS}
    appended: Dict[str, Dict[str, Any]] = {}
    for name, block in (extra_blocks or {}).items():
        if name in merged:
            merged[name].update(block)
        else:
            appended[name] = block
    return merged, appended


def render_yaml(cfg: EngineConfig, extra_blocks: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """The engine YAML text: the core blocks from *cfg*, with the keys of a plugin block named after a core block
    appended to that block, then one top-level block per other entry of *extra_blocks*."""
    merged, appended = split_extra_blocks(extra_blocks)
    extra = {name: _block_lines(block) for name, block in merged.items()}
    # Build FF-related YAML lines: always pass DMS paths when available
    # (enables FF energy reporting even without minimization)
    ff_yaml_lines = ""
    if cfg.ligand_dms and cfg.protein_dms:
        ff_yaml_lines = (
            f"  ff_minimize_steps: {cfg.ff_minimize_steps}\n"
            f"  ligand_dms: {cfg.ligand_dms}\n"
            f"  protein_dms: {cfg.protein_dms}"
        )

    # Per-ligand clustering atom indices (optional)
    cluster_yaml_line = ""
    if cfg.cluster_atom_indices_list is not None:
        # Encode as YAML sequence of sequences; None/empty => [] (heavy-atom default)
        encoded = []
        for inds in cfg.cluster_atom_indices_list:
            if inds:
                encoded.append(list(inds))
            else:
                encoded.append([])
        cluster_yaml_line = f"  cluster_atom_indices: {json.dumps(encoded)}\n"

    # Build enriched DMS input lines (for DMS-only engine path)
    dms_input_lines = "\n".join(f"  {key}: {value}" for key, value in cfg.inputs.items())

    yaml_content = f"""Advanced:
  seed: {cfg.seed}
  exhaustiveness: {cfg.exhaustiveness}
  randomize: {str(cfg.randomize).lower()}
  mc_steps: {cfg.mc_steps}
  opt_steps: {cfg.opt_steps}
  refine_steps: {cfg.refine_steps}
  rmsd_limit: {cfg.rmsd_limit}
  num_pose: {cfg.num_pose}
  rescore_pool: {cfg.rescore_pool}
  energy_range: {cfg.energy_range}
  tor_lib: {str(cfg.use_tor_lib).lower()}
{extra['Advanced']}{ff_yaml_lines}
{cluster_yaml_line}
Hardware:
  ncpu: {cfg.ncpu}
{extra['Hardware']}
Settings:
  task: {cfg.task}
  search_mode: {cfg.search_mode}
  constraint_docking: {str(cfg.constraint_docking).lower()}
  decomp: {str(cfg.decomp).lower()}
  center_x: {cfg.center[0]}
  center_y: {cfg.center[1]}
  center_z: {cfg.center[2]}
  size_x: {cfg.box_size[0]}
  size_y: {cfg.box_size[1]}
  size_z: {cfg.box_size[2]}
{extra['Settings']}
Outputs:
  dir: {cfg.output_dir}
{extra['Outputs']}
Inputs:
{dms_input_lines}
{extra['Inputs']}"""
    for name, block in appended.items():
        yaml_content += f"\n{name}:\n" + _block_lines(block)
    return yaml_content


def find_ud2_binary() -> Optional[str]:
    """Find the ud2 binary. Check PATH first, then known build locations."""
    ud2_path = which('ud2')
    if ud2_path:
        return ud2_path

    # Check known build locations relative to this package
    import unidock_engine
    engine_dir = os.path.dirname(os.path.dirname(unidock_engine.__file__))
    for build_dir in ['build_cpu', 'build']:
        candidate = os.path.join(engine_dir, 'unidock_engine', build_dir, 'bin', 'ud2')
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None


def run_engine(cfg: EngineConfig, extra_blocks: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Run the ud2 docking engine. Tries the ud2 binary (subprocess) first
    to avoid shared library conflicts (msys vs pybind11), then falls back
    to the pybind11 in-process call.
    """
    ud2_bin = find_ud2_binary()
    if ud2_bin:
        run_engine_subprocess(cfg, ud2_bin, extra_blocks)
    else:
        run_engine_pybind11(cfg, extra_blocks)


# The ud2 version (VERSION_NUMBER in unidock_engine/CMakeLists.txt, printed by `ud2 --version`) that introduced each
# top-level block a plugin may render. An older binary ignores a top-level block it does not know, warns only about
# the keys it does not find and docks without the plugin's term, exit 0: require_engine_blocks refuses that.
ENGINE_BLOCK_MIN_VERSION: Dict[str, Tuple[int, ...]] = {'bias': (0, 4, 5)}

# The ud2 version that reads the aromatic-ring bias: the bias block's `ring:` sub-block and the ligand DMS tables
# ud2_ring_bias / ud2_ring_bias_atoms (plan step 6). An older binary ignores both, warns about neither and docks
# without the ring term, exit 0: require_engine_blocks refuses that too.
RING_BIAS_MIN_VERSION: Tuple[int, ...] = (0, 4, 6)
RING_BIAS_TABLES = ('ud2_ring_bias', 'ud2_ring_bias_atoms')


def _version_text(version: Tuple[int, ...]) -> str:
    return '.'.join(str(part) for part in version)


def ud2_version(ud2_bin: str) -> Tuple[int, ...]:
    """The version `ud2 --version` prints ("UD2 C++ Engine Version: 0.4.5"), as a tuple of ints; RuntimeError when
    the binary cannot be run or prints no version."""
    try:
        result = subprocess.run([ud2_bin, '--version'], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"cannot run `{ud2_bin} --version`: {error}") from error
    output = (result.stdout or '') + (result.stderr or '')
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', output)
    if result.returncode != 0 or match is None:
        raise RuntimeError(
            f"cannot read the version of the ud2 binary {ud2_bin}: `--version` exited {result.returncode} "
            f"with output {output.strip()!r}"
        )
    return tuple(int(part) for part in match.groups())


def dms_has_ring_bias(dms_path: Optional[str]) -> bool:
    """Whether the DMS file carries one of the ring-bias tables (RING_BIAS_TABLES); False for no path or a path
    that is not an SQLite file."""
    if not dms_path or not os.path.isfile(dms_path):
        return False
    try:
        with closing(sqlite3.connect(dms_path)) as conn:
            found = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN (?, ?)", RING_BIAS_TABLES
            ).fetchone()[0]
    except sqlite3.DatabaseError:
        return False                        # not a DMS: the engine reports the file itself
    return found > 0


def require_engine_blocks(
    ud2_bin: str,
    extra_blocks: Optional[Dict[str, Dict[str, Any]]],
    ligand_dms: Optional[str] = None,
) -> None:
    """Refuse to run *extra_blocks* through *ud2_bin* when the binary predates one of their top-level blocks
    (ENGINE_BLOCK_MIN_VERSION), the bias block's ``ring:`` sub-block or, when *ligand_dms* carries them, the
    ring-bias tables (RING_BIAS_MIN_VERSION): it would ignore them and run the plain docking with exit 0. The binary
    is probed (`--version`) only when one of these is present; a plain run, a merge into a core block or a block with
    no entry makes no extra call."""
    _, appended = split_extra_blocks(extra_blocks)
    needed = {name: ENGINE_BLOCK_MIN_VERSION[name] for name in appended if name in ENGINE_BLOCK_MIN_VERSION}
    ring_items = []
    if isinstance(appended.get('bias'), dict) and 'ring' in appended['bias']:
        ring_items.append("the `ring:` sub-block of `bias:`")
    if dms_has_ring_bias(ligand_dms):
        ring_items.append(f"the ring-bias tables {' / '.join(RING_BIAS_TABLES)} of {ligand_dms}")
    if not needed and not ring_items:
        return
    version = ud2_version(ud2_bin)
    too_old = sorted(name for name, minimum in needed.items() if version < minimum)
    ring_too_old = ring_items if version < RING_BIAS_MIN_VERSION else []
    if not too_old and not ring_too_old:
        return
    parts = []
    if too_old:
        parts.append("the block(s) " + ', '.join(f"`{name}:` (ud2 >= {_version_text(needed[name])})" for name in too_old))
    parts.extend(f"{item} (ud2 >= {_version_text(RING_BIAS_MIN_VERSION)})" for item in ring_too_old)
    raise RuntimeError(
        f"the ud2 binary {ud2_bin} is version {_version_text(version)} and does not read {'; '.join(parts)}: "
        "it would ignore them and run the plain docking. Rebuild the engine or put a newer ud2 first on PATH."
    )


def run_engine_subprocess(
    cfg: EngineConfig, ud2_bin: str, extra_blocks: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    """Run engine via ud2 binary as subprocess (avoids msys library conflict). A top-level plugin block the binary
    predates, or ring-bias input it predates (the bias block's ring sub-block, the ring tables of the ligand DMS), is
    refused before anything is written (require_engine_blocks)."""
    require_engine_blocks(ud2_bin, extra_blocks, cfg.inputs.get('ligand_dms_full'))
    config_yaml_path = os.path.join(cfg.working_dir, '_ud2_engine_config.yaml')
    with open(config_yaml_path, 'w') as f:
        f.write(render_yaml(cfg, extra_blocks))

    result = subprocess.run(
        [ud2_bin, config_yaml_path],
        capture_output=False,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ud2 engine failed with return code {result.returncode}")


# The engine's top-level bias block (bias: {type, k}) and the pipeline arguments its keys map onto
BIAS_BLOCK_ARGS = {'type': 'bias', 'k': 'bias_k'}


def pipeline_version(module: Any) -> Optional[Tuple[int, ...]]:
    """The version a pybind11 pipeline module reports (``__version__``, the engine's VERSION_NUMBER), as a tuple of
    ints; None for a module without one (every module built before ud2 0.4.6) or with no version in it."""
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', str(getattr(module, '__version__', '')))
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def run_engine_pybind11(cfg: EngineConfig, extra_blocks: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Run engine via pybind11 in-process call. The keys a plugin merges into a core block (CORE_BLOCKS) are passed
    as keyword arguments of the same name; the pipeline rejects one it has no argument for (TypeError). The
    engine's ``bias`` block is mapped onto the pipeline's ``bias`` / ``bias_k`` arguments (BIAS_BLOCK_ARGS; a key the
    block cannot carry is refused, among them the ring sub-block: the pipeline takes no ring parameters). A plugin
    block the pipeline has no argument for (any other name) is refused: the run is not silently made the plain one.
    A ligand DMS with the ring-bias tables is refused against a pipeline module older than RING_BIAS_MIN_VERSION or
    without a version (pipeline_version; such a module ignores the tables and docks without the ring term, no
    warning); a DMS without them runs on any module."""
    merged, appended = split_extra_blocks(extra_blocks)
    bias_block = appended.pop('bias', None)
    if appended:
        raise RuntimeError(
            f"the pybind11 engine call has no argument for the plugin blocks {sorted(appended)}; "
            "the ud2 binary is needed for them"
        )
    extra_kwargs = {key: value for block in merged.values() for key, value in block.items()}
    if bias_block is not None:
        unknown = sorted(set(bias_block) - set(BIAS_BLOCK_ARGS))
        if unknown:
            raise RuntimeError(f"the pybind11 engine call has no argument for the bias block keys {unknown}")
        extra_kwargs.update({BIAS_BLOCK_ARGS[key]: value for key, value in bias_block.items()})
    ligand_dms = cfg.inputs.get('ligand_dms_full')
    receptor_dms = cfg.inputs.get('receptor_dms_full')
    if not ligand_dms or not receptor_dms:
        raise RuntimeError(
            "pybind11 engine requires enriched DMS files. "
            "Ensure DMS enrichment succeeded."
        )
    pipeline = importlib.import_module('unidock_engine.api.python.pipeline')
    if dms_has_ring_bias(ligand_dms):
        version = pipeline_version(pipeline)
        if version is None or version < RING_BIAS_MIN_VERSION:
            found = 'reports no version' if version is None else f'is version {_version_text(version)}'
            raise RuntimeError(
                f"the pybind11 pipeline module {getattr(pipeline, '__file__', pipeline.__name__)} {found} and does "
                f"not read the ring-bias tables {' / '.join(RING_BIAS_TABLES)} of {ligand_dms} "
                f"(ud2 >= {_version_text(RING_BIAS_MIN_VERSION)}): it would ignore them and dock without the ring "
                "term. Rebuild the engine's Python module or put a newer ud2 first on PATH."
            )
    pipeline.run_docking_pipeline(
        ligand_dms_path=ligand_dms,
        receptor_dms_path=receptor_dms,
        output_dir=cfg.output_dir,
        center_x=cfg.center[0],
        center_y=cfg.center[1],
        center_z=cfg.center[2],
        size_x=cfg.box_size[0],
        size_y=cfg.box_size[1],
        size_z=cfg.box_size[2],
        task=cfg.task,
        search_mode=cfg.search_mode,
        exhaustiveness=cfg.exhaustiveness,
        randomize=cfg.randomize,
        mc_steps=cfg.mc_steps,
        opt_steps=cfg.opt_steps,
        refine_steps=cfg.refine_steps,
        num_pose=cfg.num_pose,
        rescore_pool=cfg.rescore_pool,
        rmsd_limit=cfg.rmsd_limit,
        energy_range=cfg.energy_range,
        seed=cfg.seed,
        use_tor_lib=cfg.use_tor_lib,
        constraint_docking=cfg.constraint_docking,
        decomp=cfg.decomp,
        ff_minimize_steps=cfg.ff_minimize_steps,
        gpu_device_id=cfg.gpu_device_id,
        **extra_kwargs,
    )
