"""The plugin interface: a feature that needs a second molecule or a chemical annotation (template, covalent,
bias) implements these hooks; the runner calls each one between its stages. Every hook is a no-op here.
"""
import csv
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def write_skipped_ligands(csv_path: str, rows: Iterable[Sequence[Any]]) -> None:
    """Write the ligands a plugin left out of a run: one (source file, record index, title, reason) row each."""
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['source_file', 'record', 'title', 'reason'])
        writer.writerows(rows)


class Plugin:
    name: str = ""                                  # "template", "covalent", "bias"
    # The pydantic model of the plugin's block in the configuration YAML, which is named after the plugin
    # (``template:``); None when the plugin has no block. io.yaml reads the block with it and flattens its fields
    # into the runner keyword arguments.
    config_model: Optional[type] = None
    # True while the block's fields are also keyword arguments of the runner's compatibility constructor (the
    # one-release route of the template and covalent blocks): io.yaml flattens them into runner keyword arguments
    # when from_config returns None. False for a plugin that reads its block through from_config alone (bias): a
    # block from_config declines (all defaults) then sets nothing.
    flatten_config: bool = True

    @classmethod
    def from_config(cls, block) -> Optional['Plugin']:
        """The configured plugin for its YAML block (an instance of ``config_model``), handed to the runner through
        ``plugins=``; None when the block's keys are runner keyword arguments the compatibility constructor reads,
        which io.yaml then flattens into them (the one-release route of the template and covalent plugins)."""
        return None

    def configure(self, config: Mapping[str, Any]) -> None:
        """Called once with the run's configuration (the runner's settings as a mapping)."""

    def check_setup(self) -> Optional[str]:
        """Called once after ``configure``, before anything is prepared. Returns why the plugin can process no
        ligand at all in this run, or None; the runner then logs the reason and finishes without output. A setup
        that only degrades the plugin is logged here and returns None."""
        return None

    def skipped_ligands(self, mols: List[Any]) -> Dict[int, str]:
        """The ligands of *mols* (every usable record, in input order) the plugin cannot process, as {index in
        *mols*: reason}. The runner leaves them out of every later stage and lists them in skipped_ligands.csv."""
        return {}

    def annotate_receptor(self, builder) -> None:
        """May edit the receptor topology builder before the receptor preparation: the builder is constructed and
        nothing has run on it yet (a covalent residue list set here is read by the preparation)."""

    def prepare_ligand(self, mol, ligand_builder_kwargs: Dict[str, Any]) -> None:
        """May set ligand topology builder arguments; called once per ligand molecule."""

    def annotate_ligand_dms(self, merged_dms: str, offsets: Dict[str, int]) -> None:
        """May write extra tables into the enriched ligand DMS; *offsets* maps molecule name to atom offset."""

    def yaml_block(self) -> Dict[str, Dict[str, Any]]:
        """Engine YAML blocks: one named after a core block (core.engine.CORE_BLOCKS) is merged into it, any other
        name is a new top-level block after the core ones, e.g. the bias plugin's {"bias": {"type": "align", "k": 0.05}}
        (the engine's own bias block; run_engine_pybind11 maps it onto the pipeline's bias / bias_k arguments)."""
        return {}

    def search_policy(self) -> Dict[str, Any]:
        """Engine settings the plugin overrides, e.g. {"constraint_docking": True}."""
        return {}

    def keeps_engine_order(self) -> bool:
        """True when the engine's pose order is final for this run: the runner then skips the CNN rescoring stage
        and the engine writes num_pose poses, not the rescore pool (core.rescore.engine_num_pose). Constraint
        docking (the template and covalent plugins) keeps the engine order through its search policy instead."""
        return False

    def post_process(self, sdf_path: str) -> None:
        """Called on the final pose SDF, after the engine and the CNN stage."""

    @staticmethod
    def cli_options(parser) -> None:
        """Add the plugin's options to an argparse parser."""

    @staticmethod
    def cli_kwargs(args) -> Dict[str, Any]:
        """The runner keyword arguments set by the plugin's parsed command-line options (``cli_options``)."""
        return {}


class LigandMolHook:
    """A plugin's per-molecule hook into ``build_molecular_graph``: each method is called where the builder had a
    feature branch, in this order; every method here is the plain path, so a builder with no hooks is unchanged.
    """

    def place_core(self, mol, working_dir_name: str):
        """Before atom typing: the molecule to type, moved (onto a template core) or rebuilt (the covalent ligand
        without its dummy atoms); the plain path returns *mol* itself. The builder rebinds its molecule to the
        return value, so a hook may return a new object."""
        return mol

    def rotatable_bonds(self, mol):
        """The rotatable bond list, or None for the builder's generic finder."""
        return None

    def filter_rotatable_bonds(self, mol, rotatable_bond_info_list):
        """After the force-field step: may drop rotatable bonds (e.g. freeze the core)."""
        return rotatable_bond_info_list

    def root_fragment_idx(self, splitted_mol_list):
        """The root fragment index, or None for the builder's root-finding strategy."""
        return None


class LigandHook:
    """A plugin's hook into ``UnidockLigandTopologyBuilder``: one object per run, sliced per batch and sent
    (pickled) to the batch worker, which opens it once and binds one ``LigandMolHook`` per ligand."""

    def bind(self, num_ligands: int) -> None:
        """Called once by the builder with the number of valid ligands (validate per-ligand inputs here)."""

    def wants_static_root(self) -> bool:
        """True when the ligand root stays where the input puts it (constraint docking)."""
        return False

    def for_batch(self, begin: int, end: int) -> 'LigandHook':
        """The hook for ligands [begin, end) of the run, picklable for the batch worker."""
        return self

    def open(self) -> None:
        """Called once in the batch worker before its ligands are built (load shared inputs here)."""

    def for_ligand(self, ligand_idx: int) -> LigandMolHook:
        """The per-molecule hook for ligand *ligand_idx* of the batch."""
        return LigandMolHook()
