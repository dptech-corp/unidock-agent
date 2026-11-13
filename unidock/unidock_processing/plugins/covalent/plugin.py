"""The covalent plugin: docking a ligand bonded to a receptor residue. The ligand SDF names the residue atoms it
carries (``covalent_atom_indices`` and the matching name, residue and chain tags) and the receptor side names the
anchor, bond-start and bond-end atoms (``covalent_residue_atom_info_list``).

The runner's compatibility constructor builds it from ``covalent_ligand=True`` with ``covalent_residue_atom_info_list``;
the list alone (no covalent ligand) gives a disabled plugin that still hands the list to the receptor builder, as the
runner argument did before the plugin. The entry point ``unidock2.plugins`` / ``covalent`` gives a disabled instance
that every hook leaves as a no-op.
"""
import json
from typing import Any, Dict, List, Optional

from unidock_processing.plugins.base import LigandHook, Plugin
from unidock_processing.plugins.covalent.config import CovalentConfig
from unidock_processing.plugins.covalent.ligand import CovalentLigandHook


class CovalentPlugin(Plugin):
    name = 'covalent'
    config_model = CovalentConfig                 # the 'covalent:' block (config.py)

    def __init__(
        self,
        covalent_residue_atom_info_list: Optional[List[Any]] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.covalent_residue_atom_info_list = covalent_residue_atom_info_list
        # enabled = covalent ligand; an entry-point instance (no arguments) is disabled, the compatibility
        # constructor passes covalent_ligand
        if enabled is None:
            enabled = covalent_residue_atom_info_list is not None
        self.enabled = enabled
        self._ligand_hook: Optional[LigandHook] = None

    def ligand_hook(self) -> Optional[LigandHook]:
        """The one ligand builder hook of this run (built on first use) when the covalent ligand is on, else None."""
        if self._ligand_hook is None and self.enabled:
            self._ligand_hook = CovalentLigandHook()
        return self._ligand_hook

    def annotate_receptor(self, builder) -> None:
        """Set the covalent residue list on the constructed receptor builder (what the runner's constructor
        argument set before the plugin); the builder's analyze_receptor_topology prepares the bond from it."""
        if self.covalent_residue_atom_info_list is None:
            return
        builder.covalent_residue_atom_info_list = self.covalent_residue_atom_info_list

    def prepare_ligand(self, mol, ligand_builder_kwargs: Dict[str, Any]) -> None:
        """Set the builder hook (the runner calls this once per ligand molecule; the hook is added once)."""
        hook = self.ligand_hook()
        if hook is None:
            return
        hooks = ligand_builder_kwargs.setdefault('hooks', [])
        if hook not in hooks:
            hooks.append(hook)

    def search_policy(self) -> Dict[str, Any]:
        """Constraint docking: the engine keeps the ligand root (the covalent anchor fragment) where the input puts
        it. The engine binary forces randomize off under constraint docking itself (main.cpp), so the YAML's
        randomize line stays the run's own value, as before the plugin."""
        if not self.enabled:
            return {}
        return {'constraint_docking': True}

    @staticmethod
    def cli_options(parser) -> None:
        group = parser.add_argument_group('covalent docking (plugin "covalent")')
        group.add_argument(
            '--covalent_ligand',
            action='store_true',
            default=False,
            help='Covalent docking: the ligand SDF carries the receptor residue atoms it is bonded to '
                 '(YAML: covalent: covalent_ligand)',
        )
        group.add_argument(
            '--covalent_residue_atom_info_list',
            default=None,
            type=json.loads,
            help='JSON list of the anchor, bond-start and bond-end receptor atoms, each '
                 '[chain, residue_name, residue_idx, atom_name] (YAML: covalent: covalent_residue_atom_info_list)',
        )

    @staticmethod
    def cli_kwargs(args) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if getattr(args, 'covalent_ligand', False):
            kwargs['covalent_ligand'] = True
        covalent_residue_atom_info_list = getattr(args, 'covalent_residue_atom_info_list', None)
        if covalent_residue_atom_info_list is not None:
            kwargs['covalent_residue_atom_info_list'] = covalent_residue_atom_info_list
        return kwargs
