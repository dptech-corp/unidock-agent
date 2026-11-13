"""The template plugin: constraint docking on a core mapped to a reference pose (an automatic MCS map, a manual
core atom map per ligand, or the atom_mapper alignment that keeps the input pose).

The runner's compatibility constructor builds it from ``template_docking=True`` with ``reference_sdf_file_name``,
``core_atom_mapping_dict_list`` and ``atom_mapper_align``; ``atom_mapper_align`` alone (no template docking) gives
a disabled plugin that still swaps the ligand builder's rotatable-bond finder, as the flag did before the plugin.
The entry point ``unidock2.plugins`` / ``template`` gives a disabled instance that every hook leaves as a no-op.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from rdkit import Chem

from unidock_processing.ligand_topology import utils
from unidock_processing.plugins.base import LigandHook, Plugin
from unidock_processing.plugins.template.config import TemplateConfig
from unidock_processing.plugins.template.ligand import AtomMapperAlignHook, TemplateLigandHook

logger = logging.getLogger(__name__)


def normalize_core_atom_mapping_dict_list(
    core_atom_mapping_dict_list: Optional[List[Optional[Dict[Any, Any]]]],
) -> Optional[List[Optional[Dict[int, int]]]]:
    """The core atom maps with integer keys and values (YAML and JSON give string keys); None entries stay None."""
    if core_atom_mapping_dict_list is None:
        return None
    num_molecules: int = len(core_atom_mapping_dict_list)
    normalized: List[Optional[Dict[int, int]]] = [None] * num_molecules

    for mol_idx in range(num_molecules):
        raw_core_atom_mapping_dict = core_atom_mapping_dict_list[mol_idx]
        if raw_core_atom_mapping_dict is None:
            normalized[mol_idx] = None
        else:
            core_atom_mapping_dict = {
                int(reference_atom_idx): int(query_atom_idx)
                for reference_atom_idx, query_atom_idx \
                    in raw_core_atom_mapping_dict.items()
            }
            normalized[mol_idx] = core_atom_mapping_dict
    return normalized


class TemplatePlugin(Plugin):
    name = 'template'
    config_model = TemplateConfig                 # the 'template:' block (config.py)

    def __init__(
        self,
        reference_sdf_file_name: Optional[str] = None,
        core_atom_mapping_dict_list: Optional[List[Optional[Dict[Any, Any]]]] = None,
        atom_mapper_align: bool = False,
        enabled: Optional[bool] = None,
    ) -> None:
        if reference_sdf_file_name is not None:
            self.reference_sdf_file_name = os.path.abspath(reference_sdf_file_name)
        else:
            self.reference_sdf_file_name = None
        self.core_atom_mapping_dict_list = normalize_core_atom_mapping_dict_list(core_atom_mapping_dict_list)
        self.atom_mapper_align = atom_mapper_align
        # enabled = template docking; an entry-point instance (no arguments) is disabled, the compatibility
        # constructor passes template_docking, atom_mapper_align alone keeps the plugin disabled
        if enabled is None:
            enabled = reference_sdf_file_name is not None or core_atom_mapping_dict_list is not None
        self.enabled = enabled
        self._ligand_hook: Optional[LigandHook] = None

    def reference_center(self):
        """The centre of mass of the reference pose (hydrogens removed): the box centre of a run that gives none."""
        reference_mol = Chem.SDMolSupplier(self.reference_sdf_file_name, removeHs=True)[0]
        return tuple(utils.calculate_center_of_mass(reference_mol))

    def ligand_hook(self) -> Optional[LigandHook]:
        """The one ligand builder hook of this run (built on first use): the template hook when template docking
        is on, the atom_mapper rotatable-bond hook when only atom_mapper_align is set, else None."""
        if self._ligand_hook is None:
            if self.enabled:
                self._ligand_hook = TemplateLigandHook(
                    self.reference_sdf_file_name, self.core_atom_mapping_dict_list, self.atom_mapper_align,
                )
            elif self.atom_mapper_align:
                self._ligand_hook = AtomMapperAlignHook()
        return self._ligand_hook

    def prepare_ligand(self, mol, ligand_builder_kwargs: Dict[str, Any]) -> None:
        """Set the builder hook (the runner calls this once per ligand molecule; the hook is added once)."""
        hook = self.ligand_hook()
        if hook is None:
            return
        hooks = ligand_builder_kwargs.setdefault('hooks', [])
        if hook not in hooks:
            hooks.append(hook)

    def search_policy(self) -> Dict[str, Any]:
        """Constraint docking: the engine keeps the ligand root where the core placement put it. The engine binary
        forces randomize off under constraint docking itself (main.cpp), so the YAML's randomize line stays the
        run's own value, as before the plugin."""
        if not self.enabled:
            return {}
        return {'constraint_docking': True}

    @staticmethod
    def cli_options(parser) -> None:
        group = parser.add_argument_group('template docking (plugin "template")')
        group.add_argument(
            '--template_docking',
            action='store_true',
            default=False,
            help='Constraint docking on the core mapped to the reference pose (YAML: template: template_docking)',
        )
        group.add_argument(
            '--reference_sdf_file_name',
            default=None,
            help='Reference pose SDF for template docking (YAML: template: reference_sdf_file_name)',
        )
        group.add_argument(
            '--core_atom_mapping_dict_list',
            default=None,
            type=json.loads,
            help='JSON list with one {reference_atom_idx: query_atom_idx} map per ligand, null for the automatic '
                 'MCS map (YAML: template: core_atom_mapping_dict_list)',
        )

    @staticmethod
    def cli_kwargs(args) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if getattr(args, 'template_docking', False):
            kwargs['template_docking'] = True
        reference_sdf_file_name = getattr(args, 'reference_sdf_file_name', None)
        if reference_sdf_file_name is not None:
            kwargs['reference_sdf_file_name'] = os.path.abspath(reference_sdf_file_name)
        core_atom_mapping_dict_list = getattr(args, 'core_atom_mapping_dict_list', None)
        if core_atom_mapping_dict_list is not None:
            kwargs['core_atom_mapping_dict_list'] = core_atom_mapping_dict_list
        return kwargs
