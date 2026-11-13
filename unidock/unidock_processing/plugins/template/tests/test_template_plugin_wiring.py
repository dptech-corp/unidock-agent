"""The template plugin is wired: the search policy, the YAML block and its deprecated keys, the runner's
compatibility constructor, the ligand builder hooks (compat keyword arguments and the hooks argument give the same
graph), the CLI options, the utils re-exports and the entry point. No engine run here (test_constraint_docking.py).
"""
import argparse
import logging
import os
from copy import deepcopy

import dill
import pytest
from rdkit import Chem

from unidock_processing.ligand_topology import utils
from unidock_processing.plugins.template import mapping
from unidock_processing.plugins.template.ligand import (
    AtomMapperAlignHook, AtomMapperAlignMolHook, TemplateLigandHook, TemplateMolHook,
)
from unidock_processing.plugins.template.plugin import TemplatePlugin, normalize_core_atom_mapping_dict_list

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))          # unidock_processing/
REPO_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))
BACE_DIR = os.path.join(PACKAGE_DIR, 'test', 'data', 'constraint_docking', 'automatic_atom_mapping')
REFERENCE_SDF = os.path.join(BACE_DIR, 'reference.sdf')
CAT4A_SDF = os.path.join(BACE_DIR, 'CAT-4a.sdf')

needs_bace = pytest.mark.skipif(not os.path.isfile(REFERENCE_SDF), reason=f"missing {REFERENCE_SDF}")


def test_policy_and_yaml_block():
    plugin = TemplatePlugin(REFERENCE_SDF)
    assert plugin.enabled
    assert plugin.search_policy() == {'constraint_docking': True}
    assert plugin.yaml_block() == {}
    disabled = TemplatePlugin()                       # what the entry point instantiates
    assert not disabled.enabled
    assert disabled.search_policy() == {} and disabled.yaml_block() == {}
    kwargs = {}
    disabled.prepare_ligand(None, kwargs)
    assert kwargs == {}


def test_core_atom_maps_get_integer_keys():
    assert normalize_core_atom_mapping_dict_list(None) is None
    assert normalize_core_atom_mapping_dict_list([{'0': 14, '1': '15'}, None]) == [{0: 14, 1: 15}, None]
    plugin = TemplatePlugin('ref.sdf', [{'0': 14}], atom_mapper_align=True)
    assert plugin.core_atom_mapping_dict_list == [{0: 14}]
    assert plugin.reference_sdf_file_name == os.path.abspath('ref.sdf')
    assert plugin.atom_mapper_align is True


def test_prepare_ligand_sets_the_builder_hook_once():
    plugin = TemplatePlugin('ref.sdf', [{'0': 14}, None])
    kwargs = {}
    plugin.prepare_ligand(None, kwargs)
    plugin.prepare_ligand(None, kwargs)                # the runner calls it once per ligand molecule
    assert len(kwargs['hooks']) == 1
    hook = kwargs['hooks'][0]
    assert isinstance(hook, TemplateLigandHook) and hook is plugin.ligand_hook()
    assert hook.wants_static_root()
    assert hook.core_atom_mapping_dict_list == [{0: 14}, None]


@needs_bace
def test_ligand_hook_binds_slices_and_pickles():
    hook = TemplateLigandHook(REFERENCE_SDF, [{0: 1}, None, {2: 3}], atom_mapper_align=True)
    hook.bind(3)
    with pytest.raises(ValueError, match='core atom mapping'):
        hook.bind(2)
    with pytest.raises(ValueError, match='without reference SDF'):
        TemplateLigandHook(None).bind(1)
    automatic = TemplateLigandHook(REFERENCE_SDF)
    automatic.bind(2)
    assert automatic.core_atom_mapping_dict_list == [None, None]

    batch = dill.loads(dill.dumps(hook.for_batch(1, 3)))     # what the batch worker receives
    assert batch.core_atom_mapping_dict_list == [None, {2: 3}] and batch.atom_mapper_align
    batch.open()
    mol_hook = batch.for_ligand(1)
    assert isinstance(mol_hook, TemplateMolHook)
    assert mol_hook.core_atom_mapping_dict == {2: 3} and mol_hook.reference_mol.GetNumAtoms() > 0


def _runner(tmp_path, monkeypatch, **kwargs):
    from unidock_processing.core import runner as runner_mod
    monkeypatch.setattr(runner_mod, 'load_plugins', lambda: [])
    return runner_mod.UnidockProtocolRunner('rec.pdb', ['lig.sdf'], working_dir_name=str(tmp_path), **kwargs)


@needs_bace
def test_compat_constructor_builds_the_plugin(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, target_center=(0.0, 0.0, 0.0), template_docking=True,
                     reference_sdf_file_name=REFERENCE_SDF, core_atom_mapping_dict_list=[{'0': 14}],
                     atom_mapper_align=True, num_pose=3, rescore_pool=10, cnn_rescore=True)
    assert [plugin.name for plugin in runner.plugins] == ['template']
    plugin = runner.plugins[0]
    assert plugin.enabled and plugin.atom_mapper_align
    assert runner.core_atom_mapping_dict_list == [{0: 14}] == plugin.core_atom_mapping_dict_list
    assert runner._constraint_docking() and runner._search_policy() == {'constraint_docking': True}
    assert runner._engine_num_pose() == 3                  # the CNN pool is not raised under constraint docking

    reference_mol = Chem.SDMolSupplier(REFERENCE_SDF, removeHs=True)[0]
    assert runner.target_center == tuple(utils.calculate_center_of_mass(reference_mol))

    explicit = _runner(tmp_path, monkeypatch, target_center=(1.0, 2.0, 3.0), template_docking=True,
                       reference_sdf_file_name=REFERENCE_SDF)
    assert explicit.target_center == (1.0, 2.0, 3.0)

    plain = _runner(tmp_path, monkeypatch, target_center=(1.0, 2.0, 3.0), num_pose=3, cnn_rescore=True)
    assert plain.plugins == [] and not plain._constraint_docking() and plain._engine_num_pose() == 10


def test_compat_constructor_atom_mapper_align_alone(tmp_path, monkeypatch):
    """atom_mapper_align without template_docking (the step-1 runner forwarded the flag to the builder): the plugin
    stays disabled, no constraint docking, and the builder gets the rotatable-bond hook only."""
    runner = _runner(tmp_path, monkeypatch, target_center=(1.0, 2.0, 3.0), atom_mapper_align=True, num_pose=3,
                     rescore_pool=10, cnn_rescore=True)
    assert [plugin.name for plugin in runner.plugins] == ['template']
    plugin = runner.plugins[0]
    assert not plugin.enabled and plugin.atom_mapper_align
    assert not runner._constraint_docking() and runner._search_policy() == {} and runner._engine_num_pose() == 10
    assert runner.target_center == (1.0, 2.0, 3.0)
    kwargs = {}
    plugin.prepare_ligand(None, kwargs)
    plugin.prepare_ligand(None, kwargs)
    assert len(kwargs['hooks']) == 1 and isinstance(kwargs['hooks'][0], AtomMapperAlignHook)
    assert not kwargs['hooks'][0].wants_static_root()
    assert not TemplatePlugin(atom_mapper_align=True).enabled


def test_compat_constructor_replaces_the_registry_instance(tmp_path, monkeypatch):
    from unidock_processing.core import runner as runner_mod
    monkeypatch.setattr(runner_mod, 'load_plugins', lambda: [TemplatePlugin()])
    runner = runner_mod.UnidockProtocolRunner('rec.pdb', ['lig.sdf'], target_center=(1.0, 2.0, 3.0),
                                              working_dir_name=str(tmp_path), template_docking=True,
                                              reference_sdf_file_name='ref.sdf')
    assert [plugin.name for plugin in runner.plugins] == ['template'] and runner.plugins[0].enabled
    assert runner.reference_sdf_file_name == os.path.abspath('ref.sdf')


@needs_bace
def test_builder_compat_kwargs_build_the_hook(tmp_path):
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import UnidockLigandTopologyBuilder
    builder = UnidockLigandTopologyBuilder([CAT4A_SDF], template_docking=True, reference_sdf_file_name=REFERENCE_SDF,
                                           n_cpu=1, working_dir_name=str(tmp_path))
    assert len(builder.hooks) == 1 and isinstance(builder.hooks[0], TemplateLigandHook)
    assert builder.hooks[0].core_atom_mapping_dict_list == [None] and builder.static_root

    hook = TemplateLigandHook(REFERENCE_SDF, [{'0': 14}])
    by_hooks = UnidockLigandTopologyBuilder([CAT4A_SDF], hooks=[hook], n_cpu=1, working_dir_name=str(tmp_path))
    assert by_hooks.hooks == [hook] and by_hooks.static_root

    plain = UnidockLigandTopologyBuilder([CAT4A_SDF], n_cpu=1, working_dir_name=str(tmp_path))
    assert plain.hooks == [] and not plain.static_root

    with pytest.raises(ValueError, match='without reference SDF'):
        UnidockLigandTopologyBuilder([CAT4A_SDF], template_docking=True, n_cpu=1, working_dir_name=str(tmp_path))
    with pytest.raises(ValueError, match='core atom mapping'):
        UnidockLigandTopologyBuilder([CAT4A_SDF], core_atom_mapping_dict_list=[None, None], n_cpu=1,
                                     working_dir_name=str(tmp_path))


@needs_bace
@pytest.mark.parametrize('atom_mapper_align', [False, True])
def test_build_molecular_graph_compat_kwargs_equal_hooks(tmp_path, atom_mapper_align):
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        build_molecular_graph, load_torsion_library,
    )
    torsion_library = load_torsion_library()
    reference_mol = Chem.SDMolSupplier(REFERENCE_SDF, removeHs=False)[0]
    ligand_mol = Chem.SDMolSupplier(CAT4A_SDF, removeHs=False)[0]

    kwargs = dict(working_dir_name=str(tmp_path), skip_gaff2=True)      # Vina-only typing, no antechamber
    compat_mol = deepcopy(ligand_mol)
    compat = build_molecular_graph(compat_mol, torsion_library, template_docking=True, reference_mol=reference_mol,
                                   atom_mapper_align=atom_mapper_align, **kwargs)
    hooked_mol = deepcopy(ligand_mol)
    hook = TemplateMolHook(reference_mol, None, atom_mapper_align)
    hooked = build_molecular_graph(hooked_mol, torsion_library, hooks=[hook], **kwargs)
    assert compat == hooked
    assert hook.core_atom_idx_list and hook.core_atom_mapping_dict
    assert compat_mol.GetConformer().GetPositions().tolist() == hooked_mol.GetConformer().GetPositions().tolist()

    plain = build_molecular_graph(deepcopy(ligand_mol), torsion_library, **kwargs)
    assert plain == build_molecular_graph(deepcopy(ligand_mol), torsion_library, hooks=[], **kwargs)
    assert plain != compat                                  # the template run freezes the core


@needs_bace
def test_atom_mapper_align_alone_keeps_its_rotatable_bond_finder(tmp_path):
    """Before the plugin the builder ran the AlignRotatableBond SMARTS finder whenever atom_mapper_align was set,
    template docking or not (CAT-4a: 4 rotors instead of the generic finder's 3); the compat keyword argument and
    the hook give the same graph, with no core and no static root."""
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        UnidockLigandTopologyBuilder, build_molecular_graph, load_torsion_library,
    )
    torsion_library = load_torsion_library()
    ligand_mol = Chem.SDMolSupplier(CAT4A_SDF, removeHs=False)[0]
    kwargs = dict(working_dir_name=str(tmp_path), skip_gaff2=True)
    compat = build_molecular_graph(deepcopy(ligand_mol), torsion_library, atom_mapper_align=True, **kwargs)
    hooked = build_molecular_graph(deepcopy(ligand_mol), torsion_library, hooks=[AtomMapperAlignMolHook()], **kwargs)
    plain = build_molecular_graph(deepcopy(ligand_mol), torsion_library, **kwargs)
    assert compat == hooked
    assert len(compat[1]) == 4 and len(plain[1]) == 3

    builder = UnidockLigandTopologyBuilder([CAT4A_SDF], atom_mapper_align=True, n_cpu=1, skip_gaff2=True,
                                           working_dir_name=os.path.join(str(tmp_path), 'compat'))
    assert len(builder.hooks) == 1 and isinstance(builder.hooks[0], AtomMapperAlignHook) and not builder.static_root
    by_hooks = UnidockLigandTopologyBuilder([CAT4A_SDF], hooks=[AtomMapperAlignHook()], n_cpu=1, skip_gaff2=True,
                                            working_dir_name=os.path.join(str(tmp_path), 'hooks'))
    for b in (builder, by_hooks):
        b.generate_batch_ligand_topology()
        b.get_summary_ligand_info_dict()
    assert builder.total_ligand_info_summary_dict == by_hooks.total_ligand_info_summary_dict
    assert len(builder.total_ligand_info_summary_dict['MOL_0']['torsions']) == 4


def test_yaml_template_block_and_deprecated_preprocessing_keys(caplog):
    from unidock_processing.io.yaml import UnidockConfig, CONFIG_MAPPING, PLUGIN_CLASSES
    from unidock_processing.plugins.template.config import TemplateConfig
    assert TemplatePlugin.config_model is TemplateConfig and TemplatePlugin in PLUGIN_CLASSES
    assert CONFIG_MAPPING[-1] == ('template', 'template', TemplateConfig,
                                  ['template_docking', 'reference_sdf_file_name', 'core_atom_mapping_dict_list'])
    preprocessing_fields = next(fields for key, _, _, fields in CONFIG_MAPPING if key == 'Preprocessing')
    assert 'template_docking' not in preprocessing_fields               # no longer emitted from Preprocessing

    block = UnidockConfig.from_dict({'template': {'template_docking': True, 'reference_sdf_file_name': 'ref.sdf',
                                                  'core_atom_mapping_dict_list': [{'0': 14}]}})
    kwargs = block.to_protocol_kwargs()
    assert kwargs['template_docking'] is True and kwargs['reference_sdf_file_name'] == 'ref.sdf'
    assert kwargs['core_atom_mapping_dict_list'] == [{'0': 14}]
    assert UnidockConfig.from_dict({}).to_protocol_kwargs()['template_docking'] is False

    with caplog.at_level(logging.WARNING, logger='unidock_processing.io.yaml'):
        old = UnidockConfig.from_dict({'Preprocessing': {'template_docking': True,
                                                         'reference_sdf_file_name': 'old.sdf'}})
    assert 'deprecated' in caplog.text and "'template:'" in caplog.text
    kwargs = old.to_protocol_kwargs()
    assert kwargs['template_docking'] is True and kwargs['reference_sdf_file_name'] == 'old.sdf'
    assert kwargs['core_atom_mapping_dict_list'] is None

    both = UnidockConfig.from_dict({'Preprocessing': {'reference_sdf_file_name': 'old.sdf'},
                                    'template': {'reference_sdf_file_name': 'new.sdf'}})
    assert both.to_protocol_kwargs()['reference_sdf_file_name'] == 'new.sdf'


def test_cli_registers_the_template_options():
    from unidock_processing.cli.docking import CLICommand
    from unidock_processing.plugins.registry import plugin_classes
    assert TemplatePlugin in plugin_classes()                   # through the registry, no import in the CLI
    parser = argparse.ArgumentParser()
    CLICommand.add_arguments(parser)
    args = parser.parse_args(['--template_docking', '--reference_sdf_file_name', 'ref.sdf',
                              '--core_atom_mapping_dict_list', '[{"0": 14}, null]'])
    assert TemplatePlugin.cli_kwargs(args) == {'template_docking': True,
                                               'reference_sdf_file_name': os.path.abspath('ref.sdf'),
                                               'core_atom_mapping_dict_list': [{'0': 14}, None]}
    assert TemplatePlugin.cli_kwargs(parser.parse_args([])) == {}


def test_utils_re_exports_the_moved_mapping_functions():
    from unidock_processing.ligand_topology.utils import get_template_docking_atom_mapping
    assert get_template_docking_atom_mapping is mapping.get_template_docking_atom_mapping
    for name in ('get_template_docking_atom_mapping', 'get_core_alignment_for_template_docking',
                 'check_manual_atom_mapping_connection', 'get_full_ring_core_atoms'):
        assert getattr(utils, name) is getattr(mapping, name)
    with pytest.raises(AttributeError):
        utils.no_such_function


def test_entry_point_is_declared():
    from unidock_processing.plugins import registry
    with open(os.path.join(REPO_ROOT, 'pyproject.toml')) as f:
        text = f.read()
    assert '[project.entry-points."unidock2.plugins"]' in text
    assert 'template = "unidock_processing.plugins.template.plugin:TemplatePlugin"' in text
    assert registry.BUILTIN_PLUGINS['template'] == 'unidock_processing.plugins.template.plugin:TemplatePlugin'
    assert registry.load_plugin_class(registry.BUILTIN_PLUGINS['template']) is TemplatePlugin
    assert 'template' in registry.plugin_specs() and 'template' not in registry.missing_plugins()
