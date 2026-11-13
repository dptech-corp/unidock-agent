"""The covalent plugin is wired: the search policy, the YAML block and its deprecated keys, the runner's
compatibility constructor (the plugin, the receptor builder attribute, the ligand builder hook), the ligand builder
hooks (the compat keyword argument and the hooks argument give the same graph on the 1EWL covalent ligand), the
receptor hook called from analyze_receptor_topology at the place the branch sat before the move (stub residues), the
legacy pose writer's covalent branch, the CLI options, the utils re-export and the entry point. No engine run here
(test_covalent_docking.py).
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import types
from copy import deepcopy

import dill
import pytest
from rdkit import Chem

from unidock_processing.ligand_topology import utils
from unidock_processing.plugins.covalent import ligand as covalent_ligand_mod
from unidock_processing.plugins.covalent.ligand import (
    CovalentLigandHook, CovalentMolHook, prepare_covalent_ligand_mol,
)
from unidock_processing.plugins.covalent.plugin import CovalentPlugin
from unidock_processing.plugins.covalent.receptor import CovalentReceptorHook

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))          # unidock_processing/
REPO_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))
EWL_DIR = os.path.join(PACKAGE_DIR, 'test', 'data', 'covalent_docking', '1EWL')
COVALENT_SDF = os.path.join(EWL_DIR, 'covalent_mol.sdf')
RESIDUE_LIST = [['', 'CYX', 25, 'CA'], ['', 'CYX', 25, 'CB'], ['', 'CYX', 25, 'SG']]

needs_ewl = pytest.mark.skipif(not os.path.isfile(COVALENT_SDF), reason=f"missing {COVALENT_SDF}")


def test_policy_and_yaml_block():
    plugin = CovalentPlugin(RESIDUE_LIST)
    assert plugin.enabled and plugin.covalent_residue_atom_info_list == RESIDUE_LIST
    assert plugin.search_policy() == {'constraint_docking': True}
    assert plugin.yaml_block() == {}
    disabled = CovalentPlugin()                       # what the entry point instantiates
    assert not disabled.enabled
    assert disabled.search_policy() == {} and disabled.yaml_block() == {}
    kwargs = {}
    disabled.prepare_ligand(None, kwargs)
    assert kwargs == {}
    builder = types.SimpleNamespace(covalent_residue_atom_info_list=None)
    disabled.annotate_receptor(builder)
    assert builder.covalent_residue_atom_info_list is None
    # the residue list without covalent_ligand: what the runner argument did on its own (the list reached the
    # receptor builder, nothing else changed)
    list_only = CovalentPlugin(RESIDUE_LIST, enabled=False)
    assert not list_only.enabled and list_only.search_policy() == {} and list_only.ligand_hook() is None
    list_only.annotate_receptor(builder)
    assert builder.covalent_residue_atom_info_list == RESIDUE_LIST


def test_prepare_ligand_sets_the_builder_hook_once():
    plugin = CovalentPlugin(RESIDUE_LIST)
    kwargs = {}
    plugin.prepare_ligand(None, kwargs)
    plugin.prepare_ligand(None, kwargs)                # the runner calls it once per ligand molecule
    assert len(kwargs['hooks']) == 1
    hook = kwargs['hooks'][0]
    assert isinstance(hook, CovalentLigandHook) and hook is plugin.ligand_hook()
    assert hook.wants_static_root()
    hook.bind(3)
    batch = dill.loads(dill.dumps(hook.for_batch(1, 3)))     # what the batch worker receives
    batch.open()
    mol_hook = batch.for_ligand(0)
    assert isinstance(mol_hook, CovalentMolHook) and mol_hook.covalent_anchor_atom_info is None
    assert mol_hook.rotatable_bonds(None) is None             # the generic finder, as before
    assert mol_hook.filter_rotatable_bonds(None, [(0, 1)]) == [(0, 1)]


def _runner(tmp_path, monkeypatch, **kwargs):
    from unidock_processing.core import runner as runner_mod
    monkeypatch.setattr(runner_mod, 'load_plugins', lambda: [])
    return runner_mod.UnidockProtocolRunner('rec.pdb', ['lig.sdf'], working_dir_name=str(tmp_path), **kwargs)


def test_compat_constructor_builds_the_plugin(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, target_center=(1.0, 2.0, 3.0), covalent_ligand=True,
                     covalent_residue_atom_info_list=RESIDUE_LIST, num_pose=3, rescore_pool=10, cnn_rescore=True)
    assert [plugin.name for plugin in runner.plugins] == ['covalent']
    plugin = runner.plugins[0]
    assert plugin.enabled and plugin.covalent_residue_atom_info_list == RESIDUE_LIST
    assert runner.covalent_ligand is True and runner.covalent_residue_atom_info_list == RESIDUE_LIST
    assert runner._constraint_docking() and runner._search_policy() == {'constraint_docking': True}
    assert runner._engine_num_pose() == 3                  # the CNN pool is not raised under constraint docking

    list_only = _runner(tmp_path, monkeypatch, target_center=(1.0, 2.0, 3.0), num_pose=3, rescore_pool=10,
                        cnn_rescore=True, covalent_residue_atom_info_list=RESIDUE_LIST)
    assert [plugin.name for plugin in list_only.plugins] == ['covalent'] and not list_only.plugins[0].enabled
    assert not list_only._constraint_docking() and list_only._engine_num_pose() == 10
    kwargs = {}
    list_only.plugins[0].prepare_ligand(None, kwargs)
    assert kwargs == {}

    plain = _runner(tmp_path, monkeypatch, target_center=(1.0, 2.0, 3.0), num_pose=3, cnn_rescore=True)
    assert plain.plugins == [] and not plain._constraint_docking() and plain._engine_num_pose() == 10


def test_compat_constructor_replaces_the_registry_instance(tmp_path, monkeypatch):
    from unidock_processing.core import runner as runner_mod
    monkeypatch.setattr(runner_mod, 'load_plugins', lambda: [CovalentPlugin()])
    runner = runner_mod.UnidockProtocolRunner('rec.pdb', ['lig.sdf'], target_center=(1.0, 2.0, 3.0),
                                              working_dir_name=str(tmp_path), covalent_ligand=True,
                                              covalent_residue_atom_info_list=RESIDUE_LIST)
    assert [plugin.name for plugin in runner.plugins] == ['covalent'] and runner.plugins[0].enabled


def test_compat_constructor_orders_covalent_before_template(tmp_path, monkeypatch):
    """Both features given: the covalent hook precedes the template hook (the builder rebuilt the covalent ligand
    before any template placement), which is also the registry's name order."""
    pytest.importorskip('unidock_processing.plugins.template.plugin')
    runner = _runner(tmp_path, monkeypatch, target_center=(1.0, 2.0, 3.0), covalent_ligand=True,
                     covalent_residue_atom_info_list=RESIDUE_LIST, template_docking=True,
                     reference_sdf_file_name='ref.sdf')
    assert [plugin.name for plugin in runner.plugins] == ['covalent', 'template']
    kwargs = {}
    for plugin in runner.plugins:
        plugin.prepare_ligand(None, kwargs)
    assert [type(hook).__name__ for hook in kwargs['hooks']] == ['CovalentLigandHook', 'TemplateLigandHook']


class _RecordingReceptorBuilder:
    """Stands in for UnidockReceptorTopologyBuilder: the constructor arguments and the attribute the preparation
    reads at the time it runs."""

    def __init__(self, receptor_file_name, prepared_hydrogen=False, covalent_residue_atom_info_list=None,
                 working_dir_name='.', require_ff_params=False):
        self.constructor_covalent_residue_atom_info_list = covalent_residue_atom_info_list
        self.covalent_residue_atom_info_list = covalent_residue_atom_info_list
        self.receptor_parameterized_dms_file_name = os.path.join(working_dir_name, 'receptor_parameterized.dms')
        self.calls = []
        self.list_when_prepared = 'unset'

    def generate_receptor_topology(self):
        self.calls.append('generate_receptor_topology')
        self.list_when_prepared = self.covalent_residue_atom_info_list

    def analyze_receptor_topology_vina_only(self):
        self.calls.append('analyze_receptor_topology_vina_only')

    def get_summary_receptor_info_dict(self):
        self.calls.append('get_summary_receptor_info_dict')


def test_runner_hands_the_residue_list_to_the_receptor_builder_before_the_preparation(tmp_path, monkeypatch):
    """The runner no longer passes covalent_residue_atom_info_list to the receptor builder constructor: the plugin's
    annotate_receptor sets the same attribute on the constructed builder before generate_receptor_topology, so the
    builder state the preparation sees is the one the constructor argument gave."""
    from unidock_processing.core import runner as runner_mod
    monkeypatch.setattr(runner_mod, 'UnidockReceptorTopologyBuilder', _RecordingReceptorBuilder)
    monkeypatch.setattr(runner_mod, 'dms_has_nonbonded_params', lambda path: False)
    runner = _runner(tmp_path, monkeypatch, target_center=(1.0, 2.0, 3.0), covalent_ligand=True,
                     covalent_residue_atom_info_list=RESIDUE_LIST)
    builder = runner._prepare_receptor()
    assert builder.constructor_covalent_residue_atom_info_list is None
    assert builder.list_when_prepared == RESIDUE_LIST and builder.covalent_residue_atom_info_list == RESIDUE_LIST
    assert builder.calls == ['generate_receptor_topology', 'analyze_receptor_topology_vina_only',
                             'get_summary_receptor_info_dict']


def _covalent_mol():
    mol = Chem.SDMolSupplier(COVALENT_SDF, removeHs=False)[0]
    # the three molecule tags the builder sets before build_molecular_graph (prepare_covalent_ligand_mol keeps them)
    mol.SetProp('source_sdf_file_name', COVALENT_SDF)
    mol.SetIntProp('source_mol_idx', 0)
    mol.SetProp('ud2_molecule_name', 'MOL_0')
    return mol


@needs_ewl
def test_prepare_covalent_ligand_mol_drops_the_dummy_atoms():
    mol = _covalent_mol()
    covalent_mol, anchor, atom_info_list = prepare_covalent_ligand_mol(mol)
    assert covalent_mol.GetNumAtoms() == mol.GetNumAtoms() - 2
    assert anchor == ('A', 'CYS', 25, 'CA')
    assert atom_info_list == [('A', 'CYS', 25, name) for name in ('CA', 'CB', 'SG', 'HA', 'HB3', 'HB2')]
    assert covalent_mol.GetProp('ud2_molecule_name') == 'MOL_0'


@needs_ewl
def test_build_molecular_graph_compat_kwarg_equals_hook(tmp_path):
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        build_molecular_graph, load_torsion_library,
    )
    torsion_library = load_torsion_library()
    mol = _covalent_mol()
    kwargs = dict(working_dir_name=str(tmp_path), skip_gaff2=True)      # Vina-only typing, no antechamber
    compat = build_molecular_graph(deepcopy(mol), torsion_library, covalent_ligand=True, **kwargs)
    hook = CovalentMolHook()
    hooked = build_molecular_graph(deepcopy(mol), torsion_library, hooks=[hook], **kwargs)
    both = build_molecular_graph(deepcopy(mol), torsion_library, covalent_ligand=True, hooks=[CovalentMolHook()],
                                 **kwargs)                                # the kwarg adds no second hook
    assert compat == hooked == both
    assert hook.covalent_anchor_atom_info == ('A', 'CYS', 25, 'CA')
    atoms, torsions, root_atoms, fragments = hooked
    assert len(atoms) == mol.GetNumAtoms() - 2                            # the two dummy atoms are gone
    assert len(torsions) == 13 and 6 in root_atoms                        # the root holds the anchor atom (CA)


@needs_ewl
def test_builder_compat_kwarg_equals_hooks(tmp_path):
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import UnidockLigandTopologyBuilder
    compat = UnidockLigandTopologyBuilder([COVALENT_SDF], covalent_ligand=True, n_cpu=1, skip_gaff2=True,
                                          working_dir_name=os.path.join(str(tmp_path), 'compat'))
    assert len(compat.hooks) == 1 and isinstance(compat.hooks[0], CovalentLigandHook) and compat.static_root
    assert compat.covalent_ligand is True
    hook = CovalentLigandHook()
    by_hooks = UnidockLigandTopologyBuilder([COVALENT_SDF], hooks=[hook], n_cpu=1, skip_gaff2=True,
                                            working_dir_name=os.path.join(str(tmp_path), 'hooks'))
    assert by_hooks.hooks == [hook] and by_hooks.static_root and by_hooks.covalent_ligand is False
    both = UnidockLigandTopologyBuilder([COVALENT_SDF], covalent_ligand=True, hooks=[hook], n_cpu=1, skip_gaff2=True,
                                        working_dir_name=os.path.join(str(tmp_path), 'both'))
    assert both.hooks == [hook]                                           # the kwarg adds no second hook
    plain = UnidockLigandTopologyBuilder([COVALENT_SDF], n_cpu=1, skip_gaff2=True,
                                         working_dir_name=os.path.join(str(tmp_path), 'plain'))
    assert plain.hooks == [] and not plain.static_root

    for builder in (compat, by_hooks):
        builder.generate_batch_ligand_topology()
        builder.get_summary_ligand_info_dict()
    assert compat.total_ligand_info_summary_dict == by_hooks.total_ligand_info_summary_dict
    summary = by_hooks.total_ligand_info_summary_dict['MOL_0']
    assert len(summary['atoms']) == 59 and len(summary['torsions']) == 13 and 6 in summary['root_atoms']


def _residue(resname, resid, chain, atoms, bonds):
    """A residue molecule with the atom properties protein_topology sets: what the covalent hook and the atom-info
    loop of analyze_receptor_topology read."""
    vina_types = {'C': 'C_H', 'N': 'N_D', 'O': 'O_A', 'S': 'S_P', 'H': 'H'}
    mol = Chem.RWMol()
    for atom_idx, (name, symbol) in enumerate(atoms):
        atom = Chem.Atom(symbol)
        atom.SetProp('atom_name', name)
        atom.SetProp('residue_name', resname)
        atom.SetIntProp('residue_idx', resid)
        atom.SetProp('chain_idx', chain)
        atom.SetProp('vina_atom_type', vina_types[symbol])
        atom.SetDoubleProp('atom_charge', 0.0)
        atom.SetDoubleProp('x', float(atom_idx))
        atom.SetDoubleProp('y', float(resid))
        atom.SetDoubleProp('z', 0.0)
        mol.AddAtom(atom)
    for begin, end in bonds:
        mol.AddBond(begin, end, Chem.BondType.SINGLE)
    return mol.GetMol()


CYS_ATOMS = [('N', 'N'), ('CA', 'C'), ('CB', 'C'), ('SG', 'S'), ('C', 'C'), ('O', 'O'),
             ('HA', 'H'), ('HB2', 'H'), ('HB3', 'H'), ('HG', 'H'), ('H', 'H')]
CYS_BONDS = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (1, 6), (2, 7), (2, 8), (3, 9), (0, 10)]
GLY_ATOMS = [('N', 'N'), ('CA', 'C'), ('C', 'C'), ('O', 'O'), ('HA2', 'H'), ('HA3', 'H'), ('H', 'H')]
GLY_BONDS = [(0, 1), (1, 2), (2, 3), (1, 4), (1, 5), (0, 6)]


def _stub_receptor_builder(monkeypatch, tmp_path, **kwargs):
    """A real UnidockReceptorTopologyBuilder whose analyze_receptor_topology reads two stub residues instead of a
    DMS (no fepfixer / utop / msys conversion here)."""
    from unidock_processing.unidocktools import unidock_receptor_topology_builder as receptor_mod
    residues = [_residue('GLY', 24, '', GLY_ATOMS, GLY_BONDS), _residue('CYX', 25, '', CYS_ATOMS, CYS_BONDS)]
    monkeypatch.setattr(receptor_mod.msys, 'LoadDMS', lambda path: path)
    monkeypatch.setattr(receptor_mod, 'prepare_receptor_residue_mol_list', lambda system: (None, residues, []))
    builder = receptor_mod.UnidockReceptorTopologyBuilder('receptor.dms', working_dir_name=str(tmp_path), **kwargs)
    builder.receptor_parameterized_dms_file_name = 'receptor.dms'
    return builder


def test_receptor_hook_runs_from_analyze_receptor_topology_at_the_same_place(tmp_path, monkeypatch):
    """The builder calls the plugin's hook where its own prepare_covalent_bond_on_residue sat: after the residue
    list is built, before the atom-info loop, only when the residue list is set; the constructor argument and the
    plugin's annotate_receptor give the same builder state and the same result."""
    seen = []
    original = CovalentReceptorHook.prepare_covalent_bond_on_residue

    def spy(self):
        builder = self.builder
        seen.append((len(builder.protein_residue_property_mol_list), hasattr(builder, 'atom_info_nested_list'),
                     builder.covalent_residue_idx, list(builder.covalent_residue_atom_idx_list)))
        return original(self)

    monkeypatch.setattr(CovalentReceptorHook, 'prepare_covalent_bond_on_residue', spy)

    by_kwarg = _stub_receptor_builder(monkeypatch, tmp_path, covalent_residue_atom_info_list=RESIDUE_LIST)
    by_kwarg.analyze_receptor_topology()
    by_plugin = _stub_receptor_builder(monkeypatch, tmp_path)
    CovalentPlugin(RESIDUE_LIST).annotate_receptor(by_plugin)
    assert by_plugin.covalent_residue_atom_info_list == by_kwarg.covalent_residue_atom_info_list
    by_plugin.analyze_receptor_topology()
    plain = _stub_receptor_builder(monkeypatch, tmp_path)
    plain.analyze_receptor_topology()

    assert seen == [(2, False, None, [])] * 2                 # once per covalent builder, before the atom info
    for builder in (by_kwarg, by_plugin):
        assert builder.covalent_residue_idx == 1
        assert builder.covalent_residue_atom_idx_list == [1, 6, 2, 7, 8, 3, 9]      # CA+HA, CB+HB2+HB3, SG+HG
        residue = builder.protein_residue_property_mol_list[1]
        assert [atom.GetProp('atom_name') for atom in residue.GetAtoms()] == ['N', 'C', 'O', 'H']
        assert len(builder.atom_info_nested_list) == len(GLY_ATOMS) + 4
    assert by_kwarg.atom_info_nested_list == by_plugin.atom_info_nested_list
    assert plain.covalent_residue_idx is None and plain.covalent_residue_atom_idx_list == []
    assert len(plain.atom_info_nested_list) == len(GLY_ATOMS) + len(CYS_ATOMS)

    missing = _stub_receptor_builder(monkeypatch, tmp_path,
                                     covalent_residue_atom_info_list=[['', 'CYX', 99, 'CA']] * 3)
    with pytest.raises(ValueError, match='Cannot find covalent residues'):
        missing.analyze_receptor_topology()


@needs_ewl
def test_pose_writer_covalent_branch_uses_the_plugin(tmp_path):
    """The legacy JSON pose writer (unidock_benchmark_runner; the runner reads the engine's SDF directly) rebuilds
    the covalent ligand through the plugin's prepare_covalent_ligand_mol."""
    from unidock_processing.unidocktools.unidock_ligand_pose_writer import UnidockLigandPoseWriter
    mol = _covalent_mol()
    num_atoms = mol.GetNumAtoms() - 2
    pose_json = os.path.join(str(tmp_path), 'poses.json')
    with open(pose_json, 'w') as f:
        json.dump({'MOL_0': [{'energy': [-1.0, -0.9, -0.1, -0.8, 0.0, 0.2],
                              'coords': [float(i) for i in range(3 * num_atoms)]}]}, f)
    out_sdf = os.path.join(str(tmp_path), 'poses.sdf')
    UnidockLigandPoseWriter([mol], [pose_json], covalent_ligand=True,
                            docking_pose_sdf_file_name=out_sdf).generate_docking_pose_sdf()
    poses = list(Chem.SDMolSupplier(out_sdf, removeHs=False))
    assert len(poses) == 1 and poses[0].GetNumAtoms() == num_atoms
    assert poses[0].GetProp('vina_binding_free_energy') == '-1.0'


def test_yaml_covalent_block_and_deprecated_preprocessing_keys(caplog):
    from unidock_processing.io.yaml import UnidockConfig, CONFIG_MAPPING, PLUGIN_CLASSES, read_unidock_params_from_yaml
    from unidock_processing.plugins.covalent.config import CovalentConfig
    assert CovalentPlugin.config_model is CovalentConfig and CovalentPlugin in PLUGIN_CLASSES
    assert ('covalent', 'covalent', CovalentConfig,
            ['covalent_ligand', 'covalent_residue_atom_info_list']) in CONFIG_MAPPING
    preprocessing_fields = next(fields for key, _, _, fields in CONFIG_MAPPING if key == 'Preprocessing')
    assert 'covalent_ligand' not in preprocessing_fields           # no longer emitted from Preprocessing
    assert 'covalent_residue_atom_info_list' not in preprocessing_fields

    block = UnidockConfig.from_dict({'covalent': {'covalent_ligand': True,
                                                  'covalent_residue_atom_info_list': RESIDUE_LIST}})
    kwargs = block.to_protocol_kwargs()
    assert kwargs['covalent_ligand'] is True and kwargs['covalent_residue_atom_info_list'] == RESIDUE_LIST
    assert UnidockConfig.from_dict({}).to_protocol_kwargs()['covalent_ligand'] is False

    with caplog.at_level(logging.WARNING, logger='unidock_processing.io.yaml'):
        old = UnidockConfig.from_dict({'Preprocessing': {'covalent_ligand': True,
                                                         'covalent_residue_atom_info_list': RESIDUE_LIST}})
    assert 'deprecated' in caplog.text and "'covalent:'" in caplog.text
    kwargs = old.to_protocol_kwargs()
    assert kwargs['covalent_ligand'] is True and kwargs['covalent_residue_atom_info_list'] == RESIDUE_LIST
    assert old.preprocessing.covalent_ligand is True                 # still readable for one release

    both = UnidockConfig.from_dict({'Preprocessing': {'covalent_residue_atom_info_list': [['', 'CYX', 1, 'CA']]},
                                    'covalent': {'covalent_residue_atom_info_list': RESIDUE_LIST}})
    assert both.to_protocol_kwargs()['covalent_residue_atom_info_list'] == RESIDUE_LIST

    example = os.path.join(REPO_ROOT, 'examples', 'covalent_docking', '1EWL', 'unidock_configurations.yaml')
    if os.path.isfile(example):                                      # the shipped example still uses the old keys
        kwargs = read_unidock_params_from_yaml(example).to_protocol_kwargs()
        assert kwargs['covalent_ligand'] is True and kwargs['covalent_residue_atom_info_list'] == RESIDUE_LIST


def test_cli_registers_the_covalent_options():
    from unidock_processing.cli import docking as docking_mod
    from unidock_processing.plugins.registry import plugin_classes
    assert CovalentPlugin in plugin_classes()                        # through the registry, no import in the CLI
    parser = argparse.ArgumentParser()
    docking_mod.CLICommand.add_arguments(parser)
    args = parser.parse_args(['--covalent_ligand', '--covalent_residue_atom_info_list', json.dumps(RESIDUE_LIST)])
    assert CovalentPlugin.cli_kwargs(args) == {'covalent_ligand': True,
                                               'covalent_residue_atom_info_list': RESIDUE_LIST}
    assert CovalentPlugin.cli_kwargs(parser.parse_args([])) == {}
    assert '    covalent:\n        covalent_ligand: false\n' in docking_mod.CLICommand.__doc__


def test_utils_re_exports_prepare_covalent_ligand_mol():
    assert utils.prepare_covalent_ligand_mol is covalent_ligand_mod.prepare_covalent_ligand_mol
    from unidock_processing.ligand_topology.utils import prepare_covalent_ligand_mol as re_exported
    assert re_exported is prepare_covalent_ligand_mol
    assert 'prepare_covalent_ligand_mol' not in vars(utils)          # a lazy re-export, not a module attribute
    with pytest.raises(AttributeError):
        utils.no_such_function


def test_entry_point_is_declared():
    from unidock_processing.plugins import registry
    with open(os.path.join(REPO_ROOT, 'pyproject.toml')) as f:
        text = f.read()
    assert '[project.entry-points."unidock2.plugins"]' in text
    assert 'covalent = "unidock_processing.plugins.covalent.plugin:CovalentPlugin"' in text
    assert registry.BUILTIN_PLUGINS['covalent'] == 'unidock_processing.plugins.covalent.plugin:CovalentPlugin'
    assert registry.load_plugin_class(registry.BUILTIN_PLUGINS['covalent']) is CovalentPlugin
    assert 'covalent' in registry.plugin_specs() and 'covalent' not in registry.missing_plugins()


def test_plugin_imports_without_torch():
    code = ("import sys; sys.modules['torch'] = None; "
            "import unidock_processing.plugins.covalent.plugin, unidock_processing.plugins.covalent.receptor; "
            "import unidock_processing.core.runner, unidock_processing.io.yaml, unidock_processing.cli.docking; "
            "print('ok')")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == 'ok', result.stderr[-2000:]
