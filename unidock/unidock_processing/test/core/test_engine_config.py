"""The engine YAML of core.engine reproduces the runner's template byte for byte, and the plugin base is wired.

The golden files in data/ were written by the UnidockProtocolRunner._run_engine_subprocess template at main
1f1f2cb (before the split), with os.cpu_count patched to 8, the output dir set to /work/run1/unidock2_output and
the DMS paths below set on the runner; the four cases cover the FF lines present / absent (enriched DMS, raw DMS,
none), the cluster line, the CNN pool raise and its constraint-docking exception (through a plugin whose search
policy is constraint docking, so the core rule is tested with no plugin installed). The two template goldens were
written the same way at 2237c8d (step 1, before the template plugin): the box centre from the reference pose when
the run gives none, and an atom_mapper_align run with a manual core map; the compatibility constructor's
TemplatePlugin must reproduce them byte for byte, and they are skipped when plugins/template is not installed
(test/core passes without it). The two covalent goldens were written the same way at 705460d (step 2, before the
covalent plugin): the covalent run of test_covalent_docking.py and a covalent run with the CNN stage on (the pool is
not raised, num_pose stays 3); the compatibility constructor's CovalentPlugin must reproduce them byte for byte, and
they are skipped when plugins/covalent is not installed.
"""
import importlib.util
import logging
import os
import subprocess
import sys
import types

import pytest
import yaml

from unidock_processing.core import engine as engine_mod
from unidock_processing.core.engine import EngineConfig, render_yaml
from unidock_processing.plugins import registry
from unidock_processing.plugins.base import Plugin

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
REFERENCE_SDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                             'constraint_docking', 'automatic_atom_mapping', 'reference.sdf')
REFERENCE_CENTER = (14.79165095365234, -0.6552075201042383, -1.1135311463812374)   # its centre of mass
OUTPUT_DIR = '/work/run1/unidock2_output'
CENTER_A = (5.122, 18.327, 37.332)
HAS_PLUGIN = {name: importlib.util.find_spec(f'unidock_processing.plugins.{name}') is not None
              for name in ('template', 'covalent')}
RESIDUE_LIST = [['', 'CYX', 25, 'CA'], ['', 'CYX', 25, 'CB'], ['', 'CYX', 25, 'SG']]


class _ConstraintPlugin(Plugin):
    """A plugin whose search policy is constraint docking (the template plugin's rule), with no plugin installed."""
    name = 'constraint'

    def search_policy(self):
        return {'constraint_docking': True}


DEFAULTS = dict(
    seed=1234567, exhaustiveness=512, randomize=True, mc_steps=40, opt_steps=15, refine_steps=5,
    rmsd_limit=1.0, num_pose=10, rescore_pool=10, energy_range=5.0, use_tor_lib=False, ff_minimize_steps=0,
    ligand_dms=None, protein_dms=None, cluster_atom_indices_list=None, ncpu=8, task='screen',
    search_mode='balance', constraint_docking=False, decomp=False, center=CENTER_A,
    box_size=(30.0, 30.0, 30.0), output_dir=OUTPUT_DIR, inputs={},
)

CASES = {
    'free_dms': dict(
        ligand_dms='/work/run1/ligand_enriched.dms', protein_dms='/work/run1/receptor_enriched.dms',
        inputs={'ligand_dms_full': '/work/run1/ligand_enriched.dms',
                'receptor_dms_full': '/work/run1/receptor_enriched.dms'},
    ),
    'cnn_cluster': dict(
        seed=42, exhaustiveness=64, randomize=False, mc_steps=20, opt_steps=-1, refine_steps=3, num_pose=10,
        rescore_pool=10, rmsd_limit=2.0, energy_range=3.0, use_tor_lib=True, ff_minimize_steps=100,
        ligand_dms='/work/run1/ligand_merged.dms', protein_dms='/work/run1/receptor_enriched.dms',
        cluster_atom_indices_list=[[0, 1, 2], None, []], task='dock', search_mode='free', decomp=True,
        center=(0.5, -1.25, 10.0), box_size=(20.0, 22.5, 25.0),
        inputs={'ligand_dms_full': '/work/run1/ligand_merged.dms',
                'receptor_dms_full': '/work/run1/receptor_enriched.dms'},
    ),
    'no_dms': dict(seed=7, num_pose=10),
    'constraint_raw_dms': dict(
        exhaustiveness=128, num_pose=5, search_mode='detail', constraint_docking=True,
        ligand_dms='/work/run1/lig.dms', protein_dms='/work/run1/rec.dms',
    ),
    'template_com': dict(
        num_pose=3, constraint_docking=True, center=REFERENCE_CENTER,
        ligand_dms='/work/run1/lig.dms', protein_dms='/work/run1/rec.dms',
    ),
    'template_align': dict(
        seed=42, exhaustiveness=4, mc_steps=1, opt_steps=0, refine_steps=1, num_pose=1, use_tor_lib=True,
        search_mode='free', constraint_docking=True,
        ligand_dms='/work/run1/ligand_enriched.dms', protein_dms='/work/run1/receptor_enriched.dms',
        inputs={'ligand_dms_full': '/work/run1/ligand_enriched.dms',
                'receptor_dms_full': '/work/run1/receptor_enriched.dms'},
    ),
    'covalent': dict(
        exhaustiveness=4, mc_steps=1, opt_steps=0, refine_steps=1, search_mode='free', constraint_docking=True,
        center=(8.411, 13.047, 6.811),
        ligand_dms='/work/run1/ligand_enriched.dms', protein_dms='/work/run1/receptor_enriched.dms',
        inputs={'ligand_dms_full': '/work/run1/ligand_enriched.dms',
                'receptor_dms_full': '/work/run1/receptor_enriched.dms'},
    ),
    'covalent_cnn': dict(
        num_pose=3, constraint_docking=True,
        ligand_dms='/work/run1/lig.dms', protein_dms='/work/run1/rec.dms',
    ),
}

# the runner settings that produced the same goldens (CNN on raises num_pose 3 -> 10 and 5 -> 10, not under
# constraint docking: the core _ConstraintPlugin policy for constraint_raw_dms, the template plugin for the rest)
RUNNER_CASES = {
    'free_dms': (dict(target_center=CENTER_A, box_size=(30.0, 30.0, 30.0)),
                 dict(enriched_ligand_dms='/work/run1/ligand_enriched.dms',
                      enriched_receptor_dms='/work/run1/receptor_enriched.dms')),
    'cnn_cluster': (dict(target_center=(0.5, -1.25, 10.0), box_size=(20.0, 22.5, 25.0), task='dock',
                         search_mode='free', exhaustiveness=64, randomize=False, mc_steps=20, opt_steps=-1,
                         refine_steps=3, num_pose=3, rescore_pool=10, rmsd_limit=2.0, energy_range=3.0, seed=42,
                         use_tor_lib=True, decomp=True, ff_minimize_steps=100, cnn_rescore=True,
                         cluster_atom_indices_list=[[0, 1, 2], None, []]),
                    dict(enriched_ligand_dms='/work/run1/ligand_merged.dms',
                         enriched_receptor_dms='/work/run1/receptor_enriched.dms')),
    'no_dms': (dict(target_center=CENTER_A, num_pose=5, rescore_pool=10, cnn_rescore=True, seed=7), {}),
    'constraint_raw_dms': (dict(target_center=CENTER_A, num_pose=5, rescore_pool=10, cnn_rescore=True,
                                search_mode='detail', exhaustiveness=128, plugins=[_ConstraintPlugin()]),
                           dict(ligand_dms='/work/run1/lig.dms', protein_dms='/work/run1/rec.dms')),
    'template_com': (dict(target_center=(0.0, 0.0, 0.0), template_docking=True, reference_sdf_file_name=REFERENCE_SDF,
                          num_pose=3, rescore_pool=10, cnn_rescore=True),
                     dict(ligand_dms='/work/run1/lig.dms', protein_dms='/work/run1/rec.dms')),
    'template_align': (dict(target_center=CENTER_A, template_docking=True, reference_sdf_file_name=REFERENCE_SDF,
                            core_atom_mapping_dict_list=[{'0': 14, '1': 15}], atom_mapper_align=True,
                            randomize=True, search_mode='free', exhaustiveness=4, mc_steps=1, opt_steps=0,
                            refine_steps=1, num_pose=1, use_tor_lib=True, seed=42),
                       dict(enriched_ligand_dms='/work/run1/ligand_enriched.dms',
                            enriched_receptor_dms='/work/run1/receptor_enriched.dms')),
    'covalent': (dict(target_center=(8.411, 13.047, 6.811), covalent_ligand=True,
                      covalent_residue_atom_info_list=RESIDUE_LIST, preserve_receptor_hydrogen=True,
                      search_mode='free', exhaustiveness=4, mc_steps=1, opt_steps=0, refine_steps=1),
                 dict(enriched_ligand_dms='/work/run1/ligand_enriched.dms',
                      enriched_receptor_dms='/work/run1/receptor_enriched.dms')),
    'covalent_cnn': (dict(target_center=CENTER_A, covalent_ligand=True, covalent_residue_atom_info_list=RESIDUE_LIST,
                          num_pose=3, rescore_pool=10, cnn_rescore=True),
                     dict(ligand_dms='/work/run1/lig.dms', protein_dms='/work/run1/rec.dms')),
}
PLUGIN_OF_CASE = {'template_com': 'template', 'template_align': 'template',
                  'covalent': 'covalent', 'covalent_cnn': 'covalent'}
EXPECTED_PLUGINS = {'constraint_raw_dms': ['constraint'], 'template_com': ['template'], 'template_align': ['template'],
                    'covalent': ['covalent'], 'covalent_cnn': ['covalent']}


def golden(name):
    with open(os.path.join(DATA_DIR, f'{name}.yaml')) as f:
        return f.read()


@pytest.mark.parametrize('name', sorted(CASES))
def test_render_yaml_matches_golden(name):
    cfg = EngineConfig(**{**DEFAULTS, **CASES[name]})
    assert render_yaml(cfg, {}) == golden(name)
    assert render_yaml(cfg) == golden(name)


def test_extra_blocks_follow_the_core_blocks():
    cfg = EngineConfig(**{**DEFAULTS, **CASES['free_dms']})
    text = render_yaml(cfg, {'bias': {'type': 'align', 'k': 0.05, 'on': True, 'atoms': [1, 2]}})
    assert text == golden('free_dms') + "\nbias:\n  type: align\n  k: 0.05\n  on: true\n  atoms: [1, 2]\n"


def test_core_block_extras_merge_into_the_block():
    """A plugin block named after a core block is merged into it, its keys after the core keys (here the engine's
    deprecated Advanced.bias / Advanced.bias_k aliases); any other name stays a top-level block after the core ones
    (the bias plugin's own `bias:` block), and no extra block leaves the text as the golden."""
    cfg = EngineConfig(**{**DEFAULTS, **CASES['free_dms']})
    base = golden('free_dms')
    assert engine_mod.CORE_BLOCKS == ('Advanced', 'Hardware', 'Settings', 'Outputs', 'Inputs')
    assert engine_mod.split_extra_blocks(None) == ({name: {} for name in engine_mod.CORE_BLOCKS}, {})

    text = render_yaml(cfg, {'Advanced': {'bias': 'align', 'bias_k': 0.05}})
    assert text == base.replace('  tor_lib: false\n', '  tor_lib: false\n  bias: align\n  bias_k: 0.05\n')
    parsed = yaml.safe_load(text)
    assert parsed['Advanced']['bias'] == 'align' and parsed['Advanced']['bias_k'] == 0.05
    assert parsed['Advanced']['tor_lib'] is False and parsed['Advanced']['protein_dms'] == '/work/run1/receptor_enriched.dms'

    blocks = {'Hardware': {'gpu': 1}, 'Settings': {'inner_box': 10.0}, 'Outputs': {'keep': True},
              'Inputs': {'bias_table': '/work/run1/bias.dms'}, 'bias': {'type': 'align'}}
    text = render_yaml(cfg, blocks)
    expected = (base.replace('  ncpu: 8\n', '  ncpu: 8\n  gpu: 1\n')
                    .replace('  size_z: 30.0\n', '  size_z: 30.0\n  inner_box: 10.0\n')
                    .replace(f'  dir: {OUTPUT_DIR}\n', f'  dir: {OUTPUT_DIR}\n  keep: true\n')
                + '  bias_table: /work/run1/bias.dms\n' + '\nbias:\n  type: align\n')
    assert text == expected
    parsed = yaml.safe_load(text)
    assert parsed['Hardware'] == {'ncpu': 8, 'gpu': 1} and parsed['Settings']['inner_box'] == 10.0
    assert parsed['Outputs'] == {'dir': OUTPUT_DIR, 'keep': True} and parsed['bias'] == {'type': 'align'}
    assert parsed['Inputs']['bias_table'] == '/work/run1/bias.dms'
    assert render_yaml(cfg, {name: {} for name in engine_mod.CORE_BLOCKS}) == base


def test_pybind11_forwards_core_block_keys_and_refuses_other_blocks(monkeypatch):
    """With no ud2 binary, run_engine goes through the pybind11 pipeline with the keys merged into a core block as
    keyword arguments of the same name (the deprecated Advanced.bias / bias_k aliases, which the pipeline takes),
    maps the engine's own `bias` block onto the pipeline's bias / bias_k arguments (the block wins over an alias),
    and refuses a plugin block or a bias key the pipeline has no argument for instead of running the plain docking."""
    calls = []
    fake_pipeline = types.ModuleType('unidock_engine.api.python.pipeline')
    fake_pipeline.run_docking_pipeline = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, 'unidock_engine.api.python.pipeline', fake_pipeline)
    monkeypatch.setattr(engine_mod, 'find_ud2_binary', lambda: None)
    cfg = EngineConfig(**{**DEFAULTS, **CASES['free_dms']})

    engine_mod.run_engine(cfg, {'Advanced': {'bias': 'align', 'bias_k': 0.05}})
    assert len(calls) == 1 and calls[0]['bias'] == 'align' and calls[0]['bias_k'] == 0.05
    assert calls[0]['ligand_dms_path'] == '/work/run1/ligand_enriched.dms' and calls[0]['seed'] == 1234567
    engine_mod.run_engine(cfg)                                    # the plain call: no extra keyword
    engine_mod.run_engine(cfg, {})
    assert set(calls[1]) == set(calls[2]) == set(calls[0]) - {'bias', 'bias_k'}
    engine_mod.run_engine(cfg, {'bias': {'type': 'pos', 'k': 0.3}})            # the bias plugin's block
    assert calls[3]['bias'] == 'pos' and calls[3]['bias_k'] == 0.3 and set(calls[3]) == set(calls[0])
    engine_mod.run_engine(cfg, {'Advanced': {'bias_k': 0.05}, 'bias': {'type': 'align', 'k': 0.2}})
    assert calls[4]['bias'] == 'align' and calls[4]['bias_k'] == 0.2            # the block wins over the alias
    assert engine_mod.BIAS_BLOCK_ARGS == {'type': 'bias', 'k': 'bias_k'}
    with pytest.raises(RuntimeError, match=r"no argument for the bias block keys \['ring'\]"):
        engine_mod.run_engine(cfg, {'bias': {'type': 'align', 'ring': 1}})
    with pytest.raises(RuntimeError, match=r"no argument for the plugin blocks \['ring'\]"):
        engine_mod.run_engine(cfg, {'bias': {'type': 'align'}, 'ring': {'k': 0.1}})
    assert len(calls) == 5


def test_pybind11_refuses_ring_tables_against_a_module_without_the_ring_term(tmp_path, monkeypatch):
    """A pipeline module older than ud2 0.4.6 ignores the ring-bias tables of the ligand DMS and docks without the
    ring term, exit 0: run_engine_pybind11 refuses a DMS with them against a module that reports no __version__ (all
    modules before 0.4.6) or an older one, runs it on 0.4.6, and runs a ring-free DMS on any module."""
    import sqlite3
    calls = []
    fake_pipeline = types.ModuleType('unidock_engine.api.python.pipeline')
    fake_pipeline.run_docking_pipeline = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, 'unidock_engine.api.python.pipeline', fake_pipeline)
    monkeypatch.setattr(engine_mod, 'find_ud2_binary', lambda: None)
    plain_dms, ring_dms = str(tmp_path / 'plain.dms'), str(tmp_path / 'ring.dms')
    for path, tables in ((plain_dms, ['particle']), (ring_dms, ['particle', 'ud2_ring_bias'])):
        with sqlite3.connect(path) as conn:
            for table in tables:
                conn.execute(f"CREATE TABLE {table} (id INTEGER)")
    case = CASES['free_dms']
    plain_cfg = EngineConfig(**{**DEFAULTS, **case, 'inputs': {**case['inputs'], 'ligand_dms_full': plain_dms}})
    ring_cfg = EngineConfig(**{**DEFAULTS, **case, 'inputs': {**case['inputs'], 'ligand_dms_full': ring_dms}})
    bias = {'bias': {'type': 'align', 'k': 0.05}}

    assert engine_mod.pipeline_version(fake_pipeline) is None
    with pytest.raises(RuntimeError, match=r"pipeline module unidock_engine\.api\.python\.pipeline reports no version "
                                           r"and does not read the ring-bias tables ud2_ring_bias / ud2_ring_bias_atoms "
                                           r"of .*ring\.dms \(ud2 >= 0\.4\.6\)"):
        engine_mod.run_engine(ring_cfg, bias)
    engine_mod.run_engine(plain_cfg, bias)                                  # the step-5 run: any module
    engine_mod.run_engine(plain_cfg)
    fake_pipeline.__version__ = '0.4.5'
    assert engine_mod.pipeline_version(fake_pipeline) == (0, 4, 5)
    with pytest.raises(RuntimeError, match=r"pipeline module .* is version 0\.4\.5 and does not read the ring-bias"):
        engine_mod.run_engine(ring_cfg)
    assert len(calls) == 2
    fake_pipeline.__version__ = '0.4.6'
    engine_mod.run_engine(ring_cfg, bias)
    assert len(calls) == 3 and calls[2]['ligand_dms_path'] == ring_dms and calls[2]['bias'] == 'align'
    fake_pipeline.__version__ = '1.0.0'
    engine_mod.run_engine(ring_cfg)
    assert len(calls) == 4


def test_update_applies_known_keys_and_rejects_unknown():
    cfg = EngineConfig(**DEFAULTS)
    cfg.update({'constraint_docking': True, 'randomize': False})
    assert cfg.constraint_docking is True and cfg.randomize is False
    with pytest.raises(KeyError):
        cfg.update({'static_root': True})


class _FakePlugin(Plugin):
    name = 'fake'

    def __init__(self):
        self.calls = []

    def configure(self, config):
        self.calls.append(('configure', config['seed']))

    def annotate_receptor(self, builder):
        self.calls.append(('annotate_receptor', builder))

    def annotate_ligand_dms(self, merged_dms, offsets):
        self.calls.append(('annotate_ligand_dms', merged_dms, offsets))

    def yaml_block(self):
        return {'bias': {'type': 'align', 'k': 0.05}}

    def search_policy(self):
        return {'randomize': False}

    def post_process(self, sdf_path):
        self.calls.append(('post_process', sdf_path))


def _runner(tmp_path, monkeypatch, kwargs, attrs):
    """A runner with no registry plugins: only what the compatibility constructor builds (the template plugin)."""
    from unidock_processing.core import runner as runner_mod
    monkeypatch.setattr(runner_mod, 'load_plugins', lambda: [])
    runner = runner_mod.UnidockProtocolRunner('rec.pdb', ['lig.sdf'], working_dir_name=str(tmp_path), **kwargs)
    runner.unidock2_output_dir_name = OUTPUT_DIR
    for key, value in attrs.items():
        setattr(runner, key, value)
    return runner


def _fake_ud2_run(commands, probes, version='0.4.5', returncode=0):
    """A subprocess.run double for the ud2 binary: `--version` prints the engine's version line (recorded in
    *probes*), any other call is a run (recorded in *commands*) ending with *returncode*."""
    def run(cmd, **kw):
        if cmd[1:] == ['--version']:
            probes.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout=f'UD2 C++ Engine Version: {version}\n', stderr='')
        commands.append(cmd)
        return types.SimpleNamespace(returncode=returncode)
    return run


def _run_engine_capturing_yaml(monkeypatch, runner, probes=None):
    commands = []
    monkeypatch.setattr(engine_mod, 'find_ud2_binary', lambda: '/fake/ud2')
    monkeypatch.setattr(engine_mod.subprocess, 'run', _fake_ud2_run(commands, [] if probes is None else probes))
    monkeypatch.setattr(os, 'cpu_count', lambda: 8)
    runner._run_engine()
    config_path = os.path.join(runner.working_dir_name, '_ud2_engine_config.yaml')
    assert commands == [['/fake/ud2', config_path]]
    with open(config_path) as f:
        return f.read()


@pytest.mark.parametrize('name', sorted(RUNNER_CASES))
def test_runner_writes_the_golden_yaml(tmp_path, monkeypatch, name):
    kwargs, attrs = RUNNER_CASES[name]
    plugin_name = PLUGIN_OF_CASE.get(name)
    if plugin_name and not HAS_PLUGIN[plugin_name]:
        pytest.skip(f"the {plugin_name} plugin is not installed")
    if kwargs.get('template_docking') and not os.path.isfile(REFERENCE_SDF):
        pytest.skip(f"missing {REFERENCE_SDF}")
    runner = _runner(tmp_path, monkeypatch, kwargs, attrs)
    assert [plugin.name for plugin in runner.plugins] == EXPECTED_PLUGINS.get(name, [])
    assert _run_engine_capturing_yaml(monkeypatch, runner) == golden(name)


def test_template_run_without_the_plugin_is_a_clear_error(tmp_path, monkeypatch):
    """A requested template run degrades to one error naming the plugin, not the runner's own import failure."""
    monkeypatch.setitem(sys.modules, 'unidock_processing.plugins.template.plugin', None)     # unimportable
    for kwargs in (dict(template_docking=True, reference_sdf_file_name='ref.sdf'), dict(atom_mapper_align=True)):
        with pytest.raises(ImportError, match='template plugin'):
            _runner(tmp_path, monkeypatch, dict(target_center=CENTER_A, **kwargs), {})
    plain = _runner(tmp_path, monkeypatch, dict(target_center=CENTER_A), {})
    assert plain.plugins == [] and not plain._constraint_docking()


def test_covalent_run_without_the_plugin_is_a_clear_error(tmp_path, monkeypatch):
    """A requested covalent run (the flag, the residue list, or both) degrades to one error naming the plugin."""
    monkeypatch.setitem(sys.modules, 'unidock_processing.plugins.covalent.plugin', None)     # unimportable
    for kwargs in (dict(covalent_ligand=True, covalent_residue_atom_info_list=RESIDUE_LIST),
                   dict(covalent_ligand=True), dict(covalent_residue_atom_info_list=RESIDUE_LIST)):
        with pytest.raises(ImportError, match='covalent plugin'):
            _runner(tmp_path, monkeypatch, dict(target_center=CENTER_A, **kwargs), {})
    plain = _runner(tmp_path, monkeypatch, dict(target_center=CENTER_A), {})
    assert plain.plugins == [] and not plain._constraint_docking()


def test_yaml_plugin_keys_without_the_plugin(caplog):
    """With a known plugin not installed (the registry reports it missing), its block or its deprecated
    Preprocessing keys that set anything are one clear error naming the plugin, and a block with every value at
    its default is dropped with a warning, never silently; the Preprocessing keys stay readable."""
    from unidock_processing.io import yaml as yaml_mod
    missing = {'template': 'unidock_processing.plugins.template.plugin:TemplatePlugin',
               'covalent': 'unidock_processing.plugins.covalent.plugin:CovalentPlugin'}
    mapping = yaml_mod.build_config_mapping([])
    assert [entry[0] for entry in mapping] == ['Required', 'Advanced', 'Hardware', 'Settings', 'Preprocessing']
    assert 'template_docking' in mapping[-1][3]                # stays in Preprocessing without the plugin
    assert 'covalent_ligand' in mapping[-1][3] and 'covalent_residue_atom_info_list' in mapping[-1][3]
    UnidockConfig = yaml_mod.build_unidock_model(mapping, missing)
    with pytest.raises(ImportError, match='covalent plugin'):
        UnidockConfig.from_dict({'covalent': {'covalent_ligand': True}})
    with pytest.raises(ImportError, match='covalent plugin'):
        UnidockConfig.from_dict({'Preprocessing': {'covalent_residue_atom_info_list': RESIDUE_LIST}})
    with pytest.raises(ImportError, match='template plugin'):
        UnidockConfig.from_dict({'template': {'template_docking': True, 'reference_sdf_file_name': 'r.sdf'}})
    with pytest.raises(ImportError, match='template plugin'):
        UnidockConfig.from_dict({'Preprocessing': {'template_docking': True}})
    with pytest.raises(ImportError, match='template plugin'):
        UnidockConfig.from_dict({'Preprocessing': {'reference_sdf_file_name': 'r.sdf'}})
    with caplog.at_level(logging.WARNING, logger='unidock_processing.io.yaml'):
        UnidockConfig.from_dict({'template': {'template_docking': False, 'reference_sdf_file_name': None}})
    assert "'template:' block ignored" in caplog.text
    with caplog.at_level(logging.WARNING, logger='unidock_processing.io.yaml'):
        UnidockConfig.from_dict({'covalent': {'covalent_ligand': False, 'covalent_residue_atom_info_list': None}})
    assert "'covalent:' block ignored" in caplog.text
    kwargs = UnidockConfig.from_dict({'Preprocessing': {'covalent_ligand': False,
                                                        'template_docking': False}}).to_protocol_kwargs()
    assert kwargs['covalent_ligand'] is False and kwargs['template_docking'] is False
    assert set(DEPRECATED_KEYS_OF('template')) == {'template_docking', 'reference_sdf_file_name',
                                                   'core_atom_mapping_dict_list'}
    assert set(DEPRECATED_KEYS_OF('covalent')) == {'covalent_ligand', 'covalent_residue_atom_info_list'}


def DEPRECATED_KEYS_OF(plugin_name):
    from unidock_processing.io.yaml import DEPRECATED_PREPROCESSING_KEYS
    return [key for key, name in DEPRECATED_PREPROCESSING_KEYS.items() if name == plugin_name]


def test_yaml_blocks_follow_the_installed_plugins():
    """The YAML reader takes its plugin blocks from the registry: a plugin class with a config_model adds the block
    named after it, one without adds nothing, and the deprecated Preprocessing keys leave that block's field list
    only for an installed plugin."""
    from pydantic import BaseModel
    from unidock_processing.io import yaml as yaml_mod

    class _BiasConfig(BaseModel):
        k: float = 0.05
        type: str = 'align'

    class _Bias(Plugin):
        name = 'bias'
        config_model = _BiasConfig

    mapping = yaml_mod.build_config_mapping([_FakePlugin, _Bias])
    assert [entry[0] for entry in mapping] == ['Required', 'Advanced', 'Hardware', 'Settings', 'Preprocessing', 'bias']
    assert mapping[-1][1:] == ('bias', _BiasConfig, ['k', 'type'])
    assert 'template_docking' in mapping[-2][3]
    UnidockConfig = yaml_mod.build_unidock_model(mapping, {})
    kwargs = UnidockConfig.from_dict({'bias': {'k': 0.1}}).to_protocol_kwargs()
    assert kwargs['k'] == 0.1 and kwargs['type'] == 'align' and kwargs['template_docking'] is False


def test_yaml_plugin_block_builds_the_configured_instance():
    """A plugin whose from_config builds an instance from its block gets it in the runner's 'plugins' argument and
    its fields are not flattened (bias_k is no runner keyword argument); one whose from_config returns None is
    flattened as before (the template and covalent compatibility route), and 'plugins' is absent when no plugin
    builds one."""
    from pydantic import BaseModel
    from unidock_processing.io import yaml as yaml_mod

    class _BiasConfig(BaseModel):
        bias_type: str = 'align'
        bias_k: float = 0.05

    class _Bias(Plugin):
        name = 'bias'
        config_model = _BiasConfig

        def __init__(self, bias_type, bias_k):
            self.bias_type, self.bias_k = bias_type, bias_k

        @classmethod
        def from_config(cls, block):
            return cls(block.bias_type, block.bias_k)

    assert Plugin.from_config(None) is None
    classes = [_FakePlugin, _Bias]
    UnidockConfig = yaml_mod.build_unidock_model(yaml_mod.build_config_mapping(classes), {}, classes)
    kwargs = UnidockConfig.from_dict({'bias': {'bias_k': 0.1}, 'Advanced': {'seed': 3}}).to_protocol_kwargs()
    plugin, = kwargs['plugins']
    assert isinstance(plugin, _Bias) and (plugin.bias_type, plugin.bias_k) == ('align', 0.1)
    assert 'bias_k' not in kwargs and 'bias_type' not in kwargs and kwargs['seed'] == 3
    assert kwargs['template_docking'] is False                     # Preprocessing, flattened as before
    defaults = UnidockConfig.from_dict({}).to_protocol_kwargs()
    assert [(p.bias_type, p.bias_k) for p in defaults['plugins']] == [('align', 0.05)]

    installed = yaml_mod.UnidockConfig.from_dict({}).to_protocol_kwargs()
    assert 'plugins' not in installed                               # template / covalent flatten (from_config None)
    for plugin_cls in yaml_mod.PLUGIN_CLASSES:
        if plugin_cls.config_model is not None:
            assert plugin_cls.from_config(plugin_cls.config_model()) is None


def test_runner_applies_plugin_policy_and_blocks(tmp_path, monkeypatch):
    kwargs, attrs = RUNNER_CASES['free_dms']
    runner = _runner(tmp_path, monkeypatch, kwargs, attrs)
    runner.plugins = [_FakePlugin()]
    text = _run_engine_capturing_yaml(monkeypatch, runner)
    expected = golden('free_dms').replace('  randomize: true\n', '  randomize: false\n')
    assert text == expected + "\nbias:\n  type: align\n  k: 0.05\n"


class _AdvancedBiasPlugin(Plugin):
    """A plugin block named after a core block (the engine's deprecated Advanced.bias / bias_k aliases): merged."""
    name = 'bias'

    def yaml_block(self):
        return {'Advanced': {'bias': 'align', 'bias_k': 0.05}}


def test_runner_merges_a_plugin_block_into_the_core_block(tmp_path, monkeypatch):
    kwargs, attrs = RUNNER_CASES['free_dms']
    runner = _runner(tmp_path, monkeypatch, dict(kwargs, plugins=[_AdvancedBiasPlugin()]), attrs)
    assert [plugin.name for plugin in runner.plugins] == ['bias']
    text = _run_engine_capturing_yaml(monkeypatch, runner)
    assert text == golden('free_dms').replace('  tor_lib: false\n', '  tor_lib: false\n  bias: align\n  bias_k: 0.05\n')


class _OtherPlugin(Plugin):
    name = 'other'


def test_runner_takes_configured_plugin_instances(tmp_path, monkeypatch):
    """plugins= hands configured instances to the runner (the entry point of a plugin with no compatibility keyword
    arguments): each takes the place of the registry's instance of the same name, after the registry's others and
    in the order given; the same plugin given through its compatibility arguments as well is an error."""
    from unidock_processing.core import runner as runner_mod
    registry_fake, registry_other = _FakePlugin(), _OtherPlugin()
    monkeypatch.setattr(runner_mod, 'load_plugins', lambda: [registry_fake, registry_other])
    make = lambda **kwargs: runner_mod.UnidockProtocolRunner('rec.pdb', ['lig.sdf'], target_center=CENTER_A,
                                                              working_dir_name=str(tmp_path), **kwargs)
    assert make().plugins == [registry_fake, registry_other] == make(plugins=[]).plugins
    mine, other = _FakePlugin(), _OtherPlugin()
    runner = make(plugins=[mine])
    assert runner.plugins == [registry_other, mine] and runner._search_policy() == {'randomize': False}
    assert make(plugins=[other, mine]).plugins == [other, mine]
    for name, kwargs in (('covalent', dict(covalent_ligand=True)), ('template', dict(atom_mapper_align=True))):
        if not HAS_PLUGIN[name]:
            continue
        given = type('_Given', (Plugin,), {'name': name})()
        with pytest.raises(ValueError, match=f'{name} plugin is given twice'):
            make(plugins=[given], **kwargs)


def test_engine_failure_raises(tmp_path, monkeypatch):
    kwargs, attrs = RUNNER_CASES['free_dms']
    runner = _runner(tmp_path, monkeypatch, kwargs, attrs)
    monkeypatch.setattr(engine_mod, 'find_ud2_binary', lambda: '/fake/ud2')
    monkeypatch.setattr(engine_mod.subprocess, 'run', lambda cmd, **kw: types.SimpleNamespace(returncode=3))
    with pytest.raises(RuntimeError, match='return code 3'):
        runner._run_engine()


def _fake_ud2_script(path, version_line, body=''):
    path.write_text('#!/bin/sh\n'
                    f'case "$1" in --version) {version_line}; exit 0;; esac\n'
                    f'{body}\n')
    path.chmod(0o755)
    return str(path)


def test_ud2_version_and_require_engine_blocks(tmp_path):
    """ud2_version parses the binary's `--version` line; require_engine_blocks probes the binary only for a top-level
    block with an entry in ENGINE_BLOCK_MIN_VERSION and refuses a binary that predates it (the bias block needs
    0.4.5: an older ud2 ignores the block and docks without the bias, exit 0)."""
    assert engine_mod.ENGINE_BLOCK_MIN_VERSION == {'bias': (0, 4, 5)}
    old = _fake_ud2_script(tmp_path / 'ud2_old', 'echo "UD2 C++ Engine Version: 0.4.4"')
    new = _fake_ud2_script(tmp_path / 'ud2_new', 'echo "UD2 C++ Engine Version: 0.4.5"')
    later = _fake_ud2_script(tmp_path / 'ud2_later', 'echo "UD2 C++ Engine Version: 1.0.12"')
    mute = _fake_ud2_script(tmp_path / 'ud2_mute', 'echo "no version here"')
    assert engine_mod.ud2_version(old) == (0, 4, 4)
    assert engine_mod.ud2_version(new) == (0, 4, 5)
    assert engine_mod.ud2_version(later) == (1, 0, 12)
    with pytest.raises(RuntimeError, match='cannot read the version'):
        engine_mod.ud2_version(mute)
    with pytest.raises(RuntimeError, match='cannot run'):
        engine_mod.ud2_version(str(tmp_path / 'missing_ud2'))

    bias = {'bias': {'type': 'align', 'k': 0.05}}
    engine_mod.require_engine_blocks(new, bias)
    engine_mod.require_engine_blocks(later, bias)
    with pytest.raises(RuntimeError, match=r"version 0\.4\.4 and does not read the block\(s\) `bias:` \(ud2 >= 0\.4\.5\)"):
        engine_mod.require_engine_blocks(old, bias)
    with pytest.raises(RuntimeError, match='cannot read the version'):
        engine_mod.require_engine_blocks(mute, bias)
    # no probe without a versioned top-level block: a missing binary is never run
    missing = str(tmp_path / 'missing_ud2')
    engine_mod.require_engine_blocks(missing, None)
    engine_mod.require_engine_blocks(missing, {})
    engine_mod.require_engine_blocks(missing, {'Advanced': {'bias': 'align', 'bias_k': 0.05}})   # the aliases
    engine_mod.require_engine_blocks(missing, {'ring': {'k': 0.1}})                             # no entry
    with pytest.raises(RuntimeError, match='cannot run'):
        engine_mod.require_engine_blocks(missing, {'ring': {'k': 0.1}, **bias})


def test_require_engine_blocks_ring_bias(tmp_path):
    """The aromatic-ring bias needs ud2 0.4.6 (RING_BIAS_MIN_VERSION): a 0.4.5 binary reads the bias block but ignores
    its ring: sub-block and the ring tables of the ligand DMS, so either is refused against it and accepted against
    0.4.6; a DMS with the tables makes the probe even without a block; a ring-free or non-SQLite DMS changes nothing."""
    import sqlite3
    assert engine_mod.RING_BIAS_MIN_VERSION == (0, 4, 6)
    assert engine_mod.RING_BIAS_TABLES == ('ud2_ring_bias', 'ud2_ring_bias_atoms')
    v045 = _fake_ud2_script(tmp_path / 'ud2_045', 'echo "UD2 C++ Engine Version: 0.4.5"')
    v046 = _fake_ud2_script(tmp_path / 'ud2_046', 'echo "UD2 C++ Engine Version: 0.4.6"')
    missing = str(tmp_path / 'missing_ud2')
    bias = {'bias': {'type': 'align', 'k': 0.05}}
    ring_block = {'bias': {'type': 'align', 'k': 0.05, 'ring': {'k': 20}}}
    plain_dms = str(tmp_path / 'plain.dms')
    ring_dms = str(tmp_path / 'ring.dms')
    for path, tables in ((plain_dms, ['particle']), (ring_dms, ['particle', 'ud2_ring_bias_atoms'])):
        with sqlite3.connect(path) as conn:
            for table in tables:
                conn.execute(f"CREATE TABLE {table} (id INTEGER)")
    not_sqlite = tmp_path / 'not.dms'
    not_sqlite.write_text('not a database\n')
    assert engine_mod.dms_has_ring_bias(ring_dms)
    assert not engine_mod.dms_has_ring_bias(plain_dms) and not engine_mod.dms_has_ring_bias(str(not_sqlite))
    assert not engine_mod.dms_has_ring_bias(None) and not engine_mod.dms_has_ring_bias(str(tmp_path / 'none.dms'))

    engine_mod.require_engine_blocks(v045, bias, plain_dms)                  # the step-5 run: unchanged
    engine_mod.require_engine_blocks(v045, bias, str(not_sqlite))
    with pytest.raises(RuntimeError, match=r"version 0\.4\.5 and does not read the `ring:` sub-block of `bias:` "
                                           r"\(ud2 >= 0\.4\.6\)"):
        engine_mod.require_engine_blocks(v045, ring_block)
    with pytest.raises(RuntimeError, match=r"version 0\.4\.5 and does not read the ring-bias tables "
                                           r"ud2_ring_bias / ud2_ring_bias_atoms of .*ring\.dms \(ud2 >= 0\.4\.6\)"):
        engine_mod.require_engine_blocks(v045, bias, ring_dms)
    engine_mod.require_engine_blocks(v046, ring_block, ring_dms)
    with pytest.raises(RuntimeError, match='cannot run'):
        engine_mod.require_engine_blocks(missing, None, ring_dms)            # the tables alone make the probe
    engine_mod.require_engine_blocks(missing, None, plain_dms)
    old = _fake_ud2_script(tmp_path / 'ud2_old', 'echo "UD2 C++ Engine Version: 0.4.4"')
    with pytest.raises(RuntimeError, match=r"the block\(s\) `bias:` \(ud2 >= 0\.4\.5\); the `ring:` sub-block"):
        engine_mod.require_engine_blocks(old, ring_block)
    assert engine_mod._yaml_scalar({'k': 20, 'd0': 1.5}) == '{"k": 20, "d0": 1.5}'


def test_subprocess_route_refuses_a_binary_that_predates_the_bias_block(tmp_path, monkeypatch):
    """The runner's subprocess route against an old ud2: the bias plugin's block is refused before the YAML is
    written or the binary run; the same run against a 0.4.5 binary probes once and runs once; a run without a
    top-level block never probes."""
    kwargs, attrs = RUNNER_CASES['free_dms']
    runner = _runner(tmp_path, monkeypatch, kwargs, attrs)
    runner.plugins = [_FakePlugin()]                                         # yaml_block: the bias block
    commands, probes = [], []
    monkeypatch.setattr(engine_mod, 'find_ud2_binary', lambda: '/fake/ud2')
    monkeypatch.setattr(engine_mod.subprocess, 'run', _fake_ud2_run(commands, probes, version='0.4.4'))
    monkeypatch.setattr(os, 'cpu_count', lambda: 8)
    config_path = os.path.join(runner.working_dir_name, '_ud2_engine_config.yaml')
    with pytest.raises(RuntimeError, match=r"/fake/ud2 is version 0\.4\.4 .* `bias:` \(ud2 >= 0\.4\.5\)"):
        runner._run_engine()
    assert probes == [['/fake/ud2', '--version']] and commands == [] and not os.path.exists(config_path)

    probes = []
    text = _run_engine_capturing_yaml(monkeypatch, runner, probes)
    assert probes == [['/fake/ud2', '--version']] and text.endswith("\nbias:\n  type: align\n  k: 0.05\n")

    runner.plugins = []
    probes = []
    _run_engine_capturing_yaml(monkeypatch, runner, probes)
    assert probes == []


def test_engine_num_pose_rule():
    from unidock_processing.core.rescore import engine_num_pose
    assert engine_num_pose(3, 10, True, False) == 10
    assert engine_num_pose(3, 10, True, True) == 3
    assert engine_num_pose(30, 10, True, False) == 30
    assert engine_num_pose(3, 10, False, False) == 3
    # a plugin keeps the engine order (a biased run): no CNN stage, so no pool raise
    assert engine_num_pose(3, 10, True, False, keeps_engine_order=True) == 3
    assert engine_num_pose(3, 10, True, True, keeps_engine_order=True) == 3
    assert engine_num_pose(30, 10, True, False, keeps_engine_order=True) == 30
    assert engine_num_pose(3, 10, False, False, keeps_engine_order=True) == 3
    assert engine_num_pose(3, 10, True, False, keeps_engine_order=False) == 10


class _EngineOrderPlugin(Plugin):
    """A plugin that keeps the engine order (the bias plugin's rule, Plugin.keeps_engine_order), with no plugin
    installed."""
    name = 'engine_order'

    def keeps_engine_order(self):
        return True


def _record_cnn_calls(monkeypatch):
    """The runner's CNN stage call (core.rescore.run_cnn_rescore as the runner imported it), recorded, not run."""
    from unidock_processing.core import runner as runner_mod
    calls = []
    monkeypatch.setattr(runner_mod, 'run_cnn_rescore', lambda *args, **kwargs: calls.append(kwargs))
    return calls


def test_a_run_that_keeps_the_engine_order_has_no_cnn_stage(tmp_path, monkeypatch, caplog):
    """With the CNN stage on, a run whose plugin keeps the engine order (the core rule through _EngineOrderPlugin;
    the enabled BiasPlugin's case is tested by the external ud2align package)
    renders the requested num_pose, not the rescore pool, and the runner never calls the CNN stage; the same run
    without the plugin raises the pool and calls it."""
    name = _EngineOrderPlugin.name
    kwargs, attrs = RUNNER_CASES['no_dms']                    # num_pose 5, rescore_pool 10, cnn_rescore on
    calls = _record_cnn_calls(monkeypatch)

    plain = _runner(tmp_path, monkeypatch, kwargs, attrs)
    assert plain._engine_num_pose() == 10
    assert _run_engine_capturing_yaml(monkeypatch, plain) == golden('no_dms')
    plain._run_cnn_rescore()
    assert [(call['keep'], call['pool_written']) for call in calls] == [(5, 10)]

    calls.clear()
    plugin = _EngineOrderPlugin()
    biased = _runner(tmp_path, monkeypatch, dict(kwargs, plugins=[plugin]), attrs)
    assert biased.plugins == [plugin] and plugin.keeps_engine_order()
    assert not biased._constraint_docking() and biased._engine_num_pose() == 5
    expected = golden('no_dms').replace('  num_pose: 10\n', '  num_pose: 5\n')
    assert _run_engine_capturing_yaml(monkeypatch, biased) == expected
    with caplog.at_level(logging.INFO, logger='unidock_processing.core.runner'):
        biased._run_cnn_rescore()
    assert calls == []
    assert f"CNN rescoring skipped: plugin {name} keeps the engine order (Plugin.keeps_engine_order)" in caplog.text


@pytest.mark.parametrize('name', ['constraint_raw_dms', 'template_com', 'covalent_cnn'])
def test_constraint_docking_keeps_its_cnn_rule(tmp_path, monkeypatch, caplog, name):
    """Constraint docking (the core _ConstraintPlugin, the template and the covalent plugin) with the CNN stage on:
    the requested num_pose (the goldens), no CNN call and the constraint-docking log line, as before."""
    kwargs, attrs = RUNNER_CASES[name]
    plugin_name = PLUGIN_OF_CASE.get(name)
    if plugin_name and not HAS_PLUGIN[plugin_name]:
        pytest.skip(f"the {plugin_name} plugin is not installed")
    if kwargs.get('template_docking') and not os.path.isfile(REFERENCE_SDF):
        pytest.skip(f"missing {REFERENCE_SDF}")
    calls = _record_cnn_calls(monkeypatch)
    runner = _runner(tmp_path, monkeypatch, kwargs, attrs)
    assert runner.cnn_rescore and runner._constraint_docking()
    assert runner._engine_order_keepers() == [] and runner._engine_num_pose() == kwargs['num_pose']
    with caplog.at_level(logging.INFO, logger='unidock_processing.core.runner'):
        runner._run_cnn_rescore()
    assert calls == []
    assert "CNN rescoring skipped: constraint / template / covalent runs keep the engine order" in caplog.text
    assert "keeps the engine order" not in caplog.text


def test_registry_without_entry_points(monkeypatch):
    monkeypatch.setattr(registry, 'entry_points', lambda group: [])
    monkeypatch.setattr(registry, 'BUILTIN_PLUGINS', {})
    monkeypatch.setattr(registry, 'KNOWN_EXTERNAL_PLUGINS', {})
    assert registry.plugin_specs() == {} and registry.discover_plugins() == ([], {})
    assert registry.load_plugins() == []
    with pytest.raises(KeyError):
        registry.get_plugin('fake')


def test_registry_loads_entry_points_in_name_order(monkeypatch):
    class _Other(Plugin):
        name = 'other'

    eps = [types.SimpleNamespace(name='zz', value='tests.fake:Other'),
           types.SimpleNamespace(name='aa', value='tests.fake:Fake')]
    classes = {'tests.fake:Other': _Other, 'tests.fake:Fake': _FakePlugin}
    monkeypatch.setattr(registry, 'entry_points', lambda group: eps)
    monkeypatch.setattr(registry, 'BUILTIN_PLUGINS', {})
    monkeypatch.setattr(registry, 'KNOWN_EXTERNAL_PLUGINS', {})
    monkeypatch.setattr(registry, 'load_plugin_class', classes.__getitem__)
    assert registry.plugin_specs() == {'aa': 'tests.fake:Fake', 'zz': 'tests.fake:Other'}
    assert registry.plugin_classes() == [_FakePlugin, _Other]
    plugins = registry.load_plugins()
    assert [p.name for p in plugins] == ['fake', 'other']
    assert registry.get_plugin('other').name == 'other'


def test_registry_builtin_list_covers_a_stale_entry_point_group(monkeypatch, caplog):
    """An editable install made before the entry-point group existed lists nothing: the built-in list still
    finds the shipped plugins, an entry point of the same name wins, and a plugin whose module is not installed
    is reported missing (one warning) instead of dropped silently."""
    class _Other(Plugin):
        name = 'other'

    monkeypatch.setattr(registry, 'entry_points', lambda group: [])
    monkeypatch.setattr(registry, 'BUILTIN_PLUGINS', {'other': 'tests.fake:Other', 'gone': 'tests.gone:Gone'})
    monkeypatch.setattr(registry, 'KNOWN_EXTERNAL_PLUGINS', {})

    def load(spec):
        if spec == 'tests.fake:Other':
            return _Other
        raise ModuleNotFoundError(spec)

    monkeypatch.setattr(registry, 'load_plugin_class', load)
    with caplog.at_level(logging.WARNING, logger='unidock_processing.plugins.registry'):
        classes, missing = registry.discover_plugins()
    assert classes == [_Other] and missing == {'gone': 'tests.gone:Gone'}
    assert "plugin 'gone'" in caplog.text and 'not installed' in caplog.text
    assert registry.plugin_classes() == [_Other] and registry.missing_plugins() == missing

    eps = [types.SimpleNamespace(name='other', value='tests.fake:Other2')]
    monkeypatch.setattr(registry, 'entry_points', lambda group: eps)
    assert registry.plugin_specs() == {'gone': 'tests.gone:Gone', 'other': 'tests.fake:Other2'}


def test_registry_load_plugin_class_resolves_module_and_attribute():
    assert registry.load_plugin_class('unidock_processing.plugins.base:Plugin') is Plugin
    with pytest.raises(ImportError):
        registry.load_plugin_class('unidock_processing.plugins.no_such_module:Plugin')


def test_registry_known_external_list(monkeypatch, caplog):
    """KNOWN_EXTERNAL_PLUGINS (the plugins other distributions ship: the bias plugin is ud2align's) is consulted like
    the built-in list: a checkout without entry-point metadata finds it, an entry point of the same name wins, and
    one whose module is not installed is reported missing under its spec."""
    assert registry.BUILTIN_PLUGINS == {'covalent': 'unidock_processing.plugins.covalent.plugin:CovalentPlugin',
                                        'template': 'unidock_processing.plugins.template.plugin:TemplatePlugin'}
    assert registry.KNOWN_EXTERNAL_PLUGINS == {'bias': 'ud2align.plugin:BiasPlugin'}

    class _External(Plugin):
        name = 'external'

    monkeypatch.setattr(registry, 'entry_points', lambda group: [])
    monkeypatch.setattr(registry, 'BUILTIN_PLUGINS', {'other': 'tests.fake:Other'})
    monkeypatch.setattr(registry, 'KNOWN_EXTERNAL_PLUGINS',
                        {'external': 'tests.ext:External', 'gone': 'tests.gone:Gone'})

    def load(spec):
        if spec == 'tests.ext:External':
            return _External
        if spec == 'tests.fake:Other':
            return _FakePlugin
        raise ModuleNotFoundError(spec)

    monkeypatch.setattr(registry, 'load_plugin_class', load)
    assert registry.plugin_specs() == {'external': 'tests.ext:External', 'gone': 'tests.gone:Gone',
                                       'other': 'tests.fake:Other'}
    with caplog.at_level(logging.WARNING, logger='unidock_processing.plugins.registry'):
        classes, missing = registry.discover_plugins()
    assert classes == [_External, _FakePlugin] and missing == {'gone': 'tests.gone:Gone'}
    assert "plugin 'gone' (tests.gone:Gone) is not installed" in caplog.text

    eps = [types.SimpleNamespace(name='external', value='tests.ext:External2')]
    monkeypatch.setattr(registry, 'entry_points', lambda group: eps)
    assert registry.plugin_specs()['external'] == 'tests.ext:External2'


def _without_ud2align(monkeypatch):
    """The registry state of an install without ud2align: its modules unimportable, no entry-point metadata."""
    monkeypatch.setitem(sys.modules, 'ud2align', None)
    monkeypatch.setitem(sys.modules, 'ud2align.plugin', None)
    monkeypatch.setattr(registry, 'entry_points', lambda group: [])


def test_bias_block_without_ud2align_is_a_clear_error(monkeypatch, caplog):
    """Without ud2align the registry reports the bias plugin missing (template and covalent, when installed, load),
    and a 'bias:' block that sets anything is the clear error naming ud2align's module; an all-null block is dropped
    with a warning."""
    from unidock_processing.io import yaml as yaml_mod
    _without_ud2align(monkeypatch)
    classes, missing = registry.discover_plugins()
    assert missing == {'bias': 'ud2align.plugin:BiasPlugin'}
    installed = [name for name in ('covalent', 'template') if HAS_PLUGIN[name]]
    assert [plugin_cls.name for plugin_cls in classes] == installed
    mapping = yaml_mod.build_config_mapping(classes)
    assert 'bias' not in [entry[0] for entry in mapping]
    UnidockConfig = yaml_mod.build_unidock_model(mapping, missing, classes)
    with pytest.raises(ImportError, match=r"a 'bias:' block that sets \['template_sdf'\] needs the bias plugin "
                                          r"\(ud2align\.plugin:BiasPlugin\), which is not installed"):
        UnidockConfig.from_dict({'bias': {'template_sdf': 'ref.sdf', 'aligner': None}})
    with caplog.at_level(logging.WARNING, logger='unidock_processing.io.yaml'):
        assert 'plugins' not in UnidockConfig.from_dict({'bias': {'template_sdf': None}}).to_protocol_kwargs()
    assert "'bias:' block ignored: the bias plugin (ud2align.plugin:BiasPlugin) is not installed" in caplog.text


def test_bias_cli_options_without_ud2align_are_a_clear_error(monkeypatch, capsys):
    """Without ud2align the docking CLI has no --bias_* option, and passing one is the clear error naming ud2align's
    module (CLICommand.check_unrecognized_arguments, called by main.py before argparse's 'unrecognized arguments'),
    while any other unknown option keeps argparse's error."""
    from unidock_processing import main as main_mod
    from unidock_processing.cli.docking import CLICommand
    _without_ud2align(monkeypatch)
    parser = __import__('argparse').ArgumentParser()
    CLICommand.add_arguments(parser)
    assert not [action for action in parser._actions if any('bias' in option for option in action.option_strings)]
    CLICommand.check_unrecognized_arguments(['--other', 'x'])                  # not a plugin's option: no error here
    for arguments in (['--bias_template_sdf', 'ref.sdf'], ['--bias_k=0.2'], ['--no_bias_translate']):
        with pytest.raises(ImportError, match=r"the option\(s\) \['--(no_)?bias_[a-z_]+'\] needs the bias plugin "
                                              r"\(ud2align\.plugin:BiasPlugin\), which is not installed"):
            CLICommand.check_unrecognized_arguments(arguments)
    monkeypatch.setattr(sys, 'argv', ['unidock2', 'docking', '-r', 'rec.pdb', '--bias_template_sdf', 'ref.sdf'])
    with pytest.raises(ImportError, match=r"ud2align\.plugin:BiasPlugin"):
        main_mod.CLIDriver().run()
    monkeypatch.setattr(sys, 'argv', ['unidock2', 'docking', '-r', 'rec.pdb', '--other_option', 'x'])
    with pytest.raises(SystemExit) as exit_info:
        main_mod.CLIDriver().run()
    assert exit_info.value.code == 2
    assert 'unrecognized arguments: --other_option x' in capsys.readouterr().err


class _RecordingReceptorBuilder:
    """Stands in for UnidockReceptorTopologyBuilder: records the constructor arguments and the method calls."""
    instances = []

    def __init__(self, receptor_file_name, prepared_hydrogen=False, covalent_residue_atom_info_list=None,
                 working_dir_name='.', require_ff_params=False):
        self.kwargs = dict(prepared_hydrogen=prepared_hydrogen,
                           covalent_residue_atom_info_list=covalent_residue_atom_info_list,
                           require_ff_params=require_ff_params)
        self.receptor_parameterized_dms_file_name = os.path.join(working_dir_name, 'receptor_parameterized.dms')
        self.calls = []
        _RecordingReceptorBuilder.instances.append(self)

    def generate_receptor_topology(self):
        self.calls.append('generate_receptor_topology')

    def analyze_receptor_topology_vina_only(self):
        self.calls.append('analyze_receptor_topology_vina_only')

    def get_summary_receptor_info_dict(self):
        self.calls.append('get_summary_receptor_info_dict')


class _ReceptorPlugin(Plugin):
    """Records what the receptor builder had run when annotate_receptor was called, and edits it."""
    name = 'receptor'

    def __init__(self):
        self.seen = None

    def annotate_receptor(self, builder):
        self.seen = list(builder.calls)
        builder.covalent_residue_atom_info_list = 'set by the plugin'


def test_annotate_receptor_runs_on_the_constructed_builder_before_the_preparation(tmp_path, monkeypatch):
    """annotate_receptor sees the builder constructed and nothing run on it (the plan: 'may edit the receptor
    builder before prep'), for a receptor that needs the preparation and for an enriched DMS receptor."""
    from unidock_processing.core import runner as runner_mod
    monkeypatch.setattr(runner_mod, 'UnidockReceptorTopologyBuilder', _RecordingReceptorBuilder)
    monkeypatch.setattr(runner_mod, 'dms_has_nonbonded_params', lambda path: False)
    monkeypatch.setattr(runner_mod, 'dms_has_ud2_atom_props', lambda path: True)
    _RecordingReceptorBuilder.instances.clear()

    runner = _runner(tmp_path, monkeypatch, dict(target_center=CENTER_A, ff_minimize_steps=100), {})
    plugin = _ReceptorPlugin()
    runner.plugins = [plugin]
    builder = runner._prepare_receptor()
    assert plugin.seen == []
    assert builder.calls == ['generate_receptor_topology', 'analyze_receptor_topology_vina_only',
                             'get_summary_receptor_info_dict']
    assert builder.kwargs == dict(prepared_hydrogen=False, covalent_residue_atom_info_list=None,
                                  require_ff_params=True)                  # the constructor snapshot
    assert builder.covalent_residue_atom_info_list == 'set by the plugin'  # the attribute the preparation reads
    assert runner.enriched_receptor_dms is None

    enriched = _runner(tmp_path, monkeypatch, dict(target_center=CENTER_A), {})
    enriched.receptor_file_name = enriched.protein_dms = '/work/run1/receptor_enriched.dms'
    plugin = _ReceptorPlugin()
    enriched.plugins = [plugin]
    builder = enriched._prepare_receptor()
    assert plugin.seen == [] and builder.calls == ['generate_receptor_topology']
    assert builder.kwargs['require_ff_params'] is False
    assert builder.covalent_residue_atom_info_list == 'set by the plugin'
    assert enriched.enriched_receptor_dms == '/work/run1/receptor_enriched.dms'


def test_place_core_returns_the_molecule_the_builder_types(tmp_path):
    """A per-molecule hook may hand the builder a new molecule (the covalent ligand without its dummy atoms): the
    builder types what place_core returns; the base hook returns the molecule it was given."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from unidock_processing.plugins.base import LigandMolHook
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        build_molecular_graph, load_torsion_library,
    )

    def embedded(smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=7)
        return mol

    ethanol, propanol = embedded('CCO'), embedded('CCCO')
    assert LigandMolHook().place_core(ethanol, str(tmp_path)) is ethanol

    class _Rebuild(LigandMolHook):
        def place_core(self, mol, working_dir_name):
            return propanol

    kwargs = dict(working_dir_name=str(tmp_path), skip_gaff2=True)
    torsion_library = load_torsion_library()
    atoms, torsions, root_atoms, fragments = build_molecular_graph(ethanol, torsion_library, hooks=[_Rebuild()], **kwargs)
    assert len(atoms) == propanol.GetNumAtoms() != ethanol.GetNumAtoms()
    assert (atoms, torsions, root_atoms, fragments) == build_molecular_graph(embedded('CCCO'), torsion_library, **kwargs)


def test_plugin_base_hooks_are_no_ops():
    plugin = Plugin()
    assert plugin.yaml_block() == {} and plugin.search_policy() == {} and Plugin.from_config(None) is None
    assert plugin.keeps_engine_order() is False
    assert plugin.configure({}) is None and plugin.annotate_receptor(None) is None
    assert plugin.prepare_ligand(None, {}) is None and plugin.annotate_ligand_dms('x.dms', {}) is None
    assert plugin.post_process('x.sdf') is None and Plugin.cli_options(None) is None


def test_old_runner_module_re_exports_the_moved_names():
    from unidock_processing.unidocktools import unidock_protocol_runner as old
    from unidock_processing.core import dms, runner
    assert old.UnidockProtocolRunner is runner.UnidockProtocolRunner
    assert old.prepare_dms_lightweight is dms.prepare_dms_lightweight
    for name in old.__all__:
        if name.startswith('_'):
            assert getattr(old, name) is getattr(dms, name[1:]), name


def test_core_runner_imports_without_torch():
    code = ("import sys; import unidock_processing.core.runner; "
            "assert 'torch' not in sys.modules, 'torch imported'; print('ok')")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == 'ok', result.stderr[-2000:]
