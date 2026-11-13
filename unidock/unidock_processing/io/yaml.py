"""The configuration YAML: the core blocks, plus one block per installed plugin that declares a ``config_model``
(the block is named after the plugin: ``template:``). The plugins come from ``plugins.registry``; a block of a known
plugin that is not installed is a clear error when it sets anything, a warning when it is all defaults. A plugin
block becomes the configured instance in the runner's ``plugins`` argument when the plugin builds one from it
(``Plugin.from_config``); otherwise its fields are flattened into runner keyword arguments, for a plugin whose block
is the compatibility keyword arguments (``Plugin.flatten_config``); a block the plugin declines and does not flatten
(the bias block with no template) sets nothing.
"""
from typing import Dict, Any, Optional, List, Tuple, Type
import logging
from pydantic import BaseModel, Field, ValidationError, create_model, field_validator
import yaml

from unidock_processing.plugins.base import Plugin
from unidock_processing.plugins.registry import discover_plugins

logger = logging.getLogger(__name__)

class RequiredConfig(BaseModel):
    receptor: Optional[str] = None
    ligand: Optional[str] = None
    ligand_batch: Optional[str] = None
    center: Tuple[float, float, float] = Field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )

    @field_validator('center')
    def validate_center(cls, v):
        if len(v) != 3:
            raise ValueError('Center requires 3 elements')
        return v

class AdvancedConfig(BaseModel):
    exhaustiveness: int = 512
    randomize: bool = True
    mc_steps: int = 40
    opt_steps: int = -1
    refine_steps: int = 5
    num_pose: int = 10
    rescore_pool: int = 10
    rmsd_limit: float = 1.0
    energy_range: float = 5.0
    seed: int = 1234567
    use_tor_lib: bool = False

class HardwareConfig(BaseModel):
    n_cpu: Optional[int] = None
    gpu_device_id: int = 0

class SettingsConfig(BaseModel):
    box_size: Tuple[float, float, float] = Field(
        default_factory=lambda: (30.0, 30.0, 30.0)
    )
    task: str = 'screen'
    search_mode: str = 'balance'
    cnn_rescore: bool = False

    @field_validator('box_size')
    def validate_box_size(cls, v):
        if len(v) != 3:
            raise ValueError('Box Size requires 3 elements')
        return v

class PreprocessingConfig(BaseModel):
    # template_docking, reference_sdf_file_name and core_atom_mapping_dict_list belong to the 'template:' block and
    # covalent_ligand, covalent_residue_atom_info_list to the 'covalent:' block; they are still read here for one
    # release (DEPRECATED_PREPROCESSING_KEYS, moved by from_dict with a warning)
    template_docking: bool = False
    reference_sdf_file_name: Optional[str] = None
    core_atom_mapping_dict_list: Optional[List[Dict[str, Any]]] = None
    covalent_ligand: bool = False
    covalent_residue_atom_info_list: Optional[List[Any]] = None
    preserve_receptor_hydrogen: bool = False
    temp_dir_name: str = '/tmp'
    output_receptor_dms_file_name: str = 'receptor_parameterized.dms'
    output_docking_pose_sdf_file_name: str = 'unidock2_pose.sdf'

# Preprocessing keys read for one release that now belong to a plugin's block: key -> plugin name. With the plugin
# installed the key moves into its block (deprecation warning) and leaves the Preprocessing field list; without it
# the key stays in Preprocessing, and a value that is not the default is a clear error naming the plugin.
DEPRECATED_PREPROCESSING_KEYS: Dict[str, str] = {
    'template_docking': 'template',
    'reference_sdf_file_name': 'template',
    'core_atom_mapping_dict_list': 'template',
    'covalent_ligand': 'covalent',
    'covalent_residue_atom_info_list': 'covalent',
}

ConfigMapping = List[Tuple[str, str, Type[BaseModel], List[str]]]       # (YAML key, attribute, model, fields)


def build_config_mapping(plugin_classes: List[Type[Plugin]]) -> ConfigMapping:
    """The YAML blocks: the core ones, then the block of every plugin class with a ``config_model``."""
    installed = {plugin_cls.name for plugin_cls in plugin_classes}
    mapping: ConfigMapping = [
        ('Required', 'required', RequiredConfig,
         ['receptor', 'ligand', 'ligand_batch', 'center']),
        ('Advanced', 'advanced', AdvancedConfig, [
         'exhaustiveness', 'randomize', 'mc_steps', 'opt_steps',
         'refine_steps', 'num_pose', 'rescore_pool', 'rmsd_limit', 'energy_range',
         'seed', 'use_tor_lib']),
        ('Hardware', 'hardware', HardwareConfig, ['n_cpu', 'gpu_device_id']),
        ('Settings', 'settings', SettingsConfig, ['box_size', 'task', 'search_mode', 'cnn_rescore']),
        ('Preprocessing', 'preprocessing', PreprocessingConfig, [
         name for name in PreprocessingConfig.model_fields
         if DEPRECATED_PREPROCESSING_KEYS.get(name) not in installed]),
    ]
    for plugin_cls in plugin_classes:
        if plugin_cls.config_model is None:
            continue
        fields = list(plugin_cls.config_model.model_fields) if plugin_cls.flatten_config else []
        mapping.append((plugin_cls.name, plugin_cls.name, plugin_cls.config_model, fields))
    return mapping


def plugin_missing(name: str, spec: str, what: str) -> ImportError:
    return ImportError(f"{what} needs the {name} plugin ({spec}), which is not installed")


def check_blocks_without_plugin(data: Dict[str, Any], missing: Dict[str, str]) -> None:
    """A block of a known plugin that is not installed cannot be read: one that sets anything is a clear error,
    one with every value at its default (false / null) is dropped with a warning."""
    for name, spec in missing.items():
        if name not in data:
            continue
        block = data[name] or {}
        set_keys = sorted(key for key, value in block.items() if value)
        if set_keys:
            raise plugin_missing(name, spec, f"a '{name}:' block that sets {set_keys}")
        logger.warning("'%s:' block ignored: the %s plugin (%s) is not installed", name, name, spec)


def move_deprecated_preprocessing_keys(
    processed_data: Dict[str, BaseModel], mapping: ConfigMapping, missing: Dict[str, str],
) -> None:
    """The DEPRECATED_PREPROCESSING_KEYS set in Preprocessing go to their plugin's block, with a warning; a key set
    in both keeps the block's value. Without the plugin the key stays in Preprocessing, and a value that is not the
    default is a clear error."""
    preprocessing = processed_data.get('preprocessing')
    if preprocessing is None:
        return
    models = {attr_name: model_cls for _, attr_name, model_cls, _ in mapping}
    moved: Dict[str, Dict[str, Any]] = {}
    for key, plugin_name in DEPRECATED_PREPROCESSING_KEYS.items():
        if key not in preprocessing.model_fields_set:
            continue
        value = getattr(preprocessing, key)
        if plugin_name in missing:
            if value:
                raise plugin_missing(plugin_name, missing[plugin_name], f"'Preprocessing: {key}: {value!r}'")
            continue
        moved.setdefault(plugin_name, {})[key] = value
    for plugin_name, keys in moved.items():
        logger.warning("Preprocessing keys %s are deprecated: put them in the '%s:' block",
                       sorted(keys), plugin_name)
        block = processed_data.get(plugin_name, models[plugin_name]())
        for key, value in keys.items():
            if key not in block.model_fields_set:
                setattr(block, key, value)
        processed_data[plugin_name] = block


def build_unidock_model(mapping: ConfigMapping, missing: Dict[str, str], plugin_classes: List[Type[Plugin]] = ()):
    plugin_of_block = {plugin_cls.name: plugin_cls for plugin_cls in plugin_classes}
    field_definitions = {}
    for _, attr_name, model_cls, _ in mapping:
        field_definitions[attr_name] = (model_cls, Field(default_factory=model_cls))

    UnidockModel = create_model(
        'UnidockConfig',
        **field_definitions,
        __base__=BaseModel
    )

    class UnidockConfig(UnidockModel):
        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'UnidockConfig':
            """Create UnidockConfig from a dictionary."""
            check_blocks_without_plugin(data, missing)
            processed_data = {}
            for dict_key, attr_name, model_cls, _ in mapping:
                if dict_key in data:
                    processed_data[attr_name] = model_cls(**data[dict_key])
            move_deprecated_preprocessing_keys(processed_data, mapping, missing)

            return cls(**processed_data)

        def to_protocol_kwargs(self) -> Dict[str, Any]:
            """The runner keyword arguments: every block's fields flattened, except a plugin block its plugin
            builds an instance from (Plugin.from_config), which goes into the 'plugins' list instead, and a
            declined block of a plugin that does not flatten (an empty field list in the mapping)."""
            kwargs_dict = {}
            for _, attr_name, _, field_names in mapping:
                config = getattr(self, attr_name)
                plugin_cls = plugin_of_block.get(attr_name)
                plugin = plugin_cls.from_config(config) if plugin_cls is not None else None
                if plugin is not None:
                    kwargs_dict.setdefault('plugins', []).append(plugin)
                    continue
                for field_name in field_names:
                    kwargs_dict[field_name] = getattr(config, field_name)

            return kwargs_dict

    return UnidockConfig

PLUGIN_CLASSES, MISSING_PLUGINS = discover_plugins()
CONFIG_MAPPING = build_config_mapping(PLUGIN_CLASSES)
UnidockConfig = build_unidock_model(CONFIG_MAPPING, MISSING_PLUGINS, PLUGIN_CLASSES)

def read_unidock_params_from_yaml(yaml_file: str) -> UnidockConfig:
    """
    Read Unidock parameters from a yaml file and convert to UnidockConfig dataclass.

    Args:
        yaml_file: Path to the yaml file

    Returns:
        UnidockConfig object
    """
    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)

    try:
        return UnidockConfig.from_dict(params)
    except ValidationError as e:
        print(f'Configuration Error:\n{e.json(indent=2)}')
        raise
