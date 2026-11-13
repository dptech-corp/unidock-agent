"""Plugin discovery: the one list the runner, the CLI and the YAML reader iterate.

A plugin is known by name from the ``unidock2.plugins`` entry-point group (one entry per plugin class, "module:Class")
or from ``BUILTIN_PLUGINS``, the plugins this package ships, for an install whose entry-point metadata predates the
group (an editable install made before it lists nothing), or from ``KNOWN_EXTERNAL_PLUGINS``, the plugins other
distributions ship (the bias plugin: the ud2align package), consulted the same way so a checkout on PYTHONPATH without
entry-point metadata still finds them; an entry point of the same name wins over both lists. A known plugin whose
module does not import is "missing": skipped by the loaders, reported by ``missing_plugins`` so a configuration that
needs it gets a clear error naming its module.
"""
from importlib import import_module
from importlib.metadata import entry_points
import logging
from typing import Dict, List, Tuple, Type

from unidock_processing.plugins.base import Plugin

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "unidock2.plugins"

BUILTIN_PLUGINS: Dict[str, str] = {
    'covalent': 'unidock_processing.plugins.covalent.plugin:CovalentPlugin',
    'template': 'unidock_processing.plugins.template.plugin:TemplatePlugin',
}

# Plugins of other distributions (installed separately), declared by their own entry points
KNOWN_EXTERNAL_PLUGINS: Dict[str, str] = {
    'bias': 'ud2align.plugin:BiasPlugin',
}


def plugin_specs() -> Dict[str, str]:
    """Every known plugin, name -> "module:Class", in name order: the entry-point group's entries, plus the
    built-in and the known external list for the names the group does not list."""
    specs = {entry_point.name: entry_point.value for entry_point in entry_points(group=ENTRY_POINT_GROUP)}
    for name, spec in {**BUILTIN_PLUGINS, **KNOWN_EXTERNAL_PLUGINS}.items():
        specs.setdefault(name, spec)
    return dict(sorted(specs.items()))


def load_plugin_class(spec: str) -> Type[Plugin]:
    """The class a "module:Class" spec names (ImportError when the module is not installed)."""
    module_name, _, class_name = spec.partition(':')
    return getattr(import_module(module_name), class_name)


def discover_plugins() -> Tuple[List[Type[Plugin]], Dict[str, str]]:
    """The installed plugin classes in name order, and the known plugins whose module does not import
    (name -> spec), each logged once per call."""
    classes: List[Type[Plugin]] = []
    missing: Dict[str, str] = {}
    for name, spec in plugin_specs().items():
        try:
            classes.append(load_plugin_class(spec))
        except ImportError as error:
            logger.warning("plugin %r (%s) is not installed: %s", name, spec, error)
            missing[name] = spec
    return classes, missing


def plugin_classes() -> List[Type[Plugin]]:
    """The installed plugin classes, in name order."""
    return discover_plugins()[0]


def missing_plugins() -> Dict[str, str]:
    """The known plugins whose module does not import, name -> spec."""
    return discover_plugins()[1]


def load_plugins() -> List[Plugin]:
    """One instance of every installed plugin, in name order."""
    return [plugin_cls() for plugin_cls in plugin_classes()]


def get_plugin(name: str) -> Plugin:
    for plugin in load_plugins():
        if plugin.name == name:
            return plugin
    raise KeyError(f"no installed plugin named {name!r} (entry-point group {ENTRY_POINT_GROUP!r}, the built-in or "
                   f"the known external list)")
