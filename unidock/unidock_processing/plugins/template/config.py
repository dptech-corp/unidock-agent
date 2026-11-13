"""The ``template:`` block of the Uni-Dock2 configuration YAML (``TemplatePlugin.config_model``): the three keys the
``Preprocessing:`` block carried, which ``io.yaml`` still reads there for one release, with a deprecation warning.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TemplateConfig(BaseModel):
    template_docking: bool = False
    reference_sdf_file_name: Optional[str] = None
    core_atom_mapping_dict_list: Optional[List[Dict[str, Any]]] = None
