"""The ``covalent:`` block of the Uni-Dock2 configuration YAML (``CovalentPlugin.config_model``): the two keys the
``Preprocessing:`` block carried, which ``io.yaml`` still reads there for one release, with a deprecation warning.
"""
from typing import Any, List, Optional

from pydantic import BaseModel


class CovalentConfig(BaseModel):
    covalent_ligand: bool = False
    covalent_residue_atom_info_list: Optional[List[Any]] = None
