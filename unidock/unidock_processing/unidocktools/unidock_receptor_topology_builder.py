import os
import logging
from shutil import copyfile, which
import msys

from unidock_processing.unidocktools.protein_topology import (
    prepare_receptor_residue_mol_list,
)
from unidock_processing.unidocktools.receptor_topology_preparation import (
    ReceptorTopologyPreparation,
)
from unidock_processing.unidocktools.ccd import CCD
from unidock_processing.unidocktools.receptor_het_recovery import (
    split_receptor_pdb,
    recover_hets,
)
from unidock_processing.unidocktools.receptor_tleap_parameters import (
    parameterize_receptor_with_tleap,
)
from unidock_processing.unidocktools.unidock_vina_atom_types import VINA_ATOM_TYPE_DICT

logger = logging.getLogger(__name__)

class UnidockReceptorTopologyBuilder(object):
    def __init__(
        self,
        receptor_file_name,
        prepared_hydrogen=False,
        covalent_residue_atom_info_list=None,
        working_dir_name=".",
        require_ff_params=False,
    ):
        self.receptor_file_name = receptor_file_name
        self.prepared_hydrogen = prepared_hydrogen
        # the covalent residue list (plugins.covalent): the constructor argument stays for one release, the
        # plugin's annotate_receptor sets the same attribute; analyze_receptor_topology prepares the bond from it
        self.covalent_residue_atom_info_list = covalent_residue_atom_info_list
        self.require_ff_params = require_ff_params

        self.working_dir_name = os.path.abspath(working_dir_name)
        self.receptor_structure_dms_file_name = os.path.join(
            self.working_dir_name, "receptor_structure.dms"
        )
        self.receptor_parameterized_dms_file_name = os.path.join(
            self.working_dir_name, "receptor_parameterized.dms"
        )

    def run_protein_preparation(self):
        has_fepfixer = which("fepfixer") is not None
        has_utop = which("utop") is not None
        has_tleap = which("tleap") is not None
        if has_fepfixer and not has_utop and not has_tleap and self.require_ff_params:
            raise RuntimeError(
                "Neither utop (unitop) nor tleap (AmberTools) is on PATH: the FF path (ff_minimize_steps > 0) "
                "needs the receptor force-field parameters that one of them writes after fepfixer. Install "
                "either, or set ff_minimize_steps to 0 for Vina-only docking."
            )

        if has_fepfixer:
            # fepfixer deletes every residue it has no template for and utop aborts on the ones it
            # keeps but cannot parametrize, so only the polymer (and water) goes through them; the
            # other HETATM residues are rebuilt from the CCD and appended afterwards.
            protein_pdb = os.path.join(self.working_dir_name, "receptor_protein_input.pdb")
            hets_pdb = os.path.join(self.working_dir_name, "receptor_hets_input.pdb")
            ccd = CCD()
            counts = split_receptor_pdb(self.receptor_file_name, protein_pdb, hets_pdb, ccd)
            logger.info(f"Receptor split for fepfixer: {counts['het_atoms']} het atoms set aside, "
                        f"{counts['converted_atoms']} modified-residue atoms kept in the chain as the parent residue, "
                        f"{counts['stub_atoms']} stub atoms dropped")
            if self.prepared_hydrogen:
                fep_fixer_command = (
                    f"fepfixer -i {protein_pdb} "
                    f"-o {self.receptor_structure_dms_file_name} "
                    "--custom-protonation-states"
                )
            else:
                fep_fixer_command = (
                    f"fepfixer -i {protein_pdb} "
                    f"-o {self.receptor_structure_dms_file_name}"
                )

            os.system(fep_fixer_command)
            if has_utop:
                unitop_command = (
                    f"utop prm -i {self.receptor_structure_dms_file_name} "
                    f"-o {self.receptor_parameterized_dms_file_name}"
                )
                os.system(unitop_command)
            elif has_tleap and self.require_ff_params:
                logger.info("utop not on PATH: the receptor force-field parameters come from tleap")
                parameterize_receptor_with_tleap(
                    self.receptor_structure_dms_file_name,
                    self.working_dir_name,
                    self.receptor_parameterized_dms_file_name,
                )
            else:
                # Vina-only docking needs elements, bonds and hydrogens, which fepfixer already writes.
                logger.warning("utop not on PATH: the receptor DMS has no force-field parameters (Vina-only scoring)")
                copyfile(self.receptor_structure_dms_file_name, self.receptor_parameterized_dms_file_name)
            recover_hets(hets_pdb, self.receptor_parameterized_dms_file_name, ccd)
        else:
            receptor_topology_preparation = ReceptorTopologyPreparation(
                self.receptor_file_name, self.working_dir_name
            )
            receptor_topology_preparation.run_preparation()

    def generate_receptor_topology(self):
        receptor_file_extension = self.receptor_file_name.split(".")[-1]
        if receptor_file_extension == "pdb":
            self.run_protein_preparation()
        elif receptor_file_extension == "dms":
            self.receptor_parameterized_dms_file_name = self.receptor_file_name
        else:
            raise ValueError(
                "Only PDB and DMS are supported for receptor file extensions!!"
            )

    def analyze_receptor_topology(self):
        receptor_msys_system = msys.LoadDMS(self.receptor_parameterized_dms_file_name)
        (
            self.protein_property_mol,
            self.protein_residue_property_mol_list,
            self.cofactor_residue_property_mol_list,
        ) = prepare_receptor_residue_mol_list(receptor_msys_system)

        self.covalent_residue_idx = None
        self.covalent_residue_atom_idx_list = []
        if self.covalent_residue_atom_info_list is not None:
            from unidock_processing.plugins.covalent.receptor import CovalentReceptorHook
            CovalentReceptorHook(self).prepare_covalent_bond_on_residue()

        num_protein_residues = len(self.protein_residue_property_mol_list)
        num_cofactor_residues = len(self.cofactor_residue_property_mol_list)

        num_protein_atoms = 0
        for protein_residue_idx in range(num_protein_residues):
            protein_residue_property_mol = self.protein_residue_property_mol_list[
                protein_residue_idx
            ]
            num_protein_atoms += protein_residue_property_mol.GetNumAtoms()

        num_cofactor_atoms = 0
        for cofactor_residue_idx in range(num_cofactor_residues):
            cofactor_residue_property_mol = self.cofactor_residue_property_mol_list[
                cofactor_residue_idx
            ]
            num_cofactor_atoms += cofactor_residue_property_mol.GetNumAtoms()

        num_receptor_atoms = num_protein_atoms + num_cofactor_atoms
        self.atom_info_nested_list = [None] * num_receptor_atoms
        atom_idx = 0

        for protein_residue_idx in range(num_protein_residues):
            protein_residue_property_mol = self.protein_residue_property_mol_list[
                protein_residue_idx
            ]
            for atom in protein_residue_property_mol.GetAtoms():
                vina_atom_type = atom.GetProp("vina_atom_type")

                atom_info_list = [None] * 5
                atom_info_list[0] = atom.GetDoubleProp("x")
                atom_info_list[1] = atom.GetDoubleProp("y")
                atom_info_list[2] = atom.GetDoubleProp("z")
                atom_info_list[3] = VINA_ATOM_TYPE_DICT[vina_atom_type]
                atom_info_list[4] = atom.GetDoubleProp("atom_charge")

                self.atom_info_nested_list[atom_idx] = atom_info_list
                atom_idx += 1

        for cofactor_residue_idx in range(num_cofactor_residues):
            cofactor_residue_property_mol = self.cofactor_residue_property_mol_list[
                cofactor_residue_idx
            ]
            for atom in cofactor_residue_property_mol.GetAtoms():
                vina_atom_type = atom.GetProp("vina_atom_type")

                atom_info_list = [None] * 5
                atom_info_list[0] = atom.GetDoubleProp("x")
                atom_info_list[1] = atom.GetDoubleProp("y")
                atom_info_list[2] = atom.GetDoubleProp("z")
                atom_info_list[3] = VINA_ATOM_TYPE_DICT[vina_atom_type]
                atom_info_list[4] = atom.GetDoubleProp("atom_charge")

                self.atom_info_nested_list[atom_idx] = atom_info_list
                atom_idx += 1

    def analyze_receptor_topology_vina_only(self):
        """Fallback receptor analysis that reads the DMS via msys directly,
        without calling msys.ConvertToRdkit (which can segfault on macOS).
        Assigns Vina (X-Score) atom types from connectivity, the same rules the
        full analysis and Uni-Dock-Benchmarks inputs follow:
          C   : C_P if bonded to any heteroatom (not C/H, so N, O, S ...), else C_H
          N   : with H -> N_D (N_DA for an sp3 amine with a free lone pair, e.g. neutral Lys);
                without H -> N_A if it has a lone pair (<= 2 heavy neighbours, e.g. His), else N_P
          O   : with H -> O_DA, else O_A
          others by element (S_P, P_P, halogens, metals).
        Validated against the 736 Uni-Dock-Benchmarks receptors (0 mismatches).
        The previous force-field-name mapping typed every amide N as an acceptor and
        every sp3 carbon as hydrophobic; it is kept only for DMS files without bonds.
        Sets ff_type=0 and charge=0.0 (sufficient for Vina-only scoring).
        """
        receptor_msys_system = msys.LoadDMS(self.receptor_parameterized_dms_file_name)
        num_receptor_atoms = receptor_msys_system.natoms
        has_bonds = receptor_msys_system.nbonds > 0
        if not has_bonds:
            logger.warning(
                "Receptor DMS has no bond table; Vina types fall back to force-field names, "
                "which mistypes amide nitrogens and polar carbons."
            )
            receptor_nb_table = receptor_msys_system.getTable("nonbonded")

        ELEMENT_TO_VINA = {
            1: "H", 6: "C_H", 7: "N_D", 8: "O_A", 9: "F_H", 14: "Si", 15: "P_P",
            16: "S_P", 17: "Cl_H", 35: "Br_H", 53: "I_H",
        }
        FF_TYPE_VINA_OVERRIDES = {  # legacy mapping, only used without bonds
            "C": "C_P", "CX": "C_H", "CT": "C_H", "C*": "C_P", "CA": "C_P", "CB": "C_P",
            "CC": "C_P", "CN": "C_P", "CR": "C_P", "CV": "C_P", "CW": "C_P", "CK": "C_P",
            "CQ": "C_P", "CM": "C_P", "C8": "C_H", "2C": "C_H", "3C": "C_H",
            "N": "N_D", "NA": "N_D", "NB": "N_A", "N2": "N_D", "N3": "N_D", "N*": "N_A",
            "O": "O_A", "O2": "O_A", "OH": "O_DA", "OS": "O_A", "OW": "O_DA",
            "S": "S_P", "SH": "S_P",
        }

        def vina_type_from_connectivity(atom):
            z = atom.atomic_number
            neighbours = atom.bonded_atoms
            if z == 1:
                return "H"
            if z == 6:
                return "C_P" if any(n.atomic_number not in (1, 6) for n in neighbours) else "C_H"
            if z == 7:
                has_h = any(n.atomic_number == 1 for n in neighbours)
                heavy = [n for n in neighbours if n.atomic_number > 1]
                if has_h:
                    sp3_amine = (len(neighbours) == 3
                                 and all(n.atomic_number == 6 and len(n.bonded_atoms) == 4 for n in heavy))
                    return "N_DA" if sp3_amine else "N_D"
                return "N_A" if len(heavy) <= 2 else "N_P"
            if z == 8:
                return "O_DA" if any(n.atomic_number == 1 for n in neighbours) else "O_A"
            return ELEMENT_TO_VINA.get(z, "Met_D")

        self.atom_info_nested_list = [None] * num_receptor_atoms
        for atom_idx in range(num_receptor_atoms):
            atom = receptor_msys_system.atom(atom_idx)
            pos = atom.pos
            if has_bonds:
                vina_type_name = vina_type_from_connectivity(atom)
            else:
                ff_type_name = receptor_nb_table.term(atom_idx)["type"]
                vina_type_name = FF_TYPE_VINA_OVERRIDES.get(
                    ff_type_name, ELEMENT_TO_VINA.get(atom.atomic_number, "Met_D"))
            vina_type_int = VINA_ATOM_TYPE_DICT.get(vina_type_name, 2)

            atom_info_list = [None] * 5
            atom_info_list[0] = float(pos[0])
            atom_info_list[1] = float(pos[1])
            atom_info_list[2] = float(pos[2])
            atom_info_list[3] = vina_type_int
            atom_info_list[4] = 0.0  # charge placeholder
            self.atom_info_nested_list[atom_idx] = atom_info_list

        logger.info(
            f"Receptor analyzed (Vina-only, connectivity typing): {num_receptor_atoms} atoms"
        )

    def get_summary_receptor_info_dict(self):
        self.receptor_info_summary_dict = {}
        self.receptor_info_summary_dict["receptor"] = self.atom_info_nested_list
