import logging
import os
import json
from copy import deepcopy

import numpy as np

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D

logger = logging.getLogger(__name__)


class UnidockLigandPoseWriter(object):
    """Legacy writer of the engine's JSON poses (unidock_benchmark_runner); the runner reads the engine's SDF
    output directly. The covalent branch rebuilds the ligand through the covalent plugin."""
    def __init__(
        self,
        ligand_mol_list,
        unidock2_pose_json_file_name_list,
        covalent_ligand=False,
        docking_pose_sdf_file_name='unidock2_pose.sdf',
    ):
        self.ligand_mol_list = ligand_mol_list if ligand_mol_list else []
        self.unidock2_pose_json_file_name_list = unidock2_pose_json_file_name_list
        self.num_unidock2_batches = len(self.unidock2_pose_json_file_name_list)
        self.covalent_ligand = covalent_ligand
        self.docking_pose_sdf_file_name = os.path.abspath(docking_pose_sdf_file_name)

        self.unidock2_pose_dict = {}
        for batch_idx in range(self.num_unidock2_batches):
            unidock2_pose_json_file_name = self.unidock2_pose_json_file_name_list[
                batch_idx
            ]
            with open(unidock2_pose_json_file_name, 'r') as unidockd2_json_file:
                batch_unidock2_pose_dict = json.load(unidockd2_json_file)

            self.unidock2_pose_dict.update(batch_unidock2_pose_dict)

    def generate_docking_pose_sdf(self):
        self._write_poses_from_mol_list()

    def _set_pose_energy_properties(self, mol, pose_info):
        """Set Vina/FF energy and decomposition properties on an RDKit mol."""
        vina_scoring_list = pose_info['energy']
        mol.SetProp('vina_binding_free_energy', str(vina_scoring_list[0]))
        mol.SetProp('vina_intra_inter', str(vina_scoring_list[1]))
        mol.SetProp('vina_intra', str(vina_scoring_list[2]))
        mol.SetProp('vina_inter', str(vina_scoring_list[3]))
        mol.SetProp('vina_box_penalty', str(vina_scoring_list[4]))
        mol.SetProp('vina_torsion_number_energy', str(vina_scoring_list[5]))

        if 'ff_energy' in pose_info:
            ff_e = pose_info['ff_energy']
            mol.SetProp('ff_total_energy', str(ff_e[0]))
            mol.SetProp('ff_interaction_energy', str(ff_e[1]))
            mol.SetProp('ff_intra_ligand_energy', str(ff_e[2]))

        if 'atom_decomp' in pose_info:
            decomp = pose_info['atom_decomp']
            mol.SetProp('atom_inter_energy', ' '.join(f'{r[0]:.4f}' for r in decomp))
            mol.SetProp('atom_gauss1', ' '.join(f'{r[1]:.4f}' for r in decomp))
            mol.SetProp('atom_gauss2', ' '.join(f'{r[2]:.4f}' for r in decomp))
            mol.SetProp('atom_repulsion', ' '.join(f'{r[3]:.4f}' for r in decomp))
            mol.SetProp('atom_hydrophobic', ' '.join(f'{r[4]:.4f}' for r in decomp))
            mol.SetProp('atom_hbond', ' '.join(f'{r[5]:.4f}' for r in decomp))

        if 'ff_atom_decomp' in pose_info:
            ffd = pose_info['ff_atom_decomp']
            mol.SetProp('atom_ff_total', ' '.join(f'{r[0]:.4f}' for r in ffd))
            mol.SetProp('atom_ff_inter_lj', ' '.join(f'{r[1]:.4f}' for r in ffd))
            mol.SetProp('atom_ff_inter_coulomb', ' '.join(f'{r[2]:.4f}' for r in ffd))
            mol.SetProp('atom_ff_intra_lj', ' '.join(f'{r[3]:.4f}' for r in ffd))
            mol.SetProp('atom_ff_intra_coulomb', ' '.join(f'{r[4]:.4f}' for r in ffd))
            mol.SetProp('atom_ff_torsion', ' '.join(f'{r[5]:.4f}' for r in ffd))

    def _write_poses_from_mol_list(self):
        """Original path: use RDKit mol objects to produce SDF."""
        writer = Chem.SDWriter(self.docking_pose_sdf_file_name)

        for ligand_idx in range(len(self.ligand_mol_list)):
            ligand_mol = self.ligand_mol_list[ligand_idx]

            if self.covalent_ligand:
                from unidock_processing.plugins.covalent.ligand import prepare_covalent_ligand_mol
                ligand_mol, covalent_anchor_atom_info, covalent_atom_info_list = (
                    prepare_covalent_ligand_mol(ligand_mol)
                )

            ligand_internal_name = ligand_mol.GetProp('ud2_molecule_name')
            num_ligand_atoms = ligand_mol.GetNumAtoms()
            ligand_unidock2_pose_list = self.unidock2_pose_dict[ligand_internal_name]

            original_name = (
                ligand_mol.GetProp('_Name')
                if ligand_mol.HasProp('_Name')
                else ligand_internal_name
            )

            for pose_idx, pose_info in enumerate(ligand_unidock2_pose_list):
                ligand_mol_ud2_pose = deepcopy(ligand_mol)
                ligand_mol_ud2_pose.SetProp('ud2_molecule_name', original_name)

                self._set_pose_energy_properties(ligand_mol_ud2_pose, pose_info)

                ud2_pose_coord_list = pose_info['coords']
                num_coords_atoms = len(ud2_pose_coord_list) / 3

                if num_coords_atoms != num_ligand_atoms:
                    raise ValueError(
                        f'Atom count mismatch for {ligand_internal_name} pose {pose_idx}: '
                        f'expected {num_ligand_atoms}, got {int(num_coords_atoms)}'
                    )

                ud2_pose_coord_array = np.reshape(
                    ud2_pose_coord_list, (num_ligand_atoms, 3)
                )

                conf = ligand_mol_ud2_pose.GetConformer()
                for atom_idx in range(num_ligand_atoms):
                    conf.SetAtomPosition(
                        atom_idx,
                        Point3D(
                            float(ud2_pose_coord_array[atom_idx, 0]),
                            float(ud2_pose_coord_array[atom_idx, 1]),
                            float(ud2_pose_coord_array[atom_idx, 2]),
                        ),
                    )

                writer.write(ligand_mol_ud2_pose)
                writer.flush()

        writer.close()

