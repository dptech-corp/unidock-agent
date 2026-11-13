import os
import math
import dill as pickle
import logging
from multiprocess.pool import Pool

from rdkit import Chem
from rdkit.Chem import Descriptors, GetMolFrags, FragmentOnBonds
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from unidock_processing.ligand_topology.generic_rotatable_bond import (
    GenericRotatableBond,
)
from unidock_processing.ligand_topology import utils

from unidock_processing.torsion_library.torsion_library_driver import (
    TorsionLibraryDriver,
)
from unidock_processing.unidocktools.vina_atom_type import AtomType
from unidock_processing.unidocktools.unidock_vina_atom_types import VINA_ATOM_TYPE_DICT

logger = logging.getLogger(__name__)

_TORSION_LIBRARY_PKL = os.path.join(
    os.path.dirname(__file__), '..', 'torsion_library', 'data', 'torsion_library.pkl'
)
_torsion_library_dict = None


def load_torsion_library():
    """The torsion rule library, loaded once per process.

    The pickle holds 554 RDKit query molecules and takes 0.2-0.4 s to unpickle, while typing one ligand takes
    4 ms. Loading it per call made the topology step of a 16-ligand run 16 loads in 16 workers (0.6 s); the
    cache keeps one load per process, and a load before the pool forks is inherited by every worker.
    """
    global _torsion_library_dict
    if _torsion_library_dict is None:
        with open(_TORSION_LIBRARY_PKL, 'rb') as torsion_library_pkl_file:
            _torsion_library_dict = pickle.load(torsion_library_pkl_file)
    return _torsion_library_dict


def build_molecular_graph(
    mol,
    torsion_library_dict,
    covalent_ligand=False,
    template_docking=False,
    reference_mol=None,
    core_atom_mapping_dict=None,
    working_dir_name='.',
    has_ff=False,
    dms_path=None,
    atom_mapper_align=False,
    skip_gaff2=False,
    keep_h_rotors=None,
    hooks=None,
):
    """The atom, torsion, root and fragment lists of one ligand.

    *hooks* is a list of ``plugins.base.LigandMolHook``; each is called where the builder has a feature branch
    (``place_core``, which returns the molecule to type, ``rotatable_bonds``, ``filter_rotatable_bonds``,
    ``root_fragment_idx``), so with no hooks the plain path runs. The template keyword arguments (template_docking, reference_mol, core_atom_mapping_dict,
    atom_mapper_align) are accepted for one release and build the template hook here; atom_mapper_align without
    template_docking builds the hook that only swaps the rotatable-bond finder, as the flag did on its own.
    covalent_ligand is accepted for one release and builds the covalent hook here, first in the list (the covalent
    rebuild ran before every other branch), unless one is already given.
    """
    hooks = list(hooks) if hooks else []
    if template_docking:
        from unidock_processing.plugins.template.ligand import TemplateMolHook
        hooks.append(TemplateMolHook(reference_mol, core_atom_mapping_dict, atom_mapper_align))
    elif atom_mapper_align:
        from unidock_processing.plugins.template.ligand import AtomMapperAlignMolHook
        hooks.append(AtomMapperAlignMolHook())
    if covalent_ligand:
        from unidock_processing.plugins.covalent.ligand import CovalentMolHook
        if not any(isinstance(hook, CovalentMolHook) for hook in hooks):
            hooks.insert(0, CovalentMolHook())

    for hook in hooks:
        mol = hook.place_core(mol, working_dir_name)

    atom_typer = AtomType()
    atom_typer.assign_atom_types(mol)
    ComputeGasteigerCharges(mol)
    utils.assign_atom_properties(mol)

    rotatable_bond_info_list = None
    for hook in hooks:
        hook_rotatable_bond_info_list = hook.rotatable_bonds(mol)
        if hook_rotatable_bond_info_list is not None:
            rotatable_bond_info_list = hook_rotatable_bond_info_list
    if rotatable_bond_info_list is None:
        # H-only rotors (terminal -OH/-NH2/-NH3+) matter only when a force field
        # minimizes H positions; for Vina-only docking they are dead degrees of freedom.
        if keep_h_rotors is None:
            keep_h_rotors = not skip_gaff2
        rotatable_bond_finder = GenericRotatableBond()
        rotatable_bond_info_list = rotatable_bond_finder.identify_rotatable_bonds(
            mol, keep_h_rotors=keep_h_rotors
        )

    ## Construct nonbonded force atom pairs
    ##############################################################################
    atom_pair_12_13_nested_list, atom_pair_14_nested_list = (
        utils.calculate_nonbonded_atom_pairs(mol)
    )

    ## Construct gaff2 atom type and parameters
    ##############################################################################
    if has_ff and dms_path is not None:
        # FF params already in DMS; read GAFF2 types and charges directly
        from unidock_processing.core.dms import read_gaff2_types_from_dms
        atom_type_list, partial_charge_list = read_gaff2_types_from_dms(dms_path)
        atom_parameter_dict = {}
        torsion_parameter_nested_dict = {}
        logging.info("Skipping antechamber: FF params read from DMS %s", dms_path)
    elif skip_gaff2:
        # FF types not needed (Vina-only scoring, ff_minimize_steps=0).
        # Use placeholder GAFF2 types and RDKit Gasteiger charges to avoid
        # requiring AmberTools when only Vina scoring is used.
        num_total = mol.GetNumAtoms()
        atom_type_list = ['c3'] * num_total
        partial_charge_list = []
        for i in range(num_total):
            try:
                q = mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge')
                if not math.isfinite(q):
                    q = 0.0
            except (KeyError, RuntimeError):
                q = 0.0
            partial_charge_list.append(q)
        atom_parameter_dict = {}
        torsion_parameter_nested_dict = {}
        logging.info("Skipping antechamber: GAFF2 types not needed (Vina-only)")
    else:
        temp_ligand_sdf_file_name = os.path.join(working_dir_name, 'ligand.sdf')
        writer = Chem.SDWriter(temp_ligand_sdf_file_name)
        writer.write(mol)
        writer.close()

        (
            atom_type_list,
            partial_charge_list,
            atom_parameter_dict,
            torsion_parameter_nested_dict,
        ) = utils.record_gaff2_atom_types_and_parameters(
            temp_ligand_sdf_file_name, 'gas', working_dir_name
        )
    ###############################################################################

    ## Freeze bonds in core part to be unrotatable for template docking case
    ###############################################################################
    for hook in hooks:
        rotatable_bond_info_list = hook.filter_rotatable_bonds(mol, rotatable_bond_info_list)
    ###############################################################################

    bond_list = list(mol.GetBonds())
    rotatable_bond_idx_list = []
    for bond in bond_list:
        bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
        if (
            bond_info in rotatable_bond_info_list
            or bond_info_reversed in rotatable_bond_info_list
        ):
            rotatable_bond_idx_list.append(bond.GetIdx())

    if len(rotatable_bond_idx_list) != 0:
        splitted_mol = FragmentOnBonds(mol, rotatable_bond_idx_list, addDummies=False)
        splitted_mol_list = list(
            GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False)
        )
    else:
        splitted_mol_list = [mol]

    num_fragments = len(splitted_mol_list)

    ## Find fragment as the root node
    ##############################################################################
    root_fragment_idx = None
    for hook in hooks:
        root_fragment_idx = hook.root_fragment_idx(splitted_mol_list)
        if root_fragment_idx is not None:
            break
    if root_fragment_idx is None:
        root_fragment_idx = utils.root_finding_strategy(
            splitted_mol_list, rotatable_bond_info_list
        )
    ##############################################################################

    root_fragment = splitted_mol_list[root_fragment_idx]
    num_root_atoms = root_fragment.GetNumAtoms()

    root_atom_idx_list = []

    for root_atom_idx in range(num_root_atoms):
        root_atom = root_fragment.GetAtomWithIdx(root_atom_idx)
        root_atom_idx_list.append(root_atom.GetIntProp('internal_atom_idx'))

    ##############################################################################

    ##############################################################################
    ## Record fragment atom idx
    num_fragments = len(splitted_mol_list)
    fragment_atom_idx_nested_list = [None] * num_fragments
    for fragment_idx in range(num_fragments):
        fragment_mol = splitted_mol_list[fragment_idx]
        num_fragment_atoms = fragment_mol.GetNumAtoms()
        fragment_atom_idx_list = [None] * num_fragment_atoms

        for atom_idx in range(num_fragment_atoms):
            atom = fragment_mol.GetAtomWithIdx(atom_idx)
            fragment_atom_idx_list[atom_idx] = atom.GetIntProp('internal_atom_idx')

        fragment_atom_idx_nested_list[fragment_idx] = fragment_atom_idx_list

    ##############################################################################
    ##############################################################################
    ## Assign atom info
    num_total_atoms = mol.GetNumAtoms()
    atom_info_nested_list = [None] * num_total_atoms
    for atom_idx in range(num_total_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        vina_atom_type = atom.GetProp('vina_atom_type')

        atom_info_list = [None] * 7
        atom_info_list[0] = atom.GetDoubleProp('x')
        atom_info_list[1] = atom.GetDoubleProp('y')
        atom_info_list[2] = atom.GetDoubleProp('z')
        atom_info_list[3] = VINA_ATOM_TYPE_DICT[vina_atom_type]
        atom_info_list[4] = partial_charge_list[atom_idx]
        atom_info_list[5] = atom_pair_12_13_nested_list[atom_idx]
        atom_info_list[6] = atom_pair_14_nested_list[atom_idx]

        atom_info_nested_list[atom_idx] = atom_info_list

    ##############################################################################
    ##############################################################################
    ## Assign torsion FF parameters
    torsion_library_driver = TorsionLibraryDriver(
        mol, rotatable_bond_info_list, torsion_library_dict
    )
    torsion_library_driver.perform_torsion_matches()
    torsion_library_driver.identify_torsion_mobile_atoms(root_atom_idx_list[0])

    torsion_info_nested_list = [None] * torsion_library_driver.num_rotatable_bonds
    for torsion_idx in range(torsion_library_driver.num_rotatable_bonds):
        torsion_info_list = [None] * 5

        torsion_atom_idx_list = torsion_library_driver.torsion_atom_idx_nested_list[
            torsion_idx
        ]
        torsion_value_list = torsion_library_driver.original_torsion_value_list[
            torsion_idx
        ]
        torsion_range_list = (
            torsion_library_driver.enumerated_torsion_range_nested_list[torsion_idx]
        )
        torsion_mobile_atom_idx_list = (
            torsion_library_driver.mobile_atom_idx_nested_list[torsion_idx]
        )

        # Build torsion FF parameter list from frcmod data (when available)
        if torsion_parameter_nested_dict:
            torsion_atom_type_i = atom_type_list[torsion_atom_idx_list[0]]
            torsion_atom_type_j = atom_type_list[torsion_atom_idx_list[1]]
            torsion_atom_type_k = atom_type_list[torsion_atom_idx_list[2]]
            torsion_atom_type_l = atom_type_list[torsion_atom_idx_list[3]]

            torsion_type = (
                torsion_atom_type_i,
                torsion_atom_type_j,
                torsion_atom_type_k,
                torsion_atom_type_l,
            )
            if torsion_type in torsion_parameter_nested_dict:
                corrected_torsion_type = torsion_type
            else:
                corrected_torsion_type = tuple(reversed(torsion_type))

            torsion_parameter_dict_list = torsion_parameter_nested_dict[
                corrected_torsion_type
            ]
            num_torsion_parameters = len(torsion_parameter_dict_list)
            torsion_parameter_nested_list = [None] * num_torsion_parameters

            for torsion_parameter_idx in range(num_torsion_parameters):
                torsion_parameter_dict = torsion_parameter_dict_list[torsion_parameter_idx]
                torsion_parameter_list = [None] * 4
                torsion_parameter_list[0] = torsion_parameter_dict['barrier_factor']
                torsion_parameter_list[1] = torsion_parameter_dict['barrier_height']
                torsion_parameter_list[2] = torsion_parameter_dict['periodicity']
                torsion_parameter_list[3] = torsion_parameter_dict['phase']
                torsion_parameter_nested_list[torsion_parameter_idx] = (
                    torsion_parameter_list
                )
        else:
            # has_ff=True: param_gaff2 is not used by C++ engine; supply empty
            torsion_parameter_nested_list = []

        torsion_info_list[0] = torsion_atom_idx_list
        torsion_info_list[1] = torsion_value_list
        torsion_info_list[2] = torsion_range_list
        torsion_info_list[3] = torsion_mobile_atom_idx_list
        torsion_info_list[4] = torsion_parameter_nested_list

        torsion_info_nested_list[torsion_idx] = torsion_info_list
    ##############################################################################
    ##############################################################################

    return (
        atom_info_nested_list,
        torsion_info_nested_list,
        root_atom_idx_list,
        fragment_atom_idx_nested_list,
    )

def batch_topology_builder_process(
    ligand_sdf_file_name,
    covalent_ligand=False,
    working_dir_name='.',
    has_ff=False,
    dms_path_dict=None,
    skip_gaff2=False,
    hooks=None,
):
    """One batch of ligands in a worker process; *hooks* are the batch's ``LigandHook`` objects (opened here once,
    one per-molecule hook bound per ligand). *covalent_ligand* is accepted for one release (the builder sends the
    covalent hook instead) and forwarded to ``build_molecular_graph``."""
    torsion_library_dict = load_torsion_library()

    # iterate: list() builds the index, which loses records in some RDKit versions
    batch_ligand_mol_list = [
        mol for mol in Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)
    ]

    hooks = list(hooks) if hooks else []
    for hook in hooks:
        hook.open()

    num_batch_ligands = len(batch_ligand_mol_list)
    ligand_info_dict_list = [None] * num_batch_ligands

    for ligand_idx in range(num_batch_ligands):
        ligand_mol = batch_ligand_mol_list[ligand_idx]
        ligand_name = ligand_mol.GetProp('ud2_molecule_name')
        dms_path = dms_path_dict.get(ligand_name) if dms_path_dict else None
        (
            atom_info_nested_list,
            torsion_info_nested_list,
            root_atom_idx_list,
            fragment_atom_idx_nested_list,
        ) = build_molecular_graph(
            ligand_mol,
            torsion_library_dict,
            covalent_ligand,
            working_dir_name=working_dir_name,
            has_ff=has_ff,
            dms_path=dms_path,
            skip_gaff2=skip_gaff2,
            hooks=[hook.for_ligand(ligand_idx) for hook in hooks],
        )

        ligand_info_dict = {}
        ligand_info_dict['ligand_name'] = ligand_name
        ligand_info_dict['atom_info'] = atom_info_nested_list
        ligand_info_dict['torsion_info'] = torsion_info_nested_list
        ligand_info_dict['root_atom_idx'] = root_atom_idx_list
        ligand_info_dict['fragment_atom_idx'] = fragment_atom_idx_nested_list
        ligand_info_dict_list[ligand_idx] = ligand_info_dict

    return ligand_info_dict_list

class UnidockLigandTopologyBuilder(object):
    def __init__(
        self,
        ligand_sdf_file_name_list,
        covalent_ligand=False,
        template_docking=False,
        reference_sdf_file_name=None,
        core_atom_mapping_dict_list=None,
        n_cpu=None,
        working_dir_name='.',
        has_ff=False,
        ligand_dms_path_dict=None,
        atom_mapper_align=False,
        skip_gaff2=False,
        hooks=None,
    ):
        """*hooks* is a list of ``plugins.base.LigandHook`` (the runner's plugins set it); the template keyword
        arguments (template_docking, reference_sdf_file_name, core_atom_mapping_dict_list, atom_mapper_align) are
        accepted for one release and build the template hook here (atom_mapper_align without template_docking:
        the rotatable-bond hook only), and covalent_ligand builds the covalent hook (first in the list) unless one
        is already in *hooks*."""
        self.ligand_sdf_file_name_list = ligand_sdf_file_name_list
        self.ligand_mol_list = []
        self.mol_idx = 0

        for ligand_sdf_file_name in self.ligand_sdf_file_name_list:
            # iterate: list() builds the index, which loses records in some RDKit versions
            ligand_mol_list = [
                mol for mol in Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)
            ]

            num_source_molecules = len(ligand_mol_list)
            for source_mol_idx in range(num_source_molecules):
                ligand_mol = ligand_mol_list[source_mol_idx]
                if ligand_mol is None:
                    logging.error('Incorrect bond orders for molecule!')
                    continue
                if Descriptors.NumRadicalElectrons(ligand_mol) > 0:
                    logging.error('Molecule contains atoms with radicals!')
                    continue

                ligand_mol.SetProp('source_sdf_file_name', ligand_sdf_file_name)
                ligand_mol.SetIntProp('source_mol_idx', source_mol_idx)
                internal_molecule_name = f'MOL_{self.mol_idx}'
                ligand_mol.SetProp('ud2_molecule_name', internal_molecule_name)
                self.mol_idx += 1
                self.ligand_mol_list.append(ligand_mol)

        self.num_ligands = len(self.ligand_mol_list)
        if self.num_ligands == 0:
            raise ValueError(
                'Zero valid molecule after checking when preprocessing ligands!!'
            )

        self.covalent_ligand = covalent_ligand          # readable for one release; the covalent hook carries it

        self.hooks = list(hooks) if hooks else []
        if core_atom_mapping_dict_list is not None and len(core_atom_mapping_dict_list) != self.num_ligands:
            raise ValueError(
                'Number of user specified core atom mapping dicts does not match \
                    the number of input molecules!!'
            )
        if template_docking:
            from unidock_processing.plugins.template.ligand import TemplateLigandHook
            self.hooks.append(
                TemplateLigandHook(reference_sdf_file_name, core_atom_mapping_dict_list, atom_mapper_align)
            )
        elif atom_mapper_align:
            from unidock_processing.plugins.template.ligand import AtomMapperAlignHook
            self.hooks.append(AtomMapperAlignHook())
        if covalent_ligand:
            from unidock_processing.plugins.covalent.ligand import CovalentLigandHook
            if not any(isinstance(hook, CovalentLigandHook) for hook in self.hooks):
                self.hooks.insert(0, CovalentLigandHook())
        for hook in self.hooks:
            hook.bind(self.num_ligands)

        self.static_root = False
        if any(hook.wants_static_root() for hook in self.hooks):
            self.static_root = True

        self.has_ff = has_ff
        self.ligand_dms_path_dict = ligand_dms_path_dict or {}
        self.skip_gaff2 = skip_gaff2

        self.root_working_dir_name = os.path.abspath(working_dir_name)
        self.ligand_json_file_name = os.path.join(
            self.root_working_dir_name, 'ligands_unidock2.json'
        )

        max_n_cpu = os.cpu_count()
        if n_cpu is None:
            self.n_cpu = max_n_cpu
        elif n_cpu >= max_n_cpu:
            self.n_cpu = max_n_cpu
        else:
            self.n_cpu = n_cpu

    def generate_batch_ligand_topology(self):
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        raw_num_batches = self.n_cpu
        num_ligands_per_batch = int(self.num_ligands / raw_num_batches) + 1

        num_batches = 0
        batch_ligand_idx_tuple_list = []
        for batch_idx in range(raw_num_batches):
            begin_ligand_idx = num_ligands_per_batch * batch_idx
            end_ligand_idx = num_ligands_per_batch * (batch_idx + 1)
            num_batches += 1

            if end_ligand_idx >= self.num_ligands:
                end_ligand_idx = self.num_ligands
                batch_ligand_idx_tuple = (begin_ligand_idx, end_ligand_idx)
                batch_ligand_idx_tuple_list.append(batch_ligand_idx_tuple)
                break
            else:
                batch_ligand_idx_tuple = (begin_ligand_idx, end_ligand_idx)
                batch_ligand_idx_tuple_list.append(batch_ligand_idx_tuple)

        load_torsion_library()  # loaded here once, inherited by the forked workers
        batch_ligand_topology_builder_pool = Pool(processes=num_batches)
        batch_ligand_topology_builder_results_list = [None] * num_batches

        for batch_idx in range(num_batches):
            batch_ligand_idx_tuple = batch_ligand_idx_tuple_list[batch_idx]
            batch_ligand_mol_list = self.ligand_mol_list[
                batch_ligand_idx_tuple[0] : batch_ligand_idx_tuple[1]
            ]
            batch_hooks = [
                hook.for_batch(batch_ligand_idx_tuple[0], batch_ligand_idx_tuple[1])
                for hook in self.hooks
            ]
            working_dir_name = os.path.join(
                self.root_working_dir_name, f'ligand_batch_{batch_idx}'
            )
            os.makedirs(working_dir_name, exist_ok=True)

            batch_ligand_sdf_file_name = os.path.join(
                working_dir_name, 'ligand_batch.sdf'
            )
            batch_ligand_writer = Chem.SDWriter(batch_ligand_sdf_file_name)
            for ligand_mol in batch_ligand_mol_list:
                batch_ligand_writer.write(ligand_mol)
                batch_ligand_writer.flush()

            batch_ligand_writer.close()

            batch_dms_dict = {}
            for mol in batch_ligand_mol_list:
                name = mol.GetProp('ud2_molecule_name')
                if name in self.ligand_dms_path_dict:
                    batch_dms_dict[name] = self.ligand_dms_path_dict[name]

            batch_ligand_topology_builder_results = (
                batch_ligand_topology_builder_pool.apply_async(
                    batch_topology_builder_process,
                    kwds=dict(
                        ligand_sdf_file_name=batch_ligand_sdf_file_name,
                        working_dir_name=working_dir_name,
                        has_ff=self.has_ff,
                        dms_path_dict=batch_dms_dict,
                        skip_gaff2=self.skip_gaff2,
                        hooks=batch_hooks,
                    ),
                )
            )

            batch_ligand_topology_builder_results_list[batch_idx] = (
                batch_ligand_topology_builder_results
            )

        batch_ligand_topology_builder_pool.close()
        batch_ligand_topology_builder_pool.join()

        self.total_ligand_info_dict_list = []
        for batch_idx in range(num_batches):
            batch_ligand_topology_builder_results = (
                batch_ligand_topology_builder_results_list[batch_idx]
            )
            ligand_info_dict_list = batch_ligand_topology_builder_results.get()
            for ligand_info_dict in ligand_info_dict_list:
                self.total_ligand_info_dict_list.append(ligand_info_dict)

        if len(self.total_ligand_info_dict_list) != self.num_ligands:
            raise ValueError(
                'Collected number of batch ligands does not equal to \
                    real total number of input ligands!!'
            )

    def get_summary_ligand_info_dict(self):
        self.total_ligand_info_summary_dict = {}

        for ligand_idx in range(self.num_ligands):
            ligand_info_dict = self.total_ligand_info_dict_list[ligand_idx]
            ligand_name = ligand_info_dict['ligand_name']
            atom_info_nested_list = ligand_info_dict['atom_info']
            torsion_info_nested_list = ligand_info_dict['torsion_info']
            root_atom_idx_list = ligand_info_dict['root_atom_idx']
            fragment_atom_idx_nested_list = ligand_info_dict['fragment_atom_idx']

            self.total_ligand_info_summary_dict[ligand_name] = {
                'atoms': atom_info_nested_list,
                'torsions': torsion_info_nested_list,
                'root_atoms': root_atom_idx_list,
                'fragment_atom_idx': fragment_atom_idx_nested_list,
            }
