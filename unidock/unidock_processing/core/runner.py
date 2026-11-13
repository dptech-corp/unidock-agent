"""UnidockProtocolRunner: receptor and ligand preparation, DMS enrichment, the engine run, the CNN stage.

The stages call the core modules (core.dms, core.engine, core.rescore); the plugins loaded from the
``unidock2.plugins`` entry-point group get a hook between each pair of stages (plugins.base.Plugin).
"""
from typing import List, Optional, Tuple, Dict, Any
import os
import logging
import shutil

logger = logging.getLogger(__name__)

from rdkit import Chem
from rdkit.Chem import Descriptors

from unidock_processing.unidocktools.unidock_receptor_topology_builder import (
    UnidockReceptorTopologyBuilder,
)
from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
    UnidockLigandTopologyBuilder,
)
from unidock_processing.core.dms import (
    dms_has_ff_params,
    dms_has_ud2_atom_props,
    dms_has_nonbonded_params,
    enrich_ligand_dms,
    enrich_receptor_dms,
    merge_ligand_dms_files,
    enrich_merged_ligand_dms,
    write_scoring_annotations_safe,
    build_ligand_dms_minimal,
    build_ligand_dms_from_sdf,
    ligand_atom_offsets,
)
from unidock_processing.core.engine import EngineConfig, find_ud2_binary, run_engine
from unidock_processing.core.rescore import engine_num_pose, run_cnn_rescore
from unidock_processing.plugins.base import Plugin, write_skipped_ligands
from unidock_processing.plugins.registry import load_plugins

# Below this many ligands the minimal DMS files are built in this process: a pool costs more than it saves.
PARALLEL_DMS_MIN_LIGANDS = 16


def _build_minimal_dms_slice(working_dir_name: str, items: list) -> list:
    """``build_ligand_dms_minimal`` for each ``(temp_sdf, mol)`` of *items*: the DMS path, or None where it failed.
    Module level, so the pool workers of ``UnidockProtocolRunner._build_minimal_ligand_dms`` can run it."""
    return [build_ligand_dms_minimal(temp_sdf, working_dir_name, mol=mol, write_annotations=False)
            for temp_sdf, mol in items]


class UnidockProtocolRunner(object):
    def __init__(
        self,
        receptor_file_name: str,
        ligand_sdf_file_name_list: List[str],
        target_center: Tuple[float, float, float],
        box_size: Tuple[float, float, float] = (30.0, 30.0, 30.0),
        template_docking: bool = False,
        reference_sdf_file_name: Optional[str] = None,
        core_atom_mapping_dict_list: Optional[List[Optional[Dict[int, int]]]] = None,
        atom_mapper_align: bool = False,
        covalent_ligand: bool = False,
        covalent_residue_atom_info_list: Optional[List[Dict[str, Any]]] = None,
        preserve_receptor_hydrogen: bool = False,
        working_dir_name: str = '.',
        docking_pose_sdf_file_name: str = 'unidock2_pose.sdf',
        n_cpu: Optional[int] = None,
        gpu_device_id: int = 0,
        task: str = 'screen',
        search_mode: str = 'balance',
        exhaustiveness: int = 512,
        randomize: bool = True,
        mc_steps: int = 40,
        opt_steps: int = 15,
        refine_steps: int = 5,
        num_pose: int = 10,
        rescore_pool: int = 10,
        rmsd_limit: float = 1.0,
        energy_range: float = 5.0,
        seed: int = 1234567,
        use_tor_lib: bool = False,
        decomp: bool = False,
        ff_minimize_steps: int = 0,
        cluster_atom_indices_list: Optional[List[Optional[List[int]]]] = None,
        cnn_rescore: bool = False,
        plugins: Optional[List[Plugin]] = None,
    ) -> None:
        """*plugins*: configured plugin instances for this run (``Plugin.from_config`` builds them from the YAML
        blocks), each taking the place of the registry's instance of the same name. The template and covalent
        plugins are also built by their compatibility keyword arguments above (one release); the same plugin
        given both ways is an error."""
        self.receptor_file_name = os.path.abspath(receptor_file_name)

        self.ligand_sdf_file_name_list = [
            os.path.abspath(ligand_sdf_file_name) for ligand_sdf_file_name in ligand_sdf_file_name_list
            ]

        self.target_center = target_center

        # Configuration parameters. The template arguments build the TemplatePlugin below (compatibility
        # constructor, one release; atom_mapper_align alone gives the disabled plugin that only swaps the ligand
        # builder's rotatable-bond finder); the attributes stay readable (tests/helpers.py and ud2align read them).
        # The covalent arguments build the CovalentPlugin the same way (the residue list alone gives the disabled
        # plugin that only hands the list to the receptor builder, as the runner argument did).
        self.template_docking = template_docking
        self.atom_mapper_align = atom_mapper_align

        if reference_sdf_file_name is not None:
            self.reference_sdf_file_name = os.path.abspath(reference_sdf_file_name)
        else:
            self.reference_sdf_file_name = reference_sdf_file_name
        self.core_atom_mapping_dict_list = core_atom_mapping_dict_list

        self.covalent_ligand = covalent_ligand
        self.covalent_residue_atom_info_list = covalent_residue_atom_info_list
        self.preserve_receptor_hydrogen = preserve_receptor_hydrogen

        # Docking parameters
        self.box_size = box_size
        self.n_cpu = n_cpu
        self.gpu_device_id = gpu_device_id
        self.task = task
        self.search_mode = search_mode
        self.exhaustiveness = exhaustiveness
        self.randomize = randomize
        self.mc_steps = mc_steps
        self.opt_steps = opt_steps
        self.refine_steps = refine_steps
        self.num_pose = num_pose
        self.rescore_pool = rescore_pool
        self.rmsd_limit = rmsd_limit
        self.energy_range = energy_range
        self.seed = seed
        self.use_tor_lib = use_tor_lib
        self.decomp = decomp
        self.ff_minimize_steps = ff_minimize_steps
        self.cluster_atom_indices_list = cluster_atom_indices_list
        self.cnn_rescore = cnn_rescore

        self.working_dir_name = os.path.abspath(working_dir_name)
        self.unidock2_output_dir_name = os.path.join(
            self.working_dir_name, 'unidock2_output'
        )

        self.docking_pose_sdf_file_name = os.path.abspath(docking_pose_sdf_file_name)
        os.makedirs(self.unidock2_output_dir_name, exist_ok=True)
        self.unidock2_pose_sdf_file_name = ''

        # Auto-detect DMS inputs
        self.protein_dms = (
            self.receptor_file_name
            if self.receptor_file_name.lower().endswith('.dms')
            else None
        )
        self.ligand_dms = None
        self.ligand_dms_dict: Dict[str, str] = {}
        self.ligand_has_ff = False
        converted_list = []
        for i, lig_path in enumerate(self.ligand_sdf_file_name_list):
            if lig_path.lower().endswith('.dms'):
                self.ligand_has_ff = dms_has_ff_params(lig_path)
                logger.info(
                    "Ligand DMS %s: has_ff=%s", lig_path, self.ligand_has_ff
                )
                sdf_path = self._convert_ligand_dms_to_sdf(lig_path, i)
                converted_list.append(sdf_path)
                if self.ligand_dms is None:
                    self.ligand_dms = lig_path
                mol_name = f'MOL_{i}'
                self.ligand_dms_dict[mol_name] = lig_path
            else:
                converted_list.append(lig_path)
        self.ligand_sdf_file_name_list = converted_list

        # Paths for enriched DMS files (filled during run_unidock_protocol)
        self.enriched_ligand_dms = None
        self.enriched_receptor_dms = None

        self.plugins = load_plugins()
        for plugin in plugins or []:
            self._replace_plugin(plugin)
        given = {plugin.name for plugin in plugins or []}
        if self.covalent_ligand or self.covalent_residue_atom_info_list is not None:
            try:
                from unidock_processing.plugins.covalent.plugin import CovalentPlugin
            except ImportError as error:
                raise ImportError(
                    "covalent_ligand / covalent_residue_atom_info_list need the covalent plugin "
                    "(unidock_processing.plugins.covalent), which is not installed"
                ) from error
            covalent = CovalentPlugin(self.covalent_residue_atom_info_list, enabled=self.covalent_ligand)
            if covalent.name in given:
                raise ValueError("the covalent plugin is given twice: covalent_ligand / "
                                 "covalent_residue_atom_info_list and an instance in plugins=")
            self._replace_plugin(covalent)
        if self.template_docking or self.atom_mapper_align:
            try:
                from unidock_processing.plugins.template.plugin import TemplatePlugin
            except ImportError as error:
                raise ImportError(
                    "template_docking / atom_mapper_align need the template plugin "
                    "(unidock_processing.plugins.template), which is not installed"
                ) from error
            template = TemplatePlugin(
                self.reference_sdf_file_name, core_atom_mapping_dict_list, self.atom_mapper_align,
                enabled=self.template_docking,
            )
            if template.name in given:
                raise ValueError("the template plugin is given twice: template_docking / atom_mapper_align "
                                 "and an instance in plugins=")
            self._replace_plugin(template)
        if self.template_docking:
            self.core_atom_mapping_dict_list = template.core_atom_mapping_dict_list
            if self.target_center == (0.0, 0.0, 0.0):
                self.target_center = template.reference_center()

    def _replace_plugin(self, plugin: Plugin) -> None:
        """Put *plugin* in place of the run's instance of the same name (the registry's), last in the list."""
        self.plugins = [other for other in self.plugins if other.name != plugin.name] + [plugin]

    def _build_minimal_ligand_dms(self, ligand_items: list) -> list:
        """The minimal DMS path (or None) of each ``(mol_name, temp_sdf, mol)`` of *ligand_items*, in order. From
        ``PARALLEL_DMS_MIN_LIGANDS`` ligands on, the work is split into ``n_cpu`` contiguous slices run in a
        ``multiprocess`` pool (the topology builder's pool type); each file depends only on its own ligand, so the
        files are the ones the serial loop writes."""
        n_workers = min(self.n_cpu or os.cpu_count() or 1, len(ligand_items))
        items = [(temp_sdf, mol) for _, temp_sdf, mol in ligand_items]
        if n_workers < 2 or len(items) < PARALLEL_DMS_MIN_LIGANDS:
            return _build_minimal_dms_slice(self.working_dir_name, items)
        from multiprocess.pool import Pool

        # The molecules travel to the workers pickled. RDKit's pickle stores coordinates in single precision unless
        # CoordsAsDouble is set; rounded coordinates would change the DMS geometry and break the coordinate-based
        # atom mapping of the scoring annotations (core.dms.write_scoring_annotations_merged), which then skips them.
        previous = Chem.GetDefaultPickleProperties()
        Chem.SetDefaultPickleProperties(
            Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
        )
        try:
            size = -(-len(items) // n_workers)
            slices = [items[i:i + size] for i in range(0, len(items), size)]
            with Pool(processes=len(slices)) as pool:
                parts = pool.starmap(_build_minimal_dms_slice, [(self.working_dir_name, part) for part in slices])
        finally:
            Chem.SetDefaultPickleProperties(previous)
        return [path for part in parts for path in part]

    def _drop_skipped_ligands(self) -> bool:
        """Leave out the ligands a plugin cannot process (``Plugin.skipped_ligands``), listing them in
        skipped_ligands.csv next to the output SDF. The kept ligands are written to one SDF that replaces the
        ligand list, so the DMS loop, the topology builder and the plugins read the same molecules in the same
        order (``_iter_ligand_mols``). False when no ligand is left."""
        filtering = [plugin for plugin in self.plugins
                     if type(plugin).skipped_ligands is not Plugin.skipped_ligands and getattr(plugin, 'enabled', True)]
        if not filtering or not self.ligand_sdf_file_name_list:
            return True
        records = []                      # (source file, record index in it, mol) of every usable record
        for sdf_path in self.ligand_sdf_file_name_list:
            for record, mol in enumerate(Chem.SDMolSupplier(sdf_path, removeHs=False)):
                if mol is None or Descriptors.NumRadicalElectrons(mol) > 0:
                    continue
                records.append((sdf_path, record, mol))
        mols = [mol for _, _, mol in records]
        skipped = {}
        for plugin in filtering:
            for index, reason in plugin.skipped_ligands(mols).items():
                skipped.setdefault(index, reason)
        if not skipped:
            return True
        csv_path = os.path.join(os.path.dirname(self.docking_pose_sdf_file_name), 'skipped_ligands.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_skipped_ligands(csv_path, (
            (records[i][0], records[i][1], records[i][2].GetProp('_Name') if records[i][2].HasProp('_Name') else '',
             skipped[i])
            for i in sorted(skipped)
        ))
        logger.warning("%d of %d ligands skipped, listed in %s: %s", len(skipped), len(mols), csv_path,
                       next(iter(skipped.values())))
        kept = [mol for i, mol in enumerate(mols) if i not in skipped]
        if not kept:
            logger.warning("no ligand is left to dock")
            return False
        kept_sdf = os.path.join(self.working_dir_name, '_ud2_kept_ligands.sdf')
        writer = Chem.SDWriter(kept_sdf)
        for mol in kept:
            writer.write(mol)
        writer.close()
        self.ligand_sdf_file_name_list = [kept_sdf]
        return True

    def _iter_ligand_mols(self):
        """The ligand molecules in the order of the ligand DMS, hydrogens kept: the DMS loop of run_unidock_protocol
        and the topology builder skip a record that does not parse and a radical molecule, so the plugins are handed
        the same molecules in the same order."""
        for sdf_path in self.ligand_sdf_file_name_list:
            for mol in Chem.SDMolSupplier(sdf_path, removeHs=False):
                if mol is None or Descriptors.NumRadicalElectrons(mol) > 0:
                    continue
                yield mol

    def _convert_ligand_dms_to_sdf(self, dms_path: str, index: int) -> str:
        """Convert a ligand DMS file to SDF via msys for the topology pipeline."""
        try:
            import msys
        except ImportError:
            raise ImportError(
                "msys is required to use DMS files as ligand input. "
                "Install msys or provide ligand in SDF format."
            )

        mol_sys = msys.LoadDMS(dms_path)
        msys.AssignBondOrderAndFormalCharge(mol_sys)
        rdkit_mol = msys.ConvertToRdkit(mol_sys)

        # Set molecule name: prefer msys_name if it's a real name, else use filename stem
        mol_name = ''
        try:
            raw = mol_sys.name.strip()
            # msys.name may return the file path when msys_name is empty; skip path-like values
            if raw and not os.sep in raw and not raw.endswith('.dms'):
                mol_name = raw
        except (AttributeError, Exception):
            pass
        if not mol_name:
            mol_name = os.path.splitext(os.path.basename(dms_path))[0]
        rdkit_mol.SetProp('_Name', mol_name)

        sdf_path = os.path.join(
            self.working_dir_name, f'ligand_from_dms_{index}.sdf'
        )
        writer = Chem.SDWriter(sdf_path)
        writer.write(rdkit_mol)
        writer.close()

        logger.info("Converted ligand DMS %s -> SDF %s", dms_path, sdf_path)
        return sdf_path

    def _try_full_receptor_analysis_subprocess(self, builder):
        """Try running full receptor analysis in a subprocess.
        Returns True if successful, False if subprocess crashed (e.g. segfault).
        On success, writes atom_info to a JSON file that the parent reads.
        """
        import subprocess
        import sys
        import json

        atom_info_json_path = os.path.join(self.working_dir_name, '_receptor_atom_info.json')
        dms_path = builder.receptor_parameterized_dms_file_name

        script = f'''
import msys, json, sys
from unidock_processing.unidocktools.protein_topology import prepare_receptor_residue_mol_list
from unidock_processing.unidocktools.unidock_vina_atom_types import VINA_ATOM_TYPE_DICT

receptor_sys = msys.LoadDMS("{dms_path}")
protein_mol, protein_residues, cofactor_residues = prepare_receptor_residue_mol_list(receptor_sys)

atom_info_list = []
for res_list in [protein_residues, cofactor_residues]:
    for res_mol in res_list:
        for atom in res_mol.GetAtoms():
            atom_info_list.append([
                atom.GetDoubleProp("x"), atom.GetDoubleProp("y"), atom.GetDoubleProp("z"),
                VINA_ATOM_TYPE_DICT[atom.GetProp("vina_atom_type")],
                atom.GetDoubleProp("atom_charge"),
            ])

with open("{atom_info_json_path}", "w") as f:
    json.dump(atom_info_list, f)
'''
        try:
            result = subprocess.run(
                [sys.executable, '-c', script],
                capture_output=True, timeout=600
            )
            if result.returncode == 0 and os.path.exists(atom_info_json_path):
                with open(atom_info_json_path, 'r') as f:
                    builder.atom_info_nested_list = json.load(f)
                os.remove(atom_info_json_path)
                return True
        except Exception:
            pass
        return False

    def run_unidock_protocol(self) -> str:
        for plugin in self.plugins:
            plugin.configure(dict(vars(self)))
        for plugin in self.plugins:
            reason = plugin.check_setup()
            if reason:
                logger.warning("%s plugin: %s; nothing is docked", plugin.name, reason)
                return

        ## leave out the ligands a plugin cannot process, before any of them is prepared
        if self.ligand_dms is None and not self._drop_skipped_ligands():
            return

        ## prepare receptor
        unidock_receptor_topology_builder = self._prepare_receptor()

        ## Build per-ligand DMS from SDF when no DMS is already provided.
        ## DMS is now always required for the C++ engine (JSON path removed).
        ## When ff_minimize_steps=0, skip antechamber entirely and build
        ## minimal DMS (coords + bonds only, no GAFF2 FF params).
        if self.ligand_dms is None and self.ligand_sdf_file_name_list:
            self.ligand_dms_dict = {}
            all_succeeded = True
            last_mol = None
            use_minimal_dms = (self.ff_minimize_steps == 0)
            # (MOL_<i>, temp SDF path, mol) for every usable record, in input order
            ligand_items = []
            for sdf_path in self.ligand_sdf_file_name_list:
                for mol in Chem.SDMolSupplier(sdf_path, removeHs=False):
                    if mol is None:
                        continue
                    if Descriptors.NumRadicalElectrons(mol) > 0:
                        continue
                    mol_name = f'MOL_{len(ligand_items)}'
                    temp_sdf = os.path.join(
                        self.working_dir_name, f'_dms_input_{mol_name}.sdf'
                    )
                    ligand_items.append((mol_name, temp_sdf, mol))
            if use_minimal_dms:
                # The parsed mol goes straight in (temp_sdf only names the output). With several ligands the
                # merge rebuilds the annotation tables, so they are written below only for a single ligand.
                dms_results = self._build_minimal_ligand_dms(ligand_items)
                if ligand_items:
                    last_mol = ligand_items[-1][2]
            for k, (mol_name, temp_sdf, mol) in enumerate(ligand_items):
                if use_minimal_dms:
                    dms_result = dms_results[k]
                else:
                    # Write single-mol temp SDF for antechamber
                    writer = Chem.SDWriter(temp_sdf)
                    writer.write(mol)
                    writer.close()
                    dms_result = build_ligand_dms_from_sdf(
                        temp_sdf, self.working_dir_name
                    )
                if dms_result:
                    self.ligand_dms_dict[mol_name] = dms_result
                else:
                    all_succeeded = False
                    break
            if all_succeeded and use_minimal_dms and len(self.ligand_dms_dict) == 1:
                write_scoring_annotations_safe(
                    next(iter(self.ligand_dms_dict.values())), last_mol
                )
            if all_succeeded and self.ligand_dms_dict:
                self.ligand_has_ff = not use_minimal_dms
                self.ligand_dms = list(self.ligand_dms_dict.values())[0]
                logger.info(
                    "SDF -> DMS conversion succeeded for %d ligand(s)"
                    " (has_ff=%s, minimal=%s)",
                    len(self.ligand_dms_dict), self.ligand_has_ff,
                    use_minimal_dms,
                )
            else:
                self.ligand_dms_dict = {}
                self.ligand_has_ff = False
                logger.warning(
                    "SDF -> DMS building failed; engine may fail "
                    "without enriched DMS input"
                )

        ## prepare ligands input
        skip_gaff2 = (self.ff_minimize_steps == 0 and not self.ligand_has_ff)
        ligand_builder_kwargs = dict(
            n_cpu=self.n_cpu,
            working_dir_name=self.working_dir_name,
            has_ff=self.ligand_has_ff,
            ligand_dms_path_dict=self.ligand_dms_dict,
            skip_gaff2=skip_gaff2,
        )
        if self.plugins:
            for mol in self._iter_ligand_mols():
                for plugin in self.plugins:
                    plugin.prepare_ligand(mol, ligand_builder_kwargs)
        unidock_ligand_topology_builder = UnidockLigandTopologyBuilder(
            self.ligand_sdf_file_name_list, **ligand_builder_kwargs,
        )

        unidock_ligand_topology_builder.generate_batch_ligand_topology()
        unidock_ligand_topology_builder.get_summary_ligand_info_dict()

        ## Enrich DMS files with topology data for C++ engine
        self._produce_enriched_dms(
            unidock_ligand_topology_builder,
            unidock_receptor_topology_builder,
        )
        if self.plugins:
            offsets = ligand_atom_offsets(self.enriched_ligand_dms) if self.enriched_ligand_dms else {}
            for plugin in self.plugins:
                plugin.annotate_ligand_dms(self.enriched_ligand_dms, offsets)

        ## run ud2 engine
        self._run_engine()

        ## generate output ud2 pose sdf
        # Check if C++ engine produced SDF output directly (DMS path was set)
        sdf_files = [
            os.path.join(self.unidock2_output_dir_name, f)
            for f in os.listdir(self.unidock2_output_dir_name)
            if f.endswith('.sdf')
        ]
        if sdf_files:
            # C++ engine wrote SDF directly — concatenate batch files into
            # the final output SDF, skip Python PoseWriter entirely.
            logger.info(
                "C++ SDF output found (%d files), skipping PoseWriter",
                len(sdf_files),
            )
            sdf_files.sort()
            with open(self.docking_pose_sdf_file_name, 'w') as out:
                for sdf_f in sdf_files:
                    with open(sdf_f) as inp:
                        out.write(inp.read())

        if self.cnn_rescore:
            self._run_cnn_rescore()

        for plugin in self.plugins:
            plugin.post_process(self.docking_pose_sdf_file_name)

    def _prepare_receptor(self) -> 'UnidockReceptorTopologyBuilder':
        """The receptor topology builder, prepared: the plugins' annotate_receptor runs on the constructed builder
        before the preparation, then generate_receptor_topology and, for a receptor not already enriched, the atom
        typing (Vina-only, or the full analysis in a subprocess) and the summary."""
        # Skip full topology analysis if receptor DMS already has ud2_atom_props
        receptor_is_enriched = bool(self.protein_dms) and dms_has_ud2_atom_props(self.protein_dms)
        if receptor_is_enriched:
            logger.info("Receptor DMS already enriched (has ud2_atom_props), skipping topology analysis")
            self.enriched_receptor_dms = self.protein_dms
            unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(
                self.receptor_file_name,
                prepared_hydrogen=self.preserve_receptor_hydrogen,
                working_dir_name=self.working_dir_name,
            )
        else:
            unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(
                self.receptor_file_name,
                prepared_hydrogen=self.preserve_receptor_hydrogen,
                working_dir_name=self.working_dir_name,
                require_ff_params=self.ff_minimize_steps > 0,
            )

        for plugin in self.plugins:
            plugin.annotate_receptor(unidock_receptor_topology_builder)

        unidock_receptor_topology_builder.generate_receptor_topology()
        if receptor_is_enriched:
            return unidock_receptor_topology_builder

        receptor_dms = unidock_receptor_topology_builder.receptor_parameterized_dms_file_name

        # A DMS receptor is already prepared, and one prepared without utop has no force field for the
        # full analysis to read: both only need Vina atom types, not protein processing.
        if self.protein_dms or not dms_has_nonbonded_params(receptor_dms):
            logger.info("Receptor DMS has no Vina atom types, assigning them from connectivity")
            unidock_receptor_topology_builder.analyze_receptor_topology_vina_only()
        # Try full analysis (with GAFF2 types) in subprocess to handle segfaults
        elif not self._try_full_receptor_analysis_subprocess(unidock_receptor_topology_builder):
            logger.warning("Full receptor analysis failed (msys), using Vina-only fallback")
            unidock_receptor_topology_builder.analyze_receptor_topology_vina_only()

        unidock_receptor_topology_builder.get_summary_receptor_info_dict()
        return unidock_receptor_topology_builder

    def _search_policy(self) -> Dict[str, Any]:
        """The engine settings the plugins override (Plugin.search_policy), a later plugin's keys last."""
        policy: Dict[str, Any] = {}
        for plugin in self.plugins:
            policy.update(plugin.search_policy())
        return policy

    def _constraint_docking(self) -> bool:
        """Constraint docking: a plugin's search policy (the template and covalent plugins)."""
        return bool(self._search_policy().get('constraint_docking', False))

    def _engine_order_keepers(self) -> List[str]:
        """The plugins that keep the engine's pose order (Plugin.keeps_engine_order: an enabled bias plugin)."""
        return [plugin.name for plugin in self.plugins if plugin.keeps_engine_order()]

    def _engine_num_pose(self) -> int:
        """Poses per ligand the engine writes (core.rescore.engine_num_pose)."""
        return engine_num_pose(
            self.num_pose, self.rescore_pool, self.cnn_rescore, self._constraint_docking(),
            keeps_engine_order=bool(self._engine_order_keepers()),
        )

    def _run_cnn_rescore(self) -> None:
        """Reorder each ligand's poses with the gnina pose-CNN blend (core.rescore.run_cnn_rescore); skipped under
        constraint docking and when a plugin keeps the engine order (a biased run)."""
        if self._constraint_docking():
            logger.info("CNN rescoring skipped: constraint / template / covalent runs keep the engine order")
            return
        keepers = self._engine_order_keepers()
        if keepers:
            logger.info("CNN rescoring skipped: plugin %s keeps the engine order (Plugin.keeps_engine_order)",
                        ', '.join(keepers))
            return
        run_cnn_rescore(
            self.docking_pose_sdf_file_name, self.enriched_ligand_dms, self.enriched_receptor_dms,
            keep=self.num_pose, gpu_device_id=self.gpu_device_id, pool_written=self._engine_num_pose(),
        )

    def _produce_enriched_dms(
        self,
        ligand_builder: 'UnidockLigandTopologyBuilder',
        receptor_builder: 'UnidockReceptorTopologyBuilder',
    ) -> None:
        """Copy DMS files to working dir and write ud2_* topology tables into them.

        After this method:
          - self.enriched_ligand_dms is the path to the enriched ligand DMS
          - self.enriched_receptor_dms is the path to the enriched receptor DMS
        Both may be None if the original source was not a DMS.
        """
        # --- Ligand DMS enrichment ---
        if self.ligand_dms is not None and self.ligand_dms_dict:
            n_ligands = len(self.ligand_dms_dict)

            if n_ligands == 1:
                # Single ligand: copy and enrich directly
                enriched_path = os.path.join(
                    self.working_dir_name, 'ligand_enriched.dms'
                )
                shutil.copy2(self.ligand_dms, enriched_path)

                summary = ligand_builder.total_ligand_info_summary_dict
                lig_names = list(summary.keys())
                if lig_names:
                    lig_data = summary[lig_names[0]]
                    rdkit_mol = (
                        ligand_builder.ligand_mol_list[0]
                        if ligand_builder.ligand_mol_list else None
                    )
                    enrich_ligand_dms(
                        enriched_path,
                        lig_data['atoms'],
                        lig_data['torsions'],
                        lig_data['root_atoms'],
                        molecule_name=lig_names[0],
                        rdkit_mol=rdkit_mol,
                    )
                    self.enriched_ligand_dms = enriched_path
                    logger.info("Enriched ligand DMS: %s", enriched_path)
            else:
                # Multi-ligand: merge per-ligand DMS files, then enrich
                merged_path = os.path.join(
                    self.working_dir_name, 'ligand_merged.dms'
                )
                merge_ligand_dms_files(self.ligand_dms_dict, merged_path)
                enrich_merged_ligand_dms(merged_path, ligand_builder)
                self.enriched_ligand_dms = merged_path
                logger.info(
                    "Enriched merged ligand DMS (%d ligands): %s",
                    n_ligands, merged_path,
                )
        else:
            self.enriched_ligand_dms = None

        # --- Receptor DMS enrichment ---
        if self.enriched_receptor_dms:
            logger.info("Receptor DMS already enriched, skipping enrichment step")
        else:
            rec_dms_src = getattr(
                receptor_builder, 'receptor_parameterized_dms_file_name', None
            )
            if rec_dms_src and os.path.isfile(rec_dms_src):
                enriched_path = os.path.join(
                    self.working_dir_name, 'receptor_enriched.dms'
                )
                shutil.copy2(rec_dms_src, enriched_path)
                rec_atom_info = receptor_builder.atom_info_nested_list
                if rec_atom_info:
                    enrich_receptor_dms(enriched_path, rec_atom_info)
                    self.enriched_receptor_dms = enriched_path
                    logger.info("Enriched receptor DMS: %s", enriched_path)
                else:
                    self.enriched_receptor_dms = None
            else:
                self.enriched_receptor_dms = None

    @staticmethod
    def _find_ud2_binary():
        """Find the ud2 binary (core.engine.find_ud2_binary)."""
        return find_ud2_binary()

    def _run_engine(self):
        """Render the engine configuration and run ud2 (core.engine.run_engine): the ud2 binary as a
        subprocess when one is found, else the pybind11 in-process call. The plugins' search policies
        update the configuration and their YAML blocks follow the core blocks.
        """
        # Always pass DMS paths when available (enables FF energy reporting even without minimization)
        lig_dms = self.enriched_ligand_dms or self.ligand_dms
        prot_dms = self.enriched_receptor_dms or self.protein_dms

        # Enriched DMS inputs (for DMS-only engine path)
        inputs = {}
        if self.enriched_ligand_dms and self.enriched_receptor_dms:
            inputs = {
                'ligand_dms_full': self.enriched_ligand_dms,
                'receptor_dms_full': self.enriched_receptor_dms,
            }

        cfg = EngineConfig(
            seed=self.seed,
            exhaustiveness=self.exhaustiveness,
            randomize=self.randomize,
            mc_steps=self.mc_steps,
            opt_steps=self.opt_steps,
            refine_steps=self.refine_steps,
            rmsd_limit=self.rmsd_limit,
            num_pose=self._engine_num_pose(),
            rescore_pool=self.rescore_pool,
            energy_range=self.energy_range,
            use_tor_lib=self.use_tor_lib,
            ff_minimize_steps=self.ff_minimize_steps,
            ligand_dms=lig_dms,
            protein_dms=prot_dms,
            cluster_atom_indices_list=self.cluster_atom_indices_list,
            ncpu=os.cpu_count() or 4,
            task=self.task,
            search_mode=self.search_mode,
            constraint_docking=False,
            decomp=self.decomp,
            center=self.target_center,
            box_size=self.box_size,
            output_dir=self.unidock2_output_dir_name,
            inputs=inputs,
            working_dir=self.working_dir_name,
            gpu_device_id=self.gpu_device_id,
        )

        extra_blocks: Dict[str, Dict[str, Any]] = {}
        for plugin in self.plugins:
            extra_blocks.update(plugin.yaml_block())
        cfg.update(self._search_policy())

        run_engine(cfg, extra_blocks)
