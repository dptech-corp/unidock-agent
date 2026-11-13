from unidock_processing.plugins.registry import plugin_classes


class CLICommand:
    """Perform whole docking protocol.

    Note: Required arguments should be passed by either command line arguments or YAML Required options.
    If both sides are specified,
    the values passed in the command line arguments will overide the values passed in YAML file.

    Example input YAML configurations:

    Required:
        receptor: 1G9V_protein_water_cleaned.pdb
        ligand: ligand_prepared.sdf
        ligand_batch: null
        center: [5.122, 18.327, 37.332]
    Advanced:
        exhaustiveness: 512
        randomize: true
        mc_steps: 20
        opt_steps: -1
        refine_steps: 5
        num_pose: 10
        rescore_pool: 10
        rmsd_limit: 1.0
        energy_range: 3.0
        seed: 12345
        use_tor_lib: false
    Hardware:
        gpu_device_id: 0
    Settings:
        size: [30.0, 30.0, 30.0]
        task: screen
        search_mode: balance
    Preprocessing:
        preserve_receptor_hydrogen: false
        temp_dir_name: /tmp
        output_docking_pose_sdf_file_name: unidock2_pose.sdf
    template:
        template_docking: false
        reference_sdf_file_name: null
        core_atom_mapping_dict_list: null
    covalent:
        covalent_ligand: false
        covalent_residue_atom_info_list: null
    bias:
        template_sdf: null
        aligner: null            # null = zalign
        bias_k: null             # null = 0.05
        bias_type: null          # null = align
        ref_atom_indices: null
        translate: null          # null = true
        ring_bias: null          # null = false
        ref_ring_indices: null
        ring: null               # null = the engine's k 30, s 3, d0 1

    """

    def add_arguments(parser):
        parser.add_argument(
            '-r',
            '--receptor',
            default=None,
            help='Receptor structure file in PDB or DMS format',
        )

        parser.add_argument(
            '-l',
            '--ligand',
            default=None,
            help='Single ligand structure file in SDF or DMS format',
        )

        parser.add_argument(
            '-lb',
            '--ligand_batch',
            default=None,
            help='Recorded batch text file of ligand SDF file path',
        )

        parser.add_argument(
            '-c',
            '--center',
            nargs=3,
            type=float,
            metavar=('center_x', 'center_y', 'center_z'),
            default=[0.0, 0.0, 0.0],
            help='Docking box center coordinates',
        )

        parser.add_argument(
            '-o',
            '--output_docking_pose_sdf_file_name',
            default='unidock2_pose.sdf',
            help='Output docking pose SDF file name',
        )

        parser.add_argument(
            '-cf',
            '--configurations',
            default=None,
            help='Uni-Dock2 configuration YAML file recording all other options',
        )

        parser.add_argument(
            '--decomp',
            action='store_true',
            default=False,
            help='Enable per-atom energy decomposition (5 Vina terms per atom) in final scoring',
        )

        parser.add_argument(
            '--cnn_rescore',
            action='store_true',
            default=False,
            help='Reorder each ligand\'s poses by the engine score blended with the gnina pose-CNN ensemble '
                 '(torch, and molgrid in the CUDA builds, come with unidock2; UD2_CNN_PYTHON runs the stage in another '
                 'interpreter; '
                 'equivalent to Settings: cnn_rescore: true)',
        )

        parser.add_argument(
            '--ff_minimize_steps',
            type=int,
            default=0,
            help='Number of FF minimization steps after filtering (default 0 = disabled). '
                 'Requires receptor and ligand inputs in DMS format.',
        )

        for plugin_cls in plugin_classes():            # each installed plugin's options (plugins.registry)
            plugin_cls.cli_options(parser)

    def check_unrecognized_arguments(arguments):
        """An option of a known plugin that is not installed (``--<name>_...`` or ``--no_<name>_...``, e.g.
        ``--bias_template_sdf`` without ud2align) is the clear error naming the plugin's module, as a 'bias:' block
        is in io.yaml, instead of argparse's 'unrecognized arguments' (main.py calls this before that error)."""
        from unidock_processing.io.yaml import plugin_missing
        from unidock_processing.plugins.registry import missing_plugins
        for name, spec in missing_plugins().items():
            options = [argument.split('=', 1)[0] for argument in arguments
                       if argument.startswith((f'--{name}_', f'--no_{name}_'))]
            if options:
                raise plugin_missing(name, spec, f"the option(s) {options}")

    def run(args):
        import os
        from unidock_processing.io.yaml import UnidockConfig, read_unidock_params_from_yaml
        from unidock_processing.io.get_temp_dir_prefix import get_temp_dir_prefix
        from unidock_processing.io.tempfile import TemporaryDirectory
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )

        ## Read all arguments from user input YAML first
        kwargs_dict = {}
        if args.configurations:
            extra_params = read_unidock_params_from_yaml(args.configurations)
            kwargs_dict = extra_params.to_protocol_kwargs()
        else:
            extra_params = UnidockConfig()
            kwargs_dict = extra_params.to_protocol_kwargs()

        print(kwargs_dict)

        ## Parse receptor input
        kwargs_receptor_file_name = kwargs_dict.pop('receptor', None)
        if kwargs_receptor_file_name is not None:
            kwargs_receptor_file_name = os.path.abspath(kwargs_receptor_file_name)

        if args.receptor is not None:
            receptor_file_name = os.path.abspath(args.receptor)
        else:
            receptor_file_name = kwargs_receptor_file_name

        if receptor_file_name is None:
            raise ValueError('Receptor file name not specified !')

        ## Parse single ligand sdf file input
        kwargs_ligand_sdf_file_name = kwargs_dict.pop('ligand', None)
        if kwargs_ligand_sdf_file_name is not None:
            kwargs_ligand_sdf_file_name = os.path.abspath(kwargs_ligand_sdf_file_name)

        if args.ligand is not None:
            ligand_sdf_file_name = os.path.abspath(args.ligand)
        else:
            ligand_sdf_file_name = kwargs_ligand_sdf_file_name

        ## Parse ligand batch file input
        kwargs_ligand_batch_file_name = kwargs_dict.pop('ligand_batch', None)
        if kwargs_ligand_batch_file_name is not None:
            kwargs_ligand_batch_file_name = os.path.abspath(kwargs_ligand_batch_file_name)

        if args.ligand_batch is not None:
            ligand_batch_file_name = os.path.abspath(args.ligand_batch)
        else:
            ligand_batch_file_name = kwargs_ligand_batch_file_name

        if ligand_sdf_file_name is None and ligand_batch_file_name is None:
            raise ValueError('Ligand file input not found (provide SDF or DMS) !')

        ## Summarize ligand input sdf files
        if ligand_batch_file_name:
            with open(ligand_batch_file_name, 'r') as ligand_batch_file:
                ligand_batch_line_list = ligand_batch_file.readlines()

            batch_ligand_sdf_file_name_list = []
            for ligand_batch_line in ligand_batch_line_list:
                batch_ligand_sdf_file_name = ligand_batch_line.strip()
                if len(batch_ligand_sdf_file_name) != 0:
                    batch_ligand_sdf_file_name_list.append(
                        os.path.abspath(batch_ligand_sdf_file_name)
                    )
        else:
            batch_ligand_sdf_file_name_list = []

        if ligand_sdf_file_name:
            total_ligand_sdf_file_name_list = [
                ligand_sdf_file_name
            ] + batch_ligand_sdf_file_name_list
        else:
            total_ligand_sdf_file_name_list = batch_ligand_sdf_file_name_list

        if len(total_ligand_sdf_file_name_list) == 0:
            raise ValueError('Ligand file input not found (provide SDF or DMS) !!')

        ## Parse docking center input
        kwargs_center = kwargs_dict.pop('center', None)
        if tuple(args.center) != (0.0, 0.0, 0.0):            # argparse gives a list; the YAML centre is used otherwise
            docking_center = args.center
        else:
            docking_center = kwargs_center

        ## Specify docking pose SDF file name
        kwargs_docking_pose_sdf_file_name = kwargs_dict.pop('output_docking_pose_sdf_file_name', None)
        if args.output_docking_pose_sdf_file_name != 'unidock2_pose.sdf':
            docking_pose_sdf_file_name = args.output_docking_pose_sdf_file_name
        else:
            docking_pose_sdf_file_name = kwargs_docking_pose_sdf_file_name

        docking_pose_sdf_file_name = os.path.abspath(docking_pose_sdf_file_name)
        _ = kwargs_dict.pop('output_receptor_dms_file_name', None)

        ## Prepare temp dir
        root_temp_dir_name = os.path.abspath(kwargs_dict.pop('temp_dir_name', None))
        temp_dir_prefix = os.path.join(
            root_temp_dir_name, get_temp_dir_prefix('docking')
        )

        if root_temp_dir_name == '/tmp':
            remove_temp_dir = True
        else:
            remove_temp_dir = False

        ## Handle --decomp flag
        decomp = getattr(args, 'decomp', False)

        ff_minimize_steps = getattr(args, 'ff_minimize_steps', 0)

        if getattr(args, 'cnn_rescore', False):
            kwargs_dict['cnn_rescore'] = True

        for plugin_cls in plugin_classes():
            plugin_kwargs = plugin_cls.cli_kwargs(args)
            # a configured instance joins the YAML's 'plugins' list after it: the runner keeps the last of a name
            plugins = plugin_kwargs.pop('plugins', [])
            kwargs_dict.update(plugin_kwargs)
            if plugins:
                kwargs_dict.setdefault('plugins', []).extend(plugins)

        ## Run docking protocol
        with TemporaryDirectory(prefix=temp_dir_prefix, delete=remove_temp_dir) as temp_dir_name:
            docking_runner = UnidockProtocolRunner(
                receptor_file_name=receptor_file_name,
                ligand_sdf_file_name_list=total_ligand_sdf_file_name_list,
                target_center=tuple(docking_center),
                working_dir_name=temp_dir_name,
                docking_pose_sdf_file_name=docking_pose_sdf_file_name,
                decomp=decomp,
                ff_minimize_steps=ff_minimize_steps,
                **kwargs_dict,
            )

            docking_runner.run_unidock_protocol()
