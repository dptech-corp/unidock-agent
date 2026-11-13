#!/usr/bin/env python
"""Check a unidock2 configuration YAML before a run and print the values the run will use.

    python check_config.py config.yaml [-r receptor] [-l ligand] [-lb batch] [-c X Y Z] [-o out.sdf] [--cnn_rescore]

Give the same -r / -l / -lb / -c / -o / --cnn_rescore as the unidock2 docking command, and run it from the directory
unidock2 will run in: relative paths resolve against the current directory, as in the CLI. The file is read with the
CLI's own parser (unidock_processing.io.yaml). On top of the parser this reports what it accepts silently: unknown
or misspelt keys and engine-only keys (both ignored), search_mode presets overriding exhaustiveness and mc_steps,
a box at the origin, missing files and directories, plugin settings that are ignored, and a CNN stage that will
be skipped or cannot run. Exit status 1 when there are errors.
"""
import argparse
import difflib
import logging
import os
import shutil
import subprocess
import sys

import yaml

logger = logging.getLogger("check_config")

# Keys only the ud2 engine YAML reads (unidock_engine/src/config/parse.cpp); the CLI YAML ignores them.
ENGINE_ONLY = {
    "Advanced": {"use_grid", "grid_spacing", "grid_margin", "box_prec", "tor_prec", "tor_lib", "rng_period",
                 "ligand_dms", "protein_dms", "cluster_atom_indices", "bias", "bias_k", "ff_minimize_mode"},
    "Hardware": {"ncpu"},
    "Settings": {"inner_box", "angular_hbond", "score_weights", "rescore_weights", "affinity_weights",
                 "constraint_docking", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"},
    "bias": {"type", "k"},
}
ENGINE_ONLY_BLOCKS = {"Inputs", "Outputs"}
# Set on the command line only.
CLI_FLAG_ONLY = {"decomp": "--decomp", "ff_minimize_steps": "--ff_minimize_steps N"}
ENGINE_TASKS = ("screen", "score", "optimize", "mc", "randomize")
# search_mode presets in parse.cpp: exhaustiveness on CUDA, on CPU, and mc_steps.
PRESETS = {"fast": (128, 64, 20), "balance": (256, 128, 30), "detail": (512, 256, 40)}
COVALENT_PROPS = ("covalent_atom_indices", "covalent_atom_names", "covalent_residue_names",
                  "covalent_residue_indices", "covalent_chain_indices")


def suggestion(name, candidates):
    close = difflib.get_close_matches(name, sorted(candidates), n=1, cutoff=0.6)
    return f" (did you mean '{close[0]}'?)" if close else ""


def check_keys(raw, fields, missing_plugins, errors):
    """Blocks and keys the CLI parser would ignore without a word."""
    for block, body in raw.items():
        if block in ENGINE_ONLY_BLOCKS:
            errors.append(f"'{block}:' is an engine-YAML block; unidock2 ignores it (see ud2-engine-yaml)")
            continue
        if block not in fields and block not in missing_plugins:
            errors.append(f"unknown block '{block}:', ignored by unidock2{suggestion(block, fields)}")
            continue
        if block not in fields or body is None:
            continue
        if not isinstance(body, dict):
            errors.append(f"'{block}:' must be a mapping of keys, not {type(body).__name__}")
            continue
        for key in body:
            if key in fields[block]:
                continue
            if key in ENGINE_ONLY.get(block, ()):
                errors.append(f"{block}.{key} is read only by the ud2 engine YAML; unidock2 ignores it "
                              f"(see ud2-engine-yaml{'; the bias block here uses bias_type / bias_k' if block == 'bias' else ''})")
            elif key in CLI_FLAG_ONLY:
                errors.append(f"{block}.{key} is not read from the YAML; pass {CLI_FLAG_ONLY[key]} to unidock2 docking")
            else:
                errors.append(f"unknown key {block}.{key}, ignored by unidock2{suggestion(key, fields[block])}")
    ring = (raw.get("bias") or {}).get("ring") if isinstance(raw.get("bias"), dict) else None
    if isinstance(ring, dict):
        for key in set(ring) - {"k", "s", "d0"}:
            errors.append(f"unknown key bias.ring.{key}, ignored{suggestion(key, ('k', 's', 'd0'))}")


def resolve_file(path, what, yaml_dir, errors):
    """Absolute path of *path* as the CLI resolves it (current directory); an error when it does not exist."""
    full = os.path.abspath(path)
    if os.path.isfile(full):
        return full
    hint = ""
    if not os.path.isabs(path) and os.path.isfile(os.path.join(yaml_dir, path)):
        hint = " (it exists next to the YAML: run unidock2 from there or give an absolute path)"
    errors.append(f"{what} not found: {full}{hint}")
    return None


def sdf_records(path):
    """(records, records carrying every covalent property) of an SDF, by text scan."""
    n, n_cov, props = 0, 0, set()
    with open(path, errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                props.add(line.split("<", 1)[-1].split(">", 1)[0])
            elif line.startswith("$$$$"):
                n += 1
                n_cov += all(p in props for p in COVALENT_PROPS)
                props = set()
    return n, n_cov


def cnn_device_check(gpu_device_id):
    """None when the CNN scorer's interpreter has CUDA torch and the device exists, else the problem."""
    python = os.environ.get("UD2_CNN_PYTHON", sys.executable)
    code = f"import torch; print(torch.cuda.is_available() and torch.cuda.device_count() > {gpu_device_id})"
    try:
        result = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"cannot run the CNN interpreter {python}: {error}"
    if result.returncode != 0:
        return f"torch is not importable in {python}"
    if result.stdout.strip() != "True":
        return f"no CUDA device {gpu_device_id} for torch in {python}: the runner starts the scorer on cuda:{gpu_device_id}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("-r", "--receptor")
    ap.add_argument("-l", "--ligand")
    ap.add_argument("-lb", "--ligand_batch")
    ap.add_argument("-c", "--center", nargs=3, type=float)
    ap.add_argument("-o", "--output")
    ap.add_argument("--cnn_rescore", action="store_true")
    ap.add_argument("--decomp", action="store_true")
    ap.add_argument("--ff_minimize_steps", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("unidock_processing").setLevel(logging.ERROR)   # its warnings are reported below instead

    errors, warnings, notes = [], [], []
    try:
        with open(args.config) as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as error:
        logger.error("ERROR  cannot read %s: %s", args.config, error)
        return 1
    if not isinstance(raw, dict):
        logger.error("ERROR  %s is not a mapping of blocks", args.config)
        return 1
    yaml_dir = os.path.dirname(os.path.abspath(args.config))

    from unidock_processing.io.yaml import (CONFIG_MAPPING, DEPRECATED_PREPROCESSING_KEYS, MISSING_PLUGINS,
                                            PreprocessingConfig, UnidockConfig)
    fields = {key: set(model.model_fields) for key, _, model, _ in CONFIG_MAPPING}
    fields["Preprocessing"] = set(PreprocessingConfig.model_fields)
    check_keys(raw, fields, MISSING_PLUGINS, errors)

    try:
        config = UnidockConfig.from_dict(raw)
        kwargs = config.to_protocol_kwargs()
    except Exception as error:   # the CLI's own rejection: validation, a plugin that is not installed, a bad bias block
        errors.append(f"unidock2 rejects the file: {type(error).__name__}: {error}")
        for message in errors:
            logger.error("ERROR  %s", message)
        logger.info("result: %d errors", len(errors))
        return 1

    for name in MISSING_PLUGINS:
        if name in raw:
            warnings.append(f"'{name}:' block ignored: the {name} plugin ({MISSING_PLUGINS[name]}) is not installed")
    deprecated = sorted(k for k in (raw.get("Preprocessing") or {}) if k in DEPRECATED_PREPROCESSING_KEYS)
    if deprecated:
        warnings.append(f"Preprocessing keys {deprecated} are deprecated: move them to the "
                        f"{sorted({DEPRECATED_PREPROCESSING_KEYS[k] for k in deprecated})} block(s)")

    req, adv, hw, st, pre = config.required, config.advanced, config.hardware, config.settings, config.preprocessing
    plugin_names = [plugin.name for plugin in kwargs.get("plugins", [])]
    template_on = bool(kwargs.get("template_docking"))
    covalent_on = bool(kwargs.get("covalent_ligand"))
    bias_on = "bias" in plugin_names
    report = []

    # inputs
    receptor = args.receptor or req.receptor
    if receptor is None:
        errors.append("no receptor: set Required.receptor or pass -r")
    else:
        receptor = resolve_file(receptor, "receptor", yaml_dir, errors)
    if receptor:
        if receptor.endswith(".pdb"):
            route = "fepfixer" if shutil.which("fepfixer") else "PDBFixer + tleap (non-water HETATM residues are deleted)"
            notes.append(f"receptor is a PDB, prepared on every run via {route}; "
                         "unidock2 protein_prep -r x.pdb -o x.dms once gives a reusable DMS")
        elif not receptor.endswith(".dms"):
            errors.append(f"receptor must be .pdb or .dms: {receptor}")
        report.append(("receptor", receptor))

    ligand = args.ligand or req.ligand
    batch = args.ligand_batch or req.ligand_batch
    ligand_files = []
    if ligand is None and batch is None:
        errors.append("no ligands: set Required.ligand / ligand_batch or pass -l / -lb")
    if ligand is not None:
        ligand_files.append(resolve_file(ligand, "ligand", yaml_dir, errors))
    if batch is not None:
        batch_path = resolve_file(batch, "ligand_batch file", yaml_dir, errors)
        if batch_path:
            with open(batch_path) as handle:
                listed = [line.strip() for line in handle if line.strip()]
            missing = [p for p in listed if not os.path.isfile(os.path.abspath(p))]
            if missing:
                errors.append(f"{len(missing)} of {len(listed)} files in {batch} not found (paths resolve against "
                              f"the current directory), first: {missing[0]}")
            ligand_files += [os.path.abspath(p) for p in listed if p not in missing]
    ligand_files = [p for p in ligand_files if p]
    n_records = n_covalent = 0
    for path in ligand_files:
        if path.endswith(".sdf"):
            n, n_cov = sdf_records(path)
            n_records += n
            n_covalent += n_cov
    if ligand_files:
        report.append(("ligands", f"{len(ligand_files)} file(s), {n_records} SDF record(s)"))

    # box
    given_center = args.center is not None and tuple(args.center) != (0.0, 0.0, 0.0)
    center = tuple(args.center) if given_center else tuple(req.center)
    source = "command line" if given_center else "YAML"
    if center == (0.0, 0.0, 0.0):
        if template_on:
            source = "the reference ligand's centre (template run)"
        else:
            warnings.append("box centred at the origin: neither -c nor Required.center is set")
    report.append(("centre", f"{center}  from {source}"))
    if any(size <= 0 for size in st.box_size):
        errors.append(f"box_size must be positive: {st.box_size}")
    report.append(("box", f"{tuple(st.box_size)} A"))

    # search
    if st.task not in ENGINE_TASKS:
        errors.append(f"task '{st.task}' is not one of {ENGINE_TASKS}")
    if st.search_mode == "free":
        report.append(("search", f"free: exhaustiveness {adv.exhaustiveness}, mc_steps {adv.mc_steps}"))
    elif st.search_mode in PRESETS:
        exh_cuda, exh_cpu, mc = PRESETS[st.search_mode]
        report.append(("search", f"{st.search_mode}: exhaustiveness {exh_cuda} (CUDA) / {exh_cpu} (CPU), mc_steps {mc}"))
        overridden = [k for k in ("exhaustiveness", "mc_steps") if k in (raw.get("Advanced") or {})]
        if overridden:
            warnings.append(f"search_mode {st.search_mode} overrides Advanced {overridden}; "
                            "set search_mode: free to use your values")
    else:
        errors.append(f"search_mode '{st.search_mode}' is not one of fast, balance, detail, free")
    if st.task in ("score", "optimize"):
        notes.append(f"task {st.task}: the input pose is kept (no search); one pose per ligand")
    if adv.num_pose < 1:
        errors.append("num_pose must be at least 1")
    elif max(adv.num_pose, adv.rescore_pool) < 2:
        warnings.append("rescore_pool and num_pose are both 1: the pose pass has nothing to re-rank")
    report.append(("poses", f"{adv.num_pose} written per ligand, from {max(adv.num_pose, adv.rescore_pool)} re-ranked"
                            f"; seed {adv.seed}"))

    # plugins
    if template_on:
        reference = kwargs.get("reference_sdf_file_name")
        if reference is None:
            errors.append("template_docking is on but reference_sdf_file_name is not set")
        else:
            resolve_file(reference, "template reference", yaml_dir, errors)
    elif kwargs.get("reference_sdf_file_name") or kwargs.get("core_atom_mapping_dict_list"):
        warnings.append("template keys are set but template_docking is false: they are ignored")
    if covalent_on:
        if n_records and n_covalent < n_records:
            errors.append(f"covalent_ligand is on but {n_records - n_covalent} of {n_records} SDF records lack the "
                          f"covalent properties {list(COVALENT_PROPS)} (see ud2-covalent-docking)")
    elif kwargs.get("covalent_residue_atom_info_list"):
        warnings.append("covalent_residue_atom_info_list is set but covalent_ligand is false: it is ignored")
    on = [name for name, flag in (("template", template_on), ("covalent", covalent_on), ("bias", bias_on)) if flag]
    report.append(("plugins", ", ".join(on) if on else "none (free docking)"))

    # CNN stage
    if st.cnn_rescore or args.cnn_rescore:
        if on:
            warnings.append(f"CNN stage requested but skipped: {', '.join(on)} runs keep the engine order")
            report.append(("CNN stage", "skipped"))
        else:
            problem = cnn_device_check(hw.gpu_device_id)
            if problem and not os.environ.get("UD2_CNN_SOCKET"):
                errors.append(f"CNN stage cannot run: {problem}")
            report.append(("CNN stage", f"on, cuda:{hw.gpu_device_id}"))
    else:
        report.append(("CNN stage", "off"))

    # command-line-only stages
    if args.decomp:
        report.append(("decomp", "on: per-atom and per-residue terms in the output SDF"))
    if args.ff_minimize_steps > 0:
        report.append(("FF minim.", f"{args.ff_minimize_steps} steps after the search (serial on CUDA)"))
        if (receptor and receptor.endswith(".pdb") and shutil.which("fepfixer")
                and not shutil.which("utop") and not shutil.which("tleap")):
            errors.append("--ff_minimize_steps needs utop or tleap on PATH: the fepfixer route parameterises the "
                          "receptor with one of them")
        elif receptor and receptor.endswith(".dms"):
            from unidock_processing.core.dms import dms_has_ff_params
            if not dms_has_ff_params(receptor):
                warnings.append("the receptor DMS has no force-field parameters: the engine logs 'FF scoring "
                                "disabled' and the poses get no ff_* tags")

    # outputs and work directory
    temp_dir = os.path.abspath(pre.temp_dir_name)
    if not os.path.isdir(temp_dir):
        errors.append(f"temp_dir_name does not exist: {temp_dir} (unidock2 does not create it)")
    report.append(("work dir", f"{temp_dir}/docking_*  ({'deleted after the run' if temp_dir == '/tmp' else 'kept'})"))
    output = os.path.abspath(args.output or pre.output_docking_pose_sdf_file_name)
    if not os.path.isdir(os.path.dirname(output)):
        errors.append(f"output directory does not exist: {os.path.dirname(output)}")
    elif os.path.exists(output):
        warnings.append(f"output exists and will be overwritten: {output}")
    report.append(("output", output))
    report.append(("n_cpu", f"{hw.n_cpu or 'all'} ligand-preparation workers; the CPU engine uses every core"))

    from unidock_processing.core.engine import find_ud2_binary
    try:
        ud2 = find_ud2_binary()                  # the runner's own lookup: PATH, then in-tree build directories
    except TypeError:                            # its in-tree fallback fails on a pip-installed engine
        ud2 = None
    if ud2 is None:
        errors.append("ud2 not found (see the README: put site-packages/unidock_engine/api/python/bin on PATH)")
    else:
        version = subprocess.run([ud2, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
        report.append(("engine", f"{ud2}  {version[-1] if version else ''}"))

    logger.info("config: %s", os.path.abspath(args.config))
    for label, value in report:
        logger.info("  %-10s %s", label, value)
    for message in notes:
        logger.info("NOTE   %s", message)
    for message in warnings:
        logger.warning("WARN   %s", message)
    for message in errors:
        logger.error("ERROR  %s", message)
    logger.info("result: %d errors, %d warnings", len(errors), len(warnings))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
