from typing import Optional
import os
import math
import logging
import argparse

logger = logging.getLogger(__name__)

from rdkit import Chem

from unidock_processing.ligand_topology import utils


# Scale factor applied to IMPROPER dihedral force constants only. GAFF impropers
# are shallow near planar (~0.2 kcal/mol at 6 deg out-of-plane), so full-Cartesian
# minimization against a rigid receptor lets sp2 centers (amides, aromatics)
# pyramidalize several degrees. Stiffening the impropers holds planarity.
IMPROPER_K_SCALE = 10.0


def _add_harmonic_tables(sys_mol, parm) -> None:
    """Add stretch_harm and angle_harm TermTables to an msys System.

    Bond/angle indices index parm.atoms[i].idx == sys_mol.atom(i).id, so the
    harmonic terms share the same atom ordering as all other FF tables built
    on this system. AMBER E = k*(x-x0)^2 (k absorbs 1/2) maps directly to the
    engine ff_bond/ff_angle E = fc*(x-x0)^2, so fc = k with no factor. theta0
    is stored in degrees; the engine converts to radians on read.
    """
    sh = sys_mol.addTableFromSchema('stretch_harm')
    sh_map = {}
    for b in parm.bonds:
        r0 = b.type.req
        fc = b.type.k
        key = (round(r0, 6), round(fc, 6))
        if key not in sh_map:
            sh_map[key] = sh.params.addParam(r0=r0, fc=fc)
        sh.addTerm(
            [sys_mol.atom(b.atom1.idx), sys_mol.atom(b.atom2.idx)],
            sh_map[key],
        )

    ah = sys_mol.addTableFromSchema('angle_harm')
    ah_map = {}
    for a in parm.angles:
        theta0 = a.type.theteq
        fc = a.type.k
        key = (round(theta0, 6), round(fc, 6))
        if key not in ah_map:
            ah_map[key] = ah.params.addParam(theta0=theta0, fc=fc)
        ah.addTerm(
            [sys_mol.atom(a.atom1.idx),
             sys_mol.atom(a.atom2.idx),
             sys_mol.atom(a.atom3.idx)],
            ah_map[key],
        )


def parameterize_ligand_amber(
    input_sdf: str,
    output_dms: str,
    charge_method: str = 'gas',
    net_charge: Optional[int] = None,
    work_dir: Optional[str] = None,
) -> str:
    """Build a ligand DMS with GAFF2 FF parameters INCLUDING harmonic
    stretch_harm/angle_harm tables (drop-in replacement for the current
    ligand DMS, just WITH bond/angle terms for Cartesian FF minimization).

    Pipeline: SDF -> antechamber(gaff2, charge_method) -> parmchk2 -> tleap
              -> prmtop/inpcrd -> msys.LoadPrmTop + parmed.load_file
              -> nonbonded/exclusion/pair_12_6_es/dihedral_trig (existing)
              + stretch_harm/angle_harm (NEW) -> msys.SaveDMS
              -> _enrich_ligand_dms (ud2_atom_props vina types, bond orders,
                 formal charges, SDF tags, torsions, root atoms).

    Preserves the docked-pose 3D coordinates (restored from inpcrd/parmed,
    which keeps SDF atom order). net_charge auto-detected via
    Chem.GetFormalCharge if None. Returns output_dms on success; raises
    RuntimeError on any toolchain failure (caller decides fallback).
    """
    if not os.path.isfile(input_sdf):
        raise RuntimeError(f"Input SDF not found: {input_sdf}")

    try:
        import parmed
        import msys
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"Missing dependency for AMBER parameterization: {e}")

    if work_dir is None:
        work_dir = os.path.dirname(os.path.abspath(output_dms)) or '.'
    os.makedirs(work_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(input_sdf))[0]
    dms_dir = os.path.join(work_dir, f'_amber_build_{basename}')
    os.makedirs(dms_dir, exist_ok=True)

    # --- 1. Read SDF, determine net charge ---
    mol = Chem.SDMolSupplier(input_sdf, removeHs=False)[0]
    if mol is None:
        raise RuntimeError(f"Cannot read SDF: {input_sdf}")
    num_atoms = mol.GetNumAtoms()
    if net_charge is None:
        net_charge = Chem.GetFormalCharge(mol)

    temp_sdf = os.path.join(dms_dir, 'lig.sdf')
    utils.convert_v3000_mol_to_v2000_sdf(mol, temp_sdf)

    # --- 2. antechamber (gaff2 + charge_method) -> parmchk2 -> tleap ---
    mol2 = os.path.join(dms_dir, 'lig.mol2')
    frcmod = os.path.join(dms_dir, 'lig.frcmod')
    prmtop = os.path.join(dms_dir, 'lig.prmtop')
    inpcrd = os.path.join(dms_dir, 'lig.inpcrd')

    ante_cmd = (
        f'cd {dms_dir} && '
        f'antechamber -i {temp_sdf} -fi sdf -o {mol2} -fo mol2 '
        f'-at gaff2 -c {charge_method} -nc {net_charge} -eq 2 -pf y -dr no '
        f'> ante.log 2>&1'
    )
    os.system(ante_cmd)
    if not os.path.isfile(mol2):
        raise RuntimeError(
            f"antechamber failed for {input_sdf} (see {dms_dir}/ante.log)")

    parmchk_cmd = (
        f'cd {dms_dir} && '
        f'parmchk2 -i {mol2} -f mol2 -a Y -s 2 -o {frcmod}'
    )
    os.system(parmchk_cmd)
    if not os.path.isfile(frcmod):
        raise RuntimeError(f"parmchk2 failed for {input_sdf}")

    tleap_in = os.path.join(dms_dir, 'tleap.in')
    with open(tleap_in, 'w') as f:
        f.write(
            f'source leaprc.gaff2\n'
            f'MOL = loadmol2 {mol2}\n'
            f'loadamberparams {frcmod}\n'
            f'saveamberparm MOL {prmtop} {inpcrd}\n'
            f'quit\n'
        )
    os.system(f'cd {dms_dir} && tleap -f {tleap_in} > tleap.log 2>&1')
    if not os.path.isfile(prmtop) or not os.path.isfile(inpcrd):
        raise RuntimeError(
            f"tleap failed for {input_sdf} (see {dms_dir}/tleap.log)")

    # --- 3. Build msys system + parmed for FF parameters ---
    try:
        sys_mol = msys.LoadPrmTop(prmtop, inpcrd)
    except Exception as e:
        raise RuntimeError(f"msys.LoadPrmTop failed: {e}")

    try:
        parm = parmed.load_file(prmtop, inpcrd)
    except Exception as e:
        raise RuntimeError(f"parmed.load_file failed: {e}")

    from copy import deepcopy
    from collections import defaultdict

    # 3a. Restore docked pose (LoadPrmTop discards inpcrd coords)
    positions = np.array([[a.xx, a.xy, a.xz] for a in parm.atoms])
    sys_mol.setPositions(positions)

    # 3b. Copy Kekulized bond orders from SDF via 0.001 A coordinate matching
    kek_mol = deepcopy(mol)
    try:
        Chem.Kekulize(kek_mol, clearAromaticFlags=False)
    except Exception:
        kek_mol = mol

    conf = kek_mol.GetConformer()
    rdkit_pos_to_idx = {}
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        rdkit_pos_to_idx[
            (round(pos.x, 3), round(pos.y, 3), round(pos.z, 3))
        ] = i

    parm_to_rdkit = {}
    for a in parm.atoms:
        key = (round(a.xx, 3), round(a.xy, 3), round(a.xz, 3))
        if key in rdkit_pos_to_idx:
            parm_to_rdkit[a.idx] = rdkit_pos_to_idx[key]

    if len(parm_to_rdkit) == num_atoms:
        for bond in sys_mol.bonds:
            p0, p1 = bond.first.id, bond.second.id
            ri = parm_to_rdkit.get(p0)
            rj = parm_to_rdkit.get(p1)
            if ri is not None and rj is not None:
                rb = kek_mol.GetBondBetweenAtoms(ri, rj)
                if rb is not None:
                    bond.order = int(rb.GetBondTypeAsDouble())
        logger.info("Bond orders copied from SDF into DMS for %s", basename)
    else:
        logger.warning(
            "Atom coordinate matching incomplete (%d/%d); "
            "bond orders not updated for %s",
            len(parm_to_rdkit), num_atoms, basename,
        )

    # 3c. Charges from parmed
    for i, a in enumerate(parm.atoms):
        sys_mol.atom(i).charge = a.charge

    # 3d. nonbonded TermTable (1-atom: sigma, epsilon, type)
    nb = sys_mol.addTable('nonbonded', 1)
    nb.category = 'nonbonded'
    nb.params.addProp('sigma', float)
    nb.params.addProp('epsilon', float)
    nb.params.addProp('type', str)
    type_to_param = {}
    for i, a in enumerate(parm.atoms):
        if a.type not in type_to_param:
            type_to_param[a.type] = nb.params.addParam(
                sigma=a.sigma, epsilon=a.epsilon, type=a.type)
        nb.addTerm([sys_mol.atom(i)], type_to_param[a.type])

    # 3e. exclusion TermTable (1-2 + 1-3 + 1-4 pairs)
    excl = sys_mol.addTableFromSchema('exclusion')
    all_excl = set()
    for a in parm.atoms:
        for p in a.bond_partners:
            all_excl.add((min(a.idx, p.idx), max(a.idx, p.idx)))
        for p in a.angle_partners:
            all_excl.add((min(a.idx, p.idx), max(a.idx, p.idx)))
        for p in a.dihedral_partners:
            all_excl.add((min(a.idx, p.idx), max(a.idx, p.idx)))
    for i, j in sorted(all_excl):
        excl.addTerm([sys_mol.atom(i), sys_mol.atom(j)])

    # 3f. pair_12_6_es TermTable (1-4 pairs)
    pair = sys_mol.addTableFromSchema('pair_12_6_es')
    pairs_14 = set()
    for a in parm.atoms:
        for p in a.dihedral_partners:
            pairs_14.add((min(a.idx, p.idx), max(a.idx, p.idx)))
    SCNB = 2.0
    SCEE = 1.2
    pp_map = {}
    for i, j in sorted(pairs_14):
        ai, aj = parm.atoms[i], parm.atoms[j]
        sig_ij = (ai.sigma + aj.sigma) / 2.0
        eps_ij = math.sqrt(ai.epsilon * aj.epsilon)
        sig6 = sig_ij ** 6
        sig12 = sig6 * sig6
        a_ij = 4.0 * eps_ij * sig12 / SCNB
        b_ij = 4.0 * eps_ij * sig6 / SCNB
        q_ij = ai.charge * aj.charge / SCEE
        key = (round(a_ij, 6), round(b_ij, 6), round(q_ij, 6))
        if key not in pp_map:
            pp_map[key] = pair.params.addParam(aij=a_ij, bij=b_ij, qij=q_ij)
        pair.addTerm([sys_mol.atom(i), sys_mol.atom(j)], pp_map[key])

    # 3g. dihedral_trig TermTable (merged periodicities)
    dihe = sys_mol.addTableFromSchema('dihedral_trig')
    dihedral_groups = defaultdict(list)
    for d in parm.dihedrals:
        dihedral_groups[
            (d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx)
        ].append(d)
    dp_map = {}
    for quad, terms in dihedral_groups.items():
        fc = [0.0] * 7
        for d in terms:
            n = int(d.type.per)
            phi_k = d.type.phi_k
            if getattr(d, 'improper', False):
                phi_k *= IMPROPER_K_SCALE  # stiffen planarity terms only
            phase_rad = d.type.phase * math.pi / 180.0
            fc[0] += phi_k
            if 0 <= n <= 6:
                fc[n] += phi_k * math.cos(phase_rad)
        key = tuple(round(v, 6) for v in fc)
        if key not in dp_map:
            dp_map[key] = dihe.params.addParam(
                phi0=0.0, fc0=fc[0], fc1=fc[1], fc2=fc[2],
                fc3=fc[3], fc4=fc[4], fc5=fc[5], fc6=fc[6])
        dihe.addTerm([sys_mol.atom(q) for q in quad], dp_map[key])

    # 3h. NEW: harmonic stretch_harm + angle_harm tables
    _add_harmonic_tables(sys_mol, parm)

    # --- 4. SaveDMS with all topology + FF data ---
    msys.SaveDMS(sys_mol, output_dms)

    # --- 5. Graft ud2 tables (vina types, bond orders, formal charges, SDF
    #        tags, torsions, root atoms) via the existing enrichment path ---
    from unidock_processing.unidocktools.unidock_protocol_runner import (
        _dms_has_ff_params,
        _enrich_ligand_dms,
    )
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        build_molecular_graph,
        load_torsion_library,
    )

    if not _dms_has_ff_params(output_dms):
        raise RuntimeError(f"Built DMS missing FF tables: {output_dms}")

    torsion_dict = load_torsion_library()

    atom_info, torsion_info, root_atoms, _frag = build_molecular_graph(
        mol, torsion_dict, has_ff=True, dms_path=output_dms,
    )

    mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'MOL_0'
    _enrich_ligand_dms(
        output_dms, atom_info, torsion_info, root_atoms,
        molecule_name=mol_name, rdkit_mol=mol,
    )

    logger.info(
        "Built AMBER ligand DMS with harmonic tables: %s (%d atoms)",
        output_dms, num_atoms,
    )
    return output_dms


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Parameterize a ligand SDF with AMBER/GAFF2 and write a DMS '
            'including stretch_harm/angle_harm harmonic tables.'
        )
    )
    parser.add_argument('--sdf', required=True, help='Input ligand SDF path')
    parser.add_argument('--out', required=True, help='Output DMS path')
    parser.add_argument(
        '--charge-method', default='gas',
        help='antechamber charge method: gas (Gasteiger, fast, default) '
             'or bcc (AM1-BCC, slower, runs sqm QM)',
    )
    parser.add_argument(
        '--net-charge', type=int, default=None,
        help='Net charge (default: auto-detect via RDKit formal charge)',
    )
    parser.add_argument(
        '--work-dir', default=None,
        help='Working directory (default: output DMS directory)',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    out = parameterize_ligand_amber(
        args.sdf, args.out,
        charge_method=args.charge_method,
        net_charge=args.net_charge,
        work_dir=args.work_dir,
    )
    logger.info("Wrote: %s", out)


if __name__ == '__main__':
    main()
