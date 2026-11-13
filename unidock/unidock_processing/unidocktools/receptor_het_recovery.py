"""Keep the non-protein part of a receptor through the fepfixer + utop route.

fepfixer deletes every residue without an Amber template (hemes, FAD, NAD, SAM, metals, modified
residues, free amino acids, ...) and utop fails on the few it keeps but cannot parametrize. Both
are the wrong outcome for docking, where the pocket needs those atoms and the engine needs only
elements, bonds and hydrogens to type them. So the PDB is split first: ATOM records (plus water)
go through fepfixer, every other HETATM residue is rebuilt from the Chemical Component Dictionary
(bond orders, formal charges, hydrogens placed by RDKit) and appended to the parametrized DMS
without force-field parameters.
"""
import logging
import os

import msys
import numpy as np
from rdkit import Chem, RDLogger

from unidock_processing.unidocktools.ccd import CCD

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

WATER = {"HOH", "WAT", "DOD", "H2O", "TIP", "SOL"}
BACKBONE = {"N", "CA", "C"}
AMINO_ACIDS = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
                  "HID HIE HIP HSD HSE HSP CYX CYM ASH GLH LYN".split())
H_CLASH = 1.1  # a placed hydrogen closer than this to a foreign heavy atom sits on a covalent link

_ELEMENT_FIX = {"SE": "Se", "CL": "Cl", "BR": "Br", "FE": "Fe", "ZN": "Zn", "MG": "Mg", "MN": "Mn", "CA": "Ca",
                "NA": "Na", "CU": "Cu", "NI": "Ni", "CO": "Co", "CD": "Cd", "HG": "Hg", "SI": "Si", "AS": "As"}


def _element(line):
    el = line[76:78].strip()
    if not el:  # no element column: first letter of the atom name
        el = line[12:16].strip().lstrip("0123456789")[:1]
    return _ELEMENT_FIX.get(el.upper(), el.capitalize())


STANDARD_POLYMER = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
                       "A C G U DA DC DG DT".split())
_RENAME = {("MSE", "SE"): "SD"}  # modified-residue atoms whose parent name differs
_MET_ATOMS = {"N": "N", "CA": "C", "C": "C", "O": "O", "OXT": "O", "CB": "C", "CG": "C", "SD": "S", "CE": "C"}


def split_receptor_pdb(inp, protein_out, hets_out, ccd):
    """ATOM records and water -> protein_out; every other HETATM residue -> hets_out.
    A modified polymer residue (TPO, PTR, MSE, KCX, ...) stays in the chain as its standard parent so
    fepfixer sees no break: the atoms the parent has go to protein_out under the parent's name, the
    modification atoms go to hets_out under the original name and come back as a bonded fragment.
    ANISOU records are dropped, as are residues with an incomplete backbone. Returns counts."""
    lines = [l for l in open(inp) if l.startswith(("ATOM", "HETATM", "TER"))]
    backbone = {}
    for l in lines:
        if l.startswith("ATOM") and l[17:20].strip() in AMINO_ACIDS:
            backbone.setdefault((l[21], l[22:27]), set()).add(l[12:16].strip())
    stubs = {k for k, names in backbone.items() if not BACKBONE <= names}
    parent_atoms = {}  # resname -> (parent, {parent atom name: element})
    n_conv = n_het = n_stub = 0
    with open(protein_out, "w") as fp, open(hets_out, "w") as fh:
        for l in lines:
            if l.startswith("ATOM") and (l[21], l[22:27]) in stubs:
                n_stub += 1
                continue
            if l.startswith("HETATM") and l[17:20].strip() not in WATER:
                resname = l[17:20].strip()
                if resname not in parent_atoms:
                    parent = ccd.polymer_parent(resname)
                    template = ccd.lookup(parent) if parent in STANDARD_POLYMER else None
                    names = {a[0]: a[1] for a in template[0]} if template else {}
                    if resname == "MSE" and not names:  # no CCD index: still keep selenomethionine in the chain
                        parent, names = "MET", _MET_ATOMS
                    parent_atoms[resname] = (parent, names)
                parent, names = parent_atoms[resname]
                orig = l[12:16].strip()
                name = _RENAME.get((resname, orig), orig)
                if name in names:
                    field = l[12:16] if name == orig else f" {name:<3}"
                    l = "ATOM  " + l[6:12] + field + l[16:17] + f"{parent:>3}" + l[20:76] + f"{names[name]:>2}" + l[78:]
                    n_conv += 1
                else:
                    fh.write(l)
                    n_het += 1
                    continue
            fp.write(l)
        fp.write("END\n")
    if stubs:
        logger.warning(f"Dropped {len(stubs)} residues with an incomplete backbone: "
                       + ", ".join(f"{k[0]}:{k[1].strip()}" for k in sorted(stubs)))
    return dict(converted_atoms=n_conv, het_atoms=n_het, stub_atoms=n_stub)


def read_residues(pdb):
    """(chain, resid, icode, resname) -> [(name, element, x, y, z), ...]; first altloc per atom name."""
    residues = {}
    for l in open(pdb):
        if not l.startswith(("ATOM", "HETATM")):
            continue
        key = (l[21], int(l[22:26]), l[26].strip(), l[17:20].strip())
        atoms = residues.setdefault(key, {})
        name = l[12:16].strip()
        if name in atoms:
            continue
        atoms[name] = (name, _element(l), float(l[30:38]), float(l[38:46]), float(l[46:54]))
    return {k: list(v.values()) for k, v in residues.items()}


def residues_in_dms(dms_path):
    system = msys.LoadDMS(dms_path)
    return {(r.chain.name, r.resid, r.insertion, r.name) for r in system.residues}


def _deprotonate_acids(mol):
    """Phosphate, sulfate/sulfonate and carboxylic acid O-H are anionic at pH 7.4; the CCD stores
    many cofactors as the neutral acid. Amines are left as the template has them."""
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 8 or a.GetFormalCharge() != 0 or a.GetDegree() != 1:
            continue
        b = a.GetBonds()[0]
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        x = b.GetOtherAtom(a)
        o_neighbours = [n for n in x.GetNeighbors() if n.GetAtomicNum() == 8]
        acid = (x.GetAtomicNum() == 15
                or (x.GetAtomicNum() == 16 and len(o_neighbours) >= 3)
                or (x.GetAtomicNum() == 6 and any(mol.GetBondBetweenAtoms(x.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                                                  for n in o_neighbours if n.GetIdx() != a.GetIdx())))
        if acid:
            a.SetFormalCharge(-1)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)


def build_het_mol(atoms, resname, ccd, fragment=False):
    """RDKit mol with hydrogens for one het residue. Bond orders and formal charges come from the
    CCD template; without one the bonds are perceived from distances and no hydrogens are added.
    fragment=True (modification atoms of a residue that stayed in the chain as its parent) gets
    bonds and charges but no hydrogens."""
    template = ccd.lookup(resname)
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, (name, el, x, y, z) in enumerate(atoms):
        a = Chem.Atom(el)
        a.SetProp("name", name)
        mol.AddAtom(a)
        conf.SetAtomPosition(i, (x, y, z))
    mol.AddConformer(conf, assignId=True)
    index = {name: i for i, (name, *_ ) in enumerate(atoms)}
    templated = False
    if template is not None:
        t_atoms, t_bonds = template
        t_index = {a[0]: a for a in t_atoms}
        matched = sum(1 for n in index if n in t_index)
        if matched >= 0.8 * len(index):
            templated = True
            for n, i in index.items():
                if n in t_index:
                    mol.GetAtomWithIdx(i).SetFormalCharge(int(t_index[n][2]))
            for n1, n2, order, _ in t_bonds:
                if n1 in index and n2 in index:
                    mol.AddBond(index[n1], index[n2], Chem.BondType.values[order])
    if not templated:
        from rdkit.Chem import rdDetermineBonds
        rdDetermineBonds.DetermineConnectivity(mol)
        for a in mol.GetAtoms():
            a.SetNoImplicit(True)
        logger.warning(f"{resname}: no CCD template ({'no index' if template is None else 'atom names do not match'}); "
                       "bonds perceived from geometry, no hydrogens added")
        m = mol.GetMol()
        m.UpdatePropertyCache(strict=False)
        return m, False
    m = mol.GetMol()
    if fragment:
        for a in m.GetAtoms():
            a.SetNoImplicit(True)
        m.UpdatePropertyCache(strict=False)
        return m, True
    try:
        Chem.SanitizeMol(m)
    except Exception as e:  # noqa: BLE001 - exotic valence (metal complexes); keep heavy atoms as they are
        logger.warning(f"{resname}: sanitization failed ({str(e)[:80]}); no hydrogens added")
        for a in m.GetAtoms():
            a.SetNoImplicit(True)
        m.UpdatePropertyCache(strict=False)
        return m, True
    _deprotonate_acids(m)
    return Chem.AddHs(m, addCoords=True), True


def _prune_link_hydrogens(mol, foreign_xyz):
    """Drop hydrogens RDKit placed into a covalent link (glycan on Asn, modified residue in the chain)."""
    if mol.GetNumAtoms() == mol.GetNumHeavyAtoms() or len(foreign_xyz) == 0:
        return mol
    pos = mol.GetConformer().GetPositions()
    drop = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 1:
            continue
        d = np.linalg.norm(foreign_xyz - pos[a.GetIdx()], axis=1)
        if d.min() < H_CLASH:
            drop.append(a.GetIdx())
    if not drop:
        return mol
    rw = Chem.RWMol(mol)
    for i in sorted(drop, reverse=True):
        rw.RemoveAtom(i)
    return rw.GetMol()


def _bond_fragment(parent, fragment_atoms):
    """Join modification atoms to the parent residue's atom they sit on (P to OG1, C to NZ, ...) and
    drop the hydrogens fepfixer put on that atom in place of the substituent."""
    parent_heavy = [a for a in parent.atoms if a.atomic_number > 1 and a not in fragment_atoms]
    for f in fragment_atoms:
        if f.atomic_number == 1:
            continue
        for p in parent_heavy:
            if np.linalg.norm(np.array(f.pos) - np.array(p.pos)) < 1.9:
                f.addBond(p).order = 1
                for h in [n for n in p.bonded_atoms if n.atomic_number == 1]:
                    if p.atomic_number in (8, 16) or np.linalg.norm(np.array(h.pos) - np.array(f.pos)) < 1.3:
                        h.remove()


def append_to_dms(dms_path, hets):
    """hets: [(key, mol), ...] with key = (chain, resid, icode, resname). Rewrites dms_path in place
    (it is the route's own intermediate, never the user's input)."""
    system = msys.LoadDMS(dms_path)
    nb = system.table("nonbonded") if "nonbonded" in system.table_names else None
    zero = None
    if nb is not None:
        zero = nb.params.addParam()
        for prop in nb.params.props:
            try:
                zero[prop] = 0.0
            except Exception:  # noqa: BLE001 - string-valued property (a type label)
                zero[prop] = "UD2_HET"
    chains = {c.name: c for c in system.chains}
    heavy_xyz = np.array([a.pos for a in system.atoms if a.atomic_number > 1]) if system.natoms else np.zeros((0, 3))
    n_atoms = n_h = 0
    residues = {(r.chain.name, r.resid, r.insertion): r for r in system.residues}
    for (chain, resid, icode, resname), mol in hets:
        mol = _prune_link_hydrogens(mol, heavy_xyz)
        parent = residues.get((chain, resid, icode))
        if chain not in chains:
            chains[chain] = system.addCt().addChain()
            chains[chain].name = chain
        res = chains[chain].addResidue() if parent is None else parent
        if parent is None:
            res.name, res.resid, res.insertion = resname, resid, icode
        pos = mol.GetConformer().GetPositions()
        atoms = []
        h_count = 0
        for a in mol.GetAtoms():
            at = res.addAtom()
            at.atomic_number = a.GetAtomicNum()
            if a.HasProp("name"):
                at.name = a.GetProp("name")
            else:
                h_count += 1
                at.name = f"H{h_count}"
            at.x, at.y, at.z = (float(v) for v in pos[a.GetIdx()])
            at.formal_charge = a.GetFormalCharge()
            at.charge = 0.0
            if nb is not None:
                nb.addTerm([at], zero)
            atoms.append(at)
        for b in mol.GetBonds():
            atoms[b.GetBeginAtomIdx()].addBond(atoms[b.GetEndAtomIdx()]).order = int(b.GetBondTypeAsDouble())
        if parent is not None:
            _bond_fragment(parent, atoms)
        n_atoms += mol.GetNumHeavyAtoms()
        n_h += mol.GetNumAtoms() - mol.GetNumHeavyAtoms()
        heavy_xyz = np.vstack([heavy_xyz, pos[[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]]])
    # clone compacts the atom table: msys cannot export a system with removed atoms in place.
    # Write to a sibling and swap so a failure leaves the utop output intact.
    tmp = dms_path + ".tmp"
    msys.SaveDMS(system.clone("all"), tmp)
    os.replace(tmp, dms_path)
    return n_atoms, n_h


def recover_hets(hets_pdb, dms_path, ccd):
    """Append every residue of hets_pdb to dms_path: whole residues with hydrogens, or bonded
    fragments when the residue is already there under its parent name. Returns the appended keys."""
    if not os.path.isfile(hets_pdb) or os.path.getsize(hets_pdb) == 0:
        return []
    present = {k[:3]: k[3] for k in residues_in_dms(dms_path)}
    hets = []
    for key, atoms in read_residues(hets_pdb).items():
        if present.get(key[:3]) == key[3]:
            continue
        fragment = key[:3] in present
        mol, templated = build_het_mol(atoms, key[3], ccd, fragment=fragment)
        hets.append((key, mol))
    if not hets:
        return []
    n_heavy, n_h = append_to_dms(dms_path, hets)
    summary = {}
    for (chain, resid, icode, resname), _ in hets:
        summary[resname] = summary.get(resname, 0) + 1
    logger.info(f"Recovered {len(hets)} het residues ({n_heavy} heavy atoms, {n_h} hydrogens) into the receptor DMS: "
                + ", ".join(f"{k} x{v}" if v > 1 else k for k, v in sorted(summary.items())))
    return [h[0] for h in hets]
