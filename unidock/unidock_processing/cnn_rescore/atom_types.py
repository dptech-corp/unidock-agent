"""Atom typing for the pose CNN from the enriched DMS files (28-type gnina table), and the engine SDF record parser (no RDKit here)."""
import re
import sqlite3

# libmolgrid GninaIndexTyper order: 0 H 1 PolarH 2 AlC_H 3 AlC_P 4 ArC_H 5 ArC_P 6 N 7 N_D 8 N_DA 9 N_A 10 O 11 O_D
# 12 O_DA 13 O_A 14 S 15 S_acc 16 P 17 F 18 Cl 19 Br 20 I 21 Mg 22 Mn 23 Zn 24 Ca 25 Fe 26 GenericMetal 27 B
VINA_TO_GNINA = {1: 27, 4: 6, 5: 7, 6: 9, 7: 8, 8: 10, 9: 11, 10: 13, 11: 12, 12: 17, 13: 26, 14: 16, 15: 14,
                 16: 18, 17: 19, 18: 20, 19: 26}
METAL_BY_ANUM = {12: 21, 25: 22, 30: 23, 20: 24, 26: 25}
AROMATIC_RESIDUE_ATOMS = (
    {("PHE", a) for a in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")}
    | {("TYR", a) for a in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")}
    | {("TRP", a) for a in ("CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")}
    | {(h, a) for h in ("HIS", "HID", "HIE", "HIP") for a in ("CG", "CD2", "CE1")}
)
_PROP = re.compile(r"^>\s+<([^>]+)>\s*\n(.*)$", re.M)


def gnina_type(vina_type, anum, aromatic):
    """GninaIndexTyper index for an engine Vina type; -1 for hydrogens (not gridded)."""
    if vina_type == 0:
        return -1
    if vina_type == 2:
        return 4 if aromatic else 2
    if vina_type == 3:
        return 5 if aromatic else 3
    if vina_type == 20:
        return METAL_BY_ANUM.get(anum, 26)
    return VINA_TO_GNINA.get(vina_type, 26)


def _base_resname(name):
    return name[1:] if len(name) == 4 and name[0] in "NC" else name


def receptor_atoms(dms_path):
    """(x, y, z, type) per heavy receptor atom; aromatic carbons from the standard residue templates."""
    conn = sqlite3.connect(dms_path)
    rows = conn.execute(
        "SELECT p.anum, p.name, p.resname, p.x, p.y, p.z, a.vina_type FROM particle p "
        "JOIN ud2_atom_props a ON a.atom_id = p.id ORDER BY p.id").fetchall()
    conn.close()
    out = []
    for anum, name, resname, x, y, z, vina_type in rows:
        t = gnina_type(vina_type, anum, (_base_resname(resname), name) in AROMATIC_RESIDUE_ATOMS)
        if t >= 0:
            out.append((x, y, z, t))
    return out


def ligand_types(dms_path):
    """molecule name -> list of gnina types in particle order (single-ligand or merged DMS)."""
    conn = sqlite3.connect(dms_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "ud2_atom_aromatic" not in tables:
        raise RuntimeError(f"{dms_path}: no ud2_atom_aromatic table; the CNN typing needs the enriched ligand DMS")
    props = dict(conn.execute("SELECT atom_id, vina_type FROM ud2_atom_props"))
    aromatic = {r[0] for r in conn.execute("SELECT atom_id FROM ud2_atom_aromatic")}
    anum = dict(conn.execute("SELECT id, anum FROM particle"))
    out = {}
    if "ud2_molecules" in tables:
        names = dict(conn.execute("SELECT molecule_id, value FROM ud2_metadata WHERE key = 'original_name'")) \
            if "ud2_metadata" in tables and _has_column(conn, "ud2_metadata", "molecule_id") else {}
        for mid, mol_name, offset, natom in conn.execute("SELECT molecule_id, mol_name, atom_offset, natom FROM ud2_molecules"):
            out[names.get(mid, mol_name)] = [gnina_type(props[i], anum[i], i in aromatic) for i in range(offset, offset + natom)]
    else:
        meta = dict(conn.execute("SELECT key, value FROM ud2_metadata")) if "ud2_metadata" in tables else {}
        name = meta.get("original_name", meta.get("molecule_name", ""))
        out[name] = [gnina_type(props[i], anum[i], i in aromatic) for i in sorted(anum)]
    conn.close()
    return out


def _has_column(conn, table, column):
    return column in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def iter_records(sdf_path):
    """Yield (molecule name, properties dict, coordinates list, raw record text) per pose, in file order."""
    with open(sdf_path) as f:
        text = f.read()
    for block in text.split("$$$$\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        natom = int(lines[3][:3])
        xyz = [(float(a[0:10]), float(a[10:20]), float(a[20:30])) for a in lines[4:4 + natom]]
        props = dict(_PROP.findall(block))
        yield props.get("ud2_molecule_name", lines[0]).strip(), props, xyz, block + "$$$$\n"


def add_properties(record, items):
    """Append SD tags to a raw record (before the terminating $$$$)."""
    body = record[: -len("$$$$\n")]
    if not body.endswith("\n"):
        body += "\n"
    return body + "".join(f">  <{k}>\n{v}\n\n" for k, v in items) + "$$$$\n"
