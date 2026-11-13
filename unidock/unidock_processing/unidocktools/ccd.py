"""Chemical Component Dictionary lookup: bond orders, formal charges and hydrogen counts for any
residue the PDB knows about. The wwPDB file (components.cif.gz, ~125 MB) is indexed once into a
small sqlite database so a lookup costs nothing at run time.

    python -m unidock_processing.unidocktools.ccd components.cif.gz   # writes ~/.cache/unidock/ccd.sqlite

The database path is UD2_CCD_DB or ~/.cache/unidock/ccd.sqlite.
"""
import gzip
import json
import logging
import os
import re
import sqlite3
import sys

logger = logging.getLogger(__name__)

DEFAULT_DB = os.path.join(os.path.expanduser("~"), ".cache", "unidock", "ccd.sqlite")
_TOKEN = re.compile(r'"[^"]*"|\'[^\']*\'|\S+')
_ORDER = {"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4, "AROM": 1}


def db_path():
    return os.environ.get("UD2_CCD_DB", DEFAULT_DB)


def _tokens(line):
    return [t[1:-1] if t[0] in "\"'" and t[-1] == t[0] and len(t) > 1 else t for t in _TOKEN.findall(line)]


def _parse_block(lines):
    """One data_ block of the CCD -> (type, parent, atoms, bonds). Only the component header, the
    atom loop and the bond loop are read."""
    atoms, bonds = [], []
    header = {}
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith("_chem_comp."):
            parts = _tokens(lines[i])
            if len(parts) >= 2:
                header[parts[0].split(".")[1]] = parts[1]
        if lines[i].strip() != "loop_":
            i += 1
            continue
        cols = []
        i += 1
        while i < n and lines[i].startswith("_"):
            cols.append(lines[i].split()[0])
            i += 1
        rows = []
        while i < n and lines[i].strip() and not lines[i].startswith(("loop_", "_", "#")):
            rows.append(_tokens(lines[i]))
            i += 1
        if cols and cols[0].startswith("_chem_comp_atom."):
            ix = {c.split(".")[1]: k for k, c in enumerate(cols)}
            for r in rows:
                atoms.append([r[ix["atom_id"]], r[ix["type_symbol"]].capitalize(), int(r[ix["charge"]]) if r[ix["charge"]] not in ("?", ".") else 0,
                              r[ix.get("pdbx_leaving_atom_flag", ix["atom_id"])] == "Y"])
        elif cols and cols[0].startswith("_chem_comp_bond."):
            ix = {c.split(".")[1]: k for k, c in enumerate(cols)}
            for r in rows:
                bonds.append([r[ix["atom_id_1"]], r[ix["atom_id_2"]], _ORDER.get(r[ix["value_order"]], 1),
                              r[ix.get("pdbx_aromatic_flag", ix["atom_id_1"])] == "Y"])
    parent = header.get("mon_nstd_parent_comp_id", "?")
    return header.get("type", "?").upper(), "" if parent in ("?", ".") else parent.upper(), atoms, bonds


def _parse_single_atom(lines):
    """Single-atom components (ions) carry their atom as key/value pairs instead of a loop."""
    kv = {}
    for line in lines:
        if line.startswith("_chem_comp_atom."):
            parts = _tokens(line)
            if len(parts) >= 2:
                kv[parts[0].split(".")[1]] = parts[1]
    if "atom_id" not in kv:
        return []
    charge = kv.get("charge", "0")
    return [[kv["atom_id"], kv["type_symbol"].capitalize(), int(charge) if charge not in ("?", ".") else 0,
             kv.get("pdbx_leaving_atom_flag") == "Y"]]


def build_index(cif_gz, out=None):
    out = out or db_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    conn = sqlite3.connect(tmp)
    conn.execute("CREATE TABLE comp (id TEXT PRIMARY KEY, type TEXT, parent TEXT, atoms TEXT, bonds TEXT)")
    n = 0
    block_id, lines = None, []

    def flush():
        if block_id is None:
            return
        ctype, parent, atoms, bonds = _parse_block(lines)
        if not atoms:
            atoms = _parse_single_atom(lines)
        conn.execute("INSERT OR REPLACE INTO comp VALUES (?, ?, ?, ?, ?)",
                     (block_id, ctype, parent, json.dumps(atoms), json.dumps(bonds)))

    with gzip.open(cif_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("data_"):
                flush()
                n += 1
                block_id, lines = line[5:].strip(), []
            else:
                lines.append(line.rstrip("\n"))
        flush()
    conn.commit()
    conn.close()
    os.replace(tmp, out)
    logger.info(f"CCD index: {n} components -> {out}")
    return out


class CCD:
    """Read-only lookup over the index. lookup() returns (atoms, bonds) or None:
    atoms = [[name, element, formal_charge, leaving], ...], bonds = [[name1, name2, order, aromatic], ...]."""

    def __init__(self, path=None):
        self.path = path or db_path()
        self.conn = sqlite3.connect(self.path) if os.path.isfile(self.path) else None
        if self.conn is None:
            logger.warning(f"No CCD index at {self.path}; het residues fall back to geometric bond perception. "
                           "Build one with: python -m unidock_processing.unidocktools.ccd components.cif.gz")

    def lookup(self, resname):
        if self.conn is None:
            return None
        row = self.conn.execute("SELECT atoms, bonds FROM comp WHERE id = ?", (resname,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), json.loads(row[1])

    def polymer_parent(self, resname):
        """Standard parent of a modified polymer residue (TPO -> THR, PTR -> TYR, ...), or ''."""
        if self.conn is None:
            return ""
        row = self.conn.execute("SELECT type, parent FROM comp WHERE id = ?", (resname,)).fetchone()
        if row is None or "LINKING" not in row[0] or not row[1] or "," in row[1]:
            return ""
        return row[1]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_index(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
