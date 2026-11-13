#!/usr/bin/env python
"""Measure docked or aligned poses against a template ligand.

    python check_alignment.py --template template.sdf --poses out.sdf [--max-mcs 2.5] [--max-polar 2.0]

For each pose: the MCS RMSD to the template (RDKit FMCS over heavy atoms, complete rings only, elements compared,
any bond order; in place, no superposition) and the polar match (mean distance from each template hydrogen-bond
acceptor, donor and formally charged atom to the nearest query atom of the same kind, from ud2align's AtomProperties).
Exit status 1 when any pose is outside the limits.
"""
import argparse
import logging
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

from ud2align.properties import AtomProperties

logger = logging.getLogger("check_alignment")
RDLogger.DisableLog("rdApp.*")


def mcs_rmsd(mol, tmpl):
    a, b = Chem.RemoveHs(mol), Chem.RemoveHs(tmpl)
    res = rdFMCS.FindMCS([a, b], ringMatchesRingOnly=True, completeRingsOnly=True,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareAny, timeout=20)
    if res.numAtoms == 0:
        return float("nan"), 0
    patt = Chem.MolFromSmarts(res.smartsString)
    ma, mb = a.GetSubstructMatch(patt), b.GetSubstructMatch(patt)
    pa = a.GetConformer().GetPositions()[list(ma)]
    pb = b.GetConformer().GetPositions()[list(mb)]
    return float(np.sqrt(((pa - pb) ** 2).sum(1).mean())), res.numAtoms


def polar_match(mol, tmpl):
    pq, pt = AtomProperties(mol), AtomProperties(tmpl)
    xq, xt = mol.GetConformer().GetPositions(), tmpl.GetConformer().GetPositions()
    dists = []
    for q_idx, t_idx in ((pq.hbond_acceptors, pt.hbond_acceptors), (pq.hbond_donors, pt.hbond_donors)):
        if q_idx:
            dists += [min(np.linalg.norm(xq[i] - xt[j]) for i in q_idx) for j in t_idx]
    for j, charge in enumerate(pt.formal_charges):
        same = [i for i, c in enumerate(pq.formal_charges) if charge and c == charge]
        if same:
            dists.append(min(np.linalg.norm(xq[i] - xt[j]) for i in same))
    return float(np.mean(dists)) if dists else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--max-mcs", type=float, default=2.5)
    ap.add_argument("--max-polar", type=float, default=2.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tmpl = Chem.SDMolSupplier(args.template, removeHs=False)[0]
    if tmpl is None:
        logger.error("cannot read the template %s", args.template)
        return 2
    failed = 0
    for k, mol in enumerate(Chem.SDMolSupplier(args.poses, removeHs=False), 1):
        if mol is None:
            logger.warning("pose %d: unreadable, skipped", k)
            continue
        rmsd, n_mcs = mcs_rmsd(mol, tmpl)
        polar = polar_match(mol, tmpl)
        ok = rmsd < args.max_mcs and (np.isnan(polar) or polar < args.max_polar)
        failed += not ok
        logger.info("pose %3d  MCS RMSD %6.2f A over %2d atoms  polar match %6.2f A  %s",
                    k, rmsd, n_mcs, polar, "ok" if ok else "OUTSIDE LIMITS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
