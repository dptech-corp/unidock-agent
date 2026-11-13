"""Rescore an engine output SDF with the gnina pose-CNN ensemble and reorder each ligand's poses by the blend.

    python rescore.py --sdf unidock2_pose.sdf --ligand-dms ligand_enriched.dms --receptor-dms receptor_enriched.dms \
        --out unidock2_pose.sdf [--weight 1.5] [--batch 64] [--precision fp32|fp16] [--keep K] [--gridder auto|torch|molgrid]
    python rescore.py --serve [SOCKET]                         # resident server: models loaded once, ~10 ms per ligand

Per ligand, with z-scores taken over that ligand's poses:
    blend = -z(vina_pose_score) + weight * z(cnn_pose_score),   cnn_pose_score = mean over the 5 models of P(RMSD < 2 A)
Poses are written in decreasing blend order with the tags ud2_cnn_pose_score, ud2_cnn_blend_score and ud2_cnn_rank added;
every other tag, including ud2_pose_index, is kept. --keep K writes only the first K poses of each ligand after the reorder
(ligands the grid skips keep the engine order and are cut the same way). Needs torch (CUDA); libmolgrid, when it imports,
grids 10x faster than the torch fallback (same grids). Runs outside the main process by file path.
"""
import argparse
import logging
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):                       # run by file path: keep the package __init__ out of this process
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from atom_types import add_properties, iter_records, ligand_types, receptor_atoms
    import grid as _grid
    import models as _models
else:
    from unidock_processing.cnn_rescore.atom_types import add_properties, iter_records, ligand_types, receptor_atoms
    from unidock_processing.cnn_rescore import grid as _grid
    from unidock_processing.cnn_rescore import models as _models

logger = logging.getLogger("unidock_processing.cnn_rescore")
GRID_HALF_EDGE = 23.5 / 2.0          # molgrid dimension of the published models / 2


def zscore(values):
    v = np.asarray(values, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def fits_grid(xyz, types, half_edge=GRID_HALF_EDGE):
    """True when every heavy atom of the pose lies inside the cube the network sees (centred on the heavy-atom centroid)."""
    heavy = np.asarray([c for c, t in zip(xyz, types) if t >= 0], dtype=np.float64)
    if len(heavy) == 0:
        return False
    return bool(np.all(np.abs(heavy - heavy.mean(axis=0)) <= half_edge))


def blend_order(pose_scores, cnn_scores, weight):
    """indices of the poses in decreasing blend order (stable for ties)"""
    blend = -zscore(pose_scores) + weight * zscore(cnn_scores)
    return list(np.argsort(-blend, kind="stable")), blend


class Scorer:
    """The five networks loaded once; grids and scores poses. precision: fp32 (reference) or fp16 (autocast, 3x faster)."""

    def __init__(self, device="cuda", weights_dir=None, precision="fp16", gridder="auto"):
        import torch
        self.torch = torch
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device(device)
        self.precision = precision if self.device.type == "cuda" else "fp32"     # autocast fp16 is a CUDA path
        self.gridder, self.gridder_name = _grid.make_gridder(self.device, gridder)
        logger.info("CNN gridder: %s", self.gridder_name)
        self.models = _models.load_ensemble(self.device, weights_dir) if weights_dir else _models.load_ensemble(self.device)
        shape = (_models.GRID_CHANNELS, _models.GRID_POINTS, _models.GRID_POINTS, _models.GRID_POINTS)
        with torch.no_grad(), self._autocast():
            for m in self.models:                                   # warm-up: cuDNN algorithm selection
                m(torch.zeros((1,) + shape, dtype=torch.float32, device=self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _autocast(self):
        return self.torch.autocast("cuda", dtype=self.torch.float16, enabled=(self.precision == "fp16"))

    def score(self, rec_xyz, rec_types, poses, batch=64):
        """poses: list of (lig_xyz, lig_types); returns the mean over models of P(RMSD < 2 A) per pose."""
        torch = self.torch
        out = np.zeros((len(poses), len(self.models)), dtype=np.float32)
        with torch.no_grad(), self._autocast():
            for start in range(0, len(poses), batch):
                grids = self.gridder(rec_xyz, rec_types, poses[start:start + batch])
                for k, model in enumerate(self.models):
                    logits, _ = model(grids)
                    out[start:start + len(grids), k] = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
        return out.mean(axis=1)


def run(sdf_path, ligand_dms, receptor_dms, out_path, scorer, weight=1.5, batch=64, keep=None):
    if keep is not None and keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    t0 = time.time()
    types = ligand_types(ligand_dms)
    poses = {}                                   # molecule name -> list of (props, xyz, record), first-appearance order
    for name, props, xyz, record in iter_records(sdf_path):
        if name not in types:
            raise RuntimeError(f"pose of {name!r} has no ligand in {ligand_dms}")
        if len(xyz) != len(types[name]):
            raise RuntimeError(f"{name}: {len(xyz)} atoms in the SDF, {len(types[name])} in the DMS")
        poses.setdefault(name, []).append((props, xyz, record))
    if not poses:
        raise RuntimeError(f"no poses in {sdf_path}")
    # ligands the grid cannot hold keep the engine order and get no CNN tags
    skipped = {name for name, plist in poses.items() if not all(fits_grid(xyz, types[name]) for _, xyz, _ in plist)}
    scored = {name: plist for name, plist in poses.items() if name not in skipped}
    n_poses = sum(len(v) for v in scored.values())
    for name in sorted(skipped):
        logger.warning("CNN rescoring skipped %s: a pose extends beyond the %.1f A grid half-edge", name, GRID_HALF_EDGE)
    rec = receptor_atoms(receptor_dms)
    rec_xyz = [(x, y, z) for x, y, z, _ in rec]; rec_types = [t for _, _, _, t in rec]
    pose_atoms = [(xyz, types[name]) for name, plist in scored.items() for _, xyz, _ in plist]
    t1 = time.time()
    cnn = scorer.score(rec_xyz, rec_types, pose_atoms, batch) if n_poses else np.zeros(0, dtype=np.float32)
    t2 = time.time()
    tmp_out = out_path + ".cnn.tmp"
    i = 0
    n_written = 0
    with open(tmp_out, "w") as f:
        for name, plist in poses.items():
            if name in skipped:
                for _, _, record in plist[:keep]:          # engine order, cut like the scored ligands
                    f.write(record)
                n_written += len(plist[:keep])
                continue
            scores = cnn[i:i + len(plist)]
            i += len(plist)
            order, blend = blend_order([float(p["vina_pose_score"]) for p, _, _ in plist], scores, weight)
            n_written += len(order[:keep])
            for rank, j in enumerate(order[:keep]):
                f.write(add_properties(plist[j][2], [("ud2_cnn_pose_score", f"{scores[j]:.4f}"),
                                                    ("ud2_cnn_blend_score", f"{blend[j]:.4f}"), ("ud2_cnn_rank", str(rank))]))
    os.replace(tmp_out, out_path)
    logger.info("CNN rescoring: %d poses of %d ligands scored (%d ligands skipped), %d poses written, typing %.1f s, "
                "scoring %.1f s, %.2f ms per pose",
                n_poses, len(scored), len(skipped), n_written, t1 - t0, t2 - t1, 1000 * (t2 - t1) / max(n_poses, 1))
    return dict(n_poses=n_poses, n_ligands=len(scored), n_skipped=len(skipped), n_written=n_written, keep=keep,
                typing_s=t1 - t0, scoring_s=t2 - t1)


def serve(socket_path, scorer, batch=64):
    """Resident scorer: one JSON request per line over a Unix socket, served one at a time until SIGTERM / SIGINT."""
    import json
    import signal
    import socket
    if os.path.exists(socket_path):
        os.remove(socket_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    os.chmod(socket_path, 0o600)
    srv.listen(8)
    stop = []
    import threading
    if threading.current_thread() is threading.main_thread():       # signal handlers exist only in the main thread
        signal.signal(signal.SIGTERM, lambda *_: stop.append(1) or srv.close())
        signal.signal(signal.SIGINT, lambda *_: stop.append(1) or srv.close())
    logger.info("CNN rescoring server on %s (%s, %d models loaded)", socket_path, scorer.precision, len(scorer.models))
    try:
        while not stop:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            with conn:
                f = conn.makefile("rwb")
                line = f.readline()
                if not line:
                    continue
                try:
                    req = json.loads(line.decode())
                    if req.get("ping"):
                        reply = dict(ok=True, precision=scorer.precision)
                    else:
                        reply = dict(ok=True, **run(req["sdf"], req["ligand_dms"], req["receptor_dms"], req["out"], scorer,
                                                    float(req.get("weight", 1.5)), batch,
                                                    keep=int(req["keep"]) if req.get("keep") is not None else None))
                except Exception as e:  # noqa: BLE001 - report to the client, keep serving
                    logger.exception("request failed")
                    reply = dict(ok=False, error=f"{type(e).__name__}: {e}")
                f.write((json.dumps(reply) + "\n").encode())
                f.flush()
    finally:
        srv.close()
        if os.path.exists(socket_path):
            os.remove(socket_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdf"); ap.add_argument("--ligand-dms"); ap.add_argument("--receptor-dms"); ap.add_argument("--out")
    ap.add_argument("--weight", type=float, default=1.5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--keep", type=int, default=None,
                    help="write only the first K poses of each ligand after the reorder (default: all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp16",
                    help="fp16 autocast is the default: 3x faster, identical top-1 on udbench and PoseX (max probability shift 0.002)")
    ap.add_argument("--gridder", choices=("auto", "torch", "molgrid"), default=os.environ.get("UD2_CNN_GRIDDER", "auto"),
                    help="density grids from libmolgrid's CUDA GridMaker when it imports (auto), or always in torch; "
                         "UD2_CNN_GRIDDER sets the default")
    ap.add_argument("--serve", nargs="?", const="", default=None, metavar="SOCKET",
                    help="stay resident and serve requests on this Unix socket (default /tmp/ud2_cnn_<user>.sock)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        import torch  # noqa: F401
    except ImportError as e:
        sys.exit(f"CNN rescoring needs torch in this Python ({sys.executable}): {e}")
    scorer = Scorer(args.device, args.weights_dir, args.precision, args.gridder)
    if args.serve is not None:
        if __package__ in (None, ""):
            from client import default_socket
        else:
            from unidock_processing.cnn_rescore.client import default_socket
        serve(args.serve or default_socket(), scorer, args.batch)
        return
    if not (args.sdf and args.ligand_dms and args.receptor_dms and args.out):
        ap.error("--sdf, --ligand-dms, --receptor-dms and --out are required (or --serve)")
    run(args.sdf, args.ligand_dms, args.receptor_dms, args.out, scorer, args.weight, args.batch, keep=args.keep)


if __name__ == "__main__":
    main()
