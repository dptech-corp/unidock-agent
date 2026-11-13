"""The CNN rescoring stage glue: how many poses the engine writes for it and the call into cnn_rescore/.

The CNN itself (torch) is imported only inside ``run_cnn_rescore``'s separate process or resident server, so
importing this module never imports torch.
"""
from typing import Optional
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def engine_num_pose(requested: int, rescore_pool: int, cnn_on: bool, template_or_covalent: bool,
                    keeps_engine_order: bool = False) -> int:
    """Poses per ligand the engine writes: num_pose, raised to the rescore pool when the CNN stage will run.

    The engine re-ranks max(rescore_pool, num_pose) cluster representatives with its pose pass and writes the
    first num_pose. The CNN stage re-ranks what is written, so with the stage on the engine writes the whole
    pool and the stage writes back the first num_pose after its reorder. Measured on udbench: a pool of 10
    carries the full gain of both re-rankers, 30 adds +0.3 to +0.9 points (n.s.) at three times the CNN cost.
    The raise only feeds the CNN, so a run that skips the stage writes num_pose: constraint docking
    (*template_or_covalent*) and a run whose plugin keeps the engine order (*keeps_engine_order*, a biased run).
    """
    if cnn_on and not template_or_covalent and not keeps_engine_order:
        return max(requested, rescore_pool)
    return requested


def run_cnn_rescore(
    sdf: str,
    ligand_dms: Optional[str],
    receptor_dms: Optional[str],
    keep: int,
    gpu_device_id: int,
    pool_written: Optional[int] = None,
) -> None:
    """Reorder each ligand's poses by the engine score blended with the gnina pose-CNN ensemble.

    Uses a resident scorer when one listens on ``UD2_CNN_SOCKET`` (``rescore.py --serve``; models loaded once,
    about 20 ms per ligand), otherwise runs ``cnn_rescore/rescore.py`` by path in a separate interpreter
    (``UD2_CNN_PYTHON``, default the current one; ``UD2_CNN_PRECISION`` fp16 | fp32), which costs about 1.5 s of
    start-up per call. molgrid and RDKit cannot share a process and the script must not import this package first.
    The SDF is replaced in place; on failure the engine's SDF is left as written and the error is raised. Measured for ranking the poses
    of one ligand (re-docking top-1 +4 to +5 points); it does not improve the ranking of ligands in a screen.
    The engine ran with *pool_written* poses per ligand (``engine_num_pose``; *keep* when not given); the stage
    writes back the first *keep* of each ligand after the reorder, every tag on them as the engine wrote it.
    """
    if pool_written is None:
        pool_written = keep
    if not (ligand_dms and receptor_dms):
        raise RuntimeError("CNN rescoring needs the enriched ligand and receptor DMS files")
    from unidock_processing.cnn_rescore.client import default_socket, rescore_via_server
    request = dict(sdf=sdf, ligand_dms=ligand_dms,
                   receptor_dms=receptor_dms, out=sdf,
                   keep=keep)
    socket_path = default_socket()
    reply = rescore_via_server(socket_path, request)
    if reply is not None:                      # a resident scorer (rescore.py --serve) answered: models already loaded
        if not reply.get('ok'):
            raise RuntimeError(f"CNN rescoring server failed: {reply.get('error')}; the engine SDF is unchanged")
        if 'n_written' not in reply and keep < pool_written:
            raise RuntimeError("CNN rescoring server predates the keep count and wrote every pose; restart it from "
                               "this checkout (python unidock_processing/cnn_rescore/rescore.py --serve)")
        logger.info("CNN rescoring via server %s: %d poses of %d ligands, %d skipped, %d written, scoring %.2f s",
                    socket_path, reply['n_poses'], reply['n_ligands'], reply['n_skipped'],
                    reply.get('n_written', reply['n_poses']), reply['scoring_s'])
        return
    python = os.environ.get('UD2_CNN_PYTHON', sys.executable)
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cnn_rescore', 'rescore.py')
    cmd = [python, script,
           '--sdf', sdf, '--ligand-dms', ligand_dms,
           '--receptor-dms', receptor_dms, '--out', sdf,
           '--device', f'cuda:{gpu_device_id}', '--precision', os.environ.get('UD2_CNN_PRECISION', 'fp16'),
           '--keep', str(keep)]
    logger.info("CNN rescoring (no server on %s): %s", socket_path, ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in (result.stdout + result.stderr).strip().splitlines():
        logger.info("  %s", line)
    if result.returncode != 0:
        raise RuntimeError(f"CNN rescoring failed (exit {result.returncode}); the engine SDF is unchanged")
