"""Atom-density grids for the gnina pose networks, in torch only (reproduces libmolgrid's GridMaker to float precision).

Grid: 48 points per axis at 0.5 A, points at centre + (i - 23.5) * 0.5, centre = the ligand's heavy-atom centroid.
Each atom paints exp(-2 u^2) for u = d / r < 1, (4 u^2 - 12 u + 9) / e^2 for 1 <= u < 1.5, 0 beyond, summed per channel.
Channels: 14 receptor classes then 14 ligand classes of the gnina default typers; indices refer to the 28-type
GninaIndexTyper table used by atom_types.gnina_type. Radii and maps were read off libmolgrid 0.5.5.
"""
import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)

POINTS, RESOLUTION, DIMENSION, CHANNELS = 48, 0.5, 23.5, 28
# radius per 28-type (A): H, polar H, 4 carbons, 4 N, 4 O, S, S acc, P, F, Cl, Br, I, Mg, Mn, Zn, Ca, Fe, metal, B
RADII = np.array([0.37, 0.37, 1.9, 1.9, 1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.7, 1.7, 1.7, 1.7, 2.0, 2.0, 2.1, 1.5, 1.8, 2.0, 2.2,
                  1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.92], dtype=np.float32)
# 28-type -> channel (-1 = not gridded); receptor channels 0-13, ligand channels 14-27
REC_CHANNEL = np.array([-1, -1, 0, 1, 2, 3, 5, 6, 6, 5, 7, 8, 8, 7, 9, 9, 10, 4, 4, 4, 4, 13, 13, 12, 11, 13, 13, 13], dtype=np.int64)
LIG_CHANNEL = np.array([-1, -1, 14, 15, 16, 17, 21, 22, 22, 21, 23, 24, 24, 23, 25, 25, 26, 20, 19, 18, 18, 27, 27, 27, 27, 27, 27, 27],
                       dtype=np.int64)
_E2 = math.e ** 2
_HALF = (POINTS - 1) / 2.0                      # 23.5: grid point i sits at centre + (i - 23.5) * RESOLUTION
_REACH = int(math.ceil(1.5 * float(RADII.max()) / RESOLUTION))   # voxels an atom can reach on each side (7)
_OFFSETS = None


def density(u):
    """libmolgrid's radial profile as a function of d / r (torch)."""
    g = torch.exp(-2.0 * u * u)
    q = (4.0 * u * u - 12.0 * u + 9.0) / _E2
    return torch.where(u < 1.0, g, torch.where(u < 1.5, q, torch.zeros_like(u)))


def _offsets(device):
    global _OFFSETS
    if _OFFSETS is None or _OFFSETS.device != device:
        r = torch.arange(-_REACH, _REACH + 1, device=device)
        _OFFSETS = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)      # (K^3, 3)
    return _OFFSETS


def paint(grid, xyz, types, channel_map, centre):
    """Add the atoms' densities into grid (CHANNELS, 48, 48, 48); xyz (N, 3) and types (N,) as torch tensors."""
    device = grid.device
    typed = types >= 0                                   # hydrogens carry type -1: never index the map with them
    ch = torch.full_like(types, -1)
    ch[typed] = torch.as_tensor(channel_map, device=device)[types[typed]]
    keep = ch >= 0
    if not bool(keep.any()):
        return
    xyz, ch, r = xyz[keep], ch[keep], torch.as_tensor(RADII, device=device)[types[keep]]
    # fractional grid coordinates of each atom; nearest voxel; local cube of (2 * reach + 1)^3 voxels around it
    f = (xyz - centre) / RESOLUTION + _HALF                                      # (N, 3)
    near = torch.round(f).long()
    idx = near[:, None, :] + _offsets(device)[None, :, :]                        # (N, K^3, 3)
    d = torch.linalg.norm((idx.float() - f[:, None, :]) * RESOLUTION, dim=-1)    # (N, K^3)
    val = density(d / r[:, None])
    inside = ((idx >= 0) & (idx < POINTS)).all(-1) & (val > 0)
    flat = ((ch[:, None] * POINTS + idx[..., 0]) * POINTS + idx[..., 1]) * POINTS + idx[..., 2]
    grid.view(-1).index_add_(0, flat[inside], val[inside])


def make_grids(rec_xyz, rec_types, poses, device):
    """poses: list of (lig_xyz (n,3) float array, lig_types (n,) int array) -> tensor (len(poses), 28, 48, 48, 48)."""
    rx = torch.as_tensor(np.asarray(rec_xyz, dtype=np.float32), device=device)
    rt = torch.as_tensor(np.asarray(rec_types, dtype=np.int64), device=device)
    out = torch.zeros((len(poses), CHANNELS, POINTS, POINTS, POINTS), dtype=torch.float32, device=device)
    for k, (lx, lt) in enumerate(poses):
        lx = torch.as_tensor(np.asarray(lx, dtype=np.float32), device=device)
        lt = torch.as_tensor(np.asarray(lt, dtype=np.int64), device=device)
        gridded = lt >= 0
        gridded[gridded.clone()] = torch.as_tensor(LIG_CHANNEL, device=device)[lt[gridded]] >= 0
        centre = lx[gridded].mean(0)                                             # heavy-atom centroid, as molgrid
        # receptor atoms that can reach the cube
        reach = DIMENSION / 2 + 1.5 * float(RADII.max())
        close = (torch.abs(rx - centre) <= reach).all(1)
        paint(out[k], rx[close], rt[close], REC_CHANNEL, centre)
        paint(out[k], lx, lt, LIG_CHANNEL, centre)
    return out


def _channels(table, types):
    """28-type indices -> channel per atom, -1 for hydrogens (type -1) and unmapped types; never indexes with -1."""
    ch = np.full(types.shape, -1, dtype=np.int64)
    typed = types >= 0
    ch[typed] = table[types[typed]]
    return ch


class MolgridGridder:
    """The same grids from libmolgrid's CUDA GridMaker, 0.1-0.2 ms per pose where make_grids takes 1.8 ms.

    Same channel maps, radii, centre and geometry as make_grids; the two agree to 1e-5 on real poses. The coordinate,
    type and radius tensors stay referenced for the whole forward call: molgrid wraps their memory without holding a
    reference, and a temporary freed early is reused by the caching allocator while the kernel is still queued.
    """

    def __init__(self, device):
        import molgrid
        if device.type != "cuda":
            raise RuntimeError("the molgrid gridder runs on a CUDA device")
        self.molgrid = molgrid
        self.device = device
        self.maker = molgrid.GridMaker(resolution=RESOLUTION, dimension=DIMENSION)

    def __call__(self, rec_xyz, rec_types, poses):
        mg = self.molgrid
        rec_types = np.asarray(rec_types, dtype=np.int64)
        rec_ch = _channels(REC_CHANNEL, rec_types)
        typed = rec_ch >= 0
        rx, rec_ch, rec_r = np.asarray(rec_xyz, dtype=np.float32)[typed], rec_ch[typed], RADII[rec_types[typed]]
        reach = DIMENSION / 2 + 1.5 * float(RADII.max())
        out = torch.zeros((len(poses), CHANNELS, POINTS, POINTS, POINTS), dtype=torch.float32, device=self.device)
        for k, (lig_xyz, lig_types) in enumerate(poses):
            lig_types = np.asarray(lig_types, dtype=np.int64)
            lig_ch = _channels(LIG_CHANNEL, lig_types)
            keep = lig_ch >= 0
            lx = np.asarray(lig_xyz, dtype=np.float32)[keep]
            centre = lx.mean(0)                                                   # heavy-atom centroid, as make_grids
            close = (np.abs(rx - centre) <= reach).all(1)
            coords = torch.as_tensor(np.concatenate([rx[close], lx]), device=self.device)
            types = torch.as_tensor(np.concatenate([rec_ch[close], lig_ch[keep]]).astype(np.float32), device=self.device)
            radii = torch.as_tensor(np.concatenate([rec_r[close], RADII[lig_types[keep]]]), device=self.device)
            grid = out[k]
            self.maker.forward(mg.float3(float(centre[0]), float(centre[1]), float(centre[2])),
                               mg.tensor_as_grid(coords), mg.tensor_as_grid(types), mg.tensor_as_grid(radii),
                               mg.tensor_as_grid(grid))
            del coords, types, radii, grid                                        # after the call, not before
        return out


def make_gridder(device, choice="auto"):
    """The gridding callable (rec_xyz, rec_types, poses) -> (B, 28, 48, 48, 48) and its name.

    choice: "auto" takes libmolgrid when it imports on a CUDA device and the torch gridder otherwise; "torch" and
    "molgrid" force one, the latter raising when molgrid is unavailable.
    """
    if choice not in ("auto", "torch", "molgrid"):
        raise ValueError(f"gridder must be auto, torch or molgrid, got {choice!r}")
    if choice == "torch" or (choice == "auto" and device.type != "cuda"):
        return (lambda rec_xyz, rec_types, poses: make_grids(rec_xyz, rec_types, poses, device)), "torch"
    try:
        return MolgridGridder(device), "molgrid"
    except Exception as exc:                                                    # noqa: BLE001 - any import/CUDA failure
        if choice == "molgrid":
            raise
        logger.warning("libmolgrid unavailable (%s); gridding in torch", exc)
        return (lambda rec_xyz, rec_types, poses: make_grids(rec_xyz, rec_types, poses, device)), "torch"
