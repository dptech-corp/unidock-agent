"""Pure-Python parts of the CNN rescoring stage (no torch / molgrid needed)."""
import os
import tempfile

import numpy as np

from unidock_processing.cnn_rescore.atom_types import add_properties, gnina_type, iter_records
from unidock_processing.cnn_rescore.rescore import blend_order, fits_grid, zscore

RECORD = """lig
     UniDock2       3D

  2  1  0  0  0  0  0  0  0  0999 V2000
    1.0000    2.0000    3.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    2.5000    3.5000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
M  END
>  <ud2_molecule_name>
lig

>  <ud2_pose_index>
{k}

>  <vina_pose_score>
{score}

$$$$
"""


def test_gnina_type_mapping():
    assert gnina_type(0, 1, False) == -1                 # hydrogen: not gridded
    assert gnina_type(2, 6, False) == 2 and gnina_type(2, 6, True) == 4    # C_H aliphatic / aromatic
    assert gnina_type(3, 6, False) == 3 and gnina_type(3, 6, True) == 5    # C_P
    assert [gnina_type(t, 7, False) for t in (4, 5, 6, 7)] == [6, 7, 9, 8]  # N, N_D, N_A, N_DA
    assert [gnina_type(t, 8, False) for t in (8, 9, 10, 11)] == [10, 11, 13, 12]
    assert gnina_type(20, 30, False) == 23 and gnina_type(20, 12, False) == 21 and gnina_type(20, 29, False) == 26
    assert [gnina_type(t, 9, False) for t in (12, 16, 17, 18)] == [17, 18, 19, 20]


def test_blend_order_and_zscore():
    assert np.allclose(zscore([1.0, 1.0, 1.0]), 0.0)
    # engine prefers pose 0 (lowest score); the CNN prefers pose 2; weight 1.5 lets the CNN win, weight 0 keeps the engine
    order, blend = blend_order([-10.0, -9.5, -9.0], [0.2, 0.3, 0.9], 1.5)
    assert order[0] == 2 and len(order) == 3 and blend[2] > blend[0]
    order0, _ = blend_order([-10.0, -9.5, -9.0], [0.2, 0.3, 0.9], 0.0)
    assert order0 == [0, 1, 2]


def test_records_round_trip_and_added_tags():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.sdf")
        with open(path, "w") as f:
            f.write(RECORD.format(k=0, score=-10.0) + RECORD.format(k=1, score=-9.0))
        recs = list(iter_records(path))
        assert [r[0] for r in recs] == ["lig", "lig"]
        assert recs[1][1]["vina_pose_score"] == "-9.0" and recs[0][2][1] == (1.5, 2.5, 3.5)
        new = add_properties(recs[0][3], [("ud2_cnn_rank", "0")])
        assert new.endswith(">  <ud2_cnn_rank>\n0\n\n$$$$\n") and new.count("$$$$") == 1
        with open(path, "w") as f:
            f.write(new)
        again = list(iter_records(path))
        assert again[0][1]["ud2_cnn_rank"] == "0" and again[0][1]["ud2_pose_index"] == "0"


def test_config_plumbing():
    from unidock_processing.io.yaml import UnidockConfig
    cfg = UnidockConfig.from_dict({"Settings": {"cnn_rescore": True}})
    assert cfg.to_protocol_kwargs()["cnn_rescore"] is True
    assert UnidockConfig().to_protocol_kwargs()["cnn_rescore"] is False


def test_fits_grid_extent_rule():
    small = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 5.0, 0.0)]
    assert fits_grid(small, [2, 2, 2])
    long = [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0)]                 # 15 A from the centroid: outside the 11.75 A half-edge
    assert not fits_grid(long, [2, 2])
    assert fits_grid([(0.0, 0.0, 0.0), (30.0, 0.0, 0.0)], [-1, 2])   # a hydrogen far away does not count
    assert not fits_grid([(0.0, 0.0, 0.0)], [-1])              # no heavy atom


def test_client_without_server_and_server_error_path():
    import json
    import socket
    import threading
    from unidock_processing.cnn_rescore.client import rescore_via_server
    from unidock_processing.cnn_rescore.rescore import serve

    class FakeScorer:
        precision, models = "fp32", [None]

        def score(self, types_path, n_poses, batch=64):
            raise AssertionError("not reached")

    with tempfile.TemporaryDirectory() as tmp:
        sock = os.path.join(tmp, "s.sock")
        assert rescore_via_server(sock, {"ping": True}) is None          # nobody listening -> fall back
        t = threading.Thread(target=serve, args=(sock, FakeScorer()), daemon=True); t.start()
        reply = None
        for _ in range(200):                              # the socket file appears before listen(): retry the ping
            reply = rescore_via_server(sock, {"ping": True}, timeout=10) if os.path.exists(sock) else None
            if reply is not None:
                break
            threading.Event().wait(0.02)
        assert reply == {"ok": True, "precision": "fp32"}
        reply = rescore_via_server(sock, {"sdf": "/nonexistent.sdf", "ligand_dms": "/nonexistent.dms",
                                          "receptor_dms": "/nonexistent.dms", "out": os.path.join(tmp, "o.sdf")}, timeout=10)
        assert reply["ok"] is False and "error" in reply                 # server stays up and reports the failure
        assert rescore_via_server(sock, {"ping": True}, timeout=10)["ok"] is True


def test_grid_profile_matches_libmolgrid():
    """Values libmolgrid 0.5.5 produced for one carbon at the ligand centroid, read along x at y = z = +0.25 A."""
    import torch
    from unidock_processing.cnn_rescore import grid
    g = grid.make_grids([(50.0, 50.0, 50.0)], [2], [([(0.0, 0.0, 0.0)], [2])], torch.device("cpu"))[0]
    line = g[14, :, 24, 24].tolist()
    expected = {18: 0.0009, 19: 0.04913, 20: 0.17103, 21: 0.39262, 22: 0.68326, 23: 0.90134, 24: 0.90134, 25: 0.68326,
                26: 0.39262, 27: 0.17103, 28: 0.04913, 29: 0.0009, 30: 0.0}
    for i, v in expected.items():
        assert abs(line[i] - v) < 2e-4, (i, line[i], v)
    assert float(g[14].sum()) > 0 and float(g[0].sum()) == 0.0        # ligand carbon in channel 14, far receptor atom outside the cube


def test_grid_channel_maps():
    from unidock_processing.cnn_rescore import grid
    assert grid.REC_CHANNEL[2] == 0 and grid.LIG_CHANNEL[2] == 14           # aliphatic hydrophobic carbon
    assert grid.REC_CHANNEL[9] == 5 and grid.REC_CHANNEL[6] == 5             # N acceptor shares the plain-N channel
    assert grid.REC_CHANNEL[17] == grid.REC_CHANNEL[20] == 4                 # all halogens one receptor channel
    assert grid.LIG_CHANNEL[17] == 20 and grid.LIG_CHANNEL[18] == 19 and grid.LIG_CHANNEL[19] == grid.LIG_CHANNEL[20] == 18
    assert grid.REC_CHANNEL[0] == -1 and grid.LIG_CHANNEL[0] == -1           # hydrogens are not gridded
    assert abs(float(grid.RADII[2]) - 1.9) < 1e-6 and abs(float(grid.RADII[27]) - 1.92) < 1e-6


def test_grid_ignores_hydrogens():
    import torch
    from unidock_processing.cnn_rescore import grid
    heavy_only = grid.make_grids([(50.0, 50.0, 50.0)], [2], [([(0.0, 0.0, 0.0)], [2])], torch.device("cpu"))[0]
    with_h = grid.make_grids([(50.0, 50.0, 50.0)], [2], [([(0.0, 0.0, 0.0), (6.0, 6.0, 6.0)], [2, -1])], torch.device("cpu"))[0]
    assert torch.allclose(heavy_only, with_h)              # the far hydrogen neither paints nor shifts the centre
    assert float(with_h[27].sum()) == 0.0


def _minimal_dms_pair(tmp):
    """ligand + receptor DMS with just the tables the CNN typing reads (two heavy atoms each)"""
    import sqlite3
    lig, rec = os.path.join(tmp, "lig.dms"), os.path.join(tmp, "rec.dms")
    c = sqlite3.connect(lig)
    c.executescript("""
        CREATE TABLE particle (id INTEGER, anum INTEGER);
        CREATE TABLE ud2_atom_props (atom_id INTEGER, vina_type INTEGER);
        CREATE TABLE ud2_atom_aromatic (atom_id INTEGER);
        CREATE TABLE ud2_metadata (key TEXT, value TEXT);
        INSERT INTO particle VALUES (0, 6), (1, 8);
        INSERT INTO ud2_atom_props VALUES (0, 2), (1, 7);
        INSERT INTO ud2_metadata VALUES ('original_name', 'lig');
    """)
    c.commit(); c.close()
    c = sqlite3.connect(rec)
    c.executescript("""
        CREATE TABLE particle (id INTEGER, anum INTEGER, name TEXT, resname TEXT, x REAL, y REAL, z REAL);
        CREATE TABLE ud2_atom_props (atom_id INTEGER, vina_type INTEGER);
        INSERT INTO particle VALUES (0, 6, 'CA', 'ALA', 0.0, 0.0, 0.0), (1, 7, 'N', 'ALA', 1.0, 0.0, 0.0);
        INSERT INTO ud2_atom_props VALUES (0, 2), (1, 5);
    """)
    c.commit(); c.close()
    return lig, rec


class _LastPoseScorer:
    """CNN stand-in that likes the last pose of the file most (scores 0.1, 0.2, ..., per pose in file order)"""
    precision, models = "fp32", [None]

    def score(self, rec_xyz, rec_types, poses, batch=64):
        return np.asarray([0.1 * (k + 1) for k in range(len(poses))], dtype=np.float32)


def test_keep_writes_first_k_after_reorder():
    from unidock_processing.cnn_rescore.rescore import run
    with tempfile.TemporaryDirectory() as tmp:
        lig, rec = _minimal_dms_pair(tmp)
        src = os.path.join(tmp, "p.sdf")
        with open(src, "w") as f:                         # engine order: pose 0 best by the engine, pose 2 best by the CNN
            f.write("".join(RECORD.format(k=k, score=s) for k, s in enumerate((-10.0, -9.0, -8.0))))
        for keep, n_expected in ((None, 3), (2, 2), (1, 1)):
            out = os.path.join(tmp, f"o{keep}.sdf")
            reply = run(src, lig, rec, out, _LastPoseScorer(), keep=keep)
            recs = list(iter_records(out))
            assert reply["n_poses"] == 3 and reply["n_written"] == n_expected and len(recs) == n_expected
            assert [r[1]["ud2_cnn_rank"] for r in recs] == [str(i) for i in range(n_expected)]
            assert recs[0][1]["ud2_pose_index"] == "2"                      # the CNN-promoted pose comes first
            assert recs[0][1]["vina_pose_score"] == "-8.0"                  # its engine tags are untouched
        assert not os.path.exists(out + ".cnn.tmp")
        try:
            run(src, lig, rec, os.path.join(tmp, "bad.sdf"), _LastPoseScorer(), keep=0)
            raise AssertionError("keep=0 must be rejected")
        except ValueError:
            pass


def test_server_honours_keep():
    import threading
    from unidock_processing.cnn_rescore.client import rescore_via_server
    from unidock_processing.cnn_rescore.rescore import serve
    with tempfile.TemporaryDirectory() as tmp:
        lig, rec = _minimal_dms_pair(tmp)
        src = os.path.join(tmp, "p.sdf")
        with open(src, "w") as f:
            f.write("".join(RECORD.format(k=k, score=s) for k, s in enumerate((-10.0, -9.0, -8.0))))
        sock = os.path.join(tmp, "s.sock")
        threading.Thread(target=serve, args=(sock, _LastPoseScorer()), daemon=True).start()
        reply = None
        for _ in range(200):
            reply = rescore_via_server(sock, {"ping": True}, timeout=10) if os.path.exists(sock) else None
            if reply is not None:
                break
            threading.Event().wait(0.02)
        assert reply and reply["ok"]
        out = os.path.join(tmp, "o.sdf")
        reply = rescore_via_server(sock, dict(sdf=src, ligand_dms=lig, receptor_dms=rec, out=out, keep=1), timeout=10)
        assert reply["ok"] and reply["n_written"] == 1 and reply["keep"] == 1
        recs = list(iter_records(out))
        assert len(recs) == 1 and recs[0][1]["ud2_pose_index"] == "2" and recs[0][1]["ud2_cnn_rank"] == "0"
        reply = rescore_via_server(sock, dict(sdf=src, ligand_dms=lig, receptor_dms=rec, out=out), timeout=10)
        assert reply["ok"] and reply["n_written"] == 3 and reply["keep"] is None   # no keep: every pose, as before


def test_make_gridder_choices():
    import torch
    from unidock_processing.cnn_rescore import grid
    cpu = torch.device("cpu")
    assert grid.make_gridder(cpu, "auto")[1] == "torch"               # molgrid is a CUDA path in this stage
    assert grid.make_gridder(cpu, "torch")[1] == "torch"
    gridder, _ = grid.make_gridder(cpu, "torch")
    g = gridder([(50.0, 50.0, 50.0)], [2], [([(0.0, 0.0, 0.0)], [2])])
    assert tuple(g.shape) == (1, 28, 48, 48, 48) and float(g[0, 14].sum()) > 0
    try:
        grid.make_gridder(cpu, "nope")
        assert False, "unknown gridder accepted"
    except ValueError:
        pass


def test_molgrid_gridder_matches_torch():
    """libmolgrid's GridMaker and make_grids paint the same grids; runs where molgrid and a CUDA device are present."""
    import pytest
    torch = pytest.importorskip("torch")
    pytest.importorskip("molgrid")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from unidock_processing.cnn_rescore import grid
    rng = np.random.default_rng(7)
    rec_xyz = [tuple(v) for v in rng.uniform(-12.0, 12.0, size=(400, 3))]
    rec_types = [int(t) for t in rng.integers(0, 28, size=400)]           # all 28 types, hydrogens included
    poses = []
    for _ in range(3):
        xyz = [tuple(v) for v in rng.uniform(-5.0, 5.0, size=(30, 3))]
        types = [int(t) for t in rng.integers(-1, 28, size=30)]           # -1 = hydrogen
        poses.append((xyz, types))
    dev = torch.device("cuda")
    ours = grid.make_grids(rec_xyz, rec_types, poses, dev)
    theirs = grid.MolgridGridder(dev)(rec_xyz, rec_types, poses)
    torch.cuda.synchronize()
    assert float(ours.abs().max()) > 0.5
    assert float((ours - theirs).abs().max()) < 1e-4
