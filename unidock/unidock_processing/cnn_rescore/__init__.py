"""CNN rescoring of docking poses: gnina's published pose networks on our own atom typing, blended with the engine score.

Runs in a separate process (``cnn_rescore/rescore.py``, optionally resident with ``--serve``). The only dependency
beyond numpy is torch with CUDA; the atom-density grids are computed in torch (grid.py), no libmolgrid or OpenBabel.
"""
