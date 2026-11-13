# libmolgrid build patches

`scripts/build_molgrid.sh` builds libmolgrid 0.5.5 from the upstream source (https://github.com/gnina/libmolgrid,
Apache-2.0 / GPL-2.0) with these two patches, which are the ones the conda-forge molgrid recipe applies
(https://github.com/conda-forge/molgrid-feedstock, `recipe/`, BSD-3-Clause):

- `cmake-fix.patch` turns off `-Werror` for one warning and comments out upstream's hard-coded
  `CMAKE_CUDA_ARCHITECTURES`. conda-forge then builds without passing any, which leaves only sm_52 machine code plus
  PTX in `libmolgrid.so`; the script passes the architectures on the cmake command line instead (`MOLGRID_ARCHS`).
- `python-setup.patch` reads the version from `PKG_VERSION` instead of from git, because the source is a zip.

Nothing of libmolgrid or OpenBabel is shipped in this repository: the script downloads the source and links the
OpenBabel installed in the user's env.
