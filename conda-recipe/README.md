# Conda CPU release

This rattler-build recipe produces `unidock-agent` for Linux x86-64 / Python 3.10,
with OpenMP enabled and CUDA disabled. It preserves the `ud2`, `unidock2` and
`unidock2.plugins` interfaces. CNN rescoring needs a separately configured GPU
environment; it is outside this CPU release's acceptance criteria.

## Local build and tests

On Linux x86-64, using rattler-build **v0.55.1**:

```sh
export VERSION=0.0.0.dev0 BUILD=0
export RATTLER_ZSTD=true RATTLER_SHARDED=false RATTLER_BZ2=false
rattler-build build -r conda-recipe \
  --target-platform linux-64 --package-format conda --test native \
  --output-dir dist/conda \
  -c http://quetz.dp.tech:8088/get/baymax -c conda-forge
```

Run from the repository root. Without overrides, the recipe uses `VERSION=0.0.0`
and `BUILD=0`. The native tests install the package in a separate environment and
check both CLIs, the processing version, engine imports and the built-in
`template` / `covalent` entry points. Package contents tests also require the
pre-generated torsion library and both CLI paths. The package conflicts with
`unidock2` and `unidock2-preview`, which own the same installed paths.

## GitHub Actions

`.github/workflows/conda.yml` uses one job for building, native tests and, for
release tags, uploading to Quetz's `baymax` channel.

| Trigger | Version / build | Upload |
| --- | --- | --- |
| Pull request | `0.0.0.dev0` / `0` | No |
| Manual (`workflow_dispatch`), including a selected tag | `0.0.0.dev0` / `0` | No |
| Push `vX.Y.Z` | `X.Y.Z` / `0` | Yes, after tests pass |
| Push `vX.Y.Z-buildN` | `X.Y.Z` / `N` | Yes, after tests pass |

Tags also accept `aN`, `bN`, `rcN`, `.devN` or `.postN` after the three-part
version, for example `v1.2.3rc1-build2`. Invalid `v*` tags fail validation before
building. The first release version is a maintainer decision. The tag sets the
processing package version (`unidock2 --version`); the C++ engine keeps its own
version (`ud2 --version`).

Configure the repository variable `QUETZ_SERVER_URL` with the Quetz server root
and the repository secret `QUETZ_API_KEY` with upload access to `baymax`. Only the
tag upload step receives that secret. Before retrying an interrupted upload,
check whether that exact version and build already exists in the channel.

After the first published package passes clean-environment acceptance, update
the user installation documentation and migrate the private `ud2align` package
as described in [the release plan](../docs/conda-release-plan.md).
