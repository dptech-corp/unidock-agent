# `unidock-agent` Conda 发布与 `ud2align` 迁移方案

> 状态：待实施 · 第一期目标：通过 GitHub Actions 发布 Linux x86-64 / Python 3.10 的 `unidock-agent` CPU 包到 Quetz `baymax` channel，并完成 `ud2align` 插件依赖切换。

## 1. 背景与拆分思路

目前 `ud2` 核心（C++ 引擎 `unidock_engine` 与 Python 流程层 `unidock_processing`）已开源并迁入本仓库（GitHub `dptech-corp/unidock-agent`），而对齐插件 `ud2align` 仍保留在 GitLab 私有仓库以 Cython `.so` 形式发布。

此前为了先交付 `ud2align 2.0.0`，GitLab 私有仓库提供了 `unidock2-preview` CPU 包，并把它作为插件发布 job 的前置步骤。核心代码迁出后，两个仓库的职责需要彻底拆开：

- **GitHub（本仓库 `unidock-agent`）**：维护正式的核心 Conda 配方与发布 workflow，通过 tag 自动构建、测试并发布 `unidock-agent` 到 Quetz `baymax`。
- **GitLab（`ud2align` 私有仓库）**：只保留插件自身源码、Cython 打包与相关测试；将运行依赖从 `unidock2-preview` 改为 `unidock-agent`，并下线临时的预览引擎构建步骤。

### 1.1 命名、接口与版本约定

| 项目 | 约定 | 说明 |
| --- | --- | --- |
| 核心 Conda 包名 | `unidock-agent` | 与原版 `unidock2` 及过渡包 `unidock2-preview` 区分；第一期 build string 为 `cpu_py310_<build>`。 |
| CLI 与模块接口 | `unidock2`、`ud2`、`unidock_processing`、`unidock_engine`、`unidock2.plugins` | 全部沿用现有命令名与 Python 导入路径，保证用户脚本和 `ud2align` 无缝衔接。 |
| 插件 Conda 包名 | `ud2align` | 提供 `ud2align-cli` 与 `unidock2.plugins` 下的 `bias` 插件；构建、测试和发布均由 GitLab 独立维护。 |
| 版本号分工 | 核心与插件独立打 tag | Git tag 决定 `unidock-agent` Conda 包版本和 `unidock_processing` 元数据（即 `unidock2 --version`）；`ud2 --version` 仍显示 `unidock_engine/CMakeLists.txt` 中的 C++ 引擎版本号（当前为 `0.4.6`），无需强制与 tag 同步。 |
| 发行渠道 | `http://quetz.dp.tech:8088/get/baymax` + `conda-forge` | 与现有安装习惯保持一致，用户侧无需切换 channel。 |

---

## 2. 总体实施顺序

为避免发布窗口内破坏现有用户安装，按以下四步推进：

1. **GitHub 侧准备**：修复本仓库在 Conda Linux 工具链下的两处已知编译问题，迁移 `conda-recipe/` 至 `rattler-build`（CPU 版），并新增 `.github/workflows/conda.yml`。
2. **发布并验收核心包**：打核心发布 tag，触发 GitHub Actions 发布 `unidock-agent` 到 `baymax`；在无 GPU 的干净 Linux 环境验证安装与基础功能，然后更新安装文档。
3. **切换 GitLab 插件**：修改 `ud2align` 的 `recipe.yaml`，将依赖指向已验收的 `unidock-agent` 版本范围，移除 GitLab CI 中的 `unidock2-preview` 前置构建；发布新版 `ud2align` 并做端到端真实对接验收及旧环境升级测试。
4. **清理冗余内容**：确认无误后，清理 GitLab 仓库中已迁出到 GitHub 的核心源码、预览配方和文档引用（保留 Quetz 上已有的 `ud2align=2.0.0` 与 `unidock2-preview=0.1.0` 作为回退基线）。

---

## 3. GitHub 侧改动（本仓库）

### 3.1 修复 Linux Conda 构建所需的源码问题

已验收的 GitLab 预览包 `unidock2-preview-0.1.0-cpu_py310_0` 使用了下面两处源码修复。与 GitHub 基线 `main@61b620c` 逐文件对比，这两个文件各缺一行。迁移配方时把这两行一起移入本仓库，再用目标 Linux sysroot 和配方指定的编译器构建验收。其余引擎和处理层代码以 GitHub 版本为准。

1. **SQLite 链接 `libdl`**：在 [unidock/unidock_engine/lib/CMakeLists.txt](../unidock/unidock_engine/lib/CMakeLists.txt) 中为 `sqlite3` target 补充 `target_link_libraries(sqlite3 PUBLIC ${CMAKE_DL_LIBS})`。SQLite 源码调用 `dlclose`，目标 sysroot 为 glibc 2.17。`scripts/install.sh` 会在检测到旧 glibc 时传入 `-ldl`，而 Conda 配方直接调用 pip 构建，因此应在 CMake target 上声明链接依赖。
2. **`yaml-cpp` 补齐头文件**：在 [unidock/unidock_engine/lib/yaml-cpp/emitterutils.cpp](../unidock/unidock_engine/lib/yaml-cpp/emitterutils.cpp) 顶部补充 `#include <cstdint>`。该文件直接使用 `uint16_t` 和 `uint32_t`，显式包含头文件可以避免依赖特定工具链的间接包含。

### 3.2 迁移 `conda-recipe/` 到 `rattler-build`

当前 [conda-recipe/meta.yaml](../conda-recipe/meta.yaml) 和 [conda-recipe/conda_build_config.yaml](../conda-recipe/conda_build_config.yaml) 仍停留在旧版 `conda-build` 格式（包名为 `unidock2` 且硬编码了 CUDA）。迁移时直接复用已验收的 `unidock2-preview` 的 `recipe.yaml` 和 `build.sh`，替换这套旧配方。`build.sh` 可原样复用。配方将包名及 `about.summary` 改为 `unidock-agent`，将 `PREVIEW_VERSION` / `PREVIEW_BUILD` 改为 `VERSION` / `BUILD`，并补上与过渡包的互斥约束。新目录位于仓库根目录下一级，**`source.path` 要从 `../..` 改为 `..`**，让源文件过滤规则从当前 checkout 的根目录生效。

#### `conda-recipe/recipe.yaml` 关键配置

- **版本与构建号**：从环境变量读取（例如 `VERSION` 默认 `0.0.0`，`BUILD` 默认 `0`），由 CI 根据 tag 注入。
- **平台与 build string**：限定 `linux-64`，第一期固定 `build.string: cpu_py310_${{ build_number }}`。
- **源码过滤（`source.filter`）**：仅纳入构建所需文件（`unidock/**`、`pyproject.toml`、`README.md`、`LICENSE`），排除 `unidock/unidock_processing/**/test*/**`，沿用已验证配方的过滤范围。
- **依赖声明**：
  - `build`：`${{ compiler('c') }}`、`${{ compiler('cxx') }}`、`sysroot_linux-64 2.17.*`、`cmake`、`ninja`（打包镜像/环境不一定有 `make`，统一用 Ninja）。
  - `host`：`python 3.10.*`、`pip`、`setuptools`、`setuptools-scm`、`scikit-build-core`、`pybind11`、`wheel`；沿用预览配方的 host 依赖 `msys_viparr_lpsolve55`，以及预生成扭转库用到的 `dill`、`rdkit`、`mdanalysis`（`unidock_processing` 首次导入时会在 `site-packages` 下生成 `torsion_library.pkl`，必须在打包期完成）。
  - `run`：`python 3.10.*`、`_openmp_mutex * *_gnu`、`pyyaml`、`pydantic >=2`、`pathos`、`numpy`、`scipy`、`pandas`、`networkx`、`rdkit`、`mdanalysis`、`pdbfixer`、`openmm`、`msys_viparr_lpsolve55`、`ambertools_stable`、`pytorch * cpu*`。
- **包互斥约束（`run_constraints`）**：
  - 声明 `unidock2-preview <0a0` 和 `unidock2 <0a0`，防止用户环境中同时存在旧包导致 `bin/ud2`、`bin/unidock2` 与 Python 模块路径冲突。
- **原生冒烟测试（`tests`）**：
  - `script` 检查：`ud2 --version`、`unidock2 --version`（确认与 `${{ version }}` 一致）、`unidock2 docking --help`、导入 `unidock_engine.api.python.pipeline` 与 `unidock_processing.core.engine`、校验 `unidock2.plugins` 入口组可正常加载内置的 `template` 与 `covalent` 插件。
  - `package_contents` 检查：`bin/ud2`、`bin/unidock2`、`lib/python3.10/site-packages/unidock_processing/torsion_library/data/torsion_library.pkl`。
- **开源许可与声明（`about.license_file`）**：携带 `LICENSE`、`unidock/unidock_engine/NOTICE` 与 `unidock/unidock_processing/cnn_rescore/weights/NOTICE`。

#### `conda-recipe/build.sh` 关键步骤

```bash
#!/usr/bin/env bash
set -euo pipefail

export CMAKE_GENERATOR=Ninja

# 1. 构建并安装 C++ 引擎（CPU / OpenMP）与 Python 绑定
python -m pip install --no-deps --no-build-isolation ./unidock/unidock_engine \
  -Ccmake.define.USE_CUDA=OFF \
  -Ccmake.define.USE_OPENMP=ON \
  -Ccmake.define.BUILD_BIN=ON \
  -Ccmake.define.BUILD_TEST=OFF

# 2. 将 site-packages 下的 ud2 相对软链到环境 bin/，确保安装后开箱在 PATH 中可用
ln -s "../lib/python3.10/site-packages/unidock_engine/api/python/bin/ud2" "$PREFIX/bin/ud2"

# 3. 注入包版本并安装 unidock_processing
SETUPTOOLS_SCM_PRETEND_VERSION="$PKG_VERSION" python -m pip install --no-deps --no-build-isolation .

# 4. 打包期触发生成 torsion_library.pkl，避免只读环境运行时写入 site-packages 失败
python -c "import unidock_processing"
```

### 3.3 新增 GitHub Actions workflow（`.github/workflows/conda.yml`）

Quetz 服务公网可达，参考 [Uni-Dock2 的 conda.yml](https://github.com/dptech-corp/Uni-Dock2/blob/main/.github/workflows/conda.yml)，使用 GitHub 托管的 `ubuntu-latest` 在**单个 job** 内完成「安装工具 → 构建 → 原生测试 → 上传」即可，无需自建公开 Docker 镜像，也不需要拆成多个 job 传递 artifacts。

#### 触发与版本号规则

- **Tag 触发（构建 + 测试 + 上传）**：仅在 `push` 事件解析并校验发布 tag，将解析出的 `VERSION` 和 `BUILD` 写入 `$GITHUB_ENV` 供构建和上传使用。建议核心 tag 采用 `vX.Y.Z`（支持 `aN`/`bN`/`rcN`/`.devN`/`.postN` 后缀），默认 `BUILD=0`；同版本重打包使用 `vX.Y.Z-buildN`。首个正式版本号由维护者在发版前确定。
- **PR / 手动触发（仅构建 + 测试）**：复用同一套构建和测试步骤，跳过上传步骤（不读取 `QUETZ_API_KEY`）；这两个事件使用开发版本（例如 `VERSION=0.0.0.dev0`、`BUILD=0`），同样写入 `$GITHUB_ENV`。

#### 环境变量与权限配置

仓库中已配置好所需变量与密钥，workflow 直接引用即可：

| 名称 | 类型 | 用途 |
| --- | --- | --- |
| `QUETZ_SERVER_URL` | Repository Variable（`${{ vars.QUETZ_SERVER_URL }}`） | Quetz 服务根地址（如 `http://quetz.dp.tech:8088`）。 |
| `QUETZ_API_KEY` | Repository Secret（`${{ secrets.QUETZ_API_KEY }}`） | 仅在 tag 上传步骤注入同名环境变量，需具备 `baymax` 上传权限。 |
| `CONDA_CHANNEL` | Workflow `env`（固定为 `baymax`） | 构建依赖读取与成品上传的目标 channel。 |
| `RATTLER_ZSTD` / `RATTLER_SHARDED` / `RATTLER_BZ2` | Workflow `env` | 分别设为 `"true"` / `"false"` / `"false"`（已验证 Quetz 与 conda-forge 均支持 `.zst`，可减少约 86% 索引传输量）。 |

#### 关键执行步骤

```yaml
- name: Install rattler-build
  uses: prefix-dev/rattler-build-action@v0.2.39
  with:
    setup-only: true
    rattler-build-version: v0.55.1

- name: Build and test Conda package
  run: |
    rattler-build build -r conda-recipe \
      --target-platform linux-64 \
      --package-format conda \
      --test native \
      --output-dir dist/conda \
      -c "${{ vars.QUETZ_SERVER_URL }}/get/${{ env.CONDA_CHANNEL }}" \
      -c conda-forge

- name: Upload to Quetz
  if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
  env:
    QUETZ_API_KEY: ${{ secrets.QUETZ_API_KEY }}
  run: |
    file="dist/conda/linux-64/unidock-agent-${VERSION}-cpu_py310_${BUILD}.conda"
    test -f "$file"
    rattler-build upload quetz \
      --url "${{ vars.QUETZ_SERVER_URL }}/" \
      --channel "${{ env.CONDA_CHANNEL }}" \
      "$file"
```

> 首次运行可从 Actions 日志观察 `rattler-build` 安装、依赖求解与编译测试各自耗时；若上传步骤意外中断，重试前先检查 Quetz `baymax` 中同名同 build 的包是否已入库。

### 3.4 更新安装文档（核心验收通过后）

首个 `unidock-agent` CPU 包在干净环境验收通过后，更新 [README.md](../README.md) 与 [docs/installation.md](../docs/installation.md)：

- 补充标准 Conda 安装方式：
  ```sh
  conda install unidock-agent -c http://quetz.dp.tech:8088/get/baymax -c conda-forge
  ```
- 明确说明当前阶段发布的 Conda 包为 **Linux x86-64 / Python 3.10 CPU 版**：普通 docking 可直接在无 GPU 环境运行，`--cnn_rescore` 或 CUDA 加速仍需在带 GPU 的环境按源码/后续 CUDA 包说明配置。

---

## 4. GitLab 侧切换与端到端验收

待 GitHub 产出首个已发布的 `unidock-agent` 后，按以下顺序完成 `ud2align` 迁移：

1. **核心包独立验收**
   - 在无 GPU 的干净 Linux x86-64 环境安装指定版本的 `unidock-agent`。
   - 用 `conda list --show-channel-urls` 核对包名、`cpu_py310_*` build string 与 `baymax` 来源；验证 `ud2 --version`、`unidock2 --version`、`unidock2 docking --help` 以及内置插件加载。记录实际通过验收的版本、build string 和 channel。
2. **修改 `ud2align` 配方与 GitLab CI**
   - 在 GitLab 仓库的 `conda/ud2align/recipe.yaml` 中，将运行依赖 `unidock2-preview >=0.1,<0.2` 替换为已验收的 `unidock-agent` 兼容版本范围（例如 `>=X.Y,<X.(Y+1)`）；Python 元数据侧继续保留实际的 distribution 名 `unidock_processing`。
   - 从 GitLab `.gitlab-ci.yml` 中移除 `unidock2-preview` 的前置构建上传代码块，以及仅为此并发保护设置的 `resource_group: quetz-baymax`；保留 `ud2align-v...` tag 触发的单 job 流程（构建 Cython `.so` 扩展、`--test native`、上传 Quetz）。
3. **新包端到端验收（从用户入口）**
   - 发 tag 前，用本地候选插件包搭配 Quetz 已发布的新核心，在干净环境验证插件加载、真实对齐和归档中的 Cython 扩展。验收通过后，再发布新的 `ud2align` tag（新版本号或递增 `-buildN`）。
   - 在全新空环境中直接执行用户安装命令，确认 Conda 自动拉取新版 `ud2align` 与新发布的 `unidock-agent` CPU 包：
     ```sh
     conda create -y -n ud2align-release-check --no-default-packages -c conda-forge python=3.10
     conda activate ud2align-release-check
     conda install ud2align -c http://quetz.dp.tech:8088/get/baymax -c conda-forge
     conda list --show-channel-urls
     ```
   - 在源码目录外执行功能验证：
     - `ud2align-cli --version`
     - `unidock2 docking --help` 包含 `--bias_template_sdf`，且能正常加载 `unidock2.plugins` 下的 `bias` 入口。
     - 检查已安装或打包归档内容，确认 `ud2align` 的受保护核心模块仅以编译后的 `.so` 交付，未夹带源码。
     - 使用插件仓库现有的标准对齐基准用例在 `--ncpu 1` 下跑一次真实对齐，确认结果在既有 0.1 Å RMSD 容差内。
   - **旧环境升级测试**：在一个事先安装了旧版 `ud2align=2.0.0` + `unidock2-preview=0.1.0` 的环境中执行升级安装，确认 `unidock2-preview` 被顺利替换为 `unidock-agent` 且无文件冲突或 CLI 报错。
4. **清理 GitLab 冗余文件**
   - 以上验收全部通过后，确认所需核心修复均已合入本仓库，再从 GitLab 清理已迁出的核心源码、`conda/unidock2-preview/` 配方及过渡期文档引用；保留 `ud2align` 插件源码、Cython 构建脚本、插件测试、参考数据与 benchmark，并同步更新 GitLab 的安装文档。

---

## 5. 回退预案

Quetz `baymax` 上继续保留已发布的 `ud2align-2.0.0-py310_0` 与 `unidock2-preview-0.1.0-cpu_py310_0` 不删除。若迁移后发现兼容性问题：

1. 用户或 CI 可在干净环境中通过精确锁定恢复旧组合：
   ```sh
   conda install "ud2align=2.0.0=py310_0" "unidock2-preview=0.1.0=cpu_py310_0" -c http://quetz.dp.tech:8088/get/baymax -c conda-forge
   ```
2. 修复问题后，通过提升对应仓库的 `-buildN` 或版本号重新发包即可。

---

## 6. 后续规划：CUDA 变体

第一期 CPU 流程稳定后，继续沿用同一个 `unidock-agent` 包名与 `conda-recipe/` 目录扩展 `cuda*` 构建变体（例如通过 `variants.yaml` 区分 `cpu_*` 与 `cuda*`）：

- 开启 `USE_CUDA=ON`，补充 CUDA 编译与运行时依赖（`cuda-version`、`cuda-nvcc`、`cuda-cudart-dev`、`libcurand-dev` 及驱动 `__cuda` 兼容约束）和 GPU 版 PyTorch。
- 完成后，用户可根据环境自动解析或显式指定 `"unidock-agent[build='cpu_*']"` / `"unidock-agent[build='cuda*']"`。
- 发布前必须在具备目标 NVIDIA GPU 的环境完成真实 docking 及 `--cnn_rescore` 验收；`ud2align` 仍只依赖 `unidock-agent` 的兼容版本范围，无需为区分 CPU/GPU 改包名。
