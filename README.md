# SETSUNET-CSN

Compact research implementation of **SETSUNET**, a shared learned proposal network combined with measured curvature, Schur completion, and a two-view minimax correction for low-budget nonlinear recovery.

[中文说明](README_ZH.md)

This repository contains the core Python and C source, **one shared 49,096-parameter checkpoint**, and small synthetic tests. Examples use **five outer steps**. Inputs are generated locally; no benchmark dataset download is required.

## Quick start

Use Python 3.11 or newer and a C compiler (GCC, Clang, Zig, or TinyCC). From the repository directory:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/build_native.py --cc cc
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/run_cases.py
```

On Windows, the locally verified compiler is TinyCC 0.9.27 (x86-64):

```sh
python scripts/build_native.py --cc C:/path/to/tcc.exe --tcc
```

For Zig:

```sh
python scripts/build_native.py --cc /path/to/zig --zig
```

Use the actual `zig.exe` path on Windows. Build completion is required before running the tests. The build creates two small local shared libraries; compiled binaries and the compiler itself are not committed. A CPU PyTorch installation is sufficient. The scripts request one numerical thread and use float64 algebra.

The last command runs SETSUNET-CSN, the project's same-curvature dense LM control, and PCG16 on each of four generated cases. It writes individual scores and endpoints to `results/compact.json`. All three methods receive five outer steps and identical initial states. The examples are software demonstrations, not a replacement for the paper's task-specific comparisons.

## What is included

| Path | Purpose |
| --- | --- |
| `code/setsunet_csn/stage5.py` | Core inference loop, Schur completion, damping and step acceptance |
| `code/setsunet_csn/model_stage5.py` | Shared network with C feature/proposal construction |
| `code/setsunet_csn/core.py` | Reference completion, minimax solver and algebra |
| `code/setsunet_csn/curvature_stage*.py` | Curvature interfaces and block whitening |
| `code/native_stage4/`, `code/native_stage5/` | C source for the small-space and factor/feature kernels |
| `code/checkpoint/` | Tensor-only NPZ weights, architecture and per-tensor hashes |
| `code/compact_runtime.py` | Verified checkpoint loader and runnable example configuration |
| `code/setsunet_csn/stage6.py` | Differentiable trajectory implementation for source inspection |
| `code/setsunet_csn/stage5_curve_ablation.py` | Optional curve-envelope research implementation |
| `tests/` | Numerical properties, native/reference agreement and endpoint regression |

Earlier numbered modules remain because the core shares their numerical routines. The demonstrated entry point is `setsunet_csn.stage5.solve`; the top-level historical `setsunet_csn.solve` is an earlier implementation. Legacy adapters that require external image/geometry packages, historical `from_profile` files, and full training orchestration are outside this compact package. Use `compact_runtime.load_model()` to load the included NPZ weights.

## Small tests

- Sixty SPD completion checks: duality gap, locked measured components, basis invariance, equivariance and a differentiable proposal check.
- Twelve native/reference minimax comparisons and a rank-deficient orthogonalization check.
- Four generated tasks: balanced, noisy and biased two-view graph calibration (96 parameters), and Broyden equations (64 parameters). Each runs CSN, dense LM and PCG16 for five steps.
- Four existing five-step curve-envelope endpoints, covering dimensions 96/192 and the original development/confirmation splits. The first registered seed at contrast 4 is retained in each split/dimension; these are historical regression fixtures, not new independent experimental evidence.

The checkpoint loader checks every tensor's SHA-256 and parameter count. Tests check finite outputs and accepted objective descent, not a requirement that one method win every heterogeneous task. The original **21-task heterogeneous benchmark suite** remains distinct from these compact software tests.

## Provenance and scope

`SOURCE_PROVENANCE.json` records the original relative source paths and hashes. Numerical solver and network source is copied from the research workspace; the native loaders only change the platform-specific library suffix. `MANIFEST.json` records released file sizes and hashes, and `python scripts/check_manifest.py` verifies them.

All examples use the same original shared checkpoint, exported losslessly to NPZ. Its original checkpoint SHA-256 is:

```text
df2a4d248214b9fbf2b423ed9385cbd79dd41931dd7a778d960cc19f3fdd45e6
```

This export contains no large datasets, historical run archives, virtual environments, third-party optimizer weights, or rejected model candidates. The curve module is an explicitly selected research extension; the default examples use the incumbent straight-step solver. No training or production-model replacement is performed.

See [VALIDATION.md](VALIDATION.md) for the tested environment and checks. Linux/macOS build paths are provided but require validation on those platforms. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for the existing licensing scope.

