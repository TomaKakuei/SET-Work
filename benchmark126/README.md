# Registered 126-input package

This directory contains the 21 heterogeneous tasks with six registered inputs each (126 total), their local observation data, input preparation and scoring code, and the source for four five-step routes: `original`, `quad`, `linear_fast`, and `repair`. It uses the single shared checkpoint in `../code/checkpoint/`. No training or task-specific model selection is performed.

From the repository root, install the core requirements and the additional image decoder, build the two native libraries, then run:

```sh
python -m pip install -r benchmark126/requirements.txt
python scripts/build_native.py --cc cc
python -m benchmark126 verify-data
python -m benchmark126 audit-inputs
python -m benchmark126 list
python -m benchmark126 run --key validation_graph_noisy_view_0 --method original
python -m benchmark126 score --key validation_graph_noisy_view_0 --method original
```

On Windows, use the TinyCC or Zig build command in the root README. Omit `--key` to process all 126 cases. Choose each of the four route names with `--method`; completed per-case predictions and scores are skipped on a resumed run. Outputs go to `results/benchmark126/` and are excluded from Git. `score` reads frozen predictions; it does not call an optimizer. Solver calls use five outer steps. The `linear_fast` and `repair` registrations cap total joint Jacobians at 21; this is a separate auxiliary budget from the five-step quality comparison.

`cases.json` retains the original task IDs, slot, group, synthetic seeds, source scene identity, and image-crop origin. `data/` holds the selected HPatches source frames and homographies, selected Middlebury scene files, and exact 16×16 restoration input/target arrays. The 223 data files total about 200 MB; none exceeds 5 MB. Full external datasets and the approximately 1 GB of Python closure caches are unnecessary. `DATA_MANIFEST.json` lists size and SHA-256 for each distributed data file. `export_from_workspace.py` records how the selected crops and scenes were extracted; it is only needed by a maintainer rebuilding the data package from the historical research workspace.

`input_audit.json` is the historical signature record. The bundled `AUDIT.json` reports the independent reconstruction check performed in this repository. All 126 inputs matched the registered dimension, dtype, native-interface status and initial objective within `1e-8 + 1e-6 × max(|a|,|b|)`. Initial-state SHA-256 matched for 123/126; three multiframe SIFT-derived initial states differed in bytes under the current Python/OpenCV/PyTorch build while matching the registered initial objective. Residual and Jacobian byte hashes matched for 70/126 and 66/126 respectively. The audit reports these differences rather than asserting cross-build bitwise identity. Historical result tables and negative outcomes remain in `../evidence/`; this bundle does not replace those scores.

The actual method implementations are `../code/setsunet_csn/stage5.py`, `source/finite_models_20260925/` (quad), `source/finite_adaptation_20260926/` (linear_fast), and `source/finite_repair_20260927/` (repair). The source directory also contains the image/geometry compilers and the exact finite-model helper scripts prefixed `frozen_`. The small unprefixed helper modules expose only the functions needed by these four routes, with the original equations. The HPatches PPM reader uses `np.fromfile` plus the same OpenCV decoder to handle non-ASCII checkout paths on Windows. `SOURCE_MANIFEST.json` records source hashes and original research paths.

These are development follow-up routes, not changes to the repository's default inference policy. Case-specific quality should be read per task and registered comparison, with the paper's retained 18-task table kept distinct from the historical 21-task suite.
