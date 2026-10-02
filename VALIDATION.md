# Compact release validation

Verified on 2026-10-02 using Windows x86-64, Python 3.13.5, CPU PyTorch 2.9.1, NumPy 2.1.3, SciPy 1.16.3, threadpoolctl 3.5.0, and TinyCC 0.9.27.

Both native libraries were rebuilt from the committed C source. The build script uses relative paths and an explicit TinyCC support-directory argument so it also works when the repository's parent path contains non-ASCII characters.

```text
python scripts/build_native.py --cc <tcc executable> --tcc
python -m unittest discover -s tests -p "test_*.py" -v

test_core_properties ... ok
test_four_cases_five_steps ... ok
test_frozen_curve_endpoints ... ok
test_native_minimax_and_rank ... ok
test_shared_checkpoint ... ok
Ran 5 tests
OK

python scripts/run_cases.py
# Completed all 12 endpoints: four cases, three methods, five outer steps each.
```

The tests include 60 SPD completion cases, 12 native/reference minimax cases, rank-deficient orthogonalization, checkpoint tensor hashes, and four frozen five-step curve-envelope endpoint regressions. Endpoint tolerance is relative error below 1e-7; recovery-score tolerance is 1e-10 plus 1e-6 times the larger absolute score, matching the original standalone release check. No tolerance was changed after observing results.

The compact examples are deterministic software checks. Their scores are written separately for each task, and no aggregate success criterion or performance claim is inferred. The test wrapper uses one numerical thread and five outer steps; it does not train weights or extend the experiment budget.

An initial build attempt with the locally stored Zig distribution failed because some compiler headers were unavailable. TinyCC builds and the complete test suite then passed. Zig, GCC/Clang, Linux and macOS build paths are provided but were not qualified by this Windows check. Library dependencies are installed separately; no virtual environment or compiler is redistributed.

