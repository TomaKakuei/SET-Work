# Compact results and paper mapping

All five comparison methods use the same retained cases. The attached PDF is the reference for the methods, task scopes and limitations. Full measurements and processed data remain available upon request.

| Paper table | PDF page | Summary CSV |
|---|---:|---|
| Table 1 | 6 | [table_01.csv](table_01.csv) |
| Table 2 | 7 | [table_02.csv](table_02.csv) |
| Table 3 | 14 | [table_03.csv](table_03.csv) |
| Table 4 | 16 | [table_04.csv](table_04.csv) |
| Table 5 | 16 | [table_05.csv](table_05.csv) |
| Table 6 | 17 | [table_06.csv](table_06.csv) |
| Table 7 | 18 | [table_07.csv](table_07.csv) |
| Table 8 | 18 | [table_08.csv](table_08.csv) |
| Table 9 | 19 | [table_09.csv](table_09.csv) |

Figure 2 (page 6): [outcome counts](figure_02_outcomes.csv), 18-task and 64-case cohorts.

## Table 1: Five-step recovery on common prepared cases. Bold marks the best mean and practical ties within 2% in error (equivalent RMSE for PSNR). Arrows indicate the preferred direction.

Five outer steps on common prepared cases. PSNR higher; task errors lower. Bold source cells denote the best or practically tied means: 2% error, or 20*log10(1.02) dB for PSNR (RMSE equivalence).

| \begin{tabular}{@{}lrrrrrr@{}}\toprule Group | Scored/reg. | SETSUNET | HSLM | Ceres LM | Dogbox | VeLO |
| --- | --- | --- | --- | --- | --- | --- |
| HPatches pairs (lower better) | 579/580 | 46.2240 | 56.5578 | 46.1859 | 46.1879 | 70.8503 |
| Multiframe (lower better) | 115/116 | 15.2157 | 17.9581 | 18.7085 | 16.4495 | 27.5571 |
| Balanced graph (lower better) | 64/64 | 0.1090 | 0.5561 | 0.0101 | 0.0101 | 0.7295 |
| Biased graph (lower better) | 64/64 | 0.2387 | 0.5813 | 0.3522 | 0.3522 | 0.7338 |
| Broyden (lower better) | 64/64 | 5.37e-06 | 0.1822 | 7.02e-16 | 4.93e-16 | 0.5223 |
| SE3 (lower better) | 64/64 | 0.0049 | 0.0089 | 0.0034 | 0.0034 | 0.0093 |
| TUM (lower better) | 47/60 | 0.0592 | 0.0662 | 0.0368 | 0.0366 | 0.0678 |
| Known blur (higher better) | 66/66 | 41.2231 | 39.4490 | 41.2238 | 41.2238 | 36.9533 |
| Low light (higher better) | 100/100 | 10.3752 | 10.2626 | 10.2577 | 10.3087 | 10.3169 |
| SIDD (higher better) | 160/160 | 23.9106 | 23.8230 | 23.8927 | 23.9030 | 23.9602 |
| Stereo (lower better) | 180/180 | 2.2550 | 2.5275 | 2.6018 | 2.5328 | 2.6018 |

## Table 2: Direction and coordination comparisons with CG2. Entries are relative RMSE reductions with descriptive 95% seed-cluster intervals; each column contains 48 conditions from 16 seeds. The first row averages recovery errors over three equal-rank random completions before comparison with learned directions. The second compares minimax with mean coordination using the same learned directions.

Relative RMSE reductions (%) with descriptive 95% seed-cluster intervals; 48 conditions per column and 16 seeds. First row: learned versus three equal-rank random completions averaged before comparison, with minimax and CG2 fixed. Second row: minimax versus mean coordination with the same learned directions and CG2. The direction comparison does not isolate training from network architecture.

| \toprule Reference | Systematic bias | View noise |
| --- | --- | --- |
| Random completion (3 seeds) | +48.73% [+47.01, +50.52] | +76.30% [+72.99, +79.45] |
| Mean coordination | +9.45% [+6.57, +12.37] | -10.43% [-15.71, -5.18] |

## Table 3: Full five-step time in seconds, medians of three rotated serial repeats after one warmup. All routes share the original network and recorded host/backend. The last two cases confirm a frozen implementation. Ratios divide original time by Cholesky time. Non-stereo cases take the original fallback. The full-block stereo cases have zero exact minimax correction; these timings isolate equivalent execution and whitening cost.

Full five-step solve time, seconds; median of three rotated serial repeats after one warmup. Same weights/host/backend. Last two rows are frozen-implementation confirmation. Non-stereo cases use the original fallback. Ratios use underlying measurements and need not equal the quotient of rounded printed times.

| \begin{tabular}{@{}lrrrr@{}}\toprule Case | Original | Paired eigen | Cholesky | Ratio |
| --- | --- | --- | --- | --- |
| Biased graph | 0.0251 | 0.0247 | 0.0253 | 0.99x |
| Multiframe | 3.0780 | 3.0190 | 3.2319 | 0.95x |
| Stereo A | 7.9020 | 6.6296 | 2.7307 | 2.89x |
| Low light | 2.4179 | 2.4865 | 2.4280 | 1.00x |
| SIDD | 0.8425 | 0.8729 | 0.8691 | 0.97x |
| Stereo B | 9.3961 | 7.5215 | 2.9686 | 3.17x |
| Stereo C | 9.0197 | 7.9213 | 2.9613 | 3.05x |

## Table 4: Five-step recovery on the 18-task heterogeneous benchmark. Parentheses give state dimensions. RMSE and pixel errors are lower-is-better; PSNR is higher-is-better.

All 18 heterogeneous task configurations use all five methods. Dimensions and metrics are retained.

| \begin{tabular}{@{}lllrrrrr@{}}\toprule Task | Problem (dimension) | Metric | SETSUNET | HSLM | Ceres LM | Dogbox | VeLO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T01 | Biased graph (48) | RMSE | 0.1165 | 0.6537 | 0.2068 | 0.2068 | 0.7400 |
| T02 | Biased graph (96) | RMSE | 0.2321 | 0.7267 | 0.2857 | 0.2857 | 0.7643 |
| T03 | SE3 (18) | m | 0.0014 | 0.0026 | 0.0014 | 0.0014 | 0.0039 |
| T04 | Multiframe (16) | px | 2.5350 | 3.6085 | 3.5863 | 3.7153 | 3.5862 |
| T05 | Multiframe (16) | px | 0.1637 | 0.2540 | 0.2116 | 0.1849 | 0.2113 |
| T06 | Known blur (256) | dB | 40.1678 | 34.6637 | 40.1625 | 40.1625 | 33.0457 |
| T07 | Low light (786) | dB | 8.2994 | 8.2896 | 8.2192 | 8.2480 | 8.2079 |
| T08 | Low light (786) | dB | 8.3023 | 8.2751 | 8.2210 | 8.2841 | 8.2091 |
| T09 | SIDD (770) | dB | 34.1144 | 33.2233 | 34.1158 | 34.1152 | 34.2352 |
| T10 | Stereo (1536) | px | 0.1884 | 0.3941 | 0.4047 | 0.3501 | 0.4047 |
| T11 | Biased graph (48) | RMSE | 0.1572 | 0.7118 | 0.1978 | 0.1978 | 0.7740 |
| T12 | Biased graph (96) | RMSE | 0.2363 | 0.6827 | 0.2910 | 0.2910 | 0.7244 |
| T13 | SE3 (30) | m | 0.0038 | 0.0046 | 0.0038 | 0.0038 | 0.0040 |
| T14 | Multiframe (40) | px | 11.7885 | 14.7951 | 15.2581 | 16.6917 | 28.6592 |
| T15 | Known blur (256) | dB | 53.4392 | 53.0480 | 53.4322 | 53.4322 | 47.2617 |
| T16 | Low light (786) | dB | 8.0832 | 8.0402 | 8.0253 | 8.0547 | 8.0252 |
| T17 | SIDD (770) | dB | 35.4657 | 35.0917 | 35.4655 | 35.4651 | 35.4356 |
| T18 | Stereo (1536) | px | 0.2503 | 0.2933 | 0.3222 | 0.3020 | 0.3222 |

## Table 5: Comparison coverage. Common preparation failures affect every method. The complementary suite combines 32 extension cases and 32 biased graphs, retaining cohort identities. The derivative study counts VeLO gradients separately from Jacobians.

All five methods cover each usable case. The 64-case cohort combines 32+32 with no replacement cases. VeLO derivative counts are gradients, not Jacobians.

| \begin{tabular}{@{}llll@{}}\toprule Study | Scored/reg. | Budget | Methods |
| --- | --- | --- | --- |
| Heterogeneous tasks | 18/18 | 5 steps | All five |
| Complementary cases | 64/64 | 5 steps | All five |
| Large collections | 1,503/1,518 | 5 steps | All five |
| Graph and multiframe | 20/20 | 5 steps | All five |
| Five-target multiframe | 7/8 | 5 steps | All five |
| Optimizer transfer | 11/11 | 5 steps | All five |
| Derivative budget | 60/60 | 21 derivatives | All five |

## Table 6: Mean recovery improvement and descriptive 95% source-cluster intervals. Positive favors SETSUNET. Units follow Table 1; restoration differences are in dB.

Paired improvement favors SETSUNET; 10,000 source-cluster bootstrap draws. Intervals are descriptive. Frozen HSLM and VeLO intervals are reused; new Ceres LM and Dogbox intervals use the same source-cluster unit.

| \begin{tabularx}{\linewidth}{@{}l*{4}{>{\raggedleft\arraybackslash}X}@{}}\toprule Group | HSLM | Ceres LM | Dogbox | VeLO |
| --- | --- | --- | --- | --- |
| HPatches pairs | +10.3 [-0.111, +30.9] | -0.0381 [-0.175, +0.0456] | -0.0361 [-0.169, +0.0434] | +24.6 [-0.0399, +73.7] |
| Multiframe | +2.74 [-0.276, +7.17] | +3.49 [+0.17, +8.2] | +1.23 [-1.14, +4.41] | +12.3 [+3.72, +23] |
| Balanced graph | +0.447 [+0.399, +0.494] | -0.0989 [-0.116, -0.0819] | -0.0989 [-0.116, -0.0822] | +0.621 [+0.603, +0.637] |
| Biased graph | +0.343 [+0.291, +0.391] | +0.113 [+0.103, +0.124] | +0.113 [+0.103, +0.124] | +0.495 [+0.476, +0.515] |
| Broyden | +0.182 [+0.166, +0.201] | -5.37e-06 [-5.81e-06, -4.92e-06] | -5.37e-06 [-5.81e-06, -4.91e-06] | +0.522 [+0.492, +0.554] |
| SE3 | +0.00406 [+0.00334, +0.00482] | -0.0015 [-0.00208, -0.000968] | -0.0015 [-0.00209, -0.000973] | +0.00446 [+0.00336, +0.00566] |
| TUM | +0.00696 [+0.00119, +0.0165] | -0.0225 [-0.0621, -0.00318] | -0.0226 [-0.0625, -0.00318] | +0.00859 [+0.00428, +0.0143] |
| Known blur | +1.77 [+1.67, +1.88] | -0.00067 [-0.00189, +0.000726] | -0.00067 [-0.00191, +0.00073] | +4.27 [+3.83, +4.75] |
| Low light | +0.113 [+0.0938, +0.133] | +0.118 [+0.0942, +0.141] | +0.0665 [+0.0549, +0.0787] | +0.0583 [+0.0472, +0.0693] |
| SIDD | +0.0876 [+0.0187, +0.206] | +0.0179 [+0.0139, +0.0222] | +0.00763 [+0.00577, +0.0097] | -0.0496 [-0.0836, +0.00719] |
| Stereo | +0.273 [+0.178, +0.388] | +0.347 [+0.233, +0.48] | +0.278 [+0.18, +0.396] | +0.347 [+0.234, +0.479] |

## Table 7: Minimax versus mean coordination under systematic bias, with the same learned directions. Positive relative RMSE reduction favors minimax.

Minimax versus mean coordination with identical learned directions. Relative RMSE reduction (%) and descriptive seed-cluster intervals; positive favors minimax. CG0/CG2 are separate settings.

| \begin{tabular}{@{}lrr@{}}\toprule Bias strength | CG2 | CG0 |
| --- | --- | --- |
| 2/3 | +14.36% [10.98, 17.87] | -3.23% [-5.24, -1.26] |
| 1 | +12.08% [10.10, 14.26] | -0.72% [-2.62, 1.14] |

## Table 8: Mean relative recovery gain of learned completion over equal-rank random completion on the size-transfer instances.

Mean relative recovery gain (%) over equal-rank random completion; two new biased-graph seeds per dimension, same checkpoint and 8+8 direction budget.

| \begin{tabular}{@{}lr@{}}\toprule Variables | Versus random |
| --- | --- |
| 48 | +16.401% |
| 192 | +33.093% |
| 768 | +24.845% |

## Table 9: Recovery with a cap of 21 joint Jacobians; VeLO uses 21 joint gradients, reported separately. Ten cases per task.

Separate derivative allowance: 21 joint Jacobians for SETSUNET/HSLM/Ceres LM/Dogbox; 21 joint gradients and horizon 21 for VeLO. Latest completed endpoints are used. This is separate from the five-step comparison.

| \begin{tabular}{@{}lrrrrr@{}}\toprule Task | SETSUNET | HSLM | Ceres LM | Dogbox | VeLO |
| --- | --- | --- | --- | --- | --- |
| Biased graph | 0.2478 | 0.1862 | 0.2587 | 0.2587 | 0.7334 |
| SE3 | 0.0035 | 0.0039 | 0.0035 | 0.0035 | 0.0061 |
| Multiframe | 2.4655 | 2.0605 | 2.0399 | 3.7360 | 9.5249 |
| Stereo | 3.7066 | 3.9112 | 3.9361 | 3.8892 | 3.9438 |
| Known blur | 40.9827 | 40.9825 | 40.9825 | 40.9825 | 40.3676 |
| Low light | 9.9075 | 8.0177 | 8.0344 | 7.9807 | 8.0846 |
