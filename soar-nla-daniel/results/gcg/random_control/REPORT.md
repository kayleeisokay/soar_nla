# GCG unconditional random-substitution control

Scope: 1573 Gemma-3-12B Alpaca rollouts with sequence_token_index > 50.

Random controls match only the final internal-token Hamming distance;
positions and non-special vocabulary tokens use one unconditional uniform draw, with no model-score selection.
All intervals are paired 95% percentile bootstrap intervals over examples.

## kitft

| beta | original FVE | GCG FVE | random FVE | GCG gain | random gain | GCG − random |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.7211 [0.7100, 0.7316] | 0.7721 [0.7635, 0.7802] | 0.7032 [0.6915, 0.7143] | 0.0510 [0.0478, 0.0544] | -0.0180 [-0.0203, -0.0158] | 0.0690 [0.0647, 0.0734] |
| 0.01 | 0.7211 [0.7105, 0.7317] | 0.8141 [0.8062, 0.8220] | 0.6490 [0.6372, 0.6603] | 0.0930 [0.0895, 0.0964] | -0.0721 [-0.0767, -0.0679] | 0.1651 [0.1596, 0.1708] |
| 0 | 0.7211 [0.7104, 0.7314] | 0.8065 [0.7982, 0.8145] | 0.6341 [0.6219, 0.6459] | 0.0854 [0.0824, 0.0885] | -0.0871 [-0.0925, -0.0818] | 0.1725 [0.1665, 0.1787] |

## Edit-distance diagnostics

| beta | edit distance mean (range) |
|---:|---:|
| 0.1 | 4.36 (0–17) |
| 0.01 | 14.36 (4–20) |
| 0 | 16.41 (8–20) |
