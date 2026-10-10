| Run | Setting | Peak training-batch FVE (step <= 600) | Held-out FVE (last eval <= 600) |
|---|---|---|---|
| A | Warm start (SFT), KL 0.01 | 75.9% (0.7589 @ step 537) | 74.4% (0.744 @ step 600) |
| B | Cold start, KL 0 | 72.3% (0.7225 @ step 537) | 70.9% (0.709 @ step 590) |
| C | Cold start, KL 0.01 | 68.9% (0.6894 @ step 597) | 66.5% (0.665 @ step 600) |
