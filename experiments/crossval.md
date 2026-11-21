# Cross-validation of the layer-25 results

Date: 2026-11-20. Branch: `stack/05-causal-sets`. Tag: `exp/crossval`. Status: kept.

Config: `configs/crossval.yaml`. Results: `results/crossval/crossval.json`.

```
bash scripts/fetch_runs.sh stage2_l25
python scripts/crossval.py --config configs/crossval.yaml --device cpu   # 2 minutes, no forward pass
```

## Question

- Stage 2 used one held-out split (story_idx 5). How stable is the accuracy over other splits?
- Which shrinkage kappa gives the best held-out transfer of the sparse sets?

## Method

- 6 folds. Fold k holds out the stories with story_idx k (20 per emotion), and neutral stories 2k and 2k+1. Fold 5 is Stage 2's split.
  - Check: fold 5 reproduces Stage 2's split and its accuracy (0.829).
- Per fold:
  - the 12-way held-out accuracy of the denoised `resid_post` vectors;
  - per emotion, the greedy order of the raw MLP write for kappa 0 (plain target), 1, 4, 9 and 16;
  - the held-out transfer of each order at 64, 128, 256 and 512 neurons, and of the full MLP.

## Results

- **Accuracy per fold:** 0.825, 0.725, 0.725, 0.783, 0.746, 0.829. Mean 0.772, sd 0.048.
  - Stage 2's split is the best fold. The single split overstated the accuracy by about 0.06.
- **Full-MLP transfer:** 0.743 (sd 0.026).
- **Transfer of the greedy sets** (mean over folds and emotions; sd over folds in brackets):

  | kappa | 64 | 128 | 256 | 512 |
  |---|---|---|---|---|
  | 0 (plain) | 0.548 | 0.609 | 0.670 (0.013) | 0.724 |
  | 1 | 0.548 | 0.611 | **0.674** (0.012) | 0.734 |
  | 4 | 0.548 | 0.610 | 0.670 (0.009) | 0.729 |
  | 9 | 0.535 | 0.594 | 0.649 | 0.700 |
  | 16 | 0.509 | 0.559 | 0.605 | 0.648 |

- kappa 1 is the best, but its gain over the plain target is 0.004 at 256 neurons. That is less than the sd over folds.
- kappa 9 and 16 lower the transfer.

## Conclusion

- The layer-25 accuracy is 0.77 +- 0.05, not 0.83.
- Shrinkage does not help. The plain greedy target is as good as any kappa within noise.
