# Stage 2: emotion vectors at Qwen3-8B layer 25

Run 2026-11-18 on a Colab L4, commit `bf8ede5` (branch `stage2`), config `configs/stage2_l25.yaml`.

## Reproduce

```
python scripts/extract.py       --config configs/stage2_l25.yaml   # 182 s on an L4 (+92 s model load)
python scripts/build_vectors.py --config configs/stage2_l25.yaml   # seconds, CPU
python scripts/project.py       --config configs/stage2_l25.yaml   # 26 s
python scripts/report.py        --config configs/stage2_l25.yaml   # seconds, CPU
```

- All four steps exited 0. The outputs (`runs/stage2_l25/sums.safetensors`, `results/stage2_l25/*`, the HTML report) were **not** copied off the VM:
  - the S3 upload needs the AWS secrets, and Claude's permission classifier blocked reading them;
  - the VM was released, so the outputs are gone. The pipeline is deterministic from the commit; rerun it, then sync to S3.
- To keep them next time, run this in Colab yourself after the four steps (it reads the AWS secrets):
  ```
  import os; from google.colab import userdata
  for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]: os.environ[k] = userdata.get(k)
  !pip -q install awscli && cd /content/emotion-neurons && aws s3 sync results/stage2_l25 s3://narang99-private/emotion-neurons/stage2_l25/results && aws s3 sync runs/stage2_l25 s3://narang99-private/emotion-neurons/stage2_l25/runs
  ```
- Padding check (most-padded story alone vs batched): relative diff ≤ 4.6e-3 on the residual sites, ≤ 1.6e-2 on `mlp_act` (bf16).

## Results

- **Vectors work on held-out stories.** 12-way held-out accuracy (20 stories per emotion), story mean from token 50, nearest vector after centring on the emotional grand mean (chance 0.083):

  | site | raw | denoised |
  |---|---|---|
  | resid_mid | 0.746 | 0.838 |
  | resid_post | 0.725 | 0.829 |
  | mlp_act | 0.779 | 0.829 |
  | mlp_out | 0.738 | 0.808 |

  - Train leave-one-out (denoised): 0.802 / 0.792 / 0.814 / 0.786.
  - Denoising removes 11 / 11 / 16 / 14 neutral PCs (50% of neutral variance).
- **Stable.** Split-half cosine (topics 0–9 vs 10–19): resid_mid 0.79 (sad) to 0.92 (surprised); mlp_act 0.73 (proud) to 0.85.
- **Logit lens (resid_post, mean/denoised) names the emotion:**
  - angry: 愤怒, 怒, accusing, 侮辱, accusation, insult, angrily
  - proud: proudly, 自豪, proud, 骄傲, 成就感, pride
  - afraid: 恐惧, 恐慌, panic, 恐怖, terror
  - guilty: guilt, 良心, 愧, culpa, 疚, remorse
  - calm is the weakest: 惬意, 愉快, leisure, 悠闲, gradually, 从容
- **Valence structure** (cosine, resid_mid mean/denoised): afraid–nervous 0.64, happy–proud 0.53, happy–loving 0.48; happy–afraid −0.51, loving–nervous −0.56.
- **Errors are near neighbours** (mlp_act confusion, all 1440): nervous→afraid 13, afraid→nervous 10, sad→calm 13, sad→guilty 10, proud→calm 12, happy→loving 11. Inspired is almost never confused (118/120).

### Held-out projection matrix, mlp_act mean/denoised

Rows: held-out story label. Columns: vector. Mean over tokens 50+, centred on the train emotional grand mean, in neutral-sd units.

```
         happy  inspi  lovin  proud   calm  despe  angry  guilt    sad  afrai  nervo  surpr
  happy   1.43   0.37   0.80   0.91   0.06  -0.64  -0.52  -0.38  -0.27  -0.89  -0.84  -0.00
inspire   0.29   1.76  -0.09   0.46   0.41   0.07  -0.71   0.01  -0.12  -0.81  -0.82  -0.42
 loving   0.62  -0.12   1.68   0.34   0.26  -0.64  -0.68  -0.06   0.35  -0.79  -0.76  -0.19
  proud   0.58   0.30   0.43   0.96   0.52  -0.49  -0.48  -0.45  -0.11  -0.60  -0.64   0.13
   calm   0.20   0.31   0.18   0.19   1.06  -0.51  -0.41  -0.26   0.26  -0.46  -0.19  -0.21
despera  -0.62   0.24  -0.69  -0.57  -0.25   1.59   0.12   0.22  -0.17   0.30   0.37  -0.67
  angry  -0.56  -0.64  -0.62  -0.41  -0.63   0.24   1.67   0.09  -0.60   0.45   0.32   0.47
 guilty  -0.59  -0.09  -0.23  -0.40  -0.14   0.19   0.13   1.23   0.34   0.06   0.09  -0.41
    sad  -0.28   0.10   0.05  -0.26   0.48  -0.11  -0.50   0.28   1.11  -0.04  -0.06  -0.47
 afraid  -0.79  -0.77  -0.75  -0.87  -0.15   0.33   0.43  -0.37  -0.03   1.69   1.17  -0.08
nervous  -0.70  -0.57  -0.79  -0.76   0.14   0.35   0.21  -0.10   0.02   0.91   1.48  -0.24
surpris   0.42  -0.41   0.16   0.14  -0.30  -0.73   0.17  -0.45  -0.33  -0.06  -0.23   1.45
neutral   0.42   1.18  -0.36   0.03   1.19  -0.10  -0.88  -0.04  -0.14  -0.69  -0.71   0.17
```

- Every row peaks on its own column. Proud (0.96) is the weakest diagonal, and its stories also read happy 0.58 and calm 0.52.
- resid_mid gives the same picture with smaller diagonals (0.89 proud to 1.52 inspired).

## Surprises

- **Neutral-mean centring cost ~0.25 accuracy.** Centred on the neutral mean, held-out accuracy drops to 0.46–0.63.
  - Grand-centred projections reproduce the pooled accuracy exactly (resid_mid raw 0.746 both ways), so this is a baseline effect, not a bug.
  - Every emotion vector partly points along "emotional story vs neutral story", which adds a different constant per column. The report centres on the emotional grand mean.
- **Neutral stories read as calm (1.19) and inspired (1.18)** against the emotional grand mean, and strongly not-angry (−0.88).
  - So part of the calm and inspired vectors is "less emotional than the average story", not the emotion itself.
  - Worth a contrast before Stage 3 uses calm or inspired neurons.
- **Story labels hold up.**
  - 192 emotional stories are mis-sorted at all four sites with the denoised vectors. One Haiku agent read the 40 with the largest margin, blind (two options, random order).
  - The label fits better: 28. The model's choice fits better: 8. Both fit: 4.
  - The 8: calm-t04-s5 (loving), happy-t09-s5 (loving), proud-t13-s2 (loving), surprised-t15-s5 (happy), sad-t19-s5 (afraid), happy-t07-s4 (loving), surprised-t12-s2 (happy), happy-t04-s1 (proud).
  - Both: happy-t06-s0, guilty-t13-s0, guilty-t13-s3, sad-t19-s1.
  - Kept as they are. They are mixed emotions within a cluster, not swapped files.
