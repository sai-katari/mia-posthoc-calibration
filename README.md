# Does Post-hoc Temperature Scaling Reduce Membership Leakage?

Follow-on to [Project 1](https://github.com/sai-katari/medical-membership-privacy),
which showed that fully fine-tuned ResNet-18 models on DermaMNIST leak membership
information (loss-MIA AUROC = 0.699). At 1% FPR, both loss and entropy attacks
produce approximately 3.4% TPR after correcting for float32 output precision
(see Project 1 for details).

The question here is whether temperature scaling -- applied after training, no
retraining needed -- removes that membership information or reduces it in any
meaningful way.

## Research question

Does post-hoc temperature scaling reduce membership leakage, and is any reduction
robust to adaptive attackers?

## What I tested

Temperature scaling fits a scalar T on the validation set that divides the logits
before softmax. Higher T produces softer, higher-entropy predictions. It takes a
few seconds to apply and does not change the model's argmax predictions.

Two attacker types are included in the main evaluation:

| Attacker | Knows calibration was applied? | Knows T? | Method |
|----------|-------------------------------|----------|--------|
| Naive | No | No | Standard loss/confidence/entropy attacks on scaled outputs |
| Adaptive | Yes | Yes | Known-T reconstruction via log-space inversion, then standard attacks |

We considered a semi-adaptive attacker that estimates T from shadow samples
without knowing it directly. The NLL-based T estimation procedure is
non-identifiable: the scaled output is already approximately NLL-optimal on
data from the same distribution, so the estimator finds T near 1 regardless
of the true T. Semi-adaptive results are therefore not reported.

Known-T reconstruction via log-space inversion is algebraically exact for
complete, unrounded probability vectors. Reconstruction from stored float32
probabilities is approximate; the float64 synthetic reconstruction test
serves as the exact sanity check. Under rounding, quantization, top-k
truncation, or label-only output, reconstruction may no longer be possible.
Evaluating membership privacy under those restricted-output settings requires
separate attack models and is left for future work.

## Setup

```bash
pip install -r requirements.txt
```

At the start of each session, restore the DermaMNIST cache:

```python
import shutil, os
os.makedirs("/root/.medmnist", exist_ok=True)
shutil.copy("/your/drive/dermamnist_224.npz", "/root/.medmnist/dermamnist_224.npz")
```

## Running the pipeline

```bash
python scripts/collect_baseline.py     --config configs/baseline.yaml --p1_dir /path/to/project1
python scripts/fit_temperature.py      --config configs/baseline.yaml --p1_dir /path/to/project1
python scripts/run_naive_attacks.py    --config configs/baseline.yaml
python scripts/run_adaptive_attacks.py --config configs/baseline.yaml
python scripts/analyze_results.py      --config configs/baseline.yaml
```

Experiment artifacts are committed under this structure:

    experiments/dermamnist_resnet18_{regime}_seed{N}/
        calibration_baseline.json
        calibration_temp_scaled.json
        attack_results_baseline.json
        attack_results_naive_scaled.json
        attack_results_adaptive.json
    results/
        defense_comparison.csv
        adaptive_attacks.csv
        naive_scaled_attacks.csv

Large per-sample output files are not committed; regenerate them with the
pipeline above.

## Results

### Temperature values

T fitted by minimising NLL on the validation set only:

| Regime | T (mean +/- SD, 3 seeds) |
|--------|--------------------------|
| Frozen | 1.012 +/- 0.015 |
| Scratch | 1.107 +/- 0.044 |
| Partial FT | 1.535 +/- 0.152 |
| Full FT | 1.773 +/- 0.038 |

The fitted temperature increased across progressively more heavily fine-tuned
regimes, mirroring the leakage ordering from Project 1.

![Temperature by regime](plots/plot2_temperature_by_regime.png)

### Calibration

Temperature scaling improved ECE in scratch, partial-FT, and full-FT models
while leaving frozen-model calibration essentially unchanged:

| Regime | ECE before | ECE after | delta ECE |
|--------|-----------|-----------|-----------|
| Frozen | 0.018 | 0.018 | 0.000 |
| Scratch | 0.037 | 0.021 | -0.016 |
| Partial FT | 0.080 | 0.041 | -0.039 |
| Full FT | 0.088 | 0.050 | -0.038 |

### Loss MIA AUROC (mean +/- sample SD, 3 seeds)

| Regime | Baseline | Naive (scaled) | Adaptive (known T) |
|--------|----------|----------------|--------------------|
| Frozen | 0.516 +/- 0.001 | 0.516 +/- 0.001 | 0.516 +/- 0.001 |
| Scratch | 0.525 +/- 0.013 | 0.525 +/- 0.013 | 0.525 +/- 0.013 |
| Partial FT | 0.679 +/- 0.045 | 0.679 +/- 0.044 | 0.679 +/- 0.045 |
| Full FT | 0.699 +/- 0.002 | 0.700 +/- 0.002 | 0.699 +/- 0.002 |

### Entropy MIA AUROC and TPR @ 1% FPR

| Regime | Entropy AUC base | Entropy AUC scaled | TPR@1% base | TPR@1% scaled | Paired delta |
|--------|-----------------|-------------------|-------------|---------------|--------------|
| Frozen | 0.504 +/- 0.001 | 0.504 +/- 0.001 | 0.010 +/- 0.001 | 0.010 +/- 0.001 | -0.000 +/- 0.000 |
| Scratch | 0.510 +/- 0.007 | 0.510 +/- 0.006 | 0.010 +/- 0.001 | 0.010 +/- 0.001 | +0.000 +/- 0.001 |
| Partial FT | 0.675 +/- 0.048 | 0.674 +/- 0.048 | 0.015 +/- 0.003 | 0.017 +/- 0.003 | +0.001 +/- 0.000 |
| Full FT | 0.697 +/- 0.002 | 0.698 +/- 0.001 | 0.034 +/- 0.003 | 0.036 +/- 0.005 | +0.002 +/- 0.003 |

![Defense comparison](plots/plot1_defense_comparison.png)

### What the results show

Temperature scaling substantially improves calibration but does not materially
reduce membership leakage. The naive loss MIA AUC is essentially unchanged after
scaling across all four regimes -- the largest movement is full FT going from
0.699 to 0.700. Entropy attacks tell the same story: the paired TPR@1% change for
full FT is +0.002 +/- 0.003 across three seeds, indicating no consistent
reduction across runs.

The known-T adaptive attack recovered approximately the original AUC in all
cases. For full FT the adaptive AUC is 0.699 vs the baseline 0.699. Membership
information is preserved in the scaled distribution and is recoverable given T
and full-precision outputs.

Improved calibration and improved membership privacy are not equivalent properties.

## Sanity checks

- Known-T reconstruction: max reconstruction error < 4e-16 in float64 synthetic
  tests; on stored float32 outputs the error is approximately 3e-7
- Sample ordering: baseline and scaled attack pools have identical ordering (all 12 runs)
- Calibration metrics computed on 2005 non-members (test set) only
- ddof=1 throughout (verified by static grep)
- TPR computed consistently via roc_curve with fpr <= target_fpr

## Limitations

Known-T reconstruction relies on receiving a complete, full-precision probability
vector and knowing T. In deployed systems that round or quantize probabilities,
truncate outputs to top-k predictions, or provide only predicted labels,
reconstruction may no longer be possible. Evaluating membership privacy under
those restricted-output settings requires separate attack models and is left for
future work.

The semi-adaptive estimator has a structural identifiability problem, not a
tuning problem. NLL minimisation on honestly labeled shadow data from the same
distribution prefers the already-calibrated output, so T_hat converges near 1
regardless of T_true. A proper implementation would require disjoint shadow
data and independently trained shadow models with known membership splits.

This study covers one dataset (DermaMNIST) and one architecture (ResNet-18).
The conclusion that temperature scaling does not provide measurable membership
privacy protection applies to this experimental setting. Different training
protocols, regularization, or hyperparameter choices could change the
privacy-utility comparison.

The loss-based attacker is assumed to know the true class label. All attacks
receive the complete float32 probability vector produced by the model. Before
computing attack scores, probability vectors are converted to float64 and
renormalized. This reduces artificial ties caused by float32 softmax
saturation while preserving the probability-output black-box threat model.

## Connection to Project 1

Project 1 asked which fine-tuning regimes leak membership information and why.
Project 2 asks whether post-hoc calibration reduces that leakage.

The entropy-attack finding from Project 1 motivated this study. The entropy
results here show that temperature scaling does not reduce membership-relevant
entropy signal in any meaningful way: entropy TPR at 1% FPR changes by
+0.002 +/- 0.003 for full FT, providing no evidence of a systematic reduction.

## Related work

Chen and Pattabiraman (NDSS 2024) mitigate membership inference by enforcing
less confident predictions during training (HAMP). The finding here is the
post-hoc counterpart: softening confidence after training improves calibration
but leaves the membership ordering intact, and an attacker with T recovers the
original leakage exactly. This motivates a broader comparison between
training-time defenses such as HAMP and post-hoc calibration. The present
study evaluates only post-hoc temperature scaling and does not establish that
defense timing alone explains the difference.

## References

Guo, C. et al. On Calibration of Modern Neural Networks. ICML, 2017.

Shokri, R. et al. Membership Inference Attacks Against Machine Learning
Models. IEEE S&P, 2017.

Yeom, S. et al. Privacy Risk in Machine Learning: Analyzing the Connection
to Overfitting. CSF, 2018.

Carlini, N. et al. Membership Inference Attacks From First Principles.
IEEE S&P, 2022.

Chen, Z. and Pattabiraman, K. Overconfidence is a Dangerous Thing:
Mitigating Membership Inference Attacks by Enforcing Less Confident
Prediction. NDSS, 2024.

Yang, J. et al. MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D
and 3D Biomedical Image Classification. Scientific Data, 2023.

## Author

Sai Katari -- M.S. in Computer Science, University of Kansas.
GitHub: [sai-katari](https://github.com/sai-katari)

## License

MIT -- see [LICENSE](LICENSE).
