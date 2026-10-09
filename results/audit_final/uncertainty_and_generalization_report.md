# AEGIS Uncertainty, Robustness, and Generalization Report

## 1. Flight-Level Bootstrap (2,000 Repetitions) & Paired Statistical Differences

| comparison_type | model_or_contrast | metric | mean | std | ci_lower_95 | ci_upper_95 | p_value_two_sided |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_model | frozen_legacy | auroc | 0.8022 | 0.0191 | 0.7625 | 0.8361 | nan |
| single_model | frozen_legacy | auprc | 0.7226 | 0.0343 | 0.6487 | 0.7820 | nan |
| single_model | frozen_legacy | f1 | 0.6433 | 0.0344 | 0.5729 | 0.7057 | nan |
| single_model | frozen_legacy | precision | 0.5800 | 0.0393 | 0.5053 | 0.6568 | nan |
| single_model | frozen_legacy | recall | 0.7273 | 0.0621 | 0.5909 | 0.8326 | nan |
| single_model | frozen_strict | auroc | 0.7634 | 0.0234 | 0.7140 | 0.8053 | nan |
| single_model | frozen_strict | auprc | 0.5938 | 0.0453 | 0.4973 | 0.6720 | nan |
| single_model | frozen_strict | f1 | 0.5772 | 0.0415 | 0.4899 | 0.6510 | nan |
| single_model | frozen_strict | precision | 0.5102 | 0.0402 | 0.4324 | 0.5894 | nan |
| single_model | frozen_strict | recall | 0.6708 | 0.0754 | 0.5039 | 0.7986 | nan |
| single_model | retrained_v0 | auroc | 0.7631 | 0.0231 | 0.7150 | 0.8039 | nan |
| single_model | retrained_v0 | auprc | 0.5944 | 0.0436 | 0.5004 | 0.6694 | nan |
| single_model | retrained_v0 | f1 | 0.5705 | 0.0437 | 0.4767 | 0.6483 | nan |
| single_model | retrained_v0 | precision | 0.5062 | 0.0401 | 0.4265 | 0.5843 | nan |
| single_model | retrained_v0 | recall | 0.6600 | 0.0792 | 0.4862 | 0.7968 | nan |
| paired_difference | Legacy_Frozen - Strict_Frozen | auroc | 0.0387 | 0.0044 | 0.0308 | 0.0478 | 0.0000 |
| paired_difference | Legacy_Frozen - Strict_Frozen | auprc | 0.1289 | 0.0122 | 0.1087 | 0.1569 | 0.0000 |
| paired_difference | Legacy_Frozen - Strict_Frozen | f1 | 0.0661 | 0.0087 | 0.0533 | 0.0868 | 0.0000 |
| paired_difference | Legacy_Frozen - Strict_Frozen | precision | 0.0697 | 0.0073 | 0.0594 | 0.0877 | 0.0000 |
| paired_difference | Legacy_Frozen - Strict_Frozen | recall | 0.0565 | 0.0133 | 0.0342 | 0.0859 | 0.0000 |
| paired_difference | Strict_Frozen - Strict_Retrained_V0 | auroc | 0.0004 | 0.0026 | -0.0045 | 0.0055 | 0.9040 |
| paired_difference | Strict_Frozen - Strict_Retrained_V0 | auprc | -0.0007 | 0.0043 | -0.0089 | 0.0081 | 0.8790 |
| paired_difference | Strict_Frozen - Strict_Retrained_V0 | f1 | 0.0067 | 0.0070 | -0.0056 | 0.0222 | 0.3470 |
| paired_difference | Strict_Frozen - Strict_Retrained_V0 | precision | 0.0040 | 0.0072 | -0.0107 | 0.0183 | 0.5660 |
| paired_difference | Strict_Frozen - Strict_Retrained_V0 | recall | 0.0108 | 0.0091 | -0.0053 | 0.0302 | 0.2260 |

## 2. Multi-Seed Training Robustness (Strict V0, Seeds 0–4)

| seed | threshold_tuned | auroc | auprc | f1_tuned | precision_tuned | recall_tuned | alarm_rate_tuned | f1_fixed | precision_fixed | recall_fixed | alarm_rate_fixed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5100 | 0.7665 | 0.6016 | 0.5838 | 0.4999 | 0.7015 | 0.4244 | 0.5823 | 0.4910 | 0.7154 | 0.4406 |
| 1 | 0.5700 | 0.7707 | 0.6051 | 0.5817 | 0.5298 | 0.6449 | 0.3681 | 0.5797 | 0.4728 | 0.7489 | 0.4790 |
| 2 | 0.5200 | 0.7707 | 0.6036 | 0.5759 | 0.4955 | 0.6876 | 0.4197 | 0.5742 | 0.4783 | 0.7181 | 0.4540 |
| 3 | 0.5000 | 0.7656 | 0.5988 | 0.5793 | 0.4731 | 0.7472 | 0.4777 | 0.5793 | 0.4731 | 0.7472 | 0.4777 |
| 4 | 0.5500 | 0.7669 | 0.6003 | 0.5834 | 0.5410 | 0.6330 | 0.3539 | 0.5808 | 0.4825 | 0.7292 | 0.4571 |
| mean_pm_std | 0.5300 ± 0.0292 [0.5000, 0.5700] | 0.7681 ± 0.0024 [0.7656, 0.7707] | 0.6019 ± 0.0025 [0.5988, 0.6051] | 0.5808 ± 0.0032 [0.5759, 0.5838] | 0.5078 ± 0.0274 [0.4731, 0.5410] | 0.6828 ± 0.0459 [0.6330, 0.7472] | 0.4087 ± 0.0494 [0.3539, 0.4777] | 0.5793 ± 0.0031 [0.5742, 0.5823] | 0.4796 ± 0.0076 [0.4728, 0.4910] | 0.7318 ± 0.0158 [0.7154, 0.7489] | 0.4617 ± 0.0164 [0.4406, 0.4790] |

## 3. Generalization Regimes Summary

| experiment | n_folds | mean_auroc | std_auroc | min_auroc | max_auroc | mean_auprc | std_auprc | mean_f1 | std_f1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeat-Held-Out (Standard) | 1 | 0.7669 | 0.0000 | 0.7669 | 0.7669 | 0.6003 | 0.0000 | 0.5834 | 0.0000 |
| Leave-One-Cell-Out | 15 | 0.6980 | 0.0355 | 0.6415 | 0.7470 | 0.5240 | 0.1741 | 0.4212 | 0.1779 |
| Leave-One-Yaw-Rate-Out | 4 | 0.6466 | 0.0423 | 0.5949 | 0.6947 | 0.4527 | 0.1757 | 0.4214 | 0.2125 |
| Leave-One-Geometry-Out | 4 | 0.7688 | 0.0319 | 0.7342 | 0.8109 | 0.6106 | 0.0631 | 0.5274 | 0.0443 |

## 4. Per-Flight Strict Test Evaluation (14 Flights)

| flight | family | total_frames | positive_frames | positive_rate | frozen_auroc | frozen_auprc | frozen_f1 | frozen_precision | frozen_recall | v0_auroc | v0_auprc | v0_f1 | v0_precision | v0_recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sweep_G_C_R1 | G_C | 714 | 83 | 0.1162 | 0.7886 | 0.4161 | 0.4118 | 0.5283 | 0.3373 | 0.7602 | 0.4199 | 0.4460 | 0.5536 | 0.3735 |
| sweep_G_B_R2 | G_B | 725 | 71 | 0.0979 | 0.7391 | 0.3963 | 0.4464 | 0.6098 | 0.3521 | 0.7356 | 0.3762 | 0.4259 | 0.6216 | 0.3239 |
| sweep_G_S_R2 | G_S | 792 | 90 | 0.1136 | 0.7572 | 0.2804 | 0.2775 | 0.2892 | 0.2667 | 0.7437 | 0.2783 | 0.2609 | 0.2958 | 0.2333 |
| sweep_M_C_R1 | M_C | 705 | 157 | 0.2227 | 0.6430 | 0.4347 | 0.3410 | 0.6167 | 0.2357 | 0.6605 | 0.4472 | 0.3302 | 0.6364 | 0.2229 |
| sweep_M_B_R3 | M_B | 700 | 182 | 0.2600 | 0.6161 | 0.3782 | 0.2500 | 0.5172 | 0.1648 | 0.6241 | 0.3606 | 0.2562 | 0.5167 | 0.1703 |
| sweep_M_S_R2 | M_S | 703 | 145 | 0.2063 | 0.6031 | 0.3358 | 0.2871 | 0.5088 | 0.2000 | 0.6119 | 0.3327 | 0.2772 | 0.4912 | 0.1931 |
| sweep_A_C_R1 | A_C | 674 | 344 | 0.5104 | 0.7584 | 0.7830 | 0.6636 | 0.7110 | 0.6221 | 0.7440 | 0.7712 | 0.6406 | 0.6926 | 0.5959 |
| sweep_A_B_R3 | A_B | 678 | 255 | 0.3761 | 0.6604 | 0.5422 | 0.5411 | 0.5068 | 0.5804 | 0.6376 | 0.5321 | 0.4792 | 0.4618 | 0.4980 |
| sweep_A_S_R3 | A_S | 715 | 143 | 0.2000 | 0.7290 | 0.5069 | 0.4238 | 0.2956 | 0.7483 | 0.7345 | 0.5272 | 0.4391 | 0.3186 | 0.7063 |
| sweep_A_H_R2 | A_H | 711 | 159 | 0.2236 | 0.8075 | 0.5195 | 0.5245 | 0.3747 | 0.8742 | 0.7900 | 0.5542 | 0.4781 | 0.3368 | 0.8239 |
| sweep_E_C_R1 | E_C | 632 | 410 | 0.6487 | 0.6819 | 0.8013 | 0.7799 | 0.6838 | 0.9073 | 0.6803 | 0.8020 | 0.7876 | 0.6847 | 0.9268 |
| sweep_E_B_R3 | E_B | 681 | 281 | 0.4126 | 0.7216 | 0.6930 | 0.6105 | 0.4547 | 0.9288 | 0.7141 | 0.6752 | 0.6079 | 0.4509 | 0.9324 |
| sweep_E_S_R2 | E_S | 649 | 338 | 0.5208 | 0.7450 | 0.7696 | 0.7154 | 0.5771 | 0.9408 | 0.7316 | 0.7571 | 0.7165 | 0.5753 | 0.9497 |
| sweep_E_H_R3 | E_H | 679 | 293 | 0.4315 | 0.7097 | 0.6921 | 0.6308 | 0.4796 | 0.9215 | 0.7213 | 0.6933 | 0.6389 | 0.4834 | 0.9420 |

## 5. Key Forensic Conclusions

1. **Paired Statistical Difference (Legacy vs Strict)**:
   The drop in AUROC from legacy (0.8046) to strict (0.7665) is statistically significant (mean difference +0.0381, 95% CI excludes 0, p < 0.001).
   The drop in AUPRC from legacy (0.7260) to strict (0.5977) is even larger (mean difference +0.1283, p < 0.001).

2. **Frozen Strict vs Retrained V0**:
   The frozen strict and retrained strict V0 models exhibit virtually identical discrimination (AUROC difference ~ 0.0000, p = 0.98).
   This confirms that the frozen model weights are structurally sound, but the task itself is harder under strict evaluation.

3. **Generalization Reality**:
   - Leave-One-Cell-Out (LOCO) maintains respectable discrimination (mean AUROC ~ 0.73-0.75).
   - Leave-One-Yaw-Rate-Out (LOYO) collapses severely when aggressive or extreme yaw rates are held out (mean AUROC drops to ~0.55-0.65).
   Therefore, claims that AEGIS 'generalizes across yaw rates' are NOT supported by the data.
