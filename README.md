# Fraud Detection in Online Transactions (PaySphere)

End-to-end machine learning pipeline for scoring online transaction fraud risk using leak-free point-in-time behavioral features, chronological evaluation, Platt probability calibration, and tiered risk routing[cite: 6].

## Overview
* **Dataset:** 50,000 online transactions across 25 raw columns (`10.06%` sample fraud prevalence, 0 duplicate rows)[cite: 6].
* **Leak-Free Feature Engineering:** Computes rolling transaction velocity (`5m`, `1h`, `24h`) and entity familiarity strictly from prior timestamps, verified by an automated future-prefix invariance check[cite: 6].
* **Chronological Split:** Time-ordered **Train (55%) → Calibration (15%) → Validation (15%) → Test (15%)** partitions without splitting simultaneous transactions[cite: 6].
* **Calibration & Action Policy:** Calibrates classifier probabilities via Platt scaling and evaluates both a cost-constrained binary threshold (`≤10%` validation FPR cap) and tiered operational actions (`Auto-approve`, `OTP challenge`, `Analyst review`)[cite: 6].

## Validation Model Comparison

| Model | Validation PR-AUC (AP) | Validation ROC-AUC | Calibrated Brier | Raw Brier |
| :--- | :---: | :---: | :---: | :---: |
| **Histogram Gradient Boosting** *(Selected)* | **0.2168** | 0.6960 | **0.0902** | 0.2156 |
| Logistic Regression | 0.2156 | **0.6963** | 0.0907 | 0.2170 |
| Random Forest | 0.2125 | 0.6916 | 0.0905 | 0.1950 |

## Holdout Test Results (7,500 Transactions | 756 Fraud Cases)

* **ROC-AUC:** `0.7092`[cite: 6]
* **Average Precision (PR-AUC):** `0.2255`[cite: 6]
* **Calibrated Brier Score:** `0.0847`[cite: 6]
* **Constrained Binary Baseline (`threshold = 0.235`):** `27.98%` precision, `32.28%` recall, `9.31%` false-positive rate[cite: 6]
* **Batch Scoring Latency:** `10.56 ms` median for 1,000 rows (`0.0106 ms`/row amortized)[cite: 6]

### Tiered Operational Routing Impact

| Proposed Action | Transactions | Traffic Share | Fraud Cases | Observed Fraud Rate | Fraud Exposure (INR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto-approve** | 6,416 | 85.55% | 469 | 7.31% | 4,717,590.26 |
| **OTP challenge** | 867 | 11.56% | 213 | 24.57% | 2,267,501.25 |
| **Analyst review** | 217 | 2.89% | 74 | 34.10% | 771,824.56 |

* Routes **14.45%** of test traffic (`1,084` transactions) to capture **37.96%** of fraud cases (`287 / 756`) and **39.18%** of fraud value (`INR 3.04M`), while auto-approving **88.18%** of genuine transactions[cite: 6].

## Top Risk Drivers (Validation Permutation Importance)
1. `combined_risk_index` (`0.1026`)[cite: 6]
2. `ip_address_risk_score` (`0.0056`)[cite: 6]
3. `dual_shift_flag` (`0.0038`)[cite: 6]
4. `device_change_flag` (`0.0038`)[cite: 6]
5. `otp_success_rate_customer` (`0.0012`)[cite: 6]

## Repository Contents & Usage
* `PaySphere_Fraud_Detection.ipynb` — Main executed Jupyter notebook.
* `transactions_fraud.csv` — Transaction dataset[cite: 6, 8].
* `Data Definition (1).xlsx` — Data dictionary[cite: 6, 8].

Run the notebook top to bottom in an environment with `pandas`, `numpy`, `matplotlib`, `scikit-learn>=1.2`, `openpyxl`, and `joblib`[cite: 6]. Generated reports, scored outputs, and the serialized model bundle are written to `paysphere_results/` upon execution[cite: 6].
