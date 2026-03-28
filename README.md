# Psychosis Classification with Static fMRI Functional Network Connectivity

**Bipolar Disorder vs Schizophrenia Classification**

A systematic evaluation of machine learning pipelines for differential diagnosis of schizophrenia (SZ) and bipolar disorder (BP) using static resting-state fMRI functional network connectivity (FNC) features, with a novel Bayesian Belief Network classifier for interpretable probabilistic inference.

---

## Overview

This project was based on [2023 IEEE Signal Processing Cup Kaggle competition](https://www.kaggle.com/competitions/psychosis-classification-with-rsfmri/overview) on psychosis classification. We evaluate six classifier architectures with multiple feature selection strategies under rigorous nested cross-validation, and introduce a BBN classifier as an interpretable extension.

**Best result:** RBF SVM + 300 MI-DISR features + C=5 → **65.05% ± 6.05%** accuracy, AUC = 0.645 (5×10 repeated CV)

---

## Dataset

- **Source:** Kaggle competition: psychosis classification with rsfMRI
- **Training:** 471 subjects (183 BP, 288 SZ)
- **Test:** 315 unlabeled subjects
- **Features:** Pre-extracted static FNC: 5,460 pairwise Pearson correlations between 105 ICNs
- **Each subject folder contains:**
  - `fnc.npy` — static FNC vector (5,460 features)
  - `icn_tc.npy` — ICN time courses (105 components)

Data is not included in this repository. Downloaded from the [Kaggle competition page](https://www.kaggle.com/competitions/psychosis-classification-with-rsfmri/data).

---

## Methods

### Feature Selection
| Method | Applied To | Feature Counts |
|---|---|---|
| MI-DISR (redundancy-aware) | Linear SVM, RBF SVM | k ∈ {100, 200, 300, 400} |
| Simple top-k MI | Gradient Boosting, XGBoost, BBN | k = 400 (tree models), k = 20 (BBN) |
| None | Random Forest | All 5,460 features |

### Classifiers
- Linear SVM
- RBF SVM (with nested hyperparameter tuning: C, γ)
- Random Forest (500 trees)
- Gradient Boosting
- XGBoost
- Soft-voting Ensemble (RBF SVM + RF + GB)
- **Bayesian Belief Network** — novel contribution, first application to SZ vs BP FNC classification

### Evaluation
- 10-fold stratified cross-validation (all models)
- 5×10 repeated CV for final model (50 evaluations)
- Nested feature selection inside each fold — no data leakage
- Balanced class weights throughout

---

## Results

| Model | Features | Accuracy | AUC |
|---|---|---|---|
| **RBF SVM (MI-DISR)** | **300** | **64.55% ± 4.32%** | **0.662** |
| RBF SVM | 5,460 | 63.28% ± 3.22% | 0.656 |
| Ensemble (RBF+RF+GB) | Mixed | 63.93% ± 3.90% | 0.664 |
| Random Forest | 5,460 | 63.07% ± 2.97% | 0.626 |
| Gradient Boosting | 400 | 62.85% ± 3.05% | 0.648 |
| Linear SVM | 5,460 | 62.42% ± 6.38% | 0.633 |
| XGBoost | 400 | 62.22% ± 4.03% | 0.656 |
| BBN | 20 | 59.89% ± 5.87% | — |

**Final model (5×10 repeated CV):** 65.05% ± 6.05%, AUC = 0.645, 95% CI: [63.33%, 66.77%]

---

## Repository Structure

```
.
├── EDA_notebook.ipynb             # Exploratory data analysis
├── Final_model_code.ipynb         # All model blocks (2A–BBN)
├── figures/
│   ├── fig2_feature_count.png
│   ├── fig3_model_comparison.png
│   ├── fig5_repeated_cv.png
│   └── bbn_network_domain.png
├── saved_results/
│   ├── block4_final_5x10cv_detailed.csv
│   ├── block4_final_5x10cv_summary.csv
│   └── bbn_dag_results.csv
└── README.md
```

---

## Notebook Blocks

| Block | Description |
|---|---|
| Block 1 | Data loading and statistics |
| Block 2A | Linear SVM — all 5,460 features |
| Block 2B | RBF SVM — all 5,460 features |
| Block 2C | Random Forest — all 5,460 features |
| Block 2E1–2E4 | Linear SVM + MI-DISR (k=100,200,300,400) |
| Block 2F | RBF SVM + MI-DISR (k=100,200,300,400) |
| Block 2G | Gradient Boosting + XGBoost (nested CV, 400 MI) |
| Block 2H | Ensemble (RBF SVM + RF + GB, soft voting) |
| Block 3 | Hyperparameter tuning — RBF SVM (nested grid search) |
| Block 3A | Statistical significance test — tuned vs default |
| Block 4 | Final 5×10 repeated CV — optimised RBF SVM |
| BBN Block | Bayesian Belief Network — 20 consensus MI features |

---

## Key Findings

- **RBF kernel outperforms linear** across all feature configurations. Mild non-linearity in the class boundary
- **MI-DISR at k=300** is the optimal feature count for RBF SVM. k=100 collapses below baseline
- **Ensemble provides no accuracy gain** over the best single model. Component models share similar error patterns
- **Signal is distributed** across many FNC pairs. No dominant discriminative axis, explaining why aggressive feature reduction hurts
- **BBN** achieves lower accuracy (59.89%) but reveals interpretable conditional dependencies. PL-HC is the most stable feature (selected in 10/10 folds)

---

## Requirements

```
numpy
pandas
scikit-learn
xgboost
pgmpy
networkx
matplotlib
scipy
```

Install with:
```bash
pip install numpy pandas scikit-learn xgboost pgmpy networkx matplotlib scipy
```

---

## Authors

- **Renugambal Lakshmikanthan** — Georgia State University
- **Harshita Karmungi** — Georgia State University

---

## References

1. Rashid et al. (2016): Classification of schizophrenia and bipolar patients using static and dynamic resting-state fMRI brain connectivity. *NeuroImage*, 134, 645–657.
2. Meyer & Bontempi (2006): On the use of variable complementarity for feature selection in cancer classification. *Applications of Evolutionary Computing*.
3. Mumford & Ramsey (2014): Bayesian networks for fMRI: A primer. *NeuroImage*, 86, 573–582.
4. Orrù et al. (2012): Using Support Vector Machine to identify imaging biomarkers of neurological and psychiatric disease. *Neuroscience & Biobehavioral Reviews*, 36, 1140–1152.
