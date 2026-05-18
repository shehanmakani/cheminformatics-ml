# Cheminformatics & ML Notebooks

**Author:** Shehan Makani — Co-Founder & CEO, [ChemeNova LLC](https://chemenova.com) | NJIT Tech MBA  
**Kaggle:** [kaggle.com/shehanmakani](https://www.kaggle.com/shehanmakani) ← upvotes appreciated  
**LinkedIn:** [linkedin.com/in/shehanmakani](https://linkedin.com/in/shehanmakani)

---

Applied machine learning for chemistry and materials science. Each notebook combines real domain knowledge with rigorous modeling — the kind of work that matters in specialty chemical manufacturing, pharmaceutical formulation, and agrochemical development.

This repo is not tutorial ML. It's what happens when a chemical engineer who actually formulates products uses modern ML properly.

---

## Notebooks

| # | Topic | Key Result | Kaggle |
|---|---|---|---|
| 01 | [Aqueous Solubility Prediction (ESOL)](#01-aqueous-solubility-prediction) | XGBoost RMSE=0.524, R²=0.892 — 55% improvement over Delaney 2004 | [→ Kaggle](https://www.kaggle.com) |
| 02 | Boiling Point Prediction | *Coming soon* | — |
| 03 | Toxicity / QSAR (Ames test) | *Coming soon* | — |
| 04 | LogP Prediction from SMILES | *Coming soon* | — |
| 05 | Formulation Stability Classification | *Coming soon* | — |

---

## 01 — Aqueous Solubility Prediction

**File:** `notebooks/01_solubility_esol/solubility_prediction.ipynb`  
**Dataset:** `datasets/esol_processed.csv` (99 compounds, 19 RDKit descriptors)

### Problem
Predict the aqueous solubility (logS) of organic compounds directly from molecular structure (SMILES). Solubility determines bioavailability, environmental fate, and formulation feasibility — measuring it experimentally for every candidate is slow and expensive. A fast, accurate QSAR model fills this gap.

### Approach
- **Descriptors:** 19 molecular features computed with [RDKit](https://www.rdkit.org) — LogP, MW, HBD/HBA, TPSA, topological indices, ring counts, sp³ fraction
- **Models:** Ridge, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Validation:** 5-fold cross-validation + held-out test set (80/20 split)
- **Baseline:** Delaney (2004) linear formula — the 20-year-old benchmark everyone knows

### Results

| Model | Test RMSE | Test R² |
|---|---|---|
| Ridge | 0.886 | 0.690 |
| ElasticNet | 0.813 | 0.739 |
| Random Forest | 0.546 | 0.882 |
| Gradient Boosting | 0.555 | 0.878 |
| **XGBoost** | **0.524** | **0.892** |
| LightGBM | 0.644 | 0.836 |
| *Delaney formula (baseline)* | *1.152* | *0.547* |

XGBoost achieves **55% lower RMSE** than the Delaney formula with R² = 0.892.

### Key Findings
- LogP accounts for **63% of XGBoost feature importance** — the primary thermodynamic driver of solubility, consistent with physical organic chemistry
- PAHs (polycyclic aromatics) define the insolubility floor — pyrene hits logS = −5.18 due to high LogP + zero heteroatom content
- Gradient boosted trees outperform linear models because solubility is non-additive — LogP × AromaticRings interactions are real
- Delaney's 2004 formula is still half the answer — modern ML gets the other half

### Figures

| Model Comparison | Predicted vs Actual |
|---|---|
| ![Model Comparison](../../figures/fig1_model_comparison.png) | ![Predictions](../../figures/fig2_predictions.png) |

| Feature Importance | Chemical Space |
|---|---|
| ![Features](../../figures/fig3_features.png) | ![Space](../../figures/fig4_chemical_space.png) |

---

## Stack

```
rdkit          # molecular descriptor computation
xgboost        # gradient boosting
lightgbm       # fast gradient boosting
scikit-learn   # baseline models, cross-validation
pandas / numpy # data handling
matplotlib     # visualization
```

Full dependency list: `requirements.txt`

---

## Background

I run **ChemeNova LLC** — an AI-driven specialty chemical formulation company building [IntelliForm™](https://chemenova.com/intelliform), a formulation intelligence platform. I also operate **ChemRich Global** (ChemRich USA + ChemRich India), a specialty chemicals distribution business.

These notebooks are the public-facing side of the ML work that underpins IntelliForm™. The domain expertise behind them is real: I formulate for personal care, industrial, and agrochemical applications. When I say "LogP dominates solubility in formulation screening," that's not textbook recitation — it's the decision I make before choosing a surfactant system.

---

## References

- Delaney, J.S. (2004). ESOL: Estimating aqueous solubility directly from molecular structure. *J. Chem. Inf. Comput. Sci.*, 44(3), 1000–1005.
- Sorkun, M.C. et al. (2019). AqSolDB: A curated reference set of aqueous solubility. *Scientific Data*, 6, 143.
- RDKit: Open-source cheminformatics software. https://www.rdkit.org
- Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.

---

*If a notebook helped you, a GitHub star or Kaggle upvote keeps this going.*
