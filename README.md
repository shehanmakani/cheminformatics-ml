# Cheminformatics & ML Notebooks

**Author:** Shehan Makani — Founder & CEO, [ChemeNova LLC](https://chemenova.com) | NJIT Tech MBA  
**Kaggle:** [kaggle.com/shehanmakani](https://www.kaggle.com/shehanmakani)  
**LinkedIn:** [linkedin.com/in/shehanmakani](https://linkedin.com/in/shehanmakani)

---

Applied machine learning for chemistry and materials science. Each notebook combines real domain knowledge with rigorous modeling — the kind of work that matters in specialty chemical manufacturing, pharmaceutical formulation, and agrochemical development.

This is not tutorial ML. It's what happens when a chemical engineer who actually formulates products uses modern ML properly.

---

## Notebooks

| # | Topic | Type | Best Result | Key Finding |
|---|---|---|---|---|
| [01](#01--aqueous-solubility-prediction) | Aqueous Solubility (ESOL) | Regression | XGBoost RMSE=0.524, R²=0.892 | 55% better than Delaney 2004 formula |
| [02](#02--boiling-point-prediction) | Boiling Point | Regression | XGBoost RMSE=28.5°C, R²=0.835 | BertzCT dominates — not LogP |
| [03](#03--logp-prediction) | LogP Prediction | Regression | ElasticNet RMSE=0.750, R²=0.835 | RDKit Crippen edges out ML — explains why |
| [04](#04--ames-mutagenicity) | Ames Mutagenicity | Classification | LR AUC=0.987, Recall=0.909 | Simplest model wins; nitro groups = 100% mutagenic |

---

## 01 — Aqueous Solubility Prediction

**Notebook:** `notebooks/01_solubility_esol/solubility_prediction.ipynb`  
**Dataset:** 99 compounds, 19 RDKit descriptors, logS range −7.0 to +1.4

Predicts aqueous solubility (logS, mol/L) from SMILES. Benchmarked against the Delaney (2004) linear formula — the canonical 20-year-old baseline.

**Results:**

| Model | Test RMSE | Test R² |
|---|---|---|
| Ridge | 0.886 | 0.690 |
| ElasticNet | 0.813 | 0.739 |
| Random Forest | 0.546 | 0.882 |
| Gradient Boosting | 0.555 | 0.878 |
| **XGBoost** | **0.524** | **0.892** |
| LightGBM | 0.644 | 0.836 |
| *Delaney formula* | *1.152* | *0.547* |

**Key findings:**
- LogP accounts for 63% of XGBoost feature importance — the thermodynamic driver of water exclusion
- PAHs define the insolubility floor — pyrene (logS=−5.18) due to high LogP + zero heteroatom content
- Non-linear models outperform linear because solubility is non-additive (LogP × ring interactions)

---

## 02 — Boiling Point Prediction

**Notebook:** `notebooks/02_boiling_point/boiling_point_prediction.ipynb`  
**Dataset:** 181 compounds across 11 chemical classes, BP range −162 to +365°C

Predicts normal boiling point (°C at 1 atm). Proper train/val/test split (70/15/15). SHAP interpretability on all predictions.

**Results:**

| Model | Test RMSE (°C) | Test R² |
|---|---|---|
| Ridge | 30.9 | 0.806 |
| ElasticNet | 31.4 | 0.799 |
| Random Forest | 32.4 | 0.785 |
| Gradient Boosting | 29.4 | 0.824 |
| **XGBoost** | **28.5** | **0.835** |
| LightGBM | 39.7 | 0.678 |
| *MW-only baseline* | *47.7* | *0.536* |

**Key findings:**
- BertzCT (molecular complexity) is the top SHAP driver — not LogP. Opposite of solubility.
- More complex molecules have more surface area for van der Waals interactions → higher BP
- H-bond donors elevate BP significantly above non-polar compounds of the same MW

---

## 03 — LogP Prediction

**Notebook:** `notebooks/03_logp/logp_prediction.ipynb`  
**Dataset:** 119 compounds, experimental LogP −3.70 to +8.74 (Hansch & Leo 1979, NIST)

Predicts experimental LogP from structural descriptors alone — explicitly excluding RDKit's built-in Crippen value. The real question: can structural descriptor ML beat a purpose-built atom-contribution model?

**Results:**

| Model | Test RMSE | Test R² |
|---|---|---|
| Ridge | 0.770 | 0.826 |
| **ElasticNet** | **0.750** | **0.835** |
| Random Forest | 0.826 | 0.800 |
| Gradient Boosting | 0.799 | 0.812 |
| XGBoost | 0.778 | 0.822 |
| LightGBM | 0.900 | 0.762 |
| *RDKit Crippen* | *0.647* | *0.877* |

**Key findings:**
- RDKit Crippen wins — atom-contribution models encode exact per-fragment contributions ML can't replicate at this sample size
- MolMR (molar refractivity = electron polarizability) is the top SHAP driver — the physical basis of octanol preference
- ML is competitive in the 1–3 LogP range where most drug molecules sit

---

## 04 — Ames Mutagenicity

**Notebook:** `notebooks/04_ames_mutagenicity/ames_mutagenicity.ipynb`  
**Dataset:** 122 compounds (53 mutagens, 69 non-mutagens) — Hansen et al. 2009, Kazius et al. 2005

Predicts Ames test mutagenicity (binary classification). Structural alert feature engineering, SMOTE for class imbalance, ROC-AUC + Precision-Recall evaluation, ICH M7 regulatory context.

**Results:**

| Model | CV AUC | Test AUC | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.902** | **0.987** | **0.909** | **0.909** |
| SVM | 0.980 | 0.968 | 0.727 | 0.800 |
| XGBoost | 0.980 | 0.935 | 0.909 | 0.870 |
| Random Forest | 0.953 | 0.922 | 0.818 | 0.818 |
| *Majority baseline* | — | *0.500* | *0.000* | — |

**Structural alert rates:**

| Alert | Mutagenicity rate when present |
|---|---|
| Nitro group (−NO₂) | 100% |
| Nitroso (−N=O) | 100% |
| Aromatic amine (Ar−NH₂) | 30% |
| Alkyl halide (C−X) | 43% |

**Key findings:**
- Logistic Regression wins — mutagenicity is driven by a small number of structural features that linear boundaries capture well
- Recall is the right metric in toxicology: a missed mutagen is far more dangerous than a false positive
- Nitro and nitroso groups are 100% predictive — they directly electrophilically attack DNA
- ICH M7 requires computational mutagenicity screening for all drug impurities above 1.5 µg/day

---

## The Pattern Across All Four

| Property | Top SHAP driver | Physical meaning |
|---|---|---|
| Solubility | LogP | Hydrophobicity = thermodynamic water exclusion |
| Boiling point | BertzCT | Structural complexity = intermolecular interaction richness |
| LogP | MolMR | Electron polarizability = octanol preference |
| Mutagenicity | NumNitrogens / alerts | Electrophilic reactivity = DNA attack capacity |

These are not arbitrary ML correlations. Each top feature has a mechanistic explanation rooted in physical organic chemistry.

---

## Stack

```
rdkit              # molecular descriptor computation
xgboost            # gradient boosting
lightgbm           # fast gradient boosting
scikit-learn       # models, cross-validation, metrics
imbalanced-learn   # SMOTE for class imbalance
shap               # model interpretability
pandas / numpy     # data handling
matplotlib         # visualization
```

Full dependency list: `requirements.txt`

---

## Repo Structure

```
cheminformatics-ml/
├── notebooks/
│   ├── 01_solubility_esol/
│   │   ├── solubility_prediction.ipynb
│   │   ├── data/esol_processed.csv
│   │   └── figures/
│   ├── 02_boiling_point/
│   │   ├── boiling_point_prediction.ipynb
│   │   ├── data/bp_processed.csv
│   │   └── figures/
│   ├── 03_logp/
│   │   ├── logp_prediction.ipynb
│   │   ├── data/logp_processed.csv
│   │   └── figures/
│   └── 04_ames_mutagenicity/
│       ├── ames_mutagenicity.ipynb
│       ├── data/ames_processed.csv
│       └── figures/
├── requirements.txt
└── .gitignore
```

Each notebook folder is self-contained — data, figures, and notebook all together.

---

## Background

I run **ChemeNova LLC** — an AI-driven specialty chemical formulation company building [IntelliForm™](https://chemenova.com/intelliform), a formulation intelligence platform. I also operate **ChemRich Global** (ChemRich USA + ChemRich India), a specialty chemicals distribution business with products including D-Limonene, Calcium Chloride Dihydrate, and IPA 99%.

These notebooks are the public-facing side of the ML work that underpins IntelliForm™. The domain expertise is real — I formulate for personal care, industrial, and agrochemical applications. When I say LogP dominates solubility screening, that is the decision I make before choosing a surfactant system.

---

## References

- Delaney, J.S. (2004). ESOL. *J. Chem. Inf. Comput. Sci.*, 44(3), 1000–1005.
- Hansen, K. et al. (2009). Ames benchmark dataset. *J. Chem. Inf. Model.*, 49(9), 2077–2081.
- Kazius, J. et al. (2005). Toxicophores for mutagenicity. *J. Med. Chem.*, 48(1), 312–320.
- Wildman & Crippen (1999). Crippen LogP. *J. Chem. Inf. Comput. Sci.*, 39, 868–873.
- Lundberg & Lee (2017). SHAP. *NeurIPS.*
- RDKit: https://www.rdkit.org

---

*If a notebook was useful, a GitHub star or Kaggle upvote keeps this going.*
