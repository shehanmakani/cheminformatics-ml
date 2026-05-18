# 01 — Aqueous Solubility Prediction (ESOL Benchmark)

**Kaggle:** [View notebook →](https://www.kaggle.com/shehanmakani)  
**Status:** Published

---

## Summary

Predicts aqueous solubility (logS, mol/L) from SMILES using 19 RDKit molecular descriptors and six ML algorithms. Benchmarked against the Delaney (2004) linear formula — the canonical 20-year-old baseline.

**Best result:** XGBoost, RMSE = 0.524, R² = 0.892 — 55% improvement over Delaney.

---

## Files

| File | Description |
|---|---|
| `solubility_prediction.ipynb` | Full notebook — run top to bottom |
| `../../datasets/esol_processed.csv` | 99-compound dataset with 19 computed descriptors |
| `../../figures/fig*.png` | Pre-rendered figures |

---

## How to Run

**Locally:**
```bash
pip install -r ../../requirements.txt
jupyter notebook solubility_prediction.ipynb
```

**On Kaggle:**
- Upload `solubility_prediction.ipynb` as a new notebook
- Attach `esol_processed.csv` as a dataset
- Runtime: Python 3, GPU not needed, runs in ~2 min

---

## Dependencies

```
rdkit >= 2023.3
xgboost >= 1.7
lightgbm >= 4.0
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
```

---

## Dataset

99 diverse organic compounds spanning:
- Monoaromatics (benzene derivatives, phenols, nitro/amino)
- Polycyclic aromatic hydrocarbons (naphthalene → pyrene)
- Heteroaromatics (pyridine, quinoline, indole, thiophene)
- Aliphatics (alkanes, alcohols, ketones, acids)
- Pharmaceuticals (aspirin, ibuprofen, caffeine, acetaminophen)
- Amino acids and sugars
- Agrochemicals (atrazine, lindane)
- Solvents and halogenated compounds

logS range: −7.0 (decane) to +1.41 (methanol)

Sources: Delaney (2004), AqSolDB, PubChem experimental data

---

## Results

| Model | CV RMSE | Test RMSE | Test R² |
|---|---|---|---|
| Ridge | 0.938 ± 0.12 | 0.886 | 0.690 |
| ElasticNet | 0.794 ± 0.09 | 0.813 | 0.739 |
| Random Forest | 0.686 ± 0.11 | 0.546 | 0.882 |
| Gradient Boosting | 0.768 ± 0.10 | 0.555 | 0.878 |
| **XGBoost** | **0.733 ± 0.09** | **0.524** | **0.892** |
| LightGBM | 0.785 ± 0.13 | 0.644 | 0.836 |
| Delaney formula | — | 1.152 | 0.547 |

---

## Reference

Delaney, J.S. (2004). ESOL: Estimating aqueous solubility directly from molecular structure. *J. Chem. Inf. Comput. Sci.*, 44(3), 1000–1005. DOI: 10.1021/ci034243x
