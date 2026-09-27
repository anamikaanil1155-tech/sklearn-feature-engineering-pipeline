# Credit Risk Preprocessing Pipeline

A robust, leak-free machine learning preprocessing pipeline for tabular credit default risk data, built with scikit-learn. Covers feature encoding, scaling, missing-value imputation, and correlation / feature-importance analysis — all wrapped in a single, reusable `Pipeline` object.

## Overview

This project demonstrates how to build a preprocessing pipeline the right way: no statistic (mean, median, category vocabulary, etc.) is ever learned from data the model hasn't been trained on. The train/test split happens **before** any transformation, and every imputer, scaler, and encoder is fit exclusively on the training set.

The dataset is a synthetically generated Credit Default Risk table (5,000 rows) with a realistic mix of numeric and categorical features, injected missing values, and a genuinely learnable target signal — so the notebook demonstrates real imputation and encoding rather than a toy problem with no missing data.

## What's inside

- **`feature_engineering_pipeline.ipynb`** — the full, executed notebook (self-contained, no external files required)

## Pipeline steps

1. **Load data** — synthetic credit default dataset with numeric, categorical, and missing values
2. **Train/test split** — performed first, before any preprocessing, to prevent data leakage
3. **Preprocessing** — `ColumnTransformer` combining:
   - Numeric: median imputation → `StandardScaler`
   - Categorical: most-frequent imputation → `OneHotEncoder`
4. **Fit/transform discipline** — preprocessor is `fit()` on train only, `transform()` on both train and test
5. **Correlation analysis** — heatmap of numeric features against the target (train set only)
6. **Feature importance** — via mutual information and Random Forest importances
7. **Reusability check** — pipeline wrapped in a factory function and validated with stratified cross-validation, confirming preprocessing refits cleanly inside every fold

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

Open `feature_engineering_pipeline.ipynb` in Jupyter, JupyterLab, or Google Colab and run all cells top to bottom. No external data files or helper scripts are needed — the dataset generator is defined inline.

To use your own dataset instead of the synthetic one, replace the data-loading cell with:

```python
df = pd.read_csv("your_file.csv")
```

and update the `numeric_features` / `categorical_features` lists to match your columns. The rest of the pipeline works unchanged.

## Key design choices

| Principle | Implementation |
|---|---|
| No data leakage | Split before any `fit`/`fit_transform` call |
| Reusable | Single `Pipeline` object; also exposed via a `build_pipeline()` factory |
| Robust to unseen categories | `OneHotEncoder(handle_unknown="ignore")` |
| Validated | `StratifiedKFold` cross-validation refits preprocessing per fold |

## License

MIT (or update to match your preference)
