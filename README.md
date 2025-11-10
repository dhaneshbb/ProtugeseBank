<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![LightGBM](https://img.shields.io/badge/LightGBM-model-brightgreen.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white) ![Dataset](https://img.shields.io/badge/dataset-UCI%20ML-red.svg)

# Portuguese Bank Marketing Campaign Analysis

Machine learning classification model for predicting term deposit subscriptions from bank marketing campaigns. Achieves 87.8% test accuracy and 57.3% recall with LightGBM at threshold 0.60.

</div>

---

## Overview

Predicts customer term deposit subscription from 17 campaign and demographic features (21 original, 4 removed for data leakage). Addresses class imbalance (7.87:1 ratio), high multicollinearity (VIF > 26,000), interconnected outliers (4.45% of samples), and data leakage risks in temporal features.

**Final Model:** LightGBM Classifier (learning_rate=0.01, n_estimators=200, max_depth=6, threshold=0.60)

- Test: Accuracy = 87.8%, Precision = 46.7%, Recall = 57.3%, F1-Score = 0.515, ROC AUC = 81.1%
- Cross-validation: F1 = 0.488 +/- 0.020 (5-fold stratified)
- Features: 35 engineered (11 numerical + 1 ordinal + 2 binary + 21 one-hot)
- Class imbalance handled: is_unbalance=True, stratified split

See [Model_Comparison_Report.md](reports/Model_Comparison_Report.md) for why LightGBM was selected over XGBoost despite 1% lower test F1.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### Load Trained Model

```python
import joblib
model = joblib.load('models/final_lgbm_model.pkl')

# Make predictions with selected threshold
y_probs = model.predict_proba(X_new)[:, 1]
predictions = (y_probs >= 0.60).astype(int)  # Threshold selected for F1-score
```

### Run Full Pipeline

Open `notebooks/PRCP-1000-PortugeseBank.ipynb` for complete data preparation, modeling, and evaluation workflow.

## Dataset

| Property | Details |
|----------|---------|
| **Source** | UCI Bank Marketing Dataset (2008-2010 campaigns) |
| **Samples** | 41,188 customer contacts (36,535 no / 4,639 yes) |
| **Features** | 21 attributes (17 retained after leakage removal) |
| **Split** | 32,939 train / 8,235 test (80/20 stratified) |
| **Imbalance** | 7.87:1 ratio (88.73% no, 11.27% yes) |

See [docs/problem-statement.pdf](docs/problem-statement.pdf) for project requirements.

## Project Structure

**Core directories:**
- `data/raw/` - Original UCI dataset (bank-additional-full.csv)
- `data/processed/train_test/` - Train/test splits (X_train, X_test, y_train, y_test)
- `notebooks/` - Full ML pipeline (PRCP-1000-PortugeseBank.ipynb)
- `src/` - Reusable modules (utils, statistical analysis, model evaluation)
- `models/` - Trained model artifacts (final_lgbm_model.pkl)
- `reports/` - Detailed analysis reports
- `results/figures/` - Visualizations

## Working with the Notebook

**Import pattern used:**
The notebook imports functions from src/ modules using:
```python
from src.utils import memory_usage, dataframe_memory_usage, garbage_collection
from src.statistical_analysis import chi_square_test, calculate_vif, spearman_correlation_with_target
from src.model_evaluation import evaluate_model, threshold_analysis, cross_validation_analysis_table
```

**Running analysis:**
The notebook contains the full ML pipeline. Execute cells sequentially for:
1. Data loading and EDA (automated with insightfulpy)
2. Data cleaning (missing value imputation, duplicate removal)
3. Statistical analysis (VIF, chi-square, Spearman correlation)
4. Feature engineering (encoding, outlier capping)
5. Data leakage prevention (removal of duration, pdays, previous, poutcome)
6. Model comparison (5 algorithms tested: LightGBM, XGBoost, RandomForest, GradientBoosting, CatBoost)
7. Hyperparameter tuning (manual tuning for top 3 models)
8. Threshold optimization (0.10 to 0.90)
9. Final model selection and persistence

## Model Training Workflow

**Base model evaluation:**
```python
from src.model_evaluation import evaluate_model

results = evaluate_model(model, X_train, y_train, X_test, y_test)
# Returns: Accuracy, Precision, Recall, F1-Score, ROC AUC, CV F1, Confusion Matrix, Training Time, Overfit
```

**Cross-validation analysis:**
```python
from src.model_evaluation import cross_validation_analysis_table

cv_results = cross_validation_analysis_table(model, X_train, y_train, cv_folds=5, scoring_metric='f1')
# Returns: Fold-by-fold F1 scores, Mean, Standard Deviation
```

**Threshold selection:**
```python
from src.model_evaluation import threshold_analysis

df_threshold_results, best_threshold = threshold_analysis(model, X_test, y_test)
# Tests thresholds from 0.1 to 0.9, returns threshold with maximum F1-score
```

## Statistical Analysis Functions

**Chi-square test (categorical associations):**
```python
from src.statistical_analysis import chi_square_test

chi2_stat, p_value, dof, expected_freq = chi_square_test(data, 'job', 'y')
# Tests independence between categorical variables
```

**VIF analysis (multicollinearity detection):**
```python
from src.statistical_analysis import calculate_vif

vif_results, high_vif_features = calculate_vif(
    data,
    exclude_target='y',
    multicollinearity_threshold=5.0
)
# Returns VIF scores; VIF > 26,000 detected for economic indicators
```

**Spearman correlation (non-parametric):**
```python
from src.statistical_analysis import spearman_correlation_with_target

corr_data = spearman_correlation_with_target(
    data,
    numerical_cols=['age', 'campaign', 'euribor3m'],
    target_col='y',
    plot=True,
    table=True
)
```

## Model Persistence

**Loading the final model:**
```python
import joblib
model = joblib.load('models/final_lgbm_model.pkl')

# Predict with selected threshold (0.60)
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= 0.60).astype(int)
```

## Key Design Decisions

**Model selection criteria (weighted):**
1. CV Stability (SD) - 40%
2. Recall (minority class capture) - 30%
3. F1-Score (precision/recall balance) - 20%
4. Efficiency (speed, overfitting) - 10%

**Why LightGBM over XGBoost:**
- XGBoost achieved higher test F1 (0.519 vs 0.509) but showed 20% higher CV variance (SD = 0.024 vs 0.020)
- LightGBM provides 2.2% higher recall (60.9% vs 58.7%) - captures 20 more subscribers per 928
- Training: 3% faster (0.691s vs 0.712s)
- CV-test gap: 2.1 points vs 2.6 points (higher consistency)
- Trade-off: Accept 1% lower test F1 for 20% lower CV variance and higher stability across economic periods

**Threshold selection (0.60 vs default 0.50):**
- Improved F1-score: 0.509 -> 0.515 (+0.6%)
- Improved precision: 43.7% -> 46.7% (+3.0%)
- Reduced false positives: 728 -> 607 (-121 wasted contacts)
- Trade-off: Recall reduced 60.9% -> 57.3% (-33 captures)
- Business impact: Sacrifice 33 subscribers to avoid 121 wasted contacts (improved cost-effectiveness)

**Data leakage prevention:**
Four features removed to prevent leakage:
- `duration`: Call duration only known after call ends (post-hoc information)
- `pdays`, `previous`, `poutcome`: Previous campaign outcomes create circular prediction

**Class imbalance handling:**
- Stratified train-test split (preserves 7.87:1 ratio)
- LightGBM: `is_unbalance=True` (auto loss adjustment)
- RandomForest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight=5`

**Multicollinearity management:**
- VIF > 26,000 detected (nr.employed, cons.price.idx)
- Economic indicators highly correlated (rho = 0.94)
- Solution: Retained all features for tree-based models (robust to multicollinearity)
- No PCA needed (interpretability preserved)

## Model Performance Comparison

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC AUC | CV F1 Mean | CV F1 SD | Training Time |
|-------|--------------|-----------|--------|----------|---------|------------|----------|---------------|
| **LightGBM (tuned)** | **86.8%** | **43.7%** | **60.9%** | **0.509** | **81.1%** | **0.488** | **0.020** | **0.691s** |
| XGBoost (tuned) | 87.7% | 46.4% | 58.7% | 0.519 | 80.9% | 0.493 | 0.024 | 0.712s |
| RandomForest (tuned) | 86.8% | 43.8% | 60.2% | 0.507 | 79.7% | 0.487 | 0.026 | 11.579s |
| GradientBoosting (base) | 89.5% | 56.9% | 27.3% | 0.369 | 79.9% | 0.361 | N/A | 21.015s |
| CatBoost (base) | 89.5% | 58.6% | 23.8% | 0.339 | 80.6% | 0.329 | N/A | 3.820s |

**Performance at Selected Threshold (0.60):**
- Accuracy: 87.8%
- Precision: 46.7%
- Recall: 57.3%
- F1-Score: 0.515
- True Positives: 532 (subscribers identified)
- False Positives: 607 (wasted contacts)

## Feature Engineering

**Numerical features (11):**
- Demographics: age
- Campaign metrics: campaign (capped at 6)
- Economic indicators: emp.var.rate, cons.price.idx, cons.conf.idx (98th percentile capped), euribor3m, nr.employed

**Categorical encoding:**
- Ordinal: education (illiterate < basic < high.school < university)
- Binary: default, housing, loan
- One-hot: job (11 categories), marital (4), contact (2), month (10), day_of_week (5), poutcome (3)

**Features removed (4):**
- duration (data leakage - post-call information)
- pdays, previous, poutcome (data leakage - campaign history)

**Outlier treatment:**
- Age: Capped at 69.5 years (468 samples)
- Campaign: Capped at 6 contacts (2,406 samples)
- Cons.conf.idx: 98th percentile capping (446 samples)
- Result: 4,834 outliers capped, 0 samples deleted

## Reports

Detailed analysis in `reports/`:

- [Complete_Data_Analysis_Report.md](reports/Complete_Data_Analysis_Report.md) - Full methodology, EDA, statistical tests, and results
- [Model_Comparison_Report.md](reports/Model_Comparison_Report.md) - Model selection rationale and performance comparison
- [Challenges_Report.md](reports/Challenges_Report.md) - Technical challenges and solutions (class imbalance, data leakage, multicollinearity, outliers, model selection)
- [GALLERY.md](results/figures/GALLERY.md) - Visualizations

## Development

### Code Quality

```bash
# Format code
black .
isort .

# Lint
flake8 .

# Format notebooks
nbqa black notebooks/

# Run pre-commit hooks
pre-commit run --all-files
```

### Pre-commit Hooks

- black (88-char lines)
- isort (black-compatible)
- flake8 (ignores E203, W503, E501; max complexity 10)
- nbqa-black (notebooks)
- Validation (YAML, trailing whitespace, end-of-file)

### Code Style

- Line length: 88 characters
- Import order: FUTURE, STDLIB, THIRDPARTY, FIRSTPARTY, LOCALFOLDER
- Docstrings: NumPy format
- Target Python: 3.8, 3.9, 3.10, 3.11

## Key Takeaways

1. **Class Imbalance:** Stratified splitting + class weighting increased recall from 23.8-27.3% to 60.9% (2.6x improvement)
2. **Data Leakage:** Removed 4 temporal features despite high predictive power for production viability
3. **Multicollinearity:** Tree-based models handle VIF > 26,000 without PCA (interpretability preserved)
4. **Outliers:** Capping preserved 100% of samples while reducing skewness by 74.6% (campaign feature)
5. **Model Selection:** CV stability prioritized over test performance (LightGBM's 20% lower variance outweighs 1% lower F1)
6. **Threshold Selection:** Moving from 0.50 to 0.60 reduced wasted contacts by 121 at cost of 33 missed subscribers

---

- MIT License - Copyright (c) 2025 Dhanesh B. B.
- GitHub: [https://github.com/dhaneshbb](https://github.com/dhaneshbb)
