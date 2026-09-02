# Financial Risk for Loan Approval: An End-to-End Binary Classification Pipeline

## 1. Overview

This project implements a full credit-risk underwriting pipeline that predicts whether a loan application will be **approved or denied** from 33 demographic, employment, and credit-history features. It addresses a core financial-services problem — automating and standardizing loan-approval decisions in a way that is both accurate and auditable — by systematically comparing seven modeling strategies (from an interpretable logistic baseline through gradient-boosted trees) under a single, reusable preprocessing and evaluation harness. The final deliverable is a tuned, imbalance-aware **stacked ensemble** that reaches **0.911 test F1** and **0.998 cross-validated AUC** on a held-out set of 20,000 synthetic loan applications.

## 2. Core Methodologies & Architectural Approach

**Single-source data pipeline.** All seven models are generated from one parameterized function, `get_parameter_data(df, cat_cols, num_cols, ohe_cols, drop_cols, ohe_drop, d_poly)`, rather than duplicating preprocessing logic per model. It performs, in order: `StandardScaler` fitting on numeric columns, `OneHotEncoder` on categorical columns, a **stratified 80/20 train/test split** (`random_state=109`, fixed across every run for reproducibility), and **SMOTE oversampling applied only to the training fold** (never to test data, to avoid leakage). This single-function design means every model variant is guaranteed to see data prepared through an identical, auditable path — the only differences between runs are explicit function arguments.

**Encoding strategy differentiated by model family.** The pipeline exposes `ohe_drop` as an argument specifically because linear and tree-based models have different failure modes: logistic regression variants call the pipeline with `ohe_drop='first'` to avoid the dummy-variable trap (perfectly collinear one-hot columns destabilize linear coefficient estimates), while Random Forest and XGBoost variants call it with `ohe_drop=None`, retaining every category level since tree splits don't require full column rank and dropping a level would silently discard information a split could use.

**Imbalance handling — SMOTE over naive oversampling.** EDA revealed the target is imbalanced (76.1% not-approved vs. 23.9% approved). The notebook explicitly evaluates **SMOTE** against `RandomOverSampler` and selects SMOTE because it synthesizes new minority-class points via interpolation rather than duplicating existing rows — this smooths the decision boundary the classifier learns instead of over-weighting a fixed set of duplicated examples.

**EDA-driven feature transformation.** Histogram analysis of every numeric predictor showed strong right-skew in income-, balance-, and rate-type fields, so a log transform (`np.log(x + 1e-12)`) is applied to that subset before scaling. A full correlation heatmap was used to explicitly surface multicollinear pairs (Age/Experience, AnnualIncome/MonthlyIncome, LoanAmount/MonthlyLoanPayment, CreditScore/BaseInterestRate/InterestRate, TotalAssets/NetWorth, LoanDuration/InterestRate) — this finding directly motivates the L1 (Lasso) models below, since L1 regularization actively zeroes out redundant, collinear coefficients rather than letting them destabilize the fit.

**Optional interaction-term generation (`d_poly` flag).** Rather than a full `PolynomialFeatures` expansion (which would explode dimensionality across all pairwise combinations), the pipeline manually constructs interaction terms restricted to **categorical × numerical** pairs only. This targets a specific hypothesis — that a numeric effect (e.g., income) may behave differently within a given category (e.g., employment status) — while keeping the feature space tractable (460 features for the "poly" logistic variant, vs. ~34 for the "simple" variant).

**Standardized evaluation harness.** A second reusable function, `get_metrics(model_name, model, x_train, y_train, x_test, y_test, scoring, cv)`, computes training accuracy/F1, 5-fold cross-validated accuracy/F1 (`cross_val_score`), a CV-generated ROC curve and AUC (via `cross_val_predict(..., method='predict_proba')` to avoid optimistic in-sample ROC estimates), and true held-out test accuracy/F1. Every model's result dict is appended to a shared `model_performance` registry list, giving a single tabular source of truth for the final model comparison rather than ad hoc, per-model reporting.

**Consistent hyperparameter search protocol.** All tunable models share one `KFold(n_splits=5)` object and are tuned via `GridSearchCV` scored on **F1** (not accuracy) — an explicit choice appropriate for the imbalanced target, since accuracy alone would be inflated by the majority class.

## 3. Models & Algorithms Deployed

| # | Model | Role | Key Configuration | Why Chosen |
|---|---|---|---|---|
| 1 | **Logistic Regression (Base)** | Interpretable benchmark | No penalty, 8 hand-picked EDA-informed features (`AnnualIncome`, `MonthlyIncome`, `TotalDebtToIncomeRatio`, `LengthOfCreditHistory`, `EmploymentStatus`, `EducationLevel`, `BankruptcyHistory`, `PreviousLoanDefaults`) | Establishes a floor for model performance using only variables EDA flagged as correlated with approval |
| 2 | **Lasso Logistic — Simple** | Full-feature, regularized linear model | L1 penalty, `liblinear` solver, `GridSearchCV` over `C = logspace(-3, 2, 10)`, 5-fold CV, `f1` scoring; best `C ≈ 0.599` | L1 performs automatic feature selection and directly counteracts the multicollinearity surfaced in EDA |
| 3 | **Lasso Logistic — Poly** | Regularized linear model + interaction effects | Same as above with categorical×numerical interaction terms (460 features); best `C ≈ 0.167` | Tests whether subgroup-specific numeric effects (e.g., income effect varying by employment status) improve on the simple linear model; L1 zeroed 230/460 features automatically |
| 4 | **Random Forest — Simple** | Non-linear benchmark | `GridSearchCV`-tuned: `n_estimators=1300, max_depth=20, max_features='sqrt', criterion='gini'` | Captures non-linear feature interactions and threshold effects that a linear model cannot, without manual feature engineering |
| 5 | **Random Forest — Poly** | Non-linear model + interaction features | Same family, `max_depth=23`, trained on the interaction-augmented feature set | Checks whether explicit interaction terms add value on top of a model that already learns interactions implicitly |
| 6 | **XGBoost** | Gradient-boosted benchmark | `objective='binary:logistic'`, `eval_metric='aucpr'`, tuned `learning_rate=0.1, max_depth=7, n_estimators=800, reg_alpha=0.04` | Best standalone discriminative power (highest solo AUC/F1 of any single model) via boosted, regularized trees |
| 7 | **Stacked Ensemble (Final Model)** | Production model | `StackingClassifier(passthrough=True)` combining Lasso-Logistic-Simple + Random-Forest-Simple + XGBoost as base learners, with a `LogisticRegression` meta-learner over 5-fold CV | Blends complementary error patterns from a regularized linear model, a bagged tree ensemble, and a boosted tree ensemble; `passthrough=True` lets the meta-learner also see raw features, not just base predictions |

**Supporting algorithm:** `SMOTE` (Synthetic Minority Over-sampling Technique, from `imbalanced-learn`) is applied inside the shared preprocessing pipeline for every model above to correct the 76/24 class imbalance on the training fold only.

## 4. Tech Stack & Libraries

- **Language / Runtime:** Python 3, Jupyter Notebook (`ipykernel`)
- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Preprocessing (`scikit-learn`):** `LogisticRegression`, `RandomForestClassifier`, `StackingClassifier`, `KNeighborsClassifier`, `GridSearchCV`, `KFold`, `cross_val_score` / `cross_val_predict`, `StandardScaler`, `OneHotEncoder`, `LabelEncoder`, `train_test_split`, `roc_curve` / `roc_auc_score` / `f1_score` / `accuracy_score`, `ConfusionMatrixDisplay`
- **Imbalanced Learning:** `imbalanced-learn` (`SMOTE`)
- **Gradient Boosting:** `xgboost` (`XGBClassifier`)
- **Dataset:** `Loan.csv` — 20,000-row synthetic personal/financial dataset (not committed to this repository; expected alongside the notebook at runtime)

## 5. Key Results & Engineering Metrics

All metrics below are pulled directly from the notebook's final model-comparison table. Training/CV metrics reflect a 5-fold cross-validation protocol; test metrics reflect a single stratified 20% held-out split never seen during training or tuning.

| Model | Train Acc | CV Acc | Train F1 | CV F1 | CV AUC | Test Acc | Test F1 |
|---|---|---|---|---|---|---|---|
| logistic_base | 0.8628 | 0.8609 | 0.8647 | 0.8625 | 0.9387 | 0.8513 | 0.7326 |
| logistic_lasso_simple | 0.9544 | 0.9535 | 0.9547 | 0.9540 | 0.9915 | 0.9410 | 0.8847 |
| logistic_lasso_poly | 0.9609 | 0.9565 | 0.9613 | 0.9568 | 0.9920 | 0.9408 | 0.8830 |
| random_forest_simple | 1.0000 | 0.9589 | 1.0000 | 0.9592 | 0.9941 | 0.9295 | 0.8574 |
| random_forest_poly | 0.9999 | 0.9497 | 0.9999 | 0.9501 | 0.9913 | 0.9103 | 0.8245 |
| XGBoost_simple | 1.0000 | 0.9720 | 1.0000 | 0.9715 | 0.9973 | 0.9545 | 0.9052 |
| **stacked_model (final)** | 1.0000 | **0.9762** | 1.0000 | **0.9759** | **0.9978** | **0.9580** | **0.9112** |

**Highlights:**
- The final stacked ensemble improves test F1 by **+0.179** over the unregularized logistic baseline (0.733 → 0.911) and by **+0.006** over the best standalone model (XGBoost, 0.905).
- CV AUC of 0.998 indicates near-complete separability between approved and denied classes on the engineered feature set.
- Random Forest and XGBoost variants show a training/CV F1 gap of ~4 points (1.00 vs. ~0.95–0.97), an expected and explicitly acknowledged overfitting pattern for high-capacity tree ensembles that was not further regularized because doing so degraded validation performance.
- SMOTE-balanced training data means training-set accuracy/F1 are computed on the synthetically balanced set, while test metrics reflect the pipeline's real, imbalanced class distribution.

## 6. Repository Structure

```
CSCI_109a - Financial Risk for Loan Approval/
├── Financial_Risk_for_Loan_Approval.ipynb   # Main entry point: full EDA → preprocessing →
│                                             #   model training/tuning → evaluation pipeline
├── Project_Group47_v1.pptx                  # Project presentation summarizing findings
├── README.md                                # This file
└── Loan.csv                                 # NOT included in repo — required at runtime,
                                              #   loaded via pd.read_csv('Loan.csv')
```

## 7. Quickstart & Usage

**1. Install dependencies** (no `requirements.txt` is committed; install the libraries the notebook imports):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

**2. Provide the dataset.** Place `Loan.csv` in the same directory as the notebook — the notebook reads it via:

```python
df = pd.read_csv('Loan.csv')
```

**3. Run the notebook:**

```bash
jupyter notebook "Financial_Risk_for_Loan_Approval.ipynb"
```

Execute cells top-to-bottom. The notebook is organized to run end-to-end: EDA → feature engineering → the shared `get_parameter_data` preprocessing pipeline → per-model training/tuning (Sections "Model Training and Tuning") → the shared `get_metrics` evaluation harness → final model comparison and ROC visualization. There is no separate CLI or script entry point — this is a self-contained research/analysis notebook.
