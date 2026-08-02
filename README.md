# Credit Risk Scorecard: Lending Club Loan Default Prediction

A probability-of-default (PD) model and points-based credit scorecard built on Lending Club's public loan data (2007–2018), following industry-standard credit scoring methodology (WOE/IV feature selection, logistic regression scorecard, KS/PSI validation).

## Project Summary

This project predicts the probability that a loan will default at origination, using only information available at the time the loan was issued. The core deliverable is a FICO-style points scorecard, built by transforming a logistic regression on Weight-of-Evidence (WOE) encoded features into an interpretable point system.

**Key results (out-of-sample test set, 2016–2018):**

| Metric | Value |
|---|---|
| Test AUC | 0.685 |
| Test KS statistic | 0.269 |
| Test Gini | 0.370 |
| Score range | ~300–855 (mean 600, std 100) |
| Default rate, lowest score decile | ~49% |
| Default rate, highest score decile | ~7% |

## Methodology

### 1. Data & Target Definition
- Source: Lending Club accepted loans, 2007–2018 (`wordsforthewise/lending-club` on Kaggle)
- Target: binary default flag — `Charged Off`, `Default`, and `Late (31-120 days)` statuses labeled bad (1); `Fully Paid` labeled good (0). Loans still `Current` were excluded since their outcome is unknown.
- Base default rate: 20.0% (train), 26.3% (test) — see calibration note below.

### 2. Leakage Prevention
Only features known **at loan origination** were used (e.g., loan amount, interest rate, DTI, income, purpose). Post-origination fields such as `total_pymnt`, `recoveries`, and `last_pymnt_d` were excluded entirely, since they encode the outcome itself.

### 3. Train/Test Split
A **time-based split** (80/20 by `issue_d`) was used instead of a random split, since credit models must generalize forward in time, not just across a random sample. This also surfaced a genuine distribution shift between periods (see Calibration & Drift below).

- Train: 1,108,727 loans (Jun 2007 – Nov 2016)
- Test: 260,810 loans (Dec 2016 – Dec 2018)

### 4. Feature Selection (WOE / Information Value)
All candidate features were binned and scored using Weight of Evidence and Information Value, computed on the training set only. Low-cardinality integer features (e.g. `term`, `pub_rec`) were binned as categorical rather than run through quantile binning, after an initial pass showed `qcut` silently collapsing low-cardinality columns into a single bin and producing artificially deflated IV (caught via a manual cardinality check).

**Final features (IV > 0.02, non-redundant):**

| Feature | IV | Notes |
|---|---|---|
| int_rate | 0.449 | Strongest single predictor |
| dti | 0.076 | Debt-to-income ratio |
| loan_amnt | 0.033 | |
| annual_inc | 0.029 | |
| home_ownership | 0.025 | |
| purpose | 0.020 | |
| term | 0.207 (recomputed) | Originally miscalculated as 0.000 due to a qcut binning bug on a 2-value column |

**Explicitly excluded:**
- `grade` / `sub_grade` (IV 0.47 / 0.50) — excluded despite high IV, since these are themselves *outputs* of Lending Club's internal underwriting model. Including them would mean predicting the lender's own risk score rather than default risk directly (circularity), and using `int_rate` alone avoids this while still capturing legitimate market-priced risk information.
- `installment` — dropped after WOE correlation matrix showed 0.83 correlation with `loan_amnt` (installment is a near-deterministic function of loan amount, rate, and term).
- `revol_util` — IV 0.019 initially retained, but dropped from the final model after its multivariate logistic regression coefficient came out with the wrong sign (positive, implying higher utilization = lower risk) despite a clean, monotonically decreasing univariate WOE relationship. Diagnosed as a suppression effect from shared variance with `dti`, not a data error. Documented rather than silently fixed, then excluded to keep the final scorecard's point directions fully interpretable.
- `credit_history_months`, `open_acc`, `emp_length_num`, `delinq_2yrs`, `pub_rec`, `total_acc` — all IV < 0.02 even after correcting for the categorical-binning issue; dropped as genuinely weak predictors in this dataset.

### 5. Model
Logistic regression fit on WOE-transformed features (bins learned on train, applied to test — no leakage). This mirrors standard bank scorecard construction, prioritizing interpretability and monotonicity over raw predictive power.

### 6. Scorecard Conversion
Logistic regression coefficients were converted into a points-based scorecard using standard scaling (20 points to double the odds), then linearly rescaled to a mean of 600 / std of 100 for interpretability. This rescaling preserves all rank-ordering and discrimination metrics (AUC, KS) — it only affects presentation.

### 7. Validation

**Discrimination:** AUC 0.685 / KS 0.269 / Gini 0.370 on the out-of-sample test set. These are consistent with published benchmarks for origination-time scorecards using standard application data (no bureau/behavioral features); further gains would require data not available in this public dataset.

**Calibration:** Predicted probabilities systematically underestimate actual default rates in the test period (test base rate 26.3% vs. train 20.0%). Rank-ordering ability held up out-of-sample, but absolute probabilities did not — a model can retain strong discrimination while its calibration drifts, which is why production credit models require periodic recalibration.

**Population Stability Index (PSI):**
- Aggregate score distribution: PSI = 0.0075 (stable) — the *population* being scored did not change meaningfully between periods.
- Per-feature: most features stable (`dti` 0.0065, `annual_inc` 0.0064, `loan_amnt` 0.029), but `revol_util` showed moderate drift (0.129) and `int_rate` showed mild drift (0.056) before `revol_util` was dropped from the model.

Taken together, this suggests the test-period miscalibration reflects a shift in *borrower behavior/macro conditions* (same risk profiles defaulting more) combined with modest shifts in credit utilization and lender pricing — rather than a change in who was applying for loans. This is a realistic, multi-causal finding consistent with how drift actually presents in production credit models.

## Repository Structure
```
credit-risk-model/
├── notebooks/
│   ├── 01_eda_and_target.ipynb        # data load, leakage check, target definition
│   ├── 02_woe_iv_feature_selection.ipynb
│   ├── 03_modeling_and_scorecard.ipynb
│   └── 04_validation_and_charts.ipynb
├── src/
│   ├── woe.py            # WOE/IV calculation functions
│   ├── scorecard.py       # points conversion
│   └── metrics.py         # KS, PSI, calibration helpers
├── README.md
└── requirements.txt
```

## Limitations
- Uses historical Lending Club data (2007–2018), which spans the 2008 financial crisis; default rate dynamics and macro sensitivity may not generalize to current lending conditions.
- No bureau-level or behavioral (post-origination) data was used, by design (to avoid leakage) — this caps achievable AUC relative to models that include such data.
- The dataset does not include protected-class attributes (race, gender), so no direct fairness/disparate-impact testing was performed; this is itself worth noting, since indirect proxies (e.g. geography) were not deeply audited here.

## Reproducing
```bash
pip install -r requirements.txt
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d data/
# run notebooks in order, 01 -> 04
```
