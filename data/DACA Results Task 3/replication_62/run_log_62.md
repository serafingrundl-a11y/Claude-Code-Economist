# Run Log - Replication Study #62
## DACA Effect on Full-Time Employment

### Date: January 27, 2026

---

## Overview

This log documents all commands executed and key analytical decisions made during the independent replication study examining the causal effect of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico.

---

## 1. Data Exploration

### 1.1 Initial Data Review

**Command: Read replication instructions**
```python
from docx import Document
doc = Document('replication_instructions.docx')
print('\n'.join([p.text for p in doc.paragraphs]))
```
- Extracted research question and methodology requirements
- Identified key variables: FT (outcome), ELIGIBLE (treatment), AFTER (period)

**Command: Explore data structure**
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_labelled_version.csv', low_memory=False)
print('Shape:', df.shape)
print('Columns:', list(df.columns))
```
- Dataset: 17,382 observations, 105 variables
- Years: 2008-2016 (excluding 2012)
- Treatment group (ELIGIBLE=1): 11,382 observations (ages 26-30 in June 2012)
- Control group (ELIGIBLE=0): 6,000 observations (ages 31-35 in June 2012)

### 1.2 Key Variable Distributions

**Command: Examine treatment and period structure**
```python
print(pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True))
print(df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count']))
```

Results:
- Pre-DACA (2008-2011): 9,527 observations
- Post-DACA (2013-2016): 7,855 observations
- Full-time employment rates show clear DiD pattern

---

## 2. Analytical Decisions

### 2.1 Estimation Method

**Decision: Use Linear Probability Model (OLS/WLS)**

Rationale:
- Coefficients directly interpretable as percentage point changes
- DiD coefficient captures average treatment effect
- Robust to functional form misspecification
- Standard practice in applied microeconomics

Alternative considered: Probit/Logit models
- Rejected due to complexity in interpreting DiD coefficient and marginal effects

### 2.2 Standard Errors

**Decision: Heteroskedasticity-robust (HC1) standard errors**

Rationale:
- Binary outcome in linear model inherently produces heteroskedastic errors
- Conservative approach that does not require correct homoskedasticity assumption

Alternative considered: Clustered standard errors by state
- Not implemented as primary specification to maintain model comparability
- State fixed effects address within-state correlation

### 2.3 Survey Weights

**Decision: Report both unweighted and weighted (PERWT) estimates; prefer weighted**

Rationale:
- ACS uses complex sampling design
- Weighted estimates are population-representative
- Unweighted estimates serve as robustness check

### 2.4 Covariates

**Decision: Include demographic controls in preferred specification**

Covariates included:
- SEX (Male indicator)
- MARST (Married indicator)
- EDUC_RECODE (Education dummies: Some College, Two-Year, BA+)
- CensusRegion (Region dummies: South, Midwest, Northeast)
- AGE (Centered continuous variable)

Rationale:
- Balance tests show significant differences between groups
- Controls improve precision without changing point estimate substantially
- Theoretically relevant to employment outcomes

### 2.5 Sample Restrictions

**Decision: Use entire provided analytical sample without additional restrictions**

Rationale:
- Instructions explicitly state: "This entire file constitutes the intended analytic sample for your analysis; do not further limit the sample"
- Sample already restricted to Hispanic-Mexican individuals born in Mexico

---

## 3. Analysis Commands

### 3.1 Analysis Script Execution

**Command: Run main analysis**
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_62"
python analysis_script.py
```

### 3.2 Models Estimated

| Model | Description | Estimate | SE | p-value |
|-------|-------------|----------|-----|---------|
| 1 | Basic DiD (OLS, unweighted) | 0.0643 | 0.0153 | <0.001 |
| 2 | Basic DiD (WLS, weighted) | 0.0748 | 0.0181 | <0.001 |
| 3 | DiD + Demographics (OLS) | 0.0531 | 0.0141 | <0.001 |
| 4 | DiD + Demographics (WLS) **PREFERRED** | 0.0619 | 0.0167 | <0.001 |
| 5 | Event Study Specification | Various | Various | Various |
| 6 | DiD + State Fixed Effects | 0.0534 | 0.0142 | <0.001 |
| 7 | DiD + Year Fixed Effects | 0.0516 | 0.0141 | <0.001 |

### 3.3 Diagnostic Tests

**Parallel Trends Test:**
```python
# Pre-treatment differential trends
ELIGIBLE x YEAR_TREND coefficient: 0.0151
p-value: 0.098
```
Interpretation: Marginally fails to reject parallel trends at 10% level; parallel trends assumption receives moderate support.

**Balance Tests (Pre-Period):**
- MALE: p=0.022 (marginally significant difference)
- MARRIED: p<0.001 (significant difference)
- AGE: p<0.001 (by construction)
- Education: Generally balanced

---

## 4. Report Generation

### 4.1 LaTeX Compilation

**Commands:**
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_62"
pdflatex -interaction=nonstopmode replication_report_62.tex
pdflatex -interaction=nonstopmode replication_report_62.tex
pdflatex -interaction=nonstopmode replication_report_62.tex
```
- Three passes required for cross-references and table of contents
- Output: 24-page PDF report

---

## 5. Key Results Summary

### 5.1 Preferred Estimate

**Model 4: DiD with Demographics (WLS, Weighted)**

- Effect Size: **0.062** (6.2 percentage points)
- Standard Error: 0.017
- 95% CI: [0.029, 0.095]
- t-statistic: 3.70
- p-value: < 0.001
- Sample Size: 17,382

### 5.2 Interpretation

DACA eligibility increased the probability of full-time employment by approximately 6.2 percentage points among the eligible population (Hispanic-Mexican individuals born in Mexico, aged 26-30 in June 2012). This represents roughly a 10% increase from the baseline full-time employment rate of 63% in the pre-DACA period.

### 5.3 Robustness

The estimate is robust across specifications:
- Unweighted: 5.3-6.4 pp
- Weighted: 6.2-7.5 pp
- With state FE: 5.3 pp
- With year FE: 5.2 pp

All estimates are statistically significant at p < 0.001.

---

## 6. Files Created

| File | Description |
|------|-------------|
| `analysis_script.py` | Python script for all statistical analyses |
| `results_summary.json` | JSON file with key results for reference |
| `replication_report_62.tex` | LaTeX source for replication report |
| `replication_report_62.pdf` | Final 24-page replication report |
| `run_log_62.md` | This run log documenting commands and decisions |

---

## 7. Software Environment

- Python 3.14
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)
- scipy (statistical tests)
- pdflatex (MiKTeX 25.12) for LaTeX compilation

---

## 8. Potential Limitations Noted

1. **Linear probability model**: May produce predictions outside [0,1]
2. **Repeated cross-sections**: Cannot track same individuals over time
3. **Parallel trends**: Marginally significant pre-trend difference warrants caution
4. **Measurement**: ELIGIBLE based on observable proxies, not actual DACA receipt
5. **External validity**: Results specific to Hispanic-Mexican population

---

## End of Log
