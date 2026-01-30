# Run Log - DACA Replication Study (ID: 44)

## Date: 2026-01-27

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Instructions
- Read replication_instructions.docx
- Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the US, what was the causal impact of eligibility for DACA on full-time employment probability?
- Treatment group: Ages 26-30 at policy implementation (June 15, 2012)
- Control group: Ages 31-35 at policy implementation
- Outcome: FT (full-time employment, 35+ hours/week)
- Method: Difference-in-Differences

### 1.2 Data Files Located
- prepared_data_numeric_version.csv
- prepared_data_labelled_version.csv
- acs_data_dict.txt

### 1.3 Key Variables Identified from Data Dictionary
- **YEAR**: Survey year (2008-2016, excluding 2012)
- **FT**: Full-time employment (1=yes, 0=no) - OUTCOME
- **ELIGIBLE**: Treatment group indicator (1=ages 26-30, 0=ages 31-35) - TREATMENT
- **AFTER**: Post-policy indicator (1=2013-2016, 0=2008-2011) - TIME
- **PERWT**: Person weight for population estimates
- **AGE_IN_JUNE_2012**: Age at policy implementation
- **AGE_AT_IMMIGRATION**: Age when immigrated
- **SEX**: 1=Male, 2=Female
- **EDUC_RECODE**: Education categories (Less than HS, HS, Some College, Two-Year, BA+)
- **MARST**: Marital status
- **STATEFIP**: State identifier
- **EMPSTAT**: Employment status (1=Employed, 2=Unemployed, 3=Not in labor force)

### 1.4 DiD Design
- DiD Estimate = (E[FT|Eligible=1, After=1] - E[FT|Eligible=1, After=0]) - (E[FT|Eligible=0, After=1] - E[FT|Eligible=0, After=0])
- Regression specification: FT = β0 + β1*ELIGIBLE + β2*AFTER + β3*(ELIGIBLE*AFTER) + ε
- β3 is the DiD estimate (treatment effect of DACA on full-time employment)

---

## 2. Data Loading and Exploration

### Command: Load and explore data
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv')
```

### Key Findings:
- **Total observations**: 17,382
- **Years in data**: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- **Treatment group (ELIGIBLE=1)**: 11,382 observations
- **Control group (ELIGIBLE=0)**: 6,000 observations
- **Pre-period (AFTER=0)**: 9,527 observations
- **Post-period (AFTER=1)**: 7,855 observations

### Decision: Use provided ELIGIBLE and AFTER variables
- Per instructions, use the pre-defined ELIGIBLE variable (ages 26-30 at June 15, 2012)
- Use provided AFTER variable (1 for 2013-2016, 0 for 2008-2011)
- Do not further limit the sample

---

## 3. Descriptive Statistics

### Full-Time Employment Rates by Group (Unweighted):
| Group | Before (2008-11) | After (2013-16) | Change |
|-------|------------------|-----------------|--------|
| Treated (26-30) | 62.63% | 66.58% | +3.95 pp |
| Control (31-35) | 66.97% | 64.49% | -2.48 pp |

### Simple DiD Calculation:
- DiD = (66.58% - 62.63%) - (64.49% - 66.97%)
- DiD = 3.95% - (-2.48%)
- **DiD = 6.43 percentage points**

---

## 4. Regression Analysis

### Model Specifications:

**Model 1: Basic OLS (Unweighted)**
```
FT = β0 + β1*ELIGIBLE + β2*AFTER + β3*(ELIGIBLE*AFTER)
```
- DiD Estimate (β3): 0.0643
- Standard Error: 0.0153
- p-value: < 0.0001

**Model 2: Basic WLS (Weighted by PERWT)**
- DiD Estimate: 0.0748
- Standard Error: 0.0152
- p-value: < 0.0001

**Model 3: OLS with Robust Standard Errors**
- DiD Estimate: 0.0643
- Standard Error: 0.0153 (HC1)
- p-value: < 0.0001

**Model 4: With Demographic Covariates**
```
FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER + FEMALE + MARRIED + Education dummies + AGE
```
- DiD Estimate: 0.0535
- Standard Error: 0.0142
- p-value: 0.0002

**Model 5: With Year Fixed Effects**
- DiD Estimate: 0.0520
- Standard Error: 0.0141
- p-value: 0.0002

**Model 6: With Year and State Fixed Effects (PREFERRED)**
- DiD Estimate: 0.0520
- Standard Error: 0.0141
- 95% CI: [0.0242, 0.0797]
- p-value: 0.0002

### Decision: Model 6 selected as preferred specification
Rationale:
1. Controls for demographic differences between groups
2. Year fixed effects control for time-varying factors affecting all individuals
3. State fixed effects control for state-specific characteristics
4. Robust standard errors account for heteroskedasticity

---

## 5. Robustness Checks

### 5.1 Placebo Test (Pre-treatment trends)
- Pretending treatment started in 2010
- Placebo DiD: 0.0151 (SE: 0.0192, p=0.43)
- **Result**: No evidence of differential pre-trends

### 5.2 Event Study Analysis
- Reference year: 2011 (last pre-treatment year)
- Pre-treatment coefficients: All insignificant and near zero
- Post-treatment coefficients: Growing positive trend

### 5.3 Heterogeneous Effects
- **Male**: DiD = 0.0512 (SE: 0.0169, p=0.002)
- **Female**: DiD = 0.0388 (SE: 0.0228, p=0.088)
- **Less than BA**: DiD = 0.0510 (SE: 0.0146, p<0.001)

### 5.4 Alternative Specifications
- Probit (marginal effect): 0.0600
- Logit (marginal effect): 0.0594
- Without state FE: 0.0520

---

## 6. Key Decisions and Rationale

1. **Sample**: Used full provided sample without additional restrictions (as instructed)
2. **Weighting**: Primary analysis unweighted; weighted results reported for comparison
3. **Standard errors**: Used heteroskedasticity-robust (HC1) standard errors
4. **Covariates**: Included sex, marital status, education, age (all theoretically relevant)
5. **Fixed effects**: Included year and state FE to control for confounders
6. **Outcome**: Linear probability model (interpretable coefficients in percentage points)

---

## 7. Final Results Summary

### Preferred Estimate:
- **Effect**: 5.2 percentage point increase in full-time employment
- **Standard Error**: 1.41 percentage points
- **95% CI**: [2.42, 7.97] percentage points
- **Sample Size**: 17,382
- **Statistical Significance**: p < 0.001

### Interpretation:
DACA eligibility increased the probability of full-time employment by approximately 5.2 percentage points among eligible individuals (ages 26-30 in June 2012) compared to the control group (ages 31-35). This represents a meaningful improvement in labor market outcomes, with the effect being statistically significant and robust across specifications.

---

## 8. Files Generated

- `analysis.py` - Main analysis script
- `robustness_analysis.py` - Robustness checks and figures
- `regression_results.csv` - Main regression results
- `did_cell_means.csv` - Cell means for DiD calculation
- `event_study_results.csv` - Event study coefficients
- `balance_table.csv` - Pre-treatment balance statistics
- `heterogeneity_results.csv` - Subgroup analysis results
- `figure1_trends.png` - Time trends by treatment group
- `figure2_event_study.png` - Event study plot
- `figure3_did_visual.png` - DiD visualization
- `replication_report_44.tex` - LaTeX report
- `replication_report_44.pdf` - Final PDF report

---

## 9. Session Commands Log

```
# Check Python packages
python -c "import pandas; import statsmodels; print('OK')"

# Run main analysis
python analysis.py

# Run robustness checks
python robustness_analysis.py

# Compile LaTeX report
pdflatex replication_report_44.tex
```
