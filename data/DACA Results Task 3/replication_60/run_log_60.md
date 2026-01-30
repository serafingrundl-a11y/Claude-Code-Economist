# DACA Replication Study - Run Log

## Study Information
- **Research Question**: Effect of DACA eligibility on full-time employment among Hispanic Mexican-born individuals
- **Treatment Group**: ELIGIBLE=1, individuals aged 26-30 in June 2012
- **Control Group**: ELIGIBLE=0, individuals aged 31-35 in June 2012
- **Outcome**: FT (full-time employment, working 35+ hours/week)
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016

---

## Session Log

### 1. Initial Data Exploration

**Timestamp**: Session start

**Command**: Loaded `prepared_data_numeric_version.csv`
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
```

**Results**:
- Total observations: 17,382
- Years covered: 2008-2016 (2012 excluded)
- Number of variables: 105

**Key Variables Identified**:
- `FT`: Full-time employment indicator (0/1)
- `ELIGIBLE`: Treatment group indicator (0/1)
- `AFTER`: Post-treatment period indicator (0/1)
- `PERWT`: Person survey weight

---

### 2. Sample Characteristics

**Treatment/Control Distribution**:
- Control (ELIGIBLE=0, ages 31-35): 6,000 observations
- Treatment (ELIGIBLE=1, ages 26-30): 11,382 observations

**Pre/Post Distribution**:
- Pre-DACA (2008-2011): 9,527 observations
- Post-DACA (2013-2016): 7,855 observations

**Cross-tabulation**:
|              | Pre (0) | Post (1) | Total  |
|--------------|---------|----------|--------|
| Control (0)  | 3,294   | 2,706    | 6,000  |
| Treatment (1)| 6,233   | 5,149    | 11,382 |
| Total        | 9,527   | 7,855    | 17,382 |

---

### 3. Descriptive Statistics

**Full-Time Employment Rates by Group**:

| Group     | Period | Mean FT | Std. Dev. | N     |
|-----------|--------|---------|-----------|-------|
| Control   | Pre    | 0.670   | 0.470     | 3,294 |
| Control   | Post   | 0.645   | 0.479     | 2,706 |
| Treatment | Pre    | 0.626   | 0.484     | 6,233 |
| Treatment | Post   | 0.666   | 0.472     | 5,149 |

**Simple DiD Calculation**:
- Control group change: 0.6449 - 0.6697 = -0.0248
- Treatment group change: 0.6658 - 0.6263 = +0.0394
- DiD estimate: 0.0643 (6.43 percentage points)

---

### 4. Key Analytical Decisions

#### Decision 1: Model Specification
**Choice**: Use OLS linear probability model for binary outcome
**Rationale**:
- DiD coefficient is easily interpretable as percentage point change
- Consistent with standard practice in applied economics
- Allows straightforward comparison across specifications

#### Decision 2: Control Variables
**Included**:
- Sex (MALE indicator)
- Family size (FAMSIZE)
- Number of children (NCHILD)
- Years in USA (YRSUSA1)
- Education dummies (Some College, Two-Year Degree, BA+)
- Year fixed effects
- State-level economic controls (UNEMP, LFPR) in robustness check

**Rationale**:
- These variables capture observable differences between treatment and control groups
- Year fixed effects control for aggregate time trends
- Education affects employment probability

#### Decision 3: Preferred Specification
**Choice**: Model 4 - DiD with individual controls and year fixed effects
**Rationale**:
- Controls for observable individual characteristics
- Year fixed effects capture economy-wide trends
- Adding state controls (Model 5) has minimal impact on estimate
- More parsimonious than full model

#### Decision 4: Standard Errors
**Choice**: Report both OLS and heteroskedasticity-robust (HC1) standard errors
**Rationale**:
- Binary outcome may induce heteroskedasticity
- Robust SEs virtually identical to OLS SEs in this case

#### Decision 5: Event Study Reference Year
**Choice**: 2011 as reference year
**Rationale**:
- Last pre-treatment year
- Standard practice for event studies
- Pre-treatment coefficients show differential trends relative to this year

---

### 5. Regression Analysis Results

#### Model Progression:

| Model | Description              | Coefficient | SE     | 95% CI            | p-value |
|-------|--------------------------|-------------|--------|-------------------|---------|
| 1     | Basic DiD               | 0.0643      | 0.0153 | [0.0343, 0.0942]  | <0.001  |
| 2     | + Demographics          | 0.0536      | 0.0143 | [0.0256, 0.0816]  | <0.001  |
| 3     | + Education             | 0.0517      | 0.0143 | [0.0237, 0.0797]  | <0.001  |
| 4     | + Year FE (Preferred)   | 0.0501      | 0.0143 | [0.0221, 0.0780]  | <0.001  |
| 5     | + State Controls        | 0.0508      | 0.0143 | [0.0229, 0.0788]  | <0.001  |

#### Preferred Estimate (Model 4):
- **Effect Size**: 0.0501 (5.01 percentage points)
- **Standard Error**: 0.0143
- **95% Confidence Interval**: [0.0221, 0.0780]
- **t-statistic**: 3.511
- **p-value**: 0.0004
- **Sample Size**: 17,382

---

### 6. Robustness Checks

#### 6.1 Robust Standard Errors
- Coefficient: 0.0501
- Robust SE: 0.0141 (virtually identical to OLS SE of 0.0143)
- Conclusion: Heteroskedasticity not a major concern

#### 6.2 Weighted Analysis (PERWT)
- Weighted coefficient: 0.0586
- SE: 0.0142
- 95% CI: [0.0308, 0.0863]
- Conclusion: Slightly larger effect with weights

#### 6.3 Heterogeneity by Sex
- Males: 0.0504 (SE: 0.0172, p=0.0035)
- Females: 0.0498 (SE: 0.0228, p=0.0288)
- Conclusion: Effects similar for both genders

#### 6.4 Event Study
Year-specific coefficients (relative to 2011):

| Year | Coefficient | SE     | 95% CI              |
|------|-------------|--------|---------------------|
| 2008 | -0.0515     | 0.0272 | [-0.1049, 0.0019]   |
| 2009 | -0.0379     | 0.0273 | [-0.0915, 0.0156]   |
| 2010 | -0.0580     | 0.0271 | [-0.1111, -0.0049]  |
| 2011 | (reference) | ---    | ---                 |
| 2013 | 0.0170      | 0.0280 | [-0.0380, 0.0719]   |
| 2014 | -0.0189     | 0.0284 | [-0.0745, 0.0366]   |
| 2015 | 0.0205      | 0.0293 | [-0.0369, 0.0778]   |
| 2016 | 0.0358      | 0.0293 | [-0.0216, 0.0932]   |

**Interpretation**: Pre-treatment coefficients are negative but mostly not statistically significant (except 2010 at 5% level). Pattern suggests general support for parallel trends, though some year-to-year variation exists.

#### 6.5 Placebo Test
- Fake treatment at 2010 (using only pre-period data)
- Placebo coefficient: 0.0165
- SE: 0.0195
- p-value: 0.396
- Conclusion: No spurious effect detected, supports parallel trends

---

### 7. Commands Executed

```python
# Main analysis script
python analysis.py
```

Key Python packages used:
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)
- scipy (statistical tests)

---

### 8. Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `analysis_results.json` | Key results in JSON format |
| `analysis_output.txt` | Full console output |
| `replication_report_60.tex` | LaTeX source for report |
| `replication_report_60.pdf` | Final PDF report |
| `run_log_60.md` | This log file |

---

### 9. Interpretation and Conclusions

**Main Finding**: DACA eligibility is associated with a statistically significant 5.01 percentage point increase in the probability of full-time employment (p < 0.001).

**Interpretation**:
- This represents approximately an 8% relative increase from the baseline full-time employment rate of 62.6% for the treatment group
- The effect is economically meaningful and statistically robust
- Similar effects observed for both males and females

**Caveats**:
1. This is an intent-to-treat estimate (eligibility, not actual DACA receipt)
2. Repeated cross-section, not panel data
3. Treatment and control groups differ by age (by design)
4. Some pre-treatment year coefficients show marginally significant deviations from zero

---

### 10. Summary Statistics for Reference

**Sample Size**: 17,382
**Treatment Group (ELIGIBLE=1)**: 11,382 (65.5%)
**Control Group (ELIGIBLE=0)**: 6,000 (34.5%)
**Pre-Period**: 9,527 (54.8%)
**Post-Period**: 7,855 (45.2%)
**Overall FT Rate**: 64.9%

---

*Log completed: January 2026*
