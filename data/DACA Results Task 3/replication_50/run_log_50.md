# DACA Replication Study - Run Log

## Replication ID: 50
## Date: January 27, 2026

---

## 1. Overview

This document logs all commands, decisions, and key outputs from the independent replication study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico.

---

## 2. Data Source and Preparation

### 2.1 Data Files
- **Primary data file**: `data/prepared_data_numeric_version.csv`
- **Data dictionary**: `data/acs_data_dict.txt`
- **Labelled version** (not used): `data/prepared_data_labelled_version.csv`

### 2.2 Data Characteristics
- **Total observations**: 17,382
- **Years covered**: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- **Variables**: 105 columns

### 2.3 Key Variables Used
| Variable | Description |
|----------|-------------|
| FT | Binary: 1 = full-time employment (35+ hours/week), 0 = otherwise |
| ELIGIBLE | Binary: 1 = treatment group (ages 26-30), 0 = control (ages 31-35) |
| AFTER | Binary: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011) |
| PERWT | Person weight from ACS |
| SEX | 1 = Male, 2 = Female |
| MARST | Marital status (1-2 = married) |
| NCHILD | Number of children |
| AGE | Age of individual |
| STATEFIP | State FIPS code |
| YEAR | Survey year |

---

## 3. Sample Sizes

### 3.1 By Treatment Group and Period
```
              Pre-DACA    Post-DACA
Control        3,294       2,706
Treated        6,233       5,149
```

### 3.2 By Year
```
Year    Control    Treated
2008      848       1,506
2009      816       1,563
2010      851       1,593
2011      779       1,571
2013      747       1,377
2014      707       1,349
2015      623       1,227
2016      629       1,196
```

---

## 4. Analytical Approach

### 4.1 Research Design
- **Method**: Difference-in-Differences (DiD)
- **Treatment group**: Individuals aged 26-30 as of June 2012 (DACA-eligible)
- **Control group**: Individuals aged 31-35 as of June 2012 (ineligible due to age)
- **Outcome**: Full-time employment (FT)
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016

### 4.2 Key Decisions

1. **No sample restrictions applied**: Following instructions, the entire provided sample was used without dropping observations.

2. **Covariates included**:
   - Male (binary indicator derived from SEX)
   - Married (binary indicator from MARST)
   - Number of children (NCHILD)
   - Age (AGE)

3. **Fixed effects**: Year fixed effects included in preferred specification; state fixed effects tested as robustness check.

4. **Standard errors**: Heteroskedasticity-robust (HC1) standard errors used in all models except basic OLS.

5. **Preferred specification**: Model with year fixed effects and demographic controls (but without state fixed effects) selected as preferred based on balance between controlling for confounders and avoiding overfitting.

---

## 5. Commands Executed

### 5.1 Python Analysis Script (`analysis.py`)

```python
# Load data
df = pd.read_csv("data/prepared_data_numeric_version.csv")

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()

# Model 2: Robust SE
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')

# Model 3: With demographic controls
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + AGE',
                 data=df).fit(cov_type='HC1')

# Model 4: Year FE (PREFERRED)
model4 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + AGE + [year dummies]',
                 data=df).fit(cov_type='HC1')

# Model 5: State + Year FE
model5 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + AGE + [year dummies] + [state dummies]',
                 data=df).fit(cov_type='HC1')

# Weighted regression
model_weighted = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + AGE',
                         data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Event study
for year in years_in_data:
    df[f'YEAR_{year}_ELIGIBLE'] = ((df['YEAR'] == year) & (df['ELIGIBLE'] == 1)).astype(int)
model_event = smf.ols('FT ~ ELIGIBLE + [year dummies] + [year-eligible interactions]', data=df).fit(cov_type='HC1')

# Pre-trend test
pre_data = df[df['AFTER'] == 0]
pre_model = smf.ols('FT ~ ELIGIBLE + YEAR + YEAR*ELIGIBLE', data=pre_data).fit(cov_type='HC1')
```

### 5.2 Figure Generation Script (`create_figures.py`)

Generated the following figures:
1. `figure1_parallel_trends.pdf` - Time series by group
2. `figure2_event_study.pdf` - Event study coefficients
3. `figure3_did_bars.pdf` - DiD visualization
4. `figure4_sample_size.pdf` - Sample size by year
5. `figure5_robustness.pdf` - Robustness comparison
6. `figure6_by_gender.pdf` - Gender heterogeneity

### 5.3 LaTeX Compilation

```bash
pdflatex -interaction=nonstopmode replication_report_50.tex  # First pass
pdflatex -interaction=nonstopmode replication_report_50.tex  # Second pass (cross-refs)
pdflatex -interaction=nonstopmode replication_report_50.tex  # Third pass (final)
```

---

## 6. Key Results

### 6.1 Simple DiD Calculation
```
                    Pre-DACA    Post-DACA    Difference
Treated (26-30)      0.6263      0.6658       +0.0394
Control (31-35)      0.6697      0.6449       -0.0248
                                              --------
DiD Estimate                                   0.0643
```

### 6.2 Regression Estimates

| Specification | DiD Estimate | Std. Error | p-value |
|--------------|--------------|------------|---------|
| Basic DiD | 0.0643 | 0.0153 | <0.001 |
| Robust SE | 0.0643 | 0.0153 | <0.001 |
| Demographic controls | 0.0581 | 0.0142 | <0.001 |
| **Year FE (preferred)** | **0.0566** | **0.0142** | **<0.001** |
| State + Year FE | 0.0568 | 0.0142 | <0.001 |
| Weighted | 0.0674 | 0.0168 | <0.001 |

### 6.3 Preferred Estimate Summary
- **Effect Size**: 0.0566 (5.66 percentage points)
- **Standard Error**: 0.0142
- **95% Confidence Interval**: [0.0288, 0.0843]
- **p-value**: < 0.001
- **Sample Size**: 17,382

### 6.4 Robustness Checks

#### By Gender
- Males: DiD = 0.0615 (SE = 0.0170)
- Females: DiD = 0.0452 (SE = 0.0232)

#### Alternative Functional Forms
- Probit coefficient: 0.1683 (SE = 0.0433)
- Logit coefficient: 0.2761 (SE = 0.0720)
- Logit odds ratio: 1.318

### 6.5 Event Study Coefficients (relative to 2011)
```
Year    Coefficient    SE       p-value
2008      -0.0591     0.0289    0.041
2009      -0.0388     0.0297    0.191
2010      -0.0663     0.0294    0.024
2011       0.0000     ---       ---    (reference)
2013       0.0188     0.0306    0.539
2014      -0.0088     0.0308    0.774
2015       0.0303     0.0316    0.338
2016       0.0491     0.0314    0.118
```

### 6.6 Parallel Trends Test
- Pre-trend coefficient (YEAR × ELIGIBLE): 0.0151
- p-value: 0.098
- Interpretation: Marginally significant at 10% level; some evidence of differential pre-trends warrants cautious interpretation.

---

## 7. Files Produced

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `results.json` | Key results in JSON format |
| `yearly_means.csv` | Yearly FT rates by group |
| `figure1_parallel_trends.pdf` | Parallel trends figure |
| `figure2_event_study.pdf` | Event study figure |
| `figure3_did_bars.pdf` | DiD bar chart |
| `figure4_sample_size.pdf` | Sample size chart |
| `figure5_robustness.pdf` | Robustness comparison |
| `figure6_by_gender.pdf` | Gender heterogeneity |
| `replication_report_50.tex` | LaTeX source |
| `replication_report_50.pdf` | Final report (20 pages) |
| `run_log_50.md` | This run log |

---

## 8. Interpretation and Conclusions

### 8.1 Main Finding
DACA eligibility is estimated to have increased full-time employment by 5.66 percentage points among eligible Hispanic-Mexican individuals born in Mexico. This represents approximately an 8-9% increase relative to the pre-DACA baseline.

### 8.2 Robustness
The estimate is robust across multiple specifications:
- Stable when adding demographic controls
- Stable when adding year and state fixed effects
- Similar magnitude with survey weights
- Positive for both males and females
- Confirmed by probit and logit models

### 8.3 Limitations
1. Some evidence of differential pre-trends (marginally significant)
2. Intent-to-treat (ITT) estimate, not treatment-on-treated
3. Repeated cross-section (not panel data)
4. Possible lifecycle effects confounding age-based comparison

### 8.4 Confidence Assessment
The effect is statistically significant at conventional levels and economically meaningful. However, the presence of some differential pre-trends suggests interpreting the results with caution.

---

## 9. Software Environment

- **Python**: 3.x
- **Packages**: pandas, numpy, statsmodels, scipy, matplotlib
- **LaTeX**: pdfTeX (MiKTeX)
- **Operating System**: Windows

---

## End of Run Log
