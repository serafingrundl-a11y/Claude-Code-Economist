# Run Log: DACA Replication Study

## Overview
This log documents all commands, key decisions, and analysis steps for the independent replication of the DACA effect on full-time employment.

---

## 1. Data Exploration

### 1.1 Initial Data Inspection

**Command:** Load and inspect the dataset
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Total observations: {len(df)}")  # 17,382 observations
```

**Key Findings:**
- Dataset contains 17,382 observations
- 105 variables including demographics, employment, and state policy variables
- Years covered: 2008-2011 (pre-DACA) and 2013-2016 (post-DACA)
- 2012 data excluded (implementation year)

### 1.2 Variable Inspection

**Key Variables Identified:**
- `FT`: Full-time employment (1 = 35+ hours/week, 0 = otherwise)
- `ELIGIBLE`: Treatment indicator (1 = ages 26-30 in June 2012, 0 = ages 31-35)
- `AFTER`: Post-treatment indicator (1 = 2013-2016, 0 = 2008-2011)
- `PERWT`: ACS person weight for population estimates

**Sample Distribution:**
| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Control (ELIGIBLE=0) | 3,294 | 2,706 | 6,000 |
| Treated (ELIGIBLE=1) | 6,233 | 5,149 | 11,382 |
| **Total** | 9,527 | 7,855 | 17,382 |

---

## 2. Key Decisions

### 2.1 Identification Strategy
**Decision:** Difference-in-Differences (DiD) approach
- Treatment: DACA eligibility based on age cutoff
- Treatment group: Ages 26-30 in June 2012 (ELIGIBLE=1)
- Control group: Ages 31-35 in June 2012 (ELIGIBLE=0)
- Pre-period: 2008-2011
- Post-period: 2013-2016

**Rationale:** This follows the specified research design using age-based eligibility as a natural experiment.

### 2.2 Weighting
**Decision:** Use ACS person weights (PERWT) for all primary specifications
**Rationale:** Survey weights ensure population-representative estimates from the ACS sample.

### 2.3 Standard Errors
**Decision:** State-clustered standard errors for primary inference
**Rationale:** Accounts for within-state correlation over time and potential state-level policy effects.

### 2.4 Covariates
**Decision:** Include individual-level covariates:
- Female indicator
- Married indicator
- Has children indicator
- Education category dummies

**Rationale:** Controls for compositional differences between treatment and control groups; improves precision.

### 2.5 Fixed Effects
**Decision:** Include year fixed effects and state fixed effects
**Rationale:**
- Year FE control for aggregate time trends
- State FE control for time-invariant state characteristics

### 2.6 Sample
**Decision:** Use full provided sample without additional restrictions
**Rationale:** As instructed, the entire provided file constitutes the analytic sample.

---

## 3. Analysis Commands

### 3.1 Descriptive Statistics

```python
# Full-time employment rates by group and period
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])

# Weighted means
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE']==eligible) & (df['AFTER']==after)]
        weighted_mean = np.average(subset['FT'], weights=subset['PERWT'])
```

**Results:**
| Group | Period | Weighted Mean |
|-------|--------|---------------|
| Control | Pre | 0.6886 |
| Control | Post | 0.6629 |
| Treated | Pre | 0.6369 |
| Treated | Post | 0.6860 |

### 3.2 Simple DiD Calculation

```python
# Manual DiD calculation
diff_control = ft_01_w - ft_00_w  # Control change: -0.0257
diff_treated = ft_11_w - ft_10_w  # Treated change: +0.0491
did_weighted = diff_treated - diff_control  # DiD: 0.0748
```

### 3.3 Regression Models

**Model 1: Basic DiD (unweighted)**
```python
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
```
Result: DiD = 0.0643 (SE = 0.0153)

**Model 2: Basic DiD (weighted)**
```python
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
```
Result: DiD = 0.0748 (SE = 0.0181)

**Model 3: Year Fixed Effects**
```python
model3 = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
```
Result: DiD = 0.0721 (SE = 0.0181)

**Model 4: With Covariates**
```python
model4 = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_SOMECOL + EDUC_2YR + EDUC_BA',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
```
Result: DiD = 0.0584 (SE = 0.0167)

**Model 5: State Fixed Effects**
```python
model5 = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_SOMECOL + EDUC_2YR + EDUC_BA + C(STATE_factor)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
```
Result: DiD = 0.0578 (SE = 0.0166)

**Model 5 with Clustered SE (Preferred)**
```python
model5_cluster = smf.wls(...).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```
Result: DiD = 0.0578 (SE = 0.0210), p = 0.006

### 3.4 Event Study

```python
# Create year interaction terms
df['ELIGIBLE_2008'] = df['ELIGIBLE'] * (df['YEAR'] == 2008).astype(int)
# ... (similar for other years, 2011 as reference)

model_event = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_2008 + ELIGIBLE_2009 + ELIGIBLE_2010 + ELIGIBLE_2013 + ELIGIBLE_2014 + ELIGIBLE_2015 + ELIGIBLE_2016 + covariates + C(STATE_factor)',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')
```

**Event Study Results (relative to 2011):**
| Year | Coefficient | SE |
|------|-------------|-----|
| 2008 | -0.0665** | 0.0320 |
| 2009 | -0.0464 | 0.0328 |
| 2010 | -0.0767** | 0.0327 |
| 2011 | 0 (ref) | -- |
| 2013 | 0.0163 | 0.0339 |
| 2014 | -0.0190 | 0.0348 |
| 2015 | -0.0117 | 0.0347 |
| 2016 | 0.0576 | 0.0351 |

### 3.5 Parallel Trends Test

```python
pre_df = df[df['AFTER'] == 0].copy()
pre_df['YEAR_TREND'] = pre_df['YEAR'] - 2008
pre_df['ELIGIBLE_TREND'] = pre_df['ELIGIBLE'] * pre_df['YEAR_TREND']

model_pretrend = smf.wls('FT ~ ELIGIBLE + YEAR_TREND + ELIGIBLE_TREND + covariates',
                         data=pre_df, weights=pre_df['PERWT']).fit(cov_type='HC1')
```
Result: ELIGIBLE_TREND = 0.0162 (SE = 0.0101, p = 0.109)
**Interpretation:** Cannot reject null of parallel pre-trends at 10% level.

### 3.6 Heterogeneity Analysis

**By Gender:**
```python
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex].copy()
    model_sex = smf.wls(...).fit(cov_type='HC1')
```
- Males: DiD = 0.0590 (SE = 0.0196), p < 0.01
- Females: DiD = 0.0438 (SE = 0.0272), p > 0.10

---

## 4. Figure Generation

```python
# Figure 1: Trends
trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
trends.plot(marker='o')
plt.savefig('figure1_trends.png')

# Figure 2: Event Study
event_df = pd.read_csv('event_study_results.csv')
plt.fill_between(years, ci_lower, ci_upper, alpha=0.3)
plt.plot(years, coefs, marker='o')
plt.savefig('figure2_event_study.png')

# Figure 3: DiD Visualization
# Plots actual and counterfactual lines
plt.savefig('figure3_did_visualization.png')

# Figure 4: Sample Size
sample_by_year.plot(kind='bar')
plt.savefig('figure4_sample_size.png')
```

---

## 5. LaTeX Report Compilation

```bash
# Compile LaTeX document
pdflatex -interaction=nonstopmode replication_report_100.tex
pdflatex -interaction=nonstopmode replication_report_100.tex
pdflatex -interaction=nonstopmode replication_report_100.tex
```

Output: `replication_report_100.pdf` (20 pages)

---

## 6. Summary of Main Results

### Preferred Specification
- **Model:** WLS with year FE, state FE, individual covariates, state-clustered SE
- **Point Estimate:** 0.0578 (5.78 percentage points)
- **Standard Error:** 0.0210 (state-clustered)
- **95% Confidence Interval:** [0.0167, 0.0989]
- **p-value:** 0.006
- **Sample Size:** 17,382

### Interpretation
DACA eligibility is associated with a 5.78 percentage point increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States. This represents approximately a 9% increase relative to the pre-treatment baseline (0.6369) and is statistically significant at the 1% level.

---

## 7. Output Files

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `generate_figures.py` | Figure generation script |
| `replication_report_100.tex` | LaTeX source |
| `replication_report_100.pdf` | Final PDF report (20 pages) |
| `did_results.csv` | DiD estimates across specifications |
| `event_study_results.csv` | Event study coefficients |
| `descriptive_stats.csv` | Descriptive statistics |
| `figure1_trends.png/pdf` | Employment trends figure |
| `figure2_event_study.png/pdf` | Event study figure |
| `figure3_did_visualization.png/pdf` | DiD visualization |
| `figure4_sample_size.png/pdf` | Sample size figure |
| `run_log_100.md` | This log file |

---

## 8. Software Environment

- **Python:** 3.x
- **Key packages:** pandas, numpy, statsmodels, matplotlib, scipy
- **LaTeX:** pdfTeX (MiKTeX distribution)
- **Operating System:** Windows

---

## 9. Reproducibility Notes

1. All analysis starts from the provided `prepared_data_numeric_version.csv`
2. No external data sources were used
3. Random seed not required (no stochastic elements)
4. Code is fully deterministic and reproducible

---

*Log completed: January 27, 2026*
