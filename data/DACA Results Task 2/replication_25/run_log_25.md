# DACA Replication Study - Run Log

## Study Information
- **Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico
- **Date**: January 2026
- **Replication ID**: 25

---

## 1. Data Sources and Preparation

### 1.1 Data Files Used
- **Main data file**: `data/data.csv` (6.26 GB, 33,851,424 observations)
- **Data dictionary**: `data/acs_data_dict.txt`
- **Source**: American Community Survey (ACS) via IPUMS USA
- **Years**: 2006-2016 (1-year ACS samples)

### 1.2 Key Variables from IPUMS
| Variable | Description | Used For |
|----------|-------------|----------|
| YEAR | Census/survey year | Time period identification |
| HISPAN | Hispanic origin | Sample restriction (=1 for Mexican) |
| BPL | Birthplace | Sample restriction (=200 for Mexico) |
| CITIZEN | Citizenship status | Sample restriction (=3 for non-citizen) |
| YRIMMIG | Year of immigration | Eligibility criteria |
| BIRTHYR | Year of birth | Age calculation |
| BIRTHQTR | Quarter of birth | Precise age at DACA |
| UHRSWORK | Usual hours worked per week | Outcome variable |
| PERWT | Person weight | Survey weighting |
| SEX | Sex | Control variable |
| AGE | Age | Control variable |
| EDUC | Educational attainment | Control variable |
| MARST | Marital status | Control variable |
| STATEFIP | State FIPS code | Fixed effects |

---

## 2. Sample Construction Decisions

### 2.1 Sequential Filtering Process
```
Step 1: Load ACS 2006-2016 data
        Result: 33,851,424 observations

Step 2: Filter to Hispanic-Mexican (HISPAN = 1)
        Result: 2,945,521 observations (91.3% reduction)

Step 3: Filter to born in Mexico (BPL = 200)
        Result: 991,261 observations (66.3% reduction)

Step 4: Filter to non-citizens (CITIZEN = 3)
        Result: 701,347 observations (29.2% reduction)
        Rationale: Cannot distinguish documented vs undocumented;
                   assume non-citizens without papers are undocumented

Step 5: Filter to arrived before age 16
        Calculation: age_at_immig = YRIMMIG - BIRTHYR
        Condition: age_at_immig < 16
        Result: 205,327 observations (70.7% reduction)

Step 6: Filter to continuous residence since 2007
        Condition: YRIMMIG <= 2007
        Result: 195,023 observations (5.0% reduction)
```

### 2.2 Treatment and Control Group Definition
**Age at DACA implementation (June 15, 2012):**
- For Q1/Q2 births (Jan-Jun): age = 2012 - BIRTHYR
- For Q3/Q4 births (Jul-Dec): age = 2012 - BIRTHYR - 1

**Groups:**
- **Treatment**: Age 26-30 at DACA (27,903 obs) - DACA eligible
- **Control**: Age 31-35 at DACA (19,515 obs) - Just ineligible due to age

### 2.3 Time Period Definition
- **Pre-period**: 2006-2011 (before DACA)
- **Post-period**: 2013-2016 (after DACA)
- **Excluded**: 2012 (mid-year implementation)

**Final analytic sample**: 43,238 observations

---

## 3. Outcome Variable Definition

**Full-time employment** = UHRSWORK >= 35

- UHRSWORK measures usual hours worked per week
- Threshold of 35 hours follows standard BLS definition
- Binary outcome: 1 if full-time, 0 otherwise

---

## 4. Empirical Methods

### 4.1 Main Specification: Difference-in-Differences
```
Model: Y_it = α + β1*Treated_i + β2*Post_t + β3*(Treated_i × Post_t) + X'γ + ε_it

Where:
- Y_it = Full-time employment indicator
- Treated_i = 1 if age 26-30 at DACA
- Post_t = 1 if year >= 2013
- β3 = DiD estimate (parameter of interest)
```

### 4.2 Model Specifications Estimated
1. **Model 1**: Basic DiD (no controls)
2. **Model 2**: DiD + demographics (female, married, age, age²)
3. **Model 3**: DiD + demographics + education (preferred)
4. **Model 4**: DiD + controls + state fixed effects
5. **Model 5**: DiD + controls + state FE + year FE

### 4.3 Estimation Details
- **Estimator**: Weighted Least Squares (WLS) using PERWT
- **Standard Errors**: Heteroskedasticity-robust (HC1)
- **Software**: Python (statsmodels)

---

## 5. Key Analytical Decisions and Rationale

### Decision 1: Age Range Selection
- **Choice**: Treatment 26-30, Control 31-35
- **Rationale**: Control group meets all DACA criteria except age; provides close counterfactual

### Decision 2: Exclusion of 2012
- **Choice**: Drop all 2012 observations
- **Rationale**: DACA implemented June 15, 2012; cannot distinguish pre/post within year

### Decision 3: Full-time threshold
- **Choice**: 35+ hours/week
- **Rationale**: Standard BLS definition of full-time work

### Decision 4: Preferred specification
- **Choice**: Model 3 (DiD with demographic and education controls)
- **Rationale**: Controls for key confounders while maintaining statistical power; consistent with other specifications

### Decision 5: Treatment of citizenship
- **Choice**: Use CITIZEN = 3 (non-citizen) as proxy for unauthorized status
- **Rationale**: ACS cannot distinguish documented/undocumented; following research design instructions

---

## 6. Commands Executed

### 6.1 Python Analysis Script
```python
# Main analysis script: analysis.py
# Key commands:

# Load data in chunks (memory efficiency)
for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=500000):
    chunks.append(chunk)

# Sample restrictions
df_mex = df[df['HISPAN'] == 1]
df_mex = df_mex[df_mex['BPL'] == 200]
df_mex = df_mex[df_mex['CITIZEN'] == 3]
df_mex = df_mex[df_mex['age_at_immig'] < 16]
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007]

# DiD estimation
model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + age + age_sq + educ_hs + educ_somecoll + educ_college',
                 data=df_analysis, weights=df_analysis['PERWT'])
results3 = model3.fit(cov_type='HC1')
```

### 6.2 Figure Generation
```python
# create_figures.py
# Generated 4 figures:
# - figure1_event_study.png
# - figure2_trends.png
# - figure3_model_comparison.png
# - figure4_did_visualization.png
```

### 6.3 LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_25.tex
pdflatex -interaction=nonstopmode replication_report_25.tex  # Second pass for references
```

---

## 7. Results Summary

### 7.1 Main Finding
| Statistic | Value |
|-----------|-------|
| DiD Effect (preferred) | 0.0645 |
| Standard Error | 0.0146 |
| 95% CI Lower | 0.0359 |
| 95% CI Upper | 0.0930 |
| t-statistic | 4.42 |
| p-value | < 0.0001 |
| Sample Size | 43,238 |

**Interpretation**: DACA eligibility increased full-time employment by 6.45 percentage points.

### 7.2 Robustness
- Effect stable across Models 1-4 (5.9-6.5 pp)
- Model 5 (with year FE) shows smaller, insignificant effect (1.9 pp)
- Effect larger for women (7.8 pp) than men (5.0 pp)
- Event study shows no significant pre-trends

### 7.3 Pre-Post Means
|  | Pre-DACA | Post-DACA | Change |
|--|----------|-----------|--------|
| Treatment (26-30) | 61.47% | 63.39% | +1.92 pp |
| Control (31-35) | 64.61% | 61.36% | -3.25 pp |
| **DiD** | | | **+5.17 pp** (raw) |

---

## 8. Output Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Main Python analysis script |
| `create_figures.py` | Figure generation script |
| `regression_results.csv` | DiD coefficient estimates |
| `summary_statistics.csv` | Summary statistics by group |
| `event_study_results.csv` | Year-by-year coefficients |
| `figure1_event_study.png` | Event study plot |
| `figure2_trends.png` | Employment trends bar chart |
| `figure3_model_comparison.png` | Model comparison chart |
| `figure4_did_visualization.png` | DiD visualization |
| `replication_report_25.tex` | LaTeX report source |
| `replication_report_25.pdf` | Final PDF report (22 pages) |
| `run_log_25.md` | This run log |

---

## 9. Potential Limitations Noted

1. **Proxy for unauthorized status**: CITIZEN=3 includes all non-citizens, not just undocumented
2. **Age-based identification**: Different cohorts may face different labor market conditions
3. **Cross-sectional data**: Cannot track same individuals over time
4. **Measurement**: Self-reported hours; potential underreporting
5. **External validity**: Results specific to Hispanic-Mexican immigrants aged 26-35

---

## 10. Reproducibility Notes

To reproduce this analysis:
1. Ensure Python 3.x with pandas, numpy, statsmodels, matplotlib
2. Place data.csv in ./data/ directory
3. Run: `python analysis.py`
4. Run: `python create_figures.py`
5. Compile: `pdflatex replication_report_25.tex` (run twice)

---

*Log completed: January 2026*
