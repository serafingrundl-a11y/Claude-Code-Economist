# DACA Replication Study - Run Log 28

## Study Overview
**Research Question:** What was the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States?

**Identification Strategy:** Difference-in-Differences (DiD)
- Treatment group: Ages 26-30 at the time of DACA implementation (June 15, 2012)
- Control group: Ages 31-35 at the time of DACA implementation
- Pre-period: 2008-2011
- Post-period: 2013-2016 (2012 excluded as transition year)

## Session Log

### Initial Setup
- **Date:** 2026-01-27
- **Task:** Independent replication of DACA effects on full-time employment

### Data Files
- `prepared_data_numeric_version.csv` - Main analysis dataset
- `prepared_data_labelled_version.csv` - Labeled version for reference
- `acs_data_dict.txt` - Data dictionary

### Key Variables
- `FT`: Full-time employment (1 = yes, 0 = no) - OUTCOME
- `ELIGIBLE`: DACA eligibility (1 = eligible/treatment, 0 = control)
- `AFTER`: Post-DACA period (1 = 2013-2016, 0 = 2008-2011)
- `PERWT`: Person weight for ACS sampling

---

## Analysis Steps

### Step 1: Data Loading and Exploration
**Command:** `python analysis.py`

**Key Findings:**
- Dataset shape: 17,382 observations x 105 variables
- Full-time employment rate: 64.9% overall
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-period: 9,527 observations
- Post-period: 7,855 observations

### Step 2: Sample Statistics Calculation

**Mean Full-Time Employment (Weighted by PERWT):**

| Group | Pre-DACA | Post-DACA |
|-------|----------|-----------|
| Control (31-35) | 0.689 | 0.663 |
| Treatment (26-30) | 0.637 | 0.686 |

**Simple DiD Estimate (Weighted):** 0.075

### Step 3: Descriptive Statistics (Pre-DACA Period)

| Variable | Treatment | Control |
|----------|-----------|---------|
| Age | 25.74 | 30.52 |
| Female | 48.1% | 45.6% |
| Married | 41.1% | 52.9% |
| Number of Children | 0.94 | 1.54 |
| FT Employment | 62.6% | 67.0% |

**Education Distribution:**
- High School Degree: Treatment 70.9%, Control 73.5%
- Some College: Treatment 18.3%, Control 15.7%
- BA+: Treatment 5.5%, Control 5.6%

### Step 4: Main Difference-in-Differences Analysis

**Model Specifications:**

1. **Model 1 - Basic DiD (Unweighted):**
   - DiD Coefficient: 0.0643
   - SE: 0.0153
   - 95% CI: [0.034, 0.094]
   - t-stat: 4.20

2. **Model 2 - Basic DiD (Weighted):**
   - DiD Coefficient: 0.0748
   - SE: 0.0152
   - 95% CI: [0.045, 0.105]
   - t-stat: 4.93

3. **Model 3 - Year Fixed Effects:**
   - DiD Coefficient: 0.0721
   - SE: 0.0151
   - 95% CI: [0.042, 0.102]
   - t-stat: 4.76

4. **Model 4 - Covariates Added:**
   - Covariates: female, married, NCHILD, education dummies
   - DiD Coefficient: 0.0612
   - SE: 0.0142
   - 95% CI: [0.034, 0.089]
   - t-stat: 4.32

5. **Model 5 - State Fixed Effects:**
   - DiD Coefficient: 0.0607
   - SE: 0.0142
   - 95% CI: [0.033, 0.089]
   - t-stat: 4.28

### Step 5: Robust Standard Errors

- **Robust (HC1) SE:** 0.0166, t-stat: 3.65
- **State-Clustered SE:** 0.0216, t-stat: 2.81, p = 0.007

### Step 6: Parallel Trends Check (Event Study)

| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2008 | -0.064 | 0.027 | [-0.117, -0.010] |
| 2009 | -0.047 | 0.027 | [-0.100, 0.006] |
| 2010 | -0.076 | 0.027 | [-0.130, -0.023] |
| 2011 | 0.000 | (reference) | - |
| 2013 | 0.016 | 0.028 | [-0.039, 0.071] |
| 2014 | -0.013 | 0.028 | [-0.068, 0.042] |
| 2015 | -0.007 | 0.029 | [-0.064, 0.050] |
| 2016 | 0.065 | 0.029 | [0.007, 0.122] |

**Note:** Some pre-trend coefficients are significant, suggesting caution in interpretation.

### Step 7: Heterogeneity Analysis

**By Sex:**
- Male: DiD = 0.070 (SE: 0.017), N = 9,075
- Female: DiD = 0.049 (SE: 0.023), N = 8,307

**By Education:**
- High School Degree: DiD = 0.059 (SE: 0.018), N = 12,444
- Some College: DiD = 0.061 (SE: 0.038), N = 2,877
- BA+: DiD = 0.171 (SE: 0.060), N = 1,058

### Step 8: Visualization Generation
**Command:** `python visualizations.py`

Generated files:
- `figure1_trends.png/pdf` - Employment trends
- `figure2_event_study.png/pdf` - Event study plot
- `figure3_did_bars.png/pdf` - DiD bar chart
- `figure4_heterogeneity.png/pdf` - Heterogeneity analysis
- `figure5_model_comparison.png/pdf` - Model comparison

---

## Key Decisions

1. **Used provided ELIGIBLE and AFTER variables** as instructed, rather than constructing own eligibility criteria.

2. **Included individuals not in labor force** in the analysis (FT=0), as per instructions.

3. **Applied survey weights (PERWT)** for nationally representative estimates.

4. **Clustered standard errors at state level** to account for within-state correlation.

5. **Included covariates:** female, married, number of children, education dummies (high school, some college, two-year degree, BA+).

6. **Included fixed effects:** Year FE and State FE in preferred specification.

7. **Used linear probability model (WLS)** for interpretability.

---

## Preferred Estimate

| Metric | Value |
|--------|-------|
| Model | DiD with covariates, state/year FE, weighted, clustered SE |
| Sample Size | 17,382 |
| **DiD Coefficient** | **0.061** |
| Standard Error | 0.022 |
| 95% Confidence Interval | [0.017, 0.104] |
| p-value | 0.007 |
| Treatment N | 11,382 |
| Control N | 6,000 |

---

## Interpretation

DACA eligibility is associated with a **6.1 percentage point increase** in the probability of full-time employment among the treatment group (ages 26-30) relative to the control group (ages 31-35). This effect is **statistically significant** at the 1% level (p = 0.007).

The effect represents approximately a 9.6% increase relative to the pre-DACA treatment group mean of 63.7%.

**Caveats:**
- Event study shows some evidence of differential pre-trends
- This is an intent-to-treat estimate (not all eligible applied for DACA)
- Results apply to Hispanic-Mexican, Mexican-born individuals only

---

## Output Files

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `visualizations.py` | Visualization generation script |
| `results_summary.json` | JSON with preferred estimate |
| `model_comparison.csv` | All model specifications |
| `event_study_results.csv` | Event study coefficients |
| `heterogeneity_sex.csv` | Heterogeneity by sex |
| `heterogeneity_education.csv` | Heterogeneity by education |
| `summary_statistics.csv` | Descriptive statistics |
| `sample_sizes_by_year.csv` | Sample sizes by year |
| `figure1_trends.pdf` | Employment trends figure |
| `figure2_event_study.pdf` | Event study figure |
| `figure3_did_bars.pdf` | DiD visualization |
| `figure4_heterogeneity.pdf` | Heterogeneity analysis |
| `figure5_model_comparison.pdf` | Model comparison |
| `replication_report_28.tex` | LaTeX report |
| `replication_report_28.pdf` | Final PDF report |

---

## Session End
- **Date:** 2026-01-27
- **Status:** Complete
