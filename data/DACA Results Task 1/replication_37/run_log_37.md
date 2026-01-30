# Run Log - DACA Replication Study (Replication 37)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA employment effects study.

**Date:** January 25, 2026
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

---

## Data Sources

### Primary Data
- **File:** `data/data.csv` (6.27 GB)
- **Source:** American Community Survey (ACS) via IPUMS USA
- **Years:** 2006-2016 (one-year ACS samples)
- **Total observations:** 33,851,424

### Supporting Data
- **Data dictionary:** `data/acs_data_dict.txt`
- **State-level policy data:** `data/state_demo_policy.csv` (not used in main analysis)

---

## Key Analytical Decisions

### 1. Sample Definition

**Decision:** Restrict to Hispanic-Mexican ethnicity (HISPAN = 1) AND born in Mexico (BPL = 200)
- **Rationale:** Research question specifically asks about "ethnically Hispanic-Mexican Mexican-born people"
- **Result:** 991,261 observations

**Decision:** Restrict to non-citizens only (CITIZEN = 3)
- **Rationale:** Per instructions, assume non-citizens without immigration papers are undocumented for DACA purposes
- **Result:** 701,347 observations

**Decision:** Exclude 2012 from analysis
- **Rationale:** DACA was implemented mid-year (June 15, 2012), and ACS doesn't identify month of collection
- **Result:** 636,722 observations

**Decision:** Restrict to working-age population (ages 18-64)
- **Rationale:** Standard definition for labor force analyses
- **Result:** 547,614 observations (final analysis sample)

### 2. DACA Eligibility Definition

DACA eligibility constructed from three criteria:

1. **Arrived before age 16:**
   - Formula: `YRIMMIG - BIRTHYR < 16`
   - Result: 205,327 (29.3%) arrived before age 16

2. **Born after June 15, 1981:**
   - Formula: `BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)`
   - Rationale: Must not have turned 31 by June 15, 2012
   - Result: 230,009 (32.8%) meet this criterion

3. **In US since June 15, 2007:**
   - Formula: `YRIMMIG <= 2007`
   - Rationale: Continuous presence requirement
   - Result: 654,693 (93.3%) meet this criterion

**Combined DACA eligibility:** All three criteria must be met
- Result: 133,120 (19.0%) classified as DACA-eligible

### 3. Outcome Variable

**Decision:** Full-time employment defined as UHRSWORK >= 35
- **Rationale:** Bureau of Labor Statistics standard definition
- **Result:** 357,059 (50.9%) work full-time

### 4. Treatment Period

**Decision:** Post-treatment = 2013-2016, Pre-treatment = 2006-2011
- **Rationale:** DACA implemented June 15, 2012; 2012 excluded as transition year

### 5. Identification Strategy

**Decision:** Difference-in-differences design
- **Treatment group:** DACA-eligible individuals
- **Control group:** DACA-ineligible individuals (same population, different age/arrival characteristics)
- **Rationale:** Exploits exogenous variation in eligibility based on age at arrival and birth cohort

### 6. Model Specification

**Preferred specification (Model 6):**
```
fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married
         + hs_diploma + some_college + college_plus + C(YEAR) + C(STATEFIP)
```

With:
- Person weights (PERWT)
- Robust (HC1) standard errors

**Controls included:**
- Age and age squared (lifecycle effects)
- Female indicator (gender gap in employment)
- Married indicator (family status)
- Education indicators (human capital)
- Year fixed effects (common time shocks)
- State fixed effects (geographic heterogeneity)

---

## Commands Executed

### Data Processing and Analysis

```python
# Main analysis script: analysis.py
# Python packages used:
# - pandas (data manipulation)
# - numpy (numerical operations)
# - statsmodels (regression analysis)
# - matplotlib/seaborn (visualization)

# Key commands in analysis.py:

# 1. Load data
df = pd.read_csv("data/data.csv")

# 2. Filter to target population
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)]
df_mex = df_mex[df_mex['CITIZEN'] == 3]

# 3. Create DACA eligibility indicators
df_mex['age_at_immig'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
df_mex['arrived_before_16'] = (df_mex['age_at_immig'] < 16).astype(int)
df_mex['born_after_june_1981'] = ((df_mex['BIRTHYR'] >= 1982) |
    ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))).astype(int)
df_mex['in_us_by_2007'] = (df_mex['YRIMMIG'] <= 2007).astype(int)
df_mex['daca_eligible'] = (df_mex['arrived_before_16'] &
    df_mex['born_after_june_1981'] & df_mex['in_us_by_2007']).astype(int)

# 4. Create outcome variable
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

# 5. Create post-treatment indicator and interaction
df_mex['post_daca'] = (df_mex['YEAR'] >= 2013).astype(int)
df_analysis['eligible_x_post'] = df_analysis['daca_eligible'] * df_analysis['post_daca']

# 6. Run weighted DiD regression with robust SEs
model6 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq +
    female + married + hs_diploma + some_college + college_plus +
    C(YEAR) + C(STATEFIP)', data=df_analysis,
    weights=df_analysis['PERWT']).fit(cov_type='HC1')
```

### Shell Commands

```bash
# Run analysis script
cd "C:/Users/seraf/DACA Results Task 1/replication_37"
python analysis.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_37.tex
pdflatex -interaction=nonstopmode replication_report_37.tex
pdflatex -interaction=nonstopmode replication_report_37.tex
pdflatex -interaction=nonstopmode replication_report_37.tex
```

---

## Results Summary

### Main Finding

**Preferred estimate (Model 6):**
- DiD Coefficient: **0.0177** (1.77 percentage points)
- Standard Error: 0.0045
- 95% CI: [0.0089, 0.0265]
- p-value: < 0.001

**Interpretation:** DACA eligibility increased the probability of full-time employment by 1.77 percentage points among Mexican-born non-citizens.

### Sample Sizes
- Total analysis sample: 547,614
- DACA-eligible: 71,347
- DACA-ineligible: 476,267

### Model Comparison

| Model | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| Basic DiD (no controls) | 0.0605 | 0.0040 | < 0.001 |
| DiD + controls | 0.0263 | 0.0036 | < 0.001 |
| DiD + controls + year FE | 0.0193 | 0.0036 | < 0.001 |
| DiD + controls + year + state FE | 0.0186 | 0.0036 | < 0.001 |
| Weighted + controls + FE | 0.0177 | 0.0035 | < 0.001 |
| **Weighted + robust SE (preferred)** | **0.0177** | **0.0045** | **< 0.001** |

### Robustness Checks

| Check | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| Ages 16-55 | 0.0325 | 0.0042 | < 0.001 |
| Including 2012 | 0.0102 | 0.0043 | 0.017 |
| Men only | 0.0123 | 0.0058 | 0.033 |
| Women only | 0.0156 | 0.0069 | 0.024 |
| Placebo (fake 2009) | -0.0039 | 0.0061 | 0.515 |

---

## Output Files Generated

1. **analysis.py** - Main analysis script
2. **replication_report_37.tex** - LaTeX source for report
3. **replication_report_37.pdf** - Final PDF report (21 pages)
4. **figure1_trends.png** - Unweighted trends plot
5. **figure2_weighted_trends.png** - Weighted trends plot
6. **figure3_age_distribution.png** - Age distribution by eligibility
7. **summary_statistics.csv** - Summary statistics table
8. **yearly_trends.csv** - Year-by-year employment trends
9. **run_log_37.md** - This run log

---

## Notes and Limitations

1. **Eligibility measurement:** Cannot observe all DACA criteria in ACS (education, criminal history, continuous presence). Likely includes some ineligible individuals in "eligible" group, biasing estimates toward zero.

2. **Intent-to-treat:** Estimates reflect eligibility effect, not actual DACA receipt. True effect of receiving DACA would be larger.

3. **Parallel trends:** Placebo test (p = 0.515) supports parallel trends assumption.

4. **External validity:** Results specific to Mexican-born population; may not generalize to other origin countries.

5. **Linear probability model:** Used for ease of interpretation with fixed effects. Coefficients represent percentage point changes.

---

## Conclusion

The analysis finds a statistically significant positive effect of DACA eligibility on full-time employment (1.77 pp, p < 0.001). This effect is robust to alternative specifications and passes a placebo test. The magnitude is modest but economically meaningful given the size of the affected population.
