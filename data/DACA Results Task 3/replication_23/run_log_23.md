# Run Log - DACA Replication Study (Replication 23)

## Session Start: 2026-01-27

### Task Overview
Replicate analysis examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican Mexican-born individuals in the United States using a difference-in-differences design.

---

## Step 1: Read Replication Instructions

**Action:** Read `replication_instructions.docx`

**Key Points from Instructions:**
- **Research Question:** Causal impact of DACA eligibility on probability of full-time employment (≥35 hrs/week)
- **Treatment Group:** DACA-eligible individuals aged 26-30 at time of policy (June 15, 2012)
- **Control Group:** Individuals aged 31-35 at time of policy (would have been eligible but for age)
- **Method:** Difference-in-differences comparing treated vs control group, before (2008-2011) vs after (2013-2016)
- **Data:** ACS data from 2008-2016 (excluding 2012)
- **Key Variables:**
  - `ELIGIBLE`: 1 = treatment group (ages 26-30), 0 = control group (ages 31-35)
  - `AFTER`: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)
  - `FT`: 1 = full-time work (≥35 hrs/week), 0 = not full-time
- **Sample:** Already prepared - do not drop individuals based on characteristics
- **Note:** Binary IPUMS variables coded 1=No, 2=Yes; added variables (FT, AFTER, ELIGIBLE) coded 0=No, 1=Yes

---

## Step 2: Explore Data Structure

**Action:** Examined data folder contents

**Files Found:**
- `acs_data_dict.txt` - Data dictionary with IPUMS variable descriptions
- `prepared_data_labelled_version.csv` - Labeled version of data (~19MB)
- `prepared_data_numeric_version.csv` - Numeric version of data (~6.5MB)

**Data Dimensions:**
- 17,382 observations (excluding header)
- 105 variables

**Key Variables Confirmed:**
- `YEAR` - Survey year (2008-2016, excluding 2012)
- `PERWT` - Person weight (for weighted analysis)
- `FT` - Full-time employment (outcome)
- `AFTER` - Post-DACA indicator
- `ELIGIBLE` - Treatment group indicator
- `AGE_IN_JUNE_2012` - Age at time of DACA implementation
- Various demographic and state-level policy variables available for covariates

---

## Step 3: Analysis Plan

**Design:** Difference-in-Differences (DiD)

**Basic Model:**
$$FT_i = \beta_0 + \beta_1 ELIGIBLE_i + \beta_2 AFTER_t + \beta_3 (ELIGIBLE_i \times AFTER_t) + \epsilon_{it}$$

Where $\beta_3$ is the DiD estimator (treatment effect)

**Extensions to Consider:**
1. Unweighted vs. weighted (PERWT) estimates
2. Adding demographic covariates (sex, education, marital status)
3. Adding state fixed effects
4. Adding year fixed effects
5. Clustering standard errors by state

**Analysis Steps:**
1. Descriptive statistics by group and period
2. Visual inspection of parallel trends (pre-treatment)
3. Basic DiD regression
4. DiD with covariates
5. Robustness checks

---

## Step 4: Data Loading and Initial Exploration

**Command:**
```python
df = pd.read_csv('data/prepared_data_numeric_version.csv')
```

**Results:**
- Data loaded successfully: 17,382 observations, 105 variables
- Year distribution: 2008 (2,354), 2009 (2,379), 2010 (2,444), 2011 (2,350), 2013 (2,124), 2014 (2,056), 2015 (1,850), 2016 (1,825)
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Full-time employed (FT=1): 11,283 observations
- Not full-time (FT=0): 6,099 observations

**Age Verification:**
- Treatment group mean age in June 2012: 28.1 years (range: 26.0-30.75)
- Control group mean age in June 2012: 32.9 years (range: 31.0-35.0)

---

## Step 5: Descriptive Statistics

**2x2 DiD Table (Weighted):**

| Group | Pre-DACA | Post-DACA | Difference |
|-------|----------|-----------|------------|
| Control (31-35) | 0.689 | 0.663 | -0.026 |
| Treatment (26-30) | 0.637 | 0.686 | +0.049 |
| **DiD** | | | **0.075** |

**Sample Sizes:**

| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Control | 3,294 | 2,706 | 6,000 |
| Treatment | 6,233 | 5,149 | 11,382 |
| Total | 9,527 | 7,855 | 17,382 |

---

## Step 6: Difference-in-Differences Regression Analysis

### Model Specifications

**Model 1: Basic DiD (Unweighted)**
- DiD Estimate: 0.0643 (SE: 0.0153)
- p-value: <0.001

**Model 2: Basic DiD (Weighted)**
- DiD Estimate: 0.0748 (SE: 0.0152)
- p-value: <0.001

**Model 3: Year Fixed Effects (Weighted)**
- DiD Estimate: 0.0721 (SE: 0.0151)
- p-value: <0.001

**Model 4: Covariates (Weighted)**
- Covariates: Female, Married, Education dummies
- DiD Estimate: 0.0593 (SE: 0.0142)
- p-value: <0.001

**Model 5: State + Year FE (Weighted) [PREFERRED]**
- DiD Estimate: 0.0586 (SE: 0.0142)
- 95% CI: [0.031, 0.086]
- p-value: <0.001

**Model 6: Clustered SE (by State)**
- DiD Estimate: 0.0586 (Clustered SE: 0.0213)
- 95% CI: [0.017, 0.100]
- p-value: 0.006

---

## Step 7: Key Decisions and Justifications

### Decision 1: Use Linear Probability Model (OLS/WLS)
**Rationale:** LPM provides easily interpretable coefficients (percentage point changes). With large samples and probabilities not near 0 or 1, LPM and logit/probit yield similar marginal effects.

### Decision 2: Use Survey Weights (PERWT)
**Rationale:** ACS is a complex survey; weights ensure population representativeness and correct for sampling design.

### Decision 3: Include Year Fixed Effects
**Rationale:** Control for common time trends affecting both treatment and control groups (e.g., business cycle effects from Great Recession recovery).

### Decision 4: Include State Fixed Effects
**Rationale:** Control for time-invariant state-level factors that may affect employment (labor markets, state policies, industry mix).

### Decision 5: Include Demographic Covariates
**Rationale:**
- Sex: Large gender differences in full-time employment
- Marital status: May affect labor supply decisions
- Education: Strong predictor of employment

### Decision 6: Do Not Drop Observations
**Rationale:** Instructions explicitly state to use the full sample without dropping individuals based on characteristics.

### Decision 7: Preferred Specification
**Selection:** Model 5 with year FE, state FE, and covariates
**Rationale:** Standard in DACA literature; balances control for confounders with interpretability; robust standard errors increase confidence.

---

## Step 8: Create Figures

**Figures Generated:**
1. `figure1_parallel_trends.png/pdf` - FT employment trends by year and group
2. `figure2_did_bars.png/pdf` - Bar chart showing 2x2 DiD structure
3. `figure3_coef_plot.png/pdf` - Coefficient plot across specifications
4. `figure4_sample_dist.png/pdf` - Sample distribution by year
5. `figure5_event_study.png/pdf` - Event study visualization
6. `figure6_demographics.png/pdf` - Demographic comparisons

---

## Step 9: Write Replication Report

**Output:** `replication_report_23.tex`

**Contents:**
1. Abstract
2. Introduction (DACA background, eligibility criteria, research design)
3. Data (source, sample, key variables)
4. Descriptive Statistics (demographics, trends, 2x2 table)
5. Empirical Strategy (DiD framework, specifications, assumptions)
6. Results (main estimates, coefficient plot, covariate effects)
7. Robustness and Sensitivity (standard errors, weighting, fixed effects)
8. Discussion (interpretation, comparison to literature, limitations)
9. Conclusion
10. Appendices (additional figures, regression output, documentation)

---

## Step 10: Compile LaTeX to PDF

**Commands:**
```bash
pdflatex -interaction=nonstopmode replication_report_23.tex
pdflatex -interaction=nonstopmode replication_report_23.tex
pdflatex -interaction=nonstopmode replication_report_23.tex
```

**Output:** `replication_report_23.pdf` (22 pages)

---

## Final Results Summary

### Preferred Estimate
- **Effect:** 5.86 percentage points
- **Standard Error:** 0.0142 (robust), 0.0213 (clustered by state)
- **95% CI:** [0.031, 0.086]
- **p-value:** <0.001
- **Sample Size:** 17,382

### Interpretation
DACA eligibility is associated with a 5.86 percentage point increase in the probability of full-time employment among eligible Hispanic-Mexican Mexican-born individuals. This represents approximately a 9% increase relative to the pre-treatment mean of 64% for the treatment group.

The effect is:
- Statistically significant at conventional levels (p < 0.001)
- Robust across multiple specifications (range: 5.9 to 7.5 pp)
- Consistent with theoretical expectations about effects of legal work authorization

---

## Output Files Created

### Required Deliverables
1. `replication_report_23.tex` - LaTeX source
2. `replication_report_23.pdf` - Final report (22 pages)
3. `run_log_23.md` - This run log

### Analysis Files
4. `analysis.py` - Main analysis script
5. `create_figures.py` - Figure generation script

### Data Outputs
6. `model_results_summary.csv` - Summary of all model results
7. `detailed_model_output.txt` - Full regression output
8. `did_table_unweighted.csv` - Unweighted 2x2 table
9. `did_table_weighted.csv` - Weighted 2x2 table
10. `ft_trends_by_year.csv` - Yearly FT rates by group
11. `sample_sizes.csv` - Sample sizes by group/period

### Figures
12. `figure1_parallel_trends.png/pdf`
13. `figure2_did_bars.png/pdf`
14. `figure3_coef_plot.png/pdf`
15. `figure4_sample_dist.png/pdf`
16. `figure5_event_study.png/pdf`
17. `figure6_demographics.png/pdf`

---

## Session End: 2026-01-27
