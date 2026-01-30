# DACA Replication Study - Run Log #41

## Overview
This document logs all commands executed and key decisions made during the replication analysis of the effect of DACA eligibility on full-time employment among Mexican-born Hispanic immigrants.

---

## Session Information
- **Date**: January 26, 2026
- **Analysis Software**: Python 3.x with pandas, numpy, statsmodels, matplotlib
- **Report Generation**: LaTeX with pdflatex

---

## Step 1: Read Replication Instructions

**Command:**
```python
from docx import Document
doc = Document('replication_instructions.docx')
[print(p.text) for p in doc.paragraphs]
```

**Key Requirements Extracted:**
- Research Question: Effect of DACA eligibility on full-time employment
- Population: Hispanic-Mexican, Mexican-born individuals
- Treatment Group: Ages 26-30 at June 15, 2012 (DACA eligible)
- Control Group: Ages 31-35 at June 15, 2012 (too old for DACA)
- Outcome: Full-time employment (35+ hours per week)
- Data: ACS 2006-2016
- Method: Difference-in-differences

---

## Step 2: Explore Data Structure

**Commands:**
```bash
ls -la data/
head -n 5 data/data.csv
```

**Files Found:**
- `data.csv` - Main ACS data file (6.26 GB, 33,851,424 observations)
- `acs_data_dict.txt` - Variable definitions
- `state_demo_policy.csv` - Optional state-level data (not used)

**Key Variables Identified:**
- YEAR: Survey year (2006-2016)
- HISPAN: Hispanic origin (1 = Mexican)
- BPL: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Year of birth
- UHRSWORK: Usual hours worked per week
- PERWT: Person-level survey weight

---

## Step 3: Data Cleaning and Sample Construction

### 3.1 Filter to Hispanic-Mexican Born in Mexico

**Decision:** Require HISPAN == 1 (Mexican) AND BPL == 200 (Mexico)

**Rationale:** The instructions specify "Hispanic-Mexican Mexican-born people." This combination ensures we capture ethnic Hispanic-Mexicans who were born in Mexico.

**Result:** 991,261 observations

### 3.2 Filter to Non-Citizens

**Decision:** Require CITIZEN == 3 (Not a citizen)

**Rationale:** Per instructions, "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes." CITIZEN == 3 represents non-citizens.

**Result:** 701,347 observations

### 3.3 Apply DACA Eligibility Criteria

**Decisions:**
1. Calculate arrival_age = YRIMMIG - BIRTHYR
2. Require arrival_age < 16 (arrived before 16th birthday)
3. Require YRIMMIG <= 2007 (continuous residence since June 2007)
4. Require YRIMMIG > 0 (valid immigration year)

**Rationale:** DACA required arrival before age 16 and continuous residence since June 15, 2007. Using YRIMMIG <= 2007 ensures they were in the US by 2007.

**Result:** 191,374 observations

### 3.4 Calculate Age at Policy Implementation

**Decision:** age_2012 = 2012 - BIRTHYR

**Rationale:** Simple calculation using birth year. The BIRTHQTR variable could provide more precision, but the simpler approach is more common and reduces complexity.

### 3.5 Define Treatment and Control Groups

**Decision:**
- Treatment: age_2012 in [26, 30]
- Control: age_2012 in [31, 35]

**Rationale:**
- Ages 26-30 were eligible for DACA (under 31 as of June 15, 2012)
- Ages 31-35 would have been eligible but for exceeding the age cutoff
- Using ages close to the cutoff maximizes comparability

**Result:**
- Treatment group: 28,770 observations (across all years)
- Control group: 19,636 observations

### 3.6 Define Pre and Post Periods

**Decision:**
- Pre-DACA: 2006-2011
- Post-DACA: 2013-2016
- Exclude: 2012

**Rationale:** DACA was implemented June 15, 2012. The ACS does not record month of interview, so 2012 observations cannot be cleanly assigned to pre or post periods.

**Final Sample:** 44,161 observations (28,968 pre, 15,193 post)

---

## Step 4: Create Outcome Variable

**Decision:** fulltime = 1 if UHRSWORK >= 35, else 0

**Rationale:** The instructions define full-time employment as "usually working 35 hours per week or more." UHRSWORK captures usual weekly hours.

**Baseline Statistics:**
- Treatment pre-DACA: 61.1% full-time
- Treatment post-DACA: 63.4% full-time
- Control pre-DACA: 64.3% full-time
- Control post-DACA: 61.2% full-time

---

## Step 5: Difference-in-Differences Analysis

### 5.1 Simple DiD Calculation

```
Treatment change: 0.634 - 0.611 = +0.023
Control change:   0.612 - 0.643 = -0.031
DiD estimate:     0.023 - (-0.031) = 0.054
```

### 5.2 Regression Specifications

**Model 1: Basic DiD**
```
fulltime ~ treated + post + treated_post
```

**Model 2: + Demographics**
```
fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecoll + educ_coll
```

**Model 3: + Year Fixed Effects**
```
fulltime ~ treated + C(YEAR) + treated_post + female + married + educ_hs + educ_somecoll + educ_coll
```

**Model 4: + State Fixed Effects (Preferred)**
```
fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + educ_hs + educ_somecoll + educ_coll
```

### 5.3 Estimation Details

- **Estimation Method:** Weighted Least Squares using PERWT
- **Standard Errors:** Heteroskedasticity-robust (HC1)

---

## Step 6: Results Summary

### Main Results (Model 4 - Preferred Specification)

| Statistic | Value |
|-----------|-------|
| DiD Coefficient | 0.0471 |
| Standard Error | 0.0106 |
| t-statistic | 4.44 |
| p-value | < 0.001 |
| 95% CI | [0.0263, 0.0679] |
| Sample Size | 44,161 |

### Coefficient Stability Across Specifications

| Model | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| (1) Basic DiD | 0.0624 | 0.0117 | < 0.001 |
| (2) + Demographics | 0.0490 | 0.0107 | < 0.001 |
| (3) + Year FE | 0.0478 | 0.0106 | < 0.001 |
| (4) + State FE | 0.0471 | 0.0106 | < 0.001 |

### Heterogeneity by Gender

| Gender | Coefficient | SE | p-value |
|--------|-------------|-----|---------|
| Male | 0.0608 | 0.0125 | < 0.001 |
| Female | 0.0345 | 0.0183 | 0.060 |

### Pre-trends Test

- Differential pre-trend coefficient: 0.0033
- Standard error: 0.0040
- p-value: 0.419
- **Conclusion:** No significant differential pre-trend

---

## Step 7: Generate Figures

**Commands:**
```bash
python create_figures.py
```

**Figures Created:**
1. `figure1_employment_trends.png` - Employment rates by group over time
2. `figure2_event_study.png` - Event study coefficients
3. `figure3_did_visual.png` - DiD visualization
4. `figure4_coefficients.png` - Coefficient comparison across models

---

## Step 8: Generate LaTeX Report

**Commands:**
```bash
pdflatex -interaction=nonstopmode replication_report_41.tex
pdflatex -interaction=nonstopmode replication_report_41.tex
pdflatex -interaction=nonstopmode replication_report_41.tex
```

**Note:** Multiple passes required for table of contents and cross-references.

**Output:** `replication_report_41.pdf` (28 pages)

---

## Key Analytical Decisions and Justifications

### 1. Sample Restriction to Non-Citizens Only

**Decision:** Use CITIZEN == 3 only, excluding naturalized citizens and those born abroad to American parents.

**Justification:** Per instructions, we assume non-citizens who haven't received papers are undocumented. Naturalized citizens and those born to American parents would have legal status and thus not be affected by DACA.

### 2. Age Calculation Method

**Decision:** Use simple calculation age_2012 = 2012 - BIRTHYR rather than incorporating BIRTHQTR.

**Justification:**
- Standard practice in the literature
- Reduces complexity
- BIRTHQTR provides only quarterly (not monthly) precision anyway
- Conservative approach that may slightly attenuate effects

### 3. Age Range Selection (26-30 vs 31-35)

**Decision:** Use ages 26-30 for treatment and 31-35 for control.

**Justification:**
- Both groups are in similar life stages with comparable labor force attachment
- Close to the age cutoff (31) maximizes comparability
- Wide enough ranges to ensure adequate sample sizes

### 4. Exclusion of 2012

**Decision:** Drop all 2012 observations from analysis.

**Justification:** DACA was implemented June 15, 2012. The ACS does not record interview month, so we cannot determine whether a 2012 observation occurred before or after implementation.

### 5. Using Person Weights

**Decision:** Weight all analyses by PERWT.

**Justification:** PERWT adjusts for complex survey design and ensures estimates are representative of the target population.

### 6. Robust Standard Errors

**Decision:** Use HC1 heteroskedasticity-robust standard errors.

**Justification:** Standard practice for DiD analyses. Accounts for heteroskedasticity without making distributional assumptions.

### 7. Preferred Specification (Model 4)

**Decision:** Report Model 4 (with state and year FE) as preferred specification.

**Justification:**
- Controls for state-specific characteristics that may affect employment
- Controls for year-specific shocks (e.g., Great Recession recovery)
- Most conservative specification
- Results remain significant and economically meaningful

---

## Output Files

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `analysis_results.txt` | Console output from analysis |
| `create_figures.py` | Figure generation script |
| `regression_results.csv` | DiD coefficients across specifications |
| `event_study_results.csv` | Year-specific treatment effects |
| `summary_statistics.csv` | Summary stats by group/period |
| `figure1_employment_trends.png` | Employment trends figure |
| `figure2_event_study.png` | Event study figure |
| `figure3_did_visual.png` | DiD visualization |
| `figure4_coefficients.png` | Coefficient comparison |
| `replication_report_41.tex` | LaTeX source |
| `replication_report_41.pdf` | Final report (28 pages) |
| `run_log_41.md` | This log file |

---

## Conclusion

The analysis finds that DACA eligibility increased full-time employment by approximately 4.7 percentage points (SE = 0.011, p < 0.001), representing a 7.7% increase relative to the treatment group's pre-treatment mean. The effect is robust across specifications and supported by parallel pre-trends.
