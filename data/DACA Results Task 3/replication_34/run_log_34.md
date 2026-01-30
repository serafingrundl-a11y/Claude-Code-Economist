# Run Log - DACA Replication Study 34

## Overview
Independent replication of DACA (Deferred Action for Childhood Arrivals) employment effects analysis.

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (35+ hours/week)?

**Design:** Difference-in-Differences
- Treatment group: ELIGIBLE=1 (ages 26-30 in June 2012)
- Control group: ELIGIBLE=0 (ages 31-35 in June 2012)
- Pre-period: 2008-2011
- Post-period: 2013-2016

---

## Session: 2026-01-27

### Step 1: Read replication instructions
- Extracted text from `replication_instructions.docx` using Python docx library
- Key specifications identified:
  - Outcome: FT (full-time employment, 35+ hours/week)
  - Treatment indicator: ELIGIBLE (provided in data)
  - Time indicator: AFTER (0=2008-2011, 1=2013-2016)
  - Data excludes 2012 (implementation year)
  - Sample already restricted to relevant population (Hispanic-Mexican, Mexican-born)

### Step 2: Examine data structure
- Data file: `prepared_data_numeric_version.csv`
- Total observations: 17,382 (17,379 after cleaning missing values)
- Key variables verified:
  - FT: Full-time employment outcome (0/1) - 11,283 employed FT, 6,099 not
  - AFTER: Post-treatment indicator (0/1) - 9,527 pre, 7,855 post
  - ELIGIBLE: Treatment group indicator (0/1) - 11,382 treatment, 6,000 control
  - PERWT: Person weight for survey weighting

### Step 3: Data Exploration
- Examined distributions of all key variables
- Verified age ranges:
  - Treatment (ELIGIBLE=1): Ages 26-30.75 in June 2012 (mean: 28.1)
  - Control (ELIGIBLE=0): Ages 31-35 in June 2012 (mean: 32.9)
- Education distribution similar across groups (~71-74% high school degree)
- Geographic concentration: California (45%), Texas (21%), Illinois (6%)

### Step 4: Covariate Balance Check
Pre-treatment period comparisons:
- Female: Treatment 46.6%, Control 43.4%
- Age: Treatment 25.8, Control 30.5 (by design)
- Married: Treatment 34.5%, Control 46.3%
- Children: Treatment 0.90, Control 1.47
- Family size: Treatment 4.39, Control 4.45

### Step 5: Statistical Analysis

#### Raw DiD Calculation
| Group | Pre | Post | Difference |
|-------|-----|------|------------|
| Control (31-35) | 68.86% | 66.29% | -2.57 pp |
| Treatment (26-30) | 63.68% | 68.60% | +4.93 pp |
| **DiD** | | | **+7.49 pp** |

#### Regression Models Estimated

**Model 1: Basic OLS (unweighted)**
- DiD = 0.0644 (SE = 0.0153), p < 0.001

**Model 2: WLS with survey weights**
- DiD = 0.0749 (SE = 0.0181), p < 0.001

**Model 3: WLS + Demographics (Female, Married, Education)**
- DiD = 0.0623 (SE = 0.0167), p < 0.001

**Model 4: WLS + Demographics + Year FE**
- DiD = 0.0596 (SE = 0.0167), p < 0.001

**Model 5: WLS + Demographics + Year FE + State FE (PREFERRED)**
- DiD = 0.0589 (SE = 0.0166), p = 0.0004
- 95% CI: [0.0263, 0.0915]
- R-squared: 0.138
- N = 17,379

**Model 6: Clustered SE by State**
- DiD = 0.0589 (SE = 0.0212), p = 0.0055

### Step 6: Pre-Trends Analysis (Event Study)
Year-specific treatment effects relative to 2008:
- 2009: 0.0203 (p = 0.50) - No significant pre-trend
- 2010: -0.0089 (p = 0.77) - No significant pre-trend
- 2011: 0.0675 (p = 0.035) - Significant, possible concern
- 2013: 0.0843 (p = 0.007) - Post-DACA, significant
- 2014: 0.0494 (p = 0.13)
- 2015: 0.0566 (p = 0.08)
- 2016: 0.1258 (p < 0.001) - Post-DACA, highly significant

### Step 7: Heterogeneity Analysis
By sex:
- Males: DiD = 0.0611 (SE = 0.0196), p = 0.002
- Females: DiD = 0.0401 (SE = 0.0272), p = 0.14
- Interaction (Female x DiD): -0.0186 (not significant)

### Step 8: Robustness Checks
| Specification | Coefficient | SE | 95% CI |
|--------------|-------------|-----|--------|
| Basic DiD (WLS) | 0.0749 | 0.0181 | [0.039, 0.110] |
| + Demographics | 0.0623 | 0.0167 | [0.030, 0.095] |
| + Year FE | 0.0596 | 0.0167 | [0.027, 0.092] |
| + State FE (preferred) | 0.0589 | 0.0166 | [0.026, 0.092] |
| Clustered SE | 0.0589 | 0.0212 | [0.017, 0.100] |
| + State policies | 0.0581 | 0.0167 | [0.026, 0.091] |

---

## Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Software | Python 3.14 with statsmodels | Reproducible, widely available, full functionality |
| Weighting | Use PERWT | ACS survey weights for population-representative estimates |
| Standard Errors | Robust (HC1) preferred, clustered as sensitivity | Account for heteroskedasticity; clustering conservative |
| Covariates | Female, Married, Education | Improve precision, control for compositional differences |
| Fixed Effects | Year + State | Control for time trends and state-level confounders |
| Preferred Model | Model 5 (Full specification with robust SE) | Balances comprehensiveness with interpretability |

---

## Files Generated

### Analysis Code
- `analysis.py` - Main Python script for all analyses

### Output Files
- `regression_results.csv` - Summary of all regression models
- `event_study_results.csv` - Year-specific treatment effects
- `figure1_trends.png` - Employment trends by treatment status
- `figure2_event_study.png` - Event study coefficients with CIs
- `figure3_did_bars.png` - DiD bar chart visualization

### Report
- `replication_report_34.tex` - Full LaTeX report (~24 pages)
- `replication_report_34.pdf` - Compiled PDF report

---

## Key Results Summary

**Preferred Estimate:**
- DACA eligibility increased full-time employment by **5.89 percentage points**
- Standard Error: 0.0166
- 95% Confidence Interval: [2.63, 9.15] percentage points
- p-value: 0.0004 (statistically significant at 1% level)
- Sample Size: 17,379

**Interpretation:**
DACA eligibility is associated with a 5.89 percentage point increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born individuals aged 26-30 in June 2012, compared to those aged 31-35. This represents approximately a 9% increase relative to the baseline employment rate.

---

## Commands Executed

```bash
# Data exploration
head -1 prepared_data_numeric_version.csv
wc -l prepared_data_numeric_version.csv

# Run analysis
python analysis.py

# Compile report
pdflatex -interaction=nonstopmode replication_report_34.tex
pdflatex -interaction=nonstopmode replication_report_34.tex  # Second pass for TOC
pdflatex -interaction=nonstopmode replication_report_34.tex  # Third pass for refs
```

---

## Notes and Caveats

1. **Pre-trends concern:** The 2011 coefficient is significant, suggesting some pre-DACA divergence. This should be interpreted with caution.

2. **Intent-to-treat:** Estimates reflect eligibility effects, not actual DACA receipt. True effect on participants may be larger.

3. **Sample composition:** Sample heavily concentrated in California (45%) and Texas (21%), which may limit generalizability.

4. **Repeated cross-section:** ACS is not panel data; we cannot track same individuals over time.

---

*Log completed: 2026-01-27*
