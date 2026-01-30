# Run Log for DACA Replication Study (ID: 92)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA employment effects study.

## Date Started: 2026-01-26

---

## 1. Initial Setup and Data Review

### 1.1 Instructions Review
- Read replication_instructions.docx using Python docx library
- Research Question: Estimate causal impact of DACA eligibility on full-time employment (≥35 hours/week) among Hispanic-Mexican, Mexican-born individuals
- Treatment group: Ages 26-30 as of June 15, 2012
- Control group: Ages 31-35 as of June 15, 2012 (otherwise eligible)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (effects examined)
- 2012 is ambiguous (DACA implemented June 15, 2012) - will exclude

### 1.2 Data Files
- Main data: data/data.csv (33,851,424 rows)
- Data dictionary: data/acs_data_dict.txt
- Optional state-level data: data/state_demo_policy.csv (not used)

### 1.3 Key Variables Identified from Data Dictionary
- YEAR: Survey year (2006-2016)
- BIRTHYR: Birth year
- BIRTHQTR: Birth quarter (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- HISPAN: Hispanic origin (1=Mexican)
- BPL: Birthplace (200=Mexico)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week
- PERWT: Person weight for population estimates
- SEX: Sex (1=Male, 2=Female)
- MARST: Marital status
- EDUC: Educational attainment

---

## 2. Sample Selection Criteria

### 2.1 DACA Eligibility Requirements Applied
1. Hispanic-Mexican ethnicity (HISPAN=1)
2. Born in Mexico (BPL=200)
3. Not a citizen (CITIZEN=3, assuming non-citizens without papers are undocumented)
4. Arrived before age 16 (YRIMMIG - BIRTHYR < 16)
5. Immigrated by 2007 (YRIMMIG <= 2007 for continuous presence)
6. Ages 26-35 as of June 15, 2012 (treatment or control group)

### 2.2 Age Calculation Method
Age as of June 15, 2012 calculated accounting for birth quarter:
- If BIRTHQTR in {1, 2} (Jan-June): Age = 2012 - BIRTHYR
- If BIRTHQTR in {3, 4} (Jul-Dec): Age = 2012 - BIRTHYR - 1

### 2.3 Treatment/Control Assignment
- Treatment: Ages 26-30 as of June 15, 2012 (DACA-eligible)
- Control: Ages 31-35 as of June 15, 2012 (ineligible due to age)

### 2.4 Decision: Handling 2012 Data
- **Decision**: Exclude 2012 from analysis
- **Rationale**: DACA was implemented mid-year (June 15, 2012), and ACS does not record month of survey. Cannot distinguish pre/post treatment within 2012.

---

## 3. Analysis Commands Executed

### 3.1 Data Loading
```python
import pandas as pd
df = pd.read_csv('data/data.csv')
# Total observations: 33,851,424
```

### 3.2 Sample Selection Steps
```python
# Step 1: Hispanic-Mexican filter (HISPAN=1)
# Result: 2,945,521 observations

# Step 2: Born in Mexico (BPL=200)
# Result: 991,261 observations

# Step 3: Non-citizen (CITIZEN=3)
# Result: 701,347 observations

# Step 4: Exclude 2012
# Result: 636,722 observations

# Step 5: Ages 26-35 as of June 2012
# Result: 164,874 observations

# Step 6: Arrived before age 16 and by 2007
# Result: 43,238 observations (final sample)
```

### 3.3 Regression Models
```python
import statsmodels.formula.api as smf

# Model 1: Basic DiD (unweighted)
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_analysis)

# Model 2: Weighted DiD
model2 = smf.wls('fulltime ~ treated + post + treated_post', weights=PERWT)

# Model 3: With covariates
model3 = smf.wls('fulltime ~ treated + post + treated_post + male + married + C(EDUC)', weights=PERWT)

# Model 4: With year fixed effects (PREFERRED)
model4 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + male + married + C(EDUC)', weights=PERWT)

# Model 5: With state and year fixed effects
model5 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + male + married + C(EDUC)', weights=PERWT)
```

### 3.4 Event Study Specification
```python
# Reference year: 2011
# Created interaction terms for each year with treatment indicator
event_study = smf.wls('''fulltime ~ treated + C(YEAR) +
                         treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 +
                         treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 +
                         male + married + C(EDUC)''', weights=PERWT)
```

---

## 4. Key Analytical Decisions

### 4.1 Difference-in-Differences Design
- **Decision**: Standard two-group, two-period DiD design
- **Rationale**: Cleanly identifies effect using age-based eligibility cutoff as source of variation

### 4.2 Outcome Variable
- **Decision**: Full-time employment defined as UHRSWORK >= 35
- **Rationale**: Matches standard definition of full-time work (35+ hours/week) and aligns with research question

### 4.3 Sample Weights
- **Decision**: Use PERWT (person weights) in all primary specifications
- **Rationale**: Produces population-representative estimates; ACS is a stratified sample

### 4.4 Covariates
- **Decision**: Include sex (male indicator), marital status (married indicator), education (categorical)
- **Rationale**: Controls for observable differences that may affect employment; improves precision

### 4.5 Fixed Effects
- **Decision**: Include year fixed effects in preferred specification
- **Rationale**: Controls for time-varying shocks affecting all groups; more flexible than simple post indicator

### 4.6 Standard Errors
- **Decision**: Report both conventional and heteroskedasticity-robust (HC1) standard errors
- **Rationale**: Robust SEs account for potential heteroskedasticity in the error term

### 4.7 Preferred Specification
- **Decision**: Model 4 (weighted DiD with year FE and covariates) is the preferred specification
- **Rationale**: Balances flexibility, precision, and interpretability; year FE more conservative than simple post dummy

---

## 5. Results Summary

### 5.1 Sample Characteristics
| Group | N | Pre-DACA | Post-DACA |
|-------|---|----------|-----------|
| Treatment (26-30) | 25,470 | 16,694 | 8,776 |
| Control (31-35) | 17,768 | 11,683 | 6,085 |
| **Total** | **43,238** | **28,377** | **14,861** |

### 5.2 Simple DiD Calculation
|  | Pre-DACA | Post-DACA | Difference |
|--|----------|-----------|------------|
| Treatment | 0.631 | 0.660 | +0.029 |
| Control | 0.673 | 0.643 | -0.030 |
| **DiD** | | | **+0.059** |

### 5.3 Regression Results Summary
| Specification | DiD Estimate | SE | 95% CI |
|--------------|-------------|-----|--------|
| Basic (unweighted) | 0.052 | 0.010 | [0.032, 0.071] |
| Weighted | 0.059 | 0.010 | [0.040, 0.078] |
| + Covariates | 0.045 | 0.009 | [0.028, 0.063] |
| **+ Year FE (preferred)** | **0.044** | **0.009** | **[0.026, 0.061]** |
| + State FE | 0.043 | 0.009 | [0.025, 0.060] |
| Robust SE | 0.044 | 0.011 | [0.023, 0.065] |

### 5.4 Event Study Results
| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2006 | 0.007 | 0.023 | [-0.038, 0.051] |
| 2007 | -0.032 | 0.022 | [-0.076, 0.011] |
| 2008 | 0.009 | 0.023 | [-0.036, 0.053] |
| 2009 | -0.009 | 0.024 | [-0.055, 0.037] |
| 2010 | -0.015 | 0.023 | [-0.061, 0.030] |
| 2011 (ref) | 0.000 | -- | -- |
| 2013 | 0.033 | 0.024 | [-0.015, 0.080] |
| 2014 | 0.033 | 0.025 | [-0.015, 0.081] |
| 2015 | 0.020 | 0.025 | [-0.029, 0.069] |
| 2016 | 0.063* | 0.025 | [0.015, 0.112] |

### 5.5 Robustness Checks
| Check | DiD Estimate | SE | Notes |
|-------|-------------|-----|-------|
| 3-year bandwidth | 0.052 | 0.011 | N=27,777 |
| 4-year bandwidth | 0.049 | 0.010 | N=35,779 |
| 5-year bandwidth | 0.045 | 0.009 | N=43,238 |
| Males only | 0.034 | 0.011 | N=24,243 |
| Females only | 0.048 | 0.015 | N=18,995 |
| Placebo (2009) | -0.004 | 0.011 | Pre-period only |

---

## 6. Preferred Estimate (for Qualtrics Survey)

| Parameter | Value |
|-----------|-------|
| **Effect Size** | 0.044 (4.4 percentage points) |
| **Standard Error** | 0.009 |
| **Robust SE** | 0.011 |
| **95% CI** | [0.026, 0.061] |
| **Sample Size** | 43,238 |
| **Treatment N** | 25,470 |
| **Control N** | 17,768 |

**Interpretation**: DACA eligibility is associated with a 4.4 percentage point increase in full-time employment among eligible Hispanic-Mexican, Mexican-born non-citizens aged 26-30, compared to the ineligible control group aged 31-35.

---

## 7. Files Generated

| Filename | Description |
|----------|-------------|
| replication_report_92.tex | LaTeX source for replication report |
| replication_report_92.pdf | Compiled 23-page PDF report |
| run_log_92.md | This run log |
| analysis_script.py | Main Python analysis script |
| generate_figures.py | Script for generating figures |
| results_summary.csv | Key results in CSV format |
| figure1_parallel_trends.csv | Data for parallel trends figure |
| figure1_parallel_trends.png/pdf | Parallel trends visualization |
| figure2_event_study.csv | Data for event study figure |
| figure2_event_study.png/pdf | Event study visualization |
| figure3_did_bars.png/pdf | DiD bar chart visualization |
| figure4_robustness.png/pdf | Robustness checks visualization |

---

## 8. Session Commands Summary

```bash
# Extract instructions from docx
python -c "from docx import Document; doc = Document('replication_instructions.docx'); ..."

# Check data file structure
head -1 data/data.csv && wc -l data/data.csv

# Run main analysis
python analysis_script.py

# Generate figures
python generate_figures.py

# Compile LaTeX report (3 passes for references)
pdflatex -interaction=nonstopmode replication_report_92.tex
pdflatex -interaction=nonstopmode replication_report_92.tex
pdflatex -interaction=nonstopmode replication_report_92.tex
```

---

## 9. Notes and Caveats

1. **Undocumented status assumption**: Cannot directly identify undocumented immigrants in ACS. Used CITIZEN=3 (not a citizen) as proxy, per instructions.

2. **Immigration timing**: Used YRIMMIG <= 2007 to satisfy continuous presence requirement. Cannot verify exact date of immigration.

3. **Age calculation**: Used birth quarter to approximate age as of June 15, 2012. Some measurement error possible at quarterly boundaries.

4. **Pre-trends**: Event study shows some fluctuation in pre-treatment years but no systematic trend. 2007 coefficient is negative but not significant.

5. **Effect magnitude**: 4.4 pp effect represents ~7% increase relative to baseline (63.1% full-time employment rate).

---

*Log completed: 2026-01-26*
