# Run Log - DACA Replication Study 82

## Overview
This document logs all commands executed and key decisions made during the replication of the DACA employment effects study.

---

## 1. Data Exploration

### Initial Data Inspection
```bash
# List data folder contents
ls -la data/

# Preview data.csv
head -5 data/data.csv
```

**Findings:**
- Main data file: `data.csv` (6.2 GB, 33,851,424 observations)
- Data dictionary: `acs_data_dict.txt`
- Years covered: 2006-2016 (ACS one-year files)
- Optional supplementary file: `state_demo_policy.csv`

### Key Variables Identified
From the data dictionary:
- `YEAR`: Census year (2006-2016)
- `HISPAN`: Hispanic origin (1 = Mexican)
- `BPL`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Birth year
- `BIRTHQTR`: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- `UHRSWORK`: Usual hours worked per week
- `AGE`: Age at time of survey
- `SEX`: Sex (1=Male, 2=Female)
- `EDUC`: Education level
- `MARST`: Marital status
- `PERWT`: Person weight
- `STATEFIP`: State FIPS code

---

## 2. Sample Construction Decisions

### Decision 1: DACA Eligibility Criteria
Based on the research instructions, the following filters were applied:

| Criterion | Variable | Filter |
|-----------|----------|--------|
| Hispanic-Mexican | HISPAN | = 1 |
| Born in Mexico | BPL | = 200 |
| Not a citizen | CITIZEN | = 3 |
| Arrived before age 16 | YRIMMIG - BIRTHYR | < 16 |
| Continuous residence since 2007 | YRIMMIG | <= 2007 |

**Rationale:**
- HISPAN = 1 specifically identifies Mexican-origin Hispanics
- BPL = 200 is the IPUMS code for Mexico
- CITIZEN = 3 identifies non-citizens (we assume undocumented as instructed)
- Age at immigration calculated from immigration year minus birth year
- YRIMMIG <= 2007 approximates the continuous residence requirement

### Decision 2: Treatment and Control Groups
- **Treatment**: Ages 26-30 as of June 15, 2012
- **Control**: Ages 31-35 as of June 15, 2012

**Age Calculation:**
```python
# Calculate age as of June 15, 2012
age_june_2012 = 2012 - BIRTHYR
# Adjust for those born Jul-Dec (hadn't turned that age by June 15)
if BIRTHQTR >= 3:
    age_june_2012 -= 1
```

**Rationale:**
- The age 31 cutoff is a key eligibility threshold for DACA
- Ages 26-30 selected to have a symmetric 5-year window on each side
- Using BIRTHQTR provides more precise age calculation

### Decision 3: Time Periods
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016
- **Excluded**: 2012

**Rationale:**
- 2012 excluded because ACS doesn't record interview month, so we cannot distinguish pre- vs post-DACA observations within that year
- DACA announced June 15, 2012; applications began August 15, 2012

### Decision 4: Outcome Variable
```python
fulltime = (UHRSWORK >= 35)
```

**Rationale:**
- Standard definition of full-time employment is 35+ hours per week
- This captures the intensive margin of employment most relevant to DACA's work authorization benefit

---

## 3. Analysis Commands

### Main Analysis Script: `analysis.py`

```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_82"
python analysis.py
```

### Sample Sizes After Each Filter:
| Stage | N |
|-------|---|
| Total ACS observations (2006-2016) | 33,851,424 |
| Hispanic-Mexican, Mexico-born, non-citizen | 701,347 |
| Arrived before age 16 | 205,327 |
| Immigrated by 2007 | 195,023 |
| Ages 26-35 as of June 2012 | 47,418 |
| Excluding 2012 | 43,238 |

### Final Sample Breakdown:
- Treatment group: 25,470
- Control group: 17,768
- Pre-period: 28,377
- Post-period: 14,861

---

## 4. Key Results

### Simple Difference-in-Differences
| Group | Pre | Post | Change |
|-------|-----|------|--------|
| Treatment (26-30) | 0.6147 | 0.6339 | +0.0192 |
| Control (31-35) | 0.6461 | 0.6136 | -0.0324 |
| **DiD** | | | **+0.0516** |

### Regression Models

| Model | DiD Coefficient | Std. Error | 95% CI |
|-------|-----------------|------------|--------|
| Basic (no controls) | 0.0516 | 0.0100 | [0.032, 0.071] |
| + Demographics | 0.0451 | 0.0092 | [0.027, 0.063] |
| + Year FE | 0.0515 | 0.0099 | [0.032, 0.071] |
| + Year FE + Demographics | 0.0450 | 0.0092 | [0.027, 0.063] |
| + State FE | 0.0442 | 0.0092 | [0.026, 0.062] |
| Weighted (PERWT) | 0.0459 | 0.0107 | [0.025, 0.067] |

### Preferred Estimate (Model 4)
- **DiD Coefficient**: 0.0450
- **Standard Error**: 0.0092
- **95% CI**: [0.027, 0.063]
- **p-value**: < 0.0001
- **Sample Size**: 43,238

### Event Study Results
| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | -0.016 | 0.019 | 0.382 |
| 2007 | -0.031 | 0.019 | 0.094 |
| 2008 | 0.002 | 0.019 | 0.907 |
| 2009 | -0.017 | 0.020 | 0.382 |
| 2010 | -0.021 | 0.020 | 0.284 |
| 2011 | 0 (ref) | --- | --- |
| 2013 | 0.030 | 0.020 | 0.142 |
| 2014 | 0.025 | 0.020 | 0.224 |
| 2015 | 0.030 | 0.021 | 0.147 |
| 2016 | 0.040 | 0.021 | 0.052 |

### Robustness Checks
| Check | DiD | SE | p-value |
|-------|-----|-----|---------|
| Males only | 0.033 | 0.011 | < 0.01 |
| Females only | 0.048 | 0.015 | < 0.01 |
| Narrow bandwidth (27-29 vs 32-34) | 0.041 | 0.012 | < 0.01 |
| Wide bandwidth (25-30 vs 31-36) | 0.052 | 0.008 | < 0.001 |
| Placebo (pre-period only) | 0.003 | 0.011 | 0.766 |

---

## 5. Visualization Commands

```bash
python create_figures.py
```

### Figures Generated:
1. `figure1_parallel_trends.png/pdf` - Full-time employment trends by treatment status
2. `figure2_event_study.png/pdf` - Event study coefficients with 95% CIs
3. `figure3_did_visualization.png/pdf` - DiD calculation visualization
4. `figure4_model_comparison.png/pdf` - Coefficient estimates across models
5. `figure5_robustness.png/pdf` - Robustness check results

---

## 6. Report Compilation

```bash
# Compile LaTeX (3 passes for TOC and references)
pdflatex -interaction=nonstopmode replication_report_82.tex
pdflatex -interaction=nonstopmode replication_report_82.tex
pdflatex -interaction=nonstopmode replication_report_82.tex
```

**Output:** `replication_report_82.pdf` (21 pages)

---

## 7. Output Files

### Required Deliverables:
- `replication_report_82.tex` - LaTeX source
- `replication_report_82.pdf` - Final report
- `run_log_82.md` - This log file

### Additional Files:
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script
- `results_summary.csv` - Summary statistics
- `event_study_results.csv` - Event study coefficients
- `sample_statistics.csv` - Detailed sample statistics
- `yearly_fulltime_rates.csv` - Year-by-year employment rates
- `figure1_parallel_trends.png/pdf`
- `figure2_event_study.png/pdf`
- `figure3_did_visualization.png/pdf`
- `figure4_model_comparison.png/pdf`
- `figure5_robustness.png/pdf`

---

## 8. Key Methodological Decisions Summary

1. **Sample Definition**: Used HISPAN=1, BPL=200, CITIZEN=3 to identify potential DACA-eligible population
2. **Age Calculation**: Used BIRTHYR and BIRTHQTR to calculate precise age as of June 15, 2012
3. **Treatment/Control**: 5-year symmetric windows around age 31 cutoff
4. **Excluded 2012**: Cannot distinguish pre/post DACA within that year
5. **Outcome**: Full-time = UHRSWORK >= 35
6. **Estimation**: DiD with robust standard errors (HC1)
7. **Preferred Model**: Year FE + demographic controls (Model 4)
8. **Controls Used**: Male indicator, age, education (EDUC), married indicator

---

## 9. Interpretation

**Main Finding**: DACA eligibility is associated with a statistically significant 4.5 percentage point increase in full-time employment (95% CI: [2.7, 6.3], p < 0.001).

**Key Supporting Evidence**:
- Pre-trends are parallel (event study shows no significant pre-2012 effects)
- Placebo test (pre-period only) shows no differential trend (DiD = 0.003, p = 0.77)
- Results robust to different age bandwidths, demographic controls, and weighting

**Caveats**:
- Cannot distinguish documented vs undocumented non-citizens
- Estimate is intent-to-treat (not all eligible individuals applied)
- Sample represents those meeting observable criteria; true eligibility includes additional requirements (education, no serious criminal record) that cannot be verified in ACS

---

*Log completed: January 2026*
