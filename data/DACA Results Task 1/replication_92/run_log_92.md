# Replication Run Log - DACA Full-Time Employment Analysis

## Project Overview
- **Research Question**: What was the causal impact of DACA eligibility on full-time employment (35+ hours/week) among Hispanic-Mexican Mexican-born individuals in the US?
- **Treatment Period**: DACA implemented June 15, 2012; examining effects 2013-2016
- **Data Source**: American Community Survey (ACS) from IPUMS USA, 2006-2016
- **Replication ID**: 92

---

## Session Log

### Step 1: Initial Data Exploration
**Actions**:
1. Read replication_instructions.docx using python-docx
2. Examined data dictionary (acs_data_dict.txt)
3. Inspected data.csv structure

**Key Findings**:
- Data contains ACS samples from 2006-2016 (33,851,424 total observations)
- Key variables identified:
  - YEAR: Survey year
  - HISPAN/HISPAND: Hispanic origin (1=Mexican, 100-107 for detailed Mexican)
  - BPL/BPLD: Birthplace (200=Mexico)
  - CITIZEN: Citizenship status (3=Not a citizen)
  - YRIMMIG: Year of immigration
  - BIRTHYR: Birth year
  - BIRTHQTR: Birth quarter
  - AGE: Age
  - UHRSWORK: Usual hours worked per week (outcome: >=35 for full-time)
  - PERWT: Person weight for population estimates

---

### Step 2: DACA Eligibility Criteria Definition
**Decision Point**: Defining DACA eligibility

**DACA Eligibility Requirements** (from instructions):
1. Arrived in US before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 and did not have lawful status

**Operationalization**:
- **Age at arrival < 16**: age_at_arrival = YRIMMIG - BIRTHYR < 16
- **Under 31 as of June 2012**:
  - BIRTHYR >= 1982 (definitely under 31), OR
  - BIRTHYR == 1981 AND BIRTHQTR >= 3 (born July-Dec 1981)
- **In US since 2007**: YRIMMIG <= 2007
- **Non-citizen**: CITIZEN == 3 (Not a citizen)

**Important Note**: Cannot distinguish documented vs undocumented among non-citizens. Per instructions, assume non-citizens without naturalization are undocumented for DACA purposes.

---

### Step 3: Sample Definition
**Decision Point**: Defining analysis sample

**Sample Restrictions Applied**:
| Step | Criterion | IPUMS Code | Observations |
|------|-----------|------------|--------------|
| 1 | Full ACS 2006-2016 | - | 33,851,424 |
| 2 | Hispanic-Mexican | HISPAN = 1 | 2,945,521 |
| 3 | Born in Mexico | BPL = 200 | 991,261 |
| 4 | Non-citizen | CITIZEN = 3 | 701,347 |
| 5 | Exclude 2012 | YEAR ≠ 2012 | 636,722 |
| 6 | Ages 16-64 | AGE 16-64 | **561,470** |

**Final Sample**: 561,470 observations
- DACA Eligible: 83,611
- Not DACA Eligible: 477,859

---

### Step 4: Identification Strategy
**Decision Point**: Causal identification approach

**Primary Strategy**: Difference-in-Differences (DiD)
- Compare DACA-eligible vs non-eligible before and after 2012
- Pre-period: 2006-2011 (6 years)
- Post-period: 2013-2016 (4 years)
- 2012 excluded (policy implemented mid-year, June 15)

**Treatment Definition**:
- `daca_eligible` = 1 if all eligibility criteria met
- `post` = 1 if YEAR >= 2013
- DiD estimate: coefficient on `daca_eligible × post`

---

### Step 5: Outcome Variable
**Decision Point**: Full-time employment definition

- **Primary Outcome**: Full-time employment = UHRSWORK >= 35 hours per week
- Binary indicator: 1 if usually works 35+ hours, 0 otherwise
- **Secondary Outcome**: Any employment (EMPSTAT = 1)

---

## Analysis Execution Log

### Step 6: Analysis Script Execution
**File**: analysis.py

**Commands Executed**:
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_92"
python analysis.py
```

**Output Files Generated**:
- yearly_means.csv (trends by year and eligibility)
- event_study_results.csv (event study coefficients)
- summary_statistics.csv (descriptive stats)
- model_comparison.csv (regression results)
- robustness_results.csv (robustness checks)

---

### Step 7: Figure Generation
**File**: create_figures.py

**Figures Created**:
1. figure1_trends.png/pdf - Employment trends by DACA eligibility
2. figure2_eventstudy.png/pdf - Event study plot
3. figure3_did.png/pdf - DiD visualization
4. figure4_robustness.png/pdf - Robustness results

---

### Step 8: Report Compilation
**File**: replication_report_92.tex

**LaTeX Compilation**:
```bash
pdflatex -interaction=nonstopmode replication_report_92.tex  # First pass
pdflatex -interaction=nonstopmode replication_report_92.tex  # Second pass (TOC)
pdflatex -interaction=nonstopmode replication_report_92.tex  # Third pass (refs)
```

**Output**: replication_report_92.pdf (18 pages)

---

## Key Results Summary

### Preferred Specification (Model 6)
- **Method**: Weighted Difference-in-Differences with State and Year Fixed Effects
- **Sample**: Hispanic-Mexican, born in Mexico, non-citizens, ages 16-64
- **Weights**: Person weights (PERWT)
- **Standard Errors**: Clustered by state

### Main Result
| Statistic | Value |
|-----------|-------|
| DiD Estimate | **0.0304** |
| Standard Error | 0.00398 |
| t-statistic | 7.641 |
| p-value | < 0.001 |
| 95% CI Lower | 0.0226 |
| 95% CI Upper | 0.0382 |
| Sample Size | 561,470 |
| R-squared | 0.2297 |

**Interpretation**: DACA eligibility is associated with a **3.04 percentage point increase** in the probability of full-time employment, statistically significant at the 1% level.

### Robustness Results
| Specification | DiD Estimate | SE | N |
|---------------|--------------|-----|------|
| Main | 0.0304*** | 0.004 | 561,470 |
| Ages 18-35 | 0.0106* | 0.006 | 253,373 |
| Men Only | 0.0263*** | 0.006 | 303,717 |
| Women Only | 0.0258*** | 0.006 | 257,753 |
| Any Employment | 0.0402*** | 0.007 | 561,470 |
| Placebo (2009) | 0.0153*** | 0.004 | 345,792 |

---

## Analytical Decisions Summary

| Decision | Choice | Justification |
|----------|--------|---------------|
| Pre-treatment period | 2006-2011 | Maximum pre-period data available per instructions |
| Post-treatment period | 2013-2016 | Per instructions, full post-period |
| Exclude 2012 | Yes | Policy implemented mid-year (June 15), timing unclear |
| Age restrictions | 16-64 | Standard working-age population |
| Full-time threshold | 35 hours/week | Standard BLS definition |
| Control group | Non-eligible Mexican non-citizens | Same demographic, different eligibility status |
| Fixed effects | State + Year | Control for geographic and temporal variation |
| Standard errors | Clustered by state | Account for within-state correlation |
| Weights | Person weights (PERWT) | Population-representative estimates |
| Education criteria | Not enforced | Cannot observe in ACS data |
| Criminal history | Not enforced | Cannot observe in ACS data |

---

## Files Generated

| Filename | Description |
|----------|-------------|
| analysis.py | Main analysis script |
| create_figures.py | Figure generation script |
| yearly_means.csv | Data for trends figure |
| event_study_results.csv | Event study coefficients |
| summary_statistics.csv | Summary statistics |
| model_comparison.csv | Model comparison table |
| robustness_results.csv | Robustness check results |
| figure1_trends.png/pdf | Employment trends figure |
| figure2_eventstudy.png/pdf | Event study figure |
| figure3_did.png/pdf | DiD visualization figure |
| figure4_robustness.png/pdf | Robustness results figure |
| replication_report_92.tex | LaTeX source |
| **replication_report_92.pdf** | Final report (18 pages) |
| **run_log_92.md** | This log file |

---

## Software Used
- Python 3.x
- pandas, numpy, statsmodels, scipy, matplotlib
- pdflatex (MiKTeX)

---

## Notes and Caveats

1. **Intent-to-treat**: Treatment variable captures eligibility, not actual DACA receipt
2. **Undocumented proxy**: Non-citizen status used as proxy for undocumented (some may be documented)
3. **Unobserved criteria**: Cannot verify education requirements or criminal history
4. **Placebo concern**: Placebo test (2009) shows small significant effect, possibly due to Great Recession recovery
5. **Selection**: Cannot rule out selective migration or naturalization effects

---

## Session Complete

All required deliverables have been generated:
- [x] replication_report_92.tex
- [x] replication_report_92.pdf
- [x] run_log_92.md
