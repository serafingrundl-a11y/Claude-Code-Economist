# Run Log - DACA Replication Study (Replication 10)

## Overview

This document logs all commands executed and key decisions made during the independent replication of the DACA employment effect study.

---

## Session Information

- **Date:** January 25, 2026
- **Working Directory:** `C:\Users\seraf\DACA Results Task 1\replication_10`
- **Data Source:** American Community Survey (ACS) via IPUMS USA
- **Analysis Software:** Python 3.14 with pandas, numpy, statsmodels, matplotlib

---

## Step 1: Read and Understand Instructions

### Command:
```python
from docx import Document
doc = Document('replication_instructions.docx')
print('\n'.join([p.text for p in doc.paragraphs]))
```

### Key Findings:
- Research question: Effect of DACA eligibility on full-time employment
- Population: Hispanic-Mexican, Mexican-born individuals in the US
- Outcome: Full-time employment (35+ hours/week)
- Data: ACS 2006-2016 (excluding 2012 due to mid-year implementation)
- DACA implemented: June 15, 2012

---

## Step 2: Examine Data Structure

### Commands:
```bash
# Check data files available
ls data/

# Read data dictionary
cat data/acs_data_dict.txt

# Check data.csv structure
python -c "import pandas as pd; df = pd.read_csv('data/data.csv', nrows=5); print(df.columns.tolist())"

# Count total observations
python -c "with open('data/data.csv', 'r') as f: print('Lines:', sum(1 for _ in f))"
# Result: 33,851,425 lines (including header)

# Count observations by year
python -c "
import pandas as pd
year_counts = {}
for chunk in pd.read_csv('data/data.csv', usecols=['YEAR'], chunksize=1000000, dtype={'YEAR': 'int32'}):
    counts = chunk['YEAR'].value_counts()
    for year, count in counts.items():
        year_counts[year] = year_counts.get(year, 0) + count
for year in sorted(year_counts.keys()):
    print(f'{year}: {year_counts[year]:,}')
"
```

### Data Summary:
| Year | Observations |
|------|-------------|
| 2006 | 2,969,741 |
| 2007 | 2,994,662 |
| 2008 | 3,000,657 |
| 2009 | 3,030,728 |
| 2010 | 3,061,692 |
| 2011 | 3,112,017 |
| 2012 | 3,113,030 |
| 2013 | 3,132,795 |
| 2014 | 3,132,610 |
| 2015 | 3,147,005 |
| 2016 | 3,156,487 |

---

## Step 3: Key Analysis Decisions

### Decision 1: Sample Restriction
**Choice:** Restrict to Hispanic-Mexican (HISPAN=1), Mexican-born (BPL=200), non-citizen (CITIZEN=3) individuals aged 16-64 with valid immigration year.

**Rationale:**
- Hispanic-Mexican and Mexican-born criteria match the stated population of interest
- Non-citizen restriction focuses on the population most likely to be undocumented (cannot distinguish documented vs undocumented non-citizens in data)
- Age 16-64 is standard working-age population
- Valid immigration year required for DACA eligibility calculation

### Decision 2: DACA Eligibility Criteria
**Choice:** Define eligibility based on observable ACS variables:
1. Arrived before 16th birthday: YRIMMIG - BIRTHYR < 16
2. Under 31 as of June 2012: BIRTHYR >= 1982 OR (BIRTHYR = 1981 AND BIRTHQTR >= 3)
3. Continuous presence since 2007: YRIMMIG <= 2007
4. Non-citizen: CITIZEN = 3

**Rationale:**
- These criteria map to the official DACA requirements
- Cannot observe education/military or criminal history criteria
- Cannot distinguish documented vs undocumented among non-citizens
- Used BIRTHQTR to handle the June 15, 1981 cutoff more precisely

### Decision 3: Outcome Variable
**Choice:** Full-time employment = (EMPSTAT == 1) AND (UHRSWORK >= 35)

**Rationale:**
- EMPSTAT=1 indicates employed
- 35+ hours/week is standard definition of full-time employment
- Matches the outcome specified in instructions

### Decision 4: Identification Strategy
**Choice:** Difference-in-differences (DID) comparing DACA-eligible vs DACA-ineligible individuals, before vs after 2012.

**Rationale:**
- Natural quasi-experiment: DACA eligibility determined by pre-existing characteristics
- DID controls for time-invariant differences between groups and common time trends
- Standard approach in program evaluation literature

### Decision 5: Exclude 2012
**Choice:** Exclude all observations from 2012.

**Rationale:**
- DACA implemented June 15, 2012
- ACS does not record month of interview
- Cannot distinguish pre- vs post-treatment observations in 2012

### Decision 6: Model Specifications
**Choice:** Three main specifications:
1. Basic DID (no controls)
2. DID with demographic controls (age, age^2, sex, marital status, education)
3. DID with demographics + year FE + state FE (preferred)

**Rationale:**
- Progression shows sensitivity to controls
- Fixed effects control for aggregate time trends and state-level heterogeneity
- Demographic controls improve precision and address compositional differences

### Decision 7: Survey Weights
**Choice:** Use ACS person weights (PERWT) in all main specifications.

**Rationale:**
- Produces nationally representative estimates
- Standard practice with ACS data
- Unweighted results presented as robustness check

### Decision 8: Standard Errors
**Choice:** Heteroskedasticity-robust (HC1) standard errors.

**Rationale:**
- Binary outcome variable implies heteroskedasticity
- Conservative standard error estimates

---

## Step 4: Run Analysis

### Command:
```bash
python analysis.py
```

### Analysis Script: `analysis.py`
- Processes data in chunks (2M rows at a time) due to file size
- Filters to sample population
- Creates all variables
- Runs main DID regressions (3 specifications)
- Runs robustness checks (5 alternative specifications)
- Runs event study for parallel trends check
- Saves results to CSV files

### Results Summary:
- **Final sample size:** 561,470 observations
- **DACA eligible:** 83,611 (14.9%)
- **Preferred estimate:** 0.0654 (SE: 0.0043)
- **95% CI:** [0.0569, 0.0739]
- **Interpretation:** DACA eligibility increased full-time employment by 6.5 percentage points

---

## Step 5: Create Figures

### Command:
```bash
python create_figures.py
```

### Figures Created:
1. `event_study_plot.pdf` - Event study coefficients by year
2. `did_comparison.pdf` - Bar chart of pre/post means by treatment status
3. `robustness_forest.pdf` - Forest plot of robustness check results

---

## Step 6: Compile Report

### Commands:
```bash
# First pass
pdflatex -interaction=nonstopmode replication_report_10.tex

# Second pass (for TOC and references)
pdflatex -interaction=nonstopmode replication_report_10.tex

# Third pass (for final cross-references)
pdflatex -interaction=nonstopmode replication_report_10.tex
```

### Output:
- `replication_report_10.pdf` (21 pages)

---

## Step 7: Output Files

### Required Deliverables:
| File | Description | Status |
|------|-------------|--------|
| `replication_report_10.tex` | LaTeX source | Created |
| `replication_report_10.pdf` | PDF report | Created (21 pages) |
| `run_log_10.md` | This log file | Created |

### Supporting Files:
| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `main_results.csv` | Main regression results |
| `robustness_results.csv` | Robustness check results |
| `event_study_coefficients.csv` | Event study estimates |
| `summary_statistics.csv` | Descriptive statistics |
| `preferred_estimate.txt` | Preferred estimate summary |
| `event_study_plot.pdf` | Event study figure |
| `did_comparison.pdf` | DID comparison figure |
| `robustness_forest.pdf` | Robustness forest plot |

---

## Key Results Summary

### Preferred Estimate (Model 3: Year and State FE)
- **Coefficient:** 0.0654
- **Standard Error:** 0.0043
- **95% Confidence Interval:** [0.0569, 0.0739]
- **P-value:** < 0.0001
- **Sample Size:** 561,470

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 6.5 percentage points among Mexican-born, Hispanic-Mexican non-citizens. Given a pre-DACA baseline full-time employment rate of approximately 40% for eligible individuals, this represents a relative increase of about 16%.

### Robustness
The effect is robust to:
- Alternative age restrictions (18-35 only)
- Alternative outcome (any employment)
- Gender subsamples
- Unweighted estimation

### Parallel Trends
Event study analysis shows:
- Pre-trend coefficients close to zero in 2009-2011
- Some differential trends in 2006-2008 (caveat)
- Sharp increase in effects starting 2013
- Effects growing over time (2.6 pp in 2013 to 7.2 pp in 2016)

---

## End of Log
