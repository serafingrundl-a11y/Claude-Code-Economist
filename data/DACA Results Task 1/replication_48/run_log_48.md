# Replication Run Log - Task 48

## Overview
This log documents all commands, decisions, and analytical choices made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

---

## Session Start: 2026-01-25

### 1. Initial Data Exploration

**Files identified:**
- `data/data.csv` - Main ACS data file (6.2 GB)
- `data/acs_data_dict.txt` - Data dictionary
- `data/state_demo_policy.csv` - Optional state-level data
- `replication_instructions.docx` - Task instructions

**Data years available:** 2006-2016 (ACS 1-year files)

**Key variables identified from data dictionary:**
- `YEAR` - Census year
- `BIRTHYR` - Birth year
- `BIRTHQTR` - Quarter of birth (1-4)
- `HISPAN/HISPAND` - Hispanic origin (1 = Mexican for HISPAN)
- `BPL/BPLD` - Birthplace (200 = Mexico)
- `CITIZEN` - Citizenship status (3 = Not a citizen)
- `YRIMMIG` - Year of immigration
- `UHRSWORK` - Usual hours worked per week (outcome: >=35 = full-time)
- `EMPSTAT` - Employment status
- `PERWT` - Person weights

### 2. DACA Eligibility Criteria (from instructions)
To be DACA eligible, a person must:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

**Operationalization decisions:**
- Age criterion: Born 1982 or later (conservative - ensures under 31 on June 15, 2012)
- Age at arrival: Immigration year minus birth year < 16
- Continuous residence: Immigrated by 2007 (YRIMMIG <= 2007)
- Non-citizen status: CITIZEN = 3 (not a citizen)
- Hispanic-Mexican from Mexico: HISPAN = 1 AND BPL = 200

---

## Analysis Plan

### Research Design
Difference-in-differences (DiD) approach:
- **Treatment group:** DACA-eligible Mexican-born non-citizens
- **Control group:** DACA-ineligible Mexican-born non-citizens (non-citizens who immigrated but fail at least one eligibility criterion, primarily the age requirement)
- **Pre-period:** 2006-2011
- **Post-period:** 2013-2016

**Note:** 2012 excluded because DACA was implemented mid-year (June 15, 2012), so we cannot distinguish pre vs post in that year.

### Outcome Variable
Full-time employment: UHRSWORK >= 35 hours per week

### Estimation Strategy
1. Primary specification: DiD regression with year fixed effects and demographic controls
2. Standard errors clustered at state level
3. Robustness checks: Alternative age bandwidths, by gender, alternative outcomes

---

## Commands Executed

```bash
# View data structure
head -5 data/data.csv

# Run main analysis
python analysis.py
```

---

## Key Analytical Decisions

### Decision 1: Sample Restriction to Hispanic-Mexican, Mexican-born
**Rationale:** The research question specifically focuses on ethnically Hispanic-Mexican Mexican-born individuals. This is the population most likely to be DACA-eligible and allows for a cleaner identification.

**Implementation:**
```python
(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)
```

### Decision 2: Control Group Definition
**Rationale:** The control group consists of Mexican-born non-citizens who are NOT DACA eligible. The primary difference is the age criterion - control group members were too old (born before 1982) to qualify for DACA. This creates a plausible counterfactual as both groups face similar labor market conditions and immigration status.

**Implementation:**
```python
df['control'] = (
    (df['non_citizen'] == 1) &
    (df['daca_eligible'] == 0) &
    (df['YRIMMIG'] > 0)
)
```

### Decision 3: Excluding 2012
**Rationale:** DACA was implemented on June 15, 2012. The ACS does not record the month of interview, so we cannot determine whether observations from 2012 were collected before or after DACA implementation. Excluding 2012 ensures clean pre/post periods.

### Decision 4: Working Age Restriction (18-64)
**Rationale:** Employment outcomes are most relevant for working-age adults. This is standard in labor economics literature.

### Decision 5: Birth Year Cutoff (1982)
**Rationale:** Using born >= 1982 (rather than 1981) as the cutoff is conservative. Since we only observe birth year (not month/day), using 1982 ensures individuals were definitely under 31 on June 15, 2012.

### Decision 6: Weighted Least Squares with Person Weights
**Rationale:** ACS provides person weights (PERWT) to account for the complex survey design. Using weights provides population-representative estimates.

### Decision 7: State-Clustered Standard Errors
**Rationale:** Clustering at the state level accounts for within-state correlation in errors and is appropriate given policy variation at the state level.

---

## Results Summary

### Sample Sizes
- Total Hispanic-Mexican, Mexican-born: 991,261
- Non-citizens: 701,347
- DACA eligible: 130,799
- Final analysis sample (working age, excluding 2012): 547,614

### Main Results (Preferred Specification with Year FE)
- **DiD Coefficient:** 0.0210
- **Standard Error:** 0.0036
- **95% CI:** [0.0139, 0.0281]
- **P-value:** <0.0001

**Interpretation:** DACA eligibility is associated with a 2.1 percentage point increase in the probability of full-time employment. This is statistically significant at all conventional levels.

### Pre-Period Balance
| Variable | DACA Eligible | Control |
|----------|---------------|---------|
| Full-time | 0.520 | 0.628 |
| Employed | 0.608 | 0.685 |
| In Labor Force | 0.700 | 0.746 |
| Age | 22.1 | 37.4 |
| Female | 0.442 | 0.428 |
| Married | 0.254 | 0.622 |
| HS Education | 0.620 | 0.399 |

### Robustness Checks
| Specification | Coefficient | SE |
|---------------|-------------|-----|
| Ages 20-40 only | 0.0138 | 0.0051 |
| Men only | 0.0148 | 0.0057 |
| Women only | 0.0198 | 0.0063 |
| Any employment outcome | 0.0305 | 0.0053 |
| Labor force participation | 0.0278 | 0.0059 |

### Event Study Pre-Trends
| Year | Coefficient | SE |
|------|-------------|-----|
| 2006 | 0.0115 | 0.0122 |
| 2007 | 0.0072 | 0.0063 |
| 2008 | 0.0186 | 0.0133 |
| 2009 | 0.0199 | 0.0105 |
| 2010 | 0.0178 | 0.0152 |
| 2011 | 0.0000 | (ref) |
| 2013 | 0.0162 | 0.0091 |
| 2014 | 0.0290 | 0.0123 |
| 2015 | 0.0440 | 0.0121 |
| 2016 | 0.0440 | 0.0107 |

**Note on pre-trends:** The pre-period coefficients are relatively small and not significantly different from zero in most years, though there is some fluctuation. The post-period shows a clear increase, particularly in 2015-2016.

---

## Files Generated
- `analysis.py` - Main analysis script
- `run_log_48.md` - This log file
- `replication_report_48.tex` - LaTeX report (pending)
- `replication_report_48.pdf` - PDF report (pending)

---

## Software Environment
- Python 3.x
- pandas, numpy, statsmodels, scipy
- LaTeX (pdflatex for compilation)
