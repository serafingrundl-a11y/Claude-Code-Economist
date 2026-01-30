# DACA Replication Run Log - Session 18

## Overview
This log documents all commands and key decisions made during the DACA replication analysis.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

## Identification Strategy
- **Treatment group**: Individuals aged 26-30 at the time of DACA implementation (June 15, 2012)
- **Control group**: Individuals aged 31-35 at the time of DACA implementation
- **Method**: Difference-in-differences comparing pre-treatment (2006-2011) to post-treatment (2013-2016) periods
- **Note**: 2012 is excluded as treatment timing within that year is ambiguous

## Session Start
- Date: 2025-01-26
- Data source: American Community Survey (ACS) 2006-2016 via IPUMS

---

## Key Decisions

### 1. Sample Definition
- **Hispanic-Mexican ethnicity**: HISPAN == 1 (Mexican)
- **Born in Mexico**: BPL == 200 (Mexico)
- **Non-citizen status**: CITIZEN == 3 (Not a citizen) - proxy for undocumented status
- **Arrived before age 16**: Calculated from YRIMMIG and BIRTHYR
- **Continuous US residence since 2007**: YRIMMIG <= 2007

### 2. Age at DACA Implementation
- Treatment group: Birth year 1982-1986 (ages 26-30 on June 15, 2012)
- Control group: Birth year 1977-1981 (ages 31-35 on June 15, 2012)

### 3. Outcome Variable
- Full-time employment: UHRSWORK >= 35 (usually works 35+ hours per week)

### 4. Treatment Period
- Pre-treatment: 2006-2011
- Post-treatment: 2013-2016
- Excluded: 2012 (implementation year - ambiguous treatment status)

---

## Commands Executed

### Data Exploration
```
# Examined data dictionary for variable codes
# Key variables identified:
# - YEAR: Survey year
# - HISPAN: Hispanic origin (1 = Mexican)
# - BPL: Birthplace (200 = Mexico)
# - CITIZEN: Citizenship (3 = Not a citizen)
# - YRIMMIG: Year of immigration
# - BIRTHYR: Birth year
# - UHRSWORK: Usual hours worked per week
# - PERWT: Person weight
```

### Analysis Execution
```python
# Ran: python daca_analysis.py
# Data loaded: 33,851,424 total observations from ACS 2006-2016
# Sample filtering steps:
#   1. Hispanic-Mexican (HISPAN==1): 2,945,521 obs
#   2. Born in Mexico (BPL==200): 991,261 obs
#   3. Non-citizen (CITIZEN==3): 701,347 obs
#   4. Arrived before age 16: 205,327 obs
#   5. Present since 2007 (YRIMMIG<=2007): 195,023 obs
#   6. Age 26-35 on June 15, 2012: 47,418 obs
#   7. Excluding 2012: 43,238 obs (final sample)
```

---

## Results Summary

### Sample Distribution
| Group | Pre-Period (2006-2011) | Post-Period (2013-2016) | Total |
|-------|------------------------|-------------------------|-------|
| Control (31-35) | 11,683 | 6,085 | 17,768 |
| Treatment (26-30) | 16,694 | 8,776 | 25,470 |
| **Total** | **28,377** | **14,861** | **43,238** |

### Main Results (Preferred Specification)
- **DiD Coefficient**: 0.0456
- **Standard Error**: 0.0090 (0.0107 robust)
- **95% CI**: [0.0280, 0.0633]
- **P-value**: < 0.0001

### Interpretation
DACA eligibility is associated with a 4.56 percentage point increase in the probability of full-time employment among eligible Hispanic-Mexican immigrants. This effect is statistically significant at the 1% level.

### Mean Outcomes
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 0.631 | 0.660 | +0.029 |
| Control (31-35) | 0.673 | 0.643 | -0.030 |
| **DiD** | | | **+0.059** |

---

## Files Generated
1. `daca_analysis.py` - Main analysis script
2. `results_summary.csv` - Key results in CSV format
3. `replication_report_18.tex` - LaTeX report (49KB)
4. `replication_report_18.pdf` - Final PDF report (25 pages, 286KB)
5. `run_log_18.md` - This log file

---

## Session Completion
- Completion timestamp: 2025-01-26
- All deliverables generated successfully
- Analysis reproducible from `daca_analysis.py`

## Software Used
- Python 3.14.2
- pandas 2.3.3
- statsmodels 0.15.0
- numpy 2.3.0
- pdflatex (MiKTeX 25.12)

## Robustness Checks Performed
1. Basic DiD (no controls)
2. DiD with year fixed effects
3. DiD with year FE + individual controls
4. DiD with year FE + state FE + controls (preferred)
5. HC1 robust standard errors
6. Heterogeneity by sex
7. Event study / parallel trends check

## Conclusion
The analysis finds strong evidence that DACA eligibility increased full-time employment by approximately 4.6 percentage points (p < 0.0001). This represents a roughly 7% increase relative to baseline full-time employment rates in the treatment group.
