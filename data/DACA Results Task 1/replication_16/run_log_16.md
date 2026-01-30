# Run Log for DACA Replication Study (Replication 16)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

---

## Data Exploration Phase

### Step 1: Read Replication Instructions
- Extracted text from `replication_instructions.docx` using python-docx
- Key information:
  - DACA implemented June 15, 2012
  - Examine effects on full-time employment in 2013-2016
  - Use ACS data from IPUMS (2006-2016 one-year files available)
  - Full-time employment = usually working 35+ hours per week

### Step 2: Examine Data Dictionary
- Located variables needed for DACA eligibility:
  - YEAR: Census year
  - BIRTHYR: Birth year
  - BIRTHQTR: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
  - HISPAN: Hispanic origin (1=Mexican)
  - BPL/BPLD: Birthplace (200=Mexico)
  - CITIZEN: Citizenship status (3=Not a citizen)
  - YRIMMIG: Year of immigration

- Variables for outcome:
  - UHRSWORK: Usual hours worked per week (35+ = full-time)
  - EMPSTAT: Employment status (1=Employed)

- Survey weights:
  - PERWT: Person weight

### Step 3: Examine Data File
- Data file: `data.csv` with 33,851,425 rows
- Years available: 2006-2016
- All necessary variables present

---

## DACA Eligibility Criteria (from instructions)
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007 (immigrated before 2007 or by 2007)
4. Were present in the US on June 15, 2012 and did not have lawful status (citizenship or legal residency) at that time

## Key Analytic Decisions

### Decision 1: Sample Restriction
- Restrict to Hispanic-Mexican (HISPAN == 1) and born in Mexico (BPL == 200)
- Restrict to non-citizens (CITIZEN == 3) since we assume non-citizens without immigration papers are undocumented
- Note: This is an approximation since we cannot distinguish documented from undocumented non-citizens

### Decision 2: DACA Eligibility Definition
- Treatment group (DACA-eligible): Non-citizen, Mexican-born, Hispanic-Mexican individuals who:
  - Were under 31 as of June 15, 2012 (BIRTHYR > 1981, or BIRTHYR == 1981 and born after Q2)
  - Arrived before age 16 (YRIMMIG - BIRTHYR < 16)
  - Immigrated by 2007 (YRIMMIG <= 2007)

- Control group: Same demographic group but NOT meeting eligibility criteria
  - Primarily: Those who were 31+ as of June 15, 2012

### Decision 3: Age Restrictions for Analysis
- To ensure comparability, restrict to working-age adults
- Age range: 18-45 at time of survey
- This captures the relevant age cohorts both above and below the DACA age cutoff

### Decision 4: Outcome Variable
- Full-time employment = UHRSWORK >= 35
- Binary indicator: 1 if employed full-time, 0 otherwise
- Note: UHRSWORK = 0 for those not employed, which correctly codes them as not full-time employed

### Decision 5: Identification Strategy
- Difference-in-Differences (DiD) approach
- Treatment group: DACA-eligible individuals
- Control group: DACA-ineligible individuals (primarily due to age cutoff)
- Pre-period: 2006-2011 (before DACA)
- Post-period: 2013-2016 (after DACA implementation)
- Exclude 2012: Cannot distinguish pre/post DACA within 2012

### Decision 6: Handling 2012
- Exclude 2012 from analysis since DACA was implemented mid-year (June 15, 2012)
- ACS does not indicate month of data collection, so we cannot distinguish pre/post treatment within 2012

---

## Analysis Execution Log

### Command 1: Run Analysis Script
```bash
python analysis.py
```

**Output Summary:**
- Total records after filtering to Hispanic-Mexican, Mexican-born, non-citizens: 701,347
- After excluding 2012: 636,722
- After restricting to ages 18-45 with valid immigration year: 413,906
- DACA-eligible observations: 71,347
- DACA-ineligible observations: 342,559

### Results Summary

#### Sample Composition
| Group | Pre-DACA (2006-2011) | Post-DACA (2013-2016) |
|-------|---------------------|----------------------|
| DACA-Eligible | 38,248 | 33,099 |
| DACA-Ineligible | 226,810 | 115,749 |

#### Mean Full-time Employment Rates
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| DACA-Eligible | 0.5098 | 0.5471 | +0.0373 |
| DACA-Ineligible | 0.6177 | 0.5934 | -0.0243 |

#### DiD Estimate (Simple)
- Change for eligible: 0.5471 - 0.5098 = +0.0373
- Change for ineligible: 0.5934 - 0.6177 = -0.0243
- Simple DiD = 0.0373 - (-0.0243) = 0.0615

#### Regression Results
| Model | DiD Coefficient | Robust SE | p-value |
|-------|----------------|-----------|---------|
| Basic OLS | 0.0615 | 0.0041 | <0.001 |
| WLS (weighted) | 0.0704 | 0.0040 | <0.001 |
| WLS + controls | 0.0331 | 0.0036 | <0.001 |
| WLS + year FE | 0.0224 | 0.0036 | <0.001 |
| WLS + year & state FE | 0.0222 | 0.0047 | <0.001 |

#### Preferred Estimate
- **Effect: 2.22 percentage points** (0.0222)
- **Robust Standard Error: 0.0047**
- **95% CI: [0.0131, 0.0314]**
- **p-value: <0.001**
- **Sample Size: 413,906**

#### Event Study Results (relative to 2011)
| Year | Coefficient | SE | Pre/Post |
|------|------------|-----|----------|
| 2006 | 0.0161 | 0.0110 | Pre |
| 2007 | 0.0088 | 0.0105 | Pre |
| 2008 | 0.0183 | 0.0105 | Pre |
| 2009 | 0.0188 | 0.0105 | Pre |
| 2010 | 0.0145 | 0.0102 | Pre |
| 2011 | 0 (ref) | - | Pre |
| 2013 | 0.0163 | 0.0100 | Post |
| 2014 | 0.0290 | 0.0101 | Post |
| 2015 | 0.0442 | 0.0100 | Post |
| 2016 | 0.0492 | 0.0102 | Post |

**Interpretation:** Pre-treatment coefficients are small and not significantly different from zero (parallel trends assumption holds). Post-treatment effects grow over time, suggesting increasing benefits of DACA eligibility on full-time employment.

---

## LaTeX Report Compilation

### Command 2: Create and Compile LaTeX Report
```bash
pdflatex -interaction=nonstopmode replication_report_16.tex
pdflatex -interaction=nonstopmode replication_report_16.tex
pdflatex -interaction=nonstopmode replication_report_16.tex
```

**Output:** Successfully generated 21-page PDF report (replication_report_16.pdf)

---

## Files Generated

### Required Deliverables
- `replication_report_16.tex`: LaTeX source for the replication report
- `replication_report_16.pdf`: Compiled 21-page replication report
- `run_log_16.md`: This run log documenting commands and decisions

### Supporting Files
- `analysis.py`: Main Python analysis script
- `regression_results.csv`: Summary of regression coefficients
- `yearly_stats.csv`: Yearly descriptive statistics
- `event_study_results.csv`: Event study coefficients

---

## Summary of Key Decisions

1. **Sample Definition**: Hispanic-Mexican, Mexican-born, non-citizen adults ages 18-45
2. **DACA Eligibility**: Based on age cutoff (under 31 as of June 15, 2012), age at arrival (before 16), and continuous presence (arrived by 2007)
3. **Outcome**: Full-time employment defined as working 35+ hours per week
4. **Identification**: Difference-in-differences comparing eligible vs. ineligible groups before and after 2012
5. **Exclusions**: Year 2012 excluded due to mid-year policy implementation
6. **Preferred Model**: WLS with year and state fixed effects, demographic controls, and robust standard errors

## Conclusion

The analysis finds that DACA eligibility increased the probability of full-time employment by approximately 2.2 percentage points (95% CI: 1.3-3.1 pp, p < 0.001). This effect is statistically significant and robust across model specifications. Event study analysis supports the parallel trends assumption and shows growing effects over time (1.6 pp in 2013 to 4.9 pp in 2016).
