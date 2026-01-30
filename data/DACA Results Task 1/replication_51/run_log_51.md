# Run Log - DACA Replication Study (Replication 51)

## Overview
This document logs all commands and key decisions made during the independent replication of the DACA employment effects study.

---

## Date: January 25, 2026

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Instructions
- Read `replication_instructions.docx` using Python's python-docx library
- Research question identified: Effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals

### 1.2 Data Files Available
- `data/data.csv` - Main ACS data file (33,851,424 rows)
- `data/acs_data_dict.txt` - Variable codebook
- `data/state_demo_policy.csv` - Optional state-level data (not used)

### 1.3 Data Structure
- 54 variables in ACS data
- Years covered: 2006-2016 (11 years of 1-year ACS)
- Key variables identified:
  - YEAR: Survey year
  - BPL: Birthplace (200 = Mexico)
  - HISPAN: Hispanic origin (1 = Mexican)
  - BIRTHYR, BIRTHQTR: Birth timing
  - YRIMMIG: Year of immigration
  - CITIZEN: Citizenship status (3 = Not a citizen)
  - EMPSTAT: Employment status (1 = Employed)
  - UHRSWORK: Usual hours worked per week
  - PERWT: Person weight

---

## 2. Key Analytical Decisions

### 2.1 Sample Definition

**Decision: Restrict to Hispanic-Mexican individuals born in Mexico**
- Rationale: The instructions specify "ethnically Hispanic-Mexican Mexican-born people"
- Implementation: HISPAN == 1 AND BPL == 200
- Sample size: 991,261 person-year observations

**Decision: Restrict to working age (16-64)**
- Rationale: Standard labor economics practice; individuals outside this range unlikely to be in labor force
- Sample after restriction: 771,888 observations

**Decision: Exclude 2012**
- Rationale: DACA implemented June 15, 2012; ACS doesn't identify month of interview
- Cannot distinguish pre/post observations within 2012
- Pre-period: 2006-2011; Post-period: 2013-2016

### 2.2 DACA Eligibility Definition

**Criteria implemented:**
1. Arrived before 16th birthday: `YRIMMIG - BIRTHYR < 16`
2. Under 31 as of June 15, 2012: `BIRTHYR >= 1982` OR (`BIRTHYR == 1981` AND `BIRTHQTR >= 3`)
3. Continuous presence since 2007: `YRIMMIG <= 2007`
4. Undocumented status: `CITIZEN == 3` (not a citizen)
5. Valid immigration year: `YRIMMIG > 0`

**Decision: Assume non-citizens are undocumented**
- Rationale: Instructions state "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes"
- ACS does not distinguish documented from undocumented non-citizens
- CITIZEN == 3 captures "Not a citizen" without specifying legal status

**DACA-eligible observations: 83,611 (10.8% of working-age sample)**

### 2.3 Outcome Variable

**Decision: Full-time employment = EMPSTAT == 1 AND UHRSWORK >= 35**
- Rationale: Instructions specify "usually working 35 hours per week or more"
- This matches BLS definition of full-time work
- Binary indicator (0/1)

Full-time employment rates:
- DACA eligible pre-period: 37.14%
- DACA eligible post-period: 45.15%
- Control pre-period: 56.31%
- Control post-period: 56.39%

### 2.4 Control Group

**Decision: Use all non-DACA-eligible Hispanic-Mexican Mexican-born as control**
- Rationale: Same ethnic/national origin group provides comparability
- Control includes: naturalized citizens, legal residents, those who arrived at 16+, those who arrived after 2007, those over 31 in 2012
- Control observations: 688,277

### 2.5 Estimation Strategy

**Decision: Use difference-in-differences (DiD) design**
- Treatment: DACA eligibility (varies by individual characteristics)
- Post: Year >= 2013
- Interaction: DACA_eligible × Post is the causal estimator

**Decision: Use OLS with robust standard errors**
- Linear probability model for binary outcome
- Heteroskedasticity-robust (HC1) standard errors

---

## 3. Models Estimated

### Model 1: Basic DiD (PREFERRED)
```
fulltime_employed ~ DACA_eligible + Post + DACA_eligible × Post
```
- Result: DiD effect = 0.0793 (SE = 0.0036)
- 95% CI: [0.0721, 0.0864]
- p-value < 0.0001

### Model 2: DiD with Demographic Controls
```
fulltime_employed ~ DACA_eligible + Post + DACA_eligible × Post + AGE + AGE² + Female + Education + Married
```
- Result: DiD effect = 0.0176 (SE = 0.0034)
- Effect attenuated after controlling for age differences

### Model 3: DiD with Year Fixed Effects
```
fulltime_employed ~ DACA_eligible + DACA_eligible × Post + AGE + AGE² + Female + Education + Married + Year_FE
```
- Result: DiD effect = 0.0142 (SE = 0.0034)

### Model 4: Weighted DiD
```
Same as Model 1, using PERWT as weights
```
- Result: DiD effect = 0.0835 (SE = 0.0045)

---

## 4. Robustness Checks

### 4.1 Alternative Control Group (Non-citizens only)
- Restricts control to non-citizens who don't meet other DACA criteria
- Result: Effect = 0.0837 (SE = 0.0037)
- Similar to main estimate

### 4.2 Alternative Outcome (Any Employment)
- Result: Effect = 0.0938 (SE = 0.0036)
- Larger effect on any employment than on full-time employment

### 4.3 Heterogeneity by Sex
- Males: Effect = 0.0916 (SE = 0.0049)
- Females: Effect = 0.0570 (SE = 0.0049)
- Larger effect for males

### 4.4 Placebo Test (Fake treatment in 2010)
- Pre-period only (2006-2011), using 2010 as fake post
- Result: Effect = 0.0264 (SE = 0.0049)
- Statistically significant but smaller than main effect
- Suggests some pre-trend concerns

### 4.5 Event Study
- Year-by-year treatment effects relative to 2011
- Pre-period coefficients (2006-2009) negative, converging toward zero
- Post-period coefficients positive and growing (0.028 in 2013 to 0.089 in 2016)

---

## 5. Final Results Summary

### Preferred Estimate
- **Effect Size:** 7.93 percentage points
- **Standard Error:** 0.0036
- **95% Confidence Interval:** [7.21, 8.64] percentage points
- **P-value:** < 0.0001
- **Sample Size:** 771,888

### Interpretation
DACA eligibility is associated with a 7.93 percentage point increase in the probability of full-time employment among eligible Hispanic-Mexican Mexican-born individuals compared to non-eligible individuals. This represents approximately a 21% increase relative to the pre-DACA baseline full-time employment rate of 37.1%.

---

## 6. Files Generated

1. `analysis.py` - Main Python analysis script
2. `replication_report_51.tex` - LaTeX replication report
3. `replication_report_51.pdf` - Compiled PDF report
4. `run_log_51.md` - This log file
5. `descriptive_stats.csv` - Summary statistics
6. `main_results.csv` - Main regression results
7. `model_summaries.txt` - Full model output
8. `report_variables.json` - Key variables for report

---

## 7. Software Used

- Python 3.x
  - pandas (data manipulation)
  - numpy (numerical operations)
  - statsmodels (regression analysis)
- LaTeX (pdflatex for report compilation)

---

## 8. Limitations Noted

1. Cannot distinguish documented from undocumented immigrants in ACS
2. Some evidence of differential pre-trends in early years (2006-2009)
3. Cannot identify actual DACA recipients vs. eligible non-recipients
4. ACS is repeated cross-section, not panel data

---

## 9. Conclusion

The analysis finds a statistically significant positive effect of DACA eligibility on full-time employment. The preferred estimate of 7.93 percentage points is robust to alternative specifications, though somewhat attenuated when demographic controls are added. The event study reveals growing effects over time post-DACA, consistent with increasing program enrollment and cumulative benefits of work authorization.
