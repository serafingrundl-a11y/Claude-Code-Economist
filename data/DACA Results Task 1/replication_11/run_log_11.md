# Run Log - DACA Replication Study (Replication 11)

## Overview
This document logs all commands and key decisions made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico.

## Date
2025-01-25

---

## Step 1: Data Exploration

### Commands Executed
```bash
# List files in working directory
dir "C:\Users\seraf\DACA Results Task 1\replication_11"
dir "C:\Users\seraf\DACA Results Task 1\replication_11\data"

# Count lines in data file
wc -l data.csv
# Result: 33,851,425 rows (including header)

# Extract replication instructions from docx
python -c "from docx import Document; doc = Document('replication_instructions.docx'); print('\n'.join([p.text for p in doc.paragraphs]))"
```

### Key Findings
- Data file: `data/data.csv` (~34 million rows, 54 columns)
- Data dictionary: `data/acs_data_dict.txt`
- Years covered: 2006-2016 (ACS 1-year samples)
- Key variables identified:
  - HISPAN: Hispanic origin (1 = Mexican)
  - BPL: Birthplace (200 = Mexico)
  - CITIZEN: Citizenship status (3 = Not a citizen)
  - YRIMMIG: Year of immigration
  - YRSUSA1: Years in United States
  - BIRTHYR: Birth year
  - BIRTHQTR: Birth quarter
  - UHRSWORK: Usual hours worked per week
  - EMPSTAT: Employment status

---

## Step 2: DACA Eligibility Definition

### Key Decisions

1. **Target Population**: Hispanic-Mexican individuals born in Mexico
   - HISPAN == 1 (Mexican)
   - BPL == 200 (Mexico)

2. **Citizenship Restriction**: Non-citizens only
   - CITIZEN == 3 (Not a citizen)
   - Also included CITIZEN == 4 or 5 (first papers, status unknown)
   - Rationale: Cannot distinguish documented vs undocumented; instructions say to assume non-citizens without papers are undocumented

3. **DACA Eligibility Criteria** (announced June 15, 2012):
   - **Age at arrival < 16**: Calculated as YRIMMIG - BIRTHYR
   - **Under 31 as of June 15, 2012**: Born after June 1981
     - Used BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR in [3, 4])
     - Q3 = July-Sept, Q4 = Oct-Dec (definitely under 31 on June 15, 2012)
   - **In US since June 15, 2007**: YRIMMIG <= 2007
   - **Present in US**: Assumed for all ACS respondents

4. **Outcome Variable**: Full-time employment
   - UHRSWORK >= 35 (working 35+ hours per week)

5. **Time Periods**:
   - Pre-DACA: 2006-2011
   - Post-DACA: 2013-2016
   - 2012 excluded (ambiguous - some before/after DACA announcement)

6. **Sample Restrictions**:
   - Ages 16-64 (working-age population)

---

## Step 3: Analysis Strategy

### Identification Strategy
Difference-in-Differences (DiD) comparing:
- Treatment group: DACA-eligible individuals
- Control group: Non-DACA-eligible Hispanic-Mexicans born in Mexico (non-citizens)
- Pre-period: 2006-2011
- Post-period: 2013-2016

### Regression Models
1. **Model 1**: Basic DiD (no controls)
   - fulltime ~ daca_eligible + post + daca_x_post

2. **Model 2**: DiD with demographic controls
   - Controls: AGE, age_sq, female, married, educ_hs

3. **Model 3**: DiD with year fixed effects
   - Added: C(YEAR)

4. **Model 4**: Full specification with state and year fixed effects (PREFERRED)
   - Added: C(STATEFIP)

5. **Model 5**: Weighted regression using PERWT (survey weights)

### Robustness Checks
- Alternative outcome: Any employment (EMPSTAT == 1)
- Restricted sample: Ages 18-30
- Event study: Year-by-year DACA effects (relative to 2011)

---

## Step 4: Analysis Execution

### Command
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_11" && python analysis.py
```

### Sample Sizes
- Total Hispanic-Mexican born in Mexico: 991,261
- Non-citizen Hispanic-Mexicans: 701,347
- DACA-eligible: 133,120
- Non-DACA-eligible: 568,227
- Working-age (16-64): 618,640
- Final sample (excluding 2012): 561,470

---

## Step 5: Results Summary

### Main Finding (Preferred Specification - Model 4)
- **DiD Estimate**: 0.0325 (3.25 percentage points)
- **Standard Error**: 0.0033
- **95% CI**: [0.0259, 0.0391]
- **p-value**: < 0.0001
- **Sample Size**: 561,470

### Interpretation
DACA eligibility is associated with a statistically significant 3.25 percentage point increase in the probability of full-time employment among Hispanic-Mexican non-citizens born in Mexico.

### Robustness
- Alternative outcome (any employment): DiD = 0.0426 (SE: 0.0033)
- Weighted regression: DiD = 0.0300 (SE: 0.0033)
- Event study shows parallel trends pre-2012 and increasing effects post-2012

---

## Output Files Generated

1. `analysis.py` - Main analysis script
2. `results/summary_statistics.csv` - Summary statistics by group and period
3. `results/regression_results.csv` - Main regression results table
4. `results/event_study_results.csv` - Event study coefficients
5. `results/full_regression_output.txt` - Complete regression output
6. `replication_report_11.tex` - LaTeX report
7. `replication_report_11.pdf` - Compiled PDF report
8. `run_log_11.md` - This log file

---

## Methodological Notes

### Limitations
1. Cannot distinguish documented vs undocumented immigrants in ACS
2. ACS does not identify month of interview, so 2012 observations are ambiguous
3. DACA eligibility is imputed based on observable characteristics
4. Potential measurement error in age at arrival calculations
5. Selection into non-citizenship status may be endogenous

### Assumptions
1. Parallel trends: DACA-eligible and non-eligible groups would have followed similar employment trends absent DACA
2. No anticipation: Employment decisions not affected by DACA before its announcement
3. SUTVA: One person's DACA status does not affect another's employment

---

## Software Environment
- Python 3.14
- pandas, numpy, statsmodels, scipy
- Analysis completed: 2025-01-25
