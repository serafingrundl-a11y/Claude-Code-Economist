# Replication Run Log - Study 83

## Overview
This log documents all commands executed and key decisions made during the independent replication of the DACA employment effects study.

---

## Environment Setup
- **Date**: January 25, 2026
- **Platform**: Windows 10/11
- **Python Version**: 3.x
- **Working Directory**: `C:\Users\seraf\DACA Results Task 1\replication_83`

---

## Data Files
- **Main data**: `data/data.csv` (6.3 GB, 33,851,424 observations)
- **Data dictionary**: `data/acs_data_dict.txt`
- **State-level data**: `data/state_demo_policy.csv` (optional, not used)

---

## Analysis Steps

### Step 1: Read Replication Instructions
- Extracted text from `replication_instructions.docx` using python-docx
- Key research question: Effect of DACA eligibility on full-time employment (35+ hours/week)
- Population: Hispanic-Mexican, Mexican-born individuals in the US
- Period: 2006-2016 ACS data, with post-treatment period 2013-2016

### Step 2: Examine Data Structure
Commands:
```bash
head -1 data/data.csv  # View column headers
```

Available variables:
- YEAR, SAMPLE, SERIAL, CBSERIAL, HHWT, CLUSTER, REGION, STATEFIP, PUMA, METRO
- STRATA, GQ, FOODSTMP, PERNUM, PERWT, FAMSIZE, NCHILD, RELATE, RELATED, SEX
- AGE, BIRTHQTR, MARST, BIRTHYR, RACE, RACED, HISPAN, HISPAND, BPL, BPLD
- CITIZEN, YRIMMIG, YRSUSA1, YRSUSA2, HCOVANY, HINSEMP, HINSCAID, HINSCARE
- EDUC, EDUCD, EMPSTAT, EMPSTATD, LABFORCE, CLASSWKR, CLASSWKRD, OCC, IND
- WKSWORK1, WKSWORK2, UHRSWORK, INCTOT, FTOTINC, INCWAGE, POVERTY

### Step 3: Sample Construction Decisions

#### Decision 1: Define Hispanic-Mexican
- Used HISPAN = 1 (Mexican)
- IPUMS coding: 0=Not Hispanic, 1=Mexican, 2=Puerto Rican, 3=Cuban, 4=Other

#### Decision 2: Born in Mexico
- Used BPL = 200 (Mexico)
- Standard IPUMS birthplace code for Mexico

#### Decision 3: Non-citizen status
- Used CITIZEN = 3 (Not a citizen)
- Per instructions: assume non-citizens without immigration papers are undocumented
- Excludes naturalized citizens and those with first papers

#### Decision 4: Working age restriction
- Restricted to ages 16-64
- Rationale: Focus on population with reasonable labor force attachment potential
- Lower bound: Legal minimum working age
- Upper bound: Near retirement age

#### Decision 5: Exclude 2012
- DACA implemented June 15, 2012
- ACS does not record month of survey
- Cannot distinguish pre/post DACA within 2012
- Excluded 2012 to avoid contamination

### Step 4: DACA Eligibility Coding

Based on official DACA requirements:

1. **Arrived before 16th birthday**
   ```python
   age_at_arrival = YRIMMIG - BIRTHYR
   arrived_before_16 = (age_at_arrival < 16)
   ```

2. **Under 31 as of June 15, 2012**
   ```python
   under_31_june2012 = (BIRTHYR >= 1982) |
                       ((BIRTHYR == 1981) & (BIRTHQTR >= 2))
   ```
   - BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
   - Those born in 1981 Q2 or later would be 30 by June 2012

3. **In US since June 15, 2007**
   ```python
   in_us_since_2007 = (YRIMMIG <= 2007)
   ```
   - Using immigration year as proxy for continuous residence

4. **Combined eligibility**
   ```python
   daca_eligible = arrived_before_16 & under_31_june2012 & in_us_since_2007
   ```

### Step 5: Outcome Variable
- **Full-time employment**: UHRSWORK >= 35
- UHRSWORK = Usual hours worked per week
- Binary indicator: 1 if works 35+ hours, 0 otherwise

### Step 6: Control Variables
- Age and age-squared (AGE, AGE^2)
- Female indicator (SEX = 2)
- Married indicator (MARST in [1, 2])
- Education categories from EDUC:
  - Less than high school (EDUC < 6) - reference
  - High school (EDUC = 6)
  - Some college (EDUC in [7, 8, 9])
  - College+ (EDUC >= 10)
- State fixed effects (STATEFIP)
- Year fixed effects (YEAR)

### Step 7: Statistical Analysis

#### Model Specifications:
1. Basic DiD (no controls)
2. DiD + demographics (age, gender, marital status)
3. DiD + demographics + education
4. Full model with year and state fixed effects

#### Inference:
- Standard errors clustered at state level
- Robust to heteroskedasticity

#### Robustness checks:
- Weighted analysis using PERWT
- Prime working age subsample (25-54)
- Gender-stratified analysis
- Alternative control group (arrived after 16)
- Event study specification

---

## Key Results

### Sample Sizes
| Stage | N |
|-------|---|
| Initial ACS data | 33,851,424 |
| Hispanic-Mexican | 2,945,521 |
| Born in Mexico | 991,261 |
| Non-citizen | 701,347 |
| Ages 16-64 | 618,640 |
| Excluding 2012 | 561,470 |

### Eligibility Distribution
- DACA-eligible: 84,581 (15.1%)
- Non-eligible: 476,889 (84.9%)

### Main Results
| Model | DiD Estimate | SE | p-value |
|-------|--------------|-----|---------|
| Basic | 0.0884 | 0.0044 | <0.001 |
| + Demographics | 0.0404 | 0.0055 | <0.001 |
| + Education | 0.0371 | 0.0050 | <0.001 |
| Full (preferred) | 0.0311 | 0.0049 | <0.001 |

### Preferred Estimate
- **Effect size**: 3.1 percentage points
- **95% CI**: [2.2, 4.1]
- **Sample size**: 561,470
- **Interpretation**: DACA eligibility increased probability of full-time employment by 3.1 pp

---

## Output Files

1. `analysis.py` - Main analysis script
2. `descriptive_stats.csv` - Summary statistics
3. `regression_results.csv` - Regression results table
4. `event_study_results.csv` - Event study coefficients
5. `key_results.json` - Key results for report
6. `replication_report_83.tex` - LaTeX report
7. `replication_report_83.pdf` - Compiled PDF report
8. `run_log_83.md` - This log file

---

## Commands Executed

```bash
# Extract instructions from Word doc
python -c "from docx import Document; doc = Document('replication_instructions.docx'); print('\n'.join([p.text for p in doc.paragraphs]))"

# List data files
ls -la data/

# View data headers
head -1 data/data.csv

# Run main analysis
python analysis.py

# Compile LaTeX report
pdflatex replication_report_83.tex
pdflatex replication_report_83.tex  # Run twice for TOC
```

---

## Key Methodological Decisions

1. **Sample definition**: Focused on Mexican-born, Hispanic-Mexican non-citizens to approximate DACA-eligible population

2. **Control group**: All non-DACA-eligible individuals meeting other sample criteria (alternative: only those who arrived after 16)

3. **Eligibility coding**: Used birth year, birth quarter, and immigration year to approximate DACA criteria

4. **Treatment period**: 2013-2016 (excluded 2012 due to mid-year implementation)

5. **Inference**: State-clustered standard errors to account for within-state correlation

6. **Preferred model**: Full specification with demographic controls, education, and year/state fixed effects

---

## Notes

- The ACS is a repeated cross-section, not panel data
- Cannot observe actual DACA receipt, only eligibility
- Cannot perfectly verify continuous residence or physical presence requirements
- Non-citizen status used as proxy for undocumented status
- Results should be interpreted as intent-to-treat effects of eligibility

---

## Completion Status

- [x] Data loaded and examined
- [x] Sample restrictions applied
- [x] DACA eligibility coded
- [x] Outcome variable created
- [x] Control variables generated
- [x] Descriptive statistics computed
- [x] DiD regressions estimated
- [x] Event study conducted
- [x] Robustness checks completed
- [x] LaTeX report written
- [x] PDF compiled
- [x] Run log documented
