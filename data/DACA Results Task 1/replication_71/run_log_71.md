# Run Log - DACA Replication Study 71

## Overview
This log documents all commands and key decisions made during the replication of the DACA study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
What was the causal impact of eligibility for DACA (implemented June 15, 2012) on the probability of full-time employment (35+ hours/week) among ethnically Hispanic-Mexican, Mexican-born people living in the US, examining effects in 2013-2016?

---

## Session Log

### Step 1: Read Replication Instructions
- Read `replication_instructions.docx` using python-docx
- Key requirements identified:
  - Sample: Hispanic-Mexican ethnicity, Mexican-born, non-citizens
  - Treatment: DACA eligibility (arrived before 16, under 31 on June 15, 2012, lived continuously since June 2007, present on June 15, 2012)
  - Outcome: Full-time employment (35+ hours/week)
  - Data: ACS 2006-2016 (1-year files only)
  - Post-period: 2013-2016

### Step 2: Review Data Dictionary
- Reviewed `acs_data_dict.txt`
- Key variables identified:
  - YEAR: Census year (2006-2016)
  - HISPAN/HISPAND: Hispanic origin (1=Mexican, 100-107 for Mexican detailed)
  - BPL/BPLD: Birthplace (200=Mexico)
  - CITIZEN: Citizenship status (3=Not a citizen)
  - YRIMMIG: Year of immigration
  - BIRTHYR: Birth year
  - BIRTHQTR: Birth quarter (for age calculation)
  - UHRSWORK: Usual hours worked per week
  - EMPSTAT: Employment status
  - AGE: Age
  - PERWT: Person weight

### Step 3: Data Exploration
- Data file: `data.csv` with 33,851,425 observations
- All years 2006-2016 present
- Variables confirmed present in data

### Step 4: Define DACA Eligibility Criteria
DACA eligibility requires:
1. Arrived in US before 16th birthday (YRIMMIG - BIRTHYR < 16)
2. Under 31 on June 15, 2012 (born after June 15, 1981)
3. Present in US since June 15, 2007 (YRIMMIG <= 2007)
4. Not a citizen (CITIZEN == 3)
5. Mexican-born (BPL == 200)
6. Hispanic-Mexican ethnicity (HISPAN == 1)

### Step 5: Sample Restrictions
- Restrict to Hispanic-Mexican ethnicity (HISPAN == 1)
- Restrict to Mexican-born (BPL == 200)
- Restrict to non-citizens (CITIZEN == 3, assumed undocumented per instructions)
- Restrict to working-age population (18-55)

### Step 6: Identification Strategy
- Difference-in-Differences (DiD) approach
- Treatment group: DACA-eligible non-citizens
- Control group: Non-DACA-eligible non-citizens (similar demographics but ineligible)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (excluding 2012 as transitional year)

### Step 7: Outcome Variable
- Full-time employment: UHRSWORK >= 35

### Step 8: Analysis Steps
1. Load and clean data (chunked loading due to memory constraints)
2. Create eligibility indicators
3. Run DiD regression with controls
4. Robustness checks (different age groups, gender subsamples, employment outcome)
5. Event study analysis
6. Generate tables and figures

---

## Commands Executed

```bash
# Read instructions
python -c "from docx import Document; doc = Document('replication_instructions.docx'); [print(p.text) for p in doc.paragraphs]"

# View data structure
head -5 data/data.csv
wc -l data/data.csv

# Run main analysis
python analysis_71.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_71.tex
pdflatex -interaction=nonstopmode replication_report_71.tex
pdflatex -interaction=nonstopmode replication_report_71.tex
```

---

## Key Decisions

1. **Exclusion of 2012**: Since DACA was implemented mid-2012 (June 15) and ACS doesn't distinguish months, 2012 is excluded from analysis.

2. **Age restrictions**: Limited to working-age population (18-55) to focus on those most relevant for employment outcomes and maintain reasonable overlap between treatment and control groups.

3. **Eligibility definition**:
   - Arrived before age 16: YRIMMIG - BIRTHYR < 16
   - Under 31 on June 15, 2012: BIRTHYR >= 1982, or BIRTHYR = 1981 with BIRTHQTR >= 3
   - Present since June 2007: YRIMMIG <= 2007

4. **Control group**: Used ineligible Mexican-born non-citizen Hispanics (arrived too late, too old, or age-ineligible) as comparison group.

5. **Regression specification**: DiD with state and year fixed effects, individual demographic controls (age, age squared, female, married, education categories).

6. **Standard errors**: Clustered at state level to account for potential within-state correlation.

7. **Survey weights**: All regressions use PERWT weights for national representativeness.

---

## Results Summary

### Main Finding
- **DiD Coefficient**: 0.023 (2.3 percentage points)
- **Standard Error**: 0.004
- **95% Confidence Interval**: [0.016, 0.030]
- **p-value**: < 0.001

### Sample Size
- Total observations: 507,423
- DACA-eligible: 71,347
- Ineligible: 436,076

### Raw DiD Table
|                | Pre-DACA | Post-DACA | Difference |
|----------------|----------|-----------|------------|
| Eligible       | 0.510    | 0.547     | +0.037     |
| Ineligible     | 0.616    | 0.595     | -0.020     |
| **DiD**        |          |           | **+0.058** |

### Robustness Checks
| Specification          | Coefficient | SE     |
|-----------------------|-------------|--------|
| Main (full sample)     | 0.023***    | 0.004  |
| Employment (any)       | 0.034***    | 0.005  |
| Age 20-45 only         | 0.016***    | 0.004  |
| Males only             | 0.013**     | 0.005  |
| Females only           | 0.028***    | 0.006  |

### Event Study
Pre-period coefficients (relative to 2011) are small and insignificant, supporting parallel trends assumption. Post-period shows increasing effects: 2013 (+0.013), 2014 (+0.028**), 2015 (+0.043***), 2016 (+0.046***).

---

## Files Created
- `analysis_71.py`: Main analysis script (Python)
- `results_summary.csv`: Saved regression results
- `summary_statistics.csv`: Summary statistics by eligibility
- `event_study_results.csv`: Event study coefficients
- `replication_report_71.tex`: LaTeX report (~20 pages)
- `replication_report_71.pdf`: Compiled PDF report (19 pages)
- `run_log_71.md`: This run log

---

## Interpretation

DACA eligibility increased the probability of full-time employment by approximately 2.3 percentage points (95% CI: 1.6-3.0 pp), representing a roughly 4.5% increase relative to the pre-DACA baseline full-time employment rate of 51% for eligible individuals. This effect is:

1. Statistically significant at the 1% level
2. Robust across alternative specifications
3. Larger for women (2.8 pp) than for men (1.3 pp)
4. Consistent with work authorization enabling transition to formal full-time employment

The event study shows no significant pre-trends (supporting causal interpretation) and gradually increasing effects in the post-period (consistent with program rollout dynamics).
