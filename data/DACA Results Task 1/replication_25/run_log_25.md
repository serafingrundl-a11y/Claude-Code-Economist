# Run Log - DACA Replication Study (Replication 25)

## Overview
This log documents the commands, decisions, and key steps taken during the independent replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (35+ hours/week)?

---

## Step 1: Data Exploration

### Commands
```bash
# Check directory structure
ls -la data/

# View data dictionary
head -500 data/acs_data_dict.txt

# Check CSV header
head -1 data/data.csv

# Count rows in data file
wc -l data/data.csv  # Result: 33,851,425 rows
```

### Key Observations
- Data spans 2006-2016 (one-year ACS samples)
- 54 variables available
- ~34 million person-year observations

---

## Step 2: Sample Selection Decisions

### Population Definition
1. **Hispanic-Mexican ethnicity**: HISPAN = 1
   - Rationale: Instructions specify ethnically Hispanic-Mexican

2. **Born in Mexico**: BPL = 200
   - Rationale: Instructions specify Mexican-born

3. **Non-citizen**: CITIZEN = 3
   - Rationale: DACA targets undocumented immigrants; ACS cannot distinguish documented vs undocumented non-citizens, so using all non-citizens as proxy

4. **Working age**: AGE 16-64
   - Rationale: Standard labor force age range; DACA focused on young adults

### Resulting Sample
- Total relevant observations: 701,347
- After age restriction (16-64): 618,640
- After excluding 2012 and restricting to 2008-2016: 446,804

---

## Step 3: Variable Construction

### DACA Eligibility Criteria Implementation

1. **Arrived before age 16**:
   ```python
   age_at_arrival = YRIMMIG - BIRTHYR
   arrived_before_16 = (age_at_arrival < 16) & (age_at_arrival >= 0)
   ```

2. **Continuous presence since 2007**:
   ```python
   arrived_by_2007 = (YRIMMIG <= 2007) & (YRIMMIG > 0)
   ```

3. **Born after June 15, 1981** (under 31 as of June 15, 2012):
   ```python
   born_after_june1981 = (BIRTHYR >= 1982) |
                         ((BIRTHYR == 1981) & (BIRTHQTR in [3, 4]))
   ```
   - BIRTHQTR 3 = July-Sept, BIRTHQTR 4 = Oct-Dec
   - Conservative approach accounts for birth quarter

4. **DACA eligible** = all three criteria met

### Eligibility Breakdown
- Arrived before age 16: 153,052
- Arrived by 2007: 581,973
- Born after June 1981: 181,504
- **DACA eligible (all criteria): 91,428**

### Outcome Variable
```python
fulltime = (UHRSWORK >= 35).astype(int)
```
- Full-time employment = usually working 35+ hours/week
- Based on UHRSWORK (usual hours worked per week)

---

## Step 4: Empirical Strategy Decisions

### Difference-in-Differences Design
- **Treatment**: DACA eligibility
- **Post period**: 2013-2016
- **Pre period**: 2008-2011
- **Excluded**: 2012 (implementation year)

### Specification Choices
1. Basic DiD: No controls
2. DiD + demographics: Age, age^2, female, married
3. DiD + state FE
4. **Preferred**: DiD + state FE + year FE

### Weighting
- All regressions use person weights (PERWT)
- Heteroskedasticity-robust standard errors (HC1)

---

## Step 5: Analysis Commands

### Python Analysis Script
```bash
python analysis.py
```

### Key Package Dependencies
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)

---

## Step 6: Main Results

### Preferred Estimate (Model 4: Year + State FE)
- **Effect size**: 0.0258 (2.58 percentage points)
- **Standard error**: 0.0046
- **95% CI**: [0.0168, 0.0348]
- **P-value**: 0.0000
- **Sample size**: 446,804

### Interpretation
DACA eligibility increased full-time employment probability by 2.6 percentage points among Mexican-born, Hispanic-Mexican non-citizens.

---

## Step 7: Robustness Checks

| Specification | DiD Estimate | SE |
|--------------|--------------|-----|
| Main (full-time) | 0.0258 | 0.0046 |
| Employment (any) | 0.0406 | 0.0046 |
| Ages 18-45 only | 0.0180 | 0.0051 |
| Males only | 0.0169 | 0.0061 |
| Females only | 0.0290 | 0.0069 |

---

## Step 8: Event Study Results

### Pre-trends (relative to 2011)
- 2008: -0.0002 (SE: 0.0095) - Not significant
- 2009: 0.0074 (SE: 0.0094) - Not significant
- 2010: 0.0100 (SE: 0.0092) - Not significant

### Post-treatment effects
- 2013: 0.0132 (SE: 0.0091) - Not significant
- 2014: 0.0245 (SE: 0.0092) - Significant
- 2015: 0.0409 (SE: 0.0091) - Significant
- 2016: 0.0428 (SE: 0.0093) - Significant

### Interpretation
- Pre-trends support parallel trends assumption
- Effects emerge and grow over time after DACA implementation

---

## Step 9: Report Generation

### LaTeX Compilation
```bash
pdflatex replication_report_25.tex
pdflatex replication_report_25.tex  # Run twice for ToC
```

### Output Files
- replication_report_25.tex (LaTeX source)
- replication_report_25.pdf (Compiled report)
- run_log_25.md (This file)
- analysis.py (Python analysis code)
- results.json (Key results in JSON format)
- trends.csv (Employment trends data)

---

## Key Methodological Decisions Summary

1. **Sample restriction**: Focused on Mexican-born, Hispanic-Mexican, non-citizen population aged 16-64

2. **DACA eligibility definition**: Based on observable criteria (age at arrival, birth year/quarter, year of immigration)

3. **Outcome**: Full-time employment (35+ hours/week) using UHRSWORK

4. **Control group**: Non-DACA-eligible individuals from same population

5. **Time periods**: Pre (2008-2011), Post (2013-2016), excluding 2012

6. **Specification**: DiD with year FE, state FE, and demographic controls

7. **Inference**: Robust standard errors, survey weights

---

## Files Generated

| File | Description |
|------|-------------|
| analysis.py | Main Python analysis script |
| results.json | Key results in JSON format |
| trends.csv | Year-by-year employment trends |
| replication_report_25.tex | LaTeX source for report |
| replication_report_25.pdf | Compiled PDF report |
| run_log_25.md | This run log |

---

## Session Information

- Date: January 2026
- Platform: Windows
- Python: 3.14
- Key packages: pandas, numpy, statsmodels, scipy
