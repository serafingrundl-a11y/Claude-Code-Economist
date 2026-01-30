# Replication Run Log - Session 24

## Research Question
Estimate the causal impact of DACA eligibility on full-time employment (working 35+ hours/week) among Hispanic-Mexican, Mexican-born individuals in the US, examining effects in 2013-2016.

## Data Source
- American Community Survey (ACS) 2006-2016 via IPUMS USA
- 33,851,424 observations in data.csv
- Data dictionary: acs_data_dict.txt

---

## Key Analytical Decisions

### 1. Sample Selection

**Decision**: Restrict sample to Hispanic-Mexican, Mexican-born, non-citizen individuals.

**Variables used**:
- `HISPAN == 1` (Mexican Hispanic ethnicity)
- `BPL == 200` (Born in Mexico)
- `CITIZEN == 3` (Not a citizen - proxy for undocumented status)

**Rationale**:
- Instructions specify "ethnically Hispanic-Mexican Mexican-born people"
- Non-citizens without naturalization assumed undocumented per instructions
- Citizens and naturalized citizens excluded as they were already documented

### 2. DACA Eligibility Criteria

**Eligibility conditions from instructions**:
1. Arrived unlawfully in US before 16th birthday
2. Had not yet turned 31 as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012

**Implementation**:
- `Age at arrival < 16`: Calculate arrival age using `YEAR - YRSUSA1` or `YRIMMIG` and `BIRTHYR`
- `Age < 31 on June 15, 2012`: Born in 1982 or later (conservative: BIRTHYR >= 1982)
- `In US since 2007`: `YRIMMIG <= 2007` or `YRSUSA1 >= (YEAR - 2007)`

### 3. Treatment Definition

**Treatment variable**: `daca_eligible = 1` if person meets all DACA eligibility criteria

**Control group**: Mexican-born, Hispanic-Mexican non-citizens who do NOT meet DACA eligibility (e.g., arrived after age 16, or were over 31 in 2012)

### 4. Outcome Variable

**Full-time employment**: `fulltime = 1` if `UHRSWORK >= 35`

**Rationale**: Instructions define full-time as "usually working 35 hours per week or more"

### 5. Identification Strategy

**Approach**: Difference-in-Differences (DiD)

**Pre-period**: 2006-2011 (before DACA announcement)
**Post-period**: 2013-2016 (after DACA implementation)
**Excluded**: 2012 (cannot distinguish before/after DACA within this year per instructions)

**Model**:
```
fulltime = β0 + β1*eligible + β2*post + β3*(eligible*post) + controls + ε
```

Where β3 is the DiD estimate of DACA's effect on full-time employment.

### 6. Control Variables

- Age (AGE)
- Sex (SEX)
- Marital status (MARST)
- Education (EDUC)
- State fixed effects (STATEFIP)
- Year fixed effects (YEAR)

### 7. Working-Age Sample Restriction

**Decision**: Restrict to working-age population (18-64)

**Rationale**: Standard labor economics practice; avoids including those too young or retired.

### 8. Survey Weights

**Decision**: Use person weights (PERWT) for population estimates

**Rationale**: ACS is a weighted survey; weights necessary for representative estimates.

---

## Analysis Steps Executed

1. Load ACS data (2006-2016)
2. Filter to Hispanic-Mexican, Mexican-born, non-citizen sample
3. Construct DACA eligibility indicator
4. Construct full-time employment outcome
5. Generate descriptive statistics
6. Run DiD regression with controls
7. Conduct robustness checks
8. Generate tables and figures

---

## Execution Log

### Session Start
- Loaded replication_instructions.docx
- Examined data dictionary (acs_data_dict.txt)
- Identified key variables: YEAR, HISPAN, BPL, CITIZEN, YRIMMIG, BIRTHYR, UHRSWORK, EMPSTAT

### Data Loading
- Loaded data.csv (33,851,424 rows)
- Data spans 2006-2016 ACS

### Sample Selection (Sequential Filtering)
```
Initial observations:                 33,851,424
After excluding 2012:                 30,738,394
After Hispanic-Mexican (HISPAN=1):     2,663,503
After Mexico birthplace (BPL=200):       898,879
After non-citizen (CITIZEN=3):           636,722
After working age (18-64):               547,614
After valid immigration year:            547,614  [FINAL]
```

### Variable Construction
- DACA Eligible: 69,244 (12.6%)
- Non-Eligible: 478,370 (87.4%)
- Eligibility defined as: age_at_immig < 16 AND BIRTHYR >= 1982 AND YRIMMIG <= 2007

### Descriptive Statistics Generated
- Summary statistics by eligibility group
- Full-time employment rates by year and eligibility
- Pre/post period comparison

### Regression Models Estimated

**Model 1 (Basic DiD):**
- DiD coefficient: 0.0727
- SE: 0.0049
- p-value: <0.001

**Model 2 (With demographic controls):**
- DiD coefficient: 0.0593
- SE: 0.0047
- p-value: <0.001

**Model 3 (With controls + state/year FE) [PREFERRED]:**
- DiD coefficient: 0.0515
- SE: 0.0046
- 95% CI: [0.0424, 0.0606]
- p-value: <0.001
- N = 547,614

### Robustness Checks

1. **Alternative control group (adults only):** DiD = 0.0573 (SE = 0.0051)
2. **Placebo test (pre-period):** DiD = 0.0139 (SE = 0.0063, p = 0.027)
3. **Employment (extensive margin):** DiD = 0.0684 (SE = 0.0045)
4. **By gender:**
   - Male: DiD = 0.0552 (SE = 0.0060)
   - Female: DiD = 0.0625 (SE = 0.0071)

### Output Files Generated
- results_summary.csv
- yearly_means.csv
- sample_characteristics.csv
- replication_report_24.tex
- replication_report_24.pdf (21 pages)
- run_log_24.md (this file)

---

## Key Findings Summary

**Preferred Estimate:** DACA eligibility increased full-time employment probability by **5.15 percentage points** (95% CI: 4.24-6.06 pp), statistically significant at the 1% level.

**Interpretation:** Among Mexican-born, Hispanic-Mexican non-citizens, those eligible for DACA experienced a 5.2 percentage point larger increase in full-time employment compared to non-eligible individuals after DACA implementation in 2012. This represents approximately a 10% increase relative to the pre-period eligible group mean of 50.5%.

---

## Files in Replication Package

1. `analysis.py` - Main analysis script
2. `data/data.csv` - ACS microdata (2006-2016)
3. `data/acs_data_dict.txt` - Variable codebook
4. `results_summary.csv` - Regression results table
5. `yearly_means.csv` - Year-by-year employment means
6. `sample_characteristics.csv` - Descriptive statistics
7. `replication_report_24.tex` - LaTeX source for report
8. `replication_report_24.pdf` - Final report (21 pages)
9. `run_log_24.md` - This documentation file

