# DACA Replication Study - Run Log

## Session Started: 2026-01-26

### Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

### Study Design
- **Treatment Group**: Ages 26-30 at June 15, 2012 (birth years 1982-1986)
- **Control Group**: Ages 31-35 at June 15, 2012 (birth years 1977-1981)
- **Pre-treatment Period**: 2006-2011
- **Post-treatment Period**: 2013-2016
- **Method**: Difference-in-Differences (DiD)

---

## Step 1: Data Exploration

### Files Available:
- `data/data.csv` - Main ACS data file (6.26 GB)
- `data/acs_data_dict.txt` - Data dictionary for ACS variables
- `data/state_demo_policy.csv` - Optional state-level data
- `data/State Level Data Documentation.docx` - State data documentation

### Key Variables Identified:
- `YEAR` - Survey year (2006-2016)
- `BIRTHYR` - Birth year
- `BIRTHQTR` - Birth quarter (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- `HISPAN` - Hispanic origin (1=Mexican)
- `BPL` - Birthplace (200=Mexico)
- `CITIZEN` - Citizenship status (3=Not a citizen)
- `YRIMMIG` - Year of immigration
- `UHRSWORK` - Usual hours worked per week (>=35 = full-time)
- `PERWT` - Person weight for population estimates
- `AGE` - Age at survey time
- `SEX` - Sex (1=Male, 2=Female)
- `EDUC` - Education level
- `EMPSTAT` - Employment status (1=Employed)

### DACA Eligibility Criteria:
1. Hispanic-Mexican ethnicity (HISPAN=1)
2. Born in Mexico (BPL=200)
3. Not a citizen (CITIZEN=3)
4. Arrived in US before age 16 (YRIMMIG - BIRTHYR < 16)
5. In US by June 15, 2007 (YRIMMIG <= 2007)
6. No lawful status at June 15, 2012

### Age Group Definition:
- For June 15, 2012 reference:
  - Treatment: Ages 26-30 (born 1982-1986)
  - Control: Ages 31-35 (born 1977-1981)
  - The 31+ cutoff excluded from DACA eligibility

---

## Step 2: Data Processing Commands

### Command: Run analysis.py
```bash
cd "C:/Users/seraf/DACA Results Task 2/replication_23" && python analysis.py
```

### Data Processing Steps:
1. **Initial Load**: Read ACS data 2006-2016 (over 30 million observations)
2. **Filter Hispanic-Mexican, Mexico-born**: 991,261 observations
3. **Filter non-citizens (CITIZEN=3)**: 701,347 observations
4. **Filter arrived before age 16**: 205,327 observations
5. **Filter arrived by 2007**: 195,023 observations
6. **Select birth year cohorts (1977-1986)**: 49,019 observations
7. **Exclude 2012**: 44,725 observations (final sample)

---

## Step 3: Key Decisions Made

### Decision 1: DACA Eligibility Operationalization
- **Choice**: Use CITIZEN=3 (not a citizen) as proxy for undocumented status
- **Rationale**: Per instructions, "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes." Since we cannot observe documentation status directly, non-citizens are the best available proxy.

### Decision 2: Age at Policy Implementation
- **Choice**: Use birth year cohorts (1982-1986 for treatment, 1977-1981 for control)
- **Rationale**: DACA required individuals to be under 31 as of June 15, 2012. Birth years provide cleaner identification than age at survey since they are fixed over time.

### Decision 3: Arrival Before Age 16
- **Choice**: Calculate age at immigration as (YRIMMIG - BIRTHYR) < 16
- **Rationale**: DACA required arrival before 16th birthday. This is a conservative implementation using available data.

### Decision 4: Continuous Presence Requirement
- **Choice**: Use YRIMMIG <= 2007 for continuous presence since June 15, 2007
- **Rationale**: This ensures 5+ years of continuous presence as required by DACA. Cannot directly observe continuous presence, so immigration year is best available proxy.

### Decision 5: Exclusion of 2012
- **Choice**: Exclude year 2012 from analysis
- **Rationale**: DACA was implemented on June 15, 2012 (mid-year). ACS data does not identify survey month, so 2012 observations cannot be cleanly assigned to pre or post periods.

### Decision 6: Outcome Variable Definition
- **Choice**: Full-time employment = UHRSWORK >= 35
- **Rationale**: Per instructions, "usually working 35 hours per week or more."

### Decision 7: Control Variables
- **Choice**: Include age, age-squared, female, married status, education categories
- **Rationale**: These are standard demographic controls that may affect employment. Education captures human capital; marital status may affect labor supply decisions.

### Decision 8: Preferred Specification
- **Choice**: Model 3 (demographics + education controls) without state fixed effects
- **Rationale**: Provides good covariate balance while maintaining transparency. State fixed effects (Model 4) provide similar results but add complexity.

---

## Step 4: Analysis Results

### Sample Sizes (Unweighted):
| Group | Pre-Period | Post-Period |
|-------|------------|-------------|
| Treatment | 17,410 | 9,181 |
| Control | 11,916 | 6,218 |

### Full-Time Employment Rates (Weighted):
| Group | Pre-Period | Post-Period | Difference |
|-------|------------|-------------|------------|
| Treatment | 62.53% | 65.80% | +3.27 pp |
| Control | 67.05% | 64.12% | -2.93 pp |

### Simple DiD Estimate: 6.20 percentage points

### Regression Results:
| Model | DiD Coefficient | Std. Error | p-value |
|-------|-----------------|------------|---------|
| Basic DiD | 0.0620 | 0.0116 | <0.001 |
| + Demographics | 0.0657 | 0.0148 | <0.001 |
| + Education | 0.0658 | 0.0148 | <0.001 |
| + State FE | 0.0652 | 0.0148 | <0.001 |

### Preferred Estimate:
- **Effect Size**: 0.0658 (6.58 percentage points)
- **Standard Error**: 0.0148
- **95% CI**: [0.0368, 0.0949]
- **p-value**: <0.0001
- **Sample Size**: 44,725

---

## Step 5: Output Files Generated

1. `summary_statistics.csv` - Descriptive statistics by group and period
2. `regression_results.csv` - All regression model results
3. `final_results.csv` - Preferred estimate details
4. `event_study.png` - Event study plot
5. `parallel_trends.png` - Parallel trends visualization
6. `replication_report_23.tex` - LaTeX report
7. `replication_report_23.pdf` - Final PDF report

---

## Step 6: Robustness Checks

1. **Conditional on Employment**: DiD = 0.0450 (SE: 0.0150) - Effect on intensive margin
2. **Employment (Extensive Margin)**: DiD = 0.0579 (SE: 0.0141) - Effect on employment
3. **Males Only**: DiD = 0.0667 (SE: 0.0179)
4. **Females Only**: DiD = 0.0555 (SE: 0.0243)
5. **Tighter Bandwidth (27-29 vs 32-34)**: DiD = 0.0923 (SE: 0.0212)

All robustness checks show consistent positive effects, supporting the main finding.

---

## Conclusion

DACA eligibility is associated with a statistically significant 6.58 percentage point increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born non-citizens who meet the eligibility criteria. This effect is robust to various specifications and subgroup analyses.

