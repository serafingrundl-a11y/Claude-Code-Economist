# Run Log - Replication 94

## Project Overview
Replication study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Date Started
2026-01-26

---

## Step 1: Review Instructions and Data Dictionary
- Read replication_instructions.docx
- Research question: Effect of DACA eligibility (treatment) on probability of full-time employment (>=35 hours/week)
- Treatment group: Ages 26-30 at DACA implementation (June 15, 2012)
- Control group: Ages 31-35 at DACA implementation (would have been eligible if not for age)
- DACA implemented: June 15, 2012
- Post-treatment period: 2013-2016
- Design: Difference-in-differences using repeated cross-sections

### Key Variables from ACS Data Dictionary:
- YEAR: Census year (2006-2016)
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- HISPAN/HISPAND: Hispanic origin (1=Mexican for HISPAN, 100-107 for HISPAND detailed)
- BPL/BPLD: Birthplace (200=Mexico for BPL, 20000 for BPLD)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week (>=35 for full-time)
- EMPSTAT: Employment status (1=Employed)
- PERWT: Person weight for survey weighting
- AGE: Age at time of survey
- SEX: Sex (1=Male, 2=Female)
- EDUC/EDUCD: Education level
- MARST: Marital status
- STATEFIP: State FIPS code

### DACA Eligibility Criteria (per instructions):
1. Arrived unlawfully in US before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status
5. Assume non-citizens without immigration papers are undocumented

### Sample Selection Strategy:
- Hispanic-Mexican ethnicity (HISPAN=1 or HISPAND in 100-107)
- Born in Mexico (BPL=200)
- Non-citizen (CITIZEN=3)
- Immigrated before age 16 (need to calculate from YRIMMIG and BIRTHYR)
- Immigrated by 2007 (lived continuously since June 15, 2007)

### Age Groups for Analysis:
- Treatment: Born 1982-1986 (ages 26-30 as of June 15, 2012)
- Control: Born 1977-1981 (ages 31-35 as of June 15, 2012)

---

## Step 2: Load and Explore Data

### Data Loading
- Total rows in ACS data: 33,851,424
- Years covered: 2006-2016
- File: data/data.csv

### Sample Filtering Steps (with row counts):
1. Full ACS sample: 33,851,424
2. After filtering Hispanic-Mexican (HISPAN=1): 2,945,521
3. After filtering born in Mexico (BPL=200): 991,261
4. After filtering non-citizens (CITIZEN=3): 701,347
5. After filtering valid immigration year (YRIMMIG>0): 701,347
6. After filtering arrived before age 16: 205,327
7. After filtering immigrated by 2007: 195,023
8. After filtering to age groups 26-30 and 31-35: 49,019
9. After excluding 2012 (implementation year): 44,725

### Final Sample Characteristics:
- Total observations: 44,725
- Treatment group (ages 26-30 at DACA): 26,591
- Control group (ages 31-35 at DACA): 18,134
- Pre-period (2006-2011): 29,326
- Post-period (2013-2016): 15,399

---

## Step 3: Key Decisions Made

### Decision 1: Sample Definition
- Used HISPAN=1 (Mexican) rather than HISPAND detailed codes
- Rationale: Instructions specify "Hispanic-Mexican" which corresponds to the general version

### Decision 2: Identifying Undocumented Status
- Used CITIZEN=3 (Not a citizen) as proxy for undocumented
- Rationale: Per instructions "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented"
- Note: This includes all non-citizens, not just undocumented, which may introduce measurement error

### Decision 3: Age at DACA Calculation
- Calculated age_at_daca = 2012 - BIRTHYR
- Rationale: Simple approach using year of birth, DACA implemented mid-2012
- Treatment: ages 26-30 at DACA (born 1982-1986)
- Control: ages 31-35 at DACA (born 1977-1981)

### Decision 4: Arrived Before Age 16
- Calculated age_at_immigration = YRIMMIG - BIRTHYR
- Required age_at_immigration < 16
- Rationale: DACA eligibility criterion

### Decision 5: Continuous Presence Since 2007
- Required YRIMMIG <= 2007
- Rationale: DACA requires presence since June 15, 2007

### Decision 6: Exclusion of 2012
- Excluded 2012 from analysis
- Rationale: DACA implemented mid-year (June 15), so 2012 observations are mixed pre/post

### Decision 7: Full-time Employment Definition
- Defined as UHRSWORK >= 35
- Rationale: Standard BLS definition of full-time work

### Decision 8: Control Variables
- Included: female (SEX=2), married (MARST=1), educ_hs (EDUC>=6)
- Rationale: Standard demographic controls likely correlated with employment

### Decision 9: Fixed Effects
- Included year fixed effects (reference: 2006)
- Included state fixed effects (reference: first state alphabetically)
- Rationale: Control for aggregate time trends and state-level differences

### Decision 10: Standard Errors
- Used robust (heteroskedasticity-consistent) standard errors (HC1)
- Rationale: Standard practice for cross-sectional data

---

## Step 4: Analysis Results

### Simple DiD Calculation:
| Group | Pre-Period | Post-Period | Change |
|-------|-----------|-------------|--------|
| Treatment (26-30) | 0.6111 | 0.6339 | +0.0228 |
| Control (31-35) | 0.6431 | 0.6108 | -0.0323 |
| **Difference-in-Differences** | | | **0.0551** |

### Regression Results Summary:

| Model | DiD Estimate | Std. Error | 95% CI | p-value |
|-------|-------------|------------|--------|---------|
| 1. Basic DiD | 0.0551 | 0.0098 | [0.036, 0.074] | <0.001 |
| 2. With Controls | 0.0485 | 0.0091 | [0.031, 0.066] | <0.001 |
| 3. Year FE | 0.0485 | 0.0091 | [0.031, 0.066] | <0.001 |
| 4. Year + State FE | 0.0477 | 0.0091 | [0.030, 0.066] | <0.001 |
| 5. Weighted (PERWT) | 0.0478 | 0.0105 | [0.027, 0.069] | <0.001 |

### Pre-trends Analysis:
Year-specific treatment effects relative to 2006:
- 2007: 0.0057 (p=0.74) - Not significant
- 2008: 0.0337 (p=0.06) - Marginally significant
- 2009: 0.0259 (p=0.16) - Not significant
- 2010: 0.0231 (p=0.21) - Not significant
- 2011: 0.0308 (p=0.09) - Not significant
- 2013: 0.0622 (p=0.001) - Significant
- 2014: 0.0599 (p=0.002) - Significant
- 2015: 0.0677 (p<0.001) - Significant
- 2016: 0.0834 (p<0.001) - Significant

Pre-trend coefficients are generally small and not individually significant, though there is some suggestive evidence of a modest upward trend pre-2012.

---

## Step 5: Preferred Estimate

**Model 4: Year and State Fixed Effects (unweighted)**

- DiD Estimate: **0.0477**
- Standard Error: 0.0091
- 95% CI: [0.0299, 0.0655]
- p-value: < 0.001
- Sample Size: 44,725

**Interpretation:** DACA eligibility is associated with a 4.77 percentage point increase in the probability of full-time employment among eligible Hispanic-Mexican, Mexican-born non-citizens.

---

## Step 6: Output Files Generated

1. `output/regression_results.csv` - All regression model results
2. `output/sample_statistics.csv` - Sample sizes and means by group/period
3. `output/pretrend_analysis.csv` - Year-specific treatment effects
4. `output/detailed_summary_stats.csv` - Full summary statistics
5. `output/final_summary.txt` - Preferred estimate summary
6. `output/figure1_trends.png` - Employment trends by group
7. `output/figure2_eventstudy.png` - Event study plot
8. `output/figure3_did.png` - DiD visualization

---

## Step 7: LaTeX Report Compilation
- Created replication_report_94.tex
- Compiled to replication_report_94.pdf using pdflatex

---

## Notes and Caveats

1. **Measurement Error in Undocumented Status**: Using non-citizen status as proxy likely includes some legal permanent residents and visa holders, potentially attenuating the treatment effect estimate.

2. **Age Calculation Imprecision**: Using only birth year (not quarter) introduces some classification error at age boundaries.

3. **Pre-trends**: While not individually significant, pre-period coefficients show a mild upward trend, which could suggest violations of parallel trends assumption.

4. **Sample Restriction**: The sample is restricted to those who arrived before age 16 and were present by 2007, which may not capture all DACA-eligible individuals.

5. **Full-time Definition**: Using usual hours worked captures intensive margin but not extensive margin (employed vs not employed).

---

## Session Log

- 2026-01-26: Started replication
- Read instructions and data dictionary
- Developed sample selection criteria
- Ran analysis script (analysis.py)
- Generated visualizations
- Created LaTeX report
