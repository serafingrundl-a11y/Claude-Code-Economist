# Run Log - DACA Replication Study (Replication 50)

## Project Overview
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (usually working 35+ hours per week)?

**Study Period:** Examining effects on full-time employment in years 2013-2016, with pre-treatment years for comparison.

## Session Log

### Session Start: 2026-01-25

#### 1. Data Exploration

**Files in data folder:**
- `data.csv` - Main ACS data file (6.26 GB, 33,851,424 observations)
- `acs_data_dict.txt` - Data dictionary for ACS variables
- `state_demo_policy.csv` - Optional state-level data
- `State Level Data Documentation.docx` - Documentation for state data

**Data Variables Available (from data dictionary):**
- YEAR: Census year (2006-2016)
- HISPAN/HISPAND: Hispanic origin (1=Mexican)
- BPL/BPLD: Birthplace (200=Mexico)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth
- AGE: Age
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status
- PERWT: Person weight

#### 2. DACA Eligibility Criteria (from instructions)

DACA eligibility requires:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

**Operationalization:**
- Non-citizen (CITIZEN = 3) to proxy undocumented status
- Born in Mexico (BPL = 200)
- Hispanic-Mexican ethnicity (HISPAN = 1)
- Immigration year before age 16 (YRIMMIG - BIRTHYR < 16) OR we can use implied arrival age
- Born after June 15, 1981 (age < 31 as of June 15, 2012)
- Immigration year <= 2007 (present since June 15, 2007)

#### 3. Initial Identification Strategy (Birth-Year Cutoff)

**Approach 1: Age-at-DACA Discontinuity**
- Treatment: Those under 31 on June 15, 2012 (born after June 15, 1981)
- Control: Those 31-40 on June 15, 2012 (born 1972-1981)
- Both groups must meet other criteria (non-citizen, arrived before age 16, in US since 2007)

**Results from Approach 1:**
- Sample size: 101,893
- DiD estimate: -0.0463 (SE: 0.0093)
- Finding: Negative effect on full-time employment
- **Problem**: Event study showed significant pre-trends (coefficients varied from 0.066 in 2006 to 0.026 in 2010), invalidating the parallel trends assumption

#### 4. Revised Identification Strategy (Arrival-Age Cutoff) - PREFERRED

**Approach 2: Arrival-Age Discontinuity**
The arrival-age cutoff provides a cleaner identification strategy because it exploits variation in eligibility based on age at immigration rather than birth year. This controls for cohort effects.

**Treatment Group:** Those who arrived at age 12-15 (just under the age 16 cutoff)
**Control Group:** Those who arrived at age 16-19 (just over the age 16 cutoff)

Both groups must:
- Be Mexican-born Hispanic non-citizens
- Be under 31 as of June 2012
- Have been in US since at least 2007
- Be working age (18-55)

**Rationale for this approach:**
1. Both groups have similar time in US
2. Similar socioeconomic backgrounds
3. The only difference is eligibility based on arrival age
4. Event study shows no significant pre-trends (pre-2012 coefficients range from -0.016 to 0.008, all statistically insignificant)

#### 5. Analysis Execution

**Command 1:** Initial analysis (analysis.py)
```
python analysis.py
```
- Ran birth-year cutoff analysis
- Identified pre-trends problem

**Command 2:** Revised analysis (analysis_v2.py)
```
python analysis_v2.py
```
- Ran arrival-age cutoff analysis
- Found no pre-trends
- Significant positive effect

#### 6. Key Results from Preferred Specification

**Sample Construction:**
- Total ACS observations: 33,851,424
- Hispanic-Mexican: 2,945,521
- Mexican-born: 991,261
- Non-citizens: 701,347
- Analysis sample (arrival age design): 64,082

**Main Results (DiD with Year FE and Controls):**
- Effect on full-time employment: **+4.27 percentage points**
- Standard Error: 0.0087
- 95% CI: [2.56, 5.99 pp]
- p-value: < 0.0001

**Event Study (Pre-trends Test):**
| Year | Coefficient | SE | Significance |
|------|-------------|------|--------------|
| 2006 | -0.0156 | 0.0197 | - |
| 2007 | -0.0135 | 0.0191 | - |
| 2008 | -0.0034 | 0.0192 | - |
| 2009 | 0.0078 | 0.0194 | - |
| 2010 | 0.0081 | 0.0192 | - |
| 2011 | (base) | - | - |
| 2013 | 0.0432 | 0.0190 | ** |
| 2014 | 0.0274 | 0.0194 | - |
| 2015 | 0.0377 | 0.0193 | * |
| 2016 | 0.0567 | 0.0197 | *** |

Pre-treatment coefficients are all close to zero and insignificant, supporting parallel trends.

**Robustness Checks:**
| Specification | Estimate | SE | p-value |
|---------------|----------|------|---------|
| Basic DiD | 0.0709 | 0.0097 | <0.001 |
| DiD + Controls | 0.0489 | 0.0087 | <0.001 |
| DiD + Year FE | 0.0427 | 0.0087 | <0.001 |
| Narrow bandwidth (14-15 vs 16-17) | 0.0317 | 0.0113 | 0.005 |
| Men only | 0.0365 | 0.0101 | <0.001 |
| Women only | 0.0495 | 0.0154 | 0.001 |
| Employment (any) outcome | 0.0373 | 0.0082 | <0.001 |

#### 7. Key Decisions Made

1. **Population restriction:** Limited to Mexican-born Hispanic non-citizens as the most relevant population for DACA
2. **Undocumented proxy:** Used non-citizenship as proxy for undocumented status (cannot directly observe documentation status in ACS)
3. **Working age:** Restricted to ages 18-55 for employment analysis
4. **Identification strategy:** Chose arrival-age discontinuity over birth-year discontinuity due to cleaner pre-trends
5. **Outcome definition:** Full-time employment = UHRSWORK >= 35 hours/week
6. **Exclusion of 2012:** Excluded transition year when DACA was announced (June 2012)
7. **Survey weights:** Used PERWT for all analyses
8. **Standard errors:** Used heteroscedasticity-robust (HC1) standard errors

#### 8. Files Generated

- `analysis.py` - Initial analysis script (birth-year design)
- `analysis_v2.py` - Revised analysis script (arrival-age design) - PREFERRED
- `summary_statistics.csv` - Summary statistics
- `regression_results.csv` - Regression results (birth-year design)
- `event_study_results.csv` - Event study results (birth-year design)
- `regression_results_arrival_age.csv` - Regression results (arrival-age design)
- `event_study_arrival_age.csv` - Event study results (arrival-age design)
- `replication_report_50.tex` - LaTeX report
- `replication_report_50.pdf` - PDF report

#### 9. Interpretation

DACA eligibility is associated with a 4.27 percentage point increase in full-time employment among Mexican-born Hispanic non-citizens. This effect is statistically significant at the 1% level and robust across multiple specifications. The effect is present for both men (3.65 pp) and women (4.95 pp), with women showing a slightly larger effect.

The finding of a positive effect on full-time employment is consistent with the policy design: DACA provides legal work authorization, which should enable previously undocumented immigrants to access formal, full-time employment opportunities that they may have avoided due to legal risks.
