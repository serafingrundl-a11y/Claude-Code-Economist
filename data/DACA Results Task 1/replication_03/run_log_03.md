# Replication Run Log - Replication 03

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Data Overview
- **Source**: American Community Survey (ACS) 2006-2016 via IPUMS USA
- **Sample size**: 33,851,424 observations (full dataset)
- **Years**: 2006-2016 (DACA implemented June 15, 2012)

## Key Decisions and Steps

### Step 1: Data Understanding
- Read data dictionary from `acs_data_dict.txt`
- Identified key variables:
  - **YEAR**: Survey year (2006-2016)
  - **HISPAN**: Hispanic origin (1 = Mexican)
  - **BPL/BPLD**: Birthplace (200 = Mexico)
  - **CITIZEN**: Citizenship status (3 = Not a citizen)
  - **YRIMMIG**: Year of immigration
  - **BIRTHYR**: Year of birth
  - **AGE**: Age at survey
  - **UHRSWORK**: Usual hours worked per week
  - **EMPSTAT**: Employment status
  - **PERWT**: Person weight

### Step 2: DACA Eligibility Criteria Definition
Based on instructions, DACA eligibility requires:
1. Arrived in the US before 16th birthday
2. Had not turned 31 by June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007 (arrived by 2007)
4. Present in the US on June 15, 2012 and not a citizen

**Operationalization for ACS data:**
- HISPAN == 1 (Mexican Hispanic)
- BPL == 200 (Born in Mexico)
- CITIZEN == 3 (Not a citizen - proxy for undocumented)
- Age at arrival < 16: (YRIMMIG - BIRTHYR) < 16
- BIRTHYR > 1981 (not yet 31 on June 15, 2012)
- YRIMMIG <= 2007 (continuous residence since 2007)

### Step 3: Identification Strategy
Using a **Difference-in-Differences (DiD)** approach:
- **Treatment group**: DACA-eligible Mexican-born Hispanic non-citizens
- **Control group**: Non-eligible Mexican-born Hispanic non-citizens (arrived as adults OR born before 1981 OR arrived after 2007)
- **Pre-period**: 2006-2011 (before DACA)
- **Post-period**: 2013-2016 (after DACA - excluding 2012 as partial treatment year)

### Step 4: Outcome Variable
- **Full-time employment**: UHRSWORK >= 35 (binary indicator)
- Restricted to working-age population (18-64 years old)

### Step 5: Analysis Plan
1. Restrict sample to Mexican-born Hispanic non-citizens
2. Create DACA eligibility indicator
3. Create post-treatment indicator (YEAR >= 2013)
4. Run DiD regression with controls for age, education, sex, state, year FE
5. Use survey weights (PERWT)
6. Use heteroskedasticity-robust (HC1) standard errors

---

## Execution Log

### Data Loading and Filtering
- Used chunked reading to handle 33.8M row dataset efficiently
- Applied filters sequentially:
  - BPL == 200 (Mexico-born): 1,020,945 obs
  - HISPAN == 1 (Hispanic-Mexican): 991,261 obs
  - CITIZEN == 3 (non-citizen): 701,347 obs
  - Age 18-64: 603,425 obs
  - Excluded 2012: **547,614 observations** (final sample)

### Sample Composition
- **DACA-eligible**: 69,244 (12.6%)
- **Not DACA-eligible**: 478,370 (87.4%)
- **Pre-period (2006-2011)**: 336,493 observations
- **Post-period (2013-2016)**: 211,121 observations

### Descriptive Statistics
- Full-time employment rate: 58.7% overall
- DACA-eligible: 52.4% full-time employed
- Not eligible: 59.6% full-time employed

### Weighted Full-Time Employment Rates by Group and Period
| Group | Pre (2006-2011) | Post (2013-2016) | Change |
|-------|-----------------|------------------|--------|
| Not Eligible | 62.85% | 60.38% | -2.47 pp |
| DACA Eligible | 51.99% | 56.80% | +4.81 pp |

**Simple DiD**: 7.27 percentage points

### Regression Results

#### Model 1: Basic DiD (no controls)
- **DiD Coefficient**: 0.0727 (SE = 0.0049)
- **95% CI**: [0.0630, 0.0823]
- **p-value**: < 0.0001

#### Model 2: DiD with demographic controls
- **DiD Coefficient**: 0.0293 (SE = 0.0046)
- **95% CI**: [0.0202, 0.0383]
- **p-value**: < 0.0001
- Controls: age, age², female, married, education indicators

#### Model 3: DiD with year fixed effects (PREFERRED)
- **DiD Coefficient**: 0.0203 (SE = 0.0046)
- **95% CI**: [0.0113, 0.0294]
- **p-value**: < 0.0001
- Controls: demographic + year FE

#### Model 4: DiD with state and year fixed effects
- **DiD Coefficient**: 0.0196 (SE = 0.0046)
- **95% CI**: [0.0106, 0.0286]
- **p-value**: < 0.0001
- Controls: demographic + year FE + state FE

### Robustness Checks

#### Any Employment as Outcome
- **DiD Coefficient**: 0.0288 (SE = 0.0044)
- **95% CI**: [0.0201, 0.0375]

#### Males Only
- **DiD Coefficient**: 0.0169 (SE = 0.0059)
- **95% CI**: [0.0054, 0.0285]
- N = 296,109

#### Females Only
- **DiD Coefficient**: 0.0171 (SE = 0.0070)
- **95% CI**: [0.0034, 0.0309]
- N = 251,505

#### Placebo Test (fake treatment in 2009)
- **DiD Coefficient**: -0.0011 (SE = 0.0062)
- **p-value**: 0.861
- Result: Statistically insignificant, supporting parallel trends assumption

### Event Study Results (relative to 2011)
| Year | Coefficient | SE | Significance |
|------|-------------|-----|--------------|
| 2006 | 0.0131 | 0.0111 | |
| 2007 | 0.0089 | 0.0106 | |
| 2008 | 0.0198 | 0.0106 | * |
| 2009 | 0.0204 | 0.0105 | * |
| 2010 | 0.0184 | 0.0102 | * |
| 2011 | [Reference] | | |
| 2013 | 0.0164 | 0.0100 | * |
| 2014 | 0.0291 | 0.0100 | *** |
| 2015 | 0.0442 | 0.0099 | *** |
| 2016 | 0.0443 | 0.0101 | *** |

---

## Main Finding

**DACA eligibility is associated with a 2.0 percentage point increase in the probability of full-time employment** (95% CI: 1.1 to 2.9 pp, p < 0.0001).

This represents approximately a **3.9% relative increase** from the pre-treatment mean of 52.0% for DACA-eligible individuals.

---

## Output Files Generated
- `analysis.py` - Python analysis script
- `results.json` - Numerical results saved for reporting
- `replication_report_03.tex` - LaTeX source for the report
- `replication_report_03.pdf` - Final PDF report (25 pages)
- `run_log_03.md` - This log file

---

## Technical Notes
- Analysis performed using Python 3.14 with pandas, numpy, statsmodels
- Chunked data loading used to handle memory constraints with 33.8M row dataset
- HC1 (heteroskedasticity-robust) standard errors used throughout
- All regressions weighted by PERWT (person weights)
