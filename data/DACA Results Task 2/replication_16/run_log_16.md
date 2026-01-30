# Replication Run Log - DACA and Full-Time Employment Analysis

## Project Overview
Replicating the analysis of the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Identification Strategy
- **Treatment Group**: Ages 26-30 at time of DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at time of DACA implementation (otherwise eligible)
- **Method**: Difference-in-Differences comparing treated vs control before and after 2012

---

## Session Log

### Data Exploration

**Date**: Session start

**Files Found in data folder**:
- `data.csv` - Main ACS data file (~33.85 million observations)
- `acs_data_dict.txt` - Data dictionary for ACS variables
- `state_demo_policy.csv` - Optional state-level policy data
- `State Level Data Documentation.docx` - Documentation for state-level data

**Key Variables Identified**:
- `YEAR`: Survey year (2006-2016 available)
- `BIRTHYR`: Birth year of individual
- `BIRTHQTR`: Birth quarter
- `HISPAN`: Hispanic origin (1 = Mexican)
- `BPL`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `UHRSWORK`: Usual hours worked per week (full-time = 35+ hours)
- `PERWT`: Person weight for population estimates

### Sample Selection Criteria

**DACA Eligibility Criteria** (from instructions):
1. Arrived unlawfully in US before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 and no lawful status

**Operationalization**:
- Hispanic-Mexican ethnicity: `HISPAN == 1`
- Born in Mexico: `BPL == 200`
- Not a citizen: `CITIZEN == 3`
- Arrived before age 16: `YRIMMIG - BIRTHYR < 16`
- In US since at least 2007: `YRIMMIG <= 2007`

**Age Groups** (as of June 15, 2012):
- Treatment: Ages 26-30 on June 15, 2012 → Born 1982-1986
- Control: Ages 31-35 on June 15, 2012 → Born 1977-1981

**Time Periods**:
- Pre-treatment: 2006-2011
- Treatment year (excluded): 2012 (cannot distinguish before/after DACA)
- Post-treatment: 2013-2016

---

### Analysis Steps

#### Step 1: Data Loading
- Loaded full ACS data: 33,851,424 observations
- Data spans years 2006-2016

#### Step 2: Sample Selection
Applied sequential filters to identify DACA-eligible population:
1. Hispanic-Mexican born in Mexico: 991,261 obs
2. Non-citizens (CITIZEN == 3): 701,347 obs
3. Valid immigration year: 701,347 obs
4. Arrived before age 16: 205,327 obs
5. In US since 2007 (YRIMMIG <= 2007): 195,023 obs

#### Step 3: Define Treatment and Control
- Treatment group (born 1982-1986, ages 26-30 in 2012): 26,591 obs
- Control group (born 1977-1981, ages 31-35 in 2012): 18,134 obs
- After excluding 2012: 44,725 total observations

#### Step 4: Outcome Definition
- Full-time employment: UHRSWORK >= 35 hours per week
- Coded as binary indicator (1 = full-time, 0 = not full-time)

---

### Key Results

#### Descriptive Statistics
| Group | N | Weighted N | Full-time (%) | Female (%) | HS+ (%) | Mean Age |
|-------|---|------------|---------------|------------|---------|----------|
| Control | 18,134 | 2,530,790 | 66.1% | 43.9% | 54.3% | 31.4 |
| Treatment | 26,591 | 3,674,965 | 63.7% | 44.0% | 62.2% | 26.3 |

#### Weighted Full-Time Employment Rates by Period
| Group | Pre (2006-2011) | Post (2013-2016) | Difference |
|-------|-----------------|------------------|------------|
| Control | 67.1% | 64.1% | -2.9 pp |
| Treatment | 62.5% | 65.8% | +3.3 pp |

**Simple DiD**: (+3.3) - (-2.9) = +6.2 percentage points

#### Main Regression Results
| Model | DiD Coefficient | SE | Description |
|-------|-----------------|-----|-------------|
| 1 | 0.0551 | 0.0098 | Simple OLS |
| 2 | 0.0620 | 0.0097 | Weighted (WLS) |
| 3 | 0.0483 | 0.0089 | With demographics |
| 4 | 0.0610 | 0.0096 | Year FE |
| 5 | 0.0596 | 0.0096 | Year + State FE |
| 6 | 0.0464 | 0.0089 | Full specification |
| 7 | 0.0464 | 0.0113 | Clustered SE (preferred) |

**Preferred Estimate (Model 7)**:
- Effect: 4.64 percentage points
- SE (clustered by state): 0.0113
- 95% CI: [2.42%, 6.86%]
- p-value: < 0.001

#### Event Study (Pre-trends Check)
Coefficients relative to 2011 (reference year):
| Year | Coefficient | SE |
|------|-------------|-----|
| 2006 | -0.004 | 0.019 |
| 2007 | -0.011 | 0.020 |
| 2008 | 0.017 | 0.020 |
| 2009 | 0.019 | 0.020 |
| 2010 | 0.015 | 0.020 |
| 2013 | 0.059** | 0.021 |
| 2014 | 0.067** | 0.021 |
| 2015 | 0.041* | 0.021 |
| 2016 | 0.094** | 0.022 |

Pre-trend coefficients (2006-2010) are small and statistically insignificant, supporting parallel trends assumption.

#### Robustness Checks
1. **Narrower bandwidth** (ages 28-30 vs 31-33): 0.051 (SE: 0.013)
2. **By sex**:
   - Males: 0.062 (SE: 0.011)
   - Females: 0.031 (SE: 0.015)
3. **Logit model** (marginal effect): 0.055 (SE: 0.010)

---

### Key Decisions Made

1. **Sample Definition**: Used CITIZEN == 3 (not a citizen) to proxy for undocumented status, as we cannot distinguish documented vs undocumented in the data.

2. **Age Calculation**: Used birth year to assign treatment/control groups. Treatment = born 1982-1986 (ages 26-30 in 2012), Control = born 1977-1981 (ages 31-35 in 2012).

3. **Immigration Timing**: Required YRIMMIG <= 2007 to ensure individuals were in the US since June 2007 (per DACA requirement of continuous presence since June 15, 2007).

4. **Excluded 2012**: Cannot distinguish pre- vs post-DACA within 2012 since ACS doesn't have month of interview.

5. **Full-time Definition**: UHRSWORK >= 35 hours as specified in instructions.

6. **Weights**: Used PERWT (person weights) in weighted regressions to produce population-representative estimates.

7. **Standard Errors**: Clustered by state in preferred specification to account for within-state correlation.

8. **Fixed Effects**: Included year and state fixed effects to control for common shocks and time-invariant state characteristics.

---

### Files Generated
- `analysis.py` - Main analysis script
- `results.csv` - Key results summary
- `main_results.csv` - Regression table
- `desc_stats_by_group.csv` - Descriptive statistics
- `event_study.csv` - Event study coefficients
- `yearly_rates.csv` - Year-by-year employment rates
- `replication_report_16.tex` - LaTeX report
- `replication_report_16.pdf` - Final PDF report

