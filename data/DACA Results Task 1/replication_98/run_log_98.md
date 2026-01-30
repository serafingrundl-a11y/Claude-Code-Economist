# DACA Replication Study Run Log

## Project Overview
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (usually working 35+ hours per week)?

**Treatment:** DACA eligibility (program implemented June 15, 2012)
**Outcome:** Full-time employment (UHRSWORK >= 35)
**Post-treatment period:** 2013-2016

---

## Session Start: 2026-01-25

### Step 1: Review Instructions and Data Dictionary
- Read replication_instructions.docx
- Examined data dictionary (acs_data_dict.txt)
- Data files available:
  - data.csv (ACS data 2006-2016)
  - state_demo_policy.csv (optional supplemental state-level data)

### Key Variables Identified from Data Dictionary:
- **YEAR**: Census year (2006-2016)
- **PERWT**: Person weight
- **AGE**: Age
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- **HISPAN**: Hispanic origin (1=Mexican)
- **BPL/BPLD**: Birthplace (200/20000=Mexico)
- **CITIZEN**: Citizenship status (3=Not a citizen)
- **YRIMMIG**: Year of immigration
- **UHRSWORK**: Usual hours worked per week (outcome: >=35 for full-time)
- **EMPSTAT**: Employment status
- **SEX**: Sex
- **EDUC/EDUCD**: Educational attainment
- **MARST**: Marital status
- **STATEFIP**: State FIPS code

### DACA Eligibility Criteria (per instructions):
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

### Analytic Strategy:
**Identification approach:** Difference-in-Differences (DiD)
- Compare changes in full-time employment for DACA-eligible vs ineligible Mexican-born Hispanic non-citizens
- Pre-period: 2006-2011 (before DACA)
- Post-period: 2013-2016 (after DACA implementation)
- Note: 2012 excluded because DACA was implemented mid-year (June 15) and ACS doesn't indicate month of interview

**Treatment group:** Hispanic-Mexican, born in Mexico, non-citizen, meets DACA age/arrival criteria
**Control group:** Hispanic-Mexican, born in Mexico, non-citizen, does NOT meet DACA eligibility criteria (e.g., arrived too late or too old)

---

### Step 2: Data Processing and Analysis

**Script:** `analysis.py`

#### Sample Construction:
1. Loaded ACS data from 2006-2016 (33,851,424 observations)
2. Restricted to Hispanic-Mexican (HISPAN == 1): 2,945,521 obs
3. Restricted to born in Mexico (BPL == 200): 991,261 obs
4. Restricted to non-citizens (CITIZEN == 3): 701,347 obs
5. Excluded 2012 (mid-year DACA implementation): 636,722 obs
6. Restricted to working age 16-64: 561,470 obs
7. Removed missing YRIMMIG: 561,470 obs (final sample)

#### DACA Eligibility Construction:
- **arrived_before_16:** YRIMMIG - BIRTHYR < 16
- **under_31_june2012:** BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
- **in_us_since_2007:** YRIMMIG <= 2007
- **at_least_15_june2012:** BIRTHYR <= 1996 OR (BIRTHYR == 1997 AND BIRTHQTR <= 2)
- **daca_eligible:** All four criteria must be TRUE

#### Final Sample by Group:
| Period | Group | N | FT Rate |
|--------|-------|---|---------|
| Pre-DACA (2006-2011) | Eligible | 46,814 | 43.1% |
| Pre-DACA (2006-2011) | Ineligible | 298,978 | 60.4% |
| Post-DACA (2013-2016) | Eligible | 33,234 | 53.9% |
| Post-DACA (2013-2016) | Ineligible | 182,444 | 56.9% |

---

### Step 3: Main Results

#### Simple Difference-in-Differences:
- Change for Eligible: +10.85 pp
- Change for Ineligible: -3.44 pp
- **DiD Estimate: +14.29 pp** (SE: 0.0059, p < 0.001)

#### Preferred Model (OLS with Controls + Year FE):
- **DiD Estimate: +6.29 pp** (SE: 0.0063, p < 0.001)
- 95% CI: [5.05 pp, 7.53 pp]
- Sample size: 561,470
- R-squared: 0.215

Controls included:
- Age and age-squared
- Female indicator
- Married indicator
- High school or more education indicator
- Year fixed effects
- Clustered standard errors at state level

#### Alternative Specifications:
| Model | DiD Coefficient | SE | p-value |
|-------|-----------------|-----|---------|
| Simple DiD | 0.1429 | 0.0059 | <0.001 |
| With Controls | 0.0682 | 0.0066 | <0.001 |
| Controls + Year FE | 0.0629 | 0.0063 | <0.001 |
| Controls + State & Year FE | 0.0621 | 0.0064 | <0.001 |
| Weighted (PERWT) | 0.0600 | 0.0051 | <0.001 |

---

### Step 4: Robustness Checks

#### 4a. Alternative Outcome (Any Employment):
- DiD: +7.18 pp (SE: 0.0092, p < 0.001)

#### 4b. Placebo Test (Pre-trends with 2009 fake treatment):
- Placebo DiD: +1.61 pp (SE: 0.0042, p < 0.001)
- Note: Some evidence of pre-trends in early years

#### 4c. Event Study:
| Year | Coefficient | SE | Significant |
|------|-------------|-----|-------------|
| 2006 | -0.0198 | 0.0088 | Yes |
| 2007 | -0.0190 | 0.0054 | Yes |
| 2008 | -0.0060 | 0.0090 | No |
| 2009 | -0.0006 | 0.0064 | No |
| 2010 | +0.0023 | 0.0101 | No |
| 2011 | (reference) | - | - |
| 2013 | +0.0151 | 0.0096 | No |
| 2014 | +0.0487 | 0.0146 | Yes |
| 2015 | +0.0814 | 0.0115 | Yes |
| 2016 | +0.0888 | 0.0101 | Yes |

#### 4d. Subgroup Analysis by Gender:
- Male: DiD = +6.82 pp (SE: 0.0056, p < 0.001)
- Female: DiD = +4.76 pp (SE: 0.0086, p < 0.001)

---

### Step 5: Key Decisions and Justifications

1. **Exclusion of 2012:** DACA was implemented on June 15, 2012, and the ACS does not record interview month. Including 2012 would mix pre- and post-treatment observations.

2. **Control Group Definition:** Used Mexican-born, non-citizen, Hispanic individuals who do not meet DACA eligibility criteria (too old, arrived too late, etc.). This provides the closest counterfactual.

3. **Age Restriction (16-64):** Standard working-age population restriction. Younger individuals would not be in the labor force; older individuals have different labor market patterns.

4. **Standard Errors:** Clustered at the state level to account for within-state correlation of employment outcomes and the state-level variation in DACA implementation effects.

5. **Controls:** Included demographic characteristics (age, gender, marital status, education) that predict employment and may differ between treatment and control groups.

---

### Output Files Generated:
- `analysis.py` - Main analysis script
- `analysis_results.json` - All results in JSON format
- `summary_statistics.csv` - Descriptive statistics by group/period
- `regression_results.csv` - Regression coefficient comparison
- `event_study_results.csv` - Year-by-year treatment effects
- `preferred_model_summary.txt` - Full regression output

---

### Session End: 2026-01-25

**Preferred Estimate:**
- Effect: +6.29 percentage points increase in full-time employment
- Standard Error: 0.0063
- 95% CI: [5.05 pp, 7.53 pp]
- p-value: < 0.001
- Sample Size: 561,470

