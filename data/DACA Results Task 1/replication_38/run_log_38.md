# Run Log - Replication 38

## Project Overview
Independent replication examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican Mexican-born individuals in the United States.

## Key Dates and Timeline
- DACA announced: June 15, 2012
- Applications began: August 15, 2012
- Analysis period: 2006-2016 (ACS data)
- Post-treatment outcomes examined: 2013-2016

---

## Session Log

### Step 1: Data Review and Understanding
**Timestamp:** Session start

**Actions:**
- Read replication_instructions.docx to understand the research question
- Examined acs_data_dict.txt to understand variable definitions
- Verified data.csv structure and columns

**Key Variables Identified:**
- YEAR: Census year (2006-2016)
- HISPAN/HISPAND: Hispanic origin (1/100-107 = Mexican)
- BPL/BPLD: Birthplace (200/20000 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- AGE: Age at survey
- UHRSWORK: Usual hours worked per week (35+ = full-time)
- EMPSTAT: Employment status
- PERWT: Person weight

### Step 2: DACA Eligibility Definition
**Criteria per instructions:**
1. Arrived unlawfully before 16th birthday
2. Not yet 31 by June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status
5. Hispanic-Mexican ethnicity AND born in Mexico

**Operationalization:**
- HISPAN == 1 (Mexican) AND BPL == 200 (Mexico)
- CITIZEN == 3 (Not a citizen)
- Age at immigration < 16: (YRIMMIG - BIRTHYR) < 16
- Born after 1981 (age < 31 in 2012): BIRTHYR > 1981
- In US for 5+ years by 2012: YRIMMIG <= 2007

**Note:** Cannot distinguish documented vs undocumented, assume all non-citizens without naturalization are potentially undocumented.

### Step 3: Outcome Variable
- Full-time employment: UHRSWORK >= 35
- Conditional on being employed: EMPSTAT == 1

### Step 4: Research Design
**Approach:** Difference-in-Differences

**Treatment group:** DACA-eligible individuals (meeting all criteria above)
**Control group:** Similar Mexican-born Hispanic non-citizens who are NOT eligible
  - Option 1: Those too old (born 1981 or earlier, age 31+ in 2012)
  - Option 2: Those who arrived too recently (after 2007)
  - Option 3: Those who arrived after age 16

**Primary control:** Age-ineligible group (born 1981 or earlier) as they are most comparable

**Time periods:**
- Pre-treatment: 2006-2011 (2012 excluded due to mid-year implementation)
- Post-treatment: 2013-2016

### Step 5: Python Analysis Code
- Created analysis script with data loading, cleaning, and DiD estimation
- Used statsmodels for regression with robust standard errors
- Included demographic controls and state fixed effects

---

## Decisions Made

1. **Excluded 2012:** Mid-year implementation makes treatment assignment ambiguous
2. **Control group:** Age-ineligible (age 31+ in 2012) Mexican-born Hispanic non-citizens
3. **Sample restrictions:** Working-age adults (18-65), in labor force
4. **Clustered standard errors:** By state for policy variation
5. **Controls:** Age, sex, education, marital status, years in US, state FEs

---

## Files Created
- run_log_38.md (this file)
- analysis_38.py (main analysis script)
- generate_figures.py (figure generation script)
- results_38.json (saved results data)
- figure1_event_study.png (event study plot)
- figure2_trends.png (trends by treatment status)
- figure3_coefficients.png (coefficient comparison)
- figure4_gender.png (gender heterogeneity)
- replication_report_38.tex (LaTeX report, 22 pages)
- replication_report_38.pdf (compiled report)

---

## Final Results Summary

### Preferred Estimate (Model 4: Full Controls + State FE + Year FE)

| Metric | Value |
|--------|-------|
| Coefficient | 0.0012 (0.12 pp) |
| Standard Error | 0.0053 (0.53 pp) |
| 95% CI | [-0.0091, 0.0116] |
| P-value | 0.8166 |
| Sample Size | 123,390 |

### Interpretation
DACA eligibility had **no statistically significant effect** on full-time employment in the preferred specification. The point estimate of 0.12 percentage points is economically small and not statistically distinguishable from zero.

### Gender Heterogeneity
- **Males:** -2.16 pp (SE = 0.71 pp), statistically significant negative effect
- **Females:** +2.14 pp (SE = 0.76 pp), statistically significant positive effect

These offsetting effects may explain the null overall result.

### Key Analytical Choices
1. **Treatment group:** DACA-eligible non-citizens (arrived before 16, in US by 2007, under 31 in 2012)
2. **Control group:** Age-ineligible non-citizens (31+ in 2012) with similar characteristics
3. **Sample:** Hispanic-Mexican, born in Mexico, non-citizens, ages 18-55
4. **Period:** 2006-2011 (pre) and 2013-2016 (post), excluding 2012
5. **Estimation:** WLS with person weights, state-clustered standard errors
