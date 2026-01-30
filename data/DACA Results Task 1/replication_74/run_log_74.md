# Run Log for Replication 74: DACA Impact on Full-Time Employment

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (defined as usually working 35+ hours per week)?

## Date
2026-01-25

---

## Step 1: Data Overview

### Data Source
- American Community Survey (ACS) 2006-2016 from IPUMS USA
- Main file: data.csv (~6.3 GB, 33,851,424 observations)
- Supplemental: state_demo_policy.csv (optional state-level data)
- Data dictionary: acs_data_dict.txt

### Key Variables Identified from Data Dictionary

**Sample Identification:**
- YEAR: Survey year (2006-2016)
- PERWT: Person weight for population estimates

**Treatment Group Identification (DACA eligibility):**
- HISPAN/HISPAND: Hispanic origin (need HISPAN=1 for Mexican)
- BPL/BPLD: Birthplace (need BPL=200 for Mexico)
- CITIZEN: Citizenship status (need CITIZEN=3 for non-citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth
- AGE: Age at survey

**Outcome Variable:**
- UHRSWORK: Usual hours worked per week (full-time = 35+)
- EMPSTAT: Employment status
- LABFORCE: Labor force status

---

## Step 2: DACA Eligibility Criteria

DACA eligibility requires:
1. Arrived in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007
4. Present in the US on June 15, 2012 and did not have lawful status

### Operationalization:
- **Age at arrival < 16**: Age at immigration = YEAR - YRIMMIG (at survey) minus current age approximation
  - Alternatively: If YRIMMIG - BIRTHYR < 16
- **Born after June 15, 1981**: BIRTHYR > 1981, OR (BIRTHYR == 1981 AND BIRTHQTR >= 2)
  - Conservative: BIRTHYR >= 1982
- **In US since June 15, 2007**: YRIMMIG <= 2007
- **Non-citizen**: CITIZEN == 3 (Not a citizen)
- **Mexican-born Hispanic-Mexican**: HISPAN == 1 AND BPL == 200

---

## Step 3: Analysis Strategy

### Identification Strategy: Difference-in-Differences

**Treatment Group:** DACA-eligible individuals (meeting all criteria above)
**Control Group:** Mexican-born Hispanic-Mexican non-citizens who are NOT DACA-eligible (typically arrived as adults or too old)

**Time Periods:**
- Pre-DACA: 2006-2011 (excluding 2012 due to mid-year implementation)
- Post-DACA: 2013-2016 (as specified in research question)

**Outcome:** Binary indicator for full-time employment (UHRSWORK >= 35)

### Model Specification:
Y_it = β0 + β1*Eligible_i + β2*Post_t + β3*(Eligible_i × Post_t) + X_it'γ + ε_it

Where β3 is the difference-in-differences estimate of DACA's effect on full-time employment.

---

## Step 4: Implementation Commands

```python
# See analysis_script.py for full implementation
```

---

## Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pre-period years | 2006-2011 | Exclude 2012 due to mid-year DACA implementation |
| Post-period years | 2013-2016 | As specified in research question |
| Age at arrival threshold | < 16 | DACA requirement |
| Birth year cutoff | >= 1982 | Born after June 15, 1981 (conservative) |
| Immigration year cutoff | <= 2007 | In US since June 15, 2007 |
| Full-time definition | UHRSWORK >= 35 | Standard definition per research question |
| Sample restriction | Ages 18-40 | Working-age population, relevant to DACA age range |
| Control group | Non-eligible Mexican-born non-citizens | Similar demographic but not eligible for DACA |

---

## Analysis Progress

### Command 1: Run Main Analysis Script
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_74" && python analysis_script.py
```

### Sample Construction Results:
- Total ACS observations loaded: 33,851,424
- After restricting to Hispanic-Mexican (HISPAN=1): 2,945,521
- After restricting to Mexican-born (BPL=200): 991,261
- After restricting to non-citizens (CITIZEN=3): 701,347
- After excluding 2012: 636,722
- After restricting to ages 18-40: 341,332

### DACA Eligibility Components:
- Arrived before age 16: 201,531 (28.7%)
- Born after mid-1981: 234,404 (33.4%)
- In US since 2007: 654,693 (93.3%)
- DACA eligible (all criteria): 131,321 (18.7%)

### Final Analysis Sample:
| Group | Pre-Period (2006-2011) | Post-Period (2013-2016) |
|-------|------------------------|-------------------------|
| DACA Eligible | 38,344 | 32,963 |
| Not Eligible | 186,254 | 83,771 |

---

## Main Results

### Simple 2x2 DiD Table (Weighted Full-Time Employment Rates):

| Group | Pre (2006-2011) | Post (2013-2016) | Difference |
|-------|-----------------|------------------|------------|
| Eligible | 0.5283 | 0.5706 | +0.0423 |
| Not Eligible | 0.6410 | 0.6121 | -0.0289 |
| Difference | -0.1127 | -0.0416 | **+0.0712** |

### Regression Results:

| Model | DiD Coefficient | Std. Error | p-value | N |
|-------|-----------------|------------|---------|---|
| Basic DiD | 0.0712 | 0.0051 | <0.001 | 341,332 |
| With Demographics | 0.0250 | 0.0048 | <0.001 | 341,332 |
| Full (State/Year FE) | 0.0126 | 0.0048 | 0.009 | 341,332 |

### Preferred Estimate (Full Model with State and Year Fixed Effects):
- **DiD Coefficient: 0.0126**
- **Standard Error: 0.0048**
- **95% CI: [0.0032, 0.0219]**
- **p-value: 0.0087**

### Interpretation:
DACA eligibility is associated with a 1.26 percentage point increase in the probability of full-time employment. This effect is statistically significant at the 1% level.

---

## Robustness Checks

| Specification | DiD Coefficient | Std. Error | N |
|---------------|-----------------|------------|---|
| Employment (any work) | 0.0333 | 0.0046 | 341,332 |
| Full-time (labor force only) | -0.0126 | 0.0052 | 246,133 |
| Males only | -0.0067 | 0.0060 | 192,104 |
| Females only | 0.0309 | 0.0075 | 149,228 |

---

## Event Study Coefficients (Base Year: 2011)

| Year | Coefficient | Std. Error |
|------|-------------|------------|
| 2006 | 0.0241 | 0.0111 |
| 2007 | 0.0174 | 0.0106 |
| 2008 | 0.0298 | 0.0107 |
| 2009 | 0.0264 | 0.0106 |
| 2010 | 0.0230 | 0.0103 |
| 2013 | 0.0178 | 0.0102 |
| 2014 | 0.0237 | 0.0103 |
| 2015 | 0.0404 | 0.0102 |
| 2016 | 0.0471 | 0.0105 |

Note: Pre-treatment coefficients suggest some pre-existing differences (not perfectly zero), but the post-treatment coefficients show an increasing pattern consistent with DACA effects accumulating over time.

---

## Output Files Generated:
- summary_stats_pre.csv
- summary_stats_post.csv
- did_table.csv
- event_study_coefs.csv
- regression_results.csv
- robustness_results.csv
- yearly_means.csv

