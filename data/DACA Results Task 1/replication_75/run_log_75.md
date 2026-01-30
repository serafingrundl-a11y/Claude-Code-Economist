# Run Log for DACA Replication Study (Participant 75)

## Session Started: 2026-01-25

### Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (usually working 35+ hours per week)?

---

## Step 1: Initial Data Exploration

### Data Files Available:
- `data.csv` - Main ACS data file (6.26 GB)
- `acs_data_dict.txt` - Data dictionary with variable definitions
- `state_demo_policy.csv` - Optional state-level data (not used)

### Key Variables Identified from Data Dictionary:
- **YEAR**: Survey year (2006-2016 available)
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- **HISPAN/HISPAND**: Hispanic origin (1=Mexican for general; 100-107 for detailed Mexican)
- **BPL/BPLD**: Birthplace (200=Mexico for general; 20000 for detailed)
- **CITIZEN**: Citizenship status (3=Not a citizen)
- **YRIMMIG**: Year of immigration
- **UHRSWORK**: Usual hours worked per week
- **EMPSTAT**: Employment status
- **PERWT**: Person weight for survey weighting
- **AGE**: Age at survey

### DACA Eligibility Criteria (from instructions):
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012

---

## Step 2: Defining DACA Eligibility

### Decision Log:

**Decision 1: Sample Population**
- Focus on Hispanic-Mexican ethnicity (HISPAN=1) AND born in Mexico (BPL=200)
- Rationale: Instructions specify "ethnically Hispanic-Mexican Mexican-born people"

**Decision 2: Non-citizen assumption**
- Per instructions: "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes"
- Use CITIZEN=3 (Not a citizen) as proxy for undocumented status

**Decision 3: Age Eligibility**
- Must be under 31 as of June 15, 2012
- Birth year >= 1982 (born 1982 or later = under 31 in 2012)
- Must have arrived before age 16

**Decision 4: Arrival Age Calculation**
- Age at arrival = YRIMMIG - BIRTHYR
- Must be < 16 at time of arrival

**Decision 5: Continuous Presence**
- Must have immigrated by 2007 (YRIMMIG <= 2007) to satisfy 5-year continuous presence requirement
- Note: This is a proxy since we cannot observe actual continuous presence

**Decision 6: Treatment Period**
- DACA implemented June 15, 2012
- Pre-period: 2006-2011
- Treatment period: 2013-2016 (effects examined per instructions)
- 2012 excluded as a partial treatment year (applications started August 2012)

**Decision 7: Control Group**
- Mexican-born, Hispanic-Mexican, non-citizens who arrived AFTER age 15 (ineligible due to arrival age)
- OR those who arrived too recently (after 2007)
- OR those too old (31+ as of June 2012)

**Decision 8: Full-time Employment Outcome**
- UHRSWORK >= 35 defines full-time employment
- This matches the instructions' definition

---

## Step 3: Estimation Strategy

### Approach: Difference-in-Differences (DiD)

**Rationale:**
- DACA creates a natural experiment with clear eligibility criteria
- Can compare eligible vs. ineligible Mexican-born non-citizens before and after 2012
- This controls for common trends affecting all Mexican immigrants and for pre-existing differences between groups

**Model Specification:**
```
FullTime_it = β0 + β1*Eligible_i + β2*Post_t + β3*(Eligible_i × Post_t) + X_it'γ + ε_it
```

Where:
- FullTime_it = 1 if usually works 35+ hours/week
- Eligible_i = 1 if meets DACA eligibility criteria
- Post_t = 1 if year >= 2013
- β3 = DiD estimate (effect of DACA on full-time employment)
- X_it = control variables (age, sex, education, marital status, state, year FE)

---

## Step 4: Data Loading and Sample Construction

### Sample Construction Results:
```
1. Hispanic-Mexican, Mexican-born (all years): 991,261
2. Restrict to non-citizens (CITIZEN = 3): 701,347
3. Remove missing immigration year: 701,347
4. Restrict to working age (16-64): 618,640
5. Exclude 2012 survey year: 561,470
```

### DACA Eligibility Breakdown:
- Arrived before age 16: 155,206
- Under 31 in 2012: 171,984
- Present by 2007: 581,973
- **All three criteria (DACA eligible): 90,501**

### Final Analytic Sample:
- Total observations: 561,470
- Pre-DACA (2006-2011): 345,792
- Post-DACA (2013-2016): 215,678
- DACA Eligible: 81,508
- DACA Ineligible: 479,962

---

## Step 5: Descriptive Statistics

### Mean Full-Time Employment by Group and Period:

| Group | Pre-DACA (2006-2011) | Post-DACA (2013-2016) | Difference |
|-------|---------------------|----------------------|------------|
| DACA Eligible | 0.4248 (n=45,433) | 0.4939 (n=36,075) | +0.0691 |
| DACA Ineligible | 0.6040 (n=300,359) | 0.5791 (n=179,603) | -0.0249 |

**Simple DiD Estimate: 0.0941**
- Change in Eligible: +6.91 pp
- Change in Ineligible: -2.49 pp
- Difference-in-Differences: +9.41 pp

---

## Step 6: Regression Analysis

### Model Specifications and Results:

| Model | DiD Coefficient | Std. Error | N |
|-------|----------------|-----------|------|
| (1) Basic DiD | 0.0941 | 0.0038 | 561,470 |
| (2) + Demographics | 0.0421 | 0.0035 | 561,470 |
| (3) + Year FE | 0.0364 | 0.0035 | 561,470 |
| (4) + State & Year FE (Preferred) | **0.0358** | **0.0035** | 561,470 |

### Preferred Estimate Details:
- **Effect Size: 0.0358 (3.58 percentage points)**
- Standard Error: 0.0035
- 95% CI: [0.0289, 0.0427]
- t-statistic: 10.19
- p-value: < 0.0001
- R-squared: 0.2178

### Control Variable Coefficients (Model 4):
- Age: 0.0432 (0.0004)
- Age squared: -0.0005 (0.0000)
- Female: -0.4195 (0.0012)
- Married: -0.0249 (0.0013)
- High school: 0.0468 (0.0013)
- Some college: 0.0516 (0.0024)
- College+: 0.0781 (0.0030)

---

## Step 7: Robustness Checks

| Specification | Coefficient | Std. Error | N |
|--------------|-------------|-----------|------|
| Main (ages 16-64) | 0.0358 | 0.0035 | 561,470 |
| Ages 18-35 only | 0.0080 | 0.0043 | 253,373 |
| Any Employment outcome | 0.0453 | 0.0035 | 561,470 |
| Males only | 0.0323 | 0.0046 | 303,717 |
| Females only | 0.0295 | 0.0051 | 257,753 |
| Placebo (pre-period) | 0.0188 | 0.0046 | 345,792 |

---

## Step 8: Event Study Analysis

### Year-by-Year Coefficients (relative to 2011):

| Year | Coefficient | Std. Error | 95% CI |
|------|-------------|-----------|--------|
| 2006 | -0.0245 | 0.0080 | [-0.040, -0.009] |
| 2007 | -0.0197 | 0.0078 | [-0.035, -0.004] |
| 2008 | -0.0066 | 0.0078 | [-0.022, 0.009] |
| 2009 | 0.0004 | 0.0077 | [-0.015, 0.015] |
| 2010 | 0.0037 | 0.0075 | [-0.011, 0.018] |
| 2011 | 0.0000 | --- | (reference) |
| 2013 | 0.0095 | 0.0074 | [-0.005, 0.024] |
| 2014 | 0.0236 | 0.0074 | [0.009, 0.038] |
| 2015 | 0.0412 | 0.0074 | [0.027, 0.056] |
| 2016 | 0.0425 | 0.0075 | [0.028, 0.057] |

### Interpretation:
- Pre-trend coefficients for 2008-2010 are not significantly different from zero
- Some pre-trend in 2006-2007 (negative, significant)
- Treatment effects emerge in 2013-2014 and grow through 2015-2016
- Pattern consistent with gradual program rollout

---

## Step 9: Output Files Generated

1. **analysis_data.csv** - Cleaned analysis dataset
2. **regression_results.csv** - Main regression coefficients
3. **event_study_results.csv** - Year-by-year event study coefficients
4. **replication_report_75.tex** - LaTeX source for final report
5. **replication_report_75.pdf** - Final PDF report

---

## Summary of Key Decisions

1. **Sample Definition**: Hispanic-Mexican (HISPAN=1), Mexican-born (BPL=200), non-citizen (CITIZEN=3), working age (16-64)

2. **DACA Eligibility**: Arrived before age 16 AND born 1982+ AND immigrated by 2007

3. **Treatment Period**: 2013-2016 (post-DACA), 2006-2011 (pre-DACA), 2012 excluded

4. **Outcome**: Full-time employment = UHRSWORK >= 35

5. **Estimation Method**: Difference-in-differences with state and year fixed effects

6. **Standard Errors**: Heteroskedasticity-robust (HC1)

7. **Not used**: Survey weights in regression (following econometric best practices for causal inference); state-level supplemental data

---

## Final Results Summary

**Preferred Estimate:**
- DACA eligibility increased full-time employment by **3.58 percentage points**
- This effect is statistically significant (p < 0.0001)
- 95% confidence interval: [2.89, 4.27] percentage points
- Sample size: 561,470 observations

**Interpretation:**
DACA eligibility is associated with a meaningful increase in full-time employment among Hispanic-Mexican, Mexican-born non-citizens. Given the baseline employment rate of approximately 42.5% among eligible individuals, this represents a relative increase of about 8.4%.

---

## Session Completed: 2026-01-25
