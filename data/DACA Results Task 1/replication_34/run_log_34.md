# Replication Run Log - Study 34

## Project Overview
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

**Time Period:** Effects examined for 2013-2016 (DACA implemented June 15, 2012)

---

## Session Log

### Step 1: Initial Setup and Data Examination
**Time:** Session Start

- Read replication_instructions.docx
- Examined data files:
  - data.csv: 33,851,424 observations (ACS data 2006-2016)
  - acs_data_dict.txt: Variable descriptions and codes
  - state_demo_policy.csv: Optional state-level data (not used)

### Key Variables Identified:
| Variable | Description | Use |
|----------|-------------|-----|
| YEAR | Census/ACS year | Time periods |
| HISPAN | Hispanic origin | Sample restriction (=1 Mexican) |
| BPL | Birthplace | Sample restriction (=200 Mexico) |
| CITIZEN | Citizenship status | Eligibility (=3 Not a citizen) |
| YRIMMIG | Year of immigration | Eligibility criterion |
| BIRTHYR | Birth year | Age calculation |
| BIRTHQTR | Birth quarter | Under-31 determination |
| AGE | Age | Sample restriction, control |
| UHRSWORK | Usual hours worked | Outcome variable |
| EMPSTAT | Employment status | Alternative outcome |
| PERWT | Person weight | Survey weights |

---

## Step 2: DACA Eligibility Criteria Definition

Based on the instructions, DACA eligibility requires:

1. **Arrived before age 16:** `age_at_immig = YRIMMIG - BIRTHYR < 16`
2. **Under 31 on June 15, 2012:** `BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)`
3. **Continuous residence since June 2007:** `YRIMMIG <= 2007`
4. **Not a citizen:** `CITIZEN == 3`

**Key Decision:** We cannot distinguish documented vs. undocumented status. Following instructions, we assume non-citizens who haven't received papers are potentially undocumented.

---

## Step 3: Sample Construction

### Data Processing Flow:
1. Load data in chunks (1M rows) to manage memory
2. Filter to Hispanic-Mexican (HISPAN=1) and Mexico-born (BPL=200): **991,261 observations**
3. Exclude 2012 (DACA implemented mid-year): **898,879 observations**
4. Restrict to working age (16-64): **771,888 observations**
5. Require valid YRIMMIG > 0: **771,888 observations** (all had valid)
6. Focus on non-citizens only: **561,470 observations**

### Treatment and Control Groups:
- **Treatment (DACA Eligible):** Non-citizens meeting all eligibility criteria: **83,611 observations**
- **Control Group:** Non-citizens who arrived young AND immigrated early BUT too old (born before June 1981): **54,881 observations**
- **Main DiD Sample:** Treatment + Control: **138,492 observations**

**Rationale for Control Group:** The control group is similar to the treatment group in that they also arrived as children and have been in the US since 2007, but they fail only the age criterion. This makes them a plausible counterfactual for what would have happened to DACA-eligible individuals absent the program.

---

## Step 4: Outcome Variable

**Primary Outcome:** Full-time employment
- Definition: `UHRSWORK >= 35` (usual hours worked per week 35 or more)
- This follows the standard BLS definition of full-time work

**Secondary Outcome:** Any employment
- Definition: `EMPSTAT == 1`

---

## Step 5: Identification Strategy

### Difference-in-Differences Design:
- **Pre-period:** 2006-2011 (6 years)
- **Post-period:** 2013-2016 (4 years)
- **Treatment:** DACA-eligible individuals
- **Control:** Similar individuals ineligible due to age

### Key Assumptions:
1. **Parallel Trends:** Absent DACA, treatment and control groups would have followed similar employment trajectories
2. **No Anticipation:** No significant behavioral response before DACA implementation
3. **SUTVA:** Treatment of one individual doesn't affect others' outcomes

---

## Step 6: Analysis Results

### Main DiD Regression Results:

| Model | DACA x POST | SE | p-value | 95% CI |
|-------|-------------|-----|---------|--------|
| Basic DiD (no controls) | 0.1103 | 0.0067 | <0.001 | [0.097, 0.123] |
| + Demographics | 0.0163 | 0.0063 | 0.009 | [0.004, 0.029] |
| + Year FE | 0.0059 | 0.0063 | 0.346 | [-0.006, 0.018] |
| + State + Year FE | 0.0048 | 0.0062 | 0.446 | [-0.007, 0.017] |

**Preferred Estimate (Model 4):** 0.48 percentage points (SE = 0.62 pp, p = 0.446)

### Interpretation:
The estimated effect is small and statistically insignificant. DACA eligibility is associated with approximately a 0.5 percentage point increase in full-time employment probability, but this effect is not distinguishable from zero at conventional significance levels.

---

## Step 7: Robustness Checks

| Specification | Coefficient | SE | N |
|---------------|-------------|-----|-----|
| Any Employment (alt outcome) | 0.0042 | 0.0061 | 138,492 |
| Males Only | -0.0205 | 0.0077 | 78,318 |
| Females Only | 0.0257 | 0.0102 | 60,174 |
| Broader Control (all non-eligible non-citizens) | 0.0300 | 0.0042 | 561,470 |

### Event Study Results (Reference: 2011):

| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2006 | 0.0079 | 0.0131 | [-0.018, 0.034] |
| 2007 | 0.0070 | 0.0129 | [-0.018, 0.032] |
| 2008 | 0.0208 | 0.0130 | [-0.005, 0.046] |
| 2009 | 0.0119 | 0.0131 | [-0.014, 0.038] |
| 2010 | 0.0162 | 0.0130 | [-0.009, 0.042] |
| 2011 | 0.0000 | - | (reference) |
| 2013 | 0.0136 | 0.0134 | [-0.013, 0.040] |
| 2014 | 0.0134 | 0.0135 | [-0.013, 0.040] |
| 2015 | 0.0088 | 0.0137 | [-0.018, 0.036] |
| 2016 | 0.0263 | 0.0138 | [-0.001, 0.053] |

**Parallel Trends Assessment:** Pre-treatment coefficients (2006-2010) are all small and statistically insignificant, providing support for the parallel trends assumption. Post-treatment coefficients show a slight increase but remain statistically insignificant.

---

## Step 8: Key Decisions and Justifications

1. **Excluded 2012:** DACA was implemented mid-year (June 15, 2012). ACS data doesn't include survey month, making it impossible to distinguish pre- and post-treatment observations in 2012.

2. **Age restriction (16-64):** Standard working-age population. The lower bound ensures individuals are of legal working age; the upper bound excludes those near retirement.

3. **Control group definition:** Used non-citizens who arrived young and early but are too old, rather than all non-eligible immigrants. This provides better comparability on observable characteristics.

4. **Birth quarter for under-31 criterion:** Those born July-December 1981 (Q3-Q4) would be under 31 on June 15, 2012. Those born January-June 1981 would not qualify.

5. **Survey weights:** Used PERWT for all regressions to produce population-representative estimates.

6. **Robust standard errors:** Used HC1 heteroskedasticity-robust standard errors.

---

## Output Files Generated

1. `summary_statistics.csv` - Summary statistics by treatment/period
2. `regression_results.csv` - Main regression results
3. `robustness_results.csv` - Robustness check results
4. `event_study_results.csv` - Year-by-year coefficients
5. `final_results.txt` - Preferred estimate summary
6. `analysis.py` - Complete analysis code

---

## Final Summary

**Preferred Estimate:**
- Effect Size: 0.48 percentage points
- Standard Error: 0.62 percentage points
- 95% CI: [-0.75, 1.70] percentage points
- p-value: 0.446
- Sample Size: 138,492

**Conclusion:** We find no statistically significant effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born non-citizens. The point estimate suggests a small positive effect (about 0.5 percentage points), but this is not distinguishable from zero. This null finding is consistent across multiple specifications and robustness checks.
