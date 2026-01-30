# DACA Replication Analysis Run Log

## Project Overview
**Research Question:** Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (outcome), defined as usually working 35+ hours per week?

**Data Source:** American Community Survey (ACS) 2006-2016, via IPUMS USA
**Analysis Period:** Pre-DACA (2006-2011) and Post-DACA (2013-2016)

---

## Key Analysis Decisions

### 1. Sample Definition

**Target Population:**
- Hispanic-Mexican ethnicity (HISPAN == 1)
- Born in Mexico (BPL == 200)
- Non-citizens (CITIZEN == 3, which indicates "Not a citizen")
- We assume non-citizens without naturalization are undocumented for DACA purposes, as instructed

**DACA Eligibility Criteria (per instructions):**
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007 (at least 5 years)
4. Were present in the US on June 15, 2012 and did not have lawful status

**Operationalization:**
- Age at arrival = YRIMMIG - BIRTHYR
- Age at arrival < 16 → potentially DACA eligible
- For 2012 cutoff: Person must be under 31 as of June 15, 2012 (born 1982 or later)
- Continuous residence since June 2007: YRIMMIG <= 2007

### 2. Treatment and Control Group Definition

**Treatment Group (DACA-Eligible):**
- Hispanic-Mexican, Mexican-born, non-citizen
- Immigrated before age 16
- YRIMMIG <= 2007 (arrived at least 5 years before DACA)
- Born 1982 or later (would be under 31 as of June 15, 2012)

**Control Group (DACA-Ineligible):**
- Hispanic-Mexican, Mexican-born, non-citizen
- Either: Immigrated at age 16+ OR arrived after 2007 OR born before 1982

**Rationale:** Using a difference-in-differences design comparing eligible vs. ineligible non-citizen Mexican immigrants before and after DACA implementation.

### 3. Outcome Variable

**Full-Time Employment:** UHRSWORK >= 35
- Created binary indicator: 1 if usually works 35+ hours/week, 0 otherwise
- This follows the standard definition of full-time work in the US

### 4. Time Period

- **Pre-period:** 2006-2011 (DACA announced June 2012)
- **Post-period:** 2013-2016 (examining effects in years specified in research question)
- **Excluded:** 2012 (transitional year, cannot distinguish pre/post within 2012)

### 5. Empirical Strategy

**Difference-in-Differences Model:**
```
fulltime_it = β0 + β1*eligible_i + β2*post_t + β3*(eligible_i × post_t) + X_it'γ + ε_it
```

Where:
- β3 is the coefficient of interest (DACA effect)
- X includes controls: age, age², sex, marital status, education, state fixed effects, year fixed effects

### 6. Sample Restrictions

- Working-age population: Ages 16-64
- Non-institutionalized (GQ in 1, 2, 5 - household quarters)
- Valid immigration year (YRIMMIG > 0)

---

## Code Execution Log

### Step 1: Data Loading
- Loaded data.csv from IPUMS extract
- Total observations: 33,851,424
- Years covered: 2006-2016

### Step 2: Sample Construction
- Original sample: 33,851,424
- After Hispanic-Mexican (HISPAN==1): 2,945,521
- After Mexican-born (BPL==200): 991,261
- After non-citizens (CITIZEN==3): 701,347
- After working age 16-64: 618,640
- After non-institutional: 598,497
- After excluding 2012: 543,595
- Final sample with valid YRIMMIG: 543,595

### Step 3: DACA Eligibility
- Arrived before age 16: 134,635 (24.8%)
- Under 31 in June 2012: 148,750 (27.4%)
- Arrived by 2007: 514,326 (94.6%)
- **DACA Eligible (all criteria): 78,568 (14.5%)**

### Step 4: Outcome Variable
- Post-DACA period (2013-2016): 207,756 (38.2%)
- Full-time employment rate: 315,687 (58.1%)

### Step 5: Summary Statistics

**By DACA Eligibility:**
| Variable | Eligible | Ineligible | Difference |
|----------|----------|------------|------------|
| Age | 22.3 | 39.6 | -17.2 |
| Female | 46.1% | 47.3% | -1.2pp |
| Married | 25.7% | 66.3% | -40.5pp |
| Education < HS | 42.1% | 59.8% | -17.7pp |
| Full-time | 46.0% | 60.1% | -14.1pp |

**Difference-in-Differences Table:**
|  | Pre-DACA | Post-DACA | Difference |
|--|----------|-----------|------------|
| Eligible | 0.4272 | 0.5026 | +0.0754 |
| Ineligible | 0.6101 | 0.5857 | -0.0244 |
| DID | | | **+0.0998** |

### Step 6: Regression Results

**Model Specifications:**
1. Basic DID (no controls): β3 = 0.0998 (SE: 0.0039)
2. With demographic controls: β3 = 0.0448 (SE: 0.0036)
3. With year fixed effects: β3 = 0.0392 (SE: 0.0036)
4. Full model (unweighted): β3 = 0.0385 (SE: 0.0035)
5. **Full model (weighted): β3 = 0.0343 (SE: 0.0043)** [PREFERRED]

**Preferred Estimate (Weighted WLS with full controls):**
- Coefficient: 0.0343
- Standard Error: 0.0043
- 95% CI: [0.0259, 0.0427]
- p-value: < 0.0001

### Step 7: Robustness Checks

| Specification | Estimate | SE | p-value |
|---------------|----------|-----|---------|
| Employed only | -0.0009 | 0.0051 | 0.853 |
| Males only | 0.0331 | 0.0056 | <0.001 |
| Females only | 0.0268 | 0.0063 | <0.001 |
| Alt age cutoff (1981+) | 0.0284 | 0.0042 | <0.001 |

### Step 8: Event Study Results

| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | -0.0196 | 0.0100 | 0.051 |
| 2007 | -0.0120 | 0.0096 | 0.211 |
| 2008 | 0.0003 | 0.0097 | 0.973 |
| 2009 | 0.0092 | 0.0096 | 0.335 |
| 2010 | 0.0102 | 0.0093 | 0.274 |
| 2011 | (reference) | - | - |
| 2013 | 0.0170 | 0.0093 | 0.068 |
| 2014 | 0.0279 | 0.0094 | 0.003 |
| 2015 | 0.0437 | 0.0093 | <0.001 |
| 2016 | 0.0453 | 0.0095 | <0.001 |

---

## Final Results

**Main Finding:** DACA eligibility is associated with a **3.43 percentage point increase** in the probability of full-time employment among Hispanic-Mexican, Mexican-born non-citizens.

- This effect is statistically significant at the 1% level (p < 0.001)
- 95% Confidence Interval: [2.59pp, 4.27pp]
- Sample Size: 543,595 observations

**Interpretation:** The positive and significant effect suggests that DACA eligibility increased full-time employment among eligible individuals, consistent with the program's provision of work authorization.

---

## Files Generated
- `analysis_90.py` - Main analysis script
- `analysis_output_90.txt` - Full analysis output
- `summary_stats_90.csv` - Summary statistics
- `regression_results_90.csv` - Main regression results
- `event_study_90.csv` - Year-by-year effects
- `did_table_90.csv` - DID table
- `robustness_results_90.csv` - Robustness check results
- `replication_report_90.tex` - LaTeX report
- `replication_report_90.pdf` - Final PDF report
- `run_log_90.md` - This log file

---

## Notes and Observations

1. **Parallel Trends:** The event study shows that pre-DACA coefficients (2006-2010) are close to zero and statistically insignificant (except for a marginally significant negative coefficient in 2006), supporting the parallel trends assumption required for DID identification.

2. **Treatment Effect Dynamics:** The effect grows over time post-DACA, from 1.7pp in 2013 to 4.5pp in 2015-2016, suggesting gradual take-up or increasing program effects.

3. **Heterogeneity:** The effect is similar for males (3.3pp) and females (2.7pp), suggesting DACA benefits both genders.

4. **Intensive Margin:** When restricting to employed individuals only, the effect disappears, suggesting DACA primarily affects the extensive margin (labor force participation / employment) rather than hours among those already employed.

5. **Sample Characteristics:** DACA-eligible individuals are substantially younger (22 vs 40 years) and less likely to be married (26% vs 66%), reflecting the age requirements of the program.
