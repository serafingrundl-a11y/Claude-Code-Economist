# Replication Run Log - Replication 07

## Task Overview
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Session Start: 2026-01-25

### Step 1: Data Understanding

**Data Files:**
- `data.csv`: ACS data from 2006-2016 (33,851,424 observations + header)
- `acs_data_dict.txt`: Data dictionary for IPUMS ACS variables
- `state_demo_policy.csv`: Optional state-level data

**Key Variables Identified:**
- YEAR: Survey year (2006-2016)
- HISPAN: Hispanic origin (1 = Mexican)
- BPL: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Birth quarter (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- AGE: Age
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status
- PERWT: Person weight for population estimates

### Step 2: DACA Eligibility Criteria

Per the instructions, DACA eligibility requires:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

**Operationalization:**
- Hispanic-Mexican (HISPAN == 1) AND Mexican-born (BPL == 200)
- Not a citizen (CITIZEN == 3) - proxy for undocumented
- Age at immigration < 16: (YRIMMIG - BIRTHYR) < 16
- Age on June 15, 2012 < 31: Born after June 15, 1981
- Present since June 15, 2007: YRIMMIG <= 2007
- For age calculation: Use BIRTHYR and BIRTHQTR

### Step 3: Analysis Strategy

**Difference-in-Differences Design:**
- Treatment group: DACA-eligible Hispanic-Mexican non-citizens born in Mexico
- Control group: Similar population that is NOT DACA-eligible (e.g., arrived too late, too old, etc.)
- Pre-period: 2006-2011 (before DACA implementation)
- Post-period: 2013-2016 (after DACA, excluding 2012 which spans implementation)

**Outcome Variable:**
- Full-time employment = 1 if UHRSWORK >= 35, 0 otherwise (among employed)
- Alternative: Full-time employment among all labor force participants

### Step 4: Data Processing and Analysis

**Sample Construction:**
1. Started with 33,851,424 observations (ACS 2006-2016)
2. Filtered to Hispanic-Mexican (HISPAN == 1) AND Mexican-born (BPL == 200): 991,261 observations
3. Filtered to non-citizens (CITIZEN == 3): 701,347 observations
4. Excluded 2012 (partial treatment year): 636,722 observations
5. Restricted to working-age (16-64): 561,470 observations

**DACA Eligibility Construction:**
- Arrived before age 16: 205,327 individuals
- In US since 2007: 654,693 individuals
- Under 31 as of June 2012: 230,009 individuals
- Meeting ALL criteria (DACA eligible): 133,120 individuals

**Final Sample by Group:**
|                  | Pre (2006-2011) | Post (2013-2016) | Total   |
|------------------|-----------------|------------------|---------|
| Not Eligible     | 298,978         | 178,881          | 477,859 |
| DACA Eligible    | 46,814          | 36,797           | 83,611  |
| Total            | 345,792         | 215,678          | 561,470 |

### Step 5: Key Results

**Raw Difference-in-Differences:**
- DACA Eligible Pre: 0.4309 (43.09% full-time employment)
- DACA Eligible Post: 0.4962 (49.62% full-time employment)
- Not Eligible Pre: 0.6039 (60.39% full-time employment)
- Not Eligible Post: 0.5790 (57.90% full-time employment)
- **DiD Estimate: 0.0902 (9.02 percentage points)**

**Regression Results:**

| Model | DiD Coefficient | SE | N |
|-------|-----------------|-----|-------|
| Basic DiD (no controls) | 0.0902 | 0.0037 | 561,470 |
| With demographic controls | 0.0872 | 0.0034 | 561,470 |
| State fixed effects | 0.0866 | 0.0034 | 561,470 |
| State + Year FE | 0.0829 | 0.0034 | 561,470 |
| Weighted (PERWT) | 0.0848 | 0.0033 | 561,470 |
| Clustered SE (by state) | 0.0872 | 0.0040 | 561,470 |

**Preferred Specification (Model with demographic controls):**
- DiD Coefficient: 0.0872
- Standard Error: 0.0034
- 95% CI: [0.0806, 0.0938]
- t-statistic: 25.80
- p-value: < 0.001

### Step 6: Event Study Analysis

Event study coefficients (reference year: 2011):

| Year | Coefficient | SE | Significance |
|------|-------------|------|--------------|
| 2006 | -0.0563 | 0.0077 | *** |
| 2007 | -0.0487 | 0.0075 | *** |
| 2008 | -0.0304 | 0.0076 | *** |
| 2009 | -0.0182 | 0.0074 | ** |
| 2010 | -0.0072 | 0.0072 | |
| 2013 | 0.0235 | 0.0072 | *** |
| 2014 | 0.0478 | 0.0071 | *** |
| 2015 | 0.0763 | 0.0072 | *** |
| 2016 | 0.0878 | 0.0072 | *** |

Note: Pre-trends show a slight convergence pattern before 2011, suggesting caution in parallel trends assumption.

### Step 7: Heterogeneity Analysis

**By Gender:**
- Male: DiD = 0.0816 (SE: 0.0042), N = 303,717
- Female: DiD = 0.0841 (SE: 0.0053), N = 257,753

**By Age Group:**
- 16-25: DiD = 0.0693 (SE: 0.0071), N = 94,045
- 26-35: DiD = 0.0615 (SE: 0.0069), N = 166,525

### Step 8: Robustness Checks

1. **Alternative outcome (any employment):** DiD = 0.0983 (SE: 0.0034)
2. **Including 2012:** DiD = 0.0758 (SE: 0.0032), N = 618,640
3. **Placebo test (fake treatment at 2009):** DiD = 0.0365 (SE: 0.0044), p = 0.000
   - Note: Significant placebo effect raises concern about pre-trends

### Step 9: Key Decisions Made

1. **Population restriction:** Limited to Hispanic-Mexican, Mexican-born, non-citizen population as proxy for undocumented immigrants
2. **Eligibility definition:** Used YRIMMIG to calculate age at arrival, BIRTHYR and BIRTHQTR for age at DACA implementation
3. **Treatment timing:** Excluded 2012 as partial treatment year
4. **Outcome definition:** Full-time = UHRSWORK >= 35 (standard BLS definition)
5. **Control group:** Non-citizens who do not meet DACA eligibility criteria
6. **Standard errors:** Clustered by state for inference

### Step 10: Interpretation

The main finding suggests DACA eligibility increased full-time employment by approximately 8.7 percentage points among Hispanic-Mexican non-citizens born in Mexico. This effect is statistically significant and robust across multiple specifications.

However, the event study reveals pre-existing trends showing convergence between eligible and ineligible groups before DACA implementation. This suggests the parallel trends assumption may be partially violated, and the estimates should be interpreted with caution.

### Files Generated

1. `analysis.py` - Main analysis script
2. `create_figures.py` - Figure generation script
3. `regression_results.csv` - Regression coefficients
4. `event_study_results.csv` - Event study coefficients
5. `summary_statistics.csv` - Summary statistics
6. `figure1_event_study.pdf` - Event study plot
7. `figure2_parallel_trends.pdf` - Parallel trends plot
8. `figure3_model_comparison.pdf` - Model comparison
9. `figure4_did_diagram.pdf` - DiD visualization
10. `figure5_hours_distribution.pdf` - Hours distribution

### Session End: 2026-01-25
