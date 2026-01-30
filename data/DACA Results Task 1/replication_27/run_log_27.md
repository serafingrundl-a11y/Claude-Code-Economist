# Replication Run Log - Session 27

## Project Overview
**Research Question:** What was the causal impact of DACA eligibility on full-time employment (35+ hours/week) among Hispanic-Mexican, Mexican-born individuals in the US?

**Treatment Period:** DACA implemented June 15, 2012; examining effects 2013-2016

---

## Session Log

### Step 1: Data Exploration (Initial)
**Timestamp:** Session Start

**Data Files:**
- `data/data.csv` - Main ACS data file (~33.8 million observations)
- `data/acs_data_dict.txt` - IPUMS data dictionary
- `data/state_demo_policy.csv` - Optional state-level data

**Key Variables Identified:**
- YEAR: Survey year (2006-2016)
- HISPAN/HISPAND: Hispanic origin (1=Mexican for general version)
- BPL/BPLD: Birthplace (200=Mexico)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth (1-4)
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status (1=Employed)
- PERWT: Person weight for population estimates

### Step 2: DACA Eligibility Criteria
Based on the instructions, DACA eligibility requires:
1. Arrived unlawfully in the US before 16th birthday
2. Not yet 31 years old as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (at least 5 years by 2012)
4. Present in US on June 15, 2012 and no lawful status
5. Assume non-citizens without immigration papers are undocumented

**Operationalization:**
- HISPAN == 1 (Mexican Hispanic)
- BPL == 200 (Born in Mexico)
- CITIZEN == 3 (Not a citizen)
- YRIMMIG > 0 (has immigration year)
- Age at arrival < 16 years (YRIMMIG - BIRTHYR < 16)
- Born after June 15, 1981 (not yet 31 on June 15, 2012)
- In US at least 5 years by 2012 (YRIMMIG <= 2007)

### Step 3: Empirical Strategy - Difference-in-Differences
**Design:**
- Treatment Group: DACA-eligible Mexican-born Hispanic non-citizens
- Control Group: Similar Mexican-born Hispanic non-citizens who are NOT DACA-eligible (due to age cutoff)
- Pre-period: 2006-2011 (before DACA)
- Post-period: 2013-2016 (after DACA; excluding 2012 as implementation year)

**Outcome:** Full-time employment = 1 if UHRSWORK >= 35, 0 otherwise

**Model Specification:**
Y_it = α + β₁(DACA_eligible_i) + β₂(Post_t) + β₃(DACA_eligible_i × Post_t) + X_it'γ + ε_it

Where β₃ is the difference-in-differences estimate of DACA's effect.

### Step 4: Data Processing
Processing data in chunks due to size (~34M rows)...

**Filtering Steps:**
1. Filtered to Mexican Hispanic (HISPAN==1), Mexican-born (BPL==200), non-citizens (CITIZEN==3)
2. Result: 701,347 observations from 33.8M total
3. Constructed DACA eligibility: 130,276 eligible (18.6%)
4. Restricted to working-age (18-50): 515,666 observations
5. Excluded 2012 (implementation year): 468,582 final observations

### Step 5: Main Results

**Sample Composition:**
- Non-DACA eligible: 398,231 observations (weighted: 55.3M)
- DACA eligible: 70,351 observations (weighted: 9.7M)

**Key Findings:**

| Model | DiD Estimate | Std. Error | P-value |
|-------|-------------|------------|---------|
| Basic DiD | 0.0671 | 0.0050 | <0.001 |
| With Controls | 0.0407 | 0.0047 | <0.001 |
| With State FE | 0.0404 | 0.0047 | <0.001 |
| With Year FE | 0.0310 | 0.0047 | <0.001 |

**Preferred Estimate:** Basic DiD = 0.0671 (6.71 percentage points)
- 95% CI: [0.0573, 0.0768]
- Interpretation: DACA eligibility is associated with a 6.71 percentage point increase in full-time employment

### Step 6: Robustness Checks

| Specification | Estimate | SE |
|--------------|----------|-----|
| Age 16-45 | 0.0973 | 0.0048 |
| Employed (any hours) | 0.0680 | 0.0043 |
| In labor force | 0.0641 | 0.0044 |
| Men only | 0.0518 | 0.0060 |
| Women only | 0.0568 | 0.0072 |

### Step 7: Event Study Analysis
Pre-trend coefficients (relative to 2011) show no significant differences before DACA:
- 2006-2010: All coefficients within [-0.01, 0.02], none significant at 5%
- Post-DACA: Coefficients increase from 0.019 (2013) to 0.060 (2016)

### Key Decisions Made

1. **Sample Definition:** Restricted to Mexican Hispanic, Mexican-born, non-citizens as closest proxy for DACA-eligible population
2. **Age Restriction:** 18-50 to focus on prime working-age population
3. **Excluding 2012:** Removed implementation year to avoid contamination
4. **Control Group:** Used age-ineligible non-citizens (too old for DACA) rather than citizens
5. **Weights:** Used PERWT for population-representative estimates
6. **Standard Errors:** Heteroskedasticity-robust (HC1)

### Output Files Generated
- `analysis.py` - Main analysis script
- `results_table.csv` - Summary of regression results
- `results_summary.json` - Detailed results in JSON format
- `event_study_coefs.csv` - Event study coefficients
- `fulltime_by_year.csv` - Yearly employment rates by group
- `figure1_trends.png` - Trends visualization
- `figure2_event_study.png` - Event study plot

