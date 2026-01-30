# Run Log for DACA Replication Study - Run 09

## Project Overview
Independent replication of DACA's causal impact on full-time employment among ethnically Hispanic-Mexican, Mexican-born people in the United States.

**Research Question:** What was the causal impact of DACA eligibility on the probability of full-time employment (35+ hours/week) among eligible individuals?

**Design:** Difference-in-differences comparing:
- Treatment group: Ages 26-30 at policy implementation (June 15, 2012)
- Control group: Ages 31-35 at policy implementation (would have been eligible if younger)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (excluding 2012 due to mid-year implementation)

---

## Session Log

### Session Start: 2025-01-25

#### Step 1: Data Exploration
- **Data source:** ACS data from IPUMS (2006-2016)
- **Main data file:** data.csv (6.26 GB)
- **Data dictionary:** acs_data_dict.txt
- **State-level data:** state_demo_policy.csv (optional)

#### Key Variables Identified:
- **YEAR**: Survey year (2006-2016)
- **HISPAN/HISPAND**: Hispanic origin (1 = Mexican)
- **BPL/BPLD**: Birthplace (200 = Mexico)
- **CITIZEN**: Citizenship status (3 = Not a citizen)
- **YRIMMIG**: Year of immigration
- **BIRTHYR**: Year of birth
- **BIRTHQTR**: Quarter of birth
- **AGE**: Age at survey time
- **UHRSWORK**: Usual hours worked per week (outcome: >= 35 is full-time)
- **EMPSTAT**: Employment status
- **PERWT**: Person weight for population estimates

#### Step 2: Sample Selection Criteria
Per instructions, eligible individuals must:
1. Be ethnically Hispanic-Mexican (HISPAN = 1 or HISPAND in 100-107)
2. Be born in Mexico (BPL = 200)
3. Not be a citizen (CITIZEN = 3)
4. Have arrived before age 16
5. Have lived continuously in US since June 15, 2007 (arrived by 2007)

Age groups at policy implementation (June 15, 2012):
- Treatment: Born between June 16, 1981 and June 15, 1986 (ages 26-30)
- Control: Born between June 16, 1976 and June 15, 1981 (ages 31-35)

#### Step 3: Analysis Plan
1. Load data and filter to target population
2. Create treatment indicator (ages 26-30 vs 31-35 at policy time)
3. Create post-treatment indicator (2013-2016 vs 2006-2011)
4. Define outcome: full-time employment (UHRSWORK >= 35)
5. Run difference-in-differences regression with controls
6. Compute robust standard errors
7. Generate tables and figures

---

## Commands and Decisions Log

### Data Loading and Cleaning

**Command:** `python analysis.py`

**Key Output:**
- Total ACS records loaded: 33,851,424
- Hispanic-Mexican individuals: 2,945,521
- Born in Mexico: 1,020,945
- Not a citizen: 1,909,225
- Base eligible (Mexican-born non-citizen): 701,347
- Met DACA criteria (except age): 195,023
- Final analysis sample (ages 26-35, excl. 2012): 43,238

### Sample Construction

| Group | Pre-Period | Post-Period | Total |
|-------|------------|-------------|-------|
| Treatment (26-30) | 16,694 | 8,776 | 25,470 |
| Control (31-35) | 11,683 | 6,085 | 17,768 |
| **Total** | 28,377 | 14,861 | **43,238** |

### Main Results

**Model 1 (Basic DiD):**
- Coefficient: 0.0590
- SE: 0.0117
- p < 0.001

**Model 2 (DiD with demographics):**
- Coefficient: 0.0643
- SE: 0.0146
- p < 0.001

**Model 3 (DiD with year FE) - PREFERRED:**
- Coefficient: 0.0201
- SE: 0.0154
- 95% CI: [-0.010, 0.050]
- p = 0.192

**Model 4 (DiD with state and year FE):**
- Coefficient: 0.0191
- SE: 0.0154
- p = 0.215

### Event Study Results
Pre-trend coefficients (relative to 2011):
- 2006: 0.029 (SE: 0.025)
- 2007: -0.013 (SE: 0.024)
- 2008: 0.020 (SE: 0.024)
- 2009: -0.001 (SE: 0.024)
- 2010: -0.009 (SE: 0.023)

Post-period coefficients:
- 2013: 0.022 (SE: 0.025)
- 2014: 0.020 (SE: 0.025)
- 2015: -0.003 (SE: 0.026)
- 2016: 0.037 (SE: 0.027)

### Robustness Checks

1. **Narrower bandwidth (ages 27-29 vs 32-34):**
   - DiD: 0.0384 (SE: 0.0137), N=25,606

2. **Alternative outcome (any employment):**
   - DiD: 0.0498 (SE: 0.0139)

3. **By sex:**
   - Males: 0.0495 (SE: 0.0174)
   - Females: 0.0729 (SE: 0.0241)

---

## Key Decisions

1. **Sample Definition:**
   - Used Hispanic-Mexican ethnicity (HISPAN=1 or HISPAND 100-107)
   - Born in Mexico (BPL=200)
   - Non-citizen (CITIZEN=3)
   - Arrived before age 16 (YRIMMIG - BIRTHYR < 16)
   - Arrived by 2007 (YRIMMIG <= 2007)

2. **Age Calculation:**
   - Age at June 15, 2012 = 2012 - BIRTHYR, adjusted for birth quarter
   - If BIRTHQTR >= 3 (July onwards), subtracted 1

3. **Excluded 2012:**
   - DACA implemented mid-year (June 15, 2012)
   - Cannot distinguish pre/post observations within 2012

4. **Preferred Specification:**
   - Model 3 with year fixed effects and demographic controls
   - Controls: sex, marital status, age (quadratic), high school education

5. **Weights:**
   - Used PERWT for population-representative estimates
   - Robust (HC1) standard errors throughout

---

## Files Generated

- `analysis.py` - Main analysis script
- `results.json` - Key statistics for report
- `yearly_stats.csv` - Annual means by group
- `event_study.csv` - Event study coefficients
- `replication_report_09.tex` - LaTeX report
- `replication_report_09.pdf` - Final PDF report (22 pages)

---

## Summary of Findings

**Main Result (Preferred Specification - Model 3):**
- DiD Coefficient: 0.020 (2.0 percentage points)
- Standard Error: 0.015
- 95% CI: [-0.010, 0.050]
- P-value: 0.192

**Interpretation:**
The analysis suggests DACA eligibility is associated with a 2.0 percentage point increase in full-time employment among eligible Mexican-born non-citizens. However, this effect is not statistically significant at conventional levels after controlling for year fixed effects. The basic DiD estimate (5.9 pp, p<0.001) attenuates substantially with year fixed effects, indicating that some of the apparent effect reflects differential secular trends rather than DACA per se.

**Event Study:**
Pre-trend coefficients are small and not statistically significant, supporting the parallel trends assumption. Post-period effects are positive but mostly insignificant, with the largest effect in 2016 (3.7 pp).

**Robustness:**
- Narrower bandwidth (27-29 vs 32-34): 3.8 pp (SE=0.014)
- Alternative outcome (any employment): 5.0 pp (SE=0.014)
- Males: 5.0 pp (SE=0.017)
- Females: 7.3 pp (SE=0.024)

---

## Session End: 2025-01-25

All required deliverables generated:
1. replication_report_09.tex - LaTeX source file
2. replication_report_09.pdf - 22-page PDF report
3. run_log_09.md - This run log documenting commands and decisions
