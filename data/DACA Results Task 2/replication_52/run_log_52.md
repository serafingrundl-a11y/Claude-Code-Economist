# DACA Replication Study - Run Log 52

## Study Overview
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (≥35 hours/week)?

**Design:** Difference-in-Differences
- **Treatment Group:** Ages 26-30 at DACA implementation (June 15, 2012)
- **Control Group:** Ages 31-35 at DACA implementation
- **Pre-period:** 2006-2011 (excluding 2012 due to mid-year implementation)
- **Post-period:** 2013-2016

## Data Description
- **Source:** American Community Survey (ACS) via IPUMS USA
- **Years:** 2006-2016 (1-year ACS files)
- **Main file:** data.csv (~6GB)
- **Data dictionary:** acs_data_dict.txt

## Key Variables
- **YEAR**: Survey year
- **BIRTHYR**: Birth year (for age calculation)
- **HISPAN/HISPAND**: Hispanic origin (1 = Mexican)
- **BPL/BPLD**: Birthplace (200 = Mexico)
- **CITIZEN**: Citizenship status (3 = Not a citizen)
- **YRIMMIG**: Year of immigration (must be ≤1996 to enter before 16th birthday for oldest eligible)
- **UHRSWORK**: Usual hours worked per week (≥35 = full-time)
- **PERWT**: Person weight for population estimates

## Session Log

### Session Start: 2026-01-26

#### Step 1: Data Exploration
- Reviewed replication_instructions.docx
- Examined data dictionary (acs_data_dict.txt)
- Confirmed data file structure with 54 variables

#### Step 2: DACA Eligibility Criteria
Per instructions, DACA eligibility requires:
1. Arrived unlawfully in US before 16th birthday
2. Had not yet turned 31 as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

**Operationalization:**
- Hispanic-Mexican ethnicity: HISPAN = 1 (Mexican)
- Born in Mexico: BPL = 200
- Not a citizen: CITIZEN = 3
- Immigrated before age 16: YRIMMIG ≤ BIRTHYR + 15
- Immigrated by 2007: YRIMMIG ≤ 2007

**Age Groups:**
- Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)
- Control: Born 1977-1981 (ages 31-35 on June 15, 2012)

#### Step 3: Analysis Plan
1. Filter data to eligible population
2. Create treatment indicator (treatment = 1 if birthyr 1982-1986)
3. Create post indicator (post = 1 if year ≥ 2013)
4. Calculate full-time employment (fulltime = 1 if UHRSWORK ≥ 35)
5. Run difference-in-differences regression
6. Add robustness checks with covariates

#### Step 4: Data Processing (analysis.py)
- Loaded 6GB data file in chunks (500,000 rows per chunk)
- Initial filter: Hispanic-Mexican (HISPAN=1), born in Mexico (BPL=200), non-citizen (CITIZEN=3)
- Applied DACA eligibility criteria: immigrated before age 16 and by 2007
- Restricted to treatment (born 1982-1986) and control (born 1977-1981) groups
- Excluded year 2012
- **Final sample size: 44,725 observations**
  - Treatment group: 25,591
  - Control group: 19,134
  - Pre-period: 29,326
  - Post-period: 15,399

#### Step 5: Main Results

**Summary Statistics:**
| Group | Period | Full-time Rate | N |
|-------|--------|----------------|---|
| Control | Pre | 0.643 | 11,916 |
| Control | Post | 0.611 | 6,218 |
| Treatment | Pre | 0.611 | 17,410 |
| Treatment | Post | 0.634 | 9,181 |

**Raw DiD Calculation:**
- Treatment change: 0.634 - 0.611 = +0.023
- Control change: 0.611 - 0.643 = -0.032
- Raw DiD: 0.023 - (-0.032) = **0.055**

**Regression Results (DiD Coefficient on treat × post):**

| Model | Coefficient | SE | 95% CI | p-value |
|-------|-------------|-----|--------|---------|
| Basic (unweighted) | 0.055 | 0.010 | [0.036, 0.074] | <0.001 |
| Basic (weighted) | 0.062 | 0.012 | [0.039, 0.085] | <0.001 |
| With demographics | 0.048 | 0.011 | [0.027, 0.068] | <0.001 |
| With age controls | 0.065 | 0.015 | [0.036, 0.094] | <0.001 |
| Year FE | 0.046 | 0.011 | [0.025, 0.067] | <0.001 |
| Year + State FE | 0.045 | 0.011 | [0.025, 0.066] | <0.001 |

#### Step 6: Robustness Checks

**By Gender:**
- Males: 0.062 (SE=0.012), 95% CI [0.038, 0.086]
- Females: 0.031 (SE=0.018), 95% CI [-0.004, 0.067]

**Narrower Age Bandwidth (3 years each side):**
- Estimate: 0.052 (SE=0.017), 95% CI [0.019, 0.085]

**Placebo Test (Pre-2010 vs 2010-2011):**
- Estimate: 0.006 (SE=0.015), p=0.69 (not significant - supports parallel trends)

#### Step 7: Event Study Results

Pre-DACA coefficients (all insignificant, supporting parallel trends):
- 2006: -0.005 (p=0.827)
- 2007: -0.013 (p=0.580)
- 2008: 0.019 (p=0.452)
- 2009: 0.017 (p=0.503)
- 2010: 0.019 (p=0.450)

Post-DACA coefficients:
- 2013: 0.060 (p=0.023)*
- 2014: 0.070 (p=0.009)**
- 2015: 0.043 (p=0.108)
- 2016: 0.095 (p<0.001)***

#### Step 8: Figures Created
1. figure1_trends.pdf - Employment trends over time
2. figure2_eventstudy.pdf - Event study coefficients
3. figure3_did.pdf - DiD visualization
4. figure4_coefficients.pdf - Coefficient comparison across models
5. figure5_subgroups.pdf - Subgroup analysis

#### Step 9: Final Report
- Created replication_report_52.tex (LaTeX document)
- Compiled to replication_report_52.pdf (24 pages)

---

## Key Decisions and Justifications

1. **Excluded 2012**: DACA implemented mid-year (June 15, 2012) and ACS doesn't record month, so 2012 observations could be either pre or post.

2. **Age-at-immigration criterion**: Required YRIMMIG ≤ BIRTHYR + 15 to ensure arrived before 16th birthday.

3. **Continuous residence criterion**: Required YRIMMIG ≤ 2007 to ensure 5 years of continuous residence by June 2012.

4. **Non-citizen proxy for undocumented**: Used CITIZEN = 3 (not a citizen) as best available proxy since ACS doesn't identify undocumented status directly.

5. **Treatment/Control ages**: Followed instructions to use ages 26-30 vs 31-35 at implementation date.

6. **Weighting**: Used PERWT person weights for population-representative estimates.

7. **Preferred specification**: Weighted basic DiD (Model 2) - balances parsimony with representativeness.

---

## Final Results Summary

**Preferred Estimate:**
- Effect size: **0.062** (6.2 percentage points)
- Standard error: 0.012
- 95% CI: [0.039, 0.085]
- p-value: <0.001
- Sample size: 44,725

**Interpretation:** DACA eligibility increased the probability of full-time employment by approximately 6.2 percentage points among Hispanic-Mexican Mexican-born non-citizens aged 26-30 at implementation, relative to those aged 31-35.

---

## Files Generated

- analysis.py - Main analysis script
- create_figures.py - Figure generation script
- regression_results.csv - Model coefficients
- event_study_results.csv - Event study coefficients
- summary_statistics.csv - Summary statistics
- figure1_trends.pdf/png
- figure2_eventstudy.pdf/png
- figure3_did.pdf/png
- figure4_coefficients.pdf/png
- figure5_subgroups.pdf/png
- replication_report_52.tex - LaTeX source
- replication_report_52.pdf - Final report

