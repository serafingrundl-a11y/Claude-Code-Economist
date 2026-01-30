# Replication Run Log 49

## Overview
This log documents all commands, decisions, and key steps in the DACA replication analysis.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (defined as usually working 35+ hours per week)?

## Identification Strategy
- **Treated Group**: Ages 26-30 at time of policy implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at time of policy implementation (otherwise DACA-eligible)
- **Method**: Difference-in-Differences
- **Post-treatment period**: 2013-2016

## Data Sources
- American Community Survey (ACS) data from IPUMS USA (2006-2016)
- Main data file: data.csv
- Data dictionary: acs_data_dict.txt
- Optional state-level policy data: state_demo_policy.csv

---

## Session Log

### Step 1: Data Exploration
**Time**: Session start

**Files examined**:
1. `replication_instructions.docx` - Read and parsed research instructions
2. `acs_data_dict.txt` - Data dictionary with variable definitions
3. `data.csv` - Main ACS data file (2006-2016)
4. `state_demo_policy.csv` - Optional state-level policy variables

**Key Variables Identified**:
- `YEAR` - Survey year (2006-2016)
- `BIRTHYR` - Birth year
- `HISPAN` / `HISPAND` - Hispanic origin (1 = Mexican)
- `BPL` / `BPLD` - Birthplace (200 = Mexico)
- `CITIZEN` - Citizenship status (3 = Not a citizen)
- `YRIMMIG` - Year of immigration
- `UHRSWORK` - Usual hours worked per week
- `PERWT` - Person weight for population estimates
- `AGE` - Age at time of survey
- `SEX` - Gender
- `EDUC` / `EDUCD` - Educational attainment
- `STATEFIP` - State FIPS code

**Key Code Values**:
- HISPAN = 1: Mexican
- BPL = 200: Mexico
- CITIZEN = 3: Not a citizen
- UHRSWORK >= 35: Full-time employment

---

### Step 2: Define Sample Selection Criteria

**DACA Eligibility Requirements (from instructions)**:
1. Arrived unlawfully in the US before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

**Sample Restrictions for Analysis**:
1. Hispanic-Mexican ethnicity (HISPAN = 1)
2. Born in Mexico (BPL = 200)
3. Not a citizen (CITIZEN = 3) - proxy for undocumented status
4. Immigrated before age 16 (YRIMMIG - BIRTHYR < 16)
5. In US by 2007 (YRIMMIG <= 2007)

**Treatment/Control Assignment**:
- Treatment: Born 1982-1986 (ages 26-30 as of June 2012)
- Control: Born 1977-1981 (ages 31-35 as of June 2012)

**Time Periods**:
- Pre-treatment: 2006-2011
- Post-treatment: 2013-2016
- Excluded: 2012 (policy implementation year)

---

### Step 3: Analysis Plan

1. Load and filter data to eligible sample
2. Create treatment/control indicators
3. Create pre/post treatment indicators
4. Create outcome variable (full-time employment = UHRSWORK >= 35)
5. Run difference-in-differences regression
6. Include robustness checks with covariates
7. Generate summary statistics and visualizations

---

### Step 4: Data Processing and Analysis Execution

**Command executed**: `python analysis.py`

**Data Processing**:
- Total observations in raw ACS data (2006-2016, excluding 2012): 33,851,424
- After initial filtering (Hispanic-Mexican, Born Mexico, Non-citizen, Birth year 1977-1986): 162,283
- After arrived before age 16 restriction: 44,725
- After in US by 2007 restriction: 44,725 (Final sample)

**Sample Composition**:
- Treatment group (ages 26-30 as of 2012): 26,591 observations
- Control group (ages 31-35 as of 2012): 18,134 observations
- Pre-period observations (2006-2011): 29,326
- Post-period observations (2013-2016): 15,399

**Key Decisions Made**:
1. Used CITIZEN = 3 (not a citizen) as proxy for undocumented status
2. Required immigration before age 16 to satisfy DACA arrival requirement
3. Required immigration by 2007 to satisfy continuous residence requirement
4. Excluded 2012 as policy implementation year
5. Used person weights (PERWT) for population-representative estimates
6. Clustered standard errors by state for inference

---

### Step 5: Results Summary

**Raw Difference-in-Differences Calculation**:
- Treatment pre-period mean: 0.6111
- Treatment post-period mean: 0.6339
- Treatment change: +0.0228
- Control pre-period mean: 0.6431
- Control post-period mean: 0.6108
- Control change: -0.0323
- Raw DiD estimate: 0.0551

**Regression Results (Preferred Specification - Model 6)**:
- Model: WLS with year fixed effects, covariates, clustered SE by state
- DiD coefficient (treat_post): 0.0480
- Standard error: 0.0111
- 95% CI: [0.0262, 0.0697]
- p-value: 0.000015

**Interpretation**: DACA eligibility is associated with a 4.8 percentage point increase in the probability of full-time employment among eligible Mexican-born non-citizens, relative to the slightly older comparison group who were ineligible due to age.

**Robustness Checks**:
1. Narrower age bands (27-29 vs 32-34): coef = 0.0355, SE = 0.0117
2. By gender:
   - Males: coef = 0.0501, SE = 0.0124
   - Females: coef = 0.0344, SE = 0.0178
3. Pre-treatment parallel trends: No significant differential trends (all p > 0.28)
4. Event study: Pre-period coefficients near zero; post-period coefficients positive and increasing

---

### Step 6: Output Files Generated

1. `summary_statistics.csv` - Descriptive statistics by treatment status
2. `regression_results.csv` - Main regression results across specifications
3. `event_study.csv` - Year-by-year treatment effects
4. `did_means.csv` - 2x2 DiD means table
5. `sample_flow.csv` - Sample selection flowchart data
6. `yearly_means.csv` - Annual employment rates by treatment group
7. `gender_results.csv` - Heterogeneity analysis by gender
8. `analysis_output.txt` - Full analysis log

---

### Step 7: LaTeX Report Generation

**Command executed**: `pdflatex replication_report_49.tex` (3 passes)

**Report Structure**:
1. Abstract
2. Introduction
3. Background on DACA
4. Data (sources, sample construction, variable definitions, summary statistics)
5. Empirical Methodology (DiD framework, identification assumptions)
6. Results (raw DiD, regression results, robustness checks, event study)
7. Discussion
8. Conclusion
9. References
10. Appendix (additional tables, variable definitions, technical notes)

**Report Statistics**:
- 25 pages
- 7 tables (sample selection, summary statistics, main results, gender heterogeneity, pre-trends, event study, full model output)
- Complete documentation of methodology and results

---

### Step 8: Final Deliverables

**Required Output Files (all present in working directory)**:
1. `replication_report_49.tex` - LaTeX source file (37,510 bytes)
2. `replication_report_49.pdf` - Compiled PDF report (284,695 bytes, 25 pages)
3. `run_log_49.md` - This run log documenting all commands and decisions

**Supporting Analysis Files**:
- `analysis.py` - Main Python analysis script
- `summary_statistics.csv` - Descriptive statistics
- `regression_results.csv` - Regression results
- `event_study.csv` - Event study estimates
- `did_means.csv` - DiD means table
- `sample_flow.csv` - Sample construction
- `yearly_means.csv` - Year-by-year means
- `gender_results.csv` - Gender heterogeneity results
- `analysis_output.txt` - Full analysis output log

---

## Summary of Key Findings

**Main Result**: DACA eligibility increased full-time employment by 4.8 percentage points (95% CI: 2.6-7.0 pp)

**Key Methodological Choices**:
1. Used age-based eligibility cutoff for identification
2. Proxied undocumented status with non-citizenship
3. Applied DACA eligibility criteria to sample construction
4. Used weighted least squares with clustered standard errors
5. Verified parallel trends assumption with pre-trend tests

**Robustness**: Results stable across specifications and pass all diagnostic tests.

---

*End of Run Log*
