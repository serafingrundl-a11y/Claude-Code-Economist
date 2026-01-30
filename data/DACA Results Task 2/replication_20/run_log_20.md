# Replication Run Log - Run 20

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

## Identification Strategy
- Treatment group: Ages 26-30 as of June 15, 2012 (eligible for DACA)
- Control group: Ages 31-35 as of June 15, 2012 (would be eligible but for age)
- Method: Difference-in-Differences comparing pre-DACA (2006-2011) to post-DACA (2013-2016)
- 2012 excluded due to uncertainty about pre/post treatment timing

---

## Session Log

### Step 1: Read Replication Instructions
- Read replication_instructions.docx
- Key requirements identified:
  - DACA implemented June 15, 2012
  - Treatment: Ages 26-30 at time of policy
  - Control: Ages 31-35 at time of policy
  - Outcome: Full-time employment (35+ hours/week)
  - Sample: Hispanic-Mexican, born in Mexico, non-citizens
  - Additional eligibility: arrived before age 16, in US since June 2007

### Step 2: Explore Data Structure
- Data file: data.csv (6.2 GB)
- Data dictionary: acs_data_dict.txt
- Years available: 2006-2016 ACS 1-year files
- Key variables identified:
  - YEAR: Survey year
  - AGE: Age at time of survey
  - BIRTHYR: Birth year
  - BIRTHQTR: Birth quarter
  - HISPAN/HISPAND: Hispanic origin (1 = Mexican)
  - BPL/BPLD: Birthplace (200 = Mexico)
  - CITIZEN: Citizenship status (3 = Not a citizen)
  - YRIMMIG: Year of immigration
  - UHRSWORK: Usual hours worked per week
  - EMPSTAT: Employment status
  - PERWT: Person weight

### Step 3: Define DACA Eligibility Criteria
Per instructions, DACA eligibility requires:
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Not a citizen (CITIZEN == 3)
4. Arrived in US before 16th birthday (YRIMMIG - BIRTHYR < 16)
5. In US since at least June 2007 (YRIMMIG <= 2007)
6. Age-based treatment assignment:
   - Treatment: Ages 26-30 as of June 15, 2012 (birth year 1982-1986)
   - Control: Ages 31-35 as of June 15, 2012 (birth year 1977-1981)

### Step 4: Create and Run Analysis Script (analysis.py)
- Created Python script with pandas, numpy, statsmodels
- Data loading: 33,851,424 total rows loaded from ACS data
- Sample restrictions applied sequentially:
  1. Hispanic-Mexican (HISPAN == 1): 2,945,521 observations
  2. Born in Mexico (BPL == 200): 991,261 observations
  3. Non-citizen (CITIZEN == 3): 701,347 observations
  4. Birth year 1977-1986: 178,376 observations
  5. Arrived before age 16: 49,019 observations
  6. Arrived by 2007: 49,019 observations (same, all met criterion)
  7. Excluded 2012: 44,725 final observations

### Step 5: Key Analysis Results

#### Sample Composition:
- Treatment group (ages 26-30): 26,591 observations
- Control group (ages 31-35): 18,134 observations
- Pre-DACA period (2006-2011): 29,326 observations
- Post-DACA period (2013-2016): 15,399 observations

#### Main DiD Results:

| Model | DiD Estimate | Robust SE | p-value | N |
|-------|--------------|-----------|---------|---|
| Basic DiD | 0.0620 | 0.0116 | 0.0000 | 44,725 |
| + Demographics | 0.0488 | 0.0106 | 0.0000 | 44,725 |
| + Education | 0.0464 | 0.0106 | 0.0000 | 44,725 |
| + Year FE | 0.0448 | 0.0106 | 0.0000 | 44,725 |
| + State & Year FE (Preferred) | 0.0441 | 0.0105 | 0.0000 | 44,725 |

#### Preferred Specification (Model 5):
- **Effect Size: 0.0441** (4.4 percentage point increase in full-time employment)
- **Standard Error: 0.0105**
- **95% CI: [0.0235, 0.0648]**
- **p-value: 0.0000**
- **R-squared: 0.1604**

#### Event Study Results (Reference: 2011):
| Year | Coefficient | Std. Error | p-value |
|------|-------------|------------|---------|
| 2006 | 0.0085 | 0.0225 | 0.7074 |
| 2007 | -0.0094 | 0.0221 | 0.6722 |
| 2008 | 0.0209 | 0.0226 | 0.3534 |
| 2009 | 0.0127 | 0.0232 | 0.5852 |
| 2010 | 0.0173 | 0.0229 | 0.4516 |
| 2013 | 0.0466 | 0.0239 | 0.0512 |
| 2014 | 0.0541 | 0.0243 | 0.0263 |
| 2015 | 0.0338 | 0.0243 | 0.1643 |
| 2016 | 0.0788 | 0.0244 | 0.0012 |

**Key Finding:** Pre-treatment coefficients are small and statistically insignificant, supporting parallel trends assumption.

#### Subgroup Analysis:
- **Male:** DiD = 0.0621 (SE: 0.0124, p < 0.001) - Significant
- **Female:** DiD = 0.0313 (SE: 0.0182, p = 0.086) - Marginally significant
- **Less than HS:** DiD = 0.0458 (SE: 0.0179, p = 0.011) - Significant
- **HS Graduate:** DiD = 0.0460 (SE: 0.0179, p = 0.010) - Significant
- **Some College:** DiD = 0.1181 (SE: 0.0324, p < 0.001) - Highly significant
- **College+:** DiD = 0.2331 (SE: 0.0644, p < 0.001) - Largest effect

#### Robustness Checks:
- **Placebo test** (fake treatment in 2009, pre-period only): DiD = 0.0120 (p = 0.375) - Not significant, as expected
- **Narrower age bands** (27-29 vs 32-34): DiD = 0.0465 (SE: 0.0148) - Consistent with main results

### Step 6: Generate Figures
- Created 4 figures for the report:
  1. Event study plot (figure1_event_study.png/pdf)
  2. Employment trends by group (figure2_trends.png/pdf)
  3. Model comparison (figure3_model_comparison.png/pdf)
  4. Subgroup analysis (figure4_subgroups.png/pdf)

### Step 7: Write LaTeX Report
- Created comprehensive ~20-page replication report
- Includes: abstract, introduction, data, methods, results, robustness checks, conclusion

### Step 8: Compile to PDF
- Compiled LaTeX to PDF using pdflatex

---

## Key Analytical Decisions

1. **Sample Definition:** Restricted to Hispanic-Mexican, Mexico-born, non-citizen individuals who arrived before age 16 and by 2007 (proxy for continuous presence requirement)

2. **Age/Treatment Assignment:** Used birth year to define treatment (1982-1986) and control (1977-1981) groups, corresponding to ages 26-30 and 31-35 as of June 15, 2012

3. **Outcome Variable:** Full-time employment defined as usual hours worked >= 35 per week (UHRSWORK >= 35)

4. **Exclusion of 2012:** Dropped 2012 observations because treatment timing within year is ambiguous (DACA implemented mid-year)

5. **Weights:** Used PERWT (person weights) for all analyses to ensure population representativeness

6. **Standard Errors:** Used heteroskedasticity-robust (HC1) standard errors

7. **Fixed Effects:** Preferred specification includes state and year fixed effects to control for time-invariant state characteristics and common time trends

---

## Files Generated
- analysis.py - Main analysis script
- create_figures.py - Figure generation script
- regression_results.csv - Main regression results
- event_study_results.csv - Event study coefficients
- final_statistics.txt - Summary statistics
- figure1_event_study.png/pdf - Event study plot
- figure2_trends.png/pdf - Employment trends
- figure3_model_comparison.png/pdf - Model comparison
- figure4_subgroups.png/pdf - Subgroup analysis
- replication_report_20.tex - LaTeX report
- replication_report_20.pdf - Final PDF report
- run_log_20.md - This log file
