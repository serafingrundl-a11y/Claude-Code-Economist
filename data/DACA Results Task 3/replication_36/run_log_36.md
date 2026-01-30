# Replication Run Log - DACA Full-Time Employment Analysis

## Session Information
- Date: 2026-01-27
- Task: Independent replication of DACA effects on full-time employment
- Replication ID: 36

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Study Design
- **Treatment Group**: Eligible individuals aged 26-30 at time of DACA implementation (June 15, 2012)
- **Control Group**: Individuals aged 31-35 at time of DACA implementation (otherwise would have been eligible but for age)
- **Pre-treatment Period**: 2008-2011
- **Post-treatment Period**: 2013-2016 (2012 excluded as ambiguous)
- **Method**: Difference-in-Differences (DiD)

## Key Variables (as provided in data)
- `ELIGIBLE`: 1 = treatment group (ages 26-30 in June 2012), 0 = comparison group (ages 31-35 in June 2012)
- `AFTER`: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)
- `FT`: 1 = full-time employment (35+ hours/week), 0 = not full-time employed
- `PERWT`: Person-level survey weights from ACS

## Steps Executed

### Step 1: Read replication instructions
- Loaded replication_instructions.docx using Python docx library
- Confirmed research question and study design requirements
- Noted data files in data/ folder

### Step 2: Examine data dictionary
- Reviewed acs_data_dict.txt for variable definitions (3,851 lines)
- Noted IPUMS coding conventions (1=No, 2=Yes for binary variables from IPUMS)
- Custom variables (FT, AFTER, ELIGIBLE) coded 0=No, 1=Yes
- Identified key demographic variables: SEX, AGE, MARST, EDUC, STATEFIP

### Step 3: Preview data files
- prepared_data_labelled_version.csv - contains labelled values
- prepared_data_numeric_version.csv - contains numeric codes (used for analysis)
- Both have 105 columns including key analysis variables
- Verified YEAR values: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (no 2012)

### Step 4: Load and explore data
**Command:** `python analysis.py`

Key findings from data exploration:
- Total observations: 17,382
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-period observations: 9,527
- Post-period observations: 7,855

Age verification:
- Treatment group ages 26-30.75 in June 2012 (mean: 28.1)
- Control group ages 31-35 in June 2012 (mean: 32.9)

### Step 5: Perform difference-in-differences analysis
**Command:** `python analysis.py`

Estimated multiple model specifications:
1. Basic DiD (Unweighted): 0.0643 (SE=0.0153)
2. Basic DiD (Weighted): 0.0748 (SE=0.0152)
3. Year Fixed Effects: 0.0721 (SE=0.0151)
4. State Fixed Effects: 0.0737 (SE=0.0152)
5. Year + State FE: 0.0710 (SE=0.0152) **PREFERRED**
6. With Demographics: 0.0608 (SE=0.0142)
7. With State Policies: 0.0600 (SE=0.0142)

Standard error robustness:
- Clustered SE (State): 0.0202, p=0.0009
- Robust SE (HC1): 0.0180, p<0.0001

### Step 6: Generate figures and additional analyses
**Command:** `python generate_figures.py`

Generated files:
- figure1_parallel_trends.png/pdf - Time trends by eligibility
- figure2_did_visual.png/pdf - DiD visualization
- figure3_event_study.png/pdf - Event study with year-specific effects
- figure4_by_sex.png/pdf - Heterogeneity by sex
- figure5_by_education.png/pdf - Heterogeneity by education
- summary_statistics.csv
- placebo_results.txt
- subgroup_results.csv

Placebo test result (pre-treatment period only):
- Coefficient: 0.0203
- SE: 0.0205
- p-value: 0.324 (not significant, supporting parallel trends)

Subgroup results:
- Male: 0.0700 (SE=0.0171)
- Female: 0.0477 (SE=0.0234)
- Married: 0.0514 (SE=0.0213)
- Not Married: 0.0935 (SE=0.0219)

### Step 7: Write LaTeX report
Created replication_report_36.tex with:
- Abstract
- Introduction
- Background on DACA
- Data description
- Empirical strategy
- Main results
- Parallel trends assessment
- Robustness checks
- Heterogeneity analysis
- Discussion
- Conclusion
- Appendices

### Step 8: Compile PDF
**Commands:**
```
pdflatex replication_report_36.tex
pdflatex replication_report_36.tex (second pass for references)
pdflatex replication_report_36.tex (final pass)
```

Output: replication_report_36.pdf (22 pages)

## Key Decisions

### Decision 1: Preferred Specification
**Choice:** Model 5 - Year and State Fixed Effects with Survey Weights

**Rationale:**
- Year fixed effects control for common time trends affecting both groups
- State fixed effects control for time-invariant state differences
- Survey weights ensure population representativeness
- This specification balances control for confounders without overfitting

### Decision 2: Standard Errors
**Choice:** Report conventional SE as main, with clustered and robust as robustness

**Rationale:**
- State-level clustering accounts for within-state correlation
- All methods yield qualitatively similar conclusions
- Main result significant under all SE approaches

### Decision 3: Sample Definition
**Choice:** Use provided ELIGIBLE variable without modification

**Rationale:**
- Instructions explicitly state to use the provided ELIGIBLE variable
- Verified that age groups match specification (26-30 vs 31-35)

### Decision 4: Labor Force Participation
**Choice:** Include individuals not in labor force (as FT=0)

**Rationale:**
- Instructions specify to include those not in labor force
- This captures full employment effect including labor force entry

## Main Results

### Preferred Estimate (Model 5)
- **DiD Coefficient:** 0.0710
- **Standard Error:** 0.0152
- **95% CI:** [0.0413, 0.1007]
- **t-statistic:** 4.68
- **p-value:** <0.001
- **Sample Size:** 17,382

### Interpretation
DACA eligibility increased full-time employment by approximately 7.1 percentage points among the treatment group. Given a pre-treatment baseline of 63.7% for the treatment group, this represents an 11.1% relative increase.

## Files Produced

### Analysis Scripts
- analysis.py - Main DiD regression analysis
- generate_figures.py - Figures and additional analyses

### Output Files
- results_summary.csv - Summary of all model estimates
- summary_statistics.csv - Descriptive statistics by group
- subgroup_results.csv - Heterogeneity analysis
- placebo_results.txt - Placebo test results

### Figures
- figure1_parallel_trends.png/pdf
- figure2_did_visual.png/pdf
- figure3_event_study.png/pdf
- figure4_by_sex.png/pdf
- figure5_by_education.png/pdf

### Final Deliverables
- replication_report_36.tex - LaTeX source
- replication_report_36.pdf - Final report (22 pages)
- run_log_36.md - This log file

## Software Environment
- Python 3.x
- pandas - Data manipulation
- numpy - Numerical operations
- statsmodels - Regression analysis
- matplotlib - Visualization
- pdflatex (MiKTeX) - LaTeX compilation
