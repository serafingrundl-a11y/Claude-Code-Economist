# Replication Run Log - Replication 37

## Session Start
Date: 2026-01-27

## Task Overview
Replicate analysis examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican Mexican-born individuals in the United States.

## Research Design
- **Treatment Group**: DACA-eligible individuals aged 26-30 on June 15, 2012
- **Control Group**: Similar individuals aged 31-35 on June 15, 2012 (ineligible due to age cutoff)
- **Outcome**: Full-time employment (FT = 1 if usually working 35+ hours/week)
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016 (2012 excluded since treatment timing unclear)

## Data Files
- `prepared_data_labelled_version.csv` - Main analysis file with labels
- `prepared_data_numeric_version.csv` - Numeric version (used for analysis)
- `acs_data_dict.txt` - IPUMS data dictionary

## Key Variables (from instructions and data):
- **ELIGIBLE**: 1 = treatment group (ages 26-30), 0 = control group (ages 31-35)
- **FT**: 1 = full-time work (35+ hrs/week), 0 = not full-time
- **AFTER**: 1 = post-DACA years (2013-2016), 0 = pre-DACA years (2008-2011)
- **PERWT**: Person weights for population-representative estimates
- **SEX**: 1 = Male, 2 = Female (IPUMS coding)
- **MARST**: 1 = Married spouse present (IPUMS coding)
- **EDUC**: Educational attainment (general version)
- **NCHILD**: Number of own children in household
- **STATEFIP**: State FIPS code
- **YEAR**: Survey year

## Commands and Decisions Log

### Step 1: Read and understand data dictionary
- Extracted text from replication_instructions.docx using python-docx
- Reviewed IPUMS variable documentation in acs_data_dict.txt
- Key notes: Binary IPUMS variables coded 1=No, 2=Yes; Added variables (FT, AFTER, ELIGIBLE) coded 0=No, 1=Yes

### Step 2: Explore data structure
Command: `pd.read_csv('data/prepared_data_numeric_version.csv')` with exploration
- Dataset has 17,382 observations and 105 variables
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-DACA (AFTER=0): 9,527 observations
- Post-DACA (AFTER=1): 7,855 observations

### Step 3: Key Analytic Decisions

**Decision 1: Use survey weights (PERWT)**
- Rationale: PERWT allows estimates to be representative of the population
- Survey weights account for sampling design and non-response

**Decision 2: Create education dummy variables from EDUC**
- Created: EDUC_HS (high school, codes 6-7), EDUC_SOMECOLL (some college, codes 8-9), EDUC_BA (BA+, codes 10+)
- Reference category: Less than high school (codes 0-5)
- Note: Very few observations have less than HS in this sample

**Decision 3: Include covariates in regression**
- Demographics: FEMALE (derived from SEX==2), MARRIED (derived from MARST==1), NCHILD
- Education: EDUC_HS, EDUC_SOMECOLL, EDUC_BA
- Fixed effects: Year dummies, State dummies

**Decision 4: Use heteroskedasticity-robust standard errors (HC3)**
- Rationale: Robust to heteroskedasticity in linear probability model

**Decision 5: Preferred specification includes year and state fixed effects**
- Year FE: Controls for common time trends affecting both groups
- State FE: Controls for time-invariant state-level differences

### Step 4: Model Estimation

#### Model 1: Basic DiD (Unweighted OLS)
```
FT = b0 + b1*ELIGIBLE + b2*AFTER + b3*ELIGIBLE*AFTER + e
```
Result: DiD = 0.0643 (SE = 0.0153), p < 0.001

#### Model 2: Basic DiD (Survey Weighted WLS)
Same specification, weighted by PERWT
Result: DiD = 0.0748 (SE = 0.0181), p < 0.001

#### Model 3: DiD with Demographic Controls (Weighted)
Added: FEMALE, MARRIED, NCHILD, education dummies
Result: DiD = 0.0636 (SE = 0.0168), p < 0.001

#### Model 4: DiD with Year Fixed Effects (Weighted)
Added: Year dummies (reference: 2008)
Result: DiD = 0.0609 (SE = 0.0167), p < 0.001

#### Model 5: DiD with Year + State Fixed Effects (PREFERRED)
Added: State dummies
Result: DiD = 0.0604 (SE = 0.0168), p < 0.001

### Step 5: Robustness Checks

#### Subgroup Analysis by Sex
- Males: DiD = 0.0604 (SE = 0.0198)
- Females: DiD = 0.0517 (SE = 0.0275)
- Both positive and significant

#### Placebo Test (Pre-trends)
- Used only pre-period data (2008-2011)
- Fake treatment at 2010
- Placebo DiD = 0.0183 (SE = 0.0224), p = 0.415
- Insignificant placebo supports parallel trends assumption

#### Event Study Analysis
- Year-specific treatment effects (reference: 2011)
- Pre-period effects: 2008 (-0.065), 2009 (-0.049), 2010 (-0.075) - negative/around zero
- Post-period effects: 2013 (+0.016), 2014 (-0.015), 2015 (-0.008), 2016 (+0.064)
- No clear pre-trend; effects emerge post-DACA with 2016 showing strongest effect

### Step 6: Final Results Summary

**Preferred Estimate:**
- Effect Size: 0.0604 (6.04 percentage points)
- Standard Error: 0.0168
- 95% CI: [0.0275, 0.0932]
- Sample Size: 17,382

**Interpretation:**
DACA eligibility is associated with a 6.04 percentage point increase in the probability of full-time employment among eligible Hispanic-Mexican Mexican-born individuals aged 26-30, compared to similar individuals aged 31-35 who were ineligible due to the age cutoff. This effect is statistically significant at the 1% level.

## Output Files Generated
1. `parallel_trends.png` - Visualization of FT rates by group over time
2. `event_study.png` - Event study plot showing year-specific effects
3. `replication_report_37.tex` - LaTeX source for report
4. `replication_report_37.pdf` - Final PDF report

## Software Used
- Python 3.14 with pandas, numpy, statsmodels, matplotlib
- LaTeX (pdflatex) for document compilation
