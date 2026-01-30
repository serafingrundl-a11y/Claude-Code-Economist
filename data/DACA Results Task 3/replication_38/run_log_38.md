# Replication Run Log - Replication 38

## Overview
This log documents all key decisions and commands executed during the independent replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Study Design
- **Treatment Group**: Individuals aged 26-30 at the time of DACA implementation (June 15, 2012)
- **Control Group**: Individuals aged 31-35 at the time of DACA implementation
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016
- **Method**: Difference-in-Differences

## Data Summary
- Data file: `prepared_data_numeric_version.csv`
- Total observations: 17,382 (excluding header)
- Key variables:
  - `FT`: Full-time employment indicator (1 = 35+ hours/week, 0 = otherwise)
  - `AFTER`: Post-DACA period indicator (1 = 2013-2016, 0 = 2008-2011)
  - `ELIGIBLE`: Treatment group indicator (1 = aged 26-30, 0 = aged 31-35)
  - `PERWT`: Person weight for survey estimates

## Session Log

### Session Start: 2026-01-27

#### Step 1: Read Replication Instructions
- Extracted text from `replication_instructions.docx`
- Confirmed research question and design parameters
- Noted that IPUMS binary variables use 1=No, 2=Yes coding
- Custom variables (FT, AFTER, ELIGIBLE) use 0=No, 1=Yes coding

#### Step 2: Examine Data Structure
- Inspected column headers from prepared_data_numeric_version.csv
- Confirmed presence of all key variables
- Total observations: 17,382 individuals across years 2008-2011 and 2013-2016
- 2012 data excluded as instructed (cannot determine pre/post treatment)

#### Step 3: Exploratory Data Analysis
Executed Python analysis script. Key findings:

**Sample Distribution:**
- Pre-DACA observations: 9,527
- Post-DACA observations: 7,855
- Treatment group (ELIGIBLE=1): 11,382
- Control group (ELIGIBLE=0): 6,000

**Yearly Sample Sizes:**
| Year | N |
|------|---|
| 2008 | 2,354 |
| 2009 | 2,379 |
| 2010 | 2,444 |
| 2011 | 2,350 |
| 2013 | 2,124 |
| 2014 | 2,056 |
| 2015 | 1,850 |
| 2016 | 1,825 |

**Overall Full-Time Employment Rate:** 64.91%

**Age Verification:**
- Treatment group mean age at June 2012: 28.1 years (range: 26-30.75)
- Control group mean age at June 2012: 32.9 years (range: 31-35)

#### Step 4: Analytic Decisions

**Decision 1: Use survey weights (PERWT)**
- Rationale: ACS is a complex survey; weights needed for population-representative estimates

**Decision 2: Include year and state fixed effects**
- Rationale: Control for aggregate time trends and state-level differences

**Decision 3: Include individual-level covariates**
- Covariates included: sex (female indicator), marital status (married indicator), age, education dummies
- Rationale: Improve precision and comparability between groups

**Decision 4: Use clustered standard errors at state level**
- Rationale: Account for within-state correlation in outcomes

**Decision 5: Do not drop observations**
- As per instructions: "This entire file constitutes the intended analytic sample for your analysis; do not further limit the sample"

#### Step 5: Main Analysis Results

**Simple Difference-in-Differences (Unweighted):**
- Treatment Pre: 0.6263
- Treatment Post: 0.6658
- Treatment Change: +0.0394
- Control Pre: 0.6697
- Control Post: 0.6449
- Control Change: -0.0248
- DiD Estimate: 0.0643

**Regression Results Summary:**

| Model | DiD Estimate | SE | 95% CI | p-value |
|-------|--------------|-----|--------|---------|
| Basic (Unweighted) | 0.0643 | 0.0153 | [0.034, 0.094] | <0.001 |
| Basic (Weighted) | 0.0748 | 0.0152 | [0.045, 0.105] | <0.001 |
| Year FE (Weighted) | 0.0721 | 0.0151 | [0.042, 0.102] | <0.001 |
| Year + State FE (Weighted) | 0.0710 | 0.0152 | [0.041, 0.101] | <0.001 |
| Full Model with Covariates | 0.0592 | 0.0142 | [0.031, 0.087] | <0.001 |
| Robust SE (HC1) | 0.0592 | 0.0166 | [0.027, 0.092] | <0.001 |
| State-Clustered SE | 0.0592 | 0.0211 | [0.018, 0.101] | 0.005 |

#### Step 6: Pre-Trends Analysis
- Event study coefficients (relative to 2011):
  - 2008: -0.067 (p=0.035)
  - 2009: -0.047 (p=0.154)
  - 2010: -0.076 (p=0.020)
  - 2013: 0.017 (p=0.610)
  - 2014: -0.017 (p=0.628)
  - 2015: -0.011 (p=0.745)
  - 2016: 0.059 (p=0.093)

- Pre-trend coefficients suggest some pre-existing differences in trends between treatment and control groups, which warrants caution in interpretation

#### Step 7: Placebo Test
- Fake treatment in 2010 (pre-period only): DiD = 0.0182 (p=0.413)
- No significant placebo effect, supporting the validity of the design

#### Step 8: Heterogeneity Analysis

**By Sex:**
- Males: DiD = 0.061 (SE=0.020, p=0.002)
- Females: DiD = 0.041 (SE=0.027, p=0.129)

**By Education:**
- HS or Less: DiD = 0.046 (SE=0.019, p=0.018)
- Some College+: DiD = 0.098 (SE=0.032, p=0.002)

## Preferred Estimate

**Model:** Full model with year and state fixed effects, individual covariates, survey weights, and state-clustered standard errors

**Effect Size:** 0.0592 (5.92 percentage points)

**Standard Error:** 0.0211

**95% Confidence Interval:** [0.0177, 0.1006]

**p-value:** 0.005

**Sample Size:** 17,382

## Interpretation
DACA eligibility is associated with a statistically significant 5.9 percentage point increase in the probability of full-time employment among the treatment group (ages 26-30 at implementation) relative to the comparison group (ages 31-35), controlling for year and state fixed effects and individual characteristics. This effect is statistically significant at the 0.5% level.

## Key Files
- `analysis_script.py`: Main Python analysis script
- `replication_report_38.tex`: LaTeX replication report
- `replication_report_38.pdf`: Compiled PDF report
- `run_log_38.md`: This log file
