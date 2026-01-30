# Run Log - Replication 22: DACA Effects on Full-Time Employment

## Project Overview
- **Research Question**: What is the causal impact of DACA eligibility on full-time employment (35+ hours/week) among ethnically Hispanic-Mexican, Mexican-born individuals in the United States?
- **Treatment Period**: DACA implemented June 15, 2012; examine effects in 2013-2016
- **Data Source**: American Community Survey (ACS) 2006-2016

## Final Results Summary
- **Preferred Estimate**: 0.0327 (3.27 percentage points)
- **Standard Error**: 0.0035
- **95% Confidence Interval**: [0.0259, 0.0395]
- **p-value**: < 0.001
- **Sample Size**: 561,470 observations
- **Interpretation**: DACA eligibility increased full-time employment by 3.27 percentage points

---

## Session Log

### Step 1: Read Instructions and Understand Data
- Read `replication_instructions.docx` to understand research question and DACA eligibility criteria
- Examined data files:
  - `data/data.csv`: 33,851,424 observations, ACS 2006-2016
  - `data/acs_data_dict.txt`: Variable definitions from IPUMS
  - `data/state_demo_policy.csv`: Optional state-level data (not used)

### Step 2: Define Sample Selection Criteria
Applied sequential filters to construct analytic sample:
1. HISPAN = 1 (Mexican Hispanic ethnicity)
2. BPL = 200 (Born in Mexico)
3. CITIZEN = 3 (Non-citizen, assumed undocumented per instructions)
4. AGE 16-64 (Working age population)
5. YEAR != 2012 (Exclude transition year)
6. YRIMMIG > 0 (Valid immigration year for eligibility determination)

**Final sample**: 561,470 person-year observations

### Step 3: Define DACA Eligibility Criteria
An individual is coded as DACA-eligible if they meet ALL of:
1. Arrived in US before age 16: (YRIMMIG - BIRTHYR) < 16
2. Under 31 as of June 15, 2012: BIRTHYR >= 1982, OR (BIRTHYR = 1981 AND BIRTHQTR in {3, 4})
3. Present since June 15, 2007: YRIMMIG <= 2007

**DACA-eligible observations**: 83,611 (14.9% of sample)

### Step 4: Define Outcome Variable
- **Full-time employment**: UHRSWORK >= 35 (usual hours worked per week is 35 or more)
- 322,536 observations working full-time (57.4% of sample)

### Step 5: Estimation Strategy
- **Design**: Difference-in-Differences (DiD)
- **Treatment group**: DACA-eligible non-citizen Mexican-born Hispanic-Mexicans
- **Control group**: DACA-ineligible non-citizen Mexican-born Hispanic-Mexicans (age 31+ as of 2012)
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016

### Step 6: Run Analysis
Executed `analysis_22.py` which performs:
1. Data loading with memory-efficient chunked filtering
2. Variable construction
3. Descriptive statistics
4. Main regression models (5 specifications)
5. Robustness checks (5 tests)
6. Event study analysis

---

## Key Analytical Decisions

### Decision 1: Sample Restriction to Non-Citizens
**Choice**: Include only CITIZEN = 3 (not a citizen)
**Rationale**: Per instructions, non-citizens without immigration papers are assumed undocumented. Citizens are not eligible for DACA by definition.

### Decision 2: Age 31 Cutoff for Control Group
**Choice**: Control group defined by DACA ineligibility, primarily those born before 1981
**Rationale**: This is the natural comparison group based on DACA's age eligibility rule. The age cutoff provides quasi-random variation in eligibility.

### Decision 3: Excluding 2012
**Choice**: Drop all observations from 2012
**Rationale**: DACA implemented June 15, 2012; ACS doesn't identify month of interview, so cannot distinguish pre/post treatment within 2012.

### Decision 4: Immigration Year Requirement
**Choice**: Require YRIMMIG <= 2007 for DACA eligibility
**Rationale**: DACA requires continuous presence since June 15, 2007. Using immigration year as proxy for continuous presence.

### Decision 5: Standard Errors
**Choice**: Heteroskedasticity-robust (HC1) standard errors
**Rationale**: Linear probability model with binary outcome requires robust standard errors. HC1 provides valid inference under heteroskedasticity.

### Decision 6: Preferred Specification
**Choice**: Model with state and year fixed effects plus demographic and education controls
**Rationale**:
- Year FE controls for common macroeconomic shocks (e.g., post-recession recovery)
- State FE controls for time-invariant state-level labor market differences
- Demographic controls address compositional differences between treatment and control groups

---

## Regression Results Summary

### Main Results (All Models)
| Model | DiD Coef | SE | p-value | R-squared |
|-------|----------|-----|---------|-----------|
| (1) Basic | 0.0902 | 0.0038 | <0.001 | 0.011 |
| (2) Demographics | 0.0421 | 0.0035 | <0.001 | 0.207 |
| (3) + Education | 0.0387 | 0.0035 | <0.001 | 0.210 |
| (4) + State FE | 0.0382 | 0.0035 | <0.001 | 0.213 |
| (5) + Year FE | **0.0327** | **0.0035** | **<0.001** | 0.218 |

### Robustness Checks
| Test | DiD Coef | SE | p-value |
|------|----------|-----|---------|
| Placebo (2010) | 0.0130 | 0.0047 | 0.006 |
| Employment outcome | 0.0423 | 0.0035 | <0.001 |
| Narrow bandwidth | 0.0228 | 0.0064 | <0.001 |
| Males only | 0.0281 | 0.0046 | <0.001 |
| Females only | 0.0279 | 0.0051 | <0.001 |

### Event Study Coefficients (relative to 2011)
| Year | Coefficient | SE | Significant? |
|------|-------------|-----|--------------|
| 2006 | -0.0202 | 0.0079 | Yes |
| 2007 | -0.0186 | 0.0077 | Yes |
| 2008 | -0.0059 | 0.0077 | No |
| 2009 | -0.0009 | 0.0076 | No |
| 2010 | 0.0019 | 0.0074 | No |
| 2011 | 0.0000 | --- | Ref |
| 2013 | 0.0069 | 0.0073 | No |
| 2014 | 0.0209 | 0.0073 | Yes |
| 2015 | 0.0385 | 0.0073 | Yes |
| 2016 | 0.0392 | 0.0074 | Yes |

---

## Files Generated

### Analysis Files
- `analysis_22.py` - Main Python analysis script

### Output Files
- `results_main.csv` - Main regression results (5 models)
- `results_robustness.csv` - Robustness check results
- `results_event_study.csv` - Event study coefficients
- `results_descriptives.csv` - Descriptive statistics by group/period

### Report Files
- `replication_report_22.tex` - LaTeX source for report
- `replication_report_22.pdf` - Compiled PDF report (20 pages)

---

## Software and Packages
- Python 3.14
- pandas - Data manipulation
- numpy - Numerical operations
- statsmodels - Regression analysis

---

## Key Findings Interpretation

The preferred estimate of 0.0327 suggests that DACA eligibility increased the probability of full-time employment by approximately 3.3 percentage points among the study population. This represents:
- Approximately 7.6% increase relative to baseline full-time employment of 43.1%
- Effect is robust across specifications (range: 0.023 to 0.042 depending on model)
- Event study supports parallel trends for 2008-2011; treatment effects emerge in 2014+
- Similar effects for males and females

### Caveats
1. Placebo test shows small but significant effect, suggesting some pre-trends
2. DACA eligibility is imperfectly measured (cannot observe education/criminal history requirements)
3. Non-citizen assumption may misclassify some legal residents
4. Control group is substantially older, creating potential age-related confounds despite controls
