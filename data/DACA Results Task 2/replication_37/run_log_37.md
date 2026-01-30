# Run Log for DACA Replication Study (Replication 37)

## Overview
This log documents all commands, key decisions, and analytical choices made during the replication of the DACA study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (usually working 35+ hours per week)?

## Study Design
- **Treatment Group**: Ages 26-30 at time of DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at time of DACA implementation (otherwise eligible)
- **Method**: Difference-in-differences
- **Pre-period**: 2006-2011 (excluding 2012 due to ambiguous timing)
- **Post-period**: 2013-2016

---

## Session Log

### Step 1: Data Exploration
**Date/Time**: Session start

**Files identified**:
- `data/data.csv` - Main ACS data file (~33.8 million observations)
- `data/acs_data_dict.txt` - Data dictionary with variable definitions
- `data/state_demo_policy.csv` - State-level policy data (optional)

**Key Variables Identified**:
- YEAR: Census year (2006-2016)
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth
- HISPAN/HISPAND: Hispanic origin (1=Mexican for HISPAN)
- BPL/BPLD: Birthplace (200=Mexico)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status
- SEX, AGE, EDUC, STATEFIP: Demographic controls

### Step 2: Sample Selection Criteria

**DACA Eligibility Criteria** (from instructions):
1. Arrived in US before 16th birthday
2. Had not turned 31 by June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

**Operationalization**:
- Hispanic-Mexican: HISPAN == 1 (Mexican)
- Born in Mexico: BPL == 200
- Not a citizen: CITIZEN == 3
- Immigration before age 16: YRIMMIG - BIRTHYR < 16
- Arrived by 2007: YRIMMIG <= 2007

**Age Groups** (calculated at June 15, 2012):
- Treatment: Born 1982-1986 (ages 26-30 at DACA)
- Control: Born 1977-1981 (ages 31-35 at DACA)

### Step 3: Outcome Variable
- Full-time employment: UHRSWORK >= 35 (binary indicator)

---

## Analysis Commands

### Command: Run main analysis script
```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_37" && python analysis.py
```

### Key Analytical Decisions

1. **Exclusion of 2012**: Year 2012 was excluded from analysis because DACA was implemented mid-year (June 15, 2012), making it impossible to distinguish pre- and post-treatment observations.

2. **Age calculation**: Age at DACA implementation was calculated using birth year and birth quarter. Those born in Q3 or Q4 (July-December) were adjusted down by 1 year since they would not have reached their birthday by June 15.

3. **Sample restrictions**:
   - HISPAN == 1 (Mexican Hispanic)
   - BPL == 200 (Born in Mexico)
   - CITIZEN == 3 (Not a citizen)
   - YRIMMIG - BIRTHYR < 16 (Arrived before age 16)
   - YRIMMIG <= 2007 (In US by 2007)
   - Age 26-35 at DACA implementation

4. **Weighting**: All analyses used PERWT (person weights) from ACS.

5. **Standard errors**: Final model used clustered standard errors at the state level.

---

## Results Summary

### Sample Sizes
- Total analysis sample: 43,238 observations
- Treatment group (pre-DACA): 16,694 observations
- Treatment group (post-DACA): 8,776 observations
- Control group (pre-DACA): 11,683 observations
- Control group (post-DACA): 6,085 observations

### Main Results

| Model | DiD Estimate | SE | 95% CI | p-value |
|-------|--------------|-----|--------|---------|
| Basic DiD | 0.0590 | 0.0098 | [0.040, 0.078] | <0.001 |
| With Demographics | 0.0650 | 0.0120 | [0.041, 0.089] | <0.001 |
| With Education | 0.0644 | 0.0120 | [0.041, 0.088] | <0.001 |
| Year FE | 0.0197 | 0.0127 | [-0.005, 0.045] | 0.120 |
| Year + State FE | 0.0189 | 0.0127 | [-0.006, 0.044] | 0.136 |
| **Clustered SE** | **0.0189** | **0.0119** | **[-0.004, 0.042]** | **0.112** |

### Preferred Estimate
- **Effect Size**: 0.0189 (1.89 percentage points)
- **Standard Error**: 0.0119 (clustered by state)
- **95% Confidence Interval**: [-0.0044, 0.0423]
- **P-value**: 0.112
- **Sample Size**: 43,238

### Interpretation
The difference-in-differences estimate suggests that DACA eligibility increased the probability of full-time employment by approximately 1.9 percentage points. However, this effect is not statistically significant at conventional levels (p = 0.112).

### Pre-Trend Analysis
Event study analysis shows no clear pre-treatment trends, with year-specific coefficients fluctuating around zero in the pre-period (2006-2010 relative to 2011).

### Subgroup Results
- Males: DiD = 0.047 (SE: 0.014)
- Females: DiD = 0.076 (SE: 0.020)
- Less than HS: DiD = 0.069 (SE: 0.018)
- HS or more: DiD = 0.066 (SE: 0.016)

---

## Output Files Generated
- `results_summary.csv` - Summary of all regression results
- `model_output.txt` - Full regression output
- `descriptive_stats.csv` - Descriptive statistics
- `sample_sizes.csv` - Sample sizes by group and period
- `event_study_coefs.csv` - Event study coefficients
- `replication_report_37.tex` - LaTeX report
- `replication_report_37.pdf` - PDF report
