# Replication Run Log

## Task Overview
Independent replication of DACA (Deferred Action for Childhood Arrivals) effect on full-time employment among ethnically Hispanic-Mexican Mexican-born individuals in the United States.

## Research Design
- **Treatment group**: DACA-eligible individuals aged 26-30 as of June 15, 2012
- **Control group**: Otherwise eligible individuals aged 31-35 as of June 15, 2012 (ineligible due to age cutoff)
- **Outcome**: Full-time employment (FT=1 if usually working 35+ hours/week)
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016 (2012 excluded due to mid-year policy implementation)

---

## Session Log

### Step 1: Document Review
**Timestamp**: Session start

Read `replication_instructions.docx` containing:
- Research question specification
- Data source: American Community Survey (ACS) via IPUMS
- Variable definitions (ELIGIBLE, FT, AFTER provided)
- Instructions to use entire sample without further restrictions

### Step 2: Data Exploration

**Data file**: `prepared_data_numeric_version.csv`
- Shape: 17,382 observations x 105 variables
- Years: 2008-2011, 2013-2016 (2012 excluded as specified)

**Key variable distributions**:
- ELIGIBLE: 11,382 (1=treatment), 6,000 (0=control)
- AFTER: 9,527 (0=pre), 7,855 (1=post)
- FT: 11,283 (1=full-time), 6,099 (0=not full-time)
- AGE_IN_JUNE_2012: Range 26-35 (as expected from design)

**Initial DiD calculation (raw means)**:
| Group | Pre (AFTER=0) | Post (AFTER=1) | Change |
|-------|---------------|----------------|--------|
| Treatment (ELIGIBLE=1) | 62.63% | 66.58% | +3.95 pp |
| Control (ELIGIBLE=0) | 66.97% | 64.49% | -2.48 pp |
| **DiD Estimate** | | | **+6.43 pp** |

### Step 3: Analytical Approach Decisions

**Decision 1: Base model specification**
- Will estimate standard DiD regression: FT = β0 + β1*ELIGIBLE + β2*AFTER + β3*(ELIGIBLE×AFTER) + ε
- β3 is the DiD treatment effect estimate

**Decision 2: Sample weights**
- Will use PERWT (person weights) from ACS to produce population-representative estimates
- This is standard practice for ACS analysis

**Decision 3: Standard errors**
- Will cluster standard errors at state level (STATEFIP) to account for policy correlation within states
- Alternative: robust heteroskedasticity-consistent SEs

**Decision 4: Covariates for robustness**
- Base model: No covariates (cleanest DiD)
- Model 2: Add demographic controls (sex, age, marital status, education)
- Model 3: Add year fixed effects + demographic controls
- Model 4: Add state fixed effects + year fixed effects + demographics

**Decision 5: Parallel trends**
- Will examine pre-treatment trends visually and statistically
- Will test for differential trends by year

---

## Step 4: Full Analysis Execution

### Main Analysis Script
Created and executed `analysis_script.py` containing:
- Descriptive statistics
- 8 regression model specifications
- Parallel trends analysis with event study
- Heterogeneity analysis by sex, education, and marital status
- Robustness checks (logit, bandwidth sensitivity, placebo tests)

### Key Commands Executed

```python
# Load data
import pandas as pd
import statsmodels.formula.api as smf
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Create interaction term
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit()

# Model 2: Weighted DiD
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                 data=df, weights=df['PERWT']).fit()

# Model 3: Weighted with robust SE
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 5: Full specification with demographics and education
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)
df['EDUC_HS'] = (df['EDUC'] == 6).astype(int)
df['EDUC_SOMECOLL'] = df['EDUC'].isin([7, 8, 9]).astype(int)
df['EDUC_COLLEGE'] = df['EDUC'].isin([10, 11]).astype(int)

model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_COLLEGE',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model with clustered standard errors
model8 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_COLLEGE',
                 data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```

---

## Step 5: Results Summary

### Main DiD Estimates Across Specifications

| Model | Specification | DiD Estimate | SE | p-value |
|-------|--------------|--------------|-------|---------|
| 1 | Basic (unweighted) | 0.0643 | 0.0153 | <0.001 |
| 2 | Basic (weighted) | 0.0748 | 0.0152 | <0.001 |
| 3 | Weighted + Robust SE | 0.0748 | 0.0181 | <0.001 |
| 4 | + Demographics | 0.0674 | 0.0168 | <0.001 |
| 5 | + Education | 0.0649 | 0.0167 | <0.001 |
| 6 | + Year FE | 0.0622 | 0.0167 | <0.001 |
| 7 | + State FE | 0.0616 | 0.0166 | <0.001 |
| 8 | Clustered SE (State) | 0.0649 | 0.0217 | 0.003 |

### Preferred Estimate (Model 5)
- **DiD Effect**: 0.0649 (6.49 percentage points)
- **Standard Error**: 0.0167 (robust)
- **95% CI**: [0.032, 0.098]
- **p-value**: 0.0001
- **Sample Size**: 17,382

### Parallel Trends Check
Pre-treatment year-by-treatment interaction coefficients:
- 2009: 0.0150 (p=0.619) - not significant
- 2010: -0.0103 (p=0.733) - not significant
- 2011: 0.0648 (p=0.045) - marginally significant

Two of three pre-treatment coefficients are not statistically different from zero, providing reasonable (though not perfect) support for the parallel trends assumption.

### Placebo Test
- Fake treatment in 2010 (pre-period only): DiD = 0.0178 (p=0.461)
- Not statistically significant, supporting validity of identification

### Heterogeneity Analysis
- **By Sex**: Males (0.072, p<0.001); Females (0.053, p=0.061)
- **By Education**: Effect increases with education level
- **By Marital Status**: Larger for unmarried individuals (0.098 vs 0.057)

---

## Key Analytical Decisions and Justifications

1. **Used PERWT weights**: ACS is a stratified survey; weights are necessary for population-representative estimates.

2. **Preferred Model 5**: Includes demographic and education controls to improve precision while avoiding over-specification. State fixed effects (Model 7) did not substantially change the estimate.

3. **Robust standard errors**: Used HC1 heteroskedasticity-consistent SEs. Also checked clustered SEs at state level (larger but still significant).

4. **Did not drop any observations**: Instructions specified using entire provided sample.

5. **Linear probability model**: Used LPM as primary specification for interpretability. Logit model produced similar marginal effects (0.064).

---

## Output Files Generated
- `analysis_script.py` - Full Python analysis code
- `replication_report_01.tex` - LaTeX report (to be created)
- `replication_report_01.pdf` - Final PDF report (to be created)
- `run_log_01.md` - This log file

---

## Session End Notes
Analysis completed successfully. Main finding: DACA eligibility is associated with a statistically significant 6.5 percentage point increase in full-time employment probability among the target population, robust across specifications.
