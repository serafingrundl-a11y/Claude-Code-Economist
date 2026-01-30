# DACA Replication Study - Run Log (ID: 51)

## Overview
This log documents all commands, decisions, and analyses performed for the independent replication of the DACA effect on full-time employment study.

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome)?

**Identification Strategy:** Difference-in-differences comparing:
- Treatment group: Ages 26-30 at the time of DACA implementation (June 15, 2012) - ELIGIBLE=1
- Control group: Ages 31-35 at the time of DACA implementation - ELIGIBLE=0
- Pre-period: 2008-2011 (AFTER=0)
- Post-period: 2013-2016 (AFTER=1)

---

## Session Log

### Step 1: Read and Understand Instructions
- **Action:** Read replication_instructions.docx
- **Key findings:**
  - DACA implemented June 15, 2012
  - Data: ACS 2008-2016 (excluding 2012)
  - Outcome: FT (full-time employment, 1=35+ hours/week)
  - Treatment indicator: ELIGIBLE variable provided
  - Post-period indicator: AFTER variable provided
  - Sample: Hispanic-Mexican, Mexican-born individuals
  - Design: Repeated cross-section (not panel data)

### Step 2: Examine Data Structure
- **Action:** Explored data files
- **Files found:**
  - `prepared_data_numeric_version.csv` - 17,382 observations
  - `prepared_data_labelled_version.csv` - Same data with labels
  - `acs_data_dict.txt` - Variable documentation
- **Key variables identified:**
  - FT: Full-time employment (0/1)
  - AFTER: Post-DACA period (0/1)
  - ELIGIBLE: Treatment group indicator (0/1)
  - PERWT: Person weights for representativeness
  - Various demographic covariates (AGE, SEX, EDUC_RECODE, MARST, etc.)
  - State policy variables (DRIVERSLICENSES, INSTATETUITION, etc.)

### Step 3: Analysis Plan
**Primary Analysis:**
1. Basic difference-in-differences (DiD) without covariates
2. DiD with demographic covariates
3. DiD with state fixed effects
4. DiD with state and year fixed effects
5. DiD with state-level policy variables

**Covariates used:**
- Demographics: SEX (FEMALE), EDUC_RECODE, MARST, AGE, NCHILD
- State characteristics: STATEFIP (state fixed effects)
- State policies: DRIVERSLICENSES, INSTATETUITION, EVERIFY
- Economic conditions: LFPR, UNEMP (state-level labor force participation and unemployment)

**Statistical Approach:**
- Linear probability model (OLS)
- Standard errors clustered at state level (50 clusters)
- Unweighted regression (weights not used due to complexity with clustering)

### Step 4: Data Loading and Exploration

**Command executed:**
```python
python analysis_51_corrected.py
```

**Data Summary:**
- Total Sample Size: 17,379 (after dropping 3 observations with missing EDUC_RECODE)
- Treatment group (ELIGIBLE=1): 11,379
- Control group (ELIGIBLE=0): 6,000
- Pre-period observations: 9,524
- Post-period observations: 7,855
- Number of states: 50

**DiD Cell Counts:**
|                 | Pre (2008-2011) | Post (2013-2016) | Total  |
|-----------------|-----------------|------------------|--------|
| Control (31-35) | 3,294           | 2,706            | 6,000  |
| Treatment (26-30)| 6,230          | 5,149            | 11,379 |
| Total           | 9,524           | 7,855            | 17,379 |

### Step 5: Statistical Analysis

**Simple DiD Calculation (Weighted Means):**
- Treatment Pre: 0.6368
- Treatment Post: 0.6860
- Treatment Change: +0.0493
- Control Pre: 0.6886
- Control Post: 0.6629
- Control Change: -0.0257
- **Simple DiD: 0.0749** (7.49 percentage points)

**Regression Results (Clustered SEs at State Level):**

| Model              | DiD Estimate | Std Error | t-stat | p-value | 95% CI Lower | 95% CI Upper |
|--------------------|--------------|-----------|--------|---------|--------------|--------------|
| 1. Basic DiD       | 0.0644       | 0.0141    | 4.57   | 0.0000  | 0.0368       | 0.0921       |
| 2. Demographics    | 0.0558       | 0.0145    | 3.85   | 0.0001  | 0.0274       | 0.0842       |
| 3. State FE        | 0.0558       | 0.0150    | 3.72   | 0.0002  | 0.0265       | 0.0852       |
| 4. State+Year FE   | 0.0543       | 0.0148    | 3.67   | 0.0002  | 0.0253       | 0.0833       |
| 5. Policy Controls | 0.0555       | 0.0144    | 3.86   | 0.0001  | 0.0273       | 0.0836       |

**Parallel Trends Analysis:**

Pre-treatment trend interactions (ELIGIBLE x Year):
- 2009: 0.0203 (SE: 0.0286, p: 0.478) - Not significant
- 2010: -0.0072 (SE: 0.0158, p: 0.650) - Not significant
- 2011: 0.0593 (SE: 0.0233, p: 0.011) - Significant

Note: The 2011 interaction is statistically significant, suggesting some differential pre-trend, though this is only one year before treatment.

### Step 6: Results Summary

**Preferred Estimate (Model 3: DiD with Demographics and State Fixed Effects):**
- **Effect: 0.0558** (5.58 percentage point increase)
- **Standard Error: 0.0150** (clustered at state level)
- **95% CI: [0.0265, 0.0852]**
- **p-value: 0.0002**
- **Sample Size: 17,379**

**Interpretation:** DACA eligibility is associated with a 5.6 percentage point increase in the probability of full-time employment among eligible individuals aged 26-30 compared to similar individuals aged 31-35 who were ineligible due to age. This effect is statistically significant at conventional levels.

---

## Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| Used numeric version of data | Easier for statistical processing |
| Linear probability model | Simple interpretation, standard in DiD literature |
| Cluster SEs at state level | Account for within-state correlation; 50 clusters provides adequate inference |
| Did not use survey weights in regression | Complexity with proper clustered inference; unweighted DiD still valid |
| Used provided ELIGIBLE variable | As instructed in documentation |
| Dropped 3 observations with missing EDUC_RECODE | Required for model estimation |
| Preferred Model 3 (State FE) | Balances control for confounders with parsimony; year FE adds minimal change |

---

## Code Execution Log

### Python Analysis Commands

```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_51"
python analysis_51_corrected.py
```

### Key Python Code Snippets

**Basic DiD Model:**
```python
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}
)
```

**Model with Covariates and State FE:**
```python
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + NCHILD +
                  C(MARST) + C(EDUC_RECODE) + C(STATEFIP)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}
)
```

---

## Output Files Created

1. `replication_report_51.tex` - Main LaTeX report
2. `replication_report_51.pdf` - Compiled PDF report
3. `run_log_51.md` - This log file
4. `analysis_51_corrected.py` - Python analysis script
5. `analysis_output.txt` - Text output from analysis
6. `regression_results.csv` - Regression results table
7. `yearly_means.csv` - Yearly employment rates by group
8. `trends_plot.png/pdf` - Time series visualization
9. `did_plot.png/pdf` - DiD bar chart
10. `coefficient_plot.png/pdf` - Coefficient comparison plot

---

## Robustness and Limitations

**Robustness:**
- Results are stable across specifications (range: 0.054-0.064)
- All models show statistically significant positive effect
- Effect is robust to inclusion of state FE, year FE, and policy controls

**Limitations:**
1. Significant pre-trend in 2011 raises some concern about parallel trends assumption
2. Repeated cross-section design (not panel) limits ability to track individuals
3. Cannot directly observe DACA receipt, only eligibility
4. Age-based cutoff may have other confounds (life-cycle effects)
5. Potential selection into remaining in sample after DACA

---

## Final Preferred Estimate

| Statistic | Value |
|-----------|-------|
| Effect Size | 0.0558 |
| Standard Error | 0.0150 |
| 95% CI | [0.0265, 0.0852] |
| p-value | 0.0002 |
| Sample Size | 17,379 |
| Model | DiD with demographics and state fixed effects |
