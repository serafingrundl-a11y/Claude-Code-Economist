# DACA Replication Analysis - Run Log

## Project Information
- **Replication ID:** 19
- **Date:** January 27, 2026
- **Research Question:** Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals

---

## 1. Data Exploration

### Initial Data Inspection
```python
# Loaded and inspected data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(df.shape)  # (17382, 105)
```

### Key Findings from Data Exploration:
- **Total observations:** 17,382
- **Years included:** 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- **Treatment group (ELIGIBLE=1):** 11,382 observations (ages 26-30 in June 2012)
- **Control group (ELIGIBLE=0):** 6,000 observations (ages 31-35 in June 2012)
- **Pre-DACA period (AFTER=0):** 9,527 observations (2008-2011)
- **Post-DACA period (AFTER=1):** 7,855 observations (2013-2016)
- **Full-time employment rate:** 64.9% overall

### Variable Definitions Confirmed:
- FT: Full-time employment (1 if UHRSWORK >= 35, 0 otherwise)
- ELIGIBLE: Treatment indicator (1 = ages 26-30, 0 = ages 31-35 in June 2012)
- AFTER: Post-period indicator (1 = 2013-2016, 0 = 2008-2011)
- SEX: 1 = Male, 2 = Female (IPUMS coding)
- MARST: 1-2 = Married, 3-6 = Not married

---

## 2. Key Analytical Decisions

### Decision 1: Sample Restriction
**Choice:** Used the entire provided dataset without additional restrictions
**Rationale:** Instructions explicitly stated "This entire file constitutes the intended analytic sample for your analysis; do not further limit the sample"

### Decision 2: Outcome Variable
**Choice:** Used the pre-constructed FT variable
**Rationale:** The provided FT variable was already correctly defined (UHRSWORK >= 35). Verified: `df['FT'].mean() == (df['UHRSWORK'] >= 35).mean()` = TRUE

### Decision 3: Treatment and Control Groups
**Choice:** Used the provided ELIGIBLE variable
**Rationale:** Instructions stated "Use this variable to identify individuals in the treated and comparison groups, and do not create your own eligibility variable"

### Decision 4: Standard Errors
**Choice:** Heteroskedasticity-robust standard errors (HC1)
**Rationale:** Standard practice for DiD estimation with potential heteroskedasticity. Did not cluster at state level due to the relatively small number of observations per state-year cell in some states.

### Decision 5: Weighting
**Choice:** Primary analysis unweighted; weighted results as robustness check
**Rationale:** Unweighted OLS is the standard approach for DiD; person weights can introduce complications and were included as sensitivity analysis

### Decision 6: Preferred Specification
**Choice:** Model 4 with demographic controls, state fixed effects, and year fixed effects
**Rationale:**
- Demographic controls adjust for compositional differences between treatment and control groups
- State FE control for time-invariant state-level confounders
- Year FE control for national trends (e.g., economic recovery from 2008 recession)
- Labor market controls (Model 5) had minimal impact and may be endogenous

### Decision 7: Reference Year for Event Study
**Choice:** 2011 (year immediately before DACA implementation)
**Rationale:** Standard practice to use the period immediately before treatment as reference

### Decision 8: Statistical Software
**Choice:** Python with statsmodels
**Rationale:** Preferred coding language; provides equivalent results to Stata/R

---

## 3. Analysis Commands

### Main Analysis Script (analysis.py)

```python
# Basic DiD Model
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + TREATED_POST', data=df).fit(cov_type='HC1')

# DiD with Demographic Controls
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + TREATED_POST + SEX + married + NCHILD +
                  FAMSIZE + educ_somecoll + educ_twoyear + educ_ba',
                 data=df).fit(cov_type='HC1')

# DiD with State Fixed Effects
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + TREATED_POST + [demographics] + [state_dummies]',
                 data=df).fit(cov_type='HC1')

# DiD with Year Fixed Effects (preferred)
model4 = smf.ols('FT ~ ELIGIBLE + TREATED_POST + [demographics] + [state_dummies] + [year_dummies]',
                 data=df).fit(cov_type='HC1')

# Full Model with Labor Market Controls
model5 = smf.ols('FT ~ ELIGIBLE + TREATED_POST + [demographics] + [state_dummies] +
                  [year_dummies] + LFPR + UNEMP',
                 data=df).fit(cov_type='HC1')
```

### Event Study Specification

```python
# Create year-specific treatment interactions (2011 as reference)
event_formula = 'FT ~ ELIGIBLE + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 +
                 year_2015 + year_2016 + elig_x_2008 + elig_x_2009 + elig_x_2010 +
                 elig_x_2013 + elig_x_2014 + elig_x_2015 + elig_x_2016'
model_event = smf.ols(event_formula, data=df).fit(cov_type='HC1')
```

### Placebo Test

```python
# Using only pre-DACA data, treating 2010-2011 as "fake" post
df_pre = df[df['AFTER'] == 0].copy()
df_pre['fake_after'] = (df_pre['YEAR'].isin([2010, 2011])).astype(int)
df_pre['fake_treated_post'] = df_pre['ELIGIBLE'] * df_pre['fake_after']
model_placebo = smf.ols('FT ~ ELIGIBLE + fake_after + fake_treated_post',
                        data=df_pre).fit(cov_type='HC1')
```

---

## 4. Main Results Summary

### Simple DiD Calculation
|                  | Pre-DACA | Post-DACA | Change |
|------------------|----------|-----------|--------|
| Treated (26-30)  | 0.6263   | 0.6658    | +0.0395 |
| Control (31-35)  | 0.6697   | 0.6449    | -0.0248 |
| **DiD**          |          |           | **+0.0643** |

### Regression Results

| Model | DiD Estimate | Std. Error | P-value | N | R² |
|-------|--------------|------------|---------|---|----|
| (1) Basic | 0.0643 | 0.0153 | <0.001 | 17,382 | 0.002 |
| (2) + Demographics | 0.0519 | 0.0141 | <0.001 | 17,382 | 0.133 |
| (3) + State FE | 0.0522 | 0.0142 | <0.001 | 17,382 | 0.136 |
| **(4) + Year FE** | **0.0507** | **0.0141** | **<0.001** | **17,382** | **0.138** |
| (5) + Labor Market | 0.0505 | 0.0141 | <0.001 | 17,382 | 0.139 |

### Preferred Estimate (Model 4)
- **Effect:** 0.0507 (5.07 percentage points)
- **Standard Error:** 0.0141
- **95% CI:** [0.023, 0.078]
- **P-value:** 0.0003

### Interpretation
DACA eligibility increased full-time employment by approximately 5.1 percentage points, representing an 8.1% increase relative to the pre-treatment mean of 62.6% for the treated group.

---

## 5. Robustness Checks

### Weighted Estimation (PERWT)
- DiD: 0.0614 (SE = 0.0167, p < 0.001)

### By Gender
- Male: DiD = 0.0615 (SE = 0.0170, p < 0.001)
- Female: DiD = 0.0452 (SE = 0.0232, p = 0.051)

### By Marital Status
- Married: DiD = 0.0586 (SE = 0.0214, p = 0.006)
- Not Married: DiD = 0.0758 (SE = 0.0221, p < 0.001)

### Placebo Test (Pre-trends)
- DiD: 0.0157 (SE = 0.0205, p = 0.44) - Not significant

---

## 6. Event Study Results

| Year | Coefficient | Std. Error | Significance |
|------|-------------|------------|--------------|
| 2008 | -0.0591 | 0.0289 | ** |
| 2009 | -0.0388 | 0.0297 | |
| 2010 | -0.0663 | 0.0294 | ** |
| 2011 | (reference) | | |
| 2013 | 0.0188 | 0.0306 | |
| 2014 | -0.0088 | 0.0308 | |
| 2015 | 0.0303 | 0.0316 | |
| 2016 | 0.0491 | 0.0314 | |

**Note:** Some pre-trend variation observed (2008, 2010 significant), but placebo test reassuring.

---

## 7. Files Generated

### Scripts
- `analysis.py` - Main analysis script
- `generate_figures.py` - Figure and table generation

### Output Files
- `analysis_results.json` - Stored regression results
- `figures/fig1_ft_trends.png` - Employment trends over time
- `figures/fig2_event_study.png` - Event study plot
- `figures/fig3_did_visual.png` - DiD visualization
- `figures/fig4_state_distribution.png` - Sample by state
- `figures/fig5_model_comparison.png` - Coefficient comparison
- `figures/fig6_education.png` - Education distribution
- `figures/table1_summary.tex` - Summary statistics table
- `figures/table2_main_results.tex` - Main results table
- `figures/table3_event_study.tex` - Event study table
- `figures/table4_robustness.tex` - Robustness checks table

### Final Deliverables
- `replication_report_19.tex` - LaTeX source
- `replication_report_19.pdf` - Final report (21 pages)
- `run_log_19.md` - This file

---

## 8. LaTeX Compilation

```bash
# Compile three times for cross-references
pdflatex -interaction=nonstopmode replication_report_19.tex
pdflatex -interaction=nonstopmode replication_report_19.tex
pdflatex -interaction=nonstopmode replication_report_19.tex
```

Output: `replication_report_19.pdf` (21 pages)

---

## 9. Limitations Noted

1. **Parallel Trends:** Event study shows some pre-treatment variation (2008, 2010 coefficients significant)
2. **Repeated Cross-Section:** Cannot follow same individuals over time; no individual FE
3. **ITT vs TOT:** Estimate is intent-to-treat (eligibility), not treatment-on-treated (DACA receipt)
4. **Control Group Spillovers:** Possible general equilibrium effects on control group
5. **Geographic Concentration:** Sample concentrated in CA (45%) and TX (21%)

---

## 10. Conclusion

The analysis finds robust evidence that DACA eligibility increased full-time employment by approximately 5 percentage points. The effect is:
- Statistically significant at the 1% level
- Robust across specifications
- Present for both genders and marital statuses
- Consistent with theoretical expectations

The preferred estimate (Model 4 with demographic controls and fixed effects) is **0.0507** with a standard error of **0.0141**.
