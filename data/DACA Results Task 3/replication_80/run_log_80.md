# Replication Run Log - Study 80

## Project Overview
**Research Question**: What was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on full-time employment among ethnically Hispanic-Mexican, Mexican-born people living in the United States?

**Outcome**: Full-time employment (FT), defined as usually working 35 hours per week or more
**Treatment Group**: Ages 26-30 at June 15, 2012 (ELIGIBLE=1)
**Control Group**: Ages 31-35 at June 15, 2012 (ELIGIBLE=0)
**Pre-Period**: 2008-2011
**Post-Period**: 2013-2016 (DACA implemented June 15, 2012)

---

## Session Log

### Step 1: Read Replication Instructions
- Read `replication_instructions.docx`
- Identified key research design: Difference-in-Differences (DiD)
- Treatment: DACA eligibility for ages 26-30 at policy implementation
- Control: Ages 31-35 who would have been eligible but for age
- Outcome: Full-time employment (FT variable, binary 0/1)
- Data: ACS 2008-2011 (pre) and 2013-2016 (post), omitting 2012
- Key variables provided: ELIGIBLE, FT, AFTER

### Step 2: Examine Data Structure
- Dataset location: `data/prepared_data_numeric_version.csv`
- Observations: 17,382 (plus header)
- Key variables identified:
  - YEAR: Survey year
  - ELIGIBLE: 1 for treatment group (ages 26-30), 0 for control (ages 31-35)
  - FT: Full-time employment (1=yes, 0=no)
  - AFTER: Post-treatment indicator (1 for 2013-2016, 0 for 2008-2011)
  - PERWT: Person weight for survey weighting
  - Various demographic controls available (SEX, AGE, EDUC_RECODE, MARST, etc.)
  - State policy variables (DRIVERSLICENSES, INSTATETUITION, etc.)

### Step 3: Analysis Plan
1. Descriptive statistics by group and period
2. Basic DiD estimation: FT = β0 + β1*ELIGIBLE + β2*AFTER + β3*ELIGIBLE*AFTER + ε
3. DiD with demographic controls
4. DiD with state fixed effects
5. Parallel trends check (year-by-year analysis)
6. Robustness checks with additional controls

---

## Analysis Commands and Output

### Step 4: Run Main Analysis (Python)
```
python analysis_80.py
```

**Key Outputs:**
- Total observations: 17,382
- Treatment group (ELIGIBLE=1): 11,382
- Control group (ELIGIBLE=0): 6,000
- Pre-period observations: 9,527
- Post-period observations: 7,855

### Step 5: Main Results

**Preferred Specification: DiD with demographic controls and state-clustered standard errors**

| Specification | DiD Coefficient | Std. Error | 95% CI | P-value |
|--------------|-----------------|------------|--------|---------|
| Basic (unweighted) | 0.0643 | 0.0153 | [0.034, 0.094] | <0.001 |
| Basic (weighted) | 0.0748 | 0.0152 | [0.045, 0.105] | <0.001 |
| With controls | 0.0626 | 0.0142 | [0.035, 0.090] | <0.001 |
| With state FE | 0.0620 | 0.0142 | [0.034, 0.090] | <0.001 |
| **Clustered SE (preferred)** | **0.0626** | **0.0214** | **[0.021, 0.105]** | **0.003** |

**Interpretation:** DACA eligibility increased the probability of full-time employment by 6.26 percentage points (p = 0.003).

### Step 6: Event Study Results

| Year | Coefficient | Std. Error | P-value | Period |
|------|-------------|------------|---------|--------|
| 2008 | -0.0657 | 0.0270 | 0.015 | Pre |
| 2009 | -0.0498 | 0.0269 | 0.065 | Pre |
| 2010 | -0.0757 | 0.0317 | 0.017 | Pre |
| 2011 | 0 (ref) | - | - | Pre |
| 2013 | 0.0168 | 0.0373 | 0.653 | Post |
| 2014 | -0.0107 | 0.0211 | 0.612 | Post |
| 2015 | -0.0102 | 0.0339 | 0.762 | Post |
| 2016 | 0.0644 | 0.0295 | 0.029 | Post |

**Note:** Significant pre-treatment coefficients suggest potential parallel trends violation.

### Step 7: Robustness Checks

**By Sex:**
- Male: DiD = 0.0635 (SE: 0.0170)
- Female: DiD = 0.0517 (SE: 0.0231)

**By Education:**
- High School: DiD = 0.0476 (SE: 0.0164)
- Some College: DiD = 0.0660 (SE: 0.0370)
- BA+: DiD = 0.1319 (SE: 0.0585)

**Additional Robustness:**
- Excluding 2008: DiD = 0.0574 (SE: 0.0153)
- With state policy controls: DiD = 0.0635 (SE: 0.0142)

### Step 8: Create Figures
```
python create_figures.py
```

**Figures generated:**
1. figure1_trends.png - Time trends in FT employment by group
2. figure2_eventstudy.png - Event study coefficients
3. figure3_did_decomp.png - DiD decomposition
4. figure4_heterogeneity.png - Heterogeneity by education

### Step 9: Write LaTeX Report
Created replication_report_80.tex with full methodology and results.

### Step 10: Compile PDF
```
pdflatex replication_report_80.tex
```

---

## Key Decisions Made

1. **Model Specification:** Used Linear Probability Model (OLS/WLS) for interpretability
2. **Weighting:** Used PERWT survey weights for population-representative estimates
3. **Standard Errors:** Clustered at state level to account for within-state correlation
4. **Controls:** Included sex, marital status, age, and education as covariates
5. **Preferred Estimate:** Model with demographic controls and clustered SEs

## Final Results Summary

**Preferred Estimate:** 0.0626 (6.26 percentage points)
**Standard Error:** 0.0214 (clustered by state)
**95% CI:** [0.0207, 0.1045]
**P-value:** 0.0034
**Sample Size:** 17,382

