# DACA Replication Study - Run Log

## Study Information
- **Date**: 2026-01-26
- **Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals
- **Design**: Difference-in-Differences
- **Treatment Group**: Ages 26-30 at June 15, 2012 (DACA implementation)
- **Control Group**: Ages 31-35 at June 15, 2012 (ineligible due to age cutoff)
- **Pre-Period**: 2006-2011 (prior to DACA)
- **Post-Period**: 2013-2016 (after DACA implementation)
- **Outcome**: Full-time employment (working 35+ hours per week typically)

---

## Session Log

### Step 1: Data Exploration
**Action**: Read replication instructions and data dictionary

**Key Variables Identified**:
- `YEAR`: Survey year (2006-2016)
- `BIRTHYR`: Birth year
- `BIRTHQTR`: Birth quarter (1=Q1, 2=Q2, 3=Q3, 4=Q4)
- `BPL/BPLD`: Birthplace (200/20000 = Mexico)
- `HISPAN/HISPAND`: Hispanic origin (1/100-107 = Mexican)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `UHRSWORK`: Usual hours worked per week
- `EMPSTAT`: Employment status (1 = Employed)
- `PERWT`: Person weight for representative estimates
- `AGE`: Current age
- `SEX`: Sex (1=Male, 2=Female)
- `EDUC/EDUCD`: Educational attainment

**DACA Eligibility Criteria** (per instructions):
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Not a citizen (CITIZEN == 3) - proxy for undocumented status
4. Arrived before age 16: YRIMMIG - BIRTHYR < 16
5. Lived continuously in US since June 15, 2007: YRIMMIG <= 2007
6. Present in US on June 15, 2012 (assumed for all in sample)

**Age Groups at June 15, 2012**:
- Treatment: Ages 26-30 (eligible for DACA)
- Control: Ages 31-35 (too old for DACA by 31st birthday cutoff)

**Decision**: Exclude 2012 data since we cannot distinguish pre/post DACA within that year.

---

### Step 2: Data Processing
**Action**: Created Python script (analysis.py) for data analysis using memory-efficient chunked processing

**Sample Construction**:
1. Initial raw data: Large ACS dataset (2006-2016)
2. After Hispanic-Mexican restriction: 701,347 observations
3. After born in Mexico restriction: (filtered simultaneously)
4. After non-citizen restriction: (filtered simultaneously)
5. After arrived before age 16: 205,327 observations
6. After arrived by 2007: 195,023 observations
7. After age restriction (26-35 at DACA): 47,418 observations
8. After excluding 2012: **43,238 observations** (final analysis sample)

**Treatment and Control**:
- Treatment (ages 26-30): 27,903 observations
- Control (ages 31-35): 19,515 observations

**Time Periods**:
- Pre-DACA (2006-2011): 28,377 observations
- Post-DACA (2013-2016): 14,861 observations

---

### Step 3: Variable Construction
**Action**: Created outcome and control variables

**Outcome Variable**:
- `fulltime`: Binary indicator = 1 if employed (EMPSTAT==1) AND usual hours >= 35/week
- Employment rate: 68.2%
- Full-time employment rate: 56.6%

**Control Variables**:
- `female`: Sex indicator (43.4% female in treatment, 41.4-44.7% in control)
- `age`, `age_sq`: Current age and squared term
- `married`: Marital status indicator
- `nchild`: Number of children
- `years_in_us`: Years since immigration
- `edu_hs`, `edu_some_college`, `edu_college_plus`: Education dummies

---

### Step 4: Summary Statistics
**Action**: Calculated weighted summary statistics by group and period

| Group | Period | N | N_weighted | Fulltime_Rate | Employed_Rate |
|-------|--------|---|------------|---------------|---------------|
| Control (31-35) | Pre-DACA | 11,683 | 1,631,151 | 0.6135 | 0.7183 |
| Control (31-35) | Post-DACA | 6,085 | 845,134 | 0.6037 | 0.7219 |
| Treatment (26-30) | Pre-DACA | 16,694 | 2,280,009 | 0.5655 | 0.6844 |
| Treatment (26-30) | Post-DACA | 8,776 | 1,244,124 | 0.6198 | 0.7402 |

**Simple DiD Calculation**:
- Treatment change: 0.6198 - 0.5655 = +0.0543
- Control change: 0.6037 - 0.6135 = -0.0098
- DiD estimate: 0.0543 - (-0.0098) = **0.0641** (6.4 percentage points)

---

### Step 5: Regression Analysis
**Action**: Estimated DiD models with varying controls

| Model | DiD Coefficient | Std. Error | p-value | 95% CI |
|-------|-----------------|------------|---------|--------|
| Basic DiD | 0.0642 | 0.0121 | <0.001 | [0.041, 0.088] |
| + Demographics | 0.0689 | 0.0151 | <0.001 | [0.039, 0.099] |
| + Education | 0.0679 | 0.0151 | <0.001 | [0.038, 0.098] |
| + State FE | 0.0674 | 0.0151 | <0.001 | [0.038, 0.097] |
| + State & Year FE (Preferred) | 0.0237 | 0.0160 | 0.138 | [-0.008, 0.055] |

**Key Decision**: Selected Model 5 (state and year fixed effects) as preferred specification because:
1. Year fixed effects control for common macroeconomic shocks (e.g., recovery from 2008 recession)
2. State fixed effects control for time-invariant state characteristics
3. Most conservative specification to isolate DACA-specific effects

---

### Step 6: Robustness Checks
**Action**: Conducted additional analyses to validate findings

**Placebo Test** (fake treatment in 2009, pre-period only):
- Placebo DiD: -0.0224 (SE: 0.0158, p = 0.157)
- Result: No significant pre-trend, supports parallel trends assumption

**Heterogeneous Effects by Gender**:
- Males: 0.0131 (SE: 0.0202, p = 0.515)
- Females: 0.0272 (SE: 0.0247, p = 0.270)
- Both insignificant; point estimates suggest slightly larger effect for women

---

### Step 7: Event Study Analysis
**Action**: Estimated year-specific treatment effects (reference: 2011)

| Year | Coefficient | 95% CI |
|------|-------------|--------|
| 2006 | 0.020 | [-0.030, 0.070] |
| 2007 | -0.013 | [-0.061, 0.035] |
| 2008 | 0.013 | [-0.034, 0.061] |
| 2009 | -0.001 | [-0.049, 0.047] |
| 2010 | -0.009 | [-0.056, 0.038] |
| 2011 | 0.000 | (reference) |
| 2013 | 0.032 | [-0.018, 0.082] |
| 2014 | 0.022 | [-0.029, 0.073] |
| 2015 | 0.000 | [-0.052, 0.053] |
| 2016 | 0.032 | [-0.022, 0.086] |

**Interpretation**: Pre-treatment coefficients are close to zero and not statistically significant, supporting the parallel trends assumption. Post-treatment coefficients are positive but individually insignificant, consistent with the main DiD finding.

---

### Step 8: Figures Created
1. **figure1_trends.png**: Full-time employment rates over time for treatment vs. control
2. **figure2_event_study.png**: Event study plot with 95% confidence intervals

---

### Step 9: Final Results
**Preferred Estimate** (Model 5 with State and Year Fixed Effects):
- **DiD Coefficient**: 0.0237
- **Standard Error**: 0.0160
- **95% Confidence Interval**: [-0.008, 0.055]
- **p-value**: 0.138
- **Sample Size**: 43,238

**Interpretation**: DACA eligibility is associated with a 2.4 percentage point increase in full-time employment among eligible Mexican-born non-citizens, though this effect is not statistically significant at conventional levels. The 95% confidence interval ranges from a 0.8 percentage point decrease to a 5.5 percentage point increase, so we cannot rule out either a modest negative effect, no effect, or a meaningful positive effect.

---

## Key Analytical Decisions

1. **Sample definition**: Used HISPAN==1 (Mexican Hispanic), BPL==200 (born in Mexico), CITIZEN==3 (not a citizen) as proxy for DACA-eligible population

2. **Age calculation**: Used BIRTHYR and BIRTHQTR to calculate age at June 15, 2012

3. **Treatment definition**: Ages 26-30 at DACA (born ~1982-1986) vs. ages 31-35 (born ~1977-1981)

4. **Time periods**: Excluded 2012 due to inability to distinguish pre/post within year

5. **Outcome**: Full-time employment = employed AND usual hours >= 35/week

6. **Preferred specification**: Two-way fixed effects (state and year) with demographic and education controls

7. **Standard errors**: Heteroskedasticity-robust (HC1)

8. **Weighting**: All regressions weighted by PERWT (person weight)

---

## Files Produced
- `analysis.py`: Main analysis script
- `summary_statistics.csv`: Summary statistics by group and period
- `regression_results.csv`: Regression coefficients across all models
- `event_study_coefficients.csv`: Event study coefficients
- `final_results.csv`: Final preferred estimate
- `figure1_trends.png`: Trends figure
- `figure2_event_study.png`: Event study figure
- `replication_report_22.tex`: LaTeX report
- `replication_report_22.pdf`: Final PDF report
