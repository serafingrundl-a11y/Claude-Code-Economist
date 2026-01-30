# Replication Run Log - Participant 31

## Task Overview
Estimate the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals in the US, using a difference-in-differences approach comparing ages 26-30 (treated) vs 31-35 (control) before and after DACA implementation in 2012.

## Key Decisions and Rationale

### 1. Sample Construction

**Population Definition:**
- Hispanic-Mexican ethnicity: HISPAN = 1
- Born in Mexico: BPL = 200
- Non-citizen: CITIZEN = 3 (used as proxy for undocumented status)

**Rationale:** The instructions specify focusing on "ethnically Hispanic-Mexican Mexican-born people." Since documentation status is not directly observable in the ACS, non-citizen status is used as a proxy following standard practice in the literature.

### 2. DACA Eligibility Criteria Implementation

**Criterion 1 - Arrived before age 16:**
- Calculated as: YRIMMIG - BIRTHYR < 16
- Required YRIMMIG > 0 (valid immigration year)

**Criterion 2 - Under 31 as of June 15, 2012:**
- Age calculated using BIRTHYR and BIRTHQTR
- For Q3-Q4 births (July-December), age decremented by 1 since birthday had not occurred by June 15

**Criterion 3 - Continuous residence since June 15, 2007:**
- Approximated by YRIMMIG <= 2007
- This ensures individual was in US by 2007

**Rationale:** These criteria operationalize the DACA eligibility requirements as closely as possible given ACS data limitations. The continuous residence proxy is conservative.

### 3. Treatment and Control Groups

**Treatment Group:** Ages 26-30 as of June 15, 2012
- Would be DACA-eligible based on age
- Rationale: Per instructions, this is the specified treatment group

**Control Group:** Ages 31-35 as of June 15, 2012
- Just over the age cutoff for DACA eligibility
- Otherwise similar to treatment group in terms of other eligibility criteria
- Rationale: Per instructions, this provides a valid comparison group

### 4. Time Periods

**Pre-period:** 2006-2011
- Years before DACA implementation
- 2006 chosen as start to ensure data consistency

**Post-period:** 2013-2016
- Years after DACA implementation
- 2012 excluded because DACA was implemented mid-year (June 15)
- Rationale: Cannot distinguish pre- and post-DACA observations within 2012

### 5. Outcome Variable

**Full-time employment:** UHRSWORK >= 35
- Binary indicator: 1 if usually works 35+ hours per week, 0 otherwise
- Rationale: This follows the standard BLS definition of full-time employment

### 6. Covariates

**Included in preferred specification:**
- Female (SEX = 2): Controls for gender differences in employment
- Married (MARST = 1 or 2): Controls for household composition effects
- High school or more (EDUC >= 6): Controls for education effects
- Year fixed effects: Controls for time-varying factors affecting all individuals
- State fixed effects (STATEFIP): Controls for cross-state labor market differences

**Rationale:** These covariates improve precision and control for observable differences between groups that may affect employment.

### 7. Estimation Approach

**Primary method:** Difference-in-differences
- Compares change in outcomes over time between treatment and control groups
- Key identifying assumption: Parallel trends (absent DACA, both groups would have experienced similar changes)

**Weighting:** Person weights (PERWT) used for population-representative estimates

**Standard errors:** Heteroskedasticity-robust (HC1)

**Event study:** Estimated to test parallel trends assumption in pre-period

### 8. Specification Choices

**Preferred specification:** DiD with year FE, state FE, and demographic controls
- Provides flexible time trends and controls for geographic heterogeneity
- Balances bias-variance tradeoff

---

## Commands and Analysis Log

### Step 1: Data Exploration
```
Date: 2024
Command: Read data dictionary (acs_data_dict.txt)
Purpose: Understand variable definitions and coding
```

### Step 2: Data Loading
```python
df = pd.read_csv('data/data.csv')
# Result: 33,851,424 observations loaded
# Years: 2006-2016
```

### Step 3: Sample Restrictions
```python
# Hispanic-Mexican: HISPAN == 1
# Result: 2,945,521 observations

# Born in Mexico: BPL == 200
# Result: 991,261 observations

# Non-citizen: CITIZEN == 3
# Result: 701,347 observations

# Arrived before age 16: YRIMMIG - BIRTHYR < 16 and YRIMMIG > 0
# Result: 205,327 observations

# Continuous residence: YRIMMIG <= 2007
# Result: 195,023 observations
```

### Step 4: Define Treatment/Control Groups
```python
# Calculate age as of June 15, 2012
age_june_2012 = 2012 - BIRTHYR
# Adjust for Q3-Q4 births (birthday not reached by June 15)
if BIRTHQTR in [3, 4]: age_june_2012 -= 1

# Treatment: ages 26-30
# Control: ages 31-35
# Final analysis sample (excluding 2012): 43,238
```

### Step 5: Define Outcome
```python
fulltime = (UHRSWORK >= 35).astype(int)
```

### Step 6: Run DiD Regressions
```python
# Model 1: Basic DiD (unweighted)
# Model 2: Basic DiD (weighted)
# Model 3: DiD + demographics
# Model 4: DiD + year FE
# Model 5: DiD + year FE + state FE (PREFERRED)
```

### Step 7: Event Study
```python
# Year-specific treatment effects with 2011 as reference
# Tests parallel trends assumption
```

### Step 8: Generate Figures
```python
# Figure 1: Event study plot
# Figure 2: Employment trends
# Figure 3: DiD illustration
# Figure 4: Model comparison
```

### Step 9: Compile LaTeX Report
```bash
pdflatex -interaction=nonstopmode replication_report_31.tex
# Run 4 times for proper cross-references
# Output: replication_report_31.pdf (20 pages)
```

---

## Results Summary

### Preferred Estimate (Model 5: DiD with Year FE, State FE, Demographics)

| Metric | Value |
|--------|-------|
| DiD Effect | 0.0458 |
| Standard Error | 0.0107 |
| 95% CI | [0.0249, 0.0667] |
| t-statistic | 4.29 |
| p-value | < 0.0001 |
| Sample Size | 43,238 |

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately **4.6 percentage points** among Hispanic-Mexican Mexican-born non-citizens who met the eligibility criteria. This effect is statistically significant at the 1% level and represents a roughly 7% increase from the treatment group's baseline employment rate.

### Robustness
- Effect is consistent across all specifications (range: 0.046 to 0.059)
- Similar effects for males and females
- Larger effects for those with higher education
- Pre-trends analysis supports parallel trends assumption

---

## Files Generated

1. **analysis.py** - Main analysis script
2. **create_figures.py** - Figure generation script
3. **replication_report_31.tex** - LaTeX source
4. **replication_report_31.pdf** - Final report (20 pages)
5. **results_summary.csv** - Key results
6. **model_comparison.csv** - Model comparison table
7. **event_study_results.csv** - Event study coefficients
8. **fulltime_rates.csv** - Employment rates by group/period
9. **demographics_summary.csv** - Demographic characteristics
10. **figure1_event_study.png/pdf** - Event study figure
11. **figure2_trends.png/pdf** - Employment trends figure
12. **figure3_did_illustration.png/pdf** - DiD illustration
13. **figure4_model_comparison.png/pdf** - Model comparison figure
14. **sample_flow.txt** - Sample construction summary
15. **run_log_31.md** - This log file

---

## Verification Checklist

- [x] Data loaded correctly (33.8M observations)
- [x] Sample restrictions applied properly
- [x] Treatment/control groups defined correctly
- [x] Pre/post periods defined correctly (2012 excluded)
- [x] Outcome variable created (UHRSWORK >= 35)
- [x] Weights applied (PERWT)
- [x] Multiple specifications estimated
- [x] Event study for parallel trends
- [x] Heterogeneity analysis (by sex, education)
- [x] Figures generated
- [x] LaTeX report compiled
- [x] All required files in output folder
