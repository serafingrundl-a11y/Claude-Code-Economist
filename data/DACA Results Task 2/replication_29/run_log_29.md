# Run Log for DACA Replication Study (Participant 29)

## Project Overview
Replication study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Design
- **Treatment Group**: Ages 26-30 as of June 15, 2012 (DACA-eligible based on age)
- **Control Group**: Ages 31-35 as of June 15, 2012 (would have been eligible if younger)
- **Method**: Difference-in-Differences (DiD)
- **Pre-treatment period**: 2006-2011
- **Post-treatment period**: 2013-2016 (as specified in instructions)
- **Outcome**: Full-time employment (working 35+ hours per week)

## Data Source
American Community Survey (ACS) from IPUMS USA, 2006-2016 (1-year files)

---

## Key Decisions Log

### Decision 1: Target Population Definition
**Variables Used:**
- HISPAN = 1 (Hispanic-Mexican ethnicity)
- BPL = 200 (Born in Mexico)
- CITIZEN = 3 (Not a citizen - proxy for undocumented status)

**Rationale:** Per instructions, focus on "ethnically Hispanic-Mexican Mexican-born people." The non-citizen filter serves as the best available proxy for undocumented status, as the ACS does not directly identify documentation status.

### Decision 2: DACA Eligibility Criteria Implementation
**Criteria applied:**
1. Arrived before age 16: `YRIMMIG - BIRTHYR < 16`
2. Continuous presence since June 15, 2007: `YRIMMIG <= 2007`

**Criteria NOT fully implemented (data limitations):**
- Education requirement (school enrollment, HS diploma, GED, or military discharge)
- Clean criminal record

**Rationale:** The ACS allows implementation of age-at-arrival and year-of-arrival criteria. Education requirements are difficult to verify for enrollment status, and criminal record data is not available.

### Decision 3: Age on June 15, 2012 Calculation
**Method:**
```python
age_june_2012 = 2012 - BIRTHYR
# Adjust for those who hadn't had birthday yet by June 15
if BIRTHQTR in [3, 4]:  # July-Dec birth
    age_june_2012 -= 1
```

**Treatment Group:** Ages 26-30 on June 15, 2012
**Control Group:** Ages 31-35 on June 15, 2012

**Rationale:** DACA required applicants to be under 31 on June 15, 2012. Using birth quarter to more precisely determine age on the cutoff date. Those born in Q3 (Jul-Sep) or Q4 (Oct-Dec) had not yet had their birthday by mid-June.

### Decision 4: Exclusion of 2012
**Rationale:** The ACS does not record the month of interview, making it impossible to distinguish observations from before vs. after DACA implementation (June 15, 2012). Following instructions to examine effects in 2013-2016.

### Decision 5: Full-Time Employment Definition
**Definition:** UHRSWORK >= 35 hours per week

**Rationale:** Standard definition of full-time work in US labor statistics (35+ usual hours per week).

### Decision 6: Weighting
**Used PERWT** (person weight) for population-representative estimates.

**Rationale:** ACS is a complex survey sample; weights account for sampling design and non-response to produce nationally representative estimates.

### Decision 7: Standard Errors
**Method:** Heteroskedasticity-robust standard errors (HC1)

**Rationale:** Linear probability model with binary outcome may have heteroskedastic errors. HC1 provides consistent variance estimates.

### Decision 8: Model Specifications
Multiple specifications estimated for robustness:
1. Basic DiD (unweighted)
2. Weighted DiD
3. With covariates (sex, marital status, children, education)
4. Year fixed effects
5. State fixed effects

**Preferred Specification:** Weighted with covariates and robust SE (Column 3)
- Controls for observable differences between groups
- Uses survey weights for population representation
- Robust to heteroskedasticity

### Decision 9: Covariates Included
- Female (SEX = 2)
- Married (MARST = 1)
- Has children (NCHILD > 0)
- Education dummies: High school (EDUC 6-7), Some college (EDUC 8-9), College+ (EDUC 10+)

**Rationale:** These are strong predictors of employment that may differ between treatment/control groups and improve precision.

---

## Commands Executed

### Step 1: Data Loading
```python
import pandas as pd
needed_cols = ['YEAR', 'PERWT', 'BIRTHYR', 'BIRTHQTR', 'HISPAN', 'HISPAND',
               'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'UHRSWORK', 'EMPSTAT',
               'AGE', 'SEX', 'EDUC', 'MARST', 'STATEFIP', 'NCHILD']
df = pd.read_csv('data/data.csv', usecols=needed_cols, dtype=dtype_dict)
# Total observations: 33,851,424
```

### Step 2: Sample Filtering
```python
df = df[df['HISPAN'] == 1]           # Hispanic-Mexican: 2,945,521
df = df[df['BPL'] == 200]            # Born in Mexico: 991,261
df = df[df['CITIZEN'] == 3]          # Non-citizen: 701,347
df = df[df['YEAR'] != 2012]          # Exclude 2012: 636,722
# Age filter (26-35 on June 15, 2012): 164,874
df = df[df['YRIMMIG'] > 0]           # Valid immigration year
df = df[df['age_at_arrival'] < 16]   # Arrived before age 16
df = df[df['YRIMMIG'] <= 2007]       # Present since 2007
# Final sample: 43,238
```

### Step 3: Variable Construction
```python
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
df['post'] = (df['YEAR'] >= 2013).astype(int)
df['treated'] = ((df['age_june_2012'] >= 26) & (df['age_june_2012'] <= 30)).astype(int)
df['treat_x_post'] = df['treated'] * df['post']
```

### Step 4: DiD Estimation
```python
import statsmodels.formula.api as smf

# Preferred specification
model = smf.wls('fulltime ~ treated + post + treat_x_post + female + married +
                 has_children + educ_hs + educ_somecol + educ_college',
                data=df, weights=df['PERWT']).fit(cov_type='HC1')
```

### Step 5: Event Study
```python
# Create treatment x year interactions (reference year: 2011)
event_model = smf.wls('fulltime ~ treated + treat_x_2006 + treat_x_2007 +
                       treat_x_2008 + treat_x_2009 + treat_x_2010 +
                       treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 +
                       year_dummies + covariates',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')
```

### Step 6: LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_29.tex
pdflatex -interaction=nonstopmode replication_report_29.tex
pdflatex -interaction=nonstopmode replication_report_29.tex
```

---

## Results Summary

### Main Finding
**DACA eligibility increased full-time employment by approximately 4.27 percentage points.**

| Specification | DiD Coefficient | Std. Error | p-value | N |
|--------------|-----------------|------------|---------|-------|
| Basic DiD | 0.0516 | 0.0100 | <0.001 | 43,238 |
| Weighted | 0.0590 | 0.0098 | <0.001 | 43,238 |
| With Covariates | 0.0427 | 0.0107 | <0.001 | 43,238 |
| Year FE | 0.0405 | 0.0090 | <0.001 | 43,238 |
| State FE | 0.0420 | 0.0107 | <0.001 | 43,238 |

### Preferred Estimate
- **Coefficient:** 0.0427
- **Standard Error:** 0.0107
- **95% CI:** [0.0217, 0.0637]
- **p-value:** <0.001

### Sample Sizes
- Unweighted N: 43,238
- Treatment Group: 25,470 observations (weighted: 3,524,133)
- Control Group: 17,768 observations (weighted: 2,476,285)

### Event Study
Pre-treatment coefficients (2006-2010) are not statistically significant, supporting parallel trends assumption. Post-treatment coefficients are uniformly positive.

### Heterogeneity
- Males: 4.6 pp (p < 0.001)
- Females: 4.7 pp (p = 0.012)
- Less than HS: 3.5 pp (p = 0.056)
- HS or more: 7.9 pp (p < 0.001)

### Robustness
- Alternative outcome (any employment): 3.8 pp increase (p < 0.001)
- Alternative outcome (hours worked): 1.83 hours increase (p < 0.001)
- Wider age bandwidth (24-32 vs 33-40): 6.7 pp (p < 0.001)

---

## Files Generated

| File | Description |
|------|-------------|
| analysis.py | Main analysis script (Python) |
| replication_report_29.tex | LaTeX source for report |
| replication_report_29.pdf | Final PDF report (24 pages) |
| run_log_29.md | This log file |
| results_summary.csv | Summary of DiD coefficients |
| balance_table.csv | Pre-treatment balance check |
| event_study_results.csv | Event study coefficients |
| final_results.txt | Final results summary |
| figure1_parallel_trends.png | Trends by treatment status |
| figure2_difference.png | Treatment-control difference over time |
| figure3_sample_size.png | Sample size by year |
| figure4_event_study.png | Event study plot |

---

## Session Information
- Date: January 26, 2026
- Platform: Windows
- Python: 3.x with pandas, numpy, statsmodels, matplotlib
- LaTeX: pdfTeX (MiKTeX)

---

## Analysis Progress

| Step | Status | Notes |
|------|--------|-------|
| Read instructions | Complete | DACA study, DiD design |
| Examine data dictionary | Complete | Key variables identified |
| Load data | Complete | 33.8M observations |
| Construct treatment/control | Complete | 43,238 final sample |
| Run DiD analysis | Complete | Multiple specifications |
| Create visualizations | Complete | 4 figures |
| Write report | Complete | 24-page LaTeX report |
| Compile PDF | Complete | replication_report_29.pdf |
| Create run log | Complete | run_log_29.md |

**ALL DELIVERABLES COMPLETE**
