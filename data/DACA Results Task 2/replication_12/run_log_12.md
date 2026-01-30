# Run Log - Replication 12

## Date: 2026-01-26

## Overview
This run log documents the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico.

---

## Key Decisions

### 1. Research Design
- **Design**: Difference-in-Differences (DiD)
- **Treatment group**: Ages 26-30 at time of DACA implementation (June 15, 2012)
- **Control group**: Ages 31-35 at time of DACA implementation
- **Pre-treatment period**: 2006-2011
- **Post-treatment period**: 2013-2016 (excluding 2012 due to partial treatment)
- **Outcome**: Full-time employment (UHRSWORK >= 35 hours/week)

### 2. Sample Restrictions
The analysis sample includes individuals who:
- Have Hispanic-Mexican ethnicity (HISPAN == 1)
- Were born in Mexico (BPL == 200)
- Are not U.S. citizens (CITIZEN == 3)
- Immigrated before age 16 (derived from YRIMMIG and BIRTHYR)
- Have been in the U.S. since at least June 15, 2007 (YRIMMIG <= 2007)
- Were present in the U.S. on June 15, 2012 (implied by survey presence)
- Age 26-30 or 31-35 as of June 15, 2012

### 3. Variable Definitions
- **YEAR**: Survey year from ACS (2006-2016)
- **HISPAN**: Hispanic origin (1 = Mexican)
- **BPL**: Birthplace (200 = Mexico)
- **CITIZEN**: Citizenship status (3 = Not a citizen)
- **YRIMMIG**: Year of immigration
- **BIRTHYR**: Year of birth
- **BIRTHQTR**: Quarter of birth (used for age precision)
- **UHRSWORK**: Usual hours worked per week (>=35 = full-time)
- **PERWT**: Person weight for population estimates

### 4. Age Calculation
- Age as of June 15, 2012 calculated from BIRTHYR and BIRTHQTR
- For individuals born in quarters 3 or 4 (July-December), subtract 1 from (2012 - BIRTHYR) since they haven't had their birthday by June 15
- Treatment: Ages 26-30 as of June 15, 2012
- Control: Ages 31-35 as of June 15, 2012

---

## Commands and Steps

### Step 1: Data Exploration
```bash
# Reviewed data dictionary
head -1 "C:/Users/seraf/DACA Results Task 2/replication_12/data/data.csv"
# Output: Column headers including YEAR, PERWT, HISPAN, BPL, CITIZEN, YRIMMIG, BIRTHYR, BIRTHQTR, UHRSWORK, etc.
```

Key variables identified:
- Demographics: YEAR, AGE, BIRTHYR, BIRTHQTR, SEX, MARST
- Ethnicity/Immigration: HISPAN, HISPAND, BPL, BPLD, CITIZEN, YRIMMIG
- Employment: EMPSTAT, LABFORCE, UHRSWORK, WKSWORK2
- Weights: PERWT

### Step 2: Data Loading and Cleaning
```python
# Load data in chunks (file is ~6GB)
cols_to_load = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
                'MARST', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT',
                'LABFORCE', 'UHRSWORK']

# Filter criteria:
# 1. HISPAN == 1 (Mexican Hispanic)
# 2. BPL == 200 (Born in Mexico)
# 3. CITIZEN == 3 (Not a citizen)
```

Sample sizes at each filtering step:
- Initial filtered (Hispanic-Mexican, born in Mexico, non-citizen): 701,347 observations
- After requiring arrival before age 16: 205,327 observations
- After requiring presence since 2007: 195,023 observations
- After selecting age groups 26-30 and 31-35: 47,418 observations
- After excluding 2012: 43,238 observations (FINAL SAMPLE)

### Step 3: Variable Construction
```python
# Calculate age as of June 15, 2012
df['age_june2012'] = 2012 - df['BIRTHYR']
df.loc[df['BIRTHQTR'] >= 3, 'age_june2012'] -= 1  # Adjust for birthday timing

# Age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Treatment indicator
df['treat'] = np.where((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30), 1,
              np.where((df['age_june2012'] >= 31) & (df['age_june2012'] <= 35), 0, np.nan))

# Post-DACA indicator
df['post'] = np.where(df['YEAR'] >= 2013, 1, 0)

# Outcome: Full-time employment
df['fulltime'] = np.where(df['UHRSWORK'] >= 35, 1, 0)

# DiD interaction term
df['treat_post'] = df['treat'] * df['post']

# Control variables
df['female'] = np.where(df['SEX'] == 2, 1, 0)
df['married'] = np.where(df['MARST'] <= 2, 1, 0)
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']
```

### Step 4: Difference-in-Differences Analysis
```python
# Five model specifications were estimated:
# Model 1: Basic DiD (no controls)
# Model 2: DiD with demographic controls (female, married, years_in_us)
# Model 3: DiD with controls and year fixed effects
# Model 4: DiD with controls, year FE, and state FE
# Model 5: Weighted DiD with controls and year FE (PREFERRED)

# All models use heteroskedasticity-robust (HC1) standard errors
```

### Step 5: Robustness Checks
```python
# 1. Event study specification (dynamic DiD)
# 2. Heterogeneity analysis by sex
# 3. Placebo test with ages 36-40 vs 41-45 (both ineligible)
```

### Step 6: LaTeX Report Generation
```bash
# Compile LaTeX document
cd "C:/Users/seraf/DACA Results Task 2/replication_12"
pdflatex -interaction=nonstopmode replication_report_12.tex
pdflatex -interaction=nonstopmode replication_report_12.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_12.tex  # Third pass for stable output
```

---

## Results Summary

### Main Finding (Preferred Estimate)
- **DiD Coefficient**: 0.0470
- **Standard Error**: 0.0107 (robust)
- **95% Confidence Interval**: [0.0260, 0.0680]
- **p-value**: < 0.001

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 4.7 percentage points among eligible Hispanic-Mexican immigrants born in Mexico.

### Full-Time Employment Rates by Group and Period
|                    | Pre (2006-2011) | Post (2013-2016) |
|--------------------|-----------------|------------------|
| Control (31-35)    | 64.6%           | 61.4%            |
| Treatment (26-30)  | 61.5%           | 63.4%            |

Raw DiD: (63.4% - 61.5%) - (61.4% - 64.6%) = 1.9% + 3.2% = 5.2%

### Model Comparison
| Specification        | Coefficient | Std Error | N      |
|---------------------|-------------|-----------|--------|
| Basic DiD           | 0.0516      | 0.0100    | 43,238 |
| With Controls       | 0.0464      | 0.0093    | 43,238 |
| Year FE             | 0.0462      | 0.0092    | 43,238 |
| Year + State FE     | 0.0455      | 0.0092    | 43,238 |
| Weighted (Preferred)| 0.0470      | 0.0107    | 43,238 |

### Robustness Checks
1. **Event Study**: Pre-treatment coefficients not significantly different from zero, supporting parallel trends assumption
2. **By Sex**: Males = 0.034 (SE=0.011), Females = 0.052 (SE=0.015) - positive effects for both
3. **Placebo Test**: Ages 36-40 vs 41-45 shows coefficient of 0.019 (SE=0.012), not significant

---

## Output Files

### Required Deliverables
1. `replication_report_12.tex` - LaTeX source file
2. `replication_report_12.pdf` - 21-page PDF report
3. `run_log_12.md` - This log file

### Supporting Files
- `analysis.py` - Python analysis script
- `results_summary.csv` - Key results in CSV format
- `regression_results.txt` - Full regression output
- `event_study_results.csv` - Event study coefficients
- `summary_statistics.csv` - Descriptive statistics

---

## Software and Packages

- **Python 3.x**
  - pandas
  - numpy
  - statsmodels
  - scipy

- **LaTeX**
  - pdflatex (MiKTeX 25.12)
  - packages: amsmath, booktabs, tikz, pgfplots, hyperref, etc.

---

## Notes

1. Year 2012 was excluded because DACA was implemented mid-year (June 15, 2012), making it impossible to distinguish pre/post observations within that year in ACS data.

2. The ACS does not directly identify undocumented immigrants. The sample uses citizenship status (CITIZEN = 3, "Not a citizen") as a proxy, which likely captures most of the target population but may include some documented non-citizens.

3. The analysis focuses on intent-to-treat effects (DACA eligibility), not treatment-on-treated effects (actual DACA receipt), since DACA status is not observed in the ACS.

4. Person weights (PERWT) were used in the preferred specification to produce population-representative estimates.
