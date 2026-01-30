# Run Log - Replication 93

## Project Overview
Replication study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Date Started
2026-01-25

---

## Step 1: Data Review and Understanding

### Data Files
- `data.csv`: Main ACS data file (2006-2016, 1-year ACS samples) - 33,851,424 observations
- `acs_data_dict.txt`: Data dictionary with variable definitions
- `state_demo_policy.csv`: Optional state-level policy data (not used in main analysis)

### Key Variables Identified
- **YEAR**: Survey year (2006-2016)
- **HISPAN**: Hispanic origin (1 = Mexican)
- **HISPAND**: Detailed Hispanic origin (100-107 = Mexican variations)
- **BPL**: Birthplace (200 = Mexico)
- **BPLD**: Detailed birthplace (20000 = Mexico)
- **CITIZEN**: Citizenship status (3 = Not a citizen)
- **YRIMMIG**: Year of immigration
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Birth quarter (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- **AGE**: Age at survey
- **UHRSWORK**: Usual hours worked per week (outcome: >=35 for full-time)
- **PERWT**: Person weight for population estimates
- **EMPSTAT**: Employment status
- **STATEFIP**: State FIPS code

---

## Step 2: DACA Eligibility Criteria Definition

Based on the program requirements from the instructions:

### Eligibility Requirements:
1. **Arrived before 16th birthday**: Age at immigration < 16
   - Formula: YRIMMIG - BIRTHYR < 16

2. **Under 31 as of June 15, 2012**: Born after June 15, 1981
   - BIRTHYR > 1981, OR
   - BIRTHYR == 1981 AND BIRTHQTR >= 3 (July or later)

3. **Continuous presence since June 15, 2007**: YRIMMIG <= 2007

4. **Present in US on June 15, 2012**: Assumed true for those in ACS data who arrived by 2007

5. **No lawful status**: CITIZEN == 3 (Not a citizen, no naturalization)

### Population Restriction:
- Hispanic-Mexican ethnicity: HISPAN == 1
- Born in Mexico: BPL == 200

### Analytical Decisions:
- Exclude 2012 from main analysis (ambiguous treatment timing - DACA implemented mid-year June 15)
- Pre-period: 2006-2011
- Post-period: 2013-2016
- Will use difference-in-differences strategy comparing DACA-eligible to similar non-eligible non-citizens

---

## Step 3: Identification Strategy

### Research Design: Difference-in-Differences

**Treatment Group**: DACA-eligible Mexican non-citizens
- Met all age/arrival requirements
- Would be eligible for DACA

**Control Group**: Similar Mexican non-citizens NOT eligible for DACA
- Same demographic profile (Hispanic-Mexican, born in Mexico, non-citizen)
- Did NOT meet age/arrival requirements (e.g., arrived too late, arrived after age 16, too old)

**Key Identifying Assumption**: Parallel trends - absent DACA, the employment trends for eligible and non-eligible groups would have been similar.

### Model Specification:
```
FullTime_it = beta_0 + beta_1*Eligible_i + beta_2*Post_t + beta_3*(Eligible_i * Post_t) + X_it*gamma + epsilon_it
```

Where:
- FullTime_it: Binary indicator for full-time employment (UHRSWORK >= 35)
- Eligible_i: Binary indicator for DACA eligibility
- Post_t: Binary indicator for post-DACA period (2013-2016)
- beta_3: Causal effect of DACA eligibility on full-time employment (coefficient of interest)
- X_it: Controls (age, sex, education, marital status, state fixed effects, year fixed effects)

---

## Step 4: Data Preparation and Sample Construction

### Commands Executed:
```python
# Load data
df = pd.read_csv('data/data.csv')

# Apply sample restrictions
df = df[df['HISPAN'] == 1]      # Hispanic-Mexican
df = df[df['BPL'] == 200]       # Born in Mexico
df = df[df['CITIZEN'] == 3]     # Non-citizen
df = df[df['YEAR'] != 2012]     # Exclude 2012
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)]  # Working age
df = df[df['YRIMMIG'] > 0]      # Valid immigration year
```

### Sample Sizes After Each Restriction:
| Restriction | N |
|------------|---|
| Full ACS sample | 33,851,424 |
| Hispanic-Mexican | 2,945,521 |
| Born in Mexico | 991,261 |
| Non-citizen | 701,347 |
| Exclude 2012 | 636,722 |
| Ages 18-64 | 547,614 |

### Final Analytic Sample: 547,614 observations

---

## Step 5: Variable Construction

### DACA Eligibility Variable:
```python
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df['arrived_before_16'] = (df['age_at_immig'] < 16).astype(int)
df['under_31_2012'] = ((df['BIRTHYR'] > 1981) |
                       ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))).astype(int)
df['arrived_by_2007'] = (df['YRIMMIG'] <= 2007).astype(int)
df['daca_eligible'] = (df['arrived_before_16'] & df['under_31_2012'] & df['arrived_by_2007']).astype(int)
```

### Outcome Variable:
```python
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
```

### Control Variables:
- Age and age squared
- Female indicator (SEX == 2)
- Married indicator (MARST <= 2)
- Education indicators (less than HS, HS, some college, college+)

---

## Step 6: Analysis Execution

### Main Analysis Run:
```bash
python analysis.py
```

### Key Results:

#### Sample Composition:
- DACA eligible: 71,347 (13.0%)
- Non-eligible: 476,267 (87.0%)
- Pre-period observations: 336,493
- Post-period observations: 211,121

#### Descriptive Statistics:
| Group | Pre-FT Rate | Post-FT Rate | Change |
|-------|-------------|--------------|--------|
| Eligible | 50.98% | 54.71% | +3.73 pp |
| Non-eligible | 60.46% | 58.14% | -2.32 pp |

Raw DiD Estimate: 6.05 pp

#### Main Regression Results (Preferred Specification):
- Model: OLS with state and year fixed effects, clustered standard errors at state level
- DiD Coefficient (eligible_post): **0.0185**
- Standard Error: 0.0042
- 95% CI: [0.0102, 0.0267]
- t-statistic: 4.375
- p-value: < 0.001
- N: 547,614
- R-squared: 0.209

---

## Step 7: Robustness Checks

| Specification | Coefficient | SE | N |
|--------------|-------------|-----|---|
| Preferred (baseline) | 0.0185*** | 0.0042 | 547,614 |
| Employment outcome | 0.0296*** | 0.0075 | 547,614 |
| Labor force only | 0.0056* | 0.0030 | 390,506 |
| Males only | 0.0113*** | 0.0036 | 296,109 |
| Females only | 0.0165** | 0.0073 | 251,505 |
| Young adults (16-35) | 0.0085 | 0.0057 | 253,373 |
| Weighted (PERWT) | 0.0176*** | 0.0036 | 547,614 |

---

## Step 8: Event Study Analysis

Event study coefficients (relative to 2011):
| Year | Coefficient | SE |
|------|-------------|-----|
| 2006 | 0.0085 | 0.0093 |
| 2007 | 0.0057 | 0.0053 |
| 2008 | 0.0125 | 0.0092 |
| 2009 | 0.0116 | 0.0081 |
| 2010 | 0.0059 | 0.0104 |
| 2011 | 0 (ref) | -- |
| 2013 | 0.0043 | 0.0092 |
| 2014 | 0.0219* | 0.0128 |
| 2015 | 0.0378*** | 0.0096 |
| 2016 | 0.0380*** | 0.0077 |

Pre-treatment coefficients are small and statistically insignificant, supporting parallel trends assumption.

---

## Step 9: Figure Generation

```bash
python create_figures.py
```

### Figures Created:
1. `figure1_trends.png/pdf` - Full-time employment trends by DACA eligibility
2. `figure2_eventstudy.png/pdf` - Event study plot
3. `figure3_age_dist.png/pdf` - Age distribution by eligibility
4. `figure4_coef_comparison.png/pdf` - Model comparison
5. `figure5_robustness.png/pdf` - Robustness checks

---

## Step 10: Report Compilation

### LaTeX Compilation:
```bash
pdflatex -interaction=nonstopmode replication_report_93.tex
pdflatex -interaction=nonstopmode replication_report_93.tex
pdflatex -interaction=nonstopmode replication_report_93.tex
```

Three passes required for table of contents and cross-references.

### Output:
- `replication_report_93.pdf` - 25 pages

---

## Final Deliverables

| File | Description | Status |
|------|-------------|--------|
| `replication_report_93.tex` | LaTeX source | Complete |
| `replication_report_93.pdf` | Final report | Complete |
| `run_log_93.md` | This log file | Complete |
| `analysis.py` | Main analysis script | Complete |
| `create_figures.py` | Figure generation script | Complete |
| `results_summary.csv` | Results table | Complete |
| `key_results.txt` | Key statistics | Complete |
| `figure1-5_*.png/pdf` | Figures | Complete |

---

## Key Analytical Decisions Summary

1. **Sample Definition**: Hispanic-Mexican, Mexican-born non-citizens ages 18-64
2. **DACA Eligibility**: Based on age at arrival (<16), birth date (>June 15, 1981), arrival year (<=2007)
3. **Outcome**: Full-time employment (UHRSWORK >= 35)
4. **Method**: Difference-in-differences with state and year fixed effects
5. **Standard Errors**: Clustered at state level
6. **Year 2012**: Excluded due to mid-year DACA implementation

---

## Final Result

**DACA eligibility increased full-time employment by 1.85 percentage points (95% CI: 1.02-2.67 pp, p < 0.001).**

This represents a relative increase of approximately 3.6% from the pre-DACA baseline of 51% full-time employment among eligible individuals.

---

*Log completed: 2026-01-25*
