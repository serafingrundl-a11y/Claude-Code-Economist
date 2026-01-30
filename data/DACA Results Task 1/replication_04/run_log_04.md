# Replication Run Log - Replication 04

## Project: DACA Effect on Full-Time Employment Among Hispanic-Mexican, Mexican-Born Immigrants

### Date: 2026-01-24

---

## 1. Initial Setup and Data Understanding

### 1.1 Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

### 1.2 Data Sources
- **Main data**: `data/data.csv` - ACS data from IPUMS (2006-2016)
- **Data dictionary**: `data/acs_data_dict.txt`
- **Supplemental state data**: `data/state_demo_policy.csv` (optional, not used in main analysis)

### 1.3 Data Structure
- Total observations: ~33.8 million rows
- Years covered: 2006-2016 (1-year ACS samples)
- Key variables identified:
  - `YEAR`: Survey year
  - `HISPAN`/`HISPAND`: Hispanic origin (Mexican = 1, detailed = 100-107)
  - `BPL`/`BPLD`: Birthplace (Mexico = 200/20000)
  - `CITIZEN`: Citizenship status (3 = Not a citizen)
  - `BIRTHYR`: Birth year
  - `BIRTHQTR`: Birth quarter (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
  - `YRIMMIG`: Year of immigration
  - `UHRSWORK`: Usual hours worked per week
  - `EMPSTAT`: Employment status (1 = Employed)
  - `AGE`: Age
  - `SEX`: Sex (1 = Male, 2 = Female)
  - `MARST`: Marital status (1 = Married, spouse present)
  - `EDUC`: Education level
  - `STATEFIP`: State FIPS code
  - `PERWT`: Person weight

---

## 2. Key Analytical Decisions

### 2.1 DACA Eligibility Criteria (from instructions)
To be eligible for DACA, individuals must:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007 (immigrated by 2007)
4. Were present in the US on June 15, 2012 and did not have lawful status

### 2.2 Sample Restrictions
1. **Ethnic/origin restriction**: Hispanic-Mexican ethnicity (HISPAN = 1) AND born in Mexico (BPL = 200)
2. **Citizenship restriction**: Non-citizens (CITIZEN = 3) - assumed undocumented per instructions
3. **Working age**: Ages 16-64
4. **Valid immigration year**: YRIMMIG > 0
5. **Exclude 2012**: Due to mid-year DACA implementation

### 2.3 Treatment and Control Group Design

**Treatment group (DACA-eligible)**:
- Hispanic-Mexican, born in Mexico
- Non-citizen
- Immigrated before age 16 (YRIMMIG - BIRTHYR < 16)
- Born after June 1981 (BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3))
- Immigrated by 2007 (YRIMMIG <= 2007)

**Control group (age-ineligible)**:
- Hispanic-Mexican, born in Mexico
- Non-citizen
- Immigrated before age 16
- Born before 1981 (BIRTHYR < 1981) - too old for DACA
- Immigrated by 2007

**Rationale**: This control group leverages the arbitrary age cutoff in DACA eligibility. The comparison isolates the effect of eligibility from other characteristics that might affect employment.

### 2.4 Outcome Variable
- **Full-time employment**: Binary indicator = 1 if UHRSWORK >= 35, 0 otherwise
- **Alternative outcome**: Any employment (EMPSTAT = 1)

### 2.5 Identification Strategy
**Difference-in-Differences (DiD)**:
- Pre-period: 2006-2011 (before DACA implementation)
- Post-period: 2013-2016 (after DACA, as specified in instructions)
- 2012 excluded due to mid-year implementation (June 15, 2012)

---

## 3. Sample Construction Results

| Step | Sample Size |
|------|-------------|
| Full ACS sample (2006-2016, excl. 2012) | 33,851,424 |
| Hispanic-Mexican (HISPAN = 1) | 2,945,521 |
| Mexico-born (BPL = 200) | 991,261 |
| Non-citizens (CITIZEN = 3) | 701,347 |
| Excluding 2012 | 636,722 |
| Ages 16-64 | 561,470 |
| Valid immigration year | 561,470 |
| **Analysis Sample** | **136,637** |
| - Treatment (DACA eligible) | 83,611 |
| - Control (age-ineligible) | 53,026 |

---

## 4. Main Results

### 4.1 Raw Difference-in-Differences

| Group | Pre-DACA | Post-DACA | Difference |
|-------|----------|-----------|------------|
| Treatment (DACA Eligible) | 0.452 | 0.521 | +0.069 |
| Control (Age-Ineligible) | 0.678 | 0.635 | -0.042 |
| **DiD Estimate** | | | **0.112** |

### 4.2 Regression Results

| Model | DiD Coefficient | Std. Error | 95% CI |
|-------|----------------|------------|--------|
| Model 1: Basic DiD | 0.1117*** | 0.0056 | [0.101, 0.123] |
| Model 2: + Demographics | 0.0136** | 0.0053 | [0.003, 0.024] |
| Model 3: + Year FE | 0.0033 | 0.0052 | [-0.007, 0.014] |
| **Model 4: + State FE (Preferred)** | **0.0021** | **0.0052** | **[-0.008, 0.012]** |

### 4.3 Preferred Estimate Interpretation
- **Effect**: 0.21 percentage points increase in full-time employment
- **Statistical significance**: NOT significant at 5% level
- **Sample size**: 136,637
- **R-squared**: 0.221

### 4.4 Robustness Checks

| Specification | DiD Estimate | Std. Error | 95% CI |
|--------------|--------------|------------|--------|
| Alternative outcome (any employment) | 0.0016 | 0.0052 | [-0.009, 0.012] |
| Males only | -0.0238*** | 0.0066 | [-0.037, -0.011] |
| Females only | 0.0239*** | 0.0083 | [0.008, 0.040] |

---

## 5. Commands Executed

### 5.1 Data Loading and Processing
```python
# Load data with selected columns
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN',
               'YRIMMIG', 'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK']
df = pd.read_csv('data/data.csv', usecols=cols_needed)

# Sample restrictions
df_mex = df[df['HISPAN'] == 1]  # Hispanic-Mexican
df_mex = df_mex[df_mex['BPL'] == 200]  # Mexico-born
df_mex = df_mex[df_mex['CITIZEN'] == 3]  # Non-citizen
df_analysis = df_mex[df_mex['YEAR'] != 2012]  # Exclude 2012
df_analysis = df_analysis[(df_analysis['AGE'] >= 16) & (df_analysis['AGE'] <= 64)]
```

### 5.2 Variable Creation
```python
# DACA eligibility
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df['arrived_before_16'] = (df['age_at_immig'] < 16).astype(int)
df['under_31_2012'] = ((df['BIRTHYR'] >= 1982) |
                       ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))).astype(int)
df['continuous_presence'] = (df['YRIMMIG'] <= 2007).astype(int)
df['daca_eligible'] = (df['arrived_before_16'] & df['under_31_2012'] &
                       df['continuous_presence']).astype(int)

# Post-DACA indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# DiD interaction
df['did'] = df['daca_eligible'] * df['post']

# Outcome
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
```

### 5.3 Regression Models
```python
import statsmodels.formula.api as smf

# Model 4 (Preferred): DiD with Year and State FE
model4 = smf.wls(
    'fulltime ~ daca_eligible + C(YEAR) + C(STATEFIP) + did + AGE + age_sq + female + married + educ_hs',
    data=df_did, weights=df_did['PERWT']
).fit()
```

### 5.4 Figure Generation
```bash
python create_figures.py
# Outputs:
# - figure1_parallel_trends.pdf
# - figure2_event_study.pdf
# - figure3_did_illustration.pdf
# - figure4_sample_size.pdf
```

### 5.5 LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_04.tex
pdflatex -interaction=nonstopmode replication_report_04.tex  # Second pass for refs
```

---

## 6. Key Decisions and Rationale

### 6.1 Control Group Choice
**Decision**: Use age-ineligible immigrants (born before 1981) who otherwise meet DACA criteria.

**Rationale**: This leverages the arbitrary age cutoff in DACA eligibility. Alternative control groups (arrived after 2007, arrived at age 16+) may differ in unobservable ways that affect employment trends.

**Limitation**: Age differences between treatment and control groups are substantial (mean age 22.7 vs 39.4), which may create differential exposure to macroeconomic conditions.

### 6.2 Exclusion of 2012
**Decision**: Exclude survey year 2012 from analysis.

**Rationale**: DACA was implemented on June 15, 2012. ACS does not indicate survey month, so 2012 observations cannot be classified as pre- or post-treatment. Including 2012 would introduce measurement error.

### 6.3 Full-Time Definition
**Decision**: Define full-time as UHRSWORK >= 35 hours per week.

**Rationale**: This follows the standard BLS definition of full-time employment and matches the research question specification.

### 6.4 Weighting
**Decision**: Use ACS person weights (PERWT) for all analyses.

**Rationale**: ACS uses a complex survey design. Person weights are necessary to produce population-representative estimates.

### 6.5 Fixed Effects
**Decision**: Include both year and state fixed effects in preferred specification.

**Rationale**: Year FE control for national trends affecting all groups. State FE control for time-invariant state-level differences in labor markets and immigrant populations.

---

## 7. Files Produced

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `analysis_summary.csv` | Summary statistics |
| `trends_data.csv` | Yearly trends by treatment status |
| `event_study_results.csv` | Event study coefficients |
| `figure1_parallel_trends.pdf` | Parallel trends figure |
| `figure2_event_study.pdf` | Event study figure |
| `figure3_did_illustration.pdf` | DiD illustration |
| `figure4_sample_size.pdf` | Sample size by year |
| `replication_report_04.tex` | LaTeX report source |
| `replication_report_04.pdf` | Final report (22 pages) |
| `run_log_04.md` | This run log |

---

## 8. Conclusion

The analysis finds that DACA eligibility had a small, statistically insignificant effect on full-time employment among Hispanic-Mexican, Mexican-born non-citizens. The preferred estimate (Model 4 with year and state fixed effects) is 0.21 percentage points (95% CI: -0.82 to 1.23 pp).

The large difference between the raw DiD (11.2 pp) and the adjusted estimate (0.21 pp) highlights the importance of controlling for compositional differences between groups, particularly age.

Heterogeneity by gender is notable: DACA appears to decrease full-time employment for males (-2.4 pp) while increasing it for females (+2.4 pp), with both effects statistically significant. This warrants further investigation.

---

*Log completed: 2026-01-24*
