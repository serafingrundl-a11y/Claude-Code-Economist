# Run Log for DACA Replication Analysis (Replication 24)

## Overview
This document logs all commands, key decisions, and analytic choices made during the replication study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

---

## Date: January 26, 2026

### 1. Data Exploration

#### 1.1 Initial Data Load
```python
df = pd.read_csv('data/data.csv')
```
- Total observations: 33,851,424
- Years available: 2006-2016
- Variables: 54 columns including demographic, employment, and immigration variables

#### 1.2 Key Variables Identified from Data Dictionary (acs_data_dict.txt)
- **YEAR**: Survey year
- **HISPAN**: Hispanic origin (1 = Mexican)
- **BPL**: Birthplace (200 = Mexico)
- **CITIZEN**: Citizenship status (3 = Not a citizen)
- **YRIMMIG**: Year of immigration
- **BIRTHYR**: Year of birth
- **BIRTHQTR**: Quarter of birth
- **EMPSTAT**: Employment status (1 = Employed)
- **UHRSWORK**: Usual hours worked per week
- **SEX**: Gender (1 = Male, 2 = Female)
- **MARST**: Marital status (1 = Married, spouse present)
- **EDUC**: Educational attainment
- **STATEFIP**: State FIPS code
- **PERWT**: Person weight

---

### 2. Sample Selection Decisions

#### 2.1 Target Population
Per the research instructions, the target population is:
- Ethnically Hispanic-Mexican
- Born in Mexico
- DACA-eligible based on program criteria

#### 2.2 Sample Restrictions Applied (Sequential)

| Step | Restriction | Observations |
|------|------------|--------------|
| 0 | Full ACS 2006-2016 | 33,851,424 |
| 1 | Hispanic-Mexican (HISPAN == 1) | 2,945,521 |
| 2 | Born in Mexico (BPL == 200) | 991,261 |
| 3 | Non-citizens (CITIZEN == 3) | 701,347 |
| 4 | Arrived before age 16 | 205,327 |
| 5 | Continuous residence (arrived by 2007) | 195,023 |
| 6 | Age groups (26-35 as of June 2012) | 49,019 |
| 7 | Exclude 2012 | 44,725 |

#### 2.3 Key Decision: Non-Citizen Proxy
**Decision**: Use CITIZEN == 3 (Not a citizen) as proxy for undocumented status.

**Rationale**: The ACS does not directly identify documentation status. Following the instructions, I assume "anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes." CITIZEN == 3 captures non-citizens who are not naturalized.

**Alternative considered**: Including CITIZEN == 5 (Foreign born, citizenship status not reported). Decided against this because these individuals may include legal residents whose status is simply unreported.

#### 2.4 Key Decision: Age at Arrival Calculation
**Decision**: Calculate age at arrival as YRIMMIG - BIRTHYR.

**Rationale**: DACA requires arrival before 16th birthday. This calculation provides a reasonable approximation.

**Limitation**: YRIMMIG is year only, not month, so there is some imprecision.

#### 2.5 Key Decision: Continuous Residence Requirement
**Decision**: Require YRIMMIG <= 2007 to satisfy the "continuous residence since June 15, 2007" requirement.

**Rationale**: If someone immigrated by 2007, they would have been present continuously since mid-2007.

**Note**: The ACS cannot verify continuous residence directly; this is a proxy based on immigration year.

#### 2.6 Key Decision: Age Group Definition
**Decision**:
- Treatment group: Born 1982-1986 (ages 26-30 as of June 2012)
- Control group: Born 1977-1981 (ages 31-35 as of June 2012)

**Rationale**: DACA required applicants to be under 31 as of June 15, 2012. Those born in 1982 or later would have turned 30 or younger by June 2012. Those born 1981 would have turned 31 by June 2012 and thus be ineligible solely due to age.

**Note**: I use birth year only rather than birth year + quarter because the age cutoff is clear at the year level for this sample.

#### 2.7 Key Decision: Exclude 2012
**Decision**: Exclude all observations from 2012.

**Rationale**: DACA was implemented on June 15, 2012. Since the ACS does not include month of interview, we cannot distinguish pre- from post-implementation observations in 2012. Including 2012 would introduce measurement error in the treatment timing.

---

### 3. Outcome Variable Definition

#### 3.1 Primary Outcome: Full-Time Employment
**Definition**: Binary indicator equal to 1 if:
- EMPSTAT == 1 (Employed), AND
- UHRSWORK >= 35 (Usually works 35+ hours per week)

**Rationale**: The research question specifies "employed full-time, defined as usually working 35 hours per week or more." This standard definition aligns with BLS full-time employment thresholds.

**Code**:
```python
df_analysis['fulltime'] = ((df_analysis['EMPSTAT'] == 1) &
                           (df_analysis['UHRSWORK'] >= 35)).astype(int)
```

---

### 4. Econometric Specification

#### 4.1 Basic Difference-in-Differences
```
Y_it = β_0 + β_1*Treatment_i + β_2*Post_t + β_3*(Treatment_i × Post_t) + ε_it
```

Where:
- Y_it = Full-time employment indicator
- Treatment_i = 1 if born 1982-1986, 0 if born 1977-1981
- Post_t = 1 if year >= 2013, 0 if year <= 2011
- β_3 = DiD coefficient (causal effect of interest)

#### 4.2 Preferred Specification (Model 3)
```
Y_it = β_0 + β_1*Treatment_i + β_3*(Treatment_i × Post_t) + X_it'γ + δ_t + ε_it
```

**Covariates (X_it)**:
- Female (binary)
- Married (binary)
- Education (categorical dummies)

**Fixed Effects**:
- Year fixed effects (δ_t)

**Standard Errors**: Heteroskedasticity-robust (HC1)

**Rationale for Preferred Specification**:
1. Year FE absorb aggregate time trends
2. Demographic controls improve precision and account for compositional differences
3. State FE (Model 4) yield similar results but add complexity
4. Model balances parsimony with controlling for confounds

---

### 5. Commands Executed

#### 5.1 Main Analysis Script
```bash
python analysis.py
```

Output saved to:
- results_summary.csv
- sample_summary.csv
- event_study_results.csv
- descriptive_stats.csv

#### 5.2 Figure Generation
```bash
python create_figures.py
```

Output:
- figure1_event_study.pdf/png
- figure2_trends.pdf/png
- figure3_difference.pdf/png
- figure4_composition.pdf/png

#### 5.3 LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_24.tex
pdflatex -interaction=nonstopmode replication_report_24.tex  # Second pass for references
```

Output: replication_report_24.pdf (20 pages)

---

### 6. Key Results Summary

#### 6.1 Main Finding
**Preferred Estimate (Model 3 with Year FE and Demographics)**:
- Coefficient: 0.0486
- Standard Error: 0.0094
- 95% CI: [0.0302, 0.0669]
- p-value: < 0.001
- Sample Size: 44,725

**Interpretation**: DACA eligibility is associated with a 4.86 percentage point increase in full-time employment.

#### 6.2 Robustness Across Specifications

| Model | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| Basic DiD | 0.0592 | 0.0100 | <0.001 |
| + Demographics | 0.0487 | 0.0094 | <0.001 |
| + Year FE | 0.0486 | 0.0094 | <0.001 |
| + Year + State FE | 0.0482 | 0.0093 | <0.001 |
| Weighted | 0.0567 | 0.0110 | <0.001 |

#### 6.3 Event Study Results
Pre-treatment coefficients (2006-2010): All insignificant, supporting parallel trends
Post-treatment coefficients:
- 2013: 0.0452 (p < 0.05)
- 2014: 0.0404 (p < 0.05)
- 2015: 0.0479 (p < 0.05)
- 2016: 0.0561 (p < 0.01)

#### 6.4 Placebo Test
Fake treatment in 2009 using pre-period data only:
- Coefficient: 0.0024
- SE: 0.0111
- p-value: 0.825

Result: No significant placebo effect, supporting identification strategy.

---

### 7. Files Generated

| Filename | Description |
|----------|-------------|
| analysis.py | Main analysis script |
| create_figures.py | Figure generation script |
| results_summary.csv | Summary of regression results |
| sample_summary.csv | Sample counts by group/period |
| event_study_results.csv | Event study coefficients |
| descriptive_stats.csv | Descriptive statistics |
| figure1_event_study.pdf | Event study plot |
| figure2_trends.pdf | Employment trends by group |
| figure3_difference.pdf | Treatment-control difference |
| figure4_composition.pdf | Sample composition |
| replication_report_24.tex | LaTeX source for report |
| replication_report_24.pdf | Final PDF report (20 pages) |
| run_log_24.md | This run log |

---

### 8. Decisions Not Made / Out of Scope

The following alternative approaches were considered but not implemented:

1. **Narrower age bandwidth**: Could use 28-32 instead of 26-35 for treatment/control. Decided to follow instructions specifying 26-30 vs 31-35.

2. **Clustering standard errors**: Could cluster at state level. Used heteroskedasticity-robust SEs as sufficient for cross-sectional data.

3. **Triple-difference design**: Could add a third difference using citizens as an additional control. Not implemented as it changes the research design from what was specified.

4. **Weighting in preferred specification**: Used unweighted OLS for preferred estimate; weighted specification presented as robustness check.

5. **State-level policy controls**: State demographic and policy data was provided but not used in the main analysis. The state fixed effects specification accounts for time-invariant state differences.

---

### 9. Software and Packages

- Python 3.x
- pandas
- numpy
- statsmodels
- scipy
- matplotlib
- LaTeX (pdflatex via MiKTeX)

---

## End of Run Log
