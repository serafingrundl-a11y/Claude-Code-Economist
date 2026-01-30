# Run Log - DACA Replication Study (Replication 51)

## Overview
This log documents all commands, decisions, and key steps taken during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican immigrants.

---

## 1. Data Exploration

### Data Files Examined
- `data/data.csv` - Main ACS data file (33,851,424 observations)
- `data/acs_data_dict.txt` - Variable codebook
- `data/state_demo_policy.csv` - State-level supplementary data (not used in main analysis)

### Key Variables Identified
| Variable | Description | Role |
|----------|-------------|------|
| YEAR | Survey year | Time indicator |
| HISPAN | Hispanic origin (1=Mexican) | Sample selection |
| BPL | Birthplace (200=Mexico) | Sample selection |
| CITIZEN | Citizenship status (3=Non-citizen) | Eligibility |
| YRIMMIG | Year of immigration | Eligibility |
| BIRTHYR | Birth year | Age calculation |
| BIRTHQTR | Birth quarter | Age precision |
| UHRSWORK | Usual hours worked per week | Outcome |
| PERWT | Person weight | Survey weights |
| STATEFIP | State FIPS code | Fixed effects |
| SEX, MARST, EDUCD | Demographics | Covariates |

---

## 2. Sample Selection Decisions

### DACA Eligibility Criteria Applied
1. **Hispanic-Mexican ethnicity**: HISPAN == 1
2. **Born in Mexico**: BPL == 200
3. **Non-citizen**: CITIZEN == 3 (assuming undocumented status per instructions)
4. **Continuous US presence since 2007**: YRIMMIG <= 2007
5. **Arrived before age 16**: (YRIMMIG - BIRTHYR) < 16

### Treatment and Control Group Definitions
- **Treatment Group**: Ages 26-30 as of June 15, 2012
- **Control Group**: Ages 31-35 as of June 15, 2012 (too old for DACA)

Age calculation accounts for birth quarter:
- Q1/Q2 births: Had birthday by June 15
- Q3/Q4 births: Had not yet had 2012 birthday

### Time Period Decisions
- **Pre-treatment**: 2006-2011
- **Post-treatment**: 2013-2016
- **Excluded**: 2012 (ambiguous timing - DACA announced June 15, 2012)

---

## 3. Sample Size Progression

| Filter Applied | N Remaining |
|----------------|-------------|
| Initial data | 33,851,424 |
| Exclude 2012 | 30,738,394 |
| Hispanic-Mexican (HISPAN=1) | 2,663,503 |
| Born in Mexico (BPL=200) | 898,879 |
| Non-citizen (CITIZEN=3) | 636,722 |
| YRIMMIG <= 2007 | 595,366 |
| Arrived before age 16 | 177,294 |
| Age 26-35 in June 2012 | **43,238** |

---

## 4. Variable Construction

### Outcome Variable
- `fulltime`: Binary indicator = 1 if UHRSWORK >= 35

### Treatment Indicators
- `treated`: 1 if age 26-30 on June 15, 2012
- `post`: 1 if YEAR >= 2013
- `treated_post`: Interaction term (DiD coefficient)

### Covariates
- `female`: SEX == 2
- `married`: MARST in [1, 2]
- `educ_hs`: EDUCD 62-64 (high school)
- `educ_somecoll`: EDUCD 65-100 (some college)
- `educ_ba`: EDUCD >= 101 (bachelor's or higher)
- `age_current`, `age_current_sq`: Current age in survey year

---

## 5. Analysis Commands

### Model Specifications

**Model 1: Basic DiD**
```
fulltime ~ treated + post + treated_post
```

**Model 2: DiD with Covariates**
```
fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecoll + educ_ba + age_current + age_current_sq
```

**Model 3: DiD with Year Fixed Effects**
```
fulltime ~ treated + treated_post + [covariates] + [year_dummies]
```

**Model 4: Weighted DiD**
Same as Model 2, weighted by PERWT

**Model 5: Preferred Specification (Weighted DiD with State and Year FE)**
```
fulltime ~ treated + post + treated_post + [covariates] + [year_FE] + [state_FE]
```
Weighted by PERWT, robust standard errors (HC1)

---

## 6. Key Results

### Main DiD Estimates by Model

| Model | Coefficient | Std. Error | p-value |
|-------|-------------|------------|---------|
| Basic DiD | 0.0516 | 0.010 | <0.001 |
| + Covariates | 0.0682 | 0.012 | <0.001 |
| + Year FE | 0.0203 | 0.013 | 0.121 |
| Weighted | 0.0645 | 0.015 | <0.001 |
| **Preferred** | **0.0191** | **0.015** | **0.214** |

### Robustness Checks

| Specification | Coefficient | Std. Error | p-value |
|--------------|-------------|------------|---------|
| Narrow bandwidth (27-29 vs 32-34) | 0.0755 | 0.021 | <0.001 |
| Men only | 0.0489 | 0.017 | 0.005 |
| Women only | 0.0789 | 0.024 | 0.001 |
| Placebo (2008) | -0.0307 | 0.018 | 0.081 |

---

## 7. Output Files Generated

- `figure1_trends.png` - Full-time employment trends by treatment status
- `figure2_eventstudy.png` - Event study coefficient plot
- `figure3_models.png` - Comparison of estimates across models
- `summary_statistics.csv` - Descriptive statistics by group
- `regression_results.csv` - Main regression coefficients
- `event_study_results.csv` - Year-by-year treatment effects
- `replication_report_51.tex` - LaTeX report
- `replication_report_51.pdf` - Final PDF report

---

## 8. Key Analytical Decisions

1. **Excluding 2012**: The DACA announcement occurred mid-year (June 15, 2012), making it impossible to distinguish pre- and post-treatment observations in 2012 ACS data.

2. **Age calculation using birth quarter**: To accurately determine age as of June 15, 2012, I accounted for whether the person's birthday had occurred by that date.

3. **Assuming non-citizens are undocumented**: Per instructions, those with CITIZEN=3 (not a citizen) who arrived before age 16 are treated as potentially DACA-eligible.

4. **Preferred specification**: The model with state and year fixed effects plus survey weights provides the most credible estimate by controlling for geographic and temporal confounders.

5. **Event study design**: Reference year is 2011 (last pre-treatment year) to assess pre-trends and dynamic treatment effects.

---

## 9. Interpretation Notes

The preferred estimate suggests a positive but statistically insignificant effect of DACA eligibility on full-time employment (1.9 percentage points, p=0.214). However:
- Simpler models without state FE show larger, significant effects
- Robustness checks suggest heterogeneity by gender (larger effects for women)
- Event study shows no clear pre-trends, supporting parallel trends assumption
- Placebo test (2008) shows no significant effect, as expected

---

## 10. Software and Environment

- Python 3.x
- pandas, numpy, statsmodels, scipy, matplotlib
- Analysis run on Windows platform

---

*Log completed: Replication 51*
