# DACA Replication Study - Run Log

## Study Information
- **Replication ID**: 33
- **Date**: January 26, 2026
- **Research Question**: What is the causal effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals?

---

## 1. Data Exploration

### 1.1 Directory Contents
```
data/
├── acs_data_dict.txt      # IPUMS ACS data dictionary
├── data.csv               # Main ACS data file (6.26 GB, 33,851,424 rows)
├── state_demo_policy.csv  # Optional state-level data (not used)
└── State Level Data Documentation.docx
```

### 1.2 Data Source
- American Community Survey (ACS) one-year samples from IPUMS USA
- Years: 2006-2016 (11 years)
- Total observations: 33,851,424

### 1.3 Key Variables Identified
| Variable | Description | Values Used |
|----------|-------------|-------------|
| YEAR | Survey year | 2006-2016 |
| HISPAN | Hispanic origin | 1 = Mexican |
| BPL | Birthplace | 200 = Mexico |
| CITIZEN | Citizenship status | 3 = Not a citizen |
| BIRTHYR | Birth year | For age calculation |
| BIRTHQTR | Birth quarter | For precise age at June 15, 2012 |
| YRIMMIG | Year of immigration | For eligibility criteria |
| UHRSWORK | Usual hours worked per week | >=35 for full-time |
| PERWT | Person weight | For population estimates |
| SEX | Sex | 1=Male, 2=Female |
| EDUC | Educational attainment | For covariates |
| MARST | Marital status | For covariates |
| STATEFIP | State FIPS code | For state fixed effects |

---

## 2. Sample Construction Decisions

### 2.1 Base Population Filter
**Decision**: Restrict to Hispanic-Mexican, Mexican-born, non-citizens
- HISPAN = 1 (Mexican)
- BPL = 200 (Mexico)
- CITIZEN = 3 (Not a citizen)

**Rationale**: Per instructions, focus on "ethnically Hispanic-Mexican Mexican-born people." Non-citizen status used as proxy for undocumented status, as ACS does not directly identify documentation status.

**Result**: 701,347 observations

### 2.2 Age-Based Treatment/Control Groups
**Decision**: Define groups based on age as of June 15, 2012
- Treatment: Ages 26-30 (born approximately June 1982 - June 1986)
- Control: Ages 31-35 (born approximately June 1977 - June 1981)

**Age Calculation Method**:
```python
if birth_quarter <= 2:  # Born Jan-June
    age = 2012 - birth_year
else:  # Born July-December
    age = 2012 - birth_year - 1
```

**Rationale**: DACA required being under 31 as of June 15, 2012. Control group consists of individuals who would have been eligible except for age.

**Result**: 181,229 observations

### 2.3 DACA Eligibility Criteria
**Decision**: Apply additional eligibility requirements
1. Arrived in US before age 16: YRIMMIG <= BIRTHYR + 15
2. In US by June 2007 (continuous presence): YRIMMIG <= 2007

**Rationale**: Both treatment and control groups should satisfy all DACA criteria except age for valid comparison.

**Result**: 47,418 observations

### 2.4 Time Period Definition
**Decision**:
- Pre-treatment: 2006-2011
- Post-treatment: 2013-2016
- Exclude 2012 (ambiguous year)

**Rationale**: DACA implemented June 15, 2012. Since ACS does not identify survey month, 2012 observations cannot be classified as pre/post treatment.

**Result**: 43,238 observations (final analysis sample)

---

## 3. Variable Construction

### 3.1 Outcome Variable
**Full-time employment**: Binary indicator = 1 if UHRSWORK >= 35

**Rationale**: Standard BLS definition of full-time work

### 3.2 Treatment Variables
- `treat`: 1 if age 26-30 as of June 15, 2012
- `post`: 1 if YEAR >= 2013
- `treat_post`: treat * post (DiD interaction term)

### 3.3 Control Variables
- `male`: SEX == 1
- `married`: MARST <= 2
- `educ_hs`: EDUC >= 6 (high school or more)
- `educ_college`: EDUC >= 10 (4+ years college)

---

## 4. Analysis Commands

### 4.1 Main Analysis Script
```bash
python daca_analysis.py
```

**Processing time**: ~5 minutes for data loading and filtering

### 4.2 Figure Generation
```bash
python create_figures.py
```

### 4.3 Report Compilation
```bash
pdflatex replication_report_33.tex
pdflatex replication_report_33.tex  # Second pass for cross-references
pdflatex replication_report_33.tex  # Third pass
```

---

## 5. Model Specifications

### Model 1: Basic DiD (Unweighted)
```
fulltime ~ treat + post + treat_post
```

### Model 2: Basic DiD (Weighted)
Same as Model 1, with PERWT weights

### Model 3: Year Fixed Effects
```
fulltime ~ treat + C(YEAR) + treat_post
```

### Model 4: With Covariates
```
fulltime ~ treat + C(YEAR) + treat_post + male + married + educ_hs + educ_college
```

### Model 5: Year + State Fixed Effects
```
fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + male + married + educ_hs + educ_college
```

### Model 6: Preferred Specification (Robust SE)
Same as Model 5 with HC1 robust standard errors

### Event Study Specification
```
fulltime ~ treat + C(YEAR) + sum(treat_y[k] for k != 2011)
```
Reference year: 2011

---

## 6. Key Results Summary

### 6.1 Preferred Estimate (Model 6)
| Metric | Value |
|--------|-------|
| DiD Estimate | 0.0441 |
| Robust SE | 0.0107 |
| 95% CI | [0.0232, 0.0650] |
| p-value | < 0.001 |
| N | 43,238 |

**Interpretation**: DACA eligibility increased full-time employment by 4.4 percentage points, representing a 7.2% increase from the pre-treatment baseline rate of 61.5%.

### 6.2 Robustness Checks
| Specification | Estimate | SE | p-value |
|--------------|----------|-----|---------|
| Narrow age bands (27-29 vs 32-34) | 0.0358 | 0.0137 | 0.009 |
| Including 2012 | 0.0440 | 0.0100 | <0.001 |
| Males only | 0.0345 | 0.0124 | 0.006 |
| Females only | 0.0492 | 0.0181 | 0.007 |
| Placebo (2006-08 vs 2009-11) | -0.0020 | 0.0125 | 0.873 |

---

## 7. Output Files Generated

| File | Description |
|------|-------------|
| `daca_analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `main_results.csv` | Regression results table |
| `yearly_results.csv` | Year-by-year employment rates |
| `event_study_results.csv` | Event study coefficients |
| `descriptive_stats.csv` | Summary statistics by group/period |
| `model_summaries.txt` | Full regression output |
| `figure1_parallel_trends.pdf/png` | Parallel trends plot |
| `figure2_difference_trends.pdf/png` | Difference trends plot |
| `figure3_event_study.pdf/png` | Event study plot |
| `figure4_model_comparison.pdf/png` | Model comparison plot |
| `figure5_sample_flow.pdf/png` | Sample flow diagram |
| `replication_report_33.tex` | LaTeX source |
| `replication_report_33.pdf` | Final report (21 pages) |
| `run_log_33.md` | This log file |

---

## 8. Key Methodological Decisions

### 8.1 Why Non-citizen as Proxy for Undocumented?
The ACS does not directly identify undocumented status. Following standard practice in the literature, non-citizen status is used as a proxy. This may include some documented non-citizens, potentially attenuating estimates.

### 8.2 Why Exclude 2012?
DACA was implemented on June 15, 2012. The ACS does not record the month of survey response, so 2012 observations cannot be classified as definitively pre- or post-treatment. Excluding 2012 provides cleaner identification.

### 8.3 Why Ages 26-30 vs 31-35?
- Treatment (26-30): Young enough to be DACA-eligible (under 31 as of June 15, 2012)
- Control (31-35): Just above the age cutoff, otherwise would have been eligible
- 5-year bands provide sufficient sample size while maintaining comparability

### 8.4 Why Use Survey Weights?
PERWT weights allow for population-representative estimates. Unweighted results are also provided for comparison and are qualitatively similar.

### 8.5 Why Robust Standard Errors?
HC1 robust standard errors account for potential heteroskedasticity. Given the repeated cross-section design (not panel), clustering at the individual level is not applicable.

---

## 9. Limitations Noted

1. Non-citizen proxy for undocumented status may include documented non-citizens
2. Age-based groups differ in age, which may affect employment trajectories
3. Continuous presence criterion relies on self-reported immigration year
4. Potential selection effects if DACA affects ACS response rates
5. Some pre-treatment coefficient variation in event study (2007 marginally significant)

---

## 10. Session Information

- Python version: 3.x
- Key packages: pandas, numpy, statsmodels, matplotlib, scipy
- LaTeX distribution: MiKTeX 25.12
- OS: Windows

---

*Log completed: January 26, 2026*
