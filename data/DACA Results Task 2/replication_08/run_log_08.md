# DACA Replication Study - Run Log

## Session Information
- **Date**: January 25, 2026
- **Replication ID**: 08
- **Analysis Platform**: Python 3.x with pandas, numpy, statsmodels
- **Report Format**: LaTeX compiled to PDF

---

## 1. Data Loading and Initial Exploration

### 1.1 Data Files Examined
- **Main data**: `data/data.csv` (~6.3 GB, ~33 million observations)
- **Data dictionary**: `data/acs_data_dict.txt`
- **State-level data**: `data/state_demo_policy.csv` (not used in analysis)

### 1.2 Data Source
- American Community Survey (ACS) from IPUMS USA
- Years: 2006-2016 (one-year samples)
- File format: CSV with 54 variables

### 1.3 Key Variables Identified
From data dictionary:
- `YEAR`: Survey year (2006-2016)
- `PERWT`: Person weight for sampling
- `HISPAN`: Hispanic origin (1=Mexican)
- `BPL`: Birthplace (200=Mexico)
- `CITIZEN`: Citizenship status (3=Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Year of birth
- `UHRSWORK`: Usual hours worked per week
- `EMPSTAT`: Employment status (1=Employed)
- `EDUC`: Educational attainment
- `SEX`: Sex (1=Male, 2=Female)
- `MARST`: Marital status
- `STATEFIP`: State FIPS code

---

## 2. Sample Construction Decisions

### 2.1 Eligibility Criteria Applied
Following DACA requirements and study design:

1. **Hispanic-Mexican ethnicity**: `HISPAN == 1`
2. **Born in Mexico**: `BPL == 200`
3. **Non-citizen**: `CITIZEN == 3`
4. **Arrived before age 16**: `YRIMMIG - BIRTHYR < 16`
5. **Continuous presence since 2007**: `YRIMMIG <= 2007`
6. **Age eligibility groups**:
   - Treatment: `BIRTHYR` 1982-1986 (ages 26-30 on June 15, 2012)
   - Control: `BIRTHYR` 1977-1981 (ages 31-35 on June 15, 2012)

### 2.2 Time Period Decisions
- **Pre-period**: 2006-2011 (before DACA implementation)
- **Post-period**: 2013-2016 (after DACA implementation)
- **Excluded**: 2012 (mid-year implementation makes pre/post distinction impossible)

### 2.3 Sample Size Progression
| Filter Step | Observations |
|-------------|--------------|
| Initial ACS data | ~33 million |
| Hispanic-Mexican (HISPAN=1) | ~3.4 million |
| Born in Mexico (BPL=200) | ~1.6 million |
| Non-citizen (CITIZEN=3) | ~0.9 million |
| Arrived before age 16 | ~0.4 million |
| In US since 2007 | ~0.3 million |
| Birth year 1977-1986, excl 2012 | **44,725** |

### 2.4 Final Sample Composition
- Total observations: 44,725
- Treatment group (26-30): 26,591 (59.5%)
- Control group (31-35): 18,134 (40.5%)
- Pre-period: 29,326
- Post-period: 15,399

---

## 3. Variable Definitions

### 3.1 Outcome Variable
- **Full-time employment**: `UHRSWORK >= 35`
- Binary indicator: 1 if usually works 35+ hours/week, 0 otherwise

### 3.2 Treatment Variables
- **Treat**: 1 if BIRTHYR 1982-1986, 0 if BIRTHYR 1977-1981
- **Post**: 1 if YEAR >= 2013, 0 if YEAR <= 2011
- **Treat_Post**: Treat × Post (DID interaction term)

### 3.3 Control Variables
- **Female**: SEX == 2
- **Married**: MARST in [1, 2]
- **Age**: AGE variable
- **Education** (reference: less than HS):
  - HS: EDUC == 6
  - Some College: EDUC 7-9
  - College+: EDUC >= 10

---

## 4. Econometric Approach

### 4.1 Research Design
- **Method**: Difference-in-Differences (DID)
- **Identification**: Age-based eligibility cutoff at 31 on June 15, 2012

### 4.2 Model Specifications Estimated
1. Basic DID (no controls)
2. DID with demographic controls (female, married, age)
3. DID with demographics + education
4. **PREFERRED**: DID with year fixed effects + controls
5. DID with year + state fixed effects

### 4.3 Estimation Details
- Weighted Least Squares using PERWT
- Heteroskedasticity-robust standard errors (HC1)
- Linear probability model for binary outcome

---

## 5. Results Summary

### 5.1 Simple DID Calculation (Weighted)
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment | 62.53% | 65.80% | +3.27 pp |
| Control | 67.05% | 64.12% | -2.93 pp |
| **DID Estimate** | | | **+6.20 pp** |

### 5.2 Regression Results
| Model | Coefficient | SE | p-value | 95% CI |
|-------|-------------|-----|---------|--------|
| Basic DID | 0.0620 | 0.0116 | <0.001 | [0.039, 0.085] |
| + Demographics | 0.0504 | 0.0106 | <0.001 | [0.030, 0.071] |
| + Education | 0.0490 | 0.0106 | <0.001 | [0.028, 0.070] |
| **+ Year FE** | **0.0486** | **0.0106** | **<0.001** | **[0.028, 0.069]** |
| + State FE | 0.0479 | 0.0105 | <0.001 | [0.027, 0.069] |

### 5.3 Preferred Estimate
- **Effect size**: 4.86 percentage points
- **Standard error**: 1.06 percentage points
- **95% CI**: [2.79, 6.94]
- **p-value**: < 0.001

### 5.4 Robustness Checks
| Specification | Effect (pp) | p-value |
|--------------|-------------|---------|
| Any employment | 4.61 | <0.001 |
| Males only | 5.05 | <0.001 |
| Females only | 3.53 | 0.047 |
| Narrower age bands | 4.33 | 0.002 |
| Placebo (2009) | 0.60 | 0.627 |

---

## 6. Key Decisions and Justifications

### 6.1 Sample Selection
- **Decision**: Used CITIZEN == 3 (not a citizen) as proxy for undocumented status
- **Justification**: Instructions state to assume non-citizens without immigration papers are undocumented

### 6.2 Age Groups
- **Decision**: 5-year bands (26-30 vs 31-35) on either side of cutoff
- **Justification**: Balances sample size with comparability; provides sufficient observations while maintaining similarity

### 6.3 Immigration Timing
- **Decision**: Required YRIMMIG <= 2007 for continuous presence
- **Justification**: DACA requires continuous presence since June 15, 2007

### 6.4 Full-Time Definition
- **Decision**: UHRSWORK >= 35 hours/week
- **Justification**: Standard definition from research instructions and labor economics literature

### 6.5 Excluded Year
- **Decision**: Excluded 2012 entirely
- **Justification**: DACA implemented June 15, 2012; ACS doesn't provide interview month, so can't distinguish pre/post

### 6.6 Model Selection
- **Decision**: Year FE + controls as preferred model
- **Justification**: Controls for common time shocks while preserving degrees of freedom; state FE adds little

### 6.7 Weighting
- **Decision**: Used person weights (PERWT) throughout
- **Justification**: ACS is complex survey design; weights needed for population-representative estimates

---

## 7. Files Produced

### 7.1 Analysis Scripts
- `analysis_optimized.py`: Main analysis script (optimized for large data)
- `create_figures.py`: Figure generation script

### 7.2 Output Data Files
- `analysis_results.json`: Main results in JSON format
- `descriptive_stats.csv`: Descriptive statistics by group/period
- `figure1_data.csv`: Data for trends figure
- `figure2_event_study.csv`: Event study coefficients
- `model_summaries.txt`: Full regression output

### 7.3 Figures
- `figure1_trends.png/pdf`: Full-time employment trends
- `figure2_event_study.png/pdf`: Event study coefficients
- `figure3_did_visual.png/pdf`: DID visualization
- `figure4_by_gender.png/pdf`: Effects by gender

### 7.4 Report
- `replication_report_08.tex`: LaTeX source (23 pages)
- `replication_report_08.pdf`: Final compiled report

---

## 8. Interpretation

### 8.1 Main Finding
DACA eligibility increased full-time employment by approximately 4.86 percentage points among Hispanic-Mexican, Mexican-born non-citizens who met DACA's age and immigration timing requirements.

### 8.2 Economic Significance
- Baseline full-time rate (pre-treatment): 62.53%
- Relative increase: ~7.8%
- Effect is economically meaningful

### 8.3 Statistical Significance
- Highly significant (p < 0.001)
- 95% CI excludes zero: [2.79, 6.94]
- Robust across all specifications

### 8.4 Validity Assessment
- Event study shows no significant pre-trends
- Placebo test fails to reject null (as expected)
- Results stable across specifications
- Parallel trends assumption appears satisfied

---

## 9. Session Commands Log

```bash
# Data exploration
head -5 data/data.csv
ls -la data/

# Analysis execution
python analysis_optimized.py

# Figure generation
python create_figures.py

# Report compilation
pdflatex -interaction=nonstopmode replication_report_08.tex
pdflatex -interaction=nonstopmode replication_report_08.tex  # Second pass for TOC
```

---

## 10. Notes

1. Large dataset (~6 GB) required chunked processing for memory efficiency
2. Data loaded in 500,000-row chunks with filtering applied during load
3. All statistics weighted using person weights (PERWT)
4. Standard errors are heteroskedasticity-robust (HC1)
5. Linear probability model used for ease of interpretation

---

*Log completed: January 25, 2026*
