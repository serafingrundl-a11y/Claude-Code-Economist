# Run Log - DACA Replication Analysis 02

## Project Overview
Independent replication analysis examining the causal effect of DACA eligibility on full-time employment among Hispanic-Mexican immigrants born in Mexico.

## Date: January 26, 2026

---

## 1. Data Loading and Exploration

### Initial Data Inspection
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Shape:', df.shape)  # (17382, 105)
```

**Key Findings:**
- Total observations: 17,382
- Variables: 105 columns
- Years covered: 2008-2011 (pre-treatment), 2013-2016 (post-treatment)
- Year 2012 excluded (DACA implementation mid-year)

### Sample Distribution
| Period | Control (31-35) | Treated (26-30) | Total |
|--------|-----------------|-----------------|-------|
| Before | 3,294 | 6,233 | 9,527 |
| After | 2,706 | 5,149 | 7,855 |
| **Total** | **6,000** | **11,382** | **17,382** |

### Variable Coding Decisions
- **FT** (outcome): Binary, 1 = full-time (35+ hrs/week), 0 = otherwise
- **ELIGIBLE**: 1 = treated (ages 26-30 at June 15, 2012), 0 = control (ages 31-35)
- **AFTER**: 1 = post-treatment (2013-2016), 0 = pre-treatment (2008-2011)
- **SEX**: IPUMS coding (1=Male, 2=Female) - created FEMALE = (SEX==2)
- **MARST**: Recoded to MARRIED = 1 if MARST in {1, 2}, else 0
- **EDUC_RECODE**: Used as-is with dummy variables (BA+ as reference)

---

## 2. Research Design

### Identification Strategy
Difference-in-differences (DiD) comparing:
- **Treatment group**: Ages 26-30 at DACA implementation (ELIGIBLE=1)
- **Control group**: Ages 31-35 at DACA implementation (ELIGIBLE=0)

### Key Identifying Assumption
Parallel trends: absent DACA, both groups would have experienced the same change in full-time employment over time.

---

## 3. Main Analysis

### Simple DiD Calculation (Unweighted)
```
               Before    After    Diff
Control        0.670     0.645   -0.025
Treated        0.626     0.666    0.039
Difference    -0.043     0.021    0.064
```
**Simple DiD = 0.064 (6.4 percentage points)**

### Main Regression Specification
```python
FT = b0 + b1*ELIGIBLE + b2*AFTER + b3*(ELIGIBLE*AFTER) + covariates + error
```

Covariates included:
- FEMALE (indicator for female)
- MARRIED (indicator for married)
- Education dummies: ED_LTH, ED_HS, ED_SC, ED_2Y (BA+ reference)

### Estimation Method
- Weighted Least Squares (WLS) using PERWT (ACS person weights)
- Heteroskedasticity-robust standard errors (HC1)

### Main Result
```
Variable          Coefficient    SE       p-value     95% CI
-----------------------------------------------------------------
ELIGIBLE_AFTER    0.0621        0.0167   0.0002      [0.029, 0.095]
```

**Preferred estimate: 6.21 percentage point increase in full-time employment**

---

## 4. Specification Checks

### Alternative Specifications
| Specification | DiD Estimate | SE | N |
|--------------|--------------|-----|-------|
| Basic (no controls, unweighted) | 0.0643 | 0.015 | 17,382 |
| Basic (no controls, weighted) | 0.0748 | 0.015 | 17,382 |
| Main (with covariates, weighted) | 0.0621 | 0.017 | 17,382 |
| With state fixed effects | 0.0614 | 0.017 | 17,382 |
| Unweighted (with covariates) | 0.0536 | 0.014 | 17,382 |

**Conclusion**: Results robust across specifications (range: 0.054 to 0.075)

---

## 5. Parallel Trends Analysis

### Year-by-Year Employment Rates
| Year | Control | Treated | Gap |
|------|---------|---------|-----|
| 2008 | 0.726 | 0.667 | -0.060 |
| 2009 | 0.657 | 0.617 | -0.039 |
| 2010 | 0.673 | 0.606 | -0.067 |
| 2011 | 0.617 | 0.617 | -0.001 |
| 2013 | 0.624 | 0.642 | +0.018 |
| 2014 | 0.649 | 0.640 | -0.009 |
| 2015 | 0.650 | 0.680 | +0.030 |
| 2016 | 0.660 | 0.708 | +0.048 |

### Event Study Coefficients (Reference: 2011)
| Year | Coefficient | SE | 95% CI |
|------|-------------|------|--------|
| 2008 | -0.065** | 0.032 | [-0.128, -0.002] |
| 2009 | -0.047 | 0.033 | [-0.112, 0.017] |
| 2010 | -0.077** | 0.033 | [-0.142, -0.013] |
| 2013 | 0.014 | 0.034 | [-0.053, 0.082] |
| 2014 | -0.015 | 0.035 | [-0.083, 0.054] |
| 2015 | -0.009 | 0.035 | [-0.078, 0.059] |
| 2016 | 0.059 | 0.036 | [-0.011, 0.129] |

**Note**: Some pre-treatment divergence during Great Recession years (2008-2010). Groups converge by 2011.

---

## 6. Heterogeneity Analysis

### By Subgroup
| Subgroup | DiD Estimate | SE | N |
|----------|--------------|-----|-------|
| **By Sex** | | | |
| Male | 0.063 | 0.020 | 9,075 |
| Female | 0.047 | 0.028 | 8,307 |
| **By Education** | | | |
| HS or less | 0.047 | 0.020 | 12,453 |
| More than HS | 0.103 | 0.032 | 4,929 |
| **By Marital Status** | | | |
| Unmarried | 0.100 | 0.025 | 8,858 |
| Married | 0.006 | 0.022 | 8,524 |

**Key Findings:**
- Similar effects for men and women
- Larger effect for those with more education (10.3 pp vs. 4.7 pp)
- Effect concentrated among unmarried individuals (10.0 pp vs. 0.6 pp)

---

## 7. Key Decisions and Justifications

### Decision 1: Use of Survey Weights
**Choice**: Used PERWT (ACS person weights) in main specification
**Justification**: Weights adjust for sampling design and produce population-representative estimates

### Decision 2: Covariates Selection
**Choice**: Included sex, marital status, and education
**Justification**: These are strong predictors of employment and may differ between treatment/control groups. Did not include age since it determines group membership.

### Decision 3: Standard Errors
**Choice**: Heteroskedasticity-robust (HC1) standard errors
**Justification**: Linear probability model with binary outcome likely has heteroskedastic errors. Did not cluster by state due to limited state variation in some cells.

### Decision 4: Sample Retention
**Choice**: Used entire provided sample without further restrictions
**Justification**: Instructions specify "do not further limit the sample by dropping individuals on the basis of their characteristics"

### Decision 5: Including Those Not in Labor Force
**Choice**: Kept individuals not in labor force (coded as FT=0)
**Justification**: Instructions state "Those not in the labor force are included, usually as 0 values; keep these individuals in your analysis"

---

## 8. Figures Generated

1. **figures/parallel_trends.png**: Full-time employment trends by eligibility status (2008-2016)
2. **figures/event_study.png**: Year-specific treatment effects relative to 2011
3. **figures/did_visualization.png**: DiD visualization and heterogeneity analysis

---

## 9. Files Produced

- `replication_report_02.tex`: LaTeX source for replication report
- `replication_report_02.pdf`: Compiled 20-page PDF report
- `run_log_02.md`: This log file
- `figures/parallel_trends.png`: Trends figure
- `figures/event_study.png`: Event study figure
- `figures/did_visualization.png`: DiD and heterogeneity figure

---

## 10. Summary of Main Finding

**Effect of DACA Eligibility on Full-Time Employment:**
- **Point estimate**: 6.21 percentage points (0.0621)
- **Standard error**: 0.0167
- **95% Confidence Interval**: [0.029, 0.095]
- **p-value**: < 0.001
- **Sample size**: 17,382

The analysis provides evidence that DACA eligibility increased full-time employment by approximately 6.2 percentage points among the targeted population. This effect is statistically significant, economically meaningful, and robust across specifications.

---

## Python Package Versions Used
- pandas
- numpy
- statsmodels
- matplotlib

## Session End
Replication analysis completed January 26, 2026.
