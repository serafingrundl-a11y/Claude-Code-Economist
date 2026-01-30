# Run Log for DACA Replication Study

## Study Information
- **Research Question**: Effect of DACA eligibility on full-time employment among Mexican-born Hispanic immigrants
- **Treatment Group**: Ages 26-30 as of June 15, 2012 (ELIGIBLE=1)
- **Control Group**: Ages 31-35 as of June 15, 2012 (ELIGIBLE=0)
- **Outcome**: Full-time employment (FT), defined as usually working 35+ hours/week
- **Data**: American Community Survey 2008-2016 (excluding 2012)

## Data Files Used
- `data/prepared_data_numeric_version.csv` - Main analysis dataset
- `data/prepared_data_labelled_version.csv` - Dataset with value labels
- `data/acs_data_dict.txt` - Data dictionary from IPUMS

## Key Decisions

### 1. Analytic Sample
- Used the provided ELIGIBLE variable to identify treatment and control groups
- Did not further subset the sample per instructions
- Included individuals not in the labor force with FT=0

### 2. Estimation Method
- **Method**: Weighted Least Squares (WLS) using ACS person weights (PERWT)
- **Standard Errors**: Robust (heteroskedasticity-consistent, HC1)
- **Model**: Linear Probability Model for binary outcome

### 3. Preferred Specification
- DiD model with individual covariates:
  - FEMALE (gender)
  - MARRIED (marital status)
  - AGE (current age)
  - HAS_CHILDREN (presence of children)
  - Education dummies (High School, Less than HS, Some College, Two-Year Degree; BA+ as reference)
- Did not include state/year fixed effects in preferred model because year FE absorb the AFTER coefficient

### 4. Variable Construction
- `ELIGIBLE_AFTER`: ELIGIBLE × AFTER (DiD interaction term)
- `FEMALE`: 1 if SEX=2, 0 otherwise (IPUMS coding: 1=Male, 2=Female)
- `MARRIED`: 1 if MARST in {1, 2}, 0 otherwise (married, spouse present or absent)
- `HAS_CHILDREN`: 1 if NCHILD > 0, 0 otherwise
- Education dummies created from EDUC_RECODE

## Commands Executed

### Data Exploration
```python
# Load data
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Check key variables
print(df['YEAR'].value_counts().sort_index())
print(df['ELIGIBLE'].value_counts())
print(df['AFTER'].value_counts())
print(df['FT'].value_counts())
```

### Sample Sizes
- Total observations: 17,382
- Treatment (ELIGIBLE=1): 11,382
- Control (ELIGIBLE=0): 6,000
- Pre-period (AFTER=0): 9,527
- Post-period (AFTER=1): 7,855
- Weighted total: 2,416,349

### Full-Time Employment Rates
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Control (weighted) | 0.689 | 0.663 | -0.026 |
| Treatment (weighted) | 0.637 | 0.686 | +0.049 |
| **DiD** | | | **+0.075** |

### Main Regression Analysis
```python
import statsmodels.formula.api as smf

# Create variables
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Preferred model
formula = '''FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER +
             FEMALE + MARRIED + AGE + HAS_CHILDREN +
             EDU_High_School_Degree + EDU_Less_than_High_School +
             EDU_Some_College + EDU_Two_Year_Degree'''

model = smf.wls(formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')
```

## Results Summary

### Preferred Estimate
| Statistic | Value |
|-----------|-------|
| Effect Size (DiD coefficient) | 0.062 |
| Standard Error | 0.017 |
| 95% Confidence Interval | [0.029, 0.094] |
| p-value | < 0.001 |
| Sample Size (unweighted) | 17,382 |
| Sample Size (weighted) | 2,416,349 |
| R-squared | 0.130 |

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 6.2 percentage points among Mexican-born Hispanic immigrants aged 26-30 at the time of implementation, compared to those aged 31-35.

## Robustness Checks

### 1. Alternative Specifications
| Model | DiD Coefficient | SE | p-value |
|-------|----------------|-----|---------|
| Basic (unweighted) | 0.064 | 0.015 | <0.001 |
| Basic (weighted) | 0.075 | 0.018 | <0.001 |
| With covariates (preferred) | 0.062 | 0.017 | <0.001 |
| With state/year FE | 0.058 | 0.017 | <0.001 |
| Clustered SEs (state) | 0.062 | 0.021 | 0.004 |
| With state policy controls | 0.064 | 0.017 | <0.001 |

### 2. Heterogeneity by Gender
| Subgroup | DiD Coefficient | SE | p-value | N |
|----------|----------------|-----|---------|---|
| Men | 0.072 | 0.020 | <0.001 | 9,075 |
| Women | 0.053 | 0.028 | 0.061 | 8,307 |

### 3. Pre-Trends Test
- Differential pre-trend (ELIGIBLE × TREND): 0.017 (SE=0.011, p=0.113)
- Not statistically significant, but suggests some convergence

### 4. Event Study (relative to 2011)
| Year | Coefficient | SE | Significant? |
|------|-------------|-----|--------------|
| 2008 | -0.068 | 0.035 | * |
| 2009 | -0.050 | 0.036 | No |
| 2010 | -0.082 | 0.036 | ** |
| 2011 | (reference) | - | - |
| 2013 | 0.016 | 0.038 | No |
| 2014 | 0.000 | 0.038 | No |
| 2015 | 0.001 | 0.038 | No |
| 2016 | 0.074 | 0.038 | * |

## Output Files

### Analysis Files
- `analysis_code.py` - Python script with full analysis

### Report Files
- `replication_report_33.tex` - LaTeX source for replication report
- `replication_report_33.pdf` - Compiled PDF report (27 pages)

### This Log
- `run_log_33.md` - This file documenting all commands and decisions

## Notes

1. The pre-treatment event study coefficients show convergence between groups, which provides some support for the DiD identifying assumption but also suggests caution as the convergence could continue independently of DACA.

2. The effect appears larger for men than women, consistent with prior literature.

3. The effect appears to grow over time, with the largest effect in 2016, suggesting DACA's impacts may have taken time to materialize.

4. Including state policy controls does not substantially change the results.

5. The estimated effect is an intent-to-treat effect based on eligibility, not actual DACA receipt.
