# DACA Replication Study - Run Log

## Study Information
- **Research Question**: Effect of DACA eligibility on full-time employment among Mexican-born Hispanic immigrants
- **Method**: Difference-in-Differences (DiD)
- **Data**: American Community Survey (ACS) 2008-2011, 2013-2016 (2012 excluded)
- **Sample Size**: 17,382 observations

---

## Key Decisions

### 1. Identification Strategy
- **Decision**: Use Difference-in-Differences comparing treated group (ages 26-30 in June 2012) to control group (ages 31-35 in June 2012)
- **Rationale**: The age-based DACA eligibility cutoff (must not have turned 31 by June 15, 2012) creates a natural experiment. Those just above the cutoff serve as a plausible comparison group for those just below.

### 2. Treatment and Control Groups
- **Treated (ELIGIBLE=1)**: Individuals ages 26-30 in June 2012 (N=11,382)
- **Control (ELIGIBLE=0)**: Individuals ages 31-35 in June 2012 (N=6,000)
- **Decision**: Use the pre-constructed ELIGIBLE variable from the provided dataset rather than constructing my own

### 3. Outcome Variable
- **Decision**: Use FT (full-time employment) as provided, coded as 1 if usually works 35+ hours/week
- **Note**: Those not in the labor force are included as 0 values per instructions

### 4. Model Specification
- **Preferred Model**: Basic DiD with robust (HC1) standard errors
- **Equation**: FT = β₀ + β₁(ELIGIBLE) + β₂(AFTER) + β₃(ELIGIBLE×AFTER) + ε
- **Rationale**: Simple specification avoids potential bias from controlling for post-treatment variables while robust SEs account for heteroskedasticity

### 5. Robustness Checks
Conducted the following sensitivity analyses:
1. DiD with demographic controls (FEMALE, FAMSIZE, NCHILD)
2. DiD with year fixed effects
3. DiD with state fixed effects
4. Full model with year FE + state FE + demographics
5. Weighted analysis using PERWT
6. Heterogeneous effects by gender and education
7. Placebo test (fake treatment at 2010)
8. Event study analysis

### 6. Parallel Trends Assessment
- **Method**: Tested for differential pre-trends using ELIGIBLE × YEAR interaction in pre-period
- **Result**: Coefficient = 0.015, p = 0.103 (not significant at 5% level)
- **Conclusion**: Parallel trends assumption is supported

---

## Commands Executed

### Data Loading and Exploration
```python
# Load numeric version of data (to avoid string type issues)
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Check shape
print(df.shape)  # (17382, 105)

# Verify key variables
print(df['FT'].value_counts())  # 1: 11283, 0: 6099
print(df['ELIGIBLE'].value_counts())  # 1: 11382, 0: 6000
print(df['AFTER'].value_counts())  # 0: 9527, 1: 7855
```

### Main Analysis
```python
# Basic DiD regression
import statsmodels.formula.api as smf

df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
model = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
```

### Parallel Trends Test
```python
# Pre-period only
pre_data = df[df['AFTER']==0].copy()
pre_data['YEAR_centered'] = pre_data['YEAR'] - 2008
pre_data['ELIGIBLE_YEAR'] = pre_data['ELIGIBLE'] * pre_data['YEAR_centered']

trend_model = smf.ols('FT ~ ELIGIBLE + YEAR_centered + ELIGIBLE_YEAR', data=pre_data).fit()
```

### LaTeX Compilation
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_09"
pdflatex -interaction=nonstopmode replication_report_09.tex
pdflatex -interaction=nonstopmode replication_report_09.tex
pdflatex -interaction=nonstopmode replication_report_09.tex
```

---

## Results Summary

### Main Finding
| Metric | Value |
|--------|-------|
| DiD Coefficient | 0.0643 |
| Robust SE | 0.0153 |
| 95% CI | [0.034, 0.094] |
| p-value | < 0.001 |
| Sample Size | 17,382 |

**Interpretation**: DACA eligibility is associated with a 6.43 percentage point increase in the probability of full-time employment.

### Raw DiD Calculation
| Group | Pre (2008-11) | Post (2013-16) | Difference |
|-------|---------------|----------------|------------|
| Control (31-35) | 0.670 | 0.645 | -0.025 |
| Treated (26-30) | 0.626 | 0.666 | +0.039 |
| **DiD** | | | **0.064** |

### Robustness Across Specifications
| Model | Coefficient | SE | p-value |
|-------|------------|-----|---------|
| Basic DiD | 0.064 | 0.015 | <0.001 |
| With Demographics | 0.054 | 0.014 | <0.001 |
| Robust SE | 0.064 | 0.015 | <0.001 |
| Year FE | 0.063 | 0.015 | <0.001 |
| State FE | 0.064 | 0.015 | <0.001 |
| Full Model | 0.053 | 0.014 | <0.001 |
| Weighted | 0.075 | 0.015 | <0.001 |

### Validity Checks
- **Parallel Trends**: Differential pre-trend coefficient = 0.015, p = 0.103 (not significant)
- **Placebo Test**: Fake treatment at 2010 yields DiD = 0.016, p = 0.444 (not significant)

---

## Files Generated

### Analysis Code
- `analysis.py` - Main DiD analysis script
- `create_figures.py` - Figure generation script

### Output Files
- `analysis_results.txt` - Summary statistics and key results
- `results_table.csv` - Main results table
- `yearly_means.csv` - FT rates by year and group
- `event_study.csv` - Event study coefficients

### Figures
- `figure1_parallel_trends.pdf/png` - Parallel trends visualization
- `figure2_did_illustration.pdf/png` - DiD estimation illustration
- `figure3_event_study.pdf/png` - Event study plot
- `figure4_coefficient_plot.pdf/png` - Coefficient estimates across models
- `figure5_ft_distribution.pdf/png` - FT distribution by group/period
- `figure6_sample_composition.pdf/png` - Sample size by year

### Report
- `replication_report_09.tex` - LaTeX source
- `replication_report_09.pdf` - Final report (21 pages)

---

## Software Environment
- Python 3.x with pandas, numpy, statsmodels, matplotlib, scipy
- MiKTeX (pdfLaTeX) for PDF compilation
- Windows 10/11

---

## Notes

1. Used the numeric version of the data (`prepared_data_numeric_version.csv`) rather than the labeled version to avoid type conversion issues with categorical variables.

2. The ELIGIBLE variable was provided pre-constructed in the dataset and used as-is per instructions.

3. The year 2012 is excluded from the data because DACA was implemented mid-year (June 15, 2012), making it impossible to determine treatment status for observations in that year.

4. Survey weights (PERWT) were used for a robustness check but the unweighted estimate was chosen as the preferred specification to maintain simplicity and interpretability.

5. Heteroskedasticity-robust standard errors (HC1) were used throughout to ensure valid inference without relying on homoskedasticity assumptions.
