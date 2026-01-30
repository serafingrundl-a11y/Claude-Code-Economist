# DACA Replication Study - Run Log

## Study Information
- **Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican immigrants
- **Treatment Group**: Ages 26-30 as of June 15, 2012 (ELIGIBLE=1)
- **Control Group**: Ages 31-35 as of June 15, 2012 (ELIGIBLE=0)
- **Outcome**: Full-time employment (FT), defined as usually working 35+ hours per week
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016
- **Method**: Difference-in-differences

---

## Session Log

### 1. Data Exploration

#### Commands:
```bash
# List data files
ls -la data/

# View data structure
head -5 data/prepared_data_labelled_version.csv
head -3 data/prepared_data_numeric_version.csv

# Count observations
wc -l data/prepared_data_numeric_version.csv
# Result: 17,383 lines (17,382 observations + header)

# List all column names
head -1 data/prepared_data_numeric_version.csv | tr ',' '\n' | nl
```

#### Key Findings:
- Total observations: 17,382
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-period (AFTER=0): 9,527 observations
- Post-period (AFTER=1): 7,855 observations
- Years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 omitted)

---

### 2. Analysis Script Development

#### File: `analysis.py`

Key decisions made:
1. **Data file used**: `prepared_data_numeric_version.csv` (numeric coding for regression)
2. **Sample**: Used entire provided sample without additional restrictions (as instructed)
3. **Treatment/Control**: Used provided ELIGIBLE variable
4. **Outcome**: Used provided FT variable

#### Variable Construction:
```python
# Interaction term for DiD
data['ELIGIBLE_AFTER'] = data['ELIGIBLE'] * data['AFTER']

# Demographic controls
data['FEMALE'] = (data['SEX'] == 2).astype(int)  # IPUMS: 1=Male, 2=Female
data['AGE_centered'] = data['AGE'] - data['AGE'].mean()
data['MARRIED'] = (data['MARST'].isin([1, 2])).astype(int)  # 1,2 = married
data['HAS_CHILDREN'] = (data['NCHILD'] > 0).astype(int)

# Education dummies (reference = less than HS)
data['educ_less_than_hs'] = (data['EDUC'] < 6).astype(int)
data['educ_hs'] = (data['EDUC'] == 6).astype(int)
data['educ_some_college'] = ((data['EDUC'] >= 7) & (data['EDUC'] <= 9)).astype(int)
data['educ_ba_plus'] = (data['EDUC'] >= 10).astype(int)
```

#### Command to run analysis:
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_35"
python analysis.py
```

---

### 3. Key Results

#### Manual DiD Calculation:
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 0.6263 | 0.6658 | +0.0394 |
| Control (31-35) | 0.6697 | 0.6449 | -0.0248 |
| **DiD** | | | **0.0643** |

#### Regression Models Summary:

| Model | DiD Estimate | SE | 95% CI | p-value | R-squared |
|-------|-------------|-----|--------|---------|-----------|
| (1) Basic DiD | 0.064 | 0.015 | [0.034, 0.094] | <0.001 | 0.002 |
| (2) Demographics | 0.055 | 0.014 | [0.027, 0.083] | <0.001 | 0.126 |
| (3) Demo+Education | 0.053 | 0.014 | [0.025, 0.080] | <0.001 | 0.130 |
| (4) Year FE | 0.051 | 0.014 | [0.023, 0.079] | <0.001 | 0.133 |
| (5) Year+State FE | 0.051 | 0.014 | [0.024, 0.079] | <0.001 | 0.136 |
| (6) Weighted | 0.059 | 0.017 | [0.026, 0.091] | <0.001 | 0.138 |

#### Preferred Estimate (Model 5):
- **Effect size**: 0.051 (5.1 percentage points)
- **Standard error**: 0.014
- **95% CI**: [0.024, 0.079]
- **p-value**: 0.0003
- **Sample size**: 17,382

---

### 4. Robustness Checks

#### Event Study (relative to 2011):
| Year | Coefficient | SE | p-value |
|------|------------|-----|---------|
| 2008 | 0.060 | 0.018 | 0.001 |
| 2009 | 0.021 | 0.025 | 0.420 |
| 2010 | 0.000 | 0.025 | 0.994 |
| 2013 | 0.078 | 0.026 | 0.003 |
| 2014 | 0.041 | 0.026 | 0.123 |
| 2015 | 0.081 | 0.027 | 0.003 |
| 2016 | 0.093 | 0.027 | 0.001 |

*Note*: Pre-period coefficients for 2009-2010 support parallel trends; 2008 coefficient is significant.

#### Heterogeneity by Sex:
- Males: 0.049 (SE=0.017, p=0.003, N=9,075)
- Females: 0.050 (SE=0.023, p=0.031, N=8,307)

#### Placebo Test (2008-2009 vs 2010-2011):
- Estimate: 0.016 (SE=0.021, p=0.444)
- Interpretation: No significant pre-trend difference

---

### 5. Figure Generation

#### File: `create_figures.py`

```bash
python create_figures.py
```

#### Output files:
- `figure1_trends.png/pdf` - FT employment rates by year and treatment status
- `figure2_eventstudy.png/pdf` - Event study coefficients
- `figure3_did.png/pdf` - DiD visualization

---

### 6. Report Compilation

#### File: `replication_report_35.tex`

```bash
# Compile LaTeX (run 2-3 times for references)
pdflatex -interaction=nonstopmode replication_report_35.tex
pdflatex -interaction=nonstopmode replication_report_35.tex
```

#### Output: `replication_report_35.pdf` (18 pages)

---

## Key Analytical Decisions

1. **Model specification**: Selected Model 5 (year + state FE with demographics and education) as preferred because it controls for both temporal and geographic variation while maintaining the core DiD identification.

2. **Standard errors**: Used heteroskedasticity-robust standard errors (HC1) throughout all specifications.

3. **Weights**: Reported both unweighted (preferred) and weighted estimates. Unweighted is preferred because survey weights are designed for population inference rather than causal identification.

4. **Covariates**: Included sex, age (centered), marital status, children, and education as individual-level controls. Added year and state fixed effects.

5. **Sample**: Used entire provided sample without additional restrictions, as instructed. Individuals not in labor force are coded as FT=0 and retained.

6. **Education coding**: Used EDUC numeric variable to create categories:
   - Less than HS: EDUC < 6
   - HS degree: EDUC = 6
   - Some college: EDUC 7-9
   - BA+: EDUC >= 10

---

## Output Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `model_summary.csv` | Summary of all regression models |
| `event_study_results.csv` | Event study coefficients |
| `balance_table.csv` | Pre-treatment balance statistics |
| `yearly_stats.csv` | Year-by-year FT rates by group |
| `figure1_trends.png/pdf` | Trends figure |
| `figure2_eventstudy.png/pdf` | Event study figure |
| `figure3_did.png/pdf` | DiD visualization |
| `replication_report_35.tex` | LaTeX source |
| `replication_report_35.pdf` | Final report |
| `run_log_35.md` | This file |

---

## Summary

This replication finds that DACA eligibility increased full-time employment by approximately **5.1 percentage points** (95% CI: 2.4-7.9 pp, p<0.001). The effect is robust across specifications and consistent for both men and women. Event study analysis provides mixed support for parallel trends (2009-2010 coefficients are near zero, but 2008 is significant). The placebo test finds no significant pre-existing trends.
