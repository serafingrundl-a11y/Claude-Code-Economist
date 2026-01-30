# DACA Replication Study - Run Log

## Study Information
- **Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals
- **Date**: January 26, 2026
- **Replication ID**: 30

---

## 1. Data Source and Sample Selection

### Data Source
- American Community Survey (ACS) from IPUMS USA
- Years: 2006-2016 (one-year ACS files)
- Original data file: `data/data.csv` (33,851,424 observations)

### Sample Selection Criteria Applied

| Step | Criterion | Variable(s) | Value(s) | Observations Remaining |
|------|-----------|-------------|----------|----------------------|
| 1 | Hispanic-Mexican ethnicity | HISPAN | = 1 | 2,945,521 |
| 2 | Born in Mexico | BPL | = 200 | 991,261 |
| 3 | Not a citizen | CITIZEN | = 3 | 701,347 |
| 4 | Valid immigration year | YRIMMIG | > 0 | 701,347 |
| 5 | Exclude 2012 (ambiguous year) | YEAR | != 2012 | 636,722 |
| 6 | Birth years 1977-1986 | BIRTHYR | 1977-1986 | 162,283 |
| 7 | Arrived before age 16 | YRIMMIG - BIRTHYR | < 16 | 44,725 |
| 8 | Arrived by 2007 | YRIMMIG | <= 2007 | 44,725 |

### Final Sample
- **Unweighted N**: 44,725 person-year observations
- **Weighted N**: 6,205,755 person-years

---

## 2. Key Analytic Decisions

### Treatment and Control Group Definition

| Group | Birth Years | Age at DACA (June 15, 2012) | Rationale |
|-------|-------------|----------------------------|-----------|
| Treatment | 1982-1986 | 26-30 | Just young enough for DACA eligibility |
| Control | 1977-1981 | 31-35 | Just too old (age >= 31 cutoff) |

### DACA Eligibility Criteria Applied
1. **Arrival before age 16**: Calculated as (YRIMMIG - BIRTHYR < 16)
2. **Continuous presence since June 2007**: Proxied by (YRIMMIG <= 2007)
3. **Present in US on June 15, 2012**: Assumed for ACS respondents in 2013+
4. **Non-citizen status**: CITIZEN = 3

### Outcome Variable
- **Full-time employment**: UHRSWORK >= 35 (binary indicator)
- Definition: Usually working 35 or more hours per week

### Period Definition
- **Pre-DACA**: 2006-2011
- **Post-DACA**: 2013-2016
- **Excluded**: 2012 (mid-year implementation creates ambiguity)

---

## 3. Estimation Approach

### Main Specification
Difference-in-differences with the following model:

```
fulltime = a + b1*Treated + b2*Post + d*(Treated x Post) + X'g + year_FE + state_FE + e
```

Where:
- `fulltime`: Indicator for working 35+ hours/week
- `Treated`: Indicator for birth years 1982-1986
- `Post`: Indicator for years 2013-2016
- `Treated x Post`: DiD interaction (coefficient of interest)
- `X`: Demographic controls (female, married, age, education)
- Sample weights (PERWT) applied
- Robust (HC1) standard errors

### Alternative Specifications Estimated
1. Basic DiD (unweighted, no controls)
2. Weighted DiD (no controls)
3. DiD with demographic controls + year FE
4. DiD with demographic controls + year FE + state FE
5. DiD with robust standard errors

---

## 4. Commands Executed

### Python Analysis Script
```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_30"
python analysis.py
```

### Figure Generation
```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_30"
python create_figures.py
```

### LaTeX Compilation
```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_30"
pdflatex -interaction=nonstopmode replication_report_30.tex
pdflatex -interaction=nonstopmode replication_report_30.tex
pdflatex -interaction=nonstopmode replication_report_30.tex
```

---

## 5. Results Summary

### Main Results (Preferred Specification: Model 4)

| Parameter | Value |
|-----------|-------|
| DiD Coefficient | 0.0477 |
| Standard Error | 0.0089 |
| 95% CI | [0.0303, 0.0652] |
| t-statistic | 5.37 |
| p-value | < 0.001 |
| Sample Size | 44,725 |
| R-squared | 0.1593 |

**Interpretation**: DACA eligibility is associated with a 4.77 percentage point increase in full-time employment probability.

### Simple DiD Calculation

|  | Pre-DACA | Post-DACA | Difference |
|--|----------|-----------|------------|
| Control (31-35) | 0.6705 | 0.6412 | -0.0293 |
| Treatment (26-30) | 0.6253 | 0.6580 | +0.0327 |
| **DiD** | | | **0.0620** |

### Event Study Results (Base Year: 2011)

| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | 0.003 | 0.023 | 0.884 |
| 2007 | -0.012 | 0.022 | 0.588 |
| 2008 | 0.015 | 0.022 | 0.507 |
| 2009 | 0.011 | 0.023 | 0.635 |
| 2010 | 0.012 | 0.023 | 0.594 |
| 2011 | 0 (base) | -- | -- |
| 2013 | 0.047 | 0.024 | 0.051 |
| 2014 | 0.053 | 0.024 | 0.029 |
| 2015 | 0.033 | 0.024 | 0.175 |
| 2016 | 0.079 | 0.024 | 0.001 |

**Parallel trends**: Pre-period coefficients are all small and insignificant, supporting the identifying assumption.

### Heterogeneity Analysis

| Subgroup | Coefficient | SE |
|----------|-------------|-----|
| All | 0.048 | 0.011 |
| Male | 0.049 | 0.012 |
| Female | 0.035 | 0.018 |
| Not Married | 0.064 | 0.016 |
| Married | 0.020 | 0.014 |

### Robustness Checks

| Specification | Coefficient | SE |
|---------------|-------------|-----|
| Main | 0.048 | 0.011 |
| Narrow window (27-29 vs 32-34) | 0.050 | 0.013 |
| Donut (exclude 1981-1982) | 0.055 | 0.012 |
| Any employment outcome | 0.047 | 0.010 |
| Unweighted | 0.049 | 0.009 |

---

## 6. Output Files Generated

### Analysis Files
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script

### Results Files (CSV)
- `event_study_results.csv` - Event study coefficients
- `regression_results.csv` - Main regression table
- `descriptive_stats.csv` - Descriptive statistics
- `sample_counts.csv` - Sample sizes by group
- `fulltime_rates.csv` - Full-time employment rates
- `latex_main_table.tex` - LaTeX table fragment

### Figures
- `figure1_event_study.png` / `.pdf` - Event study plot
- `figure2_trends.png` / `.pdf` - Employment trends
- `figure3_sample_dist.png` / `.pdf` - Sample distribution
- `figure4_heterogeneity.png` / `.pdf` - Heterogeneity results
- `figure5_did_visual.png` / `.pdf` - DiD visualization

### Report Files
- `replication_report_30.tex` - LaTeX source
- `replication_report_30.pdf` - Compiled report (24 pages)

---

## 7. Key Methodological Notes

1. **Exclusion of 2012**: The year 2012 was excluded because DACA was implemented mid-year (June 15), making it impossible to classify observations as pre- or post-treatment.

2. **Citizenship assumption**: Following the instructions, all non-citizens (CITIZEN=3) were assumed to be undocumented. The data does not distinguish between undocumented immigrants and legal non-citizens with visas.

3. **Intent-to-treat interpretation**: Estimates reflect the effect of DACA eligibility, not actual DACA receipt. The ACS does not identify DACA recipients.

4. **Age at arrival criterion**: Used year of immigration minus birth year as an approximation of age at arrival. This may have some measurement error due to quarterly/monthly timing.

5. **Continuous presence**: Proxied the continuous presence requirement by requiring immigration by 2007. This is an approximation since the actual requirement is continuous presence since June 15, 2007.

6. **Sample weights**: All weighted analyses use PERWT from IPUMS. Unweighted results are similar.

---

## 8. Interpretation

The analysis finds statistically significant evidence that DACA eligibility increased full-time employment by approximately 4.8 percentage points among eligible Hispanic-Mexican immigrants born in Mexico. This represents a 7.6% relative increase from the pre-DACA baseline of 62.5%.

The effect is:
- Robust across alternative specifications and age windows
- Larger for men than women (but both positive)
- Larger for unmarried than married individuals
- Consistent with an intent-to-treat interpretation

The parallel trends assumption is supported by the event study, which shows no significant pre-trends and a clear break in 2013 when the post-DACA effects emerge.

---

## 9. Software and Packages

- Python 3.x
- pandas
- numpy
- statsmodels
- matplotlib
- LaTeX (pdflatex via MiKTeX)
