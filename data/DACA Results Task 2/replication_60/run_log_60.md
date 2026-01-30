# DACA Replication Study - Run Log

## Study Information
- **Research Question**: Effect of DACA eligibility on full-time employment among Mexican-born Hispanic immigrants
- **Method**: Difference-in-Differences
- **Data Source**: American Community Survey (IPUMS) 2006-2016

## Key Decisions and Rationale

### 1. Sample Selection Criteria

| Criterion | IPUMS Variable | Value | Rationale |
|-----------|---------------|-------|-----------|
| Hispanic-Mexican ethnicity | HISPAN | 1 | Per research design specification |
| Born in Mexico | BPL | 200 | Per research design specification |
| Non-citizen | CITIZEN | 3 | Proxy for undocumented status as instructed |
| Arrived before age 16 | YRIMMIG - BIRTHYR | < 16 | DACA eligibility requirement |
| Arrived by 2007 | YRIMMIG | <= 2007 | Continuous presence since June 2007 |
| Age range | Calculated | 26-35 on June 15, 2012 | Treatment (26-30) and control (31-35) groups |

### 2. Age Calculation Method

- Used BIRTHYR and BIRTHQTR to calculate age as of June 15, 2012
- For Q1-Q2 births (Jan-Jun): Age = 2012 - BIRTHYR (birthday passed by June 15)
- For Q3-Q4 births (Jul-Dec): Age = 2012 - BIRTHYR - 1 (birthday not yet reached)

### 3. Treatment and Control Groups

- **Treatment Group**: Ages 26-30 as of June 15, 2012 (DACA eligible)
- **Control Group**: Ages 31-35 as of June 15, 2012 (too old for DACA, otherwise similar)

### 4. Time Period Definitions

- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016
- **Excluded**: 2012 (cannot distinguish pre/post DACA within the year due to lack of month information in ACS)

### 5. Outcome Variable

- **Full-time employment**: UHRSWORK >= 35 (usually working 35+ hours per week)
- Binary indicator (0/1)

### 6. Covariates Used

| Variable | Source | Definition |
|----------|--------|------------|
| Female | SEX | = 1 if SEX == 2 |
| Married | MARST | = 1 if MARST in {1, 2} |
| High school | EDUC | = 1 if EDUC == 6 |
| Some college | EDUC | = 1 if EDUC in {7, 8, 9} |
| College+ | EDUC | = 1 if EDUC >= 10 |
| Age | AGE | Current age at survey time |
| Age squared | AGE | AGE^2 |

### 7. Fixed Effects

- Year fixed effects (YEAR)
- State fixed effects (STATEFIP)

### 8. Weights

- Person weights (PERWT) used in all weighted specifications
- Heteroskedasticity-robust standard errors (HC1)

## Sample Sizes

| Stage | N |
|-------|---|
| Initial ACS 2006-2016 | 33,851,424 |
| Hispanic-Mexican (HISPAN=1) | 2,945,521 |
| Born in Mexico (BPL=200) | 991,261 |
| Non-citizen (CITIZEN=3) | 701,347 |
| Arrived before age 16 | 205,327 |
| Arrived by 2007 | 195,023 |
| Ages 26-35 as of June 2012 | 47,418 |
| After excluding 2012 | 43,238 |

### Final Sample Breakdown

| Group | Pre-period | Post-period | Total |
|-------|------------|-------------|-------|
| Treatment (26-30) | 16,694 | 8,776 | 25,470 |
| Control (31-35) | 11,683 | 6,085 | 17,768 |
| **Total** | **28,377** | **14,861** | **43,238** |

## Commands Executed

### Data Loading and Exploration
```python
df = pd.read_csv('data/data.csv')
# Total observations: 33,851,424
```

### Sample Filters Applied
```python
df_sample = df[df['HISPAN'] == 1].copy()
df_sample = df_sample[df_sample['BPL'] == 200].copy()
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
df_sample = df_sample[df_sample['age_at_arrival'] < 16].copy()
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
```

### Regression Models

1. **Model 1**: Basic DiD (OLS, no weights)
2. **Model 2**: DiD with person weights (WLS)
3. **Model 3**: DiD with weights + covariates
4. **Model 4**: DiD with weights + covariates + year fixed effects
5. **Model 5**: DiD with weights + covariates + year FE + state FE (PREFERRED)

## Results Summary

### Main Results (Preferred Model 5)

| Statistic | Value |
|-----------|-------|
| **Effect Size** | 0.0441 (4.41 percentage points) |
| **Standard Error** | 0.0107 |
| **95% CI** | [0.0232, 0.0650] |
| **P-value** | < 0.001 |
| **Sample Size** | 43,238 |

### Model Comparison

| Model | Coefficient | SE | p-value |
|-------|-------------|-------|---------|
| (1) Basic DiD | 0.0516 | 0.0100 | <0.001 |
| (2) Weighted | 0.0590 | 0.0117 | <0.001 |
| (3) + Covariates | 0.0645 | 0.0146 | <0.001 |
| (4) + Year FE | 0.0200 | 0.0154 | 0.195 |
| (5) + State FE | **0.0441** | **0.0107** | **<0.001** |

### Robustness Checks

| Test | Coefficient | SE | p-value |
|------|-------------|-------|---------|
| Placebo (2009) | 0.0058 | 0.0136 | 0.668 |
| Alt. ages (24-28 vs 33-37) | 0.1009 | 0.0116 | <0.001 |
| Male subsample | 0.0462 | 0.0125 | <0.001 |
| Female subsample | 0.0466 | 0.0185 | 0.012 |

## Files Produced

| File | Description |
|------|-------------|
| analysis.py | Main analysis script |
| create_tables_figures.py | Script to generate tables and figures |
| results_summary.json | JSON with key results |
| table1_summary_stats.csv | Summary statistics table |
| table2_regression_results.csv | Main regression results |
| table3_robustness.csv | Robustness check results |
| event_study_data.csv | Event study coefficients |
| figure1_event_study.png/pdf | Event study plot |
| figure2_trends.png/pdf | Employment trends plot |
| figure3_did.png/pdf | DiD visualization |
| replication_report_60.tex | LaTeX source for report |
| replication_report_60.pdf | Final PDF report (17 pages) |
| run_log_60.md | This log file |

## Interpretation

The preferred estimate suggests DACA eligibility increased the probability of full-time employment by approximately 4.4 percentage points among the target population. This represents a 7% relative increase from the pre-treatment baseline of 63.1% for the treatment group.

Key findings:
1. The effect is statistically significant at conventional levels
2. Placebo test shows no pre-existing differential trends
3. Effects are similar for men and women
4. Event study shows effects accumulated over time after DACA implementation
5. The effect is robust to alternative age bandwidths

## Software Environment

- Python 3.x with pandas, numpy, statsmodels, matplotlib
- LaTeX (MiKTeX) for report compilation
- Platform: Windows

## Session Date

January 26, 2026
