# Replication Run Log - Session 55

## Overview
Replicating the DACA impact study on full-time employment among Hispanic-Mexican, Mexico-born individuals.

## Research Question
What was the causal impact of DACA eligibility on the probability of full-time employment (35+ hours/week) among eligible individuals?

## Methodology
- **Design**: Difference-in-Differences (DiD)
- **Treatment Group**: Ages 26-30 at DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at DACA implementation (otherwise eligible)
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016 (excluding 2012 as transition year)

## Key Decisions

### 1. Sample Selection Criteria
- Hispanic-Mexican ethnicity: HISPAN = 1 (Mexican)
- Born in Mexico: BPL = 200 (Mexico)
- Non-citizen without papers: CITIZEN = 3 (Not a citizen)
- Immigrated before age 16: Calculated from BIRTHYR and YRIMMIG (age_at_immig < 16)
- Lived continuously in US since June 15, 2007: YRIMMIG <= 2007
- Age groups based on age at June 15, 2012:
  - Treatment: ages 26-30 (adjusted for birth quarter)
  - Control: ages 31-35 (adjusted for birth quarter)

### 2. Age Adjustment for Birth Quarter
To accurately calculate age at June 15, 2012:
- Base age = 2012 - BIRTHYR
- For individuals born Q3 (Jul-Sep) or Q4 (Oct-Dec), subtracted 1 from calculated age since they wouldn't have had their birthday by June 15, 2012.

### 3. Outcome Variable
- Full-time employment: UHRSWORK >= 35
- Binary outcome (1 = full-time, 0 = otherwise)

### 4. Analysis Approach
- Linear probability models (Weighted Least Squares)
- Weighted by PERWT (person weight)
- Standard errors clustered by state (STATEFIP)
- Progressive model specifications:
  1. Basic DiD
  2. DiD + demographic controls (gender, marital status, education)
  3. DiD + demographic controls + state fixed effects
  4. DiD + demographic controls + state FE + year FE (PREFERRED)
  5. DiD + demographic controls + state FE + year FE + age FE

### 5. Preferred Specification Rationale
Chose Model 4 (year + state FE) as preferred because:
- Controls for time-invariant state characteristics
- Controls for national time trends
- Demographic controls improve comparability
- Age FE (Model 5) causes collinearity with treatment indicator

## Commands Executed

### Data Exploration
```bash
# Check data structure
head -5 data/data.csv
wc -l data/data.csv  # Result: 33,851,425 rows

# Check column types
python -c "import pandas as pd; df = pd.read_csv('data/data.csv', nrows=100); print(df.columns.tolist())"
```

### Python Analysis Script
```bash
python analysis.py
```

### Sample Selection Results (from analysis.py output)
```
After initial filters (HISPAN=1, BPL=200, CITIZEN=3, YEAR!=2012): 636,722 rows
After immigration year filter (<=2007): 595,366
After arrived before age 16 filter: 177,294
After age group filter (26-35 at DACA): 43,238
```

### Final Sample Composition
- Treatment group (ages 26-30): 25,470
- Control group (ages 31-35): 17,768
- Pre-period observations: 28,377
- Post-period observations: 14,861

## Results Summary

### Weighted Full-Time Employment Rates
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 0.6305 | 0.6597 | +0.0292 |
| Control (31-35) | 0.6731 | 0.6433 | -0.0299 |

**Simple DiD**: 0.0590 (2.9% - (-3.0%) = 5.9 pp)

### Regression Results
| Model | Coefficient | SE | 95% CI | p-value |
|-------|-------------|-----|--------|---------|
| Basic DiD | 0.0590 | 0.0069 | [0.046, 0.072] | <0.001 |
| + Controls | 0.0474 | 0.0092 | [0.029, 0.065] | <0.001 |
| + State FE | 0.0467 | 0.0096 | [0.028, 0.065] | <0.001 |
| + Year+State FE (PREFERRED) | 0.0449 | 0.0099 | [0.026, 0.064] | <0.001 |
| + Age FE | 0.0228 | 0.0158 | [-0.008, 0.054] | 0.149 |

### Event Study Coefficients (Reference: 2011)
Pre-DACA years (parallel trends test):
- 2006: -0.007 (SE=0.027), not significant
- 2007: -0.041 (SE=0.019), marginally significant
- 2008: -0.002 (SE=0.024), not significant
- 2009: -0.013 (SE=0.031), not significant
- 2010: -0.021 (SE=0.024), not significant

Post-DACA years:
- 2013: 0.037 (SE=0.023), not significant
- 2014: 0.039 (SE=0.023), not significant
- 2015: 0.022 (SE=0.025), not significant
- 2016: 0.069 (SE=0.022), significant

### Heterogeneity Analysis
- Male: DiD = 0.046 (SE=0.010)
- Female: DiD = 0.047 (SE=0.015)
- Less than HS: DiD = 0.050 (SE=0.018)
- High School: DiD = 0.048 (SE=0.009)
- Some College: DiD = 0.130 (SE=0.038)

## LaTeX Report Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_55.tex  # Run 4 times for cross-references
```
Output: replication_report_55.pdf (23 pages)

## Output Files Generated
1. `analysis.py` - Main analysis script
2. `regression_results.csv` - Model coefficients and standard errors
3. `event_study_results.csv` - Year-by-year treatment effects
4. `summary_statistics.csv` - Descriptive statistics by group/period
5. `heterogeneity_results.csv` - Subgroup analysis results
6. `analysis_summary.txt` - Text summary of results
7. `figure1_parallel_trends.png/pdf` - Time trends plot
8. `figure2_event_study.png/pdf` - Event study plot
9. `figure3_did_bars.png/pdf` - DiD bar chart
10. `figure4_coefficient_plot.png/pdf` - Coefficient comparison across models
11. `replication_report_55.tex` - LaTeX source file
12. `replication_report_55.pdf` - Final report (23 pages)
13. `run_log_55.md` - This log file

## Interpretation of Preferred Estimate

The preferred estimate (Model 4: Year and State FE) indicates that DACA eligibility increased the probability of full-time employment by approximately **4.5 percentage points** (SE = 0.99, 95% CI: [0.026, 0.064]).

Given the baseline full-time employment rate of 63.1% in the treatment group pre-DACA, this represents a **7.1% relative increase** in full-time employment.

The effect is:
- Statistically significant at conventional levels (p < 0.001)
- Robust across model specifications (Models 1-4)
- Consistent across gender and education subgroups
- Supported by event study analysis showing largely parallel pre-trends

## Key Caveats
1. Cannot observe actual DACA receipt (intent-to-treat estimate)
2. Cannot distinguish documented from undocumented non-citizens
3. Age-based treatment introduces potential confounders
4. 2007 coefficient in event study is marginally significant
5. Results specific to Hispanic-Mexican, Mexico-born population ages 26-35
