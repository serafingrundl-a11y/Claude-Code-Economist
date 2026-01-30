# Run Log - DACA Replication Study (ID: 72)

## Overview
This log documents all commands and key decisions for the DACA replication study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Design Summary
- **Research Question**: Effect of DACA eligibility on full-time employment (35+ hours/week)
- **Treatment Group**: DACA-eligible individuals aged 26-30 as of June 15, 2012
- **Control Group**: Individuals aged 31-35 as of June 15, 2012 (ineligible by age but otherwise meeting criteria)
- **Method**: Difference-in-Differences with WLS (weighted by PERWT), clustered SE by state
- **Pre-period**: 2006-2011 (6 years)
- **Post-period**: 2013-2016 (4 years)
- **Excluded year**: 2012 (implementation timing ambiguity)

## Data Description
- **Source**: IPUMS ACS data 2006-2016 (1-year files)
- **File**: data/data.csv (6.26 GB, 33,851,424 observations)
- **Data dictionary**: data/acs_data_dict.txt

## Key Variable Definitions (IPUMS Names)
| Variable | Definition | Values Used |
|----------|------------|-------------|
| YEAR | Survey year | 2006-2016 (excluding 2012) |
| BIRTHYR | Year of birth | Used for age calculation |
| BIRTHQTR | Quarter of birth | 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec |
| HISPAN | Hispanic origin | 1 = Mexican |
| BPL | Birthplace | 200 = Mexico |
| CITIZEN | Citizenship status | 3 = Not a citizen |
| YRIMMIG | Year of immigration | <= 2007 for continuous residence |
| UHRSWORK | Usual hours worked/week | >= 35 for full-time |
| EMPSTAT | Employment status | 1 = Employed |
| PERWT | Person weight | Used for WLS |
| STATEFIP | State FIPS code | Used for clustering SE |
| SEX | Sex | 1=Male, 2=Female |
| MARST | Marital status | 1,2 = Married |
| EDUC | Education | Categorical (0-11) |

## Sample Construction Steps

### Step 1: Load full ACS data
```
Total observations: 33,851,424
```

### Step 2: Apply eligibility filters
```
1. Hispanic-Mexican (HISPAN == 1): 2,945,521
2. Born in Mexico (BPL == 200): 991,261
3. Non-citizen (CITIZEN == 3): 701,347
4. Valid immigration year (YRIMMIG > 0): 701,347
5. Arrived before age 16: 205,327
6. Arrived by 2007: 195,023
7. Ages 26-35 as of June 15, 2012: 47,418
8. Excluding 2012: 43,238 (FINAL ANALYTIC SAMPLE)
```

### Step 3: Treatment assignment
- **Treatment (ages 26-30)**: 25,470 observations
- **Control (ages 31-35)**: 17,768 observations

## Key Analytical Decisions

### Decision 1: Age Calculation Method
**Choice**: Age as of June 15, 2012 = 2012 - BIRTHYR, minus 1 for Q3/Q4 births
**Rationale**: DACA cutoff was age 31 by June 15, 2012. Those born July-December hadn't yet had their birthday by June 15, so subtract 1 from their age.

### Decision 2: Undocumented Status
**Choice**: Treat CITIZEN == 3 (non-citizen) as undocumented
**Rationale**: Per instructions, cannot distinguish documented vs undocumented non-citizens in ACS. Instructions state to "assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes."

### Decision 3: Continuous Residence
**Choice**: YRIMMIG <= 2007
**Rationale**: DACA required continuous presence since June 15, 2007. Using year of immigration <= 2007 approximates this criterion.

### Decision 4: Excluding 2012
**Choice**: Drop all observations from 2012
**Rationale**: DACA implemented June 15, 2012. ACS doesn't record interview month, so cannot distinguish pre/post within 2012.

### Decision 5: Standard Error Clustering
**Choice**: Cluster at state level (STATEFIP)
**Rationale**: Account for within-state correlation in outcomes. 51 clusters (50 states + DC).

### Decision 6: Covariates
**Choice**: Include female, married, age, age^2, education dummies
**Rationale**: Control for demographic differences between treatment and control groups that might affect employment.

### Decision 7: Age Bandwidth
**Choice**: Main spec uses 26-30 (treatment) vs 31-35 (control)
**Rationale**: 5-year bandwidth on each side of cutoff provides reasonable sample size. Robustness check with narrower bandwidth (27-29 vs 32-34) shows larger effect.

## Commands Executed

### Data Loading and Filtering
```python
# Load data with selected columns
df = pd.read_csv('data/data.csv', usecols=cols_needed, dtype=dtypes)

# Sequential filtering
df = df[df['HISPAN'] == 1]
df = df[df['BPL'] == 200]
df = df[df['CITIZEN'] == 3]
df = df[df['YRIMMIG'] > 0]
df = df[df['age_at_arrival'] < 16]  # where age_at_arrival = YRIMMIG - BIRTHYR
df = df[df['YRIMMIG'] <= 2007]
df = df[(df['age_june2012'] >= 26) & (df['age_june2012'] <= 35)]
df = df[df['YEAR'] != 2012]
```

### Variable Creation
```python
# Age as of June 15, 2012
df['age_june2012'] = 2012 - df['BIRTHYR']
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june2012'] -= 1

# Treatment indicator
df['treated'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)

# Post-period indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Outcome variable
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)

# DiD interaction
df['treated_post'] = df['treated'] * df['post']
```

### Main Regression
```python
import statsmodels.formula.api as smf

model = smf.wls('fulltime ~ treated + post + treated_post + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_coll',
                data=df, weights=df['PERWT'])
results = model.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```

## Results Summary

### Main Findings
| Specification | DiD Estimate | SE | 95% CI | p-value |
|--------------|--------------|-----|--------|---------|
| No controls | 0.0642 | 0.0064 | [0.052, 0.077] | <0.001 |
| **With controls** | **0.0679** | **0.0110** | **[0.046, 0.089]** | **<0.001** |

### Preferred Estimate
- **Effect Size**: 0.0679 (6.79 percentage points)
- **Standard Error**: 0.0110
- **95% CI**: [0.0464, 0.0893]
- **p-value**: <0.0001
- **Sample Size**: 43,238

### Interpretation
DACA eligibility increased full-time employment by approximately 6.8 percentage points among eligible Hispanic-Mexican, Mexican-born individuals. This represents a ~12.5% increase relative to the baseline full-time employment rate of 54.1% in the treatment group.

### Robustness Checks
| Specification | Estimate | SE |
|---------------|----------|-----|
| Unweighted | 0.0686 | 0.0104 |
| Narrow bandwidth (27-29 vs 32-34) | 0.0849 | 0.0138 |
| Employment (any) outcome | 0.0499 | 0.0100 |
| Males only | 0.0612 | 0.0309 |
| Females only | 0.0688 | 0.0278 |

### Event Study
Pre-treatment coefficients (testing parallel trends):
- 2006: 0.022 (SE 0.023, p=0.34)
- 2007: -0.013 (SE 0.019, p=0.49)
- 2008: 0.017 (SE 0.023, p=0.48)
- 2009: 0.001 (SE 0.024, p=0.98)
- 2010: -0.008 (SE 0.022, p=0.73)

**Conclusion**: No significant pre-trends; parallel trends assumption appears satisfied.

## Output Files Generated
1. `replication_report_72.tex` - LaTeX report (20 pages)
2. `replication_report_72.pdf` - Compiled PDF report
3. `run_log_72.md` - This run log
4. `analysis.py` - Main analysis script
5. `create_figures.py` - Figure generation script
6. `summary_stats.csv` - Summary statistics by group/period
7. `event_study_results.csv` - Year-by-year coefficients
8. `trends_by_year.csv` - Trend data for Figure 1
9. `figure1_parallel_trends.pdf/png` - Trends visualization
10. `figure2_event_study.pdf/png` - Event study plot
11. `figure3_did_visualization.pdf/png` - DiD illustration

## Session Timeline
- **Started**: 2026-01-26
- **Data loading**: ~3 minutes
- **Analysis execution**: ~2 minutes
- **Report compilation**: ~1 minute
- **Total time**: ~15 minutes

## Software Environment
- Python 3.14.2
- pandas 2.3.3
- statsmodels
- numpy
- matplotlib
- LaTeX (MiKTeX distribution)
