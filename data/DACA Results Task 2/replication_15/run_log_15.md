# Run Log - DACA Replication Study 15

## Overview
Independent replication examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Design
- **Treatment group**: Ages 26-30 at DACA implementation (June 15, 2012)
- **Control group**: Ages 31-35 at DACA implementation (otherwise eligible)
- **Outcome**: Full-time employment (35+ hours per week usually worked)
- **Method**: Difference-in-differences
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016 (excluding 2012 due to timing ambiguity)

---

## Key Decisions Log

### 1. Data Loading and Initial Filtering
- **Date**: 2025-01-25
- **Decision**: Load only necessary columns to manage memory with 6.2GB CSV file
- **Columns used**: YEAR, STATEFIP, PERWT, SEX, AGE, BIRTHQTR, MARST, BIRTHYR, HISPAN, BPL, CITIZEN, YRIMMIG, EDUC, EMPSTAT, UHRSWORK
- **Rationale**: Full dataset has 33.8 million rows; chunked reading with immediate filtering necessary

### 2. Sample Selection Criteria
Applied in sequence:

| Step | Criterion | Variable | Value | Remaining N |
|------|-----------|----------|-------|-------------|
| 1 | Hispanic-Mexican ethnicity | HISPAN | = 1 | 991,261 |
| 2 | Born in Mexico | BPL | = 200 | (applied with step 1) |
| 3 | Non-citizen (proxy for undocumented) | CITIZEN | = 3 | 701,347 |
| 4 | Age 26-35 at DACA | Calculated | 26-35 | 181,229 |
| 5 | Arrived before age 16 | YRIMMIG <= BIRTHYR + 15 | Yes | 58,740 |
| 6 | In US since 2007 | YRIMMIG | <= 2007 | 58,740 |
| 7 | Exclude 2012 | YEAR | != 2012 | 53,490 |

**Rationale for non-citizen proxy**: The ACS does not identify undocumented status directly. Non-citizens who have not naturalized are most likely to include undocumented individuals. This is a standard approach in the literature.

### 3. Age at DACA Calculation
- **Method**: `age_at_daca = 2012 - BIRTHYR`
- **Adjustment**: Subtracted 1 year for individuals born in quarters 3 or 4 (July-December), as they would not have reached their birthday by June 15, 2012
- **Rationale**: DACA implementation was June 15, 2012; individuals born later in the year would still be one year younger at that date

### 4. Treatment Assignment
- **Treatment (Treat=1)**: age_at_daca in [26, 27, 28, 29, 30]
- **Control (Treat=0)**: age_at_daca in [31, 32, 33, 34, 35]
- **Rationale**: DACA required not having reached 31st birthday by June 15, 2012; control group is similar in age but just above the cutoff

### 5. DACA Eligibility Implementation
Applied two verifiable criteria from ACS data:
1. **Arrived before 16**: YRIMMIG <= BIRTHYR + 15
2. **Continuous residence since 2007**: YRIMMIG <= 2007

**Note**: Could not apply educational requirement (in school, HS diploma, GED) as this would require current enrollment/completion, and the sample includes people observed at various points in time.

### 6. Outcome Definition
- **Variable**: fulltime = 1 if UHRSWORK >= 35, else 0
- **UHRSWORK**: "Usual hours worked per week"
- **Threshold**: 35 hours is the standard BLS definition of full-time work

### 7. Time Period Definition
- **Pre-period**: 2006-2011 (6 years)
- **Post-period**: 2013-2016 (4 years)
- **Excluded**: 2012 (DACA implemented mid-year; cannot distinguish pre/post observations)

### 8. Standard Error Clustering
- **Clustering level**: State (STATEFIP)
- **Rationale**: Labor markets and policy environments vary by state; accounts for within-state correlation

### 9. Survey Weights
- **Used**: PERWT (person weight)
- **Rationale**: ACS is a sample survey; weights required for nationally representative estimates

---

## Commands Executed

### Python Analysis Script
```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_15"
python analysis.py
```

### Key Python Code Logic
```python
# Sample selection
df_sample = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)]  # Hispanic-Mexican, born in Mexico
df_sample = df_sample[df_sample['CITIZEN'] == 3]  # Non-citizen

# Age at DACA
df_sample['age_at_daca'] = 2012 - df_sample['BIRTHYR']
df_sample.loc[df_sample['BIRTHQTR'].isin([3, 4]), 'age_at_daca'] -= 1

# Treatment assignment
df_sample = df_sample[(df_sample['age_at_daca'] >= 26) & (df_sample['age_at_daca'] <= 35)]
df_sample['treat'] = (df_sample['age_at_daca'] <= 30).astype(int)

# DACA eligibility
df_sample = df_sample[df_sample['YRIMMIG'] <= df_sample['BIRTHYR'] + 15]  # Before age 16
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]  # Since 2007

# Outcome
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# DiD model
model = smf.wls('fulltime ~ treat + post + treat_post',
                data=df_sample, weights=df_sample['PERWT']).fit(
                cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
```

### Figure Generation
```bash
python create_figures.py
```

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_15.tex
pdflatex -interaction=nonstopmode replication_report_15.tex  # Run twice for references
```

---

## Results Summary

### Main Estimate (Preferred)
| Metric | Value |
|--------|-------|
| DiD Coefficient | 0.0444 |
| Standard Error | 0.0076 |
| 95% CI | [0.0295, 0.0593] |
| p-value | < 0.001 |
| Sample Size | 53,490 |

**Interpretation**: DACA eligibility increased full-time employment by approximately 4.4 percentage points among eligible Hispanic-Mexican, Mexican-born individuals.

### Sample Breakdown
| Group | N |
|-------|---|
| Treatment (ages 26-30) | 31,420 |
| Control (ages 31-35) | 22,070 |
| Pre-period (2006-2011) | 34,928 |
| Post-period (2013-2016) | 18,562 |

### Full-Time Employment Rates
|  | Pre (2006-2011) | Post (2013-2016) |
|--|-----------------|------------------|
| Control (31-35) | 64.4% | 62.0% |
| Treatment (26-30) | 61.9% | 63.4% |
| **Difference** | -2.5% | +1.4% |
| **DiD** | | **+3.9pp** |

### Robustness Checks
1. **Age Bandwidths**: Coefficients range from 0.033 to 0.044 for bandwidths ±3 to ±5 years
2. **By Sex**: Males (0.036***), Females (0.030**)
3. **Placebo Test**: Coefficient = 0.006 (p = 0.471) - no pre-trends
4. **Alternative Outcome (Any Employment)**: Coefficient = 0.026***

---

## Output Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `results_summary.csv` | Key numerical results |
| `event_study_results.csv` | Year-by-year coefficients |
| `summary_statistics.csv` | Descriptive statistics |
| `regression_output.txt` | Full regression tables |
| `figure1_event_study.png/pdf` | Event study plot |
| `figure2_parallel_trends.png/pdf` | Trends by group |
| `figure3_did_diagram.png/pdf` | DiD visualization |
| `figure4_bandwidth.png/pdf` | Bandwidth sensitivity |
| `replication_report_15.tex` | LaTeX source |
| `replication_report_15.pdf` | Final report (21 pages) |

---

## Variables Used (IPUMS Names)

| Variable | Description | Use |
|----------|-------------|-----|
| YEAR | Survey year | Time periods |
| HISPAN | Hispanic origin | Sample selection (=1 Mexican) |
| BPL | Birthplace | Sample selection (=200 Mexico) |
| CITIZEN | Citizenship status | Sample selection (=3 not citizen) |
| BIRTHYR | Birth year | Age calculation |
| BIRTHQTR | Birth quarter | Age adjustment |
| YRIMMIG | Year of immigration | DACA eligibility |
| UHRSWORK | Usual hours worked/week | Outcome (>=35 = full-time) |
| SEX | Sex | Covariate |
| MARST | Marital status | Covariate |
| EDUC | Education level | Covariate |
| STATEFIP | State FIPS code | Clustering, fixed effects |
| PERWT | Person weight | Survey weights |

---

## Session Info
- **Date**: 2025-01-25
- **Software**: Python 3.x with pandas, numpy, statsmodels, matplotlib
- **LaTeX**: pdfTeX (MiKTeX)
