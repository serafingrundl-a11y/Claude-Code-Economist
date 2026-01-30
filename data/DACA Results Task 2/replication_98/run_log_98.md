# Replication Run Log - Study 98

## Date: January 26, 2026

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (Deferred Action for Childhood Arrivals) on the probability of full-time employment (working 35+ hours per week)?

---

## Data Source
- American Community Survey (ACS) via IPUMS USA
- Years: 2006-2016 (one-year ACS files)
- Main data file: `data/data.csv`
- Data dictionary: `data/acs_data_dict.txt`

---

## Key Decisions and Methodology

### 1. Sample Selection Criteria

**Population Definition:**
- Hispanic-Mexican ethnicity: `HISPAN == 1`
- Born in Mexico: `BPL == 200`
- Non-citizen: `CITIZEN == 3`
- Per instructions: "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes"

**DACA Eligibility Criteria Applied:**
- Arrived in US before age 16: `YRIMMIG - BIRTHYR < 16`
- Arrived by 2007 (continuous residence since June 15, 2007): `YRIMMIG <= 2007`

**Treatment and Control Groups:**
- Treatment Group: Born 1982-1986 (ages 26-30 on June 15, 2012)
- Control Group: Born 1977-1981 (ages 31-35 on June 15, 2012)

**Rationale:** The age cutoff for DACA eligibility (under 31 on June 15, 2012) creates a natural experiment. Those just below the cutoff became eligible; those just above did not, despite otherwise meeting criteria.

### 2. Time Period Definition

- Pre-period: 2006-2011
- Post-period: 2013-2016
- **Excluded 2012**: DACA was implemented June 15, 2012 but ACS does not record month of interview, making it impossible to distinguish pre/post observations within that year.

### 3. Outcome Variable

- Full-time employment: `UHRSWORK >= 35`
- Binary indicator (1 = works 35+ hours per week, 0 otherwise)

### 4. Weighting

- All analyses use survey person weights (`PERWT`)
- Weighted least squares (WLS) regression

### 5. Model Specifications

Four models estimated with increasing controls:
1. Basic DiD: treat + post + treat*post
2. + Year fixed effects
3. + Demographic covariates (sex, marital status, education)
4. + State fixed effects (preferred specification)

### 6. Standard Errors

- Heteroskedasticity-robust (HC1) standard errors reported for inference
- Both conventional and robust SEs presented

---

## Commands Executed

### Data Processing
```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_98"
python analysis.py
```

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_98.tex
pdflatex -interaction=nonstopmode replication_report_98.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_98.tex  # Third pass for TOC
```

---

## Sample Sizes

| Stage | N |
|-------|---|
| Total ACS 2006-2016 | ~34 million |
| Hispanic-Mexican ethnicity | |
| Born in Mexico | |
| Non-citizens | |
| Birth years 1977-1986 | 178,376 |
| Arrived before age 16 | 49,019 |
| Arrived by 2007 | 49,019 |
| Excluding 2012 | **44,725** |

Final Sample by Group:
- Treatment (ages 26-30): 26,591
- Control (ages 31-35): 18,134

---

## Key Results

### Full-Time Employment Rates (Weighted)

|                    | Pre (2006-2011) | Post (2013-2016) | Change |
|--------------------|-----------------|------------------|--------|
| Control (31-35)    | 67.05%          | 64.12%           | -2.93 pp |
| Treatment (26-30)  | 62.53%          | 65.80%           | +3.27 pp |
| **Simple DiD**     |                 |                  | **+6.20 pp** |

### Regression Results

| Model | DiD Estimate | Robust SE | 95% CI | p-value |
|-------|--------------|-----------|--------|---------|
| (1) Basic DiD | 0.0620 | 0.0116 | [0.039, 0.085] | <0.001 |
| (2) + Year FE | 0.0610 | --- | [0.042, 0.080] | <0.001 |
| (3) + Covariates | 0.0467 | --- | [0.029, 0.064] | <0.001 |
| (4) + State FE | 0.0459 | 0.0105 | [0.025, 0.067] | <0.001 |

### Preferred Estimate

**Model 4 with Robust Standard Errors:**
- DiD Effect: **0.0459** (4.59 percentage points)
- Robust SE: 0.0105
- 95% CI: [0.0253, 0.0666]
- p-value: < 0.001

---

## Robustness Checks

### Pre-Trend Test
- Coefficient on (treat * year) in pre-period: 0.0034
- p-value: 0.395
- **Conclusion:** No evidence of differential pre-trends; supports parallel trends assumption

### Placebo Test (Fake Treatment in 2009)
- Placebo DiD: 0.0120
- p-value: 0.375
- **Conclusion:** No spurious effect at fake treatment timing; supports research design validity

### Event Study
- Pre-period coefficients (2006-2010 relative to 2011): All small and insignificant
- Post-period coefficients:
  - 2013: 0.0595 (p=0.023)
  - 2014: 0.0696 (p=0.009)
  - 2015: 0.0427 (p=0.108)
  - 2016: 0.0953 (p<0.001)
- **Pattern:** Effects emerge immediately post-DACA and grow over time

### Heterogeneity by Sex
- Male: DiD = 0.0621 (SE = 0.0124)
- Female: DiD = 0.0313 (SE = 0.0182)
- Effect appears larger for men but difference not statistically significant

---

## IPUMS Variables Used

| Variable | Description | Use in Analysis |
|----------|-------------|-----------------|
| YEAR | Survey year | Time period definition |
| HISPAN | Hispanic origin | Sample selection (=1 for Mexican) |
| BPL | Birthplace | Sample selection (=200 for Mexico) |
| CITIZEN | Citizenship status | Sample selection (=3 for non-citizen) |
| BIRTHYR | Birth year | Treatment/control group definition |
| YRIMMIG | Year of immigration | Eligibility check (arrived before 16, by 2007) |
| UHRSWORK | Usual hours worked per week | Outcome (>=35 for full-time) |
| PERWT | Person weight | Survey weighting |
| SEX | Sex | Covariate |
| MARST | Marital status | Covariate |
| EDUC | Education | Covariate |
| STATEFIP | State FIPS code | State fixed effects |

---

## Output Files

1. **replication_report_98.tex** - LaTeX source file
2. **replication_report_98.pdf** - Final PDF report (18 pages)
3. **run_log_98.md** - This log file
4. **analysis.py** - Python analysis script
5. **results_summary.txt** - Text summary of key results
6. **event_study_coefs.csv** - Event study coefficients
7. **yearly_employment_rates.csv** - Year-by-year employment rates

---

## Interpretation

The analysis finds that DACA eligibility increased full-time employment by approximately 4.6 percentage points among eligible Hispanic-Mexican individuals born in Mexico. This effect is statistically significant and robust to various specifications including year fixed effects, state fixed effects, and demographic controls.

The finding is economically meaningful: given a baseline full-time employment rate of about 62.5% in the treatment group pre-DACA, a 4.6 percentage point increase represents a relative increase of approximately 7.3%.

The parallel trends assumption is supported by:
1. Non-significant pre-trend test
2. Near-zero and insignificant event study coefficients in pre-period years
3. Non-significant placebo test

---

## Software Used

- Python 3.x with:
  - pandas
  - numpy
  - statsmodels
  - scipy
- pdfLaTeX (MiKTeX distribution)

---

## End of Log
