# Run Log - DACA Employment Replication Study (Run 30)

## Overview
This log documents the replication analysis examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (working 35+ hours per week)?

## Identification Strategy
- **Treatment Group**: DACA-eligible individuals aged 26-30 at the time of policy implementation (June 2012)
- **Control Group**: Individuals aged 31-35 at the time of policy implementation who would have been eligible but for their age
- **Method**: Difference-in-Differences (DiD) comparing pre-post changes between treatment and control groups
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016 (2012 excluded as implementation year)

---

## Session Log

### Step 1: Initial Setup and Data Exploration
**Timestamp**: Session start

**Actions**:
1. Read replication instructions from `replication_instructions.docx`
2. Examined data directory structure
3. Reviewed data dictionary (`acs_data_dict.txt`)
4. Identified key variables:
   - `FT`: Full-time employment (outcome, 0/1)
   - `ELIGIBLE`: DACA eligibility indicator (treatment group identifier, 0/1)
   - `AFTER`: Post-DACA indicator (2013-2016 = 1, 2008-2011 = 0)
   - `YEAR`: Survey year
   - `PERWT`: Person weight for survey estimation

**Data Files**:
- `prepared_data_labelled_version.csv`: Contains labeled categorical variables
- `prepared_data_numeric_version.csv`: Contains numeric coded variables

**Key Design Decisions**:
1. Use provided `ELIGIBLE`, `FT`, and `AFTER` variables as instructed
2. Do not create alternative eligibility definitions
3. Keep entire sample (do not drop observations)
4. Use person weights (`PERWT`) for population-representative estimates

---

### Step 2: Data Loading and Descriptive Statistics

**Sample Summary**:
- Total observations: 17,382
- Years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-DACA period: 9,527 observations
- Post-DACA period: 7,855 observations

**Sample by Group and Period**:
| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Control (31-35) | 3,294 | 2,706 | 6,000 |
| Treatment (26-30) | 6,233 | 5,149 | 11,382 |

**Demographics**:
- Mean age: 29.62 years
- Sex: 52.2% Male, 47.8% Female
- Overall FT employment rate: 64.9%

---

### Step 3: Full-Time Employment Rates

**Weighted FT Employment Rates by Group and Period**:
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 63.7% | 68.6% | +4.9 pp |
| Control (31-35) | 68.9% | 66.3% | -2.6 pp |

**Simple DiD Calculation**:
- Treatment change: +4.9 pp
- Control change: -2.6 pp
- DiD estimate: 4.9 - (-2.6) = **7.5 percentage points**

---

### Step 4: Regression Analysis

**Models Estimated**:

| Model | Specification | DiD Coef | SE | p-value |
|-------|--------------|----------|-----|---------|
| M1 | Basic OLS | 0.064 | 0.015 | <0.001 |
| M2 | OLS robust SE | 0.064 | 0.015 | <0.001 |
| M3 | WLS | 0.075 | 0.015 | <0.001 |
| M4 | WLS robust SE | 0.075 | 0.018 | <0.001 |
| M5 | + Demographics | 0.065 | 0.017 | <0.001 |
| M6 | + Education | 0.065 | 0.017 | <0.001 |
| M7 | Year FE | 0.072 | 0.018 | <0.001 |
| M8 | Year FE + Demo | 0.062 | 0.017 | <0.001 |
| M9 | State FE | 0.074 | 0.018 | <0.001 |
| M10 | Full Model | 0.061 | 0.017 | <0.001 |

**Preferred Specification (M10)**:
- Includes: Year FE, State FE, Demographics (age, sex, marital, education)
- DiD Coefficient: **0.061**
- Standard Error: 0.017
- 95% CI: [0.029, 0.094]
- p-value: <0.001
- N: 17,382
- R-squared: 0.134

---

### Step 5: Robustness Checks

**Parallel Trends Test**:
- Tested for differential pre-trends using pre-DACA data only
- Differential trend coefficient: 0.017
- p-value: 0.113
- **Conclusion**: No evidence of differential pre-trends (parallel trends assumption supported)

**Placebo Test**:
- Artificial treatment at 2010 using only pre-DACA data
- Placebo DiD: 0.018
- p-value: 0.461
- 95% CI: [-0.030, 0.065]
- **Conclusion**: No significant placebo effect (supports validity)

**Event Study Results**:
| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2008 | -0.068 | 0.035 | 0.052 |
| 2009 | -0.050 | 0.036 | 0.164 |
| 2010 | -0.082 | 0.036 | 0.021 |
| 2011 | 0.000 | (ref) | --- |
| 2013 | 0.016 | 0.038 | 0.674 |
| 2014 | 0.000 | 0.038 | 1.000 |
| 2015 | 0.001 | 0.038 | 0.970 |
| 2016 | 0.074 | 0.038 | 0.053 |

**Conclusion**: Pre-treatment coefficients mostly insignificant, supporting parallel trends. Effects appear to grow over time post-DACA.

---

### Step 6: Subgroup Analysis

| Subgroup | DiD | SE | p-value | N |
|----------|-----|-----|---------|---|
| Male | 0.072 | 0.020 | <0.001 | 9,075 |
| Female | 0.053 | 0.028 | 0.061 | 8,307 |
| Married | 0.057 | 0.026 | 0.025 | 8,524 |
| Unmarried | 0.098 | 0.026 | <0.001 | 8,858 |

**Key Findings**:
- Effect stronger for males than females
- Effect stronger for unmarried than married individuals

---

### Step 7: State Policy Interactions

| Policy | Base Effect | Interaction | Int. p-value |
|--------|-------------|-------------|--------------|
| Driver's License | 0.092 | -0.032 | 0.195 |
| E-Verify | 0.065 | 0.023 | 0.123 |

**Conclusion**: No significant heterogeneity by state policy environment.

---

### Step 8: Visualization and Report Generation

**Figures Created**:
1. `figure1_trends.png/pdf` - FT employment trends by group
2. `figure2_difference.png/pdf` - Treatment-control difference by year
3. `figure3_eventstudy.png/pdf` - Event study coefficients
4. `figure4_did.png/pdf` - DiD visualization
5. `figure5_subgroups.png/pdf` - Subgroup forest plot
6. `figure6_models.png/pdf` - Model comparison

**Report**:
- LaTeX document: `replication_report_30.tex`
- PDF output: `replication_report_30.pdf` (23 pages)

---

## Key Analytical Decisions

1. **Sample**: Used full provided sample without additional restrictions
2. **Outcome**: Used provided FT variable (35+ hours/week)
3. **Treatment Definition**: Used provided ELIGIBLE variable
4. **Weighting**: All regressions use PERWT (person weights)
5. **Standard Errors**: Heteroskedasticity-robust (HC1)
6. **Preferred Specification**: Full model with state FE, year FE, and demographics

---

## Final Results Summary

### Preferred Estimate
| Metric | Value |
|--------|-------|
| Effect Size | 0.061 (6.1 percentage points) |
| Standard Error | 0.017 |
| 95% CI | [0.029, 0.094] |
| p-value | <0.001 |
| Sample Size | 17,382 |

### Interpretation
DACA eligibility is estimated to have increased the probability of full-time employment by approximately **6.1 percentage points** among Hispanic-Mexican, Mexican-born individuals aged 26-30 (compared to those aged 31-35). This represents roughly a 10% relative increase from the baseline employment rate. The effect is statistically significant and robust across specifications.

---

## Files Produced

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Visualization script |
| `model_summary.csv` | Summary of all model results |
| `event_study.csv` | Event study coefficients |
| `ft_rates_by_year.csv` | FT rates by year and group |
| `figure1_trends.png/pdf` | Employment trends figure |
| `figure2_difference.png/pdf` | Difference by year figure |
| `figure3_eventstudy.png/pdf` | Event study figure |
| `figure4_did.png/pdf` | DiD visualization |
| `figure5_subgroups.png/pdf` | Subgroup analysis figure |
| `figure6_models.png/pdf` | Model comparison figure |
| `replication_report_30.tex` | LaTeX report source |
| `replication_report_30.pdf` | Final PDF report |
| `run_log_30.md` | This run log |

---

## Session Complete
All deliverables have been produced successfully.
