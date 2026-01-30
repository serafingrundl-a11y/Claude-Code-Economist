# Run Log - DACA Replication Study 07

## Date: 2026-01-26

## Overview
This log documents the replication analysis examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

---

## Step 1: Understanding the Research Task

**Research Question:** Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability of full-time employment (working 35+ hours/week)?

**Key Design Elements:**
- Treatment group: DACA-eligible individuals aged 26-30 at the time policy went into effect (June 2012)
- Control group: Individuals aged 31-35 at the time policy went into effect (otherwise eligible but for age)
- Pre-treatment period: 2008-2011
- Post-treatment period: 2013-2016
- Note: 2012 data excluded (cannot determine if before/after treatment)
- Outcome variable: FT (full-time employment, 1 = 35+ hours/week)

**Methodology:** Difference-in-Differences (DID) estimation

---

## Step 2: Data Exploration

**Data files available:**
- `prepared_data_numeric_version.csv` - Main analysis dataset
- `prepared_data_labelled_version.csv` - Same data with labels
- `acs_data_dict.txt` - Data dictionary

**Sample size:** 17,382 observations

**Key Variables:**
- ELIGIBLE: 1 = treatment group (ages 26-30 in June 2012), 0 = control (ages 31-35)
- AFTER: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)
- FT: 1 = full-time employment, 0 = not full-time
- PERWT: Person weights (ACS sampling weights)
- Various demographic and state-level policy controls

**Important Coding Notes:**
- Binary variables from IPUMS: 1 = No, 2 = Yes
- Constructed binary variables (FT, AFTER, ELIGIBLE, state policies): 0 = No, 1 = Yes

---

## Step 3: Analysis Implementation

### Decision Log:

1. **Estimator Choice:** Standard DID regression with interaction term (ELIGIBLE × AFTER)
2. **Weights:** Use PERWT for population-representative estimates
3. **Standard Errors:** Clustered at state level (STATEFIP) to account for within-state correlation and policy variation
4. **Covariates:** Include demographic controls (sex, marital status, children, education) and assess robustness

---

## Step 4: Data Summary

### Sample Distribution
| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Treatment (ELIGIBLE=1) | 6,233 | 5,149 | 11,382 |
| Control (ELIGIBLE=0) | 3,294 | 2,706 | 6,000 |
| **Total** | **9,527** | **7,855** | **17,382** |

### Observations by Year
| Year | Treatment | Control | Total |
|------|-----------|---------|-------|
| 2008 | 1,435 | 919 | 2,354 |
| 2009 | 1,480 | 899 | 2,379 |
| 2010 | 1,679 | 765 | 2,444 |
| 2011 | 1,639 | 711 | 2,350 |
| 2013 | 1,408 | 716 | 2,124 |
| 2014 | 1,348 | 708 | 2,056 |
| 2015 | 1,204 | 646 | 1,850 |
| 2016 | 1,189 | 636 | 1,825 |

---

## Step 5: Main Results

### Simple DID Calculation (Unweighted)
- Treatment Pre-DACA FT Rate: 0.6263
- Treatment Post-DACA FT Rate: 0.6658
- Treatment Change: +0.0394

- Control Pre-DACA FT Rate: 0.6697
- Control Post-DACA FT Rate: 0.6449
- Control Change: -0.0248

**Simple DID Estimate: 0.0643**

### Regression Results

| Model | DID Estimate | SE | p-value | Notes |
|-------|--------------|-----|---------|-------|
| (1) OLS Basic | 0.0643 | 0.0153 | <0.001 | Unweighted |
| (2) WLS Basic | 0.0748 | 0.0152 | <0.001 | Weighted with PERWT |
| (3) OLS Clustered | 0.0643 | 0.0141 | <0.001 | State-clustered SE |
| **(4) WLS Clustered** | **0.0748** | **0.0203** | **<0.001** | **Preferred specification** |
| (5) With Demographics | 0.0614 | 0.0212 | 0.004 | +Sex, marital, children, education |
| (6) State FE | 0.0609 | 0.0217 | 0.005 | +State fixed effects |
| (7) Year FE | 0.0585 | 0.0205 | 0.004 | Year fixed effects instead of AFTER |

### Preferred Estimate (Model 4)
- **DID Coefficient: 0.0748**
- **Standard Error: 0.0203**
- **95% CI: [0.0350, 0.1145]**
- **p-value: 0.0002**
- **Sample Size: 17,382**

**Interpretation:** DACA eligibility increased the probability of full-time employment by approximately 7.5 percentage points among eligible Hispanic-Mexican, Mexican-born individuals.

---

## Step 6: Subgroup Analysis

### By Sex
| Subgroup | DID Estimate | SE | 95% CI | p-value | N |
|----------|--------------|-----|--------|---------|---|
| Full Sample | 0.0748 | 0.0203 | [0.035, 0.114] | 0.0002 | 17,382 |
| Males | 0.0716 | 0.0195 | [0.033, 0.110] | 0.0002 | 9,075 |
| Females | 0.0527 | 0.0290 | [-0.004, 0.110] | 0.0696 | 8,307 |

The effect appears stronger and more precisely estimated for males.

---

## Step 7: Key Decisions and Justifications

### 1. Why Difference-in-Differences?
The DID design exploits the age-based eligibility cutoff for DACA. Individuals aged 26-30 in June 2012 were eligible while those 31-35 were not (due to the age 31 cutoff). This creates a natural comparison group.

### 2. Why State-Clustered Standard Errors?
- Individuals within states share common policy environments
- State-level policies (driver's licenses, in-state tuition) vary
- Clustering accounts for within-state correlation of errors

### 3. Why Use Weights?
ACS person weights (PERWT) make estimates representative of the target population. Weighted estimates are preferred for policy-relevant inference.

### 4. Why Include Those Not in Labor Force?
Per instructions, the FT variable includes individuals not in the labor force as zeros. This measures full-time employment among all eligible individuals, not just those in the labor force.

### 5. Why Not Further Limit the Sample?
Instructions explicitly state: "This entire file constitutes the intended analytic sample for your analysis; do not further limit the sample by dropping individuals on the basis of their characteristics."

---

## Step 8: Files Generated

### Analysis Code
- `analysis.py` - Main Python analysis script

### Visualization Code
- `create_figures.py` - Figure generation script

### Output Files
- `regression_results.csv` - Summary of all regression results
- `trends_data.csv` - Year-by-year trends data

### Figures
- `figure1_parallel_trends.pdf/png` - Employment trends by group
- `figure2_did_diagram.pdf/png` - DID visualization
- `figure3_coefficient_plot.pdf/png` - Coefficient estimates across models
- `figure4_subgroup_analysis.pdf/png` - Subgroup analysis by sex
- `figure5_sample_distribution.pdf/png` - Sample characteristics

### Report
- `replication_report_07.tex` - LaTeX source
- `replication_report_07.pdf` - Final report (19 pages)

---

## Step 9: Commands Executed

```bash
# Data exploration
python -c "import pandas as pd; df = pd.read_csv('data/prepared_data_numeric_version.csv'); print(df.shape)"

# Run main analysis
python analysis.py

# Create figures
python create_figures.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_07.tex
pdflatex -interaction=nonstopmode replication_report_07.tex
pdflatex -interaction=nonstopmode replication_report_07.tex
```

---

## Step 10: Conclusion

This independent replication finds that DACA eligibility had a statistically significant positive effect on full-time employment among Hispanic-Mexican, Mexican-born individuals. The preferred estimate suggests a 7.48 percentage point increase in the probability of full-time employment (95% CI: 3.5 to 11.4 percentage points). This effect is robust across multiple specifications and represents approximately an 11-12% relative increase from the pre-DACA baseline employment rate of the treatment group.

---

## Deliverables Checklist

- [x] `replication_report_07.tex` - LaTeX source file
- [x] `replication_report_07.pdf` - Compiled PDF report (19 pages)
- [x] `run_log_07.md` - This run log

---

*Log completed: 2026-01-26*
