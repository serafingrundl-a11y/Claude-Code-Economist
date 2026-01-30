# Run Log - DACA Replication Study (ID: 62)

## Date: January 26, 2026

---

## 1. Overview

This log documents all commands executed and key decisions made during the independent replication of the DACA full-time employment study.

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

**Research Design:** Difference-in-Differences
- **Treatment Group:** Ages 26-30 as of June 15, 2012 (born 1982-1986)
- **Control Group:** Ages 31-35 as of June 15, 2012 (born 1977-1981)

---

## 2. Data Exploration

### 2.1 Initial Data Inspection

```python
# Load data and check structure
import pandas as pd
df = pd.read_csv('data/data.csv')
print(f'Shape: {df.shape}')
# Output: Shape: (33,851,424, 54)

print(f'Years: {sorted(df["YEAR"].unique())}')
# Output: [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
```

### 2.2 Key Variables Identified

From `acs_data_dict.txt`:
- `YEAR`: Survey year (2006-2016)
- `HISPAN`: Hispanic origin (1 = Mexican)
- `BPL`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Year of birth
- `UHRSWORK`: Usual hours worked per week
- `PERWT`: Person weight for population estimates

---

## 3. Key Analytical Decisions

### Decision 1: Definition of DACA Eligibility

**Criteria Applied:**
1. Hispanic-Mexican ethnicity: `HISPAN == 1`
2. Born in Mexico: `BPL == 200`
3. Non-citizen: `CITIZEN == 3`
4. Immigrated before age 16: `YRIMMIG - BIRTHYR < 16`
5. Continuous residence since June 2007: `YRIMMIG <= 2007`

**Rationale:** These criteria follow the DACA eligibility requirements that can be identified in the ACS data. The assumption that non-citizens are undocumented follows the instructions, as the ACS cannot distinguish between documented and undocumented non-citizens.

### Decision 2: Treatment and Control Group Ages

**Treatment Group:** Birth years 1982-1986 (ages 26-30 as of June 15, 2012)
**Control Group:** Birth years 1977-1981 (ages 31-35 as of June 15, 2012)

**Rationale:** Following the research design specification. These age bands provide sufficient sample size while maintaining comparability between groups.

### Decision 3: Time Period Definition

**Pre-period:** 2006-2011
**Post-period:** 2013-2016
**Excluded:** 2012 (transition year - DACA implemented mid-year)

**Rationale:** Excluding 2012 from the main analysis creates clean pre/post periods since the ACS does not record month of survey administration, making it impossible to distinguish pre- and post-DACA observations in 2012.

### Decision 4: Outcome Variable Definition

**Full-time employment:** `UHRSWORK >= 35`

**Rationale:** Standard definition of full-time work (35+ hours per week) as specified in the research question.

### Decision 5: Model Specification

**Preferred specification (Model 4):**
```
fulltime ~ treated + post + treated*post + female + married + year_FE + state_FE
```

**Rationale:**
- Year fixed effects control for common time trends
- State fixed effects control for geographic variation in labor markets
- Demographic controls (female, married) improve precision and address compositional differences
- Did not include education as it may be endogenous (affected by DACA)

### Decision 6: Sample Weights

**Approach:** Primary analysis is unweighted; weighted analysis included as robustness check.

**Rationale:** Both approaches yield nearly identical results. Unweighted regression is preferred for the main specification for simplicity and interpretability, with weighted results confirming robustness.

---

## 4. Sample Construction

| Step | Criterion | Observations |
|------|-----------|--------------|
| 1 | Full ACS sample (2006-2016) | 33,851,424 |
| 2 | Hispanic-Mexican born in Mexico | 991,261 |
| 3 | Non-citizens | 701,347 |
| 4 | Immigrated before age 16 | 205,327 |
| 5 | Continuous residence since 2007 | 195,023 |
| 6 | Treatment or control age group | 49,019 |
| 7 | Excluding 2012 (regression sample) | 44,725 |

---

## 5. Analysis Commands

### 5.1 Main Analysis Script

```python
# Full analysis script: analysis.py
python analysis.py
```

**Output files generated:**
- `results_summary.csv`: Key regression results
- `table1_summary_stats.csv`: Summary statistics
- `table2_main_results.csv`: Main regression results
- `event_study_results.csv`: Event study coefficients
- `figure1_event_study.png/pdf`: Event study figure
- `figure2_trends.png/pdf`: Trends by treatment status

### 5.2 Key Python Commands

```python
# Filter for eligible population
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)]
df_mex = df_mex[df_mex['CITIZEN'] == 3]
df_mex['age_at_immigration'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
df_mex = df_mex[df_mex['age_at_immigration'] < 16]
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007]

# Define treatment/control groups
df_mex['treated'] = ((df_mex['BIRTHYR'] >= 1982) & (df_mex['BIRTHYR'] <= 1986)).astype(int)
df_mex['control'] = ((df_mex['BIRTHYR'] >= 1977) & (df_mex['BIRTHYR'] <= 1981)).astype(int)

# Define outcome
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Main regression
import statsmodels.formula.api as smf
model4 = smf.ols('fulltime ~ treated + post + treated_post + female + married + C(year_factor) + C(STATEFIP)',
                 data=df_reg).fit()
```

---

## 6. Results Summary

### 6.1 Raw Difference-in-Differences

| Group | Pre-period | Post-period | Difference |
|-------|------------|-------------|------------|
| Treatment | 0.608 | 0.634 | +0.026 |
| Control | 0.638 | 0.611 | -0.027 |
| **DiD** | | | **0.053** |

### 6.2 Regression Results

| Model | DiD Estimate | Std. Error | p-value | R-squared |
|-------|--------------|------------|---------|-----------|
| (1) Basic DiD | 0.0551 | 0.0098 | <0.001 | 0.001 |
| (2) Year FE | 0.0554 | 0.0098 | <0.001 | 0.004 |
| (3) + Covariates | 0.0488 | 0.0091 | <0.001 | 0.134 |
| (4) + State FE | 0.0480 | 0.0091 | <0.001 | 0.137 |
| (5) Weighted | 0.0481 | 0.0089 | <0.001 | 0.155 |

### 6.3 Preferred Estimate

- **Effect Size:** 0.0480 (4.80 percentage points)
- **Standard Error:** 0.0091
- **95% Confidence Interval:** [0.030, 0.066]
- **p-value:** < 0.001
- **Sample Size:** 44,725

### 6.4 Robustness Checks

1. **Narrower age bandwidth (27-29 vs 32-34):** DiD = 0.049 (SE = 0.012), p < 0.001
2. **Placebo test (pre-DACA):** DiD = 0.016 (SE = 0.011), p = 0.136 (not significant)
3. **Event study:** Pre-treatment coefficients not significant; effects emerge post-2012

### 6.5 Heterogeneity

- **Males:** DiD = 0.043 (SE = 0.011), p < 0.001
- **Females:** DiD = 0.045 (SE = 0.015), p = 0.003
- **Low education:** DiD = 0.021 (SE = 0.013), p = 0.101
- **High education:** DiD = 0.076 (SE = 0.013), p < 0.001

---

## 7. LaTeX Report Compilation

```bash
# Compile LaTeX document (3 passes for cross-references)
pdflatex -interaction=nonstopmode replication_report_62.tex
pdflatex -interaction=nonstopmode replication_report_62.tex
pdflatex -interaction=nonstopmode replication_report_62.tex
```

**Output:** `replication_report_62.pdf` (19 pages)

---

## 8. Files Produced

| Filename | Description |
|----------|-------------|
| `analysis.py` | Main Python analysis script |
| `replication_report_62.tex` | LaTeX source for report |
| `replication_report_62.pdf` | Final PDF report |
| `run_log_62.md` | This run log |
| `results_summary.csv` | Key results summary |
| `table1_summary_stats.csv` | Summary statistics |
| `table2_main_results.csv` | Main regression results |
| `event_study_results.csv` | Event study coefficients |
| `figure1_event_study.png/pdf` | Event study plot |
| `figure2_trends.png/pdf` | Trends by treatment status |

---

## 9. Interpretation

The preferred estimate of 4.80 percentage points suggests that DACA eligibility increased the probability of full-time employment by approximately 8% relative to the baseline rate (61%) among eligible Hispanic-Mexican individuals born in Mexico. This effect is:

1. **Statistically significant** (p < 0.001)
2. **Robust** across specifications and bandwidth choices
3. **Supported by parallel trends** evidence (placebo test insignificant, pre-trend event study coefficients near zero)
4. **Larger for higher-educated individuals**, consistent with DACA enabling formal sector employment

The findings suggest that DACA's work authorization provision had meaningful positive effects on full-time employment among eligible individuals.

---

## 10. Caveats and Limitations

1. Cannot distinguish documented from undocumented non-citizens in ACS
2. Intent-to-treat estimate (not all eligible individuals received DACA)
3. Parallel trends assumption ultimately untestable
4. Possible spillovers to control group
5. Potential selection into survey response

---

*End of Run Log*
