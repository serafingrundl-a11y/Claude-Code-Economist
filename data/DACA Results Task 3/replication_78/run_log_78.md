# Run Log - DACA Replication Study (ID: 78)

## Project Overview
Replication study examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

## Session Start: 2026-01-27

---

## Step 1: Initial Setup and Data Exploration

### 1.1 Read Replication Instructions
- **Time**: Session start
- **Action**: Extracted and read replication_instructions.docx
- **Key findings**:
  - Research Question: Causal impact of DACA eligibility on full-time employment
  - Treatment group: Ages 26-30 at time of policy (June 15, 2012)
  - Control group: Ages 31-35 at time of policy
  - Outcome: Full-time employment (FT variable, 35+ hours/week)
  - Method: Difference-in-Differences (DiD)
  - Data: ACS 2008-2011 (pre) and 2013-2016 (post), excluding 2012
  - Pre-defined variables: ELIGIBLE (treatment indicator), FT (outcome), AFTER (post-period)

### 1.2 Data Files Identified
- `prepared_data_numeric_version.csv` - 17,382 observations
- `prepared_data_labelled_version.csv` - Same data with labels
- `acs_data_dict.txt` - Variable documentation

### 1.3 Key Variable Coding
- FT: 0 = Not full-time, 1 = Full-time (35+ hours/week)
- ELIGIBLE: 0 = Control (ages 31-35), 1 = Treatment (ages 26-30)
- AFTER: 0 = Pre-DACA (2008-2011), 1 = Post-DACA (2013-2016)
- PERWT: Person weights for survey-weighted estimates

---

## Step 2: Data Loading and Validation

### 2.1 Python Environment Setup
- Using Python with pandas, numpy, statsmodels, matplotlib

### 2.2 Data Loaded
- Command: `df = pd.read_csv('prepared_data_numeric_version.csv')`
- Total observations: 17,382
- Total variables: 105
- Years in data: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)

### 2.3 Sample Distribution
| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Control (31-35) | 3,294 | 2,706 | 6,000 |
| Treatment (26-30) | 6,233 | 5,149 | 11,382 |
| Total | 9,527 | 7,855 | 17,382 |

---

## Step 3: Main Analysis - Difference-in-Differences

### 3.1 Raw DiD Calculation (Weighted)
| Group | Pre-DACA | Post-DACA | Difference |
|-------|----------|-----------|------------|
| Control | 68.86% | 66.29% | -2.57 pp |
| Treatment | 63.69% | 68.60% | +4.91 pp |
| **DiD** | | | **+7.48 pp** |

### 3.2 Regression Models
All models use FT as dependent variable.

**Model 1: Unweighted OLS**
- DiD coefficient: 0.0643 (SE: 0.0153)
- p-value: < 0.0001

**Model 2: Weighted OLS with Robust SE (PREFERRED)**
- DiD coefficient: 0.0748 (SE: 0.0181)
- 95% CI: [0.0393, 0.1102]
- p-value: < 0.0001
- N: 17,382

**Model 3: With Demographics (Female, Married, FamSize, NChild)**
- DiD coefficient: 0.0636 (SE: 0.0168)
- p-value: 0.0001
- R-squared: 0.130

**Model 4: With Year Fixed Effects**
- DiD coefficient: 0.0608 (SE: 0.0167)
- p-value: 0.0003

**Model 5: With State and Year Fixed Effects**
- DiD coefficient: 0.0607 (SE: 0.0167)
- p-value: 0.0003
- R-squared: 0.137

**Model 6: Clustered SE by State**
- DiD coefficient: 0.0636 (SE: 0.0218)
- p-value: 0.0035

### 3.3 Key Decision: Preferred Specification
**Decision**: Selected Model 2 (Weighted OLS with robust SE) as preferred specification.
**Rationale**:
- Uses survey weights for nationally representative estimates
- Robust standard errors account for heteroskedasticity
- Avoids potential bias from controlling for post-treatment variables
- Provides clean estimate of average treatment effect
- Consistent with DiD best practices

---

## Step 4: Robustness Checks

### 4.1 Parallel Trends Assessment

**Placebo Test (Pre-treatment period only, 2010-11 vs 2008-09)**
- Coefficient: 0.0178
- SE: 0.0241
- p-value: 0.461
- **Interpretation**: Null finding supports parallel trends assumption

**Event Study Coefficients (Reference: 2011)**
| Year | Coefficient | SE | p-value |
|------|------------|-----|---------|
| 2008 | -0.0681 | 0.0351 | 0.052 |
| 2009 | -0.0499 | 0.0359 | 0.164 |
| 2010 | -0.0821 | 0.0357 | 0.021 |
| 2011 | 0.0000 | -- | -- |
| 2013 | 0.0158 | 0.0375 | 0.674 |
| 2014 | 0.0000 | 0.0384 | 1.000 |
| 2015 | 0.0014 | 0.0381 | 0.970 |
| 2016 | 0.0741 | 0.0384 | 0.053 |

### 4.2 Alternative Outcomes
| Outcome | DiD | SE | p-value |
|---------|-----|-----|---------|
| Full-time employment | 0.0748 | 0.0181 | <0.001 |
| Any employment | 0.0690 | 0.0163 | <0.001 |
| Labor force participation | 0.0521 | 0.0146 | 0.0003 |
| Usual hours worked | 3.066 | 0.672 | <0.001 |

---

## Step 5: Heterogeneity Analysis

### 5.1 By Gender
| Subgroup | DiD | SE | p-value | N |
|----------|-----|-----|---------|---|
| Male | 0.0716 | 0.0199 | 0.0003 | 9,075 |
| Female | 0.0527 | 0.0281 | 0.0611 | 8,307 |

### 5.2 By Education
| Subgroup | DiD | SE | p-value | N |
|----------|-----|-----|---------|---|
| High School | 0.0608 | 0.0214 | 0.0045 | 12,444 |
| Some College | 0.0672 | 0.0437 | 0.1241 | 2,877 |
| Two-Year Degree | 0.1816 | 0.0765 | 0.0176 | 991 |
| BA+ | 0.1619 | 0.0714 | 0.0233 | 1,058 |

### 5.3 By Marital Status
| Subgroup | DiD | SE | p-value | N |
|----------|-----|-----|---------|---|
| Married | 0.0673 | 0.0266 | 0.0113 | 7,851 |
| Not Married | 0.0888 | 0.0251 | 0.0004 | 9,531 |

---

## Step 6: Figures and Tables Generated

### 6.1 Figures Created
1. `figure1_trends.png` - Full-time employment trends by treatment status over time
2. `figure2_eventstudy.png` - Event study coefficients
3. `figure3_did.png` - DiD visualization with counterfactual
4. `figure4_gender.png` - Gender-specific trends
5. `figure5_forest.png` - Forest plot of estimates across specifications

### 6.2 Tables Generated
- LaTeX tables saved to `latex_tables.tex`
- Tables included: Descriptive statistics, DiD 2x2, Main regression results, Heterogeneity, Event study, Alternative outcomes

---

## Step 7: Report Generation

### 7.1 LaTeX Report
- File: `replication_report_78.tex`
- Compiled with pdflatex (3 passes for references)
- Output: `replication_report_78.pdf` (22 pages)

### 7.2 Report Structure
1. Abstract
2. Introduction
3. Background on DACA
4. Data and Sample
5. Empirical Methodology
6. Main Results
7. Robustness Checks
8. Heterogeneous Effects
9. Additional Outcomes
10. Discussion and Conclusion
11. Appendix A: Data and Methodology Notes
12. Appendix B: Additional Tables and Figures

---

## Summary of Key Results

### Preferred Estimate
- **Effect Size**: 0.0748 (7.48 percentage points)
- **Standard Error**: 0.0181
- **95% CI**: [0.0393, 0.1102]
- **p-value**: < 0.0001
- **Sample Size**: 17,382

### Interpretation
DACA eligibility increased full-time employment by approximately 7.5 percentage points among eligible individuals. This represents an 11.7% increase relative to the pre-DACA treatment group mean of 63.7%. The effect is statistically significant at the 1% level and robust across multiple specifications.

---

## Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `generate_tables.py` | LaTeX table generation script |
| `analysis_results.json` | Key results in JSON format |
| `latex_tables.tex` | LaTeX formatted tables |
| `figure1_trends.png` | Employment trends figure |
| `figure2_eventstudy.png` | Event study figure |
| `figure3_did.png` | DiD visualization |
| `figure4_gender.png` | Gender-specific trends |
| `figure5_forest.png` | Forest plot |
| `replication_report_78.tex` | Full LaTeX report |
| `replication_report_78.pdf` | Compiled PDF report |
| `run_log_78.md` | This run log |

---

## Session End: 2026-01-27

**Status**: COMPLETE

All deliverables generated:
- [x] replication_report_78.tex
- [x] replication_report_78.pdf
- [x] run_log_78.md
