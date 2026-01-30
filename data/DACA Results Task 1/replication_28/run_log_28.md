# DACA Replication Study - Run Log

## Study Information
- **Study ID:** 28
- **Date:** January 25, 2026
- **Research Question:** What was the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born non-citizens?

---

## 1. Data Exploration

### 1.1 Initial File Inventory
```bash
ls -la "C:/Users/seraf/DACA Results Task 1/replication_28/data/"
```
**Files found:**
- `data.csv` (6.26 GB) - Main ACS data file
- `acs_data_dict.txt` (121 KB) - Data dictionary
- `state_demo_policy.csv` (37 KB) - State-level policy data (optional)

### 1.2 Data Structure
```bash
head -5 "C:/Users/seraf/DACA Results Task 1/replication_28/data/data.csv"
```
**Result:** CSV with 54 columns including:
- Survey identifiers: YEAR, SAMPLE, SERIAL
- Demographics: AGE, SEX, BIRTHYR, BIRTHQTR, MARST
- Ethnicity/origin: HISPAN, HISPAND, BPL, BPLD, CITIZEN, YRIMMIG
- Employment: EMPSTAT, UHRSWORK, LABFORCE
- Other: EDUC, STATEFIP, etc.

---

## 2. Key Analytical Decisions

### 2.1 Sample Definition
**Decision:** Restrict to Hispanic-Mexican, Mexican-born, non-citizen population

**Variables used:**
- `HISPAN = 1` (Mexican Hispanic origin)
- `BPL = 200` (Born in Mexico)
- `CITIZEN = 3` (Not a citizen)
- `AGE >= 18 AND AGE <= 64` (Working-age population)

**Rationale:**
- The research question specifically asks about ethnically Hispanic-Mexican, Mexican-born individuals
- Non-citizen status is used as a proxy for undocumented status per the instructions
- Working-age restriction focuses on labor force-relevant population

### 2.2 Exclusion of 2012
**Decision:** Exclude survey year 2012

**Rationale:** DACA was implemented on June 15, 2012. The ACS does not identify month of data collection, so 2012 observations cannot be classified as pre- or post-DACA.

### 2.3 DACA Eligibility Definition
**Decision:** Construct DACA eligibility indicator based on the following criteria:

1. **Arrived before age 16:** `YRIMMIG - BIRTHYR < 16`
2. **Born after June 15, 1981:** `BIRTHYR >= 1981`
3. **Arrived by 2007:** `YRIMMIG <= 2007`
4. **Non-citizen:** Already imposed in sample restriction

**Rationale:** These operationalize the official DACA eligibility criteria using available ACS variables.

### 2.4 Outcome Variable
**Decision:** Full-time employment defined as `UHRSWORK >= 35`

**Rationale:**
- Standard BLS definition of full-time work is 35+ hours per week
- UHRSWORK captures "usual hours worked per week"
- Binary outcome allows for linear probability model interpretation

### 2.5 Identification Strategy
**Decision:** Difference-in-Differences (DiD) design

**Treatment Group:** DACA-eligible Mexican-born non-citizens
**Control Group:** Non-DACA-eligible Mexican-born non-citizens
**Pre-Period:** 2006-2011
**Post-Period:** 2013-2016

**Rationale:**
- DiD exploits variation in eligibility within the non-citizen population
- Controls for time-invariant differences between groups
- Controls for common time trends

### 2.6 Model Specifications
**Decision:** Progressive specification building

1. Basic DiD (no controls)
2. Add demographic controls (age, age², female, married, education)
3. Add year fixed effects
4. Add state fixed effects (preferred specification)
5. Clustered standard errors by state (robustness)

**Preferred Specification:** Model 4 with year and state fixed effects, robust standard errors

---

## 3. Analysis Commands

### 3.1 Main Analysis Script
```bash
cd "C:/Users/seraf/DACA Results Task 1/replication_28"
python analysis.py
```

**Key outputs:**
- `results_summary.csv` - Regression results across specifications
- `event_study_results.csv` - Event study coefficients
- `descriptive_stats.csv` - Summary statistics

### 3.2 Figure Generation
```bash
cd "C:/Users/seraf/DACA Results Task 1/replication_28"
python create_figures.py
```

**Figures created:**
- `figure1_event_study.pdf` - Event study plot
- `figure2_trends.pdf` - Employment trends by group
- `figure3_did_illustration.pdf` - DiD visual illustration
- `figure4_coefficient_comparison.pdf` - Comparison across specifications
- `figure5_heterogeneity.pdf` - Heterogeneity analysis

### 3.3 Report Compilation
```bash
cd "C:/Users/seraf/DACA Results Task 1/replication_28"
pdflatex -interaction=nonstopmode replication_report_28.tex
pdflatex -interaction=nonstopmode replication_report_28.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_28.tex  # Third pass
```

**Output:** `replication_report_28.pdf` (22 pages)

---

## 4. Main Results Summary

### 4.1 Sample Sizes
| Group | Pre-DACA (2006-2011) | Post-DACA (2013-2016) | Total |
|-------|---------------------|----------------------|-------|
| Not Eligible | 297,004 | 177,408 | 474,412 |
| DACA Eligible | 39,489 | 33,713 | 73,202 |
| **Total** | 336,493 | 211,121 | **547,614** |

### 4.2 Full-Time Employment Rates
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| DACA Eligible | 51.3% | 54.8% | +3.5 pp |
| Not Eligible | 60.5% | 58.1% | -2.3 pp |

### 4.3 Simple DiD Estimate
```
DiD = (0.548 - 0.513) - (0.581 - 0.605) = 0.058 (5.8 pp)
```

### 4.4 Preferred Estimate (Model 4)
- **Point Estimate:** 0.0169 (1.69 percentage points)
- **Standard Error:** 0.0037
- **95% CI:** [0.0096, 0.0242]
- **P-value:** < 0.001
- **N:** 547,614
- **R²:** 0.209

### 4.5 Robustness Checks
| Specification | DiD Estimate | SE | N |
|---------------|-------------|------|--------|
| Baseline (Model 4) | 0.0169 | 0.0037 | 547,614 |
| Clustered SE | 0.0169 | 0.0045 | 547,614 |
| Ages 18-35 | 0.0091 | 0.0042 | 253,373 |
| Arrived ≥1990 | 0.0244 | 0.0041 | 400,885 |
| Any employment | 0.0294 | 0.0037 | 547,614 |

### 4.6 Event Study Results
Pre-treatment coefficients (2006-2010) are small and not statistically significant, supporting parallel trends. Post-treatment effects emerge gradually:
- 2013: 0.005 (p = 0.51)
- 2014: 0.021 (p = 0.009)
- 2015: 0.038 (p < 0.001)
- 2016: 0.038 (p < 0.001)

### 4.7 Heterogeneity
- **By Gender:** Female (0.017) > Male (0.009)
- **By Education:** HS+ (0.022) > Less than HS (0.012)
- **By Age:** 25-34 (0.017), 18-24 (0.015), 35-64 (-0.021)

---

## 5. Files Produced

### 5.1 Required Deliverables
- `replication_report_28.tex` - LaTeX source
- `replication_report_28.pdf` - Final report (22 pages)
- `run_log_28.md` - This log file

### 5.2 Analysis Scripts
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script

### 5.3 Intermediate Outputs
- `results_summary.csv`
- `event_study_results.csv`
- `descriptive_stats.csv`
- `figure1_event_study.pdf/png`
- `figure2_trends.pdf/png`
- `figure3_did_illustration.pdf/png`
- `figure4_coefficient_comparison.pdf/png`
- `figure5_heterogeneity.pdf/png`

---

## 6. Software Environment
- **Python 3.x**
  - pandas
  - numpy
  - statsmodels
  - matplotlib
- **LaTeX:** MiKTeX (pdfTeX 3.141592653-2.6-1.40.28)

---

## 7. Key Interpretation

**Main Finding:** DACA eligibility increased the probability of full-time employment by approximately 1.7 percentage points among Mexican-born non-citizens. This effect is:
- Statistically significant (p < 0.001)
- Robust across specifications
- Supported by event study showing parallel pre-trends
- Concentrated among females and those with at least high school education

**Limitations:**
- Cannot verify true undocumented status (use non-citizen as proxy)
- Cannot verify continuous presence requirement
- Intent-to-treat effect (eligibility, not actual DACA receipt)
- Repeated cross-section, not panel data
