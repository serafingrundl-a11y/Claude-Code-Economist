# DACA Replication Study - Run Log 04

## Project Overview
Independent replication analyzing the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
What was the causal impact of eligibility for DACA (treatment) on the probability that the eligible person is employed full-time (working 35+ hours/week)?

## Study Design
- **Treatment Group**: Ages 26-30 as of June 15, 2012 (DACA-eligible)
- **Control Group**: Ages 31-35 as of June 15, 2012 (ineligible due to age)
- **Method**: Difference-in-Differences
- **Pre-period**: 2006-2011 (using only 2006-2011 ACS years)
- **Post-period**: 2013-2016 (per instructions)
- **2012**: Excluded (DACA implemented mid-year, cannot distinguish pre/post)

## Session Start
- Date: 2025-01-25
- Data files located in ./data folder
- Main data: data.csv (ACS 2006-2016)
- Data dictionary: acs_data_dict.txt

---

## Key Decisions Log

### Decision 1: Sample Definition
**Criteria for DACA-like eligibility (applied to both treatment and control):**
1. Hispanic origin = Mexican (HISPAN = 1 or HISPAND in 100-107)
2. Birthplace = Mexico (BPL = 200 or BPLD = 20000)
3. Non-citizen without papers (CITIZEN = 3, meaning "Not a citizen")
4. Arrived in US before age 16 (calculated from YRIMMIG and BIRTHYR)
5. Arrived by 2007 (YRIMMIG <= 2007 and YRIMMIG > 0)
6. Present in US since June 15, 2007 (assumed if arrived by 2007)

**Age groups as of June 15, 2012:**
- Treatment: Birth years 1982-1986 (ages 26-30 in 2012)
- Control: Birth years 1977-1981 (ages 31-35 in 2012)

### Decision 2: Year of Immigration Coding
Per instructions, we cannot distinguish documented from undocumented non-citizens. We assume anyone who is not a citizen and who has not received immigration papers is undocumented. CITIZEN=3 ("Not a citizen") is our proxy.

### Decision 3: Outcome Variable
Full-time employment: UHRSWORK >= 35 hours per week (usual hours worked)

### Decision 4: Excluding 2012
DACA was implemented June 15, 2012. Since ACS doesn't record survey month, 2012 observations could be pre or post treatment. Excluding 2012 entirely.

### Decision 5: Model Specification
Preferred specification: OLS regression with year fixed effects and individual controls (gender, marital status, education categories).

---

## Analysis Steps

### Step 1: Data Loading and Initial Exploration
- Loaded data.csv with 33,851,424 observations
- ACS years 2006-2016 present
- Verified all required variables available

### Step 2: Sample Selection
Applied sequential filters:
1. Hispanic-Mexican ethnicity: 2,945,521 observations
2. Born in Mexico: 991,261 observations
3. Non-citizen: 701,347 observations
4. Arrived before age 16: 205,327 observations
5. Arrived by 2007: 195,023 observations
6. Birth year 1977-1986: 49,019 observations
7. Excluding 2012: **44,725 final observations**

### Step 3: Treatment/Control Assignment
- Treatment group (birth years 1982-1986): 26,591 observations
- Control group (birth years 1977-1981): 18,134 observations

### Step 4: Time Period Definition
- Pre-period (2006-2011): 29,326 observations
- Post-period (2013-2016): 15,399 observations

### Step 5: Outcome Variable Creation
- Full-time employment (UHRSWORK >= 35): 62.43% overall
- Treatment group: 61.90%
- Control group: 63.20%

### Step 6: Difference-in-Differences Analysis

**Simple DiD Calculation:**
| Group | Pre | Post | Change |
|-------|-----|------|--------|
| Treatment (26-30) | 0.611 | 0.634 | +0.023 |
| Control (31-35) | 0.643 | 0.611 | -0.032 |
| **DiD** | | | **+0.055** |

### Step 7: Regression Models

**Model 1: Basic DiD (No controls)**
- treat_post coefficient: 0.0551
- Standard error: 0.0098
- p-value: < 0.001

**Model 2: DiD with Year Fixed Effects**
- treat_post coefficient: 0.0554
- Standard error: 0.0098
- p-value: < 0.001

**Model 3: DiD with Year FE and Controls (PREFERRED)**
- treat_post coefficient: 0.0487
- Standard error: 0.0091
- 95% CI: [0.031, 0.067]
- p-value: < 0.001
- R-squared: 0.140
- N: 44,725

**Model 4: DiD with State + Year FE and Controls**
- treat_post coefficient: 0.0477
- Standard error: 0.0091
- R-squared: 0.144

### Step 8: Robustness Checks

| Specification | Coefficient | SE | 95% CI |
|--------------|-------------|-----|--------|
| Main (OLS) | 0.0487 | 0.0091 | [0.031, 0.067] |
| Weighted | 0.0480 | 0.0089 | [0.031, 0.065] |
| Clustered SE | 0.0487 | 0.0088 | [0.031, 0.066] |
| State + Year FE | 0.0477 | 0.0091 | [0.030, 0.066] |

### Step 9: Event Study / Parallel Trends Check

Pre-treatment coefficients (relative to 2011):
- 2006: -0.030 (p=0.101)
- 2007: -0.025 (p=0.174)
- 2008: 0.004 (p=0.852)
- 2009: -0.005 (p=0.803)
- 2010: -0.007 (p=0.691)

Post-treatment coefficients:
- 2013: 0.032 (p=0.099)
- 2014: 0.030 (p=0.132)
- 2015: 0.039 (p=0.054)
- 2016: 0.051 (p=0.011)

**Conclusion**: Pre-trends support parallel trends assumption; none significantly different from zero.

### Step 10: Heterogeneity Analysis

**By Gender:**
- Male: 0.060 (SE: 0.011), N=25,058
- Female: 0.037 (SE: 0.015), N=19,667

**By Education:**
- Less than HS: 0.028 (SE: 0.014), N=20,757
- High School: 0.052 (SE: 0.017), N=14,907
- Some College: 0.125 (SE: 0.024), N=7,620
- BA or Higher: 0.137 (SE: 0.052), N=1,441

---

## Outputs Generated

### Analysis Files
- `analysis.py`: Main analysis script
- `create_figures.py`: Figure generation script
- `analysis_sample.csv`: Cleaned analysis dataset

### Figures
- `figure1_trends.png/pdf`: Employment trends by treatment status
- `figure2_event_study.png/pdf`: Event study coefficients
- `figure3_did.png/pdf`: DiD visualization
- `figure4_heterogeneity_gender.png/pdf`: Gender heterogeneity
- `figure5_sample_size.png/pdf`: Sample size by year
- `figure6_model_comparison.png/pdf`: Model comparison

### Report
- `replication_report_04.tex`: LaTeX source (27 pages)
- `replication_report_04.pdf`: Compiled PDF report

---

## Final Results Summary

### Preferred Estimate
- **Effect size**: 0.0487 (4.87 percentage points)
- **Standard error**: 0.0091
- **95% Confidence Interval**: [0.0308, 0.0666]
- **t-statistic**: 5.34
- **p-value**: < 0.001
- **Sample size**: 44,725
- **Specification**: OLS with year fixed effects and individual controls

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 4.9 percentage points among Hispanic-Mexican, Mexican-born individuals who would have been eligible. This represents an 8% increase relative to the pre-treatment mean of 61.1%.

---

## Commands Run

```bash
# Data exploration
ls -la data/
head -5 data/data.csv

# Analysis
python analysis.py
python create_figures.py

# LaTeX compilation
pdflatex -interaction=nonstopmode replication_report_04.tex
pdflatex -interaction=nonstopmode replication_report_04.tex  # Second pass
pdflatex -interaction=nonstopmode replication_report_04.tex  # Third pass
pdflatex -interaction=nonstopmode replication_report_04.tex  # Fourth pass
```

---

## Session End
- All deliverables generated successfully
- Report: 27 pages
- Analysis fully reproducible via analysis.py
