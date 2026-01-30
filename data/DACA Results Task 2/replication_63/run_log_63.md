# DACA Replication Study - Run Log 63

## Study Overview
**Research Question:** Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of being employed full-time (35+ hours/week)?

**Method:** Difference-in-Differences (DiD)
- Treatment group: Ages 26-30 at DACA implementation (June 15, 2012)
- Control group: Ages 31-35 at DACA implementation
- Pre-period: 2006-2011
- Post-period: 2013-2016
- Transition year 2012 excluded

---

## Session Log

### Step 1: Data Exploration
**Date/Time:** Session start

**Actions:**
1. Read replication instructions from `replication_instructions.docx`
2. Examined data folder contents:
   - `data.csv` (6.3 GB) - main ACS data file
   - `acs_data_dict.txt` - variable documentation
   - `state_demo_policy.csv` - supplementary state-level data (not used)

**Key Variable Identification:**
- YEAR: Survey year (2006-2016)
- HISPAN: Hispanic origin (1 = Mexican)
- BPL: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- BIRTHYR: Year of birth
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week
- PERWT: Person weight

---

### Step 2: Sample Construction
**File:** `analysis.py`

**DACA Eligibility Criteria Applied:**
1. Hispanic-Mexican ethnicity: HISPAN == 1
2. Born in Mexico: BPL == 200
3. Non-citizen: CITIZEN == 3
4. Age 26-35 at DACA implementation: BIRTHYR between 1977-1986
5. Arrived before age 16: (YRIMMIG - BIRTHYR) < 16
6. In US since 2007: YRIMMIG <= 2007

**Sample Flow:**
| Step | Filter | Observations |
|------|--------|-------------|
| 1 | Raw ACS data (2006-2016) | 33,851,424 |
| 2 | Hispanic-Mexican (HISPAN=1) | 2,945,521 |
| 3 | Born in Mexico (BPL=200) | 991,261 |
| 4 | Non-citizen (CITIZEN=3) | 701,347 |
| 5 | Ages 26-35 at DACA | 178,376 |
| 6 | Arrived before age 16 | 49,019 |
| 7 | In US since 2007 | 49,019 |
| 8 | Exclude 2012 | 44,725 |

**Final Sample:**
- Treatment (ages 26-30): 26,591 observations
- Control (ages 31-35): 18,134 observations
- Total: 44,725 observations

---

### Step 3: Variable Construction

**Outcome Variable:**
- `fulltime` = 1 if UHRSWORK >= 35, 0 otherwise
- Full-time employment rate in sample: 62.4%

**Treatment Variable:**
- `treated` = 1 if age at DACA (2012 - BIRTHYR) <= 30
- `post` = 1 if YEAR >= 2013
- `treated_post` = treated * post (DiD interaction)

**Control Variables:**
- `female` = 1 if SEX == 2
- `married` = 1 if MARST in {1, 2}
- `age_survey` = AGE at time of survey
- `age_sq` = age_survey^2
- `educ_less_hs` = 1 if EDUC < 6
- `educ_hs` = 1 if EDUC == 6
- `educ_some_college` = 1 if EDUC in {7, 8, 9}
- `educ_college` = 1 if EDUC >= 10

---

### Step 4: Analysis Decisions

**Key Methodological Choices:**

1. **Age Groups:**
   - Treatment: 26-30 at DACA (not under 26 to avoid education enrollment confounds)
   - Control: 31-35 at DACA (the closest ineligible group)

2. **Excluding 2012:**
   - DACA implemented June 15, 2012
   - ACS does not report month of interview
   - Cannot distinguish pre- vs post-DACA observations in 2012

3. **Weighting:**
   - Used ACS person weights (PERWT) for weighted analyses
   - Unweighted results shown for robustness

4. **Standard Errors:**
   - Preferred: State-clustered standard errors
   - Robustness: Robust (HC1) standard errors

5. **Fixed Effects:**
   - Year fixed effects included in preferred specification
   - State fixed effects tested for robustness

---

### Step 5: Main Results

**Simple DiD (Raw Means):**
|  | Pre-DACA | Post-DACA | Change |
|--|----------|-----------|--------|
| Treatment | 0.611 | 0.634 | +0.023 |
| Control | 0.643 | 0.611 | -0.032 |
| **DiD** | | | **+0.055** |

**Regression Results:**

| Specification | Coefficient | SE | 95% CI |
|--------------|-------------|-----|--------|
| Basic DiD (unweighted) | 0.0551*** | 0.0098 | [0.036, 0.074] |
| Weighted DiD | 0.0620*** | 0.0116 | [0.039, 0.085] |
| With demographics | 0.0657*** | 0.0149 | [0.037, 0.095] |
| With education | 0.0656*** | 0.0148 | [0.037, 0.095] |
| With year FE | 0.0185 | 0.0157 | [-0.012, 0.049] |
| **Preferred (clustered SE)** | **0.0185** | **0.0103** | **[-0.002, 0.039]** |

**Preferred Specification:**
- DiD Estimate: 0.0185 (1.85 percentage points)
- Standard Error: 0.0103 (state-clustered)
- 95% CI: [-0.0016, 0.0386]
- p-value: 0.071

---

### Step 6: Robustness Checks

**Event Study:**
- Pre-treatment coefficients (2006-2010) are small and not individually significant
- Provides support for parallel trends assumption
- Some variation in pre-treatment coefficients suggests imperfect parallel trends

**Heterogeneity:**
| Subgroup | DiD | SE |
|----------|-----|-----|
| Male | 0.0621*** | 0.0124 |
| Female | 0.0313* | 0.0182 |
| Less than HS | 0.0458** | 0.0179 |
| HS or more | 0.0743*** | 0.0152 |

---

### Step 7: Figure Generation
**File:** `generate_figures.py`

**Figures Created:**
1. `figure1_trends.png/pdf` - Employment trends over time by treatment status
2. `figure2_eventstudy.png/pdf` - Event study plot with confidence intervals
3. `figure3_did.png/pdf` - DiD visualization with counterfactual
4. `figure4_hours.png/pdf` - Distribution of hours worked
5. `figure5_gender.png/pdf` - Heterogeneity by gender
6. `figure6_age.png/pdf` - Age distribution of sample

---

### Step 8: Report Generation

**LaTeX Report:** `replication_report_63.tex`
- Compiled with pdflatex (3 passes for cross-references)
- Final PDF: 23 pages
- Includes all tables and figures

**PDF Output:** `replication_report_63.pdf`

---

## Commands Executed

```bash
# Data exploration
head -1 data/data.csv

# Run main analysis
python analysis.py

# Generate figures
python generate_figures.py

# Compile LaTeX
pdflatex -interaction=nonstopmode replication_report_63.tex
pdflatex -interaction=nonstopmode replication_report_63.tex  # 2nd pass
pdflatex -interaction=nonstopmode replication_report_63.tex  # 3rd pass
```

---

## Files Produced

| Filename | Description |
|----------|-------------|
| `analysis.py` | Main analysis script |
| `generate_figures.py` | Figure generation script |
| `replication_report_63.tex` | LaTeX source for report |
| `replication_report_63.pdf` | Final PDF report (23 pages) |
| `run_log_63.md` | This log file |
| `figure1_trends.png/pdf` | Employment trends figure |
| `figure2_eventstudy.png/pdf` | Event study figure |
| `figure3_did.png/pdf` | DiD visualization |
| `figure4_hours.png/pdf` | Hours distribution |
| `figure5_gender.png/pdf` | Gender heterogeneity |
| `figure6_age.png/pdf` | Age distribution |

---

## Summary of Findings

**Main Result:**
DACA eligibility is estimated to have increased full-time employment by 1.85 percentage points among eligible Hispanic-Mexican, Mexican-born non-citizens. This effect is not statistically significant at the 5% level (p = 0.071) but is marginally significant at the 10% level.

**Key Observations:**
1. The effect is larger for males (6.2 pp) than females (3.1 pp)
2. The effect is larger for those with higher education (7.4 pp) than those without (4.6 pp)
3. Event study evidence provides some support for parallel trends
4. Results are sensitive to the inclusion of year fixed effects, which substantially reduces the point estimate

**Limitations:**
1. Cannot distinguish documented vs undocumented non-citizens in ACS
2. Age-based identification may confound age-specific employment trends
3. Repeated cross-section prevents individual-level analysis
4. Some pre-treatment variation in event study coefficients

---

## Software Environment

- Python 3.x with pandas, numpy, statsmodels, matplotlib
- pdfLaTeX (MiKTeX distribution)
- Windows operating system

---

*Log completed at end of session.*
