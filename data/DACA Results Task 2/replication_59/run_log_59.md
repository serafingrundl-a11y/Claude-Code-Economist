# Run Log - DACA Replication Study (ID: 59)

## Study Overview
- **Research Question**: What was the causal impact of DACA eligibility on full-time employment (>=35 hours/week) among Hispanic-Mexican, Mexican-born individuals?
- **Treatment Group**: Ages 26-30 at time of DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at time of DACA implementation
- **Method**: Difference-in-Differences
- **Post-treatment Period**: 2013-2016

---

## Session Log

### Step 1: Data Exploration
**Time**: Session start

**Actions**:
- Read replication_instructions.docx to understand the research design
- Examined acs_data_dict.txt for variable definitions
- Inspected data.csv structure (headers and sample rows)

**Data Source**: ACS 2006-2016 (1-year files from IPUMS)

**Key Variables Identified**:
| Variable | Description | Values Used |
|----------|-------------|-------------|
| YEAR | Survey year | 2006-2011 (pre), 2013-2016 (post) |
| BIRTHYR | Birth year | 1977-1986 |
| HISPAN | Hispanic origin | 1 = Mexican |
| BPL | Birthplace | 200 = Mexico |
| CITIZEN | Citizenship status | 3 = Not a citizen |
| YRIMMIG | Year of immigration | <= 2007 |
| UHRSWORK | Usual hours worked per week | >= 35 = full-time |
| PERWT | Person weight | Used for weighted estimates |
| SEX | Sex | 1 = Male, 2 = Female |
| MARST | Marital status | 1 = Married with spouse present |
| EDUC | Education level | Categorical |
| STATEFIP | State FIPS code | State fixed effects |

---

### Step 2: Sample Construction

**DACA Eligibility Criteria Applied**:
1. Hispanic-Mexican ethnicity: HISPAN = 1
2. Born in Mexico: BPL = 200
3. Non-citizen (proxy for undocumented): CITIZEN = 3
4. Arrived before age 16: YRIMMIG - BIRTHYR < 16
5. Continuous presence since 2007: YRIMMIG <= 2007
6. Age groups based on birth year:
   - Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)
   - Control: Born 1977-1981 (ages 31-35 on June 15, 2012)
7. Exclude 2012: YEAR != 2012 (DACA implemented mid-year)

**Sample Flow**:
| Step | N |
|------|---|
| Mexican-born Hispanic-Mexican individuals | 991,261 |
| After citizenship filter (non-citizen) | 701,347 |
| After arrival age filter (<16) | 188,195 |
| After continuous presence filter (<=2007) | 178,934 |
| After age group filter (born 1977-1986) | 46,669 |
| After excluding 2012 | **42,558** |

**Final Sample Breakdown**:
| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Treatment (ages 26-30) | 16,605 | 8,796 | 25,401 |
| Control (ages 31-35) | 11,267 | 5,890 | 17,157 |
| **Total** | 27,872 | 14,686 | **42,558** |

---

### Step 3: Variable Construction

**Outcome Variable**:
```
fulltime = 1 if UHRSWORK >= 35 else 0
```

**Treatment Indicators**:
```
treatment = 1 if BIRTHYR in [1982, 1986] else 0
post = 1 if YEAR >= 2013 else 0
treat_post = treatment * post  # DiD interaction term
```

**Control Variables**:
```
female = 1 if SEX == 2 else 0
married = 1 if MARST == 1 else 0
# Plus categorical education (EDUC) and state FE (STATEFIP)
```

---

### Step 4: Analysis Execution

**Command Executed**:
```bash
python analysis.py
```

**Analysis Script**: analysis.py
- Language: Python 3.14
- Libraries: pandas, numpy, statsmodels

**Models Estimated**:

| Model | Description | Coefficient | SE |
|-------|-------------|-------------|-----|
| 1 | Basic DiD (unweighted) | 0.0578 | 0.0101 |
| 2 | Basic DiD (weighted) | 0.0635 | 0.0099 |
| 3 | + Covariates | 0.0509 | 0.0091 |
| 4 | + State FE | 0.0504 | 0.0091 |
| **5** | **+ Year FE + Robust SE (PREFERRED)** | **0.0490** | **0.0108** |

---

### Step 5: Main Results

**Preferred Estimate (Model 5)**:
- **Effect**: +0.0490 (4.90 percentage points)
- **Standard Error**: 0.0108 (robust/HC1)
- **t-statistic**: 4.54
- **p-value**: < 0.0001
- **95% CI**: [0.0279, 0.0702]
- **Sample Size**: 42,558

**Interpretation**: DACA eligibility increased the probability of full-time employment by 4.9 percentage points among Hispanic-Mexican, Mexican-born non-citizens. This represents approximately a 7.8% increase relative to the pre-treatment mean of 62.5%.

---

### Step 6: Robustness Checks

**6a. Alternative Age Bandwidth (born 1975-1988)**:
- Coefficient: 0.0356
- SE: 0.0104
- Sample: 48,870
- Result: Smaller but still significant effect

**6b. By Gender**:
- Male: 0.0451 (SE: 0.0126), n=23,872
- Female: 0.0363 (SE: 0.0182), n=18,686
- Result: Effect present for both genders; slightly larger for men

**6c. Placebo Test (fake treatment in 2010)**:
- Coefficient: -0.0010
- SE: 0.0139
- p-value: 0.9451
- Result: No pre-existing differential trends (supports parallel trends assumption)

**6d. Event Study Analysis**:
| Year | Coefficient | SE | Post-DACA |
|------|-------------|-----|-----------|
| 2006 | +0.009 | 0.023 | No |
| 2007 | -0.008 | 0.023 | No |
| 2008 | +0.017 | 0.023 | No |
| 2009 | +0.015 | 0.024 | No |
| 2010 | +0.013 | 0.024 | No |
| 2011 | (ref) | - | No |
| 2013 | +0.050* | 0.024 | Yes |
| 2014 | +0.051* | 0.025 | Yes |
| 2015 | +0.042 | 0.025 | Yes |
| 2016 | +0.086*** | 0.025 | Yes |

**Event Study Findings**:
- Pre-treatment coefficients are small and insignificant (supports parallel trends)
- Post-treatment coefficients are uniformly positive
- Effect grows over time, largest in 2016

---

### Step 7: Report Generation

**LaTeX Report Created**: replication_report_59.tex

**Compilation Commands**:
```bash
pdflatex -interaction=nonstopmode replication_report_59.tex  # First pass
pdflatex -interaction=nonstopmode replication_report_59.tex  # Second pass (references)
pdflatex -interaction=nonstopmode replication_report_59.tex  # Third pass (final)
```

**Output**: replication_report_59.pdf (20 pages)

---

## Key Decisions and Justifications

### Decision 1: Non-citizen as proxy for undocumented
- **Choice**: CITIZEN = 3 (Not a citizen)
- **Justification**: ACS does not identify legal status directly. Non-citizenship is the best available proxy, though it includes some visa holders.

### Decision 2: Exclude 2012
- **Choice**: Remove all observations from 2012
- **Justification**: DACA was implemented June 15, 2012. ACS does not record interview month, making it impossible to classify 2012 observations as pre- or post-treatment.

### Decision 3: Age ranges (26-30 vs 31-35)
- **Choice**: Treatment born 1982-1986, Control born 1977-1981
- **Justification**: As specified in research design. Creates treatment group that was just under 31 and control group that was just over 31 at DACA implementation.

### Decision 4: Continuous presence filter
- **Choice**: YRIMMIG <= 2007
- **Justification**: DACA required continuous presence since June 15, 2007. Filtering on year of immigration by 2007 approximates this requirement.

### Decision 5: Preferred specification
- **Choice**: WLS with person weights, covariates, state FE, year FE, robust SE
- **Justification**: Person weights produce nationally representative estimates. Covariates improve precision. State FE control for time-invariant state characteristics. Year FE absorb common time shocks. Robust SEs address heteroskedasticity.

### Decision 6: Full-time employment definition
- **Choice**: UHRSWORK >= 35
- **Justification**: Standard BLS definition of full-time employment.

---

## Output Files

| File | Description |
|------|-------------|
| analysis.py | Python analysis script |
| replication_report_59.tex | LaTeX source for report |
| replication_report_59.pdf | Final PDF report (20 pages) |
| run_log_59.md | This run log |
| results.json | Machine-readable results |
| model_summary.txt | Full model output |
| event_study_results.json | Event study coefficients |
| latex_tables.tex | LaTeX table code |
| yearly_rates.csv | Full-time rates by year/group |

---

## Summary Statistics

**Pre-DACA Full-Time Employment Rates**:
- Treatment (ages 26-30): 62.5%
- Control (ages 31-35): 67.0%

**Post-DACA Full-Time Employment Rates**:
- Treatment (ages 26-30): 65.9%
- Control (ages 31-35): 64.1%

**Simple DiD Calculation**:
```
(0.659 - 0.625) - (0.641 - 0.670) = 0.034 - (-0.029) = 0.063
```

---

## Final Summary

**Main Finding**: DACA eligibility increased full-time employment by approximately 4.9 percentage points (robust estimate with controls).

**Statistical Significance**: p < 0.0001

**Robustness**:
- Result holds across all specifications
- Placebo test supports parallel trends assumption
- Event study shows no pre-trends and post-treatment effects

**Report**: 20-page LaTeX document with tables, figures, and technical appendix completed.
