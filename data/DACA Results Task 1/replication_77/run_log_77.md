# DACA Replication Study - Run Log

## Study Information
- **Research Question:** Effect of DACA eligibility on full-time employment (35+ hours/week) among Hispanic-Mexican Mexican-born individuals
- **Date:** January 25, 2026
- **Replication ID:** 77

---

## Step 1: Data Exploration

### Commands Executed:
```bash
# Check data folder contents
ls -la data/

# Examine CSV structure
head -5 data/data.csv

# Count total observations
python -c "with open('data/data.csv', 'r') as f: print(sum(1 for line in f))"
# Result: 33,851,425 lines (including header)
```

### Key Decision: Data Loading Strategy
- Data file is ~6GB with 33.8 million rows
- Decided to use chunked processing with pandas to filter to relevant population early
- Chunk size: 500,000 rows

---

## Step 2: Variable Definitions from Data Dictionary

### Examined Variables:
- **YEAR:** Census/ACS survey year (2006-2016 available)
- **HISPAN:** Hispanic origin (1 = Mexican)
- **BPL:** Birthplace (200 = Mexico)
- **CITIZEN:** Citizenship status (3 = Not a citizen)
- **YRIMMIG:** Year of immigration
- **BIRTHYR:** Birth year
- **BIRTHQTR:** Quarter of birth (1-4)
- **UHRSWORK:** Usual hours worked per week
- **EMPSTAT:** Employment status (1 = Employed)
- **AGE:** Age in years
- **SEX:** Sex (1 = Male, 2 = Female)
- **MARST:** Marital status (1 = Married, spouse present)
- **PERWT:** Person weight
- **STATEFIP:** State FIPS code

---

## Step 3: Sample Construction Decisions

### Decision 1: Population Definition
**Choice:** Hispanic-Mexican (HISPAN == 1) AND born in Mexico (BPL == 200)
**Rationale:**
- Research question specifically asks about "ethnically Hispanic-Mexican Mexican-born people"
- HISPAN = 1 captures Mexican ethnicity
- BPL = 200 captures Mexican birthplace
- This is the population most affected by DACA

### Decision 2: Age Restriction
**Choice:** Ages 18-64
**Rationale:**
- Working-age population for employment analysis
- Excludes children and elderly who are typically not in labor force

### Decision 3: Year Exclusion
**Choice:** Exclude 2012 from analysis
**Rationale:**
- DACA announced June 15, 2012
- ACS does not identify month of data collection
- Cannot distinguish pre- and post-DACA observations in 2012
- Pre-period: 2006-2011, Post-period: 2013-2016

---

## Step 4: DACA Eligibility Criteria

### Decision 4: Eligibility Construction
**DACA Eligible if ALL criteria met:**

1. **Arrived before age 16:**
   - Formula: `YRIMMIG - BIRTHYR < 16`
   - Conditional on valid immigration year (YRIMMIG > 0)

2. **Under 31 as of June 15, 2012:**
   - Born 1982 or later: definitely under 31
   - Born July-December 1981 (BIRTHQTR >= 3): also under 31 on June 15

3. **In US since June 15, 2007:**
   - `YRIMMIG <= 2007` with valid year

4. **Not a citizen:**
   - `CITIZEN == 3` (not a citizen)
   - Note: Cannot distinguish undocumented from documented non-citizens

**Limitations:**
- Cannot observe educational requirements (enrolled in school, HS diploma, GED)
- Cannot observe criminal history exclusions
- These will introduce measurement error in treatment assignment

---

## Step 5: Outcome Variable

### Decision 5: Full-Time Employment Definition
**Choice:** `UHRSWORK >= 35`
**Rationale:**
- Research question defines full-time as "usually working 35 hours per week or more"
- UHRSWORK captures usual hours worked per week
- Binary indicator: 1 if 35+ hours, 0 otherwise

### Decision 6: Employment Scope
**Choice:** Unconditional full-time (entire population, not just employed)
**Rationale:**
- Primary outcome includes both extensive margin (whether working) and intensive margin (hours)
- Also estimate conditional on employment as robustness check

---

## Step 6: Estimation Strategy

### Decision 7: Difference-in-Differences Design
**Treatment:** DACA eligibility (based on criteria above)
**Control:** Hispanic-Mexican Mexican-born who do not meet all eligibility criteria
**Pre-period:** 2006-2011
**Post-period:** 2013-2016

### Decision 8: Model Specifications

**Model 1:** Basic DiD
```
fulltime ~ daca_eligible + post + daca_x_post
```

**Model 2:** DiD with demographic controls
```
fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married
```

**Model 3:** DiD with year fixed effects
```
fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + C(year)
```

**Model 4 (PREFERRED):** DiD with year and state fixed effects
```
fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + C(year) + C(state)
```

### Decision 9: Estimation Method
**Choice:** Weighted Least Squares with PERWT
**Standard Errors:** Heteroskedasticity-robust (HC1)
**Rationale:**
- Person weights (PERWT) ensure nationally representative estimates
- Robust SEs account for heteroskedasticity

---

## Step 7: Analysis Execution

### Python Script: analysis.py

```python
# Key steps in analysis.py:
1. Load data in chunks, filtering to HISPAN==1 & BPL==200
2. Construct DACA eligibility indicator
3. Create outcome variable (fulltime = UHRSWORK >= 35)
4. Restrict to ages 18-64, exclude 2012
5. Run WLS regressions with robust SEs
6. Conduct robustness checks
```

### Execution Command:
```bash
python analysis.py
```

### Output File:
- `analysis_results.pkl` - Contains model objects and summary statistics

---

## Step 8: Results Summary

### Sample Sizes:
| Group | Pre (2006-2011) | Post (2013-2016) | Total |
|-------|-----------------|------------------|-------|
| DACA Ineligible | 415,802 | 268,511 | 684,313 |
| DACA Eligible | 38,248 | 33,099 | 71,347 |
| **Total** | **454,050** | **301,610** | **755,660** |

### Raw Difference-in-Differences:
| | Pre | Post | Diff |
|---|-----|------|------|
| Eligible | 0.510 | 0.547 | +0.037 |
| Ineligible | 0.622 | 0.603 | -0.019 |
| **DiD** | | | **+0.056** |

### Preferred Estimate (Model 4):
- **Coefficient:** 0.0103 (1.03 percentage points)
- **Standard Error:** 0.0044
- **95% CI:** [0.0016, 0.0190]
- **t-statistic:** 2.33
- **p-value:** 0.020
- **N:** 755,660

---

## Step 9: Robustness Checks

### Alternative 1: Full-time conditional on employment
- Coefficient: -0.008 (not significant)
- Suggests effect is on extensive margin (any employment)

### Alternative 2: Any employment
- Coefficient: 0.023***
- 95% CI: [0.015, 0.032]
- Larger effect on employment than on full-time specifically

---

## Step 10: Report Generation

### LaTeX Compilation:
```bash
pdflatex -interaction=nonstopmode replication_report_77.tex
pdflatex -interaction=nonstopmode replication_report_77.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_77.tex  # Third pass for final
```

### Output Files:
- `replication_report_77.tex` - LaTeX source (21 pages)
- `replication_report_77.pdf` - Compiled PDF report

---

## Key Analytical Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Population | Hispanic-Mexican, Mexican-born | Matches research question |
| Age range | 18-64 | Working-age population |
| Exclude 2012 | Yes | Mid-year DACA implementation |
| Full-time threshold | 35+ hours | Per research question |
| Treatment | All DACA criteria | Best available proxy |
| Control group | Same ethnicity/origin, ineligible | Within-group comparison |
| Preferred model | Year + State FE | Controls for time trends and state factors |
| Weighting | PERWT | Nationally representative |
| Standard errors | HC1 robust | Account for heteroskedasticity |

---

## Files Created

1. `analysis.py` - Main analysis script
2. `analysis_results.pkl` - Saved model results
3. `replication_report_77.tex` - LaTeX report source
4. `replication_report_77.pdf` - Final PDF report
5. `run_log_77.md` - This log file

---

## Interpretation of Results

The preferred estimate suggests DACA eligibility increased full-time employment by approximately 1.03 percentage points (95% CI: 0.16 to 1.90 pp, p = 0.020). This effect appears to operate primarily through the extensive margin (whether working at all) rather than the intensive margin (full-time vs. part-time among workers).

The effect is statistically significant but economically modest, representing approximately a 2% relative increase from a base rate of ~50% full-time employment among eligible individuals.
