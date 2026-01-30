# Run Log - Replication 07

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA employment study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of being employed full-time (35+ hours per week)?

## Study Design
- **Treatment group**: DACA-eligible individuals aged 26-30 as of June 15, 2012
- **Control group**: Individuals aged 31-35 as of June 15, 2012 who would otherwise be eligible but for age
- **Method**: Difference-in-differences comparing pre/post changes
- **Post-treatment period**: 2013-2016 (as specified)
- **Pre-treatment period**: Years before 2012 (using 2006-2011)

---

## Session Log

### Step 1: Initial Data Exploration
**Date**: 2026-01-25

**Actions taken**:
1. Read replication_instructions.docx to understand the research task
2. Explored data folder structure:
   - `data.csv` - Main ACS data file (~6GB)
   - `acs_data_dict.txt` - Variable documentation
   - `state_demo_policy.csv` - Optional state-level data
3. Reviewed ACS data dictionary for relevant variables

**Key variables identified**:
- YEAR: Survey year (2006-2016)
- HISPAN/HISPAND: Hispanic origin (1=Mexican)
- BPL/BPLD: Birthplace (200=Mexico)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Birth quarter
- UHRSWORK: Usual hours worked per week
- AGE: Age at survey time
- SEX, EDUC, MARST: Demographic controls
- PERWT: Person weight for population estimates

### Step 2: Sample Definition Decisions

**DACA Eligibility Criteria (per instructions)**:
1. Hispanic-Mexican ethnicity (HISPAN=1 or HISPAND=100-107)
2. Born in Mexico (BPL=200 or BPLD=20000)
3. Not a citizen (CITIZEN=3)
4. Arrived before age 16 (can calculate from BIRTHYR and YRIMMIG)
5. Continuous US residence since June 15, 2007 (proxied by YRIMMIG <= 2007)

**Age Groups (as of June 15, 2012)**:
- Treatment: Ages 26-30 (born 1982-1986, with adjustments for birth quarter)
- Control: Ages 31-35 (born 1977-1981)

**Key Decision**: Since we cannot distinguish documented from undocumented non-citizens, we assume all non-citizen foreign-born individuals who haven't received papers are undocumented (per instructions).

### Step 3: Outcome Variable Construction

**Full-time employment**: UHRSWORK >= 35
- This uses "usual hours worked per week"
- Coded as binary indicator (1 = full-time, 0 = not full-time)

### Step 4: Analysis Approach

**Main Specification**: Difference-in-differences regression
```
fulltime = β₀ + β₁*treat + β₂*post + β₃*treat×post + covariates + ε
```

Where:
- treat = 1 if ages 26-30 as of 2012, 0 if ages 31-35
- post = 1 if year >= 2013, 0 if year <= 2011 (excluding 2012)
- β₃ is the diff-in-diff estimate (main effect of interest)

**Covariates considered**:
- SEX: Gender
- EDUC: Educational attainment
- MARST: Marital status
- STATEFIP: State fixed effects (optional)
- YEAR: Year fixed effects (instead of simple post indicator)

**Pre-treatment period**: 2006-2011 (excluding 2012 due to timing ambiguity)
**Post-treatment period**: 2013-2016 (as specified in instructions)

---

## Analysis Execution

### Step 5: Running the Analysis

**Command executed**:
```bash
python analysis_script.py
```

**Sample construction results**:
- Total ACS observations loaded: 33,851,424
- After Hispanic-Mexican filter: 2,945,521
- After born in Mexico filter: 991,261
- After non-citizen filter: 701,347
- After age group restriction (26-35 as of 2012): 181,229
- After arrival before age 16 filter: 47,418
- After continuous residence (YRIMMIG<=2007) filter: 47,418
- After excluding 2012: 43,238

**Final sample composition**:
- Treatment group (ages 26-30): 25,470
- Control group (ages 31-35): 17,768
- Pre-period (2006-2011): 28,377
- Post-period (2013-2016): 14,861

### Step 6: Main Results

**Preferred specification (Model 2: DiD with demographics)**:
- DiD Coefficient: 0.0465
- Standard Error: 0.0093
- t-statistic: 5.018
- p-value: 0.0000
- 95% CI: [0.0283, 0.0646]
- Sample size: 43,238
- R-squared: 0.1366

**Interpretation**: DACA eligibility increased full-time employment by 4.65 percentage points, statistically significant at the 1% level.

### Step 7: Robustness Checks

| Specification | Coefficient | Std Error | 95% CI |
|--------------|-------------|-----------|--------|
| Simple DiD | 0.0516 | 0.0100 | [0.032, 0.071] |
| + Demographics | 0.0465 | 0.0093 | [0.028, 0.065] |
| + Year FE | 0.0462 | 0.0092 | [0.028, 0.064] |
| Robust SE | 0.0465 | 0.0092 | [0.028, 0.065] |
| Weighted | 0.0481 | 0.0090 | [0.030, 0.066] |
| Narrow age window | 0.0423 | 0.0120 | [0.019, 0.066] |

**Placebo test (fake treatment in 2009)**:
- Placebo DiD: 0.0030, p-value: 0.781
- Confirms no spurious effect in pre-period

**Gender heterogeneity**:
- Males: 0.0343 (SE: 0.0113)
- Females: 0.0521 (SE: 0.0150)

### Step 8: Parallel Trends Assessment

Pre-treatment employment rate differences (Treatment - Control):
- 2006: -3.40 pp
- 2007: -4.98 pp
- 2008: -1.96 pp
- 2009: -3.19 pp
- 2010: -4.02 pp
- 2011: -1.17 pp

Post-treatment differences:
- 2013: +1.20 pp
- 2014: +1.79 pp
- 2015: +1.47 pp
- 2016: +3.72 pp

Event study coefficients (vs 2011 reference):
- Pre-period coefficients not significantly different from zero
- Post-period coefficients positive (0.027-0.043)

---

## Key Decisions Log

| Decision | Choice Made | Rationale |
|----------|-------------|-----------|
| Pre-treatment period | 2006-2011 | Use all available years for trend estimation |
| Exclude 2012 | Yes | Cannot distinguish pre/post within 2012 |
| Age calculation | Adjust for birth quarter | More accurate age as of June 15, 2012 |
| Citizenship filter | CITIZEN=3 only | Per instructions, assume non-citizens are undocumented |
| DACA arrival criterion | YRIMMIG <= 2007 | Proxy for continuous residence since June 2007 |
| Arrival age | age_at_arrival < 16 | Must have arrived before 16th birthday |
| Outcome | UHRSWORK >= 35 | Standard BLS full-time definition |
| Preferred model | Model 2 with demographics | Balance of parsimony and control |
| Standard errors | OLS (not robust) | Large sample, homoscedasticity reasonable |

---

## Output Files Created

1. **analysis_script.py** - Main analysis code
2. **create_figures.py** - Figure generation code
3. **analysis_results.txt** - Summary statistics
4. **figure1_parallel_trends.png/pdf** - Trend comparison
5. **figure2_event_study.png/pdf** - Event study plot
6. **figure3_did_visualization.png/pdf** - DiD concept illustration
7. **figure4_model_comparison.png/pdf** - Robustness comparison
8. **figure5_heterogeneity.png/pdf** - Gender heterogeneity
9. **figure6_sample_composition.png/pdf** - Sample sizes by year
10. **replication_report_07.tex** - LaTeX report (21 pages)
11. **replication_report_07.pdf** - Final PDF report

---

## Summary of Findings

**Main Result**: DACA eligibility increased full-time employment by 4.65 percentage points (SE = 0.0093, p < 0.001, 95% CI: [2.83, 6.46] pp).

**Robustness**: Effect is stable across:
- Alternative specifications (4.2-5.2 pp range)
- Year fixed effects
- Robust standard errors
- Survey weights
- Narrower age windows

**Validity checks**:
- Parallel trends supported by pre-period analysis
- Placebo test shows no spurious effect (p = 0.78)
- Event study shows no pre-trends

**Interpretation**: DACA's provision of legal work authorization meaningfully improved labor market outcomes for eligible individuals, increasing full-time employment probability by about 7.6% relative to the pre-treatment baseline.

---

## Reproducibility

All analysis can be reproduced by running:
```bash
python analysis_script.py
python create_figures.py
pdflatex replication_report_07.tex
```

Required Python packages:
- pandas
- numpy
- statsmodels
- matplotlib
- scipy

---

*Log completed: 2026-01-25*
