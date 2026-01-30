# Run Log - DACA Replication Study (Run 39)

## Overview
This document logs all commands and key decisions made during the DACA replication study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (treatment) on the probability that the eligible person is employed full-time (usually working 35+ hours per week)?

## Identification Strategy
- **Treated Group**: Ages 26-30 at time of DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at time of DACA implementation (would have been eligible but for age)
- **Method**: Difference-in-Differences comparing changes from pre-DACA (2006-2011) to post-DACA (2013-2016)

---

## Session Log

### Step 1: Data Exploration and Understanding

**Files Examined**:
- `replication_instructions.docx` - Research design specifications
- `acs_data_dict.txt` - IPUMS ACS variable definitions
- `data.csv` - Main ACS data file (2006-2016)

**Key Variables Identified**:
| Variable | Description | Use in Analysis |
|----------|-------------|-----------------|
| YEAR | Survey year | Time periods |
| HISPAN | Hispanic origin (1=Mexican) | Sample restriction |
| BPL | Birthplace (200=Mexico) | Sample restriction |
| CITIZEN | Citizenship status (3=Not a citizen) | Eligibility criterion |
| YRIMMIG | Year of immigration | Eligibility criterion (arrived before 16th birthday) |
| BIRTHYR | Birth year | Age calculation |
| BIRTHQTR | Birth quarter | Precise age calculation |
| UHRSWORK | Usual hours worked per week | Outcome variable (35+ = full-time) |
| AGE | Age at survey | Age group assignment |
| PERWT | Person weight | Survey weighting |

**DACA Eligibility Criteria Applied**:
1. Hispanic-Mexican ethnicity (HISPAN = 1)
2. Born in Mexico (BPL = 200)
3. Not a citizen (CITIZEN = 3)
4. Arrived in US before 16th birthday (calculated from BIRTHYR and YRIMMIG)
5. Arrived by June 15, 2007 (YRIMMIG <= 2007)

---

### Step 2: Sample Construction

**Decision**: Exclude 2012 from analysis
- Rationale: DACA implemented June 15, 2012; cannot distinguish pre/post observations in 2012 ACS

**Pre-Period**: 2006-2011
**Post-Period**: 2013-2016

**Age Group Assignment** (based on age as of June 15, 2012):
- Treatment: Born 1982-1986 (ages 26-30 in 2012)
- Control: Born 1977-1981 (ages 31-35 in 2012)

---

### Step 3: Data Cleaning and Variable Construction

**Sample Restriction Process**:
```
Initial sample (2006-2011, 2013-2016): 33,851,424 observations
After Hispanic-Mexican restriction (HISPAN=1): 2,945,521 (dropped 30,905,903)
After Mexico birthplace restriction (BPL=200): 991,261 (dropped 1,954,260)
After non-citizen restriction (CITIZEN=3): 701,347 (dropped 289,914)
After excluding 2012: 636,722 (dropped 64,625)
After age group restriction (ages 26-35 in 2012): 162,283
After arrived-before-16 restriction: 44,725
After in-US-since-2007 restriction: 44,725 (final sample)
```

**Variable Construction**:
- `fulltime`: Binary indicator = 1 if UHRSWORK >= 35
- `employed`: Binary indicator = 1 if EMPSTAT == 1
- `treated`: Binary indicator = 1 if BIRTHYR in [1982, 1986]
- `post`: Binary indicator = 1 if YEAR >= 2013
- `treated_post`: Interaction term = treated * post
- `female`: Binary indicator = 1 if SEX == 2
- `married`: Binary indicator = 1 if MARST == 1
- `age_at_arrival`: YRIMMIG - BIRTHYR
- `arrived_before_16`: Binary = 1 if age_at_arrival < 16

---

### Step 4: Descriptive Statistics

**Sample Sizes by Group and Period**:
|                     | Pre-DACA | Post-DACA |
|---------------------|----------|-----------|
| Control (31-35)     | 11,916   | 6,218     |
| Treatment (26-30)   | 17,410   | 9,181     |

**Full-Time Employment Rates**:
|                     | Pre-DACA | Post-DACA |
|---------------------|----------|-----------|
| Control (31-35)     | 64.31%   | 61.08%    |
| Treatment (26-30)   | 61.11%   | 63.39%    |

**Simple DiD Calculation**:
- Treatment change: 63.39% - 61.11% = +2.28 pp
- Control change: 61.08% - 64.31% = -3.23 pp
- DiD estimate: 2.28 - (-3.23) = 5.51 pp

---

### Step 5: Regression Analysis

**Commands Run**:
```python
# Model 1: Basic DiD
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit(cov_type='HC1')

# Model 2: DiD with Year Fixed Effects
model2 = smf.ols('fulltime ~ treated + C(year_factor) + treated_post', data=df).fit(cov_type='HC1')

# Model 3: DiD with Year FE and Demographics
model3 = smf.ols('fulltime ~ treated + C(year_factor) + treated_post + female + married + NCHILD + educ_hs + educ_some_college + educ_college', data=df).fit(cov_type='HC1')

# Model 4: DiD with Year and State FE
model4 = smf.ols('fulltime ~ treated + C(year_factor) + C(state_factor) + treated_post', data=df).fit(cov_type='HC1')

# Model 5: Full Specification (Preferred)
model5 = smf.ols('fulltime ~ treated + C(year_factor) + C(state_factor) + treated_post + female + married + NCHILD + educ_hs + educ_some_college + educ_college', data=df).fit(cov_type='HC1')
```

**Main Results**:
| Model | Specification | Estimate | SE | p-value |
|-------|--------------|----------|-----|---------|
| 1 | Basic DiD | 0.0551 | 0.0098 | <0.001 |
| 2 | Year FE | 0.0554 | 0.0098 | <0.001 |
| 3 | Year FE + Demographics | 0.0462 | 0.0091 | <0.001 |
| 4 | Year FE + State FE | 0.0542 | 0.0098 | <0.001 |
| 5 | Full Specification | 0.0453 | 0.0091 | <0.001 |
| 6 | Weighted (PERWT) | 0.0620 | 0.0116 | <0.001 |

---

### Step 6: Robustness Checks

**Check 1: Alternative Outcome (Any Employment)**
- Estimate: 0.0431 (SE: 0.0094), p < 0.001

**Check 2: Narrower Age Bandwidth (27-29 vs 32-34)**
- Estimate: 0.0599 (SE: 0.0126), p < 0.001, N = 26,792

**Check 3: By Gender**
- Males: 0.0598 (SE: 0.0112), N = 25,058
- Females: 0.0372 (SE: 0.0150), N = 19,667

**Check 4: Pre-trend Test (Placebo 2009-2011 vs 2006-2008)**
- Estimate: 0.0164 (SE: 0.0115), p = 0.152 (not significant)
- Supports parallel trends assumption

---

### Step 7: Event Study Analysis

**Year-by-Year Treatment Effects (Reference: 2011)**:
| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2006 | -0.0347 | 0.0196 | [-0.073, 0.004] |
| 2007 | -0.0226 | 0.0198 | [-0.061, 0.016] |
| 2008 | -0.0032 | 0.0202 | [-0.043, 0.037] |
| 2009 | -0.0013 | 0.0207 | [-0.042, 0.039] |
| 2010 | -0.0112 | 0.0205 | [-0.052, 0.029] |
| 2011 | 0.0000 | -- | Reference |
| 2013 | 0.0299 | 0.0213 | [-0.012, 0.072] |
| 2014 | 0.0398 | 0.0214 | [-0.002, 0.082] |
| 2015 | 0.0390 | 0.0218 | [-0.004, 0.082] |
| 2016 | 0.0642 | 0.0219 | [0.021, 0.107] |

**Interpretation**: Pre-treatment coefficients are close to zero and not significant, supporting parallel trends. Post-DACA effects are positive and grow over time.

---

### Step 8: Figures Generated

1. `figure1_event_study.pdf` - Event study plot with 95% CIs
2. `figure2_parallel_trends.pdf` - Trends by treatment group
3. `figure3_did_illustration.pdf` - DiD visual explanation
4. `figure4_model_comparison.pdf` - Coefficient comparison across models

---

### Step 9: LaTeX Report Compilation

**Commands**:
```bash
pdflatex -interaction=nonstopmode replication_report_39.tex
pdflatex -interaction=nonstopmode replication_report_39.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_39.tex  # Third pass for TOC
```

**Output**: `replication_report_39.pdf` (21 pages)

---

## Key Decisions Summary

1. **Sample Definition**:
   - Used HISPAN=1 (Mexican) and BPL=200 (born in Mexico) per instructions
   - Used CITIZEN=3 (non-citizen) as proxy for undocumented status
   - Excluded 2012 due to mid-year DACA implementation

2. **Age Groups**:
   - Treatment: Birth years 1982-1986 (ages 26-30 on June 15, 2012)
   - Control: Birth years 1977-1981 (ages 31-35 on June 15, 2012)

3. **DACA Eligibility Criteria**:
   - Arrived before age 16: (YRIMMIG - BIRTHYR) < 16
   - In US since 2007: YRIMMIG <= 2007

4. **Outcome Variable**:
   - Full-time employment = UHRSWORK >= 35 hours

5. **Preferred Specification**:
   - Model 5 with year FE, state FE, and demographic controls
   - Estimate: 0.0453 (SE: 0.0091), 95% CI: [0.0275, 0.0632]

6. **Standard Errors**:
   - Heteroskedasticity-robust (HC1) throughout
   - No clustering needed (individual-level treatment, cross-sectional data)

---

## Final Results

**Preferred Estimate**: DACA eligibility increased full-time employment by **4.53 percentage points** (SE: 0.0091).

**95% Confidence Interval**: [2.75 pp, 6.32 pp]

**Statistical Significance**: p < 0.001

**Sample Size**: 44,725 observations

---

## Output Files

- `replication_report_39.tex` - LaTeX source
- `replication_report_39.pdf` - Final report (21 pages)
- `run_log_39.md` - This file
- `daca_analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script
- `regression_results.csv` - Model estimates
- `event_study_results.csv` - Event study coefficients
- `descriptive_stats.csv` - Summary statistics
- `figure1_event_study.pdf` - Event study figure
- `figure2_parallel_trends.pdf` - Parallel trends figure
- `figure3_did_illustration.pdf` - DiD illustration
- `figure4_model_comparison.pdf` - Model comparison figure
