# Run Log for DACA Replication Study - ID 71

## Overview
This log documents all commands executed and key decisions made during the replication study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals.

## Research Question
What was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (working 35+ hours per week)?

## Study Design
- **Treatment Group**: Ages 26-30 at time of DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at time of DACA implementation (would have been eligible except for age)
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2006-2011 (excluding 2012 due to mid-year implementation)
- **Post-period**: 2013-2016

---

## Session Log

### Step 1: Read and Understand Instructions
**Command**: Read replication_instructions.docx using python-docx

**Key findings from instructions**:
- Target population: Hispanic-Mexican, Mexican-born individuals
- Treatment: Ages 26-30 as of June 15, 2012 (DACA eligible)
- Control: Ages 31-35 as of June 15, 2012 (too old for DACA)
- Outcome: Full-time employment (35+ hours/week)
- Data: ACS 2006-2016 from IPUMS
- Analysis: Difference-in-differences

### Step 2: Review Data Dictionary
**File reviewed**: data/acs_data_dict.txt

**Key variables identified**:
| Variable | Description | Use |
|----------|-------------|-----|
| YEAR | Survey year | Time period |
| PERWT | Person weight | Weighting |
| AGE | Age at survey | Sample selection |
| BIRTHYR | Birth year | Age calculation |
| BIRTHQTR | Birth quarter | Precise age calculation |
| HISPAN | Hispanic origin | Sample: =1 (Mexican) |
| BPL | Birthplace | Sample: =200 (Mexico) |
| CITIZEN | Citizenship status | Sample: =3 (non-citizen) |
| YRIMMIG | Year of immigration | Eligibility criteria |
| UHRSWORK | Usual hours worked/week | Outcome: >=35 |
| SEX | Sex | Covariate |
| EDUC | Education | Covariate |
| MARST | Marital status | Covariate |
| STATEFIP | State FIPS code | State FE |

### Step 3: Define DACA Eligibility Criteria

**DACA eligibility requirements (from instructions)**:
1. Arrived in US before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

**Sample Selection Decisions**:
```
1. HISPAN == 1       # Mexican Hispanic ethnicity
2. BPL == 200        # Born in Mexico
3. CITIZEN == 3      # Not a citizen (proxy for undocumented)
4. YRIMMIG <= 2007   # In US since at least 2007
5. age_at_immig < 16 # Arrived before 16th birthday
   (calculated as: YRIMMIG - BIRTHYR < 16)
```

**Treatment Assignment**:
- Calculate age as of June 15, 2012 using BIRTHYR and BIRTHQTR
- Treatment: age_june2012 in [26, 30]
- Control: age_june2012 in [31, 35]

### Step 4: Data Processing
**Command**: `python analysis.py`

**Initial data**: 33,851,425 rows in data.csv

**Filtering steps**:
1. HISPAN == 1 (Mexican): Reduced to ~701,347
2. BPL == 200 (Born Mexico): Applied
3. CITIZEN == 3 (Non-citizen): Applied
4. Valid YRIMMIG (>0, !=996): Applied
5. age_at_immig < 16: Reduced to ~195,023
6. YRIMMIG <= 2007: Applied
7. Ages 26-35 on June 15, 2012: Reduced to 47,418
8. Exclude 2012: Final sample 43,238

**Final Sample Composition**:
- Total observations: 43,238 person-year
- Treatment group (ages 26-30): 25,470 (59%)
- Control group (ages 31-35): 17,768 (41%)
- Pre-period (2006-2011): 28,377 (66%)
- Post-period (2013-2016): 14,861 (34%)

### Step 5: Summary Statistics (Pre-period)

| Variable | Treatment | Control | Difference |
|----------|-----------|---------|------------|
| Full-time | 0.615 | 0.646 | -0.031 |
| Female | 0.438 | 0.434 | +0.005 |
| Married | 0.391 | 0.541 | -0.150 |
| Age | 24.71 | 29.87 | -5.16 |
| Less than HS | 0.382 | 0.462 | -0.081 |
| High School | 0.446 | 0.403 | +0.044 |
| Some College | 0.144 | 0.104 | +0.040 |
| College Degree | 0.029 | 0.031 | -0.002 |

### Step 6: Difference-in-Differences Results

**Models Estimated**:
1. Basic DiD (no covariates)
2. DiD + Demographics (female, married, age)
3. DiD + Demographics + Education
4. DiD + Year Fixed Effects
5. DiD + Year + State Fixed Effects (PREFERRED)

**Main Results Table**:

| Model | DiD Estimate | Std. Error | p-value | 95% CI |
|-------|--------------|------------|---------|--------|
| Basic DiD | 0.0590 | 0.0117 | <0.001 | [0.036, 0.082] |
| + Demographics | 0.0480 | 0.0107 | <0.001 | [0.027, 0.069] |
| + Education | 0.0463 | 0.0107 | <0.001 | [0.025, 0.067] |
| + Year FE | 0.0456 | 0.0107 | <0.001 | [0.025, 0.066] |
| + Year + State FE | **0.0448** | **0.0107** | **<0.001** | **[0.024, 0.066]** |

**Preferred Estimate**: 4.48 percentage points (SE = 0.0107)

### Step 7: Manual DiD Verification

**2x2 DiD Table (Weighted)**:
```
                  Pre-DACA    Post-DACA    Diff
Treatment (26-30): 0.6305      0.6597       +0.0292
Control (31-35):   0.6731      0.6433       -0.0299
DiD Estimate:                              +0.0590
```

Confirms basic DiD estimate of ~5.9 percentage points.

### Step 8: Robustness Checks

**8a. By Sex**:
- Male: 0.0446 (SE = 0.0125), N = 24,243
- Female: 0.0454 (SE = 0.0185), N = 18,995
- Effects similar across sexes

**8b. Placebo Test (Ages 31-35 vs 36-40)**:
- Estimate: 0.0030 (SE = 0.0135), p = 0.82
- Not significant, supports validity of main results

**8c. Event Study**:
- Pre-period effects (2006-2010): All non-significant
- Supports parallel trends assumption
- Post-period effects positive, 2016 significant at 5% level

### Step 9: Generate Report

**Commands**:
```bash
# Write LaTeX report
# File: replication_report_71.tex

# Compile to PDF (3 passes for cross-references)
pdflatex -interaction=nonstopmode replication_report_71.tex
pdflatex -interaction=nonstopmode replication_report_71.tex
pdflatex -interaction=nonstopmode replication_report_71.tex
```

**Output**: replication_report_71.pdf (19 pages)

---

## Key Decisions Summary

1. **Citizenship proxy**: Used CITIZEN == 3 (non-citizen) as proxy for undocumented status, following standard approach in literature.

2. **Age calculation**: Used birth quarter to calculate precise age as of June 15, 2012.

3. **Continuous presence**: Used YRIMMIG <= 2007 as proxy for continuous presence requirement.

4. **Arrival age**: Calculated age at immigration and required < 16.

5. **Exclude 2012**: Year 2012 excluded because DACA was implemented mid-year (June 15).

6. **Preferred specification**: Model with year and state fixed effects plus demographic and education controls.

7. **Standard errors**: Heteroskedasticity-robust (HC1) standard errors throughout.

8. **Weights**: All regressions use ACS person weights (PERWT).

---

## Output Files

| File | Description |
|------|-------------|
| analysis.py | Main analysis script |
| results_summary.csv | Regression results table |
| fulltime_rates_by_year.csv | Employment rates by year/group |
| sample_sizes.csv | Sample sizes by year/group |
| summary_statistics.csv | Pre-period summary stats |
| replication_report_71.tex | LaTeX report source |
| replication_report_71.pdf | Final PDF report (19 pages) |
| run_log_71.md | This log file |

---

## Final Results Summary

**Preferred Estimate (Model 5: Year + State FE with covariates)**:
- Effect of DACA eligibility on full-time employment: **+4.48 percentage points**
- Standard Error: 0.0107
- 95% Confidence Interval: [0.024, 0.066]
- p-value: < 0.001
- Sample Size: 43,238 person-year observations

**Interpretation**: DACA eligibility increased the probability of full-time employment by approximately 4.5 percentage points among Hispanic-Mexican individuals born in Mexico. This represents a relative increase of about 7% compared to the pre-period baseline of ~63%.
