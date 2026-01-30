# Run Log - DACA Replication Study 19

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (35+ hours/week)?

## Study Design
- **Treatment Group**: Ages 26-30 at policy implementation (June 15, 2012) - born 1982-1986
- **Control Group**: Ages 31-35 at policy implementation - born 1977-1981
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016
- **Method**: Difference-in-Differences

---

## Session Log

### Step 1: Read Instructions and Data Dictionary
**Timestamp**: Session start

**Actions**:
- Read `replication_instructions.docx` using Python docx library
- Reviewed `acs_data_dict.txt` for variable definitions

**Key Variables Identified**:
| Variable | Description | Values Used |
|----------|-------------|-------------|
| YEAR | Survey year | 2006-2016 (excl. 2012) |
| HISPAN | Hispanic origin | 1 = Mexican |
| BPL | Birthplace | 200 = Mexico |
| CITIZEN | Citizenship status | 3 = Not a citizen |
| BIRTHYR | Year of birth | 1977-1986 |
| YRIMMIG | Year of immigration | > 0, <= 2007 |
| UHRSWORK | Usual hours worked/week | >= 35 for full-time |
| PERWT | Person weight | Used for weighting |
| SEX | Sex | 1=Male, 2=Female |
| MARST | Marital status | 1=Married spouse present |
| EDUC | Education | Used for controls |
| STATEFIP | State FIPS | Used for FE |

### Step 2: Define DACA Eligibility Criteria
**Decision**: Based on instructions, eligibility defined as:
1. Hispanic-Mexican ethnicity (HISPAN = 1)
2. Born in Mexico (BPL = 200)
3. Not a citizen (CITIZEN = 3)
4. Arrived before 16th birthday (YRIMMIG - BIRTHYR < 16)
5. Present in US since June 15, 2007 (YRIMMIG <= 2007)

**Treatment/Control Definition**:
- Treatment: Birth years 1982-1986 (would be ages 26-30 on June 15, 2012)
- Control: Birth years 1977-1981 (would be ages 31-35 on June 15, 2012)

**Rationale**: The age cutoff at 31 creates a natural experiment where individuals just above and below the cutoff are similar except for DACA eligibility.

### Step 3: Data Loading and Processing

**Command**:
```python
python analysis.py
```

**Data Loading Approach**:
- Used chunked reading (500,000 rows per chunk) due to large file size (6.3 GB)
- Applied filters during loading to reduce memory usage

**Sample Selection Pipeline**:
| Filter | Remaining N |
|--------|-------------|
| Full ACS 2006-2016 | 33,851,424 |
| Hispanic-Mexican (HISPAN=1) | 701,347 |
| Born in Mexico (BPL=200) | 701,347 |
| Non-citizen (CITIZEN=3) | 701,347 |
| Birth years 1977-1986 | 178,376 |
| Excluding 2012 | 162,283 |
| Valid YRIMMIG (>0) | 162,283 |
| Arrived before age 16 | 44,725 |
| Arrived by 2007 | 44,725 |

**Final Sample**: 44,725 observations

### Step 4: Outcome Variable Definition

**Decision**: Full-time employment defined as UHRSWORK >= 35 hours per week

**Rationale**: This follows the standard Bureau of Labor Statistics definition of full-time work.

**Outcome Statistics**:
- Mean UHRSWORK: 29.97 hours
- Full-time employment rate: 62.43%

### Step 5: Summary Statistics

**Sample Distribution**:
| Period | Control (n) | Treatment (n) |
|--------|-------------|---------------|
| Pre-DACA (2006-2011) | 11,916 | 17,410 |
| Post-DACA (2013-2016) | 6,218 | 9,181 |

**Full-Time Employment Rates (Weighted)**:

|  | Control | Treatment | Difference |
|--|---------|-----------|------------|
| Pre-DACA | 0.6705 | 0.6253 | -0.0452 |
| Post-DACA | 0.6412 | 0.6580 | +0.0168 |
| Change | -0.0293 | +0.0327 | |
| **DiD** | | | **0.0620** |

### Step 6: Regression Analysis

**Model Specifications**:

| Model | Description | DiD Estimate | SE | p-value |
|-------|-------------|--------------|-----|---------|
| 1 | Basic OLS | 0.0551 | 0.0098 | <0.001 |
| 2 | Weighted (PERWT) | **0.0620** | **0.0097** | **<0.001** |
| 3 | + Covariates | 0.0485 | 0.0089 | <0.001 |
| 4 | + Year FE | 0.0473 | 0.0089 | <0.001 |
| 5 | + State FE | 0.0466 | 0.0089 | <0.001 |

**Preferred Specification**: Model 2 (Weighted DiD)
- DiD Estimate: 0.0620 (6.2 percentage points)
- 95% CI: [0.0431, 0.0810]
- t-statistic: 6.42

### Step 7: Robustness Checks

**Check 1: Heterogeneity by Gender**
| Gender | DiD Estimate | SE |
|--------|--------------|-----|
| Male | 0.0621 | 0.0107 |
| Female | 0.0313 | 0.0150 |

**Check 2: Placebo Test (Pre-DACA Only)**
- Pseudo-post period: 2009-2011
- Pseudo-pre period: 2006-2008
- Placebo DiD: 0.0120 (SE: 0.0113, p=0.291)
- **Interpretation**: No significant pre-trend, supporting parallel trends assumption

**Check 3: Event Study**
| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | -0.0053 | 0.0194 | 0.785 |
| 2007 | -0.0133 | 0.0197 | 0.499 |
| 2008 | 0.0186 | 0.0200 | 0.352 |
| 2009 | 0.0169 | 0.0201 | 0.401 |
| 2010 | 0.0189 | 0.0201 | 0.347 |
| 2011 | 0.0000 (ref) | --- | --- |
| 2013 | 0.0595 | 0.0208 | **0.004** |
| 2014 | 0.0696 | 0.0208 | **0.001** |
| 2015 | 0.0427 | 0.0214 | **0.046** |
| 2016 | 0.0953 | 0.0216 | **<0.001** |

**Interpretation**: Pre-period coefficients are small and insignificant; post-period coefficients are positive, significant, and growing.

**Check 4: Alternative Outcomes**
| Outcome | DiD Estimate | SE |
|---------|--------------|-----|
| Any employment (>0 hrs) | 0.0354 | 0.0083 |
| Hours (if employed) | 1.4213 | 0.2290 |

### Step 8: Key Decisions Summary

1. **Sample restriction**: Focused on DACA-eligible population by applying all eligibility criteria that can be observed in ACS data.

2. **Treatment definition**: Used birth year cutoffs rather than exact age calculation for simplicity and to avoid measurement error from uncertain survey timing.

3. **Excluding 2012**: Removed 2012 from analysis because DACA was implemented mid-year and survey timing is unknown.

4. **Weighting**: Used PERWT for all main analyses to account for survey design.

5. **Outcome definition**: Used 35+ hours as full-time threshold following BLS standards.

6. **Preferred model**: Selected weighted DiD without covariates as preferred specification because (a) it is the most transparent, (b) results are robust to covariate inclusion, and (c) parallel trends are supported by placebo tests.

### Step 9: Output Files Created

| File | Description |
|------|-------------|
| `analysis.py` | Main Python analysis script |
| `analysis_output.txt` | Full console output from analysis |
| `key_results.txt` | Summary of key findings |
| `summary_stats.csv` | Summary statistics by group/period |
| `event_study_results.csv` | Event study coefficients |
| `replication_report_19.tex` | LaTeX source for report |
| `replication_report_19.pdf` | Final PDF report (27 pages) |

### Step 10: Final Results

**Main Finding**: DACA eligibility is associated with a statistically significant **6.2 percentage point increase** in the probability of full-time employment.

**Preferred Estimate**:
- Effect size: 0.0620
- Standard error: 0.0097
- 95% CI: [0.043, 0.081]
- t-statistic: 6.42
- p-value: <0.001
- Sample size: 44,725

**Interpretation**: The results suggest that DACA eligibility had a meaningful positive effect on full-time employment, increasing the probability by approximately 10% relative to the pre-DACA treatment group mean of 62.5%.

---

## Computational Environment

- **Platform**: Windows 10
- **Python Version**: 3.x
- **Key Packages**:
  - pandas (data manipulation)
  - numpy (numerical computing)
  - statsmodels (regression analysis)
- **LaTeX**: MiKTeX

---

## Files in Deliverables Folder

1. `replication_report_19.tex` - LaTeX source
2. `replication_report_19.pdf` - Final report (27 pages)
3. `run_log_19.md` - This log file
4. `analysis.py` - Analysis code
5. `data/` - Data folder (not modified)
