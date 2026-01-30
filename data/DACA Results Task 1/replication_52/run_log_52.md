# DACA Replication Study Run Log

## Date: 2025-01-25

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

---

## Session 1: Data Exploration and Setup

### 1.1 Initial Data Review

**Data Files Available:**
- `data/data.csv` - Main ACS data file (6.26 GB)
- `data/acs_data_dict.txt` - Data dictionary for ACS variables
- `data/state_demo_policy.csv` - Optional state-level data
- `data/State Level Data Documentation.docx` - Documentation for state data

**Data Covers:** ACS 2006-2016 (annual one-year samples)

### 1.2 Key Variables Identified

**For DACA Eligibility:**
- `YEAR` - Survey year
- `BIRTHYR` - Birth year
- `BIRTHQTR` - Birth quarter (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- `HISPAN` - Hispanic origin (1=Mexican)
- `BPL` - Birthplace (200=Mexico)
- `CITIZEN` - Citizenship status (3=Not a citizen)
- `YRIMMIG` - Year of immigration

**For Outcome:**
- `UHRSWORK` - Usual hours worked per week (35+ = full-time)
- `EMPSTAT` - Employment status (1=Employed)

**Survey Design:**
- `PERWT` - Person weight for population estimates
- `STATEFIP` - State FIPS code

### 1.3 DACA Eligibility Criteria (from instructions)

To be DACA eligible, individuals must:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

**Key Date:** DACA implemented June 15, 2012

### 1.4 Identification Strategy

**Approach: Difference-in-Differences (DiD)**

The study uses a difference-in-differences framework comparing:
- **Treatment Group:** Hispanic-Mexican, Mexican-born, non-citizens who meet DACA age/arrival criteria
- **Control Group:** Hispanic-Mexican, Mexican-born, non-citizens who do NOT meet DACA criteria (e.g., arrived too late, too old)

**Pre-period:** 2006-2011 (before DACA)
**Post-period:** 2013-2016 (after DACA, as specified in instructions)
**Note:** 2012 is ambiguous (DACA implemented mid-year), excluded from main analysis

---

## Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sample restriction | Hispanic-Mexican (HISPAN=1), Born in Mexico (BPL=200), Non-citizens (CITIZEN=3) | Per research question |
| Treatment definition | DACA-eligible based on age at arrival (<16), birth year, and arrival timing (<=2007) | Per DACA eligibility rules |
| Control group | Same population but not meeting age/arrival criteria | Clean counterfactual |
| Outcome variable | Full-time employment (UHRSWORK >= 35) | Per research question |
| Estimation method | Difference-in-Differences with WLS (weighted by PERWT) | Standard causal inference approach |
| Exclude 2012 | Yes | DACA implemented mid-year, cannot separate pre/post |
| Age restriction | Ages 16-40 | Focus on working-age population affected by DACA |
| Standard errors | HC1 robust | Account for heteroskedasticity |

---

## Session 2: Analysis Execution

### 2.1 Data Processing

**Commands Executed:**
```python
# analysis.py - Main analysis script

# Read data in chunks due to file size (6.26 GB)
# Filter immediately to: HISPAN=1, BPL=200, CITIZEN=3
# Applied age restriction (16-40) and excluded 2012
```

**Sample Construction:**
| Step | Observations |
|------|-------------|
| Full ACS 2006-2016 | ~35.2 million |
| Hispanic-Mexican | ~2.9 million |
| Born in Mexico | ~1.5 million |
| Non-citizen | 701,347 |
| Ages 16-40 | 389,763 |
| Exclude 2012 | 355,188 |

### 2.2 Treatment Definition

**DACA Eligibility Criteria Applied:**
1. `age_at_arrival = YRIMMIG - BIRTHYR < 16`
2. `BIRTHYR > 1981` OR (`BIRTHYR = 1981` AND `BIRTHQTR >= 3`)
3. `YRIMMIG <= 2007`
4. `YRIMMIG > 0` (valid year)

**Result:** 83,611 observations classified as DACA-eligible

### 2.3 Main Results

**Difference-in-Differences Estimates:**

| Model | DiD Coefficient | SE | p-value |
|-------|----------------|-----|---------|
| (1) Basic DiD | 0.1019 | 0.0049 | <0.001 |
| (2) + Demographics | 0.0209 | 0.0044 | <0.001 |
| (3) + Full Controls | 0.0186 | 0.0044 | <0.001 |
| (4) + Year FE | 0.0078 | 0.0044 | 0.079 |
| (5) + State & Year FE | 0.0079 | 0.0044 | 0.074 |

### 2.4 Preferred Estimate (Model 5)

**Effect Size:** 0.0079 (0.79 percentage points)
**Standard Error:** 0.0044
**95% CI:** [-0.0008, 0.0166]
**P-value:** 0.074
**Sample Size:** 355,188

**Interpretation:** DACA eligibility is associated with a 0.79 percentage point increase in full-time employment probability. The effect is not statistically significant at the 5% level but is marginally significant at the 10% level.

### 2.5 Heterogeneity Analysis

| Subgroup | Coefficient | SE | N |
|----------|------------|-----|---|
| Male | -0.0131 | 0.0056 | 196,742 |
| Female | 0.0282 | 0.0068 | 158,446 |
| Ages 16-24 | 0.0232 | 0.0088 | 86,700 |
| Ages 25-32 | 0.0180 | 0.0081 | 126,103 |
| Ages 33-40 | 0.0239 | 0.0067 | 142,385 |

**Key Finding:** Strong positive effect for women (+2.8 pp), negative effect for men (-1.3 pp).

### 2.6 Robustness Checks

| Test | Coefficient | SE |
|------|------------|-----|
| Any Employment (instead of full-time) | 0.0251 | 0.0044 |
| Including 2012 | 0.0078 | 0.0044 |
| Ages 18-35 only | 0.0095 | 0.0051 |
| Unweighted | 0.0099 | 0.0037 |

### 2.7 Event Study (Parallel Trends)

| Year | Coefficient | SE | Significant? |
|------|------------|-----|--------------|
| 2006 | 0.0119 | 0.0099 | No |
| 2007 | 0.0062 | 0.0096 | No |
| 2008 | 0.0139 | 0.0097 | No |
| 2009 | 0.0183 | 0.0096 | No |
| 2010 | 0.0133 | 0.0094 | No |
| 2011 | 0.0000 | --- | (reference) |
| 2013 | 0.0107 | 0.0095 | No |
| 2014 | 0.0145 | 0.0096 | No |
| 2015 | 0.0253 | 0.0096 | Yes (p<0.05) |
| 2016 | 0.0316 | 0.0098 | Yes (p<0.01) |

**Interpretation:** Pre-treatment coefficients not significantly different from zero, supporting parallel trends assumption. Effects grow over time post-treatment.

---

## Session 3: Report Generation

### 3.1 LaTeX Report

**File:** `replication_report_52.tex`
**Compiled:** `replication_report_52.pdf` (24 pages)

**Contents:**
1. Abstract
2. Introduction
3. Background (DACA program, theoretical mechanisms)
4. Data (source, sample construction, variable definitions)
5. Empirical Strategy (DiD design, identification assumptions)
6. Results (summary stats, main results, event study, heterogeneity, robustness)
7. Discussion (interpretation, limitations)
8. Conclusion
9. Appendix (variable definitions, sample construction details)

### 3.2 Commands for PDF Generation

```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_52"
pdflatex -interaction=nonstopmode replication_report_52.tex
pdflatex -interaction=nonstopmode replication_report_52.tex  # Second pass for TOC
pdflatex -interaction=nonstopmode replication_report_52.tex  # Third pass for cross-refs
```

---

## Output Files Created

| File | Description |
|------|-------------|
| `analysis.py` | Main Python analysis script |
| `summary_statistics.csv` | Summary stats by group and period |
| `main_regression_results.csv` | DiD regression results |
| `event_study_results.csv` | Year-by-year treatment effects |
| `robustness_results.csv` | Robustness check results |
| `heterogeneity_results.csv` | Subgroup analysis results |
| `preferred_estimate.txt` | Summary of preferred estimate |
| `replication_report_52.tex` | LaTeX source for report |
| `replication_report_52.pdf` | Compiled 24-page report |
| `run_log_52.md` | This log file |

---

## Final Summary

**Research Question:** Effect of DACA eligibility on full-time employment

**Main Finding:** DACA eligibility associated with 0.79 pp increase in full-time employment (p=0.074, not significant at 5% level)

**Key Insights:**
1. Raw DiD (10.2 pp) dramatically overstates the effect; controls for age and demographics reduce it to <1 pp
2. Effects concentrated among women (+2.8 pp, significant) vs. men (-1.3 pp)
3. Effects grow over time, reaching 3.2 pp by 2016 (significant)
4. Parallel trends assumption appears satisfied (pre-treatment coefficients not significant)
5. Stronger effect on any employment (+2.5 pp) than full-time specifically

**Limitations:**
- Cannot distinguish documented vs undocumented non-citizens
- Eligibility is imputed, not directly observed
- Pre-trends show some positive pattern (though not significant)

---

## Deliverables Verification

- [x] `replication_report_52.tex` - LaTeX source
- [x] `replication_report_52.pdf` - 24-page PDF report
- [x] `run_log_52.md` - This run log

All required deliverables exist in the specified folder.
