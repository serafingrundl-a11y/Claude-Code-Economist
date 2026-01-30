# DACA Replication Analysis - Run Log

## Project Information
- **Task ID:** 68
- **Research Question:** Effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals
- **Date:** 2026-01-26

---

## 1. Data Sources

### Primary Data
- **Source:** American Community Survey (ACS) via IPUMS USA
- **File:** `data/data.csv` (6.27 GB)
- **Years:** 2006-2016 (1-year ACS files)
- **Total observations:** 33,851,424

### Supplementary Files
- **Data dictionary:** `data/acs_data_dict.txt`
- **State-level data:** `data/state_demo_policy.csv` (not used in analysis)

---

## 2. Key Variable Definitions

### Target Population
| Variable | IPUMS Name | Values Used |
|----------|------------|-------------|
| Hispanic-Mexican | HISPAN | = 1 (Mexican) |
| Born in Mexico | BPL | = 200 (Mexico) |

### DACA Eligibility Criteria
| Criterion | Construction |
|-----------|--------------|
| Arrived before age 16 | YRIMMIG - BIRTHYR < 16 |
| In U.S. since 2007 | YRIMMIG <= 2007 |
| Non-citizen | CITIZEN in {3, 4, 5} |

### Treatment/Control Groups
| Group | Birth Years | Age in 2012 |
|-------|-------------|-------------|
| Treatment | 1982-1986 | 26-30 (DACA eligible) |
| Control | 1977-1981 | 31-35 (Too old for DACA) |

### Time Periods
| Period | Years | Notes |
|--------|-------|-------|
| Pre-DACA | 2006-2011 | Before policy |
| Post-DACA | 2013-2016 | After implementation |
| Excluded | 2012 | Mid-year implementation |

### Outcome Variable
- **Full-time employment:** UHRSWORK >= 35 hours per week

---

## 3. Sample Construction

| Step | Observations | Notes |
|------|--------------|-------|
| Raw ACS data | 33,851,424 | All persons 2006-2016 |
| Hispanic-Mexican & Mexican-born | 991,261 | HISPAN=1 & BPL=200 |
| DACA eligible (base criteria) | 195,023 | Met arrival, residency, citizenship |
| Treatment or control group | 44,725 | Birth year 1977-1986 |
| Final analysis sample | 44,725 | Excluding 2012 |

### Sample by Group and Period
|  | Pre-DACA | Post-DACA | Total |
|--|----------|-----------|-------|
| Treatment | 17,410 | 9,181 | 26,591 |
| Control | 11,916 | 6,218 | 18,134 |
| Total | 29,326 | 15,399 | 44,725 |

---

## 4. Analysis Conducted

### Main Models
1. **Model 1:** Basic DiD (no controls)
2. **Model 2:** DiD + demographics (sex, marital status, age, age^2)
3. **Model 3:** DiD + demographics + education
4. **Model 4:** DiD + demographics + education + state FE + year FE (PREFERRED)

### Robustness Checks
- Alternative outcome: Any employment
- Heterogeneous effects by gender
- Placebo test with older cohorts (ages 31-35 vs 36-40)
- Event study specification

---

## 5. Main Results

### Preferred Specification (Model 4)
| Statistic | Value |
|-----------|-------|
| Effect size | 0.0141 (1.41 pp) |
| Standard error | 0.0132 |
| 95% CI | [-0.0117, 0.0399] |
| p-value | 0.285 |
| Sample size | 44,725 |

### Comparison Across Specifications
| Model | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| Model 1 (Basic) | 0.0551 | 0.0098 | <0.001 |
| Model 2 (+ Demo) | 0.0663 | 0.0125 | <0.001 |
| Model 3 (+ Educ) | 0.0657 | 0.0124 | <0.001 |
| Model 4 (+ FE) | 0.0141 | 0.0132 | 0.285 |

### Robustness Results
| Check | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| Any employment | 0.0291 | 0.0129 | 0.024 |
| Males only | 0.0240 | 0.0163 | 0.141 |
| Females only | -0.0098 | 0.0210 | 0.641 |
| Placebo test | -0.0041 | 0.0149 | 0.785 |

### Event Study (Relative to 2011)
| Year | Coefficient | SE |
|------|-------------|-----|
| 2006 | -0.0025 | 0.0205 |
| 2007 | -0.0020 | 0.0198 |
| 2008 | 0.0197 | 0.0195 |
| 2009 | 0.0070 | 0.0197 |
| 2010 | -0.0049 | 0.0193 |
| 2013 | 0.0189 | 0.0202 |
| 2014 | 0.0125 | 0.0207 |
| 2015 | 0.0138 | 0.0216 |
| 2016 | 0.0232 | 0.0222 |

---

## 6. Key Decisions and Justifications

### Decision 1: Non-citizen proxy for undocumented status
- **Choice:** Included CITIZEN values 3, 4, and 5 as potentially undocumented
- **Justification:** ACS does not identify undocumented status; following research task guidance to treat non-citizens without papers as potentially undocumented

### Decision 2: Exclusion of 2012
- **Choice:** Dropped all 2012 observations
- **Justification:** DACA was implemented June 15, 2012 (mid-year); ACS does not identify interview month, so cannot distinguish pre/post

### Decision 3: Age-based treatment/control groups
- **Choice:** Used 5-year birth cohorts (1982-1986 vs 1977-1981)
- **Justification:** Per research task specification; provides sufficient sample size while maintaining reasonable comparability

### Decision 4: State and year fixed effects
- **Choice:** Included as preferred specification
- **Justification:** Controls for geographic and temporal heterogeneity; results are sensitive to inclusion, suggesting these factors are important confounders

### Decision 5: Full-time definition
- **Choice:** UHRSWORK >= 35 hours per week
- **Justification:** Standard definition of full-time employment used by BLS and in research task

---

## 7. Files Created

### Analysis Scripts
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script

### Output Files
- `analysis_results.json` - Key statistics
- `summary_statistics.csv` - Summary stats by group/period

### Figures
- `figure1_trends.pdf` / `.png` - Employment trends by group
- `figure2_eventstudy.pdf` / `.png` - Event study coefficients
- `figure3_models.pdf` / `.png` - Model comparison
- `figure4_gender.pdf` / `.png` - Gender heterogeneity

### Report
- `replication_report_68.tex` - LaTeX source
- `replication_report_68.pdf` - Final report (23 pages)

---

## 8. Interpretation Summary

DACA eligibility is associated with a **1.4 percentage point increase** in the probability of full-time employment in the preferred specification. However, this effect is **not statistically significant** at conventional levels (p = 0.285).

Key findings:
1. Results are sensitive to fixed effects inclusion - larger effects without FE
2. No evidence of differential pre-trends (supports parallel trends assumption)
3. Effects on any employment are larger and significant (2.9 pp, p = 0.024)
4. Possible heterogeneity by gender (positive for males, negative for females)
5. Placebo test shows null effect for ineligible cohorts (supports design validity)

The wide confidence interval [-1.2, 4.0 pp] means uncertainty remains about the true effect. The point estimate suggests a modest positive effect, but we cannot rule out a null or small negative effect.

---

## 9. Software/Environment

- **Language:** Python 3.x
- **Key packages:** pandas, numpy, statsmodels, matplotlib
- **LaTeX:** pdflatex (MiKTeX)
- **Platform:** Windows

---

## 10. Reproducibility

To reproduce the analysis:
1. Ensure `data/data.csv` is in the data folder
2. Run `python analysis.py` to generate results
3. Run `python create_figures.py` to generate figures
4. Compile LaTeX: `pdflatex replication_report_68.tex` (run 3x for references)

All code runs from a clean session and produces identical results.
