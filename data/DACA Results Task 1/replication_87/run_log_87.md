# Run Log - DACA Replication Study #87

## Project Overview
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

**Time Period:** Effects examined for 2013-2016 (DACA implemented June 15, 2012)

---

## Session Log

### Session Start: 2026-01-25

#### Step 1: Data Exploration
- **Data Files:**
  - `data.csv` - Main ACS data file (6.2 GB, 33,851,424 observations)
  - `acs_data_dict.txt` - Data dictionary
  - `state_demo_policy.csv` - Optional state-level supplemental data (not used)

- **ACS Samples Included:** 2006-2016 (1-year ACS files)

- **Key Variables Identified:**
  - `YEAR` - Census year
  - `BIRTHYR` - Birth year
  - `BIRTHQTR` - Birth quarter (1-4)
  - `HISPAN` - Hispanic origin (1 = Mexican)
  - `HISPAND` - Detailed Hispanic origin (100-107 for Mexican)
  - `BPL` - Birthplace (200 = Mexico)
  - `BPLD` - Detailed birthplace (20000 = Mexico)
  - `CITIZEN` - Citizenship status (3 = Not a citizen)
  - `YRIMMIG` - Year of immigration
  - `UHRSWORK` - Usual hours worked per week
  - `PERWT` - Person weight
  - `AGE` - Age
  - `SEX` - Sex (1 = Male, 2 = Female)
  - `STATEFIP` - State FIPS code

#### Step 2: DACA Eligibility Criteria Definition
Based on the replication instructions, DACA eligibility requires:
1. Arrived in the US before 16th birthday: `(YRIMMIG - BIRTHYR) < 16`
2. Had not yet had 31st birthday as of June 15, 2012: `BIRTHYR >= 1982` OR `(BIRTHYR == 1981 AND BIRTHQTR >= 3)`
3. Lived continuously in US since June 15, 2007: `YRIMMIG <= 2007`
4. Present in US on June 15, 2012 (assumed for those in sample)
5. Did not have lawful status: `CITIZEN = 3`

**Sample Restrictions Applied Sequentially:**
| Restriction | N Remaining | N Dropped |
|------------|-------------|-----------|
| Full ACS (2006-2016) | 33,851,424 | - |
| Hispanic-Mexican (HISPAN=1) | 2,945,521 | 30,905,903 |
| Born in Mexico (BPL=200) | 991,261 | 1,954,260 |
| Non-citizen (CITIZEN=3) | 701,347 | 289,914 |
| Exclude 2012 | 636,722 | 64,625 |
| Working age (16-64) | 561,470 | 75,252 |

**Analysis Sample:**
- Treatment (DACA eligible): 83,611
- Control (too old for DACA): 54,881
- **Total analysis sample: 138,492**

#### Step 3: Identification Strategy
**Method:** Difference-in-Differences (DiD)
- **Treatment Group:** DACA-eligible individuals
- **Control Group:** Childhood arrivals (arrived before age 16) who were too old for DACA (born before June 1981), with continuous US presence since 2007
- **Pre-Period:** 2006-2011 (before DACA)
- **Post-Period:** 2013-2016 (after DACA, excluding 2012 due to timing ambiguity)

**Outcome Variable:** Full-time employment = 1 if UHRSWORK >= 35, 0 otherwise

---

## Key Decisions and Rationale

### Decision 1: Control Group Selection
**Choice:** Use "too old" childhood arrivals as control group
**Rationale:** These individuals share the experience of arriving as children and being undocumented, but were ineligible for DACA solely due to their birth year. This provides a cleaner comparison than late arrivals or other potential control groups.

### Decision 2: Exclusion of 2012
**Choice:** Exclude year 2012 from analysis
**Rationale:** DACA was announced June 15, 2012 and applications began August 15, 2012. The ACS does not identify month of interview, so 2012 observations could be pre- or post-treatment.

### Decision 3: Age Restriction
**Choice:** Restrict sample to ages 16-64
**Rationale:** Standard working-age population definition to focus on labor market participants.

### Decision 4: Weighting
**Choice:** Use ACS person weights (PERWT) in all analyses
**Rationale:** Produces population-representative estimates.

### Decision 5: Standard Errors
**Choice:** Use heteroskedasticity-robust (HC1) standard errors
**Rationale:** Accounts for potential heteroskedasticity in the linear probability model.

---

## Analysis Commands

### Command 1: Run Main Analysis
```bash
python analysis.py
```
**Output:** Full analysis including summary statistics, DiD regressions, event study, and robustness checks.

### Command 2: Generate Figures
```bash
python create_figures.py
```
**Output:** 5 figures (PNG and PDF formats)

### Command 3: Compile LaTeX Report
```bash
pdflatex -interaction=nonstopmode replication_report_87.tex
pdflatex -interaction=nonstopmode replication_report_87.tex
pdflatex -interaction=nonstopmode replication_report_87.tex
```
**Output:** `replication_report_87.pdf` (20 pages)

---

## Results Summary

### Main Results
| Specification | DiD Estimate | Std. Error | p-value |
|--------------|--------------|------------|---------|
| Basic DiD (unweighted) | 0.107 | 0.006 | <0.001 |
| Basic DiD (weighted) | 0.110 | 0.006 | <0.001 |
| With controls | 0.020 | 0.005 | <0.001 |
| With year FE | 0.010 | 0.005 | 0.065 |
| **Preferred (year + state FE, robust SE)** | **0.009** | **0.006** | **0.166** |

### Preferred Estimate
- **Effect Size:** 0.0087 (0.87 percentage points)
- **Standard Error:** 0.0063
- **95% CI:** [-0.0036, 0.0209]
- **p-value:** 0.166
- **Sample Size:** 138,492

### Interpretation
DACA eligibility is associated with a 0.87 percentage point increase in the probability of full-time employment, but this effect is not statistically significant at the 5% level. The event study shows no evidence of differential pre-trends.

---

## Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `replication_report_87.tex` | LaTeX report source |
| `replication_report_87.pdf` | Final PDF report (20 pages) |
| `results_summary.csv` | Summary of main results |
| `event_study_results.csv` | Event study coefficients |
| `summary_statistics.csv` | Summary statistics by group |
| `figure1_event_study.png/pdf` | Event study plot |
| `figure2_parallel_trends.png/pdf` | Parallel trends plot |
| `figure3_did_visual.png/pdf` | DiD visualization |
| `figure4_age_distribution.png/pdf` | Age distributions |
| `figure5_robustness.png/pdf` | Robustness comparison |
| `run_log_87.md` | This run log |

---

## Session End: 2026-01-25

**Status:** Complete
**All required deliverables generated:**
- [x] `replication_report_87.tex`
- [x] `replication_report_87.pdf`
- [x] `run_log_87.md`
