# DACA Replication Study Run Log

## Study Overview
**Research Question**: Among ethnically Hispanic-Mexican, Mexican-born people in the US, what was the causal impact of DACA eligibility on full-time employment (35+ hours/week)?

**Design**: Difference-in-differences comparing:
- Treatment group: Ages 26-30 at policy implementation (June 15, 2012)
- Control group: Ages 31-35 at policy implementation (would be eligible but for age)

**Post-treatment period**: 2013-2016

---

## Session Log

### Step 1: Data Exploration
- Read replication instructions from .docx file
- Examined ACS data dictionary (acs_data_dict.txt)
- Data file: data.csv with 33,851,424 observations (plus header)
- Data spans: ACS 2006-2016

### Key Variables Identified:
- `YEAR`: Census year (2006-2016)
- `BIRTHYR`: Birth year (for age calculation)
- `HISPAN` / `HISPAND`: Hispanic origin (1 = Mexican)
- `BPL` / `BPLD`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `UHRSWORK`: Usual hours worked per week (outcome: 35+ = full-time)
- `EMPSTAT`: Employment status (1 = employed)
- `PERWT`: Person weight for survey weighting

### Step 2: Sample Definition Decisions

**DACA Eligibility Criteria (from instructions)**:
1. Born in Mexico (BPL = 200)
2. Hispanic-Mexican ethnicity (HISPAN = 1)
3. Not a citizen (CITIZEN = 3)
4. Arrived unlawfully before 16th birthday - approximated by YRIMMIG <= BIRTHYR + 15
5. Lived in US since June 15, 2007 - approximated by YRIMMIG <= 2007
6. Age constraint for treatment assignment (based on age as of June 15, 2012)

**Treatment Group**: Born 1982-1986 (ages 26-30 on June 15, 2012)
**Control Group**: Born 1977-1981 (ages 31-35 on June 15, 2012)

**Pre-treatment Period**: 2006-2011 (excluding 2012 due to mid-year implementation)
**Post-treatment Period**: 2013-2016

### Step 3: Analytical Approach

Using a difference-in-differences (DiD) framework:
- Compare change in full-time employment rates pre/post DACA
- Between treatment group (DACA-eligible by age) vs control group (ineligible by age)
- Will use survey weights (PERWT) for population-representative estimates
- Standard errors clustered at state level for inference

---

## Commands Executed

### Data Loading and Processing
```python
# Read data in chunks, filtering for Mexican-born, Hispanic-Mexican, non-citizens
# Excluded 2012 (policy implementation year)
# Applied DACA eligibility filters:
#   - arrived_before_16: YRIMMIG <= BIRTHYR + 15
#   - in_us_since_2007: YRIMMIG <= 2007
```

### Sample Sizes After Filters:
- Initial filtered (Mexican-born, Hispanic-Mexican, non-citizen): 636,722
- After age filter (26-35 in 2012): 162,283
- After DACA eligibility filters: **44,725** (final sample)

### Outcome Variable:
```python
fulltime = (UHRSWORK >= 35) & (EMPSTAT == 1)
```

---

## Key Results

### Summary Statistics (Weighted)
| Group | Period | N | Full-Time Rate |
|-------|--------|---|----------------|
| Control (31-35) | Pre | 11,916 | 0.611 |
| Control (31-35) | Post | 6,218 | 0.598 |
| Treatment (26-30) | Pre | 17,410 | 0.560 |
| Treatment (26-30) | Post | 9,181 | 0.620 |

### Simple DiD Calculation
- Treatment change: 0.620 - 0.560 = +0.060
- Control change: 0.598 - 0.611 = -0.013
- **DiD = 0.073 (7.3 percentage points)**

### Regression Results

**Model 1 (Basic DiD):**
- DiD coefficient: 0.0731 (SE: 0.0084, p < 0.001)

**Model 2 (+ Demographics):**
- DiD coefficient: 0.0592 (SE: 0.0101, p < 0.001)

**Model 3 (+ Year/State FE) - PREFERRED:**
- DiD coefficient: **0.0578** (SE: 0.0103, p < 0.001)
- 95% CI: [0.0375, 0.0780]

### Pre-Trend Analysis
- Slope of pre-period gaps: 0.0059/year
- p-value: 0.156 (not significant)
- **Parallel trends assumption supported**

### Robustness Checks

**Alternative Bandwidths:**
- Ages 27-29 vs 32-34: DiD = 0.042 (SE: 0.009)

**By Gender:**
- Men: DiD = 0.066 (SE: 0.014)
- Women: DiD = 0.033 (SE: 0.017)

---

## Key Decisions Made

1. **Excluded 2012**: DACA implemented mid-year; ACS doesn't record survey month
2. **Used CITIZEN = 3 as proxy for undocumented**: Following instructions
3. **Clustered SEs at state level**: Accounts for within-state correlation and state-level policies
4. **Survey weights (PERWT)**: For population-representative estimates
5. **Full-time = 35+ hours**: Standard definition per BLS

---

## Files Generated

1. `analysis.py` - Main analysis script
2. `create_figures.py` - Figure generation
3. `figure1_event_study.png/pdf` - Event study plot
4. `figure2_gap_plot.png/pdf` - Treatment-control gap over time
5. `figure3_did_bars.png/pdf` - DiD visualization
6. `event_study_data.csv` - Year-by-year data
7. `summary_stats.csv` - Group means
8. `replication_report_87.tex` - LaTeX report
9. `replication_report_87.pdf` - Final PDF report (26 pages)
10. `run_log_87.md` - This file

---

## Preferred Estimate Summary

**Effect of DACA eligibility on full-time employment:**
- **Point estimate: 5.78 percentage points**
- **Standard error: 0.0103**
- **95% CI: [3.75, 7.80] percentage points**
- **Sample size: 44,725**
- **Number of state clusters: 51**

This represents approximately a 10% increase relative to the pre-treatment baseline of 56%.

---

## Session Complete
All deliverables generated successfully.
