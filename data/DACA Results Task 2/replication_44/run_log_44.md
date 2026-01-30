# Run Log for DACA Replication Study (Replication 44)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (working 35+ hours per week)?

## Identification Strategy
- **Treatment Group**: Ages 26-30 at DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at DACA implementation (would have been eligible but for age)
- **Method**: Difference-in-Differences comparing pre-period (2006-2011) to post-period (2013-2016)
- **Note**: 2012 is excluded as it contains both pre and post observations that cannot be distinguished

---

## Session Log

### Date: 2026-01-26

### Step 1: Data Exploration

**Command**: Examined data folder contents
- Found `data.csv` (6.26 GB) - main ACS data file
- Found `acs_data_dict.txt` - data dictionary with variable definitions
- Found `state_demo_policy.csv` - optional supplemental state-level data

**Key Variables Identified**:
- YEAR: Survey year (2006-2016)
- BIRTHYR: Birth year
- BIRTHQTR: Birth quarter (1-4)
- HISPAN/HISPAND: Hispanic origin (1=Mexican, 100-107 for detailed Mexican)
- BPL/BPLD: Birthplace (200=Mexico, 20000=Mexico detailed)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week (35+ = full-time)
- EMPSTAT: Employment status (1=Employed)
- PERWT: Person weight for representative estimates

### Step 2: Sample Definition Decisions

**DACA Eligibility Criteria** (per instructions):
1. Arrived unlawfully in US before 16th birthday
2. Had not yet turned 31 as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

**Operationalization**:
- Hispanic-Mexican: HISPAN == 1 (Mexican)
- Born in Mexico: BPL == 200
- Not a citizen: CITIZEN == 3 (proxy for undocumented)
- Arrived before age 16: (YRIMMIG - BIRTHYR) < 16

**Age Groups** (as of June 15, 2012):
- Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)
- Control: Born 1977-1981 (ages 31-35 on June 15, 2012)

### Step 3: Python Analysis Script Creation

Creating comprehensive analysis script to:
1. Load and filter the data
2. Create treatment/control indicators
3. Run difference-in-differences regression
4. Generate summary statistics and visualizations
5. Conduct robustness checks

**Command**: `python analysis.py`

### Step 4: Analysis Results

**Sample Selection Results**:
- Initial load (Hispanic-Mexican, born in Mexico): 991,261 observations
- After non-citizen filter: 701,347 (70.8%)
- After arrived before age 16: 205,327
- After age group filter (26-35): 47,418
- After excluding 2012: 43,238 (final analytic sample)

**Sample Breakdown**:
- Treatment group (26-30): 25,470 observations
- Control group (31-35): 17,768 observations
- Pre-period (2006-2011): 28,377 observations
- Post-period (2013-2016): 14,861 observations

### Step 5: Key Findings

**Main Difference-in-Differences Estimate** (Preferred Specification with Year and State FE):
- Coefficient: 0.0460
- Standard Error: 0.0107
- 95% CI: [0.0251, 0.0669]
- p-value: < 0.0001

**Interpretation**: DACA eligibility is associated with a 4.6 percentage point increase in the probability of full-time employment among eligible Hispanic-Mexican individuals born in Mexico.

**Robustness Checks**:
1. Intensive margin (employed only): 0.0236 (SE: 0.0105)
2. Males only: 0.0330 (SE: 0.0124)
3. Females only: 0.0487 (SE: 0.0182)
4. Placebo test (2009-2011 vs 2006-2008): -0.0001 (p=0.99) - no pre-trends
5. Narrower bandwidth (27-29 vs 32-34): 0.0397 (SE: 0.0120)
6. Wider bandwidth (24-30 vs 31-37): 0.0763 (SE: 0.0090)

### Step 6: Output Files Generated

1. `results_summary.json` - Key results in JSON format
2. `event_study_results.csv` - Event study coefficients
3. `did_table.csv` - 2x2 DiD table
4. `summary_statistics.csv` - Descriptive statistics

### Step 7: LaTeX Report Generation

Creating ~20-page replication report with:
- Abstract and Introduction
- Background on DACA
- Data and Methods
- Results (main and robustness)
- Discussion and Conclusion

**Commands**:
```bash
pdflatex -interaction=nonstopmode replication_report_44.tex
pdflatex -interaction=nonstopmode replication_report_44.tex  # Second pass for refs
pdflatex -interaction=nonstopmode replication_report_44.tex  # Third pass for ToC
```

---

## Deliverables

All required output files have been generated:

| File | Description | Size |
|------|-------------|------|
| `replication_report_44.tex` | LaTeX source for the replication report | 32 KB |
| `replication_report_44.pdf` | Final PDF report (18 pages) | 307 KB |
| `run_log_44.md` | This run log file | 4 KB |

### Additional Analysis Files

| File | Description |
|------|-------------|
| `analysis.py` | Python script for the difference-in-differences analysis |
| `results_summary.json` | Key results in JSON format |
| `event_study_results.csv` | Event study coefficients by year |
| `did_table.csv` | 2x2 difference-in-differences table |
| `summary_statistics.csv` | Descriptive statistics by treatment group |

---

## Key Decisions Summary

1. **Sample Definition**: Used HISPAN=1 (Mexican) and BPL=200 (born in Mexico) to identify Hispanic-Mexican individuals born in Mexico.

2. **Proxy for Undocumented Status**: Used CITIZEN=3 (not a citizen) as a proxy since ACS does not directly identify undocumented status.

3. **Age Calculation**: Calculated exact age on June 15, 2012 using birth year and quarter to assign treatment/control status.

4. **Pre-Period Definition**: Used 2006-2011 as pre-period; excluded 2012 because DACA was implemented mid-year.

5. **Preferred Specification**: Model with year and state fixed effects, weighted by PERWT with robust standard errors.

6. **Full-Time Definition**: Defined as UHRSWORK >= 35 hours per week, following BLS convention.

---

## Session Complete

All tasks completed successfully. The replication study finds that DACA eligibility is associated with a statistically significant 4.6 percentage point increase in full-time employment (SE=0.0107, p<0.001).
