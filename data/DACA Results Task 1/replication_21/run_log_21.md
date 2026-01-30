# Run Log for DACA Replication Study (Replication 21)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (defined as usually working 35+ hours per week)?

---

## Session Log

### Step 1: Data Exploration
**Date:** 2026-01-25

1. Read replication_instructions.docx to understand the research task
2. Examined data files:
   - data.csv: 33,851,425 observations (ACS 2006-2016)
   - acs_data_dict.txt: IPUMS variable codebook
   - state_demo_policy.csv: Optional state-level data (not used)

### Step 2: Key Variable Identification

**Outcome Variable:**
- UHRSWORK: Usual hours worked per week
- Full-time employment defined as UHRSWORK >= 35

**Treatment/DACA Eligibility Criteria (per instructions):**
1. Arrived in US before 16th birthday (YRIMMIG - BIRTHYR < 16)
2. Had not yet had 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (YRIMMIG <= 2007)
4. Present in US on June 15, 2012 (assume all observed in ACS are present)
5. Not a citizen (CITIZEN == 3)

**Sample Restrictions:**
- Hispanic-Mexican ethnicity: HISPAN == 1 (Mexican)
- Born in Mexico: BPL == 200 or BPLD == 20000
- Non-citizen: CITIZEN == 3 (not a citizen)
- Working age: Focus on ages 16-45 (reasonable labor force participation ages)

### Step 3: Identification Strategy

**Approach: Difference-in-Differences (DiD)**

The causal effect of DACA eligibility on full-time employment is estimated using a difference-in-differences design:

1. **Treatment Group:** Mexican-born Hispanic non-citizens who meet DACA eligibility criteria
2. **Control Group:** Mexican-born Hispanic non-citizens who do NOT meet DACA eligibility criteria (e.g., arrived after age 15, or arrived after 2007)
3. **Pre-Period:** 2006-2011 (before DACA implementation)
4. **Post-Period:** 2013-2016 (after DACA implementation, excluding 2012)

**Key Assumption:** Parallel trends - absent DACA, both groups would have followed similar employment trajectories.

### Step 4: Analysis Plan

1. Load and clean data
2. Create DACA eligibility indicator
3. Create full-time employment indicator
4. Restrict sample to Mexican-born Hispanic non-citizens
5. Estimate DiD regression:
   - Y = fulltime_employment
   - D = DACA_eligible * Post2012
   - Controls: age, sex, education, state, year fixed effects
6. Cluster standard errors at state level
7. Conduct robustness checks

---

## Commands Executed

### Data Reading and Exploration
```bash
# Read replication instructions
python -c "from docx import Document; doc = Document('replication_instructions.docx'); [print(p.text) for p in doc.paragraphs]"

# Examine data files
ls data/
head -50 data/data.csv
wc -l data/data.csv
# Output: 33,851,425 rows

# Read data dictionary
cat data/acs_data_dict.txt
```

### Analysis Execution
```bash
# Run main analysis script
python analysis_21.py
```

### LaTeX Compilation
```bash
# Compile report (two passes for cross-references)
pdflatex -interaction=nonstopmode replication_report_21.tex
pdflatex -interaction=nonstopmode replication_report_21.tex
```

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sample frame | Mexican-born, Hispanic-Mexican, non-citizens | Per instructions: target population most affected by DACA |
| Age restriction | 16-45 | Working age population with reasonable labor force attachment |
| Control group | Non-eligible non-citizens (arrived >16 or arrived after 2007) | Comparable immigration status but ineligible for DACA |
| Identification | Difference-in-differences | Standard approach for policy evaluation with treatment/control groups |
| Post period | 2013-2016 | DACA implemented June 2012; exclude 2012 due to partial treatment |
| Standard errors | Clustered at state level | Account for within-state correlation in treatment and outcomes |
| Full-time definition | UHRSWORK >= 35 | Per instructions |

---

## Files Created

1. `analysis_21.py` - Main analysis script
2. `replication_report_21.tex` - LaTeX report (24 pages)
3. `replication_report_21.pdf` - Final PDF report
4. `run_log_21.md` - This log file
5. `results_21.pkl` - Pickled results dictionary
6. `descriptive_stats_21.csv` - Summary statistics

---

## Main Results

### Preferred Estimate (Model 5)
- **Effect:** 0.0239 (2.39 percentage points)
- **Standard Error:** 0.0049 (clustered at state level)
- **95% CI:** [0.0143, 0.0336]
- **P-value:** < 0.001
- **Sample Size:** 427,762
- **R-squared:** 0.228

### Interpretation
DACA eligibility is associated with a 2.39 percentage point increase in the probability of full-time employment (working 35+ hours per week) among Mexican-born Hispanic non-citizens. This effect is statistically significant at the 1% level.

### Sample Composition
| Group | Pre-2012 | Post-2012 | Total |
|-------|----------|-----------|-------|
| DACA Eligible | 46,814 | 36,797 | 83,611 |
| DACA Ineligible | 227,543 | 116,608 | 344,151 |
| **Total** | **274,357** | **153,405** | **427,762** |

### Full-Time Employment Rates
| Group | Pre-2012 | Post-2012 | Change |
|-------|----------|-----------|--------|
| DACA Eligible | 43.09% | 49.62% | +6.53 pp |
| DACA Ineligible | 61.66% | 58.95% | -2.71 pp |

### Simple Difference-in-Differences
DiD = (49.62% - 43.09%) - (58.95% - 61.66%) = 6.53% + 2.71% = **9.23 pp** (unadjusted)

After controlling for demographics, year FE, state FE: **2.39 pp** (adjusted)

---

## Robustness Checks Summary

| Specification | Coefficient | SE | p-value | N |
|--------------|-------------|-------|---------|-------|
| Main (preferred) | 0.0239 | 0.0049 | <0.001 | 427,762 |
| Employed only | -0.0120 | 0.0027 | <0.001 | 276,085 |
| Age 18-35 | 0.0096 | 0.0059 | 0.107 | 253,373 |
| Males only | 0.0049 | 0.0044 | 0.262 | 234,520 |
| Females only | 0.0417 | 0.0079 | <0.001 | 193,242 |
| Placebo (pre-2010) | 0.0107 | 0.0033 | 0.001 | 274,357 |

---

## Event Study Results (Reference Year: 2011)

| Year | Coefficient | SE | Significant |
|------|-------------|-------|-------------|
| 2006 | -0.0101 | 0.0067 | No |
| 2007 | -0.0106 | 0.0054 | ** |
| 2008 | 0.0002 | 0.0088 | No |
| 2009 | 0.0034 | 0.0064 | No |
| 2010 | 0.0043 | 0.0100 | No |
| 2011 | (reference) | - | - |
| 2013 | 0.0056 | 0.0077 | No |
| 2014 | 0.0176 | 0.0119 | No |
| 2015 | 0.0330 | 0.0096 | *** |
| 2016 | 0.0342 | 0.0086 | *** |

---

## Session Completion
**Date:** 2026-01-25
**Status:** Complete

All deliverables generated:
- [x] replication_report_21.tex
- [x] replication_report_21.pdf
- [x] run_log_21.md
