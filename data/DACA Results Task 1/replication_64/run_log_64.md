# Run Log - DACA Replication Study (Replication 64)

## Overview
This log documents all commands executed and key decisions made during the independent replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

---

## Session Start: Analysis Planning

### Data Files Available
- `data/data.csv` - Main ACS data file (33,851,424 observations + header)
- `data/acs_data_dict.txt` - Data dictionary for ACS variables
- `data/state_demo_policy.csv` - State-level demographic and policy data
- `data/State Level Data Documentation.docx` - Documentation for state data

### Key Variables Identified from Data Dictionary
- **YEAR**: Census/survey year (2006-2016)
- **HISPAN/HISPAND**: Hispanic origin (1=Mexican, 100-107=Mexican detailed)
- **BPL/BPLD**: Birthplace (200/20000=Mexico)
- **CITIZEN**: Citizenship status (3=Not a citizen)
- **YRIMMIG**: Year of immigration
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth (1=Q1, 2=Q2, 3=Q3, 4=Q4)
- **AGE**: Age at survey
- **UHRSWORK**: Usual hours worked per week
- **EMPSTAT**: Employment status
- **PERWT**: Person weight

### DACA Eligibility Criteria (from instructions)
To be DACA-eligible, a person must have:
1. Arrived in the US before their 16th birthday
2. Not yet had their 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007
4. Been present in the US on June 15, 2012 and not have lawful status
5. Be non-citizen (CITIZEN=3) who has not received immigration papers (assume undocumented)

### Identification Strategy Decision
**Approach**: Difference-in-Differences (DiD)
- **Treatment group**: DACA-eligible Hispanic-Mexican, Mexican-born non-citizens
- **Control group**: Similar individuals who do not meet DACA eligibility criteria (e.g., those who arrived after age 16 or those too old for DACA)
- **Pre-period**: 2006-2011 (DACA announced June 2012, exclude 2012 as transition year)
- **Post-period**: 2013-2016 (as specified in instructions)

### Outcome Variable
- **Full-time employment**: Binary indicator = 1 if UHRSWORK >= 35

---

## Analysis Code Execution

### Step 1: Data Loading and Filtering
```bash
# Load ACS data from data/data.csv
# 33,851,424 total observations
```

### Step 2: Sample Restrictions
Applied sequentially:

| Filter | Observations Remaining |
|--------|----------------------|
| Full ACS 2006-2016 | 33,851,424 |
| Hispanic-Mexican ethnicity (HISPAN=1 or HISPAND 100-107) | 2,945,521 |
| Born in Mexico (BPL=200 or BPLD=20000) | 991,261 |
| Non-citizens (CITIZEN=3) | 701,347 |
| Exclude 2012 (transition year) | 636,722 |
| Ages 18-64 | 547,614 |

**Final analysis sample: 547,614 observations**

### Step 3: DACA Eligibility Construction
```python
# Age at arrival
age_at_arrival = YRIMMIG - BIRTHYR

# Criterion 1: Arrived before 16th birthday
arrived_before_16 = (age_at_arrival < 16)

# Criterion 2: Born after June 15, 1981 (not yet 31 on June 15, 2012)
# Using birth quarter: Q3-Q4 (July-Dec) of 1981 is after June 15
born_after_cutoff = (BIRTHYR > 1981) | ((BIRTHYR == 1981) & (BIRTHQTR >= 3))

# Criterion 3: Arrived by 2007 (continuous presence since June 15, 2007)
arrived_by_2007 = (YRIMMIG <= 2007)

# DACA eligible = all three criteria met
daca_eligible = arrived_before_16 & born_after_cutoff & arrived_by_2007
```

### Step 4: Sample by Treatment Status
| Group | Pre-DACA (2006-2011) | Post-DACA (2013-2016) | Total |
|-------|---------------------|----------------------|-------|
| DACA Ineligible | 298,245 | 178,022 | 476,267 |
| DACA Eligible | 38,248 | 33,099 | 71,347 |
| **Total** | **336,493** | **211,121** | **547,614** |

---

## Key Analytical Decisions

### Decision 1: Exclusion of 2012
**Rationale**: DACA was implemented on June 15, 2012. The ACS does not record the month of data collection, so 2012 observations could be from before or after implementation. Excluding 2012 provides cleaner identification of pre- and post-treatment periods.

### Decision 2: Non-Citizen Restriction
**Rationale**: Per instructions, we assume non-citizens who have not received immigration papers are undocumented. CITIZEN=3 identifies non-citizens. Citizens (CITIZEN in {0,1,2}) are excluded because they either have legal status or are naturalized.

### Decision 3: Age Restriction (18-64)
**Rationale**: Focus on working-age population for whom employment outcomes are most relevant. Excludes children (under 18) and typical retirement ages (65+).

### Decision 4: Birth Quarter Interpretation
**Rationale**: For the June 15, 1981 cutoff, individuals born in Q1-Q2 (Jan-June) 1981 would be 31+ on June 15, 2012. Individuals born in Q3-Q4 (July-Dec) 1981 would not yet be 31 on June 15, 2012. Thus, BIRTHYR=1981 and BIRTHQTR>=3 qualifies.

### Decision 5: Control Group Definition
**Rationale**: Control group consists of Hispanic-Mexican, Mexican-born non-citizens who do NOT meet DACA eligibility criteria. This could be due to: (a) arriving at age 16+, (b) being born before July 1981, or (c) arriving after 2007.

### Decision 6: Standard Errors
**Rationale**: Clustered at state level to account for within-state correlation in employment outcomes over time due to state-specific labor market conditions and policies.

### Decision 7: Survey Weights
**Rationale**: Used PERWT (person weights) in all regressions to produce nationally representative estimates.

---

## Commands Executed

### Python Analysis Script
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_64"
python analysis.py
```

### Figure Generation
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_64"
python create_figures.py
```

### LaTeX Compilation
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_64"
pdflatex -interaction=nonstopmode replication_report_64.tex  # Run 3 times for cross-references
```

---

## Main Results Summary

### Preferred Specification (Model 4: Year + State FE)

| Parameter | Value |
|-----------|-------|
| **DiD Coefficient (DACA x Post)** | **0.0194** |
| Standard Error | 0.0037 |
| 95% Confidence Interval | [0.0121, 0.0267] |
| t-statistic | 5.21 |
| p-value | < 0.001 |
| Sample Size | 547,614 |

**Interpretation**: DACA eligibility increased full-time employment by 1.94 percentage points, representing a 3.8% increase relative to the pre-treatment mean of 51.0%.

### Robustness Checks

| Specification | Coefficient | SE | p-value | N |
|--------------|-------------|-----|---------|---|
| Ages 18-40 | 0.0139 | 0.0050 | 0.006 | 341,332 |
| Males Only | 0.0115 | 0.0052 | 0.026 | 296,109 |
| Females Only | 0.0197 | 0.0067 | 0.004 | 251,505 |
| Any Employment | 0.0303 | 0.0058 | <0.001 | 547,614 |

### Event Study Results
Pre-DACA coefficients (2006-2010) are all statistically insignificant, supporting parallel trends assumption. Post-DACA coefficients show increasing effects over time:
- 2013: 0.012 (p=0.30)
- 2014: 0.025 (p=0.07)
- 2015: 0.041 (p=0.003)
- 2016: 0.042 (p<0.001)

---

## Output Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis Python script |
| `create_figures.py` | Figure generation script |
| `main_results.csv` | Main regression results |
| `robustness_results.csv` | Robustness check results |
| `event_study_results.csv` | Event study coefficients |
| `descriptive_stats.csv` | Descriptive statistics |
| `figure1_event_study.png/pdf` | Event study plot |
| `figure2_trends.png/pdf` | Employment trends over time |
| `figure3_did.png/pdf` | DiD visualization |
| `figure4_robustness.png/pdf` | Robustness checks summary |
| `replication_report_64.tex` | LaTeX report source |
| `replication_report_64.pdf` | Final PDF report (24 pages) |

---

## Session End

Analysis completed successfully. All required deliverables have been produced:
1. **replication_report_64.tex** - LaTeX source file
2. **replication_report_64.pdf** - Compiled PDF report (~24 pages)
3. **run_log_64.md** - This run log documenting all commands and decisions
