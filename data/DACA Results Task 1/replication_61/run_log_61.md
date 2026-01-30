# Replication Run Log - Task 61

## Project Overview
**Research Question**: What was the causal impact of DACA eligibility (treatment) on the probability of full-time employment (outcome, defined as usually working 35+ hours per week) among ethnically Hispanic-Mexican, Mexican-born individuals in the United States?

**Time Period**: DACA implemented June 15, 2012. Examining effects on full-time employment in 2013-2016.

## Key Decisions Log

### Decision 1: Sample Definition
**Date**: Session start
**Decision**: Define the target population as:
- Hispanic-Mexican ethnicity (HISPAN == 1)
- Born in Mexico (BPL == 200 or BPLD == 20000)
- Non-citizen (CITIZEN == 3, "Not a citizen") to proxy undocumented status
- Working-age population for DACA relevance

**Rationale**: The instructions specify examining "ethnically Hispanic-Mexican Mexican-born people." DACA targets undocumented immigrants, and since we cannot distinguish documented from undocumented non-citizens, we follow the instruction to assume non-citizens without papers are undocumented.

### Decision 2: DACA Eligibility Criteria
**Date**: Session start
**Decision**: Based on the program requirements, define DACA-eligible as:
1. Arrived in US before 16th birthday (YRIMMIG - BIRTHYR < 16)
2. Not yet 31 by June 15, 2012 (BIRTHYR >= 1982, accounting for mid-year cutoff)
3. Lived continuously in US since June 15, 2007 (YRIMMIG <= 2007)
4. Present in US on June 15, 2012 (YRIMMIG <= 2012)

**Rationale**: These criteria directly map to the DACA eligibility requirements listed in the instructions.

### Decision 3: Outcome Variable
**Date**: Session start
**Decision**: Define full-time employment as UHRSWORK >= 35

**Rationale**: The instructions explicitly define full-time as "usually working 35 hours per week or more." UHRSWORK is the "Usual hours worked per week" variable.

### Decision 4: Identification Strategy
**Date**: Session start
**Decision**: Use a difference-in-differences (DiD) design comparing:
- Treatment group: DACA-eligible Mexican-born non-citizens
- Control group: DACA-ineligible Mexican-born non-citizens (similar population but too old or arrived too late)
- Pre-period: 2006-2011 (before DACA)
- Post-period: 2013-2016 (after DACA implementation)
- Exclude 2012 as a transition year since DACA was implemented mid-year

**Rationale**: DiD is appropriate for policy evaluation when we have before/after data and can identify a comparable control group. The age-based eligibility cutoff provides a natural comparison group.

### Decision 5: Age Restrictions
**Date**: Session start
**Decision**: Restrict sample to ages 18-45 to focus on working-age adults
- Lower bound of 18: Legal working age, avoids child labor issues
- Upper bound of 45: Ensures both treatment and control groups have substantial overlap

**Rationale**: Employment outcomes are most relevant for working-age adults. The age bounds ensure we have comparable treatment and control observations.

### Decision 6: Treatment of 2012
**Date**: Session start
**Decision**: Exclude year 2012 from the analysis

**Rationale**: DACA was implemented on June 15, 2012, and the ACS does not identify the month of data collection. Including 2012 would mix pre- and post-treatment observations.

## Commands and Analysis Steps

### Step 1: Data Loading and Initial Exploration
```python
# Load data
import pandas as pd
data = pd.read_csv('data/data.csv')

# Examine structure
data.shape
data.columns
data.head()
```

### Step 2: Sample Construction
- Filter to Hispanic-Mexican (HISPAN == 1)
- Filter to Mexican-born (BPL == 200)
- Filter to non-citizens (CITIZEN == 3)
- Restrict to ages 18-45
- Exclude year 2012

### Step 3: Define DACA Eligibility
- Calculate age at arrival: YRIMMIG - BIRTHYR
- Arrived before 16: age_at_arrival < 16
- Age on June 15, 2012: 2012 - BIRTHYR
- Under 31 on June 15, 2012: born 1982 or later (with birth quarter adjustment)
- In US since 2007: YRIMMIG <= 2007

### Step 4: Create Analysis Variables
- Full-time employment indicator: UHRSWORK >= 35
- Post-DACA indicator: YEAR >= 2013
- Treatment group indicator: DACA-eligible
- Interaction term: treatment * post

### Step 5: Difference-in-Differences Estimation
- Linear probability model with person weights (PERWT)
- Robust standard errors clustered at state level (STATEFIP)
- Controls: age, age squared, sex, marital status, education, state FE, year FE

### Step 6: Robustness Checks
- Alternative control groups
- Different age windows
- Triple-differences if possible
- Parallel trends verification

## Output Files
- replication_report_61.tex: Full LaTeX report
- replication_report_61.pdf: Compiled PDF
- run_log_61.md: This file

## Session Progress
- [x] Read instructions
- [x] Examine data dictionary
- [x] Document key decisions
- [x] Load and clean data
- [x] Define treatment/control
- [x] Run main analysis
- [x] Robustness checks
- [ ] Create figures/tables
- [ ] Write report
- [ ] Compile PDF

## Analysis Results Summary

### Sample Construction
- Initial observations: 33,851,424
- After Hispanic-Mexican filter: 2,945,521
- After Mexican-born filter: 991,261
- After non-citizen filter: 701,347
- After excluding 2012: 636,722
- After age 18-45 restriction: 413,906
- Treatment group (DACA-eligible): 69,244
- Control group (DACA-ineligible): 344,662

### Main Results

**Simple DiD Estimate**: 0.0742

Full-time rates by group and period (weighted):
- DACA-eligible, Pre: 0.5199
- DACA-eligible, Post: 0.5680
- DACA-ineligible, Pre: 0.6431
- DACA-ineligible, Post: 0.6169

**Regression Results:**
| Model | Coefficient | Std. Error | Controls |
|-------|-------------|------------|----------|
| Model 1 | 0.0742 | 0.0038 | None |
| Model 2 | 0.0353 | 0.0043 | Demographics |
| Model 3 | 0.0345 | 0.0044 | + Education |
| Model 4 | 0.0228 | 0.0040 | + State/Year FE |

**Preferred Estimate (Model 4):**
- Coefficient: 0.0228
- Standard Error: 0.0040
- 95% CI: [0.0150, 0.0307]
- P-value: < 0.0001

### Interpretation
DACA eligibility is associated with a 2.28 percentage point increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born non-citizens. This effect is statistically significant at the 1% level.

### Robustness Checks
1. **Alternative age ranges**: Effects range from 0.027 to 0.035, all significant
2. **By gender**: Males 0.019, Females 0.051 (larger effect for women)
3. **Employment (any)**: 0.045 (larger effect on extensive margin)
4. **Labor force participation**: 0.044 (significant increase)

### Event Study / Parallel Trends
Pre-trends show some positive coefficients (0.01-0.02) relative to 2011, suggesting potential pre-existing differential trends. Post-DACA coefficients increase over time (0.022 in 2013 to 0.052 in 2016), suggesting growing effects or continued treatment take-up.

## Commands Executed

### Data Loading and Analysis
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_61"
python analysis.py
```

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_61.tex
pdflatex -interaction=nonstopmode replication_report_61.tex  # Second pass for references
```

## Output Files Produced
- `replication_report_61.tex` - LaTeX source (30,738 bytes)
- `replication_report_61.pdf` - Compiled PDF report (243,692 bytes, 25 pages)
- `run_log_61.md` - This run log
- `analysis.py` - Python analysis script
- `yearly_rates.csv` - Year-by-year full-time employment rates
- `event_study.csv` - Event study coefficients
- `summary_stats.csv` - Summary statistics by treatment status

## Session Completion
All deliverables have been produced:
- [x] replication_report_61.tex (LaTeX source)
- [x] replication_report_61.pdf (25-page compiled report)
- [x] run_log_61.md (this file)
