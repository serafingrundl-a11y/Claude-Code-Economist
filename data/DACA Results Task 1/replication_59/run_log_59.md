# Run Log - DACA Replication Study (ID: 59)

## Overview
Independent replication examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

---

## Session Log

### Step 1: Initial Setup and Data Review
- **Time**: Session start
- **Action**: Read replication_instructions.docx
- **Key findings**:
  - DACA implemented June 15, 2012
  - Examine effects on full-time employment 2013-2016
  - Data: ACS 2006-2016 from IPUMS
  - Target population: Hispanic-Mexican, Mexican-born individuals

### Step 2: Data Dictionary Review
- **Action**: Reviewed acs_data_dict.txt
- **Key variables identified**:
  - YEAR: Survey year (2006-2016)
  - HISPAN: Hispanic origin (1 = Mexican)
  - BPL: Birthplace (200 = Mexico)
  - CITIZEN: Citizenship status (3 = Not a citizen)
  - YRIMMIG: Year of immigration
  - BIRTHYR: Birth year
  - BIRTHQTR: Birth quarter
  - UHRSWORK: Usual hours worked per week
  - AGE: Age
  - PERWT: Person weight

### Step 3: DACA Eligibility Criteria (from instructions)
DACA eligibility requires:
1. Arrived in US before 16th birthday
2. Had not yet turned 31 as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012
5. Not a citizen or legal resident

### Step 4: Identification Strategy
- **Method**: Difference-in-Differences (DID)
- **Treatment group**: DACA-eligible individuals (meeting all criteria above)
- **Control group**: Similar individuals who do not meet eligibility criteria (primarily based on age cutoff)
- **Pre-period**: 2006-2011 (before DACA)
- **Post-period**: 2013-2016 (after DACA implementation)
- **Note**: 2012 excluded due to mid-year implementation

### Step 5: Sample Restrictions
- Hispanic ethnicity = Mexican (HISPAN == 1)
- Born in Mexico (BPL == 200)
- Non-citizens (CITIZEN == 3)
- Working age population (18-45 years old)
- Exclude 2012 (implementation year)
- Main analysis: Narrow bandwidth ages 26-35 in 2012

### Step 6: Outcome Variable
- Full-time employment: UHRSWORK >= 35

### Step 7: Analysis Implementation
- Python with pandas, statsmodels
- Weighted regression using PERWT
- Standard errors clustered by state
- Specifications: Basic DID, +controls, +state FE, +year FE

### Step 8: Data Processing
- **Total ACS observations**: 33,851,424
- **After sample restrictions (Hispanic-Mexican, Mexican-born, non-citizen, ages 18-45)**: 413,906
- **After arrival eligibility filter (arrived <16, by 2007)**: 113,977
- **Main analysis sample (ages 26-35 in 2012)**: 43,238

### Step 9: Analysis Execution
- Ran initial analysis (daca_analysis.py)
- Refined analysis with narrow bandwidth (daca_analysis_v2.py)
- Executed robustness checks and event study

---

## Key Decisions and Rationale

### Decision 1: Treatment/Control Group Definition
- **Treatment**: Ages 26-30 on June 15, 2012 (under 31, eligible) AND arrived before age 16 AND arrived by 2007
- **Control**: Ages 31-35 on June 15, 2012 (too old for DACA, but same arrival pattern)
- **Rationale**: Age cutoff provides clean quasi-experimental variation; narrow bandwidth improves comparability

### Decision 2: Exclusion of 2012
- **Rationale**: DACA implemented mid-2012; cannot distinguish pre/post within that year

### Decision 3: Narrow Age Bandwidth
- Focus on ages 26-35 (in 2012) for main analysis
- **Rationale**: Individuals close to the age cutoff are more comparable; minimizes confounding from lifecycle employment patterns

### Decision 4: Continuous Presence Approximation
- Use YRIMMIG <= 2007 as proxy for "present since June 2007"
- **Limitation**: Cannot verify continuous presence, only year of arrival

### Decision 5: Non-Citizen as Proxy for Undocumented
- Use CITIZEN == 3 (Not a citizen) as proxy for undocumented status
- **Limitation**: Some non-citizens may be documented (legal permanent residents)

### Decision 6: Model Specification Choice
- **Preferred specification**: State + Year fixed effects with demographic controls
- **Rationale**: Controls for state-specific factors and common time trends; year FE crucial for addressing post-recession recovery

---

## Analysis Commands Executed

```bash
# Run analysis script
cd "C:\Users\seraf\DACA Results Task 1\replication_59"
python daca_analysis_v2.py
```

```bash
# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_59.tex
pdflatex -interaction=nonstopmode replication_report_59.tex
pdflatex -interaction=nonstopmode replication_report_59.tex
```

---

## Results Summary

### Preferred Estimate (Narrow Bandwidth, State + Year FE)

| Statistic | Value |
|-----------|-------|
| **Coefficient** | 0.0194 |
| **Standard Error** | 0.0119 |
| **95% Confidence Interval** | [-0.0039, 0.0427] |
| **p-value** | 0.103 |
| **Sample Size** | 43,238 |

### Interpretation
DACA eligibility is associated with a 1.94 percentage point increase in the probability of full-time employment, though this effect is **not statistically significant** at conventional levels (p = 0.10).

### Key Findings by Specification

| Model | Coefficient | Std. Error | Notes |
|-------|-------------|------------|-------|
| Basic DID | 0.0590*** | 0.0069 | No controls |
| + Controls | 0.0646*** | 0.0096 | Demographics |
| + State FE | 0.0643*** | 0.0097 | State fixed effects |
| + Year FE (Preferred) | 0.0194 | 0.0119 | Full specification |

### Robustness Checks

| Specification | Coefficient | Std. Error | N |
|---------------|-------------|------------|---|
| Wider bandwidth (22-38) | -0.0399*** | 0.0060 | 77,295 |
| Narrower bandwidth (28-33) | 0.0258 | 0.0186 | 24,584 |
| Males only | 0.0095 | 0.0227 | 24,243 |
| Females only | 0.0195 | 0.0355 | 18,995 |
| Employment (any) | 0.0327** | 0.0137 | 43,238 |
| Placebo (pre-period) | -0.0282*** | 0.0082 | 27,666 |

### Event Study (Reference: 2011)

| Year | Coefficient | Interpretation |
|------|-------------|----------------|
| 2006 | 0.0322 | Pre-treatment |
| 2007 | -0.0082 | Pre-treatment |
| 2008 | 0.0242 | Pre-treatment |
| 2009 | 0.0035 | Pre-treatment |
| 2010 | -0.0087 | Pre-treatment |
| 2011 | 0.0000 | Reference |
| 2013 | 0.0238 | Post-treatment |
| 2014 | 0.0192 | Post-treatment |
| 2015 | -0.0014 | Post-treatment |
| 2016 | 0.0403** | Post-treatment |

---

## Conclusions

1. **Main Finding**: Positive but statistically insignificant effect of DACA eligibility on full-time employment (~2 percentage points)

2. **Parallel Trends**: Event study shows relatively flat pre-trends, supporting identification

3. **Concerns**:
   - Significant placebo coefficient raises questions about parallel trends
   - Results sensitive to bandwidth choice
   - Year fixed effects substantially reduce treatment estimate

4. **Interpretation**: The evidence is consistent with either a modest positive effect or no effect; statistical uncertainty prevents definitive conclusions

---

## Files Created

1. **run_log_59.md** - This log file
2. **daca_analysis.py** - Initial analysis script
3. **daca_analysis_v2.py** - Refined analysis script with narrow bandwidth
4. **replication_report_59.tex** - LaTeX report (20 pages)
5. **replication_report_59.pdf** - Compiled PDF report

---

## Data Files Used

- **data/data.csv** - Main ACS data (6.27 GB, 33.8M observations)
- **data/acs_data_dict.txt** - IPUMS data dictionary
- **data/state_demo_policy.csv** - Supplemental state data (not used in main analysis)

---

## Software Environment

- Python 3.x
- pandas
- numpy
- statsmodels
- pdflatex (MiKTeX)

---

## Replication Notes

To replicate the analysis:
1. Ensure Python dependencies are installed: `pip install pandas numpy statsmodels`
2. Run the analysis: `python daca_analysis_v2.py`
3. Compile the report: `pdflatex replication_report_59.tex` (run 3 times for references)

The data file (data.csv) is approximately 6 GB and requires chunked processing.
