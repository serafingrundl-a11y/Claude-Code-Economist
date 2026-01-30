# Run Log for DACA Replication Study (ID: 81)

## Overview
This log documents all commands, decisions, and key steps taken during the independent replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Data Source
- American Community Survey (ACS) data from IPUMS USA
- Years: 2006-2016 (1-year ACS files)
- Supplemental file: state_demo_policy.csv (optional, not used)

---

## Session Log

### Step 1: Environment Setup and Data Exploration
**Timestamp:** 2026-01-25

**Actions:**
- Read replication_instructions.docx to understand the research task
- Examined data dictionary (acs_data_dict.txt) to understand variable definitions
- Verified data.csv structure and column names

**Key Variables Identified:**
- `YEAR`: Survey year (2006-2016)
- `HISPAN`/`HISPAND`: Hispanic origin (1 = Mexican for HISPAN)
- `BPL`/`BPLD`: Birthplace (200 = Mexico for BPL)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Birth year
- `BIRTHQTR`: Birth quarter
- `AGE`: Age
- `UHRSWORK`: Usual hours worked per week
- `PERWT`: Person weight

---

### Step 2: Sample Definition Decisions

**Decision 1: Population Restriction**
- Restrict to Hispanic-Mexican (HISPAN == 1) AND born in Mexico (BPL == 200)
- This follows the instruction for "ethnically Hispanic-Mexican Mexican-born people"

**Decision 2: DACA Eligibility Criteria**
Based on the instructions, eligibility requires:
1. Arrived in US before age 16 (age at immigration < 16)
2. Born after June 15, 1981 (not yet 31 as of June 15, 2012)
3. Present in US since at least June 15, 2007 (continuous residence)
4. Not a citizen (CITIZEN == 3)

**Implementation Notes:**
- Age at immigration = YRIMMIG - BIRTHYR (approximation since we don't have exact dates)
- For the age 31 cutoff: used BIRTHYR >= 1982 to be conservative (under 31 on June 15, 2012)
- For continuous residence since 2007: YRIMMIG <= 2007

**Decision 3: Treatment Period**
- Pre-period: 2006-2011 (before DACA)
- Exclude 2012 as transitional year (DACA announced June 15, 2012; cannot distinguish pre/post observations)
- Post-period: 2013-2016 (after DACA implementation)

**Decision 4: Control Group**
- Mexican-born non-citizens who do not meet DACA eligibility criteria
- Similar characteristics but aged out (born before 1982) OR arrived after age 15 OR arrived after 2007

**Decision 5: Outcome Variable**
- Full-time employment = 1 if UHRSWORK >= 35, 0 otherwise
- Restrict to working-age population (ages 16-64 at time of survey)

---

### Step 3: Analysis Approach

**Primary Specification: Difference-in-Differences**
- Compare changes in full-time employment between DACA-eligible and non-eligible groups
- Before vs. after DACA implementation (2012)
- Model: Y = b0 + b1*Eligible + b2*Post + b3*(Eligible*Post) + Controls + e
- b3 is the coefficient of interest (DACA effect)

**Control Variables:**
- Age, Age squared
- Sex
- Education level
- Marital status
- State fixed effects
- Year fixed effects

**Weighting:**
- Use person weights (PERWT) for all analyses

---

### Step 4: Python Analysis Script Execution

**Command:**
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_81" && python analysis.py
```

**Script: analysis.py**
- Memory-optimized loading with chunked processing
- Filtered during load to manage 6GB CSV file
- Applied all sample restrictions
- Created DACA eligibility variable
- Ran 4 main regression specifications
- Conducted robustness checks
- Performed pre-trends analysis
- Ran event study

---

### Step 5: Analysis Results

**Sample Size After Restrictions:**
- Total observations: 561,470
- DACA-eligible: 81,508 (14.5%)
- Non-eligible: 479,962 (85.5%)
- Pre-DACA period (2006-2011): 345,792
- Post-DACA period (2013-2016): 215,678

**Full-Time Employment Rates (Raw):**
|                | Pre-DACA | Post-DACA |
|----------------|----------|-----------|
| Non-Eligible   | 60.4%    | 57.9%     |
| Eligible       | 42.5%    | 49.4%     |

**Raw DiD Estimate:** +9.4 percentage points

**Main Regression Results:**

| Model | Description | Estimate | SE |
|-------|-------------|----------|-----|
| 1 | Basic DiD (no controls) | 0.0941 | 0.0038 |
| 2 | + Individual controls | 0.0399 | 0.0034 |
| 3 | + State/Year FE | 0.0339 | 0.0034 |
| 4 | + Weights (PREFERRED) | 0.0326 | 0.0033 |

**PREFERRED ESTIMATE:**
- Coefficient: 0.0326 (3.26 percentage points)
- Standard Error: 0.0033
- 95% CI: [0.0261, 0.0391]
- t-statistic: 9.84
- p-value: < 0.0001

**Robustness Checks:**

| Specification | Estimate | SE |
|---------------|----------|-----|
| Alternative eligibility (BIRTHYR >= 1981) | 0.0269 | 0.0032 |
| Young sample (ages 18-35) | 0.0099 | 0.0041 |
| Men only | 0.0295 | 0.0041 |
| Women only | 0.0269 | 0.0053 |
| Robust standard errors | 0.0326 | 0.0042 |

**Pre-Trends Coefficients (relative to 2006):**
- 2007: 0.0038 (p=0.641)
- 2008: 0.0182 (p=0.024)
- 2009: 0.0266 (p=0.001)
- 2010: 0.0307 (p=0.000)
- 2011: 0.0192 (p=0.012)

Note: Some evidence of differential pre-trends, suggesting caution in causal interpretation.

**Event Study (reference: 2011):**
- 2006: -0.0172
- 2007: -0.0138
- 2008: 0.0001
- 2009: 0.0081
- 2010: 0.0118
- 2011: 0.0000 (reference)
- 2013: 0.0161
- 2014: 0.0267
- 2015: 0.0416
- 2016: 0.0433

---

### Step 6: Report Generation

**Command:**
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_81" && pdflatex -interaction=nonstopmode replication_report_81.tex
```
(Run 3 times for TOC and cross-references)

**Output:**
- replication_report_81.tex (LaTeX source)
- replication_report_81.pdf (16 pages)

---

## Files Generated

| File | Description |
|------|-------------|
| analysis.py | Python analysis script |
| analysis_output.log | Full console output |
| results_table.csv | Main regression results |
| event_study_results.csv | Event study coefficients |
| summary_stats.csv | Summary statistics |
| table_2x2.csv | 2x2 means table |
| detailed_results.txt | Comprehensive results summary |
| replication_report_81.tex | LaTeX report source |
| replication_report_81.pdf | Final PDF report (16 pages) |
| run_log_81.md | This log file |

---

## Key Analytical Decisions Summary

1. **Sample restriction to non-citizens only** - Required because DACA eligibility requires being undocumented. ACS cannot distinguish documented vs. undocumented, so all non-citizens are assumed potentially undocumented.

2. **Birth year cutoff of 1982** - Conservative interpretation of "not yet 31 as of June 15, 2012." Someone born in 1982 would be at most 30 in 2012.

3. **Immigration year cutoff of 2007** - Requirement for continuous presence since June 15, 2007. YRIMMIG <= 2007 ensures arrival by or before 2007.

4. **Exclusion of 2012** - DACA implemented mid-year (June 15), and ACS doesn't record interview month, so cannot cleanly assign 2012 observations to pre/post periods.

5. **Age restriction 16-64** - Standard working-age population; younger individuals not typically in labor force, older may be retired.

6. **Linear probability model with WLS** - Chosen for interpretability (coefficients as percentage points) and to incorporate survey weights.

7. **State and year fixed effects** - To control for geographic variation in labor markets and aggregate economic conditions.

---

## Interpretation

The preferred estimate suggests DACA eligibility increased full-time employment by approximately 3.3 percentage points, which is statistically significant at conventional levels. This represents a 7.8% increase relative to the baseline rate among eligible individuals (42.5%).

However, the pre-trends analysis reveals some evidence of differential trends before DACA, with the eligible group showing improved relative employment even before 2012. This suggests some caution is warranted in interpreting the results as purely causal.

The effect appears to grow over time (event study shows larger effects in 2015-2016), which is consistent with cumulative benefits of legal work status or increasing DACA uptake.

---

## Reproducibility

To reproduce this analysis:
1. Place data.csv in the data/ subfolder
2. Run: `python analysis.py`
3. Run: `pdflatex replication_report_81.tex` (3 times)

Required Python packages: pandas, numpy, statsmodels, scipy

---

*Log completed: 2026-01-25*
