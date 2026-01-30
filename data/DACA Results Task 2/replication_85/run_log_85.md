# Run Log - Replication 85

## Overview
This log documents all commands, key decisions, and analytical choices made during the replication of the DACA employment effects study.

## Session Start
Date: 2026-01-26

---

## Step 1: Understanding the Research Question

**Research Question:** Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (Deferred Action for Childhood Arrivals) on the probability of full-time employment (usually working 35+ hours per week)?

**Design:**
- Treatment group: Ages 26-30 at policy implementation (June 15, 2012)
- Control group: Ages 31-35 at policy implementation
- Pre-period: 2006-2011 (excluding 2012 due to ambiguity)
- Post-period: 2013-2016
- Method: Difference-in-Differences

---

## Step 2: Data Exploration

### Data Files Available:
- `data.csv` - Main ACS data (2006-2016, ~33.8 million rows)
- `acs_data_dict.txt` - Data dictionary
- `state_demo_policy.csv` - Optional state-level data (not used)

### Key Variables Identified:
| Variable | IPUMS Name | Description |
|----------|------------|-------------|
| Survey Year | YEAR | Census year (2006-2016) |
| Birth Year | BIRTHYR | Year of birth |
| Birth Quarter | BIRTHQTR | Quarter of birth (1-4) |
| Hispanic Origin | HISPAN | 1 = Mexican |
| Birthplace | BPL | 200 = Mexico |
| Citizenship | CITIZEN | 3 = Not a citizen |
| Year of Immigration | YRIMMIG | Year of immigration to US |
| Usual Hours Worked | UHRSWORK | Hours per week usually worked |
| Person Weight | PERWT | Survey weight for population estimates |
| Employment Status | EMPSTAT | 1 = Employed |
| Sex | SEX | 1 = Male, 2 = Female |
| Marital Status | MARST | 1-2 = Married |
| Education | EDUC | Educational attainment |
| State | STATEFIP | State FIPS code |

---

## Step 3: Sample Selection Criteria

### DACA Eligibility Requirements Applied:
1. Born in Mexico (BPL == 200)
2. Hispanic-Mexican ethnicity (HISPAN == 1)
3. Not a citizen (CITIZEN == 3) - proxies for undocumented status
4. Arrived in US before 16th birthday (YRIMMIG - BIRTHYR < 16)
5. Arrived in US by 2007 (continuous residence requirement)
6. Valid immigration year (YRIMMIG > 0)

### Age Groups (as of June 15, 2012):
- **Treated (ages 26-30):** Would have been eligible for DACA
- **Control (ages 31-35):** Would have been ineligible due to age cutoff

### Computing Age at Policy Date:
```python
# Using BIRTHYR and BIRTHQTR to determine age as of June 15, 2012
# Q1 (Jan-Mar) or Q2 (Apr-Jun): likely had birthday by June 15
# Q3 (Jul-Sep) or Q4 (Oct-Dec): had not had birthday yet
df['age_at_daca'] = np.where(
    df['BIRTHQTR'].isin([1, 2]),
    2012 - df['BIRTHYR'],
    2012 - df['BIRTHYR'] - 1
)
```

---

## Step 4: Outcome Variable

**Full-time employment:** UHRSWORK >= 35

```python
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
```

This follows the Bureau of Labor Statistics definition of full-time work.

---

## Step 5: Analysis Commands

### Python Script: analysis.py

The analysis was conducted using Python with the following key packages:
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)

### Data Loading Strategy
Due to large file size (~6GB), data was loaded in chunks with filtering applied during read:

```python
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
               'EMPSTAT', 'UHRSWORK']

chunks = []
for chunk in pd.read_csv(data_path, chunksize=500000, usecols=cols_needed):
    chunk = chunk[chunk['HISPAN'] == 1]      # Hispanic-Mexican
    chunk = chunk[chunk['BPL'] == 200]        # Born in Mexico
    chunk = chunk[chunk['CITIZEN'] == 3]      # Non-citizen
    chunk = chunk[chunk['YEAR'] != 2012]      # Exclude 2012
    if len(chunk) > 0:
        chunks.append(chunk)
```

### Sample Construction Summary
| Step | Description | N |
|------|-------------|---|
| 1 | All ACS observations 2006-2016 (excluding 2012) | 33,851,424 |
| 2 | Restrict to Hispanic-Mexican | 3,871,483 |
| 3 | Restrict to born in Mexico | 2,619,449 |
| 4 | Restrict to non-citizens | 695,094 |
| 5 | Restrict to ages 26-35 at DACA implementation | 164,874 |
| 6 | Restrict to arrived before age 16 | 43,238 |
| 7 | Final analytic sample | **43,238** |

---

## Step 6: Key Analytical Decisions

### Decision 1: Exclusion of 2012
Year 2012 was excluded because the ACS does not indicate the month of data collection. Since DACA was implemented on June 15, 2012, observations from 2012 cannot be clearly assigned to pre- or post-treatment periods.

### Decision 2: Age Calculation
Age at DACA implementation was calculated using birth year and birth quarter. Those born in Q1-Q2 (January-June) were assumed to have had their birthday by June 15, 2012.

### Decision 3: Proxy for Undocumented Status
Non-citizens (CITIZEN = 3) without legal status were assumed to be undocumented per instructions. This is imperfect but standard in the literature.

### Decision 4: Continuous Residence Requirement
Restricted to those who immigrated by 2007 to satisfy DACA's continuous residence requirement (in US since June 15, 2007).

### Decision 5: Survey Weights
All weighted specifications used PERWT (person weights) for population-representative estimates.

### Decision 6: Reference Year for Event Study
2011 was used as the reference year for the event study specification as it is the last pre-treatment year.

---

## Step 7: Results Summary

### Final Sample Sizes
| Group | Pre-Period | Post-Period | Total |
|-------|------------|-------------|-------|
| Control (31-35) | 11,683 | 6,085 | 17,768 |
| Treatment (26-30) | 16,694 | 8,776 | 25,470 |
| Total | 28,377 | 14,861 | 43,238 |

### Full-Time Employment Rates (Unweighted)
| Group | Pre-Period | Post-Period | Change |
|-------|------------|-------------|--------|
| Control | 0.6461 | 0.6136 | -0.0325 |
| Treatment | 0.6147 | 0.6339 | +0.0192 |
| **Raw DiD** | | | **+0.0516** |

### Main Regression Results

| Specification | DiD Estimate | SE | p-value |
|--------------|--------------|-----|---------|
| Basic (unweighted) | 0.0516 | 0.0100 | <0.001 |
| Weighted | 0.0590 | 0.0098 | <0.001 |
| With controls | 0.0478 | 0.0090 | <0.001 |
| Year FE (weighted) | 0.0574 | 0.0098 | <0.001 |
| Year + State FE | 0.0557 | 0.0098 | <0.001 |
| Full specification | 0.0459 | 0.0090 | <0.001 |

### Preferred Estimate
**Model with Year Fixed Effects (weighted):**
- DiD Coefficient: 0.0574
- Standard Error: 0.0098
- 95% CI: [0.0382, 0.0765]
- p-value: < 0.001

### Interpretation
DACA eligibility is associated with a 5.7 percentage point increase in the probability of full-time employment among Hispanic-Mexican immigrants born in Mexico.

---

## Step 8: Robustness Checks

### 1. Narrower Age Bands (27-29 vs 32-34)
- DiD: 0.0529 (SE: 0.0127)
- N: 25,606
- Consistent with main results

### 2. Alternative Outcome - Any Employment
- DiD: 0.0517 (SE: 0.0093)
- Similar effect on extensive margin

### 3. By Gender
- Males: DiD = 0.0446 (SE: 0.0108)
- Females: DiD = 0.0454 (SE: 0.0153)
- Effects similar across genders

### 4. Event Study (Parallel Trends Test)
| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | -0.0084 | 0.0198 | 0.670 |
| 2007 | -0.0441 | 0.0201 | 0.028 |
| 2008 | -0.0019 | 0.0203 | 0.924 |
| 2009 | -0.0142 | 0.0204 | 0.487 |
| 2010 | -0.0195 | 0.0205 | 0.340 |
| 2013 | 0.0376 | 0.0211 | 0.074 |
| 2014 | 0.0429 | 0.0212 | 0.044 |
| 2015 | 0.0227 | 0.0218 | 0.296 |
| 2016 | 0.0682 | 0.0220 | 0.002 |

Pre-treatment coefficients generally not significant (except 2007), supporting parallel trends.

---

## Step 9: Output Files Generated

| File | Description |
|------|-------------|
| results.json | All numerical results in JSON format |
| descriptive_stats.csv | Descriptive statistics by group |
| model_summaries.txt | Full regression output tables |
| replication_report_85.tex | LaTeX source for report |
| replication_report_85.pdf | Final PDF report (21 pages) |
| run_log_85.md | This run log |

---

## Step 10: Commands Executed

```bash
# Run analysis script
cd "C:\Users\seraf\DACA Results Task 2\replication_85"
python analysis.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_85.tex
pdflatex -interaction=nonstopmode replication_report_85.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_85.tex  # Third pass to finalize
```

---

## Final Notes

1. The analysis finds a statistically significant positive effect of DACA eligibility on full-time employment.
2. The effect size (4.6-5.7 percentage points) is robust across specifications.
3. Event study provides support for the parallel trends assumption.
4. Results are consistent across subgroups (males/females) and alternative outcomes.
5. All required deliverables have been produced in the specified filenames.

---

**Run completed:** 2026-01-26
