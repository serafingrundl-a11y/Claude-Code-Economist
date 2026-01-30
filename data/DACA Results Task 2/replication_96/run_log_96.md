# Run Log - Replication 96

## Overview
This log documents all commands and key decisions made during the DACA replication study.

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (>=35 hours/week)?

**Design:** Difference-in-Differences
- Treatment group: Ages 26-30 at DACA implementation (June 15, 2012)
- Control group: Ages 31-35 at DACA implementation
- Pre-period: 2006-2011
- Post-period: 2013-2016 (excluding 2012 due to implementation timing ambiguity)

---

## Session Start

### Step 1: Data Dictionary Review
- Reviewed `acs_data_dict.txt` to understand variable definitions
- Key variables identified:
  - `YEAR`: Survey year (2006-2016)
  - `BIRTHYR`: Birth year
  - `BIRTHQTR`: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
  - `HISPAN`: Hispanic origin (1=Mexican)
  - `HISPAND`: Detailed Hispanic (100-107 = Mexican variants)
  - `BPL`: Birthplace (200=Mexico)
  - `CITIZEN`: Citizenship status (3=Not a citizen)
  - `YRIMMIG`: Year of immigration
  - `UHRSWORK`: Usual hours worked per week
  - `EMPSTAT`: Employment status (1=Employed)
  - `PERWT`: Person weight for population estimates

### Step 2: DACA Eligibility Criteria
Per instructions and DACA program rules:
1. Hispanic-Mexican ethnicity (HISPAN=1 or HISPAND in 100-107)
2. Born in Mexico (BPL=200)
3. Not a citizen (CITIZEN=3)
4. Arrived in US before age 16 (YRIMMIG - BIRTHYR < 16)
5. Present in US since June 15, 2007 (YRIMMIG <= 2007)
6. Age restrictions for treatment/control based on age as of June 15, 2012

### Step 3: Age Group Definitions
- DACA required age < 31 as of June 15, 2012
- Treatment: Would be ages 26-30 on June 15, 2012
  - Birth dates: June 16, 1981 to June 15, 1986
- Control: Would be ages 31-35 on June 15, 2012
  - Birth dates: June 16, 1976 to June 15, 1981

### Step 4: Outcome Variable
- Full-time employment = UHRSWORK >= 35 (as specified in research question)

---

## Analysis Commands

### Python Analysis Script (analysis.py)

```python
# Key commands executed:

# 1. Load data in chunks (memory efficient)
for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=500000):
    # Apply eligibility filters within each chunk

# 2. Sample selection criteria
chunk['is_hispanic_mexican'] = (chunk['HISPAN'] == 1) | (chunk['HISPAND'].isin(range(100, 108)))
chunk['born_mexico'] = (chunk['BPL'] == 200) | (chunk['BPLD'] == 20000)
chunk['not_citizen'] = chunk['CITIZEN'] == 3
chunk['age_at_arrival'] = chunk['YRIMMIG'] - chunk['BIRTHYR']
chunk['arrived_before_16'] = (chunk['age_at_arrival'] < 16) & (chunk['age_at_arrival'] >= 0)
chunk['in_us_since_2007'] = chunk['YRIMMIG'] <= 2007

# 3. Age calculation as of June 15, 2012
def calc_age(row):
    if row['BIRTHQTR'] in [1, 2]:  # Born Jan-June
        return 2012 - row['BIRTHYR']
    else:  # Born Jul-Dec
        return 2012 - row['BIRTHYR'] - 1

# 4. Treatment/Control group assignment
chunk['treatment'] = (chunk['age_june_2012'] >= 26) & (chunk['age_june_2012'] <= 30)
chunk['control'] = (chunk['age_june_2012'] >= 31) & (chunk['age_june_2012'] <= 35)

# 5. Outcome variable
sample['fulltime'] = (sample['UHRSWORK'] >= 35).astype(int)

# 6. DiD Estimation
model4 = smf.wls('fulltime ~ treat + treat_post + C(YEAR)',
                  data=sample, weights=sample['PERWT']).fit(cov_type='HC1')
```

---

## Key Decisions

### Decision 1: Excluding Year 2012
**Rationale:** DACA was implemented on June 15, 2012. Since the ACS does not provide month of survey, observations in 2012 cannot be reliably classified as pre- or post-treatment. Excluding 2012 ensures clean identification.

### Decision 2: Using Birth Quarter for Age Calculation
**Rationale:** To accurately determine age as of June 15, 2012, we account for birth quarter. Those born in Q1-Q2 (Jan-Jun) have had their birthday by June 15, while those in Q3-Q4 (Jul-Dec) have not.

### Decision 3: Non-Citizen as Undocumented Proxy
**Rationale:** The ACS does not directly identify undocumented status. Using CITIZEN=3 (not a citizen) as a proxy captures most undocumented individuals while including some legal non-citizens. This is a common approach in the literature.

### Decision 4: Full-Time Threshold of 35 Hours
**Rationale:** Per the research question specification, full-time employment is defined as usually working 35 hours or more per week (UHRSWORK >= 35).

### Decision 5: Weighted Estimation with Person Weights
**Rationale:** Using PERWT (person weights) produces population-representative estimates. Weights adjust for sampling design and non-response.

### Decision 6: Heteroskedasticity-Robust Standard Errors
**Rationale:** Using HC1 (robust) standard errors accounts for potential heteroskedasticity in the error terms, providing valid inference without assuming homoskedasticity.

### Decision 7: Year Fixed Effects
**Rationale:** Year fixed effects control for aggregate time trends affecting all individuals (e.g., business cycle effects, general labor market conditions).

### Decision 8: Reference Year 2011 for Event Study
**Rationale:** 2011 is the last complete pre-treatment year and provides the most recent baseline for comparison.

---

## Results Summary

### Sample Size
- Total observations: 42,689
- Treatment group (ages 26-30): 25,174
- Control group (ages 31-35): 17,515
- Pre-period (2006-2011): 28,030
- Post-period (2013-2016): 14,659

### Main Results (Preferred Model)
- **DiD Estimate:** 0.0561 (5.61 percentage points)
- **Standard Error:** 0.0118
- **95% Confidence Interval:** [0.0330, 0.0792]
- **P-value:** < 0.001

### Interpretation
DACA eligibility increased full-time employment by approximately 5.6 percentage points among eligible Hispanic-Mexican Mexican-born individuals aged 26-30 at the time of implementation. This effect is statistically significant at conventional levels.

### Robustness Checks
| Specification | Coefficient | SE | Notes |
|--------------|-------------|-----|-------|
| Main (Year FE, weighted) | 0.0561 | 0.0118 | Preferred |
| With covariates | 0.0451 | 0.0108 | Adds female, married, education |
| Any employment | 0.0513 | 0.0110 | Alternative outcome |
| Narrow bandwidth (28-30 vs 31-33) | 0.0514 | 0.0155 | N=24,264 |
| Males only | 0.0431 | 0.0126 | N=23,936 |
| Females only | 0.0445 | 0.0186 | N=18,753 |
| With state FE | 0.0544 | 0.0117 | |
| Placebo (pre-DACA) | 0.0086 | 0.0137 | Not significant |

### Event Study (Pre-Trends)
Pre-treatment coefficients (2006-2010) are all statistically insignificant and small in magnitude, supporting the parallel trends assumption.

---

## Output Files

1. **analysis.py** - Main Python analysis script
2. **analysis_output.txt** - Full console output from analysis
3. **summary_stats.csv** - Summary statistics by group and period
4. **event_study_coefs.csv** - Event study coefficients
5. **yearly_means.csv** - Yearly means by treatment status
6. **replication_report_96.tex** - LaTeX source for report
7. **replication_report_96.pdf** - Final PDF report (21 pages)
8. **run_log_96.md** - This log file

---

## Session End

Analysis completed successfully. All required deliverables generated:
- replication_report_96.tex
- replication_report_96.pdf
- run_log_96.md
