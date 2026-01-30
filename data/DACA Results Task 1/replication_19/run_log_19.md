# Run Log - DACA Replication Study (Replication 19)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA employment effects study.

## Date: 2026-01-25

---

## 1. Data Exploration

### Commands Executed:
```bash
# Read replication instructions
python -c "from docx import Document; doc = Document('replication_instructions.docx'); print('\n'.join([p.text for p in doc.paragraphs]))"

# Explore data folder
ls -la data/

# View data dictionary
head -500 data/acs_data_dict.txt

# View first lines of data
head -5 data/data.csv
```

### Key Findings:
- Data spans 2006-2016 ACS surveys
- 33,851,424 total observations
- Key variables identified: YEAR, HISPAN, BPL, CITIZEN, YRIMMIG, BIRTHYR, BIRTHQTR, UHRSWORK, EMPSTAT, PERWT

---

## 2. Sample Definition Decisions

### Decision 1: Target Population
**Choice:** Restrict to Hispanic-Mexican (HISPAN == 1), Mexican-born (BPL == 200), non-citizens (CITIZEN == 3)

**Rationale:**
- Instructions specify "ethnically Hispanic-Mexican Mexican-born people"
- HISPAN == 1 identifies Mexican ethnicity in IPUMS
- BPL == 200 identifies Mexico as birthplace
- CITIZEN == 3 (not a citizen) serves as proxy for potentially undocumented status, as we cannot distinguish documented from undocumented non-citizens

**Sample sizes after each restriction:**
- Hispanic-Mexican: 2,945,521
- Mexican-born: 991,261
- Non-citizens: 701,347

### Decision 2: Working-Age Population
**Choice:** Restrict to ages 16-64

**Rationale:** Standard working-age definition used in labor economics literature

### Decision 3: Exclude 2012
**Choice:** Remove 2012 observations from analysis

**Rationale:** DACA was implemented June 15, 2012. ACS does not identify month of survey collection, so we cannot distinguish pre- and post-treatment observations in 2012.

---

## 3. DACA Eligibility Criteria Implementation

### Eligibility Requirements (per instructions):
1. Arrived in US before 16th birthday
2. Under 31 years old as of June 15, 2012
3. Continuously resided in US since June 15, 2007
4. Present in US on June 15, 2012, not a citizen

### Implementation:
```python
# Age as of June 2012 calculation
age_june2012 = 2012 - BIRTHYR
# Adjust for birth quarter (Q3-Q4 hasn't had birthday by June)

# Criterion 1: Under 31 as of June 2012
under_31_2012 = (age_june2012 < 31)

# Criterion 2: Arrived before 16th birthday
age_at_immig = YRIMMIG - BIRTHYR
arrived_before_16 = (age_at_immig < 16)

# Criterion 3: In US since 2007
in_us_since_2007 = (YRIMMIG <= 2007)

# Combined eligibility
daca_eligible = under_31_2012 & arrived_before_16 & in_us_since_2007
```

### Treatment/Control Sample Sizes:
- DACA-eligible (treated): 83,611
- Not eligible (control): 477,859

---

## 4. Outcome Variable Definition

### Decision: Full-Time Employment
**Definition:** UHRSWORK >= 35 AND EMPSTAT == 1

**Rationale:**
- Instructions specify "usually working 35 hours per week or more"
- UHRSWORK captures usual hours worked
- EMPSTAT == 1 confirms employment status

### Baseline Rates (Pre-DACA):
- Control group: 54.6%
- Treatment group: 37.1%

---

## 5. Estimation Strategy

### Primary Approach: Difference-in-Differences
**Formula:** Y = a + b1*Treated + b2*Post + b3*(Treated*Post) + Controls + e

**Key Design Choices:**
- Pre-period: 2006-2011
- Post-period: 2013-2016
- 2012 excluded (ambiguous treatment timing)
- Standard errors clustered by state (STATEFIP)
- Survey weights (PERWT) used in preferred specification

### Control Variables:
- Age and age-squared
- Female indicator
- Married indicator
- Education category fixed effects (EDUC)
- State fixed effects (STATEFIP)
- Year fixed effects (in some specifications)

---

## 6. Analysis Commands

```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_19"
python daca_analysis.py
```

### Output Files Generated:
- summary_statistics.csv
- regression_results.csv
- event_study_results.csv

---

## 7. Key Results Summary

### Main DiD Estimates:
| Model | Coefficient | Std. Error | 95% CI |
|-------|-------------|------------|--------|
| Simple DiD (no controls) | 0.0837 | 0.0056 | [0.073, 0.095] |
| DiD with controls | 0.0305 | 0.0061 | [0.019, 0.042] |
| DiD with year FE | 0.0256 | 0.0060 | [0.014, 0.037] |
| **Weighted DiD (preferred)** | **0.0297** | **0.0043** | **[0.021, 0.038]** |

### Interpretation:
DACA eligibility is associated with a 2.97 percentage point increase in full-time employment probability among eligible Mexican-born non-citizens, relative to the control group.

---

## 8. Robustness Checks

### Restricted Age Range (20-40):
- Coefficient: 0.0300 (SE: 0.0041)
- Consistent with main results

### Employment (Any Hours) Outcome:
- Coefficient: 0.0434 (SE: 0.0097)
- Larger effect on employment margin

### By Gender:
- Males: 0.0267 (SE: 0.0062)
- Females: 0.0242 (SE: 0.0079)
- Effects similar across genders

### Placebo Test (Fake 2008 Treatment):
- Coefficient: 0.0156 (SE: 0.0056)
- P-value: 0.006
- Some pre-trend concern, but much smaller than main effect

---

## 9. Event Study Results

Reference year: 2011

| Year | Coefficient | 95% CI |
|------|-------------|--------|
| 2006 | -0.019 | [-0.035, -0.004] |
| 2007 | -0.013 | [-0.024, -0.002] |
| 2008 | -0.005 | [-0.018, 0.009] |
| 2009 | -0.000 | [-0.011, 0.011] |
| 2010 | 0.006 | [-0.009, 0.020] |
| 2013 | 0.009 | [-0.006, 0.024] |
| 2014 | 0.015 | [-0.012, 0.043] |
| 2015 | 0.030 | [0.015, 0.045] |
| 2016 | 0.031 | [0.014, 0.048] |

**Note:** Some evidence of pre-trends in 2006-2007, converging to zero by 2009-2010. Post-DACA effects grow over time.

---

## 10. Preferred Estimate

**Specification:** Weighted DiD with demographic and state fixed effects

**Result:**
- Effect size: 0.0297 (2.97 percentage points)
- Standard error: 0.0043
- 95% CI: [0.0212, 0.0381]
- Sample size: 561,470
- Number of clusters: 51 states

**Selection Rationale:**
1. Survey weights account for complex survey design
2. Controls address compositional differences between groups
3. State fixed effects capture geographic heterogeneity
4. Clustered SEs account for within-state correlation

---

## 11. Limitations and Caveats

1. **Identification of undocumented status:** We cannot directly observe documentation status; use non-citizenship as proxy
2. **Pre-trends:** Some evidence of differential trends in 2006-2007, though effects converge before treatment
3. **Control group selection:** Control group (older non-eligible) may have different labor market dynamics
4. **YRIMMIG measurement:** Year of immigration may have reporting error
5. **Repeated cross-section:** Cannot track individuals over time

---

## 12. Software and Dependencies

- Python 3.x
- pandas
- numpy
- statsmodels
- scipy

---

## End of Log
