# Run Log - Replication 33

## DACA Impact on Full-Time Employment: Independent Replication

### Date: 2026-01-25

---

## 1. Initial Setup and Data Understanding

### 1.1 Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

### 1.2 Data Files
- **data.csv**: Main ACS data file (2006-2016), 6.26 GB, 33,851,424 observations
- **acs_data_dict.txt**: Data dictionary with variable definitions
- **state_demo_policy.csv**: Optional state-level data (not used in this analysis)

### 1.3 Key Variables Identified
From the data dictionary:
- **YEAR**: Census year (2006-2016)
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- **HISPAN**: Hispanic origin (1=Mexican)
- **BPL**: Birthplace (200=Mexico)
- **CITIZEN**: Citizenship status (3=Not a citizen)
- **YRIMMIG**: Year of immigration
- **UHRSWORK**: Usual hours worked per week
- **EMPSTAT**: Employment status (1=Employed)
- **AGE**: Age
- **SEX**: Sex (1=Male, 2=Female)
- **EDUC**: Education level
- **MARST**: Marital status
- **STATEFIP**: State FIPS code

### 1.4 DACA Eligibility Criteria (from instructions)
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007 (YRIMMIG <= 2007)
4. Were present in the US on June 15, 2012 and did not have lawful status (CITIZEN=3)

---

## 2. Identification Strategy

### 2.1 Approach: Difference-in-Differences (DiD)

Given that DACA was implemented on June 15, 2012, I use a difference-in-differences design:
- **Treatment group**: Hispanic-Mexican, Mexican-born non-citizens who meet DACA eligibility criteria
- **Control group**: Similar individuals who do NOT meet DACA eligibility criteria (arrived after age 16, or too old)
- **Pre-period**: 2006-2011 (before DACA)
- **Post-period**: 2013-2016 (after DACA implementation, as specified in instructions)

**Key Decision**: 2012 is excluded since DACA was implemented mid-year (June 15, 2012) and the ACS does not contain month of interview, so we cannot distinguish pre/post observations within 2012.

### 2.2 Sample Restrictions
1. Hispanic-Mexican ethnicity (HISPAN=1)
2. Born in Mexico (BPL=200)
3. Non-citizen (CITIZEN=3) - assuming undocumented per instructions
4. Working-age population (18-64 years old)

### 2.3 Outcome Variable
- **Full-time employment**: Binary indicator for UHRSWORK >= 35

---

## 3. Analysis Commands and Key Decisions

### 3.1 Data Loading
```python
# Used chunked reading due to file size (6.26 GB)
# Read only needed columns to reduce memory usage
use_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
            'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
            'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'LABFORCE', 'UHRSWORK']

# Filter during loading: HISPAN=1, BPL=200, CITIZEN=3
```

### 3.2 Sample Restriction Results
| Step | N |
|------|---|
| Initial sample (full ACS data) | 33,851,424 |
| Hispanic-Mexican, Mexican-born, non-citizen | 701,347 |
| Exclude 2012 (mid-implementation year) | 636,722 |
| Working age 18-64 | 547,614 |

### 3.3 DACA Eligibility Construction
```python
# Age at arrival
age_at_arrival = YRIMMIG - BIRTHYR

# Criterion 1: Arrived before 16th birthday
arrived_before_16 = (age_at_arrival < 16)

# Criterion 2: Not yet 31 as of June 15, 2012
# Born after June 15, 1981
young_enough = (BIRTHYR > 1981) OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)

# Criterion 3: In US since 2007
in_us_since_2007 = (YRIMMIG <= 2007) AND (YRIMMIG > 0)

# Combined eligibility
daca_eligible = arrived_before_16 AND young_enough AND in_us_since_2007
```

### 3.4 Key Descriptive Statistics
- DACA eligible: 71,347 observations (13.03% of sample)
- Full-time employment rate: 58.70% overall
  - Eligible: 52.71%
  - Ineligible: 59.60%
- Mean age: 37.5 years overall
  - Eligible: 23.6 years
  - Ineligible: 39.6 years

---

## 4. Main Results

### 4.1 Preferred Specification (Model 5)
Full difference-in-differences with demographic controls, year fixed effects, and state fixed effects.

**Formula:**
```
fulltime ~ daca_eligible + treat + male + AGE + age_sq + married + educ_hs +
           educ_college + years_in_us + C(YEAR) + C(STATEFIP)
```

**Treatment Effect (DACA eligible x Post):**
- Coefficient: 0.0181
- Standard Error: 0.0041 (clustered by state)
- 95% CI: [0.0101, 0.0261]
- p-value: 0.000009
- Sample Size: 547,614

### 4.2 Interpretation
DACA eligibility is associated with a statistically significant 1.81 percentage point increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born non-citizens.

### 4.3 All Model Specifications
| Model | Treatment Effect | Std. Error | p-value | N |
|-------|-----------------|------------|---------|---|
| (1) Basic | 0.0605 | 0.0032 | <0.001 | 547,614 |
| (2) Demographics | 0.0264 | 0.0043 | <0.001 | 547,614 |
| (3) Year FE | 0.0194 | 0.0041 | <0.001 | 547,614 |
| (4) State+Year FE | 0.0187 | 0.0042 | <0.001 | 547,614 |
| (5) Full (Preferred) | 0.0181 | 0.0041 | <0.001 | 547,614 |

---

## 5. Robustness Checks

### 5.1 Alternative Outcome (Any Employment)
- Treatment Effect: 0.0305 (SE: 0.0073, p<0.001)
- Larger effect on extensive margin than intensive margin

### 5.2 By Gender
- Males: 0.0113 (SE: 0.0037, p=0.002)
- Females: 0.0168 (SE: 0.0068, p=0.013)
- Effect larger for females but both significant

### 5.3 Prime Working Age (25-54)
- Treatment Effect: 0.0096 (SE: 0.0040, p=0.016)
- Smaller but still significant

### 5.4 Placebo Test (Fake 2009 Treatment)
- Using pre-period only (2006-2011) with fake treatment in 2009
- Placebo Effect: -0.0025 (SE: 0.0040, p=0.532)
- Not significant, supporting parallel trends assumption

### 5.5 Event Study
| Year | Coefficient | SE |
|------|------------|-----|
| 2006 | +0.0080 | 0.0093 |
| 2007 | +0.0053 | 0.0053 |
| 2008 | +0.0122 | 0.0092 |
| 2009 | +0.0111 | 0.0081 |
| 2010 | +0.0057 | 0.0104 |
| 2011 | 0 (ref) | - |
| 2013 | +0.0042 | 0.0091 |
| 2014 | +0.0218 | 0.0129 |
| 2015 | +0.0377 | 0.0095 |
| 2016 | +0.0381 | 0.0076 |

Pre-trends are generally flat and not significant, with effects materializing in 2014-2016.

---

## 6. Files Generated

### 6.1 Code Files
- `analysis.py`: Main Python analysis script

### 6.2 Output Files
- `results_summary.txt`: Summary of key results
- `table_main_results.csv`: Main regression results
- `table_robustness.csv`: Robustness check results
- `table_eventstudy.csv`: Event study coefficients

### 6.3 Figures
- `figure1_trends.png`: Full-time employment trends by DACA eligibility
- `figure2_eventstudy.png`: Event study plot
- `figure3_hours_distribution.png`: Distribution of usual hours worked

### 6.4 Final Report
- `replication_report_33.tex`: LaTeX source
- `replication_report_33.pdf`: Final PDF report

---

## 7. Key Methodological Decisions

1. **Excluded 2012**: DACA implemented mid-year, cannot distinguish pre/post
2. **Age restriction 18-64**: Standard working-age population
3. **Non-citizen only**: Per instructions, assume non-citizens are undocumented
4. **Clustered standard errors by state**: Account for within-state correlation
5. **Year and state fixed effects**: Control for aggregate time trends and state-level differences
6. **Control variables**: Male, age, age squared, married, high school completion, some college, years in US

---

## 8. Conclusion

The analysis finds a statistically significant positive effect of DACA eligibility on full-time employment of approximately 1.8 percentage points. This effect is robust across multiple specifications and passes placebo tests for parallel trends. The event study shows effects materializing in the years following DACA implementation (2014-2016), consistent with the expected pattern if DACA caused the improvement.
