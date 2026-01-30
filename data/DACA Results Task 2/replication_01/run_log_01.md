# Run Log - DACA Replication Study 01

## Overview
Independent replication of the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States using a difference-in-differences design.

## Key Dates and Version Info
- Analysis started: 2026-01-25
- Data source: American Community Survey (ACS) via IPUMS (2006-2016)

---

## Step 1: Data Exploration and Understanding

### Files Available:
- `data.csv`: Main ACS data file (~33.8 million rows)
- `acs_data_dict.txt`: IPUMS data dictionary
- `state_demo_policy.csv`: Optional state-level data
- `State Level Data Documentation.docx`: Documentation for state data

### Key Variables Identified from Data Dictionary:
- **YEAR**: Census year (2006-2016)
- **HISPAN/HISPAND**: Hispanic origin (1 = Mexican)
- **BPL/BPLD**: Birthplace (200 = Mexico)
- **CITIZEN**: Citizenship status (3 = Not a citizen)
- **YRIMMIG**: Year of immigration
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth
- **AGE**: Age at survey time
- **UHRSWORK**: Usual hours worked per week (>=35 = full-time)
- **EMPSTAT**: Employment status
- **PERWT**: Person weight for population estimates

### Research Design Decisions:

1. **Treatment Group**: Ages 26-30 as of June 15, 2012
   - Born between June 16, 1981 and June 15, 1986

2. **Control Group**: Ages 31-35 as of June 15, 2012
   - Born between June 16, 1976 and June 15, 1981

3. **Pre-treatment Period**: 2006-2011 (excluding 2012 due to mid-year implementation)

4. **Post-treatment Period**: 2013-2016

5. **Sample Restrictions**:
   - Hispanic-Mexican ethnicity (HISPAN = 1)
   - Born in Mexico (BPL = 200)
   - Not a citizen (CITIZEN = 3)
   - Assume non-citizens without papers are undocumented

6. **Outcome Variable**: Full-time employment
   - Defined as UHRSWORK >= 35 hours per week

---

## Step 2: Data Preparation

### Commands and Code:

Created `analysis.py` with the following key operations:

```python
# Load data in chunks (large file ~33.8M rows)
for chunk in pd.read_csv('data/data.csv', chunksize=500000):
    # Filter to Hispanic-Mexican, Mexican-born, non-citizens
    filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3)
    ]
```

### Sample Sizes After Filtering:
- Total filtered sample: 701,347 observations
- After age group assignment: 181,229 observations
  - Treatment group: 84,681
  - Control group: 96,548
- After excluding 2012: 164,874 observations
  - Pre-period (2006-2011): 102,280
  - Post-period (2013-2016): 62,594

---

## Step 3: Treatment/Control Group Assignment

### Logic for Age Group Assignment:
Using BIRTHYR and BIRTHQTR to determine precise birth timing relative to June 15, 2012:

**Treatment Group (ages 26-30 on June 15, 2012):**
- Born 1982-1985: Definitely in treatment range
- Born 1986 Q1-Q2: In treatment range (before June 16)
- Born 1981 Q3-Q4: In treatment range (after June 15)

**Control Group (ages 31-35 on June 15, 2012):**
- Born 1977-1980: Definitely in control range
- Born 1981 Q1-Q2: In control range (before June 16)
- Born 1976 Q3-Q4: In control range (after June 15)

---

## Step 4: Variable Creation

### Outcome Variable:
- `fulltime`: Binary indicator for UHRSWORK >= 35 hours per week
- Mean full-time employment rate: 60.58%

### Treatment Variables:
- `treat`: Binary indicator for treatment group
- `post`: Binary indicator for post-2012 period
- `treat_post`: Interaction term for DiD

### Control Variables:
- `female`: Binary indicator for SEX == 2
- `married`: Binary indicator for MARST in [1, 2]
- `educ_lths`: Less than high school (EDUC < 6)
- `educ_hs`: High school (EDUC == 6)
- `educ_somecol`: Some college or more (EDUC > 6)

---

## Step 5: Regression Analysis

### Models Estimated:

1. **Model 1**: Basic DiD (unweighted)
2. **Model 2**: Basic DiD (weighted by PERWT)
3. **Model 3**: DiD with demographic controls (weighted)
4. **Model 4**: DiD with year fixed effects (weighted)
5. **Model 5 (PREFERRED)**: DiD with year and state fixed effects (weighted)

### Key Results:

| Model | Coefficient | Std. Error | P-value |
|-------|-------------|------------|---------|
| Model 1 (Basic, unweighted) | 0.0160 | 0.005 | 0.001 |
| Model 2 (Basic, weighted) | 0.0256 | 0.006 | <0.001 |
| Model 3 (Controls) | 0.0185 | 0.005 | <0.001 |
| Model 4 (Year FE) | 0.0170 | 0.005 | <0.001 |
| Model 5 (Year + State FE) | 0.0169 | 0.005 | <0.001 |

### PREFERRED ESTIMATE (Model 5):
- **Effect Size**: 0.0169 (1.69 percentage points)
- **Standard Error**: 0.00505
- **95% CI**: [0.007, 0.027]
- **Sample Size**: 164,874
- **P-value**: 0.0008

---

## Step 6: Robustness Checks

### Alternative Outcomes:
1. **Any Employment (UHRSWORK > 0)**: Coef = 0.0106, SE = 0.0044, p = 0.016
2. **Employment Status (EMPSTAT == 1)**: Coef = 0.0165, SE = 0.0048, p < 0.001

### Heterogeneity by Gender:
- Males: Coef = 0.0229, SE = 0.0059 (significant)
- Females: Coef = -0.0009, SE = 0.0085 (not significant)

---

## Step 7: Parallel Trends Assessment

### Event Study Results (Reference Year: 2011):
| Year | Coefficient | Std. Error | P-value |
|------|-------------|------------|---------|
| 2006 | -0.0263 | 0.0110 | 0.017 |
| 2007 | -0.0250 | 0.0109 | 0.022 |
| 2008 | -0.0184 | 0.0111 | 0.096 |
| 2009 | -0.0064 | 0.0114 | 0.577 |
| 2010 | -0.0238 | 0.0113 | 0.036 |
| 2011 | (reference) | - | - |
| 2013 | 0.0014 | 0.0116 | 0.902 |
| 2014 | 0.0074 | 0.0114 | 0.519 |
| 2015 | -0.0044 | 0.0115 | 0.698 |
| 2016 | -0.0042 | 0.0116 | 0.718 |

Note: Pre-treatment coefficients show some variation, suggesting potential concerns about parallel trends assumption.

---

## Step 8: Output Files Generated

1. `results.json`: All numerical results
2. `model_summaries.txt`: Full regression output
3. `yearly_means.csv`: Yearly full-time employment rates by group
4. `event_study.csv`: Event study coefficients

---

## Key Decisions and Rationale

1. **Sample Definition**: Used HISPAN=1 (Mexican), BPL=200 (Mexico), CITIZEN=3 (not a citizen) to identify likely DACA-eligible population. This is consistent with IPUMS documentation and the fact that we cannot directly observe undocumented status.

2. **Excluding 2012**: Since DACA was implemented in June 2012, observations from 2012 cannot be cleanly assigned to pre or post period (ACS doesn't include month of interview).

3. **Age Groups**: Used birth year and quarter to precisely assign individuals to treatment (26-30) and control (31-35) groups as of June 15, 2012.

4. **Full-time Definition**: Standard definition of 35+ hours per week based on UHRSWORK variable.

5. **Preferred Model**: Model 5 with year and state fixed effects controls for time trends and state-level heterogeneity, providing more credible estimates.

6. **Robust Standard Errors**: Used HC1 (heteroskedasticity-consistent) standard errors for all regressions.

---

## Interpretation

The preferred estimate suggests DACA eligibility increased full-time employment by approximately 1.7 percentage points among the treatment group relative to the control group. This effect is statistically significant at conventional levels (p < 0.001).

However, the event study results show some pre-trend differences in 2006-2007 and 2010, which may raise concerns about the parallel trends assumption. The post-treatment coefficients are small and statistically insignificant when estimated year-by-year, though the pooled DiD estimate remains significant.

The effect appears to be driven primarily by males, with no significant effect found for females. This heterogeneity is worth noting in the interpretation.
