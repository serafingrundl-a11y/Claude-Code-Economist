# Run Log - DACA Replication Study 26

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA effect on full-time employment study.

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (35+ hours/week)?

**Design:** Difference-in-Differences comparing:
- Treatment group: Ages 26-30 as of June 15, 2012 (born 1982-1986)
- Control group: Ages 31-35 as of June 15, 2012 (born 1977-1981)

---

## Session Start
Date: 2024

## Step 1: Data Exploration

### 1.1 Files in data folder
- `data.csv` - Main ACS data file (6.2 GB)
- `acs_data_dict.txt` - Data dictionary
- `state_demo_policy.csv` - Optional state-level data
- `State Level Data Documentation.docx` - Documentation for state data

### 1.2 Key Variables Identified from Data Dictionary

**Identification Variables:**
- YEAR: Census year (2006-2016)
- SAMPLE: IPUMS sample identifier
- SERIAL: Household serial number
- PERNUM: Person number within household
- PERWT: Person weight

**Demographic Variables:**
- AGE: Age at time of survey
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- SEX: 1=Male, 2=Female

**Ethnicity/Immigration Variables:**
- HISPAN: Hispanic origin (1=Mexican)
- HISPAND: Detailed Hispanic (100-107 = Mexican categories)
- BPL: Birthplace (200=Mexico)
- BPLD: Detailed birthplace (20000=Mexico)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration

**Employment Variables:**
- UHRSWORK: Usual hours worked per week (0-99)
- EMPSTAT: Employment status (1=Employed, 2=Unemployed, 3=Not in labor force)
- LABFORCE: Labor force status

**Other Controls:**
- EDUC/EDUCD: Educational attainment
- MARST: Marital status
- STATEFIP: State FIPS code
- FAMSIZE: Family size
- NCHILD: Number of children

---

## Step 2: Sample Selection Criteria

### 2.1 DACA Eligibility Criteria (from instructions)
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007 (5+ years before DACA)
4. Were present in the US on June 15, 2012 and did not have lawful status

### 2.2 Operationalization Decisions

**Hispanic-Mexican Mexican-born:**
- HISPAN == 1 (Mexican) OR HISPAND in [100-107]
- BPL == 200 (Mexico) OR BPLD == 20000

**Not a citizen (proxy for undocumented):**
- CITIZEN == 3 (Not a citizen)
- Note: Cannot distinguish documented vs undocumented; per instructions, assume non-citizens without papers are undocumented

**Arrived before age 16:**
- Calculate arrival age from YRIMMIG and BIRTHYR
- arrival_age = YRIMMIG - BIRTHYR < 16

**In US by June 2007 (5 years continuous residence):**
- YRIMMIG <= 2007

**Age Groups (as of June 15, 2012):**
- Treatment: Born July 1981 - June 1986 (ages 26-30)
- Control: Born July 1976 - June 1981 (ages 31-35)

**Full-time employment:**
- UHRSWORK >= 35 (usual hours worked per week is 35 or more)

**Time periods:**
- Pre-treatment: 2006-2011 (excluding 2012 as DACA was mid-year)
- Post-treatment: 2013-2016

---

## Step 3: Data Loading and Processing

### 3.1 Command: Load data
```python
python analysis.py
```

### 3.2 Sample Construction Results

| Step | Filter Applied | Observations Remaining |
|------|----------------|------------------------|
| Initial | Full ACS 2006-2016 | 33,851,424 |
| Hispanic-Mexican | HISPAN == 1 | 2,945,521 |
| Born in Mexico | BPL == 200 | 991,261 |
| Non-citizen | CITIZEN == 3 | 701,347 |
| Valid immigration year | YRIMMIG > 0 | 701,347 |
| Arrived before age 16 | arrival_age < 16 | 205,327 |
| In US since 2007 | YRIMMIG <= 2007 | 195,023 |
| Age 26-35 in 2012 | Treatment or Control | 47,418 |
| Exclude 2012 | YEAR != 2012 | 43,238 |

### 3.3 Final Sample Composition

| Group | Pre-Period (2006-2011) | Post-Period (2013-2016) | Total |
|-------|------------------------|-------------------------|-------|
| Treatment (26-30) | 16,694 | 8,776 | 25,470 |
| Control (31-35) | 11,683 | 6,085 | 17,768 |
| **Total** | **28,377** | **14,861** | **43,238** |

---

## Step 4: Difference-in-Differences Analysis

### 4.1 Models Estimated

1. **Model 1**: Basic DiD (no controls)
2. **Model 2**: DiD + demographic controls (female, married, has_children, hs_or_more, college)
3. **Model 3**: DiD + year fixed effects
4. **Model 4**: DiD + year FE + demographic controls
5. **Model 5**: DiD + year FE + state FE + demographic controls (PREFERRED)

### 4.2 Key Results

| Model | DiD Estimate | Std. Error | 95% CI | p-value |
|-------|--------------|------------|--------|---------|
| (1) Basic DiD | 0.0590 | 0.0117 | [0.036, 0.082] | <0.001 |
| (2) + Demographics | 0.0449 | 0.0107 | [0.024, 0.066] | <0.001 |
| (3) + Year FE | 0.0574 | 0.0117 | [0.034, 0.080] | <0.001 |
| (4) + Year FE + Demo | 0.0428 | 0.0107 | [0.022, 0.064] | <0.001 |
| **(5) + State FE** | **0.0421** | **0.0107** | **[0.021, 0.063]** | **<0.001** |

### 4.3 Preferred Estimate

- **Effect Size**: 4.21 percentage points
- **Standard Error**: 0.0107
- **95% Confidence Interval**: [0.0213, 0.0630]
- **Sample Size**: 43,238
- **Interpretation**: DACA eligibility increased full-time employment by approximately 4.2 percentage points among the target population.

---

## Step 5: Robustness Checks

### 5.1 Alternative Outcome: Any Employment
- DiD Estimate: 0.0404
- SE: 0.0101
- 95% CI: [0.0205, 0.0602]

### 5.2 Heterogeneity by Sex
- **Males**: DiD = 0.0293 (SE = 0.0124), N = 24,243
- **Females**: DiD = 0.0544 (SE = 0.0181), N = 18,995
- Effect appears larger for females

### 5.3 Event Study / Pre-Trends

| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2006 | 0.0085 | 0.0228 | [-0.036, 0.053] |
| 2007 | -0.0299 | 0.0224 | [-0.074, 0.014] |
| 2008 | 0.0110 | 0.0228 | [-0.034, 0.056] |
| 2009 | -0.0061 | 0.0235 | [-0.052, 0.040] |
| 2010 | -0.0133 | 0.0233 | [-0.059, 0.032] |
| 2011 | 0 (ref) | - | - |
| 2013 | 0.0339 | 0.0242 | [-0.014, 0.081] |
| 2014 | 0.0350 | 0.0246 | [-0.013, 0.083] |
| 2015 | 0.0209 | 0.0248 | [-0.028, 0.070] |
| 2016 | 0.0637 | 0.0247 | [0.015, 0.112] |

Pre-treatment coefficients are not statistically different from zero, supporting the parallel trends assumption.

---

## Step 6: Outputs Generated

### 6.1 Figures
- `figure_event_study.png` - Event study coefficients plot
- `figure_trends.png` - Full-time employment trends by group
- `figure_difference.png` - Difference between treatment and control

### 6.2 Data Files
- `analysis_results.txt` - Detailed text output
- `event_study_data.csv` - Event study coefficients
- `trend_data.csv` - Employment trends by year and group

### 6.3 Final Deliverables
- `replication_report_26.tex` - LaTeX report
- `replication_report_26.pdf` - Compiled PDF report
- `run_log_26.md` - This run log

---

## Key Analytical Decisions

1. **Sample Definition**: Used HISPAN=1, BPL=200, CITIZEN=3 to identify target population
2. **Age Calculation**: Adjusted for birth quarter to precisely identify those aged 26-30 vs 31-35 as of June 15, 2012
3. **Treatment Definition**: Age-based eligibility (under 31 as of June 15, 2012)
4. **Outcome**: Full-time employment defined as UHRSWORK >= 35
5. **Excluded 2012**: DACA was implemented mid-year, so 2012 observations cannot be cleanly assigned to pre/post
6. **Controls**: Female, married, has children, high school+, college education
7. **Fixed Effects**: Year and state fixed effects in preferred specification
8. **Standard Errors**: Heteroskedasticity-robust (HC1)
9. **Weights**: Person weights (PERWT) used in all regressions

