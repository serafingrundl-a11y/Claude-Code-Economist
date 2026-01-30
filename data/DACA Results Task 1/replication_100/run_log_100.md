# Run Log for DACA Replication Study (Run 100)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA employment effects study.

## Date: 2026-01-25

---

## 1. Data Exploration

### 1.1 Initial File Inspection
```bash
# Listed files in data folder
cd "C:\Users\seraf\DACA Results Task 1\replication_100\data" && dir
# Result: acs_data_dict.txt, data.csv, State Level Data Documentation.docx, state_demo_policy.csv

# Counted rows in data.csv
wc -l data.csv
# Result: 33,851,425 rows (including header)
```

### 1.2 Data Dictionary Review
- Reviewed acs_data_dict.txt for variable definitions
- Key variables identified:
  - YEAR: Survey year (2006-2016)
  - HISPAN: Hispanic origin (1=Mexican)
  - BPL: Birthplace (200=Mexico)
  - CITIZEN: Citizenship status (3=Not a citizen)
  - YRIMMIG: Year of immigration
  - BIRTHYR: Birth year
  - BIRTHQTR: Birth quarter
  - UHRSWORK: Usual hours worked per week
  - EMPSTAT: Employment status (1=Employed)
  - AGE, SEX, MARST, EDUC: Demographics

---

## 2. Sample Construction Decisions

### 2.1 Target Population
- **Decision**: Focus on Hispanic-Mexican individuals born in Mexico
- **Variables used**: HISPAN == 1 AND BPL == 200
- **Rationale**: Research question specifically targets "ethnically Hispanic-Mexican Mexican-born people"
- **Result**: 991,261 observations from initial 33.8 million

### 2.2 DACA Eligibility Criteria
Based on the program requirements stated in the instructions:

1. **Arrived before 16th birthday**:
   - Calculated as: YRIMMIG - BIRTHYR < 16
   - Result: 322,246 (32.5%) of Mexican-born Hispanic sample

2. **Under 31 as of June 15, 2012**:
   - Conservative approach: BIRTHYR >= 1982, OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
   - Rationale: Q3 1981 (July-Sept) definitely born after June 15
   - Result: 274,149 (27.7%)

3. **Present in US since June 15, 2007**:
   - YRIMMIG > 0 AND YRIMMIG <= 2007
   - Result: 937,519 (94.6%)

4. **Not a citizen**:
   - CITIZEN == 3 ("Not a citizen")
   - Assumption: Non-citizens without papers are potentially undocumented
   - Result: 701,347 (70.8%)

5. **Combined DACA eligibility**:
   - All four criteria met
   - Result: 133,120 (13.4%)

### 2.3 Analytical Sample Restrictions
- **Age range**: 18-40 years old
  - Rationale: Working-age population where both eligible and ineligible groups exist
- **Citizenship**: Non-citizens only
  - Rationale: Cleaner comparison between eligible and ineligible non-citizens
- **Final analytical sample**: 374,548 observations

---

## 3. Variable Construction

### 3.1 Outcome Variable: Full-Time Employment
- **Definition**: UHRSWORK >= 35
- **Rationale**: Standard definition from instructions ("usually working 35 hours per week or more")
- **Coding**: Binary indicator (1 = full-time, 0 = otherwise)

### 3.2 Treatment Variable
- **DACA eligible (treat)**: 1 if all eligibility criteria met, 0 otherwise
- Among non-citizens: 79,359 treated (21.2%), 295,189 control (78.8%)

### 3.3 Post-Treatment Period
- **Decision**: Post = 1 if YEAR >= 2013
- **Rationale**: DACA implemented June 2012, but 2012 ACS contains mix of pre/post observations
- Pre-period: 2006-2011 (257,814 obs)
- Post-period: 2013-2016 (116,734 obs)

### 3.4 Control Variables
- AGE, AGE^2 (quadratic for nonlinear age effects)
- Female indicator (SEX == 2)
- Married indicator (MARST == 1)
- Education categories (EDUC)
- Year fixed effects
- State fixed effects (STATEFIP)

---

## 4. Empirical Strategy

### 4.1 Identification Approach
- **Method**: Difference-in-Differences (DiD)
- **Treatment group**: DACA-eligible non-citizens
- **Control group**: DACA-ineligible non-citizens (failed one or more eligibility criteria)
- **Key assumption**: Parallel trends in outcomes absent treatment

### 4.2 Model Specifications

**Model 1: Basic DiD**
```
fulltime = b0 + b1*treat + b2*post + b3*treat*post + e
```

**Model 2: DiD with Demographics**
```
fulltime = b0 + b1*treat + b2*post + b3*treat*post +
           b4*AGE + b5*AGE^2 + b6*female + b7*married + b8*EDUC + e
```

**Model 3: DiD with Year and State FE (Preferred)**
```
fulltime = b1*treat + b3*treat*post +
           Year_FE + State_FE +
           b4*AGE + b5*AGE^2 + b6*female + b7*married + e
```

**Model 4: Weighted Analysis**
- Same as Model 2 but using PERWT (person weights) from ACS

### 4.3 Standard Errors
- Clustered at state level (STATEFIP)
- Rationale: Account for within-state correlation in outcomes

---

## 5. Analysis Execution

### 5.1 Command to Run Analysis
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_100" && python analysis.py 2>&1 | tee analysis_output.txt
```

### 5.2 Key Results

**Main DiD Estimate (Model 3 - Preferred):**
- Coefficient: 0.0172
- Standard Error: 0.0060
- 95% CI: [0.0055, 0.0289]
- p-value: 0.0039
- Sample size: 374,548

**Interpretation**: DACA eligibility increased probability of full-time employment by 1.72 percentage points among eligible non-citizen Mexican-born Hispanics.

**Simple DiD Calculation:**
- Treated pre-mean: 0.5033
- Treated post-mean: 0.5471
- Control pre-mean: 0.6096
- Control post-mean: 0.5886
- Raw DiD: 0.0648 (before controls)

---

## 6. Robustness Checks

| Check | Estimate | SE | Note |
|-------|----------|-----|------|
| Employment (any) | 0.0421 | 0.0075 | Stronger effect |
| Ages 18-35 only | 0.0279 | 0.0060 | Similar |
| Males only | 0.0133 | 0.0049 | Smaller effect |
| Females only | 0.0437 | 0.0096 | Larger effect |
| Placebo (2009) | -0.0051 | 0.0039 | Insignificant (good) |

---

## 7. Event Study Results

| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | 0.0175 | 0.0063 | 0.005 |
| 2007 | 0.0113 | 0.0061 | 0.065 |
| 2008 | 0.0180 | 0.0066 | 0.006 |
| 2009 | 0.0162 | 0.0066 | 0.013 |
| 2010 | 0.0083 | 0.0094 | 0.378 |
| 2011 | (reference) | - | - |
| 2012 | -0.0056 | 0.0076 | 0.465 |
| 2013 | 0.0078 | 0.0071 | 0.272 |
| 2014 | 0.0218 | 0.0113 | 0.053 |
| 2015 | 0.0356 | 0.0100 | 0.000 |
| 2016 | 0.0393 | 0.0081 | 0.000 |

**Note on parallel trends**: Pre-treatment coefficients show some variation (not all zero), suggesting potential concerns about parallel trends assumption. Effects clearly emerge in 2015-2016.

---

## 8. Key Decisions and Rationale

1. **Excluded 2012 from main post-period**: ACS 2012 data collected throughout the year, cannot distinguish pre/post DACA within 2012.

2. **Used non-citizens only for control**: Creates cleaner counterfactual. Alternative would include naturalized citizens, but they differ systematically.

3. **Age restriction 18-40**: Ensures overlap in age distributions between treatment and control groups.

4. **State-clustered standard errors**: Standard practice for policy analysis with state-level variation.

5. **Preferred specification includes year and state FE**: Controls for aggregate time trends and state-specific factors.

---

## 9. Output Files Generated

- `analysis.py` - Main analysis script
- `analysis_output.txt` - Full console output
- `summary_stats.csv` - Summary statistics by group/period
- `event_study_coefs.csv` - Event study coefficients
- `replication_report_100.tex` - LaTeX report
- `replication_report_100.pdf` - Final PDF report

---

## 10. Software Environment

- Python 3.x
- Libraries: pandas, numpy, statsmodels, scipy
- Analysis performed: 2026-01-25
