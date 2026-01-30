# Replication Run Log - Session 26

## Project Overview
**Research Question:** What was the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican Mexican-born individuals in the United States?

**Treatment Group:** Individuals aged 26-30 at the time DACA went into effect (June 15, 2012)
**Control Group:** Individuals aged 31-35 at the time DACA went into effect
**Outcome:** Full-time employment (FT = 1 if usually working 35+ hours/week)
**Method:** Difference-in-Differences (comparing change in treated group vs. control group before/after DACA)

---

## Session Log

### Step 1: Read Instructions and Data Dictionary
- Read `replication_instructions.docx` - extracted full research specifications
- Key variables identified:
  - `ELIGIBLE`: 1 for treatment group (ages 26-30 in June 2012), 0 for control (ages 31-35)
  - `FT`: Full-time employment outcome (1 = yes, 0 = no)
  - `AFTER`: 1 for years 2013-2016, 0 for years 2008-2011
  - `PERWT`: Person weights for weighted analysis
- Data spans 2008-2016, excluding 2012
- Sample is pre-selected for Hispanic-Mexican Mexican-born individuals who meet other DACA criteria

### Step 2: Data Exploration
- Loaded `prepared_data_numeric_version.csv` using Python/pandas
- Dataset shape: 17,382 observations x 105 variables
- Years in sample: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- Sample breakdown:
  - Pre-DACA (2008-2011): 9,527 observations
  - Post-DACA (2013-2016): 7,855 observations
  - Eligible group: 11,382 observations
  - Comparison group: 6,000 observations

### Step 3: Variable Coding Verification
- Confirmed SEX coding: 1 = Male, 2 = Female (IPUMS standard)
- Confirmed MARST coding: 1-2 = Married, 3-6 = Not married
- Education recode verified: categorical strings (Less than High School, High School Degree, Some College, Two-Year Degree, BA+)

### Step 4: Basic Difference-in-Differences Calculation
- Calculated unweighted cell means:
  - Eligible, Pre: 0.626
  - Eligible, Post: 0.666
  - Comparison, Pre: 0.670
  - Comparison, Post: 0.645
- Raw unweighted DiD estimate: 0.0643
- Weighted DiD estimate: 0.0748

### Step 5: Regression Analysis

#### Model 1: Basic OLS
- FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER
- DiD coefficient: 0.0643 (SE: 0.0153), p < 0.001
- N = 17,382

#### Model 2: Weighted Least Squares with State-Clustered SE (PREFERRED)
- FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER
- Weights: PERWT
- Standard errors: Clustered by STATEFIP
- DiD coefficient: **0.0748** (SE: 0.0203), p < 0.001
- 95% CI: [0.035, 0.115]
- N = 17,382

#### Model 3: With Demographic Controls
- Added: FEMALE, MARRIED, NCHILD, education dummies (reference: Less than HS)
- DiD coefficient: 0.0640 (SE: 0.0217), p = 0.003

#### Model 4: With Year Fixed Effects
- Replaced AFTER with year dummies
- DiD coefficient: 0.0612 (SE: 0.0210), p = 0.004

#### Model 5: With State Fixed Effects
- DiD coefficient: 0.0635 (SE: 0.0167), p < 0.001

### Step 6: Robustness Checks

#### Pre-Trends Test (2008-2011 only)
- ELIGIBLE x Year 2009: 0.018 (p = 0.519)
- ELIGIBLE x Year 2010: -0.014 (p = 0.523)
- ELIGIBLE x Year 2011: 0.068 (p = 0.021) - significant, potential concern
- Conclusion: Generally supportive of parallel trends, though 2011 shows some divergence

#### Event Study
- Pre-treatment interactions mostly negative
- Post-treatment effects positive, largest in 2016 (0.074, p = 0.013)
- Suggests effects may have grown over time

#### Logit Model
- Coefficient: 0.283, Marginal effect at means: 0.064

#### Probit Model
- Coefficient: 0.174, Marginal effect at means: 0.064

### Step 7: Heterogeneity Analysis

#### By Sex
- Males: DiD = 0.072 (SE: 0.019), p < 0.001
- Females: DiD = 0.053 (SE: 0.029), p = 0.070
- Triple-difference interaction: -0.019 (p = 0.418, not significant)
- Conclusion: Both sexes benefit, no significant difference between them

### Step 8: Report Generation
- Created comprehensive LaTeX report (~20 pages)
- Compiled to PDF using pdflatex (3 passes for cross-references)
- Final output: 18 pages

---

## Key Decisions and Rationale

### 1. Preferred Specification Choice
**Decision:** Weighted least squares with state-clustered standard errors, without additional controls

**Rationale:**
- Survey weights (PERWT) are necessary for nationally representative estimates given ACS sampling design
- State-clustered standard errors account for geographic concentration (45% California, 21% Texas) and potential within-state error correlation
- No demographic controls in preferred model because: (a) controls are predetermined at baseline and should not affect treatment, (b) adding controls could introduce bias if they are affected by treatment anticipation, (c) results are robust with and without controls

### 2. Treatment of Missing Education
**Decision:** Kept observations with missing education (N=3 with NA)

**Rationale:** Very small number; dropping would not materially affect results

### 3. Not Limiting Sample Further
**Decision:** Used full provided sample without additional restrictions

**Rationale:** Instructions explicitly stated "do not further limit the sample by dropping individuals"

### 4. Including Those Not in Labor Force
**Decision:** Kept all individuals including those not in labor force (coded as FT=0)

**Rationale:** Instructions explicitly stated to keep these individuals; extensive margin effects are substantively important

---

## Summary of Results

| Specification | Effect Size | SE | 95% CI | p-value |
|--------------|-------------|------|--------|---------|
| Basic OLS | 0.0643 | 0.0153 | [0.034, 0.094] | <0.001 |
| **Weighted + Clustered (Preferred)** | **0.0748** | **0.0203** | **[0.035, 0.115]** | **<0.001** |
| With Controls | 0.0640 | 0.0217 | [0.022, 0.107] | 0.003 |
| Year Fixed Effects | 0.0612 | 0.0210 | [0.020, 0.102] | 0.004 |
| State Fixed Effects | 0.0635 | 0.0167 | [0.031, 0.096] | <0.001 |
| Logit (marginal effect) | 0.0644 | - | - | <0.001 |

**Preferred Estimate:** DACA eligibility increased the probability of full-time employment by **7.5 percentage points** (95% CI: 3.5 to 11.5 pp)

---

## Output Files Created
1. `replication_report_26.tex` - LaTeX source file
2. `replication_report_26.pdf` - Compiled PDF report (18 pages)
3. `run_log_26.md` - This run log

---

## Software Used
- Python 3.x with pandas, numpy, statsmodels
- pdflatex (MiKTeX distribution) for PDF compilation
