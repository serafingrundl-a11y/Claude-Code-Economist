# DACA Replication Study - Run Log

## Study Information
- **Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico
- **Design**: Difference-in-Differences (DiD)
- **Treatment Group**: Ages 26-30 in June 2012 (DACA-eligible)
- **Control Group**: Ages 31-35 in June 2012 (not DACA-eligible due to age)
- **Time Period**: Pre-DACA (2008-2011) vs. Post-DACA (2013-2016)

---

## Data Exploration

### Step 1: Read Instructions
- Parsed `replication_instructions.docx` using python-docx library
- Key requirements identified:
  - Use provided ELIGIBLE variable (do not create own)
  - Use provided FT variable for full-time employment (35+ hours/week)
  - Use provided AFTER variable for time period
  - Do not drop observations from the sample
  - Keep individuals not in labor force in analysis

### Step 2: Examine Data Structure
```
Dataset: prepared_data_labelled_version.csv
Observations: 17,382
Variables: 105
```

### Step 3: Key Variable Verification
- **FT**: Binary (0/1), 11,283 full-time (64.9%)
- **ELIGIBLE**: Binary, 11,382 treatment, 6,000 control
- **AFTER**: Binary, 9,527 pre-DACA, 7,855 post-DACA
- **YEAR**: 2008-2011 (pre), 2013-2016 (post), 2012 excluded
- **AGE_IN_JUNE_2012**: Range 26-35, confirms group definitions

---

## Analysis Decisions

### Decision 1: Outcome Variable
- **Choice**: Use FT (full-time employment) as provided
- **Rationale**: Instruction specifies "usually working 35 hours per week or more"
- **Note**: Those not in labor force coded as 0 and retained per instructions

### Decision 2: Treatment Definition
- **Choice**: Use ELIGIBLE variable as provided
- **Rationale**: Instructions explicitly state "use this variable to identify individuals in the treated and comparison groups, and do not create your own eligibility variable"

### Decision 3: Model Specification
- **Choice**: Linear Probability Model with OLS estimation
- **Rationale**:
  - Common in applied economics literature
  - Coefficients directly interpretable as probability changes
  - Easier to include fixed effects
  - Robust to heteroskedasticity with appropriate standard errors

### Decision 4: Standard Errors
- **Choice**: Cluster at state level
- **Rationale**:
  - Treatment varies at state-year level
  - Potential within-state correlation in employment outcomes
  - Standard practice in DiD designs
  - More conservative than HC robust SE

### Decision 5: Control Variables
- **Included**:
  - FEMALE (from SEX)
  - MARRIED (from MARST)
  - NCHILD (number of children)
  - AGE_IN_JUNE_2012
  - Education dummies (from EDUC_RECODE): Some College, Two-Year Degree, BA+
    - Reference: High School Degree (most common category)
- **Fixed Effects**:
  - State (STATEFIP)
  - Year (YEAR)
- **Rationale**: Control for observable differences between treatment/control groups; state FE absorb time-invariant state factors; year FE absorb common time trends

### Decision 6: Not Including Certain Variables
- Did NOT include: state-level policy variables in preferred specification
- **Rationale**:
  - These may be endogenous (affected by DACA implementation)
  - State and year FE capture much of this variation
  - Results robust to including them (Model 5)

---

## Commands Executed

### Python Analysis Script
File: `analysis.py`

Key operations:
1. Load data: `pd.read_csv('data/prepared_data_labelled_version.csv')`
2. Create derived variables:
   - FEMALE from SEX
   - MARRIED from MARST
   - Education dummies from EDUC_RECODE
   - ELIGIBLE_x_AFTER interaction term
3. Run regression specifications:
   - Model 1: Basic DiD (no controls)
   - Model 2: + Demographics
   - Model 3: + Education
   - Model 4: + State/Year FE (preferred)
   - Model 5: + State policies
4. Cluster standard errors at state level using `cov_type='cluster'`

### Figure Generation Script
File: `create_figures.py`

Generated:
- figure1_parallel_trends.png: Time trends by treatment group
- figure2_event_study.png: Year-specific treatment effects
- figure3_did_design.png: DiD design illustration
- figure4_covariates.png: Covariate distributions

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_94.tex  # First pass
pdflatex -interaction=nonstopmode replication_report_94.tex  # Second pass
pdflatex -interaction=nonstopmode replication_report_94.tex  # Third pass (references)
```

---

## Key Results

### Main Finding (Preferred Specification)
- **DiD Estimate**: 0.0544 (5.44 percentage points)
- **Standard Error**: 0.0150 (clustered at state)
- **95% CI**: [0.0249, 0.0839]
- **t-statistic**: 3.615
- **p-value**: 0.0003
- **Sample Size**: 17,382
- **R-squared**: 0.137

### Interpretation
DACA eligibility is associated with a statistically significant 5.44 percentage point increase in the probability of full-time employment.

### Simple DiD Calculation
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Control (31-35) | 66.97% | 64.49% | -2.48 pp |
| Treatment (26-30) | 62.63% | 66.58% | +3.94 pp |
| **DiD** | | | **+6.43 pp** |

### Robustness Checks
1. Without State FE: 0.0543 (0.0141)
2. With Survey Weights: 0.0621 (0.0167)
3. Pre-trend coefficient: 0.0142 (0.0085), p = 0.096

### Event Study Coefficients (relative to 2011)
| Year | Coefficient | SE |
|------|-------------|-----|
| 2008 | -0.0527 | 0.0270 |
| 2009 | -0.0408 | 0.0278 |
| 2010 | -0.0593 | 0.0276 |
| 2011 | 0.0 (ref) | - |
| 2013 | 0.0219 | 0.0283 |
| 2014 | -0.0139 | 0.0286 |
| 2015 | 0.0247 | 0.0290 |
| 2016 | 0.0414 | 0.0293 |

### Subgroup Analysis
- Female: DiD = 0.054 (SE: 0.023)
- Male: DiD = 0.049 (SE: 0.017)
- Married: DiD = 0.016 (SE: 0.019) - not significant
- Not Married: DiD = 0.084 (SE: 0.021) - larger effect

---

## Output Files Generated

1. **replication_report_94.tex** - LaTeX source file (~24 pages)
2. **replication_report_94.pdf** - Compiled PDF report
3. **analysis.py** - Main analysis script
4. **create_figures.py** - Figure generation script
5. **analysis_results.txt** - Summary of key results
6. **figure1_parallel_trends.png** - Parallel trends figure
7. **figure2_event_study.png** - Event study figure
8. **figure3_did_design.png** - DiD design illustration
9. **figure4_covariates.png** - Covariate distributions
10. **run_log_94.md** - This run log

---

## Threats to Validity Noted

1. **Parallel Trends**: Some evidence of differential pre-trends (pre-trend coefficient = 0.014, p = 0.096)
2. **Age-Related Confounders**: Treatment/control differ on age-correlated variables (marriage, children)
3. **Selection into DACA**: Analysis estimates eligibility effect, not actual DACA receipt
4. **Cross-sectional Design**: Cannot track same individuals over time

---

## Software Used

- Python 3.x with pandas, numpy, statsmodels, scipy, matplotlib
- pdfLaTeX (MiKTeX distribution)
- Windows operating system

---

## Timestamp
Analysis completed: 2026-01-27
