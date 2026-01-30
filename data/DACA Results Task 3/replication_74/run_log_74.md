# Replication Run Log - ID 74

## Project: DACA Effect on Full-Time Employment

### Research Question
What was the causal impact of eligibility for DACA on the probability of full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals living in the United States?

### Identification Strategy
- **Treated Group**: Eligible individuals aged 26-30 at the time DACA went into effect (June 2012)
- **Control Group**: Individuals aged 31-35 at the time DACA went into effect (otherwise would have been eligible)
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016 (2012 excluded as treatment timing unclear)

---

## Session Log

### Step 1: Initial Setup and Data Exploration
**Timestamp**: Session Start

**Actions**:
1. Read replication instructions from `replication_instructions.docx`
2. Reviewed data dictionary in `data/acs_data_dict.txt`
3. Examined data structure in `data/prepared_data_numeric_version.csv`

**Key Variables Identified**:
- `FT`: Full-time employment (1 = yes, 0 = no) - OUTCOME
- `ELIGIBLE`: Treatment group indicator (1 = treated ages 26-30, 0 = control ages 31-35)
- `AFTER`: Post-treatment period indicator (1 = 2013-2016, 0 = 2008-2011)
- `PERWT`: Person-level survey weights

**Key Design Decisions**:
1. Use provided ELIGIBLE variable (do not create own eligibility criteria)
2. Use provided AFTER variable for pre/post distinction
3. Include all observations in the sample (do not drop based on characteristics)
4. Those not in labor force are included as FT=0

---

### Step 2: Data Exploration
**Actions**:
1. Loaded `prepared_data_numeric_version.csv` (17,382 observations)
2. Confirmed years in data: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
3. Verified key variables: FT, ELIGIBLE, AFTER, PERWT

**Sample Size Distribution**:
- Control Group (31-35): 6,000 observations
  - Pre-period: 3,294
  - Post-period: 2,706
- Treatment Group (26-30): 11,382 observations
  - Pre-period: 6,233
  - Post-period: 5,149

---

### Step 3: Difference-in-Differences Analysis
**Methodology**:
- Weighted Least Squares (WLS) regression using person weights (PERWT)
- Heteroskedasticity-robust standard errors (HC1)
- Multiple model specifications for robustness

**Models Estimated**:
1. Model 1: Basic DiD (ELIGIBLE + AFTER + ELIGIBLE*AFTER)
2. Model 2: DiD + Demographics (sex, age, marital status, education)
3. Model 3: DiD + Year Fixed Effects
4. Model 4: Full Model (Demographics + Year FE)
5. Model 5: Full Model + State Fixed Effects

**Key Commands (Python)**:
```python
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Basic DiD
X = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']]
X = sm.add_constant(X)
model = sm.WLS(df['FT'], X, weights=df['PERWT']).fit(cov_type='HC1')
```

---

### Step 4: Results Summary

**Preferred Estimate (Model 1: Basic DiD)**:
- Effect: 0.0748 (7.48 percentage points)
- SE: 0.0181
- 95% CI: [0.039, 0.110]
- p-value: 0.000036
- Sample: 17,382

**Robustness Across Specifications**:
| Model | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| Basic DiD | 0.0748 | 0.0181 | 0.0000 |
| + Demographics | 0.0625 | 0.0167 | 0.0002 |
| + Year FE | 0.0721 | 0.0181 | 0.0001 |
| Full Model | 0.0599 | 0.0167 | 0.0003 |
| + State FE | 0.0592 | 0.0166 | 0.0004 |

**Parallel Trends Test**:
- Joint F-test of pre-trend coefficients: F = 1.96, p = 0.118
- Cannot reject null of parallel trends at conventional levels

**Heterogeneity**:
- Male: 0.072 (SE: 0.020)
- Female: 0.053 (SE: 0.028)
- Higher education shows larger effects

---

### Step 5: Figures Generated
1. `figure1_parallel_trends.png` - FT rates by group over time
2. `figure2_event_study.png` - Event study coefficients
3. `figure3_robustness.png` - Estimates across specifications

---

### Step 6: Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Estimator | WLS with person weights | Survey data requires weighting |
| Standard Errors | HC1 robust | Heteroskedasticity likely present |
| Reference Year (Event Study) | 2011 | Last pre-treatment year |
| Education Reference | Less than High School | Most common category |
| Model Preference | Basic DiD | Transparent, avoids over-specification |

---

### Step 7: Report Generation
**Actions**:
1. Created LaTeX replication report (`replication_report_74.tex`)
2. Compiled to PDF (24 pages)
3. Included all tables, figures, and appendices

**Report Structure**:
- Abstract and Introduction
- Data description
- Methodology
- Main results
- Robustness checks
- Heterogeneity analysis
- Discussion and limitations
- Technical appendix

---

### Step 8: Final Deliverables
**Files Created**:
1. `replication_report_74.tex` - LaTeX source
2. `replication_report_74.pdf` - Compiled report (24 pages)
3. `run_log_74.md` - This log file
4. `analysis_74.py` - Python analysis script
5. `figure1_parallel_trends.png` - Trends visualization
6. `figure2_event_study.png` - Event study plot
7. `figure3_robustness.png` - Robustness comparison
8. `regression_results.csv` - Summary results
9. `yearly_rates.csv` - Year-by-year rates

---

## Final Summary

**Research Question**: Effect of DACA eligibility on full-time employment

**Preferred Estimate**:
- Coefficient: 0.0748 (7.48 percentage points)
- Standard Error: 0.0181
- 95% CI: [0.039, 0.110]
- p-value: < 0.001
- Sample Size: 17,382

**Interpretation**: DACA eligibility increased the probability of full-time employment by approximately 7.5 percentage points among Mexican-born Hispanic individuals. This effect is statistically significant and robust across multiple model specifications.

**Key Findings**:
1. Positive and statistically significant effect across all specifications
2. Effect ranges from 5.9 to 7.5 percentage points depending on controls
3. Parallel trends assumption supported (joint F-test p = 0.118)
4. Larger effects for higher education levels
5. Slightly larger effects for men than women

---

