# DACA Replication Analysis - Run Log

## Session Information
- Date: 2026-01-27
- Replication ID: 55
- Analysis Type: Independent clean-room replication

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (defined as working 35+ hours per week)?

## Study Design
- **Treatment group**: DACA-eligible individuals ages 26-30 at time of policy (June 15, 2012)
- **Control group**: Individuals ages 31-35 at time of policy (would have been eligible but for age)
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016 (2012 excluded as treatment year)
- **Outcome**: FT (full-time employment, 1 = 35+ hours/week)

## Data Files
- `prepared_data_numeric_version.csv` - Main analysis dataset
- `prepared_data_labelled_version.csv` - Dataset with labeled values
- `acs_data_dict.txt` - Data dictionary

## Key Variables (from data dictionary)
- `ELIGIBLE`: 1 = treatment group (ages 26-30), 0 = comparison group (ages 31-35)
- `AFTER`: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)
- `FT`: 1 = full-time work (35+ hours/week), 0 = not full-time
- `PERWT`: Person weight for weighted estimates
- Various demographic and state-level policy controls available

## Analysis Steps

### Step 1: Data Loading and Exploration
- Loading prepared_data_numeric_version.csv
- Checking data dimensions and variable distributions
- Verifying ELIGIBLE, AFTER, and FT variables

### Step 2: Basic Difference-in-Differences
- Specification: FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER
- The coefficient on ELIGIBLE*AFTER is the DiD estimate (treatment effect)
- Using person weights (PERWT) for population-representative estimates
- Clustering standard errors at state level for proper inference

### Step 3: Extended Specifications
- Adding demographic controls (sex, education, marital status)
- Adding state-level policy controls
- State and year fixed effects

### Step 4: Robustness Checks
- Parallel trends analysis (pre-treatment years)
- Placebo tests
- Subgroup analysis

---

## Commands and Output Log

### Command 1: Run Analysis Script
```bash
python analysis.py
```

### Key Results Summary

#### Sample Statistics
- Total observations: 17,382
- Treatment group (ELIGIBLE=1): 11,382 (65.5%)
- Control group (ELIGIBLE=0): 6,000 (34.5%)
- Pre-period observations: 9,527
- Post-period observations: 7,855

#### Descriptive Statistics (Weighted)
| Group | Pre-DACA FT Rate | Post-DACA FT Rate | Change |
|-------|------------------|-------------------|--------|
| Treatment (26-30) | 63.69% | 68.60% | +4.91 pp |
| Control (31-35) | 68.86% | 66.29% | -2.57 pp |

**Simple DiD Estimate (Weighted):** 7.48 percentage points

#### Regression Results

| Model | DiD Estimate | Std Error | p-value |
|-------|--------------|-----------|---------|
| Basic OLS | 0.0643 | 0.0153 | <0.001 |
| Basic WLS (weighted) | 0.0748 | 0.0152 | <0.001 |
| WLS + Demographics | 0.0643 | 0.0142 | <0.001 |
| WLS + State FE | 0.0737 | 0.0152 | <0.001 |
| **Full + Clustered SE** | **0.0640** | **0.0219** | **0.004** |

#### Preferred Estimate
- **Effect Size:** 6.40 percentage points (0.064)
- **Standard Error:** 0.022 (clustered by state)
- **95% CI:** [0.021, 0.107]
- **p-value:** 0.004

#### Parallel Trends Analysis
Pre-treatment year-specific effects (reference: 2011):
- 2008: -0.068 (p=0.020)
- 2009: -0.050 (p=0.181)
- 2010: -0.082 (p=0.006)

Post-treatment year-specific effects:
- 2013: 0.016 (p=0.697)
- 2014: 0.000 (p=0.999)
- 2015: 0.001 (p=0.970)
- 2016: 0.074 (p=0.013)

**Note:** Pre-trends show some variation, suggesting caution in interpretation.

#### Placebo Test
- Fake treatment at 2010: 0.018 (p=0.486)
- Result: Not significant, consistent with parallel trends assumption

#### Subgroup Analysis
- Male: 0.072 (p<0.001)
- Female: 0.053 (p=0.070)

---

## Key Decisions Made

1. **Preferred Specification:** WLS with demographic controls and state-clustered standard errors
   - Rationale: Population weights ensure representativeness; clustering accounts for within-state correlation

2. **Controls Included:** SEX, MARST, FAMSIZE, NCHILD
   - Rationale: Standard demographic controls that may affect employment

3. **No sample restrictions beyond provided data**
   - Rationale: Per instructions, entire file constitutes the analytic sample

4. **Interpretation:** Effect is statistically significant but parallel trends show some pre-treatment variation
   - The pre-treatment gaps fluctuate, suggesting some caution in attributing the full effect to DACA

---

## Files Created
- `analysis.py` - Main analysis script (Python)
- `results.json` - Results in JSON format
- `replication_report_55.tex` - LaTeX report (required deliverable)
- `replication_report_55.pdf` - Final PDF report, 24 pages (required deliverable)
- `run_log_55.md` - This log file (required deliverable)

---

## Session Completion
- All required deliverables have been created
- Analysis completed successfully
- PDF compiled without errors (24 pages)
