# Replication Run Log - Task 43

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability of full-time employment (working 35+ hours per week)?

## Study Design
- **Treatment Group**: DACA-eligible individuals aged 26-30 in June 2012 (ELIGIBLE=1)
- **Control Group**: Individuals aged 31-35 in June 2012, otherwise eligible (ELIGIBLE=0)
- **Pre-treatment Period**: 2008-2011 (AFTER=0)
- **Post-treatment Period**: 2013-2016 (AFTER=1)
- **Outcome**: Full-time employment (FT=1 if working 35+ hours/week)
- **Method**: Difference-in-Differences

## Session Start
Date: 2025-01-27

---

## Step 1: Data Inspection

### Commands Executed
```python
# Load and inspect data
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations: {len(df):,}")  # 17,382
print(f"Number of variables: {len(df.columns)}")  # 105
```

### Data Files
- `prepared_data_numeric_version.csv`: 17,382 observations (excluding header), 105 variables
- `prepared_data_labelled_version.csv`: Same structure with labelled values
- `acs_data_dict.txt`: Data dictionary from IPUMS

### Key Variables Identified
- `FT`: Full-time employment (0/1)
- `AFTER`: Post-DACA indicator (0=2008-2011, 1=2013-2016)
- `ELIGIBLE`: Treatment group indicator (1=ages 26-30 in June 2012, 0=ages 31-35)
- `YEAR`: Survey year (2008-2011, 2013-2016; 2012 excluded)
- `PERWT`: Person weight for ACS
- Various demographic controls available (SEX, AGE, MARST, NCHILD, EDUC_RECODE, METRO, STATEFIP)

### Sample Size by Group and Period
```
                 Pre-DACA (2008-2011)  Post-DACA (2013-2016)  Total
Control (31-35)                  3294                   2706   6000
Treated (26-30)                  6233                   5149  11382
Total                            9527                   7855  17382
```

---

## Step 2: Key Analytical Decisions

### Decision 1: Use provided ELIGIBLE and FT variables
**Rationale**: The instructions specify to use the provided ELIGIBLE variable and not create our own eligibility definition. FT is the pre-constructed outcome variable.

### Decision 2: Include those not in the labor force
**Rationale**: Per instructions, "Those not in the labor force are included, usually as 0 values; keep these individuals in your analysis."

### Decision 3: Use survey weights (PERWT)
**Rationale**: ACS data should be weighted to produce population-representative estimates. Used weighted least squares (WLS) in all primary specifications.

### Decision 4: Include demographic controls
**Rationale**: Controls improve precision and address observable differences between treatment and control groups. Included:
- Female indicator (from SEX)
- Married indicator (from MARST)
- Age (continuous)
- Number of children (NCHILD)
- Education dummies (from EDUC_RECODE): High School, Some College, Two-Year Degree, BA+
- Metro area indicator (from METRO)

### Decision 5: Include state and year fixed effects
**Rationale**: State FE control for time-invariant state-level factors. Year FE control for national trends affecting both groups. This is the preferred specification.

### Decision 6: Use OLS/Linear Probability Model
**Rationale**: OLS with binary outcome is standard in DiD analysis, provides easy interpretation, and allows for straightforward inclusion of fixed effects.

---

## Step 3: Analysis Execution

### Commands Executed
```bash
# Run main analysis
cd "C:\Users\seraf\DACA Results Task 3\replication_43"
python analysis.py

# Generate figures
python create_figures.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_43.tex
pdflatex -interaction=nonstopmode replication_report_43.tex  # 2nd pass for refs
pdflatex -interaction=nonstopmode replication_report_43.tex  # 3rd pass
```

### Python Analysis Script: analysis.py
- Loaded data from CSV
- Calculated simple DiD
- Ran 6 regression models with increasing controls
- Computed robust and state-clustered standard errors
- Conducted event study analysis
- Performed placebo test
- Generated subgroup analyses by sex
- Exported results to CSV files

### Visualization Script: create_figures.py
Generated 5 figures:
1. Full-time employment trends by group over time
2. Event study plot
3. DiD illustration
4. Model comparison (coefficient stability)
5. Sample size distribution by year

---

## Step 4: Results Summary

### Simple Difference-in-Differences
```
Control group (ages 31-35):
  Pre-DACA FT rate:  0.6697
  Post-DACA FT rate: 0.6449
  Change:            -0.0248

Treatment group (ages 26-30):
  Pre-DACA FT rate:  0.6263
  Post-DACA FT rate: 0.6658
  Change:            0.0394

Difference-in-Differences: 0.0643
```

### Regression Results Summary

| Model | Coefficient | SE | p-value | Controls | Weights | State FE | Year FE |
|-------|-------------|-----|---------|----------|---------|----------|---------|
| 1 (Basic) | 0.0643 | 0.0153 | <0.001 | No | No | No | No |
| 2 (Weighted) | 0.0748 | 0.0152 | <0.001 | No | Yes | No | No |
| 3 (+Controls) | 0.0654 | 0.0142 | <0.001 | Yes | Yes | No | No |
| 4 (+State FE) | 0.0649 | 0.0142 | <0.001 | Yes | Yes | Yes | No |
| 5 (+Year FE) | 0.0627 | 0.0142 | <0.001 | Yes | Yes | No | Yes |
| **6 (Full/Preferred)** | **0.0621** | **0.0142** | **<0.001** | **Yes** | **Yes** | **Yes** | **Yes** |

### Preferred Estimate (Model 6)
- **Effect Size**: 0.0621 (6.21 percentage points)
- **Standard Error**: 0.0142
- **95% Confidence Interval**: [0.0343, 0.0899]
- **p-value**: 0.000012
- **Sample Size**: 17,382

### Robustness Checks

**Alternative Standard Errors:**
- Robust (HC1) SE: 0.0167 (p = 0.0002)
- State-Clustered SE: 0.0214 (p = 0.0055)

**Event Study:**
- Pre-treatment coefficients (relative to 2011): -0.067 (2008), -0.046 (2009), -0.075 (2010)
- Post-treatment coefficients: 0.019 (2013), -0.015 (2014), -0.008 (2015), 0.065 (2016)
- Some pre-trend variation but no clear systematic trend

**Placebo Test:**
- Coefficient: 0.018
- p-value: 0.340
- Interpretation: No significant effect at placebo cutoff, supports parallel trends

**Subgroup Analysis:**
- Males: 0.061 (SE = 0.017)
- Females: 0.052 (SE = 0.023)
- Effects similar across genders

---

## Step 5: Interpretation

### Main Finding
DACA eligibility is associated with a 6.2 percentage point increase in the probability of full-time employment among DACA-eligible Mexican-born Hispanic individuals, relative to the trend experienced by the control group.

### Economic Significance
- Baseline (pre-DACA treatment group): 62.6% full-time employment
- Effect represents approximately a 10% increase relative to baseline
- Statistically significant at the 1% level in all specifications

### Caveats
1. Event study shows some pre-treatment variation, though no clear systematic trend
2. Intent-to-treat estimate (not all eligible applied/received DACA)
3. Age-based comparison has inherent differences
4. Cross-sectional data, not panel

---

## Step 6: Output Files Generated

### Analysis Output
- `regression_results.csv`: Main regression results
- `event_study_results.csv`: Event study coefficients
- `summary_statistics.csv`: Descriptive statistics

### Figures
- `figure1_ft_trends.png/pdf`: Employment trends over time
- `figure2_event_study.png/pdf`: Event study plot
- `figure3_did_illustration.png/pdf`: DiD diagram
- `figure4_model_comparison.png/pdf`: Coefficient comparison
- `figure5_sample_distribution.png/pdf`: Sample sizes by year

### Report
- `replication_report_43.tex`: LaTeX source (18 pages)
- `replication_report_43.pdf`: Final PDF report

---

## Deliverables Checklist

- [x] `replication_report_43.tex` - LaTeX source file
- [x] `replication_report_43.pdf` - Compiled PDF report (~18 pages)
- [x] `run_log_43.md` - This run log

---

## Session End
Completed: 2025-01-27
