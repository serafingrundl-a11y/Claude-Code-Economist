# DACA Replication Study - Run Log

## Project Overview
Independent replication study examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Design
- **Treatment Group**: DACA-eligible individuals aged 26-30 on June 15, 2012
- **Control Group**: Individuals aged 31-35 on June 15, 2012 (would have been eligible but for age)
- **Outcome Variable**: Full-time employment (>=35 hours/week)
- **Method**: Difference-in-Differences (DiD)
- **Pre-treatment Period**: 2006-2011
- **Post-treatment Period**: 2013-2016 (excluding 2012 due to mid-year implementation)

## Key Dates
- DACA Implementation: June 15, 2012
- Applications started: August 15, 2012

---

## Session Log

### Step 1: Data Examination
**Date/Time**: 2026-01-26

**Actions**:
- Read replication_instructions.docx using Python docx library
- Examined data dictionary (acs_data_dict.txt)
- Reviewed data structure (data.csv header)

**Data Files**:
- `data/data.csv`: Main ACS data file (2006-2016), 33.8 million rows
- `data/acs_data_dict.txt`: Data dictionary with variable definitions
- `data/state_demo_policy.csv`: Optional state-level data (not used)

**Key Variables Identified**:
- `YEAR`: Survey year (2006-2016)
- `BIRTHYR`: Birth year
- `BIRTHQTR`: Birth quarter (1-4)
- `HISPAN`: Hispanic origin (1 = Mexican)
- `BPL`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `UHRSWORK`: Usual hours worked per week
- `PERWT`: Person weight for population estimates
- `AGE`: Age at survey
- `EMPSTAT`: Employment status

---

### Step 2: Data Processing and Sample Construction
**Date/Time**: 2026-01-26

**Actions Performed**:
1. Loaded data in chunks (500,000 rows per chunk) to handle memory constraints
2. Applied sequential filters to identify DACA-eligible population
3. Created treatment and control groups based on age on June 15, 2012
4. Excluded 2012 observations (implementation year)

**Sample Restrictions Applied**:
1. HISPAN = 1 (Hispanic-Mexican ethnicity)
2. BPL = 200 (Born in Mexico)
3. CITIZEN = 3 (Non-citizen)
4. YRIMMIG > 0 and YRIMMIG <= YEAR (Valid immigration year)
5. YRIMMIG - BIRTHYR < 16 (Arrived before age 16)
6. YRIMMIG <= 2007 (Continuously in US since 2007)
7. Age on June 15, 2012 in range 26-35 (Treatment or control age)
8. YEAR != 2012 (Exclude implementation year)

**Final Sample**:
- Total observations: 43,238
- Treatment group (ages 26-30): 25,470
- Control group (ages 31-35): 17,768
- Pre-period (2006-2011): 28,377
- Post-period (2013-2016): 14,861

---

### Step 3: Statistical Analysis
**Date/Time**: 2026-01-26

**Analyses Conducted**:

1. **Summary Statistics**
   - Calculated weighted means for all variables by group and period
   - Covariate balance table for pre-period

2. **Simple Difference-in-Differences**
   - Treatment pre-period mean: 0.6305
   - Treatment post-period mean: 0.6597
   - Control pre-period mean: 0.6731
   - Control post-period mean: 0.6433
   - Simple DiD estimate: 0.0590

3. **Regression-Based DiD**
   - Model 1 (Basic): 0.059 (SE: 0.012, p < 0.001)
   - Model 2 (Year FE): 0.057 (SE: 0.012, p < 0.001)
   - Model 3 (+ Controls): 0.044 (SE: 0.011, p < 0.001)
   - Model 4 (+ State FE): 0.043 (SE: 0.011, p < 0.001) **[PREFERRED]**

4. **Robustness Checks**
   - Pre-trend test: coef = 0.003, p = 0.528 (parallel trends supported)
   - Placebo test (2009): coef = 0.006, p = 0.668 (no spurious effect)
   - Alternative outcome (any employment): 0.040 (p < 0.001)
   - Heterogeneity by gender: Males 0.033 (p=0.007), Females 0.049 (p=0.007)

5. **Event Study**
   - Year-specific effects estimated with 2011 as reference
   - Pre-treatment coefficients all insignificant (parallel trends)
   - Post-treatment coefficients positive, largest in 2016 (0.063, p=0.010)

---

### Step 4: Results Documentation
**Date/Time**: 2026-01-26

**Outputs Generated**:
- `results.json`: All numerical results in JSON format
- `yearly_means.csv`: Year-by-year means for plotting
- `summary_stats.csv`: Summary statistics by group/period
- `model4_full.txt`: Full regression output for preferred model
- `replication_report_95.tex`: LaTeX report (~24 pages)
- `replication_report_95.pdf`: Compiled PDF report

---

## Decisions and Justifications

### Decision 1: Sample Restrictions
- **Choice**: Restrict to non-citizens (CITIZEN = 3) to proxy for undocumented status
- **Justification**: The ACS does not directly identify documentation status. Non-citizen status is the best available proxy, as citizens and naturalized immigrants would not be eligible for DACA.

### Decision 2: Age Group Definition
- **Choice**: Use birth year and quarter to precisely calculate age as of June 15, 2012
- **Justification**: The DACA age cutoff was exactly age 31 on June 15, 2012. Using birth quarter accounts for whether the individual's birthday had occurred by mid-June.
- **Formula**: Age = 2012 - BIRTHYR - 1[BIRTHQTR >= 3]

### Decision 3: Full-time Employment Definition
- **Choice**: Full-time defined as UHRSWORK >= 35 hours per week
- **Justification**: This follows the standard Bureau of Labor Statistics (BLS) definition of full-time work.

### Decision 4: Treatment of 2012 Data
- **Choice**: Exclude 2012 from analysis entirely
- **Justification**: DACA was implemented on June 15, 2012. Since the ACS does not record the month of interview, observations from 2012 cannot be classified as pre- or post-treatment.

### Decision 5: Arrival Before Age 16
- **Choice**: Calculate arrival age as YRIMMIG - BIRTHYR and require < 16
- **Justification**: This is a direct implementation of the DACA requirement that applicants arrived in the US before their 16th birthday.

### Decision 6: Continuous Presence Since 2007
- **Choice**: Require YRIMMIG <= 2007
- **Justification**: DACA requires continuous presence in the US since June 15, 2007. Using year of immigration <= 2007 is the closest available proxy.

### Decision 7: Control Group Age Range
- **Choice**: Use ages 31-35 on June 15, 2012 as the control group
- **Justification**: These individuals meet all DACA criteria except the age cutoff, making them a natural comparison group. The 5-year bandwidth matches the treatment group (26-30) and provides adequate sample size while maintaining similarity.

### Decision 8: Preferred Specification
- **Choice**: Model 4 with state and year fixed effects plus individual controls
- **Justification**: Year fixed effects absorb common shocks; state fixed effects account for geographic variation in labor markets and policy environments; individual controls improve precision and account for compositional differences.

### Decision 9: Standard Errors
- **Choice**: Robust (HC1) standard errors
- **Justification**: Heteroskedasticity-robust standard errors are standard practice in applied microeconomics with individual-level data.

### Decision 10: Survey Weights
- **Choice**: Use person weights (PERWT) in all analyses
- **Justification**: ACS sampling is not simple random sampling. Weights are necessary for estimates to be representative of the population.

---

## Commands Executed

```bash
# Read replication instructions
python -c "from docx import Document; doc = Document('replication_instructions.docx'); print('\n'.join([p.text for p in doc.paragraphs]))"

# View data files
ls -la "data/"
head -5 "data/data.csv"

# Run main analysis
python analysis.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_95.tex
pdflatex -interaction=nonstopmode replication_report_95.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_95.tex  # Third pass to finalize
```

---

## Final Results Summary

**Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born non-citizens in the US.

**Preferred Estimate (Model 4)**:
- Effect Size: **0.043** (4.3 percentage points)
- Standard Error: 0.011
- 95% Confidence Interval: [0.022, 0.064]
- p-value: < 0.001

**Interpretation**: DACA eligibility is associated with a statistically significant 4.3 percentage point increase in the probability of full-time employment. This represents approximately a 6.9% increase relative to the pre-period baseline rate of 63.1%.

**Robustness**: The finding is robust to:
- Different specifications (with/without controls and fixed effects)
- Pre-trend test (parallel trends assumption supported)
- Placebo test (no effect at fake treatment date)
- Alternative outcomes (any employment shows similar pattern)
- Subgroup analysis (effects present for both males and females)

---

## Files Generated

| File | Description |
|------|-------------|
| analysis.py | Main Python analysis script |
| results.json | Numerical results in JSON format |
| yearly_means.csv | Year-by-year outcome means |
| summary_stats.csv | Summary statistics |
| model4_full.txt | Full regression output |
| replication_report_95.tex | LaTeX source for report |
| replication_report_95.pdf | Compiled PDF report |
| run_log_95.md | This log file |

---

## Session Complete
**End Time**: 2026-01-26
**Status**: All deliverables generated successfully
