# Run Log - DACA Replication Study (ID: 32)

## Overview
This document logs all commands and key decisions made during the independent replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (defined as usually working 35+ hours per week)?

## Data Source
- American Community Survey (ACS) 2006-2016 via IPUMS
- Data file: data.csv (~33.8 million observations, ~34GB)
- Data dictionary: acs_data_dict.txt

---

## Key Analytical Decisions

### 1. Sample Definition
- **Hispanic-Mexican ethnicity**: HISPAN == 1
- **Born in Mexico**: BPL == 200
- **Non-citizen (proxy for undocumented)**: CITIZEN == 3
- **Working age**: Ages 18-64
- **Decision rationale**: Follow instructions exactly; assume non-citizens without papers are undocumented

### 2. DACA Eligibility Definition
Eligibility requires ALL of:
1. **Age at arrival < 16**: (YRIMMIG - BIRTHYR) < 16
2. **Born after June 15, 1981**: BIRTHYR > 1981 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
3. **Immigrated by 2007**: YRIMMIG <= 2007 AND YRIMMIG > 0

**Decision rationale**: These are the observable DACA criteria in ACS data. Educational enrollment and criminal history cannot be observed.

### 3. Treatment Period Definition
- **Pre-period**: 2006-2011 (6 years)
- **Post-period**: 2013-2016 (4 years)
- **Excluded**: 2012 (transition year - DACA announced June 15, 2012, but ACS doesn't record month)
- **Decision rationale**: Cannot distinguish pre/post within 2012; clean separation of periods

### 4. Outcome Variable
- **Full-time employment**: UHRSWORK >= 35 (binary indicator)
- **Decision rationale**: Follows exact definition in research question

### 5. Control Group
- Mexican-born, Hispanic-Mexican non-citizens who do NOT meet DACA eligibility criteria
- Includes: Those who arrived after age 16, too old as of 2012, or arrived after 2007
- **Decision rationale**: Same demographic group but ineligible for DACA provides counterfactual

### 6. Identification Strategy
- **Difference-in-Differences (DiD)** design
- Compare changes in full-time employment between eligible vs non-eligible before vs after DACA
- **Decision rationale**: Standard approach for policy evaluation with discrete eligibility cutoffs

### 7. Model Specifications
1. Basic DiD (no controls)
2. DiD with demographics (age, age², female)
3. DiD with year FE + demographics (PREFERRED)
4. DiD with state FE + year FE + demographics

**Preferred specification rationale**: Year FE flexibly controls for common time trends; demographics control for compositional differences; state FE not needed (results nearly identical)

### 8. Statistical Methods
- Weighted least squares using PERWT (ACS person weights)
- Heteroskedasticity-robust (HC1) standard errors
- **Decision rationale**: Population-representative estimates with appropriate standard errors

---

## Session Commands

### Data Exploration
```bash
# Check file structure
dir data
head -5 data/data.csv
wc -l data/data.csv  # Result: 33,851,425 rows
```

### Analysis Script Execution
```bash
python analysis_script.py
```

Key output:
- Total filtered observations (Mexican-born, Hispanic, non-citizen): 701,347
- Working-age sample (18-64): 603,425
- Analysis sample (excluding 2012): 547,614
- DACA eligible: ~70,000 in analysis sample
- Not DACA eligible: ~477,000 in analysis sample

### Figure Generation
```bash
python create_figures.py
```

Generated:
- figure1_event_study.png/pdf
- figure2_trends.png/pdf
- figure3_specifications.png/pdf

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_32.tex  # Run 3 times for references
```

Output: replication_report_32.pdf (24 pages)

---

## Main Results Summary

### Preferred Estimate (Model 3: Year FE + Demographics)
| Statistic | Value |
|-----------|-------|
| DiD Coefficient | 0.0189 |
| Standard Error | 0.0046 |
| 95% CI | [0.0099, 0.0278] |
| t-statistic | 4.14 |
| p-value | < 0.001 |
| Sample Size | 547,614 |

**Interpretation**: DACA eligibility increased the probability of full-time employment by approximately 1.9 percentage points.

### Results Across Specifications
| Model | DiD Coef | SE | N |
|-------|----------|-----|------|
| (1) Basic | 0.0683 | 0.0049 | 547,614 |
| (2) Demographics | 0.0272 | 0.0046 | 547,614 |
| (3) Year FE (PREFERRED) | 0.0189 | 0.0046 | 547,614 |
| (4) State + Year FE | 0.0182 | 0.0045 | 547,614 |

### Event Study Results
Pre-treatment coefficients (2006-2010): All insignificant (supports parallel trends)
Post-treatment coefficients:
- 2013: 0.0123 (NS)
- 2014: 0.0266 (p<0.05)
- 2015: 0.0435 (p<0.05)
- 2016: 0.0438 (p<0.05)

### Robustness Checks
| Specification | DiD Coef | SE |
|---------------|----------|-----|
| Main (ages 18-64) | 0.0189 | 0.0046 |
| Ages 16-35 | 0.0062 | 0.0047 |
| Any employment | 0.0290 | 0.0044 |
| Hours worked (continuous) | 1.1114 | 0.1715 |
| Shorter pre-period | 0.0190 | 0.0054 |

### Heterogeneity
- Males: 0.0118 (SE: 0.0059)
- Females: 0.0215 (SE: 0.0071)
- Less than HS: 0.0113 (NS)
- High school: 0.0182 (p<0.01)
- Some college+: 0.0406 (p<0.01)

---

## Output Files Generated

### Required Deliverables
- [x] replication_report_32.tex
- [x] replication_report_32.pdf (24 pages)
- [x] run_log_32.md (this file)

### Additional Output Files
- analysis_script.py - Main analysis code
- create_figures.py - Figure generation code
- results_table.csv - Summary of main results
- event_study_results.csv - Year-by-year coefficients
- model_summaries.txt - Full regression output
- figure1_event_study.png/pdf
- figure2_trends.png/pdf
- figure3_specifications.png/pdf

---

## Key Decisions Documentation

### Why DiD over other methods?
DiD is appropriate because:
1. DACA created a discrete eligibility boundary
2. We observe both treatment and control groups before and after
3. Can test parallel trends assumption with event study

### Why exclude 2012?
ACS doesn't record interview month. DACA announced June 15, 2012; applications began August 15, 2012. Cannot distinguish pre- vs post-DACA observations within 2012.

### Why use non-DACA-eligible as control?
- Same country of origin and immigration status
- Facing similar economic conditions
- Main difference is DACA eligibility criteria
- Alternative controls (e.g., legal immigrants) face different labor market conditions

### Why age 18-64?
Standard working-age population. Excluding very young (not typically working) and elderly (near retirement).

### Why Model 3 as preferred?
- Year FE captures aggregate time trends (recession recovery, etc.)
- Demographics capture compositional differences
- State FE adds little explanatory power
- Balance of parsimony and control

---

## Session Information

Analysis completed successfully.

All required deliverables generated:
1. replication_report_32.tex - LaTeX source
2. replication_report_32.pdf - 24-page report
3. run_log_32.md - This documentation file
