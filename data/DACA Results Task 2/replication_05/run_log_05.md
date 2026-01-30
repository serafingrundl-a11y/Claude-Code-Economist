# DACA Replication Study - Run Log

## Study Information
- **Research Question:** Effect of DACA eligibility on full-time employment among Hispanic-Mexican immigrants born in Mexico
- **Date:** January 2026
- **Analysis Type:** Difference-in-Differences

---

## Data Exploration

### Initial Data Review
- Examined `replication_instructions.docx` for research specifications
- Reviewed `acs_data_dict.txt` for variable definitions
- Checked `state_demo_policy.csv` for supplemental state-level data (not used in final analysis)
- Inspected `data.csv` structure (33,851,424 observations)

### Key Variables Identified
| Variable | Description | Usage |
|----------|-------------|-------|
| YEAR | Survey year | Time period identification |
| PERWT | Person weight | Survey weighting |
| HISPAN | Hispanic origin | Sample restriction (=1 for Mexican) |
| BPL | Birthplace | Sample restriction (=200 for Mexico) |
| CITIZEN | Citizenship status | Sample restriction (=3 for non-citizen) |
| BIRTHYR | Birth year | Age calculation |
| BIRTHQTR | Birth quarter | Age adjustment |
| YRIMMIG | Year of immigration | DACA eligibility |
| UHRSWORK | Usual hours worked | Outcome variable |

---

## Sample Construction Decisions

### Decision 1: Identifying DACA-Eligible Population
- **Filter:** Hispanic-Mexican ethnicity (HISPAN = 1)
- **Rationale:** Research question specifically targets this population
- **Impact:** 33.9M → 2.95M observations

### Decision 2: Mexico-Born Restriction
- **Filter:** BPL = 200 (Mexico)
- **Rationale:** Research question focuses on Mexican-born individuals
- **Impact:** 2.95M → 991K observations

### Decision 3: Non-Citizen Status as Proxy for Undocumented
- **Filter:** CITIZEN = 3 (Not a citizen)
- **Rationale:** ACS does not directly identify documentation status; non-citizen status serves as proxy per research instructions
- **Caveat:** May include some legal permanent residents
- **Impact:** 991K → 701K observations

### Decision 4: Year Restrictions
- **Filter:** Years 2006-2011 (pre), 2013-2016 (post)
- **Excluded:** 2012 (ambiguous transition year - DACA implemented June 15)
- **Rationale:** Cannot distinguish pre/post observations in 2012
- **Impact:** 701K → 637K observations

### Decision 5: Age Group Definition
- **Treatment:** Ages 26-30 as of June 15, 2012
- **Control:** Ages 31-35 as of June 15, 2012
- **Age Calculation:**
  ```
  age_june2012 = 2012 - BIRTHYR
  if BIRTHQTR in [3, 4]: age_june2012 -= 1  # Hadn't had birthday by June 15
  ```
- **Rationale:** Those 31+ were ineligible due to age cutoff
- **Impact:** 637K → 165K observations

### Decision 6: DACA Eligibility Criteria
- **Arrived before age 16:** YRIMMIG - BIRTHYR < 16
- **Continuous residence since 2007:** YRIMMIG <= 2007
- **Rationale:** Core DACA eligibility requirements
- **Impact:** 165K → 43,238 final observations

---

## Outcome Variable Definition

### Full-Time Employment
```python
fulltime = 1 if UHRSWORK >= 35 else 0
```
- **Threshold:** 35 hours per week (standard full-time definition)
- **Source:** Research instructions specify "usually working 35 hours per week or more"

---

## Empirical Strategy

### Primary Specification
**Difference-in-Differences Model:**
```
Y_ist = α + β₁(Treated) + β₂(Post) + β₃(Treated × Post) + ε_ist
```

### Extended Specifications
1. **With covariates:** Female, married, education (HS+), age
2. **Year fixed effects:** Replace Post with year dummies
3. **State fixed effects:** Add state dummies
4. **Clustered standard errors:** Cluster at state level

### Event Study Specification
```
Y_ist = α + β₁(Treated) + Σ_k δ_k(Treated × Year_k) + λ_t + X'γ + ε_ist
```
- Reference year: 2011 (year before DACA)

---

## Commands Executed

### Data Analysis
```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_05"
python analysis.py
```

### Figure Generation
```bash
mkdir -p figures
python create_figures.py
```

### Report Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_05.tex  # First pass
pdflatex -interaction=nonstopmode replication_report_05.tex  # Second pass
pdflatex -interaction=nonstopmode replication_report_05.tex  # Third pass
```

---

## Key Results

### Sample Composition
| Group | N |
|-------|---|
| Treatment (ages 26-30) | 25,470 |
| Control (ages 31-35) | 17,768 |
| Pre-period (2006-2011) | 28,377 |
| Post-period (2013-2016) | 14,861 |
| **Total** | **43,238** |

### Full-Time Employment Rates
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment | 0.631 | 0.660 | +0.029 |
| Control | 0.673 | 0.643 | -0.030 |
| **Simple DiD** | | | **0.059** |

### Preferred Estimate (State-Clustered SE)
| Statistic | Value |
|-----------|-------|
| DiD Coefficient | 0.0458 |
| Standard Error | 0.0097 |
| 95% CI | [0.0268, 0.0648] |
| p-value | < 0.001 |

### Interpretation
DACA eligibility is associated with a **4.6 percentage point increase** in full-time employment for the treatment group relative to the control group. This effect is statistically significant at the 1% level.

---

## Robustness Checks

| Specification | Estimate | SE |
|---------------|----------|-----|
| Basic DiD | 0.052 | 0.010 |
| Weighted DiD | 0.059 | 0.010 |
| With Covariates | 0.047 | 0.009 |
| Year FE | 0.047 | 0.009 |
| State + Year FE | 0.046 | 0.009 |
| Clustered SE (Preferred) | 0.046 | 0.010 |
| Labor Force Only | 0.028 | 0.011 |
| Narrow Bandwidth (27-29 vs 32-34) | 0.040 | 0.012 |

---

## Event Study Results (Pre-Trends Check)

| Year | Coefficient | SE | Significant |
|------|-------------|-----|-------------|
| 2006 | 0.006 | 0.023 | No |
| 2007 | -0.032 | 0.022 | No |
| 2008 | 0.008 | 0.023 | No |
| 2009 | -0.009 | 0.024 | No |
| 2010 | -0.014 | 0.023 | No |
| 2011 | 0 (ref) | - | - |
| 2013 | 0.035 | 0.024 | No |
| 2014 | 0.037 | 0.025 | No |
| 2015 | 0.020 | 0.025 | No |
| 2016 | 0.067 | 0.025 | Yes |

**Conclusion:** No significant pre-trends detected; parallel trends assumption supported.

---

## Heterogeneity Analysis

### By Sex
| Subgroup | Estimate | SE |
|----------|----------|-----|
| Male | 0.045 | 0.013 |
| Female | 0.045 | 0.019 |

### By Education
| Subgroup | Estimate | SE |
|----------|----------|-----|
| Less than HS | 0.034 | 0.018 |
| HS or more | 0.077 | 0.016 |

---

## Output Files

### Analysis Files
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script

### Results
- `results/regression_results.csv` - Regression coefficients
- `results/descriptive_stats.csv` - Summary statistics
- `results/event_study.csv` - Event study coefficients

### Figures
- `figures/event_study.pdf` - Event study plot
- `figures/trends_by_group.pdf` - Employment trends
- `figures/did_visualization.pdf` - DiD schematic
- `figures/robustness_checks.pdf` - Forest plot of estimates

### Report
- `replication_report_05.tex` - LaTeX source
- `replication_report_05.pdf` - Final report (22 pages)

---

## Notes and Limitations

1. **Proxy for undocumented status:** Using non-citizen status as proxy may include some legal residents
2. **Age differences:** Control group is 5 years older than treatment group
3. **Intent-to-treat:** Estimates eligibility effect, not actual DACA receipt
4. **2012 exclusion:** Loses observations from transition year
5. **Geographic concentration:** Sample concentrated in states with large Mexican immigrant populations

---

## Session Information

- **Python version:** 3.x
- **Key packages:** pandas, numpy, statsmodels, matplotlib
- **LaTeX distribution:** MiKTeX
- **Operating system:** Windows
