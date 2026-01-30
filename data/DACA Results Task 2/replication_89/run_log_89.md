# DACA Replication Study - Run Log

## Study Information
- **Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals
- **Method**: Difference-in-Differences
- **Data**: American Community Survey (ACS) 2006-2016

---

## Session Log

### Step 1: Read Replication Instructions
- Read `replication_instructions.docx` to understand the research task
- Key requirements identified:
  - Treatment group: Ages 26-30 as of June 15, 2012 (born 1982-1986)
  - Control group: Ages 31-35 as of June 15, 2012 (born 1977-1981)
  - Outcome: Full-time employment (35+ hours/week)
  - Pre-period: 2006-2011
  - Post-period: 2013-2016 (excluding 2012)

### Step 2: Explore Data Structure
- Examined `data/acs_data_dict.txt` for variable definitions
- Key variables identified:
  - `HISPAN`: Hispanic origin (1 = Mexican)
  - `BPL`: Birthplace (200 = Mexico)
  - `CITIZEN`: Citizenship status (3 = Not a citizen)
  - `YRIMMIG`: Year of immigration
  - `BIRTHYR`: Birth year
  - `UHRSWORK`: Usual hours worked per week
  - `PERWT`: Person-level sampling weight

### Step 3: Create Analysis Script
- Created `analysis.py` with the following components:
  - Data loading with chunked processing (due to large file size)
  - Sample selection filters
  - Variable construction
  - Difference-in-differences regression
  - Robustness checks
  - Event study analysis

### Step 4: Run Analysis
- Executed `python analysis.py`
- Processing time: Several minutes (chunked processing of 33.8M rows)
- Final sample size: 44,725 observations
  - Treatment group: 26,591
  - Control group: 18,134

### Step 5: Key Results

#### Sample Construction
| Filter | Observations |
|--------|-------------|
| Hispanic-Mexican (HISPAN=1) | (filtered) |
| Born in Mexico (BPL=200) | (filtered) |
| Non-citizen (CITIZEN=3) | (filtered) |
| Arrived before age 16 | (filtered) |
| Immigrated by 2007 | (filtered) |
| Birth years 1977-1986 | (filtered) |
| Excluding 2012 | 44,725 |

#### Raw Difference-in-Differences
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 0.611 | 0.634 | +0.023 |
| Control (31-35) | 0.643 | 0.611 | -0.032 |
| **DiD** | | | **0.055** |

#### Main Regression Results
| Model | DiD Coefficient | Std Error | P-value | R-squared |
|-------|----------------|-----------|---------|-----------|
| Basic DiD | 0.0620 | 0.0116 | <0.001 | 0.002 |
| Demographics | 0.0657 | 0.0148 | <0.001 | 0.148 |
| Full Covariates | 0.0654 | 0.0148 | <0.001 | 0.152 |
| Year FE | 0.0187 | 0.0157 | 0.236 | 0.156 |
| State+Year FE | 0.0170 | 0.0157 | 0.278 | 0.160 |

#### Preferred Estimate (Model 5: State + Year Fixed Effects)
- **Effect Size**: 0.0170 (1.70 percentage points)
- **Standard Error**: 0.0157
- **95% CI**: [-0.0137, 0.0478]
- **P-value**: 0.278
- **Sample Size**: 44,725

#### Robustness Checks
| Specification | DiD Coefficient | SE |
|--------------|----------------|-----|
| Males only | 0.0674 | 0.0178 |
| Females only | 0.0528 | 0.0244 |
| Employment (any) | 0.0573 | 0.0141 |
| Narrow age bands (28-30 vs 31-33) | 0.0866 | 0.0185 |

### Step 6: Write Replication Report
- Created `replication_report_89.tex` (LaTeX document, ~19 pages)
- Sections include:
  - Introduction
  - Background (DACA program, mechanisms, prior literature)
  - Data (source, sample selection, variables)
  - Methodology (DiD design, specifications, event study)
  - Results (summary statistics, main results, robustness)
  - Discussion
  - Conclusion
  - Appendix

### Step 7: Compile PDF
- Compiled `replication_report_89.pdf` using pdflatex
- Final output: 19 pages

---

## Key Analytical Decisions

1. **Sample Definition**:
   - Used HISPAN=1 (Mexican Hispanic) rather than broader Hispanic definition
   - Restricted to BPL=200 (born in Mexico)
   - Used CITIZEN=3 (non-citizen) as proxy for undocumented status
   - Calculated age at immigration to verify arrival before age 16
   - Required immigration by 2007 for continuous residence requirement

2. **Age Groups**:
   - Treatment: Born 1982-1986 (ages 26-30 in June 2012)
   - Control: Born 1977-1981 (ages 31-35 in June 2012)
   - These groups differ only by DACA eligibility due to age cutoff

3. **Time Periods**:
   - Pre-treatment: 2006-2011
   - Post-treatment: 2013-2016
   - Excluded 2012 (DACA implemented mid-year)

4. **Outcome Variable**:
   - Full-time employment defined as UHRSWORK >= 35 hours/week
   - Binary indicator (1 = full-time, 0 = not full-time)

5. **Model Specifications**:
   - Progressive covariate inclusion
   - Year fixed effects to control for common shocks
   - State fixed effects for geographic heterogeneity
   - Heteroskedasticity-robust (HC1) standard errors
   - Person-level sampling weights (PERWT)

6. **Preferred Specification**:
   - Model 5 with state and year fixed effects
   - Reasoning: Year FE account for differential trends; state FE for geographic heterogeneity

---

## Output Files Created

1. `analysis.py` - Main analysis script
2. `results_summary.csv` - Regression results summary
3. `yearly_fulltime_means.csv` - Annual employment rates by group
4. `event_study_results.csv` - Event study coefficients
5. `summary_statistics.csv` - Sample characteristics
6. `analysis_output.txt` - Detailed text output
7. `replication_report_89.tex` - LaTeX source
8. `replication_report_89.pdf` - Final report (19 pages)
9. `run_log_89.md` - This run log

---

## Interpretation Notes

The preferred estimate of 1.70 percentage points is not statistically significant at conventional levels (p = 0.278). However, models without year fixed effects yield larger, statistically significant estimates (~6.5 pp). This sensitivity suggests:

1. The parallel trends assumption may be problematic
2. Age-specific secular trends may confound the basic DiD estimate
3. The true effect is likely positive but modest in magnitude

The event study shows:
- Pre-treatment coefficients generally close to zero (supporting parallel trends)
- Post-treatment coefficients positive but mostly insignificant
- Largest effect in 2016 (0.052, borderline significant)

---

## Session End
All deliverables completed successfully.
