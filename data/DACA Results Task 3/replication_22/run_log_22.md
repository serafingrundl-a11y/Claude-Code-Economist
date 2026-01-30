# Replication Run Log - Replication 22

## Session Start
Date: 2026-01-27

## Task Overview
Replicate analysis examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Design
- **Treatment group**: DACA-eligible individuals aged 26-30 at time of policy (June 15, 2012)
- **Control group**: Individuals aged 31-35 at time of policy (would have been eligible except for age)
- **Outcome**: Full-time employment (usually working 35+ hours/week)
- **Method**: Difference-in-differences comparing treated (26-30) to untreated (31-35) group, before (2008-2011) vs after (2013-2016) DACA implementation

## Key Variables (from instructions)
- `ELIGIBLE`: 1 = eligible (treated group, ages 26-30), 0 = comparison group (ages 31-35)
- `FT`: 1 = full-time work, 0 = not full-time work
- `AFTER`: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)

## Steps Performed

### Step 1: Read Instructions
- Read replication_instructions.docx
- Confirmed research question and methodology requirements
- Note: 2012 data excluded (cannot determine pre/post treatment status)
- Note: ACS is repeated cross-section, not panel data

### Step 2: Explore Data
- Data file: prepared_data_numeric_version.csv (17,382 observations, 105 variables)
- Key variables confirmed:
  - ELIGIBLE: 1 = treated (ages 26-30), 0 = control (ages 31-35) at June 15, 2012
  - AFTER: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)
  - FT: 1 = full-time employment (35+ hours/week), 0 = not full-time
  - PERWT: Person weight for population estimates
- 2012 data excluded as specified
- Sample composition:
  - Treated group (ELIGIBLE=1): 11,382 observations
  - Control group (ELIGIBLE=0): 6,000 observations
  - Pre-period: 9,527 observations
  - Post-period: 7,855 observations

### Step 3: Conduct Analysis

#### Analytical Approach
- Method: Difference-in-Differences (DiD)
- Treatment: DACA eligibility (ELIGIBLE=1, ages 26-30 at June 2012)
- Control: ELIGIBLE=0 (ages 31-35 at June 2012)
- Pre-period: 2008-2011 (AFTER=0)
- Post-period: 2013-2016 (AFTER=1)
- Outcome: Full-time employment (FT)

#### Key Decisions
1. Used person weights (PERWT) to weight observations for population representativeness
2. Used heteroskedasticity-robust standard errors (HC1)
3. Included demographic controls: sex, marital status, education level, number of children
4. Included year fixed effects to control for common time trends
5. Included state fixed effects to control for time-invariant state characteristics
6. Preferred specification: Model 7 (demographics + year FE + state FE)

#### Results Summary
- Simple DiD (unweighted): 0.0643 (SE: 0.0153)
- Simple DiD (weighted): 0.0748 (SE: 0.0181)
- Preferred model (Model 7): 0.0586 (SE: 0.0166)
- 95% CI: [0.0261, 0.0911]
- p-value: 0.0004 (statistically significant)

#### Interpretation
DACA eligibility is associated with a 5.9 percentage point increase in the probability of full-time employment among eligible Hispanic-Mexican, Mexican-born individuals.

#### Robustness Checks
- Results robust across multiple specifications (Models 1-8)
- Heterogeneity analysis by gender, education, marital status
- Event study shows post-treatment effects positive

### Step 4: Create Report
- Created LaTeX document: replication_report_22.tex
- Report includes:
  - Abstract
  - Table of contents
  - Introduction with DACA background
  - Data description and summary statistics
  - Methodology section
  - Results with multiple regression specifications
  - Robustness checks (parallel trends, heterogeneity analysis)
  - Discussion and policy implications
  - Conclusion
  - Appendices with full regression results, variable definitions, sensitivity analysis

### Step 5: Compile PDF
- Compiled LaTeX to PDF: replication_report_22.pdf
- PDF is 21 pages (meets ~20 page requirement)
- Three compilation passes for proper cross-references and table of contents

## Final Deliverables
All required output files are present in the replication folder:
1. `replication_report_22.tex` - LaTeX source file
2. `replication_report_22.pdf` - Compiled PDF report (21 pages)
3. `run_log_22.md` - This run log

## Additional Files Created
- `analysis_script.py` - Python analysis script
- `analysis_results.json` - JSON file with key results

## Key Results Summary
| Model | DiD Estimate | Robust SE | p-value | N |
|-------|-------------|-----------|---------|-------|
| Basic DiD (OLS) | 0.0643 | 0.0153 | <0.001 | 17,382 |
| Weighted DiD | 0.0748 | 0.0181 | <0.001 | 17,382 |
| + Demographics | 0.0620 | 0.0167 | <0.001 | 17,382 |
| + Year FE + State FE (PREFERRED) | 0.0586 | 0.0166 | <0.001 | 17,382 |

## Conclusion
The analysis finds that DACA eligibility increased full-time employment by approximately 5.9 percentage points among Hispanic-Mexican, Mexican-born individuals. This effect is statistically significant and robust across multiple specifications.
