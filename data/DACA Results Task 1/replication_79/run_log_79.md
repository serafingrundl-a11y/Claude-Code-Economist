# Run Log - Replication 79: DACA Impact on Full-Time Employment

## Project Overview
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

**Analysis Period:** DACA implemented June 15, 2012; examining effects on full-time employment 2013-2016.

**Execution Date:** January 25, 2026

---

## Key Decisions Log

### 1. Sample Definition
**Decision:** Restrict sample to Hispanic-Mexican ethnicity (HISPAN=1) AND born in Mexico (BPL=200).
**Rationale:** The research question specifically focuses on "ethnically Hispanic-Mexican Mexican-born people." This is the population most likely to be affected by DACA given the structure of undocumented immigration to the US.

### 2. DACA Eligibility Criteria Implementation
Per the instructions, DACA eligibility requires:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007 (at least 5 years by June 2012)
4. Were present in the US on June 15, 2012 and did not have lawful status

**Implementation:**
- **Age at arrival < 16:** Calculate age at arrival = YRIMMIG - BIRTHYR. Eligible if arrival_age < 16.
- **Age on June 15, 2012 < 31:** BIRTHYR > 1981 (born after June 15, 1981). Using BIRTHYR >= 1982 as conservative criterion.
- **Continuous presence since June 15, 2007:** YRIMMIG <= 2007
- **Not a citizen:** CITIZEN == 3 (Not a citizen)

**Note:** Cannot distinguish documented vs undocumented non-citizens. Per instructions, assume non-citizens without naturalization are undocumented.

### 3. Treatment and Control Group Definition
**Treatment Group (DACA-eligible):**
- Hispanic-Mexican, born in Mexico
- Non-citizen (CITIZEN=3)
- Arrived before age 16
- Born after 1981 (age < 31 on June 15, 2012)
- Arrived by 2007 (present for at least 5 years)

**Control Group:**
- Hispanic-Mexican, born in Mexico
- Non-citizen (CITIZEN=3)
- NOT meeting one or more DACA criteria (arrived age 16+, or born 1981 or earlier, etc.)

### 4. Outcome Variable
**Full-time employment:** UHRSWORK >= 35 (usually working 35+ hours per week)
- Binary indicator: 1 if full-time employed, 0 otherwise
- Following the research question definition

### 5. Estimation Strategy
**Difference-in-Differences (DiD) Design:**
- Pre-treatment period: 2006-2011 (DACA announced June 2012)
- Post-treatment period: 2013-2016 (per instructions)
- Excluding 2012 due to treatment timing uncertainty within the year

**Model Specification:**
Y_it = alpha + beta1 * Eligible_i + beta2 * Post_t + beta3 * (Eligible_i * Post_t) + X_it * gamma + epsilon_it

Where:
- Y_it = full-time employment indicator
- Eligible_i = DACA eligibility indicator
- Post_t = indicator for years 2013-2016
- beta3 = DiD estimate (causal effect of DACA eligibility)
- X_it = control variables

### 6. Control Variables
- Age and age squared (to capture life-cycle employment patterns)
- Sex (SEX)
- Marital status (MARST)
- Education level (EDUC)
- State fixed effects (STATEFIP)
- Year fixed effects (YEAR)

### 7. Working-Age Sample Restriction
**Decision:** Restrict to ages 16-64 to focus on working-age population.
**Rationale:** Standard labor economics practice; individuals under 16 are largely not in the labor force, and those 65+ often retire.

### 8. Weighting
**Decision:** Use person weights (PERWT) in all analyses.
**Rationale:** ACS provides sampling weights to make estimates representative of the population.

### 9. Year 2012 Exclusion
**Decision:** Exclude 2012 from the analysis sample entirely.
**Rationale:** DACA was announced June 15, 2012, and applications began August 15, 2012. The ACS does not record the month of interview, so 2012 observations cannot be classified as pre- or post-treatment.

---

## Analysis Steps

### Step 1: Data Loading and Initial Exploration
- Loaded data.csv containing ACS data from 2006-2016
- Total observations: 33,851,424
- Verified variable availability and coding

### Step 2: Sample Construction
Sequential filtering applied:
1. Hispanic-Mexican filter (HISPAN=1): 2,945,521 observations
2. Mexican-born filter (BPL=200): 991,261 observations
3. Non-citizen filter (CITIZEN=3): 701,347 observations
4. Working-age filter (16-64): 618,640 observations
5. Exclude 2012: 561,470 observations (final sample)

### Step 3: Variable Construction
- Created DACA eligibility indicator (80,300 eligible, 481,170 non-eligible)
- Created full-time employment outcome (57.4% full-time employed)
- Created post-DACA period indicator (2013-2016)
- Created interaction term for DiD estimation

### Step 4: Descriptive Statistics
- Generated summary statistics by treatment status and time period
- Pre-treatment FT employment: 44.7% (eligible), 62.7% (control)
- Post-treatment FT employment: 52.0% (eligible), 60.1% (control)

### Step 5: Main DiD Estimation
Five models estimated with progressively more controls:
1. Basic DiD: 0.0994 (SE: 0.0047)
2. + Demographics: 0.0442 (SE: 0.0043)
3. + Education: 0.0410 (SE: 0.0043)
4. + Year FE: 0.0335 (SE: 0.0043)
5. + State FE (PREFERRED): 0.0329 (SE: 0.0043)

### Step 6: Event Study Specification
- Estimated year-specific effects relative to 2011 (reference year)
- Pre-treatment coefficients: small and statistically insignificant (supports parallel trends)
- Post-treatment coefficients: positive and growing (2014: 0.028*, 2015: 0.045***, 2016: 0.045***)

### Step 7: Robustness Checks
- Males only: 0.0301 (SE: 0.0056)
- Females only: 0.0268 (SE: 0.0063)
- Ages 18-55: 0.0255 (SE: 0.0047)
- Any employment outcome: 0.0423 (SE: 0.0042)

---

## Commands Executed

```bash
# Python analysis script execution
cd "C:\Users\seraf\DACA Results Task 1\replication_79"
python analysis_79.py

# LaTeX compilation (three passes for references)
pdflatex -interaction=nonstopmode replication_report_79.tex
pdflatex -interaction=nonstopmode replication_report_79.tex
pdflatex -interaction=nonstopmode replication_report_79.tex
```

---

## Results Summary

### PREFERRED ESTIMATE (Model 5: Full specification with state and year FE)

| Statistic | Value |
|-----------|-------|
| DiD Estimate | 0.0329 |
| Robust Standard Error | 0.0043 |
| t-statistic | 7.73 |
| p-value | < 0.0001 |
| 95% Confidence Interval | [0.0246, 0.0413] |
| Sample Size | 561,470 |
| R-squared | 0.230 |

### Interpretation
DACA eligibility increased the probability of full-time employment by 3.29 percentage points among Mexican-born Hispanic non-citizens. This represents a 7.4% increase relative to the pre-treatment mean for eligible individuals (44.7%). The effect is statistically significant at the 1% level.

---

## IPUMS Variables Used

| Variable | Description | Values Used |
|----------|-------------|-------------|
| YEAR | Survey year | 2006-2011, 2013-2016 |
| HISPAN | Hispanic origin | 1 (Mexican) |
| BPL | Birthplace | 200 (Mexico) |
| CITIZEN | Citizenship status | 3 (Not a citizen) |
| BIRTHYR | Birth year | >=1982 for eligibility |
| YRIMMIG | Year of immigration | <=2007 for eligibility |
| UHRSWORK | Usual hours worked | >=35 for full-time |
| AGE | Age | 16-64 |
| SEX | Sex | 1=Male, 2=Female |
| MARST | Marital status | 1,2=Married |
| EDUC | Education | Categories 1-11 |
| STATEFIP | State FIPS code | All states |
| PERWT | Person weight | Used for all estimates |

---

## Files Created

### Required Deliverables
- `run_log_79.md` - This log file
- `replication_report_79.tex` - LaTeX source for report
- `replication_report_79.pdf` - Compiled 21-page report

### Analysis Files
- `analysis_79.py` - Main Python analysis script

### Output Directories
- `figures/` - Contains visualization outputs:
  - `fulltime_emp_trends.png` / `.pdf` - Time trends plot
  - `diff_trends.png` / `.pdf` - Difference plot
  - `event_study.png` / `.pdf` - Event study coefficients

- `tables/` - Contains tabular outputs:
  - `summary_statistics.csv` - Descriptive statistics
  - `yearly_trends.csv` - Year-by-year employment rates
  - `regression_results.csv` - Main DiD results
  - `robustness_results.csv` - Robustness check results
  - `event_study_coefficients.csv` - Event study coefficients
  - `key_results.txt` - Preferred estimate summary

---

## Software Environment

- Python 3.x
- pandas, numpy, statsmodels, matplotlib
- MiKTeX (pdflatex) for LaTeX compilation

---

## End of Log
