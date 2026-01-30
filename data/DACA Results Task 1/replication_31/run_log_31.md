# DACA Replication Study - Run Log

## Date: 2026-01-25

## Study Objective
Estimate the causal impact of DACA eligibility on full-time employment (35+ hours/week) among Hispanic-Mexican, Mexican-born individuals in the United States, examining effects in 2013-2016.

---

## Session 1: Initial Setup and Data Exploration

### Step 1: Read Replication Instructions
- Read replication_instructions.docx
- Key research question: Effect of DACA eligibility (treatment) on probability of full-time employment (outcome)
- DACA implemented June 15, 2012
- Target years for outcome analysis: 2013-2016

### Step 2: DACA Eligibility Criteria (from instructions)
1. Arrived unlawfully in US before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status (not citizen/legal resident)

### Step 3: Data Sources Identified
- Main data: data/data.csv (ACS data from IPUMS, 2006-2016)
- Data dictionary: data/acs_data_dict.txt
- Optional: data/state_demo_policy.csv

### Step 4: Key Variables Identified from Data Dictionary
- YEAR: Census year (2006-2016)
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- HISPAN/HISPAND: Hispanic origin (1=Mexican for general)
- BPL/BPLD: Birthplace (200=Mexico for general)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status (1=Employed)
- AGE: Age
- PERWT: Person weight for survey

### Step 5: Sample Restrictions Planned
1. Hispanic-Mexican ethnicity (HISPAN==1)
2. Born in Mexico (BPL==200)
3. Non-citizen (CITIZEN==3) - approximation for undocumented status
4. Age restrictions based on DACA eligibility

### Step 6: Identification Strategy
- Difference-in-differences approach
- Treatment group: DACA-eligible individuals
- Control group: Similar individuals not DACA-eligible (age-based cutoffs)
- Pre-period: 2006-2011 (before DACA)
- Post-period: 2013-2016 (after DACA)
- 2012 excluded due to mid-year implementation

---

## Commands and Analysis Log

### Command 1: Data Exploration
```bash
head -5 data/data.csv
```
Output: Confirmed data structure with 54 columns including all key variables

### Command 2: Check Data Size
```python
import os
file_size = os.path.getsize('data/data.csv') / (1024**3)
# Result: 5.83 GB, 33,851,424 rows
```

### Command 3: Run Main Analysis
```bash
python analysis.py
```
Key output:
- Total rows processed: 33,851,424
- Rows matching sample criteria (Hispanic-Mexican, Mexico-born, non-citizen): 701,347
- Working-age sample (18-64): 547,614
- DACA eligible: 71,347 (13.0% of working-age sample)

### Command 4: Create Figures
```bash
python create_figures.py
```
Created: figure1_event_study.pdf, figure2_trends.pdf, figure3_did.pdf, figure4_age_dist.pdf

---

## Key Decisions and Justifications

### Decision 1: Sample Definition
**Choice**: Hispanic-Mexican ethnicity (HISPAN=1) AND born in Mexico (BPL=200) AND non-citizen (CITIZEN=3)
**Justification**: The instructions specifically target "ethnically Hispanic-Mexican Mexican-born people." Using non-citizen status as proxy for undocumented status follows the instruction that "anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes."

### Decision 2: DACA Eligibility Criteria
**Choice**: Eligible if (1) arrived before age 16, (2) under 31 as of June 2012, (3) in US since at least 2007
**Justification**: These operationalize the DACA requirements using available ACS variables. Cannot directly observe unlawful status or presence on June 15, 2012, so using non-citizen status and immigration year as proxies.

### Decision 3: Exclusion of 2012
**Choice**: Exclude 2012 from analysis
**Justification**: DACA was implemented mid-year (June 15, 2012) and the ACS does not provide month of interview, making it impossible to determine pre/post status for 2012 observations.

### Decision 4: Working-Age Restriction
**Choice**: Ages 18-64
**Justification**: Standard labor economics definition of working-age population. Below 18 would be in school; above 64 approaching retirement.

### Decision 5: Full-Time Definition
**Choice**: UHRSWORK >= 35 hours per week
**Justification**: Standard Bureau of Labor Statistics definition of full-time employment.

### Decision 6: Identification Strategy
**Choice**: Difference-in-differences comparing DACA-eligible vs non-eligible individuals, before vs after 2012
**Justification**: Classic causal inference design for policy evaluation. Pre-trends test (event study) shows no differential trends before 2012, supporting parallel trends assumption.

### Decision 7: Preferred Specification
**Choice**: Model 4 with year and state fixed effects plus demographic controls
**Justification**: Most conservative specification controlling for common year trends (macroeconomic conditions) and state-level differences (local labor markets, policy environments).

---

## Results Summary

### Preferred Estimate (Model 4)
- **Effect size**: 0.0173 (1.73 percentage points)
- **Standard error**: 0.0045
- **95% CI**: [0.0085, 0.0262]
- **P-value**: 0.0001
- **Sample size**: 547,614

### Interpretation
DACA eligibility is associated with a 1.73 percentage point increase in the probability of full-time employment (35+ hours/week), statistically significant at the 1% level.

### Robustness
- Effect robust across specifications (1.7-2.7 pp depending on controls)
- Similar effects found for any employment outcome (2.7 pp)
- Effects positive for both males (1.1 pp) and females (1.6 pp)
- Event study shows no pre-trends, with effects emerging 2014-2016

---

## Files Generated
1. analysis.py - Main analysis script
2. create_figures.py - Figure generation script
3. main_results.csv - Regression results table
4. event_study_results.csv - Year-by-year coefficients
5. model_output.txt - Detailed regression output
6. figure1_event_study.pdf - Event study plot
7. figure2_trends.pdf - Employment trends by eligibility
8. figure3_did.pdf - DiD visualization
9. figure4_age_dist.pdf - Age distribution by eligibility
10. replication_report_31.tex - LaTeX report
11. replication_report_31.pdf - Final PDF report

