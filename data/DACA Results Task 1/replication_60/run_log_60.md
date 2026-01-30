# Run Log - Replication 60

## DACA Impact on Full-Time Employment: Independent Replication

### Session Start
- Date: 2025-01-25
- Task: Estimate the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the US

---

## Data Understanding

### Data Sources
1. **data.csv** - Main ACS data file (2006-2016, 1-year samples, 33.8 million observations)
2. **acs_data_dict.txt** - Data dictionary for IPUMS ACS variables
3. **state_demo_policy.csv** - Optional supplemental state-level data (not used)

### Key Variables Identified

#### Identifying DACA Eligibility
- **YEAR**: Survey year (2006-2016)
- **BIRTHYR**: Year of birth
- **BIRTHQTR**: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- **HISPAN/HISPAND**: Hispanic origin (1=Mexican, detailed codes 100-107 for Mexican)
- **BPL/BPLD**: Birthplace (200=Mexico, detailed 20000=Mexico)
- **CITIZEN**: Citizenship status (3=Not a citizen)
- **YRIMMIG**: Year of immigration

#### Outcome Variable
- **UHRSWORK**: Usual hours worked per week (full-time = 35+ hours)

#### Control Variables
- **AGE**: Age
- **SEX**: Sex (1=Male, 2=Female)
- **EDUC/EDUCD**: Educational attainment
- **MARST**: Marital status
- **STATEFIP**: State FIPS code
- **PERWT**: Person weight

---

## DACA Eligibility Criteria (from instructions)

To be DACA eligible, an individual must:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 without lawful status

### Operationalization
- **Age at arrival < 16**: YRIMMIG - BIRTHYR < 16
- **Age on June 15, 2012 < 31**: Born after June 15, 1981 (BIRTHYR >= 1982, or BIRTHYR=1981 and BIRTHQTR >= 3)
- **In US since June 2007**: YRIMMIG <= 2007
- **At least 15 by August 2012**: BIRTHYR <= 1997
- **Not a citizen**: CITIZEN == 3

### Treatment Definition
- Post-DACA period: 2013-2016 (DACA implemented mid-2012, effects visible from 2013)
- Pre-DACA period: 2006-2011 (exclude 2012 due to mid-year implementation)

---

## Analytical Approach

### Difference-in-Differences Design
- **Treatment group**: DACA-eligible non-citizens (meet all criteria above)
- **Control group**: DACA-ineligible non-citizens (similar demographics but fail at least one criterion)
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016

### Sample Restrictions
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Non-citizen (CITIZEN == 3) - assuming undocumented status per instructions
4. Working-age population (16-45 years old to capture relevant age groups)
5. Exclude 2012 (DACA implemented mid-year)
6. Valid year of immigration (YRIMMIG > 0)

---

## Sample Construction Results

| Restriction | Observations | Percent |
|-------------|--------------|---------|
| Starting sample (2006-2016 ACS) | 33,851,424 | 100.0% |
| Hispanic-Mexican ethnicity | 2,945,521 | 8.7% |
| Born in Mexico | 991,261 | 2.9% |
| Non-citizen | 701,347 | 2.1% |
| Working age (16-45) | 470,312 | 1.4% |
| Exclude 2012 | 427,762 | 1.3% |
| **Final analytical sample** | **427,762** | **1.3%** |

### Treatment/Control Group Distribution
- DACA Eligible: 81,097 (19.0%)
- Not Eligible: 346,665 (81.0%)

---

## Key Results

### Main Finding (Preferred Specification - Model 4)
- **DiD Coefficient**: 0.0409 (4.09 percentage points)
- **Standard Error**: 0.0035
- **95% Confidence Interval**: [0.0339, 0.0479]
- **P-value**: < 0.0001
- **Sample Size**: 427,762

### Interpretation
DACA eligibility is associated with a **4.09 percentage point INCREASE** in the probability of full-time employment. This effect is **statistically significant** at the 5% level.

### Raw Difference-in-Differences
| Group | Pre-DACA (2006-2011) | Post-DACA (2013-2016) | Difference |
|-------|---------------------|----------------------|------------|
| Not Eligible | 0.617 | 0.578 | -0.038 |
| DACA Eligible | 0.431 | 0.528 | +0.097 |
| **DiD** | | | **0.135** |

### Regression Results Summary

| Model | DiD Coefficient | Std Error | Controls |
|-------|----------------|-----------|----------|
| (1) Basic | 0.1352 | (0.0039) | None |
| (2) Demographics | 0.0490 | (0.0036) | Age, Sex, Marriage, Education |
| (3) Year FE | 0.0410 | (0.0036) | + Year FE |
| (4) Year+State FE | 0.0409 | (0.0035) | + State FE |
| (5) Weighted | 0.0383 | (0.0035) | + Person Weights |

### Event Study Results (Reference: 2011)

| Year | Coefficient | Std Error | Significance |
|------|-------------|-----------|--------------|
| 2006 | -0.0088 | (0.0077) | NS |
| 2007 | -0.0110 | (0.0075) | NS |
| 2008 | 0.0013 | (0.0076) | NS |
| 2009 | 0.0048 | (0.0074) | NS |
| 2010 | 0.0060 | (0.0073) | NS |
| 2011 | [Reference] | | |
| 2013 | 0.0060 | (0.0072) | NS |
| 2014 | 0.0305 | (0.0073) | *** |
| 2015 | 0.0611 | (0.0074) | *** |
| 2016 | 0.0726 | (0.0076) | *** |

**Interpretation**: Pre-treatment coefficients are small and insignificant, supporting parallel trends assumption. Effect grows over time in post-period.

### Robustness Checks

| Specification | Coefficient | Std Error | N |
|---------------|-------------|-----------|-----|
| Main result (Model 4) | 0.0409 | (0.0035) | 427,762 |
| Age 18-35 only | 0.0118 | (0.0041) | 253,373 |
| Tighter control group | 0.0419 | (0.0037) | 347,901 |
| Any employment outcome | 0.0641 | (0.0033) | 427,762 |
| Clustered SE by state | 0.0410 | (0.0059) | 427,762 |

### Heterogeneity Analysis

| Subgroup | DiD Coefficient | Std Error | N |
|----------|----------------|-----------|-----|
| **By Gender** | | | |
| Male | 0.0264 | (0.0044) | 234,520 |
| Female | 0.0524 | (0.0057) | 193,242 |
| **By Education** | | | |
| Less than HS | 0.0476 | (0.0055) | 228,612 |
| HS or more | 0.0253 | (0.0048) | 199,150 |

---

## Commands Executed

### Step 1: Extract Text from Instructions
```bash
python -c "from docx import Document; doc = Document('replication_instructions.docx'); [print(p.text) for p in doc.paragraphs]"
```

### Step 2: Explore Data Files
```bash
ls -la data/
```

### Step 3: Run Main Analysis
```bash
python analysis.py
```
- Loaded 33.8 million observations
- Constructed analytical sample of 427,762
- Estimated 5 DiD specifications
- Ran robustness checks and event study
- Exported results to CSV files

### Step 4: Create LaTeX Report
```bash
pdflatex -interaction=nonstopmode replication_report_60.tex
```
- Generated 24-page report
- Includes tables, results, and methodology

---

## Files Created

1. **analysis.py** - Main Python analysis script
2. **results_summary.csv** - Key results in CSV format
3. **table1_summary_stats.csv** - Summary statistics table
4. **table2_did_results.csv** - DiD regression results
5. **table3_robustness.csv** - Robustness check results
6. **table4_event_study.csv** - Event study coefficients
7. **replication_report_60.tex** - LaTeX source for report
8. **replication_report_60.pdf** - Final PDF report (24 pages)
9. **run_log_60.md** - This log file

---

## Key Decisions Made

1. **Age range 16-45**: Chose to focus on working-age population while capturing both young DACA-eligible individuals and slightly older controls
2. **Exclude 2012**: DACA implemented mid-year (June 15, 2012), cannot distinguish pre/post within that year
3. **Non-citizen = undocumented**: Per instructions, assuming non-citizens without naturalization have not received papers
4. **Control group**: All non-eligible non-citizens rather than matched controls, to maximize power
5. **Preferred specification**: Model 4 with year and state fixed effects, capturing time and location-specific factors
6. **Full-time threshold**: 35+ hours per week, consistent with BLS definition

---

## Session End
- All deliverables completed
- PDF report generated (24 pages)
- Analysis code and outputs saved
