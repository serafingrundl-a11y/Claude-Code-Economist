# Replication Run Log - ID 65

## Project: DACA Impact on Full-Time Employment

### Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (usually working 35 hours per week or more)?

### Identification Strategy
- **Treatment Group**: Ages 26-30 at time of DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at time of DACA implementation
- **Design**: Difference-in-differences comparing treated vs untreated groups, before vs after DACA
- **Post-treatment Period**: 2013-2016

---

## Session Log

### Step 1: Data Exploration
- **Action**: Read replication instructions from replication_instructions.docx
- **Data Files Found**:
  - data.csv: Main ACS data file (33,851,425 observations including header)
  - acs_data_dict.txt: Data dictionary for IPUMS ACS variables
  - state_demo_policy.csv: Optional state-level data (not used)

### Key Variables Identified:
1. **YEAR**: Census/ACS year (2006-2016)
2. **BIRTHYR**: Birth year
3. **BIRTHQTR**: Quarter of birth
4. **HISPAN/HISPAND**: Hispanic origin (1=Mexican)
5. **BPL/BPLD**: Birthplace (200=Mexico)
6. **CITIZEN**: Citizenship status (3=Not a citizen)
7. **YRIMMIG**: Year of immigration
8. **UHRSWORK**: Usual hours worked per week (35+ = full-time)
9. **EMPSTAT**: Employment status
10. **PERWT**: Person weight for survey weighting
11. **AGE**: Age at time of survey

---

### Step 2: Sample Definition and DACA Eligibility Criteria

**DACA Eligibility Requirements (from instructions):**
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

**Operationalization:**
- Hispanic-Mexican: HISPAN == 1 (Mexican)
- Born in Mexico: BPL == 200
- Not a citizen: CITIZEN == 3
- Arrived before age 16: (YRIMMIG - BIRTHYR) < 16
- Lived in US since 2007: YRIMMIG <= 2007
- Age 26-30 on June 15, 2012 (Treatment): Born 1982-1986
- Age 31-35 on June 15, 2012 (Control): Born 1977-1981

**Key Decision**: Since DACA was implemented June 15, 2012, and had a strict cutoff of not yet turning 31:
- Treatment (ages 26-30 on June 15, 2012): Birth years 1982-1986
- Control (ages 31-35 on June 15, 2012): Birth years 1977-1981

---

### Step 3: Sample Construction (executed in analysis.py)

**Commands and Results:**
```python
# Load data in chunks and filter
for chunk in pd.read_csv(DATA_PATH, usecols=cols_needed, chunksize=1000000):
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
```

**Sample Flow:**
| Step | Observations | Dropped |
|------|-------------|---------|
| Full ACS data (2006-2016) | 33,851,424 | --- |
| Hispanic-Mexican, Born in Mexico | 991,261 | 32,860,163 |
| Non-citizens (CITIZEN == 3) | 701,347 | 289,914 |
| Arrived before age 16 | 205,327 | 496,020 |
| In US since 2007 (YRIMMIG <= 2007) | 195,023 | 10,304 |
| Treatment or Control age group | 49,019 | 145,004 |
| Exclude 2012 | 44,725 | 4,294 |

---

### Step 4: Variable Construction

**Outcome Variable:**
```python
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)
```

**Treatment Indicator:**
```python
df['treatment'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)
df['control'] = ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)).astype(int)
df['treated'] = df['treatment']
```

**Time Period:**
```python
df = df[df['YEAR'] != 2012]  # Exclude implementation year
df['post'] = (df['YEAR'] >= 2013).astype(int)
```

**Interaction Term:**
```python
df['treated_post'] = df['treated'] * df['post']
```

---

### Step 5: Analysis Execution

**Main Analysis Script:** `analysis.py`

**Models Estimated:**
1. Basic DiD (unweighted OLS)
2. Weighted DiD (using PERWT)
3. Weighted DiD with Year Fixed Effects
4. Weighted DiD with Demographic Controls
5. Weighted DiD with State Fixed Effects
6. Preferred: Weighted DiD with Controls and Robust (HC1) Standard Errors

**Command:**
```bash
python analysis.py
```

---

### Step 6: Results Summary

**Main Results (Preferred Estimate - Model 6):**
| Metric | Value |
|--------|-------|
| DiD Estimate | 0.0590 |
| Standard Error (robust) | 0.0110 |
| 95% CI | [0.037, 0.080] |
| P-value | < 0.001 |
| Sample Size | 44,725 |

**Interpretation:** DACA eligibility increased the probability of full-time employment by 5.9 percentage points.

**Heterogeneity Results:**
| Subgroup | Estimate | SE |
|----------|----------|-----|
| Male | 0.081 | 0.014 |
| Female | 0.032 | 0.018 |
| Less than HS | 0.056 | 0.017 |
| HS or More | 0.088 | 0.017 |
| Married | 0.066 | 0.017 |
| Not Married | 0.094 | 0.017 |

---

### Step 7: Figure and Table Generation

**Script:** `figures_and_tables.py`

**Command:**
```bash
python figures_and_tables.py
```

**Outputs Generated:**
- figure1_parallel_trends.png/pdf
- figure2_event_study.png/pdf
- figure3_did_illustration.png/pdf
- figure4_heterogeneity.png/pdf
- table1_sample_chars.tex
- table2_main_results.tex
- table3_heterogeneity.tex
- results_summary.csv
- descriptive_stats.csv
- event_study_results.csv
- heterogeneity_results.csv
- model_summary.txt

---

### Step 8: Report Compilation

**LaTeX Document:** `replication_report_65.tex`

**Compilation Commands:**
```bash
pdflatex -interaction=nonstopmode replication_report_65.tex  # First pass
pdflatex -interaction=nonstopmode replication_report_65.tex  # Second pass
pdflatex -interaction=nonstopmode replication_report_65.tex  # Third pass (for cross-refs)
```

**Output:** `replication_report_65.pdf` (21 pages)

---

## Key Analytical Decisions

1. **Treatment Group Definition**: Used birth years 1982-1986 to approximate ages 26-30 on June 15, 2012. This is an approximation since we cannot observe exact birth dates.

2. **Control Group Definition**: Used birth years 1977-1981 to approximate ages 31-35 on June 15, 2012.

3. **Undocumented Status**: Cannot distinguish documented from undocumented non-citizens. Used CITIZEN == 3 (not a citizen) as proxy, assuming non-citizens who arrived as children and have not naturalized are likely undocumented.

4. **Continuous Residence Requirement**: Operationalized as YRIMMIG <= 2007, which captures having been in the US since at least 2007.

5. **Outcome Definition**: Full-time employment defined as EMPSTAT == 1 (employed) AND UHRSWORK >= 35 (usual hours 35+).

6. **Exclusion of 2012**: The year of DACA implementation (June 2012) was excluded because we cannot distinguish pre- from post-implementation within that year.

7. **Weighting**: Used ACS person weights (PERWT) for population-representative estimates.

8. **Standard Errors**: Preferred specification uses heteroskedasticity-robust (HC1) standard errors.

9. **Controls in Preferred Model**: Female, married, education dummies (HS, some college, college+), number of children, and year fixed effects.

---

## Files Produced

| File | Description |
|------|-------------|
| analysis.py | Main analysis script |
| figures_and_tables.py | Figure and table generation |
| replication_report_65.tex | LaTeX source for report |
| replication_report_65.pdf | Compiled PDF report (21 pages) |
| run_log_65.md | This run log |
| results_summary.csv | Summary of all model estimates |
| descriptive_stats.csv | Descriptive statistics by group |
| event_study_results.csv | Event study coefficients |
| heterogeneity_results.csv | Subgroup analysis results |
| model_summary.txt | Full regression output |
| figure1_parallel_trends.png/pdf | Parallel trends figure |
| figure2_event_study.png/pdf | Event study figure |
| figure3_did_illustration.png/pdf | DiD illustration |
| figure4_heterogeneity.png/pdf | Heterogeneity analysis figure |
| table1_sample_chars.tex | Sample characteristics table |
| table2_main_results.tex | Main results table |
| table3_heterogeneity.tex | Heterogeneity table |

---

## Summary

This replication estimated the effect of DACA eligibility on full-time employment using a difference-in-differences design. The preferred estimate indicates that DACA eligibility increased full-time employment by 5.9 percentage points (95% CI: [0.037, 0.080]). The effect is statistically significant (p < 0.001) and robust across specifications. Heterogeneity analysis shows larger effects for males (8.1 pp) than females (3.2 pp), and for those with more education.
