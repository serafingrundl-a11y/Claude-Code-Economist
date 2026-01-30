# Run Log - DACA Replication Study (ID: 69)

## Overview
This document logs all key commands, decisions, and steps taken during the replication analysis of the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

---

## Session Log

### Step 1: Data Exploration

**Files in data folder:**
- `data.csv` - Main ACS data file (~6.26 GB, 33,851,425 rows)
- `acs_data_dict.txt` - Data dictionary for IPUMS variables
- `state_demo_policy.csv` - Optional state-level supplementary data
- `State Level Data Documentation.docx` - Documentation for state data

**Key Variables Identified:**
- `YEAR` - Census year (2006-2016)
- `HISPAN`/`HISPAND` - Hispanic origin (1 = Mexican)
- `BPL`/`BPLD` - Birthplace (200 = Mexico)
- `CITIZEN` - Citizenship status (3 = Not a citizen)
- `YRIMMIG` - Year of immigration
- `BIRTHYR` - Birth year
- `BIRTHQTR` - Birth quarter (1-4)
- `UHRSWORK` - Usual hours worked per week
- `AGE` - Age at survey
- `PERWT` - Person weight for population estimates

### Step 2: DACA Eligibility Criteria Definition

Based on instructions, DACA eligibility requires:

1. **Arrived unlawfully in the US before their 16th birthday**
   - Cannot directly observe lawful vs. unlawful arrival
   - Proxy: Non-citizens who have not received immigration papers (CITIZEN == 3)
   - Age at arrival = YRIMMIG - BIRTHYR < 16

2. **Had not yet had their 31st birthday as of June 15, 2012**
   - Born after June 15, 1981
   - Operational definition: BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR in [3, 4])

3. **Lived continuously in the US since June 15, 2007**
   - YRIMMIG <= 2007

4. **Present in the US on June 15, 2012 and did not have lawful status**
   - CITIZEN == 3 (Not a citizen, not naturalized)

5. **Hispanic-Mexican ethnicity and born in Mexico**
   - HISPAN == 1 (Mexican)
   - BPL == 200 (Mexico)

### Step 3: Identification Strategy

**Design: Difference-in-Differences (DiD)**

- **Treatment Group**: DACA-eligible individuals (meet all criteria above)
- **Control Group**: Childhood arrivals (arrived < age 16) who were too old for DACA (31+ by June 2012)

- **Time Periods**:
  - Pre-treatment: 2006-2011 (before DACA implementation)
  - Post-treatment: 2013-2016 (after DACA, as specified in instructions)
  - Note: 2012 is excluded because DACA implemented mid-year (June 15, 2012)

**Outcome Variable**: Full-time employment = 1 if UHRSWORK >= 35, 0 otherwise

### Step 4: Sample Construction

Sample restrictions applied:
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Non-citizens (CITIZEN == 3)
4. Arrived by 2007 (YRIMMIG <= 2007)
5. Arrived before age 16 (age_at_arrival < 16)
6. Working age at survey (18 <= AGE <= 50)

**Final Sample:**
- Total observations: 119,839
- Treatment group (DACA-eligible): 71,347
- Control group (not eligible): 48,492
- Weighted population: ~16.3 million

---

## Analysis Code

Created `analysis.py` with the following components:

### Data Loading
- Used chunked reading due to large file size (~6 GB)
- Applied filters early to reduce memory usage
- Loaded only necessary columns

### Variable Construction
```python
# DACA eligibility
df['under_31_june2012'] = (df['BIRTHYR'] >= 1982) | \
                          ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'].isin([3, 4])))
df['daca_eligible'] = df['under_31_june2012'].astype(int)

# Post-DACA indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# DiD interaction
df['treat_post'] = df['daca_eligible'] * df['post']

# Outcome: Full-time employment
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
```

### Models Estimated

1. **Model 1**: Basic DiD (no controls)
2. **Model 2**: DiD + demographics (age, age², female, married)
3. **Model 3**: DiD + demographics + education
4. **Model 4**: DiD + demographics + education + year FE
5. **Model 5**: DiD + demographics + education + year FE + state FE (PREFERRED)

### Robustness Checks
- Alternative age restriction (18-45)
- Men only
- Women only
- Any employment as outcome
- Clustered standard errors at state level

### Event Study
- Estimated treatment × year interactions
- Reference year: 2011 (year before DACA)

---

## Key Results

### Main Finding (Preferred Specification)

**DACA Eligible × Post coefficient: -0.003**
- Standard Error: 0.007
- 95% CI: [-0.016, 0.011]
- p-value: 0.666

**Interpretation**: No statistically significant effect of DACA eligibility on full-time employment. The 95% CI rules out effects larger than ~1.6 percentage points in either direction.

### 2x2 DiD Table

|                | Control | Treatment | Difference |
|----------------|---------|-----------|------------|
| Pre-DACA       | 0.686   | 0.525     | -0.161     |
| Post-DACA      | 0.651   | 0.569     | -0.082     |
| Change         | -0.035  | 0.044     | **0.079**  |

Raw DiD estimate (0.079) is large but confounded. Controlled estimate is ~0 after adding year FE.

### Model Progression

| Model | Coefficient | SE | p-value |
|-------|------------|-----|---------|
| (1) Basic DiD | 0.079 | 0.007 | 0.000 |
| (2) + Demographics | 0.015 | 0.007 | 0.026 |
| (3) + Education | 0.012 | 0.007 | 0.070 |
| (4) + Year FE | -0.002 | 0.007 | 0.806 |
| (5) + State FE | **-0.003** | 0.007 | 0.666 |

### Robustness Checks

| Specification | Coefficient | SE | N |
|--------------|------------|-----|------|
| Main (Model 5) | -0.003 | 0.007 | 119,839 |
| Age 18-45 | -0.008 | 0.007 | 113,977 |
| Men Only | -0.028*** | 0.008 | 67,825 |
| Women Only | 0.020* | 0.011 | 52,014 |
| Any Employment | 0.005 | 0.007 | 119,839 |
| Clustered SE | -0.003 | 0.005 | 119,839 |

### Event Study Pre-Trends

Some evidence of pre-existing convergence (coefficients for 2006-2008 are positive and significant relative to 2011). This raises concerns about the parallel trends assumption.

---

## Key Decisions and Justifications

### 1. Control Group Definition
**Decision**: Use childhood arrivals (arrived < age 16) who were 31+ by June 2012
**Justification**: These individuals share the experience of childhood migration but are excluded from DACA solely due to the age cutoff, providing a natural comparison group.

### 2. Exclusion of 2012
**Decision**: Exclude 2012 from analysis
**Justification**: DACA was implemented mid-year (June 15, 2012), so observations from 2012 cannot be cleanly classified as pre- or post-treatment.

### 3. Citizenship Proxy for Undocumented Status
**Decision**: Use CITIZEN == 3 (not a citizen) as proxy for undocumented
**Justification**: ACS does not directly identify undocumented immigrants. Non-citizens who have not received immigration papers are the best available proxy.

### 4. Age Restriction (18-50)
**Decision**: Restrict to ages 18-50
**Justification**: Ensures overlap between treatment and control groups while focusing on working-age population. Treatment group is young (mean age 24), so including older ages provides a suitable control group.

### 5. Preferred Specification
**Decision**: Model with year FE and state FE
**Justification**: Year FE absorb aggregate time trends affecting both groups. State FE absorb time-invariant state characteristics. This provides the most credible identification of the DACA effect.

### 6. Survey Weights
**Decision**: Use PERWT (person weight) for all analyses
**Justification**: Ensures population-representative estimates accounting for ACS sampling design.

### 7. Robust Standard Errors
**Decision**: Use HC1 (Huber-White) robust standard errors
**Justification**: Provides valid inference under heteroskedasticity. Also report clustered SEs as robustness check.

---

## Files Produced

1. `analysis.py` - Main analysis script
2. `results_summary.txt` - Text summary of main results
3. `latex_data.json` - Data for LaTeX tables
4. `descriptive_stats.json` - Descriptive statistics
5. `replication_report_69.tex` - LaTeX source for report
6. `replication_report_69.pdf` - Final PDF report (24 pages)
7. `run_log_69.md` - This log file

---

## Conclusion

The analysis finds no statistically significant effect of DACA eligibility on full-time employment among Hispanic-Mexican immigrants born in Mexico. The preferred estimate is -0.003 (SE = 0.007), with a 95% CI of [-0.016, 0.011]. The null result is robust to alternative specifications and sample definitions.

Notable finding: Heterogeneous effects by gender (negative for men, positive for women) approximately cancel in the pooled sample.

Limitation: Event study reveals some pre-existing convergence, raising concerns about the parallel trends assumption.

---

## Commands Run

```bash
# List data files
ls -la data/

# Check data dimensions
wc -l data/data.csv

# View data dictionary
head -100 data/acs_data_dict.txt

# Run analysis
python analysis.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_69.tex
pdflatex -interaction=nonstopmode replication_report_69.tex  # Second pass for references
```

---

*Log completed: DACA Replication Study ID 69*
