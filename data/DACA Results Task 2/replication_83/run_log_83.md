# Replication Run Log - Task 83

## Overview
Independent replication of DACA impact on full-time employment among Hispanic-Mexican individuals born in Mexico.

## Research Design
- **Treatment Group**: Ages 26-30 at policy implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at policy implementation (June 15, 2012)
- **Outcome**: Full-time employment (35+ hours/week)
- **Method**: Difference-in-differences
- **Pre-period**: 2006-2011 (excluding 2012 due to inability to distinguish pre/post)
- **Post-period**: 2013-2016

## Data Source
American Community Survey (ACS) from IPUMS USA, years 2006-2016.

---

## Session Log

### Step 1: Initial Setup and Data Exploration
- Read replication instructions from `replication_instructions.docx`
- Examined data dictionary (`acs_data_dict.txt`)
- Confirmed data structure with 54 variables

### Key Variables Identified:
- `YEAR`: Survey year (2006-2016)
- `HISPAN`: Hispanic origin (1 = Mexican)
- `BPL`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Birth year
- `BIRTHQTR`: Birth quarter
- `UHRSWORK`: Usual hours worked per week
- `PERWT`: Person weight for population estimates

### Sample Eligibility Criteria (based on DACA requirements):
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Not a citizen (CITIZEN == 3)
4. Arrived in US before age 16
5. Present in US since at least June 15, 2007 (arrived by 2007)
6. Age groups:
   - Treatment: Age 26-30 as of June 15, 2012 (born 1982-1986)
   - Control: Age 31-35 as of June 15, 2012 (born 1977-1981)

### Step 2: Data Processing Decisions

**Decision 1: Exclude 2012**
- Rationale: ACS does not record month of data collection, so observations in 2012 cannot be classified as pre- or post-DACA (implemented June 15, 2012).

**Decision 2: Age Calculation**
- Age at DACA implementation calculated using BIRTHYR and BIRTHQTR
- If birth quarter <= 2 (Jan-Jun): Age = 2012 - BIRTHYR
- If birth quarter >= 3 (Jul-Dec): Age = 2012 - BIRTHYR - 1 (hadn't had birthday by June 15)

**Decision 3: Documentation Status**
- Per instructions: "anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes"
- Used CITIZEN = 3 (Not a citizen) as proxy for undocumented status

**Decision 4: Arrival Age Requirement**
- DACA requires arrival before 16th birthday
- Calculated as: age_at_arrival = YRIMMIG - BIRTHYR
- Kept observations where age_at_arrival < 16

**Decision 5: Continuous Presence Requirement**
- DACA requires continuous presence since June 15, 2007
- Approximated by requiring YRIMMIG <= 2007

---

## Sample Construction Results

| Restriction | Observations | Dropped |
|-------------|--------------|---------|
| Raw data (2006-2016) | 33,851,424 | -- |
| Exclude 2012 | 30,738,394 | 3,113,030 |
| Hispanic-Mexican | 2,663,503 | 28,074,891 |
| Born in Mexico | 898,879 | 1,764,624 |
| Non-citizen | 636,722 | 262,157 |
| Age 26-35 at DACA | 164,874 | 471,848 |
| Arrived before age 16 | 43,238 | 121,636 |
| **Final Sample** | **43,238** | -- |

---

## Step 3: Difference-in-Differences Analysis

### Models Estimated

1. **Model 1**: Basic OLS (unweighted)
2. **Model 2**: Weighted DID (using PERWT)
3. **Model 3**: Weighted DID + covariates (female, married, education)
4. **Model 4**: Weighted DID + year fixed effects
5. **Model 5 (Preferred)**: Weighted DID + year FE + covariates

### Results Summary

| Model | DID Estimate | Std. Error | P-value |
|-------|-------------|------------|---------|
| Model 1 (Basic) | 0.0516 | 0.0100 | <0.001 |
| Model 2 (Weighted) | 0.0590 | 0.0117 | <0.001 |
| Model 3 (+ Covariates) | 0.0466 | 0.0107 | <0.001 |
| Model 4 (+ Year FE) | 0.0574 | 0.0117 | <0.001 |
| Model 5 (Full) | **0.0449** | **0.0107** | **<0.001** |

### Preferred Estimate (Model 5)

- **Effect Size**: 0.0449 (4.49 percentage points)
- **Standard Error**: 0.0107
- **95% CI**: [0.0239, 0.0658]
- **P-value**: <0.0001
- **Sample Size**: 43,238

### Interpretation

DACA eligibility increased full-time employment by approximately 4.5 percentage points among Hispanic-Mexican individuals born in Mexico who met the program's other eligibility requirements. This represents about a 7% increase relative to the treatment group's pre-period employment rate of 63.1%.

---

## Subgroup Analysis

| Subgroup | N | DID Estimate | SE |
|----------|---|--------------|-----|
| Male | 24,243 | 0.0345 | 0.0124 |
| Female | 18,995 | 0.0492 | 0.0181 |
| Full Sample | 43,238 | 0.0449 | 0.0107 |

Both subgroups show positive and statistically significant effects.

---

## Files Generated

1. `analysis.py` - Main analysis script
2. `did_results.csv` - DID estimates across models
3. `yearly_trends.csv` - Year-by-year employment rates
4. `summary_stats.csv` - Summary statistics
5. `model5_full_results.csv` - Full preferred model output
6. `replication_report_83.tex` - LaTeX report
7. `replication_report_83.pdf` - Final PDF report

---

## Commands Run

```python
# Main analysis
python analysis.py
```

```bash
# PDF compilation
pdflatex replication_report_83.tex
pdflatex replication_report_83.tex  # Second pass for TOC
```
