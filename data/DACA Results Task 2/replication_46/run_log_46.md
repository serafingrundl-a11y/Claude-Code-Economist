# DACA Replication Study - Run Log

## Study Overview
**Research Question:** Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals.

**Treatment Group:** Ages 26-30 on June 15, 2012
**Control Group:** Ages 31-35 on June 15, 2012
**Outcome:** Full-time employment (35+ hours/week)
**Method:** Difference-in-Differences

---

## Data Description
- **Source:** American Community Survey (ACS) via IPUMS
- **Years:** 2006-2016 (pre-treatment: 2006-2011; post-treatment: 2013-2016; 2012 excluded)
- **Main file:** data.csv (~6GB)
- **Data dictionary:** acs_data_dict.txt

---

## Key Variable Definitions (from IPUMS data dictionary)

| Variable | Description | Values Used |
|----------|-------------|-------------|
| YEAR | Census/survey year | 2006-2016 (excl. 2012) |
| HISPAN | Hispanic origin | 1 = Mexican |
| BPL | Birthplace | 200 = Mexico |
| CITIZEN | Citizenship status | 3 = Not a citizen |
| YRIMMIG | Year of immigration | <= 2007 |
| BIRTHYR | Birth year | Used for age calculation |
| BIRTHQTR | Quarter of birth | 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec |
| UHRSWORK | Usual hours worked per week | >= 35 for full-time |
| EMPSTAT | Employment status | 1 = Employed |
| PERWT | Person weight | Used for weighting |

---

## Session Log

### Step 1: Initial Setup and Data Exploration
**Date:** 2026-01-26

**Commands:**
```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_46\data"
ls -la
head -5 data.csv
```

**Result:** Confirmed data structure:
- data.csv: ~6GB with 54 columns
- Columns include all required IPUMS variables
- Data dictionary available in acs_data_dict.txt

---

### Step 2: Sample Definition Decisions

**DACA Eligibility Criteria Applied:**
1. Hispanic-Mexican ethnicity: `HISPAN == 1`
2. Born in Mexico: `BPL == 200`
3. Not a citizen (proxy for undocumented): `CITIZEN == 3`
4. Arrived before age 16: `YRIMMIG - BIRTHYR < 16`
5. Arrived by 2007 (continuous residence since June 15, 2007): `YRIMMIG <= 2007`
6. Present in US on June 15, 2012 (assumed for all in sample)

**Age Group Determination:**
- DACA cutoff: Must not have reached 31st birthday by June 15, 2012
- Treatment group: Born between June 16, 1981 and June 15, 1986 (ages 26-30 on June 15, 2012)
- Control group: Born between June 16, 1976 and June 15, 1981 (ages 31-35 on June 15, 2012)

**Birth date approximation using BIRTHQTR:**
- Q1 (Jan-Mar) & Q2 (Apr-Jun): Assumed birthday occurred before June 15
- Q3 (Jul-Sep) & Q4 (Oct-Dec): Assumed birthday occurs after June 15

**Age on June 15, 2012 calculation:**
```python
age_june2012 = np.where(
    BIRTHQTR.isin([1, 2]),
    2012 - BIRTHYR,      # Birthday already passed
    2012 - BIRTHYR - 1   # Birthday not yet passed
)
```

---

### Step 3: Data Processing

**Analysis script:** `analysis.py`

**Processing steps:**
1. Load data in chunks (500,000 rows at a time) due to file size
2. Filter to Hispanic-Mexican (HISPAN=1), Mexican-born (BPL=200), non-citizens (CITIZEN=3)
3. Calculate age on June 15, 2012
4. Restrict to ages 26-35
5. Apply DACA eligibility: arrived before age 16 AND by 2007
6. Exclude 2012 observations
7. Create treatment/control and pre/post indicators

**Sample sizes:**
- Initial ACS data: ~30 million observations (2006-2016)
- After Hispanic-Mexican + Mexican-born + non-citizen: 701,347
- After age restriction (26-35): 181,229
- After arrived before age 16: 47,418
- After arrived by 2007: 47,418
- After excluding 2012: 43,238 (final sample)

---

### Step 4: Outcome Variable Definition

**Full-time employment:**
```python
fulltime = (EMPSTAT == 1) & (UHRSWORK >= 35)
```

- Must be employed (EMPSTAT = 1)
- Must usually work 35+ hours per week (UHRSWORK >= 35)

**Baseline rates (weighted):**
- Overall: 59.52%
- Treatment pre-period: 56.55%
- Treatment post-period: 61.99%
- Control pre-period: 61.35%
- Control post-period: 60.37%

---

### Step 5: Main Analysis Results

**Simple Difference-in-Differences:**
```
Treatment change: +5.43 pp (56.55% -> 61.99%)
Control change:   -0.99 pp (61.35% -> 60.37%)
DiD estimate:     +6.42 pp
```

**Regression Results:**

| Model | DiD Estimate | Std. Error | 95% CI |
|-------|--------------|------------|--------|
| Basic | 6.42 pp | 1.21 pp | [4.05, 8.79] |
| With covariates | 5.09 pp | 1.11 pp | [2.91, 7.27] |
| Year FE | 4.95 pp | 1.11 pp | [2.77, 7.12] |
| State+Year FE | 4.92 pp | 1.11 pp | [2.75, 7.09] |

**Preferred estimate:** 5.09 pp with demographic controls
- Statistically significant at p < 0.001
- Represents ~9% increase relative to pre-treatment mean

---

### Step 6: Event Study Analysis

Testing parallel trends assumption (reference year: 2011)

| Year | Coefficient | SE | 95% CI | Significant? |
|------|-------------|-----|--------|--------------|
| 2006 | -0.019 | 0.025 | [-0.069, 0.031] | No |
| 2007 | -0.047 | 0.025 | [-0.096, 0.003] | No |
| 2008 | -0.010 | 0.026 | [-0.061, 0.041] | No |
| 2009 | -0.017 | 0.026 | [-0.069, 0.035] | No |
| 2010 | -0.019 | 0.026 | [-0.070, 0.032] | No |
| 2011 | 0 (ref) | --- | --- | --- |
| 2013 | +0.047 | 0.027 | [-0.006, 0.101] | No |
| 2014 | +0.045 | 0.028 | [-0.009, 0.100] | No |
| 2015 | +0.024 | 0.028 | [-0.031, 0.078] | No |
| 2016 | +0.059 | 0.028 | [0.004, 0.113] | Yes |

**Interpretation:** Pre-treatment coefficients are small and insignificant, supporting parallel trends.

---

### Step 7: Robustness Checks

**By Gender:**
- Male: DiD = 5.54 pp (SE: 1.38), N = 24,243
- Female: DiD = 4.72 pp (SE: 1.83), N = 18,995

**Alternative Age Bandwidth (27-29 vs 32-34):**
- DiD = 6.05 pp (SE: 1.56), N = 25,606
- Similar to main estimate

**Placebo Test (fake treatment in 2008, pre-period only):**
- DiD = 1.30 pp (SE: 1.41), p = 0.357
- Not significant, supporting no pre-trends

---

### Step 8: Output Files Generated

**Data outputs:**
- `summary_stats.csv`: Summary statistics by group and period
- `event_study.csv`: Event study coefficients
- `main_results.csv`: All DiD estimates

**Report:**
- `replication_report_46.tex`: LaTeX source (~800 lines)
- `replication_report_46.pdf`: Final report (21 pages)

**Code:**
- `analysis.py`: Main analysis script

---

## Key Decisions Summary

1. **Sample restriction:** Used non-citizenship as proxy for undocumented status since ACS does not directly identify documentation.

2. **Age calculation:** Used birth quarter to approximate whether birthday occurred before/after June 15, 2012.

3. **Arrival criterion:** Required arrival by 2007 (not just before June 2007) as proxy for continuous residence.

4. **Excluded 2012:** Cannot distinguish pre/post within 2012 since ACS doesn't record interview month.

5. **Weighting:** Used PERWT person weights throughout for population-representative estimates.

6. **Standard errors:** Heteroskedasticity-robust (HC1) standard errors.

7. **Preferred specification:** Model with demographic controls (gender, marital status, education) - balances precision with transparency.

---

## Final Results Summary

**Preferred estimate:** DACA eligibility increased full-time employment by **5.09 percentage points** (95% CI: 2.91-7.27 pp).

This represents approximately a **9% increase** relative to the treatment group's pre-treatment mean of 56.55%.

The effect is:
- Statistically significant at p < 0.001
- Robust across specifications
- Consistent for both men and women
- Supported by parallel pre-trends

---

## Command Reference

```bash
# Data exploration
head -5 data.csv

# Run analysis
python analysis.py

# Compile LaTeX report
pdflatex replication_report_46.tex
pdflatex replication_report_46.tex  # Second pass for references
pdflatex replication_report_46.tex  # Third pass
```

---

## Software Environment

- Python 3.x with pandas, numpy, statsmodels, scipy
- pdfTeX (MiKTeX)
- Operating System: Windows
