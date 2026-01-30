# Replication Run Log - ID 86

## Date: 2026-01-25

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Key Decisions and Commands

### Step 1: Data Exploration

**Files available:**
- `data/data.csv` - Main ACS data file (33,851,425 rows including header)
- `data/acs_data_dict.txt` - Data dictionary for ACS variables
- `data/state_demo_policy.csv` - Optional state-level data
- `data/State Level Data Documentation.docx` - Documentation for state data

**Data spans:** 2006-2016 ACS one-year samples

### Step 2: Variable Identification

**Key variables from data dictionary:**
- `YEAR` - Census year (2006-2016)
- `HISPAN` / `HISPAND` - Hispanic origin (1 = Mexican, detailed codes 100-107)
- `BPL` / `BPLD` - Birthplace (200 = Mexico)
- `CITIZEN` - Citizenship status (3 = Not a citizen)
- `YRIMMIG` - Year of immigration
- `BIRTHYR` - Birth year
- `BIRTHQTR` - Birth quarter (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- `AGE` - Age
- `UHRSWORK` - Usual hours worked per week
- `EMPSTAT` - Employment status (1=Employed, 2=Unemployed, 3=Not in labor force)
- `PERWT` - Person weight

### Step 3: DACA Eligibility Criteria

Per instructions, eligibility requires:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007 (arrived before mid-2007)
4. Were present in the US on June 15, 2012 and did not have lawful status

**Operationalization:**
- Mexican-born: `BPL == 200` (Mexico)
- Hispanic-Mexican ethnicity: `HISPAN == 1`
- Non-citizen: `CITIZEN == 3`
- Arrived before age 16: `YRIMMIG - BIRTHYR < 16`
- Born after June 15, 1981: For conservative estimate, `BIRTHYR >= 1982` OR (`BIRTHYR == 1981` AND `BIRTHQTR >= 3`)
- In US since mid-2007: `YRIMMIG <= 2007`

### Step 4: Outcome Variable

**Full-time employment:** `UHRSWORK >= 35` (usual hours worked per week)

### Step 5: Identification Strategy

**Difference-in-Differences (DiD) Design:**
- Treatment group: DACA-eligible individuals (meeting all criteria above)
- Control group: Non-DACA-eligible Mexican-born Hispanic-Mexican non-citizens (e.g., arrived too late or too old)
- Pre-period: 2006-2011 (pre-DACA)
- Post-period: 2013-2016 (post-DACA implementation)
- 2012 excluded: Cannot distinguish pre/post within 2012 since DACA implemented June 15, 2012

**Model specification:**
$$Y_{it} = \alpha + \beta_1 \cdot Eligible_i + \beta_2 \cdot Post_t + \beta_3 \cdot (Eligible_i \times Post_t) + X_{it}\gamma + \epsilon_{it}$$

Where $\beta_3$ is the DiD estimate of DACA's effect on full-time employment.

### Step 6: Analysis Implementation

Using Python with pandas, statsmodels for regression analysis.

---

## Commands Executed

```bash
# Initial exploration
head -5 data/data.csv
wc -l data/data.csv

# Run analysis
python analysis.py
```

---

## Analysis Results Summary

### Sample Construction
- Total ACS observations: 33,851,424
- After Hispanic-Mexican filter: 2,945,521
- After Mexico birthplace filter: 991,261
- After non-citizen filter: 701,347
- After excluding 2012: 636,722
- After working-age (16-64) filter: 561,470 (final sample)

### DACA Eligibility
- DACA eligible: 83,611
- Not eligible: 477,859

### Key Results

**Simple DiD (2x2 table):**
|                | Pre      | Post     | Difference |
|----------------|----------|----------|------------|
| Not Eligible   | 0.6276   | 0.6013   | -0.0263    |
| Eligible       | 0.4522   | 0.5214   | +0.0692    |
| **DiD**        |          |          | **+0.0956**|

**Regression Results:**
| Model | Specification | DiD Estimate | Std. Error | 95% CI |
|-------|--------------|--------------|------------|--------|
| 1 | Basic DiD | 0.0956 | 0.0046 | [0.0866, 0.1046] |
| 2 | + Demographics | 0.0414 | 0.0042 | [0.0331, 0.0497] |
| 3 | + Education | 0.0383 | 0.0042 | [0.0301, 0.0466] |
| 4 | + Year FE | 0.0311 | 0.0042 | - |
| 5 | + Year + State FE | 0.0305 | 0.0042 | - |

**Preferred Estimate:** 0.0956 (basic DiD) or 0.0305-0.0414 with controls

### Robustness Checks
- Employment (any work) effect: 0.0484 (SE: 0.0042)
- Male-specific effect: 0.0363 (SE: 0.0055)
- Female-specific effect: 0.0381 (SE: 0.0063)

### Event Study Results
Pre-treatment years show small, statistically insignificant coefficients, supporting parallel trends assumption. Post-treatment years show increasing positive effects.

---

## Files Created
- `analysis.py` - Main analysis script
- `results_summary.csv` - Summary of key results
- `model_results.txt` - Detailed regression output
- `replication_report_86.tex` - LaTeX report
- `replication_report_86.pdf` - Final PDF report
- `run_log_86.md` - This log file
