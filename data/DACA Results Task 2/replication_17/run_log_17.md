# Replication Run Log - DACA Employment Effects Study

## Project Overview
**Research Question:** Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for the DACA program (treatment) on the probability of full-time employment (35+ hours/week)?

**Identification Strategy:** Difference-in-differences comparing:
- Treatment group: Ages 26-30 at time of DACA (June 15, 2012)
- Control group: Ages 31-35 at time of DACA (otherwise would have been eligible but for age)

**Outcome:** Full-time employment (UHRSWORK >= 35)

---

## Session Log

### 2026-01-26: Initial Setup and Data Exploration

#### Step 1: Read Replication Instructions
- Reviewed `replication_instructions.docx`
- Key parameters:
  - DACA implemented: June 15, 2012
  - Post-treatment years to analyze: 2013-2016
  - Pre-treatment years available: 2006-2011
  - 2012 excluded (cannot distinguish pre/post DACA within year)
  - Target population: Hispanic-Mexican, born in Mexico, non-citizens

#### Step 2: Review Data Dictionary (`acs_data_dict.txt`)
Key variables identified:
- `YEAR`: Survey year (2006-2016 available)
- `BIRTHYR`: Birth year
- `BIRTHQTR`: Quarter of birth
- `HISPAN`/`HISPAND`: Hispanic origin (1 = Mexican, 100-107 = Mexican detailed)
- `BPL`/`BPLD`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `UHRSWORK`: Usual hours worked per week
- `EMPSTAT`: Employment status (1 = Employed)
- `PERWT`: Person weight for survey estimates
- `AGE`: Age at time of survey
- `SEX`: Sex (1 = Male, 2 = Female)
- `EDUC`: Education level
- `MARST`: Marital status
- `STATEFIP`: State FIPS code

#### Step 3: DACA Eligibility Criteria
Per instructions, eligible if:
1. Arrived in US before 16th birthday
2. Not yet 31 by June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

For our analysis:
- Treatment group: Born 1982-1986 (ages 26-30 on June 15, 2012)
- Control group: Born 1977-1981 (ages 31-35 on June 15, 2012)
- Both groups must meet other eligibility criteria

---

## Analysis Commands and Code

### Data Loading and Cleaning

**File:** `analysis.py`

```python
# Load ACS data (33.8 million records)
df = pd.read_csv('data/data.csv', usecols=cols_needed)

# Sample selection filters:
# 1. Hispanic-Mexican ethnicity (HISPAN == 1 or HISPAND 100-107)
# 2. Born in Mexico (BPL == 200)
# 3. Non-citizens (CITIZEN == 3)
# 4. Arrived before age 16 (YRIMMIG - BIRTHYR < 16)
# 5. Continuous presence (YRIMMIG <= 2007)
# 6. Exclude 2012 (ambiguous timing)
```

### Sample Size Progression
| Filter | N |
|--------|---|
| Total ACS records | 33,851,424 |
| Hispanic-Mexican | 2,945,521 |
| Born in Mexico | 991,261 |
| Non-citizens | 701,347 |
| Ages 26-35 at DACA | 181,229 |
| Arrived before 16 | 47,418 |
| In US by 2007 | 47,418 |
| Excluding 2012 | **43,238** |

### Age Calculation
Age at DACA (June 15, 2012) calculated as:
- `age_at_daca = 2012 - BIRTHYR`
- Adjusted down by 1 for those born in Q3 or Q4 (birthday not yet reached)

---

## Key Decisions

1. **Sample Definition:**
   - Used CITIZEN == 3 (not a citizen) as proxy for undocumented status
   - Applied all DACA eligibility criteria to both treatment and control groups

2. **Treatment/Control Groups:**
   - Treatment: Age 26-30 at DACA (DACA-eligible)
   - Control: Age 31-35 at DACA (would be eligible but for age cutoff)

3. **Outcome Variable:**
   - Full-time employment: UHRSWORK >= 35 hours per week

4. **Time Periods:**
   - Pre-treatment: 2006-2011
   - Post-treatment: 2013-2016
   - 2012 excluded (DACA announced mid-year)

5. **Estimation:**
   - Weighted least squares using PERWT (person weights)
   - Robust standard errors (HC1)
   - Controls: sex, marital status, education, age
   - Fixed effects: year, state

---

## Results Summary

### Sample Composition (Final Sample N = 43,238)
| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Treatment (26-30) | 16,694 | 8,776 | 25,470 |
| Control (31-35) | 11,683 | 6,085 | 17,768 |

### Full-time Employment Rates (Weighted)
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 0.631 | 0.660 | +0.029 |
| Control (31-35) | 0.673 | 0.643 | -0.030 |
| **DiD Estimate** | | | **+0.059** |

### Regression Results
| Model | DiD Coefficient | SE | p-value |
|-------|-----------------|-----|---------|
| Basic DiD | 0.059 | 0.012 | <0.001 |
| + Demographics | 0.048 | 0.011 | <0.001 |
| + Year FE | 0.047 | 0.011 | <0.001 |
| + Year + State FE | **0.046** | **0.011** | **<0.001** |

### Preferred Estimate
- **DiD coefficient: 0.046 (SE = 0.011)**
- 95% CI: [0.025, 0.067]
- Interpretation: DACA eligibility increased full-time employment probability by 4.6 percentage points

### Robustness Checks
1. Employment (any) as outcome: 0.044 (SE = 0.010)
2. Males only: 0.035 (SE = 0.012)
3. Females only: 0.052 (SE = 0.018)
4. Narrower age bands (27-29 vs 32-34): 0.040 (SE = 0.012)

### Event Study (Reference: 2011)
| Year | Coefficient | SE | Significant |
|------|-------------|-----|-------------|
| 2006 | 0.004 | 0.023 | |
| 2007 | -0.033 | 0.022 | |
| 2008 | 0.008 | 0.023 | |
| 2009 | -0.009 | 0.024 | |
| 2010 | -0.013 | 0.023 | |
| 2011 | 0 (ref) | - | |
| 2013 | 0.036 | 0.024 | |
| 2014 | 0.037 | 0.025 | |
| 2015 | 0.021 | 0.025 | |
| 2016 | 0.067 | 0.025 | *** |

Pre-trends appear relatively flat, supporting parallel trends assumption.

---

## Output Files
- `analysis.py` - Main analysis script
- `results_summary.csv` - Cell means by group and period
- `model_coefficients.csv` - Regression coefficients from all models
- `event_study.csv` - Event study coefficients
- `replication_report_17.tex` - LaTeX report
- `replication_report_17.pdf` - Final PDF report

---

## Conclusion
DACA eligibility is associated with a statistically significant 4.6 percentage point increase in full-time employment among eligible Hispanic-Mexican, Mexican-born non-citizens. The effect is robust to inclusion of demographic controls and fixed effects for year and state. Event study analysis shows relatively flat pre-trends, supporting the validity of the difference-in-differences design.
