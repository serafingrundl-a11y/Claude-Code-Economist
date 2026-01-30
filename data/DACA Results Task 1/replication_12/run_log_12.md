# Run Log - Replication 12

## Project: DACA Impact on Full-Time Employment
## Date Started: 2026-01-25

---

## Task Overview
Replicate analysis examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

---

## Key Decisions and Commands Log

### Step 1: Read Replication Instructions
- **Action**: Extracted text from `replication_instructions.docx`
- **Key findings**:
  - Research question: Effect of DACA eligibility on full-time employment (35+ hours/week)
  - Population: Hispanic-Mexican, Mexican-born individuals in the US
  - Treatment: DACA eligibility (program implemented June 15, 2012)
  - Outcome period: 2013-2016
  - Data: American Community Survey (ACS) from IPUMS, years 2006-2016
  - DACA eligibility criteria:
    1. Arrived in US before 16th birthday
    2. Had not turned 31 by June 15, 2012
    3. Lived continuously in US since June 15, 2007
    4. Present in US on June 15, 2012 without lawful status

### Step 2: Explore Data Structure
- **Action**: Examined data folder contents
- **Files found**:
  - `data/data.csv` - Main ACS data file (~34 million rows)
  - `data/acs_data_dict.txt` - Data dictionary for ACS variables
  - `data/state_demo_policy.csv` - Optional state-level data
  - `data/State Level Data Documentation.docx` - Documentation

### Step 3: Data Loading Strategy
- **Decision**: Due to large file size (34M rows), implemented chunked processing
- **Command**: Used pandas `read_csv` with `chunksize=1000000`
- **Columns retained**: YEAR, SAMPLE, PERWT, SEX, AGE, BIRTHQTR, BIRTHYR, HISPAN, HISPAND, BPL, BPLD, CITIZEN, YRIMMIG, YRSUSA1, YRSUSA2, EDUC, EDUCD, EMPSTAT, EMPSTATD, LABFORCE, UHRSWORK, MARST, NCHILD, METRO, STATEFIP, REGION, FAMSIZE

### Step 4: Sample Selection Criteria
- **Hispanic-Mexican ethnicity**: HISPAN == 1 OR (HISPAND >= 100 AND HISPAND <= 107)
- **Mexican-born**: BPL == 200
- **Non-citizen**: CITIZEN == 3
- **Working age**: AGE >= 16 AND AGE <= 64
- **Final sample size**: 618,640 observations

### Step 5: DACA Eligibility Variable Construction
- **Condition 1**: Arrived before 16th birthday
  - Calculated as: AGE - YRSUSA1 < 16
- **Condition 2**: Born on or after June 15, 1981 (under 31 on June 15, 2012)
  - Implemented as: BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR >= 2)
- **Condition 3**: Continuous presence since June 15, 2007
  - Implemented as: YRIMMIG <= 2007
- **DACA eligible**: 93,883 (15.2%)

### Step 6: Outcome Variable
- **Full-time employment**: UHRSWORK >= 35
- **Employed (any)**: EMPSTAT == 1
- **Full-time employment rate**: 57.1%

### Step 7: Time Period Definition
- **Pre-DACA**: 2006-2011 (345,792 observations)
- **Post-DACA**: 2013-2016 (215,678 observations)
- **Transition year**: 2012 (excluded from main analysis)

### Step 8: Control Variables
- Female (SEX == 2): 46.0%
- Age and Age-squared
- Married (MARST == 1): 54.2%
- High school or more (EDUC >= 6): 42.7%
- College (EDUC >= 10): 4.0%
- Metropolitan area (METRO >= 2): 87.5%
- Has children (NCHILD > 0): 63.2%
- State fixed effects (STATEFIP)
- Year fixed effects (YEAR)

### Step 9: Identification Strategy
- **Method**: Difference-in-Differences (DiD)
- **Treatment group**: DACA-eligible individuals
- **Control group**: Non-eligible Mexican-born non-citizens
- **Standard errors**: Clustered at state level

### Step 10: Main Results
| Model | DiD Coefficient | SE | p-value | N |
|-------|----------------|-----|---------|---|
| Basic DiD | 0.0884 | 0.0044 | <0.001 | 561,470 |
| DiD + Controls | 0.0371 | 0.0050 | <0.001 | 561,470 |
| DiD + State/Year FE | 0.0310 | 0.0048 | <0.001 | 561,470 |

**Preferred estimate**: 3.1 percentage point increase in full-time employment (Model 3)

### Step 11: Event Study Results
| Year | Coefficient | SE | Significant |
|------|-------------|-----|-------------|
| 2006 | -0.0193 | 0.0094 | ** |
| 2007 | -0.0158 | 0.0058 | *** |
| 2008 | -0.0038 | 0.0087 | |
| 2009 | -0.0009 | 0.0069 | |
| 2010 | 0.0040 | 0.0103 | |
| 2011 | 0 (ref) | - | |
| 2013 | 0.0069 | 0.0092 | |
| 2014 | 0.0198 | 0.0137 | |
| 2015 | 0.0382 | 0.0090 | *** |
| 2016 | 0.0390 | 0.0092 | *** |

### Step 12: Robustness Checks
| Specification | Coefficient | SE | N |
|--------------|-------------|-----|---|
| Age 18-40 | 0.0131 | 0.0047 | 341,332 |
| Male only | 0.0273 | 0.0045 | 303,717 |
| Female only | 0.0262 | 0.0073 | 257,753 |
| Any employment | 0.0414 | 0.0095 | 561,470 |
| Include 2012 | 0.0242 | 0.0034 | 618,640 |
| Survey weights | 0.0284 | 0.0038 | 561,470 |

### Step 13: Placebo Test
- **Fake treatment date**: 2009 (pre-DACA period)
- **Placebo coefficient**: 0.0147
- **p-value**: 0.0005
- **Interpretation**: Significant placebo effect suggests some pre-existing differential trends

### Step 14: Output Files
- `analysis_script.py` - Main analysis code
- `results_summary.csv` - Summary of all regression results
- `event_study_results.csv` - Event study coefficients
- `descriptive_stats.csv` - Descriptive statistics by group
- `figure1_main_results.png/pdf` - Main results visualization
- `figure2_robustness.png/pdf` - Robustness checks visualization
- `figure3_pretrends.png/pdf` - Pre-trends visualization
- `replication_report_12.tex` - LaTeX report
- `replication_report_12.pdf` - Final PDF report

---

## Key Methodological Decisions

1. **Sample restriction to non-citizens**: Following the instructions, assumed non-citizens are undocumented for DACA purposes since we cannot distinguish documented vs undocumented immigrants in the data.

2. **Age restriction 16-64**: Standard working-age population definition to focus on labor market outcomes.

3. **Exclusion of 2012**: DACA was implemented mid-year (June 15), making 2012 a transition year with mixed pre/post observations.

4. **State-clustered standard errors**: Account for correlation of errors within states over time.

5. **Event study specification**: Reference year 2011 (last pre-DACA year), allowing examination of pre-trends and dynamic treatment effects.

---

## Interpretation Notes

The preferred estimate of 3.1 percentage points should be interpreted with caution due to:
1. Some evidence of pre-trends in 2006-2007
2. Significant placebo test result
3. Inability to distinguish documented vs undocumented immigrants

The effect appears to grow stronger over time (2015-2016 show larger coefficients than 2013-2014), possibly reflecting increased DACA uptake and labor market adjustment.
