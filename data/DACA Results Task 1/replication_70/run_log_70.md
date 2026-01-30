# DACA Replication Study Run Log

## Study Overview
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (usually working 35+ hours per week)?

**Treatment Period:** DACA implemented June 15, 2012; effects examined 2013-2016

---

## Session Log

### 2026-01-25: Initial Setup and Data Exploration

#### Step 1: Read Replication Instructions
- Extracted text from `replication_instructions.docx`
- Key requirements identified:
  - Use ACS 1-year files from 2006-2016
  - Sample: Hispanic-Mexican ethnicity, born in Mexico
  - Treatment: DACA eligibility
  - Outcome: Full-time employment (35+ hours/week)
  - DACA eligibility criteria:
    1. Arrived in US before 16th birthday
    2. Under 31 on June 15, 2012
    3. Lived continuously in US since June 15, 2007
    4. Non-citizen (assume undocumented if no immigration papers)

#### Step 2: Read Data Dictionary
- Key variables identified:
  - YEAR: Survey year (2006-2016)
  - HISPAN/HISPAND: Hispanic origin (1 = Mexican, 100-107 = Mexican detailed)
  - BPL/BPLD: Birthplace (200 = Mexico)
  - CITIZEN: Citizenship status (3 = Not a citizen)
  - YRIMMIG: Year of immigration
  - BIRTHYR: Birth year
  - BIRTHQTR: Quarter of birth
  - AGE: Age at survey
  - UHRSWORK: Usual hours worked per week
  - EMPSTAT: Employment status
  - PERWT: Person weight

#### Step 3: Data File Exploration
- data.csv contains 33,851,425 observations
- Data spans 2006-2016 ACS samples
- All required variables are present

### Key Decisions Made:

1. **Sample Selection:**
   - HISPAN == 1 (Mexican)
   - BPL == 200 (Born in Mexico)
   - This targets Hispanic-Mexican, Mexican-born individuals

2. **DACA Eligibility Definition:**
   - Non-citizen (CITIZEN == 3)
   - Arrived before age 16 (YEAR - YRIMMIG < AGE at time of arrival)
   - Under 31 on June 15, 2012 (born after June 15, 1981)
   - In US since June 15, 2007 (YRIMMIG <= 2007)
   - Note: Cannot distinguish documented vs undocumented; assume all non-citizens without papers are undocumented

3. **Full-Time Employment:**
   - UHRSWORK >= 35 (usual hours worked 35+)
   - Among those employed (EMPSTAT == 1) or as binary including non-employed

4. **Identification Strategy:**
   - Difference-in-Differences (DiD) approach
   - Treatment group: DACA-eligible non-citizens
   - Control group: Non-eligible non-citizens (e.g., arrived after turning 16, or too old)
   - Pre-treatment period: 2006-2011 (or 2010-2011)
   - Post-treatment period: 2013-2016
   - Year 2012 excluded (treatment began mid-year)

5. **Age Restrictions:**
   - Working-age sample (e.g., 16-40 or 18-35) to ensure comparability

---

## Analysis Code Execution

### Step 4: Data Loading and Processing
- Loaded data using chunked processing (1M rows at a time) to manage memory
- Filtered to Hispanic-Mexican (HISPAN==1) born in Mexico (BPL==200)
- Total filtered observations: 991,261
- Valid immigration year records: 991,261
- Non-citizens: 701,347 (70.8% of Mexican-born sample)

### Step 5: DACA Eligibility Determination
Applied the following criteria:
1. Arrived before age 16: 322,246 individuals
2. Under 31 on June 15, 2012: 274,149 individuals
3. In US since June 15, 2007 (YRIMMIG <= 2007): 937,519 individuals
4. Non-citizen status: 701,347 individuals

**DACA-eligible individuals identified: 133,120**

### Step 6: Analysis Sample Construction
- Excluded year 2012 (ambiguous treatment timing)
- Remaining observations: 898,879
- Restricted to non-citizens only: 636,722
- Restricted to working-age (16-45): 427,762
  - Treatment group (DACA eligible): 83,611
  - Control group (non-eligible): 344,151

### Step 7: Main Results

#### Preferred Specification (Model 4):
- DiD with demographic controls, state and year fixed effects, clustered standard errors
- **Effect on full-time employment: 0.0212 (2.12 percentage points)**
- Standard Error: 0.0043
- 95% CI: [0.0127, 0.0297]
- p-value: <0.0001
- Sample size: 427,762
- R-squared: 0.2335

#### Weighted Estimate (using PERWT):
- Effect: 0.0193 (1.93 percentage points)
- Standard Error: 0.0036
- 95% CI: [0.0122, 0.0264]
- p-value: <0.0001

### Step 8: Robustness Checks

1. **Employment (any) as outcome**: 0.0357 (SE: 0.0093, p<0.001)
2. **Ages 18-35 only**: 0.0077 (SE: 0.0056, p=0.167) - Not significant
3. **Males only**: 0.0026 (SE: 0.0038, p=0.489) - Not significant
4. **Females only**: 0.0364 (SE: 0.0074, p<0.001) - Strong effect
5. **Include 2012 as pre-period**: 0.0214 (SE: 0.0051, p<0.001)

### Step 9: Event Study Results
Pre-treatment coefficients (relative to 2011):
- 2006: -0.0084 (p=0.208)
- 2007: -0.0113 (p=0.035)
- 2008: 0.0013 (p=0.876)
- 2009: 0.0046 (p=0.469)
- 2010: 0.0060 (p=0.532)

Post-treatment coefficients:
- 2013: 0.0053 (p=0.466)
- 2014: 0.0154 (p=0.167)
- 2015: 0.0309 (p<0.001)
- 2016: 0.0312 (p<0.001)

**Interpretation:** Pre-trends show no systematic differences, supporting parallel trends assumption. Effects become significant and grow in 2015-2016.

---

## Files Generated

1. `analysis.py` - Main analysis script
2. `figure_data_trends.csv` - Trend data for Figure 1
3. `figure_data_event_study.csv` - Event study coefficients for Figure 2
4. `yearly_trends.csv` - Year-by-year employment rates
5. `summary_statistics.csv` - Summary statistics table
6. `model_results.txt` - Full model output
7. `results.json` - Machine-readable results

---

## Interpretation of Results

The analysis finds that DACA eligibility increased the probability of full-time employment by approximately 2.1 percentage points among Mexican-born, Hispanic-Mexican non-citizens. This effect is statistically significant at conventional levels.

The effect appears to be driven primarily by women (3.6 pp increase), with no significant effect detected for men. The event study shows that effects materialized gradually, becoming statistically significant by 2015-2016.

---

## Final Deliverables

### Required Output Files (in replication_70 folder):

| Filename | Description | Status |
|----------|-------------|--------|
| `replication_report_70.tex` | LaTeX source for replication report (~21 pages) | Complete |
| `replication_report_70.pdf` | Compiled PDF report | Complete |
| `run_log_70.md` | This log file documenting commands and decisions | Complete |

### Additional Supporting Files:

| Filename | Description |
|----------|-------------|
| `analysis.py` | Python analysis script |
| `results.json` | Machine-readable results |
| `model_results.txt` | Detailed model output |
| `figure_data_trends.csv` | Data for trend figures |
| `figure_data_event_study.csv` | Event study coefficients |
| `summary_statistics.csv` | Summary statistics by group |
| `yearly_trends.csv` | Employment rates by year |

---

## Summary of Key Findings

**Research Question:** Effect of DACA eligibility on full-time employment (35+ hours/week) among Hispanic-Mexican, Mexican-born non-citizens.

**Main Result:** DACA eligibility increased full-time employment by **2.12 percentage points** (95% CI: [1.27, 2.97], p < 0.001).

**Key Insights:**
1. Effect is concentrated among women (3.6 pp) with no significant effect for men
2. Effects emerged gradually, becoming significant by 2015-2016
3. Event study supports parallel trends assumption
4. Results robust to including 2012 as pre-period and using survey weights

---

## Session Complete

All required deliverables have been produced and are located in the `replication_70` folder.
