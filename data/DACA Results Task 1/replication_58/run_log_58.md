# Run Log for DACA Replication Study (ID: 58)

## Project Overview
**Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

**Treatment Period**: DACA was implemented June 15, 2012. Examining effects in years 2013-2016.

---

## Session Log

### 2026-01-25: Initial Setup and Data Exploration

#### 1. Data Files Identified
- `data.csv` (6.26 GB): Main ACS data file with years 2006-2016
- `acs_data_dict.txt`: IPUMS data dictionary
- `state_demo_policy.csv`: Optional state-level data (not used in this analysis)

#### 2. Key Variables Identified from Data Dictionary

**For identifying Hispanic-Mexican, Mexican-born individuals:**
- `HISPAN`: Hispanic origin (1 = Mexican)
- `HISPAND`: Detailed Hispanic origin (100-107 = Mexican variants)
- `BPL`: Birthplace (200 = Mexico)
- `BPLD`: Detailed birthplace (20000 = Mexico)

**For DACA eligibility:**
- `BIRTHYR`: Birth year
- `BIRTHQTR`: Quarter of birth (1-4)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration

**For outcome variable (full-time employment):**
- `UHRSWORK`: Usual hours worked per week (35+ = full-time)
- `EMPSTAT`: Employment status (1 = Employed)

**Survey design variables:**
- `YEAR`: Census/survey year
- `PERWT`: Person weight
- `CLUSTER`: PSU cluster
- `STRATA`: Sampling strata

#### 3. DACA Eligibility Criteria (from instructions)
To be eligible for DACA, individuals must have:
1. Arrived in the US before their 16th birthday
2. Not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

**Operational definitions:**
- Arrived before age 16: `YEAR - YRIMMIG + 1` gives approximate years in US; combined with AGE to infer arrival age
- Under 31 as of June 15, 2012: `BIRTHYR >= 1982` (could be born in 1981 Q3-Q4 but approximating)
- Present since 2007: `YRIMMIG <= 2007`
- Non-citizen: `CITIZEN == 3`

#### 4. Methodological Approach: Difference-in-Differences
- **Treatment group**: Mexican-born, Hispanic-Mexican, non-citizen individuals who arrived before age 16, under 31 as of 2012, and in US since 2007
- **Control group**: Similar Mexican-born, Hispanic-Mexican individuals who do NOT meet one or more DACA criteria (e.g., arrived at age 16+, or too old)
- **Pre-period**: 2006-2011 (before DACA)
- **Post-period**: 2013-2016 (after DACA)
- **Excluded year**: 2012 (implementation year - cannot distinguish before/after)

---

### Commands Executed

```bash
# Check data file structure
head -5 data/data.csv
head -1 data/data.csv | tr ',' '\n'
```

---

## Key Decisions Made

1. **Exclude 2012 from analysis**: Per instructions, ACS does not list month of data collection, so we cannot distinguish before/after DACA implementation.

2. **Define full-time employment as UHRSWORK >= 35**: This matches the standard Bureau of Labor Statistics definition and the research question specification.

3. **Treatment vs Control distinction**: Using age at arrival threshold (under 16 vs 16+) as the main treatment/control separator within the eligible age cohort.

4. **Use person weights (PERWT)**: To ensure nationally representative estimates.

5. **Restrict sample to working-age population**: Focus on ages 18-35 to have relevant labor market participants who could plausibly be DACA-eligible.

---

## Analysis Pipeline

1. Load and filter data for Hispanic-Mexican, Mexican-born individuals
2. Create DACA eligibility indicator
3. Create full-time employment outcome
4. Implement difference-in-differences analysis
5. Generate summary statistics and regression results
6. Create visualizations
7. Compile LaTeX report

---

### 2026-01-25: Analysis Execution

#### Analysis Script Execution
```bash
cd "C:/Users/seraf/DACA Results Task 1/replication_58" && python analysis.py
```

#### Key Results

**Sample Construction:**
- Total Hispanic-Mexican, Mexico-born observations: 991,261
- After excluding 2012: 898,879
- Working-age sample (18-45): 519,609
- Final DiD sample: 140,565
  - Treatment group (DACA eligible): 69,244
  - Control group (arrived at 16+): 71,321

**DACA Eligibility Breakdown:**
- Arrived before age 16: 292,083
- Born 1982 or later: 238,235
- In US since 2007: 851,155
- Non-citizen: 636,722
- Fully DACA eligible: 118,852

**Main Results (Preferred Specification - DiD with Year and State Fixed Effects):**
- DiD Estimate: 0.0685 (6.85 percentage points)
- Standard Error: 0.0043
- 95% CI: [0.0602, 0.0769]
- p-value: <0.0001

**Interpretation:** DACA eligibility is associated with a 6.85 percentage point increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born non-citizens.

#### Robustness Checks
1. **By Gender:**
   - Male: 0.0684 (SE: 0.0072)
   - Female: 0.0657 (SE: 0.0061)

2. **Broader Age Range (18-50):** 0.0467 (SE: 0.0030)

3. **Any Employment (instead of full-time):** 0.0729 (SE: 0.0052)

#### Event Study Results (Reference: 2011)
Pre-treatment coefficients suggest some pre-trends in 2006-2007, but effects stabilize near zero in 2008-2010. Post-treatment effects increase steadily from 2013 to 2016.

#### Output Files Generated
- `summary_statistics.csv`
- `trends_data.csv`
- `event_study_coefs.csv`
- `regression_results.csv`
- `model_summary.txt`
- `figure1_trends.png/pdf`
- `figure2_event_study.png/pdf`
- `figure3_difference.png/pdf`

---

## Technical Notes

### Data Processing
- Used chunked reading (500,000 rows per chunk) to handle 6.26 GB file
- Applied memory-efficient data types (int8, int16, float32) where possible
- Filtered to Hispanic-Mexican (HISPAN=1) and Mexico-born (BPL=200) during chunk loading

### Statistical Methods
- Weighted least squares using person weights (PERWT)
- Clustered standard errors at the state level (STATEFIP)
- Year and state fixed effects in preferred specification

### Limitations
1. Cannot verify continuous residence requirement with precision
2. Age at arrival is approximated from survey year, current age, and immigration year
3. Cannot distinguish documented vs undocumented non-citizens
4. 2012 excluded entirely due to inability to determine before/after DACA timing

---

### 2026-01-25: Report Generation

#### LaTeX Report Compilation
```bash
cd "C:/Users/seraf/DACA Results Task 1/replication_58" && pdflatex -interaction=nonstopmode replication_report_58.tex
# Run 3 times for cross-references
```

#### Final Deliverables
All required output files successfully generated:
- `replication_report_58.tex` - LaTeX source (37,372 bytes)
- `replication_report_58.pdf` - Compiled report (358,438 bytes, 22 pages)
- `run_log_58.md` - This run log

---

## Summary of Final Results

| Metric | Value |
|--------|-------|
| **DiD Estimate** | 0.0685 (6.85 percentage points) |
| **Standard Error** | 0.0043 |
| **95% Confidence Interval** | [0.0602, 0.0769] |
| **p-value** | < 0.0001 |
| **Sample Size** | 140,565 |
| **Treatment Group N** | 69,244 |
| **Control Group N** | 71,321 |

**Conclusion:** DACA eligibility is associated with a statistically significant 6.85 percentage point increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born non-citizens, comparing those who meet all DACA eligibility criteria to a control group of similar immigrants who arrived at age 16 or older.

