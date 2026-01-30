# Replication Run Log - DACA Full-Time Employment Analysis

## Project Overview
**Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

**Analysis Period**: Effects examined for 2013-2016 (post-DACA implementation on June 15, 2012)

---

## Step 1: Data Exploration and Understanding

### Data Files
- `data.csv`: Main ACS data file (~33.8 million observations, 2006-2016)
- `acs_data_dict.txt`: Variable documentation from IPUMS
- `state_demo_policy.csv`: State-level policy data (optional, not used in main analysis)

### Key Variables Identified
| Variable | Description | Values/Notes |
|----------|-------------|--------------|
| YEAR | Survey year | 2006-2016 |
| HISPAN | Hispanic origin | 1 = Mexican |
| HISPAND | Hispanic origin (detailed) | 100-107 = Mexican varieties |
| BPL | Birthplace | 200 = Mexico |
| BPLD | Birthplace (detailed) | 20000 = Mexico |
| CITIZEN | Citizenship status | 3 = Not a citizen |
| YRIMMIG | Year of immigration | Year value |
| BIRTHYR | Birth year | Year value |
| BIRTHQTR | Birth quarter | 1-4 |
| AGE | Age | Integer |
| UHRSWORK | Usual hours worked per week | 0-99 |
| EMPSTAT | Employment status | 1 = Employed |
| PERWT | Person weight | For weighted analysis |

---

## Step 2: DACA Eligibility Criteria Definition

Based on instructions, DACA eligibility requires:
1. **Arrived before 16th birthday**: (YRIMMIG - BIRTHYR) < 16
2. **Under 31 as of June 15, 2012**: Born after June 15, 1981
   - Approximation: BIRTHYR >= 1982 (conservative) or use BIRTHQTR for finer control
3. **In US since June 15, 2007**: YRIMMIG <= 2007
4. **Not a citizen**: CITIZEN == 3

### Sample Restrictions
- Hispanic-Mexican ethnicity: HISPAN == 1 (or HISPAND in 100-107)
- Born in Mexico: BPL == 200 (or BPLD == 20000)
- Non-citizen: CITIZEN == 3 (assume undocumented)

---

## Step 3: Identification Strategy

### Difference-in-Differences Design
- **Treatment Group**: DACA-eligible non-citizen Mexican-born Hispanic individuals
- **Control Group**: Non-eligible non-citizen Mexican-born Hispanic individuals (similar population but not meeting age/arrival criteria)
- **Pre-Period**: 2006-2011 (before DACA)
- **Post-Period**: 2013-2016 (after DACA implementation)
- **Note**: 2012 excluded due to mid-year DACA implementation

### Outcome Variable
- Full-time employment: UHRSWORK >= 35 (binary indicator)

---

## Step 4: Commands Executed

### Python Analysis Script
```bash
python daca_analysis.py
```

---

## Key Decisions Log

1. **Sample Definition**: Restricted to Hispanic-Mexican (HISPAN==1) born in Mexico (BPL==200) who are non-citizens (CITIZEN==3)

2. **Age Restriction for Working Population**: Limited to ages 18-64 for labor force analysis

3. **DACA Eligibility**:
   - Arrived before age 16: YRIMMIG - BIRTHYR < 16
   - Under 31 on June 15, 2012: BIRTHYR >= 1982
   - In US since 2007: YRIMMIG <= 2007
   - Non-citizen: CITIZEN == 3

4. **Control Group Definition**: Non-citizen Mexican-born individuals who do not meet DACA eligibility criteria (older arrivals or arrived after 2007)

5. **Exclusion of 2012**: Due to mid-year implementation, 2012 is excluded to avoid contamination

6. **Weighting**: Using PERWT for nationally representative estimates

7. **Standard Errors**: Clustered at state level (STATEFIP) to account for within-state correlation

---

## Step 5: Analysis Output

### Sample Sizes
- Total ACS records: ~33.8 million (2006-2016)
- After filtering to Hispanic-Mexican born in Mexico, non-citizens: 701,347
- After age restriction (18-64): 603,425
- After excluding 2012: 547,614
- After requiring valid YRIMMIG: 547,614 (final sample)

### Treatment/Control Groups
- DACA Eligible: 71,347 (13.0%)
- Not Eligible: 476,267 (87.0%)

### Outcome Summary (Weighted Full-Time Employment Rates)
| Group | Pre-DACA | Post-DACA |
|-------|----------|-----------|
| Not Eligible | 62.83% | 60.37% |
| DACA Eligible | 52.53% | 56.95% |

**Simple DiD Calculation**: (56.95 - 52.53) - (60.37 - 62.83) = 4.42 + 2.46 = 6.88 pp

---

## Step 6: Main Regression Results

### Model 1: Simple DiD (No Controls)
| Variable | Coefficient | Std. Error | p-value |
|----------|-------------|------------|---------|
| DACA Eligible x Post | 0.0688 | 0.0035 | <0.001 |
| DACA Eligible | -0.1030 | 0.0039 | <0.001 |
| Post | -0.0247 | 0.0026 | <0.001 |

### Model 2: DiD with Demographic Controls (PREFERRED)
| Variable | Coefficient | Std. Error | p-value |
|----------|-------------|------------|---------|
| **DACA Eligible x Post** | **0.0266** | **0.0040** | **<0.001** |
| DACA Eligible | -0.0270 | 0.0042 | <0.001 |
| Post | -0.0151 | 0.0022 | <0.001 |
| Age | 0.0333 | 0.0012 | <0.001 |
| Age² | -0.0004 | 0.0000 | <0.001 |
| Female | -0.4377 | 0.0154 | <0.001 |
| Married | -0.0364 | 0.0057 | <0.001 |
| High School | 0.0356 | 0.0028 | <0.001 |
| Some College | 0.0310 | 0.0058 | <0.001 |
| College+ | 0.0652 | 0.0042 | <0.001 |

### Model 3: DiD with Year and State Fixed Effects
| Variable | Coefficient | Std. Error | p-value |
|----------|-------------|------------|---------|
| DACA Eligible x Post | 0.0173 | 0.0036 | <0.001 |

---

## Step 7: Robustness Checks

| Specification | Coefficient | Std. Error | N |
|---------------|-------------|------------|---|
| Main (ages 18-64) | 0.0266 | 0.0040 | 547,614 |
| Ages 18-35 only | 0.0261 | 0.0055 | 253,373 |
| Males only | 0.0214 | 0.0057 | 296,109 |
| Females only | 0.0236 | 0.0065 | 251,505 |

---

## Step 8: Event Study (Pre-Trends Check)

| Year | Coefficient | 95% CI | Significant? |
|------|-------------|--------|--------------|
| 2006 | 0.0141 | [-0.015, 0.043] | No |
| 2007 | 0.0060 | [-0.007, 0.019] | No |
| 2008 | 0.0175 | [-0.010, 0.045] | No |
| 2009 | 0.0171 | [-0.007, 0.041] | No |
| 2010 | 0.0136 | [-0.018, 0.045] | No |
| 2011 | 0 (reference) | -- | -- |
| 2013 | 0.0118 | [-0.010, 0.033] | No |
| 2014 | 0.0244 | [-0.002, 0.051] | * |
| 2015 | 0.0401 | [0.014, 0.067] | *** |
| 2016 | 0.0401 | [0.018, 0.063] | *** |

Pre-trends: No significant pre-treatment differences, supporting parallel trends assumption.

---

## FINAL RESULTS

### Preferred Estimate
- **Effect Size**: 0.0266 (2.66 percentage points)
- **Standard Error**: 0.0040 (clustered at state level)
- **95% Confidence Interval**: [0.0187, 0.0344]
- **p-value**: <0.001
- **Sample Size**: 547,614

### Interpretation
DACA eligibility is associated with a **2.66 percentage point increase** in the probability of full-time employment among Hispanic-Mexican individuals born in Mexico. This effect is statistically significant at the 1% level.

---

## Deliverables Created

1. `daca_analysis.py` - Main analysis script
2. `summary_statistics.csv` - Summary statistics by group
3. `regression_results.csv` - Main regression results
4. `event_study_results.csv` - Event study coefficients
5. `replication_report_23.tex` - LaTeX source for report
6. `replication_report_23.pdf` - Final PDF report (18 pages)
7. `run_log_23.md` - This log file

---

## Commands Log

```bash
# Data exploration
head -3 data/data.csv
wc -l data/data.csv

# Run analysis
python daca_analysis.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_23.tex
pdflatex -interaction=nonstopmode replication_report_23.tex
pdflatex -interaction=nonstopmode replication_report_23.tex
```

