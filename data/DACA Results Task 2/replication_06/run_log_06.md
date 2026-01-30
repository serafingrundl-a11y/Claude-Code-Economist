# DACA Replication Study - Run Log

## Study Information
- **Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals
- **Method**: Difference-in-Differences
- **Date Completed**: January 25, 2026

---

## 1. Data Exploration

### 1.1 Initial Data Review
**Files examined:**
- `data/data.csv` - Main dataset (6.3 GB, ~34 million records)
- `data/acs_data_dict.txt` - Data dictionary with variable definitions
- `data/state_demo_policy.csv` - State-level data (not used)

**Command:**
```bash
head -5 data/data.csv
```

**Key finding:** Dataset contains 54 variables including YEAR, HISPAN, BPL, CITIZEN, UHRSWORK, and demographic variables.

### 1.2 Data Dictionary Analysis
Reviewed IPUMS variable codes for:
- HISPAN: Hispanic origin (1 = Mexican)
- BPL: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status (1 = Employed)
- BIRTHYR: Birth year
- YEAR: Survey year (2006-2016)

---

## 2. Key Analytical Decisions

### 2.1 Sample Selection Criteria

| Criterion | IPUMS Variable | Value | Rationale |
|-----------|---------------|-------|-----------|
| Hispanic-Mexican | HISPAN | = 1 | Per instructions |
| Born in Mexico | BPL | = 200 | Per instructions |
| Non-citizen | CITIZEN | = 3 | Proxy for undocumented status |
| Treatment age | BIRTHYR | 1982-1986 | Ages 26-30 at DACA implementation |
| Control age | BIRTHYR | 1977-1981 | Ages 31-35 at DACA implementation |

### 2.2 Time Period Decisions
- **Pre-DACA period**: 2006-2011
- **Post-DACA period**: 2013-2016
- **Excluded**: 2012 (DACA implemented mid-year on June 15, 2012)

**Rationale:** Cannot distinguish pre/post within 2012 as ACS doesn't record interview month.

### 2.3 Outcome Variable Definition
- **Full-time employment**: UHRSWORK >= 35 hours/week
- Binary indicator (1 = full-time, 0 = not full-time)

### 2.4 Treatment Assignment
- Treatment (treat = 1): Age 26-30 at DACA implementation
- Control (treat = 0): Age 31-35 at DACA implementation
- Age calculated as: 2012 - BIRTHYR

### 2.5 Statistical Choices
- **Weights**: Person weights (PERWT) used for population-representative estimates
- **Standard errors**: Clustered at state level (STATEFIP)
- **Control variables**: Sex, marital status, education categories
- **Fixed effects**: State fixed effects in preferred specification

---

## 3. Analysis Commands

### 3.1 Python Analysis Script
Created `analysis.py` with the following workflow:

```python
# Data loading with chunked processing (due to large file size)
chunksize = 500000
for chunk in pd.read_csv('data/data.csv', chunksize=chunksize):
    # Filter each chunk
    chunk = chunk[(chunk['YEAR'] >= 2006) & (chunk['YEAR'] <= 2016)]
    chunk = chunk[chunk['HISPAN'] == 1]  # Hispanic-Mexican
    chunk = chunk[chunk['BPL'] == 200]   # Born in Mexico
    chunk = chunk[chunk['CITIZEN'] == 3] # Non-citizen
```

**Command to run analysis:**
```bash
python analysis.py
```

### 3.2 Model Specifications

**Model 1: Basic DD**
```python
fulltime ~ treat + post + treat_post
```

**Model 2: DD with Demographics**
```python
fulltime ~ treat + post + treat_post + female + married + C(EDUC)
```

**Model 3: DD with Demographics + State FE (Preferred)**
```python
fulltime ~ treat + post + treat_post + female + married + C(EDUC) + C(state)
```

**Model 4: Event Study**
```python
fulltime ~ treat + C(YEAR) + treat:C(YEAR) + female + married + C(EDUC) + C(state)
```

### 3.3 Figure Generation
Created `create_figures.py` to generate:
- `figure1_parallel_trends.png/pdf` - Time series of employment by group
- `figure2_event_study.png/pdf` - Event study coefficients
- `figure3_dd_visual.png/pdf` - DD visualization
- `figure4_robustness.png/pdf` - Robustness comparison

**Command:**
```bash
python create_figures.py
```

---

## 4. Results Summary

### 4.1 Sample Sizes
| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Control (31-35) | 54,133 | 32,837 | 86,970 |
| Treatment (26-30) | 46,371 | 28,942 | 75,313 |
| **Total** | **100,504** | **61,779** | **162,283** |

### 4.2 Full-Time Employment Rates
| Period | Control | Treatment |
|--------|---------|-----------|
| Pre-DACA | 60.99% | 61.32% |
| Post-DACA | 58.42% | 60.82% |

### 4.3 Main Results (Preferred Specification)
| Statistic | Value |
|-----------|-------|
| DD Estimate | 0.0236 |
| Standard Error | 0.0040 |
| 95% CI | [0.0157, 0.0315] |
| P-value | < 0.001 |
| Sample Size | 162,283 |

**Interpretation:** DACA eligibility increased the probability of full-time employment by 2.36 percentage points (95% CI: 1.57 to 3.15 pp).

### 4.4 Robustness Checks
| Specification | Estimate | SE | P-value |
|--------------|----------|-----|---------|
| Basic DD | 0.0308 | 0.0047 | < 0.001 |
| DD + Demographics | 0.0238 | 0.0040 | < 0.001 |
| DD + Demographics + State FE | 0.0236 | 0.0040 | < 0.001 |
| Employment (any) outcome | 0.0193 | 0.0041 | < 0.001 |
| Males only | 0.0319 | 0.0065 | < 0.001 |
| Females only | 0.0018 | 0.0059 | 0.760 |
| DACA-eligible proxy | 0.0484 | 0.0106 | < 0.001 |

---

## 5. Report Generation

### 5.1 LaTeX Compilation
**Commands:**
```bash
pdflatex -interaction=nonstopmode replication_report_06.tex
pdflatex -interaction=nonstopmode replication_report_06.tex  # Second pass for references
```

### 5.2 Output Files
- `replication_report_06.tex` - LaTeX source (22 pages)
- `replication_report_06.pdf` - Final PDF report

---

## 6. Files Produced

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `descriptive_stats.csv` | Summary statistics by group |
| `regression_results.csv` | Regression coefficients |
| `parallel_trends_data.csv` | Data for trends plot |
| `final_summary.csv` | Key results summary |
| `figure1_parallel_trends.png/pdf` | Parallel trends plot |
| `figure2_event_study.png/pdf` | Event study plot |
| `figure3_dd_visual.png/pdf` | DD visualization |
| `figure4_robustness.png/pdf` | Robustness comparison |
| `replication_report_06.tex` | LaTeX report source |
| `replication_report_06.pdf` | Final report |
| `run_log_06.md` | This log file |

---

## 7. Key Methodological Notes

### 7.1 Identification Strategy
The DD design exploits the age-based eligibility cutoff for DACA:
- Treatment: Those just young enough to qualify (ages 26-30 on June 15, 2012)
- Control: Those just too old to qualify (ages 31-35 on June 15, 2012)

Both groups share:
- Hispanic-Mexican ethnicity
- Born in Mexico
- Non-citizen status
- Similar labor market characteristics

### 7.2 Threats to Validity
1. **Non-citizen proxy**: CITIZEN=3 includes both documented and undocumented non-citizens
2. **Pre-trends**: Some marginally significant pre-2012 coefficients in event study
3. **Age-related confounds**: 5-10 year age difference could affect employment dynamics
4. **Selection into DACA**: Not all eligible individuals applied

### 7.3 Interpretation
The estimate of 2.36 percentage points represents:
- 3.9% increase relative to pre-DACA treatment mean (61.3%)
- Statistically significant at p < 0.001
- Robust to controls and fixed effects
- Concentrated among men (3.19 pp), not significant for women

---

## 8. Software and Environment

- **Python**: 3.x with pandas, numpy, statsmodels, matplotlib
- **LaTeX**: pdfTeX (MiKTeX distribution)
- **Platform**: Windows

---

## 9. Conclusion

This replication study finds that DACA eligibility had a statistically significant positive effect on full-time employment among Hispanic-Mexican, Mexican-born non-citizens. The preferred estimate of 2.36 percentage points (95% CI: 1.57-3.15) is robust across specifications and is primarily driven by effects among men.
