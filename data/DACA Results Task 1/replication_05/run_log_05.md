# DACA Replication Study - Run Log

## Study Information
- **Research Question:** What is the causal effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born non-citizens?
- **Date:** January 2026
- **Replication ID:** 05

---

## 1. Data Preparation

### Data Files Used
- `data/data.csv` - Main ACS data file (33,851,424 observations)
- `data/acs_data_dict.txt` - Variable definitions and coding
- `data/state_demo_policy.csv` - Optional state-level data (not used in main analysis)

### Sample Construction Steps

| Step | Restriction | Observations |
|------|-------------|--------------|
| 1 | Raw ACS data (2006-2016) | 33,851,424 |
| 2 | Hispanic-Mexican (HISPAN=1) | 2,945,521 |
| 3 | Born in Mexico (BPL=200) | 991,261 |
| 4 | Non-citizen (CITIZEN=3) | 701,347 |
| 5 | Working age (16-64) | 618,640 |
| 6 | Exclude 2012 | 561,470 |

---

## 2. Key Decisions

### Decision 1: DACA Eligibility Definition
**Criteria Used:**
1. Arrived in US before 16th birthday: `YRIMMIG - BIRTHYR < 16`
2. Under 31 on June 15, 2012: `BIRTHYR >= 1981`
3. Present since June 15, 2007: `YRIMMIG <= 2007`

**Rationale:** These criteria follow the official DACA requirements as closely as possible given ACS data limitations. Education and criminal history requirements cannot be observed in the data.

### Decision 2: Treatment Period Definition
- **Pre-period:** 2006-2011
- **Post-period:** 2013-2016
- **Excluded:** 2012 (ambiguous - DACA announced mid-year)

**Rationale:** DACA was announced June 15, 2012 and applications began August 15, 2012. Since ACS does not report interview month, 2012 observations have ambiguous treatment status.

### Decision 3: Outcome Variable
- **Definition:** Full-time employment = UHRSWORK >= 35
- **Rationale:** Standard BLS definition of full-time work is 35+ hours per week.

### Decision 4: Sample Restrictions
- **Age range:** 16-64 (standard working age)
- **Non-citizens only:** Assumed undocumented per instructions
- **Mexican-born Hispanic-Mexican:** Target population for DACA

### Decision 5: Standard Errors
- **Clustering:** State level (STATEFIP)
- **Rationale:** Accounts for within-state correlation in labor market conditions and serial correlation over time.

### Decision 6: Estimation Method
- **Method:** Weighted Least Squares with person weights (PERWT)
- **Rationale:** ACS weights ensure representative estimates; WLS with clustering is standard for survey data.

---

## 3. Analysis Commands

### Python Script: `daca_analysis.py`

```python
# Main analysis script executed with:
python daca_analysis.py
```

### Key Operations:

1. **Load data:**
   ```python
   df = pd.read_csv("data/data.csv")
   ```

2. **Sample restrictions:**
   ```python
   df_sample = df[df['HISPAN'] == 1]
   df_sample = df_sample[df_sample['BPL'] == 200]
   df_sample = df_sample[df_sample['CITIZEN'] == 3]
   ```

3. **DACA eligibility:**
   ```python
   df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
   df_sample['arrived_before_16'] = df_sample['age_at_immig'] < 16
   df_sample['under_31_in_2012'] = df_sample['BIRTHYR'] >= 1981
   df_sample['present_since_2007'] = df_sample['YRIMMIG'] <= 2007
   df_sample['daca_eligible'] = (
       df_sample['arrived_before_16'] &
       df_sample['under_31_in_2012'] &
       df_sample['present_since_2007']
   )
   ```

4. **Outcome variable:**
   ```python
   df_sample['fulltime_employed'] = (df_sample['UHRSWORK'] >= 35).astype(int)
   ```

5. **DiD regression:**
   ```python
   model4 = smf.wls(
       'fulltime_employed ~ treatment + daca_x_post + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
       data=df_analysis_main,
       weights=df_analysis_main['PERWT']
   ).fit(cov_type='cluster', cov_kwds={'groups': df_analysis_main['STATEFIP']})
   ```

---

## 4. Results Summary

### Preferred Estimate (Model 4: Full DiD with state and year FE)

| Statistic | Value |
|-----------|-------|
| Effect size | 0.0305 (3.05 pp) |
| Standard error | 0.0041 |
| 95% CI | [0.0225, 0.0385] |
| P-value | < 0.001 |
| Sample size | 561,470 |

### All Model Specifications

| Model | Coefficient | Std. Error | Controls |
|-------|-------------|------------|----------|
| Model 1 (Basic DiD) | 0.0919 | 0.0039 | None |
| Model 2 (+Demographics) | 0.0379 | 0.0044 | Age, Sex, Marital |
| Model 3 (+Year FE) | 0.0311 | 0.0040 | + Year FE |
| Model 4 (Full) | 0.0305 | 0.0041 | + State FE |

### Robustness Checks

| Specification | Coefficient | SE | N |
|---------------|-------------|-----|------|
| Ages 18-45 | 0.0197 | 0.0035 | 413,906 |
| Males only | 0.0254 | 0.0055 | 303,717 |
| Females only | 0.0284 | 0.0064 | 257,753 |
| Strict eligibility | 0.0310 | 0.0043 | 561,470 |
| Include 2012 | 0.0310 | 0.0047 | 618,640 |

### Event Study Coefficients (relative to 2011)

| Year | Coefficient | SE |
|------|-------------|-----|
| 2006 | -0.0159 | 0.0135 |
| 2007 | -0.0106 | 0.0077 |
| 2008 | 0.0008 | 0.0120 |
| 2009 | 0.0050 | 0.0111 |
| 2010 | 0.0095 | 0.0160 |
| 2013 | 0.0129 | 0.0105 |
| 2014 | 0.0231 | 0.0150 |
| 2015 | 0.0405 | 0.0137 |
| 2016 | 0.0419 | 0.0122 |

### Placebo Test (fake treatment at 2009)

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.0141 |
| SE | 0.0040 |
| P-value | 0.0004 |

**Note:** The significant placebo coefficient suggests potential pre-trend violations.

---

## 5. Output Files

| File | Description |
|------|-------------|
| `replication_report_05.tex` | LaTeX source for replication report |
| `replication_report_05.pdf` | Compiled PDF report (21 pages) |
| `run_log_05.md` | This run log file |
| `daca_analysis.py` | Main Python analysis script |
| `analysis_results.txt` | Summary of main results |
| `stats_for_latex.json` | Statistics formatted for LaTeX |
| `descriptive_stats.csv` | Descriptive statistics by group |
| `pre_trends.csv` | Pre-period employment rates |
| `post_trends.csv` | Post-period employment rates |

---

## 6. LaTeX Compilation

```bash
# First pass
pdflatex -interaction=nonstopmode replication_report_05.tex

# Second pass (for cross-references)
pdflatex -interaction=nonstopmode replication_report_05.tex

# Third pass (final)
pdflatex -interaction=nonstopmode replication_report_05.tex
```

---

## 7. Interpretation

The main finding is that DACA eligibility is associated with a 3.05 percentage point increase in full-time employment probability. This represents approximately a 7% increase relative to the baseline employment rate of 43.6% among eligible individuals in the pre-period.

**Caveats:**
1. Parallel trends assumption may be violated (significant placebo test)
2. DACA eligibility is imperfectly measured (cannot observe education/criminal requirements)
3. Cannot distinguish documented from undocumented non-citizens

---

## 8. Session Information

- **Language:** Python 3.x
- **Key packages:** pandas, numpy, statsmodels
- **LaTeX compiler:** pdflatex (MiKTeX)
- **Platform:** Windows

---

*End of Run Log*
