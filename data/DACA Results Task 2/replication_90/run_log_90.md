# Run Log - DACA Replication Study (ID: 90)

## Date: 2026-01-26

---

## 1. Initial Setup and Data Exploration

### 1.1 Understanding the Task
- **Research Question**: What was the causal impact of DACA eligibility on the probability of full-time employment (35+ hours/week) among Hispanic-Mexican, Mexican-born individuals in the US?
- **Treatment Group**: Ages 26-30 as of June 15, 2012 (DACA implementation date)
- **Control Group**: Ages 31-35 as of June 15, 2012 (ineligible due to age cutoff)
- **Method**: Difference-in-Differences (DiD)
- **Pre-treatment Period**: 2006-2011
- **Post-treatment Period**: 2013-2016 (2012 excluded due to ambiguity)

### 1.2 DACA Eligibility Criteria
1. Arrived unlawfully in the US before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Present in the US on June 15, 2012 and did not have lawful status

### 1.3 Data Files
- `data.csv`: Main ACS data file (~6GB)
- `acs_data_dict.txt`: Data dictionary
- `state_demo_policy.csv`: Optional state-level supplemental data
- Data covers ACS years 2006-2016 (1-year files)

### 1.4 Key Variables from Data Dictionary
| Variable | Description |
|----------|-------------|
| YEAR | Census year |
| HISPAN | Hispanic origin (1=Mexican) |
| BPL | Birthplace (200=Mexico) |
| CITIZEN | Citizenship status (3=Not a citizen) |
| YRIMMIG | Year of immigration |
| BIRTHYR | Birth year |
| BIRTHQTR | Quarter of birth |
| UHRSWORK | Usual hours worked per week |
| EMPSTAT | Employment status |
| PERWT | Person weight |
| AGE | Age |
| SEX | Sex (1=Male, 2=Female) |
| EDUC | Education level |
| MARST | Marital status |
| STATEFIP | State FIPS code |

---

## 2. Sample Construction

### 2.1 Filter Criteria (Applied Sequentially)
1. **Ethnicity**: HISPAN == 1 (Mexican) → 991,261 observations
2. **Birthplace**: BPL == 200 (Mexico) → Already satisfied
3. **Citizenship**: CITIZEN == 3 (Not a citizen) → 701,347 observations
4. **Immigration timing**: Arrived before age 16 → 201,531 observations
5. **Continuous residence**: YRIMMIG <= 2007 → 191,374 observations
6. **Age restriction**: Ages 26-35 as of June 2012 → 46,817 observations
7. **Exclude 2012**: Remove ambiguous year → 42,689 observations (FINAL)

### 2.2 Age Group Definition
- DACA was implemented on June 15, 2012
- Age calculation uses BIRTHYR and BIRTHQTR to determine exact age
- If born Q1-Q2 (Jan-Jun): age = 2012 - BIRTHYR
- If born Q3-Q4 (Jul-Dec): age = 2012 - BIRTHYR - 1
- Treatment group: ages 26-30 as of June 15, 2012
- Control group: ages 31-35 as of June 15, 2012

### 2.3 Weighted Population
- Pre-DACA treatment group: ~2,256,000
- Pre-DACA control group: ~1,609,000
- Post-DACA treatment group: ~1,231,000
- Post-DACA control group: ~832,000
- Total weighted: ~5,928,000 person-years

---

## 3. Analysis Decisions

### 3.1 Outcome Variable
- **Full-time employment**: UHRSWORK >= 35 (binary indicator)
- Secondary: Any employment (EMPSTAT == 1)

### 3.2 Model Specifications
1. **Model 1**: Basic DiD (no controls)
2. **Model 2**: DiD + demographic controls (sex, marital status, education, age)
3. **Model 3**: Model 2 + year fixed effects
4. **Model 4 (Preferred)**: Model 3 + state fixed effects

### 3.3 Standard Errors
- Clustered by state (STATEFIP) to account for within-state correlation
- 51 clusters (50 states + DC)

---

## 4. Commands Executed

### 4.1 Data Loading and Processing
```python
# Load data in chunks, filtering to Hispanic-Mexican Mexican-born
usecols = ['YEAR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'BIRTHYR', 'BIRTHQTR',
           'UHRSWORK', 'EMPSTAT', 'PERWT', 'AGE', 'SEX', 'EDUC', 'MARST',
           'STATEFIP', 'LABFORCE']

for chunk in pd.read_csv(data_path, usecols=usecols, dtype=dtypes, chunksize=500000):
    chunk_filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk_filtered)
```

### 4.2 Sample Construction
```python
# Non-citizens
df = df[df['CITIZEN'] == 3]

# Age at immigration < 16
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[(df['age_at_immigration'] >= 0) & (df['age_at_immigration'] < 16)]

# Continuous residence
df = df[df['YRIMMIG'] <= 2007]

# Age as of June 2012
df['age_june2012'] = 2012 - df['BIRTHYR']
df.loc[df['BIRTHQTR'] >= 3, 'age_june2012'] -= 1

# Treatment/control groups
df['treat'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)
df = df[(df['age_june2012'] >= 26) & (df['age_june2012'] <= 35)]

# Exclude 2012
df = df[df['YEAR'] != 2012]

# Post indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)
```

### 4.3 Regression Analysis
```python
# Preferred specification (Model 4)
model4 = smf.wls('fulltime ~ treat + treat_post + female + married + educ_hs +
                  educ_college + AGE + age_sq + C(year) + C(state)',
                  data=df, weights=df['PERWT']).fit(
                      cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```

### 4.4 Figure Generation
```python
python generate_figures.py
# Created: figure1_trends.png/pdf, figure2_eventstudy.png/pdf,
#          figure3_did.png/pdf, figure4_sample.png/pdf
```

### 4.5 LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_90.tex
pdflatex -interaction=nonstopmode replication_report_90.tex
pdflatex -interaction=nonstopmode replication_report_90.tex
```

---

## 5. Results Summary

### 5.1 Main Finding
| Specification | Estimate | SE | 95% CI |
|--------------|----------|-----|--------|
| Basic DiD | 0.0577 | 0.007 | [0.044, 0.071] |
| + Controls | 0.0377 | 0.009 | [0.020, 0.055] |
| + Year FE | 0.0366 | 0.009 | [0.019, 0.055] |
| + State FE (Preferred) | 0.0356 | 0.010 | [0.017, 0.055] |

### 5.2 Preferred Estimate
- **Effect**: 0.0356 (3.56 percentage points)
- **Standard Error**: 0.0097
- **t-statistic**: 3.69
- **p-value**: 0.0002
- **95% CI**: [0.0167, 0.0545]
- **Sample Size**: 42,689

### 5.3 Event Study Coefficients (relative to 2011)
| Year | Coefficient | SE |
|------|-------------|-----|
| 2006 | 0.007 | 0.029 |
| 2007 | -0.023 | 0.016 |
| 2008 | 0.017 | 0.021 |
| 2009 | 0.002 | 0.024 |
| 2010 | -0.003 | 0.024 |
| 2013 | 0.034 | 0.023 |
| 2014 | 0.032 | 0.018 |
| 2015 | 0.016 | 0.020 |
| 2016 | 0.061 | 0.021 |

### 5.4 Robustness Checks
- **Any employment outcome**: 0.0377 (SE: 0.007)
- **Males only**: 0.0242 (SE: 0.010)
- **Females only**: 0.0367 (SE: 0.017)
- **Pre-trend test (treat*year)**: 0.0004 (SE: 0.004, p=0.93) - No differential pre-trend

---

## 6. Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| Exclude 2012 | Cannot distinguish pre/post DACA within 2012; mid-year implementation |
| CITIZEN==3 as undocumented proxy | ACS doesn't identify undocumented; non-citizen is best available proxy |
| Age calculated from BIRTHYR+BIRTHQTR | More precise than AGE for determining DACA age eligibility at specific date |
| Cluster SE by state | Standard practice for policy evaluation; accounts for within-state correlation |
| YRIMMIG <= 2007 for continuous residence | Satisfies "continuously since June 15, 2007" requirement |
| age_at_immigration < 16 | Satisfies "arrived before 16th birthday" requirement |
| Model 4 as preferred | Most rigorous specification with year and state FE; accounts for confounders |
| Weighted regression | PERWT accounts for complex survey design; nationally representative |

---

## 7. Files Generated

| File | Description |
|------|-------------|
| analysis.py | Main analysis script |
| generate_figures.py | Figure generation script |
| results_summary.txt | Key results in text format |
| fulltime_rates_by_year.csv | Employment rates by group/year |
| event_study_coefs.csv | Event study coefficients |
| figure1_trends.png/pdf | Employment trends over time |
| figure2_eventstudy.png/pdf | Event study plot |
| figure3_did.png/pdf | DiD illustration |
| figure4_sample.png/pdf | Sample composition |
| replication_report_90.tex | LaTeX source |
| replication_report_90.pdf | Final report (20 pages) |
| run_log_90.md | This log file |

---

## 8. Interpretation

DACA eligibility increased the probability of full-time employment by approximately 3.6 percentage points among Mexican-born Hispanic non-citizens who otherwise met DACA eligibility criteria. This represents a 5.7% increase relative to the pre-DACA baseline for the treatment group (0.0356/0.631 = 0.056).

The event study shows:
1. No significant differential pre-trends (parallel trends assumption supported)
2. Effects emerge in 2013-2014 (immediate post-DACA period)
3. Largest effect in 2016 (6.1 pp), suggesting cumulative benefits

The effect is statistically significant (p=0.0002) and robust across specifications.

---

*Log completed: 2026-01-26*
