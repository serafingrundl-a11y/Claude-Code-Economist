# Run Log - Replication 77

## Project: DACA Impact on Full-Time Employment Replication Study

### Date Started: 2026-01-26

---

## 1. Data Understanding

### 1.1 Data Sources
- **Main data file**: `data/data.csv` (33,851,424 observations, ACS 2006-2016)
- **Data dictionary**: `data/acs_data_dict.txt`
- **Optional state-level data**: `data/state_demo_policy.csv`

### 1.2 Key Variables Identified
| Variable | Description | Usage |
|----------|-------------|-------|
| YEAR | Survey year | Time dimension |
| HISPAN/HISPAND | Hispanic origin | Sample selection (Mexican = 1) |
| BPL/BPLD | Birthplace | Sample selection (Mexico = 200) |
| CITIZEN | Citizenship status | DACA eligibility (non-citizen = 3) |
| YRIMMIG | Year of immigration | DACA eligibility (before age 16) |
| BIRTHYR | Birth year | Age calculation, DACA eligibility |
| BIRTHQTR | Birth quarter | Refine age calculations |
| UHRSWORK | Usual hours worked per week | Outcome (35+ = full-time) |
| EMPSTAT | Employment status | Employment filtering |
| PERWT | Person weight | Survey weights |
| AGE | Age at survey | Age group classification |

### 1.3 Research Design
- **Treatment group**: Ages 26-30 as of June 15, 2012 (born 1982-1986)
- **Control group**: Ages 31-35 as of June 15, 2012 (born 1977-1981)
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016 (excluding 2012 as implementation year)
- **Method**: Difference-in-differences

### 1.4 DACA Eligibility Criteria
1. Arrived unlawfully in the US before 16th birthday
2. Had not had 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status
5. Non-citizen (CITIZEN = 3 assumed undocumented)

---

## 2. Data Cleaning and Sample Selection

### 2.1 Sample Restrictions Applied
| Step | Restriction | Observations | % Retained |
|------|-------------|--------------|------------|
| 0 | Total ACS data | 33,851,424 | 100% |
| 1 | Hispanic-Mexican (HISPAN=1) | 2,945,521 | 8.7% |
| 2 | Born in Mexico (BPL=200) | 991,261 | 33.7% of step 1 |
| 3 | Non-citizen (CITIZEN=3) | 701,347 | 70.8% of step 2 |
| 4 | Arrived before age 16 | 205,327 | 29.3% of step 3 |
| 5 | Age 26-35 in 2012 | 49,019 | 23.9% of step 4 |
| 6 | Exclude 2012 | 44,725 | Final sample |

### 2.2 Key Decisions
- **Decision 1**: Use HISPAN = 1 for Mexican ethnicity (not detailed HISPAND)
- **Decision 2**: Use BPL = 200 for Mexico birthplace
- **Decision 3**: Assume CITIZEN = 3 (non-citizen) represents undocumented status
- **Decision 4**: Exclude 2012 from analysis due to mid-year implementation
- **Decision 5**: Define full-time employment as UHRSWORK >= 35
- **Decision 6**: Use survey weights (PERWT) for weighted estimation
- **Decision 7**: Cluster standard errors at state level (STATEFIP)

---

## 3. Analysis Steps

### 3.1 Step 1: Load and Filter Data
```python
df = pd.read_csv('data/data.csv')
# Apply sequential filters for sample selection
```

### 3.2 Step 2: Create Analysis Variables
- `treated`: 1 if born 1982-1986 (ages 26-30 in 2012)
- `post`: 1 if year >= 2013
- `treated_post`: interaction term (DiD coefficient)
- `fulltime`: 1 if UHRSWORK >= 35

### 3.3 Step 3: Difference-in-Differences Estimation
- Model specification: `fulltime ~ treated + post + treated_post`
- Weighted by PERWT
- Clustered SE at STATEFIP level
- Multiple specifications tested (basic, weighted, covariates, FE)

### 3.4 Step 4: Robustness Checks
- Employment outcome (EMPSTAT == 1)
- Continuous hours outcome (UHRSWORK)
- Heterogeneity by gender
- Pre-trends test (event study)
- Full event study (2006 as reference)

---

## 4. Commands Executed

```bash
# Run main analysis
cd "C:\Users\seraf\DACA Results Task 2\replication_77"
python analysis.py
```

### Python Script: analysis.py
- Loads data from data/data.csv
- Applies sample restrictions
- Creates analysis variables
- Runs 5 DiD model specifications
- Runs robustness checks
- Saves results to CSV files

---

## 5. Results Summary

### 5.1 Sample Characteristics
- N (observations): 44,725
- N (weighted): 6,205,755
- Full-time employment rate: 64.7%
- Employment rate: 70.8%
- Female: 43.1%
- Married: 45.5%
- Mean age: 28.4 years

### 5.2 Difference-in-Differences Summary
|  | Pre-DACA | Post-DACA | Difference |
|--|----------|-----------|------------|
| Treatment (26-30) | 0.6253 | 0.6580 | +0.0327 |
| Control (31-35) | 0.6705 | 0.6412 | -0.0293 |
| **DiD Estimate** | | | **+0.0620** |

### 5.3 Main Regression Results
| Model | Coefficient | SE | 95% CI | p-value |
|-------|-------------|-----|--------|---------|
| Basic DiD | 0.0551 | 0.0065 | [0.0425, 0.0678] | <0.001 |
| Weighted | 0.0620 | 0.0089 | [0.0445, 0.0795] | <0.001 |
| + Covariates | 0.0483 | 0.0109 | [0.0269, 0.0697] | <0.001 |
| + State/Year FE | 0.0596 | 0.0094 | [0.0412, 0.0780] | <0.001 |
| Full Model | 0.0465 | 0.0113 | [0.0243, 0.0688] | <0.001 |

### 5.4 Preferred Estimate
- **Effect size**: 0.0596 (5.96 percentage points)
- **Standard error**: 0.0094
- **95% CI**: [0.0412, 0.0780]
- **p-value**: <0.001
- **Sample size**: 44,725

### 5.5 Robustness Checks
- Employment effect: 0.0547 (SE: 0.0076, p<0.001)
- Hours worked effect: 2.42 hours (SE: 0.34, p<0.001)
- Male effect: 0.0590 (SE: 0.0127)
- Female effect: 0.0272 (SE: 0.0181)

### 5.6 Pre-trends Test
All pre-trend coefficients insignificant (p > 0.20), supporting parallel trends assumption.

### 5.7 Event Study Results
| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2007 | -0.0069 | 0.0201 | 0.732 |
| 2008 | 0.0215 | 0.0189 | 0.254 |
| 2009 | 0.0234 | 0.0202 | 0.246 |
| 2010 | 0.0197 | 0.0198 | 0.319 |
| 2011 | 0.0045 | 0.0206 | 0.828 |
| 2013 | 0.0639* | 0.0265 | 0.016 |
| 2014 | 0.0710* | 0.0209 | 0.001 |
| 2015 | 0.0455* | 0.0197 | 0.021 |
| 2016 | 0.0989* | 0.0195 | <0.001 |

---

## 6. Notes and Issues

### 6.1 Methodological Notes
- Cannot distinguish documented from undocumented immigrants in ACS data
- Assumed all non-citizens (CITIZEN=3) are potentially undocumented
- 2012 excluded due to mid-year DACA implementation (June 15, 2012)
- Linear probability model used (standard in DiD literature)

### 6.2 Limitations
- ACS is repeated cross-section, not panel data
- Cannot observe same individuals before and after
- Age eligibility cutoff may have measurement error
- Continuous residence requirement cannot be verified in data

### 6.3 Files Generated
- `analysis.py`: Main analysis script
- `results_main.csv`: Main regression results
- `results_event_study.csv`: Event study coefficients
- `results_descriptive.csv`: Descriptive statistics
- `replication_report_77.tex`: LaTeX report
- `replication_report_77.pdf`: Final PDF report
