# DACA Replication Study - Run Log (ID: 85)

## Date: 2026-01-25

---

## 1. Project Overview

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

**Data Source:** American Community Survey (ACS) 2006-2016, provided via IPUMS

---

## 2. Data Files Used

| File | Description | Size |
|------|-------------|------|
| `data/data.csv` | Main ACS data file | 6.26 GB, ~33.8 million rows |
| `data/acs_data_dict.txt` | Variable codebook | 121 KB |
| `data/state_demo_policy.csv` | State-level policy data (not used) | 37 KB |

---

## 3. Key Variables from ACS

| Variable | Description | Used For |
|----------|-------------|----------|
| YEAR | Survey year | Time period |
| HISPAN | Hispanic origin (1=Mexican) | Target population |
| BPL | Birthplace (200=Mexico) | Target population |
| CITIZEN | Citizenship (3=Non-citizen) | DACA eligibility |
| YRIMMIG | Year of immigration | DACA eligibility |
| BIRTHYR | Birth year | DACA eligibility |
| AGE | Current age | Sample restriction, controls |
| UHRSWORK | Usual hours worked per week | Outcome |
| EMPSTAT | Employment status | Outcome |
| SEX | Gender | Control variable |
| MARST | Marital status | Control variable |
| EDUC | Educational attainment | Control variable |
| STATEFIP | State FIPS code | Fixed effects |
| PERWT | Person weight | Weighted analysis |

---

## 4. DACA Eligibility Criteria Implementation

Per the DACA program requirements (as of June 15, 2012):

1. **Arrived before 16th birthday:** `age_at_immigration < 16`
   - Calculated as: `YRIMMIG - BIRTHYR`

2. **Under 31 as of June 15, 2012:** `BIRTHYR >= 1982`
   - Conservative implementation using birth year only

3. **Continuous residence since June 15, 2007:** `YRIMMIG <= 2007`
   - Must have arrived in US by 2007

4. **Not a citizen (assumed undocumented):** `CITIZEN == 3`
   - Cannot distinguish documented from undocumented; assuming non-citizens are potentially undocumented

---

## 5. Sample Construction

### Step-by-step filtering:

1. **Full dataset:** 33,851,425 observations
2. **After filtering to Mexican-born Hispanic-Mexican (HISPAN=1, BPL=200):** 991,261 observations
3. **After restricting to working age (16-40):** 473,012 observations
4. **After excluding 2012 (transition year):** 431,062 observations
5. **After restricting to non-citizens:** 355,188 observations (final sample)

### Final sample composition:
- Treatment group (DACA-eligible): 80,300 observations
- Control group (not DACA-eligible): 274,888 observations

---

## 6. Identification Strategy

**Method:** Difference-in-Differences (DiD)

**Treatment:** DACA eligibility (binary indicator based on eligibility criteria)

**Control Group:** Mexican-born Hispanic non-citizens who do not meet DACA eligibility criteria (primarily due to: arriving after age 16, or being too old in 2012, or arriving after 2007)

**Pre-period:** 2006-2011 (before DACA implementation)

**Post-period:** 2013-2016 (after DACA implementation; 2012 excluded as transition year)

**Outcome:** Full-time employment = 1 if employed AND usually works 35+ hours per week

### Key Assumption:
Parallel trends assumption - absent DACA, the full-time employment trends for DACA-eligible and non-eligible groups would have been similar.

---

## 7. Model Specifications

### Model 1: Basic DiD
```
Y = β₀ + β₁(Treated) + β₂(Post) + β₃(Treated × Post) + ε
```

### Model 2: DiD with Demographics
```
Y = β₀ + β₁(Treated) + β₂(Post) + β₃(Treated × Post) +
    β₄(Age) + β₅(Age²) + β₆(Male) + β₇(Married) +
    γ(Education Dummies) + ε
```

### Model 3: DiD with State and Year Fixed Effects (PREFERRED)
```
Y = β₀ + β₁(Treated) + β₃(Treated × Post) +
    β₄(Age) + β₅(Age²) + β₆(Male) + β₇(Married) +
    γ(Education Dummies) + δ(State FE) + θ(Year FE) + ε
```

### Model 4: Weighted Analysis
Same as Model 3 but using person weights (PERWT) via weighted least squares.

All standard errors are heteroskedasticity-robust (HC1).

---

## 8. Main Results

### Descriptive Statistics - Full-Time Employment Rates:

| Group | Pre-DACA (2006-2011) | Post-DACA (2013-2016) | Change |
|-------|---------------------|----------------------|--------|
| DACA-Eligible | 36.67% | 44.99% | +8.32 pp |
| Control | 55.53% | 54.58% | -0.95 pp |

**Simple DiD:** 8.32 - (-0.95) = 9.27 percentage points

### Regression Results:

| Model | DiD Estimate | Std. Error | 95% CI | p-value |
|-------|-------------|------------|--------|---------|
| Basic DiD | 0.0927 | 0.0040 | [0.085, 0.101] | <0.001 |
| + Demographics | 0.0185 | 0.0037 | [0.011, 0.026] | <0.001 |
| + State & Year FE | 0.0094 | 0.0037 | [0.002, 0.017] | 0.012 |
| Weighted with FE | 0.0059 | 0.0045 | [-0.003, 0.015] | 0.192 |

---

## 9. Preferred Estimate

**Model 3 (DiD with State and Year Fixed Effects):**

- **Effect:** 0.94 percentage points (0.0094)
- **Standard Error:** 0.0037
- **95% CI:** [0.21, 1.67 percentage points]
- **p-value:** 0.012
- **Sample Size:** 355,188

**Interpretation:** DACA eligibility is associated with a statistically significant 0.94 percentage point increase in the probability of full-time employment among Mexican-born Hispanic non-citizens.

---

## 10. Robustness Checks

### 10.1 Placebo Test (Fake treatment in 2010)
- Coefficient: 0.0061
- SE: 0.0049
- p-value: 0.216
- **Result:** No significant pre-trend, supporting parallel trends assumption

### 10.2 Alternative Age Restrictions (18-35)
- Coefficient: 0.0230
- SE: 0.0043
- p-value: <0.001
- N: 253,373
- **Result:** Larger effect with tighter age restrictions

### 10.3 Heterogeneity by Gender
- Male: 0.0035 (SE=0.0050), not significant
- Female: 0.0307 (SE=0.0053), highly significant
- **Result:** Effects driven primarily by women

### 10.4 Event Study
| Year | Coefficient | SE | Significant? |
|------|-------------|-----|--------------|
| 2006 | 0.001 | 0.008 | No |
| 2007 | 0.003 | 0.008 | No |
| 2008 | 0.013 | 0.008 | No |
| 2009 | 0.015 | 0.008 | Marginal |
| 2010 | 0.017 | 0.008 | Yes* |
| **2011** | **ref** | - | - |
| 2013 | 0.009 | 0.008 | No |
| 2014 | 0.011 | 0.008 | No |
| 2015 | 0.024 | 0.008 | Yes** |
| 2016 | 0.026 | 0.008 | Yes** |

**Result:** Effects grow over time post-DACA, consistent with gradual program take-up. Some pre-trends present (2010 marginally significant), which warrants caution in interpretation.

---

## 11. Key Analytical Decisions

1. **Control Group Definition:** Used Mexican-born Hispanic non-citizens who failed DACA criteria (arrived after 16, too old, or arrived after 2007) rather than citizens or different ethnicity.

2. **Age Restriction:** Limited to ages 16-40 to maintain reasonable overlap between treatment and control groups and focus on prime working age.

3. **Exclusion of 2012:** Dropped transition year since DACA was announced mid-year (June 15, 2012) and applications began August 2012.

4. **Birth Year Cutoff:** Used BIRTHYR >= 1982 for age requirement (conservative approach since exact birth month unknown).

5. **Full-time Definition:** 35+ usual hours worked per week, consistent with standard full-time employment definition.

6. **Standard Errors:** Used heteroskedasticity-robust (HC1) standard errors throughout.

7. **Preferred Specification:** Model 3 with state and year fixed effects to control for location-specific and time-varying factors.

---

## 12. Limitations

1. Cannot distinguish documented from undocumented non-citizens in the data.
2. Cannot verify continuous presence in the US (required for DACA).
3. No exact month of immigration, so age at arrival is approximate.
4. No exact birth month, so age cutoffs are approximate.
5. Some evidence of pre-trends in event study (2010) suggests caution in causal interpretation.
6. Repeated cross-section design (not panel) prevents individual-level tracking.

---

## 13. Files Produced

| File | Description |
|------|-------------|
| `analysis.py` | Main Python analysis script |
| `results.json` | JSON file with key results |
| `replication_report_85.tex` | LaTeX report |
| `replication_report_85.pdf` | Compiled PDF report |
| `run_log_85.md` | This run log |

---

## 14. Software Environment

- Python 3.x
- pandas
- numpy
- statsmodels
- scipy

---

## 15. Commands Executed

```bash
# Read first rows of data
head -5 data/data.csv

# Count data rows
wc -l data/data.csv

# Run analysis
python analysis.py

# Compile LaTeX report
pdflatex replication_report_85.tex
```

---

## End of Run Log
