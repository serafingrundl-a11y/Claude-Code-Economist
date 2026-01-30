# DACA Replication Study - Run Log 79

## Date
January 26, 2026

## Overview
This log documents the key decisions, commands, and analytical choices made during the replication study examining the effect of DACA eligibility on full-time employment.

---

## 1. Research Question

**Main Question:** Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (defined as usually working 35+ hours per week)?

**Research Design:** Difference-in-Differences
- **Treatment Group:** Ages 26-30 as of June 15, 2012 (DACA-eligible)
- **Control Group:** Ages 31-35 as of June 15, 2012 (would be eligible but for age)
- **Pre-period:** 2006-2011
- **Post-period:** 2013-2016 (2012 excluded due to mid-year implementation)

---

## 2. Data Source and Preparation

### Data Files Used
- `data/data.csv` - American Community Survey (ACS) data from IPUMS USA (2006-2016)
- `data/acs_data_dict.txt` - Variable codebook

### IPUMS Variables Used
| Variable | Description | Use in Analysis |
|----------|-------------|-----------------|
| YEAR | Survey year | Time period indicator |
| BIRTHYR | Birth year | Calculate age as of June 2012 |
| BIRTHQTR | Birth quarter (1-4) | Adjust age calculation |
| HISPAN | Hispanic origin | Filter for Mexican (HISPAN=1) |
| BPL | Birthplace | Filter for Mexico (BPL=200) |
| CITIZEN | Citizenship status | Filter for non-citizens (CITIZEN=3) |
| YRIMMIG | Year of immigration | Check residence since 2007 |
| UHRSWORK | Usual hours worked per week | Outcome variable (>=35 = full-time) |
| SEX | Sex | Covariate and heterogeneity |
| AGE | Age at survey | Covariate |
| EDUC | Education | Covariate |
| MARST | Marital status | Covariate |
| STATEFIP | State FIPS code | Fixed effects |
| PERWT | Person weight | Weighted estimates |

---

## 3. Sample Construction

### Sequential Filtering Steps

| Step | Criterion | Observations | Dropped |
|------|-----------|--------------|---------|
| 1 | Initial ACS sample (2006-2016) | 33,851,424 | - |
| 2 | Hispanic-Mexican (HISPAN=1) | 2,945,521 | 30,905,903 |
| 3 | Born in Mexico (BPL=200) | 991,261 | 1,954,260 |
| 4 | Non-citizen (CITIZEN=3) | 701,347 | 289,914 |
| 5 | Exclude 2012 | 636,722 | 64,625 |
| 6 | Arrived before age 16 | 186,357 | 450,365 |
| 7 | US resident since 2007 (YRIMMIG<=2007) | 177,294 | 9,063 |
| 8 | Ages 26-35 as of June 2012 | **43,238** | 134,056 |

### Final Sample
- **Total N:** 43,238
- **Treatment group (ages 26-30):** 25,470
- **Control group (ages 31-35):** 17,768
- **Pre-period (2006-2011):** 28,377
- **Post-period (2013-2016):** 14,861

---

## 4. Key Analytical Decisions

### 4.1 Age Calculation
Age as of June 15, 2012 was calculated as:
```
age_june2012 = 2012 - BIRTHYR
If BIRTHQTR > 2 (July-December): age_june2012 = age_june2012 - 1
```
**Rationale:** June 15 falls in Q2. People born in Q3/Q4 had not yet had their birthday by June 15, 2012.

### 4.2 DACA Eligibility Criteria Applied
1. Hispanic-Mexican ethnicity (HISPAN = 1)
2. Born in Mexico (BPL = 200)
3. Non-citizen (CITIZEN = 3) - assumed undocumented per instructions
4. Arrived before age 16: (YRIMMIG - BIRTHYR) < 16
5. Continuous US residence since 2007: YRIMMIG <= 2007
6. Age 26-35 as of June 15, 2012 (for treatment/control group definition)

### 4.3 Outcome Variable
- **Full-time employment:** UHRSWORK >= 35
- Binary indicator: 1 if working 35+ hours/week, 0 otherwise
- Overall full-time rate in sample: 62.69%

### 4.4 Treatment and Control Assignment
- **Treatment (treated=1):** age_june2012 between 26 and 30 (inclusive)
- **Control (treated=0):** age_june2012 between 31 and 35 (inclusive)

### 4.5 Pre/Post Period Definition
- **Pre (post=0):** YEAR in {2006, 2007, 2008, 2009, 2010, 2011}
- **Post (post=1):** YEAR in {2013, 2014, 2015, 2016}
- **Excluded:** 2012 (DACA implemented mid-year, cannot distinguish timing)

---

## 5. Model Specifications

### Model 1: Basic DiD
```
fulltime ~ treated + post + treated*post
```

### Model 2: DiD with Demographics
```
fulltime ~ treated + post + treated*post + female + age + married
```

### Model 3: DiD with Demographics and Education
```
fulltime ~ treated + post + treated*post + female + age + married + edu_hs + edu_some_college + edu_college_plus
```

### Model 4: DiD with Year Fixed Effects
```
fulltime ~ treated + treated*post + C(YEAR) + female + age + married + edu_hs + edu_some_college + edu_college_plus
```

### Model 5: DiD with Year and State Fixed Effects (PREFERRED)
```
fulltime ~ treated + treated*post + C(YEAR) + C(state) + female + age + married + edu_hs + edu_some_college + edu_college_plus
```

### Model 6: Weighted DiD with Year and State Fixed Effects
Same as Model 5, but using WLS with PERWT as weights.

---

## 6. Results Summary

### Main DiD Estimates

| Model | DiD Estimate | Std Error | p-value | 95% CI |
|-------|-------------|-----------|---------|--------|
| Basic DiD | 0.0516 | 0.0100 | <0.001 | [0.032, 0.071] |
| +Demographics | 0.0464 | 0.0093 | <0.001 | [0.028, 0.065] |
| +Education | 0.0450 | 0.0092 | <0.001 | [0.027, 0.063] |
| +Year FE | 0.0449 | 0.0092 | <0.001 | [0.027, 0.063] |
| **+Year+State FE** | **0.0441** | **0.0092** | **<0.001** | **[0.026, 0.062]** |
| Weighted | 0.0448 | 0.0107 | <0.001 | [0.024, 0.066] |

### Preferred Estimate
- **Effect:** 4.41 percentage points (0.0441)
- **Standard Error:** 0.0092
- **95% Confidence Interval:** [0.026, 0.062]
- **Interpretation:** DACA eligibility increased the probability of full-time employment by approximately 4.4 percentage points.

### Simple 2x2 DiD Calculation
```
Treatment Pre:  0.6147
Treatment Post: 0.6339
Treatment Diff: +0.0192

Control Pre:    0.6461
Control Post:   0.6136
Control Diff:   -0.0324

DiD = 0.0192 - (-0.0324) = 0.0516
```

---

## 7. Robustness Checks

### 7.1 Narrower Age Bands (27-29 vs 32-34)
- DiD Estimate: 0.0409 (SE: 0.0119, p=0.0006)
- Consistent with main findings

### 7.2 Alternative Full-Time Definition (40+ hours)
- DiD Estimate: 0.0535 (SE: 0.0094, p<0.001)
- Effect is actually larger with stricter definition

### 7.3 Pre-Trend Placebo Test (Fake Treatment in 2009)
- Placebo DiD: 0.0048 (SE: 0.0108, p=0.655)
- No significant pre-trend, supporting parallel trends assumption

---

## 8. Event Study Results

| Year | Coefficient | Std Error | p-value |
|------|-------------|-----------|---------|
| 2006 | -0.017 | 0.019 | 0.351 |
| 2007 | -0.033 | 0.019 | 0.083 |
| 2008 | 0.002 | 0.019 | 0.929 |
| 2009 | -0.017 | 0.020 | 0.384 |
| 2010 | -0.020 | 0.020 | 0.317 |
| 2011 | (Reference) | - | - |
| 2013 | 0.030 | 0.020 | 0.142 |
| 2014 | 0.027 | 0.020 | 0.182 |
| 2015 | 0.030 | 0.021 | 0.145 |
| 2016 | 0.041 | 0.021 | 0.046 |

**Key Finding:** Pre-period coefficients are small and statistically insignificant, supporting the parallel trends assumption. Post-period coefficients are uniformly positive and grow over time.

---

## 9. Heterogeneity Analysis

### By Gender
| Gender | DiD Estimate | Std Error | p-value |
|--------|-------------|-----------|---------|
| Male | 0.031 | 0.011 | 0.005 |
| Female | 0.049 | 0.015 | 0.001 |

### By Education
| Education | DiD Estimate | Std Error | p-value | N |
|-----------|-------------|-----------|---------|---|
| Less than HS | 0.017 | 0.014 | 0.233 | 18,057 |
| High School | 0.048 | 0.014 | 0.001 | 18,353 |
| Some College | 0.120 | 0.028 | <0.001 | 5,435 |
| College+ | 0.120 | 0.051 | 0.017 | 1,393 |

**Key Finding:** Effects are larger for women and for more educated individuals.

---

## 10. Files Generated

### Analysis Scripts
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script

### Output Files
- `summary_statistics.csv` - Descriptive statistics
- `regression_results.csv` - Main regression results
- `event_study_results.csv` - Event study coefficients
- `model_summaries.txt` - Full model output

### Figures
- `figure1_trends.pdf/png` - Employment trends by treatment status
- `figure2_event_study.pdf/png` - Event study plot
- `figure3_gender.pdf/png` - Heterogeneity by gender
- `figure4_did_illustration.pdf/png` - DiD graphical illustration
- `figure5_coef_plot.pdf/png` - Coefficient plot across specifications
- `figure6_age_distribution.pdf/png` - Sample distribution by age

### Final Deliverables
- `replication_report_79.tex` - LaTeX source
- `replication_report_79.pdf` - Final report (24 pages)
- `run_log_79.md` - This log file

---

## 11. Software and Commands

### Environment
- Python 3.x with pandas, numpy, statsmodels, matplotlib
- LaTeX (MiKTeX/pdflatex)

### Key Commands
```bash
# Run main analysis
python analysis.py

# Generate figures
python create_figures.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_79.tex
pdflatex -interaction=nonstopmode replication_report_79.tex  # Second pass for references
```

---

## 12. Conclusions

The analysis finds that DACA eligibility had a statistically significant positive effect on full-time employment. The preferred estimate suggests that DACA increased full-time employment by approximately 4.4 percentage points (95% CI: [0.026, 0.062]). This result is robust across multiple specifications and consistent with the parallel trends assumption as evidenced by the event study analysis and placebo tests.

The heterogeneity analysis reveals that effects are particularly pronounced for women and for individuals with higher levels of education, suggesting that DACA may have been especially important for enabling skilled workers to access formal employment opportunities.

---

*End of Run Log 79*
