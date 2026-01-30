# Run Log - DACA Replication Study (ID: 61)

## Overview
This log documents all commands, key decisions, and analytical choices made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on full-time employment (usually working 35+ hours per week)?

## Research Design
- **Design**: Difference-in-Differences (DiD)
- **Treatment Group**: Ages 26-30 as of June 15, 2012 (birth years 1982-1986)
- **Control Group**: Ages 31-35 as of June 15, 2012 (birth years 1977-1981)
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016
- **Note**: 2012 excluded due to DACA implementation mid-year

---

## Step 1: Data Exploration and Understanding

### 1.1 Data Files Examined
- `data.csv`: Main ACS data file (33,851,424 observations)
- `acs_data_dict.txt`: Data dictionary with variable definitions
- `state_demo_policy.csv`: Optional state-level data (not used in main analysis)

### 1.2 Key Variables Identified
From the data dictionary:

| Variable | Description | Key Values |
|----------|-------------|------------|
| YEAR | Survey year | 2006-2016 |
| BIRTHYR | Birth year | Numeric |
| BIRTHQTR | Birth quarter | 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec |
| HISPAN | Hispanic origin | 1=Mexican |
| BPL | Birthplace | 200=Mexico |
| CITIZEN | Citizenship status | 3=Not a citizen |
| YRIMMIG | Year of immigration | Year values |
| UHRSWORK | Usual hours worked per week | 0-99 |
| EMPSTAT | Employment status | 1=Employed, 2=Unemployed, 3=NILF |
| PERWT | Person weight | For weighted analysis |
| AGE | Age at survey | Numeric |
| SEX | Sex | 1=Male, 2=Female |
| EDUC | Education | Categorical |
| MARST | Marital status | Categorical |

### 1.3 DACA Eligibility Criteria (from instructions)
1. Arrived in US before 16th birthday
2. Had not had 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status
5. Hispanic-Mexican ethnicity, born in Mexico
6. Not a citizen (proxy for undocumented status)

---

## Step 2: Sample Construction Decisions

### 2.1 Age Group Definition
- **Treatment (Young)**: Born 1982-1986 → Ages 26-30 on June 15, 2012
- **Control (Old)**: Born 1977-1981 → Ages 31-35 on June 15, 2012
- **Rationale**: Control group would be DACA-eligible except for age cutoff at 31

### 2.2 Eligibility Criteria Implementation
1. `HISPAN == 1` (Mexican Hispanic)
2. `BPL == 200` (Born in Mexico)
3. `CITIZEN == 3` (Not a citizen - proxy for undocumented)
4. Arrived before age 16: `YRIMMIG - BIRTHYR < 16`
5. In US since 2007: `YRIMMIG <= 2007`
6. Age restrictions based on birth year cohorts

### 2.3 Sample Construction Results
```
Initial sample size:           33,851,424
After HISPAN==1:                2,945,521
After BPL==200:                   991,261
After CITIZEN==3:                 701,347
After birth year filter:          178,376
After age_at_immig < 16:           48,406
After YRIMMIG <= 2007:             48,406
After excluding 2012:              44,161
```

### 2.4 Time Period Definition
- **Pre-DACA**: 2006-2011
- **Post-DACA**: 2013-2016
- **Excluded**: 2012 (implementation year - cannot distinguish before/after within year)

### 2.5 Outcome Variable
- **Full-time employment**: `UHRSWORK >= 35` hours per week AND `EMPSTAT == 1`
- Binary indicator (1 if full-time employed, 0 otherwise)

---

## Step 3: Analysis Commands

### 3.1 Python Analysis Script
Created `analysis.py` to perform:
1. Data loading and cleaning
2. Sample construction with eligibility criteria
3. Descriptive statistics
4. Difference-in-differences estimation
5. Robustness checks
6. Export of results for LaTeX report

### 3.2 Key Analytical Decisions

**Decision 1: Handling of 2012**
- Excluded from analysis as DACA was implemented mid-year (June 15, 2012)
- ACS does not provide month of interview, so pre/post cannot be distinguished

**Decision 2: Age calculation**
- Used birth year cohorts rather than AGE variable to maintain consistent treatment/control group definitions across survey years
- Treatment: Birth year 1982-1986
- Control: Birth year 1977-1981

**Decision 3: Immigration timing**
- Require YRIMMIG <= 2007 to satisfy "present since June 2007" requirement
- Require YRIMMIG - BIRTHYR < 16 for "arrived before age 16"

**Decision 4: Full-time definition**
- UHRSWORK >= 35 per standard BLS definition
- Set to 0 for those not employed (EMPSTAT != 1)

**Decision 5: Weighting**
- All analyses use PERWT (person weights) for population representativeness
- Standard errors clustered at state level (STATEFIP)

---

## Step 4: Model Specifications

### 4.1 Main DiD Model
```
fulltime_i = β0 + β1*treat_i + β2*post_t + β3*(treat_i × post_t) + ε_it
```
Where:
- `treat` = 1 if birth year 1982-1986, 0 if 1977-1981
- `post` = 1 if year ≥ 2013, 0 if year ≤ 2011
- β3 is the DiD estimate of DACA effect

### 4.2 Extended Models
- Model 1: Basic DiD (no controls)
- Model 2: Add demographic controls (female, married)
- Model 3: Add state fixed effects
- Model 4: Add year fixed effects (PREFERRED SPECIFICATION)

### 4.3 Robustness Checks
1. Event study with year-specific treatment effects
2. Placebo test using fake 2009 treatment
3. Alternative outcomes (employment, labor force participation)
4. Heterogeneity by sex

---

## Step 5: Results Summary

### 5.1 Sample Characteristics

| Group | Period | N | Full-time Rate |
|-------|--------|---|----------------|
| Control (31-35) | Pre-DACA | 11,757 | 0.611 |
| Control (31-35) | Post-DACA | 6,110 | 0.598 |
| Treatment (26-30) | Pre-DACA | 17,211 | 0.560 |
| Treatment (26-30) | Post-DACA | 9,083 | 0.621 |

### 5.2 Simple DiD Calculation
```
Treatment change:  0.621 - 0.560 = +0.061
Control change:    0.598 - 0.611 = -0.013
DiD estimate:      0.061 - (-0.013) = 0.073
```

### 5.3 Main Regression Results

| Model | DiD Estimate | SE | p-value |
|-------|--------------|-----|---------|
| (1) Basic | 0.0734 | 0.0090 | <0.001 |
| (2) Demographics | 0.0618 | 0.0103 | <0.001 |
| (3) State FE | 0.0615 | 0.0103 | <0.001 |
| (4) Year FE | **0.0605** | **0.0105** | **<0.001** |

### 5.4 Preferred Estimate (Model 4)
- **Effect Size**: 0.0605 (6.05 percentage points)
- **Standard Error**: 0.0105
- **95% CI**: [0.040, 0.081]
- **p-value**: < 0.001
- **Sample Size**: 44,161

### 5.5 Event Study Results
| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2006 | -0.010 | 0.015 | [-0.038, 0.019] |
| 2007 | -0.010 | 0.015 | [-0.040, 0.020] |
| 2008 | 0.016 | 0.017 | [-0.018, 0.049] |
| 2009 | 0.014 | 0.018 | [-0.022, 0.050] |
| 2010 | 0.021 | 0.016 | [-0.011, 0.053] |
| 2011 | 0.000 | --- | Reference |
| 2013 | 0.067 | 0.024 | [0.020, 0.113] |
| 2014 | 0.069 | 0.021 | [0.029, 0.110] |
| 2015 | 0.045 | 0.016 | [0.014, 0.076] |
| 2016 | 0.079 | 0.017 | [0.046, 0.112] |

### 5.6 Robustness Checks

**Alternative Outcomes:**
- Employment (any): DiD = 0.047 (SE = 0.008)
- Labor force participation: DiD = 0.031 (SE = 0.006)

**Heterogeneity by Sex:**
- Males: DiD = 0.065 (SE = 0.016)
- Females: DiD = 0.040 (SE = 0.017)

**Placebo Test (fake treatment in 2009):**
- Placebo DiD = 0.014 (SE = 0.008, p = 0.079)
- Not statistically significant - supports parallel trends assumption

---

## Step 6: Files Generated

1. `analysis.py` - Main analysis script
2. `replication_report_61.tex` - LaTeX report (~23 pages)
3. `replication_report_61.pdf` - Compiled PDF report
4. `run_log_61.md` - This log file
5. `figures/fig1_trends.png` - Employment trends by group
6. `figures/fig2_event_study.png` - Event study coefficients
7. `figures/fig3_did_bars.png` - Pre/post comparison bars
8. `tables/summary_stats.csv` - Summary statistics
9. `tables/main_results.csv` - Regression results
10. `tables/event_study.csv` - Event study coefficients
11. `tables/heterogeneity.csv` - Heterogeneity results
12. `tables/key_numbers.txt` - Key numbers for report

---

## Computational Environment
- Platform: Windows (win32)
- Python: Used for all analysis
- Key packages: pandas, numpy, statsmodels, matplotlib, seaborn
- LaTeX: pdflatex (MiKTeX)

---

## Key Commands Executed

```bash
# Run analysis
python analysis.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_61.tex
pdflatex -interaction=nonstopmode replication_report_61.tex  # Second pass for refs
```

---

## Interpretation Notes

1. **Main finding**: DACA eligibility increased full-time employment by approximately 6.1 percentage points among Hispanic-Mexican individuals born in Mexico who met the other eligibility criteria.

2. **Statistical significance**: The effect is highly statistically significant (p < 0.001) across all specifications.

3. **Parallel trends**: Event study shows no significant pre-trends (2006-2010 coefficients all near zero and not significant), supporting the validity of the DiD design.

4. **Mechanisms**: The effect appears to operate through multiple channels:
   - Increased labor force participation (+3.1 pp)
   - Increased employment (+4.7 pp)
   - Shift toward full-time work (full-time effect > employment effect)

5. **Heterogeneity**: Effects are present for both men and women, with somewhat larger effects for men (6.5 pp vs 4.0 pp).

6. **Robustness**: Results are robust to adding demographic controls, state fixed effects, and year fixed effects.

---

## Limitations Acknowledged

1. Cannot distinguish documented from undocumented non-citizens
2. Education/criminal history eligibility requirements not verifiable in data
3. 2012 exclusion reduces sample size
4. Age-based identification assumes no age-specific employment trends
5. Potential spillovers to control group could bias estimates toward zero

---

## Conclusion

The replication finds a statistically significant and economically meaningful positive effect of DACA eligibility on full-time employment. The preferred estimate of 6.05 percentage points (SE = 1.05) represents an approximately 11% increase relative to the pre-DACA baseline for the treatment group.
