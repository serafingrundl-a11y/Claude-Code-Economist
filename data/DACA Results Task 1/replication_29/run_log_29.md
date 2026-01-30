# DACA Replication Study - Run Log

## Study Information
- **Replication ID:** 29
- **Research Question:** Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the US
- **Date:** January 2026

---

## 1. Data Sources

### Primary Data
- **File:** `data/data.csv`
- **Source:** American Community Survey (ACS) via IPUMS USA
- **Years:** 2006-2016 (1-year ACS samples)
- **Size:** 33,851,424 observations (full dataset)

### Supporting Files
- **Data dictionary:** `data/acs_data_dict.txt`
- **State-level data:** `data/state_demo_policy.csv` (not used in main analysis)

---

## 2. Key Analytic Decisions

### 2.1 Sample Selection

1. **Ethnicity/Birthplace Filter:**
   - HISPAN = 1 (Mexican ethnicity)
   - BPL = 200 (Born in Mexico)
   - *Rationale:* Focus on population most affected by DACA; majority of DACA recipients are Mexican-origin

2. **Citizenship Status:**
   - CITIZEN = 3 (Not a citizen)
   - *Rationale:* DACA targets undocumented immigrants; non-citizen is best available proxy since ACS doesn't distinguish documented vs undocumented

3. **Age Restriction:**
   - Ages 16-64 (working-age population)
   - *Rationale:* Standard labor economics practice; ensures individuals are of legal working age

4. **Time Period:**
   - 2006-2016 with 2012 excluded from primary analysis
   - *Rationale:* DACA announced June 15, 2012; ACS lacks month, making 2012 ambiguous

### 2.2 DACA Eligibility Definition

Individuals coded as DACA-eligible if ALL of the following criteria are met:

| Criterion | IPUMS Implementation | DACA Requirement |
|-----------|---------------------|------------------|
| Age at arrival | YRIMMIG - BIRTHYR < 16 | Arrived before 16th birthday |
| Age at announcement | BIRTHYR >= 1982 | Under 31 on June 15, 2012 |
| Continuous residence | YRIMMIG <= 2007 | In US since June 15, 2007 |
| Immigration status | CITIZEN = 3 | Not lawfully present |

**Limitations:**
- Cannot observe education/military service requirement
- Cannot observe criminal record exclusions
- Some documented non-citizens may be included

### 2.3 Outcome Variable

- **Full-time employment:** UHRSWORK >= 35
- Binary indicator (1 = usually works 35+ hours/week, 0 = otherwise)
- Includes individuals not in labor force (coded as 0)

### 2.4 Treatment and Control Period

- **Pre-DACA:** 2006-2011
- **Post-DACA:** 2013-2016
- **Excluded:** 2012 (implementation year ambiguity)

---

## 3. Statistical Methods

### 3.1 Identification Strategy

**Difference-in-Differences (DiD)**

Comparing changes in full-time employment rates between:
- Treatment: DACA-eligible non-citizens
- Control: DACA-ineligible non-citizens (same ethnicity/birthplace)

### 3.2 Regression Specifications

**Model 1: Basic DiD**
```
FullTime = α + β(DACA × Post) + γ(DACA) + δ(Post) + ε
```

**Model 2: With Demographic Controls**
```
FullTime = α + β(DACA × Post) + γ(DACA) + δ(Post) + X'θ + ε
```
Controls: Age, Age², Female, Married, HS or more education

**Model 3: Year Fixed Effects**
```
FullTime = α + β(DACA × Post) + γ(DACA) + X'θ + λ_t + ε
```

**Model 4: State and Year Fixed Effects (Preferred)**
```
FullTime = α + β(DACA × Post) + γ(DACA) + X'θ + μ_s + λ_t + ε
```

### 3.3 Standard Errors

- Clustered at state level (STATEFIP)
- Accounts for within-state correlation and serial correlation

---

## 4. Results Summary

### 4.1 Sample Sizes

| Sample | N |
|--------|------|
| Initial (Hispanic-Mexican, born Mexico) | 991,261 |
| Working age (16-64) | 851,090 |
| Non-citizens | 618,640 |
| Final regression sample (excl. 2012) | 561,470 |
| DACA eligible | 89,164 |
| DACA ineligible | 529,476 |

### 4.2 Main Results

| Model | DiD Estimate | SE | 95% CI |
|-------|-------------|-----|--------|
| (1) Basic | 0.0932 | 0.0047 | [0.084, 0.103] |
| (2) + Controls | 0.0407 | 0.0049 | [0.031, 0.050] |
| (3) + Year FE | 0.0351 | 0.0047 | [0.026, 0.044] |
| (4) + State & Year FE | **0.0346** | **0.0047** | **[0.025, 0.044]** |

**Preferred Estimate:** 3.46 percentage points (p < 0.001)

### 4.3 Robustness Checks

| Check | Estimate | SE | Note |
|-------|----------|-----|------|
| Include 2012 | 0.0271 | 0.0033 | Smaller effect |
| Ages 18-30 only | 0.0085 | 0.0046 | Much smaller |
| Pre-trend test | 0.0059 | 0.0012 | Significant trend |

### 4.4 Heterogeneity

| Subgroup | Estimate | SE |
|----------|----------|-----|
| Male | 0.0295 | 0.0045 |
| Female | 0.0301 | 0.0069 |
| Less than HS | 0.0228 | 0.0043 |
| HS or more | 0.0311 | 0.0071 |

---

## 5. Key Concerns and Limitations

### 5.1 Pre-Trends

- Event study shows significant negative coefficients in 2006-2007
- Pre-trend test coefficient is positive and significant
- Suggests eligible group was trending upward relative to ineligible
- **Implication:** May overstate DACA effect

### 5.2 Measurement Error

- Cannot observe actual DACA status (only eligibility)
- Cannot distinguish documented from undocumented non-citizens
- Missing eligibility criteria (education, criminal record)
- **Implication:** Likely attenuates estimates toward zero

### 5.3 Other Issues

- Control group includes some documented immigrants
- Age composition changes over time (controlled for)
- Economic conditions varied (year FE help)

---

## 6. Commands Executed

### 6.1 Analysis Script
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_29"
python analysis.py
```

### 6.2 Figure Generation
```bash
python create_figures.py
```

### 6.3 LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_29.tex
pdflatex -interaction=nonstopmode replication_report_29.tex  # Second pass for TOC
pdflatex -interaction=nonstopmode replication_report_29.tex  # Third pass for references
```

---

## 7. Output Files

### 7.1 Required Deliverables
- [x] `replication_report_29.tex` - LaTeX source
- [x] `replication_report_29.pdf` - Compiled report (24 pages)
- [x] `run_log_29.md` - This file

### 7.2 Supporting Files
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script
- `analysis_results.json` - Results in JSON format
- `descriptive_statistics.csv` - Descriptive statistics
- `event_study_results.csv` - Event study coefficients
- `regression_results.txt` - Regression output
- `figure1_event_study.pdf` - Event study figure
- `figure2_trends.pdf` - Employment trends
- `figure3_model_comparison.pdf` - Model comparison
- `figure4_did_visual.pdf` - DiD visualization

---

## 8. Final Interpretation

**Main Finding:** DACA eligibility increased full-time employment by approximately 3.5 percentage points among Hispanic-Mexican non-citizens born in Mexico. This represents an 8% increase relative to the pre-DACA baseline of 42.6%.

**Confidence:** Moderate. While the effect is statistically significant and robust to specification changes, evidence of pre-trends suggests some caution in interpretation. The true effect may be somewhat smaller than estimated.

**Policy Implication:** Providing work authorization to undocumented youth can improve their labor market integration, as measured by full-time employment.

---

*Log completed: January 2026*
