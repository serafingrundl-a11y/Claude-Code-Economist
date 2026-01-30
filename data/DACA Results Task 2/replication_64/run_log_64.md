# DACA Replication Study - Run Log

## Session Information
- **Date**: January 26, 2026
- **Task**: Independent replication of DACA effects on full-time employment
- **Data Source**: American Community Survey (ACS) 2006-2016 via IPUMS

---

## Step 1: Read and Understand Instructions

### Command
```
Read replication_instructions.docx
```

### Key Information Extracted
- **Research Question**: Effect of DACA eligibility on full-time employment (35+ hours/week)
- **Target Population**: Hispanic-Mexican, Mexican-born individuals in the US
- **Treatment Group**: Ages 26-30 as of June 15, 2012
- **Control Group**: Ages 31-35 as of June 15, 2012
- **Method**: Difference-in-differences
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016 (2012 excluded due to mid-year implementation)

---

## Step 2: Examine Data Structure

### Command
```bash
head -5 data/data.csv
```

### Data Overview
- **Total rows**: 33,851,424
- **Variables**: 54 columns
- **Key variables identified**:
  - YEAR: Survey year
  - HISPAN: Hispanic origin (1 = Mexican)
  - BPL: Birthplace (200 = Mexico)
  - CITIZEN: Citizenship status (3 = Not a citizen)
  - YRIMMIG: Year of immigration
  - BIRTHYR: Birth year
  - UHRSWORK: Usual hours worked per week
  - EMPSTAT: Employment status
  - PERWT: Person weights

---

## Step 3: Sample Selection Decisions

### DACA Eligibility Criteria Applied

| Filter | IPUMS Code | Rationale |
|--------|------------|-----------|
| Hispanic-Mexican ethnicity | HISPAN == 1 | Target population per instructions |
| Born in Mexico | BPL == 200 | DACA primarily affected Mexican immigrants |
| Non-citizen | CITIZEN == 3 | Proxy for undocumented status |
| Arrived before age 16 | (YRIMMIG - BIRTHYR) < 16 | DACA requirement |
| Continuous residence since 2007 | YRIMMIG <= 2007 | DACA requirement |

### Sample Construction Summary

| Stage | N |
|-------|---|
| Raw data (2006-2016) | 33,851,424 |
| Hispanic-Mexican | 2,945,521 |
| Born in Mexico | 991,261 |
| Non-citizen | 701,347 |
| Arrived before age 16 | 205,327 |
| Residence since 2007 | 195,023 |
| Age 26-35 in 2012 | 49,019 |
| Excluding 2012 | 44,725 |

---

## Step 4: Analysis Code Development

### File Created
`analysis.py`

### Key Design Decisions

1. **Treatment Definition**
   - Treatment: age_2012 in [26, 30]
   - Control: age_2012 in [31, 35]
   - where age_2012 = 2012 - BIRTHYR

2. **Outcome Definition**
   - fulltime = 1 if UHRSWORK >= 35, else 0

3. **Chunked Data Loading**
   - Used pandas chunked reading due to large file size (6.3 GB)
   - Applied filters during chunk processing to reduce memory

4. **Regression Specifications**
   - Model 1: Basic DiD (treat + post + treat_post)
   - Model 2: DiD + year fixed effects
   - Model 3: DiD + year FE + demographics
   - Model 4: DiD + year FE + demographics + state FE
   - Model 5: Weighted version of Model 4 (PREFERRED)

5. **Standard Errors**
   - Heteroskedasticity-robust (HC1) throughout

---

## Step 5: Run Analysis

### Command
```bash
cd "C:/Users/seraf/DACA Results Task 2/replication_64"
python analysis.py 2>&1 | tee analysis_output.txt
```

### Key Results

#### Raw Difference-in-Differences
| Group | Pre | Post | Diff |
|-------|-----|------|------|
| Treatment (26-30) | 0.6111 | 0.6339 | +0.0228 |
| Control (31-35) | 0.6431 | 0.6108 | -0.0323 |
| **DiD** | | | **+0.0551** |

#### Regression Estimates

| Model | DiD Coef | SE | p-value |
|-------|----------|-----|---------|
| 1. Basic | 0.0551 | 0.0098 | <0.001 |
| 2. Year FE | 0.0554 | 0.0098 | <0.001 |
| 3. Controls | 0.0160 | 0.0132 | 0.226 |
| 4. + State FE | 0.0143 | 0.0132 | 0.278 |
| 5. Weighted (PREFERRED) | 0.0171 | 0.0157 | 0.275 |

#### Event Study Coefficients (relative to 2011)

| Year | Coef | SE | Significant? |
|------|------|-----|--------------|
| 2006 | -0.035 | 0.020 | * |
| 2007 | -0.023 | 0.020 | |
| 2008 | -0.003 | 0.020 | |
| 2009 | -0.001 | 0.021 | |
| 2010 | -0.011 | 0.021 | |
| 2013 | +0.030 | 0.021 | |
| 2014 | +0.040 | 0.021 | * |
| 2015 | +0.039 | 0.022 | * |
| 2016 | +0.064 | 0.022 | *** |

#### Robustness Checks

| Check | Coefficient | p-value |
|-------|-------------|---------|
| Placebo (2009 fake treatment) | 0.0164 | 0.152 |
| Any employment outcome | 0.0433 | <0.001 |
| Males only | 0.0598 | <0.001 |
| Females only | 0.0365 | 0.015 |
| Alt control (32-36) | 0.0579 | <0.001 |

---

## Step 6: Generate Report

### Commands
```bash
# Create LaTeX file
# (Written via file write operation)

# Compile PDF
pdflatex -interaction=nonstopmode replication_report_64.tex
pdflatex -interaction=nonstopmode replication_report_64.tex  # Second pass for references
```

### Output
- `replication_report_64.tex` (LaTeX source)
- `replication_report_64.pdf` (21 pages)

---

## Files Generated

| File | Description |
|------|-------------|
| analysis.py | Main analysis script |
| analysis_output.txt | Full console output from analysis |
| yearly_means.csv | Employment rates by year and group |
| model_results.txt | Full regression outputs |
| summary_statistics.csv | Key statistics in CSV format |
| replication_report_64.tex | LaTeX report source |
| replication_report_64.pdf | Final report (21 pages) |
| run_log_64.md | This file |

---

## Key Analytical Decisions

### 1. Why exclude 2012?
DACA was implemented on June 15, 2012. The ACS does not record interview month, so 2012 observations cannot be cleanly classified as pre- or post-treatment.

### 2. Why use non-citizenship as proxy for undocumented?
The ACS does not directly identify undocumented status. Non-citizenship (CITIZEN=3) is a standard proxy in the literature. This includes some legal residents but captures most undocumented immigrants.

### 3. Why does effect attenuate with controls?
The treatment and control groups differ systematically by age. Age controls (age, age^2) absorb variation partly attributable to lifecycle effects on employment. The "true" DACA effect may be somewhere between the controlled and uncontrolled estimates.

### 4. Why use weighted regression as preferred?
ACS provides survey weights (PERWT) that make estimates representative of the US population. Weighted regression accounts for the complex survey design.

### 5. Interpretation of results
- Basic DiD: +5.5 pp, highly significant
- Preferred (weighted + controls): +1.7 pp, not significant (p=0.275)
- Event study shows growing effects, significant by 2016 (+6.4 pp)
- Effect on any employment larger (+4.3 pp) than full-time specifically

---

## Preferred Estimate Summary

| Metric | Value |
|--------|-------|
| DiD Coefficient | 0.0171 |
| Standard Error | 0.0157 |
| 95% CI | [-0.0136, 0.0479] |
| p-value | 0.275 |
| Sample Size | 44,725 |
| Treatment N | 26,591 |
| Control N | 18,134 |

**Interpretation**: The effect of DACA eligibility on full-time employment is positive (1.71 percentage points) but not statistically significant at conventional levels in the preferred specification.

---

## Session Complete
All required deliverables generated:
- [x] replication_report_64.tex
- [x] replication_report_64.pdf
- [x] run_log_64.md
