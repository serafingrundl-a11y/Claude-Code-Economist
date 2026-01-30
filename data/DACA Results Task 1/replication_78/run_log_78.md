# Run Log - DACA Replication Study (Replication 78)

## Project Overview
- **Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?
- **Time Period**: Pre-DACA (2006-2011) vs Post-DACA (2013-2016), excluding 2012
- **Identification Strategy**: Difference-in-Differences

---

## Key Decisions and Justifications

### 1. Sample Definition
- **Population**: Hispanic-Mexican ethnicity (HISPAN==1) AND born in Mexico (BPL==200)
- **Age restriction**: 18-64 (working-age population)
- **Excluded 2012**: DACA was implemented mid-year (June 15, 2012), so we cannot distinguish pre/post treatment within 2012

### 2. DACA Eligibility Criteria (Treatment Definition)
Applied the following criteria as of June 15, 2012:
1. **Arrived before 16th birthday**: age_at_arrival = YRIMMIG - BIRTHYR < 16
2. **Under 31 on June 15, 2012**: BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
3. **In US since June 2007**: YRIMMIG <= 2007
4. **Not a citizen**: CITIZEN == 3

**Note**: We cannot distinguish documented vs undocumented non-citizens in ACS data. Following instruction guidance, we assume non-citizens without naturalization are undocumented.

### 3. Outcome Variable
- **Full-time employment**: UHRSWORK >= 35 (usual hours worked per week >= 35)
- Binary outcome (0/1)

### 4. Control Variables
- Age and age-squared (quadratic)
- Female indicator (SEX == 2)
- Married indicator (MARST <= 2)
- Education categories (less than HS, some HS, HS grad, some college, college+)
- Metro area indicator (METRO >= 2)

### 5. Fixed Effects
- Year fixed effects (to control for common time trends)
- State fixed effects (STATEFIP, to control for geographic differences)

---

## Commands Executed

### Data Loading and Preparation
```python
# Loaded data in chunks due to file size (~34M rows)
# Filtered during load: HISPAN == 1 and BPL == 200
# Final sample: 755,660 observations (after age and year restrictions)
```

### Main Analysis
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_78"
python analysis.py
```

---

## Sample Statistics

| Group | Period | Full-Time Rate | N |
|-------|--------|----------------|------|
| Control | Pre-DACA | 62.2% | 415,802 |
| Control | Post-DACA | 60.3% | 268,511 |
| Treatment | Pre-DACA | 51.0% | 38,248 |
| Treatment | Post-DACA | 54.7% | 33,099 |

**Simple DID**: (54.7 - 51.0) - (60.3 - 62.2) = 3.7 - (-1.9) = 5.6 pp

---

## Regression Results Summary

| Model | Coefficient | SE | p-value | N |
|-------|-------------|------|---------|------|
| Basic DID (no controls) | 0.0561 | 0.0039 | <0.001 | 755,660 |
| + Demographics | 0.0122 | 0.0037 | 0.001 | 755,660 |
| + Year FE | 0.0070 | 0.0037 | 0.057 | 755,660 |
| + State FE (Preferred) | 0.0063 | 0.0037 | 0.089 | 755,660 |

### Robustness Checks
| Specification | Coefficient | SE | p-value | N |
|--------------|-------------|------|---------|------|
| Age 18-40 only | 0.0048 | 0.0039 | 0.218 | 414,834 |
| Males only | 0.0012 | 0.0048 | 0.807 | 399,807 |
| Females only | 0.0026 | 0.0055 | 0.634 | 355,853 |
| Any employment | 0.0201 | 0.0036 | <0.001 | 755,660 |

---

## Event Study Results (Relative to 2011)

| Year | Coefficient | SE | p-value |
|------|-------------|------|---------|
| 2006 | 0.0293 | 0.0087 | 0.001 |
| 2007 | 0.0252 | 0.0084 | 0.003 |
| 2008 | 0.0250 | 0.0084 | 0.003 |
| 2009 | 0.0177 | 0.0083 | 0.033 |
| 2010 | 0.0098 | 0.0080 | 0.225 |
| 2013 | 0.0024 | 0.0078 | 0.753 |
| 2014 | 0.0202 | 0.0077 | 0.009 |
| 2015 | 0.0334 | 0.0077 | <0.001 |
| 2016 | 0.0346 | 0.0078 | <0.001 |

**Note**: Pre-trends show some statistically significant differences in 2006-2009, which raises concerns about the parallel trends assumption. However, the magnitude decreases approaching 2011.

---

## Output Files Generated
1. `analysis.py` - Main analysis script
2. `analysis_output.txt` - Full console output
3. `regression_results.csv` - Summary of regression coefficients
4. `event_study_results.csv` - Event study coefficients
5. `descriptive_statistics.csv` - Sample characteristics by treatment status
6. `summary_by_group.csv` - Outcome means by treatment x period
7. `replication_report_78.tex` - LaTeX report
8. `replication_report_78.pdf` - Final PDF report

---

## Preferred Estimate

**Effect of DACA eligibility on full-time employment:**
- Coefficient: 0.0063 (0.63 percentage points)
- Standard Error: 0.0037
- 95% CI: [-0.0009, 0.0135]
- p-value: 0.089
- Sample Size: 755,660

**Interpretation**: DACA eligibility is associated with a 0.63 percentage point increase in the probability of full-time employment, though this effect is not statistically significant at the 5% level (p = 0.089).

---

## Methodological Notes

1. **Robust standard errors (HC1)** used throughout to account for heteroskedasticity
2. **Linear probability model** (OLS) used for ease of interpretation
3. **State and year fixed effects** control for location-specific and time-specific unobserved factors
4. **No sample weights applied** - unweighted analysis

---

## Date: January 2025
