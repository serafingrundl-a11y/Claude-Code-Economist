# DACA Replication Study - Run Log

## Study Information
- **Research Question**: What was the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico?
- **Data Source**: American Community Survey (ACS) 2006-2016 via IPUMS
- **Analysis Date**: January 25, 2026

---

## Session Log

### 1. Data Review and Understanding

**Timestamp**: Start of session

**Actions**:
- Read replication instructions from `replication_instructions.docx`
- Examined data dictionary in `acs_data_dict.txt`
- Reviewed structure of `data.csv` (approximately 34 million observations)

**Key Variables Identified**:
| Variable | Description | Use |
|----------|-------------|-----|
| YEAR | Survey year | Time period |
| HISPAN | Hispanic origin (1=Mexican) | Sample selection |
| BPL | Birthplace (200=Mexico) | Sample selection |
| CITIZEN | Citizenship status (3=not citizen) | DACA eligibility |
| YRIMMIG | Year of immigration | DACA eligibility |
| BIRTHYR | Birth year | DACA eligibility |
| UHRSWORK | Usual hours worked per week | Outcome |
| EMPSTAT | Employment status | Alternative outcome |
| SEX | Gender | Control |
| MARST | Marital status | Control |
| STATEFIP | State FIPS code | Fixed effects |
| PERWT | Person weight | Weighting |

---

### 2. Sample Construction Decisions

**Decision 1: Sample Population**
- **Choice**: Hispanic-Mexican (HISPAN=1) born in Mexico (BPL=200)
- **Rationale**: Instructions specify "ethnically Hispanic-Mexican Mexican-born people"
- **Result**: 991,261 observations from initial filter

**Decision 2: Exclude 2012**
- **Choice**: Exclude year 2012 from analysis
- **Rationale**: DACA was announced June 15, 2012 and applications started August 15, 2012. The ACS does not identify month of interview, so 2012 observations have ambiguous treatment status.
- **Result**: 898,879 observations after exclusion

**Decision 3: Working Age Restriction**
- **Choice**: Restrict to ages 18-64
- **Rationale**: Standard definition of working-age population; excludes minors and retirees

---

### 3. DACA Eligibility Criteria Implementation

**Treatment Group Definition** (DACA-Eligible):
All of the following must be true:
1. `age_at_arrival = YRIMMIG - BIRTHYR < 16` (arrived before age 16)
2. `BIRTHYR >= 1982` (under 31 on June 15, 2012)
3. `YRIMMIG <= 2007` (in US by 2007)
4. `CITIZEN == 3` (non-citizen)

**Control Group Definition**:
Non-citizens who narrowly miss DACA eligibility:

*Subgroup 1: Arrived too old*
- `age_at_arrival >= 16 AND age_at_arrival <= 25`
- `CITIZEN == 3`
- `YRIMMIG <= 2007`

*Subgroup 2: Born too early (too old in 2012)*
- `age_at_arrival < 16`
- `BIRTHYR >= 1972 AND BIRTHYR <= 1981`
- `CITIZEN == 3`
- `YRIMMIG <= 2007`

**Final Sample**:
- Treatment group: 69,244 observations
- Control group: 286,355 observations
- Total: 355,599 observations

---

### 4. Outcome Variable Definition

**Primary Outcome**: Full-time employment
- **Definition**: `UHRSWORK >= 35`
- **Rationale**: Standard BLS definition of full-time work is 35+ hours/week

**Secondary Outcome**: Any employment
- **Definition**: `EMPSTAT == 1`
- **Use**: Robustness check

---

### 5. Identification Strategy

**Method**: Difference-in-Differences (DiD)

**Design**:
- Treatment: DACA-eligible individuals
- Control: Similar non-citizens who narrowly miss eligibility
- Pre-period: 2006-2011
- Post-period: 2013-2016

**Identifying Assumption**: Parallel trends
- In absence of DACA, employment trends would have been similar between treatment and control groups

**Validation**: Event study analysis to test for pre-trends

---

### 6. Model Specifications

**Model 1: Basic DiD**
```
fulltime_employed ~ treat + post + treat_post
```

**Model 2: DiD with Demographics**
```
fulltime_employed ~ treat + post + treat_post + female + married + C(age_group)
```

**Model 3: DiD with Fixed Effects (Preferred)**
```
fulltime_employed ~ treat + treat_post + female + married + C(age_group) + C(STATEFIP) + C(YEAR)
```

**Model 4: Weighted DiD**
- Same as Model 3 but with person weights (PERWT)

**Standard Errors**: Heteroskedasticity-robust (HC1)

---

### 7. Commands Executed

```bash
# Data loading and analysis
python analysis.py

# Figure generation
python create_figures.py

# LaTeX compilation
pdflatex -interaction=nonstopmode replication_report_15.tex
pdflatex -interaction=nonstopmode replication_report_15.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_15.tex  # Third pass
```

---

### 8. Results Summary

**Main Results**:

| Model | DiD Estimate | Std. Error | 95% CI | N |
|-------|--------------|------------|--------|---|
| Basic DiD | 0.0631 | 0.0043 | [0.055, 0.071] | 355,599 |
| + Demographics | 0.0439 | 0.0041 | [0.036, 0.052] | 355,599 |
| + Fixed Effects | 0.0351 | 0.0041 | [0.027, 0.043] | 355,599 |
| Weighted | 0.0335 | 0.0049 | [0.024, 0.043] | 355,599 |

**Preferred Estimate**: 3.51 percentage points (Model 3)
- 95% CI: [2.71, 4.31]
- p-value < 0.001

**Event Study Results**:
| Year | Coefficient | SE | Pre/Post |
|------|-------------|-----|----------|
| 2006 | -0.017 | 0.010 | Pre |
| 2007 | -0.011 | 0.009 | Pre |
| 2008 | -0.003 | 0.009 | Pre |
| 2009 | 0.003 | 0.009 | Pre |
| 2010 | 0.001 | 0.009 | Pre |
| 2011 | 0.000 | -- | Reference |
| 2013 | 0.017 | 0.009 | Post |
| 2014 | 0.041 | 0.009 | Post |
| 2015 | 0.064 | 0.009 | Post |
| 2016 | 0.073 | 0.009 | Post |

**Key Finding**: Pre-trends are flat and near zero, supporting parallel trends assumption. Post-treatment effects grow over time.

**Robustness**:
| Specification | Estimate | SE |
|---------------|----------|-----|
| Control: Arrived 16+ only | 0.051 | 0.004 |
| Control: Too old only | 0.058 | 0.006 |
| Outcome: Any employment | 0.064 | 0.004 |

---

### 9. Output Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `main_results.csv` | Main regression results |
| `robustness_results.csv` | Robustness check results |
| `event_study_results.csv` | Event study coefficients |
| `descriptive_stats.csv` | Summary statistics |
| `yearly_trends.csv` | Year-by-year employment rates |
| `yearly_means_for_plot.csv` | Data for Figure 1 |
| `analysis_summary.txt` | Summary of preferred estimate |
| `figure1_trends.png/pdf` | Employment trends figure |
| `figure2_eventstudy.png/pdf` | Event study figure |
| `figure3_did.png/pdf` | DiD visualization |
| `replication_report_15.tex` | LaTeX report source |
| `replication_report_15.pdf` | Final report (16 pages) |

---

### 10. Key Methodological Decisions and Rationale

1. **Non-citizen proxy for undocumented status**: The ACS does not distinguish between documented and undocumented non-citizens. Using non-citizenship as a proxy likely includes some documented immigrants, which would attenuate estimates toward zero.

2. **Conservative age cutoff**: Used BIRTHYR >= 1982 rather than trying to incorporate birth quarter. This may exclude some eligible 1981-born individuals but avoids false positives.

3. **Combined control group**: Used both "arrived too old" and "born too early" groups. This provides a larger control group and leverages both dimensions of eligibility criteria. Robustness checks show results hold with either control group alone.

4. **Unweighted preferred specification**: While weighted results are similar, unweighted OLS with robust standard errors is more standard in the literature and less sensitive to outliers in weights.

5. **Excluding 2012**: Rather than attempting to model partial treatment in 2012, excluded the year entirely. This is conservative but avoids measurement error.

---

### 11. Interpretation

**Main Finding**: DACA eligibility increased full-time employment by approximately 3.5 percentage points among Hispanic-Mexican individuals born in Mexico. This represents a 7% relative increase from the pre-treatment mean of 50.5%.

**Mechanisms**:
- Direct work authorization enabling formal employment
- Access to jobs requiring employment verification
- Reduced fear of deportation enabling more visible employment
- Access to driver's licenses facilitating commuting

**Limitations**:
- Intent-to-treat effect (not all eligible individuals received DACA)
- Cannot identify undocumented status directly
- Potential spillovers to control group

---

## Session End

**Final Deliverables**:
- `replication_report_15.tex` - LaTeX source
- `replication_report_15.pdf` - 16-page report
- `run_log_15.md` - This file

**Analysis completed successfully.**
