# DACA Replication Study - Run Log

## Study Information
- **Research Question**: What was the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born non-citizens in the United States?
- **Analysis Period**: 2006-2016
- **Outcome**: Full-time employment (working >= 35 hours/week)

---

## Commands Executed

### 1. Data Exploration

```bash
# Check folder contents
ls -la data/

# Examine CSV header
head -1 data/data.csv

# Count total observations
wc -l data/data.csv
# Result: 33,851,425 rows (including header)
```

### 2. Run Main Analysis

```bash
python analysis.py
```

**Output Summary**:
- Total observations loaded: 33,851,424
- After filtering to Hispanic-Mexican (HISPAN=1): 2,945,521
- After filtering to Mexico-born (BPL=200): 991,261
- After filtering to non-citizens (CITIZEN=3): 701,347
- After filtering to ages 16-64: 618,640 (final sample)
- DACA-eligible: 92,822 (15.0%)
- Full-time employment rate: 57.1%

### 3. Create Figures

```bash
python create_figures.py
```

**Output**:
- figure1_trends.png/pdf (employment trends)
- figure2_eventstudy.png/pdf (event study plot)
- figure3_gap.png/pdf (employment gap)
- figure4_sample.png/pdf (sample composition)

### 4. Compile LaTeX Report

```bash
pdflatex -interaction=nonstopmode replication_report_80.tex
pdflatex -interaction=nonstopmode replication_report_80.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_80.tex  # Third pass
```

**Output**: replication_report_80.pdf (22 pages)

---

## Key Decisions

### Sample Selection

1. **Ethnicity**: Filtered to HISPAN = 1 (Mexican ethnicity)
   - Rationale: Research question specifies Hispanic-Mexican population

2. **Birthplace**: Filtered to BPL = 200 (Mexico)
   - Rationale: Research question specifies Mexican-born individuals

3. **Citizenship**: Filtered to CITIZEN = 3 (not a citizen)
   - Rationale: DACA is for undocumented immigrants; per instructions, assumed non-citizens without papers are undocumented

4. **Age Range**: Restricted to 16-64 years old
   - Rationale: Standard working-age population definition; younger individuals not typically in labor force

### DACA Eligibility Criteria

Created eligibility indicator based on observable criteria:

1. **Arrived before age 16**: YRIMMIG - BIRTHYR < 16
   - Rationale: DACA requires arrival before 16th birthday

2. **Under 31 on June 15, 2012**: BIRTHYR > 1981, or BIRTHYR = 1981 and BIRTHQTR >= 3
   - Rationale: DACA requires being under 31 on June 15, 2012
   - Conservative approach using birth quarter information

3. **Continuous residence since 2007**: YRIMMIG <= 2007
   - Rationale: DACA requires continuous US presence since June 15, 2007

### Treatment of 2012

- Excluded 2012 from post-treatment period (post_daca = 1 for 2013-2016 only)
- Rationale: DACA announced June 15, 2012; ACS doesn't identify collection month
- 2012 included in sample but treated as transition year

### Outcome Definition

- **Full-time employment**: UHRSWORK >= 35
- Rationale: Standard BLS definition of full-time work

### Econometric Specifications

1. **Basic DiD**: No controls, simple interaction
2. **DiD with Controls**: Added age, age^2, male, married, education, years in US, metro
3. **Preferred Specification**: Added state and year fixed effects
4. **Standard Errors**: Clustered by state (51 clusters)
   - Rationale: Account for within-state correlation in outcomes and policy implementation

### Variables Used (IPUMS Names)

| Purpose | Variable | Values Used |
|---------|----------|-------------|
| Year | YEAR | 2006-2016 |
| State | STATEFIP | All states |
| Person weight | PERWT | For weighted analysis |
| Sex | SEX | 1=Male, 2=Female |
| Age | AGE | 16-64 |
| Birth quarter | BIRTHQTR | 1-4 |
| Birth year | BIRTHYR | For eligibility |
| Hispanic origin | HISPAN | 1=Mexican |
| Birthplace | BPL | 200=Mexico |
| Citizenship | CITIZEN | 3=Not a citizen |
| Immigration year | YRIMMIG | For eligibility |
| Education | EDUCD | Detailed codes |
| Employment status | EMPSTAT | 1=Employed |
| Labor force | LABFORCE | 2=In labor force |
| Usual hours worked | UHRSWORK | >=35 for full-time |
| Metropolitan | METRO | 2,3,4=Metro |

---

## Main Results

### Preferred Estimate (Model with Year and State Fixed Effects)

| Metric | Value |
|--------|-------|
| DiD Estimate | 0.0312 |
| Standard Error | 0.0055 |
| 95% CI | [0.0206, 0.0419] |
| t-statistic | 5.730 |
| p-value | <0.0001 |
| Sample Size | 618,640 |
| R-squared | 0.217 |

**Interpretation**: DACA eligibility increased full-time employment by 3.12 percentage points (7.3% relative increase from 42.8% baseline).

### Robustness Checks

| Specification | Estimate | SE |
|--------------|----------|-----|
| Basic DiD (no controls) | 0.0868 | 0.0051 |
| DiD + controls | 0.0382 | 0.0058 |
| DiD + controls + FE (preferred) | 0.0312 | 0.0055 |
| Employment outcome | 0.0423 | 0.0093 |
| LFP outcome | 0.0417 | 0.0096 |
| Males only | 0.0272 | 0.0055 |
| Females only | 0.0289 | 0.0070 |
| Weighted (PERWT) | 0.0385 | 0.0051 |
| Placebo (2010 fake treatment) | 0.0129 | 0.0037 |

### Event Study Results

| Year | Coefficient | SE |
|------|-------------|-----|
| 2006 | -0.0167 | 0.0087 |
| 2007 | -0.0150 | 0.0048 |
| 2008 | -0.0023 | 0.0088 |
| 2009 | 0.0023 | 0.0059 |
| 2010 | 0.0050 | 0.0101 |
| 2011 | (reference) | -- |
| 2012 | (omitted) | -- |
| 2013 | 0.0095 | 0.0089 |
| 2014 | 0.0233 | 0.0128 |
| 2015 | 0.0403 | 0.0098 |
| 2016 | 0.0400 | 0.0087 |

---

## Files Generated

### Analysis Files
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script
- `analysis_results.pkl` - Pickled regression results
- `summary_stats.pkl` - Pickled summary statistics
- `event_study_results.pkl` - Pickled event study results

### Figures
- `figure1_trends.png/pdf` - Employment trends by eligibility
- `figure2_eventstudy.png/pdf` - Event study plot
- `figure3_gap.png/pdf` - Employment gap over time
- `figure4_sample.png/pdf` - Sample composition

### Report
- `replication_report_80.tex` - LaTeX source (22 pages)
- `replication_report_80.pdf` - Compiled report

---

## Caveats and Limitations

1. **Eligibility measurement error**: Cannot observe all DACA criteria (education requirements, criminal history, physical presence)

2. **Non-citizen assumption**: Some non-citizens may have legal status (visa holders) and not need DACA

3. **Parallel trends concern**: Placebo test shows significant pre-trend (p=0.0005), suggesting caution in causal interpretation

4. **2012 timing**: Cannot identify pre/post within 2012 due to lack of month information

5. **External validity**: Results specific to Hispanic-Mexican, Mexico-born non-citizens

---

## Session Information

- **Date**: January 2026
- **Python Version**: 3.x with pandas, numpy, statsmodels, matplotlib
- **LaTeX Distribution**: MiKTeX

---

## Conclusion

The analysis finds that DACA eligibility is associated with a 3.12 percentage point increase in full-time employment (95% CI: 2.06-4.19 pp). The effect is statistically significant and robust across specifications, though pre-trend concerns warrant some caution in interpretation. Effects appear to grow over time (event study shows larger effects in 2015-2016 than 2013-2014), consistent with gradual program uptake.
