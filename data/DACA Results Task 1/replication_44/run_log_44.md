# Run Log for DACA Replication Study (Replication 44)

## Overview
This log documents all commands, key decisions, and analytical choices made during the independent replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (>=35 hours/week)?

---

## Session Start: 2026-01-25

### Step 1: Data Exploration

**Commands:**
```bash
ls -la data/
head -1 data/data.csv | tr ',' '\n'
wc -l data/data.csv
```

**Data Files:**
- `data.csv`: Main ACS data file (~34 million observations, 6.3GB)
- `acs_data_dict.txt`: Data dictionary with variable definitions
- `state_demo_policy.csv`: Optional supplemental state-level data (not used)

**Years Available:** 2006-2016 (ACS 1-year samples)

**Key Variables Identified:**
- `YEAR`: Survey year
- `HISPAN`/`HISPAND`: Hispanic origin (1 = Mexican)
- `BPL`/`BPLD`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Birth year
- `BIRTHQTR`: Birth quarter
- `AGE`: Age
- `UHRSWORK`: Usual hours worked per week
- `EMPSTAT`: Employment status
- `PERWT`: Person weight
- `STATEFIP`: State FIPS code

---

### Step 2: DACA Eligibility Criteria Definition

**DACA Eligibility Requirements (per instructions):**
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

**Operationalization Decisions:**

1. **Age at arrival < 16**:
   - Calculate age at arrival = YRIMMIG - BIRTHYR
   - Require age_at_arrival >= 0 AND age_at_arrival < 16

2. **Under 31 on June 15, 2012**:
   - Must be born after June 15, 1981
   - Using BIRTHYR and BIRTHQTR:
     - BIRTHYR > 1981, OR
     - BIRTHYR = 1981 AND BIRTHQTR >= 3 (July-Sept or later, to be conservative)

3. **Continuous presence since June 15, 2007**:
   - YRIMMIG <= 2007

4. **Non-citizen status**:
   - CITIZEN = 3 (Not a citizen)
   - Note: Per instructions, assume non-citizens without papers are undocumented

**Key Decision:** I cannot observe the education/enrollment eligibility criterion (must be in school, graduated HS, or have GED), so this criterion is not applied. This may lead to some misclassification of eligibility.

---

### Step 3: Sample Construction

**Sample Restrictions Applied:**
1. Hispanic-Mexican: HISPAN = 1
2. Mexican-born: BPL = 200
3. Exclude 2012 (cannot distinguish pre/post DACA within the year)
4. Working age: Ages 16-64 for initial restriction
5. Non-citizens with valid YRIMMIG for main analysis
6. Ages 16-45 for comparability between treatment and control

**Final Sample:**
- Total: 427,762 person-year observations
- Treatment (DACA-eligible): 82,351
- Control (not DACA-eligible): 345,411

---

### Step 4: Identification Strategy

**Difference-in-Differences Design:**
- **Treatment Group**: DACA-eligible Mexican-born non-citizens
- **Control Group**: DACA-ineligible Mexican-born non-citizens (fail at least one eligibility criterion: arrived too old, arrived after 2007, or too old in 2012)
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016
- **Excluded**: 2012 (implementation year)

**Outcome Variable:**
- `fulltime`: Binary indicator = 1 if UHRSWORK >= 35

**Control Variables:**
- Age, Age-squared
- Female indicator
- Married indicator
- Year fixed effects
- State fixed effects

---

### Step 5: Analysis Commands

**Main Analysis Script:** `analysis.py`

```bash
python analysis.py
```

**Key Results:**

| Model | Specification | DiD Estimate | Std. Error |
|-------|--------------|--------------|------------|
| 1 | Basic DiD | 0.0973 | 0.0048 |
| 2 | + Demographics | 0.0305 | 0.0044 |
| 3 | + Year FE | 0.0211 | 0.0044 |
| 4 | + Year + State FE (Preferred) | 0.0212 | 0.0044 |

**Preferred Estimate:** 2.12 percentage points (SE = 0.44 pp)
**95% CI:** [1.26, 2.97] percentage points

---

### Step 6: Robustness Checks

1. **Alternative Control Group (Naturalized Citizens):**
   - Estimate: 0.024 (SE: 0.004)
   - Supports main finding

2. **Narrow Age Bandwidth (18-35):**
   - Estimate: 0.026 (SE: 0.005)
   - Slightly larger but consistent

3. **Placebo Test (Fake Policy 2009, Pre-Period Only):**
   - Estimate: 0.012 (SE: 0.006)
   - Much smaller than actual effect

4. **Event Study:**
   - Pre-period coefficients near zero (supports parallel trends)
   - Post-period coefficients grow over time (2014-2016)

---

### Step 7: Figure Generation

**Script:** `create_figures.py`

```bash
python create_figures.py
```

**Figures Created:**
1. `figure1_trends.png/pdf`: Full-time employment trends by eligibility
2. `figure2_eventstudy.png/pdf`: Event study coefficients
3. `figure3_did_bars.png/pdf`: Pre/post employment by group

---

### Step 8: Report Generation

**LaTeX Compilation:**
```bash
pdflatex -interaction=nonstopmode replication_report_44.tex
pdflatex -interaction=nonstopmode replication_report_44.tex
pdflatex -interaction=nonstopmode replication_report_44.tex
```

**Output:** `replication_report_44.pdf` (21 pages)

---

## Key Analytical Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target population | HISPAN=1 AND BPL=200 | As specified in instructions |
| Undocumented proxy | CITIZEN=3 | Per instructions, assume non-citizens without papers are undocumented |
| Age at arrival calculation | YRIMMIG - BIRTHYR | Direct calculation from available variables |
| Under-31 criterion | BIRTHYR>1981 OR (BIRTHYR=1981 AND BIRTHQTR>=3) | Conservative interpretation using birth quarter |
| Continuous presence | YRIMMIG <= 2007 | Must have arrived by 2007 |
| Exclude 2012 | Yes | Cannot distinguish pre/post within year |
| Age range | 16-45 | Balance treatment/control comparability |
| Control group | Non-citizen, non-eligible | Same immigration background, different eligibility |
| Preferred specification | Year + State FE | Controls for time trends and state heterogeneity |
| Weighting | PERWT | Survey weights for population representativeness |
| Standard errors | HC1 (robust) | Account for heteroskedasticity |

---

## Files Produced

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `results_summary.pkl` | Saved results for figures |
| `figure1_trends.png/pdf` | Trends figure |
| `figure2_eventstudy.png/pdf` | Event study figure |
| `figure3_did_bars.png/pdf` | DiD bar chart |
| `replication_report_44.tex` | LaTeX source |
| `replication_report_44.pdf` | Final report (21 pages) |
| `run_log_44.md` | This log file |

---

## Final Results Summary

**Preferred Estimate:**
- Effect size: 0.0212 (2.12 percentage points)
- Standard error: 0.0044 (0.44 percentage points)
- 95% CI: [0.0126, 0.0297]
- Sample size: 427,762
- p-value: < 0.001

**Interpretation:** DACA eligibility is associated with a 2.12 percentage point increase in the probability of full-time employment (working 35+ hours per week) among Mexican-born non-citizens. This represents approximately a 5% relative increase from the pre-DACA baseline rate of 45%.

---

## Session End: 2026-01-25
