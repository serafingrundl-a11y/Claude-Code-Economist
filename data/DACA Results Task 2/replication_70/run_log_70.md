# Run Log for DACA Replication Study (Replication 70)

## Overview
This log documents all commands, decisions, and analytical choices made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (outcome, defined as usually working 35+ hours per week)?

## Identification Strategy
- **Treatment group**: Ages 26-30 at time of policy implementation (June 15, 2012)
- **Control group**: Ages 31-35 at time of policy implementation (June 15, 2012)
- **Method**: Difference-in-differences comparing pre-treatment (2006-2011) to post-treatment (2013-2016) periods
- **Note**: 2012 is excluded as treatment timing within year is unclear

---

## Session Log

### Step 1: Initial Setup and Data Exploration
- **Date**: 2026-01-26
- Read `replication_instructions.docx` to understand the research task
- Reviewed data dictionary (`acs_data_dict.txt`) to understand variable definitions
- Confirmed data availability: ACS data from 2006-2016 in `data.csv` (33,851,424 observations)

### Step 2: Variable Identification

#### Key Variables Used:
| Variable | Description | Values Used |
|----------|-------------|-------------|
| YEAR | Survey year | 2006-2011, 2013-2016 |
| HISPAN | Hispanic origin | 1 (Mexican) |
| BPL | Birthplace | 200 (Mexico) |
| CITIZEN | Citizenship status | 3 (Not a citizen) |
| YRIMMIG | Year of immigration | <= 2007 |
| BIRTHYR | Birth year | 1977-1986 |
| UHRSWORK | Usual hours worked/week | >= 35 for full-time |
| PERWT | Person weight | Used for weighting |
| SEX | Sex | 1=Male, 2=Female |
| MARST | Marital status | 1,2=Married |
| NCHILD | Number of children | >0 = has children |
| EDUC | Education attainment | Various levels |

### Step 3: Sample Selection Process

**Command executed:** `python analysis.py`

**Sample Selection Criteria Applied:**

| Filter | Description | Sample Size After |
|--------|-------------|-------------------|
| 1 | Full ACS sample (2006-2016) | 33,851,424 |
| 2 | Exclude 2012 | 30,738,394 |
| 3 | Hispanic-Mexican (HISPAN=1) | 2,663,503 |
| 4 | Born in Mexico (BPL=200) | 898,879 |
| 5 | Non-citizen (CITIZEN=3) | 636,722 |
| 6 | Arrived by 2007 (YRIMMIG<=2007) | 595,366 |
| 7 | Arrived before age 16 | 177,294 |
| 8 | Ages 26-35 at DACA | 44,725 |

**Final Sample:**
- Treatment group (ages 26-30): 26,591 observations
- Control group (ages 31-35): 18,134 observations
- Pre-period (2006-2011): 29,326 observations
- Post-period (2013-2016): 15,399 observations

### Step 4: Key Analytical Decisions

#### Decision 1: Proxy for Undocumented Status
- **Choice**: Use CITIZEN = 3 (Not a citizen) as proxy
- **Rationale**: ACS does not identify undocumented status directly. Non-citizens who arrived before 2007 and have not naturalized are likely undocumented.
- **Limitation**: May include some legal permanent residents, potentially attenuating effects.

#### Decision 2: Continuous Presence Requirement
- **Choice**: YRIMMIG <= 2007
- **Rationale**: Proxies requirement that individuals be present in US since June 15, 2007
- **Limitation**: Cannot verify continuous presence, only arrival year

#### Decision 3: Arrived Before Age 16
- **Choice**: (YRIMMIG - BIRTHYR) < 16
- **Rationale**: Direct application of eligibility criterion
- **Note**: Uses year-level approximation

#### Decision 4: Treatment Period Definition
- **Pre-treatment**: 2006-2011 (6 years)
- **Post-treatment**: 2013-2016 (4 years)
- **Excluded**: 2012 (DACA implemented June 15, ACS doesn't identify survey month)

#### Decision 5: Full-Time Employment Definition
- **Choice**: UHRSWORK >= 35
- **Rationale**: Per instructions - "usually working 35 hours per week or more"

#### Decision 6: Control Variables
Included in preferred specification:
- Female indicator
- Married indicator
- Has children indicator
- Age and age squared
- Education indicators (HS, some college, college+)

#### Decision 7: Survey Weights
- **Choice**: Use PERWT (person weights) in preferred specification
- **Rationale**: Generates population-representative estimates

#### Decision 8: Standard Errors
- **Choice**: Heteroskedasticity-robust (HC1)
- **Rationale**: Appropriate for cross-sectional data with potential heteroskedasticity

### Step 5: Analysis Results

#### Main Result (Preferred Specification):
- **DiD Coefficient**: 0.0654
- **Standard Error**: 0.0148
- **95% CI**: [0.0364, 0.0944]
- **t-statistic**: 4.43
- **p-value**: < 0.001
- **Sample Size**: 44,725

**Interpretation**: DACA eligibility increased full-time employment by 6.54 percentage points.

#### Robustness Checks:

1. **Simple DiD (no controls, no weights)**: 0.0551 (SE: 0.0098)
2. **DiD with controls (no weights)**: 0.0658 (SE: 0.0124)
3. **Event study**: Pre-treatment coefficients small and insignificant, supporting parallel trends
4. **Placebo test (2009 fake treatment)**: -0.027 (SE: 0.015), p=0.082 - not significant
5. **By gender**: Male 0.0649 (SE: 0.0178), Female 0.0525 (SE: 0.0242)

#### Additional Outcomes:
- Any employment: DiD = 0.0573 (SE: 0.0140)
- Labor force participation: DiD = 0.0189 (SE: 0.0124), not significant

### Step 6: Figure Generation

**Command executed:** `python create_figures.py`

**Figures Created:**
1. `figure1_event_study.png` - Event study plot with year-by-year treatment effects
2. `figure2_trends.png` - Full-time employment trends by treatment status
3. `figure3_did.png` - Difference-in-differences illustration

### Step 7: Report Generation

**Command executed:** `pdflatex replication_report_70.tex` (run twice for references)

**Output:**
- `replication_report_70.tex` (LaTeX source)
- `replication_report_70.pdf` (23 pages)

---

## Files Generated

| Filename | Description |
|----------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `summary_stats.csv` | Descriptive statistics |
| `event_study_results.csv` | Event study coefficients |
| `figure1_event_study.png` | Event study plot |
| `figure2_trends.png` | Employment trends plot |
| `figure3_did.png` | DiD illustration |
| `replication_report_70.tex` | LaTeX report source |
| `replication_report_70.pdf` | Final PDF report (23 pages) |
| `run_log_70.md` | This run log |

---

## Summary of Findings

### Main Finding
DACA eligibility increased full-time employment by **6.54 percentage points** (95% CI: 3.64-9.44 pp) among Hispanic-Mexican, Mexican-born individuals ages 26-30 at policy implementation, relative to those ages 31-35.

### Robustness
- Results robust to inclusion of demographic controls
- Results robust to use of survey weights
- Event study shows no clear pre-treatment differential trends
- Placebo test supports parallel trends assumption
- Effects present for both men and women

### Interpretation
The estimated effect is consistent with DACA's provision of legal work authorization, which enabled eligible individuals to work in the formal labor market and access full-time employment opportunities. The effect represents approximately a 10.5% increase relative to the pre-treatment mean of 62.5% full-time employment for the treatment group.

---

## Technical Notes

### Software Used
- Python 3.x with pandas, numpy, statsmodels, scipy, matplotlib
- LaTeX (pdflatex) for report compilation

### Computation Time
- Data loading: ~30 seconds
- Full analysis: ~2 minutes
- LaTeX compilation: ~10 seconds

### Data Size
- Input: 6.3 GB CSV file
- Final sample: 44,725 observations

---

*Run log completed: 2026-01-26*
