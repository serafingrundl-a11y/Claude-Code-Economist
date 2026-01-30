# Run Log - Replication 97

## Project: DACA Impact on Full-Time Employment

### Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (>=35 hours/week)?

### Design
- **Treatment group**: Ages 26-30 at time of DACA (June 15, 2012) - DACA eligible
- **Control group**: Ages 31-35 at time of DACA (June 15, 2012) - too old for DACA
- **Method**: Difference-in-Differences
- **Pre-period**: 2006-2011 (or subset)
- **Post-period**: 2013-2016

---

## Session Log

### 2024-01-26 - Initial Setup

**Time**: Session Start

1. **Read replication instructions** from `replication_instructions.docx`
   - Key outcome: Full-time employment (UHRSWORK >= 35)
   - Treatment: DACA eligibility based on age at June 15, 2012
   - Target population: Hispanic-Mexican, born in Mexico, non-citizen immigrants

2. **Examined data files**:
   - `data.csv` - Main ACS data file (~6.2 GB)
   - `acs_data_dict.txt` - Data dictionary
   - `state_demo_policy.csv` - Optional supplemental data (not used)

3. **Key variables identified**:
   - `YEAR`: Survey year (2006-2016)
   - `BIRTHYR`: Year of birth (for age calculation)
   - `HISPAN`/`HISPAND`: Hispanic origin (HISPAN=1 for Mexican)
   - `BPL`/`BPLD`: Birthplace (BPL=200 for Mexico)
   - `CITIZEN`: Citizenship status (3 = not a citizen)
   - `YRIMMIG`: Year of immigration
   - `UHRSWORK`: Usual hours worked per week
   - `EMPSTAT`: Employment status
   - `PERWT`: Person weight

4. **DACA Eligibility Criteria** (from instructions):
   - Arrived unlawfully before 16th birthday
   - Had not yet turned 31 as of June 15, 2012
   - Lived continuously in US since June 15, 2007
   - Present in US on June 15, 2012 without lawful status

5. **Analytical approach**:
   - Limit sample to Hispanic-Mexican (HISPAN=1), born in Mexico (BPL=200)
   - Non-citizens only (CITIZEN=3 - proxy for undocumented status)
   - Calculate age as of June 15, 2012: age_2012 = 2012 - BIRTHYR
   - Treatment group: age_2012 between 26-30
   - Control group: age_2012 between 31-35
   - Outcome: fulltime = 1 if UHRSWORK >= 35, 0 otherwise
   - Additional filter: arrived before age 16 (YRIMMIG - BIRTHYR < 16)
   - Continuous residence since 2007: YRIMMIG <= 2007

---

### Data Processing Steps

**Step 1: Load and Filter Data**
- Loaded ACS data from data.csv in chunks (500,000 rows at a time) for memory efficiency
- Initial filter: Hispanic-Mexican (HISPAN=1), born in Mexico (BPL=200)
- Result: 991,261 observations

**Step 2: Apply DACA Eligibility Filters**
- Filter to non-citizens only (CITIZEN=3): 701,347 observations
- Filter to those who arrived before age 16 (YRIMMIG - BIRTHYR < 16): 205,327 observations
- Filter to continuous residence since 2007 (YRIMMIG <= 2007): 195,023 observations

**Step 3: Define Treatment and Control Groups**
- Treatment: Age 26-30 as of June 15, 2012 (DACA eligible)
- Control: Age 31-35 as of June 15, 2012 (just over cutoff)
- Result: 49,019 observations in treatment/control groups

**Step 4: Define Time Periods**
- Pre-period: 2006-2011 (before DACA)
- Post-period: 2013-2016 (after DACA fully in effect)
- Excluded 2012 (DACA implemented mid-year)
- Final sample: 44,725 observations

---

### Key Decisions and Rationale

1. **Proxy for Undocumented Status**: Used CITIZEN=3 (not a citizen) as proxy for undocumented status, as ACS does not distinguish documented from undocumented non-citizens. This is a common approach in the literature.

2. **Age Calculation**: Used age_2012 = 2012 - BIRTHYR. This is approximate since we don't have exact birth month, but the error is at most ~1 year.

3. **Continuous Residence Requirement**: Operationalized as YRIMMIG <= 2007, meaning immigrant arrived by 2007 (5 years before DACA).

4. **Arrival Before Age 16**: Required YRIMMIG - BIRTHYR < 16.

5. **Exclusion of 2012**: Dropped 2012 observations since DACA was implemented mid-year (June 15, 2012) and ACS does not report survey month.

6. **Full-Time Definition**: UHRSWORK >= 35 hours per week, consistent with standard BLS definition.

7. **Weighting**: Used PERWT (person weights) for all estimates to ensure population representativeness.

8. **Standard Errors**: Used heteroskedasticity-robust standard errors (HC1).

---

### Analysis Results

**Sample Sizes (Unweighted)**
| Group | Pre (2006-2011) | Post (2013-2016) | Total |
|-------|-----------------|------------------|-------|
| Control (31-35) | 11,916 | 6,218 | 18,134 |
| Treatment (26-30) | 17,410 | 9,181 | 26,591 |
| Total | 29,326 | 15,399 | 44,725 |

**Full-Time Employment Rates (Weighted)**
| Group | Pre | Post | Change |
|-------|-----|------|--------|
| Control (31-35) | 67.05% | 64.12% | -2.93 pp |
| Treatment (26-30) | 62.53% | 65.80% | +3.27 pp |

**Simple DiD**: 6.20 percentage points

**Regression Results**
| Model | Coefficient | SE | 95% CI |
|-------|-------------|-----|--------|
| 1. Basic DiD | 0.0620 | 0.0116 | [0.039, 0.085] |
| 2. Year FE | 0.0610 | 0.0115 | [0.038, 0.084] |
| 3. Covariates | 0.0474 | 0.0106 | [0.027, 0.068] |
| 4. State FE (Preferred) | 0.0472 | 0.0105 | [0.027, 0.068] |

**Preferred Estimate**
- Effect: 4.72 percentage points
- Standard Error: 0.0105
- 95% CI: [2.66, 6.79] pp
- p-value: < 0.001
- Sample Size: 44,725

**Robustness Checks**
| Specification | Coefficient | SE |
|---------------|-------------|-----|
| Any Employment | 0.0451 | 0.0100 |
| Men Only | 0.0482 | 0.0123 |
| Women Only | 0.0319 | 0.0176 |
| Narrower Ages (27-29 vs 32-34) | 0.0480 | 0.0134 |

**Parallel Trends Test**
- Joint F-test for pre-trends: F=0.78, p=0.56
- Conclusion: Cannot reject null of parallel pre-trends

---

### Files Generated

1. `analysis_97.py` - Main analysis script
2. `results_summary.json` - JSON summary of all results
3. `figure1_parallel_trends.png` - Parallel trends visualization
4. `figure2_event_study.png` - Event study/dynamic effects plot
5. `replication_report_97.tex` - LaTeX report
6. `replication_report_97.pdf` - Final PDF report

---

### Conclusion

DACA eligibility is associated with a statistically significant increase in full-time employment of approximately 4.7 percentage points (95% CI: 2.7-6.8 pp) among Hispanic-Mexican individuals born in Mexico who met the eligibility criteria. The effect is robust across multiple specifications and subgroups. The parallel trends assumption appears to hold based on the pre-period analysis.
