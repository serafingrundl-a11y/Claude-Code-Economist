"""
DACA Impact on Full-Time Employment: Difference-in-Differences Analysis
Replication Study - Run 09
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS SCRIPT")
print("=" * 70)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1] Loading ACS data...")

# Read data in chunks due to large file size
chunk_size = 500000
chunks = []

# Define columns we need to minimize memory usage
columns_needed = [
    'YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN',
    'YRIMMIG', 'YRSUSA1', 'SEX', 'MARST', 'EDUC', 'EDUCD',
    'UHRSWORK', 'EMPSTAT', 'EMPSTATD', 'LABFORCE',
    'STATEFIP', 'METRO'
]

for chunk in pd.read_csv('data/data.csv', usecols=columns_needed, chunksize=chunk_size):
    chunks.append(chunk)
    print(f"  Loaded {len(chunks) * chunk_size:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"  Total rows loaded: {len(df):,}")

# ============================================================================
# STEP 2: Identify DACA-Eligible Population
# ============================================================================
print("\n[2] Filtering to DACA-eligible population...")

# Step 2a: Hispanic-Mexican ethnicity
# HISPAN = 1 means Mexican, or HISPAND in 100-107 (Mexican detailed codes)
df['is_hispanic_mexican'] = (df['HISPAN'] == 1) | (df['HISPAND'].isin(range(100, 108)))
print(f"  Hispanic-Mexican: {df['is_hispanic_mexican'].sum():,}")

# Step 2b: Born in Mexico
# BPL = 200 is Mexico
df['born_mexico'] = df['BPL'] == 200
print(f"  Born in Mexico: {df['born_mexico'].sum():,}")

# Step 2c: Not a citizen (proxy for undocumented status)
# CITIZEN = 3 means "Not a citizen"
df['not_citizen'] = df['CITIZEN'] == 3
print(f"  Not a citizen: {df['not_citizen'].sum():,}")

# Combine base eligibility criteria
df['base_eligible'] = df['is_hispanic_mexican'] & df['born_mexico'] & df['not_citizen']
print(f"  Base eligible (Mexican-born non-citizen): {df['base_eligible'].sum():,}")

# Step 2d: Filter to base eligible population for further analysis
df_eligible = df[df['base_eligible']].copy()
print(f"  Working with {len(df_eligible):,} potentially eligible individuals")

# ============================================================================
# STEP 3: Determine Age at Policy Implementation
# ============================================================================
print("\n[3] Calculating age at DACA implementation (June 15, 2012)...")

# Calculate age at June 15, 2012
# Age depends on birth year and quarter
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# For June 15, 2012:
# - If born Q1 (Jan-Mar) or Q2 (Apr-Jun before Jun 15): already had birthday in 2012
# - If born Q3 or Q4 or Q2 (after Jun 15): haven't had birthday yet in 2012

# Simplified: Use birth year to calculate age, adjust for birth quarter
# Age at June 15, 2012 = 2012 - BIRTHYR - adjustment
# If BIRTHQTR >= 3, they haven't had their birthday yet by June 15, so subtract 1
# Q1, Q2 (Jan-Jun): most have had birthday by June 15, use 2012 - BIRTHYR
# Q3, Q4 (Jul-Dec): haven't had birthday by June 15, use 2012 - BIRTHYR - 1

# More precise: Q2 spans Apr-Jun, so roughly half have had birthday
# For simplicity, treat Q1-Q2 as already had birthday, Q3-Q4 as not yet
df_eligible['age_at_daca'] = 2012 - df_eligible['BIRTHYR']
df_eligible.loc[df_eligible['BIRTHQTR'] >= 3, 'age_at_daca'] -= 1

print(f"  Age at DACA range: {df_eligible['age_at_daca'].min()} to {df_eligible['age_at_daca'].max()}")

# ============================================================================
# STEP 4: Define Treatment and Control Groups
# ============================================================================
print("\n[4] Defining treatment and control groups...")

# Treatment group: Ages 26-30 at June 15, 2012 (eligible for DACA based on age)
# Control group: Ages 31-35 at June 15, 2012 (too old for DACA)

# Note: DACA required being under 31 on June 15, 2012
# Treatment: 26-30 (could potentially benefit from DACA)
# Control: 31-35 (just over the age cutoff)

df_eligible['treatment'] = (df_eligible['age_at_daca'] >= 26) & (df_eligible['age_at_daca'] <= 30)
df_eligible['control'] = (df_eligible['age_at_daca'] >= 31) & (df_eligible['age_at_daca'] <= 35)
df_eligible['in_sample'] = df_eligible['treatment'] | df_eligible['control']

print(f"  Treatment group (26-30): {df_eligible['treatment'].sum():,}")
print(f"  Control group (31-35): {df_eligible['control'].sum():,}")

# ============================================================================
# STEP 5: Additional Eligibility Criteria
# ============================================================================
print("\n[5] Applying additional DACA eligibility criteria...")

# DACA requires:
# - Arrived in US before 16th birthday
# - Present in US on June 15, 2012 (we proxy with being in the survey)
# - Lived continuously in US since June 15, 2007

# Age at arrival can be approximated from YRIMMIG and BIRTHYR
df_eligible['age_at_arrival'] = df_eligible['YRIMMIG'] - df_eligible['BIRTHYR']

# Arrived before 16th birthday
df_eligible['arrived_before_16'] = df_eligible['age_at_arrival'] < 16

# Arrived by 2007 (continuous residence since June 15, 2007)
# Use YRIMMIG <= 2007
df_eligible['arrived_by_2007'] = df_eligible['YRIMMIG'] <= 2007

# Some YRIMMIG values might be 0 (N/A) - these need to be handled
# Check distribution
print(f"  YRIMMIG = 0 (N/A): {(df_eligible['YRIMMIG'] == 0).sum():,}")

# For analysis, keep only those with valid immigration year
df_eligible['valid_yrimmig'] = df_eligible['YRIMMIG'] > 0

# Combined eligibility for DACA-like criteria
# Note: For the control group (age 31-35), they wouldn't actually be DACA eligible
# but we want them to be similar in other respects
df_eligible['daca_criteria'] = (
    df_eligible['arrived_before_16'] &
    df_eligible['arrived_by_2007'] &
    df_eligible['valid_yrimmig']
)

print(f"  Valid immigration year: {df_eligible['valid_yrimmig'].sum():,}")
print(f"  Arrived before age 16: {(df_eligible['arrived_before_16'] & df_eligible['valid_yrimmig']).sum():,}")
print(f"  Arrived by 2007: {(df_eligible['arrived_by_2007'] & df_eligible['valid_yrimmig']).sum():,}")
print(f"  Meet DACA criteria (except age): {df_eligible['daca_criteria'].sum():,}")

# ============================================================================
# STEP 6: Create Analysis Sample
# ============================================================================
print("\n[6] Creating final analysis sample...")

# Final sample: in treatment or control age group AND meets DACA criteria (except age)
df_analysis = df_eligible[df_eligible['in_sample'] & df_eligible['daca_criteria']].copy()
print(f"  Analysis sample size: {len(df_analysis):,}")

# Exclude 2012 (DACA implemented mid-year)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"  After excluding 2012: {len(df_analysis):,}")

# Create period indicator
# Pre-period: 2006-2011
# Post-period: 2013-2016
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Create treated indicator (1 for treatment group, 0 for control)
df_analysis['treated'] = df_analysis['treatment'].astype(int)

# Create interaction term
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

print(f"\n  Sample by group and period:")
print(f"    Treatment, Pre (2006-2011): {((df_analysis['treated']==1) & (df_analysis['post']==0)).sum():,}")
print(f"    Treatment, Post (2013-2016): {((df_analysis['treated']==1) & (df_analysis['post']==1)).sum():,}")
print(f"    Control, Pre (2006-2011): {((df_analysis['treated']==0) & (df_analysis['post']==0)).sum():,}")
print(f"    Control, Post (2013-2016): {((df_analysis['treated']==0) & (df_analysis['post']==1)).sum():,}")

# ============================================================================
# STEP 7: Define Outcome Variable
# ============================================================================
print("\n[7] Defining outcome: Full-time employment...")

# Full-time employment: usually works 35+ hours per week
# UHRSWORK = 0 typically means not working or N/A
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator for reference
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)

print(f"  Overall full-time rate: {df_analysis['fulltime'].mean():.3f}")
print(f"  Overall employment rate: {df_analysis['employed'].mean():.3f}")

# ============================================================================
# STEP 8: Create Control Variables
# ============================================================================
print("\n[8] Creating control variables...")

# Age (at survey time) - centered
df_analysis['age_centered'] = df_analysis['AGE'] - df_analysis['AGE'].mean()
df_analysis['age_sq'] = df_analysis['age_centered'] ** 2

# Sex (1=Male, 2=Female)
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Marital status
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Education categories
# EDUC: 0=N/A, 1=None, 2=Primary, 3=Some secondary, 4=HS diploma, 5=Some college, 6+=College+
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 4).astype(int)  # HS or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 7).astype(int)  # BA or more

# Years in US
df_analysis['years_in_us'] = df_analysis['YEAR'] - df_analysis['YRIMMIG']

# State fixed effects (convert to categorical)
df_analysis['state'] = df_analysis['STATEFIP'].astype('category')

# Year fixed effects
df_analysis['year_fe'] = df_analysis['YEAR'].astype('category')

print("  Control variables created: age, sex, marital status, education, years in US")

# ============================================================================
# STEP 9: Descriptive Statistics
# ============================================================================
print("\n[9] Descriptive statistics...")

# By treatment group
print("\n  Pre-treatment means:")
pre_treat = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==0)]
pre_ctrl = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==0)]

print(f"    Treatment group (N={len(pre_treat):,}):")
print(f"      Full-time: {pre_treat['fulltime'].mean():.3f}")
print(f"      Employed: {pre_treat['employed'].mean():.3f}")
print(f"      Female: {pre_treat['female'].mean():.3f}")
print(f"      Married: {pre_treat['married'].mean():.3f}")
print(f"      HS or more: {pre_treat['educ_hs'].mean():.3f}")
print(f"      Age: {pre_treat['AGE'].mean():.1f}")

print(f"\n    Control group (N={len(pre_ctrl):,}):")
print(f"      Full-time: {pre_ctrl['fulltime'].mean():.3f}")
print(f"      Employed: {pre_ctrl['employed'].mean():.3f}")
print(f"      Female: {pre_ctrl['female'].mean():.3f}")
print(f"      Married: {pre_ctrl['married'].mean():.3f}")
print(f"      HS or more: {pre_ctrl['educ_hs'].mean():.3f}")
print(f"      Age: {pre_ctrl['AGE'].mean():.1f}")

# ============================================================================
# STEP 10: Difference-in-Differences Analysis
# ============================================================================
print("\n" + "=" * 70)
print("[10] DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 70)

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls(
    'fulltime ~ treated + post + treated_post',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
model2 = smf.wls(
    'fulltime ~ treated + post + treated_post + female + married + age_centered + age_sq + educ_hs',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with year fixed effects ---")
model3 = smf.wls(
    'fulltime ~ treated + treated_post + female + married + age_centered + age_sq + educ_hs + C(YEAR)',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')
print(f"\n  DiD Coefficient (treated_post): {model3.params['treated_post']:.4f}")
print(f"  Std Error: {model3.bse['treated_post']:.4f}")
print(f"  95% CI: [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")
print(f"  P-value: {model3.pvalues['treated_post']:.4f}")

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with state and year fixed effects ---")
model4 = smf.wls(
    'fulltime ~ treated + treated_post + female + married + age_centered + age_sq + educ_hs + C(YEAR) + C(STATEFIP)',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')
print(f"\n  DiD Coefficient (treated_post): {model4.params['treated_post']:.4f}")
print(f"  Std Error: {model4.bse['treated_post']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"  P-value: {model4.pvalues['treated_post']:.4f}")

# ============================================================================
# STEP 11: Calculate Simple DiD Manually for Verification
# ============================================================================
print("\n" + "=" * 70)
print("[11] MANUAL DiD CALCULATION (Verification)")
print("=" * 70)

# Weighted means
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

treat_pre = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==0)]
treat_post = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==1)]
ctrl_pre = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==0)]
ctrl_post = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==1)]

y_treat_pre = weighted_mean(treat_pre, 'fulltime', 'PERWT')
y_treat_post = weighted_mean(treat_post, 'fulltime', 'PERWT')
y_ctrl_pre = weighted_mean(ctrl_pre, 'fulltime', 'PERWT')
y_ctrl_post = weighted_mean(ctrl_post, 'fulltime', 'PERWT')

did_estimate = (y_treat_post - y_treat_pre) - (y_ctrl_post - y_ctrl_pre)

print(f"\n  Treatment group:")
print(f"    Pre-period mean:  {y_treat_pre:.4f}")
print(f"    Post-period mean: {y_treat_post:.4f}")
print(f"    Change: {y_treat_post - y_treat_pre:.4f}")

print(f"\n  Control group:")
print(f"    Pre-period mean:  {y_ctrl_pre:.4f}")
print(f"    Post-period mean: {y_ctrl_post:.4f}")
print(f"    Change: {y_ctrl_post - y_ctrl_pre:.4f}")

print(f"\n  DIFFERENCE-IN-DIFFERENCES ESTIMATE: {did_estimate:.4f}")

# ============================================================================
# STEP 12: Event Study / Parallel Trends
# ============================================================================
print("\n" + "=" * 70)
print("[12] EVENT STUDY ANALYSIS (Parallel Trends Check)")
print("=" * 70)

# Create year indicators interacted with treatment
df_analysis['year'] = df_analysis['YEAR']
years = sorted(df_analysis['YEAR'].unique())
base_year = 2011  # Reference year (last pre-treatment year)

# Create interaction terms for each year
for yr in years:
    if yr != base_year:
        df_analysis[f'treat_yr_{yr}'] = (df_analysis['treated'] * (df_analysis['YEAR'] == yr)).astype(int)

# Build formula for event study
event_terms = ' + '.join([f'treat_yr_{yr}' for yr in years if yr != base_year])
event_formula = f'fulltime ~ treated + C(YEAR) + {event_terms} + female + married + age_centered + age_sq + educ_hs'

model_event = smf.wls(
    event_formula,
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')

print("\n  Event study coefficients (relative to 2011):")
for yr in years:
    if yr != base_year:
        coef = model_event.params[f'treat_yr_{yr}']
        se = model_event.bse[f'treat_yr_{yr}']
        pval = model_event.pvalues[f'treat_yr_{yr}']
        sig = '*' if pval < 0.05 else ''
        print(f"    {yr}: {coef:8.4f} (SE: {se:.4f}){sig}")

# ============================================================================
# STEP 13: Robustness Checks
# ============================================================================
print("\n" + "=" * 70)
print("[13] ROBUSTNESS CHECKS")
print("=" * 70)

# Robustness 1: Different age bandwidths
print("\n--- Narrower bandwidth: Ages 27-29 vs 32-34 ---")
df_narrow = df_analysis[
    ((df_analysis['age_at_daca'] >= 27) & (df_analysis['age_at_daca'] <= 29)) |
    ((df_analysis['age_at_daca'] >= 32) & (df_analysis['age_at_daca'] <= 34))
].copy()
df_narrow['treated'] = ((df_narrow['age_at_daca'] >= 27) & (df_narrow['age_at_daca'] <= 29)).astype(int)
df_narrow['treated_post'] = df_narrow['treated'] * df_narrow['post']

model_narrow = smf.wls(
    'fulltime ~ treated + post + treated_post + female + married + age_centered + educ_hs',
    data=df_narrow,
    weights=df_narrow['PERWT']
).fit(cov_type='HC1')
print(f"  DiD Coefficient: {model_narrow.params['treated_post']:.4f} (SE: {model_narrow.bse['treated_post']:.4f})")
print(f"  Sample size: {len(df_narrow):,}")

# Robustness 2: Employment as outcome
print("\n--- Alternative outcome: Any employment ---")
model_emp = smf.wls(
    'employed ~ treated + post + treated_post + female + married + age_centered + age_sq + educ_hs',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')
print(f"  DiD Coefficient: {model_emp.params['treated_post']:.4f} (SE: {model_emp.bse['treated_post']:.4f})")

# Robustness 3: By sex
print("\n--- Heterogeneity by sex ---")
df_male = df_analysis[df_analysis['female'] == 0]
df_female = df_analysis[df_analysis['female'] == 1]

model_male = smf.wls(
    'fulltime ~ treated + post + treated_post + married + age_centered + age_sq + educ_hs',
    data=df_male,
    weights=df_male['PERWT']
).fit(cov_type='HC1')

model_female = smf.wls(
    'fulltime ~ treated + post + treated_post + married + age_centered + age_sq + educ_hs',
    data=df_female,
    weights=df_female['PERWT']
).fit(cov_type='HC1')

print(f"  Males - DiD: {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f}), N={len(df_male):,}")
print(f"  Females - DiD: {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f}), N={len(df_female):,}")

# ============================================================================
# STEP 14: Save Results for Report
# ============================================================================
print("\n" + "=" * 70)
print("[14] SUMMARY OF RESULTS")
print("=" * 70)

print(f"""
PREFERRED SPECIFICATION: Model 3 (DiD with year fixed effects and controls)

  Sample: Mexican-born, Hispanic-Mexican, non-citizen individuals
          who arrived in US before age 16 and by 2007

  Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
  Control: Ages 31-35 at DACA implementation

  Outcome: Full-time employment (35+ hours/week)

  MAIN RESULT:
    DiD Coefficient: {model3.params['treated_post']:.4f}
    Standard Error:  {model3.bse['treated_post']:.4f}
    95% CI: [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]
    P-value: {model3.pvalues['treated_post']:.4f}

  Sample Size: {len(df_analysis):,}

  Interpretation:
    DACA eligibility is associated with a {model3.params['treated_post']*100:.2f} percentage point
    {'increase' if model3.params['treated_post'] > 0 else 'decrease'} in the probability of full-time employment.
""")

# Save key statistics to file for LaTeX report
results_dict = {
    'n_total': len(df_analysis),
    'n_treatment_pre': len(treat_pre),
    'n_treatment_post': len(treat_post),
    'n_control_pre': len(ctrl_pre),
    'n_control_post': len(ctrl_post),
    'y_treat_pre': y_treat_pre,
    'y_treat_post': y_treat_post,
    'y_ctrl_pre': y_ctrl_pre,
    'y_ctrl_post': y_ctrl_post,
    'did_simple': did_estimate,
    'did_model1': model1.params['treated_post'],
    'se_model1': model1.bse['treated_post'],
    'did_model2': model2.params['treated_post'],
    'se_model2': model2.bse['treated_post'],
    'did_model3': model3.params['treated_post'],
    'se_model3': model3.bse['treated_post'],
    'ci_low_model3': model3.conf_int().loc['treated_post', 0],
    'ci_high_model3': model3.conf_int().loc['treated_post', 1],
    'pval_model3': model3.pvalues['treated_post'],
    'did_model4': model4.params['treated_post'],
    'se_model4': model4.bse['treated_post'],
}

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print("\nResults saved to results.json")

# ============================================================================
# STEP 15: Generate Data for Plots
# ============================================================================
print("\n[15] Generating data for plots...")

# Time series of outcome by group
yearly_stats = df_analysis.groupby(['YEAR', 'treated']).apply(
    lambda x: pd.Series({
        'fulltime_mean': np.average(x['fulltime'], weights=x['PERWT']),
        'fulltime_se': x['fulltime'].std() / np.sqrt(len(x)),
        'n': len(x)
    })
).reset_index()

yearly_stats.to_csv('yearly_stats.csv', index=False)
print("  Saved yearly_stats.csv")

# Event study coefficients
event_coefs = []
for yr in years:
    if yr == base_year:
        event_coefs.append({'year': yr, 'coef': 0, 'se': 0})
    else:
        event_coefs.append({
            'year': yr,
            'coef': model_event.params[f'treat_yr_{yr}'],
            'se': model_event.bse[f'treat_yr_{yr}']
        })
event_df = pd.DataFrame(event_coefs)
event_df.to_csv('event_study.csv', index=False)
print("  Saved event_study.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
