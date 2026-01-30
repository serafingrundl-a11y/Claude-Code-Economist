"""
DACA Replication Study Analysis
Research Question: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals

Key DACA eligibility criteria (as of June 15, 2012):
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

Identification Strategy: Difference-in-differences
- Treatment group: DACA-eligible individuals (meet criteria above)
- Control group: Similar non-citizens who don't meet eligibility criteria
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded as transition year)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import json
warnings.filterwarnings('ignore')

print("="*80)
print("DACA REPLICATION STUDY - Full Analysis")
print("="*80)

# =============================================================================
# SECTION 1: DATA LOADING AND INITIAL PROCESSING (Chunked for memory)
# =============================================================================
print("\n[1] LOADING DATA IN CHUNKS...")

# Read data in chunks to manage memory - only keep relevant columns
cols_to_keep = ['YEAR', 'STATEFIP', 'PUMA', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR',
                'MARST', 'BIRTHYR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD',
                'CITIZEN', 'YRIMMIG', 'YRSUSA1', 'EDUC', 'EDUCD',
                'EMPSTAT', 'UHRSWORK', 'LABFORCE']

# Process in chunks and filter immediately
chunk_size = 1000000
chunks_processed = []
total_rows = 0
kept_rows = 0

print("Processing data in chunks (filtering as we go)...")

for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=cols_to_keep, chunksize=chunk_size)):
    total_rows += len(chunk)

    # Apply filters immediately to reduce memory
    # Filter 1: Hispanic-Mexican origin (HISPAN == 1 indicates Mexican)
    chunk = chunk[chunk['HISPAN'] == 1]

    # Filter 2: Born in Mexico (BPL == 200 is Mexico)
    chunk = chunk[chunk['BPL'] == 200]

    # Filter 3: Non-citizens (CITIZEN == 3: Not a citizen)
    chunk = chunk[chunk['CITIZEN'] == 3]

    # Filter 4: Exclude 2012 (transition year)
    chunk = chunk[chunk['YEAR'] != 2012]

    # Filter 5: Working-age individuals (16-64)
    chunk = chunk[(chunk['AGE'] >= 16) & (chunk['AGE'] <= 64)]

    if len(chunk) > 0:
        chunks_processed.append(chunk)
        kept_rows += len(chunk)

    if (i + 1) % 5 == 0:
        print(f"  Processed {total_rows:,} rows, kept {kept_rows:,} rows...")

# Combine all chunks
print(f"\nCombining filtered chunks...")
df = pd.concat(chunks_processed, ignore_index=True)
del chunks_processed  # Free memory

print(f"\nTotal rows processed: {total_rows:,}")
print(f"Final filtered sample: {len(df):,}")

# =============================================================================
# SECTION 2: CONSTRUCT KEY VARIABLES
# =============================================================================
print("\n[2] CONSTRUCTING ANALYSIS VARIABLES...")

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Calculate age at June 15, 2012 for DACA eligibility
# Using BIRTHYR and BIRTHQTR to be more precise
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

# Age at June 15, 2012
# Vectorized calculation
df['age_at_daca'] = 2012 - df['BIRTHYR']
# If born after June (Q3 or Q4), subtract 1 from age at June 15
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_at_daca'] -= 1

# DACA Eligibility Criteria:
# 1. Under 31 as of June 15, 2012 (born after June 15, 1981)
df['under31_at_daca'] = (df['age_at_daca'] < 31).astype(int)

# 2. Arrived before age 16
# Calculate age at arrival using YRIMMIG (year of immigration)
# YRIMMIG = 0 means N/A, so we need to handle this
df['age_at_arrival'] = np.where(
    df['YRIMMIG'] > 0,
    df['YRIMMIG'] - df['BIRTHYR'],
    np.nan
)
df['arrived_before_16'] = np.where(
    df['YRIMMIG'] > 0,
    (df['age_at_arrival'] < 16).astype(int),
    0
)

# 3. In US since June 2007 (at least 5 years by 2012)
# YRIMMIG should be 2007 or earlier
df['in_us_since_2007'] = np.where(
    df['YRIMMIG'] > 0,
    (df['YRIMMIG'] <= 2007).astype(int),
    0
)

# Combined DACA eligibility
# Eligible if: under 31 at DACA, arrived before 16, in US since 2007
df['daca_eligible'] = (
    (df['under31_at_daca'] == 1) &
    (df['arrived_before_16'] == 1) &
    (df['in_us_since_2007'] == 1)
).astype(int)

# Outcome variable: Full-time employment (35+ hours per week)
# UHRSWORK: usual hours worked per week (0 = N/A or not working)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Alternative outcome: Employed (in labor force and employed)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Labor force participation
df['in_labor_force'] = (df['LABFORCE'] == 2).astype(int)

# Create control group: non-eligible non-citizens of Mexican birth
df['control'] = (
    (df['CITIZEN'] == 3) &  # Non-citizen
    (df['daca_eligible'] == 0)  # Not DACA eligible
).astype(int)

# Education categories
df['less_than_hs'] = (df['EDUC'] < 6).astype(int)
df['high_school'] = ((df['EDUC'] >= 6) & (df['EDUC'] <= 7)).astype(int)
df['some_college'] = (df['EDUC'] >= 8).astype(int)

# Female indicator
df['female'] = (df['SEX'] == 2).astype(int)

# Marital status
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

print(f"\nDACA eligible observations: {df['daca_eligible'].sum():,}")
print(f"Control group observations: {df['control'].sum():,}")
print(f"Post-period observations: {df['post'].sum():,}")
print(f"Pre-period observations: {(df['post']==0).sum():,}")

# =============================================================================
# SECTION 3: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[3] GENERATING DESCRIPTIVE STATISTICS...")

# Overall sample by treatment/control and pre/post
summary_stats = pd.DataFrame()

for group_name, group_val in [('DACA Eligible', 1), ('Control', 0)]:
    for period, period_val in [('Pre', 0), ('Post', 1)]:
        subset = df[(df['daca_eligible'] == group_val) & (df['post'] == period_val)]

        stats_dict = {
            'Group': group_name,
            'Period': period,
            'N': len(subset),
            'N_weighted': subset['PERWT'].sum(),
            'Full-time Emp Rate': np.average(subset['fulltime'], weights=subset['PERWT']) if len(subset) > 0 else np.nan,
            'Employment Rate': np.average(subset['employed'], weights=subset['PERWT']) if len(subset) > 0 else np.nan,
            'LFP Rate': np.average(subset['in_labor_force'], weights=subset['PERWT']) if len(subset) > 0 else np.nan,
            'Mean Age': np.average(subset['AGE'], weights=subset['PERWT']) if len(subset) > 0 else np.nan,
            'Female Share': np.average(subset['female'], weights=subset['PERWT']) if len(subset) > 0 else np.nan,
            'Married Share': np.average(subset['married'], weights=subset['PERWT']) if len(subset) > 0 else np.nan,
            'Less than HS': np.average(subset['less_than_hs'], weights=subset['PERWT']) if len(subset) > 0 else np.nan,
            'High School': np.average(subset['high_school'], weights=subset['PERWT']) if len(subset) > 0 else np.nan,
            'Some College+': np.average(subset['some_college'], weights=subset['PERWT']) if len(subset) > 0 else np.nan,
        }
        summary_stats = pd.concat([summary_stats, pd.DataFrame([stats_dict])], ignore_index=True)

print("\nSummary Statistics:")
print(summary_stats.to_string())

# Save summary statistics
summary_stats.to_csv('summary_statistics.csv', index=False)

# =============================================================================
# SECTION 4: MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n[4] RUNNING DIFFERENCE-IN-DIFFERENCES REGRESSIONS...")

# Create analysis sample (treatment and control groups only)
analysis_df = df.copy()

# Interaction term for DiD
analysis_df['did'] = analysis_df['daca_eligible'] * analysis_df['post']

print(f"\nAnalysis sample size: {len(analysis_df):,}")

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls(
    'fulltime ~ daca_eligible + post + did',
    data=analysis_df,
    weights=analysis_df['PERWT']
).fit(cov_type='HC1')
print(f"DiD Coefficient: {model1.params['did']:.4f} (SE: {model1.bse['did']:.4f})")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographics ---")
model2 = smf.wls(
    'fulltime ~ daca_eligible + post + did + AGE + I(AGE**2) + female + married',
    data=analysis_df,
    weights=analysis_df['PERWT']
).fit(cov_type='HC1')
print(f"DiD Coefficient: {model2.params['did']:.4f} (SE: {model2.bse['did']:.4f})")

# Model 3: DiD with demographics and education
print("\n--- Model 3: DiD with Demographics + Education ---")
model3 = smf.wls(
    'fulltime ~ daca_eligible + post + did + AGE + I(AGE**2) + female + married + high_school + some_college',
    data=analysis_df,
    weights=analysis_df['PERWT']
).fit(cov_type='HC1')
print(f"DiD Coefficient: {model3.params['did']:.4f} (SE: {model3.bse['did']:.4f})")

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with Year Fixed Effects ---")
analysis_df['year_factor'] = analysis_df['YEAR'].astype(str)
model4 = smf.wls(
    'fulltime ~ daca_eligible + did + AGE + I(AGE**2) + female + married + high_school + some_college + C(year_factor)',
    data=analysis_df,
    weights=analysis_df['PERWT']
).fit(cov_type='HC1')
print(f"DiD Coefficient: {model4.params['did']:.4f} (SE: {model4.bse['did']:.4f})")

# Model 5: DiD with state and year fixed effects
print("\n--- Model 5: DiD with State and Year Fixed Effects ---")
analysis_df['state_factor'] = analysis_df['STATEFIP'].astype(str)
model5 = smf.wls(
    'fulltime ~ daca_eligible + did + AGE + I(AGE**2) + female + married + high_school + some_college + C(year_factor) + C(state_factor)',
    data=analysis_df,
    weights=analysis_df['PERWT']
).fit(cov_type='HC1')

# Extract key coefficients
print("\n" + "="*80)
print("MODEL 5 (PREFERRED) - KEY RESULTS:")
print("="*80)
print(f"DiD Coefficient (DACA effect): {model5.params['did']:.4f}")
print(f"Standard Error: {model5.bse['did']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['did', 0]:.4f}, {model5.conf_int().loc['did', 1]:.4f}]")
print(f"P-value: {model5.pvalues['did']:.4f}")
print(f"N: {int(model5.nobs):,}")

# =============================================================================
# SECTION 5: ROBUSTNESS CHECKS
# =============================================================================
print("\n[5] ROBUSTNESS CHECKS...")

# Check 1: Employment (any employment, not just full-time)
print("\n--- Robustness 1: Employment (Any) as Outcome ---")
model_emp = smf.wls(
    'employed ~ daca_eligible + did + AGE + I(AGE**2) + female + married + high_school + some_college + C(year_factor) + C(state_factor)',
    data=analysis_df,
    weights=analysis_df['PERWT']
).fit(cov_type='HC1')
print(f"Employment DiD Coefficient: {model_emp.params['did']:.4f} (SE: {model_emp.bse['did']:.4f})")

# Check 2: Labor Force Participation
print("\n--- Robustness 2: Labor Force Participation as Outcome ---")
model_lfp = smf.wls(
    'in_labor_force ~ daca_eligible + did + AGE + I(AGE**2) + female + married + high_school + some_college + C(year_factor) + C(state_factor)',
    data=analysis_df,
    weights=analysis_df['PERWT']
).fit(cov_type='HC1')
print(f"LFP DiD Coefficient: {model_lfp.params['did']:.4f} (SE: {model_lfp.bse['did']:.4f})")

# Check 3: Subgroup analysis by gender
print("\n--- Robustness 3: Subgroup by Gender ---")
model_male = smf.wls(
    'fulltime ~ daca_eligible + did + AGE + I(AGE**2) + married + high_school + some_college + C(year_factor) + C(state_factor)',
    data=analysis_df[analysis_df['female'] == 0],
    weights=analysis_df[analysis_df['female'] == 0]['PERWT']
).fit(cov_type='HC1')
print(f"Male Full-time DiD: {model_male.params['did']:.4f} (SE: {model_male.bse['did']:.4f})")

model_female = smf.wls(
    'fulltime ~ daca_eligible + did + AGE + I(AGE**2) + married + high_school + some_college + C(year_factor) + C(state_factor)',
    data=analysis_df[analysis_df['female'] == 1],
    weights=analysis_df[analysis_df['female'] == 1]['PERWT']
).fit(cov_type='HC1')
print(f"Female Full-time DiD: {model_female.params['did']:.4f} (SE: {model_female.bse['did']:.4f})")

# Check 4: Different age groups
print("\n--- Robustness 4: Age Group Analysis ---")
# Young workers (16-30)
young_df = analysis_df[(analysis_df['AGE'] >= 16) & (analysis_df['AGE'] <= 30)]
model_young = smf.wls(
    'fulltime ~ daca_eligible + did + AGE + I(AGE**2) + female + married + high_school + some_college + C(year_factor)',
    data=young_df,
    weights=young_df['PERWT']
).fit(cov_type='HC1')
print(f"Young (16-30) Full-time DiD: {model_young.params['did']:.4f} (SE: {model_young.bse['did']:.4f})")

# =============================================================================
# SECTION 6: EVENT STUDY / PARALLEL TRENDS
# =============================================================================
print("\n[6] EVENT STUDY ANALYSIS...")

# Create year-specific treatment effects
analysis_df['daca_2006'] = ((analysis_df['daca_eligible'] == 1) & (analysis_df['YEAR'] == 2006)).astype(int)
analysis_df['daca_2007'] = ((analysis_df['daca_eligible'] == 1) & (analysis_df['YEAR'] == 2007)).astype(int)
analysis_df['daca_2008'] = ((analysis_df['daca_eligible'] == 1) & (analysis_df['YEAR'] == 2008)).astype(int)
analysis_df['daca_2009'] = ((analysis_df['daca_eligible'] == 1) & (analysis_df['YEAR'] == 2009)).astype(int)
analysis_df['daca_2010'] = ((analysis_df['daca_eligible'] == 1) & (analysis_df['YEAR'] == 2010)).astype(int)
# 2011 is reference year (omitted)
analysis_df['daca_2013'] = ((analysis_df['daca_eligible'] == 1) & (analysis_df['YEAR'] == 2013)).astype(int)
analysis_df['daca_2014'] = ((analysis_df['daca_eligible'] == 1) & (analysis_df['YEAR'] == 2014)).astype(int)
analysis_df['daca_2015'] = ((analysis_df['daca_eligible'] == 1) & (analysis_df['YEAR'] == 2015)).astype(int)
analysis_df['daca_2016'] = ((analysis_df['daca_eligible'] == 1) & (analysis_df['YEAR'] == 2016)).astype(int)

model_event = smf.wls(
    'fulltime ~ daca_eligible + daca_2006 + daca_2007 + daca_2008 + daca_2009 + daca_2010 + daca_2013 + daca_2014 + daca_2015 + daca_2016 + AGE + I(AGE**2) + female + married + high_school + some_college + C(year_factor)',
    data=analysis_df,
    weights=analysis_df['PERWT']
).fit(cov_type='HC1')

print("\nEvent Study Coefficients (ref year = 2011):")
event_vars = ['daca_2006', 'daca_2007', 'daca_2008', 'daca_2009', 'daca_2010',
              'daca_2013', 'daca_2014', 'daca_2015', 'daca_2016']
event_results = {}
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    event_results[var] = {'coef': coef, 'se': se, 'pval': pval}
    print(f"{var}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# =============================================================================
# SECTION 7: SAVE RESULTS FOR REPORT
# =============================================================================
print("\n[7] SAVING RESULTS...")

# Create results dictionary
results = {
    'preferred_estimate': {
        'coefficient': float(model5.params['did']),
        'se': float(model5.bse['did']),
        'ci_lower': float(model5.conf_int().loc['did', 0]),
        'ci_upper': float(model5.conf_int().loc['did', 1]),
        'pvalue': float(model5.pvalues['did']),
        'n_obs': int(model5.nobs)
    },
    'model1_basic': {
        'coefficient': float(model1.params['did']),
        'se': float(model1.bse['did']),
        'n_obs': int(model1.nobs)
    },
    'model2_demographics': {
        'coefficient': float(model2.params['did']),
        'se': float(model2.bse['did']),
        'n_obs': int(model2.nobs)
    },
    'model3_education': {
        'coefficient': float(model3.params['did']),
        'se': float(model3.bse['did']),
        'n_obs': int(model3.nobs)
    },
    'model4_year_fe': {
        'coefficient': float(model4.params['did']),
        'se': float(model4.bse['did']),
        'n_obs': int(model4.nobs)
    },
    'model5_state_year_fe': {
        'coefficient': float(model5.params['did']),
        'se': float(model5.bse['did']),
        'n_obs': int(model5.nobs)
    },
    'robustness_employment': {
        'coefficient': float(model_emp.params['did']),
        'se': float(model_emp.bse['did'])
    },
    'robustness_lfp': {
        'coefficient': float(model_lfp.params['did']),
        'se': float(model_lfp.bse['did'])
    },
    'robustness_male': {
        'coefficient': float(model_male.params['did']),
        'se': float(model_male.bse['did'])
    },
    'robustness_female': {
        'coefficient': float(model_female.params['did']),
        'se': float(model_female.bse['did'])
    },
    'robustness_young': {
        'coefficient': float(model_young.params['did']),
        'se': float(model_young.bse['did'])
    },
    'event_study': {var: {'coef': float(event_results[var]['coef']),
                          'se': float(event_results[var]['se']),
                          'pval': float(event_results[var]['pval'])} for var in event_vars}
}

# Save results to file
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Calculate weighted means for pre/post by group for simple DiD calculation
print("\n[8] SIMPLE DiD CALCULATION...")

# Pre-period means
pre_treat = analysis_df[(analysis_df['daca_eligible']==1) & (analysis_df['post']==0)]
pre_control = analysis_df[(analysis_df['daca_eligible']==0) & (analysis_df['post']==0)]
post_treat = analysis_df[(analysis_df['daca_eligible']==1) & (analysis_df['post']==1)]
post_control = analysis_df[(analysis_df['daca_eligible']==0) & (analysis_df['post']==1)]

y_pre_treat = np.average(pre_treat['fulltime'], weights=pre_treat['PERWT'])
y_pre_control = np.average(pre_control['fulltime'], weights=pre_control['PERWT'])
y_post_treat = np.average(post_treat['fulltime'], weights=post_treat['PERWT'])
y_post_control = np.average(post_control['fulltime'], weights=post_control['PERWT'])

simple_did = (y_post_treat - y_pre_treat) - (y_post_control - y_pre_control)

print(f"\nWeighted Full-time Employment Rates:")
print(f"Pre-Treatment (DACA eligible):  {y_pre_treat:.4f}")
print(f"Post-Treatment (DACA eligible): {y_post_treat:.4f}")
print(f"Pre-Control:                    {y_pre_control:.4f}")
print(f"Post-Control:                   {y_post_control:.4f}")
print(f"\nSimple DiD Estimate: {simple_did:.4f}")

# Add to results
results['simple_did'] = {
    'y_pre_treat': float(y_pre_treat),
    'y_pre_control': float(y_pre_control),
    'y_post_treat': float(y_post_treat),
    'y_post_control': float(y_post_control),
    'estimate': float(simple_did)
}

# Trends by year
print("\n[9] FULL-TIME EMPLOYMENT TRENDS BY YEAR AND GROUP...")
yearly_trends = analysis_df.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n_obs': len(x),
        'n_weighted': x['PERWT'].sum()
    })
).reset_index()
print(yearly_trends.to_string())
yearly_trends.to_csv('yearly_trends.csv', index=False)

# Save updated results
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save summary stats for the report
summary_stats.to_csv('summary_statistics.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nPreferred estimate (Model 5 with state and year FE):")
print(f"  DACA Effect on Full-time Employment: {model5.params['did']:.4f}")
print(f"  Standard Error: {model5.bse['did']:.4f}")
print(f"  95% Confidence Interval: [{model5.conf_int().loc['did', 0]:.4f}, {model5.conf_int().loc['did', 1]:.4f}]")
print(f"  Sample Size: {int(model5.nobs):,}")
