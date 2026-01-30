"""
DACA Replication Study - Analysis Script
=========================================

Research Question:
Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on the probability of full-time employment?

Identification Strategy:
- Treatment Group: Ages 26-30 at DACA implementation (June 15, 2012)
- Control Group: Ages 31-35 at DACA implementation
- Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("DACA REPLICATION STUDY - FULL ANALYSIS")
print("="*80)

# =============================================================================
# Step 1: Load Data
# =============================================================================
print("\n[1] Loading data...")

# Read the CSV in chunks due to large file size
chunks = []
chunksize = 500000

# Define dtypes to reduce memory usage
dtypes = {
    'YEAR': 'int16',
    'SAMPLE': 'int32',
    'STATEFIP': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'UHRSWORK': 'int8',
    'EMPSTAT': 'int8',
    'PERWT': 'float32',
    'SEX': 'int8',
    'MARST': 'int8',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'LABFORCE': 'int8'
}

# First pass: filter the data as we read it
for chunk in pd.read_csv('data/data.csv', chunksize=chunksize, dtype=dtypes,
                          usecols=['YEAR', 'STATEFIP', 'AGE', 'BIRTHQTR', 'BIRTHYR',
                                   'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN',
                                   'YRIMMIG', 'UHRSWORK', 'EMPSTAT', 'PERWT',
                                   'SEX', 'MARST', 'EDUC', 'EDUCD', 'LABFORCE']):

    # Apply initial filters to reduce memory
    # Hispanic-Mexican (HISPAN == 1)
    # Born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]

    if len(filtered) > 0:
        chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"   Loaded {len(df):,} observations (Hispanic-Mexican, born in Mexico)")

# =============================================================================
# Step 2: Define Sample Selection
# =============================================================================
print("\n[2] Applying sample selection criteria...")

# Save initial count
n_initial = len(df)

# Criterion 1: Not a citizen (proxy for undocumented)
# CITIZEN == 3 means "Not a citizen"
df = df[df['CITIZEN'] == 3]
print(f"   After non-citizen filter: {len(df):,} ({len(df)/n_initial*100:.1f}%)")

# Criterion 2: Arrived before age 16
# Need valid immigration year
df = df[df['YRIMMIG'] > 0]  # Filter out N/A (0)
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_immigration'] < 16]
print(f"   After arrived before 16: {len(df):,}")

# Criterion 3: Define treatment and control groups based on age on June 15, 2012
# Treatment: ages 26-30 on June 15, 2012 (born 1982-1986)
# Control: ages 31-35 on June 15, 2012 (born 1977-1981)

# Calculate age on June 15, 2012
# If born Q1-Q2 (before June 15), age = 2012 - birthyr
# If born Q3-Q4 (after June 15), age = 2012 - birthyr - 1

df['age_june2012'] = 2012 - df['BIRTHYR']
# Adjust for those born after June 15 (Q3-Q4)
df.loc[df['BIRTHQTR'] >= 3, 'age_june2012'] = df.loc[df['BIRTHQTR'] >= 3, 'age_june2012'] - 1

# Create treatment and control groups
df['treated'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)
df['control'] = ((df['age_june2012'] >= 31) & (df['age_june2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df = df[(df['treated'] == 1) | (df['control'] == 1)]
print(f"   After age group filter (26-35): {len(df):,}")

# Criterion 4: In US since June 15, 2007 (5 years continuous residence)
# This means immigration year <= 2007
df = df[df['YRIMMIG'] <= 2007]
print(f"   After continuous residence (arrived <= 2007): {len(df):,}")

# Criterion 5: Exclude 2012 (cannot distinguish pre/post)
df = df[df['YEAR'] != 2012]
print(f"   After excluding 2012: {len(df):,}")

# =============================================================================
# Step 3: Create Outcome and Analysis Variables
# =============================================================================
print("\n[3] Creating analysis variables...")

# Define full-time employment (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Define post-DACA period (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term for DiD
df['treat_post'] = df['treated'] * df['post']

# Create age categories for fixed effects
df['age_group'] = pd.cut(df['AGE'], bins=[0, 25, 30, 35, 40, 100],
                          labels=['<26', '26-30', '31-35', '36-40', '>40'])

# Education categories
df['educ_cat'] = pd.cut(df['EDUC'], bins=[-1, 5, 6, 9, 11],
                         labels=['Less than HS', 'HS/GED', 'Some College', 'College+'])

# Gender
df['female'] = (df['SEX'] == 2).astype(int)

# Marital status (married vs not)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

print(f"   Final analytic sample: {len(df):,} observations")
print(f"   Treatment group (26-30): {df['treated'].sum():,}")
print(f"   Control group (31-35): {(1-df['treated']).sum():,}")
print(f"   Pre-period observations: {(1-df['post']).sum():,}")
print(f"   Post-period observations: {df['post'].sum():,}")

# =============================================================================
# Step 4: Summary Statistics
# =============================================================================
print("\n[4] Summary Statistics")
print("="*80)

# Overall summary
print("\n4.1 Full Sample Summary:")
print("-"*60)

summary_vars = ['fulltime', 'female', 'married', 'AGE', 'UHRSWORK']
summary_stats = df[summary_vars].describe().T[['mean', 'std', 'min', 'max']]
summary_stats['count'] = df[summary_vars].count()
print(summary_stats.to_string())

# Summary by treatment status
print("\n4.2 Summary by Treatment Group:")
print("-"*60)

def weighted_stats(group, var, weight):
    """Calculate weighted mean and standard error."""
    weights = group[weight]
    values = group[var]
    weighted_mean = np.average(values, weights=weights)
    weighted_var = np.average((values - weighted_mean)**2, weights=weights)
    weighted_std = np.sqrt(weighted_var)
    n = len(values)
    return pd.Series({'mean': weighted_mean, 'std': weighted_std, 'n': n})

print("\nTreatment Group (ages 26-30 on June 15, 2012):")
treat_df = df[df['treated'] == 1]
for var in ['fulltime', 'female', 'AGE', 'UHRSWORK']:
    stats = weighted_stats(treat_df, var, 'PERWT')
    print(f"   {var}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={int(stats['n']):,}")

print("\nControl Group (ages 31-35 on June 15, 2012):")
ctrl_df = df[df['treated'] == 0]
for var in ['fulltime', 'female', 'AGE', 'UHRSWORK']:
    stats = weighted_stats(ctrl_df, var, 'PERWT')
    print(f"   {var}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={int(stats['n']):,}")

# 2x2 DiD Table
print("\n4.3 Difference-in-Differences Table (Full-Time Employment Rate):")
print("-"*60)

did_table = df.groupby(['treated', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
did_table.columns = ['Pre-DACA', 'Post-DACA']
did_table.index = ['Control (31-35)', 'Treatment (26-30)']
did_table['Difference'] = did_table['Post-DACA'] - did_table['Pre-DACA']
print(did_table.round(4).to_string())
print(f"\nDiD Estimate: {did_table.loc['Treatment (26-30)', 'Difference'] - did_table.loc['Control (31-35)', 'Difference']:.4f}")

# =============================================================================
# Step 5: Main Difference-in-Differences Analysis
# =============================================================================
print("\n[5] Difference-in-Differences Regression Analysis")
print("="*80)

# Model 1: Basic DiD without controls
print("\n5.1 Basic DiD Model (No Controls):")
print("-"*60)

model1 = smf.wls('fulltime ~ treated + post + treat_post',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n5.2 DiD with Demographic Controls:")
print("-"*60)

model2 = smf.wls('fulltime ~ treated + post + treat_post + female + married',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n5.3 DiD with Year Fixed Effects:")
print("-"*60)

df['year_fe'] = df['YEAR'].astype(str)
model3 = smf.wls('fulltime ~ treated + treat_post + female + married + C(year_fe)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Extract key coefficients
key_vars = ['Intercept', 'treated', 'treat_post', 'female', 'married']
coef_df = pd.DataFrame({
    'coef': model3.params[key_vars],
    'std_err': model3.bse[key_vars],
    'pvalue': model3.pvalues[key_vars]
})
print(coef_df.round(4).to_string())

# Model 4: DiD with state fixed effects
print("\n5.4 DiD with State Fixed Effects:")
print("-"*60)

df['state_fe'] = df['STATEFIP'].astype(str)
model4 = smf.wls('fulltime ~ treated + treat_post + female + married + C(year_fe) + C(state_fe)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

key_vars = ['Intercept', 'treated', 'treat_post', 'female', 'married']
coef_df = pd.DataFrame({
    'coef': model4.params[key_vars],
    'std_err': model4.bse[key_vars],
    'pvalue': model4.pvalues[key_vars]
})
print(coef_df.round(4).to_string())

# =============================================================================
# Step 6: Preferred Specification
# =============================================================================
print("\n[6] Preferred Specification (Model 4 with Year and State FE)")
print("="*80)

# Extract the DiD coefficient
did_coef = model4.params['treat_post']
did_se = model4.bse['treat_post']
did_pval = model4.pvalues['treat_post']
ci_low = did_coef - 1.96 * did_se
ci_high = did_coef + 1.96 * did_se

print(f"\nDiD Estimate (treat_post coefficient):")
print(f"   Coefficient: {did_coef:.4f}")
print(f"   Standard Error: {did_se:.4f}")
print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"   p-value: {did_pval:.4f}")
print(f"   Sample Size: {int(model4.nobs):,}")

# =============================================================================
# Step 7: Robustness Checks
# =============================================================================
print("\n[7] Robustness Checks")
print("="*80)

# 7.1: Restrict to employed individuals (intensive margin)
print("\n7.1 Among Employed Only (Intensive Margin):")
df_emp = df[df['EMPSTAT'] == 1]
model_emp = smf.wls('fulltime ~ treated + treat_post + female + married + C(year_fe) + C(state_fe)',
                     data=df_emp, weights=df_emp['PERWT']).fit(cov_type='HC1')
print(f"   DiD Coefficient: {model_emp.params['treat_post']:.4f}")
print(f"   Standard Error: {model_emp.bse['treat_post']:.4f}")
print(f"   Sample Size: {int(model_emp.nobs):,}")

# 7.2: By gender
print("\n7.2 Heterogeneous Effects by Gender:")
df_male = df[df['female'] == 0]
df_female = df[df['female'] == 1]

model_male = smf.wls('fulltime ~ treated + treat_post + married + C(year_fe) + C(state_fe)',
                      data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ treated + treat_post + married + C(year_fe) + C(state_fe)',
                        data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"   Males:   DiD = {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f}), n={int(model_male.nobs):,}")
print(f"   Females: DiD = {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f}), n={int(model_female.nobs):,}")

# 7.3: Placebo test with pre-period only
print("\n7.3 Placebo Test (Pre-Period Trends, 2006-2011):")
df_pre = df[df['post'] == 0].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_treat_post'] = df_pre['treated'] * df_pre['placebo_post']

model_placebo = smf.wls('fulltime ~ treated + placebo_post + placebo_treat_post + female + married',
                         data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"   Placebo DiD Coefficient: {model_placebo.params['placebo_treat_post']:.4f}")
print(f"   Standard Error: {model_placebo.bse['placebo_treat_post']:.4f}")
print(f"   p-value: {model_placebo.pvalues['placebo_treat_post']:.4f}")

# 7.4: Alternative age bandwidths
print("\n7.4 Alternative Age Bandwidths:")

# Narrower: 27-29 vs 32-34
df_narrow = df[(df['age_june2012'] >= 27) & (df['age_june2012'] <= 34)]
df_narrow['treated_narrow'] = ((df_narrow['age_june2012'] >= 27) & (df_narrow['age_june2012'] <= 29)).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treated_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treated_narrow + treat_post_narrow + female + married + C(year_fe)',
                        data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"   Narrower (27-29 vs 32-34): DiD = {model_narrow.params['treat_post_narrow']:.4f} (SE: {model_narrow.bse['treat_post_narrow']:.4f})")

# Wider: 24-30 vs 31-37
df_full = pd.concat(chunks, ignore_index=True)
df_full = df_full[df_full['CITIZEN'] == 3]
df_full = df_full[df_full['YRIMMIG'] > 0]
df_full['age_at_immigration'] = df_full['YRIMMIG'] - df_full['BIRTHYR']
df_full = df_full[df_full['age_at_immigration'] < 16]
df_full['age_june2012'] = 2012 - df_full['BIRTHYR']
df_full.loc[df_full['BIRTHQTR'] >= 3, 'age_june2012'] = df_full.loc[df_full['BIRTHQTR'] >= 3, 'age_june2012'] - 1
df_full = df_full[df_full['YRIMMIG'] <= 2007]
df_full = df_full[df_full['YEAR'] != 2012]

df_wide = df_full[(df_full['age_june2012'] >= 24) & (df_full['age_june2012'] <= 37)]
df_wide['treated_wide'] = ((df_wide['age_june2012'] >= 24) & (df_wide['age_june2012'] <= 30)).astype(int)
df_wide['post'] = (df_wide['YEAR'] >= 2013).astype(int)
df_wide['treat_post_wide'] = df_wide['treated_wide'] * df_wide['post']
df_wide['fulltime'] = (df_wide['UHRSWORK'] >= 35).astype(int)
df_wide['female'] = (df_wide['SEX'] == 2).astype(int)
df_wide['married'] = (df_wide['MARST'].isin([1, 2])).astype(int)
df_wide['year_fe'] = df_wide['YEAR'].astype(str)

model_wide = smf.wls('fulltime ~ treated_wide + treat_post_wide + female + married + C(year_fe)',
                      data=df_wide, weights=df_wide['PERWT']).fit(cov_type='HC1')
print(f"   Wider (24-30 vs 31-37): DiD = {model_wide.params['treat_post_wide']:.4f} (SE: {model_wide.bse['treat_post_wide']:.4f})")

# =============================================================================
# Step 8: Event Study
# =============================================================================
print("\n[8] Event Study Analysis")
print("="*80)

# Create year indicators for event study
df['year_2006'] = (df['YEAR'] == 2006).astype(int)
df['year_2007'] = (df['YEAR'] == 2007).astype(int)
df['year_2008'] = (df['YEAR'] == 2008).astype(int)
df['year_2009'] = (df['YEAR'] == 2009).astype(int)
df['year_2010'] = (df['YEAR'] == 2010).astype(int)
df['year_2011'] = (df['YEAR'] == 2011).astype(int)
# 2012 excluded
df['year_2013'] = (df['YEAR'] == 2013).astype(int)
df['year_2014'] = (df['YEAR'] == 2014).astype(int)
df['year_2015'] = (df['YEAR'] == 2015).astype(int)
df['year_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions with treatment (2011 is reference)
df['treat_2006'] = df['treated'] * df['year_2006']
df['treat_2007'] = df['treated'] * df['year_2007']
df['treat_2008'] = df['treated'] * df['year_2008']
df['treat_2009'] = df['treated'] * df['year_2009']
df['treat_2010'] = df['treated'] * df['year_2010']
df['treat_2013'] = df['treated'] * df['year_2013']
df['treat_2014'] = df['treated'] * df['year_2014']
df['treat_2015'] = df['treated'] * df['year_2015']
df['treat_2016'] = df['treated'] * df['year_2016']

event_model = smf.wls('''fulltime ~ treated +
                          year_2006 + year_2007 + year_2008 + year_2009 + year_2010 +
                          year_2013 + year_2014 + year_2015 + year_2016 +
                          treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 +
                          treat_2013 + treat_2014 + treat_2015 + treat_2016 +
                          female + married''',
                       data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-"*60)
event_vars = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
              'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
event_coefs = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016],
    'Coefficient': [event_model.params[v] for v in event_vars],
    'Std_Error': [event_model.bse[v] for v in event_vars],
    'CI_Low': [event_model.params[v] - 1.96*event_model.bse[v] for v in event_vars],
    'CI_High': [event_model.params[v] + 1.96*event_model.bse[v] for v in event_vars]
})
# Add 2011 as reference
event_coefs = pd.concat([event_coefs, pd.DataFrame({'Year': [2011], 'Coefficient': [0],
                                                      'Std_Error': [0], 'CI_Low': [0], 'CI_High': [0]})],
                        ignore_index=True).sort_values('Year')
print(event_coefs.to_string(index=False))

# =============================================================================
# Step 9: Export Results for Report
# =============================================================================
print("\n[9] Saving Results for Report")
print("="*80)

# Save key results to a file
results = {
    'main_estimate': {
        'coefficient': did_coef,
        'std_error': did_se,
        'ci_lower': ci_low,
        'ci_upper': ci_high,
        'p_value': did_pval,
        'n_obs': int(model4.nobs)
    },
    'sample_info': {
        'total_obs': len(df),
        'treatment_obs': int(df['treated'].sum()),
        'control_obs': int((1-df['treated']).sum()),
        'pre_period_obs': int((1-df['post']).sum()),
        'post_period_obs': int(df['post'].sum())
    },
    'model_comparison': {
        'model1_basic': {'coef': model1.params['treat_post'], 'se': model1.bse['treat_post']},
        'model2_demographics': {'coef': model2.params['treat_post'], 'se': model2.bse['treat_post']},
        'model3_year_fe': {'coef': model3.params['treat_post'], 'se': model3.bse['treat_post']},
        'model4_state_fe': {'coef': model4.params['treat_post'], 'se': model4.bse['treat_post']}
    }
}

import json
with open('results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to results_summary.json")

# Save event study results
event_coefs.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save 2x2 DiD table
did_table.to_csv('did_table.csv')
print("DiD table saved to did_table.csv")

# =============================================================================
# Step 10: Summary Statistics Table for Report
# =============================================================================
print("\n[10] Generating Summary Statistics Table")
print("="*80)

# Create comprehensive summary statistics
def compute_weighted_stats(data, var, weight):
    """Compute weighted mean, std, and count."""
    w = data[weight]
    v = data[var]
    wmean = np.average(v, weights=w)
    wvar = np.average((v - wmean)**2, weights=w)
    wstd = np.sqrt(wvar)
    return wmean, wstd, len(v)

# Variables for summary table
summary_vars = {
    'fulltime': 'Full-time Employed',
    'UHRSWORK': 'Usual Hours Worked',
    'female': 'Female',
    'married': 'Married',
    'AGE': 'Age',
    'age_at_immigration': 'Age at Immigration'
}

# Create summary table by group
summary_data = []
for group_name, group_df in [('Treatment (26-30)', df[df['treated']==1]),
                              ('Control (31-35)', df[df['treated']==0]),
                              ('Full Sample', df)]:
    for var_name, var_label in summary_vars.items():
        if var_name in group_df.columns:
            mean, std, n = compute_weighted_stats(group_df, var_name, 'PERWT')
            summary_data.append({
                'Group': group_name,
                'Variable': var_label,
                'Mean': mean,
                'Std Dev': std,
                'N': n
            })

summary_df = pd.DataFrame(summary_data)
summary_pivot = summary_df.pivot(index='Variable', columns='Group', values=['Mean', 'Std Dev'])
print(summary_pivot.round(3).to_string())

summary_pivot.to_csv('summary_statistics.csv')
print("\nSummary statistics saved to summary_statistics.csv")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print(f"""
PREFERRED ESTIMATE (Model with Year and State Fixed Effects):
-------------------------------------------------------------
Effect of DACA Eligibility on Full-Time Employment:
  Coefficient:    {did_coef:.4f}
  Standard Error: {did_se:.4f}
  95% CI:         [{ci_low:.4f}, {ci_high:.4f}]
  p-value:        {did_pval:.4f}
  Sample Size:    {int(model4.nobs):,}

INTERPRETATION:
DACA eligibility is associated with a {did_coef*100:.2f} percentage point
{"increase" if did_coef > 0 else "decrease"} in the probability of full-time
employment among eligible Hispanic-Mexican individuals born in Mexico.

This effect is {"statistically significant" if did_pval < 0.05 else "not statistically significant"} at the 5% level.
""")

print("\nFiles generated:")
print("  - results_summary.json")
print("  - event_study_results.csv")
print("  - did_table.csv")
print("  - summary_statistics.csv")
