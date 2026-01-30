"""
DACA Replication Study Analysis
==============================
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born non-citizens.

Author: Anonymous (Replication 23)
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*70)

# =============================================================================
# STEP 1: Load and filter data
# =============================================================================
print("\n[STEP 1] Loading data...")

# Define columns we need
cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
               'UHRSWORK', 'EMPSTAT', 'EDUC', 'MARST', 'STATEFIP', 'METRO']

# Read data in chunks to handle large file
chunk_size = 500000
chunks = []

print("Reading data in chunks...")
for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size)):
    # Filter: Hispanic-Mexican (HISPAN=1) and Born in Mexico (BPL=200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    if (i+1) % 20 == 0:
        print(f"  Processed {(i+1)*chunk_size:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"Total Hispanic-Mexican, Mexico-born observations: {len(df):,}")

# =============================================================================
# STEP 2: Define DACA eligibility criteria
# =============================================================================
print("\n[STEP 2] Applying DACA eligibility criteria...")

# Keep only non-citizens (CITIZEN=3)
# Note: We cannot distinguish documented vs undocumented, assume non-citizens
# without naturalization are undocumented per instructions
df = df[df['CITIZEN'] == 3].copy()
print(f"After citizenship filter (non-citizens only): {len(df):,}")

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA requirement: Arrived before age 16
df = df[df['age_at_immig'] < 16].copy()
print(f"After arrival before age 16 filter: {len(df):,}")

# DACA requirement: In US since June 15, 2007 (use YRIMMIG <= 2007)
df = df[df['YRIMMIG'] <= 2007].copy()
print(f"After arrival by 2007 filter: {len(df):,}")

# =============================================================================
# STEP 3: Define treatment and control groups based on birth year
# =============================================================================
print("\n[STEP 3] Defining treatment and control groups...")

# DACA was implemented June 15, 2012
# Treatment: Ages 26-30 on June 15, 2012 -> Birth years 1982-1986
# Control: Ages 31-35 on June 15, 2012 -> Birth years 1977-1981

# Define treatment group (eligible for DACA based on age)
df['treat'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)

# Define control group (just missed cutoff)
df['control'] = ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control groups
df = df[(df['treat'] == 1) | (df['control'] == 1)].copy()
print(f"After selecting treatment/control birth year cohorts: {len(df):,}")

# =============================================================================
# STEP 4: Define pre and post periods
# =============================================================================
print("\n[STEP 4] Defining time periods...")

# Pre-treatment: 2006-2011
# Post-treatment: 2013-2016
# Exclude 2012 (DACA implemented mid-year)

df = df[df['YEAR'] != 2012].copy()
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"After excluding 2012: {len(df):,}")

# =============================================================================
# STEP 5: Create outcome variable
# =============================================================================
print("\n[STEP 5] Creating outcome variable...")

# Full-time employment: Usually works 35+ hours per week
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employment indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# =============================================================================
# STEP 6: Create control variables
# =============================================================================
print("\n[STEP 6] Creating control variables...")

# Age at survey (for controls)
df['age'] = df['AGE']

# Sex indicator (female=1)
df['female'] = (df['SEX'] == 2).astype(int)

# Education categories
df['educ_less_hs'] = (df['EDUC'] < 6).astype(int)  # Less than high school
df['educ_hs'] = (df['EDUC'] == 6).astype(int)       # High school
df['educ_some_col'] = (df['EDUC'] == 7).astype(int) # Some college
df['educ_college'] = (df['EDUC'] >= 10).astype(int) # College+

# Marital status
df['married'] = (df['MARST'] <= 2).astype(int)  # Married (spouse present or absent)

# Metro status
df['metro'] = (df['METRO'] >= 2).astype(int)  # In metro area

# Years in US
df['yrs_in_us'] = df['YRSUSA1']

# =============================================================================
# STEP 7: Summary Statistics
# =============================================================================
print("\n[STEP 7] Computing summary statistics...")

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

# Sample sizes by group and period
print("\nSample Sizes (unweighted):")
print(df.groupby(['treat', 'post']).size().unstack())

# Weighted sample sizes
print("\nWeighted Sample Sizes:")
print(df.groupby(['treat', 'post'])['PERWT'].sum().unstack().round(0))

# Mean full-time employment by group and period
print("\nFull-time Employment Rate (weighted):")
ft_means = df.groupby(['treat', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
print(ft_means.round(4))

# Pre-post differences
print("\nPre-Post Differences:")
pre_treat = df[(df['treat']==1) & (df['post']==0)]
post_treat = df[(df['treat']==1) & (df['post']==1)]
pre_ctrl = df[(df['treat']==0) & (df['post']==0)]
post_ctrl = df[(df['treat']==0) & (df['post']==1)]

ft_pre_treat = np.average(pre_treat['fulltime'], weights=pre_treat['PERWT'])
ft_post_treat = np.average(post_treat['fulltime'], weights=post_treat['PERWT'])
ft_pre_ctrl = np.average(pre_ctrl['fulltime'], weights=pre_ctrl['PERWT'])
ft_post_ctrl = np.average(post_ctrl['fulltime'], weights=post_ctrl['PERWT'])

print(f"Treatment group: {ft_post_treat:.4f} - {ft_pre_treat:.4f} = {ft_post_treat - ft_pre_treat:.4f}")
print(f"Control group:   {ft_post_ctrl:.4f} - {ft_pre_ctrl:.4f} = {ft_post_ctrl - ft_pre_ctrl:.4f}")
print(f"DiD estimate:    {(ft_post_treat - ft_pre_treat) - (ft_post_ctrl - ft_pre_ctrl):.4f}")

# =============================================================================
# STEP 8: Difference-in-Differences Regression
# =============================================================================
print("\n" + "="*70)
print("REGRESSION ANALYSIS")
print("="*70)

# Create interaction term
df['treat_post'] = df['treat'] * df['post']

# Model 1: Basic DiD (no controls)
print("\n[Model 1] Basic DiD (no controls):")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n[Model 2] DiD with demographic controls:")
model2 = smf.wls('fulltime ~ treat + post + treat_post + female + age + I(age**2) + married',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with demographic + education controls
print("\n[Model 3] DiD with demographic + education controls:")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + age + I(age**2) + married + educ_hs + educ_some_col + educ_college',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: Full model with state fixed effects
print("\n[Model 4] DiD with controls + state fixed effects:")
df['state'] = df['STATEFIP'].astype('category')
model4 = smf.wls('fulltime ~ treat + post + treat_post + female + age + I(age**2) + married + educ_hs + educ_some_col + educ_college + C(state)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
# Print only key coefficients
print("\nKey Coefficients (Model 4):")
key_vars = ['Intercept', 'treat', 'post', 'treat_post', 'female', 'age',
            'I(age ** 2)', 'married', 'educ_hs', 'educ_some_col', 'educ_college']
for var in key_vars:
    if var in model4.params.index:
        print(f"  {var:20s}: {model4.params[var]:8.4f} (SE: {model4.bse[var]:.4f})")

# Model 5: Year fixed effects instead of single post dummy
print("\n[Model 5] DiD with year fixed effects:")
df['year_cat'] = df['YEAR'].astype('category')
model5 = smf.wls('fulltime ~ treat + C(year_cat) + treat:C(year_cat) + female + age + I(age**2) + married + educ_hs + educ_some_col + educ_college',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nYear-specific treatment effects (relative to 2006):")
for year in [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    var_name = f'treat:C(year_cat)[T.{year}]'
    if var_name in model5.params.index:
        print(f"  {year}: {model5.params[var_name]:8.4f} (SE: {model5.bse[var_name]:.4f})")

# =============================================================================
# STEP 9: Robustness Checks
# =============================================================================
print("\n" + "="*70)
print("ROBUSTNESS CHECKS")
print("="*70)

# Robustness 1: Restrict to employed individuals only
print("\n[Robustness 1] Effect on full-time conditional on employment:")
df_emp = df[df['employed'] == 1].copy()
model_r1 = smf.wls('fulltime ~ treat + post + treat_post + female + age + I(age**2) + married + educ_hs + educ_some_col + educ_college',
                    data=df_emp, weights=df_emp['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient: {model_r1.params['treat_post']:.4f} (SE: {model_r1.bse['treat_post']:.4f})")
print(f"  N = {len(df_emp):,}")

# Robustness 2: Effect on employment (extensive margin)
print("\n[Robustness 2] Effect on employment (extensive margin):")
model_r2 = smf.wls('employed ~ treat + post + treat_post + female + age + I(age**2) + married + educ_hs + educ_some_col + educ_college',
                    data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient: {model_r2.params['treat_post']:.4f} (SE: {model_r2.bse['treat_post']:.4f})")

# Robustness 3: By sex
print("\n[Robustness 3] Heterogeneous effects by sex:")
df_male = df[df['female'] == 0].copy()
df_female = df[df['female'] == 1].copy()

model_male = smf.wls('fulltime ~ treat + post + treat_post + age + I(age**2) + married + educ_hs + educ_some_col + educ_college',
                      data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ treat + post + treat_post + age + I(age**2) + married + educ_hs + educ_some_col + educ_college',
                        data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"  Males:   DiD = {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f}), N = {len(df_male):,}")
print(f"  Females: DiD = {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f}), N = {len(df_female):,}")

# Robustness 4: Alternative age bandwidth (tighter: 27-29 vs 32-34)
print("\n[Robustness 4] Tighter age bandwidth (27-29 vs 32-34):")
df_tight = df[(df['BIRTHYR'].isin([1983, 1984, 1985])) | (df['BIRTHYR'].isin([1978, 1979, 1980]))].copy()
df_tight['treat'] = df_tight['BIRTHYR'].isin([1983, 1984, 1985]).astype(int)
df_tight['treat_post'] = df_tight['treat'] * df_tight['post']
model_r4 = smf.wls('fulltime ~ treat + post + treat_post + female + age + I(age**2) + married + educ_hs + educ_some_col + educ_college',
                    data=df_tight, weights=df_tight['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient: {model_r4.params['treat_post']:.4f} (SE: {model_r4.bse['treat_post']:.4f})")
print(f"  N = {len(df_tight):,}")

# =============================================================================
# STEP 10: Create output tables and figures
# =============================================================================
print("\n" + "="*70)
print("CREATING OUTPUT FOR REPORT")
print("="*70)

# Save summary statistics to file
summary_stats = {
    'Group': ['Treatment (Pre)', 'Treatment (Post)', 'Control (Pre)', 'Control (Post)'],
    'N_unweighted': [len(pre_treat), len(post_treat), len(pre_ctrl), len(post_ctrl)],
    'N_weighted': [pre_treat['PERWT'].sum(), post_treat['PERWT'].sum(),
                   pre_ctrl['PERWT'].sum(), post_ctrl['PERWT'].sum()],
    'FT_Employment': [ft_pre_treat, ft_post_treat, ft_pre_ctrl, ft_post_ctrl],
    'Mean_Age': [np.average(pre_treat['age'], weights=pre_treat['PERWT']),
                 np.average(post_treat['age'], weights=post_treat['PERWT']),
                 np.average(pre_ctrl['age'], weights=pre_ctrl['PERWT']),
                 np.average(post_ctrl['age'], weights=post_ctrl['PERWT'])],
    'Pct_Female': [np.average(pre_treat['female'], weights=pre_treat['PERWT']),
                   np.average(post_treat['female'], weights=post_treat['PERWT']),
                   np.average(pre_ctrl['female'], weights=pre_ctrl['PERWT']),
                   np.average(post_ctrl['female'], weights=post_ctrl['PERWT'])],
    'Pct_Married': [np.average(pre_treat['married'], weights=pre_treat['PERWT']),
                    np.average(post_treat['married'], weights=post_treat['PERWT']),
                    np.average(pre_ctrl['married'], weights=pre_ctrl['PERWT']),
                    np.average(post_ctrl['married'], weights=post_ctrl['PERWT'])]
}
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_statistics.csv', index=False)
print("Saved: summary_statistics.csv")

# Save regression results
reg_results = {
    'Model': ['Model 1 (Basic)', 'Model 2 (Demographics)', 'Model 3 (+ Education)', 'Model 4 (+ State FE)'],
    'DiD_Coefficient': [model1.params['treat_post'], model2.params['treat_post'],
                        model3.params['treat_post'], model4.params['treat_post']],
    'Std_Error': [model1.bse['treat_post'], model2.bse['treat_post'],
                  model3.bse['treat_post'], model4.bse['treat_post']],
    'P_Value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
                model3.pvalues['treat_post'], model4.pvalues['treat_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs)]
}
reg_df = pd.DataFrame(reg_results)
reg_df.to_csv('regression_results.csv', index=False)
print("Saved: regression_results.csv")

# Create event study plot
print("\nCreating event study figure...")
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
year_effects = []
year_ses = []

# Get coefficients from model5 (year interactions)
for year in years:
    if year == 2006:
        year_effects.append(0)  # Reference year
        year_ses.append(0)
    else:
        var_name = f'treat:C(year_cat)[T.{year}]'
        if var_name in model5.params.index:
            year_effects.append(model5.params[var_name])
            year_ses.append(model5.bse[var_name])

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(years, year_effects, yerr=[1.96*se for se in year_ses],
            fmt='o-', capsize=5, capthick=2, markersize=8, color='navy')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
ax.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect on Full-Time Employment', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend()
ax.set_xticks(years)
plt.tight_layout()
plt.savefig('event_study.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: event_study.png")

# Create parallel trends figure
print("Creating parallel trends figure...")
ft_by_year_treat = df[df['treat']==1].groupby('YEAR').apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT']))
ft_by_year_ctrl = df[df['treat']==0].groupby('YEAR').apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT']))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ft_by_year_treat.index, ft_by_year_treat.values, 'o-',
        label='Treatment (Ages 26-30 at 2012)', color='blue', markersize=8)
ax.plot(ft_by_year_ctrl.index, ft_by_year_ctrl.values, 's-',
        label='Control (Ages 31-35 at 2012)', color='red', markersize=8)
ax.axvline(x=2012, color='gray', linestyle='--', alpha=0.7, label='DACA (June 2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by Treatment Status', fontsize=14)
ax.legend()
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('parallel_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: parallel_trends.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)

print(f"""
PREFERRED ESTIMATE (Model 3 - DiD with demographic and education controls):
  Effect Size:     {model3.params['treat_post']:.4f}
  Standard Error:  {model3.bse['treat_post']:.4f}
  95% CI:          [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]
  P-value:         {model3.pvalues['treat_post']:.4f}
  Sample Size:     {int(model3.nobs):,}

INTERPRETATION:
  DACA eligibility is associated with a {model3.params['treat_post']*100:.2f} percentage point
  {'increase' if model3.params['treat_post'] > 0 else 'decrease'} in the probability of full-time employment
  among Hispanic-Mexican, Mexican-born non-citizens who arrived before age 16
  and have been in the US since at least 2007.

  This effect is {'statistically significant' if model3.pvalues['treat_post'] < 0.05 else 'not statistically significant'}
  at the 5% level (p = {model3.pvalues['treat_post']:.4f}).
""")

# Save final results for report
final_results = {
    'effect_size': model3.params['treat_post'],
    'std_error': model3.bse['treat_post'],
    'ci_lower': model3.conf_int().loc['treat_post', 0],
    'ci_upper': model3.conf_int().loc['treat_post', 1],
    'p_value': model3.pvalues['treat_post'],
    'sample_size': int(model3.nobs),
    'model': 'Model 3 (Demographics + Education)'
}
pd.Series(final_results).to_csv('final_results.csv')
print("\nSaved: final_results.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
