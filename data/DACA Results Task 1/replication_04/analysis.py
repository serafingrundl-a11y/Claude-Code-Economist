"""
DACA Replication Analysis - Replication 04
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born non-citizens in the US.

Identification: Difference-in-Differences comparing DACA-eligible to
age-ineligible non-citizen Mexican immigrants.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("="*80)
print("DACA REPLICATION ANALYSIS - REPLICATION 04")
print("="*80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load only necessary columns to save memory
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN',
               'YRIMMIG', 'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK']

print("Loading data...")
df = pd.read_csv('data/data.csv', usecols=cols_needed)
print(f"Total observations loaded: {len(df):,}")

# =============================================================================
# 2. SAMPLE SELECTION
# =============================================================================
print("\n" + "="*80)
print("STEP 2: SAMPLE SELECTION")
print("="*80)

# 2.1 Restrict to Hispanic-Mexican ethnicity
print(f"\nInitial sample: {len(df):,}")
df_mex = df[df['HISPAN'] == 1].copy()
print(f"After restricting to Hispanic-Mexican (HISPAN=1): {len(df_mex):,}")

# 2.2 Restrict to Mexico-born
df_mex = df_mex[df_mex['BPL'] == 200].copy()
print(f"After restricting to Mexico-born (BPL=200): {len(df_mex):,}")

# 2.3 Restrict to non-citizens (assumed undocumented per instructions)
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"After restricting to non-citizens (CITIZEN=3): {len(df_mex):,}")

# 2.4 Exclude 2012 (mid-year DACA implementation)
df_analysis = df_mex[df_mex['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

# 2.5 Restrict to working-age population (16-64)
df_analysis = df_analysis[(df_analysis['AGE'] >= 16) & (df_analysis['AGE'] <= 64)].copy()
print(f"After restricting to ages 16-64: {len(df_analysis):,}")

# =============================================================================
# 3. CREATE VARIABLES
# =============================================================================
print("\n" + "="*80)
print("STEP 3: CREATING VARIABLES")
print("="*80)

# 3.1 Calculate age at immigration
# YRIMMIG = 0 means N/A (likely native born, but we've already filtered to foreign-born)
df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0].copy()
print(f"After removing missing immigration year: {len(df_analysis):,}")

df_analysis['age_at_immig'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']

# 3.2 DACA Eligibility Criteria
# Criterion 1: Arrived before age 16
df_analysis['arrived_before_16'] = (df_analysis['age_at_immig'] < 16).astype(int)

# Criterion 2: Born after June 15, 1981 (under 31 as of June 15, 2012)
# Using BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# June 15 is in Q2, so conservatively: born in 1982+ OR (born 1981 and Q3 or Q4)
df_analysis['under_31_2012'] = ((df_analysis['BIRTHYR'] >= 1982) |
                                 ((df_analysis['BIRTHYR'] == 1981) &
                                  (df_analysis['BIRTHQTR'] >= 3))).astype(int)

# Criterion 3: Continuous presence since June 15, 2007 (immigrated by 2007)
df_analysis['continuous_presence'] = (df_analysis['YRIMMIG'] <= 2007).astype(int)

# 3.3 Combined DACA eligibility
df_analysis['daca_eligible'] = ((df_analysis['arrived_before_16'] == 1) &
                                 (df_analysis['under_31_2012'] == 1) &
                                 (df_analysis['continuous_presence'] == 1)).astype(int)

print(f"\nDACA eligibility breakdown:")
print(f"  Arrived before age 16: {df_analysis['arrived_before_16'].sum():,} ({df_analysis['arrived_before_16'].mean()*100:.1f}%)")
print(f"  Under 31 as of 2012: {df_analysis['under_31_2012'].sum():,} ({df_analysis['under_31_2012'].mean()*100:.1f}%)")
print(f"  Continuous presence (immigrated by 2007): {df_analysis['continuous_presence'].sum():,} ({df_analysis['continuous_presence'].mean()*100:.1f}%)")
print(f"  All criteria (DACA eligible): {df_analysis['daca_eligible'].sum():,} ({df_analysis['daca_eligible'].mean()*100:.1f}%)")

# 3.4 Post-DACA indicator (2013-2016)
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# 3.5 Difference-in-differences interaction
df_analysis['did'] = df_analysis['daca_eligible'] * df_analysis['post']

# 3.6 Outcome: Full-time employment (35+ hours per week)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# 3.7 Alternative outcome: Employed (EMPSTAT = 1)
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)

# 3.8 Control group definition
# Control group: Similar immigrants who are NOT DACA eligible due to age (older)
# Those who arrived before 16, by 2007, but were born before 1981 (too old for DACA)
df_analysis['control_eligible'] = ((df_analysis['arrived_before_16'] == 1) &
                                    (df_analysis['continuous_presence'] == 1) &
                                    (df_analysis['BIRTHYR'] < 1981)).astype(int)

# Restrict to either DACA eligible or control group (exclude those who don't meet other criteria)
df_did = df_analysis[(df_analysis['daca_eligible'] == 1) |
                      (df_analysis['control_eligible'] == 1)].copy()
print(f"\nAnalysis sample (DACA eligible + age-based control): {len(df_did):,}")
print(f"  Treatment group (DACA eligible): {df_did['daca_eligible'].sum():,}")
print(f"  Control group (age-ineligible): {(df_did['control_eligible'] == 1).sum():,}")

# 3.9 Create additional control variables
df_did['age_sq'] = df_did['AGE'] ** 2
df_did['female'] = (df_did['SEX'] == 2).astype(int)
df_did['married'] = (df_did['MARST'] == 1).astype(int)

# Education categories
df_did['educ_hs'] = (df_did['EDUC'] >= 6).astype(int)  # HS or higher
df_did['educ_college'] = (df_did['EDUC'] >= 10).astype(int)  # Some college or higher

print(f"\nOutcome variable (full-time employment):")
print(f"  Overall rate: {df_did['fulltime'].mean()*100:.1f}%")
print(f"  Treated (DACA eligible): {df_did[df_did['daca_eligible']==1]['fulltime'].mean()*100:.1f}%")
print(f"  Control (age-ineligible): {df_did[df_did['daca_eligible']==0]['fulltime'].mean()*100:.1f}%")

# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("STEP 4: DESCRIPTIVE STATISTICS")
print("="*80)

def weighted_mean(df, var, weight='PERWT'):
    return np.average(df[var], weights=df[weight])

def weighted_std(df, var, weight='PERWT'):
    avg = weighted_mean(df, var, weight)
    return np.sqrt(np.average((df[var] - avg)**2, weights=df[weight]))

# Summary statistics by treatment status
print("\n--- Weighted Summary Statistics by Treatment Status ---")
treat = df_did[df_did['daca_eligible'] == 1]
control = df_did[df_did['daca_eligible'] == 0]

stats_vars = ['AGE', 'female', 'married', 'educ_hs', 'fulltime', 'employed']
print(f"\n{'Variable':<20} {'Treated Mean':>15} {'Control Mean':>15} {'Difference':>12}")
print("-"*65)
for var in stats_vars:
    t_mean = weighted_mean(treat, var)
    c_mean = weighted_mean(control, var)
    diff = t_mean - c_mean
    print(f"{var:<20} {t_mean:>15.3f} {c_mean:>15.3f} {diff:>12.3f}")

# Pre/Post comparison
print("\n--- Full-time Employment by Treatment and Period ---")
pre = df_did[df_did['post'] == 0]
post = df_did[df_did['post'] == 1]

pre_treat = pre[pre['daca_eligible'] == 1]
pre_control = pre[pre['daca_eligible'] == 0]
post_treat = post[post['daca_eligible'] == 1]
post_control = post[post['daca_eligible'] == 0]

print(f"\n{'Group':<25} {'Pre-DACA':>15} {'Post-DACA':>15} {'Difference':>12}")
print("-"*70)
pre_t = weighted_mean(pre_treat, 'fulltime')
post_t = weighted_mean(post_treat, 'fulltime')
print(f"{'DACA Eligible (Treated)':<25} {pre_t:>15.3f} {post_t:>15.3f} {(post_t-pre_t):>12.3f}")

pre_c = weighted_mean(pre_control, 'fulltime')
post_c = weighted_mean(post_control, 'fulltime')
print(f"{'Age-Ineligible (Control)':<25} {pre_c:>15.3f} {post_c:>15.3f} {(post_c-pre_c):>12.3f}")

raw_did = (post_t - pre_t) - (post_c - pre_c)
print(f"\n{'Raw DiD Estimate:':<25} {raw_did:>15.3f}")

# =============================================================================
# 5. DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "="*80)
print("STEP 5: DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("="*80)

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls('fulltime ~ daca_eligible + post + did',
                  data=df_did, weights=df_did['PERWT']).fit()
print(model1.summary().tables[1])
print(f"\nDiD Coefficient: {model1.params['did']:.4f}")
print(f"Standard Error: {model1.bse['did']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['did', 0]:.4f}, {model1.conf_int().loc['did', 1]:.4f}]")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls('fulltime ~ daca_eligible + post + did + AGE + age_sq + female + married + educ_hs',
                  data=df_did, weights=df_did['PERWT']).fit()
print(model2.summary().tables[1])
print(f"\nDiD Coefficient: {model2.params['did']:.4f}")
print(f"Standard Error: {model2.bse['did']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['did', 0]:.4f}, {model2.conf_int().loc['did', 1]:.4f}]")

# Model 3: DiD with Year Fixed Effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
df_did['year_factor'] = pd.Categorical(df_did['YEAR'])
model3 = smf.wls('fulltime ~ daca_eligible + C(YEAR) + did + AGE + age_sq + female + married + educ_hs',
                  data=df_did, weights=df_did['PERWT']).fit()
print(f"DiD Coefficient: {model3.params['did']:.4f}")
print(f"Standard Error: {model3.bse['did']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['did', 0]:.4f}, {model3.conf_int().loc['did', 1]:.4f}]")

# Model 4: DiD with Year and State Fixed Effects (Preferred)
print("\n--- Model 4: DiD with Year and State Fixed Effects (PREFERRED) ---")
model4 = smf.wls('fulltime ~ daca_eligible + C(YEAR) + C(STATEFIP) + did + AGE + age_sq + female + married + educ_hs',
                  data=df_did, weights=df_did['PERWT']).fit()
print(f"DiD Coefficient: {model4.params['did']:.4f}")
print(f"Standard Error: {model4.bse['did']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['did', 0]:.4f}, {model4.conf_int().loc['did', 1]:.4f}]")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"Number of observations: {int(model4.nobs):,}")

# =============================================================================
# 6. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("STEP 6: ROBUSTNESS CHECKS")
print("="*80)

# 6.1 Alternative outcome: Employment (any)
print("\n--- Robustness: Employment (Any) as Outcome ---")
model_emp = smf.wls('employed ~ daca_eligible + C(YEAR) + C(STATEFIP) + did + AGE + age_sq + female + married + educ_hs',
                     data=df_did, weights=df_did['PERWT']).fit()
print(f"DiD Coefficient: {model_emp.params['did']:.4f}")
print(f"Standard Error: {model_emp.bse['did']:.4f}")
print(f"95% CI: [{model_emp.conf_int().loc['did', 0]:.4f}, {model_emp.conf_int().loc['did', 1]:.4f}]")

# 6.2 Subgroup: Males only
print("\n--- Robustness: Males Only ---")
df_male = df_did[df_did['female'] == 0]
model_male = smf.wls('fulltime ~ daca_eligible + C(YEAR) + C(STATEFIP) + did + AGE + age_sq + married + educ_hs',
                      data=df_male, weights=df_male['PERWT']).fit()
print(f"DiD Coefficient: {model_male.params['did']:.4f}")
print(f"Standard Error: {model_male.bse['did']:.4f}")
print(f"95% CI: [{model_male.conf_int().loc['did', 0]:.4f}, {model_male.conf_int().loc['did', 1]:.4f}]")
print(f"N: {int(model_male.nobs):,}")

# 6.3 Subgroup: Females only
print("\n--- Robustness: Females Only ---")
df_female = df_did[df_did['female'] == 1]
model_female = smf.wls('fulltime ~ daca_eligible + C(YEAR) + C(STATEFIP) + did + AGE + age_sq + married + educ_hs',
                        data=df_female, weights=df_female['PERWT']).fit()
print(f"DiD Coefficient: {model_female.params['did']:.4f}")
print(f"Standard Error: {model_female.bse['did']:.4f}")
print(f"95% CI: [{model_female.conf_int().loc['did', 0]:.4f}, {model_female.conf_int().loc['did', 1]:.4f}]")
print(f"N: {int(model_female.nobs):,}")

# 6.4 Event Study
print("\n--- Event Study: Year-by-Year Effects ---")
df_did['treat_X_year'] = df_did['daca_eligible'].astype(str) + '_' + df_did['YEAR'].astype(str)
# Reference year: 2011 (last pre-treatment year)
df_did['treat_2006'] = ((df_did['daca_eligible'] == 1) & (df_did['YEAR'] == 2006)).astype(int)
df_did['treat_2007'] = ((df_did['daca_eligible'] == 1) & (df_did['YEAR'] == 2007)).astype(int)
df_did['treat_2008'] = ((df_did['daca_eligible'] == 1) & (df_did['YEAR'] == 2008)).astype(int)
df_did['treat_2009'] = ((df_did['daca_eligible'] == 1) & (df_did['YEAR'] == 2009)).astype(int)
df_did['treat_2010'] = ((df_did['daca_eligible'] == 1) & (df_did['YEAR'] == 2010)).astype(int)
# 2011 is reference
df_did['treat_2013'] = ((df_did['daca_eligible'] == 1) & (df_did['YEAR'] == 2013)).astype(int)
df_did['treat_2014'] = ((df_did['daca_eligible'] == 1) & (df_did['YEAR'] == 2014)).astype(int)
df_did['treat_2015'] = ((df_did['daca_eligible'] == 1) & (df_did['YEAR'] == 2015)).astype(int)
df_did['treat_2016'] = ((df_did['daca_eligible'] == 1) & (df_did['YEAR'] == 2016)).astype(int)

model_event = smf.wls('fulltime ~ daca_eligible + C(YEAR) + C(STATEFIP) + treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2013 + treat_2014 + treat_2015 + treat_2016 + AGE + age_sq + female + married + educ_hs',
                       data=df_did, weights=df_did['PERWT']).fit()

print(f"\n{'Year':<10} {'Coefficient':>12} {'Std Err':>12} {'95% CI':>25}")
print("-"*65)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_low, ci_high = model_event.conf_int().loc[var]
    print(f"{year:<10} {coef:>12.4f} {se:>12.4f} [{ci_low:>10.4f}, {ci_high:>10.4f}]")
print("Reference year: 2011")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================
print("\n" + "="*80)
print("STEP 7: SAVING RESULTS")
print("="*80)

# Save key results to file
results = {
    'preferred_estimate': {
        'coefficient': model4.params['did'],
        'std_error': model4.bse['did'],
        'ci_lower': model4.conf_int().loc['did', 0],
        'ci_upper': model4.conf_int().loc['did', 1],
        'n_obs': int(model4.nobs),
        'r_squared': model4.rsquared
    },
    'raw_did': raw_did,
    'model1_basic': model1.params['did'],
    'model2_controls': model2.params['did'],
    'model3_yearfe': model3.params['did'],
    'model4_preferred': model4.params['did'],
    'robustness_employment': model_emp.params['did'],
    'robustness_male': model_male.params['did'],
    'robustness_female': model_female.params['did']
}

# Print final summary
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"\nPreferred Estimate (Model 4 with Year and State FE):")
print(f"  Effect of DACA eligibility on full-time employment: {results['preferred_estimate']['coefficient']:.4f}")
print(f"  Standard Error: {results['preferred_estimate']['std_error']:.4f}")
print(f"  95% Confidence Interval: [{results['preferred_estimate']['ci_lower']:.4f}, {results['preferred_estimate']['ci_upper']:.4f}]")
print(f"  Sample Size: {results['preferred_estimate']['n_obs']:,}")
print(f"  R-squared: {results['preferred_estimate']['r_squared']:.4f}")

print("\nInterpretation:")
if results['preferred_estimate']['coefficient'] > 0:
    print(f"  DACA eligibility increased the probability of full-time employment by")
    print(f"  {results['preferred_estimate']['coefficient']*100:.2f} percentage points among Hispanic-Mexican,")
    print(f"  Mexican-born non-citizens who met the eligibility criteria.")
else:
    print(f"  DACA eligibility decreased the probability of full-time employment by")
    print(f"  {abs(results['preferred_estimate']['coefficient'])*100:.2f} percentage points among Hispanic-Mexican,")
    print(f"  Mexican-born non-citizens who met the eligibility criteria.")

if results['preferred_estimate']['ci_lower'] > 0:
    print("  This effect is statistically significant at the 5% level.")
elif results['preferred_estimate']['ci_upper'] < 0:
    print("  This effect is statistically significant at the 5% level.")
else:
    print("  This effect is NOT statistically significant at the 5% level.")

# =============================================================================
# 8. CREATE DATA FOR FIGURES
# =============================================================================
print("\n" + "="*80)
print("STEP 8: CREATING DATA FOR FIGURES")
print("="*80)

# Trends by year and treatment status
trends = df_did.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'employed_rate': np.average(x['employed'], weights=x['PERWT']),
        'n': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()

print("\nTrends in Full-Time Employment by Year and Treatment Status:")
print(trends.pivot(index='YEAR', columns='daca_eligible', values='fulltime_rate').round(4))

# Save trends for plotting
trends.to_csv('trends_data.csv', index=False)
print("\nTrends data saved to trends_data.csv")

# Event study coefficients
event_study_results = []
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_{year}'
    event_study_results.append({
        'year': year,
        'coefficient': model_event.params[var],
        'std_error': model_event.bse[var],
        'ci_lower': model_event.conf_int().loc[var, 0],
        'ci_upper': model_event.conf_int().loc[var, 1]
    })
# Add reference year
event_study_results.append({
    'year': 2011,
    'coefficient': 0,
    'std_error': 0,
    'ci_lower': 0,
    'ci_upper': 0
})
event_df = pd.DataFrame(event_study_results).sort_values('year')
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Sample composition
print("\n--- Sample Composition ---")
print(f"Treatment group (DACA eligible): {len(df_did[df_did['daca_eligible']==1]):,}")
print(f"Control group (age-ineligible): {len(df_did[df_did['daca_eligible']==0]):,}")
print(f"Pre-period observations: {len(df_did[df_did['post']==0]):,}")
print(f"Post-period observations: {len(df_did[df_did['post']==1]):,}")

# Year distribution
print("\n--- Observations by Year ---")
print(df_did.groupby('YEAR').size())

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save a summary file with key statistics
summary_stats = {
    'Total_Sample': len(df_did),
    'Treatment_N': len(df_did[df_did['daca_eligible']==1]),
    'Control_N': len(df_did[df_did['daca_eligible']==0]),
    'Pre_Period_N': len(df_did[df_did['post']==0]),
    'Post_Period_N': len(df_did[df_did['post']==1]),
    'Pre_Treat_FT_Rate': weighted_mean(pre_treat, 'fulltime'),
    'Post_Treat_FT_Rate': weighted_mean(post_treat, 'fulltime'),
    'Pre_Control_FT_Rate': weighted_mean(pre_control, 'fulltime'),
    'Post_Control_FT_Rate': weighted_mean(post_control, 'fulltime'),
    'Raw_DiD': raw_did,
    'Model1_DiD': model1.params['did'],
    'Model2_DiD': model2.params['did'],
    'Model3_DiD': model3.params['did'],
    'Model4_DiD_Preferred': model4.params['did'],
    'Model4_SE': model4.bse['did'],
    'Model4_CI_Lower': model4.conf_int().loc['did', 0],
    'Model4_CI_Upper': model4.conf_int().loc['did', 1],
    'Model4_N': int(model4.nobs)
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv('analysis_summary.csv', index=False)
print("\nSummary statistics saved to analysis_summary.csv")
