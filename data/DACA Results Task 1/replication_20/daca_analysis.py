"""
DACA Replication Analysis
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on full-time employment (35+ hours/week)?

DACA eligibility criteria:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

# Load data in chunks and filter to relevant sample only
print("\n1. LOADING DATA (chunked processing for memory efficiency)...")

# Define columns we need
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT', 'UHRSWORK']

# Process in chunks, keeping only Hispanic-Mexican Mexican-born
chunks = []
chunk_size = 500000
total_rows = 0

for chunk in pd.read_csv('data/data.csv', chunksize=chunk_size, usecols=cols_needed):
    total_rows += len(chunk)
    # Filter to Hispanic-Mexican (HISPAN==1) and Mexican-born (BPL==200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"   Processed {total_rows:,} rows...", end='\r')

print(f"\n   Total rows processed: {total_rows:,}")

df_mex = pd.concat(chunks, ignore_index=True)
print(f"   Hispanic-Mexican Mexican-born observations: {len(df_mex):,}")

# Check key variables
print("\n2. DATA EXPLORATION...")
print(f"   Years in dataset: {sorted(df_mex['YEAR'].unique())}")
print(f"\n   Citizenship distribution in Mexican-born Hispanic sample:")
citizen_counts = df_mex['CITIZEN'].value_counts().sort_index()
for val, count in citizen_counts.items():
    pct = 100 * count / len(df_mex)
    print(f"      CITIZEN={val}: {count:,} ({pct:.1f}%)")

# Step 1: Define non-citizen status
print("\n3. DEFINING DACA ELIGIBILITY...")
print("   CITIZEN == 3 indicates non-citizen (potentially undocumented)")
print("   CITIZEN == 4 indicates received first papers (documented, not eligible)")

# Calculate age at June 15, 2012
def calc_age_june2012(row):
    """Calculate age as of June 15, 2012"""
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']
    # Q1 (Jan-Mar), Q2 (Apr-Jun) -> birthday passed by June 15
    # Q3, Q4 -> birthday not yet passed
    if birth_qtr in [1, 2]:
        age = 2012 - birth_year
    else:
        age = 2012 - birth_year - 1
    return age

df_mex['age_at_june2012'] = df_mex.apply(calc_age_june2012, axis=1)

# Calculate age at arrival
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

print("\n   DACA eligibility requirements:")
print("   - Arrived before 16th birthday")
print("   - Under age 31 as of June 15, 2012")
print("   - In US since June 15, 2007 (immigrated by 2007)")
print("   - Non-citizen without lawful status")

# Eligibility indicators
df_mex['noncitizen'] = (df_mex['CITIZEN'] == 3).astype(int)
df_mex['arrived_young'] = (df_mex['age_at_arrival'] < 16).astype(int)
df_mex['under_31_2012'] = (df_mex['age_at_june2012'] < 31).astype(int)
df_mex['arrived_by_2007'] = ((df_mex['YRIMMIG'] <= 2007) & (df_mex['YRIMMIG'] > 0)).astype(int)

# DACA eligible if all criteria met
df_mex['daca_eligible'] = (
    (df_mex['noncitizen'] == 1) &
    (df_mex['arrived_young'] == 1) &
    (df_mex['under_31_2012'] == 1) &
    (df_mex['arrived_by_2007'] == 1)
).astype(int)

print(f"\n   DACA eligible observations: {df_mex['daca_eligible'].sum():,} ({100*df_mex['daca_eligible'].mean():.1f}%)")

# Focus on non-citizens only for main comparison
print("\n4. DEFINING TREATMENT AND COMPARISON GROUPS...")
df_noncit = df_mex[df_mex['noncitizen'] == 1].copy()
print(f"   Non-citizen Mexican-born Hispanic sample: {len(df_noncit):,}")
print(f"   DACA eligible among non-citizens: {df_noncit['daca_eligible'].sum():,} ({100*df_noncit['daca_eligible'].mean():.1f}%)")

# Define outcome variable
print("\n5. DEFINING OUTCOME VARIABLE...")
print("   Full-time employment = UHRSWORK >= 35 (usually works 35+ hours per week)")

df_noncit['employed'] = (df_noncit['EMPSTAT'] == 1).astype(int)
df_noncit['fulltime'] = ((df_noncit['UHRSWORK'] >= 35) & (df_noncit['employed'] == 1)).astype(int)

print(f"   Employment rate: {100*df_noncit['employed'].mean():.1f}%")
print(f"   Full-time employment rate: {100*df_noncit['fulltime'].mean():.1f}%")

# Define time periods
print("\n6. DEFINING TIME PERIODS...")
df_noncit['post_daca'] = (df_noncit['YEAR'] >= 2013).astype(int)

# Exclude 2012 for cleaner identification
df_analysis = df_noncit[df_noncit['YEAR'] != 2012].copy()
print(f"   Excluding 2012 (treatment year with mixed before/after)")
print(f"   Pre-DACA years (2006-2011): {sorted(df_analysis[df_analysis['post_daca']==0]['YEAR'].unique())}")
print(f"   Post-DACA years (2013-2016): {sorted(df_analysis[df_analysis['post_daca']==1]['YEAR'].unique())}")
print(f"   Analysis sample size: {len(df_analysis):,}")

# Create control variables
print("\n7. PREPARING CONTROL VARIABLES...")

df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Education categories
df_analysis['educ_less_hs'] = (df_analysis['EDUC'] < 6).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] == 6).astype(int)
df_analysis['educ_some_college'] = (df_analysis['EDUC'].isin([7, 8, 9])).astype(int)
df_analysis['educ_college_plus'] = (df_analysis['EDUC'] >= 10).astype(int)

# Age squared
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Create interaction term for DiD
df_analysis['treat_post'] = df_analysis['daca_eligible'] * df_analysis['post_daca']

print(f"   Female: {100*df_analysis['female'].mean():.1f}%")
print(f"   Married: {100*df_analysis['married'].mean():.1f}%")
print(f"   Average age: {df_analysis['AGE'].mean():.1f}")

# Summary Statistics
print("\n" + "="*80)
print("8. SUMMARY STATISTICS")
print("="*80)

def weighted_mean(df, var, weight='PERWT'):
    return np.average(df[var], weights=df[weight])

summary_vars = ['fulltime', 'employed', 'AGE', 'female', 'married',
                'educ_less_hs', 'educ_hs', 'educ_some_college', 'educ_college_plus']

print("\n   Pre-DACA Period (2006-2011):")
pre = df_analysis[df_analysis['post_daca'] == 0]
pre_treat = pre[pre['daca_eligible'] == 1]
pre_control = pre[pre['daca_eligible'] == 0]

print(f"\n   {'Variable':<25} {'Eligible':<15} {'Not Eligible':<15} {'Difference':<15}")
print("   " + "-"*70)
for var in summary_vars:
    t_mean = weighted_mean(pre_treat, var)
    c_mean = weighted_mean(pre_control, var)
    diff = t_mean - c_mean
    print(f"   {var:<25} {t_mean:>12.3f}   {c_mean:>12.3f}   {diff:>12.3f}")

print(f"\n   N (DACA eligible): {len(pre_treat):,}")
print(f"   N (Not eligible): {len(pre_control):,}")

print("\n   Post-DACA Period (2013-2016):")
post = df_analysis[df_analysis['post_daca'] == 1]
post_treat = post[post['daca_eligible'] == 1]
post_control = post[post['daca_eligible'] == 0]

print(f"\n   {'Variable':<25} {'Eligible':<15} {'Not Eligible':<15} {'Difference':<15}")
print("   " + "-"*70)
for var in summary_vars:
    t_mean = weighted_mean(post_treat, var)
    c_mean = weighted_mean(post_control, var)
    diff = t_mean - c_mean
    print(f"   {var:<25} {t_mean:>12.3f}   {c_mean:>12.3f}   {diff:>12.3f}")

print(f"\n   N (DACA eligible): {len(post_treat):,}")
print(f"   N (Not eligible): {len(post_control):,}")

# Simple 2x2 DiD
print("\n" + "="*80)
print("9. DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

print("\n   Simple 2x2 DiD (unweighted):")
pre_treat_ft = pre_treat['fulltime'].mean()
pre_control_ft = pre_control['fulltime'].mean()
post_treat_ft = post_treat['fulltime'].mean()
post_control_ft = post_control['fulltime'].mean()

diff_treat = post_treat_ft - pre_treat_ft
diff_control = post_control_ft - pre_control_ft
did_estimate = diff_treat - diff_control

print(f"\n   Full-time employment rates:")
print(f"   {'Group':<20} {'Pre-DACA':<15} {'Post-DACA':<15} {'Difference':<15}")
print("   " + "-"*65)
print(f"   {'DACA Eligible':<20} {pre_treat_ft:>12.4f}   {post_treat_ft:>12.4f}   {diff_treat:>12.4f}")
print(f"   {'Not Eligible':<20} {pre_control_ft:>12.4f}   {post_control_ft:>12.4f}   {diff_control:>12.4f}")
print("   " + "-"*65)
print(f"   {'DiD Estimate':<20} {'':<15} {'':<15} {did_estimate:>12.4f}")

# Regression-based DiD
print("\n   Regression-based DiD Analysis:")

# Model 1: Basic DiD without controls
print("\n   Model 1: Basic DiD (no controls)")
model1 = smf.wls('fulltime ~ daca_eligible + post_daca + treat_post',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient (treat_post): {model1.params['treat_post']:.4f}")
print(f"   Standard error: {model1.bse['treat_post']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model1.pvalues['treat_post']:.4f}")

# Model 2: DiD with demographic controls
print("\n   Model 2: DiD with demographic controls")
model2 = smf.wls('fulltime ~ daca_eligible + post_daca + treat_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient (treat_post): {model2.params['treat_post']:.4f}")
print(f"   Standard error: {model2.bse['treat_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model2.pvalues['treat_post']:.4f}")

# Model 3: DiD with state fixed effects
print("\n   Model 3: DiD with state fixed effects")
model3 = smf.wls('fulltime ~ daca_eligible + post_daca + treat_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(STATEFIP)',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"   Standard error: {model3.bse['treat_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: Full model with year and state FEs
print("\n   Model 4: DiD with year and state fixed effects (PREFERRED)")
model4 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(STATEFIP) + C(YEAR)',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"   Standard error: {model4.bse['treat_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model4.pvalues['treat_post']:.4f}")
print(f"   R-squared: {model4.rsquared:.4f}")
print(f"   N: {int(model4.nobs):,}")

# Save preferred model results
preferred_coef = model4.params['treat_post']
preferred_se = model4.bse['treat_post']
preferred_ci = model4.conf_int().loc['treat_post']
preferred_pval = model4.pvalues['treat_post']
preferred_n = int(model4.nobs)

# Robustness Checks
print("\n" + "="*80)
print("10. ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Employment (any) as outcome
print("\n   Robustness 1: Employment (any) as outcome")
model_emp = smf.wls('employed ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(STATEFIP) + C(YEAR)',
                     data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_emp.params['treat_post']:.4f}")
print(f"   Standard error: {model_emp.bse['treat_post']:.4f}")
print(f"   p-value: {model_emp.pvalues['treat_post']:.4f}")

# Robustness 2: Working age restriction (18-64)
print("\n   Robustness 2: Working age restriction (18-64)")
df_work_age = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 64)]
model_age = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(STATEFIP) + C(YEAR)',
                     data=df_work_age, weights=df_work_age['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_age.params['treat_post']:.4f}")
print(f"   Standard error: {model_age.bse['treat_post']:.4f}")
print(f"   p-value: {model_age.pvalues['treat_post']:.4f}")
print(f"   N: {int(model_age.nobs):,}")

# Robustness 3: Placebo test (fake treatment in 2009)
print("\n   Robustness 3: Placebo test (fake treatment in 2009)")
df_placebo = df_analysis[(df_analysis['YEAR'] <= 2011)].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treat_post_placebo'] = df_placebo['daca_eligible'] * df_placebo['post_placebo']
model_placebo = smf.wls('fulltime ~ daca_eligible + post_placebo + treat_post_placebo + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(STATEFIP) + C(YEAR)',
                         data=df_placebo, weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"   Placebo DiD coefficient: {model_placebo.params['treat_post_placebo']:.4f}")
print(f"   Standard error: {model_placebo.bse['treat_post_placebo']:.4f}")
print(f"   p-value: {model_placebo.pvalues['treat_post_placebo']:.4f}")

# Robustness 4: Comparison with naturalized citizens
print("\n   Robustness 4: Comparison with naturalized citizens as control")
df_citizen_compare = df_mex[df_mex['YEAR'] != 2012].copy()
df_citizen_compare = df_citizen_compare[df_citizen_compare['CITIZEN'].isin([2, 3])].copy()
df_citizen_compare['post_daca'] = (df_citizen_compare['YEAR'] >= 2013).astype(int)
df_citizen_compare['treatment'] = ((df_citizen_compare['CITIZEN'] == 3) &
                                    (df_citizen_compare['arrived_young'] == 1) &
                                    (df_citizen_compare['under_31_2012'] == 1) &
                                    (df_citizen_compare['arrived_by_2007'] == 1)).astype(int)
df_citizen_compare['treat_post_cit'] = df_citizen_compare['treatment'] * df_citizen_compare['post_daca']
df_citizen_compare['employed'] = (df_citizen_compare['EMPSTAT'] == 1).astype(int)
df_citizen_compare['fulltime'] = ((df_citizen_compare['UHRSWORK'] >= 35) & (df_citizen_compare['employed'] == 1)).astype(int)
df_citizen_compare['female'] = (df_citizen_compare['SEX'] == 2).astype(int)
df_citizen_compare['married'] = (df_citizen_compare['MARST'].isin([1, 2])).astype(int)
df_citizen_compare['educ_hs'] = (df_citizen_compare['EDUC'] == 6).astype(int)
df_citizen_compare['educ_some_college'] = (df_citizen_compare['EDUC'].isin([7, 8, 9])).astype(int)
df_citizen_compare['educ_college_plus'] = (df_citizen_compare['EDUC'] >= 10).astype(int)
df_citizen_compare['age_sq'] = df_citizen_compare['AGE'] ** 2

model_cit = smf.wls('fulltime ~ treatment + treat_post_cit + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(STATEFIP) + C(YEAR)',
                     data=df_citizen_compare, weights=df_citizen_compare['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_cit.params['treat_post_cit']:.4f}")
print(f"   Standard error: {model_cit.bse['treat_post_cit']:.4f}")
print(f"   p-value: {model_cit.pvalues['treat_post_cit']:.4f}")

# Event Study
print("\n" + "="*80)
print("11. EVENT STUDY ANALYSIS")
print("="*80)

for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype(int)
    df_analysis[f'treat_x_{year}'] = df_analysis['daca_eligible'] * df_analysis[f'year_{year}']

model_event = smf.wls('fulltime ~ daca_eligible + treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(STATEFIP) + C(YEAR)',
                       data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\n   Event study coefficients (reference year: 2011):")
print(f"   {'Year':<10} {'Coefficient':<15} {'Std Error':<15} {'95% CI':<25}")
print("   " + "-"*65)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_x_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_low, ci_high = model_event.conf_int().loc[var]
    print(f"   {year:<10} {coef:>12.4f}   {se:>12.4f}   [{ci_low:.4f}, {ci_high:.4f}]")

# Subgroup Analysis
print("\n" + "="*80)
print("12. SUBGROUP ANALYSIS")
print("="*80)

# By gender
print("\n   By Gender:")
for gender, name in [(0, 'Male'), (1, 'Female')]:
    df_sub = df_analysis[df_analysis['female'] == gender]
    model_sub = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + married + educ_hs + educ_some_college + educ_college_plus + C(STATEFIP) + C(YEAR)',
                         data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"   {name}: DiD = {model_sub.params['treat_post']:.4f} (SE = {model_sub.bse['treat_post']:.4f}), N = {int(model_sub.nobs):,}")

# By education
print("\n   By Education Level:")
for educ_var, educ_name in [('educ_less_hs', 'Less than HS'), ('educ_hs', 'High School'),
                             ('educ_some_college', 'Some College'), ('educ_college_plus', 'College+')]:
    df_sub = df_analysis[df_analysis[educ_var] == 1]
    if len(df_sub) > 1000:
        try:
            model_sub = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + C(STATEFIP) + C(YEAR)',
                                 data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
            print(f"   {educ_name}: DiD = {model_sub.params['treat_post']:.4f} (SE = {model_sub.bse['treat_post']:.4f}), N = {int(model_sub.nobs):,}")
        except:
            print(f"   {educ_name}: Could not estimate (insufficient variation)")
    else:
        print(f"   {educ_name}: Insufficient sample size (N = {len(df_sub)})")

# Final Summary
print("\n" + "="*80)
print("13. FINAL RESULTS SUMMARY")
print("="*80)

print(f"\n   PREFERRED ESTIMATE (Model 4: Full DiD with Year and State FEs):")
print(f"   " + "-"*60)
print(f"   Effect of DACA eligibility on full-time employment:")
print(f"   Coefficient: {preferred_coef:.4f}")
print(f"   Standard Error: {preferred_se:.4f}")
print(f"   95% Confidence Interval: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"   p-value: {preferred_pval:.4f}")
print(f"   Sample Size: {preferred_n:,}")

print(f"\n   Interpretation:")
if preferred_pval < 0.05:
    direction = "increased" if preferred_coef > 0 else "decreased"
    print(f"   DACA eligibility is associated with a statistically significant")
    print(f"   {abs(preferred_coef)*100:.2f} percentage point {direction} in the probability")
    print(f"   of full-time employment among Mexican-born non-citizen Hispanics.")
else:
    print(f"   The effect of DACA eligibility on full-time employment is not")
    print(f"   statistically significant at the 5% level.")

# Save results
results_df = pd.DataFrame({
    'Model': ['(1) Basic DiD', '(2) + Demographics', '(3) + State FEs', '(4) + Year FEs'],
    'Coefficient': [model1.params['treat_post'], model2.params['treat_post'],
                    model3.params['treat_post'], model4.params['treat_post']],
    'Std_Error': [model1.bse['treat_post'], model2.bse['treat_post'],
                  model3.bse['treat_post'], model4.bse['treat_post']],
    'CI_Lower': [model1.conf_int().loc['treat_post', 0], model2.conf_int().loc['treat_post', 0],
                 model3.conf_int().loc['treat_post', 0], model4.conf_int().loc['treat_post', 0]],
    'CI_Upper': [model1.conf_int().loc['treat_post', 1], model2.conf_int().loc['treat_post', 1],
                 model3.conf_int().loc['treat_post', 1], model4.conf_int().loc['treat_post', 1]],
    'P_value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
                model3.pvalues['treat_post'], model4.pvalues['treat_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs)],
    'R2': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared]
})

results_df.to_csv('regression_results.csv', index=False)
print("\n   Regression results saved to regression_results.csv")

# Save summary statistics
summary_stats = {
    'total_obs_original': total_rows,
    'mexican_born_hispanic': len(df_mex),
    'noncitizen_sample': len(df_noncit),
    'analysis_sample': len(df_analysis),
    'daca_eligible': int(df_analysis['daca_eligible'].sum()),
    'pct_eligible': 100*df_analysis['daca_eligible'].mean(),
    'pre_period_obs': len(pre),
    'post_period_obs': len(post),
    'mean_fulltime': df_analysis['fulltime'].mean(),
    'mean_employed': df_analysis['employed'].mean(),
    'mean_age': df_analysis['AGE'].mean(),
    'pct_female': 100*df_analysis['female'].mean(),
    'pct_married': 100*df_analysis['married'].mean(),
    'pre_treat_ft': pre_treat_ft,
    'pre_control_ft': pre_control_ft,
    'post_treat_ft': post_treat_ft,
    'post_control_ft': post_control_ft,
    'simple_did': did_estimate
}

pd.DataFrame([summary_stats]).to_csv('summary_statistics.csv', index=False)
print("   Summary statistics saved to summary_statistics.csv")

# Save event study results
event_results = []
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_x_{year}'
    event_results.append({
        'year': year,
        'coefficient': model_event.params[var],
        'std_error': model_event.bse[var],
        'ci_lower': model_event.conf_int().loc[var, 0],
        'ci_upper': model_event.conf_int().loc[var, 1],
        'pvalue': model_event.pvalues[var]
    })
event_results.append({'year': 2011, 'coefficient': 0, 'std_error': 0, 'ci_lower': 0, 'ci_upper': 0, 'pvalue': 1})
event_df = pd.DataFrame(event_results).sort_values('year')
event_df.to_csv('event_study_results.csv', index=False)
print("   Event study results saved to event_study_results.csv")

# Save robustness check results
robustness_results = pd.DataFrame({
    'Check': ['Any Employment', 'Working Age 18-64', 'Placebo 2009', 'vs Naturalized Citizens'],
    'Coefficient': [model_emp.params['treat_post'], model_age.params['treat_post'],
                    model_placebo.params['treat_post_placebo'], model_cit.params['treat_post_cit']],
    'Std_Error': [model_emp.bse['treat_post'], model_age.bse['treat_post'],
                  model_placebo.bse['treat_post_placebo'], model_cit.bse['treat_post_cit']],
    'P_value': [model_emp.pvalues['treat_post'], model_age.pvalues['treat_post'],
                model_placebo.pvalues['treat_post_placebo'], model_cit.pvalues['treat_post_cit']]
})
robustness_results.to_csv('robustness_results.csv', index=False)
print("   Robustness check results saved to robustness_results.csv")

# Save subgroup analysis
subgroup_results = []
for gender, name in [(0, 'Male'), (1, 'Female')]:
    df_sub = df_analysis[df_analysis['female'] == gender]
    model_sub = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + married + educ_hs + educ_some_college + educ_college_plus + C(STATEFIP) + C(YEAR)',
                         data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    subgroup_results.append({'Group': name, 'Coefficient': model_sub.params['treat_post'],
                             'Std_Error': model_sub.bse['treat_post'], 'N': int(model_sub.nobs)})

subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv('subgroup_results.csv', index=False)
print("   Subgroup analysis results saved to subgroup_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
