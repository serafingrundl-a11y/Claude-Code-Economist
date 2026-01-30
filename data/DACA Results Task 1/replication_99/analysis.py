"""
DACA Replication Analysis - ID 99
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexico-born individuals
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
print("DACA REPLICATION ANALYSIS - ID 99")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations loaded: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")
print(f"   Columns: {list(df.columns)}")

# ============================================================================
# 2. SAMPLE RESTRICTION TO TARGET POPULATION
# ============================================================================
print("\n2. Restricting sample to target population...")

# Step 2a: Hispanic-Mexican ethnicity
print(f"   Initial sample: {len(df):,}")
df_target = df[df['HISPAN'] == 1].copy()
print(f"   After Hispanic-Mexican filter (HISPAN==1): {len(df_target):,}")

# Step 2b: Born in Mexico
df_target = df_target[df_target['BPL'] == 200].copy()
print(f"   After Mexico birthplace filter (BPL==200): {len(df_target):,}")

# Step 2c: Non-citizens (potential undocumented)
# CITIZEN == 3 means "Not a citizen"
df_target = df_target[df_target['CITIZEN'] == 3].copy()
print(f"   After non-citizen filter (CITIZEN==3): {len(df_target):,}")

# Step 2d: Working age population (16-64) for employment analysis
df_target = df_target[(df_target['AGE'] >= 16) & (df_target['AGE'] <= 64)].copy()
print(f"   After working age filter (16-64): {len(df_target):,}")

# Step 2e: Exclude 2012 (mid-year implementation creates ambiguity)
df_target = df_target[df_target['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_target):,}")

# ============================================================================
# 3. CONSTRUCT DACA ELIGIBILITY INDICATOR
# ============================================================================
print("\n3. Constructing DACA eligibility indicator...")

# Calculate age at arrival
# Using YRIMMIG as year of immigration
df_target['age_at_arrival'] = df_target['YRIMMIG'] - df_target['BIRTHYR']

# DACA eligibility criteria:
# 1. Arrived before 16th birthday: age_at_arrival < 16
# 2. Born on or after June 15, 1981 (not yet 31 on June 15, 2012)
# 3. Arrived by June 15, 2007 (continuous presence requirement) - use YRIMMIG <= 2007
# 4. Already filtered for non-citizens

# Create DACA eligibility indicator
df_target['daca_eligible'] = (
    (df_target['age_at_arrival'] < 16) &                    # Arrived before 16
    (df_target['age_at_arrival'] >= 0) &                     # Valid age at arrival
    (df_target['BIRTHYR'] >= 1981) &                         # Not yet 31 on June 15, 2012
    (df_target['YRIMMIG'] <= 2007) &                         # In US since at least 2007
    (df_target['YRIMMIG'] > 0)                               # Valid immigration year
).astype(int)

print(f"   DACA eligible observations: {df_target['daca_eligible'].sum():,}")
print(f"   Non-eligible observations: {(df_target['daca_eligible']==0).sum():,}")

# ============================================================================
# 4. CONSTRUCT OUTCOME AND TREATMENT VARIABLES
# ============================================================================
print("\n4. Constructing outcome and treatment variables...")

# Full-time employment: UHRSWORK >= 35 (usual hours worked per week)
df_target['fulltime'] = (df_target['UHRSWORK'] >= 35).astype(int)

# Post-DACA indicator (2013 onwards)
df_target['post'] = (df_target['YEAR'] >= 2013).astype(int)

# Interaction term for DiD
df_target['daca_x_post'] = df_target['daca_eligible'] * df_target['post']

print(f"   Full-time employed: {df_target['fulltime'].sum():,} ({100*df_target['fulltime'].mean():.1f}%)")
print(f"   Post-DACA period observations: {df_target['post'].sum():,}")

# ============================================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n5. Descriptive Statistics")
print("="*80)

# Summary by treatment group
print("\n5a. Sample sizes by group and period:")
summary = df_target.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['count', 'mean'],
    'PERWT': 'sum'
}).round(4)
summary.columns = ['N', 'FT_Rate', 'Pop_Weight']
print(summary)

# Pre-treatment means
print("\n5b. Pre-treatment characteristics by eligibility status:")
pre_df = df_target[df_target['post'] == 0]
pre_stats = pre_df.groupby('daca_eligible').agg({
    'AGE': 'mean',
    'fulltime': 'mean',
    'UHRSWORK': 'mean',
    'EDUC': 'mean',
    'SEX': lambda x: (x==1).mean(),  # Male proportion
    'PERWT': 'sum'
}).round(3)
pre_stats.columns = ['Mean_Age', 'FT_Rate', 'Mean_Hours', 'Mean_Educ', 'Male_Pct', 'Pop_Weight']
print(pre_stats)

# ============================================================================
# 6. MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n6. Main DiD Analysis")
print("="*80)

# 6a. Simple DiD without controls
print("\n6a. Basic DiD (no controls):")
model1 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post', data=df_target).fit()
print(f"   DiD estimate (daca_x_post): {model1.params['daca_x_post']:.4f}")
print(f"   Standard Error: {model1.bse['daca_x_post']:.4f}")
print(f"   p-value: {model1.pvalues['daca_x_post']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['daca_x_post', 0]:.4f}, {model1.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"   N: {int(model1.nobs):,}")

# 6b. DiD with demographic controls
print("\n6b. DiD with demographic controls:")
# Create control variables
df_target['male'] = (df_target['SEX'] == 1).astype(int)
df_target['age_sq'] = df_target['AGE'] ** 2

# Education categories
df_target['educ_hs'] = (df_target['EDUC'] >= 6).astype(int)  # High school or more
df_target['educ_college'] = (df_target['EDUC'] >= 10).astype(int)  # College or more

model2 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + male + educ_hs + educ_college',
                  data=df_target).fit()
print(f"   DiD estimate (daca_x_post): {model2.params['daca_x_post']:.4f}")
print(f"   Standard Error: {model2.bse['daca_x_post']:.4f}")
print(f"   p-value: {model2.pvalues['daca_x_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['daca_x_post', 0]:.4f}, {model2.conf_int().loc['daca_x_post', 1]:.4f}]")

# 6c. DiD with year and state fixed effects
print("\n6c. DiD with year and state fixed effects:")
# Create year dummies
year_dummies = pd.get_dummies(df_target['YEAR'], prefix='year', drop_first=True)
state_dummies = pd.get_dummies(df_target['STATEFIP'], prefix='state', drop_first=True)

df_model = pd.concat([df_target[['fulltime', 'daca_eligible', 'post', 'daca_x_post',
                                  'AGE', 'age_sq', 'male', 'educ_hs', 'educ_college']],
                      year_dummies, state_dummies], axis=1)
df_model = df_model.dropna()

# Prepare formula with fixed effects
year_vars = ' + '.join(year_dummies.columns)
state_vars = ' + '.join(state_dummies.columns)
formula3 = f'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + male + educ_hs + educ_college + {year_vars} + {state_vars}'

model3 = smf.ols(formula3, data=df_model).fit()
print(f"   DiD estimate (daca_x_post): {model3.params['daca_x_post']:.4f}")
print(f"   Standard Error: {model3.bse['daca_x_post']:.4f}")
print(f"   p-value: {model3.pvalues['daca_x_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['daca_x_post', 0]:.4f}, {model3.conf_int().loc['daca_x_post', 1]:.4f}]")

# 6d. Weighted regression (using PERWT)
print("\n6d. DiD with controls and weights (PERWT):")
model4 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + male + educ_hs + educ_college',
                  data=df_target, weights=df_target['PERWT']).fit()
print(f"   DiD estimate (daca_x_post): {model4.params['daca_x_post']:.4f}")
print(f"   Standard Error: {model4.bse['daca_x_post']:.4f}")
print(f"   p-value: {model4.pvalues['daca_x_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['daca_x_post', 0]:.4f}, {model4.conf_int().loc['daca_x_post', 1]:.4f}]")

# ============================================================================
# 7. ROBUST STANDARD ERRORS
# ============================================================================
print("\n7. Preferred Model with Robust Standard Errors")
print("="*80)

# Model with robust (HC1) standard errors
model_robust = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + male + educ_hs + educ_college',
                        data=df_target).fit(cov_type='HC1')
print(f"\n   DiD estimate (daca_x_post): {model_robust.params['daca_x_post']:.4f}")
print(f"   Robust SE: {model_robust.bse['daca_x_post']:.4f}")
print(f"   t-statistic: {model_robust.tvalues['daca_x_post']:.4f}")
print(f"   p-value: {model_robust.pvalues['daca_x_post']:.4f}")
print(f"   95% CI: [{model_robust.conf_int().loc['daca_x_post', 0]:.4f}, {model_robust.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"   R-squared: {model_robust.rsquared:.4f}")
print(f"   N: {int(model_robust.nobs):,}")

# ============================================================================
# 8. ROBUSTNESS CHECKS
# ============================================================================
print("\n8. Robustness Checks")
print("="*80)

# 8a. Alternative age restrictions
print("\n8a. Restricted to ages 18-35 (core DACA-eligible ages):")
df_young = df_target[(df_target['AGE'] >= 18) & (df_target['AGE'] <= 35)]
model_young = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + male + educ_hs + educ_college',
                       data=df_young).fit(cov_type='HC1')
print(f"   DiD estimate: {model_young.params['daca_x_post']:.4f} (SE: {model_young.bse['daca_x_post']:.4f})")
print(f"   p-value: {model_young.pvalues['daca_x_post']:.4f}")
print(f"   N: {int(model_young.nobs):,}")

# 8b. Placebo test: Pre-trends using 2009 as fake treatment
print("\n8b. Placebo test (fake treatment in 2009):")
df_pre = df_target[df_target['YEAR'] <= 2011].copy()
df_pre['post_fake'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['daca_x_post_fake'] = df_pre['daca_eligible'] * df_pre['post_fake']
model_placebo = smf.ols('fulltime ~ daca_eligible + post_fake + daca_x_post_fake + AGE + age_sq + male + educ_hs + educ_college',
                         data=df_pre).fit(cov_type='HC1')
print(f"   Placebo DiD estimate: {model_placebo.params['daca_x_post_fake']:.4f} (SE: {model_placebo.bse['daca_x_post_fake']:.4f})")
print(f"   p-value: {model_placebo.pvalues['daca_x_post_fake']:.4f}")

# 8c. Event study - Year-by-year effects
print("\n8c. Event study (year-by-year effects):")
df_target['year_2006'] = (df_target['YEAR'] == 2006).astype(int)
df_target['year_2007'] = (df_target['YEAR'] == 2007).astype(int)
df_target['year_2008'] = (df_target['YEAR'] == 2008).astype(int)
df_target['year_2009'] = (df_target['YEAR'] == 2009).astype(int)
df_target['year_2010'] = (df_target['YEAR'] == 2010).astype(int)
# 2011 is reference year
df_target['year_2013'] = (df_target['YEAR'] == 2013).astype(int)
df_target['year_2014'] = (df_target['YEAR'] == 2014).astype(int)
df_target['year_2015'] = (df_target['YEAR'] == 2015).astype(int)
df_target['year_2016'] = (df_target['YEAR'] == 2016).astype(int)

# Interactions
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_target[f'daca_x_{year}'] = df_target['daca_eligible'] * df_target[f'year_{year}']

event_formula = 'fulltime ~ daca_eligible + ' + \
                ' + '.join([f'year_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]) + \
                ' + ' + ' + '.join([f'daca_x_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]) + \
                ' + AGE + age_sq + male + educ_hs + educ_college'

model_event = smf.ols(event_formula, data=df_target).fit(cov_type='HC1')
print("   Year-specific DACA effects (relative to 2011):")
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'daca_x_{year}']
    se = model_event.bse[f'daca_x_{year}']
    pval = model_event.pvalues[f'daca_x_{year}']
    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    print(f"   {year}: {coef:7.4f} ({se:.4f}) {sig}")

# ============================================================================
# 9. HETEROGENEITY ANALYSIS
# ============================================================================
print("\n9. Heterogeneity Analysis")
print("="*80)

# 9a. By sex
print("\n9a. Effect by sex:")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_target[df_target['SEX'] == sex_val]
    model_sex = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + educ_hs + educ_college',
                        data=df_sex).fit(cov_type='HC1')
    print(f"   {sex_name}: {model_sex.params['daca_x_post']:.4f} (SE: {model_sex.bse['daca_x_post']:.4f}), p={model_sex.pvalues['daca_x_post']:.4f}, N={int(model_sex.nobs):,}")

# 9b. By education
print("\n9b. Effect by education level:")
df_lowedu = df_target[df_target['educ_hs'] == 0]
df_highedu = df_target[df_target['educ_hs'] == 1]
model_lowedu = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + male',
                       data=df_lowedu).fit(cov_type='HC1')
model_highedu = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + male',
                        data=df_highedu).fit(cov_type='HC1')
print(f"   Less than HS: {model_lowedu.params['daca_x_post']:.4f} (SE: {model_lowedu.bse['daca_x_post']:.4f}), p={model_lowedu.pvalues['daca_x_post']:.4f}")
print(f"   HS or more: {model_highedu.params['daca_x_post']:.4f} (SE: {model_highedu.bse['daca_x_post']:.4f}), p={model_highedu.pvalues['daca_x_post']:.4f}")

# ============================================================================
# 10. SAVE RESULTS FOR REPORT
# ============================================================================
print("\n10. Saving results...")

# Create results summary
results_summary = {
    'Model': ['Basic DiD', 'With Controls', 'With FE', 'Weighted', 'Robust SE (Preferred)'],
    'Estimate': [model1.params['daca_x_post'], model2.params['daca_x_post'],
                 model3.params['daca_x_post'], model4.params['daca_x_post'],
                 model_robust.params['daca_x_post']],
    'SE': [model1.bse['daca_x_post'], model2.bse['daca_x_post'],
           model3.bse['daca_x_post'], model4.bse['daca_x_post'],
           model_robust.bse['daca_x_post']],
    'p_value': [model1.pvalues['daca_x_post'], model2.pvalues['daca_x_post'],
                model3.pvalues['daca_x_post'], model4.pvalues['daca_x_post'],
                model_robust.pvalues['daca_x_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model_robust.nobs)]
}
results_df = pd.DataFrame(results_summary)
results_df.to_csv('results_summary.csv', index=False)
print("   Results saved to results_summary.csv")

# Save full model output
with open('model_output.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("PREFERRED MODEL: DiD with Controls and Robust SEs\n")
    f.write("="*80 + "\n\n")
    f.write(model_robust.summary().as_text())
    f.write("\n\n")
    f.write("="*80 + "\n")
    f.write("EVENT STUDY MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(model_event.summary().as_text())
print("   Full model output saved to model_output.txt")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY - PREFERRED ESTIMATE")
print("="*80)
print(f"\nResearch Question: Effect of DACA eligibility on full-time employment")
print(f"Sample: Hispanic-Mexican, Mexico-born, non-citizen, ages 16-64")
print(f"Method: Difference-in-Differences")
print(f"\nPreferred Estimate (DiD coefficient):")
print(f"   Effect size: {model_robust.params['daca_x_post']:.4f}")
print(f"   Robust SE: {model_robust.bse['daca_x_post']:.4f}")
print(f"   95% CI: [{model_robust.conf_int().loc['daca_x_post', 0]:.4f}, {model_robust.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"   p-value: {model_robust.pvalues['daca_x_post']:.4f}")
print(f"   Sample size: {int(model_robust.nobs):,}")
print(f"\nInterpretation: DACA eligibility is associated with a")
print(f"   {model_robust.params['daca_x_post']*100:.2f} percentage point change in the probability")
print(f"   of full-time employment after program implementation.")
print("="*80)

# Save key statistics for LaTeX
with open('key_stats.txt', 'w') as f:
    f.write(f"main_effect,{model_robust.params['daca_x_post']:.4f}\n")
    f.write(f"main_se,{model_robust.bse['daca_x_post']:.4f}\n")
    f.write(f"main_pvalue,{model_robust.pvalues['daca_x_post']:.4f}\n")
    f.write(f"main_ci_low,{model_robust.conf_int().loc['daca_x_post', 0]:.4f}\n")
    f.write(f"main_ci_high,{model_robust.conf_int().loc['daca_x_post', 1]:.4f}\n")
    f.write(f"main_n,{int(model_robust.nobs)}\n")
    f.write(f"n_eligible,{df_target['daca_eligible'].sum()}\n")
    f.write(f"n_noneligible,{(df_target['daca_eligible']==0).sum()}\n")
    f.write(f"ft_rate_overall,{df_target['fulltime'].mean():.4f}\n")

print("\nAnalysis complete!")
