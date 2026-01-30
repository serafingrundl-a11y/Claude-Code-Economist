"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the US.

Identification Strategy: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DACA REPLICATION ANALYSIS")
print("=" * 60)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n1. Loading data...")
dtype_dict = {
    'YEAR': 'int32',
    'STATEFIP': 'int16',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8',
    'MARST': 'int8',
    'PERWT': 'float64'
}

# Read only necessary columns
cols_to_use = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT',
               'UHRSWORK', 'MARST']

df = pd.read_csv('data/data.csv', usecols=cols_to_use, dtype=dtype_dict)
print(f"Total observations loaded: {len(df):,}")

# =============================================================================
# 2. SAMPLE RESTRICTIONS
# =============================================================================
print("\n2. Applying sample restrictions...")

# Working age (16-64)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)]
print(f"After age restriction (16-64): {len(df):,}")

# Hispanic-Mexican ethnicity
df = df[df['HISPAN'] == 1]
print(f"After Hispanic-Mexican restriction: {len(df):,}")

# Born in Mexico
df = df[df['BPL'] == 200]
print(f"After Mexico birthplace restriction: {len(df):,}")

# Non-citizens (CITIZEN == 3: not a citizen, or 5: foreign born citizenship unknown)
df = df[(df['CITIZEN'] == 3) | (df['CITIZEN'] == 5)]
print(f"After non-citizen restriction: {len(df):,}")

# Valid immigration year (must have immigrated)
df = df[(df['YRIMMIG'] > 0) & (df['YRIMMIG'] <= df['YEAR'])]
print(f"After valid immigration year: {len(df):,}")

# Exclude 2012 (DACA implemented mid-year)
df = df[df['YEAR'] != 2012]
print(f"After excluding 2012: {len(df):,}")

# =============================================================================
# 3. CONSTRUCT VARIABLES
# =============================================================================
print("\n3. Constructing variables...")

# Age at immigration (approximate - using year of immigration)
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']

# Correct for those who immigrated in birth year or before
df.loc[df['age_at_immigration'] < 0, 'age_at_immigration'] = 0

# DACA eligibility criteria
# 1. Arrived before 16th birthday
df['arrived_before_16'] = (df['age_at_immigration'] < 16).astype(int)

# 2. Not yet 31 as of June 15, 2012 (born after June 15, 1981)
# Conservative: born 1982 or later definitely qualifies
# Born 1981: only if born after June 15
df['under_31_in_2012'] = ((df['BIRTHYR'] >= 1982) |
                          ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))).astype(int)

# 3. Continuous residence since June 15, 2007 (arrived by 2007)
df['arrived_by_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# DACA eligible: meets all criteria (and is non-citizen, already filtered)
df['daca_eligible'] = ((df['arrived_before_16'] == 1) &
                       (df['under_31_in_2012'] == 1) &
                       (df['arrived_by_2007'] == 1)).astype(int)

# Post-DACA period (2013 onwards)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term (DiD estimator)
df['daca_x_post'] = df['daca_eligible'] * df['post']

# Outcome: Full-time employment (35+ hours/week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Education categories
df['educ_cat'] = pd.cut(df['EDUC'], bins=[-1, 2, 6, 8, 11],
                        labels=['less_hs', 'hs', 'some_college', 'college_plus'])

# Age squared
df['age_sq'] = df['AGE'] ** 2

# Female indicator
df['female'] = (df['SEX'] == 2).astype(int)

# Married indicator
df['married'] = (df['MARST'] <= 2).astype(int)

print(f"DACA eligible: {df['daca_eligible'].sum():,} ({df['daca_eligible'].mean()*100:.1f}%)")
print(f"Post period: {df['post'].sum():,} ({df['post'].mean()*100:.1f}%)")
print(f"Full-time employed: {df['fulltime'].sum():,} ({df['fulltime'].mean()*100:.1f}%)")

# =============================================================================
# 4. SUMMARY STATISTICS
# =============================================================================
print("\n4. Summary Statistics")
print("-" * 60)

# By DACA eligibility and period
summary = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'PERWT': 'sum'
}).round(3)

print("\nSummary by DACA eligibility and period:")
print(summary)

# Pre-post comparison by group
print("\n" + "=" * 60)
print("Full-time employment rates:")
print("=" * 60)
pre_inelig = df[(df['daca_eligible'] == 0) & (df['post'] == 0)]['fulltime'].mean()
post_inelig = df[(df['daca_eligible'] == 0) & (df['post'] == 1)]['fulltime'].mean()
pre_elig = df[(df['daca_eligible'] == 1) & (df['post'] == 0)]['fulltime'].mean()
post_elig = df[(df['daca_eligible'] == 1) & (df['post'] == 1)]['fulltime'].mean()

print(f"\nIneligible group:")
print(f"  Pre-DACA (2006-2011):  {pre_inelig:.4f}")
print(f"  Post-DACA (2013-2016): {post_inelig:.4f}")
print(f"  Change:                {post_inelig - pre_inelig:.4f}")

print(f"\nEligible group:")
print(f"  Pre-DACA (2006-2011):  {pre_elig:.4f}")
print(f"  Post-DACA (2013-2016): {post_elig:.4f}")
print(f"  Change:                {post_elig - pre_elig:.4f}")

did_simple = (post_elig - pre_elig) - (post_inelig - pre_inelig)
print(f"\nSimple DiD estimate: {did_simple:.4f}")

# =============================================================================
# 5. MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("5. REGRESSION ANALYSIS")
print("=" * 60)

# Create year dummies
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
df = pd.concat([df, year_dummies], axis=1)

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)

# Create education dummies
educ_dummies = pd.get_dummies(df['educ_cat'], prefix='educ', drop_first=True)
df = pd.concat([df, educ_dummies], axis=1)

# Model 1: Basic DiD
print("\nModel 1: Basic DiD")
model1 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model1.params['daca_x_post']:.4f}")
print(f"Standard error:  {model1.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['daca_x_post', 0]:.4f}, {model1.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"N = {int(model1.nobs):,}")

# Model 2: Add demographics
print("\nModel 2: Add demographics (age, sex, married)")
model2 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model2.params['daca_x_post']:.4f}")
print(f"Standard error:  {model2.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['daca_x_post', 0]:.4f}, {model2.conf_int().loc['daca_x_post', 1]:.4f}]")

# Model 3: Add education
year_cols = [c for c in df.columns if c.startswith('year_')]
educ_cols = [c for c in df.columns if c.startswith('educ_')]

formula3 = 'fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + ' + ' + '.join(educ_cols)
print("\nModel 3: Add education")
model3 = smf.wls(formula3, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model3.params['daca_x_post']:.4f}")
print(f"Standard error:  {model3.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['daca_x_post', 0]:.4f}, {model3.conf_int().loc['daca_x_post', 1]:.4f}]")

# Model 4: Add year fixed effects
formula4 = 'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + ' + ' + '.join(educ_cols) + ' + ' + ' + '.join(year_cols)
print("\nModel 4: Add year fixed effects")
model4 = smf.wls(formula4, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model4.params['daca_x_post']:.4f}")
print(f"Standard error:  {model4.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['daca_x_post', 0]:.4f}, {model4.conf_int().loc['daca_x_post', 1]:.4f}]")

# Model 5: Full model with state FE (main specification)
state_cols = [c for c in df.columns if c.startswith('state_')]
formula5 = 'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + ' + ' + '.join(educ_cols) + ' + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)
print("\nModel 5: Full specification (state + year FE)")
model5 = smf.wls(formula5, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model5.params['daca_x_post']:.4f}")
print(f"Standard error:  {model5.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['daca_x_post', 0]:.4f}, {model5.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"p-value: {model5.pvalues['daca_x_post']:.4f}")

# =============================================================================
# 6. EVENT STUDY
# =============================================================================
print("\n" + "=" * 60)
print("6. EVENT STUDY")
print("=" * 60)

# Create year-specific treatment effects
years = sorted(df['YEAR'].unique())
base_year = 2011  # Last pre-treatment year

for y in years:
    if y != base_year:
        df[f'daca_x_{y}'] = (df['daca_eligible'] * (df['YEAR'] == y)).astype(int)

event_vars = [f'daca_x_{y}' for y in years if y != base_year]
formula_event = 'fulltime ~ daca_eligible + ' + ' + '.join(event_vars) + ' + AGE + age_sq + female + married + ' + ' + '.join(educ_cols) + ' + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)

model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent study coefficients (relative to 2011):")
event_results = []
for y in years:
    if y == base_year:
        event_results.append({'year': y, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
    else:
        var = f'daca_x_{y}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        ci = model_event.conf_int().loc[var]
        event_results.append({'year': y, 'coef': coef, 'se': se, 'ci_low': ci[0], 'ci_high': ci[1]})
        print(f"  {y}: {coef:.4f} (SE: {se:.4f})")

event_df = pd.DataFrame(event_results)

# =============================================================================
# 7. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 60)
print("7. ROBUSTNESS CHECKS")
print("=" * 60)

# 7a. Restrict to ages 18-30 (prime DACA-eligible ages)
print("\n7a. Restricted sample: Ages 18-30")
df_young = df[(df['AGE'] >= 18) & (df['AGE'] <= 30)]
formula_young = 'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + ' + ' + '.join(educ_cols) + ' + ' + ' + '.join(year_cols) + ' + ' + ' + '.join([c for c in state_cols if c in df_young.columns])
model_young = smf.wls(formula_young, data=df_young, weights=df_young['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_young.params['daca_x_post']:.4f}")
print(f"Standard error:  {model_young.bse['daca_x_post']:.4f}")
print(f"N = {int(model_young.nobs):,}")

# 7b. Males only
print("\n7b. Males only")
df_male = df[df['female'] == 0]
formula_male = 'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + married + ' + ' + '.join(educ_cols) + ' + ' + ' + '.join(year_cols) + ' + ' + ' + '.join([c for c in state_cols if c in df_male.columns])
model_male = smf.wls(formula_male, data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_male.params['daca_x_post']:.4f}")
print(f"Standard error:  {model_male.bse['daca_x_post']:.4f}")

# 7c. Females only
print("\n7c. Females only")
df_female = df[df['female'] == 1]
formula_female = 'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + married + ' + ' + '.join(educ_cols) + ' + ' + ' + '.join(year_cols) + ' + ' + ' + '.join([c for c in state_cols if c in df_female.columns])
model_female = smf.wls(formula_female, data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_female.params['daca_x_post']:.4f}")
print(f"Standard error:  {model_female.bse['daca_x_post']:.4f}")

# 7d. Employment (any) instead of full-time
print("\n7d. Outcome: Any employment (instead of full-time)")
formula_emp = 'employed ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + ' + ' + '.join(educ_cols) + ' + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)
model_emp = smf.wls(formula_emp, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_emp.params['daca_x_post']:.4f}")
print(f"Standard error:  {model_emp.bse['daca_x_post']:.4f}")

# =============================================================================
# 8. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n" + "=" * 60)
print("8. SAVING RESULTS")
print("=" * 60)

# Save key results
results_dict = {
    'main_did_coef': model5.params['daca_x_post'],
    'main_did_se': model5.bse['daca_x_post'],
    'main_did_ci_low': model5.conf_int().loc['daca_x_post', 0],
    'main_did_ci_high': model5.conf_int().loc['daca_x_post', 1],
    'main_did_pvalue': model5.pvalues['daca_x_post'],
    'n_obs': int(model5.nobs),
    'simple_did': did_simple,
    'pre_elig': pre_elig,
    'post_elig': post_elig,
    'pre_inelig': pre_inelig,
    'post_inelig': post_inelig,
    'n_eligible': int(df['daca_eligible'].sum()),
    'n_ineligible': int((1-df['daca_eligible']).sum()),
    'model1_coef': model1.params['daca_x_post'],
    'model1_se': model1.bse['daca_x_post'],
    'model2_coef': model2.params['daca_x_post'],
    'model2_se': model2.bse['daca_x_post'],
    'model3_coef': model3.params['daca_x_post'],
    'model3_se': model3.bse['daca_x_post'],
    'model4_coef': model4.params['daca_x_post'],
    'model4_se': model4.bse['daca_x_post'],
    'robust_young_coef': model_young.params['daca_x_post'],
    'robust_young_se': model_young.bse['daca_x_post'],
    'robust_male_coef': model_male.params['daca_x_post'],
    'robust_male_se': model_male.bse['daca_x_post'],
    'robust_female_coef': model_female.params['daca_x_post'],
    'robust_female_se': model_female.bse['daca_x_post'],
    'robust_emp_coef': model_emp.params['daca_x_post'],
    'robust_emp_se': model_emp.bse['daca_x_post'],
}

# Save results to file
import json
with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("Results saved to results.json")

# Save event study results
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# =============================================================================
# 9. CREATE FIGURES
# =============================================================================
print("\n9. Creating figures...")

# Figure 1: Event study
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(event_df['year'], event_df['coef'],
            yerr=[event_df['coef'] - event_df['ci_low'],
                  event_df['ci_high'] - event_df['coef']],
            fmt='o-', capsize=5, capthick=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
ax.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation (2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Effect on Full-Time Employment (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend()
ax.set_xticks(years)
plt.tight_layout()
plt.savefig('figure_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_event_study.pdf', bbox_inches='tight')
print("Event study figure saved")

# Figure 2: Trends by group
fig, ax = plt.subplots(figsize=(10, 6))
trends = df.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
trends.columns = ['Ineligible', 'DACA Eligible']
trends.plot(ax=ax, marker='o', markersize=8, linewidth=2)
ax.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA (2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility', fontsize=14)
ax.legend()
ax.set_xticks(years)
plt.tight_layout()
plt.savefig('figure_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_trends.pdf', bbox_inches='tight')
print("Trends figure saved")

# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\nPreferred Estimate (Full specification with state and year FE):")
print(f"  Effect of DACA eligibility on full-time employment: {model5.params['daca_x_post']:.4f}")
print(f"  Standard error: {model5.bse['daca_x_post']:.4f}")
print(f"  95% Confidence Interval: [{model5.conf_int().loc['daca_x_post', 0]:.4f}, {model5.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"  p-value: {model5.pvalues['daca_x_post']:.4f}")
print(f"  Sample size: {int(model5.nobs):,}")
print(f"\nInterpretation:")
if model5.params['daca_x_post'] > 0:
    print(f"  DACA eligibility is associated with a {model5.params['daca_x_post']*100:.2f} percentage point")
    print(f"  increase in the probability of full-time employment.")
else:
    print(f"  DACA eligibility is associated with a {abs(model5.params['daca_x_post'])*100:.2f} percentage point")
    print(f"  decrease in the probability of full-time employment.")

if model5.pvalues['daca_x_post'] < 0.05:
    print("  This effect is statistically significant at the 5% level.")
else:
    print("  This effect is NOT statistically significant at the 5% level.")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
