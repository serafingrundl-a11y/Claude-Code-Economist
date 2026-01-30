"""
DACA Replication Study - Robustness Checks and Additional Analyses
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Create variables
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)
df['ED_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['ED_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['ED_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['ED_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Year dummies
for year in df['YEAR'].unique():
    if year != 2008:
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

# State dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)

year_vars = [col for col in df.columns if col.startswith('YEAR_')]
state_vars = [col for col in df.columns if col.startswith('STATE_')]

print("=" * 80)
print("ROBUSTNESS CHECKS AND ADDITIONAL ANALYSES")
print("=" * 80)

# ============================================================================
# 1. HETEROGENEOUS EFFECTS BY GENDER
# ============================================================================

print("\n1. HETEROGENEOUS EFFECTS BY GENDER")
print("-" * 40)

# Male only
df_male = df[df['FEMALE'] == 0]
formula_male = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + ED_BA + AGE + ' + ' + '.join(year_vars) + ' + ' + ' + '.join([s for s in state_vars if s in df_male.columns])
model_male = smf.ols(formula_male, data=df_male).fit(cov_type='HC1')

# Female only
df_female = df[df['FEMALE'] == 1]
formula_female = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + ED_BA + AGE + ' + ' + '.join(year_vars) + ' + ' + ' + '.join([s for s in state_vars if s in df_female.columns])
model_female = smf.ols(formula_female, data=df_female).fit(cov_type='HC1')

print(f"\nMale sample (N={int(model_male.nobs):,}):")
print(f"  DiD estimate: {model_male.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Std error: {model_male.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model_male.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_male.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  p-value: {model_male.pvalues['ELIGIBLE_AFTER']:.4f}")

print(f"\nFemale sample (N={int(model_female.nobs):,}):")
print(f"  DiD estimate: {model_female.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Std error: {model_female.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model_female.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_female.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  p-value: {model_female.pvalues['ELIGIBLE_AFTER']:.4f}")

# ============================================================================
# 2. HETEROGENEOUS EFFECTS BY EDUCATION
# ============================================================================

print("\n2. HETEROGENEOUS EFFECTS BY EDUCATION")
print("-" * 40)

# Less than BA
df_no_ba = df[df['ED_BA'] == 0]
formula_no_ba = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + AGE + ' + ' + '.join(year_vars) + ' + ' + ' + '.join([s for s in state_vars if s in df_no_ba.columns])
model_no_ba = smf.ols(formula_no_ba, data=df_no_ba).fit(cov_type='HC1')

# BA or higher
df_ba = df[df['ED_BA'] == 1]
if len(df_ba) > 100:
    formula_ba = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + ' + ' + '.join(year_vars)
    model_ba = smf.ols(formula_ba, data=df_ba).fit(cov_type='HC1')
    print(f"\nBA+ sample (N={int(model_ba.nobs):,}):")
    print(f"  DiD estimate: {model_ba.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  Std error: {model_ba.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"  p-value: {model_ba.pvalues['ELIGIBLE_AFTER']:.4f}")

print(f"\nLess than BA sample (N={int(model_no_ba.nobs):,}):")
print(f"  DiD estimate: {model_no_ba.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Std error: {model_no_ba.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model_no_ba.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_no_ba.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  p-value: {model_no_ba.pvalues['ELIGIBLE_AFTER']:.4f}")

# ============================================================================
# 3. PLACEBO TEST - PRE-TREATMENT TRENDS
# ============================================================================

print("\n3. PLACEBO TEST - PRE-TREATMENT PERIOD")
print("-" * 40)

# Use only pre-treatment data and define placebo treatment at 2010
df_pre = df[df['AFTER'] == 0].copy()
df_pre['PLACEBO_AFTER'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['ELIGIBLE_PLACEBO'] = df_pre['ELIGIBLE'] * df_pre['PLACEBO_AFTER']

# Placebo test
formula_placebo = 'FT ~ ELIGIBLE + PLACEBO_AFTER + ELIGIBLE_PLACEBO + FEMALE + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + ED_BA + AGE'
model_placebo = smf.ols(formula_placebo, data=df_pre).fit(cov_type='HC1')

print(f"\nPlacebo test (pretending treatment starts in 2010):")
print(f"  Sample: Pre-treatment period only (2008-2011)")
print(f"  N = {int(model_placebo.nobs):,}")
print(f"  Placebo DiD estimate: {model_placebo.params['ELIGIBLE_PLACEBO']:.4f}")
print(f"  Std error: {model_placebo.bse['ELIGIBLE_PLACEBO']:.4f}")
print(f"  95% CI: [{model_placebo.conf_int().loc['ELIGIBLE_PLACEBO', 0]:.4f}, {model_placebo.conf_int().loc['ELIGIBLE_PLACEBO', 1]:.4f}]")
print(f"  p-value: {model_placebo.pvalues['ELIGIBLE_PLACEBO']:.4f}")
print(f"  (Null hypothesis: no differential pre-trends)")

# ============================================================================
# 4. EVENT STUDY ANALYSIS
# ============================================================================

print("\n4. EVENT STUDY ANALYSIS")
print("-" * 40)

# Create year-specific treatment effects (relative to 2011, the last pre-treatment year)
years_to_analyze = [2008, 2009, 2010, 2013, 2014, 2015, 2016]  # 2011 is reference
for year in years_to_analyze:
    df[f'YEAR_{year}_ELIG'] = ((df['YEAR'] == year) & (df['ELIGIBLE'] == 1)).astype(int)

year_elig_vars = [f'YEAR_{y}_ELIG' for y in years_to_analyze]
formula_event = 'FT ~ ELIGIBLE + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(year_elig_vars) + ' + FEMALE + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + ED_BA + AGE + ' + ' + '.join(state_vars)
model_event = smf.ols(formula_event, data=df).fit(cov_type='HC1')

print("\nEvent study coefficients (reference: 2011):")
print(f"{'Year':<10} {'Coef':>10} {'SE':>10} {'95% CI':>25}")
print("-" * 55)
for year in years_to_analyze:
    var = f'YEAR_{year}_ELIG'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_low = model_event.conf_int().loc[var, 0]
    ci_high = model_event.conf_int().loc[var, 1]
    print(f"{year:<10} {coef:>10.4f} {se:>10.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}]")

# ============================================================================
# 5. ALTERNATIVE SPECIFICATIONS
# ============================================================================

print("\n5. ALTERNATIVE SPECIFICATIONS")
print("-" * 40)

# 5a. Probit model
print("\n5a. Probit model:")
try:
    probit_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + ED_BA + AGE'
    model_probit = smf.probit(probit_formula, data=df).fit(disp=False)

    # Calculate marginal effect at mean
    margeff = model_probit.get_margeff(at='mean')
    print(f"  DiD coefficient: {model_probit.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  Marginal effect at mean: {margeff.margeff[2]:.4f}")  # ELIGIBLE_AFTER is 3rd var
except:
    print("  Probit model did not converge")

# 5b. Logit model
print("\n5b. Logit model:")
try:
    logit_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + ED_BA + AGE'
    model_logit = smf.logit(logit_formula, data=df).fit(disp=False)

    # Calculate marginal effect at mean
    margeff_logit = model_logit.get_margeff(at='mean')
    print(f"  DiD coefficient: {model_logit.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  Marginal effect at mean: {margeff_logit.margeff[2]:.4f}")
except:
    print("  Logit model did not converge")

# 5c. Without state fixed effects
print("\n5c. Without state fixed effects:")
formula_no_state = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + ED_BA + AGE + ' + ' + '.join(year_vars)
model_no_state = smf.ols(formula_no_state, data=df).fit(cov_type='HC1')
print(f"  DiD estimate: {model_no_state.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Std error: {model_no_state.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model_no_state.pvalues['ELIGIBLE_AFTER']:.4f}")

# ============================================================================
# 6. BALANCE TABLE
# ============================================================================

print("\n6. BALANCE TABLE (Pre-treatment characteristics)")
print("-" * 40)

pre_treatment = df[df['AFTER'] == 0]
balance_vars = ['AGE', 'FEMALE', 'MARRIED', 'ED_HS', 'ED_SOMECOLL', 'ED_TWOYEAR', 'ED_BA']

print(f"\n{'Variable':<15} {'Treated':>10} {'Control':>10} {'Diff':>10} {'p-value':>10}")
print("-" * 55)

for var in balance_vars:
    treated_mean = pre_treatment[pre_treatment['ELIGIBLE'] == 1][var].mean()
    control_mean = pre_treatment[pre_treatment['ELIGIBLE'] == 0][var].mean()
    diff = treated_mean - control_mean

    # T-test
    t_stat, p_val = stats.ttest_ind(
        pre_treatment[pre_treatment['ELIGIBLE'] == 1][var],
        pre_treatment[pre_treatment['ELIGIBLE'] == 0][var]
    )

    print(f"{var:<15} {treated_mean:>10.3f} {control_mean:>10.3f} {diff:>10.3f} {p_val:>10.4f}")

# ============================================================================
# 7. CREATE FIGURES
# ============================================================================

print("\n7. CREATING FIGURES")
print("-" * 40)

# Figure 1: Trends in full-time employment
fig1, ax1 = plt.subplots(figsize=(10, 6))

years = sorted(df['YEAR'].unique())
treated_means = []
control_means = []

for year in years:
    year_data = df[df['YEAR'] == year]
    treated_means.append(year_data[year_data['ELIGIBLE'] == 1]['FT'].mean())
    control_means.append(year_data[year_data['ELIGIBLE'] == 0]['FT'].mean())

ax1.plot(years, treated_means, 'o-', label='Treated (Age 26-30 in 2012)', color='blue', linewidth=2, markersize=8)
ax1.plot(years, control_means, 's--', label='Control (Age 31-35 in 2012)', color='red', linewidth=2, markersize=8)
ax1.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Trends by Treatment Group', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.set_ylim(0.5, 0.8)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(years)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure1_trends.png")

# Figure 2: Event study coefficients
fig2, ax2 = plt.subplots(figsize=(10, 6))

coefs = []
ses = []
for year in years_to_analyze:
    var = f'YEAR_{year}_ELIG'
    coefs.append(model_event.params[var])
    ses.append(model_event.bse[var])

# Add 2011 as reference (coefficient = 0)
all_years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
all_coefs = coefs[:3] + [0] + coefs[3:]
all_ses = ses[:3] + [0] + ses[3:]

ax2.errorbar(all_years, all_coefs, yerr=[1.96*s for s in all_ses], fmt='o-',
             color='blue', capsize=5, capthick=2, linewidth=2, markersize=8)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Treatment Effect by Year', fontsize=14)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(all_years)
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure2_event_study.png")

# Figure 3: DiD visualization
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Pre and post means
pre_treated = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treated = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

# Plot lines
ax3.plot([0, 1], [pre_treated, post_treated], 'o-', color='blue', linewidth=2, markersize=10, label='Treated (Age 26-30)')
ax3.plot([0, 1], [pre_control, post_control], 's--', color='red', linewidth=2, markersize=10, label='Control (Age 31-35)')

# Counterfactual line
counterfactual = pre_treated + (post_control - pre_control)
ax3.plot([0, 1], [pre_treated, counterfactual], ':', color='blue', linewidth=2, alpha=0.5, label='Treated Counterfactual')

# DiD arrow
ax3.annotate('', xy=(1, post_treated), xytext=(1, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.05, (post_treated + counterfactual)/2, f'DiD = {post_treated - counterfactual:.3f}',
         fontsize=12, color='green', va='center')

ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Before DACA\n(2008-2011)', 'After DACA\n(2013-2016)'])
ax3.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences Visualization', fontsize=14)
ax3.legend(loc='lower right', fontsize=10)
ax3.set_ylim(0.5, 0.75)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure3_did_visual.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure3_did_visual.png")

# ============================================================================
# 8. SAVE ROBUSTNESS RESULTS
# ============================================================================

print("\n8. SAVING RESULTS")
print("-" * 40)

# Event study results
event_results = pd.DataFrame({
    'Year': years_to_analyze,
    'Coefficient': [model_event.params[f'YEAR_{y}_ELIG'] for y in years_to_analyze],
    'Std_Error': [model_event.bse[f'YEAR_{y}_ELIG'] for y in years_to_analyze],
    'CI_Lower': [model_event.conf_int().loc[f'YEAR_{y}_ELIG', 0] for y in years_to_analyze],
    'CI_Upper': [model_event.conf_int().loc[f'YEAR_{y}_ELIG', 1] for y in years_to_analyze]
})
event_results.to_csv('event_study_results.csv', index=False)
print("  Saved event_study_results.csv")

# Balance table results
balance_results = []
for var in balance_vars:
    treated_mean = pre_treatment[pre_treatment['ELIGIBLE'] == 1][var].mean()
    control_mean = pre_treatment[pre_treatment['ELIGIBLE'] == 0][var].mean()
    diff = treated_mean - control_mean
    t_stat, p_val = stats.ttest_ind(
        pre_treatment[pre_treatment['ELIGIBLE'] == 1][var],
        pre_treatment[pre_treatment['ELIGIBLE'] == 0][var]
    )
    balance_results.append({
        'Variable': var,
        'Treated_Mean': treated_mean,
        'Control_Mean': control_mean,
        'Difference': diff,
        'p_value': p_val
    })
balance_df = pd.DataFrame(balance_results)
balance_df.to_csv('balance_table.csv', index=False)
print("  Saved balance_table.csv")

# Heterogeneity results
hetero_results = pd.DataFrame({
    'Subgroup': ['Male', 'Female', 'Less than BA'],
    'DiD_Estimate': [model_male.params['ELIGIBLE_AFTER'],
                     model_female.params['ELIGIBLE_AFTER'],
                     model_no_ba.params['ELIGIBLE_AFTER']],
    'Std_Error': [model_male.bse['ELIGIBLE_AFTER'],
                  model_female.bse['ELIGIBLE_AFTER'],
                  model_no_ba.bse['ELIGIBLE_AFTER']],
    'N': [int(model_male.nobs), int(model_female.nobs), int(model_no_ba.nobs)]
})
hetero_results.to_csv('heterogeneity_results.csv', index=False)
print("  Saved heterogeneity_results.csv")

print("\n" + "=" * 80)
print("ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 80)
