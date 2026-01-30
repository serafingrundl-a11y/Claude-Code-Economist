"""
DACA Replication Analysis Script - Replication 31
Difference-in-Differences Analysis of DACA Impact on Full-Time Employment

Research Question:
Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA (treatment) on the probability
of full-time employment (35+ hours/week)?

Treatment: Ages 26-30 at June 15, 2012 (ELIGIBLE=1)
Control: Ages 31-35 at June 15, 2012 (ELIGIBLE=0)
Pre-period: 2008-2011 (AFTER=0)
Post-period: 2013-2016 (AFTER=1)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r'C:\Users\seraf\DACA Results Task 3\replication_31')

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('tables', exist_ok=True)

print("="*70)
print("DACA REPLICATION ANALYSIS - Replication 31")
print("="*70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n" + "="*70)
print("1. LOADING DATA")
print("="*70)

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Dataset shape: {df.shape}")
print(f"Total observations: {len(df):,}")

# Handle missing education values - fill with mode or create indicator
df['EDUC_RECODE'] = df['EDUC_RECODE'].fillna('Unknown')

# Create numeric education variable
educ_map = {
    'Less than High School': 1,
    'High School Degree': 2,
    'Some College': 3,
    'Two-Year Degree': 4,
    'BA+': 5,
    'Unknown': 0
}
df['EDUC_NUM'] = df['EDUC_RECODE'].map(educ_map)

# =============================================================================
# 2. DATA EXPLORATION
# =============================================================================
print("\n" + "="*70)
print("2. DATA EXPLORATION")
print("="*70)

# Check key variables
print("\n--- Key Variables Summary ---")
print(f"\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())
print(f"\nAFTER distribution:")
print(df['AFTER'].value_counts())
print(f"\nFT (Full-time) distribution:")
print(df['FT'].value_counts())

# Year distribution
print(f"\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

# Check AGE_IN_JUNE_2012 for treatment/control
print(f"\nAGE_IN_JUNE_2012 by ELIGIBLE:")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe())

# Check SEX coding (1=Male, 2=Female in IPUMS)
print(f"\nSEX distribution:")
print(df['SEX'].value_counts())

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*70)
print("3. DESCRIPTIVE STATISTICS")
print("="*70)

# Create summary statistics table
def weighted_mean(x, weights):
    """Calculate weighted mean"""
    return np.average(x, weights=weights)

def weighted_std(x, weights):
    """Calculate weighted standard deviation"""
    avg = np.average(x, weights=weights)
    var = np.average((x - avg)**2, weights=weights)
    return np.sqrt(var)

# Summary by group (ELIGIBLE) and period (AFTER)
groups = df.groupby(['ELIGIBLE', 'AFTER'])

summary_stats = []
for (elig, after), group in groups:
    n = len(group)
    n_weighted = group['PERWT'].sum()
    ft_mean = weighted_mean(group['FT'], group['PERWT'])
    ft_std = weighted_std(group['FT'], group['PERWT'])
    age_mean = weighted_mean(group['AGE'], group['PERWT'])

    summary_stats.append({
        'Group': 'Treatment' if elig == 1 else 'Control',
        'Period': 'Post' if after == 1 else 'Pre',
        'N (unweighted)': n,
        'N (weighted)': int(n_weighted),
        'FT Rate': ft_mean,
        'FT SD': ft_std,
        'Mean Age': age_mean
    })

summary_df = pd.DataFrame(summary_stats)
print("\n--- Summary Statistics by Group and Period ---")
print(summary_df.to_string(index=False))

# Calculate simple DiD
print("\n--- Simple Difference-in-Differences ---")
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]
pre_ctrl = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]
post_ctrl = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]

ft_pre_treat = weighted_mean(pre_treat['FT'], pre_treat['PERWT'])
ft_post_treat = weighted_mean(post_treat['FT'], post_treat['PERWT'])
ft_pre_ctrl = weighted_mean(pre_ctrl['FT'], pre_ctrl['PERWT'])
ft_post_ctrl = weighted_mean(post_ctrl['FT'], post_ctrl['PERWT'])

diff_treat = ft_post_treat - ft_pre_treat
diff_ctrl = ft_post_ctrl - ft_pre_ctrl
did_simple = diff_treat - diff_ctrl

print(f"\nTreatment group (Ages 26-30):")
print(f"  Pre-DACA FT rate:  {ft_pre_treat:.4f} ({ft_pre_treat*100:.2f}%)")
print(f"  Post-DACA FT rate: {ft_post_treat:.4f} ({ft_post_treat*100:.2f}%)")
print(f"  Change:            {diff_treat:.4f} ({diff_treat*100:.2f} pp)")

print(f"\nControl group (Ages 31-35):")
print(f"  Pre-DACA FT rate:  {ft_pre_ctrl:.4f} ({ft_pre_ctrl*100:.2f}%)")
print(f"  Post-DACA FT rate: {ft_post_ctrl:.4f} ({ft_post_ctrl*100:.2f}%)")
print(f"  Change:            {diff_ctrl:.4f} ({diff_ctrl*100:.2f} pp)")

print(f"\nSimple DiD estimate: {did_simple:.4f} ({did_simple*100:.2f} percentage points)")

# =============================================================================
# 4. BASIC DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "="*70)
print("4. BASIC DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("="*70)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Create female indicator (SEX=2 in IPUMS is female)
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Basic DiD without covariates, with clustered SEs at state level
print("\n--- Model 1: Basic DiD (no covariates) ---")
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster',
                                           cov_kwds={'groups': df['STATEFIP']})
print(model1.summary())

# Save key results
did_coef_basic = model1.params['ELIGIBLE_AFTER']
did_se_basic = model1.bse['ELIGIBLE_AFTER']
did_pval_basic = model1.pvalues['ELIGIBLE_AFTER']
did_ci_basic = model1.conf_int().loc['ELIGIBLE_AFTER']

print(f"\n*** Basic DiD Estimate ***")
print(f"Coefficient: {did_coef_basic:.4f}")
print(f"Standard Error: {did_se_basic:.4f}")
print(f"95% CI: [{did_ci_basic[0]:.4f}, {did_ci_basic[1]:.4f}]")
print(f"P-value: {did_pval_basic:.4f}")

# =============================================================================
# 5. DiD WITH DEMOGRAPHIC CONTROLS
# =============================================================================
print("\n" + "="*70)
print("5. DiD WITH DEMOGRAPHIC CONTROLS")
print("="*70)

# Model with individual-level covariates using numeric variables
print("\n--- Model 2: DiD with demographic controls ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + C(MARST) + C(EDUC_NUM)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster',
                                           cov_kwds={'groups': df['STATEFIP']})
print(model2.summary())

did_coef_demo = model2.params['ELIGIBLE_AFTER']
did_se_demo = model2.bse['ELIGIBLE_AFTER']
did_pval_demo = model2.pvalues['ELIGIBLE_AFTER']
did_ci_demo = model2.conf_int().loc['ELIGIBLE_AFTER']

print(f"\n*** DiD Estimate with Demographics ***")
print(f"Coefficient: {did_coef_demo:.4f}")
print(f"Standard Error: {did_se_demo:.4f}")
print(f"95% CI: [{did_ci_demo[0]:.4f}, {did_ci_demo[1]:.4f}]")

# =============================================================================
# 6. DiD WITH STATE AND YEAR FIXED EFFECTS
# =============================================================================
print("\n" + "="*70)
print("6. DiD WITH STATE AND YEAR FIXED EFFECTS")
print("="*70)

# Model with state and year fixed effects
print("\n--- Model 3: DiD with state and year fixed effects ---")
# Note: AFTER is absorbed by YEAR fixed effects, so we drop it
model3 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + AGE + C(MARST) + C(EDUC_NUM) + C(STATEFIP) + C(YEAR)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster',
                                           cov_kwds={'groups': df['STATEFIP']})

did_coef_fe = model3.params['ELIGIBLE_AFTER']
did_se_fe = model3.bse['ELIGIBLE_AFTER']
did_ci_fe = model3.conf_int().loc['ELIGIBLE_AFTER']
did_pval_fe = model3.pvalues['ELIGIBLE_AFTER']

print(f"\n*** DiD Estimate with Fixed Effects (PREFERRED) ***")
print(f"Coefficient: {did_coef_fe:.4f}")
print(f"Standard Error: {did_se_fe:.4f}")
print(f"95% CI: [{did_ci_fe[0]:.4f}, {did_ci_fe[1]:.4f}]")
print(f"P-value: {did_pval_fe:.4f}")

# =============================================================================
# 7. EVENT STUDY / PRE-TRENDS ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("7. EVENT STUDY / PRE-TRENDS ANALYSIS")
print("="*70)

# Create year-specific treatment effects (event study)
years = sorted(df['YEAR'].unique())
for year in years:
    df[f'ELIGIBLE_YEAR_{year}'] = ((df['ELIGIBLE'] == 1) & (df['YEAR'] == year)).astype(int)

# Reference year is 2011 (last pre-treatment year)
year_vars = [f'ELIGIBLE_YEAR_{y}' for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]]

formula_event = 'FT ~ ELIGIBLE + ' + ' + '.join(year_vars) + ' + FEMALE + AGE + C(MARST) + C(EDUC_NUM) + C(STATEFIP) + C(YEAR)'

model_event = smf.wls(formula_event,
                       data=df,
                       weights=df['PERWT']).fit(cov_type='cluster',
                                                cov_kwds={'groups': df['STATEFIP']})

print("\n--- Event Study Coefficients ---")
print("(Reference year: 2011)")
event_results = []
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'ELIGIBLE_YEAR_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci = model_event.conf_int().loc[var]
    event_results.append({
        'Year': year,
        'Coefficient': coef,
        'SE': se,
        'CI_low': ci[0],
        'CI_high': ci[1]
    })
    print(f"Year {year}: {coef:.4f} (SE: {se:.4f}), 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

event_df = pd.DataFrame(event_results)

# =============================================================================
# 8. HETEROGENEITY ANALYSIS BY SEX
# =============================================================================
print("\n" + "="*70)
print("8. HETEROGENEITY ANALYSIS BY SEX")
print("="*70)

# Males (SEX=1 in IPUMS)
df_male = df[df['SEX'] == 1].copy()
model_male = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + AGE + C(MARST) + C(EDUC_NUM) + C(STATEFIP) + C(YEAR)',
                      data=df_male,
                      weights=df_male['PERWT']).fit(cov_type='cluster',
                                                    cov_kwds={'groups': df_male['STATEFIP']})

print(f"\n--- Males (N = {len(df_male):,}) ---")
print(f"DiD Estimate: {model_male.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model_male.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_male.conf_int().loc['ELIGIBLE_AFTER'][0]:.4f}, {model_male.conf_int().loc['ELIGIBLE_AFTER'][1]:.4f}]")

# Females (SEX=2 in IPUMS)
df_female = df[df['SEX'] == 2].copy()
model_female = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + AGE + C(MARST) + C(EDUC_NUM) + C(STATEFIP) + C(YEAR)',
                        data=df_female,
                        weights=df_female['PERWT']).fit(cov_type='cluster',
                                                        cov_kwds={'groups': df_female['STATEFIP']})

print(f"\n--- Females (N = {len(df_female):,}) ---")
print(f"DiD Estimate: {model_female.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model_female.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_female.conf_int().loc['ELIGIBLE_AFTER'][0]:.4f}, {model_female.conf_int().loc['ELIGIBLE_AFTER'][1]:.4f}]")

# =============================================================================
# 9. ROBUSTNESS: LOGIT MODEL
# =============================================================================
print("\n" + "="*70)
print("9. ROBUSTNESS: LOGIT MODEL")
print("="*70)

# Use unweighted logit with robust standard errors
from statsmodels.discrete.discrete_model import Logit

# Create design matrix
X = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']].copy()
X = sm.add_constant(X)
y = df['FT']

logit_model = Logit(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}, disp=0)
print("\n--- Logit Model Results ---")
print(logit_model.summary())

# Calculate marginal effect for ELIGIBLE_AFTER
# Average marginal effect
prob = logit_model.predict(X)
marg_effect = (logit_model.params['ELIGIBLE_AFTER'] * prob * (1 - prob)).mean()
print(f"\nAverage Marginal Effect of ELIGIBLE_AFTER: {marg_effect:.4f}")

# =============================================================================
# 10. ADDITIONAL COVARIATES: STATE POLICY VARIABLES
# =============================================================================
print("\n" + "="*70)
print("10. MODEL WITH STATE POLICY CONTROLS")
print("="*70)

# Model with state policy variables
policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID', 'EVERIFY', 'SECURECOMMUNITIES']
policy_formula = ' + '.join(policy_vars)

model_policy = smf.wls(f'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + AGE + C(MARST) + C(EDUC_NUM) + {policy_formula} + C(STATEFIP) + C(YEAR)',
                        data=df,
                        weights=df['PERWT']).fit(cov_type='cluster',
                                                 cov_kwds={'groups': df['STATEFIP']})

did_coef_policy = model_policy.params['ELIGIBLE_AFTER']
did_se_policy = model_policy.bse['ELIGIBLE_AFTER']
did_ci_policy = model_policy.conf_int().loc['ELIGIBLE_AFTER']

print(f"\n*** DiD Estimate with State Policy Controls ***")
print(f"Coefficient: {did_coef_policy:.4f}")
print(f"Standard Error: {did_se_policy:.4f}")
print(f"95% CI: [{did_ci_policy[0]:.4f}, {did_ci_policy[1]:.4f}]")

# =============================================================================
# 11. SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("11. SUMMARY OF ALL MODELS")
print("="*70)

results_summary = [
    {'Model': '1. Basic DiD', 'Estimate': did_coef_basic, 'SE': did_se_basic,
     'CI_low': did_ci_basic[0], 'CI_high': did_ci_basic[1], 'P_value': did_pval_basic},
    {'Model': '2. + Demographics', 'Estimate': did_coef_demo, 'SE': did_se_demo,
     'CI_low': did_ci_demo[0], 'CI_high': did_ci_demo[1], 'P_value': did_pval_demo},
    {'Model': '3. + State/Year FE (Preferred)', 'Estimate': did_coef_fe, 'SE': did_se_fe,
     'CI_low': did_ci_fe[0], 'CI_high': did_ci_fe[1], 'P_value': did_pval_fe},
    {'Model': '4. + State Policies', 'Estimate': did_coef_policy,
     'SE': did_se_policy,
     'CI_low': did_ci_policy[0],
     'CI_high': did_ci_policy[1],
     'P_value': model_policy.pvalues['ELIGIBLE_AFTER']},
    {'Model': '5. Males Only', 'Estimate': model_male.params['ELIGIBLE_AFTER'],
     'SE': model_male.bse['ELIGIBLE_AFTER'],
     'CI_low': model_male.conf_int().loc['ELIGIBLE_AFTER'][0],
     'CI_high': model_male.conf_int().loc['ELIGIBLE_AFTER'][1],
     'P_value': model_male.pvalues['ELIGIBLE_AFTER']},
    {'Model': '6. Females Only', 'Estimate': model_female.params['ELIGIBLE_AFTER'],
     'SE': model_female.bse['ELIGIBLE_AFTER'],
     'CI_low': model_female.conf_int().loc['ELIGIBLE_AFTER'][0],
     'CI_high': model_female.conf_int().loc['ELIGIBLE_AFTER'][1],
     'P_value': model_female.pvalues['ELIGIBLE_AFTER']},
]

results_df = pd.DataFrame(results_summary)
print("\n" + results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('tables/model_results.csv', index=False)
event_df.to_csv('tables/event_study_results.csv', index=False)
summary_df.to_csv('tables/summary_statistics.csv', index=False)

# =============================================================================
# 12. ADDITIONAL DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*70)
print("12. ADDITIONAL DESCRIPTIVE STATISTICS")
print("="*70)

# Education distribution by group
print("\n--- Education Distribution by Group ---")
for elig_val, elig_name in [(1, 'Treatment'), (0, 'Control')]:
    sub = df[df['ELIGIBLE'] == elig_val]
    print(f"\n{elig_name} group:")
    educ_dist = sub.groupby('EDUC_RECODE')['PERWT'].sum()
    educ_dist = educ_dist / educ_dist.sum() * 100
    print(educ_dist.round(1))

# State distribution (top 10)
print("\n--- Top 10 States by Sample Size ---")
state_counts = df.groupby('STATEFIP').size().sort_values(ascending=False).head(10)
print(state_counts)

# =============================================================================
# 13. PREFERRED ESTIMATE SUMMARY
# =============================================================================
print("\n" + "="*70)
print("13. PREFERRED ESTIMATE (Model 3)")
print("="*70)

n_total = len(df)
n_treat = len(df[df['ELIGIBLE']==1])
n_ctrl = len(df[df['ELIGIBLE']==0])

print(f"\nSample Size: {n_total:,}")
print(f"  - Treatment group (ages 26-30): {n_treat:,}")
print(f"  - Control group (ages 31-35): {n_ctrl:,}")
print(f"\nPreferred DiD Estimate: {did_coef_fe:.4f}")
print(f"Standard Error: {did_se_fe:.4f}")
print(f"95% Confidence Interval: [{did_ci_fe[0]:.4f}, {did_ci_fe[1]:.4f}]")
print(f"P-value: {did_pval_fe:.4f}")

# Interpretation
print("\n--- Interpretation ---")
if did_pval_fe < 0.05:
    direction = "increase" if did_coef_fe > 0 else "decrease"
    print(f"DACA eligibility is associated with a statistically significant {direction}")
    print(f"in the probability of full-time employment of {abs(did_coef_fe)*100:.2f} percentage points.")
else:
    print(f"The effect of DACA eligibility on full-time employment is not statistically significant")
    print(f"at the 5% level (estimate = {did_coef_fe*100:.2f} pp, p = {did_pval_fe:.3f}).")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

# Store key results for LaTeX export
key_results = {
    'n_total': n_total,
    'n_treat': n_treat,
    'n_ctrl': n_ctrl,
    'did_estimate': did_coef_fe,
    'did_se': did_se_fe,
    'did_ci_low': did_ci_fe[0],
    'did_ci_high': did_ci_fe[1],
    'did_pval': did_pval_fe,
    'ft_pre_treat': ft_pre_treat,
    'ft_post_treat': ft_post_treat,
    'ft_pre_ctrl': ft_pre_ctrl,
    'ft_post_ctrl': ft_post_ctrl,
    'did_simple': did_simple,
    'did_basic': did_coef_basic,
    'did_basic_se': did_se_basic,
    'did_demo': did_coef_demo,
    'did_demo_se': did_se_demo,
    'did_policy': did_coef_policy,
    'did_policy_se': did_se_policy,
    'did_male': model_male.params['ELIGIBLE_AFTER'],
    'did_male_se': model_male.bse['ELIGIBLE_AFTER'],
    'did_female': model_female.params['ELIGIBLE_AFTER'],
    'did_female_se': model_female.bse['ELIGIBLE_AFTER'],
    'event_study': event_df,
    'results_summary': results_df,
    'summary_stats': summary_df
}

# Save to pickle for later use
import pickle
with open('tables/key_results.pkl', 'wb') as f:
    pickle.dump(key_results, f)

print("\nKey results saved to tables/key_results.pkl")
