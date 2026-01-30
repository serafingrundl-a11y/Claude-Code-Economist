"""
DACA Replication Analysis
Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals

Difference-in-Differences Design:
- Treatment: DACA-eligible individuals aged 26-30 at June 2012
- Control: Individuals aged 31-35 at June 2012 (otherwise eligible)
- Pre-period: 2008-2011
- Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# Basic data exploration
print("\n" + "="*60)
print("DATA EXPLORATION")
print("="*60)

print(f"\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())

print(f"\nAFTER distribution:")
print(df['AFTER'].value_counts())

print(f"\nFT distribution:")
print(df['FT'].value_counts())

print(f"\nYear distribution:")
print(df['YEAR'].value_counts().sort_index())

# Check age distribution
print(f"\nAge in June 2012 distribution:")
print(df['AGE_IN_JUNE_2012'].value_counts().sort_index())

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Summary statistics by group
print("\n" + "="*60)
print("SUMMARY STATISTICS BY GROUP")
print("="*60)

# Create group labels
df['Group'] = np.where(df['ELIGIBLE'] == 1, 'Eligible (26-30)', 'Comparison (31-35)')
df['Period'] = np.where(df['AFTER'] == 1, 'Post (2013-16)', 'Pre (2008-11)')

# Summary table
summary = df.groupby(['Group', 'Period']).agg({
    'FT': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
print(summary)

# Weighted means
print("\n" + "="*60)
print("WEIGHTED FULL-TIME EMPLOYMENT RATES")
print("="*60)

for group in ['Eligible (26-30)', 'Comparison (31-35)']:
    for period in ['Pre (2008-11)', 'Post (2013-16)']:
        subset = df[(df['Group'] == group) & (df['Period'] == period)]
        weighted_mean = np.average(subset['FT'], weights=subset['PERWT'])
        n = len(subset)
        print(f"{group}, {period}: {weighted_mean:.4f} (n={n:,})")

# Calculate simple DiD
print("\n" + "="*60)
print("SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("="*60)

# Unweighted
pre_treat = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]['FT'].mean()
pre_control = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]['FT'].mean()
post_control = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]['FT'].mean()

diff_treat = post_treat - pre_treat
diff_control = post_control - pre_control
did_simple = diff_treat - diff_control

print(f"\nUnweighted:")
print(f"Eligible pre:  {pre_treat:.4f}")
print(f"Eligible post: {post_treat:.4f}")
print(f"Change eligible: {diff_treat:.4f}")
print(f"\nComparison pre:  {pre_control:.4f}")
print(f"Comparison post: {post_control:.4f}")
print(f"Change comparison: {diff_control:.4f}")
print(f"\nDiD estimate: {did_simple:.4f}")

# Weighted
def weighted_mean(x, w):
    return np.average(x, weights=w)

pre_treat_w = weighted_mean(df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]['FT'],
                            df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]['PERWT'])
post_treat_w = weighted_mean(df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]['FT'],
                             df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]['PERWT'])
pre_control_w = weighted_mean(df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]['FT'],
                              df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]['PERWT'])
post_control_w = weighted_mean(df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]['FT'],
                               df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]['PERWT'])

diff_treat_w = post_treat_w - pre_treat_w
diff_control_w = post_control_w - pre_control_w
did_weighted = diff_treat_w - diff_control_w

print(f"\nWeighted:")
print(f"Eligible pre:  {pre_treat_w:.4f}")
print(f"Eligible post: {post_treat_w:.4f}")
print(f"Change eligible: {diff_treat_w:.4f}")
print(f"\nComparison pre:  {pre_control_w:.4f}")
print(f"Comparison post: {post_control_w:.4f}")
print(f"Change comparison: {diff_control_w:.4f}")
print(f"\nDiD estimate: {did_weighted:.4f}")

# REGRESSION ANALYSIS
print("\n" + "="*60)
print("REGRESSION ANALYSIS")
print("="*60)

# Model 1: Basic DiD (OLS, no weights, no controls)
print("\n--- Model 1: Basic DiD (no weights, no controls) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])

# Model 2: DiD with survey weights
print("\n--- Model 2: DiD with survey weights (WLS) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with state clustered standard errors
print("\n--- Model 3: DiD with state-clustered standard errors ---")
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='cluster',
                                                                        cov_kwds={'groups': df['STATEFIP']})
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with year fixed effects ---")
df['YEAR_str'] = df['YEAR'].astype(str)
model4 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR_str)', data=df).fit(cov_type='cluster',
                                                                               cov_kwds={'groups': df['STATEFIP']})
print("DiD coefficient (ELIGIBLE_AFTER):", model4.params['ELIGIBLE_AFTER'])
print("Standard error:", model4.bse['ELIGIBLE_AFTER'])
print("p-value:", model4.pvalues['ELIGIBLE_AFTER'])
print("95% CI:", model4.conf_int().loc['ELIGIBLE_AFTER'].values)

# Model 5: DiD with state and year fixed effects
print("\n--- Model 5: DiD with state and year fixed effects ---")
df['STATE_str'] = df['STATEFIP'].astype(str)
model5 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR_str) + C(STATE_str)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print("DiD coefficient (ELIGIBLE_AFTER):", model5.params['ELIGIBLE_AFTER'])
print("Standard error:", model5.bse['ELIGIBLE_AFTER'])
print("p-value:", model5.pvalues['ELIGIBLE_AFTER'])
print("95% CI:", model5.conf_int().loc['ELIGIBLE_AFTER'].values)

# Prepare covariates
print("\n" + "="*60)
print("PREPARING COVARIATES")
print("="*60)

# Recode SEX (IPUMS: 1=Male, 2=Female)
df['FEMALE'] = (df['SEX'] == 2).astype(int)
print(f"Female proportion: {df['FEMALE'].mean():.3f}")

# Marital status (IPUMS codes: 1=Married spouse present, 2=Married spouse absent, etc.)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)
print(f"Married proportion: {df['MARRIED'].mean():.3f}")

# Education categories
print(f"\nEducation distribution:")
print(df['EDUC_RECODE'].value_counts())

# Create education dummies
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_AA'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Model 6: DiD with demographic controls
print("\n--- Model 6: DiD with demographic controls ---")
model6 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR_str) + C(STATE_str) + '
                 'FEMALE + MARRIED + AGE + EDUC_HS + EDUC_SOMECOL + EDUC_AA + EDUC_BA',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print("DiD coefficient (ELIGIBLE_AFTER):", model6.params['ELIGIBLE_AFTER'])
print("Standard error:", model6.bse['ELIGIBLE_AFTER'])
print("p-value:", model6.pvalues['ELIGIBLE_AFTER'])
print("95% CI:", model6.conf_int().loc['ELIGIBLE_AFTER'].values)

# Model 7: Full model with additional state policies
print("\n--- Model 7: Full model with state policies ---")
state_policies = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID',
                  'EVERIFY', 'SECURECOMMUNITIES']
policy_formula = ' + '.join(state_policies)

model7 = smf.ols(f'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR_str) + C(STATE_str) + '
                 f'FEMALE + MARRIED + AGE + EDUC_HS + EDUC_SOMECOL + EDUC_AA + EDUC_BA + '
                 f'{policy_formula}',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print("DiD coefficient (ELIGIBLE_AFTER):", model7.params['ELIGIBLE_AFTER'])
print("Standard error:", model7.bse['ELIGIBLE_AFTER'])
print("p-value:", model7.pvalues['ELIGIBLE_AFTER'])
print("95% CI:", model7.conf_int().loc['ELIGIBLE_AFTER'].values)

# PARALLEL TRENDS ANALYSIS
print("\n" + "="*60)
print("PARALLEL TRENDS CHECK (EVENT STUDY)")
print("="*60)

# Create year-specific treatment effects
for year in df['YEAR'].unique():
    df[f'ELIGIBLE_Y{year}'] = (df['ELIGIBLE'] * (df['YEAR'] == year)).astype(int)

year_vars = [f'ELIGIBLE_Y{y}' for y in sorted(df['YEAR'].unique()) if y != 2011]  # 2011 as reference
year_formula = ' + '.join(year_vars)

model_event = smf.ols(f'FT ~ ELIGIBLE + {year_formula} + C(YEAR_str) + C(STATE_str)',
                      data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("\nYear-specific treatment effects (reference: 2011):")
for var in year_vars:
    print(f"{var}: {model_event.params[var]:.4f} (SE: {model_event.bse[var]:.4f})")

# HETEROGENEITY ANALYSIS
print("\n" + "="*60)
print("HETEROGENEITY ANALYSIS")
print("="*60)

# By gender
print("\n--- By Gender ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex]
    model_het = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR_str)',
                        data=subset).fit(cov_type='cluster', cov_kwds={'groups': subset['STATEFIP']})
    print(f"{label}: DiD = {model_het.params['ELIGIBLE_AFTER']:.4f} "
          f"(SE = {model_het.bse['ELIGIBLE_AFTER']:.4f}), p = {model_het.pvalues['ELIGIBLE_AFTER']:.4f}")

# By education
print("\n--- By Education ---")
for educ in df['EDUC_RECODE'].unique():
    subset = df[df['EDUC_RECODE'] == educ]
    if len(subset) > 100:
        try:
            model_het = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR_str)',
                                data=subset).fit(cov_type='cluster', cov_kwds={'groups': subset['STATEFIP']})
            print(f"{educ}: DiD = {model_het.params['ELIGIBLE_AFTER']:.4f} "
                  f"(SE = {model_het.bse['ELIGIBLE_AFTER']:.4f})")
        except:
            print(f"{educ}: Could not estimate")

# ROBUSTNESS CHECKS
print("\n" + "="*60)
print("ROBUSTNESS CHECKS")
print("="*60)

# Linear probability model vs probit
print("\n--- Probit model ---")
try:
    # Probit marginal effects
    probit_model = smf.probit('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(disp=0)
    mfx = probit_model.get_margeff()
    print(f"DiD marginal effect: {mfx.margeff[2]:.4f}")
    print(f"Standard error: {mfx.margeff_se[2]:.4f}")
except Exception as e:
    print(f"Probit failed: {e}")

# Weighted regression
print("\n--- Weighted DiD with controls ---")
model_weighted = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR_str) + C(STATE_str) + '
                         'FEMALE + MARRIED + AGE',
                         data=df, weights=df['PERWT']).fit()
print("DiD coefficient (ELIGIBLE_AFTER):", model_weighted.params['ELIGIBLE_AFTER'])

# FINAL RESULTS SUMMARY
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

results_summary = {
    'Model': ['Basic DiD', 'DiD + Year FE', 'DiD + State + Year FE',
              'DiD + Demographics', 'DiD + State Policies'],
    'DiD Estimate': [model1.params['ELIGIBLE_AFTER'],
                     model4.params['ELIGIBLE_AFTER'],
                     model5.params['ELIGIBLE_AFTER'],
                     model6.params['ELIGIBLE_AFTER'],
                     model7.params['ELIGIBLE_AFTER']],
    'Std. Error': [model1.bse['ELIGIBLE_AFTER'],
                   model4.bse['ELIGIBLE_AFTER'],
                   model5.bse['ELIGIBLE_AFTER'],
                   model6.bse['ELIGIBLE_AFTER'],
                   model7.bse['ELIGIBLE_AFTER']],
    'p-value': [model1.pvalues['ELIGIBLE_AFTER'],
                model4.pvalues['ELIGIBLE_AFTER'],
                model5.pvalues['ELIGIBLE_AFTER'],
                model6.pvalues['ELIGIBLE_AFTER'],
                model7.pvalues['ELIGIBLE_AFTER']]
}

results_df = pd.DataFrame(results_summary)
results_df['95% CI Lower'] = results_df['DiD Estimate'] - 1.96 * results_df['Std. Error']
results_df['95% CI Upper'] = results_df['DiD Estimate'] + 1.96 * results_df['Std. Error']
print(results_df.to_string(index=False))

# Save key results
print("\n" + "="*60)
print("PREFERRED SPECIFICATION: Model 6 (DiD with Demographics)")
print("="*60)
print(f"Sample size: {len(df):,}")
print(f"DiD effect estimate: {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence interval: [{model6.conf_int().loc['ELIGIBLE_AFTER'].values[0]:.4f}, "
      f"{model6.conf_int().loc['ELIGIBLE_AFTER'].values[1]:.4f}]")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")

# Save results to file
with open('analysis_results.txt', 'w') as f:
    f.write("DACA REPLICATION ANALYSIS RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Sample size: {len(df):,}\n")
    f.write(f"Treatment group (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum():,}\n")
    f.write(f"Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum():,}\n\n")
    f.write("PREFERRED SPECIFICATION: Model 6 (DiD with Demographics)\n")
    f.write(f"DiD effect estimate: {model6.params['ELIGIBLE_AFTER']:.4f}\n")
    f.write(f"Standard error: {model6.bse['ELIGIBLE_AFTER']:.4f}\n")
    f.write(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER'].values[0]:.4f}, "
            f"{model6.conf_int().loc['ELIGIBLE_AFTER'].values[1]:.4f}]\n")
    f.write(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}\n")

print("\nAnalysis complete. Results saved to analysis_results.txt")
