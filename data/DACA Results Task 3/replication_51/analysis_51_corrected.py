"""
DACA Replication Study - Corrected Analysis Script (ID: 51)
Research Question: Effect of DACA eligibility on full-time employment

This script implements a difference-in-differences analysis to estimate
the causal effect of DACA eligibility on full-time employment.
Using proper clustered standard errors.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set working directory path
data_path = "C:/Users/seraf/DACA Results Task 3/replication_51/data/"
output_path = "C:/Users/seraf/DACA Results Task 3/replication_51/"

print("=" * 60)
print("DACA REPLICATION STUDY - CORRECTED ANALYSIS")
print("=" * 60)

# Load data
print("\nLoading data...")
df = pd.read_csv(data_path + "prepared_data_numeric_version.csv")
print(f"Data dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop rows with missing values in key variables
df = df.dropna(subset=['FT', 'ELIGIBLE', 'AFTER', 'AGE', 'SEX', 'NCHILD', 'MARST', 'EDUC_RECODE', 'STATEFIP'])
print(f"Sample size after dropping missing: {len(df)}")

# Create necessary variables
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)

# ===== SUMMARY STATISTICS =====
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

# Calculate means by group and period (weighted)
means_table = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: pd.Series({
        'n': len(x),
        'mean_FT_unweighted': x['FT'].mean(),
        'mean_FT_weighted': np.average(x['FT'], weights=x['PERWT'])
    })
).reset_index()

print("\nMean Full-Time Employment by Group and Period:")
print(means_table.to_string(index=False))

# Calculate simple DiD
treat_post = means_table.loc[(means_table['ELIGIBLE']==1) & (means_table['AFTER']==1), 'mean_FT_weighted'].values[0]
treat_pre = means_table.loc[(means_table['ELIGIBLE']==1) & (means_table['AFTER']==0), 'mean_FT_weighted'].values[0]
control_post = means_table.loc[(means_table['ELIGIBLE']==0) & (means_table['AFTER']==1), 'mean_FT_weighted'].values[0]
control_pre = means_table.loc[(means_table['ELIGIBLE']==0) & (means_table['AFTER']==0), 'mean_FT_weighted'].values[0]

did_simple = (treat_post - treat_pre) - (control_post - control_pre)

print(f"\nSimple DiD Estimate: {did_simple:.4f}")
print(f"  Treatment Change: {treat_post - treat_pre:.4f}")
print(f"  Control Change: {control_post - control_pre:.4f}")

# ===== REGRESSION WITH PROPER CLUSTERED SEs =====
print("\n" + "=" * 60)
print("REGRESSION ANALYSIS WITH CLUSTERED STANDARD ERRORS")
print("=" * 60)

# Model 1: Basic DiD with WLS and robust cluster SE
print("\n--- Model 1: Basic DiD ---")

# Use OLS first as baseline (unweighted)
model1_ols = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}
)
print("\nOLS with Clustered SEs (unweighted):")
print(model1_ols.summary().tables[1])

# For weighted analysis, we can use the formula interface with weights
# but need to be careful about the clustered SEs

# Create a proper analysis approach using linearmodels for panel data
# Since we're doing repeated cross-section DiD, standard approaches apply

# Let's use statsmodels with proper clustering
# First aggregate to state-year level for cluster-robust inference

print("\n--- Alternative: Aggregated Analysis ---")
# Aggregate to state-year-eligibility level
agg_df = df.groupby(['STATEFIP', 'YEAR', 'ELIGIBLE', 'AFTER']).agg({
    'FT': 'mean',
    'PERWT': 'sum',
    'FEMALE': 'mean',
    'AGE': 'mean',
    'NCHILD': 'mean',
    'MARST': lambda x: (x.isin([1,2])).mean(),
    'DRIVERSLICENSES': 'first',
    'INSTATETUITION': 'first',
    'EVERIFY': 'first',
    'LFPR': 'first',
    'UNEMP': 'first'
}).reset_index()

agg_df['ELIGIBLE_AFTER'] = agg_df['ELIGIBLE'] * agg_df['AFTER']
agg_df.rename(columns={'MARST': 'MARRIED_SHARE'}, inplace=True)

# Weighted by number of observations
agg_df['n_obs'] = df.groupby(['STATEFIP', 'YEAR', 'ELIGIBLE', 'AFTER']).size().values

print(f"Aggregated data: {len(agg_df)} state-year-eligibility cells")

# Model with aggregated data, weighted by cell size
model_agg = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                    data=agg_df, weights=agg_df['n_obs']).fit(
    cov_type='cluster', cov_kwds={'groups': agg_df['STATEFIP']}
)
print("\nAggregated WLS with Clustered SEs:")
print(model_agg.summary().tables[1])

# Back to individual level with proper approach
print("\n--- Individual-Level Analysis ---")

# Model 1: Basic DiD
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}
)

# Model 2: With demographics
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + NCHILD + C(MARST) + C(EDUC_RECODE)',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

# Model 3: With state FE
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + NCHILD + C(MARST) + C(EDUC_RECODE) + C(STATEFIP)',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

# Model 4: With state and year FE (drop AFTER)
model4 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + AGE + NCHILD + C(MARST) + C(EDUC_RECODE) + C(STATEFIP) + C(YEAR)',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

# Model 5: With policy controls
model5 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + NCHILD + C(MARST) + C(EDUC_RECODE) + DRIVERSLICENSES + INSTATETUITION + EVERIFY + LFPR + UNEMP',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

# Extract results
results = []
for name, model, did_var in [
    ('1. Basic DiD', model1, 'ELIGIBLE_AFTER'),
    ('2. Demographics', model2, 'ELIGIBLE_AFTER'),
    ('3. State FE', model3, 'ELIGIBLE_AFTER'),
    ('4. State+Year FE', model4, 'ELIGIBLE_AFTER'),
    ('5. Policy Controls', model5, 'ELIGIBLE_AFTER')
]:
    est = model.params[did_var]
    se = model.bse[did_var]
    pval = model.pvalues[did_var]
    ci_l = model.conf_int().loc[did_var, 0]
    ci_u = model.conf_int().loc[did_var, 1]
    results.append({
        'Model': name,
        'DiD_Estimate': est,
        'Std_Error': se,
        't_stat': est/se,
        'p_value': pval,
        '95% CI Lower': ci_l,
        '95% CI Upper': ci_u
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("SUMMARY OF DIFFERENCE-IN-DIFFERENCES ESTIMATES")
print("(Clustered SEs at State Level)")
print("=" * 60)
print(results_df.round(4).to_string(index=False))

# ===== YEARLY MEANS =====
print("\n" + "=" * 60)
print("YEARLY EMPLOYMENT RATES BY GROUP")
print("=" * 60)

yearly_means = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'mean_FT': np.average(x['FT'], weights=x['PERWT']),
        'n': len(x)
    })
).reset_index()

yearly_pivot = yearly_means.pivot(index='YEAR', columns='ELIGIBLE', values='mean_FT')
yearly_pivot.columns = ['Control (31-35)', 'Treatment (26-30)']
print(yearly_pivot.round(4))

# ===== PRE-TREND TEST =====
print("\n" + "=" * 60)
print("PRE-TREATMENT PARALLEL TRENDS TEST")
print("=" * 60)

df_pre = df[df['AFTER'] == 0].copy()
df_pre['YEAR_2009'] = (df_pre['YEAR'] == 2009).astype(int)
df_pre['YEAR_2010'] = (df_pre['YEAR'] == 2010).astype(int)
df_pre['YEAR_2011'] = (df_pre['YEAR'] == 2011).astype(int)
df_pre['ELIG_2009'] = df_pre['ELIGIBLE'] * df_pre['YEAR_2009']
df_pre['ELIG_2010'] = df_pre['ELIGIBLE'] * df_pre['YEAR_2010']
df_pre['ELIG_2011'] = df_pre['ELIGIBLE'] * df_pre['YEAR_2011']

pretrend_model = smf.ols('FT ~ ELIGIBLE + YEAR_2009 + YEAR_2010 + YEAR_2011 + ELIG_2009 + ELIG_2010 + ELIG_2011',
                         data=df_pre).fit(cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})

print("\nDifferential Pre-Trends (ELIGIBLE x Year interactions):")
for var in ['ELIG_2009', 'ELIG_2010', 'ELIG_2011']:
    print(f"  {var}: {pretrend_model.params[var]:.4f} (SE: {pretrend_model.bse[var]:.4f}, p: {pretrend_model.pvalues[var]:.4f})")

# Joint F-test
from statsmodels.stats.anova import anova_lm
r_matrix = np.zeros((3, len(pretrend_model.params)))
for i, var in enumerate(['ELIG_2009', 'ELIG_2010', 'ELIG_2011']):
    r_matrix[i, list(pretrend_model.params.index).index(var)] = 1

try:
    f_test = pretrend_model.f_test(r_matrix)
    print(f"\nJoint F-test of pre-trends: F = {f_test.fvalue[0][0]:.4f}, p = {f_test.pvalue:.4f}")
except:
    print("\nJoint F-test could not be computed")

# ===== SAVE RESULTS =====
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save summary
with open(output_path + 'analysis_output.txt', 'w') as f:
    f.write("DACA Replication Analysis Output\n")
    f.write("=" * 60 + "\n\n")

    f.write("SAMPLE INFORMATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Sample Size: {len(df)}\n")
    f.write(f"Treatment group (ELIGIBLE=1): {sum(df['ELIGIBLE']==1)}\n")
    f.write(f"Control group (ELIGIBLE=0): {sum(df['ELIGIBLE']==0)}\n")
    f.write(f"Pre-period observations: {sum(df['AFTER']==0)}\n")
    f.write(f"Post-period observations: {sum(df['AFTER']==1)}\n")
    f.write(f"Number of states: {df['STATEFIP'].nunique()}\n\n")

    f.write("SIMPLE DiD CALCULATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"Treatment Pre: {treat_pre:.4f}\n")
    f.write(f"Treatment Post: {treat_post:.4f}\n")
    f.write(f"Control Pre: {control_pre:.4f}\n")
    f.write(f"Control Post: {control_post:.4f}\n")
    f.write(f"DiD Estimate: {did_simple:.4f}\n\n")

    f.write("REGRESSION RESULTS SUMMARY\n")
    f.write("-" * 40 + "\n")
    f.write(results_df.round(4).to_string(index=False))
    f.write("\n\n")

    # Preferred estimate (Model 3)
    pref = results_df[results_df['Model'] == '3. State FE'].iloc[0]
    f.write("PREFERRED ESTIMATE (Model 3: State FE)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Effect: {pref['DiD_Estimate']:.4f}\n")
    f.write(f"Std. Error: {pref['Std_Error']:.4f}\n")
    f.write(f"95% CI: [{pref['95% CI Lower']:.4f}, {pref['95% CI Upper']:.4f}]\n")
    f.write(f"p-value: {pref['p_value']:.4f}\n")

results_df.to_csv(output_path + 'regression_results.csv', index=False)
yearly_pivot.to_csv(output_path + 'yearly_means.csv')

print("Saved: analysis_output.txt, regression_results.csv, yearly_means.csv")

# ===== FINAL RESULTS =====
print("\n" + "=" * 60)
print("FINAL RESULTS - PREFERRED ESTIMATE")
print("=" * 60)
pref = results_df[results_df['Model'] == '3. State FE'].iloc[0]
print(f"\nModel: DiD with demographic controls and state fixed effects")
print(f"DiD Estimate: {pref['DiD_Estimate']:.4f}")
print(f"Standard Error (Clustered at State): {pref['Std_Error']:.4f}")
print(f"95% CI: [{pref['95% CI Lower']:.4f}, {pref['95% CI Upper']:.4f}]")
print(f"p-value: {pref['p_value']:.4f}")
print(f"Sample Size: {len(df)}")
print("=" * 60)

print("\nAnalysis complete!")
