"""
DACA Replication Study - Analysis Script (ID: 51)
Research Question: Effect of DACA eligibility on full-time employment

This script implements a difference-in-differences analysis to estimate
the causal effect of DACA eligibility on full-time employment.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set working directory path
data_path = "C:/Users/seraf/DACA Results Task 3/replication_51/data/"
output_path = "C:/Users/seraf/DACA Results Task 3/replication_51/"

print("=" * 60)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 60)

# Load data
print("\nLoading data...")
df = pd.read_csv(data_path + "prepared_data_numeric_version.csv")

print(f"Data dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

# ===== BASIC DATA EXPLORATION =====
print("\n" + "=" * 60)
print("DATA EXPLORATION")
print("=" * 60)

print("\nKey Variables Summary:")
print("-" * 40)

print("\nFT (Full-time employment):")
print(df['FT'].value_counts().sort_index())

print("\nELIGIBLE (Treatment group):")
print(df['ELIGIBLE'].value_counts().sort_index())

print("\nAFTER (Post-DACA period):")
print(df['AFTER'].value_counts().sort_index())

print("\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

# DiD Cell Counts
print("\n" + "=" * 60)
print("DIFFERENCE-IN-DIFFERENCES CELL STRUCTURE")
print("=" * 60)
did_table = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
did_table.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
did_table.columns = ['Pre (2008-2011)', 'Post (2013-2016)', 'Total']
print(did_table)

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
print("-" * 60)
print(means_table.to_string(index=False))

# Calculate simple DiD
treat_post = means_table.loc[(means_table['ELIGIBLE']==1) & (means_table['AFTER']==1), 'mean_FT_weighted'].values[0]
treat_pre = means_table.loc[(means_table['ELIGIBLE']==1) & (means_table['AFTER']==0), 'mean_FT_weighted'].values[0]
control_post = means_table.loc[(means_table['ELIGIBLE']==0) & (means_table['AFTER']==1), 'mean_FT_weighted'].values[0]
control_pre = means_table.loc[(means_table['ELIGIBLE']==0) & (means_table['AFTER']==0), 'mean_FT_weighted'].values[0]

did_simple = (treat_post - treat_pre) - (control_post - control_pre)

print("\n" + "=" * 60)
print("SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION")
print("=" * 60)
print(f"\nTreatment Group (ELIGIBLE=1, ages 26-30):")
print(f"  Pre-DACA mean:  {treat_pre:.4f}")
print(f"  Post-DACA mean: {treat_post:.4f}")
print(f"  Change:         {treat_post - treat_pre:.4f}")

print(f"\nControl Group (ELIGIBLE=0, ages 31-35):")
print(f"  Pre-DACA mean:  {control_pre:.4f}")
print(f"  Post-DACA mean: {control_post:.4f}")
print(f"  Change:         {control_post - control_pre:.4f}")

print(f"\nDifference-in-Differences: {did_simple:.4f}")

# ===== DEMOGRAPHIC SUMMARY =====
print("\n" + "=" * 60)
print("COVARIATE SUMMARY BY TREATMENT STATUS")
print("=" * 60)

# Create binary/numeric versions of key variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)

# Summarize covariates by ELIGIBLE
covar_summary = df.groupby('ELIGIBLE').agg({
    'AGE': 'mean',
    'FEMALE': 'mean',
    'MARRIED': 'mean',
    'NCHILD': 'mean',
    'AGE_AT_IMMIGRATION': 'mean'
}).round(3)
covar_summary.index = ['Control (31-35)', 'Treatment (26-30)']
print("\nMean Characteristics by Treatment Status:")
print(covar_summary)

# ===== REGRESSION ANALYSIS =====
print("\n" + "=" * 60)
print("REGRESSION ANALYSIS")
print("=" * 60)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Create dummy variables for categorical variables using get_dummies
# This avoids issues with the formula interface

# For EDUC_RECODE, create dummies (reference: first category)
educ_dummies = pd.get_dummies(df['EDUC_RECODE'], prefix='EDUC', drop_first=True).astype(float)
marst_dummies = pd.get_dummies(df['MARST'], prefix='MARST', drop_first=True).astype(float)
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True).astype(float)
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True).astype(float)

# Combine all into analysis dataframe
df_analysis = pd.concat([df[['FT', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE', 'NCHILD',
                             'PERWT', 'STATEFIP', 'DRIVERSLICENSES', 'INSTATETUITION', 'EVERIFY',
                             'LFPR', 'UNEMP', 'YEAR']],
                         educ_dummies, marst_dummies, state_dummies, year_dummies], axis=1)

# Drop any rows with missing values
df_analysis = df_analysis.dropna()
print(f"\nAnalysis sample size after dropping missing: {len(df_analysis)}")

# Function to run weighted regression with clustered SEs
def run_wls_clustered(y, X, weights, clusters):
    """Run WLS regression with cluster-robust standard errors."""
    # Add constant
    X = sm.add_constant(X)

    # Fit WLS model
    model = sm.WLS(y, X, weights=weights).fit()

    # Get cluster-robust SEs
    # Group data by cluster
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    n_params = X.shape[1]

    # Compute meat of the sandwich
    resid = model.resid
    xu = X.values * resid.values[:, np.newaxis] * np.sqrt(weights.values[:, np.newaxis])

    # Sum residuals within clusters
    S = np.zeros((n_params, n_params))
    for c in unique_clusters:
        mask = clusters == c
        xu_c = xu[mask].sum(axis=0)
        S += np.outer(xu_c, xu_c)

    # Finite sample correction
    n = len(y)
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - n_params))

    # Bread of sandwich
    bread = np.linalg.inv(X.T @ (X * weights.values[:, np.newaxis]))

    # Clustered variance-covariance matrix
    vcov = correction * bread @ S @ bread

    # Extract standard errors
    se_clustered = np.sqrt(np.diag(vcov))

    return model, se_clustered, vcov

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD ---")
X1 = df_analysis[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']]
y = df_analysis['FT']
weights = df_analysis['PERWT']
clusters = df_analysis['STATEFIP'].values

model1, se1, vcov1 = run_wls_clustered(y, X1, weights, clusters)

from scipy import stats

n_clust = len(np.unique(clusters))

print(f"{'Variable':<20} {'Coef':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
var_names = ['Intercept', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']
for i, var in enumerate(var_names):
    coef = model1.params.iloc[i]
    se = se1[i]
    t = coef / se
    p = 2 * stats.t.sf(abs(t), n_clust - 1)
    print(f"{var:<20} {coef:>10.4f} {se:>10.4f} {t:>10.4f} {p:>10.4f}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographics ---")
demo_cols = ['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE', 'NCHILD'] + [c for c in educ_dummies.columns] + [c for c in marst_dummies.columns]
X2 = df_analysis[demo_cols]
model2, se2, vcov2 = run_wls_clustered(y, X2, weights, clusters)

print("Key coefficients:")
key_vars_idx = {'ELIGIBLE': 1, 'AFTER': 2, 'ELIGIBLE_AFTER': 3, 'FEMALE': 4, 'AGE': 5, 'NCHILD': 6}
for var, idx in key_vars_idx.items():
    coef = model2.params.iloc[idx]
    se = se2[idx]
    t = coef / se
    p = 2 * stats.t.sf(abs(t), n_clust - 1)
    print(f"  {var}: {coef:.4f} (SE: {se:.4f}, t: {t:.2f}, p: {p:.4f})")

# Model 3: DiD with state fixed effects
print("\n--- Model 3: DiD with State Fixed Effects ---")
state_cols = ['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE', 'NCHILD'] + [c for c in educ_dummies.columns] + [c for c in marst_dummies.columns] + [c for c in state_dummies.columns]
X3 = df_analysis[state_cols]
model3, se3, vcov3 = run_wls_clustered(y, X3, weights, clusters)

print("Key DiD coefficients:")
for var, idx in [('ELIGIBLE', 1), ('AFTER', 2), ('ELIGIBLE_AFTER', 3)]:
    coef = model3.params.iloc[idx]
    se = se3[idx]
    t = coef / se
    p = 2 * stats.t.sf(abs(t), n_clust - 1)
    print(f"  {var}: {coef:.4f} (SE: {se:.4f}, t: {t:.2f}, p: {p:.4f})")

# Model 4: DiD with state and year FE (drop AFTER since it's collinear with year FE)
print("\n--- Model 4: DiD with State and Year Fixed Effects ---")
year_fe_cols = ['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE', 'NCHILD'] + [c for c in educ_dummies.columns] + [c for c in marst_dummies.columns] + [c for c in state_dummies.columns] + [c for c in year_dummies.columns]
X4 = df_analysis[year_fe_cols]
model4, se4, vcov4 = run_wls_clustered(y, X4, weights, clusters)

print("Key DiD coefficients:")
for var, idx in [('ELIGIBLE', 1), ('ELIGIBLE_AFTER', 2)]:
    coef = model4.params.iloc[idx]
    se = se4[idx]
    t = coef / se
    p = 2 * stats.t.sf(abs(t), n_clust - 1)
    print(f"  {var}: {coef:.4f} (SE: {se:.4f}, t: {t:.2f}, p: {p:.4f})")

# Model 5: DiD with state policy variables
print("\n--- Model 5: DiD with State Policy Controls ---")
policy_cols = ['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE', 'NCHILD',
               'DRIVERSLICENSES', 'INSTATETUITION', 'EVERIFY', 'LFPR', 'UNEMP'] + [c for c in educ_dummies.columns] + [c for c in marst_dummies.columns]
X5 = df_analysis[policy_cols]
model5, se5, vcov5 = run_wls_clustered(y, X5, weights, clusters)

print("Key coefficients:")
policy_vars = ['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'DRIVERSLICENSES', 'INSTATETUITION', 'EVERIFY', 'LFPR', 'UNEMP']
for var in policy_vars:
    idx = list(X5.columns).index(var) + 1  # +1 for intercept
    coef = model5.params.iloc[idx]
    se = se5[idx]
    t = coef / se
    p = 2 * stats.t.sf(abs(t), n_clust - 1)
    print(f"  {var}: {coef:.4f} (SE: {se:.4f})")

# ===== RESULTS SUMMARY TABLE =====
print("\n" + "=" * 60)
print("SUMMARY OF DIFFERENCE-IN-DIFFERENCES ESTIMATES")
print("=" * 60)

# Extract DiD coefficient and SE for each model
results_summary = pd.DataFrame({
    'Model': ['1. Basic DiD', '2. Demographics', '3. State FE', '4. State+Year FE', '5. Policy Controls'],
    'DiD_Estimate': [
        model1.params.iloc[3],  # ELIGIBLE_AFTER
        model2.params.iloc[3],
        model3.params.iloc[3],
        model4.params.iloc[2],  # Different index due to dropped AFTER
        model5.params.iloc[3]
    ],
    'Std_Error': [
        se1[3],
        se2[3],
        se3[3],
        se4[2],
        se5[3]
    ]
})

results_summary['t_stat'] = results_summary['DiD_Estimate'] / results_summary['Std_Error']
results_summary['p_value'] = 2 * stats.t.sf(np.abs(results_summary['t_stat']), n_clust - 1)
results_summary['95% CI Lower'] = results_summary['DiD_Estimate'] - 1.96 * results_summary['Std_Error']
results_summary['95% CI Upper'] = results_summary['DiD_Estimate'] + 1.96 * results_summary['Std_Error']

print(results_summary.round(4).to_string(index=False))

# ===== PARALLEL TRENDS ANALYSIS =====
print("\n" + "=" * 60)
print("PARALLEL TRENDS ANALYSIS")
print("=" * 60)

# Calculate yearly means by group
yearly_means = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'mean_FT': np.average(x['FT'], weights=x['PERWT']),
        'n': len(x)
    })
).reset_index()

print("\nYearly Full-Time Employment Rates by Group:")
yearly_pivot = yearly_means.pivot(index='YEAR', columns='ELIGIBLE', values='mean_FT')
yearly_pivot.columns = ['Control (31-35)', 'Treatment (26-30)']
print(yearly_pivot.round(4))

# Pre-trend test
print("\nPre-Treatment Trends Test (2008-2011):")
df_pre = df[df['AFTER'] == 0].copy()
df_pre['YEAR_2009'] = (df_pre['YEAR'] == 2009).astype(int)
df_pre['YEAR_2010'] = (df_pre['YEAR'] == 2010).astype(int)
df_pre['YEAR_2011'] = (df_pre['YEAR'] == 2011).astype(int)
df_pre['ELIG_2009'] = df_pre['ELIGIBLE'] * df_pre['YEAR_2009']
df_pre['ELIG_2010'] = df_pre['ELIGIBLE'] * df_pre['YEAR_2010']
df_pre['ELIG_2011'] = df_pre['ELIGIBLE'] * df_pre['YEAR_2011']

X_pre = df_pre[['ELIGIBLE', 'YEAR_2009', 'YEAR_2010', 'YEAR_2011', 'ELIG_2009', 'ELIG_2010', 'ELIG_2011']]
y_pre = df_pre['FT']
w_pre = df_pre['PERWT']
c_pre = df_pre['STATEFIP'].values

pretrend_model, pretrend_se, pretrend_vcov = run_wls_clustered(y_pre, X_pre, w_pre, c_pre)

print("\nDifferential pre-trend interactions:")
for i, var in enumerate(['ELIG_2009', 'ELIG_2010', 'ELIG_2011']):
    idx = i + 5  # After intercept and main effects
    coef = pretrend_model.params.iloc[idx]
    se = pretrend_se[idx]
    t = coef / se
    p = 2 * stats.t.sf(abs(t), len(np.unique(c_pre)) - 1)
    print(f"  {var}: {coef:.4f} (SE: {se:.4f}, p: {p:.4f})")

# ===== CREATE VISUALIZATIONS =====
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# Plot 1: Trends over time
fig, ax = plt.subplots(figsize=(10, 6))

for eligible in [0, 1]:
    data = yearly_means[yearly_means['ELIGIBLE'] == eligible]
    label = 'Treatment (26-30)' if eligible == 1 else 'Control (31-35)'
    color = 'red' if eligible == 1 else 'blue'
    ax.plot(data['YEAR'], data['mean_FT'], marker='o', label=label, color=color, linewidth=2, markersize=8)

ax.axvline(x=2012, color='gray', linestyle='--', alpha=0.7, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment by DACA Eligibility Group\n(Ages 26-30 vs. 31-35)', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig(output_path + 'trends_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(output_path + 'trends_plot.pdf', bbox_inches='tight')
plt.close()
print("Saved: trends_plot.png, trends_plot.pdf")

# Plot 2: DiD bar chart
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(2)
width = 0.35

pre_means = [control_pre, treat_pre]
post_means = [control_post, treat_post]

bars1 = ax.bar(x - width/2, pre_means, width, label='Pre-DACA (2008-2011)', color='steelblue')
bars2 = ax.bar(x + width/2, post_means, width, label='Post-DACA (2013-2016)', color='coral')

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: Full-Time Employment', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(['Control (31-35)', 'Treatment (26-30)'])
ax.legend()
ax.set_ylim(0, max(max(pre_means), max(post_means)) * 1.15)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_path + 'did_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(output_path + 'did_plot.pdf', bbox_inches='tight')
plt.close()
print("Saved: did_plot.png, did_plot.pdf")

# Plot 3: Coefficient plot
fig, ax = plt.subplots(figsize=(10, 6))

models = results_summary['Model'].tolist()
estimates = results_summary['DiD_Estimate'].tolist()
ci_lower = results_summary['95% CI Lower'].tolist()
ci_upper = results_summary['95% CI Upper'].tolist()

y_pos = np.arange(len(models))

ax.errorbar(estimates, y_pos, xerr=[np.array(estimates) - np.array(ci_lower),
                                      np.array(ci_upper) - np.array(estimates)],
            fmt='o', capsize=5, capthick=2, markersize=8, color='darkblue')

ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Estimate (Effect on Full-Time Employment)', fontsize=12)
ax.set_title('DACA Effect Estimates Across Specifications\n(with 95% Confidence Intervals)', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_path + 'coefficient_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(output_path + 'coefficient_plot.pdf', bbox_inches='tight')
plt.close()
print("Saved: coefficient_plot.png, coefficient_plot.pdf")

# ===== SAVE OUTPUT FOR LATEX =====
print("\n" + "=" * 60)
print("SAVING RESULTS FOR REPORT")
print("=" * 60)

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
    f.write(results_summary.round(4).to_string(index=False))
    f.write("\n\n")

    f.write("PREFERRED ESTIMATE (Model 3: State FE)\n")
    f.write("-" * 40 + "\n")
    pref_est = results_summary.loc[results_summary['Model'] == '3. State FE', 'DiD_Estimate'].values[0]
    pref_se = results_summary.loc[results_summary['Model'] == '3. State FE', 'Std_Error'].values[0]
    pref_ci_l = results_summary.loc[results_summary['Model'] == '3. State FE', '95% CI Lower'].values[0]
    pref_ci_u = results_summary.loc[results_summary['Model'] == '3. State FE', '95% CI Upper'].values[0]
    pref_p = results_summary.loc[results_summary['Model'] == '3. State FE', 'p_value'].values[0]
    f.write(f"Effect: {pref_est:.4f}\n")
    f.write(f"Std. Error: {pref_se:.4f}\n")
    f.write(f"95% CI: [{pref_ci_l:.4f}, {pref_ci_u:.4f}]\n")
    f.write(f"p-value: {pref_p:.4f}\n")

print("Saved: analysis_output.txt")

# Save results for LaTeX tables
results_summary.to_csv(output_path + 'regression_results.csv', index=False)
yearly_pivot.to_csv(output_path + 'yearly_means.csv')
print("Saved: regression_results.csv, yearly_means.csv")

# ===== FINAL RESULTS =====
print("\n" + "=" * 60)
print("FINAL RESULTS - PREFERRED ESTIMATE")
print("=" * 60)
print(f"\nModel: DiD with demographic controls and state fixed effects")
print(f"DiD Estimate (DACA effect on full-time employment): {pref_est:.4f}")
print(f"Clustered Standard Error: {pref_se:.4f}")
print(f"95% Confidence Interval: [{pref_ci_l:.4f}, {pref_ci_u:.4f}]")
print(f"p-value: {pref_p:.4f}")
print(f"Sample Size: {len(df_analysis)}")
print("=" * 60)

print("\nAnalysis complete!")
