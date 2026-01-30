"""
DACA Replication Study - Analysis Script
Research Question: Impact of DACA eligibility on full-time employment
Method: Difference-in-Differences (DiD)
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

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n1. LOADING DATA...")
print("-" * 40)

df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Columns: {df.shape[1]}")

# =============================================================================
# 2. DATA EXPLORATION
# =============================================================================
print("\n2. DATA EXPLORATION")
print("-" * 40)

# Check key variables
print("\nKey variables summary:")
print(f"  ELIGIBLE: {df['ELIGIBLE'].value_counts().to_dict()}")
print(f"  AFTER: {df['AFTER'].value_counts().to_dict()}")
print(f"  FT: {df['FT'].value_counts().to_dict()}")
print(f"  Years in data: {sorted(df['YEAR'].unique())}")

# Check for missing values in key variables
key_vars = ['ELIGIBLE', 'AFTER', 'FT', 'PERWT', 'YEAR', 'AGE', 'SEX', 'STATEFIP']
print("\nMissing values in key variables:")
for var in key_vars:
    missing = df[var].isna().sum()
    print(f"  {var}: {missing}")

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 40)

# Group sizes
print("\nSample sizes by group:")
print(df.groupby(['ELIGIBLE', 'AFTER']).size().unstack())

# Weighted sample sizes
print("\nWeighted sample sizes (population estimates):")
weighted_n = df.groupby(['ELIGIBLE', 'AFTER'])['PERWT'].sum()
print(weighted_n.unstack())

# Full-time employment rates by group and period
print("\nFull-time employment rates (weighted):")
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

ft_rates = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean)
print(ft_rates.unstack())

# Simple DiD calculation
ft_matrix = ft_rates.unstack()
simple_did = (ft_matrix.loc[1, 1] - ft_matrix.loc[1, 0]) - (ft_matrix.loc[0, 1] - ft_matrix.loc[0, 0])
print(f"\nSimple DiD estimate: {simple_did:.4f}")

# Descriptive stats by eligible status
print("\nDescriptive statistics by ELIGIBLE status:")
print("\nTreatment group (ELIGIBLE=1):")
treated = df[df['ELIGIBLE'] == 1]
print(f"  N: {len(treated):,}")
print(f"  Mean age: {treated['AGE'].mean():.1f}")
print(f"  Age range: {treated['AGE'].min()}-{treated['AGE'].max()}")
print(f"  % Female: {(treated['SEX'] == 2).mean()*100:.1f}%")
print(f"  FT rate (pre): {treated[treated['AFTER']==0]['FT'].mean()*100:.1f}%")
print(f"  FT rate (post): {treated[treated['AFTER']==1]['FT'].mean()*100:.1f}%")

print("\nControl group (ELIGIBLE=0):")
control = df[df['ELIGIBLE'] == 0]
print(f"  N: {len(control):,}")
print(f"  Mean age: {control['AGE'].mean():.1f}")
print(f"  Age range: {control['AGE'].min()}-{control['AGE'].max()}")
print(f"  % Female: {(control['SEX'] == 2).mean()*100:.1f}%")
print(f"  FT rate (pre): {control[control['AFTER']==0]['FT'].mean()*100:.1f}%")
print(f"  FT rate (post): {control[control['AFTER']==1]['FT'].mean()*100:.1f}%")

# Education distribution by group
print("\nEducation distribution by ELIGIBLE status (unweighted %):")
educ_cross = pd.crosstab(df['EDUC_RECODE'], df['ELIGIBLE'], normalize='columns') * 100
print(educ_cross)

# =============================================================================
# 4. PARALLEL TRENDS ANALYSIS
# =============================================================================
print("\n4. PARALLEL TRENDS ANALYSIS")
print("-" * 40)

# FT rates by year and group
print("\nFull-time employment rates by year and ELIGIBLE status (weighted):")
yearly_ft = df.groupby(['YEAR', 'ELIGIBLE']).apply(weighted_mean).unstack()
print(yearly_ft)

# Calculate year-over-year changes in pre-period
print("\nYear-over-year changes in FT rate (pre-period):")
pre_years = [2008, 2009, 2010, 2011]
for i in range(1, len(pre_years)):
    year_prev = pre_years[i-1]
    year_curr = pre_years[i]
    change_treated = yearly_ft.loc[year_curr, 1] - yearly_ft.loc[year_prev, 1]
    change_control = yearly_ft.loc[year_curr, 0] - yearly_ft.loc[year_prev, 0]
    diff_in_changes = change_treated - change_control
    print(f"  {year_prev}-{year_curr}: Treated={change_treated:.4f}, Control={change_control:.4f}, Diff={diff_in_changes:.4f}")

# =============================================================================
# 5. PREPARE VARIABLES FOR REGRESSION
# =============================================================================
print("\n5. PREPARING VARIABLES FOR REGRESSION...")
print("-" * 40)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Create female indicator (SEX=2 is female)
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Create married indicator (MARST=1 or 2 is married)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

# Convert EDUC_RECODE to numeric categories
educ_map = {
    'Less than High School': 1,
    'High School Degree': 2,
    'Some College': 3,
    'Two-Year Degree': 4,
    'BA+': 5
}
df['EDUC_NUM'] = df['EDUC_RECODE'].map(educ_map).fillna(2).astype(int)

# Create education dummies manually
df['EDUC_HS'] = (df['EDUC_NUM'] == 2).astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_NUM'] == 3).astype(int)
df['EDUC_2YR'] = (df['EDUC_NUM'] == 4).astype(int)
df['EDUC_BA'] = (df['EDUC_NUM'] == 5).astype(int)

print("Variables created: ELIGIBLE_AFTER, FEMALE, MARRIED, EDUC dummies")

# =============================================================================
# 6. DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n6. DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("-" * 40)

y = df['FT'].values.astype(float)
weights = df['PERWT'].values.astype(float)
clusters = df['STATEFIP'].values

# Model 1: Basic DiD (no covariates)
print("\nModel 1: Basic DiD (no covariates)")
print("-" * 40)

X1 = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']].values.astype(float)
X1 = sm.add_constant(X1)

model1 = sm.WLS(y, X1, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': clusters})

# Extract results
col_names1 = ['const', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']
print(f"\n{'Variable':<20} {'Coef':>10} {'Std Err':>10} {'z':>10} {'P>|z|':>10} {'[0.025':>10} {'0.975]':>10}")
print("-" * 80)
for i, var in enumerate(col_names1):
    print(f"{var:<20} {model1.params[i]:>10.4f} {model1.bse[i]:>10.4f} {model1.tvalues[i]:>10.3f} {model1.pvalues[i]:>10.4f} {model1.conf_int()[i,0]:>10.4f} {model1.conf_int()[i,1]:>10.4f}")

did_coef_m1 = model1.params[3]
did_se_m1 = model1.bse[3]
did_pval_m1 = model1.pvalues[3]
did_ci_m1 = model1.conf_int()[3]

print(f"\n*** DiD Estimate (Model 1): {did_coef_m1:.4f} ***")
print(f"    Standard Error: {did_se_m1:.4f}")
print(f"    95% CI: [{did_ci_m1[0]:.4f}, {did_ci_m1[1]:.4f}]")
print(f"    p-value: {did_pval_m1:.4f}")

# Model 2: DiD with demographic covariates
print("\n\nModel 2: DiD with demographic covariates")
print("-" * 40)

X2_cols = ['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD',
           'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_2YR', 'EDUC_BA']
X2 = df[X2_cols].values.astype(float)
X2 = sm.add_constant(X2)

model2 = sm.WLS(y, X2, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': clusters})

col_names2 = ['const'] + X2_cols
print(f"\n{'Variable':<20} {'Coef':>10} {'Std Err':>10} {'z':>10} {'P>|z|':>10} {'[0.025':>10} {'0.975]':>10}")
print("-" * 90)
for i, var in enumerate(col_names2):
    print(f"{var:<20} {model2.params[i]:>10.4f} {model2.bse[i]:>10.4f} {model2.tvalues[i]:>10.3f} {model2.pvalues[i]:>10.4f} {model2.conf_int()[i,0]:>10.4f} {model2.conf_int()[i,1]:>10.4f}")

did_idx_m2 = 3  # ELIGIBLE_AFTER index
did_coef_m2 = model2.params[did_idx_m2]
did_se_m2 = model2.bse[did_idx_m2]
did_pval_m2 = model2.pvalues[did_idx_m2]
did_ci_m2 = model2.conf_int()[did_idx_m2]

print(f"\n*** DiD Estimate (Model 2): {did_coef_m2:.4f} ***")
print(f"    Standard Error: {did_se_m2:.4f}")
print(f"    95% CI: [{did_ci_m2[0]:.4f}, {did_ci_m2[1]:.4f}]")
print(f"    p-value: {did_pval_m2:.4f}")

# Model 3: DiD with state and year fixed effects
print("\n\nModel 3: DiD with state and year fixed effects")
print("-" * 40)

# Create state and year dummies
states = sorted(df['STATEFIP'].unique())
years = sorted(df['YEAR'].unique())

# Reference: first state and first year
state_dummies = np.zeros((len(df), len(states)-1))
for i, state in enumerate(states[1:]):
    state_dummies[:, i] = (df['STATEFIP'] == state).astype(float)

year_dummies = np.zeros((len(df), len(years)-1))
for i, year in enumerate(years[1:]):
    year_dummies[:, i] = (df['YEAR'] == year).astype(float)

# Build X3: Note we drop AFTER because year dummies absorb it
X3_base = df[['ELIGIBLE', 'ELIGIBLE_AFTER', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD',
              'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_2YR', 'EDUC_BA']].values.astype(float)
X3 = np.column_stack([np.ones(len(df)), X3_base, state_dummies, year_dummies])

model3 = sm.WLS(y, X3, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': clusters})

# Print key coefficients only
key_vars_idx = list(range(11))  # First 11 variables (including constant)
key_var_names = ['const', 'ELIGIBLE', 'ELIGIBLE_AFTER', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD',
                 'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_2YR', 'EDUC_BA']

print(f"\n{'Variable':<20} {'Coef':>10} {'Std Err':>10} {'z':>10} {'P>|z|':>10} {'[0.025':>10} {'0.975]':>10}")
print("-" * 90)
for i, var in enumerate(key_var_names):
    print(f"{var:<20} {model3.params[i]:>10.4f} {model3.bse[i]:>10.4f} {model3.tvalues[i]:>10.3f} {model3.pvalues[i]:>10.4f} {model3.conf_int()[i,0]:>10.4f} {model3.conf_int()[i,1]:>10.4f}")
print(f"... plus {len(states)-1} state dummies and {len(years)-1} year dummies")

did_idx_m3 = 2  # ELIGIBLE_AFTER index in model 3
did_coef_m3 = model3.params[did_idx_m3]
did_se_m3 = model3.bse[did_idx_m3]
did_pval_m3 = model3.pvalues[did_idx_m3]
did_ci_m3 = model3.conf_int()[did_idx_m3]

print(f"\n*** DiD Estimate (Model 3): {did_coef_m3:.4f} ***")
print(f"    Standard Error: {did_se_m3:.4f}")
print(f"    95% CI: [{did_ci_m3[0]:.4f}, {did_ci_m3[1]:.4f}]")
print(f"    p-value: {did_pval_m3:.4f}")

# =============================================================================
# 7. ADDITIONAL ROBUSTNESS: Event Study / Pre-trends Test
# =============================================================================
print("\n7. EVENT STUDY ANALYSIS")
print("-" * 40)

# Create year-specific treatment effects (event study)
# Reference year: 2011 (last pre-treatment year)
event_years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in event_years:
    df[f'ELIGIBLE_Y{year}'] = (df['ELIGIBLE'] * (df['YEAR'] == year)).astype(int)

event_cols = [f'ELIGIBLE_Y{y}' for y in event_years]
X_event_base = df[['ELIGIBLE', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD',
                   'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_2YR', 'EDUC_BA'] + event_cols].values.astype(float)
X_event = np.column_stack([np.ones(len(df)), X_event_base, state_dummies, year_dummies])

model_event = sm.WLS(y, X_event, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': clusters})

print("\nEvent Study Coefficients (relative to 2011):")
print(f"{'Year':<10} {'Coef':>10} {'Std Err':>10} {'95% CI':>25}")
print("-" * 60)
event_results = {}
for i, year in enumerate(event_years):
    idx = 10 + i  # After base covariates
    coef = model_event.params[idx]
    se = model_event.bse[idx]
    ci = model_event.conf_int()[idx]
    event_results[year] = {'coef': coef, 'se': se, 'ci_lo': ci[0], 'ci_hi': ci[1]}
    print(f"{year:<10} {coef:>10.4f} {se:>10.4f} [{ci[0]:>10.4f}, {ci[1]:>10.4f}]")

# =============================================================================
# 8. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("8. SUMMARY OF RESULTS")
print("=" * 80)

print("\n" + "-" * 90)
print("DIFFERENCE-IN-DIFFERENCES ESTIMATES OF DACA EFFECT ON FULL-TIME EMPLOYMENT")
print("-" * 90)
print(f"{'Model':<50} {'Coef':>10} {'SE':>10} {'95% CI':>25} {'p-val':>10}")
print("-" * 90)
print(f"{'Model 1: Basic DiD':<50} {did_coef_m1:>10.4f} {did_se_m1:>10.4f} [{did_ci_m1[0]:>8.4f}, {did_ci_m1[1]:>8.4f}] {did_pval_m1:>10.4f}")
print(f"{'Model 2: + Demographics':<50} {did_coef_m2:>10.4f} {did_se_m2:>10.4f} [{did_ci_m2[0]:>8.4f}, {did_ci_m2[1]:>8.4f}] {did_pval_m2:>10.4f}")
print(f"{'Model 3: + State/Year FE':<50} {did_coef_m3:>10.4f} {did_se_m3:>10.4f} [{did_ci_m3[0]:>8.4f}, {did_ci_m3[1]:>8.4f}] {did_pval_m3:>10.4f}")
print("-" * 90)

# Sample size
print(f"\nTotal sample size: {len(df):,}")
print(f"Treatment group (ELIGIBLE=1): {len(df[df['ELIGIBLE']==1]):,}")
print(f"Control group (ELIGIBLE=0): {len(df[df['ELIGIBLE']==0]):,}")

# Preferred estimate
print("\n" + "=" * 80)
print("PREFERRED ESTIMATE")
print("=" * 80)
print(f"\nPreferred specification: Model 3 (DiD with covariates and state/year fixed effects)")
print(f"  Effect size: {did_coef_m3:.4f}")
print(f"  Standard error: {did_se_m3:.4f}")
print(f"  95% Confidence interval: [{did_ci_m3[0]:.4f}, {did_ci_m3[1]:.4f}]")
print(f"  Sample size: {len(df):,}")

# Interpretation
effect_pct = did_coef_m3 * 100
print(f"\nInterpretation:")
print(f"  DACA eligibility is associated with a {abs(effect_pct):.2f} percentage point")
if did_coef_m3 > 0:
    print(f"  INCREASE in the probability of full-time employment.")
else:
    print(f"  DECREASE in the probability of full-time employment.")

if did_pval_m3 < 0.05:
    print(f"  This effect is statistically significant at the 5% level (p={did_pval_m3:.4f}).")
elif did_pval_m3 < 0.10:
    print(f"  This effect is statistically significant at the 10% level (p={did_pval_m3:.4f}).")
else:
    print(f"  This effect is NOT statistically significant at conventional levels (p={did_pval_m3:.4f}).")

# =============================================================================
# 9. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n9. SAVING RESULTS...")

# Save key statistics to a file for the LaTeX report
results = {
    'simple_did': simple_did,
    'model1_coef': did_coef_m1,
    'model1_se': did_se_m1,
    'model1_ci_lo': did_ci_m1[0],
    'model1_ci_hi': did_ci_m1[1],
    'model1_pval': did_pval_m1,
    'model2_coef': did_coef_m2,
    'model2_se': did_se_m2,
    'model2_ci_lo': did_ci_m2[0],
    'model2_ci_hi': did_ci_m2[1],
    'model2_pval': did_pval_m2,
    'model3_coef': did_coef_m3,
    'model3_se': did_se_m3,
    'model3_ci_lo': did_ci_m3[0],
    'model3_ci_hi': did_ci_m3[1],
    'model3_pval': did_pval_m3,
    'n_total': len(df),
    'n_treated': len(df[df['ELIGIBLE']==1]),
    'n_control': len(df[df['ELIGIBLE']==0]),
    'ft_rate_treated_pre': treated[treated['AFTER']==0]['FT'].mean(),
    'ft_rate_treated_post': treated[treated['AFTER']==1]['FT'].mean(),
    'ft_rate_control_pre': control[control['AFTER']==0]['FT'].mean(),
    'ft_rate_control_post': control[control['AFTER']==1]['FT'].mean(),
}

# Save yearly FT rates
for year in sorted(df['YEAR'].unique()):
    for elig in [0, 1]:
        subset = df[(df['YEAR']==year) & (df['ELIGIBLE']==elig)]
        results[f'ft_rate_y{year}_e{elig}'] = np.average(subset['FT'], weights=subset['PERWT'])

# Add event study results
for year, res in event_results.items():
    results[f'event_y{year}_coef'] = res['coef']
    results[f'event_y{year}_se'] = res['se']

# Save to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('analysis_results.csv', index=False)
print("Results saved to analysis_results.csv")

# Save yearly rates for plotting
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'ft_rate': np.average(x['FT'], weights=x['PERWT']),
        'n': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()
yearly_rates.to_csv('yearly_ft_rates.csv', index=False)
print("Yearly rates saved to yearly_ft_rates.csv")

# Save descriptive statistics
desc_stats = {
    'Variable': ['N (total)', 'N (treated)', 'N (control)',
                 'FT rate - Treated Pre', 'FT rate - Treated Post',
                 'FT rate - Control Pre', 'FT rate - Control Post',
                 'Mean Age - Treated', 'Mean Age - Control',
                 '% Female - Treated', '% Female - Control'],
    'Value': [len(df), len(treated), len(control),
              f"{results['ft_rate_treated_pre']*100:.1f}%",
              f"{results['ft_rate_treated_post']*100:.1f}%",
              f"{results['ft_rate_control_pre']*100:.1f}%",
              f"{results['ft_rate_control_post']*100:.1f}%",
              f"{treated['AGE'].mean():.1f}",
              f"{control['AGE'].mean():.1f}",
              f"{(treated['SEX']==2).mean()*100:.1f}%",
              f"{(control['SEX']==2).mean()*100:.1f}%"]
}
pd.DataFrame(desc_stats).to_csv('descriptive_stats.csv', index=False)
print("Descriptive stats saved to descriptive_stats.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
