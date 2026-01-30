"""
DACA Replication Analysis - Study 88
Independent replication examining the effect of DACA eligibility on full-time employment
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
pd.set_option('display.float_format', '{:.4f}'.format)

print("=" * 80)
print("DACA REPLICATION ANALYSIS - STUDY 88")
print("=" * 80)

# Load data
print("\n[1] LOADING DATA...")
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_88\data\prepared_data_labelled_version.csv"
df = pd.read_csv(data_path)
print(f"Total observations loaded: {len(df):,}")

# Basic data overview
print("\n[2] DATA OVERVIEW")
print("-" * 40)
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Year range: {df['YEAR'].min()} - {df['YEAR'].max()}")

# Check key variables
print("\n[3] KEY VARIABLE DISTRIBUTIONS")
print("-" * 40)
print(f"\nELIGIBLE (Treatment indicator):")
print(df['ELIGIBLE'].value_counts())
print(f"\nAFTER (Post-treatment indicator):")
print(df['AFTER'].value_counts())
print(f"\nFT (Full-time employment outcome):")
print(df['FT'].value_counts())

# Sample size by year and group
print("\n[4] SAMPLE SIZE BY YEAR AND TREATMENT GROUP")
print("-" * 40)
year_group = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack(fill_value=0)
year_group.columns = ['Control (31-35)', 'Treatment (26-30)']
print(year_group)
print(f"\nTotal by group:")
print(df.groupby('ELIGIBLE').size())

# Check age distribution to verify groups
print("\n[5] AGE DISTRIBUTION CHECK")
print("-" * 40)
print("Age at June 2012 by ELIGIBLE status:")
age_stats = df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe()
print(age_stats)

# Descriptive statistics - Demographics
print("\n[6] DEMOGRAPHIC CHARACTERISTICS BY GROUP")
print("-" * 40)

# Sex distribution
print("\nSex distribution:")
sex_by_group = pd.crosstab(df['ELIGIBLE'], df['SEX'], normalize='index') * 100
print(sex_by_group)

# Education distribution
print("\nEducation distribution:")
educ_by_group = pd.crosstab(df['ELIGIBLE'], df['EDUC_RECODE'], normalize='index') * 100
print(educ_by_group)

# Marital status
print("\nMarital status distribution:")
marst_by_group = pd.crosstab(df['ELIGIBLE'], df['MARST'], normalize='index') * 100
print(marst_by_group)

# =============================================
# MAIN ANALYSIS: DIFFERENCE-IN-DIFFERENCES
# =============================================

print("\n" + "=" * 80)
print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# Calculate basic means for DiD table
print("\n[7] 2x2 TABLE OF MEAN FULL-TIME EMPLOYMENT RATES")
print("-" * 40)

# Unweighted means
means_table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
means_table.index = ['Control (31-35)', 'Treatment (26-30)']
means_table.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print("\nUnweighted means:")
print(means_table)

# Calculate DiD manually
pre_control = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]['FT'].mean()
post_control = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]['FT'].mean()
pre_treat = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]['FT'].mean()

did_control = post_control - pre_control
did_treat = post_treat - pre_treat
did_estimate = did_treat - did_control

print(f"\nControl group change: {post_control:.4f} - {pre_control:.4f} = {did_control:.4f}")
print(f"Treatment group change: {post_treat:.4f} - {pre_treat:.4f} = {did_treat:.4f}")
print(f"DiD estimate (unweighted): {did_estimate:.4f}")

# Weighted means
print("\n[8] WEIGHTED 2x2 TABLE")
print("-" * 40)

def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

weighted_means = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean)
weighted_means_table = weighted_means.unstack()
weighted_means_table.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_means_table.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print("\nWeighted means:")
print(weighted_means_table)

# Calculate weighted DiD
pre_control_w = weighted_means.loc[(0, 0)]
post_control_w = weighted_means.loc[(0, 1)]
pre_treat_w = weighted_means.loc[(1, 0)]
post_treat_w = weighted_means.loc[(1, 1)]

did_control_w = post_control_w - pre_control_w
did_treat_w = post_treat_w - pre_treat_w
did_estimate_w = did_treat_w - did_control_w

print(f"\nControl group change (weighted): {post_control_w:.4f} - {pre_control_w:.4f} = {did_control_w:.4f}")
print(f"Treatment group change (weighted): {post_treat_w:.4f} - {pre_treat_w:.4f} = {did_treat_w:.4f}")
print(f"DiD estimate (weighted): {did_estimate_w:.4f}")

# =============================================
# REGRESSION ANALYSIS
# =============================================

print("\n" + "=" * 80)
print("REGRESSION-BASED DiD ESTIMATION")
print("=" * 80)

# Model 1: Basic DiD (unweighted)
print("\n[9] MODEL 1: Basic DiD (Unweighted OLS)")
print("-" * 40)
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])

# Model 2: Weighted DiD using WLS
print("\n[10] MODEL 2: Weighted DiD (WLS with PERWT)")
print("-" * 40)

model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with robust standard errors
print("\n[11] MODEL 3: Weighted DiD with Robust SE")
print("-" * 40)
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: DiD with covariates using matrix approach
print("\n[12] MODEL 4: Weighted DiD with Demographic Controls")
print("-" * 40)

# Create proper binary variables
df['MALE'] = (df['SEX'] == 'Male').astype(int) if df['SEX'].dtype == 'object' else (df['SEX'] == 1).astype(int)

# Simple model with sex control
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary().tables[1])

# Model 5: DiD with year fixed effects using matrix approach
print("\n[13] MODEL 5: Weighted DiD with Year Fixed Effects")
print("-" * 40)

# Create year dummies manually with clean names
for year in df['YEAR'].unique():
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)

# Reference year is 2008
year_cols = [f'year_{y}' for y in sorted(df['YEAR'].unique()) if y != 2008]

# Build design matrix manually
X_vars = ['ELIGIBLE', 'ELIGIBLE_AFTER'] + year_cols
X = df[X_vars].copy()
X = sm.add_constant(X)
y = df['FT']
weights = df['PERWT']

model5 = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')
print("\nCoefficients:")
for var in ['const', 'ELIGIBLE', 'ELIGIBLE_AFTER']:
    coef = model5.params[var]
    se = model5.bse[var]
    pval = model5.pvalues[var]
    print(f"{var:20s}: {coef:8.4f} (SE: {se:.4f}, p: {pval:.4f})")

print("\nYear Fixed Effects:")
for col in year_cols:
    coef = model5.params[col]
    se = model5.bse[col]
    print(f"{col:15s}: {coef:8.4f} (SE: {se:.4f})")

# Model 6: Full model with controls and year FE
print("\n[14] MODEL 6: Full Model (Year FE + Sex Control)")
print("-" * 40)

X_full = df[['ELIGIBLE', 'ELIGIBLE_AFTER', 'MALE'] + year_cols].copy()
X_full = sm.add_constant(X_full)

model6 = sm.WLS(y, X_full, weights=weights).fit(cov_type='HC1')
print("\nKey Coefficients:")
for var in ['const', 'ELIGIBLE', 'ELIGIBLE_AFTER', 'MALE']:
    coef = model6.params[var]
    se = model6.bse[var]
    pval = model6.pvalues[var]
    print(f"{var:20s}: {coef:8.4f} (SE: {se:.4f}, p: {pval:.4f})")

# Model 7: Add education controls
print("\n[15] MODEL 7: Full Model with Education Controls")
print("-" * 40)

# Create education dummies with clean names
educ_categories = df['EDUC_RECODE'].dropna().unique()
for cat in educ_categories:
    if pd.notna(cat):
        clean_name = str(cat).replace(' ', '_').replace('-', '_').replace("'", "")
        df[f'educ_{clean_name}'] = (df['EDUC_RECODE'] == cat).astype(int)

# Drop one for reference (Less than High School as reference)
educ_cols = [col for col in df.columns if col.startswith('educ_') and 'Less' not in col]

X_full2 = df[['ELIGIBLE', 'ELIGIBLE_AFTER', 'MALE'] + educ_cols + year_cols].copy()
X_full2 = sm.add_constant(X_full2)

model7 = sm.WLS(y, X_full2, weights=weights).fit(cov_type='HC1')
print("\nKey Coefficients:")
for var in ['const', 'ELIGIBLE', 'ELIGIBLE_AFTER', 'MALE']:
    coef = model7.params[var]
    se = model7.bse[var]
    pval = model7.pvalues[var]
    print(f"{var:20s}: {coef:8.4f} (SE: {se:.4f}, p: {pval:.4f})")

print("\nEducation Effects:")
for col in educ_cols:
    coef = model7.params[col]
    se = model7.bse[col]
    print(f"{col:30s}: {coef:8.4f} (SE: {se:.4f})")

# =============================================
# ROBUSTNESS CHECKS
# =============================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# Check 1: Parallel trends - pre-treatment trends
print("\n[16] PRE-TREATMENT TREND CHECK")
print("-" * 40)

pre_data = df[df['AFTER'] == 0].copy()
pre_trends = pre_data.groupby(['YEAR', 'ELIGIBLE']).apply(weighted_mean).unstack()
pre_trends.columns = ['Control', 'Treatment']
print("\nFull-time employment rates by year (pre-treatment):")
print(pre_trends)
print(f"\nDifference (Treatment - Control) by year:")
print(pre_trends['Treatment'] - pre_trends['Control'])

# Check 2: Year-by-year effects
print("\n[17] YEAR-BY-YEAR TREATMENT EFFECTS")
print("-" * 40)

yearly_effects = []
for year in sorted(df['YEAR'].unique()):
    year_data = df[df['YEAR'] == year]
    treat_mean = np.average(year_data[year_data['ELIGIBLE'] == 1]['FT'],
                            weights=year_data[year_data['ELIGIBLE'] == 1]['PERWT'])
    ctrl_mean = np.average(year_data[year_data['ELIGIBLE'] == 0]['FT'],
                           weights=year_data[year_data['ELIGIBLE'] == 0]['PERWT'])
    yearly_effects.append({
        'Year': year,
        'Treatment': treat_mean,
        'Control': ctrl_mean,
        'Difference': treat_mean - ctrl_mean
    })

yearly_df = pd.DataFrame(yearly_effects)
print(yearly_df.to_string(index=False))

# Calculate average pre-treatment difference
pre_diff = yearly_df[yearly_df['Year'] < 2012]['Difference'].mean()
post_diff = yearly_df[yearly_df['Year'] > 2012]['Difference'].mean()
print(f"\nAverage pre-treatment difference: {pre_diff:.4f}")
print(f"Average post-treatment difference: {post_diff:.4f}")
print(f"Change in difference (DiD): {post_diff - pre_diff:.4f}")

# Check 3: Effect by sex
print("\n[18] HETEROGENEOUS EFFECTS BY SEX")
print("-" * 40)

for sex_val in df['SEX'].unique():
    sex_data = df[df['SEX'] == sex_val]
    sex_model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=sex_data, weights=sex_data['PERWT']).fit(cov_type='HC1')
    n_obs = len(sex_data)
    print(f"\n{sex_val} (n={n_obs}):")
    print(f"  DiD estimate: {sex_model.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  SE: {sex_model.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"  P-value: {sex_model.pvalues['ELIGIBLE_AFTER']:.4f}")

# Check 4: Effect by education level
print("\n[19] HETEROGENEOUS EFFECTS BY EDUCATION")
print("-" * 40)

for educ in df['EDUC_RECODE'].unique():
    educ_data = df[df['EDUC_RECODE'] == educ]
    if len(educ_data) > 100:  # Only if sufficient sample size
        try:
            educ_model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                                data=educ_data, weights=educ_data['PERWT']).fit(cov_type='HC1')
            print(f"\n{educ} (n={len(educ_data)}):")
            print(f"  DiD estimate: {educ_model.params['ELIGIBLE_AFTER']:.4f}")
            print(f"  SE: {educ_model.bse['ELIGIBLE_AFTER']:.4f}")
            print(f"  P-value: {educ_model.pvalues['ELIGIBLE_AFTER']:.4f}")
        except Exception as e:
            print(f"\n{educ}: Could not estimate ({e})")

# Check 5: Placebo test - artificial treatment in 2010
print("\n[20] PLACEBO TEST (Pre-treatment period only)")
print("-" * 40)

pre_only = df[df['AFTER'] == 0].copy()
pre_only['PLACEBO_AFTER'] = (pre_only['YEAR'] >= 2010).astype(int)
pre_only['PLACEBO_INTERACTION'] = pre_only['ELIGIBLE'] * pre_only['PLACEBO_AFTER']

placebo_model = smf.wls('FT ~ ELIGIBLE + PLACEBO_AFTER + PLACEBO_INTERACTION',
                        data=pre_only, weights=pre_only['PERWT']).fit(cov_type='HC1')
print(f"\nPlacebo DiD (2008-2009 vs 2010-2011):")
print(f"  Coefficient: {placebo_model.params['PLACEBO_INTERACTION']:.4f}")
print(f"  SE: {placebo_model.bse['PLACEBO_INTERACTION']:.4f}")
print(f"  P-value: {placebo_model.pvalues['PLACEBO_INTERACTION']:.4f}")

if placebo_model.pvalues['PLACEBO_INTERACTION'] > 0.1:
    print("  Interpretation: No significant pre-trend (supports parallel trends assumption)")
else:
    print("  Interpretation: Significant pre-trend detected (parallel trends may be violated)")

# =============================================
# SUMMARY OF RESULTS
# =============================================

print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print("\n[21] MAIN RESULTS TABLE")
print("-" * 40)

results_summary = []
results_summary.append({
    'Model': '1. Basic DiD (Unweighted)',
    'Coefficient': model1.params['ELIGIBLE_AFTER'],
    'Std_Error': model1.bse['ELIGIBLE_AFTER'],
    'P_value': model1.pvalues['ELIGIBLE_AFTER'],
    'N': int(model1.nobs)
})
results_summary.append({
    'Model': '2. Weighted DiD',
    'Coefficient': model2.params['ELIGIBLE_AFTER'],
    'Std_Error': model2.bse['ELIGIBLE_AFTER'],
    'P_value': model2.pvalues['ELIGIBLE_AFTER'],
    'N': int(model2.nobs)
})
results_summary.append({
    'Model': '3. Weighted DiD (Robust SE)',
    'Coefficient': model3.params['ELIGIBLE_AFTER'],
    'Std_Error': model3.bse['ELIGIBLE_AFTER'],
    'P_value': model3.pvalues['ELIGIBLE_AFTER'],
    'N': int(model3.nobs)
})
results_summary.append({
    'Model': '4. With Sex Control',
    'Coefficient': model4.params['ELIGIBLE_AFTER'],
    'Std_Error': model4.bse['ELIGIBLE_AFTER'],
    'P_value': model4.pvalues['ELIGIBLE_AFTER'],
    'N': int(model4.nobs)
})
results_summary.append({
    'Model': '5. With Year FE',
    'Coefficient': model5.params['ELIGIBLE_AFTER'],
    'Std_Error': model5.bse['ELIGIBLE_AFTER'],
    'P_value': model5.pvalues['ELIGIBLE_AFTER'],
    'N': int(model5.nobs)
})
results_summary.append({
    'Model': '6. Year FE + Sex',
    'Coefficient': model6.params['ELIGIBLE_AFTER'],
    'Std_Error': model6.bse['ELIGIBLE_AFTER'],
    'P_value': model6.pvalues['ELIGIBLE_AFTER'],
    'N': int(model6.nobs)
})
results_summary.append({
    'Model': '7. Full (Year + Sex + Educ)',
    'Coefficient': model7.params['ELIGIBLE_AFTER'],
    'Std_Error': model7.bse['ELIGIBLE_AFTER'],
    'P_value': model7.pvalues['ELIGIBLE_AFTER'],
    'N': int(model7.nobs)
})

results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False))

# Preferred estimate
print("\n" + "=" * 80)
print("PREFERRED ESTIMATE")
print("=" * 80)

preferred_model = model5  # Year FE model as preferred
preferred_coef = preferred_model.params['ELIGIBLE_AFTER']
preferred_se = preferred_model.bse['ELIGIBLE_AFTER']
preferred_ci_low = preferred_coef - 1.96 * preferred_se
preferred_ci_high = preferred_coef + 1.96 * preferred_se
preferred_pval = preferred_model.pvalues['ELIGIBLE_AFTER']

print(f"\nPreferred Model: Weighted DiD with Year Fixed Effects (Model 5)")
print(f"Effect of DACA eligibility on full-time employment:")
print(f"  Coefficient: {preferred_coef:.4f}")
print(f"  Standard Error: {preferred_se:.4f}")
print(f"  95% CI: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"  P-value: {preferred_pval:.4f}")
print(f"  Sample Size: {int(preferred_model.nobs):,}")

# Effect size interpretation
print(f"\nInterpretation:")
print(f"  DACA eligibility is associated with a {preferred_coef*100:.2f} percentage point")
if preferred_coef > 0:
    print(f"  INCREASE in the probability of full-time employment.")
else:
    print(f"  DECREASE in the probability of full-time employment.")

if preferred_pval < 0.01:
    print(f"  This effect is statistically significant at the 1% level (p = {preferred_pval:.4f}).")
elif preferred_pval < 0.05:
    print(f"  This effect is statistically significant at the 5% level (p = {preferred_pval:.4f}).")
elif preferred_pval < 0.10:
    print(f"  This effect is statistically significant at the 10% level (p = {preferred_pval:.4f}).")
else:
    print(f"  This effect is NOT statistically significant at conventional levels (p = {preferred_pval:.4f}).")

# Save key results to file for LaTeX
print("\n\nSaving results for LaTeX report...")
results_for_latex = {
    'n_obs': len(df),
    'n_treatment': len(df[df['ELIGIBLE'] == 1]),
    'n_control': len(df[df['ELIGIBLE'] == 0]),
    'pre_treat_ft': pre_treat,
    'post_treat_ft': post_treat,
    'pre_ctrl_ft': pre_control,
    'post_ctrl_ft': post_control,
    'pre_treat_ft_w': pre_treat_w,
    'post_treat_ft_w': post_treat_w,
    'pre_ctrl_ft_w': pre_control_w,
    'post_ctrl_ft_w': post_control_w,
    'did_unweighted': did_estimate,
    'did_weighted': did_estimate_w,
    'model1_coef': model1.params['ELIGIBLE_AFTER'],
    'model1_se': model1.bse['ELIGIBLE_AFTER'],
    'model1_pval': model1.pvalues['ELIGIBLE_AFTER'],
    'model2_coef': model2.params['ELIGIBLE_AFTER'],
    'model2_se': model2.bse['ELIGIBLE_AFTER'],
    'model2_pval': model2.pvalues['ELIGIBLE_AFTER'],
    'model3_coef': model3.params['ELIGIBLE_AFTER'],
    'model3_se': model3.bse['ELIGIBLE_AFTER'],
    'model3_pval': model3.pvalues['ELIGIBLE_AFTER'],
    'model4_coef': model4.params['ELIGIBLE_AFTER'],
    'model4_se': model4.bse['ELIGIBLE_AFTER'],
    'model4_pval': model4.pvalues['ELIGIBLE_AFTER'],
    'model5_coef': model5.params['ELIGIBLE_AFTER'],
    'model5_se': model5.bse['ELIGIBLE_AFTER'],
    'model5_pval': model5.pvalues['ELIGIBLE_AFTER'],
    'model6_coef': model6.params['ELIGIBLE_AFTER'],
    'model6_se': model6.bse['ELIGIBLE_AFTER'],
    'model6_pval': model6.pvalues['ELIGIBLE_AFTER'],
    'model7_coef': model7.params['ELIGIBLE_AFTER'],
    'model7_se': model7.bse['ELIGIBLE_AFTER'],
    'model7_pval': model7.pvalues['ELIGIBLE_AFTER'],
    'preferred_coef': preferred_coef,
    'preferred_se': preferred_se,
    'preferred_ci_low': preferred_ci_low,
    'preferred_ci_high': preferred_ci_high,
    'preferred_pval': preferred_pval,
    'preferred_n': int(preferred_model.nobs),
    'placebo_coef': placebo_model.params['PLACEBO_INTERACTION'],
    'placebo_se': placebo_model.bse['PLACEBO_INTERACTION'],
    'placebo_pval': placebo_model.pvalues['PLACEBO_INTERACTION'],
    'male_effect': model6.params['MALE'],
    'male_se': model6.bse['MALE']
}

# Save to CSV for easy reading
results_df_save = pd.DataFrame([results_for_latex])
results_df_save.to_csv(r"C:\Users\seraf\DACA Results Task 3\replication_88\results_summary.csv", index=False)

# Also save the yearly effects
yearly_df.to_csv(r"C:\Users\seraf\DACA Results Task 3\replication_88\yearly_effects.csv", index=False)

print("\nResults saved to results_summary.csv and yearly_effects.csv")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
