"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
ethnically Hispanic-Mexican Mexican-born people living in the US.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("="*70)
print("DACA REPLICATION ANALYSIS")
print("="*70)

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"\nDataset loaded: {df.shape[0]} observations, {df.shape[1]} variables")

# Convert key variables to numeric
df['FT'] = pd.to_numeric(df['FT'], errors='coerce')
df['ELIGIBLE'] = pd.to_numeric(df['ELIGIBLE'], errors='coerce')
df['AFTER'] = pd.to_numeric(df['AFTER'], errors='coerce')
df['PERWT'] = pd.to_numeric(df['PERWT'], errors='coerce')
df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
df['SEX'] = pd.to_numeric(df['SEX'], errors='coerce')
df['MARST'] = pd.to_numeric(df['MARST'], errors='coerce')
df['NCHILD'] = pd.to_numeric(df['NCHILD'], errors='coerce')
df['EDUC'] = pd.to_numeric(df['EDUC'], errors='coerce')
df['UHRSWORK'] = pd.to_numeric(df['UHRSWORK'], errors='coerce')
df['STATEFIP'] = pd.to_numeric(df['STATEFIP'], errors='coerce')

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# =============================================================================
# 2. DATA EXPLORATION AND SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*70)
print("SECTION 2: DATA EXPLORATION")
print("="*70)

print("\n2.1 Years in dataset:")
print(sorted(df['YEAR'].unique()))

print("\n2.2 Sample sizes by year:")
print(df.groupby('YEAR').size())

print("\n2.3 Treatment and control group sizes:")
print(f"Treatment (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum()}")
print(f"Control (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum()}")

print("\n2.4 Pre/Post period sizes:")
print(f"Pre-DACA (AFTER=0): {(df['AFTER']==0).sum()}")
print(f"Post-DACA (AFTER=1): {(df['AFTER']==1).sum()}")

print("\n2.5 Full-time employment rates:")
print(f"Overall FT rate: {df['FT'].mean():.4f}")

# =============================================================================
# 3. DESCRIPTIVE STATISTICS BY GROUP
# =============================================================================
print("\n" + "="*70)
print("SECTION 3: DESCRIPTIVE STATISTICS")
print("="*70)

# Key variables for balance table
vars_for_balance = ['AGE', 'SEX', 'MARST', 'NCHILD', 'EDUC', 'UHRSWORK', 'FT']

print("\n3.1 Summary statistics by treatment status (pre-period only):")
pre_df = df[df['AFTER']==0]
for v in vars_for_balance:
    if v in df.columns:
        treat_mean = pre_df[pre_df['ELIGIBLE']==1][v].mean()
        ctrl_mean = pre_df[pre_df['ELIGIBLE']==0][v].mean()
        diff = treat_mean - ctrl_mean
        print(f"{v:15s}: Treat={treat_mean:8.3f}, Control={ctrl_mean:8.3f}, Diff={diff:8.3f}")

# Full-time employment by group and period
print("\n3.2 Full-time employment by group and period:")
ft_summary = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
print(ft_summary)

# Weighted means
print("\n3.3 Weighted full-time employment by group and period:")
for elig in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE']==elig) & (df['AFTER']==after)]
        weighted_mean = np.average(subset['FT'], weights=subset['PERWT'])
        print(f"ELIGIBLE={elig}, AFTER={after}: {weighted_mean:.4f}")

# =============================================================================
# 4. SIMPLE DIFFERENCE-IN-DIFFERENCES
# =============================================================================
print("\n" + "="*70)
print("SECTION 4: SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("="*70)

# Unweighted DiD
treated_before = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
treated_after = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
control_before = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
control_after = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

did_estimate = (treated_after - treated_before) - (control_after - control_before)

print("\n4.1 Unweighted DiD:")
print(f"Treated before: {treated_before:.4f}")
print(f"Treated after:  {treated_after:.4f}")
print(f"Control before: {control_before:.4f}")
print(f"Control after:  {control_after:.4f}")
print(f"Treated change: {treated_after - treated_before:.4f}")
print(f"Control change: {control_after - control_before:.4f}")
print(f"DiD estimate:   {did_estimate:.4f}")

# Weighted DiD
def weighted_mean(data, weights):
    return np.average(data, weights=weights)

t_b_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'],
                       df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['PERWT'])
t_a_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'],
                       df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['PERWT'])
c_b_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'],
                       df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['PERWT'])
c_a_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'],
                       df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['PERWT'])

did_weighted = (t_a_w - t_b_w) - (c_a_w - c_b_w)

print("\n4.2 Weighted DiD:")
print(f"Treated before: {t_b_w:.4f}")
print(f"Treated after:  {t_a_w:.4f}")
print(f"Control before: {c_b_w:.4f}")
print(f"Control after:  {c_a_w:.4f}")
print(f"DiD estimate (weighted): {did_weighted:.4f}")

# =============================================================================
# 5. REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES
# =============================================================================
print("\n" + "="*70)
print("SECTION 5: REGRESSION-BASED DiD MODELS")
print("="*70)

# Model 1: Basic DiD
print("\n5.1 Model 1: Basic DiD (FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER)")
X1 = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']].astype(float)
y = df['FT'].astype(float)
model1 = sm.OLS(y, sm.add_constant(X1)).fit()
print(model1.summary().tables[1])

# Model 1 with robust SEs
print("\n5.2 Model 1 with heteroskedasticity-robust standard errors (HC1):")
model1_robust = sm.OLS(y, sm.add_constant(X1)).fit(cov_type='HC1')
print(model1_robust.summary().tables[1])

# Model 2: DiD with year fixed effects
print("\n5.3 Model 2: DiD with year fixed effects")
year_dummies = pd.get_dummies(df['YEAR'], prefix='Y', drop_first=True).astype(float)
X2 = pd.concat([df[['ELIGIBLE', 'ELIGIBLE_AFTER']].astype(float), year_dummies], axis=1)
model2 = sm.OLS(y, sm.add_constant(X2)).fit(cov_type='HC1')
print(f"DiD coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model2.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 3: DiD with covariates
print("\n5.4 Model 3: DiD with covariates")
# Create dummies for categorical variables
df['MALE'] = (df['SEX'] == 1).astype(float)
df['MARRIED'] = (df['MARST'] == 1).astype(float)  # Married, spouse present

# Education recodes
educ_dummies = pd.get_dummies(df['EDUC_RECODE'], prefix='EDUC', drop_first=True).astype(float)

covariates_df = pd.concat([
    df[['ELIGIBLE', 'ELIGIBLE_AFTER', 'MALE', 'MARRIED', 'AGE', 'NCHILD']].astype(float),
    year_dummies,
    educ_dummies
], axis=1)

model3 = sm.OLS(y, sm.add_constant(covariates_df)).fit(cov_type='HC1')
print(f"DiD coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model3.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 4: DiD with state fixed effects
print("\n5.5 Model 4: DiD with state and year fixed effects")
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True).astype(float)
covariates_state_df = pd.concat([
    covariates_df,
    state_dummies
], axis=1)

model4 = sm.OLS(y, sm.add_constant(covariates_state_df)).fit(cov_type='HC1')
print(f"DiD coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# =============================================================================
# 6. WEIGHTED REGRESSION MODELS
# =============================================================================
print("\n" + "="*70)
print("SECTION 6: WEIGHTED REGRESSION MODELS")
print("="*70)

# Model 5: Basic DiD with survey weights
print("\n6.1 Model 5: Weighted DiD (using PERWT)")
weights = df['PERWT'].astype(float)
model5 = sm.WLS(y, sm.add_constant(X1), weights=weights).fit(cov_type='HC1')
print(f"DiD coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model5.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 6: Weighted DiD with full covariates and state FEs
print("\n6.2 Model 6: Weighted DiD with covariates and state FEs")
model6 = sm.WLS(y, sm.add_constant(covariates_state_df), weights=weights).fit(cov_type='HC1')
print(f"DiD coefficient (ELIGIBLE_AFTER): {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model6.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# =============================================================================
# 7. PARALLEL TRENDS ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("SECTION 7: PARALLEL TRENDS ANALYSIS")
print("="*70)

# Calculate FT rates by year and eligibility
print("\n7.1 Full-time employment rates by year and eligibility status:")
trends = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
print(trends)

# Weighted trends
print("\n7.2 Weighted full-time employment rates by year and eligibility:")
weighted_trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT']), include_groups=False
).unstack()
print(weighted_trends)

# Event study specification
print("\n7.3 Event study regression (pre-treatment trends test):")
# Create year-specific treatment effects
df['Y2008'] = (df['YEAR'] == 2008).astype(float)
df['Y2009'] = (df['YEAR'] == 2009).astype(float)
df['Y2010'] = (df['YEAR'] == 2010).astype(float)
df['Y2011'] = (df['YEAR'] == 2011).astype(float)
df['Y2013'] = (df['YEAR'] == 2013).astype(float)
df['Y2014'] = (df['YEAR'] == 2014).astype(float)
df['Y2015'] = (df['YEAR'] == 2015).astype(float)
df['Y2016'] = (df['YEAR'] == 2016).astype(float)

# Interactions with ELIGIBLE (2011 is reference)
df['ELIG_2008'] = df['ELIGIBLE'] * df['Y2008']
df['ELIG_2009'] = df['ELIGIBLE'] * df['Y2009']
df['ELIG_2010'] = df['ELIGIBLE'] * df['Y2010']
df['ELIG_2013'] = df['ELIGIBLE'] * df['Y2013']
df['ELIG_2014'] = df['ELIGIBLE'] * df['Y2014']
df['ELIG_2015'] = df['ELIGIBLE'] * df['Y2015']
df['ELIG_2016'] = df['ELIGIBLE'] * df['Y2016']

event_vars = ['ELIGIBLE', 'Y2008', 'Y2009', 'Y2010', 'Y2013', 'Y2014', 'Y2015', 'Y2016',
              'ELIG_2008', 'ELIG_2009', 'ELIG_2010', 'ELIG_2013', 'ELIG_2014', 'ELIG_2015', 'ELIG_2016']
X_event = df[event_vars].astype(float)
model_event = sm.OLS(y, sm.add_constant(X_event)).fit(cov_type='HC1')

print("\nEvent study coefficients (ELIGIBLE x YEAR, reference: 2011):")
event_coefs = ['ELIG_2008', 'ELIG_2009', 'ELIG_2010', 'ELIG_2013', 'ELIG_2014', 'ELIG_2015', 'ELIG_2016']
for coef in event_coefs:
    print(f"{coef}: coef={model_event.params[coef]:.4f}, se={model_event.bse[coef]:.4f}, p={model_event.pvalues[coef]:.4f}")

# =============================================================================
# 8. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*70)
print("SECTION 8: ROBUSTNESS CHECKS")
print("="*70)

# 8.1 By gender
print("\n8.1 Heterogeneity by gender:")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex_val].copy()
    X_sex = df_sex[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']].astype(float)
    y_sex = df_sex['FT'].astype(float)
    model_sex = sm.OLS(y_sex, sm.add_constant(X_sex)).fit(cov_type='HC1')
    print(f"{sex_name}: DiD = {model_sex.params['ELIGIBLE_AFTER']:.4f} (SE={model_sex.bse['ELIGIBLE_AFTER']:.4f}, p={model_sex.pvalues['ELIGIBLE_AFTER']:.4f}), N={len(df_sex)}")

# 8.2 By marital status
print("\n8.2 Heterogeneity by marital status:")
for married_val, married_name in [(1, 'Married'), (0, 'Not Married')]:
    df_marr = df[df['MARRIED'] == married_val].copy()
    if len(df_marr) > 100:
        X_marr = df_marr[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']].astype(float)
        y_marr = df_marr['FT'].astype(float)
        model_marr = sm.OLS(y_marr, sm.add_constant(X_marr)).fit(cov_type='HC1')
        print(f"{married_name}: DiD = {model_marr.params['ELIGIBLE_AFTER']:.4f} (SE={model_marr.bse['ELIGIBLE_AFTER']:.4f}, p={model_marr.pvalues['ELIGIBLE_AFTER']:.4f}), N={len(df_marr)}")

# 8.3 Placebo test: pre-period only with fake treatment year
print("\n8.3 Placebo test (pre-period only, fake treatment at 2010):")
df_pre = df[df['AFTER'] == 0].copy()
df_pre['PLACEBO_AFTER'] = (df_pre['YEAR'] >= 2010).astype(float)
df_pre['ELIGIBLE_PLACEBO'] = df_pre['ELIGIBLE'] * df_pre['PLACEBO_AFTER']
X_placebo = df_pre[['ELIGIBLE', 'PLACEBO_AFTER', 'ELIGIBLE_PLACEBO']].astype(float)
y_placebo = df_pre['FT'].astype(float)
model_placebo = sm.OLS(y_placebo, sm.add_constant(X_placebo)).fit(cov_type='HC1')
print(f"Placebo DiD coefficient: {model_placebo.params['ELIGIBLE_PLACEBO']:.4f}")
print(f"Standard error: {model_placebo.bse['ELIGIBLE_PLACEBO']:.4f}")
print(f"p-value: {model_placebo.pvalues['ELIGIBLE_PLACEBO']:.4f}")

# 8.4 Narrower age bandwidth (closer to cutoff)
print("\n8.4 Narrower bandwidth (ages 28-33 in June 2012):")
df['AGE_IN_JUNE_2012'] = pd.to_numeric(df['AGE_IN_JUNE_2012'], errors='coerce')
df_narrow = df[(df['AGE_IN_JUNE_2012'] >= 28) & (df['AGE_IN_JUNE_2012'] <= 33)].copy()
X_narrow = df_narrow[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']].astype(float)
y_narrow = df_narrow['FT'].astype(float)
model_narrow = sm.OLS(y_narrow, sm.add_constant(X_narrow)).fit(cov_type='HC1')
print(f"Narrower bandwidth DiD: {model_narrow.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model_narrow.bse['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model_narrow.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample size: {len(df_narrow)}")

# =============================================================================
# 9. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "="*70)
print("SECTION 9: SUMMARY OF RESULTS")
print("="*70)

print("\nMain DiD estimates (ELIGIBLE_AFTER coefficient):")
print("-" * 70)
print(f"{'Model':<45} {'Coef':>8} {'SE':>8} {'p-value':>10}")
print("-" * 70)
print(f"{'1. Basic DiD':<45} {model1_robust.params['ELIGIBLE_AFTER']:>8.4f} {model1_robust.bse['ELIGIBLE_AFTER']:>8.4f} {model1_robust.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'2. Year FEs':<45} {model2.params['ELIGIBLE_AFTER']:>8.4f} {model2.bse['ELIGIBLE_AFTER']:>8.4f} {model2.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'3. Year FEs + covariates':<45} {model3.params['ELIGIBLE_AFTER']:>8.4f} {model3.bse['ELIGIBLE_AFTER']:>8.4f} {model3.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'4. Year + State FEs + covariates':<45} {model4.params['ELIGIBLE_AFTER']:>8.4f} {model4.bse['ELIGIBLE_AFTER']:>8.4f} {model4.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'5. Weighted basic DiD':<45} {model5.params['ELIGIBLE_AFTER']:>8.4f} {model5.bse['ELIGIBLE_AFTER']:>8.4f} {model5.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'6. Weighted with FEs + covariates':<45} {model6.params['ELIGIBLE_AFTER']:>8.4f} {model6.bse['ELIGIBLE_AFTER']:>8.4f} {model6.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print("-" * 70)

# Preferred specification
print("\n*** PREFERRED ESTIMATE (Model 4: Year + State FEs + covariates) ***")
print(f"Effect size: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"Sample size: {int(model4.nobs)}")
print(f"R-squared: {model4.rsquared:.4f}")

# =============================================================================
# 10. SAVE RESULTS FOR LATEX REPORT
# =============================================================================
print("\n" + "="*70)
print("SECTION 10: SAVING RESULTS")
print("="*70)

# Save key results to a file for the LaTeX report
results = {
    'did_simple': float(did_estimate),
    'did_weighted': float(did_weighted),
    'model1_coef': float(model1_robust.params['ELIGIBLE_AFTER']),
    'model1_se': float(model1_robust.bse['ELIGIBLE_AFTER']),
    'model1_pval': float(model1_robust.pvalues['ELIGIBLE_AFTER']),
    'model2_coef': float(model2.params['ELIGIBLE_AFTER']),
    'model2_se': float(model2.bse['ELIGIBLE_AFTER']),
    'model2_pval': float(model2.pvalues['ELIGIBLE_AFTER']),
    'model3_coef': float(model3.params['ELIGIBLE_AFTER']),
    'model3_se': float(model3.bse['ELIGIBLE_AFTER']),
    'model3_pval': float(model3.pvalues['ELIGIBLE_AFTER']),
    'model4_coef': float(model4.params['ELIGIBLE_AFTER']),
    'model4_se': float(model4.bse['ELIGIBLE_AFTER']),
    'model4_pval': float(model4.pvalues['ELIGIBLE_AFTER']),
    'model4_ci_low': float(model4.conf_int().loc['ELIGIBLE_AFTER', 0]),
    'model4_ci_high': float(model4.conf_int().loc['ELIGIBLE_AFTER', 1]),
    'model4_rsq': float(model4.rsquared),
    'model5_coef': float(model5.params['ELIGIBLE_AFTER']),
    'model5_se': float(model5.bse['ELIGIBLE_AFTER']),
    'model5_pval': float(model5.pvalues['ELIGIBLE_AFTER']),
    'model6_coef': float(model6.params['ELIGIBLE_AFTER']),
    'model6_se': float(model6.bse['ELIGIBLE_AFTER']),
    'model6_pval': float(model6.pvalues['ELIGIBLE_AFTER']),
    'n_total': int(len(df)),
    'n_treated': int((df['ELIGIBLE']==1).sum()),
    'n_control': int((df['ELIGIBLE']==0).sum()),
    'n_pre': int((df['AFTER']==0).sum()),
    'n_post': int((df['AFTER']==1).sum()),
    'ft_rate_overall': float(df['FT'].mean()),
    'treated_before': float(treated_before),
    'treated_after': float(treated_after),
    'control_before': float(control_before),
    'control_after': float(control_after),
    'placebo_coef': float(model_placebo.params['ELIGIBLE_PLACEBO']),
    'placebo_pval': float(model_placebo.pvalues['ELIGIBLE_PLACEBO']),
    'narrow_coef': float(model_narrow.params['ELIGIBLE_AFTER']),
    'narrow_se': float(model_narrow.bse['ELIGIBLE_AFTER']),
    'narrow_pval': float(model_narrow.pvalues['ELIGIBLE_AFTER']),
    'narrow_n': int(len(df_narrow)),
}

# Event study coefficients
for coef in event_coefs:
    results[f'event_{coef}'] = float(model_event.params[coef])
    results[f'event_{coef}_se'] = float(model_event.bse[coef])
    results[f'event_{coef}_pval'] = float(model_event.pvalues[coef])

# Save results as Python file for import
with open('analysis_results.py', 'w') as f:
    f.write("# Analysis results for LaTeX report\n")
    f.write("results = ")
    f.write(repr(results))

print("Results saved to analysis_results.py")

# Also save trends data for plotting
trends.to_csv('trends_data.csv')
weighted_trends.to_csv('weighted_trends_data.csv')
print("Trends data saved to trends_data.csv and weighted_trends_data.csv")

# Save full model summaries
with open('model_summaries.txt', 'w') as f:
    f.write("MODEL 1: Basic DiD\n")
    f.write("="*70 + "\n")
    f.write(str(model1_robust.summary()) + "\n\n")
    f.write("MODEL 4: Full specification (Year + State FEs + covariates)\n")
    f.write("="*70 + "\n")
    f.write(str(model4.summary()) + "\n\n")
    f.write("EVENT STUDY MODEL\n")
    f.write("="*70 + "\n")
    f.write(str(model_event.summary()) + "\n")

print("Model summaries saved to model_summaries.txt")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
