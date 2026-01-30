"""
DACA Replication Analysis - Replication 17
============================================
Analysis of the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the United States.

Research Design: Difference-in-Differences
- Treatment Group: Ages 26-30 at time of DACA implementation (June 2012)
- Control Group: Ages 31-35 at time of DACA implementation
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

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("=" * 80)
print("DACA REPLICATION ANALYSIS - REPLICATION 17")
print("=" * 80)
print()

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("1. LOADING DATA")
print("-" * 40)

data_path = r"C:\Users\seraf\DACA Results Task 3\replication_17\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"Total observations: {len(df):,}")
print(f"Number of variables: {df.shape[1]}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print()

# =============================================================================
# 2. DATA VERIFICATION
# =============================================================================
print("2. DATA VERIFICATION")
print("-" * 40)

# Check key variables
print("Key variable distributions:")
print(f"\nELIGIBLE (Treatment indicator):")
print(df['ELIGIBLE'].value_counts().sort_index())
print(f"  0 = Control group (ages 31-35 in June 2012)")
print(f"  1 = Treatment group (ages 26-30 in June 2012)")

print(f"\nAFTER (Post-treatment indicator):")
print(df['AFTER'].value_counts().sort_index())
print(f"  0 = Pre-DACA (2008-2011)")
print(f"  1 = Post-DACA (2013-2016)")

print(f"\nFT (Full-time employment):")
print(df['FT'].value_counts().sort_index())
print(f"  0 = Not full-time")
print(f"  1 = Full-time (35+ hours/week)")

print(f"\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())
print()

# =============================================================================
# 3. SUMMARY STATISTICS BY GROUP
# =============================================================================
print("3. SUMMARY STATISTICS BY GROUP")
print("-" * 40)

# Create 2x2 table of means
summary = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
print("\nUnweighted Summary:")
print(summary)

# Weighted means
def weighted_mean(group, value_col, weight_col):
    return np.average(group[value_col], weights=group[weight_col])

def weighted_std(group, value_col, weight_col):
    weights = group[weight_col]
    values = group[value_col]
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance)

print("\nWeighted Full-Time Employment Rates by Group:")
print("-" * 60)

groups = df.groupby(['ELIGIBLE', 'AFTER'])
for (eligible, after), group in groups:
    wt_mean = weighted_mean(group, 'FT', 'PERWT')
    wt_std = weighted_std(group, 'FT', 'PERWT')
    n = len(group)
    wt_n = group['PERWT'].sum()
    label_elig = "Treatment (26-30)" if eligible == 1 else "Control (31-35)"
    label_after = "Post-DACA" if after == 1 else "Pre-DACA"
    print(f"{label_elig}, {label_after}: {wt_mean:.4f} (SD={wt_std:.4f}), N={n:,}, Weighted N={wt_n:,.0f}")

print()

# =============================================================================
# 4. SIMPLE DIFFERENCE-IN-DIFFERENCES
# =============================================================================
print("4. SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("-" * 40)

# Calculate means for each cell
means = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()

print("\n2x2 Table of Weighted Mean Full-Time Employment:")
print("-" * 50)
print(f"                    Pre-DACA    Post-DACA    Diff")
print(f"Treatment (26-30):  {means.loc[1, 0]:.4f}      {means.loc[1, 1]:.4f}       {means.loc[1, 1] - means.loc[1, 0]:.4f}")
print(f"Control (31-35):    {means.loc[0, 0]:.4f}      {means.loc[0, 1]:.4f}       {means.loc[0, 1] - means.loc[0, 0]:.4f}")
print("-" * 50)

# DiD estimate
diff_treatment = means.loc[1, 1] - means.loc[1, 0]
diff_control = means.loc[0, 1] - means.loc[0, 0]
did_estimate = diff_treatment - diff_control

print(f"\nFirst Difference (Treatment): {diff_treatment:.4f}")
print(f"First Difference (Control):   {diff_control:.4f}")
print(f"\nDifference-in-Differences Estimate: {did_estimate:.4f}")
print(f"  Interpretation: DACA eligibility is associated with a")
print(f"  {did_estimate:.4f} ({did_estimate*100:.2f} percentage point) change")
print(f"  in full-time employment probability.")
print()

# =============================================================================
# 5. REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES
# =============================================================================
print("5. REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES")
print("-" * 40)

# Create interaction term
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls)
print("\nModel 1: Basic DiD (No Controls)")
print("-" * 40)
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model1.summary().tables[1])

print(f"\nDiD Coefficient (ELIGIBLE_X_AFTER): {model1.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model1.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# Model 2: DiD with demographic controls
print("\nModel 2: DiD with Demographic Controls")
print("-" * 40)

# Create dummy for sex (SEX: 1=Male, 2=Female in IPUMS)
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Education recodes - create dummies
# From data: 'Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+'
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Marital status (MARST: 1=Married spouse present, 2=Married spouse absent, 3=Separated, 4=Divorced, 5=Widowed, 6=Never married)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)

# Number of children
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

print(f"\nDiD Coefficient (ELIGIBLE_X_AFTER): {model2.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model2.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# Model 3: DiD with year fixed effects
print("\nModel 3: DiD with Year Fixed Effects")
print("-" * 40)

# Create year dummies
for year in df['YEAR'].unique():
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

year_vars = [f'YEAR_{y}' for y in sorted(df['YEAR'].unique()) if y != 2008]  # 2008 as reference

model3_formula = 'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA + ' + ' + '.join(year_vars)
model3 = smf.wls(model3_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Print key coefficients
print("Key coefficients:")
key_vars = ['Intercept', 'ELIGIBLE', 'ELIGIBLE_X_AFTER', 'FEMALE', 'MARRIED', 'HAS_CHILDREN']
for var in key_vars:
    if var in model3.params:
        print(f"  {var}: {model3.params[var]:.4f} (SE={model3.bse[var]:.4f}, p={model3.pvalues[var]:.4f})")

print(f"\nDiD Coefficient (ELIGIBLE_X_AFTER): {model3.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model3.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# Model 4: DiD with state fixed effects
print("\nModel 4: DiD with State Fixed Effects")
print("-" * 40)

# Create state dummies
states = df['STATEFIP'].unique()
ref_state = sorted(states)[0]  # Use first state as reference
for state in states:
    df[f'STATE_{state}'] = (df['STATEFIP'] == state).astype(int)

state_vars = [f'STATE_{s}' for s in sorted(states) if s != ref_state]

model4_formula = 'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(state_vars)
model4 = smf.wls(model4_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("Key coefficients:")
for var in key_vars:
    if var in model4.params:
        print(f"  {var}: {model4.params[var]:.4f} (SE={model4.bse[var]:.4f}, p={model4.pvalues[var]:.4f})")

print(f"\nDiD Coefficient (ELIGIBLE_X_AFTER): {model4.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model4.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# =============================================================================
# 6. PARALLEL TRENDS CHECK
# =============================================================================
print("6. PARALLEL TRENDS CHECK")
print("-" * 40)

# Calculate year-by-year means for each group
yearly_means = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
yearly_means.columns = ['Control (31-35)', 'Treatment (26-30)']

print("\nWeighted Full-Time Employment Rate by Year and Group:")
print(yearly_means.round(4))
print()

# Pre-treatment trend test
df_pre = df[df['AFTER'] == 0].copy()
df_pre['ELIGIBLE_X_YEAR'] = df_pre['ELIGIBLE'] * df_pre['YEAR']

trend_model = smf.wls('FT ~ ELIGIBLE + YEAR + ELIGIBLE_X_YEAR',
                       data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')

print("Pre-Treatment Trend Test (2008-2011):")
print(f"  ELIGIBLE x YEAR coefficient: {trend_model.params['ELIGIBLE_X_YEAR']:.6f}")
print(f"  Standard Error: {trend_model.bse['ELIGIBLE_X_YEAR']:.6f}")
print(f"  P-value: {trend_model.pvalues['ELIGIBLE_X_YEAR']:.4f}")
print(f"  Interpretation: {'Parallel trends assumption appears satisfied (p > 0.05)' if trend_model.pvalues['ELIGIBLE_X_YEAR'] > 0.05 else 'Evidence of differential pre-trends (p < 0.05)'}")
print()

# =============================================================================
# 7. EVENT STUDY ANALYSIS
# =============================================================================
print("7. EVENT STUDY ANALYSIS")
print("-" * 40)

# Create year-specific treatment effects (relative to 2011)
years = sorted(df['YEAR'].unique())
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)

# Create interaction terms for each year except 2011 (reference year)
for year in years:
    if year != 2011:
        df[f'ELIGIBLE_X_YEAR_{year}'] = df['ELIGIBLE'] * (df['YEAR'] == year).astype(int)

event_vars = [f'ELIGIBLE_X_YEAR_{y}' for y in years if y != 2011]
event_formula = 'FT ~ ELIGIBLE + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(event_vars)
event_model = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("Event Study Coefficients (relative to 2011):")
print("-" * 50)
for year in sorted(years):
    if year != 2011:
        var = f'ELIGIBLE_X_YEAR_{year}'
        coef = event_model.params[var]
        se = event_model.bse[var]
        ci_low, ci_high = event_model.conf_int().loc[var]
        pval = event_model.pvalues[var]
        period = "Pre-DACA" if year < 2012 else "Post-DACA"
        print(f"  {year} ({period}): {coef:.4f} (SE={se:.4f}), 95% CI=[{ci_low:.4f}, {ci_high:.4f}], p={pval:.4f}")
    else:
        print(f"  {year} (Reference): 0.0000")
print()

# =============================================================================
# 8. ROBUSTNESS CHECKS
# =============================================================================
print("8. ROBUSTNESS CHECKS")
print("-" * 40)

# 8.1 Unweighted regression
print("\n8.1 Unweighted DiD Regression:")
model_unweighted = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                            data=df).fit(cov_type='HC1')
print(f"  DiD Coefficient: {model_unweighted.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"  Standard Error: {model_unweighted.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"  P-value: {model_unweighted.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# 8.2 Linear probability model vs logit comparison
print("\n8.2 Logistic Regression DiD:")
try:
    # Use sampling for faster computation
    df_sample = df.sample(n=min(50000, len(df)), random_state=42) if len(df) > 50000 else df
    logit_model = smf.logit('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN',
                             data=df_sample).fit(disp=False)
    # Calculate marginal effect at means
    print(f"  DiD Log-Odds Coefficient: {logit_model.params['ELIGIBLE_X_AFTER']:.4f}")
    print(f"  Standard Error: {logit_model.bse['ELIGIBLE_X_AFTER']:.4f}")
    print(f"  P-value: {logit_model.pvalues['ELIGIBLE_X_AFTER']:.4f}")
except Exception as e:
    print(f"  Logit model could not be estimated: {e}")

# 8.3 By gender subgroup
print("\n8.3 Subgroup Analysis by Gender:")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex]
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                         data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"  {label}: DiD = {model_sub.params['ELIGIBLE_X_AFTER']:.4f} (SE={model_sub.bse['ELIGIBLE_X_AFTER']:.4f}, p={model_sub.pvalues['ELIGIBLE_X_AFTER']:.4f})")

# 8.4 Clustered standard errors by state
print("\n8.4 DiD with State-Clustered Standard Errors:")
model_clustered = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                           data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD Coefficient: {model_clustered.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"  Clustered SE: {model_clustered.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"  P-value: {model_clustered.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"  95% CI: [{model_clustered.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model_clustered.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")

print()

# =============================================================================
# 9. SUMMARY OF RESULTS
# =============================================================================
print("=" * 80)
print("9. SUMMARY OF RESULTS")
print("=" * 80)

print("\n--- PREFERRED ESTIMATE ---")
print(f"Model: Weighted Linear Probability DiD with Demographic Controls")
print(f"DiD Estimate: {model2.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model2.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model2.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"Sample Size: {len(df):,}")
print(f"Weighted Sample Size: {df['PERWT'].sum():,.0f}")

print("\n--- INTERPRETATION ---")
effect = model2.params['ELIGIBLE_X_AFTER']
if effect > 0:
    print(f"DACA eligibility is associated with a {effect:.4f} ({effect*100:.2f} percentage point)")
    print("INCREASE in the probability of full-time employment.")
else:
    print(f"DACA eligibility is associated with a {abs(effect):.4f} ({abs(effect)*100:.2f} percentage point)")
    print("DECREASE in the probability of full-time employment.")

if model2.pvalues['ELIGIBLE_X_AFTER'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
elif model2.pvalues['ELIGIBLE_X_AFTER'] < 0.10:
    print("This effect is statistically significant at the 10% level but not the 5% level.")
else:
    print("This effect is NOT statistically significant at conventional levels.")

print("\n--- MODEL COMPARISON ---")
print(f"{'Model':<50} {'DiD Estimate':>12} {'SE':>10} {'P-value':>10}")
print("-" * 82)
print(f"{'1. Basic DiD (no controls)':<50} {model1.params['ELIGIBLE_X_AFTER']:>12.4f} {model1.bse['ELIGIBLE_X_AFTER']:>10.4f} {model1.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'2. DiD + Demographics (PREFERRED)':<50} {model2.params['ELIGIBLE_X_AFTER']:>12.4f} {model2.bse['ELIGIBLE_X_AFTER']:>10.4f} {model2.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'3. DiD + Year FE':<50} {model3.params['ELIGIBLE_X_AFTER']:>12.4f} {model3.bse['ELIGIBLE_X_AFTER']:>10.4f} {model3.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'4. DiD + Year FE + State FE':<50} {model4.params['ELIGIBLE_X_AFTER']:>12.4f} {model4.bse['ELIGIBLE_X_AFTER']:>10.4f} {model4.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'5. Unweighted DiD + Demographics':<50} {model_unweighted.params['ELIGIBLE_X_AFTER']:>12.4f} {model_unweighted.bse['ELIGIBLE_X_AFTER']:>10.4f} {model_unweighted.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'6. DiD + Demographics (Clustered SE)':<50} {model_clustered.params['ELIGIBLE_X_AFTER']:>12.4f} {model_clustered.bse['ELIGIBLE_X_AFTER']:>10.4f} {model_clustered.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# =============================================================================
# 10. SAVE RESULTS FOR LATEX
# =============================================================================

# Save key results to file for LaTeX
results_dict = {
    'simple_did': did_estimate,
    'model1_coef': model1.params['ELIGIBLE_X_AFTER'],
    'model1_se': model1.bse['ELIGIBLE_X_AFTER'],
    'model1_pval': model1.pvalues['ELIGIBLE_X_AFTER'],
    'model1_ci_low': model1.conf_int().loc['ELIGIBLE_X_AFTER', 0],
    'model1_ci_high': model1.conf_int().loc['ELIGIBLE_X_AFTER', 1],
    'model2_coef': model2.params['ELIGIBLE_X_AFTER'],
    'model2_se': model2.bse['ELIGIBLE_X_AFTER'],
    'model2_pval': model2.pvalues['ELIGIBLE_X_AFTER'],
    'model2_ci_low': model2.conf_int().loc['ELIGIBLE_X_AFTER', 0],
    'model2_ci_high': model2.conf_int().loc['ELIGIBLE_X_AFTER', 1],
    'model3_coef': model3.params['ELIGIBLE_X_AFTER'],
    'model3_se': model3.bse['ELIGIBLE_X_AFTER'],
    'model3_pval': model3.pvalues['ELIGIBLE_X_AFTER'],
    'model4_coef': model4.params['ELIGIBLE_X_AFTER'],
    'model4_se': model4.bse['ELIGIBLE_X_AFTER'],
    'model4_pval': model4.pvalues['ELIGIBLE_X_AFTER'],
    'model_clustered_coef': model_clustered.params['ELIGIBLE_X_AFTER'],
    'model_clustered_se': model_clustered.bse['ELIGIBLE_X_AFTER'],
    'model_clustered_pval': model_clustered.pvalues['ELIGIBLE_X_AFTER'],
    'n_obs': len(df),
    'n_weighted': df['PERWT'].sum(),
    'n_treatment': len(df[df['ELIGIBLE'] == 1]),
    'n_control': len(df[df['ELIGIBLE'] == 0]),
    'mean_ft_treatment_pre': means.loc[1, 0],
    'mean_ft_treatment_post': means.loc[1, 1],
    'mean_ft_control_pre': means.loc[0, 0],
    'mean_ft_control_post': means.loc[0, 1],
    'trend_test_coef': trend_model.params['ELIGIBLE_X_YEAR'],
    'trend_test_pval': trend_model.pvalues['ELIGIBLE_X_YEAR'],
}

# Save to CSV for reference
results_df = pd.DataFrame([results_dict])
results_df.to_csv(r"C:\Users\seraf\DACA Results Task 3\replication_17\analysis_results.csv", index=False)
print("\nResults saved to analysis_results.csv")

# Save yearly means
yearly_means.to_csv(r"C:\Users\seraf\DACA Results Task 3\replication_17\yearly_means.csv")
print("Yearly means saved to yearly_means.csv")

# Save event study coefficients
event_coefs = []
for year in sorted(years):
    if year != 2011:
        var = f'ELIGIBLE_X_YEAR_{year}'
        event_coefs.append({
            'year': year,
            'coefficient': event_model.params[var],
            'se': event_model.bse[var],
            'ci_low': event_model.conf_int().loc[var, 0],
            'ci_high': event_model.conf_int().loc[var, 1],
            'pval': event_model.pvalues[var]
        })
    else:
        event_coefs.append({
            'year': year,
            'coefficient': 0,
            'se': 0,
            'ci_low': 0,
            'ci_high': 0,
            'pval': 1
        })

event_df = pd.DataFrame(event_coefs)
event_df.to_csv(r"C:\Users\seraf\DACA Results Task 3\replication_17\event_study.csv", index=False)
print("Event study coefficients saved to event_study.csv")

# Create summary statistics table
summary_stats = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['mean', 'std'],
    'FEMALE': 'mean',
    'MARRIED': 'mean',
    'HAS_CHILDREN': 'mean',
    'AGE': 'mean',
    'PERWT': ['count', 'sum']
}).round(4)
summary_stats.to_csv(r"C:\Users\seraf\DACA Results Task 3\replication_17\summary_stats.csv")
print("Summary statistics saved to summary_stats.csv")

# Full model summaries
with open(r"C:\Users\seraf\DACA Results Task 3\replication_17\model_summaries.txt", 'w') as f:
    f.write("MODEL 1: Basic DiD\n")
    f.write("=" * 80 + "\n")
    f.write(model1.summary().as_text())
    f.write("\n\n")
    f.write("MODEL 2: DiD with Demographics (PREFERRED)\n")
    f.write("=" * 80 + "\n")
    f.write(model2.summary().as_text())
    f.write("\n\n")
    f.write("MODEL 3: DiD with Year Fixed Effects\n")
    f.write("=" * 80 + "\n")
    f.write(model3.summary().as_text())
    f.write("\n\n")
    f.write("MODEL 4: DiD with Year and State Fixed Effects\n")
    f.write("=" * 80 + "\n")
    f.write(model4.summary().as_text())
    f.write("\n\n")
    f.write("EVENT STUDY MODEL\n")
    f.write("=" * 80 + "\n")
    f.write(event_model.summary().as_text())
print("Model summaries saved to model_summaries.txt")
