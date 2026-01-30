"""
DACA Replication Study - Difference-in-Differences Analysis
Estimating the effect of DACA eligibility on full-time employment

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the US,
what was the causal impact of DACA eligibility on full-time employment probability?

Treatment: ELIGIBLE=1 (Ages 26-30 in June 2012)
Control: ELIGIBLE=0 (Ages 31-35 in June 2012)
Pre-period: 2008-2011 (AFTER=0)
Post-period: 2013-2016 (AFTER=1)
Outcome: FT (Full-time employment, 35+ hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 70)
print("DACA REPLICATION STUDY - DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 70)

df = pd.read_csv('data/prepared_data_labelled_version.csv', low_memory=False)

print(f"\nTotal observations: {len(df):,}")
print(f"Years covered: {df['YEAR'].min()}-{df['YEAR'].max()} (excluding 2012)")

# ============================================================================
# SECTION 1: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("=" * 70)

# Group structure
print("\n--- Table 1: Sample Distribution by Treatment and Period ---")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'],
                        rownames=['ELIGIBLE (Treatment)'],
                        colnames=['AFTER (Post-Period)'],
                        margins=True)
crosstab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
crosstab.columns = ['Pre (2008-11)', 'Post (2013-16)', 'Total']
print(crosstab)

# Summary statistics by group
print("\n--- Table 2: Summary Statistics by Treatment Group ---")
summary_stats = df.groupby('ELIGIBLE').agg({
    'FT': ['mean', 'std', 'count'],
    'AGE': ['mean', 'std'],
    'PERWT': 'mean'
}).round(3)
print(summary_stats)

# Full-time employment rates by group and period
print("\n--- Table 3: Full-Time Employment Rates by Group and Period ---")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count']).round(4)
ft_rates.columns = ['FT Rate', 'N']
print(ft_rates)

# Compute weighted means
print("\n--- Table 4: Weighted Full-Time Employment Rates ---")
def weighted_mean(group, value_col, weight_col):
    return np.average(group[value_col], weights=group[weight_col])

weighted_ft = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT')
).unstack()
weighted_ft.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_ft.columns = ['Pre (2008-11)', 'Post (2013-16)']
print(weighted_ft.round(4))

# ============================================================================
# SECTION 2: BASIC DIFFERENCE-IN-DIFFERENCES (UNWEIGHTED)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: BASIC DIFFERENCE-IN-DIFFERENCES")
print("=" * 70)

# Calculate the 2x2 DiD manually
ft_00 = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()  # Control, Pre
ft_01 = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()  # Control, Post
ft_10 = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()  # Treatment, Pre
ft_11 = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()  # Treatment, Post

diff_control = ft_01 - ft_00
diff_treatment = ft_11 - ft_10
did_effect = diff_treatment - diff_control

print("\n--- Simple 2x2 DiD Calculation (Unweighted) ---")
print(f"Control Pre:    {ft_00:.4f}")
print(f"Control Post:   {ft_01:.4f}")
print(f"Treatment Pre:  {ft_10:.4f}")
print(f"Treatment Post: {ft_11:.4f}")
print(f"\nDifference (Control):   {diff_control:.4f}")
print(f"Difference (Treatment): {diff_treatment:.4f}")
print(f"\nDiD Effect: {did_effect:.4f} ({did_effect*100:.2f} percentage points)")

# ============================================================================
# SECTION 3: REGRESSION-BASED DiD
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES")
print("=" * 70)

# Model 1: Basic DiD (OLS without weights)
print("\n--- Model 1: Basic DiD (OLS, Unweighted) ---")
df['ELIGIBLE_x_AFTER'] = df['ELIGIBLE'] * df['AFTER']
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_x_AFTER']])
model1 = sm.OLS(df['FT'], X).fit(cov_type='HC1')
print(model1.summary())

# Model 2: Basic DiD (WLS with survey weights)
print("\n--- Model 2: Basic DiD (WLS, Weighted) ---")
model2 = sm.WLS(df['FT'], X, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# ============================================================================
# SECTION 4: DiD WITH COVARIATES
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 4: DiD WITH COVARIATES")
print("=" * 70)

# Create dummy variables for covariates
# SEX: Male=1, Female=0
df['MALE'] = (df['SEX'] == 'Male').astype(int)

# MARST dummies (reference: Never married/single)
df['MARRIED'] = df['MARST'].isin(['Married, spouse present', 'Married, spouse absent']).astype(int)

# Education dummies (reference: High School Degree)
df['LESS_THAN_HS'] = (df['EDUC_RECODE'] == 'Less than High School').astype(int)
df['SOME_COLLEGE'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['TWO_YEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['BA_PLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Region dummies (reference: West)
df['SOUTH'] = (df['CensusRegion'] == 'South').astype(int)
df['MIDWEST'] = (df['CensusRegion'] == 'Midwest').astype(int)
df['NORTHEAST'] = (df['CensusRegion'] == 'Northeast').astype(int)

# Age (continuous)
df['AGE_CENTERED'] = df['AGE'] - df['AGE'].mean()

# Model 3: DiD with demographic covariates (unweighted)
print("\n--- Model 3: DiD with Demographics (OLS, Unweighted) ---")
X3 = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_x_AFTER',
                          'MALE', 'MARRIED', 'SOME_COLLEGE', 'TWO_YEAR', 'BA_PLUS',
                          'SOUTH', 'MIDWEST', 'NORTHEAST', 'AGE_CENTERED']])
model3 = sm.OLS(df['FT'], X3).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with demographics (weighted)
print("\n--- Model 4: DiD with Demographics (WLS, Weighted) ---")
model4 = sm.WLS(df['FT'], X3, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# ============================================================================
# SECTION 5: YEAR-SPECIFIC EFFECTS (EVENT STUDY)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 5: YEAR-SPECIFIC EFFECTS (EVENT STUDY)")
print("=" * 70)

# Create year dummies (reference: 2011)
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

# Create interaction terms
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_x_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Model 5: Event study specification (unweighted)
print("\n--- Model 5: Event Study Specification (OLS, Unweighted) ---")
event_vars = ['ELIGIBLE', 'YEAR_2008', 'YEAR_2009', 'YEAR_2010',
              'YEAR_2013', 'YEAR_2014', 'YEAR_2015', 'YEAR_2016',
              'ELIGIBLE_x_YEAR_2008', 'ELIGIBLE_x_YEAR_2009', 'ELIGIBLE_x_YEAR_2010',
              'ELIGIBLE_x_YEAR_2013', 'ELIGIBLE_x_YEAR_2014', 'ELIGIBLE_x_YEAR_2015', 'ELIGIBLE_x_YEAR_2016']
X5 = sm.add_constant(df[event_vars])
model5 = sm.OLS(df['FT'], X5).fit(cov_type='HC1')
print(model5.summary())

# Extract event study coefficients
print("\n--- Event Study Coefficients (Relative to 2011) ---")
event_coefs = {}
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model5.params[f'ELIGIBLE_x_YEAR_{year}']
    se = model5.bse[f'ELIGIBLE_x_YEAR_{year}']
    event_coefs[year] = (coef, se)
    print(f"Year {year}: {coef:.4f} (SE: {se:.4f})")

# ============================================================================
# SECTION 6: ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 6: ROBUSTNESS CHECKS")
print("=" * 70)

# Model 6: State fixed effects (using state dummies)
print("\n--- Model 6: With State Fixed Effects (OLS) ---")
# Create state dummies - use statename column
state_dummies = pd.get_dummies(df['statename'], prefix='STATE', drop_first=True)
# Convert to numeric
state_dummies = state_dummies.astype(float)
X6 = pd.concat([X3.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
model6 = sm.OLS(df['FT'].reset_index(drop=True), X6).fit(cov_type='HC1')
print(f"\nDiD Coefficient (ELIGIBLE_x_AFTER): {model6.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Robust SE: {model6.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"t-stat: {model6.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"p-value: {model6.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")

# Model 7: With year fixed effects
print("\n--- Model 7: With Year Fixed Effects (OLS) ---")
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True)
year_dummies = year_dummies.astype(float)
X7_base = sm.add_constant(df[['ELIGIBLE', 'ELIGIBLE_x_AFTER',
                              'MALE', 'MARRIED', 'SOME_COLLEGE', 'TWO_YEAR', 'BA_PLUS',
                              'SOUTH', 'MIDWEST', 'NORTHEAST', 'AGE_CENTERED']])
X7 = pd.concat([X7_base.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1)
model7 = sm.OLS(df['FT'].reset_index(drop=True), X7).fit(cov_type='HC1')
print(f"\nDiD Coefficient (ELIGIBLE_x_AFTER): {model7.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Robust SE: {model7.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"t-stat: {model7.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"p-value: {model7.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")

# ============================================================================
# SECTION 7: HETEROGENEITY ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 7: HETEROGENEITY ANALYSIS")
print("=" * 70)

# By Sex
print("\n--- DiD by Sex ---")
for sex in ['Male', 'Female']:
    df_sex = df[df['SEX'] == sex]
    X_sex = sm.add_constant(df_sex[['ELIGIBLE', 'AFTER', 'ELIGIBLE_x_AFTER']])
    model_sex = sm.OLS(df_sex['FT'], X_sex).fit(cov_type='HC1')
    print(f"\n{sex}:")
    print(f"  N = {len(df_sex):,}")
    print(f"  DiD Effect: {model_sex.params['ELIGIBLE_x_AFTER']:.4f}")
    print(f"  SE: {model_sex.bse['ELIGIBLE_x_AFTER']:.4f}")
    print(f"  p-value: {model_sex.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# By Education
print("\n--- DiD by Education ---")
for educ in ['High School Degree', 'Some College', 'BA+']:
    df_educ = df[df['EDUC_RECODE'] == educ]
    if len(df_educ) > 100:
        X_educ = sm.add_constant(df_educ[['ELIGIBLE', 'AFTER', 'ELIGIBLE_x_AFTER']])
        model_educ = sm.OLS(df_educ['FT'], X_educ).fit(cov_type='HC1')
        print(f"\n{educ}:")
        print(f"  N = {len(df_educ):,}")
        print(f"  DiD Effect: {model_educ.params['ELIGIBLE_x_AFTER']:.4f}")
        print(f"  SE: {model_educ.bse['ELIGIBLE_x_AFTER']:.4f}")
        print(f"  p-value: {model_educ.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# By Region
print("\n--- DiD by Region ---")
for region in df['CensusRegion'].unique():
    df_region = df[df['CensusRegion'] == region]
    if len(df_region) > 100:
        X_region = sm.add_constant(df_region[['ELIGIBLE', 'AFTER', 'ELIGIBLE_x_AFTER']])
        model_region = sm.OLS(df_region['FT'], X_region).fit(cov_type='HC1')
        print(f"\n{region}:")
        print(f"  N = {len(df_region):,}")
        print(f"  DiD Effect: {model_region.params['ELIGIBLE_x_AFTER']:.4f}")
        print(f"  SE: {model_region.bse['ELIGIBLE_x_AFTER']:.4f}")
        print(f"  p-value: {model_region.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# ============================================================================
# SECTION 8: BALANCE TESTS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 8: BALANCE TESTS (Pre-Treatment Period)")
print("=" * 70)

df_pre = df[df['AFTER'] == 0]

print("\n--- Covariate Balance in Pre-Period ---")
covariates = ['MALE', 'MARRIED', 'AGE', 'SOME_COLLEGE', 'TWO_YEAR', 'BA_PLUS']
for cov in covariates:
    treat_mean = df_pre[df_pre['ELIGIBLE']==1][cov].mean()
    ctrl_mean = df_pre[df_pre['ELIGIBLE']==0][cov].mean()
    diff = treat_mean - ctrl_mean
    # T-test
    treat_vals = df_pre[df_pre['ELIGIBLE']==1][cov]
    ctrl_vals = df_pre[df_pre['ELIGIBLE']==0][cov]
    t_stat, p_val = stats.ttest_ind(treat_vals, ctrl_vals)
    print(f"{cov:15s}: Treatment={treat_mean:.4f}, Control={ctrl_mean:.4f}, Diff={diff:.4f}, p={p_val:.4f}")

# ============================================================================
# SECTION 9: PARALLEL TRENDS TEST
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 9: PARALLEL TRENDS TEST")
print("=" * 70)

# Test: Are pre-treatment trends similar between groups?
df_pre = df[df['AFTER'] == 0]

# Regress FT on ELIGIBLE, YEAR, ELIGIBLE*YEAR in pre-period
df_pre['YEAR_TREND'] = df_pre['YEAR'] - 2008
df_pre['ELIGIBLE_x_TREND'] = df_pre['ELIGIBLE'] * df_pre['YEAR_TREND']

X_trend = sm.add_constant(df_pre[['ELIGIBLE', 'YEAR_TREND', 'ELIGIBLE_x_TREND']])
model_trend = sm.OLS(df_pre['FT'], X_trend).fit(cov_type='HC1')
print("\n--- Pre-Treatment Trend Test ---")
print(f"ELIGIBLE x YEAR_TREND coefficient: {model_trend.params['ELIGIBLE_x_TREND']:.4f}")
print(f"Robust SE: {model_trend.bse['ELIGIBLE_x_TREND']:.4f}")
print(f"t-stat: {model_trend.tvalues['ELIGIBLE_x_TREND']:.4f}")
print(f"p-value: {model_trend.pvalues['ELIGIBLE_x_TREND']:.4f}")
print("\nInterpretation: A non-significant coefficient suggests parallel pre-trends.")

# ============================================================================
# SECTION 10: SUMMARY OF RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 10: SUMMARY OF RESULTS")
print("=" * 70)

print("\n" + "-" * 70)
print("TABLE: Summary of DiD Estimates")
print("-" * 70)
print(f"{'Model':<45} {'Estimate':>10} {'SE':>10} {'p-value':>10}")
print("-" * 70)
print(f"{'1. Basic DiD (OLS, unweighted)':<45} {model1.params['ELIGIBLE_x_AFTER']:>10.4f} {model1.bse['ELIGIBLE_x_AFTER']:>10.4f} {model1.pvalues['ELIGIBLE_x_AFTER']:>10.4f}")
print(f"{'2. Basic DiD (WLS, weighted)':<45} {model2.params['ELIGIBLE_x_AFTER']:>10.4f} {model2.bse['ELIGIBLE_x_AFTER']:>10.4f} {model2.pvalues['ELIGIBLE_x_AFTER']:>10.4f}")
print(f"{'3. DiD + Demographics (OLS)':<45} {model3.params['ELIGIBLE_x_AFTER']:>10.4f} {model3.bse['ELIGIBLE_x_AFTER']:>10.4f} {model3.pvalues['ELIGIBLE_x_AFTER']:>10.4f}")
print(f"{'4. DiD + Demographics (WLS)':<45} {model4.params['ELIGIBLE_x_AFTER']:>10.4f} {model4.bse['ELIGIBLE_x_AFTER']:>10.4f} {model4.pvalues['ELIGIBLE_x_AFTER']:>10.4f}")
print(f"{'6. DiD + Demographics + State FE':<45} {model6.params['ELIGIBLE_x_AFTER']:>10.4f} {model6.bse['ELIGIBLE_x_AFTER']:>10.4f} {model6.pvalues['ELIGIBLE_x_AFTER']:>10.4f}")
print(f"{'7. DiD + Demographics + Year FE':<45} {model7.params['ELIGIBLE_x_AFTER']:>10.4f} {model7.bse['ELIGIBLE_x_AFTER']:>10.4f} {model7.pvalues['ELIGIBLE_x_AFTER']:>10.4f}")
print("-" * 70)

# Preferred estimate
print("\n" + "=" * 70)
print("PREFERRED ESTIMATE")
print("=" * 70)
print("\nBased on the analysis, the preferred estimate is Model 4")
print("(DiD with demographic covariates, using survey weights)")
print(f"\nEffect Size: {model4.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"Sample Size: {len(df):,}")
print(f"t-statistic: {model4.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"p-value: {model4.pvalues['ELIGIBLE_x_AFTER']:.4f}")

print("\nInterpretation:")
print(f"DACA eligibility increased the probability of full-time employment")
print(f"by approximately {model4.params['ELIGIBLE_x_AFTER']*100:.1f} percentage points")
print(f"among the eligible population (ages 26-30 in June 2012).")

# Save key results for LaTeX
results_dict = {
    'n_total': len(df),
    'n_treatment': len(df[df['ELIGIBLE']==1]),
    'n_control': len(df[df['ELIGIBLE']==0]),
    'n_pre': len(df[df['AFTER']==0]),
    'n_post': len(df[df['AFTER']==1]),
    'ft_00': ft_00,
    'ft_01': ft_01,
    'ft_10': ft_10,
    'ft_11': ft_11,
    'did_simple': did_effect,
    'model1_coef': model1.params['ELIGIBLE_x_AFTER'],
    'model1_se': model1.bse['ELIGIBLE_x_AFTER'],
    'model1_pval': model1.pvalues['ELIGIBLE_x_AFTER'],
    'model2_coef': model2.params['ELIGIBLE_x_AFTER'],
    'model2_se': model2.bse['ELIGIBLE_x_AFTER'],
    'model2_pval': model2.pvalues['ELIGIBLE_x_AFTER'],
    'model3_coef': model3.params['ELIGIBLE_x_AFTER'],
    'model3_se': model3.bse['ELIGIBLE_x_AFTER'],
    'model3_pval': model3.pvalues['ELIGIBLE_x_AFTER'],
    'model4_coef': model4.params['ELIGIBLE_x_AFTER'],
    'model4_se': model4.bse['ELIGIBLE_x_AFTER'],
    'model4_pval': model4.pvalues['ELIGIBLE_x_AFTER'],
    'model4_ci_low': model4.conf_int().loc['ELIGIBLE_x_AFTER', 0],
    'model4_ci_high': model4.conf_int().loc['ELIGIBLE_x_AFTER', 1],
    'model6_coef': model6.params['ELIGIBLE_x_AFTER'],
    'model6_se': model6.bse['ELIGIBLE_x_AFTER'],
    'model6_pval': model6.pvalues['ELIGIBLE_x_AFTER'],
    'model7_coef': model7.params['ELIGIBLE_x_AFTER'],
    'model7_se': model7.bse['ELIGIBLE_x_AFTER'],
    'model7_pval': model7.pvalues['ELIGIBLE_x_AFTER'],
    'pretrend_coef': model_trend.params['ELIGIBLE_x_TREND'],
    'pretrend_pval': model_trend.pvalues['ELIGIBLE_x_TREND'],
}

# Save to file
import json
with open('results_summary.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n\nResults saved to results_summary.json")
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
