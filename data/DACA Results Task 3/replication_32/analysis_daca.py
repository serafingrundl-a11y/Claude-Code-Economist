"""
DACA Replication Study - Analysis Script
=========================================
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US.

Identification: Difference-in-Differences
- Treatment: Ages 26-30 as of June 15, 2012 (ELIGIBLE=1)
- Control: Ages 31-35 as of June 15, 2012 (ELIGIBLE=0)
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
print("DACA REPLICATION STUDY - DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n1. LOADING DATA...")
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"   Total observations: {len(df):,}")
print(f"   Variables: {df.shape[1]}")

# =============================================================================
# 2. DATA EXPLORATION
# =============================================================================
print("\n2. DATA EXPLORATION")
print("-" * 40)

# Check key variables
print("\n2.1 Distribution of Key Variables:")
print(f"\n   ELIGIBLE (Treatment indicator):")
print(df['ELIGIBLE'].value_counts().sort_index())

print(f"\n   AFTER (Post-treatment period):")
print(df['AFTER'].value_counts().sort_index())

print(f"\n   FT (Full-time employment):")
print(df['FT'].value_counts().sort_index())

print(f"\n   YEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

# Sample sizes by group
print("\n2.2 Sample Sizes by Group:")
group_counts = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
group_counts.index = ['Control (31-35)', 'Treatment (26-30)']
group_counts.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print(group_counts)

# Weighted sample sizes
print("\n2.3 Weighted Sample Sizes:")
weighted_counts = df.groupby(['ELIGIBLE', 'AFTER'])['PERWT'].sum().unstack()
weighted_counts.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_counts.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print(weighted_counts.round(0))

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 40)

# FT rates by group and period
print("\n3.1 Full-Time Employment Rates by Group and Period:")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print((ft_rates * 100).round(2))
print("(Weighted percentages)")

# Calculate simple DiD
diff_control = ft_rates.loc['Control (31-35)', 'Post (2013-2016)'] - ft_rates.loc['Control (31-35)', 'Pre (2008-2011)']
diff_treatment = ft_rates.loc['Treatment (26-30)', 'Post (2013-2016)'] - ft_rates.loc['Treatment (26-30)', 'Pre (2008-2011)']
simple_did = diff_treatment - diff_control

print(f"\n3.2 Simple Difference-in-Differences Calculation:")
print(f"   Control group change:   {diff_control*100:.2f} pp")
print(f"   Treatment group change: {diff_treatment*100:.2f} pp")
print(f"   DiD estimate:           {simple_did*100:.2f} pp")

# FT rates by year
print("\n3.3 Full-Time Employment Rates by Year:")
ft_by_year = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
ft_by_year.columns = ['Control (31-35)', 'Treatment (26-30)']
print((ft_by_year * 100).round(2))

# Demographic characteristics
print("\n3.4 Demographic Characteristics (Pre-Period):")
pre_df = df[df['AFTER'] == 0]

# Age
print("\n   Mean Age:")
for elig in [0, 1]:
    group = pre_df[pre_df['ELIGIBLE'] == elig]
    label = 'Treatment (26-30)' if elig == 1 else 'Control (31-35)'
    mean_age = np.average(group['AGE'], weights=group['PERWT'])
    print(f"   {label}: {mean_age:.2f}")

# Sex (1=Male, 2=Female in IPUMS)
print("\n   Proportion Male (weighted):")
for elig in [0, 1]:
    group = pre_df[pre_df['ELIGIBLE'] == elig]
    label = 'Treatment (26-30)' if elig == 1 else 'Control (31-35)'
    prop_male = np.average(group['SEX'] == 1, weights=group['PERWT'])
    print(f"   {label}: {prop_male*100:.1f}%")

# Marital status (1 = married spouse present)
print("\n   Proportion Married (weighted):")
for elig in [0, 1]:
    group = pre_df[pre_df['ELIGIBLE'] == elig]
    label = 'Treatment (26-30)' if elig == 1 else 'Control (31-35)'
    prop_married = np.average(group['MARST'] == 1, weights=group['PERWT'])
    print(f"   {label}: {prop_married*100:.1f}%")

# =============================================================================
# 4. MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n4. REGRESSION ANALYSIS")
print("-" * 40)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls)
print("\n4.1 Model 1: Basic Difference-in-Differences")
print("    FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER")

# Using WLS with person weights
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df['STATEFIP']})
print(model1.summary().tables[1])
print(f"\n   DiD Estimate (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"   SE (clustered by state):       {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value:                       {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   N: {int(model1.nobs):,}")

# Model 2: DiD with demographic controls
print("\n4.2 Model 2: DiD with Demographic Controls")
print("    FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER + SEX + MARST + EDUC_RECODE + NCHILD")

# Create dummy variables for categorical controls
df['MALE'] = (df['SEX'] == 1).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Education dummies (reference: Less than High School)
educ_dummies = pd.get_dummies(df['EDUC_RECODE'], prefix='EDUC', drop_first=False)
# Drop one category for reference
educ_cols = [c for c in educ_dummies.columns if 'Less' not in c]
for col in educ_cols:
    df[col] = educ_dummies[col]

model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + Q("EDUC_High School Degree") + Q("EDUC_Some College") + Q("EDUC_Two-Year Degree") + Q("EDUC_BA+")',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df['STATEFIP']})
print(model2.summary().tables[1])
print(f"\n   DiD Estimate (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"   SE (clustered by state):       {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value:                       {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 3: DiD with Year Fixed Effects
print("\n4.3 Model 3: DiD with Year Fixed Effects")
print("    FT ~ ELIGIBLE + ELIGIBLE*AFTER + Year FEs + Demographics")

# Create year dummies (excluding 2008 as reference)
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

year_fe_formula = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + Q("EDUC_High School Degree") + Q("EDUC_Some College") + Q("EDUC_Two-Year Degree") + Q("EDUC_BA+")'

model3 = smf.wls(year_fe_formula,
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df['STATEFIP']})
print(model3.summary().tables[1])
print(f"\n   DiD Estimate (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"   SE (clustered by state):       {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value:                       {model3.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 4: DiD with State Fixed Effects
print("\n4.4 Model 4: DiD with State and Year Fixed Effects")

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
for col in state_dummies.columns:
    df[col] = state_dummies[col]

# Build formula with state FEs
state_fe_cols = [c for c in df.columns if c.startswith('STATE_')]
state_fe_terms = ' + '.join(state_fe_cols)

formula4 = f'FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + Q("EDUC_High School Degree") + Q("EDUC_Some College") + Q("EDUC_Two-Year Degree") + Q("EDUC_BA+") + {state_fe_terms}'

model4 = smf.wls(formula4,
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df['STATEFIP']})
print(f"\n   DiD Estimate (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"   SE (clustered by state):       {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value:                       {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   N: {int(model4.nobs):,}")

# =============================================================================
# 5. ROBUSTNESS CHECKS
# =============================================================================
print("\n5. ROBUSTNESS CHECKS")
print("-" * 40)

# 5.1 Unweighted analysis
print("\n5.1 Unweighted OLS Analysis")
model_unweighted = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                            data=df).fit(cov_type='cluster',
                                         cov_kwds={'groups': df['STATEFIP']})
print(f"   DiD Estimate: {model_unweighted.params['ELIGIBLE_AFTER']:.4f}")
print(f"   SE (clustered): {model_unweighted.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   p-value: {model_unweighted.pvalues['ELIGIBLE_AFTER']:.4f}")

# 5.2 By sex subgroups
print("\n5.2 Heterogeneity by Sex")
for sex, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex]
    model_sex = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=df_sex, weights=df_sex['PERWT']).fit(cov_type='cluster',
                                                                   cov_kwds={'groups': df_sex['STATEFIP']})
    print(f"   {sex_label}: DiD = {model_sex.params['ELIGIBLE_AFTER']:.4f} (SE = {model_sex.bse['ELIGIBLE_AFTER']:.4f}), p = {model_sex.pvalues['ELIGIBLE_AFTER']:.4f}")

# 5.3 By education
print("\n5.3 Heterogeneity by Education")
df['HS_DEGREE_VAR'] = df['HS_DEGREE'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0)
for hs, hs_label in [(1, 'High School+'), (0, 'Less than HS')]:
    df_hs = df[df['HS_DEGREE_VAR'] == hs]
    if len(df_hs) > 100:
        model_hs = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                           data=df_hs, weights=df_hs['PERWT']).fit(cov_type='cluster',
                                                                    cov_kwds={'groups': df_hs['STATEFIP']})
        print(f"   {hs_label}: DiD = {model_hs.params['ELIGIBLE_AFTER']:.4f} (SE = {model_hs.bse['ELIGIBLE_AFTER']:.4f}), p = {model_hs.pvalues['ELIGIBLE_AFTER']:.4f}")

# 5.4 Placebo test - check pre-trends
print("\n5.4 Pre-Trends Test (Pre-Period Only)")
pre_df = df[df['AFTER'] == 0].copy()
pre_df['YEAR_TREND'] = pre_df['YEAR'] - 2008

model_pretrend = smf.wls('FT ~ ELIGIBLE * YEAR_TREND',
                          data=pre_df, weights=pre_df['PERWT']).fit(cov_type='cluster',
                                                                     cov_kwds={'groups': pre_df['STATEFIP']})
print(f"   Interaction (ELIGIBLE x YEAR_TREND): {model_pretrend.params['ELIGIBLE:YEAR_TREND']:.4f}")
print(f"   SE: {model_pretrend.bse['ELIGIBLE:YEAR_TREND']:.4f}")
print(f"   p-value: {model_pretrend.pvalues['ELIGIBLE:YEAR_TREND']:.4f}")
print("   (Non-significant coefficient suggests parallel pre-trends)")

# 5.5 Event study
print("\n5.5 Event Study Analysis")
# Create year-by-treatment interactions (relative to 2011)
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * (df['YEAR'] == year).astype(int)

event_formula = 'FT ~ ELIGIBLE + ELIGIBLE_YEAR_2008 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 + ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016 + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016'

model_event = smf.wls(event_formula,
                       data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                          cov_kwds={'groups': df['STATEFIP']})

print("   Year-specific treatment effects (relative to 2011):")
event_years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in event_years:
    coef = model_event.params[f'ELIGIBLE_YEAR_{year}']
    se = model_event.bse[f'ELIGIBLE_YEAR_{year}']
    pval = model_event.pvalues[f'ELIGIBLE_YEAR_{year}']
    print(f"   {year}: {coef:.4f} (SE = {se:.4f}, p = {pval:.4f})")

# =============================================================================
# 6. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("6. SUMMARY OF RESULTS")
print("=" * 80)

print("\n6.1 Main Results Summary Table:")
print("-" * 70)
print(f"{'Model':<40} {'DiD Estimate':>12} {'SE':>10} {'p-value':>10}")
print("-" * 70)
print(f"{'(1) Basic DiD':<40} {model1.params['ELIGIBLE_AFTER']:>12.4f} {model1.bse['ELIGIBLE_AFTER']:>10.4f} {model1.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'(2) + Demographics':<40} {model2.params['ELIGIBLE_AFTER']:>12.4f} {model2.bse['ELIGIBLE_AFTER']:>10.4f} {model2.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'(3) + Year FEs':<40} {model3.params['ELIGIBLE_AFTER']:>12.4f} {model3.bse['ELIGIBLE_AFTER']:>10.4f} {model3.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'(4) + State & Year FEs':<40} {model4.params['ELIGIBLE_AFTER']:>12.4f} {model4.bse['ELIGIBLE_AFTER']:>10.4f} {model4.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print("-" * 70)
print("Note: Standard errors clustered by state. All models weighted by PERWT.")

# Preferred estimate
print("\n6.2 PREFERRED ESTIMATE:")
print("-" * 70)
preferred_model = model4
print(f"   Model: DiD with State and Year Fixed Effects + Demographics")
print(f"   DiD Estimate: {preferred_model.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {preferred_model.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{preferred_model.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {preferred_model.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-statistic: {preferred_model.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   p-value: {preferred_model.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   Sample Size: {int(preferred_model.nobs):,}")

# Interpretation
print(f"\n6.3 INTERPRETATION:")
effect_pp = preferred_model.params['ELIGIBLE_AFTER'] * 100
print(f"   DACA eligibility is associated with a {effect_pp:.2f} percentage point")
print(f"   change in full-time employment probability.")
if preferred_model.pvalues['ELIGIBLE_AFTER'] < 0.05:
    print(f"   This effect is statistically significant at the 5% level.")
else:
    print(f"   This effect is NOT statistically significant at the 5% level.")

# Save key results for report
results_dict = {
    'preferred_estimate': preferred_model.params['ELIGIBLE_AFTER'],
    'preferred_se': preferred_model.bse['ELIGIBLE_AFTER'],
    'preferred_ci_lower': preferred_model.conf_int().loc['ELIGIBLE_AFTER', 0],
    'preferred_ci_upper': preferred_model.conf_int().loc['ELIGIBLE_AFTER', 1],
    'preferred_pvalue': preferred_model.pvalues['ELIGIBLE_AFTER'],
    'sample_size': int(preferred_model.nobs),
    'model1_estimate': model1.params['ELIGIBLE_AFTER'],
    'model2_estimate': model2.params['ELIGIBLE_AFTER'],
    'model3_estimate': model3.params['ELIGIBLE_AFTER'],
    'model4_estimate': model4.params['ELIGIBLE_AFTER'],
}

# Save results
pd.Series(results_dict).to_csv('analysis_results.csv')
print("\n   Results saved to analysis_results.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
