"""
DACA Replication Study - Difference-in-Differences Analysis
Examining the effect of DACA eligibility on full-time employment

Author: Anonymous
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

print("="*80)
print("DACA REPLICATION STUDY: DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"\nTotal observations: {len(df):,}")
print(f"Total variables: {len(df.columns)}")

# =============================================================================
# 2. DATA VALIDATION AND DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "="*80)
print("DATA VALIDATION AND DESCRIPTIVE STATISTICS")
print("="*80)

# Check year distribution
print("\n--- Year Distribution ---")
print(df['YEAR'].value_counts().sort_index())

# Check key variable distributions
print("\n--- FT (Full-time) Distribution ---")
print(df['FT'].value_counts())

print("\n--- AFTER (Post-treatment) Distribution ---")
print(df['AFTER'].value_counts())

print("\n--- ELIGIBLE (Treatment group) Distribution ---")
print(df['ELIGIBLE'].value_counts())

# Cross-tabulation: ELIGIBLE x AFTER
print("\n--- Treatment x Period Cross-tabulation ---")
cross_tab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
cross_tab.columns = ['Pre (2008-2011)', 'Post (2013-2016)', 'Total']
cross_tab.index = ['Comparison (31-35)', 'Treatment (26-30)', 'Total']
print(cross_tab)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# =============================================================================
# 3. RAW MEANS BY GROUP AND PERIOD
# =============================================================================

print("\n" + "="*80)
print("RAW MEANS BY GROUP AND PERIOD")
print("="*80)

# Calculate means for each cell
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
means.index = pd.MultiIndex.from_tuples([
    ('Comparison (31-35)', 'Pre (2008-2011)'),
    ('Comparison (31-35)', 'Post (2013-2016)'),
    ('Treatment (26-30)', 'Pre (2008-2011)'),
    ('Treatment (26-30)', 'Post (2013-2016)')
])
print("\n--- Full-time Employment Rates (Unweighted) ---")
print(means.round(4))

# Calculate weighted means
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

weighted_means = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean)
print("\n--- Full-time Employment Rates (Weighted by PERWT) ---")
print(weighted_means.round(4))

# Manual DiD calculation
pre_treat_unw = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treat_unw = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
pre_control_unw = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control_unw = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

did_unweighted = (post_treat_unw - pre_treat_unw) - (post_control_unw - pre_control_unw)

print("\n--- DiD Components (Unweighted) ---")
print(f"Treatment (26-30) Pre:  {pre_treat_unw:.4f}")
print(f"Treatment (26-30) Post: {post_treat_unw:.4f}")
print(f"Treatment Change:       {post_treat_unw - pre_treat_unw:.4f}")
print(f"\nComparison (31-35) Pre:  {pre_control_unw:.4f}")
print(f"Comparison (31-35) Post: {post_control_unw:.4f}")
print(f"Comparison Change:       {post_control_unw - pre_control_unw:.4f}")
print(f"\nDiD Estimate (Unweighted): {did_unweighted:.4f}")

# Weighted DiD
def wmean(d, col, w):
    return np.average(d[col], weights=d[w])

pre_treat_w = wmean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)], 'FT', 'PERWT')
post_treat_w = wmean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)], 'FT', 'PERWT')
pre_control_w = wmean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)], 'FT', 'PERWT')
post_control_w = wmean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)], 'FT', 'PERWT')

did_weighted = (post_treat_w - pre_treat_w) - (post_control_w - pre_control_w)

print("\n--- DiD Components (Weighted) ---")
print(f"Treatment (26-30) Pre:  {pre_treat_w:.4f}")
print(f"Treatment (26-30) Post: {post_treat_w:.4f}")
print(f"Treatment Change:       {post_treat_w - pre_treat_w:.4f}")
print(f"\nComparison (31-35) Pre:  {pre_control_w:.4f}")
print(f"Comparison (31-35) Post: {post_control_w:.4f}")
print(f"Comparison Change:       {post_control_w - pre_control_w:.4f}")
print(f"\nDiD Estimate (Weighted): {did_weighted:.4f}")

# =============================================================================
# 4. REGRESSION ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD (OLS, no weights, no clustering)
print("\n--- Model 1: Basic DiD (OLS, no weights, no clustering) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary())

# Model 2: DiD with robust standard errors
print("\n--- Model 2: DiD with Robust Standard Errors (HC3) ---")
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC3')
print(model2.summary())

# Model 3: DiD with state-clustered standard errors
print("\n--- Model 3: DiD with State-Clustered Standard Errors ---")
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}
)
print(model3.summary())

# Model 4: Weighted DiD with state-clustered standard errors
print("\n--- Model 4: Weighted DiD with State-Clustered Standard Errors ---")
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}
)
print(model4.summary())

# =============================================================================
# 5. MODELS WITH COVARIATES
# =============================================================================

print("\n" + "="*80)
print("MODELS WITH COVARIATES")
print("="*80)

# Create demographic covariates
# SEX: 1=Male, 2=Female (IPUMS coding)
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# MARST: 1=Married spouse present, 6=Never married
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Education recodes - already provided as EDUC_RECODE
# Create dummies for education
df['ED_LESS_HS'] = (df['EDUC_RECODE'] == 'Less than High School').astype(int)
df['ED_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['ED_SOME_COLLEGE'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['ED_TWO_YEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['ED_BA_PLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Number of children
df['NCHILD_capped'] = df['NCHILD'].clip(upper=5)

# State unemployment rate (already in data as UNEMP)

# Model 5: DiD with basic demographic covariates (weighted, clustered)
print("\n--- Model 5: Weighted DiD with Demographics (clustered SEs) ---")
model5 = smf.wls(
    'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD_capped',
    data=df, weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model5.summary())

# Model 6: DiD with demographics + education (weighted, clustered)
print("\n--- Model 6: Weighted DiD with Demographics + Education (clustered SEs) ---")
model6 = smf.wls(
    'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD_capped + ED_HS + ED_SOME_COLLEGE + ED_TWO_YEAR + ED_BA_PLUS',
    data=df, weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model6.summary())

# Model 7: DiD with demographics + education + state labor market (weighted, clustered)
print("\n--- Model 7: Weighted DiD with Demographics + Education + State Labor Market (clustered SEs) ---")
model7 = smf.wls(
    'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD_capped + ED_HS + ED_SOME_COLLEGE + ED_TWO_YEAR + ED_BA_PLUS + UNEMP + LFPR',
    data=df, weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model7.summary())

# =============================================================================
# 6. MODELS WITH FIXED EFFECTS
# =============================================================================

print("\n" + "="*80)
print("MODELS WITH FIXED EFFECTS")
print("="*80)

# Model 8: Year fixed effects
print("\n--- Model 8: Year Fixed Effects (weighted, clustered SEs) ---")
model8 = smf.wls(
    'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR)',
    data=df, weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model8.summary())

# Model 9: State fixed effects
print("\n--- Model 9: State Fixed Effects (weighted, clustered SEs) ---")
model9 = smf.wls(
    'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)',
    data=df, weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model9.summary())

# Model 10: Year and State fixed effects
print("\n--- Model 10: Year + State Fixed Effects (weighted, clustered SEs) ---")
model10 = smf.wls(
    'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP)',
    data=df, weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model10.summary())

# =============================================================================
# 7. FULLY SPECIFIED MODEL (PREFERRED)
# =============================================================================

print("\n" + "="*80)
print("PREFERRED MODEL SPECIFICATION")
print("="*80)

# Model 11: Full specification with year FE, state FE, demographics, education
print("\n--- Model 11: Preferred Model (Year FE + State FE + Demographics + Education) ---")
model11 = smf.wls(
    'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD_capped + ED_HS + ED_SOME_COLLEGE + ED_TWO_YEAR + ED_BA_PLUS + C(YEAR) + C(STATEFIP)',
    data=df, weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model11.summary())

# =============================================================================
# 8. PARALLEL TRENDS CHECK
# =============================================================================

print("\n" + "="*80)
print("PARALLEL TRENDS CHECK")
print("="*80)

# Calculate yearly means by treatment group
yearly_means = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'FT_mean': np.average(x['FT'], weights=x['PERWT']),
        'N': len(x)
    })
).reset_index()
yearly_means_pivot = yearly_means.pivot(index='YEAR', columns='ELIGIBLE', values='FT_mean')
yearly_means_pivot.columns = ['Comparison (31-35)', 'Treatment (26-30)']
print("\n--- Full-time Employment Rates by Year (Weighted) ---")
print(yearly_means_pivot.round(4))

# Event study approach - create year indicators interacted with treatment
pre_years = [2008, 2009, 2010, 2011]
post_years = [2013, 2014, 2015, 2016]

# Use 2011 as reference year
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
# 2011 is reference
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions
df['ELIG_2008'] = df['ELIGIBLE'] * df['YEAR_2008']
df['ELIG_2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIG_2010'] = df['ELIGIBLE'] * df['YEAR_2010']
df['ELIG_2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIG_2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIG_2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIG_2016'] = df['ELIGIBLE'] * df['YEAR_2016']

# Event study regression
print("\n--- Event Study Regression (weighted, clustered SEs) ---")
event_model = smf.wls(
    'FT ~ ELIGIBLE + ELIG_2008 + ELIG_2009 + ELIG_2010 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016 + C(YEAR) + C(STATEFIP)',
    data=df, weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(event_model.summary())

# Extract event study coefficients
event_coefs = {}
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = event_model.params.get(f'ELIG_{year}', np.nan)
    se = event_model.bse.get(f'ELIG_{year}', np.nan)
    event_coefs[year] = {'coef': coef, 'se': se}

print("\n--- Event Study Coefficients (relative to 2011) ---")
print("Year\tCoefficient\tStd. Error\t95% CI")
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    if year == 2011:
        print(f"{year}\t0.0000\t\t(reference)")
    else:
        coef = event_coefs[year]['coef']
        se = event_coefs[year]['se']
        ci_lo = coef - 1.96*se
        ci_hi = coef + 1.96*se
        print(f"{year}\t{coef:.4f}\t\t{se:.4f}\t\t[{ci_lo:.4f}, {ci_hi:.4f}]")

# =============================================================================
# 9. ROBUSTNESS CHECKS
# =============================================================================

print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Linear Probability Model vs Logit
print("\n--- Logit Model (for comparison) ---")
try:
    logit_model = smf.glm(
        'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
        data=df, family=sm.families.Binomial(),
        freq_weights=df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
    print("Logit coefficients:")
    print(logit_model.params.round(4))
    print("\nMarginal effect at mean (ELIGIBLE_AFTER):")
    # Approximate marginal effect
    mean_prob = logit_model.predict().mean()
    logit_me = logit_model.params['ELIGIBLE_AFTER'] * mean_prob * (1 - mean_prob)
    print(f"Approximate marginal effect: {logit_me:.4f}")
except Exception as e:
    print(f"Logit model failed: {e}")

# Robustness 2: Different age bandwidths
print("\n--- Robustness: Different Age Bandwidths ---")
# The data already restricts to ages 26-30 (treatment) and 31-35 (comparison)
# Check age distribution
print("Age distribution in data:")
print(df['AGE'].value_counts().sort_index())

# Robustness 3: By sex
print("\n--- Heterogeneity: By Sex ---")
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    model_sub = smf.wls(
        'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
        data=df_sub, weights=df_sub['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': df_sub['STATEFIP']})
    print(f"\n{sex_label} (N={len(df_sub):,}):")
    print(f"  DiD Coefficient: {model_sub.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  Std. Error: {model_sub.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"  95% CI: [{model_sub.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_sub.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
    print(f"  p-value: {model_sub.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# 10. SUMMARY OF RESULTS
# =============================================================================

print("\n" + "="*80)
print("SUMMARY OF KEY RESULTS")
print("="*80)

results_summary = []

# Model descriptions and results
models_info = [
    ("Model 1: Basic OLS", model1),
    ("Model 2: Robust SE", model2),
    ("Model 3: Clustered SE", model3),
    ("Model 4: Weighted + Clustered", model4),
    ("Model 5: + Demographics", model5),
    ("Model 6: + Education", model6),
    ("Model 7: + Labor Market", model7),
    ("Model 8: Year FE", model8),
    ("Model 9: State FE", model9),
    ("Model 10: Year + State FE", model10),
    ("Model 11: Preferred (Full)", model11),
]

print("\n--- DiD Coefficient (ELIGIBLE x AFTER) Across Specifications ---")
print(f"{'Model':<35} {'Coef':>10} {'SE':>10} {'95% CI':>25} {'p-value':>10}")
print("-"*95)

for name, model in models_info:
    coef = model.params['ELIGIBLE_AFTER']
    se = model.bse['ELIGIBLE_AFTER']
    ci = model.conf_int().loc['ELIGIBLE_AFTER']
    pval = model.pvalues['ELIGIBLE_AFTER']
    print(f"{name:<35} {coef:>10.4f} {se:>10.4f} [{ci[0]:>9.4f}, {ci[1]:>8.4f}] {pval:>10.4f}")
    results_summary.append({
        'Model': name,
        'Coefficient': coef,
        'Std_Error': se,
        'CI_Lower': ci[0],
        'CI_Upper': ci[1],
        'p_value': pval
    })

# =============================================================================
# 11. PREFERRED ESTIMATE
# =============================================================================

print("\n" + "="*80)
print("PREFERRED ESTIMATE")
print("="*80)

# Use Model 11 as preferred specification
preferred_model = model11
preferred_coef = preferred_model.params['ELIGIBLE_AFTER']
preferred_se = preferred_model.bse['ELIGIBLE_AFTER']
preferred_ci = preferred_model.conf_int().loc['ELIGIBLE_AFTER']
preferred_pval = preferred_model.pvalues['ELIGIBLE_AFTER']
n_obs = preferred_model.nobs

print(f"\nPreferred Model: Year FE + State FE + Demographics + Education")
print(f"Sample Size: {int(n_obs):,}")
print(f"\nDiD Estimate (Effect of DACA eligibility on full-time employment):")
print(f"  Coefficient: {preferred_coef:.4f}")
print(f"  Standard Error: {preferred_se:.4f}")
print(f"  95% Confidence Interval: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"  t-statistic: {preferred_coef/preferred_se:.4f}")
print(f"  p-value: {preferred_pval:.4f}")

print(f"\nInterpretation:")
if preferred_coef > 0:
    print(f"  DACA eligibility is associated with a {preferred_coef*100:.2f} percentage point")
    print(f"  increase in the probability of full-time employment.")
else:
    print(f"  DACA eligibility is associated with a {abs(preferred_coef)*100:.2f} percentage point")
    print(f"  decrease in the probability of full-time employment.")

if preferred_pval < 0.05:
    print(f"  This effect is statistically significant at the 5% level (p={preferred_pval:.4f}).")
elif preferred_pval < 0.10:
    print(f"  This effect is statistically significant at the 10% level (p={preferred_pval:.4f}).")
else:
    print(f"  This effect is NOT statistically significant at conventional levels (p={preferred_pval:.4f}).")

# =============================================================================
# 12. SAVE RESULTS
# =============================================================================

results_df = pd.DataFrame(results_summary)
results_df.to_csv('regression_results_summary.csv', index=False)
print("\n\nResults saved to: regression_results_summary.csv")

# Save detailed output
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
