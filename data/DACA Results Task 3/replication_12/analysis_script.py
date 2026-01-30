"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US

Author: [Anonymized]
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================

print("=" * 80)
print("DACA REPLICATION STUDY - FULL ANALYSIS")
print("=" * 80)

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

print(f"\n1. DATA OVERVIEW")
print(f"   Total observations: {len(df):,}")
print(f"   Variables: {len(df.columns)}")
print(f"   Years: {sorted(df['YEAR'].unique())}")

# ============================================================================
# 2. SAMPLE DESCRIPTION
# ============================================================================

print(f"\n2. SAMPLE DESCRIPTION")
print("-" * 40)

# Treatment and control groups
print(f"\n   Treatment assignment (ELIGIBLE):")
print(f"   - Treatment (ages 26-30): {(df['ELIGIBLE']==1).sum():,} ({100*(df['ELIGIBLE']==1).mean():.1f}%)")
print(f"   - Control (ages 31-35): {(df['ELIGIBLE']==0).sum():,} ({100*(df['ELIGIBLE']==0).mean():.1f}%)")

print(f"\n   Time periods (AFTER):")
print(f"   - Pre-DACA (2008-2011): {(df['AFTER']==0).sum():,} ({100*(df['AFTER']==0).mean():.1f}%)")
print(f"   - Post-DACA (2013-2016): {(df['AFTER']==1).sum():,} ({100*(df['AFTER']==1).mean():.1f}%)")

# Outcome variable
print(f"\n   Outcome - Full-time employment (FT):")
print(f"   - Full-time (FT=1): {(df['FT']==1).sum():,} ({100*(df['FT']==1).mean():.1f}%)")
print(f"   - Not full-time (FT=0): {(df['FT']==0).sum():,} ({100*(df['FT']==0).mean():.1f}%)")

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# ============================================================================
# 3. SUMMARY STATISTICS
# ============================================================================

print(f"\n3. SUMMARY STATISTICS BY GROUP")
print("-" * 40)

# Demographics
demo_vars = ['SEX', 'AGE', 'MARST', 'NCHILD', 'FAMSIZE']
print("\n   Key demographic distributions:")

for var in ['SEX', 'MARST']:
    print(f"\n   {var}:")
    print(df.groupby(['ELIGIBLE', 'AFTER'])[var].value_counts(normalize=True).unstack().round(3))

# Education distribution
print("\n   Education (EDUC_RECODE):")
print(df.groupby(['ELIGIBLE', 'AFTER'])['EDUC_RECODE'].value_counts(normalize=True).unstack(level=[0,1]).round(3))

# ============================================================================
# 4. FULL-TIME EMPLOYMENT RATES BY GROUP
# ============================================================================

print(f"\n4. FULL-TIME EMPLOYMENT RATES")
print("-" * 40)

# Unweighted rates
print("\n   A. Unweighted FT rates:")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
ft_rates.columns = ['FT_Rate', 'Std_Dev', 'N']
print(ft_rates.round(4))

# Weighted rates
print("\n   B. Weighted FT rates (using PERWT):")
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

def weighted_std(group):
    weights = group['PERWT']
    values = group['FT']
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance)

ft_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: pd.Series({
        'FT_Rate': weighted_mean(x),
        'Weighted_N': x['PERWT'].sum()
    })
)
print(ft_weighted.round(4))

# ============================================================================
# 5. SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION
# ============================================================================

print(f"\n5. SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("-" * 40)

# Extract rates
control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
treat_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
treat_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

print(f"\n   Unweighted calculations:")
print(f"   Control group (ages 31-35):")
print(f"     Pre-DACA:  {control_pre:.4f}")
print(f"     Post-DACA: {control_post:.4f}")
print(f"     Change:    {control_post - control_pre:.4f}")

print(f"\n   Treatment group (ages 26-30):")
print(f"     Pre-DACA:  {treat_pre:.4f}")
print(f"     Post-DACA: {treat_post:.4f}")
print(f"     Change:    {treat_post - treat_pre:.4f}")

did_simple = (treat_post - treat_pre) - (control_post - control_pre)
print(f"\n   DiD Estimate: {did_simple:.4f} ({did_simple*100:.2f} percentage points)")

# Weighted DiD
def get_weighted_ft(eligible, after):
    subset = df[(df['ELIGIBLE']==eligible) & (df['AFTER']==after)]
    return np.average(subset['FT'], weights=subset['PERWT'])

control_pre_w = get_weighted_ft(0, 0)
control_post_w = get_weighted_ft(0, 1)
treat_pre_w = get_weighted_ft(1, 0)
treat_post_w = get_weighted_ft(1, 1)

did_weighted = (treat_post_w - treat_pre_w) - (control_post_w - control_pre_w)
print(f"\n   Weighted calculations:")
print(f"   Control: Pre={control_pre_w:.4f}, Post={control_post_w:.4f}, Change={control_post_w-control_pre_w:.4f}")
print(f"   Treatment: Pre={treat_pre_w:.4f}, Post={treat_post_w:.4f}, Change={treat_post_w-treat_pre_w:.4f}")
print(f"   Weighted DiD Estimate: {did_weighted:.4f} ({did_weighted*100:.2f} percentage points)")

# ============================================================================
# 6. REGRESSION ANALYSIS
# ============================================================================

print(f"\n6. REGRESSION ANALYSIS")
print("-" * 40)

# Model 1: Basic DiD (unweighted)
print("\n   MODEL 1: Basic DiD (OLS, unweighted)")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(f"   FT = {model1.params['Intercept']:.4f} + {model1.params['ELIGIBLE']:.4f}*ELIGIBLE + {model1.params['AFTER']:.4f}*AFTER + {model1.params['ELIGIBLE_AFTER']:.4f}*ELIGIBLE×AFTER")
print(f"\n   DiD coefficient (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust SE: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-stat: {model1.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   N: {int(model1.nobs)}")
print(f"   R-squared: {model1.rsquared:.4f}")

# Model 2: Basic DiD (weighted)
print("\n   MODEL 2: Basic DiD (WLS, using PERWT)")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust SE: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-stat: {model2.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 3: DiD with individual controls (unweighted)
print("\n   MODEL 3: DiD with individual controls (OLS)")
# Create dummy variables for categorical controls
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)  # 1=married spouse present, 2=married spouse absent

# Education dummies (reference: Less than High School)
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_AA'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA',
                 data=df).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust SE: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-stat: {model3.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model3.rsquared:.4f}")

# Model 4: DiD with individual controls (weighted)
print("\n   MODEL 4: DiD with individual controls (WLS)")
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust SE: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-stat: {model4.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 5: DiD with state fixed effects (weighted)
print("\n   MODEL 5: DiD with state fixed effects (WLS)")
df['STATE_FE'] = pd.Categorical(df['STATEFIP'])
model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust SE: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-stat: {model5.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 6: DiD with year fixed effects (weighted)
print("\n   MODEL 6: DiD with year and state fixed effects (WLS)")
model6 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + C(STATEFIP) + C(YEAR)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust SE: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-stat: {model6.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")

# ============================================================================
# 7. ROBUSTNESS CHECKS
# ============================================================================

print(f"\n7. ROBUSTNESS CHECKS")
print("-" * 40)

# 7.1 Separate pre-trends check
print("\n   7.1 Pre-trends check (restricting to pre-period only)")
df_pre = df[df['AFTER'] == 0].copy()
df_pre['YEAR_TREND'] = df_pre['YEAR'] - 2008

# Test for differential pre-trends
model_pretrend = smf.wls('FT ~ ELIGIBLE * YEAR_TREND', data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"   Interaction (ELIGIBLE × Year): {model_pretrend.params['ELIGIBLE:YEAR_TREND']:.4f}")
print(f"   SE: {model_pretrend.bse['ELIGIBLE:YEAR_TREND']:.4f}")
print(f"   p-value: {model_pretrend.pvalues['ELIGIBLE:YEAR_TREND']:.4f}")
if model_pretrend.pvalues['ELIGIBLE:YEAR_TREND'] > 0.05:
    print("   --> No evidence of differential pre-trends (p > 0.05)")
else:
    print("   --> Evidence of differential pre-trends (p <= 0.05)")

# 7.2 Event study / dynamic effects
print("\n   7.2 Event study - Year-by-year effects")
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions (2011 is reference year)
for yr in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIG_X_{yr}'] = df['ELIGIBLE'] * df[f'YEAR_{yr}']

model_event = smf.wls('FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + '
                      'ELIG_X_2008 + ELIG_X_2009 + ELIG_X_2010 + ELIG_X_2013 + ELIG_X_2014 + ELIG_X_2015 + ELIG_X_2016',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("   Year-specific treatment effects (relative to 2011):")
for yr in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'ELIG_X_{yr}']
    se = model_event.bse[f'ELIG_X_{yr}']
    pval = model_event.pvalues[f'ELIG_X_{yr}']
    sig = "*" if pval < 0.05 else ("+" if pval < 0.10 else "")
    print(f"   {yr}: {coef:7.4f} (SE: {se:.4f}) {sig}")

# 7.3 Subgroup analysis by gender
print("\n   7.3 Subgroup analysis by gender")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex]
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"   {label}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}), "
          f"p = {model_sub.pvalues['ELIGIBLE_AFTER']:.4f}, N = {int(model_sub.nobs)}")

# 7.4 Sensitivity to age bandwidth
print("\n   7.4 Sensitivity to age bandwidth (narrower bandwidth)")
# Use ages 27-29 vs 32-34
df['NARROW_BAND'] = ((df['AGE_IN_JUNE_2012'] >= 27) & (df['AGE_IN_JUNE_2012'] < 30)) | \
                    ((df['AGE_IN_JUNE_2012'] >= 32) & (df['AGE_IN_JUNE_2012'] < 35))
df_narrow = df[df['NARROW_BAND']].copy()
model_narrow = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                       data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"   Narrower bandwidth (27-29 vs 32-34):")
print(f"   DiD = {model_narrow.params['ELIGIBLE_AFTER']:.4f} (SE: {model_narrow.bse['ELIGIBLE_AFTER']:.4f}), "
      f"p = {model_narrow.pvalues['ELIGIBLE_AFTER']:.4f}, N = {int(model_narrow.nobs)}")

# ============================================================================
# 8. SUMMARY TABLE
# ============================================================================

print(f"\n8. SUMMARY OF MAIN RESULTS")
print("-" * 80)
print(f"{'Model':<45} {'DiD Est':>10} {'SE':>10} {'95% CI':>20} {'p-value':>10}")
print("-" * 80)

models = [
    ('1. Basic DiD (unweighted)', model1),
    ('2. Basic DiD (weighted)', model2),
    ('3. With controls (unweighted)', model3),
    ('4. With controls (weighted)', model4),
    ('5. + State FE (weighted)', model5),
    ('6. + Year FE (weighted)', model6),
]

for name, model in models:
    coef = model.params['ELIGIBLE_AFTER']
    se = model.bse['ELIGIBLE_AFTER']
    ci_low = model.conf_int().loc['ELIGIBLE_AFTER', 0]
    ci_high = model.conf_int().loc['ELIGIBLE_AFTER', 1]
    pval = model.pvalues['ELIGIBLE_AFTER']
    print(f"{name:<45} {coef:>10.4f} {se:>10.4f} [{ci_low:>7.4f}, {ci_high:>7.4f}] {pval:>10.4f}")

print("-" * 80)

# ============================================================================
# 9. PREFERRED ESTIMATE
# ============================================================================

print(f"\n9. PREFERRED ESTIMATE")
print("=" * 80)
preferred = model4  # Weighted DiD with individual controls
print(f"   Model: Weighted DiD with individual-level controls (Model 4)")
print(f"   Specification: WLS regression with PERWT weights, robust (HC1) standard errors")
print(f"   Controls: Female, Married, Age, Number of children, Education dummies")
print(f"")
print(f"   Effect size (DiD estimate): {preferred.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard error: {preferred.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% Confidence interval: [{preferred.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {preferred.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-statistic: {preferred.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {preferred.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   Sample size: {int(preferred.nobs)}")
print(f"")
print(f"   Interpretation: DACA eligibility increased the probability of full-time")
print(f"   employment by approximately {preferred.params['ELIGIBLE_AFTER']*100:.1f} percentage points among")
print(f"   Hispanic-Mexican Mexican-born individuals aged 26-30 in June 2012,")
print(f"   compared to similar individuals aged 31-35.")
if preferred.pvalues['ELIGIBLE_AFTER'] < 0.05:
    print(f"   This effect is statistically significant at the 5% level.")
elif preferred.pvalues['ELIGIBLE_AFTER'] < 0.10:
    print(f"   This effect is marginally significant at the 10% level.")
else:
    print(f"   This effect is not statistically significant at conventional levels.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# ============================================================================
# 10. SAVE RESULTS FOR LATEX
# ============================================================================

# Save detailed regression output
with open('regression_results.txt', 'w') as f:
    f.write("MODEL 1: Basic DiD (OLS, unweighted)\n")
    f.write(model1.summary().as_text())
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("MODEL 2: Basic DiD (WLS, weighted)\n")
    f.write(model2.summary().as_text())
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("MODEL 3: DiD with controls (OLS)\n")
    f.write(model3.summary().as_text())
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("MODEL 4: DiD with controls (WLS) - PREFERRED\n")
    f.write(model4.summary().as_text())
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("MODEL 5: DiD with state FE (WLS)\n")
    f.write(model5.summary().as_text())
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("MODEL 6: DiD with year and state FE (WLS)\n")
    f.write(model6.summary().as_text())

print("\nRegression results saved to 'regression_results.txt'")
