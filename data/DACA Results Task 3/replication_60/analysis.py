"""
DACA Replication Study: Effect of DACA Eligibility on Full-Time Employment
===========================================================================
This script performs a difference-in-differences analysis to estimate the causal
effect of DACA eligibility on full-time employment among Hispanic-Mexican
Mexican-born individuals in the United States.

Research Design:
- Treatment Group: ELIGIBLE=1, individuals aged 26-30 in June 2012
- Control Group: ELIGIBLE=0, individuals aged 31-35 in June 2012
- Pre-period: 2008-2011 (AFTER=0)
- Post-period: 2013-2016 (AFTER=1)
- Outcome: FT (full-time employment, defined as usually working 35+ hours/week)
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

# =============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

print(f"\n1. DATA OVERVIEW")
print(f"   Total observations: {len(df):,}")
print(f"   Years covered: {df['YEAR'].min()} - {df['YEAR'].max()}")
print(f"   Number of variables: {len(df.columns)}")

# =============================================================================
# 2. SAMPLE DESCRIPTION
# =============================================================================

print(f"\n2. SAMPLE CHARACTERISTICS")
print("-"*40)

# Group sizes
print("\n   Treatment/Control Group Distribution:")
eligible_counts = df['ELIGIBLE'].value_counts().sort_index()
print(f"   Control (ELIGIBLE=0, ages 31-35): {eligible_counts[0]:,}")
print(f"   Treatment (ELIGIBLE=1, ages 26-30): {eligible_counts[1]:,}")

# Pre/Post distribution
print("\n   Pre/Post Period Distribution:")
after_counts = df['AFTER'].value_counts().sort_index()
print(f"   Pre-DACA (2008-2011): {after_counts[0]:,}")
print(f"   Post-DACA (2013-2016): {after_counts[1]:,}")

# Cross-tabulation
print("\n   Cross-tabulation (ELIGIBLE x AFTER):")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (0)', 'Treatment (1)', 'Total']
crosstab.columns = ['Pre (0)', 'Post (1)', 'Total']
print(crosstab.to_string())

# =============================================================================
# 3. DESCRIPTIVE STATISTICS BY GROUP
# =============================================================================

print(f"\n3. DESCRIPTIVE STATISTICS")
print("-"*40)

# Full-time employment rates by group
print("\n   Full-Time Employment Rates by Group:")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
ft_rates.index = pd.MultiIndex.from_tuples(
    [('Control', 'Pre'), ('Control', 'Post'), ('Treatment', 'Pre'), ('Treatment', 'Post')]
)
print(ft_rates.to_string())

# Calculate simple DiD
control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
treat_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
treat_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

did_simple = (treat_post - treat_pre) - (control_post - control_pre)

print(f"\n   Simple DiD Calculation:")
print(f"   Control group change: {control_post:.4f} - {control_pre:.4f} = {control_post - control_pre:.4f}")
print(f"   Treatment group change: {treat_post:.4f} - {treat_pre:.4f} = {treat_post - treat_pre:.4f}")
print(f"   DiD estimate: {did_simple:.4f}")

# Demographics by group
print("\n   Demographics by Treatment Group:")
for group, name in [(0, 'Control'), (1, 'Treatment')]:
    subset = df[df['ELIGIBLE'] == group]
    print(f"\n   {name} Group:")
    print(f"     Mean age in June 2012: {subset['AGE_IN_JUNE_2012'].mean():.2f}")
    print(f"     Male proportion: {(subset['SEX']==1).mean():.3f}")
    print(f"     Mean years in USA: {subset['YRSUSA1'].mean():.2f}")

# =============================================================================
# 4. BASELINE BALANCE CHECK
# =============================================================================

print(f"\n4. BASELINE BALANCE CHECK (Pre-period only)")
print("-"*40)

pre_data = df[df['AFTER'] == 0].copy()

balance_vars = {
    'AGE_IN_JUNE_2012': 'Age in June 2012',
    'SEX': 'Sex (1=Male, 2=Female)',
    'YRSUSA1': 'Years in USA',
    'FAMSIZE': 'Family Size',
    'NCHILD': 'Number of Children',
}

print("\n   Variable               Control Mean  Treatment Mean  Difference")
print("   " + "-"*65)
for var, label in balance_vars.items():
    control_mean = pre_data[pre_data['ELIGIBLE']==0][var].mean()
    treat_mean = pre_data[pre_data['ELIGIBLE']==1][var].mean()
    diff = treat_mean - control_mean
    print(f"   {label:22s}  {control_mean:12.3f}  {treat_mean:14.3f}  {diff:10.3f}")

# =============================================================================
# 5. MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================

print(f"\n5. MAIN DiD REGRESSION ANALYSIS")
print("-"*40)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls)
print("\n   Model 1: Basic DiD (no controls)")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(f"   DiD coefficient (ELIGIBLE x AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-statistic: {model1.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model1.rsquared:.4f}")
print(f"   N: {int(model1.nobs):,}")

# Model 2: DiD with demographic controls
print("\n   Model 2: DiD with demographic controls")
# Create dummy variables for categorical controls
df['MALE'] = (df['SEX'] == 1).astype(int)

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + FAMSIZE + NCHILD + YRSUSA1',
                 data=df).fit()
print(f"   DiD coefficient (ELIGIBLE x AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-statistic: {model2.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model2.rsquared:.4f}")
print(f"   N: {int(model2.nobs):,}")

# Model 3: DiD with demographic + education controls
print("\n   Model 3: DiD with demographic + education controls")
# Create education dummies
df['HS_ONLY'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['SOME_COLLEGE'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['TWO_YEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['BA_PLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + FAMSIZE + NCHILD + YRSUSA1 + SOME_COLLEGE + TWO_YEAR + BA_PLUS',
                 data=df).fit()
print(f"   DiD coefficient (ELIGIBLE x AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-statistic: {model3.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model3.rsquared:.4f}")
print(f"   N: {int(model3.nobs):,}")

# Model 4: DiD with year fixed effects
print("\n   Model 4: DiD with year fixed effects")
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

model4 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + FAMSIZE + NCHILD + YRSUSA1 + SOME_COLLEGE + TWO_YEAR + BA_PLUS + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016',
                 data=df).fit()
print(f"   DiD coefficient (ELIGIBLE x AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model4.rsquared:.4f}")
print(f"   N: {int(model4.nobs):,}")

# Model 5: Full model with state controls
print("\n   Model 5: Full model with state-level controls")
model5 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + FAMSIZE + NCHILD + YRSUSA1 + SOME_COLLEGE + TWO_YEAR + BA_PLUS + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + UNEMP + LFPR',
                 data=df).fit()
print(f"   DiD coefficient (ELIGIBLE x AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-statistic: {model5.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model5.rsquared:.4f}")
print(f"   N: {int(model5.nobs):,}")

# =============================================================================
# 6. ROBUST STANDARD ERRORS
# =============================================================================

print(f"\n6. PREFERRED MODEL WITH ROBUST STANDARD ERRORS")
print("-"*40)

# Preferred model (Model 4) with HC1 robust standard errors
model_robust = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + FAMSIZE + NCHILD + YRSUSA1 + SOME_COLLEGE + TWO_YEAR + BA_PLUS + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016',
                       data=df).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE x AFTER): {model_robust.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust Standard Error: {model_robust.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model_robust.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_robust.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-statistic: {model_robust.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model_robust.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# 7. WEIGHTED ANALYSIS (Using Survey Weights)
# =============================================================================

print(f"\n7. WEIGHTED ANALYSIS (Using PERWT)")
print("-"*40)

# WLS with person weights
from statsmodels.regression.linear_model import WLS

# Prepare variables
y = df['FT'].values
X = df[['ELIGIBLE', 'ELIGIBLE_AFTER', 'MALE', 'FAMSIZE', 'NCHILD', 'YRSUSA1',
        'SOME_COLLEGE', 'TWO_YEAR', 'BA_PLUS',
        'YEAR_2009', 'YEAR_2010', 'YEAR_2011', 'YEAR_2013', 'YEAR_2014', 'YEAR_2015', 'YEAR_2016']].copy()
X = sm.add_constant(X)
weights = df['PERWT'].values

model_weighted = WLS(y, X, weights=weights).fit()
did_idx = list(X.columns).index('ELIGIBLE_AFTER')
conf_int = model_weighted.conf_int()
print(f"   Weighted DiD coefficient: {model_weighted.params[did_idx]:.4f}")
print(f"   Standard Error: {model_weighted.bse[did_idx]:.4f}")
print(f"   95% CI: [{conf_int.iloc[did_idx, 0]:.4f}, {conf_int.iloc[did_idx, 1]:.4f}]")

# =============================================================================
# 8. HETEROGENEITY ANALYSIS BY SEX
# =============================================================================

print(f"\n8. HETEROGENEITY ANALYSIS BY SEX")
print("-"*40)

# Males
df_male = df[df['SEX'] == 1].copy()
model_male = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FAMSIZE + NCHILD + YRSUSA1 + SOME_COLLEGE + TWO_YEAR + BA_PLUS + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016',
                     data=df_male).fit()
print(f"   Males (N={int(model_male.nobs):,}):")
print(f"   DiD coefficient: {model_male.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model_male.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model_male.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_male.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value: {model_male.pvalues['ELIGIBLE_AFTER']:.4f}")

# Females
df_female = df[df['SEX'] == 2].copy()
model_female = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FAMSIZE + NCHILD + YRSUSA1 + SOME_COLLEGE + TWO_YEAR + BA_PLUS + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016',
                       data=df_female).fit()
print(f"\n   Females (N={int(model_female.nobs):,}):")
print(f"   DiD coefficient: {model_female.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model_female.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model_female.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_female.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value: {model_female.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# 9. EVENT STUDY (Year-by-Year Effects)
# =============================================================================

print(f"\n9. EVENT STUDY: Year-by-Year Effects")
print("-"*40)

# Create year-specific treatment interactions
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * (df['YEAR'] == year).astype(int)

# Event study regression (2011 as reference year)
event_formula = 'FT ~ ELIGIBLE + ELIGIBLE_YEAR_2008 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 + ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016 + MALE + FAMSIZE + NCHILD + YRSUSA1 + SOME_COLLEGE + TWO_YEAR + BA_PLUS + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016'
model_event = smf.ols(event_formula, data=df).fit()

print("   Year     Coefficient    Std.Error    95% CI")
print("   " + "-"*55)
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'ELIGIBLE_YEAR_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_lo = model_event.conf_int().loc[var, 0]
    ci_hi = model_event.conf_int().loc[var, 1]
    print(f"   {year}     {coef:10.4f}    {se:9.4f}    [{ci_lo:.4f}, {ci_hi:.4f}]")
print("   2011     (reference year)")

# =============================================================================
# 10. PLACEBO TEST: Pre-treatment trends
# =============================================================================

print(f"\n10. PLACEBO TEST: Pre-treatment trends")
print("-"*40)

# Use only pre-period data and create a fake "treatment" at 2010
pre_only = df[df['AFTER'] == 0].copy()
pre_only['FAKE_POST'] = (pre_only['YEAR'] >= 2010).astype(int)
pre_only['FAKE_TREAT'] = pre_only['ELIGIBLE'] * pre_only['FAKE_POST']

placebo_model = smf.ols('FT ~ ELIGIBLE + FAKE_POST + FAKE_TREAT + MALE + FAMSIZE + NCHILD + YRSUSA1',
                        data=pre_only).fit()
print(f"   Placebo DiD (fake treatment at 2010): {placebo_model.params['FAKE_TREAT']:.4f}")
print(f"   Standard Error: {placebo_model.bse['FAKE_TREAT']:.4f}")
print(f"   p-value: {placebo_model.pvalues['FAKE_TREAT']:.4f}")
print("   (Insignificant result supports parallel trends assumption)")

# =============================================================================
# 11. SUMMARY TABLE OF REGRESSION RESULTS
# =============================================================================

print(f"\n11. SUMMARY TABLE OF ALL MODELS")
print("-"*80)
print(f"{'Model':<35} {'Coef':>10} {'SE':>10} {'95% CI':>25} {'p-value':>10} {'N':>10}")
print("-"*80)

models_summary = [
    ('(1) Basic DiD', model1, 'ELIGIBLE_AFTER'),
    ('(2) + Demographics', model2, 'ELIGIBLE_AFTER'),
    ('(3) + Education', model3, 'ELIGIBLE_AFTER'),
    ('(4) + Year FE (Preferred)', model4, 'ELIGIBLE_AFTER'),
    ('(5) + State Controls', model5, 'ELIGIBLE_AFTER'),
]

for name, model, var in models_summary:
    coef = model.params[var]
    se = model.bse[var]
    ci = f"[{model.conf_int().loc[var, 0]:.4f}, {model.conf_int().loc[var, 1]:.4f}]"
    pval = model.pvalues[var]
    n = int(model.nobs)
    print(f"{name:<35} {coef:>10.4f} {se:>10.4f} {ci:>25} {pval:>10.4f} {n:>10,}")

print("-"*80)

# =============================================================================
# 12. PREFERRED ESTIMATE SUMMARY
# =============================================================================

print(f"\n12. PREFERRED ESTIMATE SUMMARY")
print("="*80)
print(f"   Model: DiD with individual controls and year fixed effects (Model 4)")
print(f"   Sample: All DACA-eligible Hispanic-Mexican Mexican-born individuals")
print(f"           Treatment: Ages 26-30 in June 2012")
print(f"           Control: Ages 31-35 in June 2012")
print(f"   Outcome: Full-time employment (working 35+ hours/week)")
print(f"   Pre-period: 2008-2011")
print(f"   Post-period: 2013-2016")
print(f"\n   EFFECT SIZE: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"   This represents a {model4.params['ELIGIBLE_AFTER']*100:.2f} percentage point increase")
print(f"   in the probability of full-time employment")
print(f"\n   Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% Confidence Interval: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   Sample Size: {int(model4.nobs):,}")
print("="*80)

# =============================================================================
# 13. EXPORT RESULTS FOR LATEX
# =============================================================================

# Save key statistics to a file for LaTeX
results_dict = {
    'n_total': int(len(df)),
    'n_treatment': int(eligible_counts[1]),
    'n_control': int(eligible_counts[0]),
    'n_pre': int(after_counts[0]),
    'n_post': int(after_counts[1]),
    'ft_control_pre': float(control_pre),
    'ft_control_post': float(control_post),
    'ft_treat_pre': float(treat_pre),
    'ft_treat_post': float(treat_post),
    'did_simple': float(did_simple),
    'preferred_coef': float(model4.params['ELIGIBLE_AFTER']),
    'preferred_se': float(model4.bse['ELIGIBLE_AFTER']),
    'preferred_ci_lo': float(model4.conf_int().loc['ELIGIBLE_AFTER', 0]),
    'preferred_ci_hi': float(model4.conf_int().loc['ELIGIBLE_AFTER', 1]),
    'preferred_pval': float(model4.pvalues['ELIGIBLE_AFTER']),
    'preferred_n': int(model4.nobs),
    'male_coef': float(model_male.params['ELIGIBLE_AFTER']),
    'male_se': float(model_male.bse['ELIGIBLE_AFTER']),
    'female_coef': float(model_female.params['ELIGIBLE_AFTER']),
    'female_se': float(model_female.bse['ELIGIBLE_AFTER']),
}

# Save results
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\nResults saved to analysis_results.json")

# =============================================================================
# 14. DETAILED REGRESSION OUTPUT
# =============================================================================

print(f"\n14. DETAILED OUTPUT: PREFERRED MODEL")
print("="*80)
print(model4.summary())

# Full model output for appendix
print(f"\n15. FULL MODEL WITH ALL CONTROLS (Model 5)")
print("="*80)
print(model5.summary())
