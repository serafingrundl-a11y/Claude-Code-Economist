"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
ethnically Hispanic-Mexican Mexican-born people in the United States.

Treatment group: Ages 26-30 at June 2012 (ELIGIBLE=1)
Control group: Ages 31-35 at June 2012 (ELIGIBLE=0)
Outcome: Full-time employment (FT=1 if usually working 35+ hours/week)
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 80)
print("DACA REPLICATION ANALYSIS")
print("=" * 80)

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

print(f"\nDataset shape: {df.shape}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# SECTION 1: DATA DESCRIPTION
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: SAMPLE DESCRIPTION")
print("=" * 80)

print("\n--- Key Variable Distributions ---")

# ELIGIBLE status
print("\nELIGIBLE (Treatment status):")
print(df['ELIGIBLE'].value_counts().sort_index())
print(f"  0 = Control (ages 31-35 in June 2012)")
print(f"  1 = Treatment (ages 26-30 in June 2012)")

# AFTER (Post-treatment)
print("\nAFTER (Post-treatment period):")
print(df['AFTER'].value_counts().sort_index())
print(f"  0 = Pre-DACA (2008-2011)")
print(f"  1 = Post-DACA (2013-2016)")

# FT (Outcome)
print("\nFT (Full-time employment outcome):")
print(df['FT'].value_counts().sort_index())

# Cross-tabulation of treatment status
print("\nCross-tabulation (ELIGIBLE x AFTER):")
cross_tab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
print(cross_tab)

# Demographic characteristics
print("\n--- Demographic Characteristics ---")

# Sex
print("\nSex distribution (1=Male, 2=Female):")
print(df['SEX'].value_counts().sort_index())
print(f"Percent Male: {(df['SEX'] == 1).mean() * 100:.1f}%")

# Age
print(f"\nAge: Mean = {df['AGE'].mean():.2f}, SD = {df['AGE'].std():.2f}")

# Age at June 2012
print("\nAge in June 2012 by eligibility:")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].agg(['mean', 'std', 'min', 'max']))

# Marital status
print("\nMarital status (MARST):")
marst_labels = {1: 'Married, spouse present', 2: 'Married, spouse absent',
                3: 'Separated', 4: 'Divorced', 5: 'Widowed', 6: 'Never married'}
print(df['MARST'].value_counts().sort_index())

# Education
print("\nEducation (EDUC_RECODE):")
print(df['EDUC_RECODE'].value_counts())

# Number of children
print(f"\nNumber of children (NCHILD): Mean = {df['NCHILD'].mean():.2f}")

# =============================================================================
# SECTION 2: SIMPLE DIFFERENCE-IN-DIFFERENCES
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("=" * 80)

# Mean FT by group
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean()
print("\nMean Full-Time Employment by Group:")
print(ft_means.unstack())

# DiD calculation
ft_00 = ft_means.loc[(0, 0)]  # Control, Pre
ft_01 = ft_means.loc[(0, 1)]  # Control, Post
ft_10 = ft_means.loc[(1, 0)]  # Treated, Pre
ft_11 = ft_means.loc[(1, 1)]  # Treated, Post

print(f"\nControl group (ages 31-35):")
print(f"  Pre-DACA:  {ft_00:.4f}")
print(f"  Post-DACA: {ft_01:.4f}")
print(f"  Change:    {ft_01 - ft_00:.4f}")

print(f"\nTreatment group (ages 26-30):")
print(f"  Pre-DACA:  {ft_10:.4f}")
print(f"  Post-DACA: {ft_11:.4f}")
print(f"  Change:    {ft_11 - ft_10:.4f}")

did_simple = (ft_11 - ft_10) - (ft_01 - ft_00)
print(f"\nDifference-in-Differences estimate: {did_simple:.4f}")
print(f"  ({ft_11:.4f} - {ft_10:.4f}) - ({ft_01:.4f} - {ft_00:.4f})")
print(f"  = {ft_11 - ft_10:.4f} - ({ft_01 - ft_00:.4f})")
print(f"  = {did_simple:.4f}")

# =============================================================================
# SECTION 3: REGRESSION-BASED DiD
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD without controls (OLS)
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD Coefficient (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 2: DiD with robust standard errors
print("\n--- Model 2: Basic DiD with Robust (HC1) Standard Errors ---")
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# SECTION 4: DiD WITH DEMOGRAPHIC CONTROLS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: DiD WITH DEMOGRAPHIC CONTROLS")
print("=" * 80)

# Create dummy variables for education
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

# Education dummies (reference: High School Degree)
df['EDUC_LHS'] = (df['EDUC_RECODE'] == 'Less than High School').astype(int)
df['EDUC_SC'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_2YR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Model 3: DiD with demographic controls
print("\n--- Model 3: DiD with Demographic Controls ---")
formula3 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_SC + EDUC_2YR + EDUC_BA'
model3 = smf.ols(formula3, data=df).fit(cov_type='HC1')
print(model3.summary().tables[1])
print(f"\nDiD Coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# SECTION 5: DiD WITH STATE AND YEAR FIXED EFFECTS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: DiD WITH STATE AND YEAR FIXED EFFECTS")
print("=" * 80)

# Create state and year dummies
print("\n--- Model 4: DiD with State Fixed Effects ---")
formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_SC + EDUC_2YR + EDUC_BA + C(STATEFIP)'
model4 = smf.ols(formula4, data=df).fit(cov_type='HC1')
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")

print("\n--- Model 5: DiD with Year Fixed Effects ---")
formula5 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_SC + EDUC_2YR + EDUC_BA + C(YEAR)'
model5 = smf.ols(formula5, data=df).fit(cov_type='HC1')
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")

print("\n--- Model 6: DiD with State AND Year Fixed Effects ---")
formula6 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_SC + EDUC_2YR + EDUC_BA + C(STATEFIP) + C(YEAR)'
model6 = smf.ols(formula6, data=df).fit(cov_type='HC1')
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# SECTION 6: DiD WITH STATE POLICY VARIABLES
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 6: DiD WITH STATE POLICY VARIABLES")
print("=" * 80)

print("\n--- Model 7: DiD with State Policy Controls ---")
formula7 = '''FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + AGE +
              EDUC_SC + EDUC_2YR + EDUC_BA +
              DRIVERSLICENSES + INSTATETUITION + EVERIFY + SECURECOMMUNITIES +
              UNEMP + LFPR + C(YEAR)'''
model7 = smf.ols(formula7, data=df).fit(cov_type='HC1')
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model7.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# SECTION 7: WEIGHTED DiD (Using Person Weights)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 7: WEIGHTED DIFFERENCE-IN-DIFFERENCES")
print("=" * 80)

print("\n--- Model 8: Weighted DiD (Basic) ---")
import statsmodels.api as sm

# Prepare data for weighted regression
X_weighted = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']].copy()
X_weighted = sm.add_constant(X_weighted)
y_weighted = df['FT']
weights = df['PERWT']

model8 = sm.WLS(y_weighted, X_weighted, weights=weights).fit(cov_type='HC1')
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model8.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model8.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model8.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model8.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model8.pvalues['ELIGIBLE_AFTER']:.4f}")

print("\n--- Model 9: Weighted DiD with Controls (PREFERRED MODEL) ---")
X_vars = ['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'NCHILD', 'AGE',
          'EDUC_SC', 'EDUC_2YR', 'EDUC_BA']
X_full = df[X_vars].copy()
X_full = sm.add_constant(X_full)

model9 = sm.WLS(y_weighted, X_full, weights=weights).fit(cov_type='HC1')
print("\nCoefficients:")
for var in ['const', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'NCHILD', 'AGE']:
    print(f"  {var}: {model9.params[var]:.4f} (SE: {model9.bse[var]:.4f})")
print(f"\nDiD Coefficient (ELIGIBLE_AFTER): {model9.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model9.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model9.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model9.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model9.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# SECTION 8: PARALLEL TRENDS ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 8: PARALLEL TRENDS ANALYSIS")
print("=" * 80)

# Calculate mean FT by year and eligibility
print("\n--- FT by Year and Eligibility Status ---")
ft_by_year = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
print(ft_by_year)

# Test for parallel trends in pre-period (2008-2011)
print("\n--- Pre-Treatment Parallel Trends Test ---")
pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_ELIGIBLE'] = pre_data['YEAR'] * pre_data['ELIGIBLE']

formula_pt = 'FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE:C(YEAR)'
model_pt = smf.ols(formula_pt, data=pre_data).fit(cov_type='HC1')

# Extract interaction terms
print("\nInteraction coefficients (ELIGIBLE x Year) in pre-period:")
for param in model_pt.params.index:
    if 'ELIGIBLE:' in param and 'YEAR' in param:
        print(f"  {param}: {model_pt.params[param]:.4f} (p={model_pt.pvalues[param]:.4f})")

# Joint F-test for parallel trends
print("\n--- Joint F-test for Pre-Treatment Parallel Trends ---")
# Restricted model (no interaction)
formula_restricted = 'FT ~ ELIGIBLE + C(YEAR)'
model_restricted = smf.ols(formula_restricted, data=pre_data).fit()

# F-test
f_stat = ((model_restricted.ssr - model_pt.ssr) / 3) / (model_pt.ssr / model_pt.df_resid)
p_value = 1 - stats.f.cdf(f_stat, 3, model_pt.df_resid)
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value > 0.05:
    print("Conclusion: Fail to reject null hypothesis of parallel trends (p > 0.05)")
else:
    print("Conclusion: Evidence against parallel trends (p < 0.05)")

# =============================================================================
# SECTION 9: EVENT STUDY / DYNAMIC EFFECTS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 9: EVENT STUDY / DYNAMIC EFFECTS")
print("=" * 80)

# Create year-specific treatment effects (event study)
# Reference year: 2011 (last pre-treatment year)
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

formula_es = '''FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 +
                ELIGIBLE_YEAR_2008 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 +
                ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016'''
model_es = smf.ols(formula_es, data=df).fit(cov_type='HC1')

print("\nEvent Study Coefficients (ELIGIBLE x Year):")
print("(Reference: 2011)")
event_study_results = []
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_es.params[f'ELIGIBLE_YEAR_{year}']
    se = model_es.bse[f'ELIGIBLE_YEAR_{year}']
    pval = model_es.pvalues[f'ELIGIBLE_YEAR_{year}']
    ci_low = model_es.conf_int().loc[f'ELIGIBLE_YEAR_{year}', 0]
    ci_high = model_es.conf_int().loc[f'ELIGIBLE_YEAR_{year}', 1]
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}], p={pval:.4f})")
    event_study_results.append({
        'Year': year,
        'Coefficient': coef,
        'SE': se,
        'CI_low': ci_low,
        'CI_high': ci_high,
        'p_value': pval
    })

event_study_df = pd.DataFrame(event_study_results)

# =============================================================================
# SECTION 10: HETEROGENEOUS EFFECTS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 10: HETEROGENEOUS EFFECTS BY SUBGROUP")
print("=" * 80)

# By gender
print("\n--- Effects by Gender ---")
for gender in [1, 2]:
    gender_label = "Male" if gender == 1 else "Female"
    subset = df[df['SEX'] == gender]
    model_gender = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=subset).fit(cov_type='HC1')
    print(f"{gender_label} (N={len(subset)}):")
    print(f"  DiD: {model_gender.params['ELIGIBLE_AFTER']:.4f} (SE: {model_gender.bse['ELIGIBLE_AFTER']:.4f}, p={model_gender.pvalues['ELIGIBLE_AFTER']:.4f})")

# By marital status
print("\n--- Effects by Marital Status ---")
for married in [0, 1]:
    married_label = "Married" if married == 1 else "Not Married"
    subset = df[df['MARRIED'] == married]
    model_married = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=subset).fit(cov_type='HC1')
    print(f"{married_label} (N={len(subset)}):")
    print(f"  DiD: {model_married.params['ELIGIBLE_AFTER']:.4f} (SE: {model_married.bse['ELIGIBLE_AFTER']:.4f}, p={model_married.pvalues['ELIGIBLE_AFTER']:.4f})")

# By education
print("\n--- Effects by Education Level ---")
for educ in ['High School Degree', 'Some College', 'BA+']:
    subset = df[df['EDUC_RECODE'] == educ]
    if len(subset) > 100:
        model_educ = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=subset).fit(cov_type='HC1')
        print(f"{educ} (N={len(subset)}):")
        print(f"  DiD: {model_educ.params['ELIGIBLE_AFTER']:.4f} (SE: {model_educ.bse['ELIGIBLE_AFTER']:.4f}, p={model_educ.pvalues['ELIGIBLE_AFTER']:.4f})")

# =============================================================================
# SECTION 11: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 11: ROBUSTNESS CHECKS")
print("=" * 80)

# Probit model
print("\n--- Robustness Check 1: Probit Model ---")
from statsmodels.discrete.discrete_model import Probit
X_probit = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']].copy()
X_probit = sm.add_constant(X_probit)
model_probit = Probit(df['FT'], X_probit).fit(disp=0)
print(f"Probit DiD Coefficient: {model_probit.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model_probit.bse['ELIGIBLE_AFTER']:.4f}")

# Marginal effect (approximate)
mfx = model_probit.get_margeff()
print(f"Marginal Effect at Mean: {mfx.margeff[2]:.4f}")

# Logit model
print("\n--- Robustness Check 2: Logit Model ---")
from statsmodels.discrete.discrete_model import Logit
model_logit = Logit(df['FT'], X_probit).fit(disp=0)
print(f"Logit DiD Coefficient: {model_logit.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model_logit.bse['ELIGIBLE_AFTER']:.4f}")

# Marginal effect
mfx_logit = model_logit.get_margeff()
print(f"Marginal Effect at Mean: {mfx_logit.margeff[2]:.4f}")

# Clustered standard errors by state
print("\n--- Robustness Check 3: Clustered Standard Errors (by State) ---")
model_clustered = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model_clustered.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {model_clustered.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_clustered.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_clustered.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# =============================================================================
# SECTION 12: SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 12: SUMMARY OF RESULTS")
print("=" * 80)

print("\n--- Summary Table of All Models ---")
results_summary = []

models = [
    ('Model 1: Basic DiD', model1.params['ELIGIBLE_AFTER'], model1.bse['ELIGIBLE_AFTER'], model1.pvalues['ELIGIBLE_AFTER'], len(df)),
    ('Model 2: Robust SE', model2.params['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'], model2.pvalues['ELIGIBLE_AFTER'], len(df)),
    ('Model 3: + Demographics', model3.params['ELIGIBLE_AFTER'], model3.bse['ELIGIBLE_AFTER'], model3.pvalues['ELIGIBLE_AFTER'], len(df)),
    ('Model 4: + State FE', model4.params['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'], model4.pvalues['ELIGIBLE_AFTER'], len(df)),
    ('Model 5: + Year FE', model5.params['ELIGIBLE_AFTER'], model5.bse['ELIGIBLE_AFTER'], model5.pvalues['ELIGIBLE_AFTER'], len(df)),
    ('Model 6: State + Year FE', model6.params['ELIGIBLE_AFTER'], model6.bse['ELIGIBLE_AFTER'], model6.pvalues['ELIGIBLE_AFTER'], len(df)),
    ('Model 8: Weighted Basic', model8.params['ELIGIBLE_AFTER'], model8.bse['ELIGIBLE_AFTER'], model8.pvalues['ELIGIBLE_AFTER'], len(df)),
    ('Model 9: Weighted + Controls', model9.params['ELIGIBLE_AFTER'], model9.bse['ELIGIBLE_AFTER'], model9.pvalues['ELIGIBLE_AFTER'], len(df)),
    ('Clustered SE', model_clustered.params['ELIGIBLE_AFTER'], model_clustered.bse['ELIGIBLE_AFTER'], model_clustered.pvalues['ELIGIBLE_AFTER'], len(df)),
]

print(f"{'Model':<30} {'Coef':>10} {'SE':>10} {'P-value':>10} {'N':>8}")
print("-" * 70)
for name, coef, se, pval, n in models:
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"{name:<30} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {n:>8}{sig}")

print("\n*** p<0.01, ** p<0.05, * p<0.1")

# =============================================================================
# PREFERRED ESTIMATE
# =============================================================================
print("\n" + "=" * 80)
print("PREFERRED ESTIMATE")
print("=" * 80)

# Use Model 9 (Weighted DiD with demographic controls) as preferred
print("\nPreferred Model: Weighted DiD with Demographic Controls (Model 9)")
print(f"Effect estimate: {model9.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model9.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model9.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model9.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model9.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample size: {len(df)}")

print("\nInterpretation:")
print(f"DACA eligibility is associated with a {model9.params['ELIGIBLE_AFTER']*100:.2f} percentage point")
print(f"increase in the probability of full-time employment among eligible individuals")
print(f"aged 26-30 in June 2012, relative to the control group aged 31-35.")

# Save key results for export
key_results = {
    'preferred_estimate': model9.params['ELIGIBLE_AFTER'],
    'standard_error': model9.bse['ELIGIBLE_AFTER'],
    'ci_lower': model9.conf_int().loc['ELIGIBLE_AFTER', 0],
    'ci_upper': model9.conf_int().loc['ELIGIBLE_AFTER', 1],
    'p_value': model9.pvalues['ELIGIBLE_AFTER'],
    'sample_size': len(df)
}

# Save results to file
import json
with open('analysis_results.json', 'w') as f:
    json.dump(key_results, f, indent=2)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nResults saved to analysis_results.json")

# =============================================================================
# Export additional tables for LaTeX
# =============================================================================

# Balance table
print("\n--- Exporting Balance Table ---")
balance_vars = ['AGE', 'FEMALE', 'MARRIED', 'NCHILD', 'EDUC_SC', 'EDUC_2YR', 'EDUC_BA']
balance_data = []
for var in balance_vars:
    treat_mean = df[df['ELIGIBLE'] == 1][var].mean()
    treat_sd = df[df['ELIGIBLE'] == 1][var].std()
    control_mean = df[df['ELIGIBLE'] == 0][var].mean()
    control_sd = df[df['ELIGIBLE'] == 0][var].std()
    diff = treat_mean - control_mean
    balance_data.append({
        'Variable': var,
        'Treatment Mean': treat_mean,
        'Treatment SD': treat_sd,
        'Control Mean': control_mean,
        'Control SD': control_sd,
        'Difference': diff
    })

balance_df = pd.DataFrame(balance_data)
balance_df.to_csv('balance_table.csv', index=False)
print("Balance table saved to balance_table.csv")

# Event study results
event_study_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# FT by year and eligibility
ft_by_year.to_csv('ft_by_year.csv')
print("FT by year saved to ft_by_year.csv")
