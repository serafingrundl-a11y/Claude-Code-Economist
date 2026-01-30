"""
DACA Replication Analysis
Research Question: Impact of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the US.

Treatment group: Ages 26-30 at time of policy (June 2012)
Control group: Ages 31-35 at time of policy (June 2012)
Outcome: Full-time employment (FT = 1 if usually working 35+ hours/week)
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

df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\n1. DATA OVERVIEW")
print(f"   Total observations: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")
print(f"   Variables: {len(df.columns)}")

# Check key variables
print(f"\n2. KEY VARIABLES")
print(f"   ELIGIBLE distribution:")
print(f"     0 (Control, ages 31-35): {(df['ELIGIBLE']==0).sum():,}")
print(f"     1 (Treated, ages 26-30): {(df['ELIGIBLE']==1).sum():,}")
print(f"   AFTER distribution:")
print(f"     0 (Pre-DACA 2008-2011): {(df['AFTER']==0).sum():,}")
print(f"     1 (Post-DACA 2013-2016): {(df['AFTER']==1).sum():,}")
print(f"   FT (Full-time employment) distribution:")
print(f"     0: {(df['FT']==0).sum():,}")
print(f"     1: {(df['FT']==1).sum():,}")

# Summary statistics by group
print(f"\n3. FULL-TIME EMPLOYMENT RATES BY GROUP")
groups = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
groups.columns = ['Mean FT', 'Std', 'N']
print(groups)

# Calculate simple DiD estimate manually
ft_treated_after = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
ft_treated_before = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
ft_control_after = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
ft_control_before = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()

did_manual = (ft_treated_after - ft_treated_before) - (ft_control_after - ft_control_before)

print(f"\n4. SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION")
print(f"   Treated (26-30) After:  {ft_treated_after:.4f}")
print(f"   Treated (26-30) Before: {ft_treated_before:.4f}")
print(f"   Change in Treated:      {ft_treated_after - ft_treated_before:.4f}")
print(f"   Control (31-35) After:  {ft_control_after:.4f}")
print(f"   Control (31-35) Before: {ft_control_before:.4f}")
print(f"   Change in Control:      {ft_control_after - ft_control_before:.4f}")
print(f"   DiD Estimate:           {did_manual:.4f}")

# DiD Regression - Basic
print(f"\n5. DIFFERENCE-IN-DIFFERENCES REGRESSION (Basic)")
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary())

# DiD Regression - with robust standard errors
print(f"\n6. DiD REGRESSION WITH ROBUST STANDARD ERRORS (HC1)")
model1_robust = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model1_robust.summary())

# Sample sizes for the 2x2 table
print(f"\n7. SAMPLE SIZES BY GROUP")
sample_sizes = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
sample_sizes.index = ['Control (31-35)', 'Treated (26-30)']
sample_sizes.columns = ['Pre-DACA', 'Post-DACA']
print(sample_sizes)

# Year-by-year analysis
print(f"\n8. FULL-TIME EMPLOYMENT BY YEAR AND ELIGIBILITY")
year_analysis = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
year_analysis.columns = ['Control (31-35)', 'Treated (26-30)']
print(year_analysis)

# Sex distribution
print(f"\n9. DEMOGRAPHIC CHARACTERISTICS")
print(f"   SEX (1=Male, 2=Female):")
print(df.groupby('SEX').size())
print(f"   Percent Male: {(df['SEX']==1).mean()*100:.1f}%")

# Age distribution
print(f"\n   AGE distribution:")
print(df['AGE'].describe())

# Education distribution
print(f"\n   Education (EDUC_RECODE):")
print(df['EDUC_RECODE'].value_counts())

# MarST
print(f"\n   Marital Status (MARST):")
print(df['MARST'].value_counts())

# DiD with covariates
print(f"\n10. DiD WITH COVARIATES")

# Create dummy variables for education
df['educ_hs'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['educ_somecol'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['educ_2yr'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['educ_ba'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Create dummy for female
df['female'] = (df['SEX'] == 2).astype(int)

# Create married dummy
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Create children dummy
df['has_children'] = (df['NCHILD'] > 0).astype(int)

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + female + married + has_children + educ_hs + educ_somecol + educ_2yr + educ_ba',
                 data=df).fit(cov_type='HC1')
print(model2.summary())

# DiD with year fixed effects instead of single AFTER dummy
print(f"\n11. DiD WITH YEAR FIXED EFFECTS")
# Create year dummies
for year in df['YEAR'].unique():
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)

# Interaction with each post year
df['ELIG_2013'] = df['ELIGIBLE'] * df['year_2013']
df['ELIG_2014'] = df['ELIGIBLE'] * df['year_2014']
df['ELIG_2015'] = df['ELIGIBLE'] * df['year_2015']
df['ELIG_2016'] = df['ELIGIBLE'] * df['year_2016']

model3 = smf.ols('FT ~ ELIGIBLE + year_2009 + year_2010 + year_2011 + year_2013 + year_2014 + year_2015 + year_2016 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016',
                 data=df).fit(cov_type='HC1')
print(model3.summary())

# DiD with state fixed effects
print(f"\n12. DiD WITH STATE FIXED EFFECTS")
model4 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)', data=df).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Full model with covariates and state FE
print(f"\n13. FULL MODEL: DiD + COVARIATES + STATE FE")
model5 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + female + married + has_children + educ_hs + educ_somecol + educ_2yr + educ_ba + C(STATEFIP)',
                 data=df).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   t-statistic: {model5.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model with clustered standard errors at state level
print(f"\n14. DiD WITH CLUSTERED STANDARD ERRORS (BY STATE)")
model6 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='cluster',
                  cov_kwds={'groups': df['STATEFIP']})
print(model6.summary())

# Full model with clustered SE
print(f"\n15. FULL MODEL WITH CLUSTERED SE (BY STATE)")
model7 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + female + married + has_children + educ_hs + educ_somecol + educ_2yr + educ_ba',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model7.summary())

# Pre-trend test - check if treated and control have similar trends before DACA
print(f"\n16. PRE-TRENDS ANALYSIS")
pre_data = df[df['AFTER'] == 0].copy()
pre_data['year_trend'] = pre_data['YEAR'] - 2008
pre_data['ELIG_trend'] = pre_data['ELIGIBLE'] * pre_data['year_trend']
pretrend_model = smf.ols('FT ~ ELIGIBLE + year_trend + ELIG_trend', data=pre_data).fit(cov_type='HC1')
print(f"   Testing differential pre-trend (ELIGIBLE x year_trend):")
print(f"   Coefficient: {pretrend_model.params['ELIG_trend']:.4f}")
print(f"   Standard Error: {pretrend_model.bse['ELIG_trend']:.4f}")
print(f"   p-value: {pretrend_model.pvalues['ELIG_trend']:.4f}")
if pretrend_model.pvalues['ELIG_trend'] > 0.05:
    print("   -> No evidence of differential pre-trends (p > 0.05)")
else:
    print("   -> Evidence of differential pre-trends (p < 0.05)")

# Event study
print(f"\n17. EVENT STUDY COEFFICIENTS (Pre-trends check)")
# Create pre-period interactions too
df['ELIG_2009'] = df['ELIGIBLE'] * df['year_2009']
df['ELIG_2010'] = df['ELIGIBLE'] * df['year_2010']
df['ELIG_2011'] = df['ELIGIBLE'] * df['year_2011']

event_model = smf.ols('FT ~ ELIGIBLE + year_2009 + year_2010 + year_2011 + year_2013 + year_2014 + year_2015 + year_2016 + ELIG_2009 + ELIG_2010 + ELIG_2011 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016',
                      data=df).fit(cov_type='HC1')
print("   Interaction coefficients (relative to 2008):")
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    coef = event_model.params[f'ELIG_{year}']
    se = event_model.bse[f'ELIG_{year}']
    pval = event_model.pvalues[f'ELIG_{year}']
    print(f"   {year}: {coef:7.4f} (SE: {se:.4f}, p: {pval:.3f})")

# Weighted analysis
print(f"\n18. WEIGHTED DiD ANALYSIS (USING PERWT)")
model_weighted = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model_weighted.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model_weighted.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   p-value: {model_weighted.pvalues['ELIGIBLE_AFTER']:.4f}")

# Subgroup analysis by sex
print(f"\n19. HETEROGENEITY ANALYSIS BY SEX")
male_data = df[df['SEX'] == 1]
female_data = df[df['SEX'] == 2]

model_male = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=male_data).fit(cov_type='HC1')
model_female = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=female_data).fit(cov_type='HC1')

print(f"   Males (N={len(male_data):,}):")
print(f"     DiD coefficient: {model_male.params['ELIGIBLE_AFTER']:.4f} (SE: {model_male.bse['ELIGIBLE_AFTER']:.4f}, p: {model_male.pvalues['ELIGIBLE_AFTER']:.4f})")
print(f"   Females (N={len(female_data):,}):")
print(f"     DiD coefficient: {model_female.params['ELIGIBLE_AFTER']:.4f} (SE: {model_female.bse['ELIGIBLE_AFTER']:.4f}, p: {model_female.pvalues['ELIGIBLE_AFTER']:.4f})")

# Summary table
print(f"\n" + "=" * 80)
print("SUMMARY OF MAIN RESULTS")
print("=" * 80)
print(f"\nPreferred Specification: Basic DiD with Robust SE (HC1)")
print(f"   Sample Size: {len(df):,}")
print(f"   DiD Estimate (ELIGIBLE_AFTER): {model1_robust.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model1_robust.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   t-statistic: {model1_robust.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model1_robust.pvalues['ELIGIBLE_AFTER']:.4f}")
ci = model1_robust.conf_int().loc['ELIGIBLE_AFTER']
print(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

print(f"\nInterpretation:")
if model1_robust.pvalues['ELIGIBLE_AFTER'] < 0.05:
    effect_dir = "increase" if model1_robust.params['ELIGIBLE_AFTER'] > 0 else "decrease"
    print(f"   DACA eligibility is associated with a statistically significant {effect_dir}")
    print(f"   in the probability of full-time employment of {abs(model1_robust.params['ELIGIBLE_AFTER'])*100:.2f} percentage points.")
else:
    print(f"   DACA eligibility is NOT associated with a statistically significant change")
    print(f"   in the probability of full-time employment (p = {model1_robust.pvalues['ELIGIBLE_AFTER']:.3f}).")

# Create output tables for the report
print(f"\n" + "=" * 80)
print("TABLE OUTPUT FOR REPORT")
print("=" * 80)

# Table 1: Summary Statistics
print("\nTABLE 1: Summary Statistics")
print("-" * 60)
summary_stats = df[['FT', 'AGE', 'female', 'married', 'has_children', 'NCHILD']].describe()
print(summary_stats)

# Table 2: 2x2 DiD table
print("\nTABLE 2: Mean Full-Time Employment by Group")
print("-" * 60)
did_table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
print(did_table)

# Table 3: Regression results
print("\nTABLE 3: DiD Regression Results")
print("-" * 60)
results_summary = pd.DataFrame({
    'Model': ['(1) Basic', '(2) With Covariates', '(3) With State FE', '(4) Full Model', '(5) Clustered SE'],
    'DiD Estimate': [model1_robust.params['ELIGIBLE_AFTER'],
                     model2.params['ELIGIBLE_AFTER'],
                     model4.params['ELIGIBLE_AFTER'],
                     model5.params['ELIGIBLE_AFTER'],
                     model6.params['ELIGIBLE_AFTER']],
    'SE': [model1_robust.bse['ELIGIBLE_AFTER'],
           model2.bse['ELIGIBLE_AFTER'],
           model4.bse['ELIGIBLE_AFTER'],
           model5.bse['ELIGIBLE_AFTER'],
           model6.bse['ELIGIBLE_AFTER']],
    'p-value': [model1_robust.pvalues['ELIGIBLE_AFTER'],
                model2.pvalues['ELIGIBLE_AFTER'],
                model4.pvalues['ELIGIBLE_AFTER'],
                model5.pvalues['ELIGIBLE_AFTER'],
                model6.pvalues['ELIGIBLE_AFTER']]
})
print(results_summary.to_string(index=False))

# Save key results for LaTeX
results_dict = {
    'n_total': len(df),
    'n_treated': (df['ELIGIBLE']==1).sum(),
    'n_control': (df['ELIGIBLE']==0).sum(),
    'n_pre': (df['AFTER']==0).sum(),
    'n_post': (df['AFTER']==1).sum(),
    'did_estimate': model1_robust.params['ELIGIBLE_AFTER'],
    'did_se': model1_robust.bse['ELIGIBLE_AFTER'],
    'did_pval': model1_robust.pvalues['ELIGIBLE_AFTER'],
    'did_ci_low': ci[0],
    'did_ci_high': ci[1],
    'ft_treated_after': ft_treated_after,
    'ft_treated_before': ft_treated_before,
    'ft_control_after': ft_control_after,
    'ft_control_before': ft_control_before,
}

print("\n\nKEY RESULTS DICTIONARY:")
for k, v in results_dict.items():
    print(f"  {k}: {v}")

# Export year-by-year data for plotting
year_means = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
year_means.columns = ['Control', 'Treated']
year_means.to_csv('year_means.csv')
print("\nYear means saved to year_means.csv")

# Export regression tables to CSV
results_summary.to_csv('regression_results.csv', index=False)
print("Regression results saved to regression_results.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
