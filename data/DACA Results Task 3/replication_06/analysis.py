"""
DACA Replication Analysis - Replication 06
Effect of DACA eligibility on full-time employment
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"Total observations: {len(df)}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"ELIGIBLE values: {df['ELIGIBLE'].unique()}")
print(f"AFTER values: {df['AFTER'].unique()}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Group means
print("\nFull-time employment rates by group and period:")
grouped = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: pd.Series({
        'mean_FT': np.average(x['FT'], weights=x['PERWT']),
        'n_obs': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()
print(grouped)

# Calculate simple DiD
pre_control = grouped[(grouped['ELIGIBLE']==0) & (grouped['AFTER']==0)]['mean_FT'].values[0]
post_control = grouped[(grouped['ELIGIBLE']==0) & (grouped['AFTER']==1)]['mean_FT'].values[0]
pre_treat = grouped[(grouped['ELIGIBLE']==1) & (grouped['AFTER']==0)]['mean_FT'].values[0]
post_treat = grouped[(grouped['ELIGIBLE']==1) & (grouped['AFTER']==1)]['mean_FT'].values[0]

did_simple = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\nSimple DiD calculation:")
print(f"  Pre-period treated (ELIGIBLE=1, AFTER=0): {pre_treat:.4f}")
print(f"  Post-period treated (ELIGIBLE=1, AFTER=1): {post_treat:.4f}")
print(f"  Pre-period control (ELIGIBLE=0, AFTER=0): {pre_control:.4f}")
print(f"  Post-period control (ELIGIBLE=0, AFTER=1): {post_control:.4f}")
print(f"  Change in treated: {post_treat - pre_treat:.4f}")
print(f"  Change in control: {post_control - pre_control:.4f}")
print(f"  DiD estimate: {did_simple:.4f}")

# Demographic summary
print("\n" + "="*60)
print("SAMPLE CHARACTERISTICS")
print("="*60)

print(f"\nSample size by group:")
print(f"  Treatment group (ELIGIBLE=1): {len(df[df['ELIGIBLE']==1])}")
print(f"  Control group (ELIGIBLE=0): {len(df[df['ELIGIBLE']==0])}")
print(f"  Pre-period (AFTER=0): {len(df[df['AFTER']==0])}")
print(f"  Post-period (AFTER=1): {len(df[df['AFTER']==1])}")

# Sex distribution
print(f"\nSex distribution (1=Male, 2=Female):")
print(df.groupby(['ELIGIBLE', 'SEX']).size().unstack(fill_value=0))

# Education distribution
print(f"\nEducation distribution by eligibility:")
print(df.groupby(['ELIGIBLE', 'EDUC_RECODE']).size().unstack(fill_value=0))

# Age distribution
print(f"\nAge in June 2012 distribution:")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe())

print("\n" + "="*60)
print("REGRESSION ANALYSIS")
print("="*60)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (unweighted) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD coefficient (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"N: {int(model1.nobs)}")

# Model 2: Basic DiD (weighted)
print("\n--- Model 2: Basic DiD (weighted by PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])
print(f"\nDiD coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 3: DiD with demographic controls (weighted)
print("\n--- Model 3: DiD with demographic controls (weighted) ---")
# Create age variable centered
df['AGE_CENTERED'] = df['AGE'] - df['AGE'].mean()
df['AGE_CENTERED_SQ'] = df['AGE_CENTERED']**2

# Create education dummies - handle True/False and NaN values
df['HS_DEGREE_NUM'] = df['HS_DEGREE'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0).astype(int)

model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(SEX) + AGE_CENTERED + AGE_CENTERED_SQ + C(MARST) + HS_DEGREE_NUM',
                  data=df, weights=df['PERWT']).fit()
print(model3.summary().tables[1])
print(f"\nDiD coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 4: DiD with state fixed effects (weighted)
print("\n--- Model 4: DiD with state fixed effects (weighted) ---")
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)',
                  data=df, weights=df['PERWT']).fit()
print(f"DiD coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 5: Full model with demographic controls and state FE (weighted)
print("\n--- Model 5: Full model (demographics + state FE, weighted) ---")
model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(SEX) + AGE_CENTERED + AGE_CENTERED_SQ + C(MARST) + HS_DEGREE_NUM + C(STATEFIP)',
                  data=df, weights=df['PERWT']).fit()
print(f"DiD coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")

# Year-by-year event study
print("\n" + "="*60)
print("EVENT STUDY ANALYSIS")
print("="*60)

# Create year dummies interacted with ELIGIBLE
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions (2011 as reference year)
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

event_formula = 'FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_YEAR_2008 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 + ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016'
event_model = smf.wls(event_formula, data=df, weights=df['PERWT']).fit()

print("\nEvent study coefficients (relative to 2011):")
event_years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_results = []
for year in event_years:
    coef = event_model.params[f'ELIGIBLE_YEAR_{year}']
    se = event_model.bse[f'ELIGIBLE_YEAR_{year}']
    ci_low = event_model.conf_int().loc[f'ELIGIBLE_YEAR_{year}', 0]
    ci_high = event_model.conf_int().loc[f'ELIGIBLE_YEAR_{year}', 1]
    pval = event_model.pvalues[f'ELIGIBLE_YEAR_{year}']
    event_results.append({'Year': year, 'Coefficient': coef, 'SE': se, 'CI_Low': ci_low, 'CI_High': ci_high, 'p_value': pval})
    print(f"  Year {year}: {coef:.4f} (SE: {se:.4f}), 95% CI: [{ci_low:.4f}, {ci_high:.4f}], p={pval:.4f}")

event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

# Subgroup analysis by sex
print("\n" + "="*60)
print("SUBGROUP ANALYSIS BY SEX")
print("="*60)

for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub, weights=df_sub['PERWT']).fit()
    print(f"\n{sex_name}:")
    print(f"  DiD coefficient: {model_sub.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  Standard error: {model_sub.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"  95% CI: [{model_sub.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_sub.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
    print(f"  p-value: {model_sub.pvalues['ELIGIBLE_AFTER']:.4f}")
    print(f"  N: {len(df_sub)}")

# Parallel trends test
print("\n" + "="*60)
print("PARALLEL TRENDS TEST (Pre-period only)")
print("="*60)

df_pre = df[df['AFTER'] == 0].copy()
df_pre['TIME_TREND'] = df_pre['YEAR'] - 2008

# Test if trends differ pre-treatment
df_pre['ELIGIBLE_TREND'] = df_pre['ELIGIBLE'] * df_pre['TIME_TREND']
trend_model = smf.wls('FT ~ ELIGIBLE + TIME_TREND + ELIGIBLE_TREND', data=df_pre, weights=df_pre['PERWT']).fit()

print(f"Differential pre-trend coefficient (ELIGIBLE_TREND): {trend_model.params['ELIGIBLE_TREND']:.4f}")
print(f"Standard error: {trend_model.bse['ELIGIBLE_TREND']:.4f}")
print(f"p-value: {trend_model.pvalues['ELIGIBLE_TREND']:.4f}")
print(f"Interpretation: A non-significant coefficient supports the parallel trends assumption")

# Save key results for LaTeX
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_summary = {
    'Model': ['Basic DiD (unweighted)', 'Basic DiD (weighted)', 'With demographics (weighted)',
              'With state FE (weighted)', 'Full model (weighted)'],
    'Coefficient': [model1.params['ELIGIBLE_AFTER'], model2.params['ELIGIBLE_AFTER'],
                    model3.params['ELIGIBLE_AFTER'], model4.params['ELIGIBLE_AFTER'],
                    model5.params['ELIGIBLE_AFTER']],
    'SE': [model1.bse['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'],
           model3.bse['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'],
           model5.bse['ELIGIBLE_AFTER']],
    'CI_Low': [model1.conf_int().loc['ELIGIBLE_AFTER', 0], model2.conf_int().loc['ELIGIBLE_AFTER', 0],
               model3.conf_int().loc['ELIGIBLE_AFTER', 0], model4.conf_int().loc['ELIGIBLE_AFTER', 0],
               model5.conf_int().loc['ELIGIBLE_AFTER', 0]],
    'CI_High': [model1.conf_int().loc['ELIGIBLE_AFTER', 1], model2.conf_int().loc['ELIGIBLE_AFTER', 1],
                model3.conf_int().loc['ELIGIBLE_AFTER', 1], model4.conf_int().loc['ELIGIBLE_AFTER', 1],
                model5.conf_int().loc['ELIGIBLE_AFTER', 1]],
    'p_value': [model1.pvalues['ELIGIBLE_AFTER'], model2.pvalues['ELIGIBLE_AFTER'],
                model3.pvalues['ELIGIBLE_AFTER'], model4.pvalues['ELIGIBLE_AFTER'],
                model5.pvalues['ELIGIBLE_AFTER']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('regression_results.csv', index=False)
print("Results saved to regression_results.csv")

# Summary statistics for tables
summary_stats = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['mean', 'std', 'count'],
    'PERWT': 'sum',
    'AGE': 'mean',
    'SEX': lambda x: (x == 2).mean(),  # Proportion female
}).reset_index()
summary_stats.columns = ['ELIGIBLE', 'AFTER', 'FT_mean', 'FT_std', 'N', 'Weighted_N', 'Mean_Age', 'Prop_Female']
summary_stats.to_csv('summary_statistics.csv', index=False)
print("Summary statistics saved to summary_statistics.csv")

# Year-by-year means for plotting
yearly_means = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'FT_mean': np.average(x['FT'], weights=x['PERWT']),
        'N': len(x)
    })
).reset_index()
yearly_means.to_csv('yearly_means.csv', index=False)
print("Yearly means saved to yearly_means.csv")

print("\n" + "="*60)
print("PREFERRED ESTIMATE")
print("="*60)
print(f"\nPreferred model: Basic DiD (weighted by PERWT)")
print(f"Effect size: {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"Sample size: {int(model2.nobs)}")
print(f"p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

print("\nAnalysis complete!")
