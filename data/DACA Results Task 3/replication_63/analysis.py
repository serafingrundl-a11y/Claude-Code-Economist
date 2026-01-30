"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Load Data
# =============================================================================
print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\n1. DATA LOADED")
print(f"   Total observations: {len(df):,}")
print(f"   Variables: {len(df.columns)}")

# =============================================================================
# 2. Data Summary
# =============================================================================
print("\n" + "="*80)
print("2. DATA SUMMARY")
print("="*80)

# Year distribution
print("\nObservations by Year:")
year_counts = df.groupby('YEAR').size()
for year, count in year_counts.items():
    print(f"   {year}: {count:,}")

# Treatment/Control groups
print("\nObservations by ELIGIBLE status:")
print(f"   Treatment (ELIGIBLE=1, ages 26-30): {(df['ELIGIBLE']==1).sum():,}")
print(f"   Control (ELIGIBLE=0, ages 31-35): {(df['ELIGIBLE']==0).sum():,}")

# Before/After
print("\nObservations by AFTER status:")
print(f"   Pre-DACA (AFTER=0, 2008-2011): {(df['AFTER']==0).sum():,}")
print(f"   Post-DACA (AFTER=1, 2013-2016): {(df['AFTER']==1).sum():,}")

# Full-time employment
print("\nFull-time Employment (FT):")
print(f"   FT=0 (Not full-time): {(df['FT']==0).sum():,}")
print(f"   FT=1 (Full-time): {(df['FT']==1).sum():,}")
print(f"   Overall FT rate: {df['FT'].mean():.4f}")

# 2x2 Table
print("\n" + "-"*60)
print("2x2 DiD Table - Full-time Employment Rates (Unweighted)")
print("-"*60)

did_table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'sum'])
print(did_table)

# Calculate simple DiD
ft_pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
ft_post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
ft_pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
ft_post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"\n   Treatment group (26-30) pre-DACA:  {ft_pre_treat:.4f}")
print(f"   Treatment group (26-30) post-DACA: {ft_post_treat:.4f}")
print(f"   Control group (31-35) pre-DACA:    {ft_pre_control:.4f}")
print(f"   Control group (31-35) post-DACA:   {ft_post_control:.4f}")

simple_did = (ft_post_treat - ft_pre_treat) - (ft_post_control - ft_pre_control)
print(f"\n   Simple DiD estimate (unweighted): {simple_did:.4f}")

# =============================================================================
# 3. Weighted 2x2 DiD
# =============================================================================
print("\n" + "-"*60)
print("2x2 DiD Table - Full-time Employment Rates (Weighted)")
print("-"*60)

def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

ft_pre_treat_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)])
ft_post_treat_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)])
ft_pre_control_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)])
ft_post_control_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)])

print(f"   Treatment group (26-30) pre-DACA:  {ft_pre_treat_w:.4f}")
print(f"   Treatment group (26-30) post-DACA: {ft_post_treat_w:.4f}")
print(f"   Control group (31-35) pre-DACA:    {ft_pre_control_w:.4f}")
print(f"   Control group (31-35) post-DACA:   {ft_post_control_w:.4f}")

weighted_did = (ft_post_treat_w - ft_pre_treat_w) - (ft_post_control_w - ft_pre_control_w)
print(f"\n   Simple DiD estimate (weighted): {weighted_did:.4f}")

# =============================================================================
# 4. Regression Analysis - Primary Specification
# =============================================================================
print("\n" + "="*80)
print("3. REGRESSION ANALYSIS")
print("="*80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (Unweighted, No Clustering) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])

# Model 2: Weighted DiD
print("\n--- Model 2: Weighted DiD (Survey Weights) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: Weighted DiD with clustered SEs at state level
print("\n--- Model 3: Weighted DiD with State-Clustered SEs ---")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model3.summary().tables[1])

# =============================================================================
# 5. Robustness Checks
# =============================================================================
print("\n" + "="*80)
print("4. ROBUSTNESS CHECKS")
print("="*80)

# Model 4: With demographic controls
print("\n--- Model 4: With Demographic Controls (SEX, MARST, NCHILD) ---")
# Recode SEX: 1=Male, 2=Female -> create female indicator
df['FEMALE'] = (df['SEX'] == 2).astype(int)
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARST + NCHILD',
                  data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model4.summary().tables[1])

# Model 5: With education controls
print("\n--- Model 5: With Education Controls ---")
# Create education dummies
df['educ_hs'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['educ_somecol'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['educ_twoyear'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['educ_ba'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARST + NCHILD + educ_hs + educ_somecol + educ_twoyear + educ_ba',
                  data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model5.summary().tables[1])

# Model 6: With state fixed effects
print("\n--- Model 6: With State Fixed Effects ---")
model6 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)',
                  data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
# Only print the main coefficients
print("   ELIGIBLE_AFTER coefficient: {:.4f}".format(model6.params['ELIGIBLE_AFTER']))
print("   ELIGIBLE_AFTER std error:   {:.4f}".format(model6.bse['ELIGIBLE_AFTER']))
print("   ELIGIBLE_AFTER p-value:     {:.4f}".format(model6.pvalues['ELIGIBLE_AFTER']))

# Model 7: With year fixed effects
print("\n--- Model 7: With Year Fixed Effects ---")
model7 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE:C(YEAR)',
                  data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
# Print year-specific treatment effects
print("Year-specific treatment effects (ELIGIBLE x YEAR interactions):")
for param in model7.params.index:
    if 'ELIGIBLE:C(YEAR)' in param:
        year = param.split('[T.')[1].split(']')[0]
        coef = model7.params[param]
        se = model7.bse[param]
        print(f"   {year}: {coef:.4f} (SE: {se:.4f})")

# Model 8: Full specification (state FE + year FE + demographics + education)
print("\n--- Model 8: Full Specification (State FE + Year FE + Controls) ---")
model8 = smf.wls('FT ~ ELIGIBLE_AFTER + FEMALE + MARST + NCHILD + educ_hs + educ_somecol + educ_twoyear + educ_ba + C(STATEFIP) + C(YEAR)',
                  data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print("   ELIGIBLE_AFTER coefficient: {:.4f}".format(model8.params['ELIGIBLE_AFTER']))
print("   ELIGIBLE_AFTER std error:   {:.4f}".format(model8.bse['ELIGIBLE_AFTER']))
print("   ELIGIBLE_AFTER p-value:     {:.4f}".format(model8.pvalues['ELIGIBLE_AFTER']))
ci_low = model8.params['ELIGIBLE_AFTER'] - 1.96 * model8.bse['ELIGIBLE_AFTER']
ci_high = model8.params['ELIGIBLE_AFTER'] + 1.96 * model8.bse['ELIGIBLE_AFTER']
print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# =============================================================================
# 6. Subgroup Analysis
# =============================================================================
print("\n" + "="*80)
print("5. SUBGROUP ANALYSIS")
print("="*80)

# By gender
print("\n--- By Gender ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex]
    model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                    data=subset, weights=subset['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': subset['STATEFIP']})
    print(f"   {label}: DiD = {model.params['ELIGIBLE_AFTER']:.4f} (SE: {model.bse['ELIGIBLE_AFTER']:.4f})")

# By education
print("\n--- By Education Level ---")
for educ in df['EDUC_RECODE'].unique():
    if pd.isna(educ):
        continue
    subset = df[df['EDUC_RECODE'] == educ]
    if len(subset) > 100:
        model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=subset, weights=subset['PERWT']).fit(
            cov_type='cluster', cov_kwds={'groups': subset['STATEFIP']})
        print(f"   {educ}: DiD = {model.params['ELIGIBLE_AFTER']:.4f} (SE: {model.bse['ELIGIBLE_AFTER']:.4f})")

# By marital status
print("\n--- By Marital Status ---")
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)
for married, label in [(1, 'Married'), (0, 'Not Married')]:
    subset = df[df['MARRIED'] == married]
    model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                    data=subset, weights=subset['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': subset['STATEFIP']})
    print(f"   {label}: DiD = {model.params['ELIGIBLE_AFTER']:.4f} (SE: {model.bse['ELIGIBLE_AFTER']:.4f})")

# =============================================================================
# 7. Parallel Trends Check
# =============================================================================
print("\n" + "="*80)
print("6. PARALLEL TRENDS CHECK")
print("="*80)

# Calculate FT rates by year and eligible status
yearly_ft = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
yearly_ft.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\nWeighted Full-time Employment Rates by Year:")
print(yearly_ft.round(4))

# Calculate year-to-year changes for both groups (pre-period)
print("\nYear-to-year changes in FT rates (Pre-DACA period):")
pre_years = [2008, 2009, 2010, 2011]
for i in range(1, len(pre_years)):
    y1, y2 = pre_years[i-1], pre_years[i]
    treat_change = yearly_ft.loc[y2, 'Treatment (26-30)'] - yearly_ft.loc[y1, 'Treatment (26-30)']
    control_change = yearly_ft.loc[y2, 'Control (31-35)'] - yearly_ft.loc[y1, 'Control (31-35)']
    print(f"   {y1}-{y2}: Treatment change = {treat_change:.4f}, Control change = {control_change:.4f}")

# =============================================================================
# 8. Summary of Results
# =============================================================================
print("\n" + "="*80)
print("7. SUMMARY OF MAIN RESULTS")
print("="*80)

print("\n*** PREFERRED SPECIFICATION: Model 3 (Weighted DiD with State-Clustered SEs) ***")
print(f"\n   Treatment Effect (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   t-statistic: {model3.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
ci_low3 = model3.params['ELIGIBLE_AFTER'] - 1.96 * model3.bse['ELIGIBLE_AFTER']
ci_high3 = model3.params['ELIGIBLE_AFTER'] + 1.96 * model3.bse['ELIGIBLE_AFTER']
print(f"   95% Confidence Interval: [{ci_low3:.4f}, {ci_high3:.4f}]")
print(f"   Sample Size: {len(df):,}")

# Save key results for LaTeX
results = {
    'n_obs': len(df),
    'n_treat_pre': len(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]),
    'n_treat_post': len(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]),
    'n_control_pre': len(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]),
    'n_control_post': len(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]),
    'ft_treat_pre': ft_pre_treat_w,
    'ft_treat_post': ft_post_treat_w,
    'ft_control_pre': ft_pre_control_w,
    'ft_control_post': ft_post_control_w,
    'did_simple': weighted_did,
    'did_model3': model3.params['ELIGIBLE_AFTER'],
    'se_model3': model3.bse['ELIGIBLE_AFTER'],
    'pval_model3': model3.pvalues['ELIGIBLE_AFTER'],
    'ci_low_model3': ci_low3,
    'ci_high_model3': ci_high3,
    'did_model8': model8.params['ELIGIBLE_AFTER'],
    'se_model8': model8.bse['ELIGIBLE_AFTER'],
}

# Save results to file
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("ANALYSIS COMPLETE - Results saved to analysis_results.json")
print("="*80)

# =============================================================================
# 9. Generate Tables for LaTeX
# =============================================================================

# Table 1: Sample Descriptive Statistics
print("\n\nLATEX TABLE DATA:")
print("\n--- Table 1: Sample Size by Group and Period ---")
print(f"Treatment Pre: {results['n_treat_pre']:,}")
print(f"Treatment Post: {results['n_treat_post']:,}")
print(f"Control Pre: {results['n_control_pre']:,}")
print(f"Control Post: {results['n_control_post']:,}")

# Descriptive statistics
print("\n--- Variable Means ---")
desc_vars = ['FT', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD', 'UHRSWORK']
for var in desc_vars:
    if var in df.columns:
        mean_val = np.average(df[var].dropna(), weights=df.loc[df[var].notna(), 'PERWT'])
        print(f"{var}: {mean_val:.4f}")

# Education distribution
print("\n--- Education Distribution ---")
educ_dist = df.groupby('EDUC_RECODE').apply(lambda x: x['PERWT'].sum()).sort_values(ascending=False)
educ_dist = educ_dist / educ_dist.sum()
print(educ_dist.round(4))

# State distribution (top 10)
print("\n--- Top 10 States by Population ---")
state_dist = df.groupby('statename').apply(lambda x: x['PERWT'].sum()).sort_values(ascending=False).head(10)
state_dist = state_dist / df['PERWT'].sum()
print(state_dist.round(4))

print("\n\n*** END OF ANALYSIS ***")
