"""
DACA Replication Study - Analysis Script
Research Question: Impact of DACA eligibility on full-time employment
Method: Difference-in-Differences

Author: Anonymous Replicator
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)
print("\n--- STEP 1: Loading Data ---")

# Load the numeric version for analysis
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"Dataset loaded: {len(df):,} observations")
print(f"Columns: {len(df.columns)}")

# ============================================================================
# DATA EXPLORATION
# ============================================================================

print("\n--- STEP 2: Data Exploration ---")

# Check key variables
print("\n[Key Variables Summary]")
print(f"FT (Full-time): {df['FT'].value_counts().to_dict()}")
print(f"AFTER (Post-DACA): {df['AFTER'].value_counts().to_dict()}")
print(f"ELIGIBLE (Treatment): {df['ELIGIBLE'].value_counts().to_dict()}")

# Year distribution
print(f"\nYear distribution:")
print(df['YEAR'].value_counts().sort_index())

# Age at June 2012 distribution
print(f"\nAge at June 2012 distribution:")
print(df['AGE_IN_JUNE_2012'].describe())

# Check ELIGIBLE by AGE_IN_JUNE_2012
print("\nELIGIBLE by AGE_IN_JUNE_2012:")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].agg(['min', 'max', 'mean', 'count']))

# Sex distribution
print("\nSEX distribution (1=Male, 2=Female):")
print(df['SEX'].value_counts().sort_index())

# FT rates by group and period
print("\n[Full-time Employment Rates]")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'sum', 'count'])
ft_rates.columns = ['FT_rate', 'FT_count', 'N']
print(ft_rates)

# Weighted FT rates
print("\n[Weighted Full-time Employment Rates]")
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE'] == eligible) & (df['AFTER'] == after)]
        weighted_mean = np.average(subset['FT'], weights=subset['PERWT'])
        print(f"ELIGIBLE={eligible}, AFTER={after}: FT rate = {weighted_mean:.4f} (N={len(subset):,})")

# ============================================================================
# DIFFERENCE-IN-DIFFERENCES CALCULATION
# ============================================================================

print("\n" + "="*80)
print("--- STEP 3: Simple DiD Calculation ---")
print("="*80)

# Calculate unweighted DiD
ft_00 = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
ft_01 = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
ft_10 = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
ft_11 = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

did_simple = (ft_11 - ft_10) - (ft_01 - ft_00)

print(f"\n[Unweighted 2x2 DiD]")
print(f"Control Pre  (ELIGIBLE=0, AFTER=0): {ft_00:.4f}")
print(f"Control Post (ELIGIBLE=0, AFTER=1): {ft_01:.4f}")
print(f"Treated Pre  (ELIGIBLE=1, AFTER=0): {ft_10:.4f}")
print(f"Treated Post (ELIGIBLE=1, AFTER=1): {ft_11:.4f}")
print(f"\nControl change (Post - Pre): {ft_01 - ft_00:.4f}")
print(f"Treated change (Post - Pre): {ft_11 - ft_10:.4f}")
print(f"\nDiD Estimate: {did_simple:.4f} ({did_simple*100:.2f} percentage points)")

# Calculate weighted DiD
def weighted_mean(data, weight_col='PERWT', value_col='FT'):
    return np.average(data[value_col], weights=data[weight_col])

ft_00_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)])
ft_01_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)])
ft_10_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)])
ft_11_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)])

did_weighted = (ft_11_w - ft_10_w) - (ft_01_w - ft_00_w)

print(f"\n[Weighted 2x2 DiD]")
print(f"Control Pre  (ELIGIBLE=0, AFTER=0): {ft_00_w:.4f}")
print(f"Control Post (ELIGIBLE=0, AFTER=1): {ft_01_w:.4f}")
print(f"Treated Pre  (ELIGIBLE=1, AFTER=0): {ft_10_w:.4f}")
print(f"Treated Post (ELIGIBLE=1, AFTER=1): {ft_11_w:.4f}")
print(f"\nControl change (Post - Pre): {ft_01_w - ft_00_w:.4f}")
print(f"Treated change (Post - Pre): {ft_11_w - ft_10_w:.4f}")
print(f"\nWeighted DiD Estimate: {did_weighted:.4f} ({did_weighted*100:.2f} percentage points)")

# ============================================================================
# REGRESSION ANALYSIS - BASIC DiD
# ============================================================================

print("\n" + "="*80)
print("--- STEP 4: Regression Analysis ---")
print("="*80)

# Model 1: Basic DiD (unweighted)
print("\n[Model 1: Basic DiD - OLS, Unweighted]")
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: Basic DiD (weighted)
print("\n[Model 2: Basic DiD - WLS, Weighted by PERWT]")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with Year Fixed Effects (weighted)
print("\n[Model 3: DiD with Year Fixed Effects - WLS]")
df['YEAR_cat'] = pd.Categorical(df['YEAR'])
model3 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR)', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nKey coefficients (Year FE model):")
print(f"ELIGIBLE:        {model3.params['ELIGIBLE']:.4f} (SE: {model3.bse['ELIGIBLE']:.4f})")
print(f"ELIGIBLE_AFTER:  {model3.params['ELIGIBLE_AFTER']:.4f} (SE: {model3.bse['ELIGIBLE_AFTER']:.4f})")

# Model 4: DiD with State Fixed Effects (weighted)
print("\n[Model 4: DiD with State Fixed Effects - WLS]")
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nKey coefficients (State FE model):")
print(f"ELIGIBLE:        {model4.params['ELIGIBLE']:.4f} (SE: {model4.bse['ELIGIBLE']:.4f})")
print(f"AFTER:           {model4.params['AFTER']:.4f} (SE: {model4.bse['AFTER']:.4f})")
print(f"ELIGIBLE_AFTER:  {model4.params['ELIGIBLE_AFTER']:.4f} (SE: {model4.bse['ELIGIBLE_AFTER']:.4f})")

# Model 5: DiD with Year and State Fixed Effects (weighted)
print("\n[Model 5: DiD with Year + State Fixed Effects - WLS]")
model5 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP)', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nKey coefficients (Two-way FE model):")
print(f"ELIGIBLE:        {model5.params['ELIGIBLE']:.4f} (SE: {model5.bse['ELIGIBLE']:.4f})")
print(f"ELIGIBLE_AFTER:  {model5.params['ELIGIBLE_AFTER']:.4f} (SE: {model5.bse['ELIGIBLE_AFTER']:.4f})")

# ============================================================================
# REGRESSION ANALYSIS - WITH DEMOGRAPHIC CONTROLS
# ============================================================================

print("\n" + "="*80)
print("--- STEP 5: Regression with Demographic Controls ---")
print("="*80)

# Create dummy variables for categorical controls
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Model 6: DiD with demographic controls (weighted)
print("\n[Model 6: DiD with Demographic Controls - WLS]")
# Check what variables are available
demo_vars = ['FEMALE', 'MARST', 'NCHILD', 'FAMSIZE']
print("Adding controls: SEX, MARST, NCHILD, FAMSIZE")
model6 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + C(MARST) + NCHILD + FAMSIZE',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nKey coefficients (with demographic controls):")
print(f"ELIGIBLE:        {model6.params['ELIGIBLE']:.4f} (SE: {model6.bse['ELIGIBLE']:.4f})")
print(f"AFTER:           {model6.params['AFTER']:.4f} (SE: {model6.bse['AFTER']:.4f})")
print(f"ELIGIBLE_AFTER:  {model6.params['ELIGIBLE_AFTER']:.4f} (SE: {model6.bse['ELIGIBLE_AFTER']:.4f})")
print(f"FEMALE:          {model6.params['FEMALE']:.4f} (SE: {model6.bse['FEMALE']:.4f})")

# Model 7: Full model with all controls
print("\n[Model 7: Full Model - Year FE + State FE + Demographics - WLS]")
model7 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + C(MARST) + NCHILD + FAMSIZE + C(YEAR) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nKey coefficients (full model):")
print(f"ELIGIBLE:        {model7.params['ELIGIBLE']:.4f} (SE: {model7.bse['ELIGIBLE']:.4f})")
print(f"ELIGIBLE_AFTER:  {model7.params['ELIGIBLE_AFTER']:.4f} (SE: {model7.bse['ELIGIBLE_AFTER']:.4f})")
print(f"FEMALE:          {model7.params['FEMALE']:.4f} (SE: {model7.bse['FEMALE']:.4f})")

# ============================================================================
# REGRESSION ANALYSIS - WITH EDUCATION
# ============================================================================

print("\n" + "="*80)
print("--- STEP 6: Regression with Education Controls ---")
print("="*80)

# Model 8: Adding education (general)
print("\n[Model 8: Full Model + Education - WLS]")
model8 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + C(MARST) + NCHILD + FAMSIZE + C(EDUC) + C(YEAR) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nKey coefficients (with education):")
print(f"ELIGIBLE:        {model8.params['ELIGIBLE']:.4f} (SE: {model8.bse['ELIGIBLE']:.4f})")
print(f"ELIGIBLE_AFTER:  {model8.params['ELIGIBLE_AFTER']:.4f} (SE: {model8.bse['ELIGIBLE_AFTER']:.4f})")

# ============================================================================
# PREFERRED SPECIFICATION
# ============================================================================

print("\n" + "="*80)
print("--- STEP 7: PREFERRED SPECIFICATION ---")
print("="*80)

# My preferred specification: DiD with two-way fixed effects and demographic controls
print("\n[PREFERRED MODEL: DiD with Year FE, State FE, and Demographic Controls]")
print("Weighted by PERWT, HC1 robust standard errors")

preferred_model = smf.wls(
    'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + C(MARST) + NCHILD + C(YEAR) + C(STATEFIP)',
    data=df,
    weights=df['PERWT']
).fit(cov_type='HC1')

print("\n" + "-"*60)
print("MAIN RESULTS:")
print("-"*60)
did_coef = preferred_model.params['ELIGIBLE_AFTER']
did_se = preferred_model.bse['ELIGIBLE_AFTER']
did_t = preferred_model.tvalues['ELIGIBLE_AFTER']
did_p = preferred_model.pvalues['ELIGIBLE_AFTER']
ci_low = did_coef - 1.96 * did_se
ci_high = did_coef + 1.96 * did_se

print(f"\nDiD Estimate (ELIGIBLE_AFTER coefficient):")
print(f"  Coefficient: {did_coef:.4f}")
print(f"  Std. Error:  {did_se:.4f}")
print(f"  t-statistic: {did_t:.2f}")
print(f"  p-value:     {did_p:.4f}")
print(f"  95% CI:      [{ci_low:.4f}, {ci_high:.4f}]")
print(f"\nInterpretation: DACA eligibility {'increased' if did_coef > 0 else 'decreased'} full-time employment")
print(f"  by {abs(did_coef*100):.2f} percentage points")

if did_p < 0.01:
    sig_level = "statistically significant at the 1% level"
elif did_p < 0.05:
    sig_level = "statistically significant at the 5% level"
elif did_p < 0.10:
    sig_level = "statistically significant at the 10% level"
else:
    sig_level = "not statistically significant at conventional levels"
print(f"  This effect is {sig_level}.")

print(f"\nN = {int(preferred_model.nobs):,}")
print(f"R-squared = {preferred_model.rsquared:.4f}")

# ============================================================================
# ROBUSTNESS CHECKS
# ============================================================================

print("\n" + "="*80)
print("--- STEP 8: Robustness Checks ---")
print("="*80)

# 8a: Linear probability model vs probit (margins)
print("\n[8a: Comparison of Model Specifications]")
print("Main model is linear probability model (OLS/WLS)")
print("Comparing key coefficient across specifications:\n")

results_table = []
results_table.append(("Basic DiD (unweighted)", model1.params['ELIGIBLE_AFTER'], model1.bse['ELIGIBLE_AFTER']))
results_table.append(("Basic DiD (weighted)", model2.params['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER']))
results_table.append(("Year FE only", model3.params['ELIGIBLE_AFTER'], model3.bse['ELIGIBLE_AFTER']))
results_table.append(("State FE only", model4.params['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER']))
results_table.append(("Two-way FE", model5.params['ELIGIBLE_AFTER'], model5.bse['ELIGIBLE_AFTER']))
results_table.append(("Demographics", model6.params['ELIGIBLE_AFTER'], model6.bse['ELIGIBLE_AFTER']))
results_table.append(("Full w/ Educ", model8.params['ELIGIBLE_AFTER'], model8.bse['ELIGIBLE_AFTER']))
results_table.append(("Preferred", preferred_model.params['ELIGIBLE_AFTER'], preferred_model.bse['ELIGIBLE_AFTER']))

print(f"{'Model':<25} {'Coefficient':>12} {'Std. Error':>12} {'95% CI':>24}")
print("-" * 75)
for name, coef, se in results_table:
    ci = f"[{coef-1.96*se:.4f}, {coef+1.96*se:.4f}]"
    print(f"{name:<25} {coef:>12.4f} {se:>12.4f} {ci:>24}")

# 8b: Clustered standard errors by state
print("\n[8b: State-Clustered Standard Errors]")
model_clustered = smf.wls(
    'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + C(MARST) + NCHILD + C(YEAR) + C(STATEFIP)',
    data=df,
    weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"ELIGIBLE_AFTER with HC1 SE:      {preferred_model.params['ELIGIBLE_AFTER']:.4f} ({preferred_model.bse['ELIGIBLE_AFTER']:.4f})")
print(f"ELIGIBLE_AFTER with Cluster SE:  {model_clustered.params['ELIGIBLE_AFTER']:.4f} ({model_clustered.bse['ELIGIBLE_AFTER']:.4f})")

# 8c: By gender subgroups
print("\n[8c: Heterogeneity by Gender]")
model_male = smf.wls(
    'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(MARST) + NCHILD + C(YEAR) + C(STATEFIP)',
    data=df[df['SEX']==1],
    weights=df[df['SEX']==1]['PERWT']
).fit(cov_type='HC1')

model_female = smf.wls(
    'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(MARST) + NCHILD + C(YEAR) + C(STATEFIP)',
    data=df[df['SEX']==2],
    weights=df[df['SEX']==2]['PERWT']
).fit(cov_type='HC1')

print(f"Male subsample:   DACA effect = {model_male.params['ELIGIBLE_AFTER']:.4f} (SE: {model_male.bse['ELIGIBLE_AFTER']:.4f}), N={int(model_male.nobs):,}")
print(f"Female subsample: DACA effect = {model_female.params['ELIGIBLE_AFTER']:.4f} (SE: {model_female.bse['ELIGIBLE_AFTER']:.4f}), N={int(model_female.nobs):,}")

# 8d: Pre-trends test (year-specific effects)
print("\n[8d: Pre-Trends Analysis]")
# Create year-specific treatment interactions
df['ELIGIBLE_2008'] = df['ELIGIBLE'] * (df['YEAR'] == 2008).astype(int)
df['ELIGIBLE_2009'] = df['ELIGIBLE'] * (df['YEAR'] == 2009).astype(int)
df['ELIGIBLE_2010'] = df['ELIGIBLE'] * (df['YEAR'] == 2010).astype(int)
df['ELIGIBLE_2011'] = df['ELIGIBLE'] * (df['YEAR'] == 2011).astype(int)
# 2012 excluded from data
df['ELIGIBLE_2013'] = df['ELIGIBLE'] * (df['YEAR'] == 2013).astype(int)
df['ELIGIBLE_2014'] = df['ELIGIBLE'] * (df['YEAR'] == 2014).astype(int)
df['ELIGIBLE_2015'] = df['ELIGIBLE'] * (df['YEAR'] == 2015).astype(int)
df['ELIGIBLE_2016'] = df['ELIGIBLE'] * (df['YEAR'] == 2016).astype(int)

model_pretrends = smf.wls(
    'FT ~ ELIGIBLE + ELIGIBLE_2008 + ELIGIBLE_2009 + ELIGIBLE_2010 + ELIGIBLE_2013 + ELIGIBLE_2014 + ELIGIBLE_2015 + ELIGIBLE_2016 + FEMALE + C(MARST) + NCHILD + C(YEAR) + C(STATEFIP)',
    data=df,
    weights=df['PERWT']
).fit(cov_type='HC1')

print("Year-specific effects (reference year: 2011):")
pretrend_years = ['ELIGIBLE_2008', 'ELIGIBLE_2009', 'ELIGIBLE_2010', 'ELIGIBLE_2013', 'ELIGIBLE_2014', 'ELIGIBLE_2015', 'ELIGIBLE_2016']
print(f"{'Year':<15} {'Coefficient':>12} {'Std. Error':>12} {'p-value':>10}")
print("-" * 50)
for var in pretrend_years:
    print(f"{var:<15} {model_pretrends.params[var]:>12.4f} {model_pretrends.bse[var]:>12.4f} {model_pretrends.pvalues[var]:>10.4f}")

# ============================================================================
# ADDITIONAL SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("--- STEP 9: Summary Statistics for Report ---")
print("="*80)

print("\n[Sample Composition]")
print(f"Total observations: {len(df):,}")
print(f"Treatment group (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum():,} ({100*(df['ELIGIBLE']==1).mean():.1f}%)")
print(f"Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum():,} ({100*(df['ELIGIBLE']==0).mean():.1f}%)")
print(f"Pre-period (AFTER=0): {(df['AFTER']==0).sum():,} ({100*(df['AFTER']==0).mean():.1f}%)")
print(f"Post-period (AFTER=1): {(df['AFTER']==1).sum():,} ({100*(df['AFTER']==1).mean():.1f}%)")

print("\n[Weighted Population Estimates]")
total_weight = df['PERWT'].sum()
treated_weight = df[df['ELIGIBLE']==1]['PERWT'].sum()
control_weight = df[df['ELIGIBLE']==0]['PERWT'].sum()
print(f"Total weighted population: {total_weight:,.0f}")
print(f"Treatment group weighted: {treated_weight:,.0f} ({100*treated_weight/total_weight:.1f}%)")
print(f"Control group weighted: {control_weight:,.0f} ({100*control_weight/total_weight:.1f}%)")

print("\n[Demographics by Treatment Group]")
for eligible in [0, 1]:
    subset = df[df['ELIGIBLE'] == eligible]
    group_name = "Treatment (ages 26-30)" if eligible == 1 else "Control (ages 31-35)"
    print(f"\n{group_name}:")
    print(f"  N = {len(subset):,}")
    print(f"  Mean age at June 2012: {subset['AGE_IN_JUNE_2012'].mean():.1f}")
    print(f"  % Female: {100*(subset['SEX']==2).mean():.1f}%")
    print(f"  Mean FT rate: {100*subset['FT'].mean():.1f}%")
    print(f"  Weighted FT rate: {100*np.average(subset['FT'], weights=subset['PERWT']):.1f}%")

print("\n[State Distribution - Top 10]")
state_counts = df.groupby('STATEFIP').agg({
    'PERWT': 'sum',
    'FT': lambda x: np.average(x, weights=df.loc[x.index, 'PERWT'])
}).sort_values('PERWT', ascending=False).head(10)
state_counts.columns = ['Weighted_Pop', 'FT_Rate']
print(state_counts)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("--- STEP 10: Saving Results ---")
print("="*80)

# Create results dictionary for export
results = {
    'preferred_coefficient': did_coef,
    'preferred_se': did_se,
    'preferred_ci_low': ci_low,
    'preferred_ci_high': ci_high,
    'preferred_pvalue': did_p,
    'sample_size': int(preferred_model.nobs),
    'r_squared': preferred_model.rsquared
}

# Save to file
import json
with open('results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to results_summary.json")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Print final preferred estimate for reference
print(f"\n*** PREFERRED ESTIMATE ***")
print(f"Effect of DACA eligibility on full-time employment:")
print(f"  {did_coef:.4f} ({did_se:.4f})")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  N = {int(preferred_model.nobs):,}")
