"""
DACA Replication Analysis
Research Question: Impact of DACA eligibility on full-time employment probability
among Hispanic-Mexican Mexican-born individuals in the US

Analysis: Difference-in-Differences
- Treatment group: Ages 26-30 in June 2012 (ELIGIBLE=1)
- Control group: Ages 31-35 in June 2012 (ELIGIBLE=0)
- Pre-period: 2008-2011 (AFTER=0)
- Post-period: 2013-2016 (AFTER=1)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

data_path = r"C:\Users\seraf\DACA Results Task 3\replication_96\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"\n1. DATA OVERVIEW")
print("-"*40)
print(f"Total observations: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Columns: {df.shape[1]}")

# Check key variables
print(f"\n2. KEY VARIABLES")
print("-"*40)
print(f"ELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())
print(f"\nAFTER distribution:")
print(df['AFTER'].value_counts())
print(f"\nFT distribution:")
print(df['FT'].value_counts())

# Check sample sizes by treatment-period groups
print(f"\n3. SAMPLE SIZES BY GROUP")
print("-"*40)
group_counts = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
print(group_counts)

# Calculate weighted means for FT by group
print(f"\n4. WEIGHTED MEAN FT RATES BY GROUP")
print("-"*40)

def weighted_mean(group, value_col='FT', weight_col='PERWT'):
    return np.average(group[value_col], weights=group[weight_col])

def weighted_std(group, value_col='FT', weight_col='PERWT'):
    avg = np.average(group[value_col], weights=group[weight_col])
    variance = np.average((group[value_col] - avg)**2, weights=group[weight_col])
    return np.sqrt(variance)

# Calculate weighted FT rates for each group
groups = df.groupby(['ELIGIBLE', 'AFTER'])
ft_rates = {}
for name, group in groups:
    eligible, after = name
    ft_rate = weighted_mean(group)
    n = len(group)
    ft_rates[(eligible, after)] = {'rate': ft_rate, 'n': n}

print("\nFull-Time Employment Rates (Weighted):")
print(f"{'Group':<30} {'FT Rate':<15} {'N':<10}")
print("-"*55)
for (eligible, after), vals in ft_rates.items():
    group_name = f"ELIGIBLE={eligible}, AFTER={after}"
    print(f"{group_name:<30} {vals['rate']:.4f}          {vals['n']:,}")

# Calculate simple DiD estimate
print(f"\n5. SIMPLE DIFFERENCE-IN-DIFFERENCES ESTIMATE")
print("-"*40)

# Pre-period difference
pre_treated = ft_rates[(1, 0)]['rate']
pre_control = ft_rates[(0, 0)]['rate']
pre_diff = pre_treated - pre_control

# Post-period difference
post_treated = ft_rates[(1, 1)]['rate']
post_control = ft_rates[(0, 1)]['rate']
post_diff = post_treated - post_control

# DiD estimate
did_simple = post_diff - pre_diff

print(f"Pre-period (2008-2011):")
print(f"  Treated (ELIGIBLE=1): {pre_treated:.4f}")
print(f"  Control (ELIGIBLE=0): {pre_control:.4f}")
print(f"  Difference: {pre_diff:.4f}")

print(f"\nPost-period (2013-2016):")
print(f"  Treated (ELIGIBLE=1): {post_treated:.4f}")
print(f"  Control (ELIGIBLE=0): {post_control:.4f}")
print(f"  Difference: {post_diff:.4f}")

print(f"\nDifference-in-Differences Estimate: {did_simple:.4f}")
print(f"  (Change in treated - Change in control)")
print(f"  = ({post_treated:.4f} - {pre_treated:.4f}) - ({post_control:.4f} - {pre_control:.4f})")
print(f"  = {post_treated - pre_treated:.4f} - {post_control - pre_control:.4f}")
print(f"  = {did_simple:.4f}")

# Regression-based DiD
print(f"\n6. REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES")
print("-"*40)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD with person weights
print("\nModel 1: Basic DiD (FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER)")
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')
print(model1.summary())

# Store results for the report
results = {
    'model1': {
        'coef': model1.params['ELIGIBLE_AFTER'],
        'se': model1.bse['ELIGIBLE_AFTER'],
        'ci_low': model1.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model1.conf_int().loc['ELIGIBLE_AFTER', 1],
        'pvalue': model1.pvalues['ELIGIBLE_AFTER'],
        'n': len(df)
    }
}

print(f"\n*** DiD Coefficient (ELIGIBLE_AFTER): {results['model1']['coef']:.6f} ***")
print(f"    Standard Error: {results['model1']['se']:.6f}")
print(f"    95% CI: [{results['model1']['ci_low']:.6f}, {results['model1']['ci_high']:.6f}]")
print(f"    p-value: {results['model1']['pvalue']:.6f}")

# Model 2: DiD with year fixed effects
print("\n" + "="*80)
print("\nModel 2: DiD with Year Fixed Effects")
df['YEAR_cat'] = df['YEAR'].astype('category')
model2 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

results['model2'] = {
    'coef': model2.params['ELIGIBLE_AFTER'],
    'se': model2.bse['ELIGIBLE_AFTER'],
    'ci_low': model2.conf_int().loc['ELIGIBLE_AFTER', 0],
    'ci_high': model2.conf_int().loc['ELIGIBLE_AFTER', 1],
    'pvalue': model2.pvalues['ELIGIBLE_AFTER'],
    'n': len(df)
}

print(f"\n*** DiD Coefficient with Year FE: {results['model2']['coef']:.6f} ***")
print(f"    Standard Error: {results['model2']['se']:.6f}")
print(f"    95% CI: [{results['model2']['ci_low']:.6f}, {results['model2']['ci_high']:.6f}]")
print(f"    p-value: {results['model2']['pvalue']:.6f}")

# Model 3: DiD with covariates
print("\n" + "="*80)
print("\nModel 3: DiD with Demographic Covariates")

# Check for available covariates
print("\nAvailable covariates:")
covariates_to_check = ['SEX', 'MARST', 'EDUC', 'NCHILD', 'STATEFIP', 'METRO']
for cov in covariates_to_check:
    if cov in df.columns:
        print(f"  {cov}: {df[cov].nunique()} unique values")

# Create covariates for the model
# SEX: 1=Male, 2=Female
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Marital status: simplified to married vs not
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['EDUC_cat'] = pd.cut(df['EDUC'], bins=[-1, 5, 6, 9, 11], labels=['Less_than_HS', 'HS', 'Some_College', 'BA_plus'])

# Model 3: With demographics
model3 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + FEMALE + MARRIED + NCHILD + C(EDUC_cat)',
                  data=df.dropna(subset=['EDUC_cat']),
                  weights=df.dropna(subset=['EDUC_cat'])['PERWT']).fit(cov_type='HC1')

results['model3'] = {
    'coef': model3.params['ELIGIBLE_AFTER'],
    'se': model3.bse['ELIGIBLE_AFTER'],
    'ci_low': model3.conf_int().loc['ELIGIBLE_AFTER', 0],
    'ci_high': model3.conf_int().loc['ELIGIBLE_AFTER', 1],
    'pvalue': model3.pvalues['ELIGIBLE_AFTER'],
    'n': model3.nobs
}

print(f"\n*** DiD Coefficient with Covariates: {results['model3']['coef']:.6f} ***")
print(f"    Standard Error: {results['model3']['se']:.6f}")
print(f"    95% CI: [{results['model3']['ci_low']:.6f}, {results['model3']['ci_high']:.6f}]")
print(f"    p-value: {results['model3']['pvalue']:.6f}")
print(f"    N: {int(results['model3']['n']):,}")

# Model 4: Full model with state fixed effects
print("\n" + "="*80)
print("\nModel 4: DiD with State Fixed Effects + Covariates")

model4 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP) + FEMALE + MARRIED + NCHILD + C(EDUC_cat)',
                  data=df.dropna(subset=['EDUC_cat']),
                  weights=df.dropna(subset=['EDUC_cat'])['PERWT']).fit(cov_type='HC1')

results['model4'] = {
    'coef': model4.params['ELIGIBLE_AFTER'],
    'se': model4.bse['ELIGIBLE_AFTER'],
    'ci_low': model4.conf_int().loc['ELIGIBLE_AFTER', 0],
    'ci_high': model4.conf_int().loc['ELIGIBLE_AFTER', 1],
    'pvalue': model4.pvalues['ELIGIBLE_AFTER'],
    'n': model4.nobs
}

print(f"\n*** DiD Coefficient with State FE: {results['model4']['coef']:.6f} ***")
print(f"    Standard Error: {results['model4']['se']:.6f}")
print(f"    95% CI: [{results['model4']['ci_low']:.6f}, {results['model4']['ci_high']:.6f}]")
print(f"    p-value: {results['model4']['pvalue']:.6f}")

# Additional analysis: by sex
print("\n" + "="*80)
print("\n7. HETEROGENEITY ANALYSIS: BY SEX")
print("-"*40)

# Males
df_male = df[df['SEX'] == 1]
model_male = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                      data=df_male,
                      weights=df_male['PERWT']).fit(cov_type='HC1')

# Females
df_female = df[df['SEX'] == 2]
model_female = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=df_female,
                        weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"Males (N={len(df_male):,}):")
print(f"  DiD Coefficient: {model_male.params['ELIGIBLE_AFTER']:.6f}")
print(f"  SE: {model_male.bse['ELIGIBLE_AFTER']:.6f}")
print(f"  p-value: {model_male.pvalues['ELIGIBLE_AFTER']:.6f}")

print(f"\nFemales (N={len(df_female):,}):")
print(f"  DiD Coefficient: {model_female.params['ELIGIBLE_AFTER']:.6f}")
print(f"  SE: {model_female.bse['ELIGIBLE_AFTER']:.6f}")
print(f"  p-value: {model_female.pvalues['ELIGIBLE_AFTER']:.6f}")

results['male'] = {
    'coef': model_male.params['ELIGIBLE_AFTER'],
    'se': model_male.bse['ELIGIBLE_AFTER'],
    'n': len(df_male)
}
results['female'] = {
    'coef': model_female.params['ELIGIBLE_AFTER'],
    'se': model_female.bse['ELIGIBLE_AFTER'],
    'n': len(df_female)
}

# Pre-trend analysis: Year-by-year effects
print("\n" + "="*80)
print("\n8. PRE-TREND ANALYSIS")
print("-"*40)

# Create year-specific interactions with ELIGIBLE
df_pretrend = df.copy()
years = sorted(df['YEAR'].unique())
base_year = 2011  # Last pre-treatment year

for year in years:
    df_pretrend[f'YEAR_{year}'] = (df_pretrend['YEAR'] == year).astype(int)
    df_pretrend[f'ELIGIBLE_YEAR_{year}'] = df_pretrend['ELIGIBLE'] * df_pretrend[f'YEAR_{year}']

# Run event study regression (excluding base year)
year_vars = [f'YEAR_{y}' for y in years if y != base_year]
interaction_vars = [f'ELIGIBLE_YEAR_{y}' for y in years if y != base_year]

formula = 'FT ~ ELIGIBLE + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(interaction_vars)
model_event = smf.wls(formula, data=df_pretrend, weights=df_pretrend['PERWT']).fit(cov_type='HC1')

print(f"Event Study Coefficients (Base Year: {base_year}):")
print(f"{'Year':<10} {'Coefficient':<15} {'SE':<15} {'p-value':<15}")
print("-"*55)
for year in sorted(years):
    if year != base_year:
        var = f'ELIGIBLE_YEAR_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        pval = model_event.pvalues[var]
        marker = "*" if pval < 0.05 else ""
        print(f"{year:<10} {coef:<15.6f} {se:<15.6f} {pval:<15.6f} {marker}")

# Summary statistics table
print("\n" + "="*80)
print("\n9. SUMMARY STATISTICS")
print("-"*40)

# Summary stats by treatment group (pre-period only)
df_pre = df[df['AFTER'] == 0]

summary_vars = ['FT', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD']
print("\nPre-Period Summary Statistics (2008-2011):")
print(f"{'Variable':<15} {'Treated Mean':<15} {'Control Mean':<15} {'Difference':<15}")
print("-"*60)

for var in summary_vars:
    if var in df_pre.columns:
        treated_mean = np.average(df_pre[df_pre['ELIGIBLE']==1][var],
                                   weights=df_pre[df_pre['ELIGIBLE']==1]['PERWT'])
        control_mean = np.average(df_pre[df_pre['ELIGIBLE']==0][var],
                                   weights=df_pre[df_pre['ELIGIBLE']==0]['PERWT'])
        diff = treated_mean - control_mean
        print(f"{var:<15} {treated_mean:<15.4f} {control_mean:<15.4f} {diff:<15.4f}")

# Final preferred estimate
print("\n" + "="*80)
print("\n10. PREFERRED ESTIMATE (MODEL 4)")
print("="*80)

print(f"""
PREFERRED ESTIMATE:
===================
Effect Size: {results['model4']['coef']:.6f}
Standard Error: {results['model4']['se']:.6f}
95% Confidence Interval: [{results['model4']['ci_low']:.6f}, {results['model4']['ci_high']:.6f}]
p-value: {results['model4']['pvalue']:.6f}
Sample Size: {int(results['model4']['n']):,}

This estimate comes from a difference-in-differences regression with:
- Year fixed effects (to control for common time trends)
- State fixed effects (to control for time-invariant state characteristics)
- Demographic covariates: sex, marital status, number of children, education
- Heteroskedasticity-robust standard errors
- ACS person weights (PERWT)

INTERPRETATION:
DACA eligibility is associated with a {results['model4']['coef']*100:.2f} percentage point
{"increase" if results['model4']['coef'] > 0 else "decrease"} in the probability of full-time
employment among eligible individuals, compared to the control group.
This effect is {"statistically significant" if results['model4']['pvalue'] < 0.05 else "not statistically significant"} at the 5% level.
""")

# Save results to file
import json
with open(r'C:\Users\seraf\DACA Results Task 3\replication_96\analysis_results.json', 'w') as f:
    # Convert results to serializable format
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                     for k, v in value.items()}
    json.dump(results_serializable, f, indent=2)

print("\nResults saved to analysis_results.json")

# Create visualizations
print("\n" + "="*80)
print("\n11. CREATING VISUALIZATIONS")
print("-"*40)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Figure 1: Trends in FT employment by group
fig1, ax1 = plt.subplots(figsize=(10, 6))

ft_by_year_group = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()

ft_by_year_group.columns = ['Control (Ages 31-35)', 'Treatment (Ages 26-30)']
ft_by_year_group.plot(ax=ax1, marker='o', linewidth=2)

ax1.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation (2012)')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Rates by Treatment Status and Year', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_96\figure1_trends.png', dpi=300, bbox_inches='tight')
print("Saved: figure1_trends.png")

# Figure 2: Event study plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

event_coefs = []
event_ses = []
event_years = []

for year in sorted(years):
    if year == base_year:
        event_coefs.append(0)
        event_ses.append(0)
    else:
        var = f'ELIGIBLE_YEAR_{year}'
        event_coefs.append(model_event.params[var])
        event_ses.append(model_event.bse[var])
    event_years.append(year)

event_coefs = np.array(event_coefs)
event_ses = np.array(event_ses)

ax2.errorbar(event_years, event_coefs, yerr=1.96*event_ses, fmt='o-', capsize=5,
             linewidth=2, markersize=8, color='blue')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation')
ax2.fill_between([2008, 2012], -0.1, 0.1, alpha=0.2, color='gray', label='Pre-period')
ax2.fill_between([2012, 2016], -0.1, 0.1, alpha=0.2, color='lightblue', label='Post-period')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_96\figure2_event_study.png', dpi=300, bbox_inches='tight')
print("Saved: figure2_event_study.png")

# Figure 3: Difference in FT rates (Treated - Control) by year
fig3, ax3 = plt.subplots(figsize=(10, 6))

diff_by_year = ft_by_year_group['Treatment (Ages 26-30)'] - ft_by_year_group['Control (Ages 31-35)']
diff_by_year.plot(ax=ax3, marker='o', linewidth=2, color='green')

ax3.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Difference in FT Rate (Treatment - Control)', fontsize=12)
ax3.set_title('Difference in Full-Time Employment Rates Over Time', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_96\figure3_difference.png', dpi=300, bbox_inches='tight')
print("Saved: figure3_difference.png")

plt.close('all')

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
