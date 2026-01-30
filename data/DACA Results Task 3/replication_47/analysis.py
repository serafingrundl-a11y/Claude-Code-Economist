"""
DACA Replication Study: Effect on Full-Time Employment
Analysis Script
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 3\replication_47")

# Load data
print("Loading data...")
df = pd.read_csv('data/prepared_data_labelled_version.csv', low_memory=False)
print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} variables")

# Create additional variables
df['MALE'] = (df['SEX'] == 'Male').astype(int)
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# =============================================================================
# TABLE 1: Summary Statistics
# =============================================================================
print("\n" + "="*60)
print("TABLE 1: Summary Statistics")
print("="*60)

summary_vars = ['FT', 'AGE', 'MALE', 'PERWT']
summary_stats = df.groupby(['ELIGIBLE', 'AFTER'])[summary_vars].agg(['mean', 'std', 'count'])
print(summary_stats)

# Group-specific statistics
groups = [
    ('Treatment (Ages 26-30), Pre-DACA', (df['ELIGIBLE']==1) & (df['AFTER']==0)),
    ('Treatment (Ages 26-30), Post-DACA', (df['ELIGIBLE']==1) & (df['AFTER']==1)),
    ('Control (Ages 31-35), Pre-DACA', (df['ELIGIBLE']==0) & (df['AFTER']==0)),
    ('Control (Ages 31-35), Post-DACA', (df['ELIGIBLE']==0) & (df['AFTER']==1)),
]

print("\nDetailed Summary by Group:")
for name, mask in groups:
    print(f"\n{name}:")
    print(f"  N = {mask.sum()}")
    print(f"  FT rate = {df.loc[mask, 'FT'].mean():.3f}")
    print(f"  Mean age = {df.loc[mask, 'AGE'].mean():.1f}")
    print(f"  Male % = {df.loc[mask, 'MALE'].mean()*100:.1f}%")

# =============================================================================
# TABLE 2: Simple DID Calculation
# =============================================================================
print("\n" + "="*60)
print("TABLE 2: Simple Difference-in-Differences")
print("="*60)

means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
print("\nUnweighted FT means:")
print(means.round(4))

did_simple = (means.loc[1,1] - means.loc[1,0]) - (means.loc[0,1] - means.loc[0,0])
print(f"\nSimple DID estimate: {did_simple:.4f} ({did_simple*100:.2f} percentage points)")

# Weighted means
weighted_means = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
print("\nWeighted FT means:")
print(weighted_means.round(4))

did_weighted = (weighted_means.loc[1,1] - weighted_means.loc[1,0]) - (weighted_means.loc[0,1] - weighted_means.loc[0,0])
print(f"\nWeighted DID estimate: {did_weighted:.4f} ({did_weighted*100:.2f} percentage points)")

# =============================================================================
# TABLE 3: DID Regression Results
# =============================================================================
print("\n" + "="*60)
print("TABLE 3: DID Regression Results")
print("="*60)

# Model 1: Basic OLS
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE:AFTER', data=df).fit(cov_type='HC1')

# Model 2: WLS with person weights
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE:AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 3: WLS with covariates
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE:AFTER + MALE + C(EDUC_RECODE) + C(MARST) + C(CensusRegion)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 4: WLS with year and state FE
model4 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE:AFTER + MALE + C(YEAR) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 5: Full specification with all controls
model5 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE:AFTER + MALE + C(EDUC_RECODE) + C(MARST) + C(YEAR) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nModel comparison:")
print("-"*80)
print(f"{'Model':<30} {'DID Coef':<12} {'SE':<12} {'p-value':<12} {'N':<10}")
print("-"*80)

models = [
    ('(1) Basic OLS', model1),
    ('(2) Weighted', model2),
    ('(3) + Demographics', model3),
    ('(4) + Year/State FE', model4),
    ('(5) Full Specification', model5),
]

for name, m in models:
    coef = m.params['ELIGIBLE:AFTER']
    se = m.bse['ELIGIBLE:AFTER']
    pval = m.pvalues['ELIGIBLE:AFTER']
    n = int(m.nobs)
    print(f"{name:<30} {coef:>10.4f}   {se:>10.4f}   {pval:>10.4f}   {n:>8}")

# =============================================================================
# TABLE 4: Year-by-Year FT Employment Rates
# =============================================================================
print("\n" + "="*60)
print("TABLE 4: Full-Time Employment Rates by Year")
print("="*60)

yearly = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'FT_rate': np.average(x['FT'], weights=x['PERWT']),
        'N': len(x)
    })
).unstack()
print(yearly)

# =============================================================================
# FIGURE 1: Event Study Plot Data
# =============================================================================
print("\n" + "="*60)
print("Event Study Coefficients")
print("="*60)

# Create year dummies for event study
for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'YEAR_{y}'] = (df['YEAR'] == y).astype(int)
    df[f'ELIGIBLE_YEAR_{y}'] = df['ELIGIBLE'] * df[f'YEAR_{y}']

formula_event = 'FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + '
formula_event += 'ELIGIBLE_YEAR_2008 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 + ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016 + MALE'

model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

event_results = []
for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'ELIGIBLE_YEAR_{y}']
    se = model_event.bse[f'ELIGIBLE_YEAR_{y}']
    ci_low = coef - 1.96*se
    ci_high = coef + 1.96*se
    event_results.append({'year': y, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})
    print(f"{y}: coef={coef:.4f}, SE={se:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]")

# Add reference year
event_results.append({'year': 2011, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
event_df = pd.DataFrame(event_results).sort_values('year')

# Create event study plot
plt.figure(figsize=(10, 6))
plt.errorbar(event_df['year'], event_df['coef'],
             yerr=1.96*event_df['se'],
             fmt='o-', capsize=5, capthick=2, markersize=8)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Treatment Effect (relative to 2011)', fontsize=12)
plt.title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nEvent study plot saved to event_study.png")

# =============================================================================
# FIGURE 2: Trends by Treatment Group
# =============================================================================
plt.figure(figsize=(10, 6))

# Calculate weighted means by year
for eligible, label, color in [(1, 'Treatment (Ages 26-30)', 'blue'),
                                (0, 'Control (Ages 31-35)', 'red')]:
    subset = df[df['ELIGIBLE'] == eligible]
    yearly_ft = subset.groupby('YEAR').apply(lambda x: np.average(x['FT'], weights=x['PERWT']))
    plt.plot(yearly_ft.index, yearly_ft.values, 'o-', label=label, color=color, linewidth=2, markersize=8)

plt.axvline(x=2012, color='gray', linestyle='--', alpha=0.7, label='DACA (2012)')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Full-Time Employment Trends by DACA Eligibility', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.55, 0.75)
plt.tight_layout()
plt.savefig('trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Trends plot saved to trends.png")

# =============================================================================
# Pre-Trends Test
# =============================================================================
print("\n" + "="*60)
print("Pre-Trends Test")
print("="*60)

pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_centered'] = pre_data['YEAR'] - 2008

model_pretrend = smf.wls('FT ~ ELIGIBLE * YEAR_centered', data=pre_data,
                         weights=pre_data['PERWT']).fit(cov_type='HC1')
print(f"Differential pre-trend coefficient: {model_pretrend.params['ELIGIBLE:YEAR_centered']:.4f}")
print(f"Standard error: {model_pretrend.bse['ELIGIBLE:YEAR_centered']:.4f}")
print(f"p-value: {model_pretrend.pvalues['ELIGIBLE:YEAR_centered']:.4f}")

# =============================================================================
# Heterogeneity Analysis: By Sex
# =============================================================================
print("\n" + "="*60)
print("Heterogeneity Analysis: By Sex")
print("="*60)

for sex, label in [(1, 'Male'), (0, 'Female')]:
    subset = df[df['MALE'] == sex]
    model_sex = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE:AFTER',
                        data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"\n{label}:")
    print(f"  DID coefficient: {model_sex.params['ELIGIBLE:AFTER']:.4f}")
    print(f"  SE: {model_sex.bse['ELIGIBLE:AFTER']:.4f}")
    print(f"  p-value: {model_sex.pvalues['ELIGIBLE:AFTER']:.4f}")
    print(f"  N: {int(model_sex.nobs)}")

# =============================================================================
# Save Key Results
# =============================================================================
print("\n" + "="*60)
print("PREFERRED ESTIMATE (Model 2: Weighted DID)")
print("="*60)
print(f"Effect size: {model2.params['ELIGIBLE:AFTER']:.4f}")
print(f"Standard error: {model2.bse['ELIGIBLE:AFTER']:.4f}")
print(f"95% CI: [{model2.params['ELIGIBLE:AFTER'] - 1.96*model2.bse['ELIGIBLE:AFTER']:.4f}, {model2.params['ELIGIBLE:AFTER'] + 1.96*model2.bse['ELIGIBLE:AFTER']:.4f}]")
print(f"Sample size: {int(model2.nobs)}")

print("\nAnalysis complete!")
