"""
DACA Replication Analysis - Study 97
Effect of DACA eligibility on full-time employment

Research Design: Difference-in-Differences
Treatment: DACA eligibility (ages 26-30 at implementation)
Control: Ages 31-35 at implementation (otherwise eligible)
Outcome: Full-time employment (FT)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_97\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print("="*80)
print("DACA REPLICATION STUDY - DATA EXPLORATION")
print("="*80)

# Basic info
print(f"\nDataset shape: {df.shape}")
print(f"Number of observations: {len(df)}")
print(f"\nColumns ({len(df.columns)}):")
print(df.columns.tolist())

# Key variables
print("\n" + "="*80)
print("KEY VARIABLES")
print("="*80)

# Year distribution
print("\nYear distribution:")
print(df['YEAR'].value_counts().sort_index())

# ELIGIBLE distribution
print("\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())
print(f"  1 = Treatment group (ages 26-30 at DACA): {(df['ELIGIBLE']==1).sum()}")
print(f"  0 = Control group (ages 31-35 at DACA): {(df['ELIGIBLE']==0).sum()}")

# AFTER distribution
print("\nAFTER distribution:")
print(df['AFTER'].value_counts())
print(f"  0 = Pre-period (2008-2011): {(df['AFTER']==0).sum()}")
print(f"  1 = Post-period (2013-2016): {(df['AFTER']==1).sum()}")

# FT distribution
print("\nFT (Full-time employment) distribution:")
print(df['FT'].value_counts())
print(f"  Overall FT rate: {df['FT'].mean():.4f}")

# Create interaction term for DiD
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

print("\n" + "="*80)
print("SAMPLE CHARACTERISTICS")
print("="*80)

# Check AGE_IN_JUNE_2012
print("\nAge distribution at June 2012 (AGE_IN_JUNE_2012):")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe())

# Sex distribution (1=Male, 2=Female in IPUMS)
print("\nSex distribution by group:")
print(df.groupby('ELIGIBLE')['SEX'].value_counts().unstack(fill_value=0))

# Education distribution
print("\nEducation (EDUC_RECODE) by group:")
print(df.groupby('ELIGIBLE')['EDUC_RECODE'].value_counts().unstack(fill_value=0))

# Marital status
print("\nMarital status (MARST) by group:")
print(df.groupby('ELIGIBLE')['MARST'].value_counts().unstack(fill_value=0))

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS - 2x2 TABLE")
print("="*80)

# Create 2x2 table of FT rates
table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
print("\nFull-time employment rates by group and period:")
print(table)

# Unweighted DiD estimate
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
print("\n2x2 Mean Table (unweighted):")
print(ft_means)

did_unweighted = (ft_means.loc[1, 1] - ft_means.loc[1, 0]) - (ft_means.loc[0, 1] - ft_means.loc[0, 0])
print(f"\nUnweighted DiD estimate: {did_unweighted:.4f}")

# Weighted means
print("\n" + "="*80)
print("WEIGHTED DESCRIPTIVE STATISTICS")
print("="*80)

def weighted_mean(x, weights):
    return np.average(x, weights=weights)

def weighted_std(x, weights):
    avg = np.average(x, weights=weights)
    variance = np.average((x - avg)**2, weights=weights)
    return np.sqrt(variance)

# Calculate weighted means for each cell
print("\nWeighted full-time employment rates:")
for eligible in [0, 1]:
    for after in [0, 1]:
        mask = (df['ELIGIBLE'] == eligible) & (df['AFTER'] == after)
        subset = df[mask]
        wt_mean = weighted_mean(subset['FT'], subset['PERWT'])
        wt_std = weighted_std(subset['FT'], subset['PERWT'])
        n = len(subset)
        group = "Treatment" if eligible == 1 else "Control"
        period = "Post" if after == 1 else "Pre"
        print(f"{group}, {period}: FT = {wt_mean:.4f} (SD = {wt_std:.4f}), N = {n}")

# Calculate weighted DiD manually
wt_means = {}
for eligible in [0, 1]:
    for after in [0, 1]:
        mask = (df['ELIGIBLE'] == eligible) & (df['AFTER'] == after)
        subset = df[mask]
        wt_means[(eligible, after)] = weighted_mean(subset['FT'], subset['PERWT'])

did_weighted = (wt_means[(1,1)] - wt_means[(1,0)]) - (wt_means[(0,1)] - wt_means[(0,0)])
print(f"\nWeighted DiD estimate: {did_weighted:.4f}")

# Within-group changes
print("\nWithin-group changes:")
treat_change = wt_means[(1,1)] - wt_means[(1,0)]
control_change = wt_means[(0,1)] - wt_means[(0,0)]
print(f"  Treatment (ages 26-30): {wt_means[(1,0)]:.4f} -> {wt_means[(1,1)]:.4f} (change: {treat_change:+.4f})")
print(f"  Control (ages 31-35): {wt_means[(0,0)]:.4f} -> {wt_means[(0,1)]:.4f} (change: {control_change:+.4f})")
print(f"  Difference-in-Differences: {did_weighted:.4f}")

print("\n" + "="*80)
print("MAIN REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD (unweighted OLS)
print("\n--- Model 1: Basic DiD (OLS, unweighted) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])

# Model 2: Basic DiD with survey weights (WLS)
print("\n--- Model 2: Basic DiD (WLS, weighted by PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with robust standard errors
print("\n--- Model 3: Basic DiD (OLS, HC1 robust SE) ---")
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: WLS with robust SE
print("\n--- Model 4: Basic DiD (WLS, HC1 robust SE) - PREFERRED ---")
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary().tables[1])

print("\n" + "="*80)
print("MODELS WITH COVARIATES")
print("="*80)

# Create dummy variables for categorical vars
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)  # Married spouse present

# Model 5: DiD with demographic controls
print("\n--- Model 5: DiD with demographic controls (WLS, robust SE) ---")
covariate_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD'
model5 = smf.wls(covariate_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary().tables[1])

# Model 6: DiD with demographic + education controls using matrix approach
print("\n--- Model 6: DiD with demographics + education (WLS, robust SE) ---")
# Create education dummies with numeric names
educ_map = {'Less than High School': 1, 'High School Degree': 2, 'Some College': 3, 'Two-Year Degree': 4, 'BA+': 5}
df['EDUC_NUM'] = df['EDUC_RECODE'].map(educ_map).fillna(0).astype(int)

# Build design matrix
X6_cols = ['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'NCHILD']
X6 = df[X6_cols].copy().astype(float)
# Add education dummies (drop first = less than HS)
for ed in [2, 3, 4, 5]:
    X6[f'educ_{ed}'] = (df['EDUC_NUM'] == ed).astype(float)
X6 = sm.add_constant(X6)
model6 = sm.WLS(df['FT'].astype(float), X6.astype(float), weights=df['PERWT'].astype(float)).fit(cov_type='HC1')
print(f"Key coefficients:")
print(f"  Intercept: {model6.params['const']:.4f}")
print(f"  ELIGIBLE: {model6.params['ELIGIBLE']:.4f} (SE: {model6.bse['ELIGIBLE']:.4f})")
print(f"  AFTER: {model6.params['AFTER']:.4f} (SE: {model6.bse['AFTER']:.4f})")
print(f"  ELIGIBLE_AFTER: {model6.params['ELIGIBLE_AFTER']:.4f} (SE: {model6.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  FEMALE: {model6.params['FEMALE']:.4f} (SE: {model6.bse['FEMALE']:.4f})")
print(f"  Education dummies included (HS, Some College, Two-Year, BA+)")

# Model 7: Add year fixed effects
print("\n--- Model 7: DiD with year fixed effects (WLS, robust SE) ---")
X7_cols = ['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'NCHILD']
X7 = df[X7_cols].copy().astype(float)
# Add education dummies
for ed in [2, 3, 4, 5]:
    X7[f'educ_{ed}'] = (df['EDUC_NUM'] == ed).astype(float)
# Add year dummies (drop 2008)
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    X7[f'year_{year}'] = (df['YEAR'] == year).astype(float)
X7 = sm.add_constant(X7)
model7 = sm.WLS(df['FT'].astype(float), X7.astype(float), weights=df['PERWT'].astype(float)).fit(cov_type='HC1')
print(f"Key coefficients:")
print(f"  ELIGIBLE: {model7.params['ELIGIBLE']:.4f} (SE: {model7.bse['ELIGIBLE']:.4f})")
print(f"  ELIGIBLE_AFTER (DiD): {model7.params['ELIGIBLE_AFTER']:.4f} (SE: {model7.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  FEMALE: {model7.params['FEMALE']:.4f}")
print(f"  Year fixed effects included")

# Model 8: Add state fixed effects
print("\n--- Model 8: DiD with state + year FE (WLS, robust SE) ---")
X8_cols = ['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'NCHILD']
X8 = df[X8_cols].copy().astype(float)
# Add education dummies
for ed in [2, 3, 4, 5]:
    X8[f'educ_{ed}'] = (df['EDUC_NUM'] == ed).astype(float)
# Add year dummies
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    X8[f'year_{year}'] = (df['YEAR'] == year).astype(float)
# Add state dummies (drop first state)
states = sorted(df['STATEFIP'].unique())[1:]  # drop first
for state in states:
    X8[f'state_{state}'] = (df['STATEFIP'] == state).astype(float)
X8 = sm.add_constant(X8)
model8 = sm.WLS(df['FT'].astype(float), X8.astype(float), weights=df['PERWT'].astype(float)).fit(cov_type='HC1')
print(f"Key coefficients:")
print(f"  ELIGIBLE: {model8.params['ELIGIBLE']:.4f} (SE: {model8.bse['ELIGIBLE']:.4f})")
print(f"  ELIGIBLE_AFTER (DiD): {model8.params['ELIGIBLE_AFTER']:.4f} (SE: {model8.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  State and year fixed effects included")
print(f"  N states: {len(states)+1}, N obs: {int(model8.nobs)}")

print("\n" + "="*80)
print("PREFERRED MODEL SUMMARY")
print("="*80)

# Use Model 4 as preferred (basic DiD with weights and robust SE)
preferred = model4
print("\nPreferred Model: Basic DiD with survey weights and robust SE")
print(f"\nDiD Estimate (ELIGIBLE_AFTER): {preferred.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {preferred.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{preferred.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {preferred.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"t-statistic: {preferred.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {preferred.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"\nSample size: {int(preferred.nobs)}")
print(f"R-squared: {preferred.rsquared:.4f}")

print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Parallel trends check - separate pre-trends
print("\n--- Pre-trend Analysis ---")
pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_CENTERED'] = pre_data['YEAR'] - 2008
model_pretrend = smf.wls('FT ~ ELIGIBLE * YEAR_CENTERED', data=pre_data, weights=pre_data['PERWT']).fit(cov_type='HC1')
print("Pre-period trend interaction (ELIGIBLE:YEAR_CENTERED):")
print(f"  Coefficient: {model_pretrend.params['ELIGIBLE:YEAR_CENTERED']:.4f}")
print(f"  SE: {model_pretrend.bse['ELIGIBLE:YEAR_CENTERED']:.4f}")
print(f"  p-value: {model_pretrend.pvalues['ELIGIBLE:YEAR_CENTERED']:.4f}")
if model_pretrend.pvalues['ELIGIBLE:YEAR_CENTERED'] > 0.05:
    print("  -> Parallel trends assumption supported (no significant differential pre-trend)")
else:
    print("  -> WARNING: Evidence of differential pre-trends")

# Subgroup analysis by sex
print("\n--- Subgroup Analysis by Sex ---")
for sex, sex_name in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex]
    model_sex = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"\n{sex_name}:")
    print(f"  DiD estimate: {model_sex.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  SE: {model_sex.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"  95% CI: [{model_sex.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_sex.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
    print(f"  p-value: {model_sex.pvalues['ELIGIBLE_AFTER']:.4f}")
    print(f"  N: {int(model_sex.nobs)}")

# Event study style - by year
print("\n--- Year-by-Year Effects (Event Study) ---")
print("\nWeighted difference (Treatment - Control) by year:")
year_effects = []
for year in sorted(df['YEAR'].unique()):
    subset = df[df['YEAR'] == year]
    treat_mask = subset['ELIGIBLE'] == 1
    ctrl_mask = subset['ELIGIBLE'] == 0
    if treat_mask.sum() > 0 and ctrl_mask.sum() > 0:
        wt_diff = weighted_mean(subset[treat_mask]['FT'], subset[treat_mask]['PERWT']) - \
                  weighted_mean(subset[ctrl_mask]['FT'], subset[ctrl_mask]['PERWT'])
        year_effects.append((year, wt_diff))
        period = "Pre" if year < 2012 else "Post"
        print(f"  {year} ({period}): {wt_diff:+.4f}")

# Average pre and post differences
pre_avg = np.mean([e[1] for e in year_effects if e[0] < 2012])
post_avg = np.mean([e[1] for e in year_effects if e[0] > 2012])
print(f"\n  Average pre-period difference: {pre_avg:+.4f}")
print(f"  Average post-period difference: {post_avg:+.4f}")
print(f"  Difference (DiD): {post_avg - pre_avg:+.4f}")

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY TABLE")
print("="*80)

print("\n{:<50} {:>12} {:>12} {:>10}".format("Model", "Estimate", "SE", "p-value"))
print("-"*84)
models_summary = [
    ("(1) OLS, unweighted", model1),
    ("(2) WLS, weighted", model2),
    ("(3) OLS, robust SE", model3),
    ("(4) WLS, robust SE (preferred)", model4),
    ("(5) WLS + demographics", model5),
    ("(6) WLS + demographics + education", model6),
    ("(7) WLS + year FE", model7),
    ("(8) WLS + state + year FE", model8),
]
for name, model in models_summary:
    est = model.params['ELIGIBLE_AFTER']
    se = model.bse['ELIGIBLE_AFTER']
    pval = model.pvalues['ELIGIBLE_AFTER']
    print(f"{name:<50} {est:>12.4f} {se:>12.4f} {pval:>10.4f}")

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print("\n" + "-"*60)
print("MAIN FINDING")
print("-"*60)
print(f"\nEffect of DACA eligibility on full-time employment:")
print(f"  Point estimate: {preferred.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard error: {preferred.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% Confidence interval: [{preferred.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {preferred.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  p-value: {preferred.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  Sample size: {int(preferred.nobs)}")

stat_sig = "statistically significant" if preferred.pvalues['ELIGIBLE_AFTER'] < 0.05 else "not statistically significant"
direction = "positive" if preferred.params['ELIGIBLE_AFTER'] > 0 else "negative"
print(f"\nThe estimated effect is {direction} and {stat_sig} at the 5% level.")
print(f"\nInterpretation: DACA eligibility increased the probability of full-time")
print(f"employment by {preferred.params['ELIGIBLE_AFTER']*100:.2f} percentage points among the treatment group")
print(f"(ages 26-30 at implementation) relative to the control group (ages 31-35).")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save key results
results = {
    'did_estimate': preferred.params['ELIGIBLE_AFTER'],
    'se': preferred.bse['ELIGIBLE_AFTER'],
    'ci_lower': preferred.conf_int().loc['ELIGIBLE_AFTER', 0],
    'ci_upper': preferred.conf_int().loc['ELIGIBLE_AFTER', 1],
    'pvalue': preferred.pvalues['ELIGIBLE_AFTER'],
    'n': int(preferred.nobs)
}
print("\nKey results dictionary:")
print(results)
