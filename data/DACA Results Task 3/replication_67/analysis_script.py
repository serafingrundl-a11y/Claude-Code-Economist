"""
DACA Replication Study - Analysis Script
Research Question: Impact of DACA eligibility on full-time employment

Treatment: ELIGIBLE=1 (ages 26-30 at June 2012)
Control: ELIGIBLE=0 (ages 31-35 at June 2012)
Pre-period: 2008-2011 (AFTER=0)
Post-period: 2013-2016 (AFTER=1)
Outcome: FT (full-time employment)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"\nTotal observations loaded: {len(df):,}")

# =============================================================================
# 2. DATA EXPLORATION
# =============================================================================
print("\n" + "=" * 70)
print("DATA EXPLORATION")
print("=" * 70)

# Basic info
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")
print(f"ELIGIBLE values: {df['ELIGIBLE'].unique()}")
print(f"AFTER values: {df['AFTER'].unique()}")

# Sample sizes by group
print("\n--- Sample Size by Group ---")
group_counts = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
group_counts.index = ['Control (31-35)', 'Treatment (26-30)']
group_counts.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print(group_counts)

# Weighted sample sizes
print("\n--- Weighted Sample Size by Group ---")
weighted_counts = df.groupby(['ELIGIBLE', 'AFTER'])['PERWT'].sum().unstack()
weighted_counts.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_counts.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print(weighted_counts.round(0))

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

# Outcome variable by group
print("\n--- Full-Time Employment Rate by Group (Weighted) ---")

def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

ft_rates = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean).unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print(ft_rates.round(4))

# Simple DiD calculation
print("\n--- Simple Difference-in-Differences Calculation ---")
dd_treatment = ft_rates.loc['Treatment (26-30)', 'Post (2013-2016)'] - ft_rates.loc['Treatment (26-30)', 'Pre (2008-2011)']
dd_control = ft_rates.loc['Control (31-35)', 'Post (2013-2016)'] - ft_rates.loc['Control (31-35)', 'Pre (2008-2011)']
dd_estimate = dd_treatment - dd_control
print(f"Treatment change (post - pre): {dd_treatment:.4f}")
print(f"Control change (post - pre): {dd_control:.4f}")
print(f"DiD estimate: {dd_estimate:.4f}")

# Demographic characteristics
print("\n--- Demographic Characteristics by Treatment Status (Pre-period) ---")
pre_data = df[df['AFTER'] == 0]

# Sex (1=Male, 2=Female in IPUMS)
def pct_female(g):
    return np.average(g['SEX'] == 2, weights=g['PERWT'])

# Mean age
def mean_age(g):
    return np.average(g['AGE'], weights=g['PERWT'])

# Married (MARST: 1=married spouse present, 2=married spouse absent)
def pct_married(g):
    return np.average(g['MARST'].isin([1, 2]), weights=g['PERWT'])

# Has children
def pct_children(g):
    return np.average(g['NCHILD'] > 0, weights=g['PERWT'])

# High school degree or more
def pct_hs(g):
    return np.average(g['HS_DEGREE'] == True, weights=g['PERWT'])

demographics = pre_data.groupby('ELIGIBLE').agg({
    'AGE': lambda x: np.average(x, weights=pre_data.loc[x.index, 'PERWT']),
    'PERWT': 'sum'
})

print("\nSelected characteristics (pre-period, weighted):")
for elig in [0, 1]:
    sub = pre_data[pre_data['ELIGIBLE'] == elig]
    group_name = 'Treatment (26-30)' if elig == 1 else 'Control (31-35)'
    print(f"\n{group_name}:")
    print(f"  Mean age: {np.average(sub['AGE'], weights=sub['PERWT']):.2f}")
    print(f"  % Female: {pct_female(sub)*100:.1f}%")
    print(f"  % Married: {pct_married(sub)*100:.1f}%")
    print(f"  % Has children: {pct_children(sub)*100:.1f}%")
    print(f"  % High school+: {np.average(sub['HS_DEGREE']==True, weights=sub['PERWT'])*100:.1f}%")

# =============================================================================
# 4. YEARLY TRENDS (for parallel trends check)
# =============================================================================
print("\n" + "=" * 70)
print("YEARLY FULL-TIME EMPLOYMENT RATES")
print("=" * 70)

yearly_rates = df.groupby(['YEAR', 'ELIGIBLE']).apply(weighted_mean)
yearly_rates = yearly_rates.unstack()
yearly_rates.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\nFull-time employment rate by year:")
print(yearly_rates.round(4))

# Calculate the gap
yearly_rates['Gap'] = yearly_rates['Treatment (26-30)'] - yearly_rates['Control (31-35)']
print("\nGap (Treatment - Control):")
print(yearly_rates['Gap'].round(4))

# =============================================================================
# 5. REGRESSION ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("REGRESSION ANALYSIS")
print("=" * 70)

# Create interaction term
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                      cov_kwds={'groups': df['STATEFIP']})
print(model1.summary().tables[1])
print(f"\nDiD Estimate: {model1.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Std Error (clustered): {model1.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model1.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")

# Create dummy variables for categorical controls
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Education dummies (using EDUC_RECODE)
educ_dummies = pd.get_dummies(df['EDUC_RECODE'], prefix='EDUC', drop_first=True)
df = pd.concat([df, educ_dummies], axis=1)

model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                      cov_kwds={'groups': df['STATEFIP']})
print(model2.summary().tables[1])
print(f"\nDiD Estimate: {model2.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Std Error (clustered): {model2.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 3: DiD with state and year fixed effects
print("\n--- Model 3: DiD with State and Year Fixed Effects ---")

# Create state and year dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True)

# Merge with df for regression
df_fe = pd.concat([df, state_dummies, year_dummies], axis=1)

# Build formula with fixed effects
state_vars = ' + '.join([c for c in state_dummies.columns])
year_vars = ' + '.join([c for c in year_dummies.columns])

formula3 = f'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + {state_vars} + {year_vars}'

model3 = smf.wls(formula3, data=df_fe, weights=df_fe['PERWT']).fit(cov_type='cluster',
                                                                     cov_kwds={'groups': df_fe['STATEFIP']})

print(f"\nDiD Estimate: {model3.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Std Error (clustered): {model3.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# =============================================================================
# 6. EVENT STUDY SPECIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("EVENT STUDY SPECIFICATION")
print("=" * 70)

# Create year-specific treatment effects (relative to 2011)
for year in df['YEAR'].unique():
    df[f'ELIGIBLE_X_{year}'] = (df['ELIGIBLE'] == 1) & (df['YEAR'] == year)
    df[f'ELIGIBLE_X_{year}'] = df[f'ELIGIBLE_X_{year}'].astype(int)

# Drop 2011 as reference year
event_vars = [f'ELIGIBLE_X_{y}' for y in sorted(df['YEAR'].unique()) if y != 2011]
event_formula = 'FT ~ ELIGIBLE + ' + ' + '.join(event_vars)

# Add year dummies (drop 2011)
for year in df['YEAR'].unique():
    if year != 2011:
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

year_vars_event = [f'YEAR_{y}' for y in sorted(df['YEAR'].unique()) if y != 2011]
event_formula += ' + ' + ' + '.join(year_vars_event)

model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                                         cov_kwds={'groups': df['STATEFIP']})

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
for year in sorted(df['YEAR'].unique()):
    if year != 2011:
        var = f'ELIGIBLE_X_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        ci_low, ci_high = model_event.conf_int().loc[var]
        print(f"{year}: {coef:7.4f} (SE: {se:.4f}) 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# =============================================================================
# 7. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF RESULTS")
print("=" * 70)

print("\n--- Main DiD Estimates ---")
print(f"{'Model':<40} {'Estimate':>10} {'SE':>10} {'p-value':>10}")
print("-" * 70)
print(f"{'Model 1: Basic DiD':<40} {model1.params['ELIGIBLE_X_AFTER']:>10.4f} {model1.bse['ELIGIBLE_X_AFTER']:>10.4f} {model1.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'Model 2: + Demographics':<40} {model2.params['ELIGIBLE_X_AFTER']:>10.4f} {model2.bse['ELIGIBLE_X_AFTER']:>10.4f} {model2.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'Model 3: + State & Year FE':<40} {model3.params['ELIGIBLE_X_AFTER']:>10.4f} {model3.bse['ELIGIBLE_X_AFTER']:>10.4f} {model3.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")

# =============================================================================
# 8. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save key results to a file for the report
results_dict = {
    'n_obs': len(df),
    'n_treatment': len(df[df['ELIGIBLE'] == 1]),
    'n_control': len(df[df['ELIGIBLE'] == 0]),
    'ft_rate_treat_pre': ft_rates.loc['Treatment (26-30)', 'Pre (2008-2011)'],
    'ft_rate_treat_post': ft_rates.loc['Treatment (26-30)', 'Post (2013-2016)'],
    'ft_rate_ctrl_pre': ft_rates.loc['Control (31-35)', 'Pre (2008-2011)'],
    'ft_rate_ctrl_post': ft_rates.loc['Control (31-35)', 'Post (2013-2016)'],
    'simple_dd': dd_estimate,
    'model1_coef': model1.params['ELIGIBLE_X_AFTER'],
    'model1_se': model1.bse['ELIGIBLE_X_AFTER'],
    'model1_pval': model1.pvalues['ELIGIBLE_X_AFTER'],
    'model1_ci_low': model1.conf_int().loc['ELIGIBLE_X_AFTER', 0],
    'model1_ci_high': model1.conf_int().loc['ELIGIBLE_X_AFTER', 1],
    'model2_coef': model2.params['ELIGIBLE_X_AFTER'],
    'model2_se': model2.bse['ELIGIBLE_X_AFTER'],
    'model2_pval': model2.pvalues['ELIGIBLE_X_AFTER'],
    'model2_ci_low': model2.conf_int().loc['ELIGIBLE_X_AFTER', 0],
    'model2_ci_high': model2.conf_int().loc['ELIGIBLE_X_AFTER', 1],
    'model3_coef': model3.params['ELIGIBLE_X_AFTER'],
    'model3_se': model3.bse['ELIGIBLE_X_AFTER'],
    'model3_pval': model3.pvalues['ELIGIBLE_X_AFTER'],
    'model3_ci_low': model3.conf_int().loc['ELIGIBLE_X_AFTER', 0],
    'model3_ci_high': model3.conf_int().loc['ELIGIBLE_X_AFTER', 1],
}

# Event study coefficients
for year in sorted(df['YEAR'].unique()):
    if year != 2011:
        var = f'ELIGIBLE_X_{year}'
        results_dict[f'event_{year}_coef'] = model_event.params[var]
        results_dict[f'event_{year}_se'] = model_event.bse[var]

# Save to file
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("Results saved to analysis_results.json")

# Save yearly rates for plotting
yearly_rates.to_csv('yearly_ft_rates.csv')
print("Yearly rates saved to yearly_ft_rates.csv")

# Save group counts
group_counts.to_csv('group_counts.csv')
print("Group counts saved to group_counts.csv")

# Full model summaries
with open('model_summaries.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("MODEL 1: BASIC DID\n")
    f.write("=" * 70 + "\n")
    f.write(str(model1.summary()) + "\n\n")
    f.write("=" * 70 + "\n")
    f.write("MODEL 2: DID WITH DEMOGRAPHIC CONTROLS\n")
    f.write("=" * 70 + "\n")
    f.write(str(model2.summary()) + "\n\n")

print("Model summaries saved to model_summaries.txt")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
