"""
DACA Replication Study - Analysis Script
Independent Replication #25

Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States?

Method: Difference-in-Differences (DiD)
- Treatment: Ages 26-30 at DACA implementation (June 2012)
- Control: Ages 31-35 at DACA implementation
- Pre-period: 2008-2011
- Post-period: 2013-2016
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

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# 1. Load and Explore Data
# =============================================================================
print("\n" + "="*80)
print("SECTION 1: DATA LOADING AND EXPLORATION")
print("="*80)

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\nDataset dimensions: {df.shape[0]:,} observations x {df.shape[1]} variables")

# Key variables
print("\n--- Key Variables ---")
print(f"ELIGIBLE: {df['ELIGIBLE'].value_counts().to_dict()}")
print(f"AFTER: {df['AFTER'].value_counts().to_dict()}")
print(f"FT: {df['FT'].value_counts().to_dict()}")

# Year distribution
print("\n--- Year Distribution ---")
print(df['YEAR'].value_counts().sort_index())

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# =============================================================================
# 2. Summary Statistics
# =============================================================================
print("\n" + "="*80)
print("SECTION 2: SUMMARY STATISTICS")
print("="*80)

# Overall sample statistics
print("\n--- Overall Sample Statistics ---")
print(f"Total observations: {len(df):,}")
print(f"Treatment group (ELIGIBLE=1): {df['ELIGIBLE'].sum():,} ({100*df['ELIGIBLE'].mean():.1f}%)")
print(f"Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum():,} ({100*(1-df['ELIGIBLE'].mean()):.1f}%)")
print(f"Pre-period (AFTER=0): {(df['AFTER']==0).sum():,}")
print(f"Post-period (AFTER=1): {df['AFTER'].sum():,}")

# Full-time employment rates by group
print("\n--- Full-Time Employment Rates (Unweighted) ---")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)']
print(ft_rates.round(4))

# Calculate simple DiD
did_simple = (ft_rates.iloc[1,1] - ft_rates.iloc[1,0]) - (ft_rates.iloc[0,1] - ft_rates.iloc[0,0])
print(f"\nSimple DiD estimate (unweighted): {did_simple:.4f}")

# Weighted employment rates
print("\n--- Full-Time Employment Rates (Weighted by PERWT) ---")
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

ft_rates_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean).unstack()
ft_rates_weighted.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates_weighted.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)']
print(ft_rates_weighted.round(4))

did_weighted = (ft_rates_weighted.iloc[1,1] - ft_rates_weighted.iloc[1,0]) - (ft_rates_weighted.iloc[0,1] - ft_rates_weighted.iloc[0,0])
print(f"\nSimple DiD estimate (weighted): {did_weighted:.4f}")

# Demographics by group
print("\n--- Demographics by Treatment Group ---")
for eligible in [0, 1]:
    group_name = "Treatment (26-30)" if eligible == 1 else "Control (31-35)"
    subset = df[df['ELIGIBLE'] == eligible]
    print(f"\n{group_name}:")
    print(f"  N = {len(subset):,}")
    print(f"  Mean Age: {subset['AGE'].mean():.2f}")
    print(f"  Female %: {100*(subset['SEX']==2).mean():.1f}%")
    print(f"  Mean FT rate: {100*subset['FT'].mean():.1f}%")

# Year by Year employment rates
print("\n--- Full-Time Employment Rates by Year ---")
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_rates.columns = ['Control (31-35)', 'Treatment (26-30)']
print(yearly_rates.round(4))

# =============================================================================
# 3. Main DiD Regression Analysis
# =============================================================================
print("\n" + "="*80)
print("SECTION 3: DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("="*80)

# Model 1: Basic OLS (unweighted)
print("\n--- Model 1: Basic DiD (Unweighted, Robust SE) ---")
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
model1 = sm.OLS(df['FT'], X).fit(cov_type='HC1')
print(model1.summary())

# Model 2: Weighted OLS
print("\n--- Model 2: Basic DiD (Weighted by PERWT, Robust SE) ---")
model2 = sm.WLS(df['FT'], X, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# =============================================================================
# 4. Extended Models with Covariates
# =============================================================================
print("\n" + "="*80)
print("SECTION 4: EXTENDED MODELS WITH COVARIATES")
print("="*80)

# Prepare covariates
# Sex: 1=Male, 2=Female in IPUMS
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Education recode - create dummies
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Marital status - married vs not
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Model 3: With demographic covariates (unweighted)
print("\n--- Model 3: DiD with Covariates (Unweighted, Robust SE) ---")
covariates = ['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE', 'MARRIED',
              'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_TWOYEAR', 'EDUC_BA']
X3 = sm.add_constant(df[covariates])
model3 = sm.OLS(df['FT'], X3).fit(cov_type='HC1')
print(model3.summary())

# Model 4: With covariates (weighted)
print("\n--- Model 4: DiD with Covariates (Weighted, Robust SE) ---")
model4 = sm.WLS(df['FT'], X3, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# =============================================================================
# 5. State-Clustered Standard Errors
# =============================================================================
print("\n" + "="*80)
print("SECTION 5: STATE-CLUSTERED STANDARD ERRORS")
print("="*80)

# Model 5: Basic DiD with state-clustered SE
print("\n--- Model 5: Basic DiD (Clustered SE by State) ---")
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
model5 = sm.OLS(df['FT'], X).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model5.summary())

# Model 6: With covariates and state-clustered SE
print("\n--- Model 6: DiD with Covariates (Clustered SE by State) ---")
model6 = sm.OLS(df['FT'], X3).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model6.summary())

# =============================================================================
# 6. Year Fixed Effects Model
# =============================================================================
print("\n" + "="*80)
print("SECTION 6: YEAR FIXED EFFECTS MODEL")
print("="*80)

# Create year dummies (excluding 2008 as reference)
years = sorted(df['YEAR'].unique())
for year in years[1:]:  # Skip first year as reference
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

year_vars = [f'YEAR_{year}' for year in years[1:]]

# Model 7: With year fixed effects
print("\n--- Model 7: DiD with Year Fixed Effects (Robust SE) ---")
X7_vars = ['ELIGIBLE'] + year_vars + ['ELIGIBLE_AFTER']
X7 = sm.add_constant(df[X7_vars])
model7 = sm.OLS(df['FT'], X7).fit(cov_type='HC1')
print(model7.summary())

# =============================================================================
# 7. Heterogeneity Analysis by Sex
# =============================================================================
print("\n" + "="*80)
print("SECTION 7: HETEROGENEITY ANALYSIS BY SEX")
print("="*80)

# Males only
print("\n--- DiD for Males Only ---")
df_male = df[df['SEX'] == 1]
X_male = sm.add_constant(df_male[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
model_male = sm.OLS(df_male['FT'], X_male).fit(cov_type='HC1')
print(f"N = {len(df_male):,}")
print(f"Treatment effect: {model_male.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model_male.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_male.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_male.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Females only
print("\n--- DiD for Females Only ---")
df_female = df[df['SEX'] == 2]
X_female = sm.add_constant(df_female[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
model_female = sm.OLS(df_female['FT'], X_female).fit(cov_type='HC1')
print(f"N = {len(df_female):,}")
print(f"Treatment effect: {model_female.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model_female.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_female.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_female.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# =============================================================================
# 8. Pre-Trend Analysis
# =============================================================================
print("\n" + "="*80)
print("SECTION 8: PRE-TREND ANALYSIS")
print("="*80)

# Test for parallel trends in pre-period
df_pre = df[df['AFTER'] == 0].copy()

print("\n--- Pre-Period Trends by Year ---")
pre_trends = df_pre.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
pre_trends.columns = ['Control', 'Treatment']
pre_trends['Difference'] = pre_trends['Treatment'] - pre_trends['Control']
print(pre_trends.round(4))

# Test: Year interaction in pre-period
print("\n--- Pre-Trend Test: Year x Eligible Interactions ---")
df_pre['YEAR_CENTERED'] = df_pre['YEAR'] - 2008
X_pretrend = sm.add_constant(df_pre[['ELIGIBLE', 'YEAR_CENTERED']])
df_pre['ELIG_TREND'] = df_pre['ELIGIBLE'] * df_pre['YEAR_CENTERED']
X_pretrend = sm.add_constant(df_pre[['ELIGIBLE', 'YEAR_CENTERED', 'ELIG_TREND']])
model_pretrend = sm.OLS(df_pre['FT'], X_pretrend).fit(cov_type='HC1')
print(f"Eligible x Year trend coefficient: {model_pretrend.params['ELIG_TREND']:.4f}")
print(f"SE: {model_pretrend.bse['ELIG_TREND']:.4f}")
print(f"p-value: {model_pretrend.pvalues['ELIG_TREND']:.4f}")

# =============================================================================
# 9. Summary of All Results
# =============================================================================
print("\n" + "="*80)
print("SECTION 9: SUMMARY OF RESULTS")
print("="*80)

print("\n" + "-"*80)
print("MAIN RESULTS TABLE")
print("-"*80)
print(f"{'Model':<45} {'Estimate':>10} {'SE':>10} {'95% CI':>20}")
print("-"*80)

results = [
    ("1. Basic DiD (unweighted, robust SE)", model1.params['ELIGIBLE_AFTER'], model1.bse['ELIGIBLE_AFTER'], model1.conf_int().loc['ELIGIBLE_AFTER']),
    ("2. Basic DiD (weighted, robust SE)", model2.params['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'], model2.conf_int().loc['ELIGIBLE_AFTER']),
    ("3. With covariates (unweighted)", model3.params['ELIGIBLE_AFTER'], model3.bse['ELIGIBLE_AFTER'], model3.conf_int().loc['ELIGIBLE_AFTER']),
    ("4. With covariates (weighted)", model4.params['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'], model4.conf_int().loc['ELIGIBLE_AFTER']),
    ("5. Basic DiD (state-clustered SE)", model5.params['ELIGIBLE_AFTER'], model5.bse['ELIGIBLE_AFTER'], model5.conf_int().loc['ELIGIBLE_AFTER']),
    ("6. With covariates (state-clustered)", model6.params['ELIGIBLE_AFTER'], model6.bse['ELIGIBLE_AFTER'], model6.conf_int().loc['ELIGIBLE_AFTER']),
    ("7. Year fixed effects", model7.params['ELIGIBLE_AFTER'], model7.bse['ELIGIBLE_AFTER'], model7.conf_int().loc['ELIGIBLE_AFTER']),
]

for name, est, se, ci in results:
    ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    print(f"{name:<45} {est:>10.4f} {se:>10.4f} {ci_str:>20}")

print("-"*80)

# =============================================================================
# 10. Preferred Estimate
# =============================================================================
print("\n" + "="*80)
print("SECTION 10: PREFERRED ESTIMATE")
print("="*80)

# Model 2 (weighted, robust SE) is the preferred specification
preferred = model2
print("\nPreferred Model: Basic DiD with person weights and robust standard errors")
print("\nRationale:")
print("- Uses ACS survey weights for population-representative estimates")
print("- Heteroskedasticity-robust standard errors account for non-constant variance")
print("- Parsimonious specification without potentially endogenous covariates")
print("- DiD design controls for time-invariant unobserved heterogeneity")

print("\n" + "-"*60)
print("PREFERRED ESTIMATE DETAILS")
print("-"*60)
print(f"Treatment Effect (ELIGIBLE x AFTER): {preferred.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust Standard Error: {preferred.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {preferred.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"p-value: {preferred.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence Interval: [{preferred.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {preferred.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"\nSample Size: {int(df['PERWT'].sum()):,} (weighted), {len(df):,} (unweighted)")
print("-"*60)

# Interpretation
effect_pct = preferred.params['ELIGIBLE_AFTER'] * 100
print(f"\nInterpretation:")
print(f"DACA eligibility is associated with a {effect_pct:.2f} percentage point")
if effect_pct > 0:
    print("increase in the probability of full-time employment.")
else:
    print("decrease in the probability of full-time employment.")

# =============================================================================
# 11. Save Results for LaTeX
# =============================================================================
print("\n" + "="*80)
print("SECTION 11: EXPORTING RESULTS")
print("="*80)

# Create results dataframe for export
results_df = pd.DataFrame({
    'Model': ['(1) Basic', '(2) Weighted', '(3) +Covariates', '(4) Weighted+Cov',
              '(5) Clustered', '(6) Clust+Cov', '(7) Year FE'],
    'Estimate': [r[1] for r in results],
    'SE': [r[2] for r in results],
    'CI_Lower': [r[3][0] for r in results],
    'CI_Upper': [r[3][1] for r in results]
})
results_df.to_csv('regression_results.csv', index=False)
print("Regression results saved to: regression_results.csv")

# Employment rates table
ft_table = pd.DataFrame({
    'Group': ['Control (31-35)', 'Control (31-35)', 'Treatment (26-30)', 'Treatment (26-30)'],
    'Period': ['Pre-DACA', 'Post-DACA', 'Pre-DACA', 'Post-DACA'],
    'Unweighted': [ft_rates.iloc[0,0], ft_rates.iloc[0,1], ft_rates.iloc[1,0], ft_rates.iloc[1,1]],
    'Weighted': [ft_rates_weighted.iloc[0,0], ft_rates_weighted.iloc[0,1],
                 ft_rates_weighted.iloc[1,0], ft_rates_weighted.iloc[1,1]]
})
ft_table.to_csv('employment_rates.csv', index=False)
print("Employment rates saved to: employment_rates.csv")

# Yearly trends
yearly_rates.to_csv('yearly_trends.csv')
print("Yearly trends saved to: yearly_trends.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
