"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
ethnically Hispanic-Mexican Mexican-born people living in the United States.

Difference-in-Differences Design:
- Treatment group: Eligible individuals aged 26-30 at time of DACA (June 2012)
- Control group: Individuals aged 31-35 at time of DACA (otherwise would be eligible)
- Pre-period: 2008-2011
- Post-period: 2013-2016 (2012 excluded)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 70)
print("DACA REPLICATION ANALYSIS")
print("=" * 70)

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Basic data summary
print("\n1. DATA OVERVIEW")
print("-" * 50)
print(f"Total observations: {len(df):,}")
print(f"Years covered: {sorted(df['YEAR'].unique())}")
print(f"Treatment group (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum():,} observations")
print(f"Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum():,} observations")

# Create interaction term
df['TREAT_POST'] = df['ELIGIBLE'] * df['AFTER']

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n2. SUMMARY STATISTICS")
print("-" * 50)

# By group and period
summary = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['count', 'mean', 'std'],
    'PERWT': ['sum', 'mean'],
    'AGE': 'mean',
    'SEX': lambda x: (x == 2).mean()  # Proportion female (SEX=2 is female in IPUMS)
}).round(4)
print("\nSummary by Treatment Group and Period:")
print(summary)

# Calculate unweighted DiD
print("\n3. SIMPLE DIFFERENCE-IN-DIFFERENCES (Unweighted)")
print("-" * 50)

# Mean FT by group/period
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean()
print(f"Treatment group, Pre (ELIGIBLE=1, AFTER=0):  {ft_means[(1, 0)]:.4f}")
print(f"Treatment group, Post (ELIGIBLE=1, AFTER=1): {ft_means[(1, 1)]:.4f}")
print(f"Control group, Pre (ELIGIBLE=0, AFTER=0):    {ft_means[(0, 0)]:.4f}")
print(f"Control group, Post (ELIGIBLE=0, AFTER=1):   {ft_means[(0, 1)]:.4f}")

# DiD calculation
did_treat = ft_means[(1, 1)] - ft_means[(1, 0)]
did_control = ft_means[(0, 1)] - ft_means[(0, 0)]
did_estimate = did_treat - did_control

print(f"\nChange in Treatment group: {did_treat:.4f}")
print(f"Change in Control group:   {did_control:.4f}")
print(f"Difference-in-Differences: {did_estimate:.4f}")

# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================
print("\n4. REGRESSION ANALYSIS")
print("-" * 50)

# Model 1: Basic DiD (OLS, unweighted)
print("\n--- Model 1: Basic DiD (OLS, no weights, no controls) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + TREAT_POST', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: Basic DiD with survey weights
print("\n--- Model 2: Basic DiD (WLS with PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + TREAT_POST', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: With year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects (WLS) ---")
df['YEAR_CAT'] = pd.Categorical(df['YEAR'])
model3 = smf.wls('FT ~ ELIGIBLE + TREAT_POST + C(YEAR)', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: With demographic controls
print("\n--- Model 4: DiD with Demographic Controls (WLS) ---")
# Recode SEX to binary (Female = 1 if SEX == 2)
df['FEMALE'] = (df['SEX'] == 2).astype(int)
# Create age squared
df['AGE_SQ'] = df['AGE'] ** 2
# Married indicator (MARST = 1 is married, spouse present; 2 is married, spouse absent)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

model4 = smf.wls('FT ~ ELIGIBLE + TREAT_POST + C(YEAR) + AGE + AGE_SQ + FEMALE + MARRIED + NCHILD',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nCoefficients for key variables:")
print(f"TREAT_POST: {model4.params['TREAT_POST']:.4f} (SE: {model4.bse['TREAT_POST']:.4f})")
print(f"t-stat: {model4.tvalues['TREAT_POST']:.3f}, p-value: {model4.pvalues['TREAT_POST']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['TREAT_POST', 0]:.4f}, {model4.conf_int().loc['TREAT_POST', 1]:.4f}]")

# Model 5: Full model with education and state FE
print("\n--- Model 5: Full Model with Education and State FE (WLS) ---")
model5 = smf.wls('FT ~ ELIGIBLE + TREAT_POST + C(YEAR) + C(STATEFIP) + AGE + AGE_SQ + FEMALE + MARRIED + NCHILD + C(EDUC_RECODE)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nPreferred Specification Coefficients for key variables:")
print(f"TREAT_POST: {model5.params['TREAT_POST']:.4f} (SE: {model5.bse['TREAT_POST']:.4f})")
print(f"t-stat: {model5.tvalues['TREAT_POST']:.3f}, p-value: {model5.pvalues['TREAT_POST']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['TREAT_POST', 0]:.4f}, {model5.conf_int().loc['TREAT_POST', 1]:.4f}]")

# ============================================================================
# ROBUSTNESS: LINEAR PROBABILITY MODEL WITH CLUSTERED SE
# ============================================================================
print("\n5. ROBUSTNESS CHECKS")
print("-" * 50)

# Cluster by state - need to handle missing values
print("\n--- Model 6: Preferred Model with State-Clustered SE ---")
# Create a clean dataset without missing values for clustering
df_clean = df.dropna(subset=['FT', 'ELIGIBLE', 'TREAT_POST', 'YEAR', 'STATEFIP', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD', 'EDUC_RECODE']).copy()
model6 = smf.wls('FT ~ ELIGIBLE + TREAT_POST + C(YEAR) + C(STATEFIP) + AGE + AGE_SQ + FEMALE + MARRIED + NCHILD + C(EDUC_RECODE)',
                 data=df_clean, weights=df_clean['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_clean['STATEFIP']})
print(f"TREAT_POST: {model6.params['TREAT_POST']:.4f} (SE: {model6.bse['TREAT_POST']:.4f})")
print(f"t-stat: {model6.tvalues['TREAT_POST']:.3f}, p-value: {model6.pvalues['TREAT_POST']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['TREAT_POST', 0]:.4f}, {model6.conf_int().loc['TREAT_POST', 1]:.4f}]")

# ============================================================================
# PARALLEL TRENDS CHECK - Pre-treatment period trends
# ============================================================================
print("\n6. PARALLEL TRENDS ANALYSIS")
print("-" * 50)

# Create year-by-treatment interactions for event study
pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_TREAT'] = pre_data['YEAR'].astype(str) + '_ELIGIBLE' + pre_data['ELIGIBLE'].astype(str)

# Calculate weighted means by year and treatment
yearly_means = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
yearly_means.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\nWeighted FT Rates by Year and Group:")
print(yearly_means.round(4))

# Test for pre-trends using pre-period only
print("\n--- Pre-trend Test (Pre-period only, 2008-2011) ---")
pre_df = df[df['AFTER'] == 0].copy()
pre_df['YEAR_CENTERED'] = pre_df['YEAR'] - 2011  # Center at last pre-period year
pre_df['YEAR_TREAT_INTERACT'] = pre_df['YEAR_CENTERED'] * pre_df['ELIGIBLE']

pre_model = smf.wls('FT ~ ELIGIBLE + YEAR_CENTERED + YEAR_TREAT_INTERACT',
                    data=pre_df, weights=pre_df['PERWT']).fit(cov_type='HC1')
print(f"Year x Treatment interaction: {pre_model.params['YEAR_TREAT_INTERACT']:.4f}")
print(f"SE: {pre_model.bse['YEAR_TREAT_INTERACT']:.4f}")
print(f"p-value: {pre_model.pvalues['YEAR_TREAT_INTERACT']:.4f}")
if pre_model.pvalues['YEAR_TREAT_INTERACT'] > 0.05:
    print("Parallel trends assumption supported (no significant differential pre-trend)")
else:
    print("Warning: Evidence of differential pre-trends")

# ============================================================================
# EVENT STUDY ANALYSIS
# ============================================================================
print("\n7. EVENT STUDY / DYNAMIC EFFECTS")
print("-" * 50)

# Create year dummies interacted with treatment
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions (omit 2011 as reference year)
df['T_2008'] = df['YEAR_2008'] * df['ELIGIBLE']
df['T_2009'] = df['YEAR_2009'] * df['ELIGIBLE']
df['T_2010'] = df['YEAR_2010'] * df['ELIGIBLE']
df['T_2013'] = df['YEAR_2013'] * df['ELIGIBLE']
df['T_2014'] = df['YEAR_2014'] * df['ELIGIBLE']
df['T_2015'] = df['YEAR_2015'] * df['ELIGIBLE']
df['T_2016'] = df['YEAR_2016'] * df['ELIGIBLE']

event_model = smf.wls('FT ~ ELIGIBLE + C(YEAR) + T_2008 + T_2009 + T_2010 + T_2013 + T_2014 + T_2015 + T_2016',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Reference: 2011):")
print("-" * 40)
event_vars = ['T_2008', 'T_2009', 'T_2010', 'T_2013', 'T_2014', 'T_2015', 'T_2016']
for var in event_vars:
    coef = event_model.params[var]
    se = event_model.bse[var]
    pval = event_model.pvalues[var]
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))
    print(f"{var}: {coef:7.4f} (SE: {se:.4f}) {sig}")

# ============================================================================
# HETEROGENEITY ANALYSIS
# ============================================================================
print("\n8. HETEROGENEITY ANALYSIS")
print("-" * 50)

# By gender
print("\n--- Effect by Gender ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    sub = df[df['SEX'] == sex_val]
    sub_model = smf.wls('FT ~ ELIGIBLE + TREAT_POST + C(YEAR)',
                        data=sub, weights=sub['PERWT']).fit(cov_type='HC1')
    print(f"{sex_name}: DiD = {sub_model.params['TREAT_POST']:.4f} (SE: {sub_model.bse['TREAT_POST']:.4f}), p = {sub_model.pvalues['TREAT_POST']:.4f}")

# By education
print("\n--- Effect by Education Level ---")
for edu in df['EDUC_RECODE'].dropna().unique():
    sub = df[df['EDUC_RECODE'] == edu]
    if len(sub) > 100:
        sub_model = smf.wls('FT ~ ELIGIBLE + TREAT_POST + C(YEAR)',
                            data=sub, weights=sub['PERWT']).fit(cov_type='HC1')
        print(f"{edu}: DiD = {sub_model.params['TREAT_POST']:.4f} (SE: {sub_model.bse['TREAT_POST']:.4f}), N = {len(sub)}")

# ============================================================================
# FINAL PREFERRED ESTIMATE
# ============================================================================
print("\n" + "=" * 70)
print("PREFERRED ESTIMATE SUMMARY")
print("=" * 70)
print(f"\nModel: WLS with PERWT weights, Year and State FE, Demographic Controls")
print(f"Sample Size: {len(df):,}")
print(f"\nTreatment Effect (TREAT_POST coefficient):")
print(f"  Point Estimate: {model5.params['TREAT_POST']:.4f}")
print(f"  Standard Error: {model5.bse['TREAT_POST']:.4f}")
print(f"  95% CI: [{model5.conf_int().loc['TREAT_POST', 0]:.4f}, {model5.conf_int().loc['TREAT_POST', 1]:.4f}]")
print(f"  p-value: {model5.pvalues['TREAT_POST']:.4f}")

# With clustered SE
print(f"\nWith State-Clustered SE:")
print(f"  Point Estimate: {model6.params['TREAT_POST']:.4f}")
print(f"  Standard Error: {model6.bse['TREAT_POST']:.4f}")
print(f"  95% CI: [{model6.conf_int().loc['TREAT_POST', 0]:.4f}, {model6.conf_int().loc['TREAT_POST', 1]:.4f}]")
print(f"  p-value: {model6.pvalues['TREAT_POST']:.4f}")

# ============================================================================
# SAVE RESULTS FOR REPORT
# ============================================================================
results = {
    'n_obs': len(df),
    'n_treated': (df['ELIGIBLE']==1).sum(),
    'n_control': (df['ELIGIBLE']==0).sum(),
    'did_simple': did_estimate,
    'model1_coef': model1.params['TREAT_POST'],
    'model1_se': model1.bse['TREAT_POST'],
    'model2_coef': model2.params['TREAT_POST'],
    'model2_se': model2.bse['TREAT_POST'],
    'model5_coef': model5.params['TREAT_POST'],
    'model5_se': model5.bse['TREAT_POST'],
    'model5_ci_low': model5.conf_int().loc['TREAT_POST', 0],
    'model5_ci_high': model5.conf_int().loc['TREAT_POST', 1],
    'model5_pval': model5.pvalues['TREAT_POST'],
    'model6_coef': model6.params['TREAT_POST'],
    'model6_se': model6.bse['TREAT_POST'],
    'model6_ci_low': model6.conf_int().loc['TREAT_POST', 0],
    'model6_ci_high': model6.conf_int().loc['TREAT_POST', 1],
    'model6_pval': model6.pvalues['TREAT_POST'],
    'yearly_means': yearly_means,
    'pre_trend_coef': pre_model.params['YEAR_TREAT_INTERACT'],
    'pre_trend_pval': pre_model.pvalues['YEAR_TREAT_INTERACT']
}

# Save to pickle for later use
import pickle
with open('analysis_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nResults saved to analysis_results.pkl")
print("\n" + "=" * 70)
