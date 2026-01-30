"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among ethnically Hispanic-Mexican Mexican-born individuals in the United States

This script performs a difference-in-differences analysis comparing
individuals aged 26-30 at policy implementation (treatment group, ELIGIBLE=1)
to those aged 31-35 (control group, ELIGIBLE=0).
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
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# ============================================================
# DATA PREPARATION
# ============================================================

# Create interaction term for DiD
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Create covariates
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Education dummies
edu_dummies = pd.get_dummies(df['EDUC_RECODE'], prefix='EDU', drop_first=True)
df = pd.concat([df, edu_dummies], axis=1)

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_')

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

print("\n--- Sample sizes ---")
print(f"Total observations: {len(df):,}")
print(f"Total weighted sample: {df['PERWT'].sum():,.0f}")
print("\nBy group:")
print(pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True))

print("\n--- Full-time employment rates ---")
print("\nUnweighted:")
print(df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack())

print("\nWeighted:")
def weighted_mean(group):
    return (group['FT'] * group['PERWT']).sum() / group['PERWT'].sum()
print(df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean, include_groups=False).unstack())

print("\n--- Demographics by eligibility ---")
print("\nSEX (1=Male, 2=Female):")
print(pd.crosstab(df['ELIGIBLE'], df['SEX'], normalize='index'))

print("\nEducation distribution:")
print(df.groupby('ELIGIBLE')['EDUC_RECODE'].value_counts(normalize=True).unstack().fillna(0))

print("\nAge in June 2012:")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe())

# ============================================================
# MAIN ANALYSIS: DIFFERENCE-IN-DIFFERENCES
# ============================================================

print("\n" + "="*60)
print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*60)

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (unweighted) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: Basic DiD (weighted)
print("\n--- Model 2: Basic DiD (weighted) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: With individual covariates (weighted)
print("\n--- Model 3: DiD with covariates (weighted) ---")
formula3 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + HAS_CHILDREN + EDU_High_School_Degree + EDU_Less_than_High_School + EDU_Some_College + EDU_Two_Year_Degree'
model3 = smf.wls(formula3, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: With state and year fixed effects
print("\n--- Model 4: DiD with state and year FE ---")
formula4 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + HAS_CHILDREN + EDU_High_School_Degree + EDU_Less_than_High_School + EDU_Some_College + EDU_Two_Year_Degree + C(YEAR) + C(STATEFIP)'
model4 = smf.wls(formula4, data=df, weights=df['PERWT']).fit(cov_type='HC1')
key_vars = ['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'AGE']
print("Key coefficients:")
for var in key_vars:
    coef = model4.params[var]
    se = model4.bse[var]
    pval = model4.pvalues[var]
    print(f"  {var}: {coef:.4f} ({se:.4f}), p={pval:.4f}")

# ============================================================
# ROBUSTNESS CHECKS
# ============================================================

print("\n" + "="*60)
print("ROBUSTNESS CHECKS")
print("="*60)

# Clustered standard errors at state level
print("\n--- Clustered SEs at state level ---")
model_cluster = smf.wls(formula3, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
coef = model_cluster.params['ELIGIBLE_AFTER']
se = model_cluster.bse['ELIGIBLE_AFTER']
pval = model_cluster.pvalues['ELIGIBLE_AFTER']
print(f"ELIGIBLE_AFTER: {coef:.4f} ({se:.4f}), p={pval:.4f}")

# By gender
print("\n--- By Gender ---")
for gender, name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == gender].copy()
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    coef = model_sub.params['ELIGIBLE_AFTER']
    se = model_sub.bse['ELIGIBLE_AFTER']
    pval = model_sub.pvalues['ELIGIBLE_AFTER']
    print(f"  {name}: DiD={coef:.4f} ({se:.4f}), p={pval:.4f}, N={len(df_sub)}")

# Pre-trends test
print("\n--- Pre-trend test ---")
df_pre = df[df['YEAR'] <= 2011].copy()
df_pre['YEAR_TREND'] = df_pre['YEAR'] - 2008
df_pre['ELIGIBLE_TREND'] = df_pre['ELIGIBLE'] * df_pre['YEAR_TREND']
model_trend = smf.wls('FT ~ ELIGIBLE + YEAR_TREND + ELIGIBLE_TREND',
                      data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
coef = model_trend.params['ELIGIBLE_TREND']
se = model_trend.bse['ELIGIBLE_TREND']
pval = model_trend.pvalues['ELIGIBLE_TREND']
print(f"Differential pre-trend (ELIGIBLE x TREND): {coef:.4f} ({se:.4f}), p={pval:.4f}")

# Event study
print("\n--- Event Study (relative to 2011) ---")
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_Y{year}'] = df['ELIGIBLE'] * (df['YEAR'] == year).astype(int)

year_vars = ['ELIGIBLE_Y2008', 'ELIGIBLE_Y2009', 'ELIGIBLE_Y2010',
             'ELIGIBLE_Y2013', 'ELIGIBLE_Y2014', 'ELIGIBLE_Y2015', 'ELIGIBLE_Y2016']
formula_event = 'FT ~ ELIGIBLE + C(YEAR) + ' + ' + '.join(year_vars)
model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("Year-specific effects:")
for var in year_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    year = var.replace('ELIGIBLE_Y', '')
    marker = "*" if pval < 0.05 else " "
    print(f"  {year}: {coef:7.4f} ({se:.4f}){marker}")

# State policy controls
print("\n--- State policy controls ---")
formula_pol = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + DRIVERSLICENSES + INSTATETUITION + EVERIFY + FEMALE + MARRIED + AGE'
model_pol = smf.wls(formula_pol, data=df, weights=df['PERWT']).fit(cov_type='HC1')
coef = model_pol.params['ELIGIBLE_AFTER']
se = model_pol.bse['ELIGIBLE_AFTER']
pval = model_pol.pvalues['ELIGIBLE_AFTER']
print(f"ELIGIBLE_AFTER: {coef:.4f} ({se:.4f}), p={pval:.4f}")

# ============================================================
# PREFERRED ESTIMATE SUMMARY
# ============================================================

print("\n" + "="*60)
print("PREFERRED ESTIMATE SUMMARY")
print("="*60)

# Preferred model: Model 3 (weighted DiD with covariates)
preferred = model3

coef = preferred.params['ELIGIBLE_AFTER']
se = preferred.bse['ELIGIBLE_AFTER']
ci_lo, ci_hi = preferred.conf_int().loc['ELIGIBLE_AFTER']
pval = preferred.pvalues['ELIGIBLE_AFTER']

print(f"\nEffect of DACA eligibility on full-time employment:")
print(f"  Coefficient: {coef:.4f}")
print(f"  Standard Error: {se:.4f}")
print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  p-value: {pval:.4f}")
print(f"  Sample size: {int(preferred.nobs):,}")
print(f"  R-squared: {preferred.rsquared:.4f}")

print("\nInterpretation:")
print(f"  DACA eligibility increased the probability of full-time employment")
print(f"  by approximately {coef*100:.1f} percentage points.")

print("\n" + "="*60)
print("Analysis complete.")
print("="*60)
