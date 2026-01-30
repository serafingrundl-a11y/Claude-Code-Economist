"""
DACA Replication Study - Robustness Checks and Heterogeneity Analysis
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Create covariate dummies
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC'] >= 7).fillna(0).astype(int)
df['EDUC_BA'] = (df['EDUC'] >= 10).fillna(0).astype(int)
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

print("=" * 80)
print("ROBUSTNESS CHECKS AND HETEROGENEITY ANALYSIS")
print("=" * 80)

#############################################################################
# 1. PARALLEL TRENDS CHECK - Year-by-Year Effects
#############################################################################
print("\n" + "=" * 80)
print("1. PARALLEL TRENDS CHECK - EVENT STUDY")
print("=" * 80)

# Create year-by-eligible interactions
years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    if year != 2008:  # 2008 is the reference year
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
        df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Event study formula
formula_es = 'FT ~ ELIGIBLE + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 + ELIGIBLE_YEAR_2011 + ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016 + FEMALE + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE'

model_es = smf.wls(formula_es, data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("\nEvent Study Results (Reference Year: 2008)")
print("-" * 60)
print(f"{'Year':<8} {'Coef':>10} {'SE':>10} {'p-value':>10}")
print("-" * 60)

# Pre-treatment periods
print("PRE-TREATMENT:")
for year in [2009, 2010, 2011]:
    coef = model_es.params[f'ELIGIBLE_YEAR_{year}']
    se = model_es.bse[f'ELIGIBLE_YEAR_{year}']
    p = model_es.pvalues[f'ELIGIBLE_YEAR_{year}']
    stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
    print(f"{year:<8} {coef:>10.4f} {se:>10.4f} {p:>10.4f} {stars}")

# Post-treatment periods
print("\nPOST-TREATMENT:")
for year in [2013, 2014, 2015, 2016]:
    coef = model_es.params[f'ELIGIBLE_YEAR_{year}']
    se = model_es.bse[f'ELIGIBLE_YEAR_{year}']
    p = model_es.pvalues[f'ELIGIBLE_YEAR_{year}']
    stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
    print(f"{year:<8} {coef:>10.4f} {se:>10.4f} {p:>10.4f} {stars}")

#############################################################################
# 2. HETEROGENEITY BY SEX
#############################################################################
print("\n" + "=" * 80)
print("2. HETEROGENEITY BY SEX")
print("=" * 80)

# Males only
df_male = df[df['SEX'] == 1].copy()
model_male = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                     data=df_male, weights=df_male['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})

# Females only
df_female = df[df['SEX'] == 2].copy()
model_female = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                       data=df_female, weights=df_female['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})

print(f"\nMales (N={len(df_male):,}):")
print(f"  DiD Estimate: {model_male.params['ELIGIBLE_AFTER']:.4f}")
print(f"  SE (clustered): {model_male.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model_male.pvalues['ELIGIBLE_AFTER']:.4f}")

print(f"\nFemales (N={len(df_female):,}):")
print(f"  DiD Estimate: {model_female.params['ELIGIBLE_AFTER']:.4f}")
print(f"  SE (clustered): {model_female.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model_female.pvalues['ELIGIBLE_AFTER']:.4f}")

#############################################################################
# 3. HETEROGENEITY BY EDUCATION
#############################################################################
print("\n" + "=" * 80)
print("3. HETEROGENEITY BY EDUCATION")
print("=" * 80)

# Less than high school vs HS+ based on EDUC variable
df['NO_HS'] = (df['EDUC'] < 6).astype(int)

# No HS
df_nohs = df[df['EDUC'] < 6].copy()
if len(df_nohs) > 100:
    model_nohs = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE',
                         data=df_nohs, weights=df_nohs['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_nohs['STATEFIP']})
    print(f"\nLess than High School (N={len(df_nohs):,}):")
    print(f"  DiD Estimate: {model_nohs.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  SE (clustered): {model_nohs.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"  p-value: {model_nohs.pvalues['ELIGIBLE_AFTER']:.4f}")
else:
    print(f"\nLess than High School: Too few observations (N={len(df_nohs)})")

# HS or more
df_hs = df[df['EDUC'] >= 6].copy()
model_hs = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                   data=df_hs, weights=df_hs['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_hs['STATEFIP']})
print(f"\nHigh School or More (N={len(df_hs):,}):")
print(f"  DiD Estimate: {model_hs.params['ELIGIBLE_AFTER']:.4f}")
print(f"  SE (clustered): {model_hs.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model_hs.pvalues['ELIGIBLE_AFTER']:.4f}")

#############################################################################
# 4. HETEROGENEITY BY MARITAL STATUS
#############################################################################
print("\n" + "=" * 80)
print("4. HETEROGENEITY BY MARITAL STATUS")
print("=" * 80)

# Married
df_married = df[df['MARRIED'] == 1].copy()
model_married = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                        data=df_married, weights=df_married['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_married['STATEFIP']})

# Not married
df_notmarried = df[df['MARRIED'] == 0].copy()
model_notmarried = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                           data=df_notmarried, weights=df_notmarried['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_notmarried['STATEFIP']})

print(f"\nMarried (N={len(df_married):,}):")
print(f"  DiD Estimate: {model_married.params['ELIGIBLE_AFTER']:.4f}")
print(f"  SE (clustered): {model_married.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model_married.pvalues['ELIGIBLE_AFTER']:.4f}")

print(f"\nNot Married (N={len(df_notmarried):,}):")
print(f"  DiD Estimate: {model_notmarried.params['ELIGIBLE_AFTER']:.4f}")
print(f"  SE (clustered): {model_notmarried.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model_notmarried.pvalues['ELIGIBLE_AFTER']:.4f}")

#############################################################################
# 5. PLACEBO TEST - Fake Treatment Year (2010)
#############################################################################
print("\n" + "=" * 80)
print("5. PLACEBO TEST - Fake Treatment in 2010")
print("=" * 80)

# Use only pre-treatment data
df_preonly = df[df['YEAR'] <= 2011].copy()
df_preonly['FAKE_AFTER'] = (df_preonly['YEAR'] >= 2010).astype(int)
df_preonly['ELIGIBLE_FAKE_AFTER'] = df_preonly['ELIGIBLE'] * df_preonly['FAKE_AFTER']

model_placebo = smf.wls('FT ~ ELIGIBLE + FAKE_AFTER + ELIGIBLE_FAKE_AFTER + FEMALE + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                        data=df_preonly, weights=df_preonly['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_preonly['STATEFIP']})

print(f"\nPlacebo test (fake treatment at 2010, using only 2008-2011 data):")
print(f"  N = {len(df_preonly):,}")
print(f"  Placebo DiD Estimate: {model_placebo.params['ELIGIBLE_FAKE_AFTER']:.4f}")
print(f"  SE (clustered): {model_placebo.bse['ELIGIBLE_FAKE_AFTER']:.4f}")
print(f"  p-value: {model_placebo.pvalues['ELIGIBLE_FAKE_AFTER']:.4f}")

#############################################################################
# 6. TRIPLE DIFFERENCE (DiDiD) by Sex
#############################################################################
print("\n" + "=" * 80)
print("6. TRIPLE DIFFERENCE (DiDiD) BY SEX")
print("=" * 80)

# Create triple interaction
df['ELIGIBLE_AFTER_FEMALE'] = df['ELIGIBLE'] * df['AFTER'] * df['FEMALE']
df['ELIGIBLE_FEMALE'] = df['ELIGIBLE'] * df['FEMALE']
df['AFTER_FEMALE'] = df['AFTER'] * df['FEMALE']

model_ddd = smf.wls('FT ~ ELIGIBLE + AFTER + FEMALE + ELIGIBLE_AFTER + ELIGIBLE_FEMALE + AFTER_FEMALE + ELIGIBLE_AFTER_FEMALE + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                    data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"\nDiD effect for Males (ELIGIBLE_AFTER): {model_ddd.params['ELIGIBLE_AFTER']:.4f}")
print(f"  SE: {model_ddd.bse['ELIGIBLE_AFTER']:.4f}, p-value: {model_ddd.pvalues['ELIGIBLE_AFTER']:.4f}")

print(f"\nDifferential effect for Females (ELIGIBLE_AFTER_FEMALE): {model_ddd.params['ELIGIBLE_AFTER_FEMALE']:.4f}")
print(f"  SE: {model_ddd.bse['ELIGIBLE_AFTER_FEMALE']:.4f}, p-value: {model_ddd.pvalues['ELIGIBLE_AFTER_FEMALE']:.4f}")

#############################################################################
# 7. BALANCE CHECK - Covariates in Pre-Period
#############################################################################
print("\n" + "=" * 80)
print("7. BALANCE CHECK - Pre-Period Covariate Differences")
print("=" * 80)

df_pre = df[df['AFTER'] == 0].copy()

covariates = ['AGE', 'FEMALE', 'MARRIED', 'EDUC_SOMECOLL', 'EDUC_BA', 'HAS_CHILDREN']

print(f"\n{'Variable':<15} {'Treatment':>12} {'Control':>12} {'Difference':>12}")
print("-" * 55)

for cov in covariates:
    treat_mean = np.average(df_pre[df_pre['ELIGIBLE'] == 1][cov], weights=df_pre[df_pre['ELIGIBLE'] == 1]['PERWT'])
    ctrl_mean = np.average(df_pre[df_pre['ELIGIBLE'] == 0][cov], weights=df_pre[df_pre['ELIGIBLE'] == 0]['PERWT'])
    diff = treat_mean - ctrl_mean
    print(f"{cov:<15} {treat_mean:>12.4f} {ctrl_mean:>12.4f} {diff:>12.4f}")

#############################################################################
# 8. WEIGHTED MEAN FT BY YEAR AND GROUP (for plotting)
#############################################################################
print("\n" + "=" * 80)
print("8. WEIGHTED MEAN FT BY YEAR AND GROUP (for parallel trends plot)")
print("=" * 80)

print(f"\n{'Year':<6} {'Treatment':>12} {'Control':>12} {'Difference':>12}")
print("-" * 45)

for year in years:
    df_year = df[df['YEAR'] == year]
    treat_ft = np.average(df_year[df_year['ELIGIBLE'] == 1]['FT'],
                          weights=df_year[df_year['ELIGIBLE'] == 1]['PERWT'])
    ctrl_ft = np.average(df_year[df_year['ELIGIBLE'] == 0]['FT'],
                         weights=df_year[df_year['ELIGIBLE'] == 0]['PERWT'])
    diff = treat_ft - ctrl_ft
    print(f"{year:<6} {treat_ft:>12.4f} {ctrl_ft:>12.4f} {diff:>12.4f}")

#############################################################################
# 9. SAMPLE SIZE BY YEAR AND GROUP
#############################################################################
print("\n" + "=" * 80)
print("9. SAMPLE SIZE BY YEAR AND GROUP")
print("=" * 80)

crosstab = pd.crosstab(df['YEAR'], df['ELIGIBLE'], margins=True)
print(crosstab)
