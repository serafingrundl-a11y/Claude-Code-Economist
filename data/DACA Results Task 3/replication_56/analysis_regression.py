"""
DACA Replication Study - Regression Analysis
Difference-in-Differences with various specifications
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print("=" * 80)
print("DACA REPLICATION STUDY - REGRESSION ANALYSIS")
print("=" * 80)

#############################################################################
# MODEL 1: Basic DiD without weights
#############################################################################
print("\n" + "=" * 80)
print("MODEL 1: Basic DiD (Unweighted, no covariates)")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Basic OLS
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary())

#############################################################################
# MODEL 2: DiD with survey weights
#############################################################################
print("\n" + "=" * 80)
print("MODEL 2: DiD with Survey Weights (PERWT)")
print("=" * 80)

# Weighted least squares
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary())

#############################################################################
# MODEL 3: DiD with covariates (weighted)
#############################################################################
print("\n" + "=" * 80)
print("MODEL 3: DiD with Covariates (Weighted)")
print("=" * 80)

# Create dummy variables for categorical variables
# SEX: 1=Male, 2=Female - create female dummy
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# MARST: Create married dummy (MARST 1-2 = married)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

# Education categories (based on EDUC)
# EDUC: 6=Grade 12, 7=1 yr college, 8=2 yrs, 10=4 yrs, 11=5+ yrs
df['EDUC_HS'] = (df['EDUC'] >= 6).astype(int)  # At least high school
df['EDUC_SOMECOLL'] = (df['EDUC'] >= 7).astype(int)  # Some college
df['EDUC_BA'] = (df['EDUC'] >= 10).astype(int)  # BA or more

# Handle NaN in EDUC
df['EDUC_HS'] = df['EDUC_HS'].fillna(0)
df['EDUC_SOMECOLL'] = df['EDUC_SOMECOLL'].fillna(0)
df['EDUC_BA'] = df['EDUC_BA'].fillna(0)

# Number of children
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Model with covariates
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                 data=df, weights=df['PERWT']).fit()
print(model3.summary())

#############################################################################
# MODEL 4: DiD with Year Fixed Effects
#############################################################################
print("\n" + "=" * 80)
print("MODEL 4: DiD with Year Fixed Effects (Weighted)")
print("=" * 80)

# Create year dummies
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

model4 = smf.wls('FT ~ ELIGIBLE + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIGIBLE_AFTER',
                 data=df, weights=df['PERWT']).fit()
print(model4.summary())

#############################################################################
# MODEL 5: Full model with Year FE and Covariates
#############################################################################
print("\n" + "=" * 80)
print("MODEL 5: Full Model (Year FE + Covariates, Weighted)")
print("=" * 80)

model5 = smf.wls('FT ~ ELIGIBLE + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIGIBLE_AFTER + FEMALE + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                 data=df, weights=df['PERWT']).fit()
print(model5.summary())

#############################################################################
# MODEL 6: Add State Fixed Effects
#############################################################################
print("\n" + "=" * 80)
print("MODEL 6: Full Model with State Fixed Effects (Weighted)")
print("=" * 80)

# Get state dummies
df_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df = pd.concat([df, df_dummies], axis=1)

# Create formula with state dummies
state_cols = [col for col in df.columns if col.startswith('STATE_')]
state_formula = ' + '.join(state_cols)

formula6 = f'FT ~ ELIGIBLE + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIGIBLE_AFTER + FEMALE + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE + {state_formula}'

model6 = smf.wls(formula6, data=df, weights=df['PERWT']).fit()

# Print only key coefficients
print("\nKey coefficients from Model 6:")
print(f"ELIGIBLE_AFTER (DiD estimate): {model6.params['ELIGIBLE_AFTER']:.6f}")
print(f"Standard Error: {model6.bse['ELIGIBLE_AFTER']:.6f}")
print(f"t-statistic: {model6.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")
print(f"\nR-squared: {model6.rsquared:.4f}")
print(f"Number of observations: {model6.nobs:.0f}")

#############################################################################
# SUMMARY TABLE
#############################################################################
print("\n" + "=" * 80)
print("SUMMARY OF DiD ESTIMATES ACROSS MODELS")
print("=" * 80)

models = [model1, model2, model3, model4, model5, model6]
model_names = ['Basic OLS', 'Weighted', 'Covariates', 'Year FE', 'Full', 'State FE']

print(f"{'Model':<15} {'Estimate':>10} {'Std Err':>10} {'t-stat':>10} {'p-value':>10} {'95% CI'}")
print("-" * 80)

for name, model in zip(model_names, models):
    est = model.params['ELIGIBLE_AFTER']
    se = model.bse['ELIGIBLE_AFTER']
    t = model.tvalues['ELIGIBLE_AFTER']
    p = model.pvalues['ELIGIBLE_AFTER']
    ci_low = model.conf_int().loc['ELIGIBLE_AFTER', 0]
    ci_high = model.conf_int().loc['ELIGIBLE_AFTER', 1]
    print(f"{name:<15} {est:>10.4f} {se:>10.4f} {t:>10.2f} {p:>10.4f} [{ci_low:.4f}, {ci_high:.4f}]")

#############################################################################
# ROBUST STANDARD ERRORS
#############################################################################
print("\n" + "=" * 80)
print("MODEL WITH HETEROSKEDASTICITY-ROBUST STANDARD ERRORS (HC1)")
print("=" * 80)

# Re-run preferred model with robust SE
model_robust = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                       data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nKey results with robust standard errors:")
print(f"ELIGIBLE_AFTER coefficient: {model_robust.params['ELIGIBLE_AFTER']:.6f}")
print(f"Robust Std Error: {model_robust.bse['ELIGIBLE_AFTER']:.6f}")
print(f"t-statistic: {model_robust.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model_robust.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI: [{model_robust.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model_robust.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")

#############################################################################
# CLUSTERED STANDARD ERRORS BY STATE
#############################################################################
print("\n" + "=" * 80)
print("MODEL WITH STATE-CLUSTERED STANDARD ERRORS")
print("=" * 80)

model_cluster = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE',
                        data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print("\nKey results with clustered standard errors:")
print(f"ELIGIBLE_AFTER coefficient: {model_cluster.params['ELIGIBLE_AFTER']:.6f}")
print(f"Clustered Std Error: {model_cluster.bse['ELIGIBLE_AFTER']:.6f}")
print(f"t-statistic: {model_cluster.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model_cluster.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI: [{model_cluster.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model_cluster.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")

print("\n" + "=" * 80)
print("PREFERRED SPECIFICATION DETAILS")
print("=" * 80)
print("\nPreferred specification: Model with covariates, survey weights, and clustered SEs")
print(model_cluster.summary())
