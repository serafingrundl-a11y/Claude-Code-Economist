"""
DACA Replication Study - Analysis Script
Estimates the causal effect of DACA eligibility on full-time employment

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for DACA on
the probability of being employed full-time (35+ hours/week)?

Treatment: Ages 26-30 as of June 15, 2012 (DACA age-eligible)
Control: Ages 31-35 as of June 15, 2012 (too old for DACA)
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 2\replication_02")

print("="*70)
print("DACA Replication Analysis")
print("="*70)

# -----------------------------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------------------------
print("\n[1] Loading data...")

# Read the data in chunks due to large file size
dtypes = {
    'YEAR': 'int16',
    'SAMPLE': 'int32',
    'SERIAL': 'int32',
    'HHWT': 'float32',
    'STATEFIP': 'int8',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'YRSUSA1': 'int8',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'EMPSTAT': 'int8',
    'EMPSTATD': 'int8',
    'LABFORCE': 'int8',
    'UHRSWORK': 'int8',
    'MARST': 'int8',
    'NCHILD': 'int8',
    'FAMSIZE': 'int8'
}

# Columns we need
cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
               'EDUC', 'EMPSTAT', 'LABFORCE', 'UHRSWORK', 'MARST',
               'NCHILD', 'FAMSIZE', 'STATEFIP']

# Read data
df = pd.read_csv('data/data.csv', usecols=cols_needed, dtype={k: v for k, v in dtypes.items() if k in cols_needed})
print(f"   Total records loaded: {len(df):,}")

# -----------------------------------------------------------------------------
# 2. Apply Sample Restrictions
# -----------------------------------------------------------------------------
print("\n[2] Applying sample restrictions...")

# Keep only years we need: 2006-2011 (pre), 2013-2016 (post)
# Exclude 2012 because DACA was implemented mid-year
df = df[df['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])]
print(f"   After year restriction (2006-2011, 2013-2016): {len(df):,}")

# Hispanic-Mexican (HISPAN == 1)
df = df[df['HISPAN'] == 1]
print(f"   After Hispanic-Mexican restriction: {len(df):,}")

# Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200]
print(f"   After Mexico birthplace restriction: {len(df):,}")

# Non-citizen without papers (CITIZEN == 3)
# This is our proxy for undocumented status
df = df[df['CITIZEN'] == 3]
print(f"   After non-citizen restriction: {len(df):,}")

# -----------------------------------------------------------------------------
# 3. Calculate Age on June 15, 2012
# -----------------------------------------------------------------------------
print("\n[3] Calculating age as of June 15, 2012...")

# Create approximate age on June 15, 2012
# Using birth year and birth quarter
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

def calc_age_june_2012(row):
    """Calculate age as of June 15, 2012 based on birth year and quarter"""
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']

    # June 15 is in Q2 (April-June)
    # If born in Q1 or Q2, they've had their birthday by June 15
    # If born in Q3 or Q4, they haven't had their birthday yet
    if birth_qtr in [1, 2]:
        age = 2012 - birth_year
    else:
        age = 2012 - birth_year - 1
    return age

df['AGE_JUNE_2012'] = df.apply(calc_age_june_2012, axis=1)

# -----------------------------------------------------------------------------
# 4. Define Treatment and Control Groups
# -----------------------------------------------------------------------------
print("\n[4] Defining treatment and control groups...")

# Treatment group: Ages 26-30 on June 15, 2012
# Control group: Ages 31-35 on June 15, 2012
df = df[(df['AGE_JUNE_2012'] >= 26) & (df['AGE_JUNE_2012'] <= 35)]
print(f"   After age restriction (26-35 as of June 2012): {len(df):,}")

# Create treatment indicator (1 if ages 26-30, 0 if ages 31-35)
df['TREATED'] = (df['AGE_JUNE_2012'] <= 30).astype(int)

# -----------------------------------------------------------------------------
# 5. Apply Additional DACA Eligibility Criteria
# -----------------------------------------------------------------------------
print("\n[5] Applying DACA eligibility criteria...")

# DACA requires: Arrived in US before age 16
# Calculate age at immigration
# YRIMMIG == 0 means N/A (likely native born, but we already filtered for Mexico-born)
df = df[df['YRIMMIG'] > 0]  # Must have valid immigration year
df['AGE_AT_IMMIG'] = df['YRIMMIG'] - df['BIRTHYR']

# Arrived before turning 16
df = df[df['AGE_AT_IMMIG'] < 16]
print(f"   After arrived-before-16 restriction: {len(df):,}")

# Continuously in US since June 15, 2007
# This means immigration year must be 2006 or earlier
# (since if they immigrated in 2007, we can't be sure they were present by June 15)
df = df[df['YRIMMIG'] <= 2006]
print(f"   After continuous residence (arrived by 2006): {len(df):,}")

# -----------------------------------------------------------------------------
# 6. Create Outcome and Analysis Variables
# -----------------------------------------------------------------------------
print("\n[6] Creating outcome and analysis variables...")

# Outcome: Full-time employment (35+ hours per week)
# UHRSWORK: usual hours worked per week
df['FULLTIME'] = (df['UHRSWORK'] >= 35).astype(int)

# Post-treatment indicator (2013-2016 vs 2006-2011)
df['POST'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term for DiD
df['TREATED_POST'] = df['TREATED'] * df['POST']

# Additional covariates
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Education categories
df['EDUC_HS'] = (df['EDUC'] >= 6).astype(int)  # High school or more
df['EDUC_COLLEGE'] = (df['EDUC'] >= 10).astype(int)  # Some college or more

print(f"\n   Final sample size: {len(df):,}")
print(f"   Treatment group (26-30): {df['TREATED'].sum():,}")
print(f"   Control group (31-35): {(1-df['TREATED']).sum():,}")

# -----------------------------------------------------------------------------
# 7. Descriptive Statistics
# -----------------------------------------------------------------------------
print("\n[7] Descriptive Statistics")
print("-" * 70)

# Summary by treatment group
print("\n   === Summary by Treatment Group ===")
summary_treat = df.groupby('TREATED').agg({
    'FULLTIME': ['mean', 'std'],
    'AGE': 'mean',
    'FEMALE': 'mean',
    'MARRIED': 'mean',
    'EDUC_HS': 'mean',
    'PERWT': 'sum'
}).round(3)
print(summary_treat)

# Summary by period
print("\n   === Summary by Period ===")
summary_period = df.groupby('POST').agg({
    'FULLTIME': ['mean', 'std'],
    'PERWT': 'sum'
}).round(3)
print(summary_period)

# 2x2 table for DiD
print("\n   === Full-time Employment Rates (Weighted) ===")
did_table = df.groupby(['TREATED', 'POST']).apply(
    lambda x: np.average(x['FULLTIME'], weights=x['PERWT'])
).unstack()
did_table.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
did_table.index = ['Control (31-35)', 'Treatment (26-30)']
print(did_table.round(4))

# Calculate raw DiD
pre_treat = did_table.iloc[1, 0]
post_treat = did_table.iloc[1, 1]
pre_control = did_table.iloc[0, 0]
post_control = did_table.iloc[0, 1]

raw_did = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n   Raw DiD estimate: {raw_did:.4f}")
print(f"   Treatment change: {post_treat - pre_treat:.4f}")
print(f"   Control change: {post_control - pre_control:.4f}")

# -----------------------------------------------------------------------------
# 8. Difference-in-Differences Regression
# -----------------------------------------------------------------------------
print("\n[8] Difference-in-Differences Regression Analysis")
print("-" * 70)

# Model 1: Basic DiD
print("\n   === Model 1: Basic DiD ===")
X1 = df[['TREATED', 'POST', 'TREATED_POST']]
X1 = sm.add_constant(X1)
y = df['FULLTIME']
weights = df['PERWT']

model1 = sm.WLS(y, X1, weights=weights).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n   === Model 2: DiD with Demographics ===")
X2 = df[['TREATED', 'POST', 'TREATED_POST', 'FEMALE', 'MARRIED', 'EDUC_HS']]
X2 = sm.add_constant(X2)

model2 = sm.WLS(y, X2, weights=weights).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n   === Model 3: DiD with Year Fixed Effects ===")
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True, dtype=float)
X3 = pd.concat([df[['TREATED', 'TREATED_POST', 'FEMALE', 'MARRIED', 'EDUC_HS']].astype(float), year_dummies], axis=1)
X3 = sm.add_constant(X3)

model3 = sm.WLS(y.astype(float), X3.astype(float), weights=weights).fit(cov_type='HC1')
print("\n   Key coefficients from Model 3:")
key_vars = ['const', 'TREATED', 'TREATED_POST', 'FEMALE', 'MARRIED', 'EDUC_HS']
for var in key_vars:
    if var in model3.params.index:
        coef = model3.params[var]
        se = model3.bse[var]
        pval = model3.pvalues[var]
        sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"   {var:20s}: {coef:8.4f} ({se:.4f}) {sig}")

# Model 4: DiD with year and state fixed effects
print("\n   === Model 4: DiD with Year and State Fixed Effects ===")
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True, dtype=float)
X4 = pd.concat([df[['TREATED', 'TREATED_POST', 'FEMALE', 'MARRIED', 'EDUC_HS']].astype(float),
                year_dummies, state_dummies], axis=1)
X4 = sm.add_constant(X4)

model4 = sm.WLS(y.astype(float), X4.astype(float), weights=weights).fit(cov_type='HC1')
print("\n   Key coefficients from Model 4:")
for var in key_vars:
    if var in model4.params.index:
        coef = model4.params[var]
        se = model4.bse[var]
        pval = model4.pvalues[var]
        sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"   {var:20s}: {coef:8.4f} ({se:.4f}) {sig}")

# -----------------------------------------------------------------------------
# 9. Preferred Specification Summary
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("PREFERRED ESTIMATE (Model 4 - Full specification)")
print("="*70)

preferred_coef = model4.params['TREATED_POST']
preferred_se = model4.bse['TREATED_POST']
preferred_pval = model4.pvalues['TREATED_POST']
preferred_ci = model4.conf_int().loc['TREATED_POST']

print(f"\n   DiD Estimate (TREATED_POST): {preferred_coef:.4f}")
print(f"   Standard Error (robust):     {preferred_se:.4f}")
print(f"   95% Confidence Interval:     [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"   P-value:                     {preferred_pval:.4f}")
print(f"   Sample Size:                 {len(df):,}")
print(f"   Weighted Sample Size:        {df['PERWT'].sum():,.0f}")

# Statistical significance
if preferred_pval < 0.01:
    sig_level = "statistically significant at the 1% level"
elif preferred_pval < 0.05:
    sig_level = "statistically significant at the 5% level"
elif preferred_pval < 0.1:
    sig_level = "statistically significant at the 10% level"
else:
    sig_level = "not statistically significant at conventional levels"

print(f"\n   The effect is {sig_level}.")

# -----------------------------------------------------------------------------
# 10. Save Results for Report
# -----------------------------------------------------------------------------
print("\n[10] Saving results...")

# Save summary statistics
summary_stats = pd.DataFrame({
    'Variable': ['Full-time employment', 'Female', 'Married', 'HS or more education', 'Age'],
    'Mean': [df['FULLTIME'].mean(), df['FEMALE'].mean(), df['MARRIED'].mean(),
             df['EDUC_HS'].mean(), df['AGE'].mean()],
    'Std': [df['FULLTIME'].std(), df['FEMALE'].std(), df['MARRIED'].std(),
            df['EDUC_HS'].std(), df['AGE'].std()]
})
summary_stats.to_csv('summary_stats.csv', index=False)

# Save DiD table
did_table.to_csv('did_table.csv')

# Save regression results
results_df = pd.DataFrame({
    'Model': ['Model 1', 'Model 2', 'Model 3', 'Model 4'],
    'Coefficient': [model1.params['TREATED_POST'], model2.params['TREATED_POST'],
                   model3.params['TREATED_POST'], model4.params['TREATED_POST']],
    'Std_Error': [model1.bse['TREATED_POST'], model2.bse['TREATED_POST'],
                 model3.bse['TREATED_POST'], model4.bse['TREATED_POST']],
    'P_Value': [model1.pvalues['TREATED_POST'], model2.pvalues['TREATED_POST'],
               model3.pvalues['TREATED_POST'], model4.pvalues['TREATED_POST']],
    'R_Squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared],
    'N': [model1.nobs, model2.nobs, model3.nobs, model4.nobs]
})
results_df.to_csv('regression_results.csv', index=False)

# -----------------------------------------------------------------------------
# 11. Event Study / Year-by-Year Effects
# -----------------------------------------------------------------------------
print("\n[11] Event Study Analysis")
print("-" * 70)

# Create year-specific treatment effects (relative to 2011)
df['YEAR_2006'] = (df['YEAR'] == 2006).astype(int)
df['YEAR_2007'] = (df['YEAR'] == 2007).astype(int)
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
# 2011 is reference year
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'TREAT_X_{yr}'] = df['TREATED'] * df[f'YEAR_{yr}']

# Event study regression
event_vars = ['TREATED'] + [f'TREAT_X_{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]
year_vars = [f'YEAR_{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]

X_event = df[event_vars + year_vars + ['FEMALE', 'MARRIED', 'EDUC_HS']].astype(float)
X_event = pd.concat([X_event, state_dummies], axis=1)
X_event = sm.add_constant(X_event)

model_event = sm.WLS(y.astype(float), X_event.astype(float), weights=weights).fit(cov_type='HC1')

print("\n   Year-by-Year Treatment Effects (relative to 2011):")
event_results = []
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'TREAT_X_{yr}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    ci = model_event.conf_int().loc[var]
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"   {yr}: {coef:7.4f} ({se:.4f}) [{ci[0]:.4f}, {ci[1]:.4f}] {sig}")
    event_results.append({
        'Year': yr,
        'Coefficient': coef,
        'SE': se,
        'CI_Lower': ci[0],
        'CI_Upper': ci[1],
        'P_Value': pval
    })

event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

# -----------------------------------------------------------------------------
# 12. Robustness Checks
# -----------------------------------------------------------------------------
print("\n[12] Robustness Checks")
print("-" * 70)

# Check 1: Males only
print("\n   === Males Only ===")
df_male = df[df['FEMALE'] == 0]
X_male = df_male[['TREATED', 'TREATED_POST', 'MARRIED', 'EDUC_HS']].astype(float)
X_male = pd.concat([X_male, pd.get_dummies(df_male['YEAR'], prefix='year', drop_first=True, dtype=float),
                   pd.get_dummies(df_male['STATEFIP'], prefix='state', drop_first=True, dtype=float)], axis=1)
X_male = sm.add_constant(X_male)
y_male = df_male['FULLTIME'].astype(float)
w_male = df_male['PERWT']
model_male = sm.WLS(y_male, X_male.astype(float), weights=w_male).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_male.params['TREATED_POST']:.4f} (SE: {model_male.bse['TREATED_POST']:.4f})")
print(f"   N = {len(df_male):,}")

# Check 2: Females only
print("\n   === Females Only ===")
df_female = df[df['FEMALE'] == 1]
X_female = df_female[['TREATED', 'TREATED_POST', 'MARRIED', 'EDUC_HS']].astype(float)
X_female = pd.concat([X_female, pd.get_dummies(df_female['YEAR'], prefix='year', drop_first=True, dtype=float),
                     pd.get_dummies(df_female['STATEFIP'], prefix='state', drop_first=True, dtype=float)], axis=1)
X_female = sm.add_constant(X_female)
y_female = df_female['FULLTIME'].astype(float)
w_female = df_female['PERWT']
model_female = sm.WLS(y_female, X_female.astype(float), weights=w_female).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_female.params['TREATED_POST']:.4f} (SE: {model_female.bse['TREATED_POST']:.4f})")
print(f"   N = {len(df_female):,}")

# Check 3: Alternative outcome - In labor force
print("\n   === Alternative Outcome: Labor Force Participation ===")
df['IN_LF'] = (df['LABFORCE'] == 2).astype(int)
y_lf = df['IN_LF'].astype(float)
model_lf = sm.WLS(y_lf, X4.astype(float), weights=weights).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_lf.params['TREATED_POST']:.4f} (SE: {model_lf.bse['TREATED_POST']:.4f})")

# Check 4: Alternative outcome - Employed
print("\n   === Alternative Outcome: Employment ===")
df['EMPLOYED'] = (df['EMPSTAT'] == 1).astype(int)
y_emp = df['EMPLOYED'].astype(float)
model_emp = sm.WLS(y_emp, X4.astype(float), weights=weights).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_emp.params['TREATED_POST']:.4f} (SE: {model_emp.bse['TREATED_POST']:.4f})")

# Save robustness results
robustness_df = pd.DataFrame({
    'Specification': ['Main (Full sample)', 'Males only', 'Females only',
                     'Labor force participation', 'Employment'],
    'Coefficient': [model4.params['TREATED_POST'], model_male.params['TREATED_POST'],
                   model_female.params['TREATED_POST'], model_lf.params['TREATED_POST'],
                   model_emp.params['TREATED_POST']],
    'SE': [model4.bse['TREATED_POST'], model_male.bse['TREATED_POST'],
          model_female.bse['TREATED_POST'], model_lf.bse['TREATED_POST'],
          model_emp.bse['TREATED_POST']],
    'N': [len(df), len(df_male), len(df_female), len(df), len(df)]
})
robustness_df.to_csv('robustness_results.csv', index=False)

# -----------------------------------------------------------------------------
# 13. Summary Table for Years
# -----------------------------------------------------------------------------
print("\n[13] Summary by Year")
print("-" * 70)

yearly_summary = df.groupby('YEAR').agg({
    'FULLTIME': lambda x: np.average(x, weights=df.loc[x.index, 'PERWT']),
    'PERWT': 'sum',
    'TREATED': 'mean'
}).round(4)
yearly_summary.columns = ['FT_Rate', 'Weighted_N', 'Prop_Treated']
print(yearly_summary)
yearly_summary.to_csv('yearly_summary.csv')

print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)
print("\nOutput files created:")
print("  - summary_stats.csv")
print("  - did_table.csv")
print("  - regression_results.csv")
print("  - event_study_results.csv")
print("  - robustness_results.csv")
print("  - yearly_summary.csv")
