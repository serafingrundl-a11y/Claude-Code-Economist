"""
DACA Eligibility Effect on Full-Time Employment
Replication Analysis Script
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
print("DACA REPLICATION STUDY - FULL-TIME EMPLOYMENT ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading ACS data...")

# Read data in chunks due to large file size
dtype_spec = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'PUMA': 'int32',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int8',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'YRSUSA1': 'int8',
    'EDUC': 'int8',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8'
}

# Columns needed for analysis
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
               'EDUC', 'EMPSTAT', 'UHRSWORK']

# Load data
df = pd.read_csv('data/data.csv', usecols=cols_needed, dtype=dtype_spec)
print(f"   Total observations loaded: {len(df):,}")

# ============================================================================
# 2. SAMPLE SELECTION
# ============================================================================
print("\n[2] Applying sample restrictions...")

# Exclude 2012 (cannot distinguish pre/post DACA within year)
df = df[df['YEAR'] != 2012]
print(f"   After excluding 2012: {len(df):,}")

# Hispanic-Mexican ethnicity (HISPAN == 1)
df = df[df['HISPAN'] == 1]
print(f"   After Hispanic-Mexican filter: {len(df):,}")

# Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200]
print(f"   After Mexico birthplace filter: {len(df):,}")

# Non-citizen (CITIZEN == 3) - proxy for undocumented
df = df[df['CITIZEN'] == 3]
print(f"   After non-citizen filter: {len(df):,}")

# Working age (18-64)
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)]
print(f"   After working age (18-64) filter: {len(df):,}")

# Need valid immigration year for eligibility determination
df = df[df['YRIMMIG'] > 0]
print(f"   After valid immigration year filter: {len(df):,}")

# ============================================================================
# 3. CONSTRUCT VARIABLES
# ============================================================================
print("\n[3] Constructing analysis variables...")

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Age on June 15, 2012 (use birth year; conservative approach)
df['age_2012'] = 2012 - df['BIRTHYR']

# Years in US as of 2007
df['in_us_by_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# DACA Eligibility criteria:
# 1. Arrived before age 16
# 2. Under 31 on June 15, 2012 (born 1982 or later)
# 3. In US since June 2007 (arrived 2007 or earlier)
# 4. Present in US on June 15, 2012 (assumed if in sample)

df['daca_eligible'] = (
    (df['age_at_immig'] < 16) &
    (df['BIRTHYR'] >= 1982) &
    (df['YRIMMIG'] <= 2007)
).astype(int)

print(f"   DACA eligible: {df['daca_eligible'].sum():,} ({df['daca_eligible'].mean()*100:.1f}%)")
print(f"   Non-eligible: {(1-df['daca_eligible']).sum():,} ({(1-df['daca_eligible'].mean())*100:.1f}%)")

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term
df['eligible_post'] = df['daca_eligible'] * df['post']

# Full-time employment outcome (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# ============================================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[4] Generating descriptive statistics...")

# Summary by eligibility status
summary_stats = df.groupby('daca_eligible').agg({
    'fulltime': ['mean', 'std'],
    'employed': ['mean', 'std'],
    'AGE': ['mean', 'std'],
    'SEX': lambda x: (x == 1).mean(),  # Proportion male
    'EDUC': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\n   Summary Statistics by DACA Eligibility:")
print(summary_stats)

# Pre-post comparison by eligibility
print("\n   Full-time Employment Rate by Eligibility and Period:")
prepost = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': 'mean',
    'PERWT': 'sum'
}).round(4)
print(prepost)

# Year-by-year trends
print("\n   Full-time Employment Rate by Year and Eligibility:")
yearly = df.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
print(yearly.round(4))

# ============================================================================
# 5. DIFFERENCE-IN-DIFFERENCES ESTIMATION
# ============================================================================
print("\n[5] Running Difference-in-Differences Estimation...")

# Create dummy variables for categorical controls
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)

# Education categories
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # High school or more
df['educ_college'] = (df['EDUC'] >= 7).astype(int)  # Some college or more

# Model 1: Basic DiD
print("\n   Model 1: Basic DiD (no controls)")
model1 = smf.wls(
    'fulltime ~ daca_eligible + post + eligible_post',
    data=df,
    weights=df['PERWT']
).fit(cov_type='HC1')
print(f"   DiD coefficient (eligible_post): {model1.params['eligible_post']:.4f}")
print(f"   Standard error: {model1.bse['eligible_post']:.4f}")
print(f"   p-value: {model1.pvalues['eligible_post']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['eligible_post', 0]:.4f}, {model1.conf_int().loc['eligible_post', 1]:.4f}]")

# Model 2: DiD with demographic controls
print("\n   Model 2: DiD with demographic controls")
model2 = smf.wls(
    'fulltime ~ daca_eligible + post + eligible_post + AGE + I(AGE**2) + female + married + educ_hs + educ_college',
    data=df,
    weights=df['PERWT']
).fit(cov_type='HC1')
print(f"   DiD coefficient (eligible_post): {model2.params['eligible_post']:.4f}")
print(f"   Standard error: {model2.bse['eligible_post']:.4f}")
print(f"   p-value: {model2.pvalues['eligible_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['eligible_post', 0]:.4f}, {model2.conf_int().loc['eligible_post', 1]:.4f}]")

# Model 3: DiD with state and year fixed effects
print("\n   Model 3: DiD with demographic controls + state + year FE")

# Create state and year dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True, dtype=float)
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True, dtype=float)

X_vars = ['daca_eligible', 'eligible_post', 'AGE', 'female', 'married', 'educ_hs', 'educ_college']
X = df[X_vars].astype(float).copy()
X['age_sq'] = (df['AGE'] ** 2).astype(float)
X = pd.concat([X, state_dummies, year_dummies], axis=1)
X = sm.add_constant(X)

model3 = sm.WLS(df['fulltime'].astype(float), X, weights=df['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient (eligible_post): {model3.params['eligible_post']:.4f}")
print(f"   Standard error: {model3.bse['eligible_post']:.4f}")
print(f"   p-value: {model3.pvalues['eligible_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['eligible_post', 0]:.4f}, {model3.conf_int().loc['eligible_post', 1]:.4f}]")

# ============================================================================
# 6. ROBUSTNESS CHECKS
# ============================================================================
print("\n[6] Robustness Checks...")

# Alternative control group: Older Mexican-born immigrants (arrived as adults)
print("\n   Robustness 1: Restrict control to those who arrived as adults (age 25+)")
df_robust1 = df[(df['daca_eligible'] == 1) | (df['age_at_immig'] >= 25)]
model_r1 = smf.wls(
    'fulltime ~ daca_eligible + post + eligible_post + AGE + I(AGE**2) + female + married',
    data=df_robust1,
    weights=df_robust1['PERWT']
).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_r1.params['eligible_post']:.4f} (SE: {model_r1.bse['eligible_post']:.4f})")

# Placebo test: pre-period trends (2006-2008 vs 2009-2011)
print("\n   Robustness 2: Placebo test (Pre-period: 2006-2008 vs 2009-2011)")
df_pre = df[df['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_interaction'] = df_pre['daca_eligible'] * df_pre['placebo_post']

model_placebo = smf.wls(
    'fulltime ~ daca_eligible + placebo_post + placebo_interaction + AGE + female + married',
    data=df_pre,
    weights=df_pre['PERWT']
).fit(cov_type='HC1')
print(f"   Placebo DiD coefficient: {model_placebo.params['placebo_interaction']:.4f} (SE: {model_placebo.bse['placebo_interaction']:.4f})")
print(f"   p-value: {model_placebo.pvalues['placebo_interaction']:.4f}")

# Employment rate (extensive margin)
print("\n   Robustness 3: Extensive margin (any employment)")
model_emp = smf.wls(
    'employed ~ daca_eligible + post + eligible_post + AGE + I(AGE**2) + female + married',
    data=df,
    weights=df['PERWT']
).fit(cov_type='HC1')
print(f"   DiD coefficient (employment): {model_emp.params['eligible_post']:.4f} (SE: {model_emp.bse['eligible_post']:.4f})")

# By gender
print("\n   Robustness 4: Heterogeneity by gender")
for gender, label in [(1, 'Male'), (2, 'Female')]:
    df_gender = df[df['SEX'] == gender]
    model_g = smf.wls(
        'fulltime ~ daca_eligible + post + eligible_post + AGE + married',
        data=df_gender,
        weights=df_gender['PERWT']
    ).fit(cov_type='HC1')
    print(f"   {label}: DiD = {model_g.params['eligible_post']:.4f} (SE: {model_g.bse['eligible_post']:.4f})")

# ============================================================================
# 7. GENERATE OUTPUT FOR REPORT
# ============================================================================
print("\n[7] Generating output tables...")

# Create results summary
results_summary = {
    'Model': ['Basic DiD', 'With Controls', 'With Controls + FE'],
    'DiD Estimate': [model1.params['eligible_post'], model2.params['eligible_post'], model3.params['eligible_post']],
    'Std. Error': [model1.bse['eligible_post'], model2.bse['eligible_post'], model3.bse['eligible_post']],
    'p-value': [model1.pvalues['eligible_post'], model2.pvalues['eligible_post'], model3.pvalues['eligible_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs)]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results_summary.csv', index=False)
print("   Results saved to results_summary.csv")

# Year-by-year means for plotting
yearly_means = df.groupby(['YEAR', 'daca_eligible']).agg({
    'fulltime': 'mean',
    'employed': 'mean'
}).reset_index()
yearly_means.to_csv('yearly_means.csv', index=False)
print("   Yearly means saved to yearly_means.csv")

# Sample characteristics table
sample_chars = df.groupby('daca_eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'educ_college': 'mean',
    'fulltime': 'mean',
    'employed': 'mean',
    'PERWT': ['sum', 'count']
}).round(3)
sample_chars.to_csv('sample_characteristics.csv')
print("   Sample characteristics saved to sample_characteristics.csv")

# ============================================================================
# 8. FINAL RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"\nPreferred Estimate (Model 3: DiD with controls and fixed effects):")
print(f"   Effect of DACA eligibility on full-time employment: {model3.params['eligible_post']:.4f}")
print(f"   Standard error: {model3.bse['eligible_post']:.4f}")
print(f"   95% Confidence Interval: [{model3.conf_int().loc['eligible_post', 0]:.4f}, {model3.conf_int().loc['eligible_post', 1]:.4f}]")
print(f"   p-value: {model3.pvalues['eligible_post']:.4f}")
print(f"   Sample size: {int(model3.nobs):,}")

print("\nInterpretation:")
did_est = model3.params['eligible_post']
if did_est > 0:
    print(f"   DACA eligibility is associated with a {did_est*100:.2f} percentage point")
    print(f"   increase in the probability of full-time employment.")
else:
    print(f"   DACA eligibility is associated with a {abs(did_est)*100:.2f} percentage point")
    print(f"   decrease in the probability of full-time employment.")

# Statistical significance
if model3.pvalues['eligible_post'] < 0.01:
    print("   This effect is statistically significant at the 1% level.")
elif model3.pvalues['eligible_post'] < 0.05:
    print("   This effect is statistically significant at the 5% level.")
elif model3.pvalues['eligible_post'] < 0.10:
    print("   This effect is statistically significant at the 10% level.")
else:
    print("   This effect is not statistically significant at conventional levels.")

print("\n" + "="*80)
print("Analysis complete.")
print("="*80)
