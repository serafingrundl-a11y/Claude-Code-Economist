"""
DACA Replication Study - Analysis Script
Replication 45

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US.

Identification: Difference-in-Differences comparing ages 26-30 (treatment)
vs ages 31-35 (control) at time of DACA implementation (June 15, 2012).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# For output
import json
import os

print("=" * 60)
print("DACA REPLICATION STUDY - Analysis Script")
print("=" * 60)

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n[1] Loading data...")

# Read CSV in chunks due to large file size
data_path = "data/data.csv"

# Define dtypes to reduce memory usage
dtypes = {
    'YEAR': 'int16',
    'SAMPLE': 'int32',
    'SERIAL': 'int64',
    'HHWT': 'float32',
    'REGION': 'int8',
    'STATEFIP': 'int8',
    'METRO': 'int8',
    'GQ': 'int8',
    'PERNUM': 'int16',
    'PERWT': 'float32',
    'FAMSIZE': 'int8',
    'NCHILD': 'int8',
    'RELATE': 'int8',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'BIRTHYR': 'int16',
    'RACE': 'int8',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'YRSUSA1': 'int8',
    'YRSUSA2': 'int8',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'EMPSTAT': 'int8',
    'EMPSTATD': 'int8',
    'LABFORCE': 'int8',
    'CLASSWKR': 'int8',
    'CLASSWKRD': 'int8',
    'UHRSWORK': 'int8',
    'POVERTY': 'int16'
}

# Columns we need
use_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
            'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
            'EMPSTAT', 'LABFORCE', 'UHRSWORK']

# Load data
df = pd.read_csv(data_path, usecols=use_cols, dtype={col: dtypes.get(col, 'float32') for col in use_cols})
print(f"   Total observations loaded: {len(df):,}")

# =============================================================================
# STEP 2: Filter Sample
# =============================================================================
print("\n[2] Filtering sample...")

# Initial count
n_total = len(df)
print(f"   Initial sample size: {n_total:,}")

# Filter: Hispanic-Mexican (HISPAN == 1)
df = df[df['HISPAN'] == 1]
n_hispan = len(df)
print(f"   After Hispanic-Mexican filter: {n_hispan:,}")

# Filter: Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200]
n_mexico = len(df)
print(f"   After Mexico birthplace filter: {n_mexico:,}")

# Filter: Non-citizen (CITIZEN == 3)
df = df[df['CITIZEN'] == 3]
n_noncit = len(df)
print(f"   After non-citizen filter: {n_noncit:,}")

# Filter: Exclude 2012 (implementation year - cannot distinguish before/after)
df = df[df['YEAR'] != 2012]
n_no2012 = len(df)
print(f"   After excluding 2012: {n_no2012:,}")

# =============================================================================
# STEP 3: Define Treatment and Control Groups
# =============================================================================
print("\n[3] Defining treatment and control groups...")

# Age at DACA implementation (June 15, 2012)
# Treatment: ages 26-30 on June 15, 2012 -> born 1982-1986
# Control: ages 31-35 on June 15, 2012 -> born 1977-1981

# More precise: accounting for birth quarter
# June 15 is in Q2, so:
# - Someone born in Q1-Q2 1982 would be 30 years old
# - Someone born in Q3-Q4 1981 would also be 30 (just turned)

# For simplicity, use birth year approach:
# Age on June 15, 2012 = 2012 - BIRTHYR (approximately)

# Treatment group: born 1982-1986 (ages 26-30 on June 15, 2012)
treatment_mask = (df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)

# Control group: born 1977-1981 (ages 31-35 on June 15, 2012)
control_mask = (df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)

# Filter to only treatment and control
df = df[treatment_mask | control_mask]
n_groups = len(df)
print(f"   After selecting age groups: {n_groups:,}")

# Create treatment indicator
df['treatment'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)
print(f"   Treatment group (26-30): {df['treatment'].sum():,}")
print(f"   Control group (31-35): {(df['treatment'] == 0).sum():,}")

# =============================================================================
# STEP 4: Apply DACA Eligibility Criteria
# =============================================================================
print("\n[4] Applying DACA eligibility criteria...")

# Criterion: Arrived in US before 16th birthday
# Age at arrival = YRIMMIG - BIRTHYR
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Keep only those who arrived before age 16
# Handle missing/invalid YRIMMIG (coded as 0)
df = df[(df['YRIMMIG'] > 0) & (df['age_at_arrival'] < 16)]
n_arr16 = len(df)
print(f"   After arrival before age 16 filter: {n_arr16:,}")

# Criterion: Continuous presence since June 15, 2007
# Must have arrived by 2007
df = df[df['YRIMMIG'] <= 2007]
n_cont = len(df)
print(f"   After continuous presence (YRIMMIG <= 2007) filter: {n_cont:,}")

print(f"\n   FINAL ANALYTIC SAMPLE: {len(df):,}")

# =============================================================================
# STEP 5: Create Outcome and Key Variables
# =============================================================================
print("\n[5] Creating analysis variables...")

# Outcome: Full-time employment (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Post-treatment indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term
df['treat_post'] = df['treatment'] * df['post']

# Additional controls
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] == 1).astype(int)

# Education categories
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # High school or more
df['educ_college'] = (df['EDUC'] >= 7).astype(int)  # Some college or more

# Age at survey (for controls)
df['age_survey'] = df['YEAR'] - df['BIRTHYR']

print(f"   Full-time employment rate: {df['fulltime'].mean():.3f}")
print(f"   Treatment group share: {df['treatment'].mean():.3f}")
print(f"   Post-period share: {df['post'].mean():.3f}")

# =============================================================================
# STEP 6: Descriptive Statistics
# =============================================================================
print("\n[6] Generating descriptive statistics...")

# Summary by treatment status
print("\n   Summary by Treatment Status:")
print("-" * 50)
summary = df.groupby('treatment').agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'age_survey': 'mean',
    'PERWT': 'sum'
}).round(3)
print(summary)

# Summary by period and treatment (2x2 table)
print("\n   Full-time Employment by Group and Period:")
print("-" * 50)
crosstab = df.groupby(['treatment', 'post'])['fulltime'].agg(['mean', 'count']).round(3)
print(crosstab)

# Calculate raw DiD
ft_treat_pre = df[(df['treatment']==1) & (df['post']==0)]['fulltime'].mean()
ft_treat_post = df[(df['treatment']==1) & (df['post']==1)]['fulltime'].mean()
ft_ctrl_pre = df[(df['treatment']==0) & (df['post']==0)]['fulltime'].mean()
ft_ctrl_post = df[(df['treatment']==0) & (df['post']==1)]['fulltime'].mean()

raw_did = (ft_treat_post - ft_treat_pre) - (ft_ctrl_post - ft_ctrl_pre)
print(f"\n   Raw DiD estimate: {raw_did:.4f}")
print(f"   Treatment pre: {ft_treat_pre:.4f}, post: {ft_treat_post:.4f}")
print(f"   Control pre: {ft_ctrl_pre:.4f}, post: {ft_ctrl_post:.4f}")

# =============================================================================
# STEP 7: Difference-in-Differences Regression
# =============================================================================
print("\n[7] Running DiD regressions...")

results_dict = {}

# Model 1: Basic DiD (no controls)
print("\n   Model 1: Basic DiD")
print("-" * 50)
model1 = smf.ols('fulltime ~ treatment + post + treat_post', data=df).fit()
print(model1.summary().tables[1])
results_dict['model1'] = {
    'coef': model1.params['treat_post'],
    'se': model1.bse['treat_post'],
    'pval': model1.pvalues['treat_post'],
    'ci_low': model1.conf_int().loc['treat_post', 0],
    'ci_high': model1.conf_int().loc['treat_post', 1],
    'n': int(model1.nobs),
    'r2': model1.rsquared
}

# Model 2: With demographic controls
print("\n   Model 2: DiD with demographic controls")
print("-" * 50)
model2 = smf.ols('fulltime ~ treatment + post + treat_post + female + married + educ_hs',
                 data=df).fit()
print(model2.summary().tables[1])
results_dict['model2'] = {
    'coef': model2.params['treat_post'],
    'se': model2.bse['treat_post'],
    'pval': model2.pvalues['treat_post'],
    'ci_low': model2.conf_int().loc['treat_post', 0],
    'ci_high': model2.conf_int().loc['treat_post', 1],
    'n': int(model2.nobs),
    'r2': model2.rsquared
}

# Model 3: With state fixed effects
print("\n   Model 3: DiD with state fixed effects")
print("-" * 50)
df['state_fe'] = df['STATEFIP'].astype(str)
model3 = smf.ols('fulltime ~ treatment + post + treat_post + female + married + educ_hs + C(state_fe)',
                 data=df).fit()
# Print just the key coefficients
print(f"   treat_post coefficient: {model3.params['treat_post']:.4f}")
print(f"   Standard error: {model3.bse['treat_post']:.4f}")
print(f"   p-value: {model3.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
results_dict['model3'] = {
    'coef': model3.params['treat_post'],
    'se': model3.bse['treat_post'],
    'pval': model3.pvalues['treat_post'],
    'ci_low': model3.conf_int().loc['treat_post', 0],
    'ci_high': model3.conf_int().loc['treat_post', 1],
    'n': int(model3.nobs),
    'r2': model3.rsquared
}

# Model 4: With year fixed effects
print("\n   Model 4: DiD with year and state fixed effects")
print("-" * 50)
df['year_fe'] = df['YEAR'].astype(str)
model4 = smf.ols('fulltime ~ treatment + treat_post + female + married + educ_hs + C(state_fe) + C(year_fe)',
                 data=df).fit()
print(f"   treat_post coefficient: {model4.params['treat_post']:.4f}")
print(f"   Standard error: {model4.bse['treat_post']:.4f}")
print(f"   p-value: {model4.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
results_dict['model4'] = {
    'coef': model4.params['treat_post'],
    'se': model4.bse['treat_post'],
    'pval': model4.pvalues['treat_post'],
    'ci_low': model4.conf_int().loc['treat_post', 0],
    'ci_high': model4.conf_int().loc['treat_post', 1],
    'n': int(model4.nobs),
    'r2': model4.rsquared
}

# Model 5: Clustered standard errors at state level
print("\n   Model 5: DiD with clustered standard errors (state)")
print("-" * 50)
model5 = smf.ols('fulltime ~ treatment + treat_post + female + married + educ_hs + C(state_fe) + C(year_fe)',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"   treat_post coefficient: {model5.params['treat_post']:.4f}")
print(f"   Clustered SE: {model5.bse['treat_post']:.4f}")
print(f"   p-value: {model5.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
results_dict['model5'] = {
    'coef': model5.params['treat_post'],
    'se': model5.bse['treat_post'],
    'pval': model5.pvalues['treat_post'],
    'ci_low': model5.conf_int().loc['treat_post', 0],
    'ci_high': model5.conf_int().loc['treat_post', 1],
    'n': int(model5.nobs),
    'r2': model5.rsquared
}

# =============================================================================
# STEP 8: Weighted Analysis
# =============================================================================
print("\n[8] Running weighted DiD regression...")
print("-" * 50)

# Weighted regression using person weights - simpler approach
# Remove any NaN weights
df_weighted = df[~(df['PERWT'].isna()) & (df['PERWT'] > 0)].copy()

# Simple weighted regression without all the fixed effects (for tractability)
model_weighted = smf.wls('fulltime ~ treatment + post + treat_post + female + married + educ_hs',
                         data=df_weighted, weights=df_weighted['PERWT']).fit()
print(f"   Weighted treat_post coefficient: {model_weighted.params['treat_post']:.4f}")
print(f"   Standard error: {model_weighted.bse['treat_post']:.4f}")
print(f"   p-value: {model_weighted.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model_weighted.conf_int().loc['treat_post', 0]:.4f}, {model_weighted.conf_int().loc['treat_post', 1]:.4f}]")

results_dict['model_weighted'] = {
    'coef': model_weighted.params['treat_post'],
    'se': model_weighted.bse['treat_post'],
    'pval': model_weighted.pvalues['treat_post'],
    'ci_low': model_weighted.conf_int().loc['treat_post', 0],
    'ci_high': model_weighted.conf_int().loc['treat_post', 1],
    'n': int(model_weighted.nobs),
    'r2': model_weighted.rsquared
}

# =============================================================================
# STEP 9: Event Study / Pre-trends Check
# =============================================================================
print("\n[9] Event study analysis (parallel trends check)...")
print("-" * 50)

# Create year-specific treatment effects
df['year_2006'] = (df['YEAR'] == 2006).astype(int)
df['year_2007'] = (df['YEAR'] == 2007).astype(int)
df['year_2008'] = (df['YEAR'] == 2008).astype(int)
df['year_2009'] = (df['YEAR'] == 2009).astype(int)
df['year_2010'] = (df['YEAR'] == 2010).astype(int)
df['year_2011'] = (df['YEAR'] == 2011).astype(int)  # Reference year
df['year_2013'] = (df['YEAR'] == 2013).astype(int)
df['year_2014'] = (df['YEAR'] == 2014).astype(int)
df['year_2015'] = (df['YEAR'] == 2015).astype(int)
df['year_2016'] = (df['YEAR'] == 2016).astype(int)

# Create interactions (omit 2011 as reference)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'treat_year_{year}'] = df['treatment'] * df[f'year_{year}']

event_formula = ('fulltime ~ treatment + ' +
                 ' + '.join([f'treat_year_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]) +
                 ' + female + married + educ_hs + C(state_fe) + C(year_fe)')

model_event = smf.ols(event_formula, data=df).fit()

print("\n   Year-specific treatment effects (ref: 2011):")
event_results = {}
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treat_year_{year}']
    se = model_event.bse[f'treat_year_{year}']
    pval = model_event.pvalues[f'treat_year_{year}']
    print(f"   {year}: coef = {coef:.4f}, SE = {se:.4f}, p = {pval:.4f}")
    event_results[year] = {'coef': coef, 'se': se, 'pval': pval}

results_dict['event_study'] = event_results

# =============================================================================
# STEP 10: Subgroup Analysis
# =============================================================================
print("\n[10] Subgroup analysis...")
print("-" * 50)

# By sex
print("\n   By Sex:")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex]
    model_sub = smf.ols('fulltime ~ treatment + post + treat_post + married + educ_hs',
                        data=subset).fit()
    print(f"   {label}: coef = {model_sub.params['treat_post']:.4f}, SE = {model_sub.bse['treat_post']:.4f}, n = {int(model_sub.nobs)}")
    results_dict[f'subgroup_{label.lower()}'] = {
        'coef': model_sub.params['treat_post'],
        'se': model_sub.bse['treat_post'],
        'n': int(model_sub.nobs)
    }

# By education
print("\n   By Education:")
for educ_cond, label in [((df['EDUC'] < 6), 'Less than HS'), ((df['EDUC'] >= 6), 'HS or more')]:
    subset = df[educ_cond]
    model_sub = smf.ols('fulltime ~ treatment + post + treat_post + female + married',
                        data=subset).fit()
    print(f"   {label}: coef = {model_sub.params['treat_post']:.4f}, SE = {model_sub.bse['treat_post']:.4f}, n = {int(model_sub.nobs)}")

# =============================================================================
# STEP 11: Save Results
# =============================================================================
print("\n[11] Saving results...")

# Save detailed results to JSON
with open('analysis_results.json', 'w') as f:
    # Convert numpy types to Python types
    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return o

    results_serializable = {}
    for key, val in results_dict.items():
        if isinstance(val, dict):
            results_serializable[key] = {k: convert(v) for k, v in val.items()}
        else:
            results_serializable[key] = convert(val)

    json.dump(results_serializable, f, indent=2)

# Create summary statistics table
summary_stats = df.groupby(['treatment', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'age_survey': 'mean',
    'PERWT': 'sum'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')

# Create yearly means for plotting
yearly_means = df.groupby(['YEAR', 'treatment']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
yearly_means.to_csv('yearly_means.csv')

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nPreferred estimate (Model 5 with clustered SEs):")
print(f"   Effect of DACA eligibility on full-time employment: {results_dict['model5']['coef']:.4f}")
print(f"   Standard Error (clustered): {results_dict['model5']['se']:.4f}")
print(f"   95% Confidence Interval: [{results_dict['model5']['ci_low']:.4f}, {results_dict['model5']['ci_high']:.4f}]")
print(f"   Sample Size: {results_dict['model5']['n']:,}")
print(f"   p-value: {results_dict['model5']['pval']:.4f}")

# Final sample composition
print(f"\nSample Composition:")
print(f"   Treatment group (ages 26-30 at DACA): {df['treatment'].sum():,}")
print(f"   Control group (ages 31-35 at DACA): {(df['treatment'] == 0).sum():,}")
print(f"   Pre-period observations: {(df['post'] == 0).sum():,}")
print(f"   Post-period observations: {(df['post'] == 1).sum():,}")
