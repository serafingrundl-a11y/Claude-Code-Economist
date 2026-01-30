"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
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
print("DACA REPLICATION STUDY - DATA ANALYSIS")
print("="*80)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\n1. Loading data...")
data_path = "data/data.csv"

# Read data in chunks due to large file size
chunks = []
chunksize = 500000

# Read necessary columns only
usecols = ['YEAR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'BIRTHYR', 'BIRTHQTR',
           'UHRSWORK', 'EMPSTAT', 'PERWT', 'AGE', 'SEX', 'EDUC', 'MARST',
           'STATEFIP', 'LABFORCE']

dtypes = {
    'YEAR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'BIRTHYR': 'int16',
    'BIRTHQTR': 'int8',
    'UHRSWORK': 'int8',
    'EMPSTAT': 'int8',
    'PERWT': 'float32',
    'AGE': 'int8',
    'SEX': 'int8',
    'EDUC': 'int8',
    'MARST': 'int8',
    'STATEFIP': 'int8',
    'LABFORCE': 'int8'
}

for chunk in pd.read_csv(data_path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
    # Initial filter to reduce memory: Hispanic-Mexican born in Mexico
    chunk_filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk_filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df):,} Hispanic-Mexican Mexican-born individuals")

# ==============================================================================
# 2. SAMPLE CONSTRUCTION
# ==============================================================================
print("\n2. Constructing analysis sample...")

# Filter to non-citizens (proxy for undocumented)
df = df[df['CITIZEN'] == 3]
print(f"After filtering to non-citizens: {len(df):,}")

# Calculate age at immigration
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']

# Filter: arrived before age 16
df = df[(df['age_at_immigration'] >= 0) & (df['age_at_immigration'] < 16)]
print(f"After filtering to arrival before age 16: {len(df):,}")

# Filter: continuous residence since June 15, 2007 (YRIMMIG <= 2007)
df = df[df['YRIMMIG'] <= 2007]
print(f"After filtering for continuous residence (YRIMMIG <= 2007): {len(df):,}")

# Calculate age as of June 15, 2012
# Using BIRTHYR and BIRTHQTR
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born in Q1-Q2, had birthday by June 15
# If born in Q3-Q4, had not had birthday by June 15
df['age_june2012'] = 2012 - df['BIRTHYR']
# Adjust for birth quarter (if born after June, subtract 1)
df.loc[df['BIRTHQTR'] >= 3, 'age_june2012'] = df.loc[df['BIRTHQTR'] >= 3, 'age_june2012'] - 1

# Define treatment and control groups based on age at DACA implementation
# Treatment: ages 26-30 as of June 15, 2012
# Control: ages 31-35 as of June 15, 2012
df['treat_group'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)
df['control_group'] = ((df['age_june2012'] >= 31) & (df['age_june2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df = df[(df['treat_group'] == 1) | (df['control_group'] == 1)]
print(f"After restricting to age 26-35 as of June 2012: {len(df):,}")

# Exclude 2012 (cannot distinguish pre/post DACA)
df = df[df['YEAR'] != 2012]
print(f"After excluding 2012: {len(df):,}")

# Define post-treatment indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Create treatment indicator (1 if in treatment group, 0 if control)
df['treat'] = df['treat_group']

# Create interaction term
df['treat_post'] = df['treat'] * df['post']

# ==============================================================================
# 3. OUTCOME VARIABLE
# ==============================================================================
print("\n3. Creating outcome variable...")

# Full-time employment: UHRSWORK >= 35
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Alternative: employed at all
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# ==============================================================================
# 4. DESCRIPTIVE STATISTICS
# ==============================================================================
print("\n4. Descriptive Statistics")
print("="*60)

# Summary by treatment status and period
print("\n4.1 Sample sizes by group and period:")
sample_counts = df.groupby(['treat', 'post']).agg({
    'PERWT': ['count', 'sum']
}).round(0)
sample_counts.columns = ['Unweighted N', 'Weighted N']
print(sample_counts)

# Full-time employment rates
print("\n4.2 Full-time employment rates (weighted):")
def weighted_mean(x, w):
    return np.average(x, weights=w)

ft_rates = df.groupby(['treat', 'post']).apply(
    lambda x: weighted_mean(x['fulltime'], x['PERWT'])
).unstack()
ft_rates.columns = ['Pre-DACA', 'Post-DACA']
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
print(ft_rates.round(4))

# Calculate simple DiD
did_simple = (ft_rates.loc['Treatment (26-30)', 'Post-DACA'] -
              ft_rates.loc['Treatment (26-30)', 'Pre-DACA']) - \
             (ft_rates.loc['Control (31-35)', 'Post-DACA'] -
              ft_rates.loc['Control (31-35)', 'Pre-DACA'])
print(f"\nSimple DiD estimate: {did_simple:.4f}")

# Demographics by group
print("\n4.3 Demographics by treatment group:")
demo_vars = ['SEX', 'AGE', 'MARST', 'EDUC']
for var in demo_vars:
    print(f"\n{var} distribution:")
    print(df.groupby('treat')[var].describe().round(2))

# ==============================================================================
# 5. REGRESSION ANALYSIS
# ==============================================================================
print("\n5. Regression Analysis")
print("="*60)

# Prepare control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # High school or more
df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # Some college or more

# Year dummies
df['year'] = df['YEAR']

# State as categorical
df['state'] = df['STATEFIP'].astype('category')

# Age controls (use age at survey)
df['age_sq'] = df['AGE'] ** 2

print("\n5.1 Model 1: Basic DiD (no controls)")
# Basic DiD regression with weights
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df['STATEFIP']})
print(model1.summary().tables[1])

print("\n5.2 Model 2: DiD with demographic controls")
model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + educ_college + AGE + age_sq',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df['STATEFIP']})
print(model2.summary().tables[1])

print("\n5.3 Model 3: DiD with year fixed effects")
model3 = smf.wls('fulltime ~ treat + treat_post + female + married + educ_hs + educ_college + AGE + age_sq + C(year)',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df['STATEFIP']})
print("\nTreatment effect (treat_post) coefficient:")
print(f"  Estimate: {model3.params['treat_post']:.4f}")
print(f"  Std. Error: {model3.bse['treat_post']:.4f}")
print(f"  t-stat: {model3.tvalues['treat_post']:.4f}")
print(f"  p-value: {model3.pvalues['treat_post']:.4f}")

print("\n5.4 Model 4: DiD with year and state fixed effects")
model4 = smf.wls('fulltime ~ treat + treat_post + female + married + educ_hs + educ_college + AGE + age_sq + C(year) + C(state)',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df['STATEFIP']})
print("\nTreatment effect (treat_post) coefficient:")
print(f"  Estimate: {model4.params['treat_post']:.4f}")
print(f"  Std. Error: {model4.bse['treat_post']:.4f}")
print(f"  t-stat: {model4.tvalues['treat_post']:.4f}")
print(f"  p-value: {model4.pvalues['treat_post']:.4f}")

# ==============================================================================
# 6. PREFERRED SPECIFICATION
# ==============================================================================
print("\n6. Preferred Specification")
print("="*60)

# Model 4 is the preferred specification
preferred_model = model4
coef = preferred_model.params['treat_post']
se = preferred_model.bse['treat_post']
ci_low = coef - 1.96 * se
ci_high = coef + 1.96 * se

print(f"\nPreferred Estimate (Model 4 with year and state FE):")
print(f"  Effect of DACA eligibility on full-time employment: {coef:.4f}")
print(f"  Standard Error (clustered by state): {se:.4f}")
print(f"  95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  Sample Size: {len(df):,}")

# ==============================================================================
# 7. ROBUSTNESS CHECKS
# ==============================================================================
print("\n7. Robustness Checks")
print("="*60)

# 7.1 Alternative outcome: Employment (any)
print("\n7.1 Alternative outcome: Employment (any)")
model_emp = smf.wls('employed ~ treat + treat_post + female + married + educ_hs + educ_college + AGE + age_sq + C(year) + C(state)',
                     data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                        cov_kwds={'groups': df['STATEFIP']})
print(f"  Estimate: {model_emp.params['treat_post']:.4f}")
print(f"  Std. Error: {model_emp.bse['treat_post']:.4f}")

# 7.2 By gender
print("\n7.2 By gender subgroups:")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex]
    model_sub = smf.wls('fulltime ~ treat + treat_post + married + educ_hs + educ_college + AGE + age_sq + C(year) + C(state)',
                         data=df_sub, weights=df_sub['PERWT']).fit(cov_type='cluster',
                                                                    cov_kwds={'groups': df_sub['STATEFIP']})
    print(f"  {label}: {model_sub.params['treat_post']:.4f} (SE: {model_sub.bse['treat_post']:.4f})")

# 7.3 Pre-trends check
print("\n7.3 Pre-trends analysis (pre-period only):")
df_pre = df[df['post'] == 0].copy()
df_pre['year_centered'] = df_pre['YEAR'] - 2006
df_pre['treat_year'] = df_pre['treat'] * df_pre['year_centered']

model_pretrend = smf.wls('fulltime ~ treat + year_centered + treat_year + female + married + educ_hs + educ_college + AGE + age_sq',
                          data=df_pre, weights=df_pre['PERWT']).fit(cov_type='cluster',
                                                                     cov_kwds={'groups': df_pre['STATEFIP']})
print(f"  Differential pre-trend (treat*year): {model_pretrend.params['treat_year']:.4f}")
print(f"  Std. Error: {model_pretrend.bse['treat_year']:.4f}")
print(f"  p-value: {model_pretrend.pvalues['treat_year']:.4f}")

# ==============================================================================
# 8. EVENT STUDY
# ==============================================================================
print("\n8. Event Study Analysis")
print("="*60)

# Create year dummies interacted with treatment
df['year_factor'] = df['YEAR'].astype('category')
years = sorted(df['YEAR'].unique())
base_year = 2011  # last pre-treatment year

# Create interactions
for y in years:
    if y != base_year:
        df[f'treat_year{y}'] = df['treat'] * (df['YEAR'] == y).astype(int)

# Build formula
year_terms = ' + '.join([f'treat_year{y}' for y in years if y != base_year])
event_formula = f'fulltime ~ treat + {year_terms} + female + married + educ_hs + educ_college + AGE + age_sq + C(year) + C(state)'

model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                                        cov_kwds={'groups': df['STATEFIP']})

print("Event study coefficients (relative to 2011):")
for y in years:
    if y != base_year:
        param_name = f'treat_year{y}'
        print(f"  {y}: {model_event.params[param_name]:.4f} (SE: {model_event.bse[param_name]:.4f})")

# ==============================================================================
# 9. SAVE RESULTS
# ==============================================================================
print("\n9. Saving results...")

# Create results dictionary
results = {
    'preferred_estimate': coef,
    'standard_error': se,
    'ci_lower': ci_low,
    'ci_upper': ci_high,
    'sample_size': len(df),
    't_statistic': preferred_model.tvalues['treat_post'],
    'p_value': preferred_model.pvalues['treat_post']
}

# Save to file
with open('results_summary.txt', 'w') as f:
    f.write("DACA Replication Study - Results Summary\n")
    f.write("="*50 + "\n\n")
    f.write("Preferred Specification: DiD with year and state FE\n")
    f.write(f"Effect estimate: {coef:.4f}\n")
    f.write(f"Standard error: {se:.4f}\n")
    f.write(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n")
    f.write(f"t-statistic: {results['t_statistic']:.4f}\n")
    f.write(f"p-value: {results['p_value']:.4f}\n")
    f.write(f"Sample size: {len(df):,}\n")

# Save full-time rates by group/period for report
ft_rates_df = df.groupby(['treat', 'post', 'YEAR']).apply(
    lambda x: pd.Series({
        'fulltime_rate': weighted_mean(x['fulltime'], x['PERWT']),
        'n_unweighted': len(x),
        'n_weighted': x['PERWT'].sum()
    })
).reset_index()
ft_rates_df.to_csv('fulltime_rates_by_year.csv', index=False)

# Save event study coefficients
event_coefs = []
for y in years:
    if y == base_year:
        event_coefs.append({'year': y, 'coef': 0, 'se': 0})
    else:
        param_name = f'treat_year{y}'
        event_coefs.append({
            'year': y,
            'coef': model_event.params[param_name],
            'se': model_event.bse[param_name]
        })
event_df = pd.DataFrame(event_coefs)
event_df.to_csv('event_study_coefs.csv', index=False)

print("\nResults saved to:")
print("  - results_summary.txt")
print("  - fulltime_rates_by_year.csv")
print("  - event_study_coefs.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
