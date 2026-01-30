"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US.

Identification Strategy: Difference-in-differences
- Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
- Control: Ages 31-35 at DACA implementation
- Pre-period: 2006-2011
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

print("=" * 80)
print("DACA REPLICATION ANALYSIS")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
print("\n1. LOADING DATA...")
print("-" * 40)

# Load data in chunks due to large file size
chunksize = 500000
chunks = []

# Columns we need
cols_needed = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX', 'MARST',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK', 'STATEFIP']

# Read data
for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunksize):
    chunks.append(chunk)
    print(f"  Loaded chunk: {len(chunk):,} rows")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal observations loaded: {len(df):,}")

# -----------------------------------------------------------------------------
# 2. DEFINE SAMPLE ELIGIBILITY CRITERIA
# -----------------------------------------------------------------------------
print("\n2. DEFINING SAMPLE ELIGIBILITY CRITERIA...")
print("-" * 40)

# DACA was implemented on June 15, 2012
# Eligibility requirements (simplified for analysis):
# 1. Hispanic-Mexican ethnicity (HISPAN == 1)
# 2. Born in Mexico (BPL == 200)
# 3. Not a citizen (CITIZEN == 3) - proxy for undocumented
# 4. Arrived in US before their 16th birthday
# 5. Lived continuously in US since June 15, 2007 (arrived by 2007)

# Step 1: Hispanic-Mexican
print("\nStep 1: Hispanic-Mexican ethnicity (HISPAN == 1)")
df_sample = df[df['HISPAN'] == 1].copy()
print(f"  Observations: {len(df_sample):,}")

# Step 2: Born in Mexico
print("\nStep 2: Born in Mexico (BPL == 200)")
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"  Observations: {len(df_sample):,}")

# Step 3: Not a citizen (proxy for undocumented)
print("\nStep 3: Not a citizen (CITIZEN == 3)")
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"  Observations: {len(df_sample):,}")

# Step 4: Calculate age at arrival and filter for arrived before age 16
# Using YRIMMIG and BIRTHYR
print("\nStep 4: Arrived before age 16")
# Remove observations with missing immigration year
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_arrival'] < 16].copy()
print(f"  Observations: {len(df_sample):,}")

# Step 5: Lived continuously in US since 2007 (arrived by 2007)
print("\nStep 5: Present in US since June 15, 2007 (YRIMMIG <= 2007)")
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
print(f"  Observations: {len(df_sample):,}")

# -----------------------------------------------------------------------------
# 3. DEFINE TREATMENT AND CONTROL GROUPS
# -----------------------------------------------------------------------------
print("\n3. DEFINING TREATMENT AND CONTROL GROUPS...")
print("-" * 40)

# Age on June 15, 2012:
# Treatment: 26-30 years old -> Birth years 1982-1986 (approximately)
# Control: 31-35 years old -> Birth years 1977-1981 (approximately)

# More precise calculation using birth year and quarter
# June 15, 2012 falls in Q2
# If born in Q1-Q2 of a year, age in June 2012 = 2012 - BIRTHYR
# If born in Q3-Q4 of a year, age in June 2012 = 2012 - BIRTHYR - 1

def age_on_june_2012(row):
    """Calculate age on June 15, 2012"""
    if row['BIRTHQTR'] <= 2:  # Born Jan-June
        return 2012 - row['BIRTHYR']
    else:  # Born July-Dec
        return 2012 - row['BIRTHYR'] - 1

df_sample['age_june_2012'] = df_sample.apply(age_on_june_2012, axis=1)

# Define treatment (26-30) and control (31-35)
df_sample['treatment_group'] = ((df_sample['age_june_2012'] >= 26) &
                                 (df_sample['age_june_2012'] <= 30)).astype(int)
df_sample['control_group'] = ((df_sample['age_june_2012'] >= 31) &
                               (df_sample['age_june_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_sample = df_sample[(df_sample['treatment_group'] == 1) |
                       (df_sample['control_group'] == 1)].copy()

print(f"Observations in treatment group (26-30): {df_sample['treatment_group'].sum():,}")
print(f"Observations in control group (31-35): {(df_sample['control_group'] == 1).sum():,}")
print(f"Total observations: {len(df_sample):,}")

# -----------------------------------------------------------------------------
# 4. DEFINE TIME PERIODS
# -----------------------------------------------------------------------------
print("\n4. DEFINING TIME PERIODS...")
print("-" * 40)

# Pre-treatment: 2006-2011
# Post-treatment: 2013-2016
# Exclude 2012 (implementation year)

df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

print(f"Pre-treatment years (2006-2011): {df_sample[df_sample['post']==0]['YEAR'].unique()}")
print(f"Post-treatment years (2013-2016): {df_sample[df_sample['post']==1]['YEAR'].unique()}")
print(f"Total observations after excluding 2012: {len(df_sample):,}")

# Distribution by period and treatment status
print("\nObservation counts by group and period:")
crosstab = pd.crosstab(df_sample['treatment_group'], df_sample['post'],
                       margins=True, margins_name='Total')
crosstab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
crosstab.columns = ['Pre (2006-2011)', 'Post (2013-2016)', 'Total']
print(crosstab)

# -----------------------------------------------------------------------------
# 5. DEFINE OUTCOME VARIABLE
# -----------------------------------------------------------------------------
print("\n5. DEFINING OUTCOME VARIABLE...")
print("-" * 40)

# Full-time employment: UHRSWORK >= 35
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

print(f"Full-time employment rate (overall): {df_sample['fulltime'].mean():.3f}")
print(f"  Treatment group: {df_sample[df_sample['treatment_group']==1]['fulltime'].mean():.3f}")
print(f"  Control group: {df_sample[df_sample['control_group']==1]['fulltime'].mean():.3f}")

# -----------------------------------------------------------------------------
# 6. SUMMARY STATISTICS
# -----------------------------------------------------------------------------
print("\n6. SUMMARY STATISTICS...")
print("-" * 40)

# Create covariates
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'] <= 2).astype(int)  # 1 or 2 = married
df_sample['hs_or_more'] = (df_sample['EDUCD'] >= 62).astype(int)  # HS diploma or higher
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# Summary by treatment status (pre-period only)
pre_sample = df_sample[df_sample['post'] == 0]

def weighted_stats(group, var, weight):
    """Calculate weighted mean and std"""
    d = group
    w = d[weight]
    x = d[var]
    mean = np.average(x, weights=w)
    variance = np.average((x - mean)**2, weights=w)
    return pd.Series({'mean': mean, 'std': np.sqrt(variance)})

print("\nPre-treatment characteristics (weighted):")
print("\n                          Treatment    Control")
print("                          (26-30)      (31-35)")
print("-" * 50)

vars_to_summarize = ['fulltime', 'employed', 'female', 'married', 'hs_or_more', 'AGE']
var_labels = ['Full-time employed', 'Employed', 'Female', 'Married',
              'HS diploma or more', 'Age']

for var, label in zip(vars_to_summarize, var_labels):
    treat_mean = np.average(pre_sample[pre_sample['treatment_group']==1][var],
                           weights=pre_sample[pre_sample['treatment_group']==1]['PERWT'])
    ctrl_mean = np.average(pre_sample[pre_sample['control_group']==1][var],
                          weights=pre_sample[pre_sample['control_group']==1]['PERWT'])
    print(f"{label:25s}   {treat_mean:.3f}        {ctrl_mean:.3f}")

# Sample sizes
print(f"\n{'N (unweighted)':25s}   {len(pre_sample[pre_sample['treatment_group']==1]):,}        {len(pre_sample[pre_sample['control_group']==1]):,}")

# -----------------------------------------------------------------------------
# 7. DIFFERENCE-IN-DIFFERENCES ESTIMATION
# -----------------------------------------------------------------------------
print("\n7. DIFFERENCE-IN-DIFFERENCES ESTIMATION...")
print("-" * 40)

# Create interaction term
df_sample['treat_post'] = df_sample['treatment_group'] * df_sample['post']

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.wls('fulltime ~ treatment_group + post + treat_post',
                 data=df_sample, weights=df_sample['PERWT']).fit()
print(f"\nDiD coefficient (treat_post): {model1.params['treat_post']:.4f}")
print(f"Standard error: {model1.bse['treat_post']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model1.pvalues['treat_post']:.4f}")
print(f"N: {int(model1.nobs):,}")

# Model 2: DiD with year fixed effects
print("\n--- Model 2: DiD with Year Fixed Effects ---")
df_sample['year_factor'] = df_sample['YEAR'].astype(str)
model2 = smf.wls('fulltime ~ treatment_group + C(YEAR) + treat_post',
                 data=df_sample, weights=df_sample['PERWT']).fit()
print(f"\nDiD coefficient (treat_post): {model2.params['treat_post']:.4f}")
print(f"Standard error: {model2.bse['treat_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model2.pvalues['treat_post']:.4f}")

# Model 3: DiD with controls and year fixed effects
print("\n--- Model 3: DiD with Controls + Year FE ---")
model3 = smf.wls('fulltime ~ treatment_group + C(YEAR) + treat_post + female + married + hs_or_more',
                 data=df_sample, weights=df_sample['PERWT']).fit()
print(f"\nDiD coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"Standard error: {model3.bse['treat_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: DiD with controls, year FE, and state FE (preferred specification)
print("\n--- Model 4: DiD with Controls + Year FE + State FE (Preferred) ---")
model4 = smf.wls('fulltime ~ treatment_group + C(YEAR) + C(STATEFIP) + treat_post + female + married + hs_or_more',
                 data=df_sample, weights=df_sample['PERWT']).fit()
print(f"\nDiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"Standard error: {model4.bse['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model4.pvalues['treat_post']:.4f}")

# -----------------------------------------------------------------------------
# 8. ROBUSTNESS CHECKS
# -----------------------------------------------------------------------------
print("\n8. ROBUSTNESS CHECKS...")
print("-" * 40)

# Clustered standard errors by state (using HC1 as approximation)
print("\n--- Robust Standard Errors (HC1) ---")
model4_robust = smf.wls('fulltime ~ treatment_group + C(YEAR) + C(STATEFIP) + treat_post + female + married + hs_or_more',
                        data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model4_robust.params['treat_post']:.4f}")
print(f"Robust SE: {model4_robust.bse['treat_post']:.4f}")
print(f"95% CI: [{model4_robust.conf_int().loc['treat_post', 0]:.4f}, {model4_robust.conf_int().loc['treat_post', 1]:.4f}]")

# By sex
print("\n--- Heterogeneity by Sex ---")
for sex, label in [(0, 'Male'), (1, 'Female')]:
    subset = df_sample[df_sample['female'] == sex]
    model_sex = smf.wls('fulltime ~ treatment_group + C(YEAR) + treat_post',
                        data=subset, weights=subset['PERWT']).fit()
    print(f"  {label}: {model_sex.params['treat_post']:.4f} (SE: {model_sex.bse['treat_post']:.4f})")

# -----------------------------------------------------------------------------
# 9. EVENT STUDY / PARALLEL TRENDS
# -----------------------------------------------------------------------------
print("\n9. PARALLEL TRENDS CHECK (Event Study)...")
print("-" * 40)

# Create year-specific treatment effects (excluding 2011 as reference)
df_sample['ref_year'] = 2011
year_dummies = pd.get_dummies(df_sample['YEAR'], prefix='year')
df_sample = pd.concat([df_sample, year_dummies], axis=1)

# Create interactions
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_sample[f'treat_year_{year}'] = df_sample['treatment_group'] * (df_sample['YEAR'] == year).astype(int)

# Event study regression
formula = 'fulltime ~ treatment_group + C(YEAR) + '
formula += ' + '.join([f'treat_year_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])
event_model = smf.wls(formula, data=df_sample, weights=df_sample['PERWT']).fit()

print("\nEvent study coefficients (2011 = reference):")
print("Year     Coef      SE       95% CI")
print("-" * 50)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_year_{year}'
    coef = event_model.params[var]
    se = event_model.bse[var]
    ci_low, ci_high = event_model.conf_int().loc[var]
    marker = "*" if event_model.pvalues[var] < 0.05 else ""
    print(f"{year}    {coef:7.4f}  {se:6.4f}   [{ci_low:7.4f}, {ci_high:7.4f}] {marker}")

# -----------------------------------------------------------------------------
# 10. SAVE RESULTS
# -----------------------------------------------------------------------------
print("\n10. SAVING RESULTS...")
print("-" * 40)

# Save key results to file for the report
results = {
    'sample_size': int(model4.nobs),
    'n_treatment_pre': len(df_sample[(df_sample['treatment_group']==1) & (df_sample['post']==0)]),
    'n_treatment_post': len(df_sample[(df_sample['treatment_group']==1) & (df_sample['post']==1)]),
    'n_control_pre': len(df_sample[(df_sample['control_group']==1) & (df_sample['post']==0)]),
    'n_control_post': len(df_sample[(df_sample['control_group']==1) & (df_sample['post']==1)]),
    'did_coef': model4.params['treat_post'],
    'did_se': model4.bse['treat_post'],
    'did_ci_low': model4.conf_int().loc['treat_post', 0],
    'did_ci_high': model4.conf_int().loc['treat_post', 1],
    'did_pvalue': model4.pvalues['treat_post'],
    'did_robust_se': model4_robust.bse['treat_post'],
}

# Calculate mean outcomes by group and period
means = df_sample.groupby(['treatment_group', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
results['mean_treat_pre'] = means.loc[1, 0]
results['mean_treat_post'] = means.loc[1, 1]
results['mean_ctrl_pre'] = means.loc[0, 0]
results['mean_ctrl_post'] = means.loc[0, 1]

# Save to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)

# Print final summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
print(f"\nPreferred Estimate (Model 4: Controls + Year FE + State FE)")
print(f"  DiD Coefficient: {results['did_coef']:.4f}")
print(f"  Standard Error:  {results['did_se']:.4f}")
print(f"  Robust SE (HC1): {results['did_robust_se']:.4f}")
print(f"  95% CI: [{results['did_ci_low']:.4f}, {results['did_ci_high']:.4f}]")
print(f"  P-value: {results['did_pvalue']:.4f}")
print(f"  Sample size: {results['sample_size']:,}")

print(f"\nMean Full-Time Employment Rates:")
print(f"                    Pre-DACA    Post-DACA    Change")
print(f"  Treatment (26-30):  {results['mean_treat_pre']:.3f}       {results['mean_treat_post']:.3f}       {results['mean_treat_post']-results['mean_treat_pre']:+.3f}")
print(f"  Control (31-35):    {results['mean_ctrl_pre']:.3f}       {results['mean_ctrl_post']:.3f}       {results['mean_ctrl_post']-results['mean_ctrl_pre']:+.3f}")
print(f"  Difference-in-Differences: {(results['mean_treat_post']-results['mean_treat_pre'])-(results['mean_ctrl_post']-results['mean_ctrl_pre']):+.3f}")

print("\n" + "=" * 80)
print("Analysis complete. Results saved to results_summary.csv")
print("=" * 80)

# Create detailed regression table for latex
print("\n\n--- LATEX TABLE DATA ---")
print("Model 1 (Basic):", model1.params['treat_post'], model1.bse['treat_post'], int(model1.nobs))
print("Model 2 (Year FE):", model2.params['treat_post'], model2.bse['treat_post'], int(model2.nobs))
print("Model 3 (Controls+Year):", model3.params['treat_post'], model3.bse['treat_post'], int(model3.nobs))
print("Model 4 (Full):", model4.params['treat_post'], model4.bse['treat_post'], int(model4.nobs))
