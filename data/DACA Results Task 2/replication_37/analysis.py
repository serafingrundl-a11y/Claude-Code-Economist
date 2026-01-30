"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals

Method: Difference-in-Differences
Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
Control: Ages 31-35 at DACA implementation
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
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: Load and Filter Data
# ============================================================================
print("\n[1] Loading data...")

# Define columns to load (to reduce memory usage)
cols_to_use = [
    'YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'UHRSWORK', 'MARST', 'NCHILD'
]

# Load data in chunks to handle large file
chunk_size = 500000
chunks = []

print("Reading data in chunks...")
for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=cols_to_use, chunksize=chunk_size)):
    # Apply initial filters within each chunk to reduce memory
    # Filter 1: Hispanic-Mexican (HISPAN == 1)
    chunk = chunk[chunk['HISPAN'] == 1]
    # Filter 2: Born in Mexico (BPL == 200)
    chunk = chunk[chunk['BPL'] == 200]
    # Filter 3: Not a citizen (CITIZEN == 3)
    chunk = chunk[chunk['CITIZEN'] == 3]
    # Filter 4: Years 2006-2016 (should be all data, but confirm)
    chunk = chunk[(chunk['YEAR'] >= 2006) & (chunk['YEAR'] <= 2016)]

    if len(chunk) > 0:
        chunks.append(chunk)

    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"After initial filters: {len(df):,} observations")

# ============================================================================
# STEP 2: Create DACA Eligibility Variables
# ============================================================================
print("\n[2] Creating DACA eligibility variables...")

# Calculate age at DACA implementation (June 15, 2012)
# Using birth year and quarter to approximate
# Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec
# For June 15, 2012, people born before June 15 in 2012 are a certain age
# We'll use birth year to calculate approximate age at June 2012

df['age_at_daca'] = 2012 - df['BIRTHYR']

# Adjust for birth quarter (approximate)
# If born in Q3 or Q4 (after June), they would be 1 year younger at June 2012
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_at_daca'] = df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_at_daca'] - 1

# Treatment group: ages 26-30 at DACA (born ~1982-1986)
# Control group: ages 31-35 at DACA (born ~1977-1981)
df['treatment_group'] = ((df['age_at_daca'] >= 26) & (df['age_at_daca'] <= 30)).astype(int)
df['control_group'] = ((df['age_at_daca'] >= 31) & (df['age_at_daca'] <= 35)).astype(int)
df['in_sample'] = df['treatment_group'] | df['control_group']

# Filter to only include treatment and control groups
df = df[df['in_sample'] == 1].copy()
print(f"After age filtering (26-35 at DACA): {len(df):,} observations")

# ============================================================================
# STEP 3: Apply Additional DACA Eligibility Criteria
# ============================================================================
print("\n[3] Applying additional DACA eligibility criteria...")

# Criterion: Arrived in US before 16th birthday
# YRIMMIG - BIRTHYR < 16
df['arrived_before_16'] = (df['YRIMMIG'] - df['BIRTHYR']) < 16

# Criterion: Lived in US since June 15, 2007
# YRIMMIG <= 2007
df['in_us_by_2007'] = df['YRIMMIG'] <= 2007

# Combine eligibility criteria (aside from age which defines groups)
df['other_eligible'] = df['arrived_before_16'] & df['in_us_by_2007']

# Filter to those meeting other eligibility criteria
df_eligible = df[df['other_eligible']].copy()
print(f"After applying DACA eligibility criteria: {len(df_eligible):,} observations")

# ============================================================================
# STEP 4: Create Outcome and Treatment Variables
# ============================================================================
print("\n[4] Creating outcome and treatment variables...")

# Outcome: Full-time employment (35+ hours per week)
df_eligible['fulltime'] = (df_eligible['UHRSWORK'] >= 35).astype(int)

# Post-DACA indicator (2013-2016)
# Excluding 2012 because DACA was implemented mid-year
df_eligible['post'] = (df_eligible['YEAR'] >= 2013).astype(int)

# Treatment indicator (1 = treatment group, 0 = control group)
df_eligible['treat'] = df_eligible['treatment_group']

# Interaction term for DiD
df_eligible['treat_post'] = df_eligible['treat'] * df_eligible['post']

# Create analysis sample: exclude 2012
df_analysis = df_eligible[df_eligible['YEAR'] != 2012].copy()
print(f"Analysis sample (excluding 2012): {len(df_analysis):,} observations")

# ============================================================================
# STEP 5: Create Control Variables
# ============================================================================
print("\n[5] Creating control variables...")

# Education categories
df_analysis['educ_lesshs'] = (df_analysis['EDUC'] < 6).astype(int)  # Less than high school
df_analysis['educ_hs'] = (df_analysis['EDUC'] == 6).astype(int)  # High school
df_analysis['educ_somecol'] = ((df_analysis['EDUC'] >= 7) & (df_analysis['EDUC'] <= 9)).astype(int)  # Some college
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # College+

# Sex (female indicator)
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Marital status (married indicator)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Has children
df_analysis['has_children'] = (df_analysis['NCHILD'] > 0).astype(int)

# Age at survey (for additional control)
df_analysis['age'] = df_analysis['AGE']
df_analysis['age_sq'] = df_analysis['age'] ** 2

# Years in US
df_analysis['years_in_us'] = df_analysis['YEAR'] - df_analysis['YRIMMIG']

print("Control variables created.")

# ============================================================================
# STEP 6: Descriptive Statistics
# ============================================================================
print("\n" + "=" * 80)
print("[6] DESCRIPTIVE STATISTICS")
print("=" * 80)

# Sample sizes by group and period
print("\n--- Sample Sizes ---")
sample_sizes = df_analysis.groupby(['treat', 'post']).agg(
    n=('fulltime', 'count'),
    n_weighted=('PERWT', 'sum')
).round(0)
sample_sizes.index = pd.MultiIndex.from_tuples(
    [(f"{'Treatment' if t else 'Control'}", f"{'Post' if p else 'Pre'}")
     for t, p in sample_sizes.index],
    names=['Group', 'Period']
)
print(sample_sizes)

# Mean full-time employment by group and period
print("\n--- Mean Full-Time Employment Rate ---")
ft_means = df_analysis.groupby(['treat', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
ft_means.index = ['Control (31-35)', 'Treatment (26-30)']
ft_means.columns = ['Pre-DACA (2006-11)', 'Post-DACA (2013-16)']
print(ft_means.round(4))

# Simple DiD calculation
print("\n--- Simple Difference-in-Differences ---")
# Pre means
pre_treat = df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==0)]
pre_ctrl = df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==0)]
post_treat = df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==1)]
post_ctrl = df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==1)]

mean_pre_treat = np.average(pre_treat['fulltime'], weights=pre_treat['PERWT'])
mean_pre_ctrl = np.average(pre_ctrl['fulltime'], weights=pre_ctrl['PERWT'])
mean_post_treat = np.average(post_treat['fulltime'], weights=post_treat['PERWT'])
mean_post_ctrl = np.average(post_ctrl['fulltime'], weights=post_ctrl['PERWT'])

diff_treat = mean_post_treat - mean_pre_treat
diff_ctrl = mean_post_ctrl - mean_pre_ctrl
did_estimate = diff_treat - diff_ctrl

print(f"Treatment group change: {diff_treat:.4f}")
print(f"Control group change: {diff_ctrl:.4f}")
print(f"DiD estimate: {did_estimate:.4f}")

# Covariate balance
print("\n--- Covariate Means by Group (Pre-Period) ---")
pre_data = df_analysis[df_analysis['post'] == 0]
covariates = ['female', 'married', 'has_children', 'educ_lesshs', 'educ_hs', 'educ_somecol', 'educ_college', 'years_in_us']

balance_table = pd.DataFrame()
for var in covariates:
    treat_mean = np.average(pre_data[pre_data['treat']==1][var], weights=pre_data[pre_data['treat']==1]['PERWT'])
    ctrl_mean = np.average(pre_data[pre_data['treat']==0][var], weights=pre_data[pre_data['treat']==0]['PERWT'])
    balance_table.loc[var, 'Treatment'] = treat_mean
    balance_table.loc[var, 'Control'] = ctrl_mean
    balance_table.loc[var, 'Difference'] = treat_mean - ctrl_mean

print(balance_table.round(4))

# ============================================================================
# STEP 7: Difference-in-Differences Regression Models
# ============================================================================
print("\n" + "=" * 80)
print("[7] REGRESSION ANALYSIS")
print("=" * 80)

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"DiD Coefficient (treat_post): {model1.params['treat_post']:.4f}")
print(f"Standard Error: {model1.bse['treat_post']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model1.pvalues['treat_post']:.4f}")
print(f"N: {int(model1.nobs):,}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + has_children + age + age_sq',
                  data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"DiD Coefficient (treat_post): {model2.params['treat_post']:.4f}")
print(f"Standard Error: {model2.bse['treat_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model2.pvalues['treat_post']:.4f}")

# Model 3: DiD with demographic + education controls
print("\n--- Model 3: DiD with Demographic + Education Controls ---")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + has_children + age + age_sq + educ_hs + educ_somecol + educ_college',
                  data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"DiD Coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"Standard Error: {model3.bse['treat_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: Full model with year and state fixed effects
print("\n--- Model 4: Full Model with Year Fixed Effects ---")
# Create year dummies
for year in df_analysis['YEAR'].unique():
    if year != 2006:  # Reference year
        df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype(int)

year_dummies = ' + '.join([f'year_{y}' for y in sorted(df_analysis['YEAR'].unique()) if y != 2006])
formula4 = f'fulltime ~ treat + treat_post + female + married + has_children + age + age_sq + educ_hs + educ_somecol + educ_college + {year_dummies}'

model4 = smf.wls(formula4, data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"DiD Coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"Standard Error: {model4.bse['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model4.pvalues['treat_post']:.4f}")

# Model 5: Add state fixed effects (using top states)
print("\n--- Model 5: Full Model with Year and State Fixed Effects ---")
# Get top 10 states by sample size
top_states = df_analysis['STATEFIP'].value_counts().head(10).index.tolist()
for state in top_states:
    df_analysis[f'state_{state}'] = (df_analysis['STATEFIP'] == state).astype(int)

state_dummies = ' + '.join([f'state_{s}' for s in top_states[:-1]])  # Exclude one for reference
formula5 = f'fulltime ~ treat + treat_post + female + married + has_children + age + age_sq + educ_hs + educ_somecol + educ_college + years_in_us + {year_dummies} + {state_dummies}'

model5 = smf.wls(formula5, data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"DiD Coefficient (treat_post): {model5.params['treat_post']:.4f}")
print(f"Standard Error: {model5.bse['treat_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model5.pvalues['treat_post']:.4f}")

# ============================================================================
# STEP 8: Robustness Check - Clustered Standard Errors
# ============================================================================
print("\n" + "=" * 80)
print("[8] ROBUSTNESS: Clustered Standard Errors by State")
print("=" * 80)

# Model with clustered standard errors
model_cluster = smf.wls(formula5, data=df_analysis, weights=df_analysis['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"DiD Coefficient (treat_post): {model_cluster.params['treat_post']:.4f}")
print(f"Clustered SE: {model_cluster.bse['treat_post']:.4f}")
print(f"95% CI: [{model_cluster.conf_int().loc['treat_post', 0]:.4f}, {model_cluster.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model_cluster.pvalues['treat_post']:.4f}")

# ============================================================================
# STEP 9: Event Study / Pre-Trend Analysis
# ============================================================================
print("\n" + "=" * 80)
print("[9] EVENT STUDY - Year-by-Year Effects")
print("=" * 80)

# Create year-specific treatment interactions
for year in sorted(df_analysis['YEAR'].unique()):
    df_analysis[f'treat_year_{year}'] = df_analysis['treat'] * (df_analysis['YEAR'] == year).astype(int)

# Exclude 2011 as reference year (last pre-treatment year)
event_vars = ' + '.join([f'treat_year_{y}' for y in sorted(df_analysis['YEAR'].unique()) if y != 2011])
formula_event = f'fulltime ~ treat + {year_dummies} + {event_vars} + female + married + has_children + age + age_sq + educ_hs + educ_somecol + educ_college'

model_event = smf.wls(formula_event, data=df_analysis, weights=df_analysis['PERWT']).fit()

print("\nYear-by-Year Treatment Effects (Reference: 2011):")
print("-" * 50)
for year in sorted(df_analysis['YEAR'].unique()):
    if year != 2011:
        var = f'treat_year_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        pval = model_event.pvalues[var]
        print(f"{year}: {coef:7.4f} (SE: {se:.4f}, p={pval:.3f})")

# ============================================================================
# STEP 10: Subgroup Analysis
# ============================================================================
print("\n" + "=" * 80)
print("[10] SUBGROUP ANALYSIS")
print("=" * 80)

# By sex
print("\n--- By Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex_val]
    model_sub = smf.wls('fulltime ~ treat + post + treat_post + married + has_children + age + age_sq + educ_hs + educ_somecol + educ_college',
                        data=df_sub, weights=df_sub['PERWT']).fit()
    print(f"{sex_name}: DiD = {model_sub.params['treat_post']:.4f} (SE: {model_sub.bse['treat_post']:.4f}), N = {int(model_sub.nobs):,}")

# By education
print("\n--- By Education Level ---")
for educ_val, educ_name in [(1, 'Less than HS'), (0, 'HS or more')]:
    if educ_val == 1:
        df_sub = df_analysis[df_analysis['educ_lesshs'] == 1]
    else:
        df_sub = df_analysis[df_analysis['educ_lesshs'] == 0]
    model_sub = smf.wls('fulltime ~ treat + post + treat_post + female + married + has_children + age + age_sq',
                        data=df_sub, weights=df_sub['PERWT']).fit()
    print(f"{educ_name}: DiD = {model_sub.params['treat_post']:.4f} (SE: {model_sub.bse['treat_post']:.4f}), N = {int(model_sub.nobs):,}")

# ============================================================================
# STEP 11: Save Results
# ============================================================================
print("\n" + "=" * 80)
print("[11] SAVING RESULTS")
print("=" * 80)

# Create results summary
results_summary = {
    'Model': ['Basic DiD', 'With Demographics', 'With Education', 'Year FE', 'Year + State FE', 'Clustered SE'],
    'Coefficient': [model1.params['treat_post'], model2.params['treat_post'], model3.params['treat_post'],
                   model4.params['treat_post'], model5.params['treat_post'], model_cluster.params['treat_post']],
    'SE': [model1.bse['treat_post'], model2.bse['treat_post'], model3.bse['treat_post'],
           model4.bse['treat_post'], model5.bse['treat_post'], model_cluster.bse['treat_post']],
    'CI_lower': [model1.conf_int().loc['treat_post', 0], model2.conf_int().loc['treat_post', 0],
                 model3.conf_int().loc['treat_post', 0], model4.conf_int().loc['treat_post', 0],
                 model5.conf_int().loc['treat_post', 0], model_cluster.conf_int().loc['treat_post', 0]],
    'CI_upper': [model1.conf_int().loc['treat_post', 1], model2.conf_int().loc['treat_post', 1],
                 model3.conf_int().loc['treat_post', 1], model4.conf_int().loc['treat_post', 1],
                 model5.conf_int().loc['treat_post', 1], model_cluster.conf_int().loc['treat_post', 1]],
    'P_value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'], model3.pvalues['treat_post'],
               model4.pvalues['treat_post'], model5.pvalues['treat_post'], model_cluster.pvalues['treat_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs), int(model_cluster.nobs)]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save full model output
with open('model_output.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DACA REPLICATION STUDY - FULL MODEL OUTPUT\n")
    f.write("=" * 80 + "\n\n")

    f.write("PREFERRED MODEL (Model 5: Year + State Fixed Effects)\n")
    f.write("-" * 80 + "\n")
    f.write(model5.summary().as_text())
    f.write("\n\n")

    f.write("MODEL WITH CLUSTERED STANDARD ERRORS\n")
    f.write("-" * 80 + "\n")
    f.write(model_cluster.summary().as_text())
    f.write("\n\n")

    f.write("EVENT STUDY MODEL\n")
    f.write("-" * 80 + "\n")
    f.write(model_event.summary().as_text())

print("Full model output saved to model_output.txt")

# ============================================================================
# STEP 12: Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
PREFERRED ESTIMATE (Model 5 with Clustered SEs):
------------------------------------------------
Effect Size (DiD coefficient): {model_cluster.params['treat_post']:.4f}
Standard Error (Clustered):    {model_cluster.bse['treat_post']:.4f}
95% Confidence Interval:       [{model_cluster.conf_int().loc['treat_post', 0]:.4f}, {model_cluster.conf_int().loc['treat_post', 1]:.4f}]
P-value:                       {model_cluster.pvalues['treat_post']:.4f}
Sample Size:                   {int(model_cluster.nobs):,}

INTERPRETATION:
The difference-in-differences estimate suggests that DACA eligibility
{"increased" if model_cluster.params['treat_post'] > 0 else "decreased"} the probability of full-time employment by
{abs(model_cluster.params['treat_post']):.1%} ({abs(model_cluster.params['treat_post'])*100:.2f} percentage points).
This effect is {"statistically significant" if model_cluster.pvalues['treat_post'] < 0.05 else "not statistically significant"} at the 5% level.
""")

# Save descriptive statistics for LaTeX
desc_stats = {
    'Statistic': ['Treatment Pre-Mean', 'Treatment Post-Mean', 'Control Pre-Mean', 'Control Post-Mean',
                  'Treatment Change', 'Control Change', 'DiD Estimate'],
    'Value': [mean_pre_treat, mean_post_treat, mean_pre_ctrl, mean_post_ctrl,
              diff_treat, diff_ctrl, did_estimate]
}
pd.DataFrame(desc_stats).to_csv('descriptive_stats.csv', index=False)
print("\nDescriptive statistics saved to descriptive_stats.csv")

# Save sample sizes
sample_sizes_full = df_analysis.groupby(['treat', 'post']).agg(
    n_unweighted=('fulltime', 'count'),
    n_weighted=('PERWT', 'sum'),
    mean_fulltime=('fulltime', lambda x: np.average(x, weights=df_analysis.loc[x.index, 'PERWT']))
).reset_index()
sample_sizes_full['treat'] = sample_sizes_full['treat'].map({0: 'Control', 1: 'Treatment'})
sample_sizes_full['post'] = sample_sizes_full['post'].map({0: 'Pre-DACA', 1: 'Post-DACA'})
sample_sizes_full.to_csv('sample_sizes.csv', index=False)
print("Sample sizes saved to sample_sizes.csv")

# Save event study coefficients
event_coefs = []
for year in sorted(df_analysis['YEAR'].unique()):
    if year != 2011:
        var = f'treat_year_{year}'
        event_coefs.append({
            'Year': year,
            'Coefficient': model_event.params[var],
            'SE': model_event.bse[var],
            'CI_lower': model_event.conf_int().loc[var, 0],
            'CI_upper': model_event.conf_int().loc[var, 1]
        })
    else:
        event_coefs.append({
            'Year': 2011,
            'Coefficient': 0,
            'SE': 0,
            'CI_lower': 0,
            'CI_upper': 0
        })
pd.DataFrame(event_coefs).to_csv('event_study_coefs.csv', index=False)
print("Event study coefficients saved to event_study_coefs.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
