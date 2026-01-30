"""
DACA Replication Analysis - Replication 12
Difference-in-Differences Analysis of DACA Effect on Full-Time Employment

Research Question:
Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for the Deferred Action for Childhood
Arrivals (DACA) program on the probability of full-time employment (>=35 hrs/week)?

Treatment group: Ages 26-30 as of June 15, 2012
Control group: Ages 31-35 as of June 15, 2012
Pre-period: 2006-2011
Post-period: 2013-2016
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
print("DACA REPLICATION ANALYSIS - REPLICATION 12")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load and Filter Data
# ============================================================================
print("STEP 1: Loading and filtering data...")
print("-" * 40)

# Define columns to load (only what we need to save memory)
cols_to_load = [
    'YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
    'MARST', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EMPSTAT', 'LABFORCE', 'UHRSWORK'
]

# Load data in chunks to handle large file
chunk_size = 500000
chunks = []

print("Loading data in chunks...")
for i, chunk in enumerate(pd.read_csv(
    'C:/Users/seraf/DACA Results Task 2/replication_12/data/data.csv',
    usecols=cols_to_load,
    chunksize=chunk_size,
    low_memory=False
)):
    # Filter to relevant population:
    # 1. Hispanic-Mexican ethnicity (HISPAN == 1)
    # 2. Born in Mexico (BPL == 200)
    # 3. Non-citizen (CITIZEN == 3 indicates "Not a citizen")
    filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Mexican Hispanic
        (chunk['BPL'] == 200) &   # Born in Mexico
        (chunk['CITIZEN'] == 3)   # Not a citizen
    ].copy()

    if len(filtered) > 0:
        chunks.append(filtered)

    if (i + 1) % 20 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

print(f"  Finished loading all chunks.")

# Combine all chunks
df = pd.concat(chunks, ignore_index=True)
print(f"  Initial filtered sample: {len(df):,} observations")

# ============================================================================
# STEP 2: Calculate Age as of June 15, 2012
# ============================================================================
print()
print("STEP 2: Calculating age as of June 15, 2012...")
print("-" * 40)

# Calculate age as of June 15, 2012
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# For June 15, 2012:
# - If born in Q1 (Jan-Mar) or Q2 (Apr-Jun before June 15), they've had their birthday
# - If born in Q3 or Q4, they haven't had their birthday yet

# Approximate: Use midpoint of birth quarter
# Q1 midpoint: Feb 15, Q2 midpoint: May 15, Q3 midpoint: Aug 15, Q4 midpoint: Nov 15
# For simplicity, if birthqtr <= 2, they've likely had birthday by June 15

df['age_june2012'] = 2012 - df['BIRTHYR']
# Adjust for those who haven't had their birthday by June 15
# If born in Q3 or Q4, subtract 1 (they haven't had birthday by June 15)
df.loc[df['BIRTHQTR'] >= 3, 'age_june2012'] = df.loc[df['BIRTHQTR'] >= 3, 'age_june2012'] - 1

print(f"  Age calculation complete.")
print(f"  Age range in data: {df['age_june2012'].min()} to {df['age_june2012'].max()}")

# ============================================================================
# STEP 3: Apply DACA Eligibility Criteria
# ============================================================================
print()
print("STEP 3: Applying DACA eligibility criteria...")
print("-" * 40)

# DACA eligibility requirements:
# 1. Arrived before 16th birthday
# 2. Under 31 as of June 15, 2012 (for treatment; control is 31-35)
# 3. Continuous presence since June 15, 2007 (YRIMMIG <= 2007)
# 4. Present in US on June 15, 2012 (assumed from survey)
# 5. No lawful status (CITIZEN == 3, already filtered)

# Calculate age at immigration
# Use year of immigration and birth year
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Filter: arrived before age 16
df = df[df['age_at_immig'] < 16].copy()
print(f"  After requiring arrival before age 16: {len(df):,} observations")

# Filter: in US since at least 2007
df = df[df['YRIMMIG'] <= 2007].copy()
print(f"  After requiring presence since 2007: {len(df):,} observations")

# ============================================================================
# STEP 4: Define Treatment and Control Groups
# ============================================================================
print()
print("STEP 4: Defining treatment and control groups...")
print("-" * 40)

# Treatment group: age 26-30 as of June 15, 2012
# Control group: age 31-35 as of June 15, 2012

df['treat'] = np.where(
    (df['age_june2012'] >= 26) & (df['age_june2012'] <= 30),
    1,
    np.where(
        (df['age_june2012'] >= 31) & (df['age_june2012'] <= 35),
        0,
        np.nan
    )
)

# Keep only treatment and control groups
df = df[df['treat'].notna()].copy()
df['treat'] = df['treat'].astype(int)

print(f"  After selecting age groups 26-30 and 31-35: {len(df):,} observations")
print(f"  Treatment group (26-30): {(df['treat'] == 1).sum():,}")
print(f"  Control group (31-35): {(df['treat'] == 0).sum():,}")

# ============================================================================
# STEP 5: Define Pre/Post Period and Outcome
# ============================================================================
print()
print("STEP 5: Defining pre/post periods and outcome variable...")
print("-" * 40)

# Pre-period: 2006-2011
# Post-period: 2013-2016 (exclude 2012 due to mid-year implementation)
df = df[df['YEAR'] != 2012].copy()

df['post'] = np.where(df['YEAR'] >= 2013, 1, 0)

print(f"  After excluding 2012: {len(df):,} observations")
print(f"  Pre-period (2006-2011): {(df['post'] == 0).sum():,}")
print(f"  Post-period (2013-2016): {(df['post'] == 1).sum():,}")

# Define outcome: full-time employment (UHRSWORK >= 35)
df['fulltime'] = np.where(df['UHRSWORK'] >= 35, 1, 0)

# Also create employed indicator
df['employed'] = np.where(df['EMPSTAT'] == 1, 1, 0)

print()
print(f"  Full-time employment rate: {df['fulltime'].mean():.4f}")
print(f"  Employment rate: {df['employed'].mean():.4f}")

# ============================================================================
# STEP 6: Create Analysis Variables
# ============================================================================
print()
print("STEP 6: Creating analysis variables...")
print("-" * 40)

# Interaction term for DiD
df['treat_post'] = df['treat'] * df['post']

# Create additional covariates
df['female'] = np.where(df['SEX'] == 2, 1, 0)
df['married'] = np.where(df['MARST'] <= 2, 1, 0)  # 1=married spouse present, 2=married spouse absent

# Education categories
df['educ_cat'] = pd.cut(
    df['EDUC'],
    bins=[-1, 5, 6, 7, 10, 11],
    labels=['less_than_hs', 'hs_diploma', 'some_college', 'bachelors_plus', 'grad_degree']
)

# Years in US as of survey year
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']

print("  Created variables: treat_post, female, married, educ_cat, years_in_us")

# ============================================================================
# STEP 7: Summary Statistics
# ============================================================================
print()
print("=" * 80)
print("STEP 7: Summary Statistics")
print("=" * 80)

# Overall sample size by year and treatment status
print()
print("Sample size by year and treatment status:")
print("-" * 50)
crosstab = pd.crosstab(df['YEAR'], df['treat'], margins=True)
crosstab.columns = ['Control (31-35)', 'Treatment (26-30)', 'Total']
print(crosstab)

# Summary statistics by treatment and period
print()
print("Summary Statistics by Treatment Group and Period:")
print("-" * 70)

summary_vars = ['fulltime', 'employed', 'female', 'married', 'AGE', 'years_in_us', 'UHRSWORK']
summary = df.groupby(['treat', 'post'])[summary_vars].agg(['mean', 'std', 'count'])
print(summary.round(4))

# Mean full-time employment by group and period
print()
print("Full-time Employment Rate by Group and Period:")
print("-" * 50)
ft_means = df.groupby(['treat', 'post'])['fulltime'].mean().unstack()
ft_means.index = ['Control (31-35)', 'Treatment (26-30)']
ft_means.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(ft_means.round(4))

# Calculate raw DiD
raw_did = (ft_means.loc['Treatment (26-30)', 'Post (2013-2016)'] -
           ft_means.loc['Treatment (26-30)', 'Pre (2006-2011)']) - \
          (ft_means.loc['Control (31-35)', 'Post (2013-2016)'] -
           ft_means.loc['Control (31-35)', 'Pre (2006-2011)'])
print()
print(f"Raw DiD estimate: {raw_did:.4f}")

# Weighted means
print()
print("Weighted Full-time Employment Rate by Group and Period:")
print("-" * 50)
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

ft_weighted = df.groupby(['treat', 'post']).apply(weighted_mean).unstack()
ft_weighted.index = ['Control (31-35)', 'Treatment (26-30)']
ft_weighted.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(ft_weighted.round(4))

weighted_did = (ft_weighted.loc['Treatment (26-30)', 'Post (2013-2016)'] -
                ft_weighted.loc['Treatment (26-30)', 'Pre (2006-2011)']) - \
               (ft_weighted.loc['Control (31-35)', 'Post (2013-2016)'] -
                ft_weighted.loc['Control (31-35)', 'Pre (2006-2011)'])
print()
print(f"Weighted DiD estimate: {weighted_did:.4f}")

# ============================================================================
# STEP 8: Difference-in-Differences Regression
# ============================================================================
print()
print("=" * 80)
print("STEP 8: Difference-in-Differences Regression Analysis")
print("=" * 80)

# Model 1: Basic DiD (no controls)
print()
print("Model 1: Basic DiD (no controls)")
print("-" * 50)
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print()
print("Model 2: DiD with demographic controls")
print("-" * 50)
model2 = smf.ols('fulltime ~ treat + post + treat_post + female + married + years_in_us',
                 data=df).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with demographic controls and year fixed effects
print()
print("Model 3: DiD with demographic controls and year fixed effects")
print("-" * 50)
df['year_factor'] = pd.Categorical(df['YEAR'])
model3 = smf.ols('fulltime ~ treat + treat_post + female + married + years_in_us + C(YEAR)',
                 data=df).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with demographic controls, year FE, and state FE
print()
print("Model 4: DiD with controls, year FE, and state FE")
print("-" * 50)
model4 = smf.ols('fulltime ~ treat + treat_post + female + married + years_in_us + C(YEAR) + C(STATEFIP)',
                 data=df).fit(cov_type='HC1')
print("Coefficients for main variables:")
print(model4.params[['Intercept', 'treat', 'treat_post', 'female', 'married', 'years_in_us']].round(4))
print()
print("Standard errors:")
print(model4.bse[['Intercept', 'treat', 'treat_post', 'female', 'married', 'years_in_us']].round(4))
print()
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {model4.nobs:.0f}")

# Model 5: Weighted regression (preferred specification)
print()
print("Model 5: Weighted DiD with controls and year FE (PREFERRED)")
print("-" * 50)
model5 = smf.wls('fulltime ~ treat + treat_post + female + married + years_in_us + C(YEAR)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary())

# ============================================================================
# STEP 9: Robustness Checks
# ============================================================================
print()
print("=" * 80)
print("STEP 9: Robustness Checks")
print("=" * 80)

# 9a: Event study / dynamic DiD
print()
print("9a: Event Study Specification")
print("-" * 50)
df['year_rel'] = df['YEAR'].map({
    2006: -6, 2007: -5, 2008: -4, 2009: -3, 2010: -2, 2011: -1,
    2013: 1, 2014: 2, 2015: 3, 2016: 4
})
# Create interaction for each year (omit 2011 as reference)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'treat_yr{yr}'] = df['treat'] * (df['YEAR'] == yr).astype(int)

event_formula = 'fulltime ~ treat + C(YEAR) + ' + ' + '.join([f'treat_yr{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])
event_model = smf.ols(event_formula, data=df).fit(cov_type='HC1')

print("Event study coefficients (relative to 2011):")
event_coefs = event_model.params[[f'treat_yr{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]]
event_se = event_model.bse[[f'treat_yr{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]]
event_results = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016],
    'Coefficient': event_coefs.values,
    'Std Error': event_se.values
})
event_results['95% CI Lower'] = event_results['Coefficient'] - 1.96 * event_results['Std Error']
event_results['95% CI Upper'] = event_results['Coefficient'] + 1.96 * event_results['Std Error']
print(event_results.round(4))

# 9b: Analysis by sex
print()
print("9b: Heterogeneity by Sex")
print("-" * 50)

# Males
df_male = df[df['female'] == 0].copy()
model_male = smf.ols('fulltime ~ treat + post + treat_post + married + years_in_us',
                     data=df_male).fit(cov_type='HC1')
print(f"Males - DiD estimate: {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f})")

# Females
df_female = df[df['female'] == 1].copy()
model_female = smf.ols('fulltime ~ treat + post + treat_post + married + years_in_us',
                       data=df_female).fit(cov_type='HC1')
print(f"Females - DiD estimate: {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f})")

# 9c: Placebo test with different age groups
print()
print("9c: Placebo Test (ages 36-40 vs 41-45)")
print("-" * 50)

# Reload for placebo - need different age group
chunks_placebo = []
for i, chunk in enumerate(pd.read_csv(
    'C:/Users/seraf/DACA Results Task 2/replication_12/data/data.csv',
    usecols=cols_to_load,
    chunksize=chunk_size,
    low_memory=False
)):
    filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3)
    ].copy()
    if len(filtered) > 0:
        chunks_placebo.append(filtered)

df_placebo = pd.concat(chunks_placebo, ignore_index=True)
df_placebo['age_june2012'] = 2012 - df_placebo['BIRTHYR']
df_placebo.loc[df_placebo['BIRTHQTR'] >= 3, 'age_june2012'] -= 1
df_placebo['age_at_immig'] = df_placebo['YRIMMIG'] - df_placebo['BIRTHYR']
df_placebo = df_placebo[
    (df_placebo['age_at_immig'] < 16) &
    (df_placebo['YRIMMIG'] <= 2007)
].copy()

# Placebo groups: 36-40 vs 41-45 (both ineligible)
df_placebo['treat'] = np.where(
    (df_placebo['age_june2012'] >= 36) & (df_placebo['age_june2012'] <= 40), 1,
    np.where(
        (df_placebo['age_june2012'] >= 41) & (df_placebo['age_june2012'] <= 45), 0, np.nan
    )
)
df_placebo = df_placebo[df_placebo['treat'].notna()].copy()
df_placebo = df_placebo[df_placebo['YEAR'] != 2012].copy()
df_placebo['post'] = (df_placebo['YEAR'] >= 2013).astype(int)
df_placebo['treat_post'] = df_placebo['treat'] * df_placebo['post']
df_placebo['fulltime'] = (df_placebo['UHRSWORK'] >= 35).astype(int)
df_placebo['female'] = (df_placebo['SEX'] == 2).astype(int)
df_placebo['married'] = (df_placebo['MARST'] <= 2).astype(int)
df_placebo['years_in_us'] = df_placebo['YEAR'] - df_placebo['YRIMMIG']

placebo_model = smf.ols('fulltime ~ treat + post + treat_post + female + married + years_in_us',
                        data=df_placebo).fit(cov_type='HC1')
print(f"Placebo DiD estimate: {placebo_model.params['treat_post']:.4f} (SE: {placebo_model.bse['treat_post']:.4f})")
print(f"N = {placebo_model.nobs:.0f}")

# ============================================================================
# STEP 10: Final Summary Results
# ============================================================================
print()
print("=" * 80)
print("STEP 10: Summary of Main Results")
print("=" * 80)

print()
print("MAIN FINDINGS:")
print("-" * 70)
print(f"Sample size: {len(df):,}")
print(f"Treatment group (ages 26-30 in 2012): {(df['treat'] == 1).sum():,}")
print(f"Control group (ages 31-35 in 2012): {(df['treat'] == 0).sum():,}")
print()

# Preferred specification (Model 5 - weighted with year FE)
preferred = model5
print("PREFERRED ESTIMATE (Weighted OLS with year FE):")
print("-" * 50)
print(f"DiD Coefficient (treat x post): {preferred.params['treat_post']:.4f}")
print(f"Standard Error (robust): {preferred.bse['treat_post']:.4f}")
print(f"95% Confidence Interval: [{preferred.params['treat_post'] - 1.96*preferred.bse['treat_post']:.4f}, {preferred.params['treat_post'] + 1.96*preferred.bse['treat_post']:.4f}]")
print(f"t-statistic: {preferred.params['treat_post']/preferred.bse['treat_post']:.4f}")
print(f"p-value: {preferred.pvalues['treat_post']:.4f}")
print()

# Table of all specifications
print("Comparison of Specifications:")
print("-" * 70)
results_table = pd.DataFrame({
    'Model': ['Basic DiD', 'With Controls', 'Year FE', 'Year + State FE', 'Weighted (Preferred)'],
    'Coefficient': [
        model1.params['treat_post'],
        model2.params['treat_post'],
        model3.params['treat_post'],
        model4.params['treat_post'],
        model5.params['treat_post']
    ],
    'Std Error': [
        model1.bse['treat_post'],
        model2.bse['treat_post'],
        model3.bse['treat_post'],
        model4.bse['treat_post'],
        model5.bse['treat_post']
    ],
    'N': [
        model1.nobs,
        model2.nobs,
        model3.nobs,
        model4.nobs,
        model5.nobs
    ]
})
results_table['95% CI'] = results_table.apply(
    lambda x: f"[{x['Coefficient']-1.96*x['Std Error']:.4f}, {x['Coefficient']+1.96*x['Std Error']:.4f}]",
    axis=1
)
print(results_table.to_string(index=False))

# ============================================================================
# STEP 11: Export Results for Report
# ============================================================================
print()
print("=" * 80)
print("STEP 11: Exporting results...")
print("=" * 80)

# Save key results to file
results_output = {
    'sample_size': len(df),
    'n_treatment': (df['treat'] == 1).sum(),
    'n_control': (df['treat'] == 0).sum(),
    'did_coef': preferred.params['treat_post'],
    'did_se': preferred.bse['treat_post'],
    'did_ci_lower': preferred.params['treat_post'] - 1.96*preferred.bse['treat_post'],
    'did_ci_upper': preferred.params['treat_post'] + 1.96*preferred.bse['treat_post'],
    'did_pvalue': preferred.pvalues['treat_post'],
    'ft_rate_treat_pre': ft_means.loc['Treatment (26-30)', 'Pre (2006-2011)'],
    'ft_rate_treat_post': ft_means.loc['Treatment (26-30)', 'Post (2013-2016)'],
    'ft_rate_control_pre': ft_means.loc['Control (31-35)', 'Pre (2006-2011)'],
    'ft_rate_control_post': ft_means.loc['Control (31-35)', 'Post (2013-2016)'],
}

# Save to CSV
pd.DataFrame([results_output]).to_csv(
    'C:/Users/seraf/DACA Results Task 2/replication_12/results_summary.csv',
    index=False
)

# Save detailed regression results
with open('C:/Users/seraf/DACA Results Task 2/replication_12/regression_results.txt', 'w') as f:
    f.write("DACA REPLICATION ANALYSIS - REGRESSION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write("Model 1: Basic DiD\n")
    f.write(model1.summary().as_text())
    f.write("\n\n" + "=" * 80 + "\n\n")
    f.write("Model 5: Weighted DiD (Preferred)\n")
    f.write(model5.summary().as_text())

# Save event study results
event_results.to_csv(
    'C:/Users/seraf/DACA Results Task 2/replication_12/event_study_results.csv',
    index=False
)

# Save summary statistics
summary_stats = df.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean', 'std'],
    'female': ['mean'],
    'married': ['mean'],
    'AGE': ['mean', 'std'],
    'years_in_us': ['mean', 'std'],
    'PERWT': ['sum']
}).round(4)
summary_stats.to_csv(
    'C:/Users/seraf/DACA Results Task 2/replication_12/summary_statistics.csv'
)

print("Results exported to:")
print("  - results_summary.csv")
print("  - regression_results.txt")
print("  - event_study_results.csv")
print("  - summary_statistics.csv")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
