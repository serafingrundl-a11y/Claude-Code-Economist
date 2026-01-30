"""
DACA Replication Analysis
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on full-time employment (35+ hours/week)?

Treatment group: Ages 26-30 at the time of DACA implementation (June 15, 2012)
Control group: Ages 31-35 at the time of DACA implementation

Method: Difference-in-Differences
Pre-period: 2006-2011 (excluding 2012 due to ambiguity about survey timing)
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

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")

# Display basic info
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")
print(f"\nColumn names: {list(df.columns)}")

# Summary statistics
print("\n2. INITIAL DATA EXPLORATION...")
print(f"\nYear distribution:")
print(df['YEAR'].value_counts().sort_index())

# Step 3: Define sample restrictions
print("\n3. APPLYING SAMPLE RESTRICTIONS...")

# DACA eligibility criteria based on instructions:
# - Hispanic-Mexican ethnicity (HISPAN == 1)
# - Born in Mexico (BPL == 200)
# - Not a citizen (CITIZEN == 3 - "Not a citizen")
# - Arrived before age 16
# - Lived in US continuously since June 15, 2007 (arrived by 2007)
# - Present in US on June 15, 2012

# First, let's check the values for key variables
print("\nChecking HISPAN values:")
print(df['HISPAN'].value_counts().head(10))

print("\nChecking BPL values (showing top values):")
print(df['BPL'].value_counts().head(10))

print("\nChecking CITIZEN values:")
print(df['CITIZEN'].value_counts())

# Filter for Hispanic-Mexican
df_sample = df[df['HISPAN'] == 1].copy()
print(f"\nAfter filtering for Hispanic-Mexican (HISPAN==1): {len(df_sample):,}")

# Filter for born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After filtering for born in Mexico (BPL==200): {len(df_sample):,}")

# Filter for non-citizens (CITIZEN == 3)
# Per instructions: "Assume that anyone who is not a citizen and who has not received
# immigration papers is undocumented for DACA purposes"
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After filtering for non-citizens (CITIZEN==3): {len(df_sample):,}")

# DACA eligibility by age: must not have turned 31 before June 15, 2012
# Treatment group: ages 26-30 as of June 15, 2012
# Control group: ages 31-35 as of June 15, 2012

# Calculate age as of June 15, 2012
# Since we only have BIRTHYR and BIRTHQTR, we'll approximate
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

# A person's age on June 15, 2012:
# If born in Q1 or Q2 (Jan-Jun), they've had their birthday by June 15
# If born in Q3 or Q4 (Jul-Dec), they haven't had their birthday yet

def age_on_june15_2012(birthyr, birthqtr):
    """Calculate age on June 15, 2012"""
    # June 15 falls in Q2
    if birthqtr in [1, 2]:  # Born Jan-Jun, birthday has passed
        return 2012 - birthyr
    else:  # Born Jul-Dec, birthday hasn't passed
        return 2012 - birthyr - 1

df_sample['age_june2012'] = df_sample.apply(
    lambda x: age_on_june15_2012(x['BIRTHYR'], x['BIRTHQTR']), axis=1
)

print(f"\nAge (as of June 15, 2012) distribution in sample:")
print(df_sample['age_june2012'].describe())

# DACA required arrival before 16th birthday
# We need YRIMMIG (year of immigration) and compare to BIRTHYR
# Age at arrival = YRIMMIG - BIRTHYR (approximately)

df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

print(f"\nAge at arrival distribution:")
print(df_sample['age_at_arrival'].describe())

# Filter for those who arrived before age 16
df_sample = df_sample[df_sample['age_at_arrival'] < 16].copy()
print(f"After filtering for arrived before age 16: {len(df_sample):,}")

# DACA required continuous presence since June 15, 2007 (arrived by 2007)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
print(f"After filtering for arrived by 2007: {len(df_sample):,}")

# Define treatment and control groups based on age on June 15, 2012
# Treatment: 26-30 (DACA eligible)
# Control: 31-35 (too old for DACA but otherwise similar)

df_sample['treat_group'] = np.where(
    (df_sample['age_june2012'] >= 26) & (df_sample['age_june2012'] <= 30), 1,
    np.where((df_sample['age_june2012'] >= 31) & (df_sample['age_june2012'] <= 35), 0, np.nan)
)

# Keep only treatment and control age ranges
df_analysis = df_sample[df_sample['treat_group'].notna()].copy()
print(f"\nAfter keeping only ages 26-35 (as of June 2012): {len(df_analysis):,}")

print(f"\nTreatment group (ages 26-30): {(df_analysis['treat_group']==1).sum():,}")
print(f"Control group (ages 31-35): {(df_analysis['treat_group']==0).sum():,}")

# Define pre/post periods
# Pre: 2006-2011, Post: 2013-2016
# Exclude 2012 because we can't distinguish before/after DACA implementation

df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = np.where(df_analysis['YEAR'] >= 2013, 1, 0)

print(f"\nAfter excluding 2012: {len(df_analysis):,}")
print(f"Pre-period (2006-2011): {(df_analysis['post']==0).sum():,}")
print(f"Post-period (2013-2016): {(df_analysis['post']==1).sum():,}")

# Define outcome: Full-time employment (35+ hours/week)
# UHRSWORK: Usual hours worked per week
# Full-time = UHRSWORK >= 35

df_analysis['fulltime'] = np.where(df_analysis['UHRSWORK'] >= 35, 1, 0)

print(f"\n4. OUTCOME VARIABLE: FULL-TIME EMPLOYMENT (35+ hours/week)")
print(f"Overall full-time employment rate: {df_analysis['fulltime'].mean():.4f}")

# Check distribution by group and period
print("\nFull-time employment rates by group and period:")
ft_rates = df_analysis.groupby(['treat_group', 'post'])['fulltime'].agg(['mean', 'count', 'sum'])
print(ft_rates)

# Calculate raw DiD
ft_by_group_period = df_analysis.groupby(['treat_group', 'post'])['fulltime'].mean().unstack()
print("\nFull-time employment rates (mean):")
print(ft_by_group_period)

# Raw DiD calculation
treat_diff = ft_by_group_period.loc[1, 1] - ft_by_group_period.loc[1, 0]
control_diff = ft_by_group_period.loc[0, 1] - ft_by_group_period.loc[0, 0]
raw_did = treat_diff - control_diff

print(f"\nRaw Difference-in-Differences:")
print(f"Treatment group change (post - pre): {treat_diff:.4f}")
print(f"Control group change (post - pre): {control_diff:.4f}")
print(f"DiD estimate: {raw_did:.4f}")

# 5. REGRESSION ANALYSIS
print("\n5. REGRESSION ANALYSIS")

# Create interaction term
df_analysis['treat_post'] = df_analysis['treat_group'] * df_analysis['post']

# Model 1: Basic DiD without covariates
print("\n--- Model 1: Basic DiD (no covariates, no weights) ---")
model1 = smf.ols('fulltime ~ treat_group + post + treat_post', data=df_analysis).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with person weights (PERWT)
print("\n--- Model 2: DiD with person weights ---")
model2 = smf.wls('fulltime ~ treat_group + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with covariates (weighted)
print("\n--- Model 3: DiD with covariates (weighted) ---")

# Create covariates
df_analysis['female'] = np.where(df_analysis['SEX'] == 2, 1, 0)
df_analysis['married'] = np.where(df_analysis['MARST'].isin([1, 2]), 1, 0)

# Education categories
df_analysis['educ_hs'] = np.where(df_analysis['EDUC'] == 6, 1, 0)  # High school
df_analysis['educ_somecoll'] = np.where(df_analysis['EDUC'].isin([7, 8, 9]), 1, 0)  # Some college
df_analysis['educ_coll'] = np.where(df_analysis['EDUC'] >= 10, 1, 0)  # College+

# Age (current age in survey year)
df_analysis['age'] = df_analysis['AGE']
df_analysis['age_sq'] = df_analysis['age'] ** 2

model3 = smf.wls('fulltime ~ treat_group + post + treat_post + female + married + educ_hs + educ_somecoll + educ_coll + age + age_sq',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with year fixed effects (weighted)
print("\n--- Model 4: DiD with year fixed effects (weighted) ---")
df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)

model4 = smf.wls('fulltime ~ treat_group + treat_post + C(year_factor) + female + married + educ_hs + educ_somecoll + educ_coll + age + age_sq',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# Model 5: Add state fixed effects
print("\n--- Model 5: DiD with year and state fixed effects (weighted) ---")
df_analysis['state_factor'] = df_analysis['STATEFIP'].astype(str)

model5 = smf.wls('fulltime ~ treat_group + treat_post + C(year_factor) + C(state_factor) + female + married + educ_hs + educ_somecoll + educ_coll',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model5.summary2().tables[1].to_string())

# 6. ROBUSTNESS CHECKS
print("\n6. ROBUSTNESS CHECKS")

# 6a. Placebo test: Use only pre-period data with "fake" treatment at 2009
print("\n--- 6a. Placebo Test (pre-period only, fake treatment at 2009) ---")
df_pre = df_analysis[df_analysis['post'] == 0].copy()
df_pre['placebo_post'] = np.where(df_pre['YEAR'] >= 2009, 1, 0)
df_pre['placebo_treat_post'] = df_pre['treat_group'] * df_pre['placebo_post']

model_placebo = smf.wls('fulltime ~ treat_group + placebo_post + placebo_treat_post',
                         data=df_pre,
                         weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD coefficient: {model_placebo.params['placebo_treat_post']:.4f}")
print(f"Placebo DiD std error: {model_placebo.bse['placebo_treat_post']:.4f}")
print(f"Placebo DiD p-value: {model_placebo.pvalues['placebo_treat_post']:.4f}")

# 6b. Alternative age windows
print("\n--- 6b. Alternative age windows (24-28 vs 33-37) ---")
df_sample_alt = df_sample.copy()
df_sample_alt['treat_group_alt'] = np.where(
    (df_sample_alt['age_june2012'] >= 24) & (df_sample_alt['age_june2012'] <= 28), 1,
    np.where((df_sample_alt['age_june2012'] >= 33) & (df_sample_alt['age_june2012'] <= 37), 0, np.nan)
)
df_analysis_alt = df_sample_alt[df_sample_alt['treat_group_alt'].notna()].copy()
df_analysis_alt = df_analysis_alt[df_analysis_alt['YEAR'] != 2012]
df_analysis_alt['post'] = np.where(df_analysis_alt['YEAR'] >= 2013, 1, 0)
df_analysis_alt['fulltime'] = np.where(df_analysis_alt['UHRSWORK'] >= 35, 1, 0)
df_analysis_alt['treat_post_alt'] = df_analysis_alt['treat_group_alt'] * df_analysis_alt['post']

model_alt = smf.wls('fulltime ~ treat_group_alt + post + treat_post_alt',
                     data=df_analysis_alt,
                     weights=df_analysis_alt['PERWT']).fit(cov_type='HC1')
print(f"Alternative window DiD coefficient: {model_alt.params['treat_post_alt']:.4f}")
print(f"Alternative window DiD std error: {model_alt.bse['treat_post_alt']:.4f}")
print(f"Alternative window DiD p-value: {model_alt.pvalues['treat_post_alt']:.4f}")

# 6c. By gender
print("\n--- 6c. Effects by gender ---")
df_analysis['female'] = np.where(df_analysis['SEX'] == 2, 1, 0)

for gender, label in [(0, 'Male'), (1, 'Female')]:
    df_gender = df_analysis[df_analysis['female'] == gender]
    model_g = smf.wls('fulltime ~ treat_group + post + treat_post',
                       data=df_gender,
                       weights=df_gender['PERWT']).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_g.params['treat_post']:.4f} (SE: {model_g.bse['treat_post']:.4f}, p: {model_g.pvalues['treat_post']:.4f})")

# 7. EVENT STUDY
print("\n7. EVENT STUDY (Year-by-Year Effects)")

# Create year dummies interacted with treatment
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    df_analysis[f'year_{year}'] = np.where(df_analysis['YEAR'] == year, 1, 0)
    df_analysis[f'treat_year_{year}'] = df_analysis['treat_group'] * df_analysis[f'year_{year}']

# Use 2011 as reference year
year_terms = ' + '.join([f'year_{y}' for y in years if y != 2011])
treat_year_terms = ' + '.join([f'treat_year_{y}' for y in years if y != 2011])

formula_event = f'fulltime ~ treat_group + {year_terms} + {treat_year_terms}'
model_event = smf.wls(formula_event, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Treatment x Year, reference: 2011):")
for year in years:
    if year != 2011:
        coef = model_event.params.get(f'treat_year_{year}', np.nan)
        se = model_event.bse.get(f'treat_year_{year}', np.nan)
        pval = model_event.pvalues.get(f'treat_year_{year}', np.nan)
        print(f"  {year}: {coef:.4f} (SE: {se:.4f}, p: {pval:.4f})")

# 8. SUMMARY STATISTICS
print("\n8. SUMMARY STATISTICS")

# Create summary stats table
summary_vars = ['fulltime', 'female', 'married', 'age', 'educ_hs', 'educ_somecoll', 'educ_coll']

print("\nPre-period summary statistics:")
pre_stats = df_analysis[df_analysis['post'] == 0].groupby('treat_group')[summary_vars].mean()
print(pre_stats.T)

print("\nPost-period summary statistics:")
post_stats = df_analysis[df_analysis['post'] == 1].groupby('treat_group')[summary_vars].mean()
print(post_stats.T)

# 9. FINAL RESULTS SUMMARY
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"\nSample size: {len(df_analysis):,}")
print(f"Treatment group (ages 26-30 on June 15, 2012): {(df_analysis['treat_group']==1).sum():,}")
print(f"Control group (ages 31-35 on June 15, 2012): {(df_analysis['treat_group']==0).sum():,}")

print("\n--- PREFERRED ESTIMATE: Model 4 (Year FE + Covariates, Weighted) ---")
preferred_coef = model4.params['treat_post']
preferred_se = model4.bse['treat_post']
preferred_ci = model4.conf_int().loc['treat_post']
preferred_pval = model4.pvalues['treat_post']

print(f"Effect size (coefficient on treat x post): {preferred_coef:.4f}")
print(f"Standard error: {preferred_se:.4f}")
print(f"95% Confidence interval: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"P-value: {preferred_pval:.4f}")

print("\n--- ALL MODEL COMPARISONS ---")
models = [
    ('Model 1: Basic DiD', model1, 'treat_post'),
    ('Model 2: Weighted', model2, 'treat_post'),
    ('Model 3: Weighted + Covariates', model3, 'treat_post'),
    ('Model 4: Year FE + Covariates', model4, 'treat_post'),
    ('Model 5: Year + State FE', model5, 'treat_post'),
]

print(f"\n{'Model':<35} {'Coef':>10} {'SE':>10} {'p-value':>10} {'N':>10}")
print("-" * 75)
for name, model, var in models:
    print(f"{name:<35} {model.params[var]:>10.4f} {model.bse[var]:>10.4f} {model.pvalues[var]:>10.4f} {int(model.nobs):>10}")

# Save key results to a file for the report
results_dict = {
    'sample_size': len(df_analysis),
    'n_treatment': int((df_analysis['treat_group']==1).sum()),
    'n_control': int((df_analysis['treat_group']==0).sum()),
    'preferred_coef': preferred_coef,
    'preferred_se': preferred_se,
    'preferred_ci_low': preferred_ci[0],
    'preferred_ci_high': preferred_ci[1],
    'preferred_pval': preferred_pval,
    'raw_did': raw_did,
    'ft_treat_pre': ft_by_group_period.loc[1, 0],
    'ft_treat_post': ft_by_group_period.loc[1, 1],
    'ft_control_pre': ft_by_group_period.loc[0, 0],
    'ft_control_post': ft_by_group_period.loc[0, 1],
}

import json
with open('results_summary.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\nResults saved to results_summary.json")
print("\nAnalysis complete!")
