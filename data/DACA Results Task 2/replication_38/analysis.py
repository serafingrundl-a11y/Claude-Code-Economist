"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals.

Design: Difference-in-Differences
Treatment: Ages 26-30 as of June 15, 2012
Control: Ages 31-35 as of June 15, 2012
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = "data/data.csv"
OUTPUT_DIR = "."

# DACA implementation date reference
DACA_DATE_YEAR = 2012
DACA_DATE_MONTH = 6
DACA_DATE_DAY = 15

print("=" * 60)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 60)

# -----------------------------------------------------------------------------
# STEP 1: Load and filter data
# -----------------------------------------------------------------------------
print("\n[Step 1] Loading data...")

# Define columns we need
cols_needed = [
    'YEAR', 'PERWT', 'STATEFIP', 'AGE', 'BIRTHYR', 'BIRTHQTR',
    'SEX', 'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'UHRSWORK', 'MARST',
    'NCHILD', 'FAMSIZE'
]

# Read data in chunks to handle large file
chunks = []
chunk_size = 500000

for chunk in pd.read_csv(DATA_PATH, usecols=cols_needed, chunksize=chunk_size):
    # Filter to Hispanic-Mexican born in Mexico
    # HISPAN == 1 means Mexican
    # BPL == 200 means Mexico
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)].copy()
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"  Processed chunk, kept {len(filtered)} Hispanic-Mexican born in Mexico")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal Hispanic-Mexican born in Mexico: {len(df):,}")

# -----------------------------------------------------------------------------
# STEP 2: Apply DACA eligibility criteria
# -----------------------------------------------------------------------------
print("\n[Step 2] Applying DACA eligibility criteria...")

# Criterion: Not a citizen (CITIZEN == 3)
# Per instructions: assume non-citizens without papers are undocumented
df = df[df['CITIZEN'] == 3].copy()
print(f"  After non-citizen filter: {len(df):,}")

# Calculate age as of June 15, 2012
# Age on June 15, 2012 = 2012 - BIRTHYR (adjusted by birth quarter)
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# June 15 is in Q2, so:
# - If born in Q1 or Q2 (Jan-Jun), they would have had birthday by June 15
# - If born in Q3 or Q4 (Jul-Dec), birthday is after June 15

def calc_age_june2012(row):
    """Calculate age as of June 15, 2012"""
    base_age = 2012 - row['BIRTHYR']
    # If born after June (Q3 or Q4), subtract 1 year
    if row['BIRTHQTR'] in [3, 4]:
        base_age -= 1
    return base_age

df['age_june2012'] = df.apply(calc_age_june2012, axis=1)

# Filter to relevant age groups (26-35 as of June 2012)
# Treatment: 26-30, Control: 31-35
df = df[(df['age_june2012'] >= 26) & (df['age_june2012'] <= 35)].copy()
print(f"  After age 26-35 filter (as of June 2012): {len(df):,}")

# Create treatment indicator (1 = treated group, ages 26-30)
df['treated'] = (df['age_june2012'] <= 30).astype(int)

# Criterion: Arrived before age 16
# Age at immigration = YRIMMIG - BIRTHYR
# Must be < 16
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[(df['age_at_immigration'] < 16) & (df['age_at_immigration'] >= 0)].copy()
print(f"  After arrived before age 16 filter: {len(df):,}")

# Criterion: Present in US since June 15, 2007
# Using YRIMMIG <= 2007 as proxy for continuous residence
df = df[df['YRIMMIG'] <= 2007].copy()
print(f"  After residence since 2007 filter: {len(df):,}")

# Exclude 2012 (cannot distinguish pre/post within year)
df = df[df['YEAR'] != 2012].copy()
print(f"  After excluding 2012: {len(df):,}")

# Keep only years 2006-2016
df = df[(df['YEAR'] >= 2006) & (df['YEAR'] <= 2016)].copy()
print(f"  After keeping 2006-2011, 2013-2016: {len(df):,}")

# -----------------------------------------------------------------------------
# STEP 3: Create outcome and analysis variables
# -----------------------------------------------------------------------------
print("\n[Step 3] Creating outcome and analysis variables...")

# Outcome: Full-time employment (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Post-DACA indicator (2013 onwards)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term for DID
df['treated_post'] = df['treated'] * df['post']

# Create control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
def educ_category(educd):
    if educd <= 61:  # Less than high school
        return 1
    elif educd <= 64:  # High school
        return 2
    elif educd <= 100:  # Some college
        return 3
    else:  # Bachelor's or higher
        return 4

df['educ_cat'] = df['EDUCD'].apply(educ_category)

print(f"\nFinal sample size: {len(df):,}")
print(f"Treated group (ages 26-30): {df['treated'].sum():,}")
print(f"Control group (ages 31-35): {(1-df['treated']).sum():,}")

# -----------------------------------------------------------------------------
# STEP 4: Descriptive Statistics
# -----------------------------------------------------------------------------
print("\n[Step 4] Descriptive Statistics")
print("=" * 60)

# Summary by treatment and time period
summary_vars = ['fulltime', 'employed', 'UHRSWORK', 'female', 'married', 'AGE']

print("\n--- Pre-DACA Period (2006-2011) ---")
pre_df = df[df['post'] == 0]
print(f"\nTreatment Group (Ages 26-30):")
print(pre_df[pre_df['treated'] == 1][summary_vars].describe().round(3))
print(f"\nControl Group (Ages 31-35):")
print(pre_df[pre_df['treated'] == 0][summary_vars].describe().round(3))

print("\n--- Post-DACA Period (2013-2016) ---")
post_df = df[df['post'] == 1]
print(f"\nTreatment Group (Ages 26-30):")
print(post_df[post_df['treated'] == 1][summary_vars].describe().round(3))
print(f"\nControl Group (Ages 31-35):")
print(post_df[post_df['treated'] == 0][summary_vars].describe().round(3))

# Calculate cell means
print("\n--- Full-time Employment Rates by Group and Period ---")
cell_means = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
print(cell_means)

# Weighted means
def weighted_mean(df, var, weight):
    return np.average(df[var], weights=df[weight])

print("\n--- Weighted Full-time Employment Rates ---")
for treated in [0, 1]:
    for post in [0, 1]:
        subset = df[(df['treated'] == treated) & (df['post'] == post)]
        wt_mean = weighted_mean(subset, 'fulltime', 'PERWT')
        group = "Treatment" if treated == 1 else "Control"
        period = "Post" if post == 1 else "Pre"
        print(f"{group}, {period}: {wt_mean:.4f} (n={len(subset):,})")

# Simple DID calculation
pre_treat = weighted_mean(df[(df['treated']==1) & (df['post']==0)], 'fulltime', 'PERWT')
post_treat = weighted_mean(df[(df['treated']==1) & (df['post']==1)], 'fulltime', 'PERWT')
pre_control = weighted_mean(df[(df['treated']==0) & (df['post']==0)], 'fulltime', 'PERWT')
post_control = weighted_mean(df[(df['treated']==0) & (df['post']==1)], 'fulltime', 'PERWT')

did_simple = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n--- Simple DID Estimate ---")
print(f"Treatment change: {post_treat:.4f} - {pre_treat:.4f} = {post_treat - pre_treat:.4f}")
print(f"Control change: {post_control:.4f} - {pre_control:.4f} = {post_control - pre_control:.4f}")
print(f"DID estimate: {did_simple:.4f}")

# -----------------------------------------------------------------------------
# STEP 5: Main Regression Analysis
# -----------------------------------------------------------------------------
print("\n[Step 5] Regression Analysis")
print("=" * 60)

# Model 1: Basic DID (no controls, no weights)
print("\n--- Model 1: Basic DID (OLS, no controls) ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit()
print(model1.summary())

# Model 2: DID with controls
print("\n--- Model 2: DID with demographic controls ---")
model2 = smf.ols('fulltime ~ treated + post + treated_post + female + married + C(educ_cat)',
                 data=df).fit()
print(model2.summary())

# Model 3: DID with year fixed effects
print("\n--- Model 3: DID with year fixed effects ---")
model3 = smf.ols('fulltime ~ treated + treated_post + female + married + C(educ_cat) + C(YEAR)',
                 data=df).fit()
print(model3.summary())

# Model 4: DID with year and state fixed effects
print("\n--- Model 4: DID with year and state fixed effects ---")
model4 = smf.ols('fulltime ~ treated + treated_post + female + married + C(educ_cat) + C(YEAR) + C(STATEFIP)',
                 data=df).fit(cov_type='HC1')
print(model4.summary())

# Model 5: Weighted regression (preferred specification)
print("\n--- Model 5: Weighted DID with controls and fixed effects (PREFERRED) ---")
model5 = smf.wls('fulltime ~ treated + treated_post + female + married + C(educ_cat) + C(YEAR) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary())

# Extract key results
did_coef = model5.params['treated_post']
did_se = model5.bse['treated_post']
did_ci = model5.conf_int().loc['treated_post']
did_pval = model5.pvalues['treated_post']

print("\n" + "=" * 60)
print("PREFERRED ESTIMATE SUMMARY")
print("=" * 60)
print(f"DID Coefficient (treated_post): {did_coef:.4f}")
print(f"Standard Error: {did_se:.4f}")
print(f"95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
print(f"P-value: {did_pval:.4f}")
print(f"Sample Size: {int(model5.nobs):,}")

# -----------------------------------------------------------------------------
# STEP 6: Robustness Checks
# -----------------------------------------------------------------------------
print("\n[Step 6] Robustness Checks")
print("=" * 60)

# Robustness 1: Linear probability model with clustered standard errors by state
print("\n--- Robustness 1: Clustered SE by state ---")
# Using grouped adjustment for clusters
from statsmodels.stats.sandwich_covariance import cov_cluster
model_robust1 = smf.wls('fulltime ~ treated + treated_post + female + married + C(educ_cat) + C(YEAR) + C(STATEFIP)',
                        data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                          cov_kwds={'groups': df['STATEFIP']})
print(f"DID with clustered SE: {model_robust1.params['treated_post']:.4f} (SE: {model_robust1.bse['treated_post']:.4f})")

# Robustness 2: Different age bandwidths
print("\n--- Robustness 2: Narrower age bandwidth (27-29 vs 32-34) ---")
df_narrow = df[(df['age_june2012'].isin([27, 28, 29])) | (df['age_june2012'].isin([32, 33, 34]))].copy()
df_narrow['treated'] = (df_narrow['age_june2012'] <= 29).astype(int)
df_narrow['treated_post'] = df_narrow['treated'] * df_narrow['post']
model_narrow = smf.wls('fulltime ~ treated + treated_post + female + married + C(educ_cat) + C(YEAR)',
                       data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"Narrow bandwidth DID: {model_narrow.params['treated_post']:.4f} (SE: {model_narrow.bse['treated_post']:.4f})")

# Robustness 3: Employed outcome instead of full-time
print("\n--- Robustness 3: Any employment outcome ---")
model_employed = smf.wls('employed ~ treated + treated_post + female + married + C(educ_cat) + C(YEAR) + C(STATEFIP)',
                         data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Employment DID: {model_employed.params['treated_post']:.4f} (SE: {model_employed.bse['treated_post']:.4f})")

# Robustness 4: By sex
print("\n--- Robustness 4: Heterogeneity by sex ---")
df_male = df[df['female'] == 0]
df_female = df[df['female'] == 1]

model_male = smf.wls('fulltime ~ treated + treated_post + married + C(educ_cat) + C(YEAR)',
                     data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ treated + treated_post + married + C(educ_cat) + C(YEAR)',
                       data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"Male DID: {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f})")
print(f"Female DID: {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f})")

# Robustness 5: Pre-trends (placebo test)
print("\n--- Robustness 5: Pre-trends test (2009 placebo) ---")
df_pretrends = df[df['YEAR'] <= 2011].copy()
df_pretrends['post_placebo'] = (df_pretrends['YEAR'] >= 2009).astype(int)
df_pretrends['treated_post_placebo'] = df_pretrends['treated'] * df_pretrends['post_placebo']
model_placebo = smf.wls('fulltime ~ treated + treated_post_placebo + female + married + C(educ_cat) + C(YEAR)',
                        data=df_pretrends, weights=df_pretrends['PERWT']).fit(cov_type='HC1')
print(f"Placebo DID (2009 cutoff): {model_placebo.params['treated_post_placebo']:.4f} (SE: {model_placebo.bse['treated_post_placebo']:.4f})")

# -----------------------------------------------------------------------------
# STEP 7: Event Study
# -----------------------------------------------------------------------------
print("\n[Step 7] Event Study")
print("=" * 60)

# Create year dummies interacted with treatment
years = sorted(df['YEAR'].unique())
base_year = 2011  # Last pre-treatment year

for year in years:
    if year != base_year:
        df[f'treated_year_{year}'] = (df['treated'] * (df['YEAR'] == year)).astype(int)

# Construct formula for event study
year_interactions = ' + '.join([f'treated_year_{y}' for y in years if y != base_year])
event_formula = f'fulltime ~ treated + {year_interactions} + female + married + C(educ_cat) + C(YEAR)'

model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
event_results = []
for year in years:
    if year != base_year:
        coef = model_event.params[f'treated_year_{year}']
        se = model_event.bse[f'treated_year_{year}']
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        event_results.append({'Year': year, 'Coefficient': coef, 'SE': se, 'CI_low': ci_low, 'CI_high': ci_high})
        print(f"  {year}: {coef:.4f} (SE: {se:.4f})")

event_df = pd.DataFrame(event_results)

# -----------------------------------------------------------------------------
# STEP 8: Save Results
# -----------------------------------------------------------------------------
print("\n[Step 8] Saving Results")
print("=" * 60)

# Save summary statistics
summary_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'UHRSWORK': 'mean',
    'female': 'mean',
    'married': 'mean',
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')

# Save event study results
event_df.to_csv('event_study_results.csv', index=False)

# Save yearly means for plotting
yearly_means = df.groupby(['YEAR', 'treated']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x),
        'total_weight': x['PERWT'].sum()
    })
).reset_index()
yearly_means.to_csv('yearly_means.csv', index=False)

# Create results dictionary for report
results = {
    'preferred_estimate': {
        'coefficient': did_coef,
        'standard_error': did_se,
        'ci_lower': did_ci[0],
        'ci_upper': did_ci[1],
        'p_value': did_pval,
        'sample_size': int(model5.nobs)
    },
    'simple_did': did_simple,
    'cell_means': {
        'pre_treat': pre_treat,
        'post_treat': post_treat,
        'pre_control': pre_control,
        'post_control': post_control
    },
    'robustness': {
        'clustered_se': model_robust1.bse['treated_post'],
        'narrow_bandwidth': model_narrow.params['treated_post'],
        'employment_outcome': model_employed.params['treated_post'],
        'male': model_male.params['treated_post'],
        'female': model_female.params['treated_post'],
        'placebo': model_placebo.params['treated_post_placebo']
    }
}

# Save as JSON-like format
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to:")
print("  - summary_statistics.csv")
print("  - event_study_results.csv")
print("  - yearly_means.csv")
print("  - analysis_results.json")

# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"""
PREFERRED SPECIFICATION: Weighted Linear Probability Model
- Outcome: Full-time employment (35+ hours/week)
- Treatment: DACA-eligible ages 26-30 vs control ages 31-35
- Method: Difference-in-Differences
- Controls: Sex, marital status, education, year FE, state FE
- Weights: ACS person weights (PERWT)

MAIN RESULT:
- DID Estimate: {did_coef:.4f} ({did_coef*100:.2f} percentage points)
- Standard Error: {did_se:.4f}
- 95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]
- P-value: {did_pval:.4f}
- Sample Size: {int(model5.nobs):,}

INTERPRETATION:
DACA eligibility is associated with a {abs(did_coef)*100:.2f} percentage point
{"increase" if did_coef > 0 else "decrease"} in the probability of full-time employment
among eligible Hispanic-Mexican immigrants aged 26-30 compared to the
slightly older control group (ages 31-35).
This effect is {"statistically significant" if did_pval < 0.05 else "not statistically significant"} at the 5% level.

ROBUSTNESS:
- Clustered SE: {model_robust1.params['treated_post']:.4f} (SE: {model_robust1.bse['treated_post']:.4f})
- Narrow bandwidth (27-29 vs 32-34): {model_narrow.params['treated_post']:.4f}
- Any employment outcome: {model_employed.params['treated_post']:.4f}
- Males only: {model_male.params['treated_post']:.4f}
- Females only: {model_female.params['treated_post']:.4f}
- Pre-trends placebo (2009): {model_placebo.params['treated_post_placebo']:.4f}
""")

print("Analysis complete!")
