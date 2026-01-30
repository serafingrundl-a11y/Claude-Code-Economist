"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican immigrants born in Mexico

Author: Replication 85
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA (with filtering during read to manage memory)
# ============================================================================
print("\n[1] Loading ACS data (filtering during read)...")
data_path = "data/data.csv"

# Define columns we need
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT', 'UHRSWORK']

# Load and filter in chunks
chunks = []
chunk_size = 500000
total_rows = 0
for chunk in pd.read_csv(data_path, chunksize=chunk_size, usecols=cols_needed, low_memory=False):
    total_rows += len(chunk)
    # Apply filters immediately
    chunk = chunk[chunk['HISPAN'] == 1]  # Hispanic-Mexican
    chunk = chunk[chunk['BPL'] == 200]    # Born in Mexico
    chunk = chunk[chunk['CITIZEN'] == 3]  # Non-citizen
    chunk = chunk[chunk['YEAR'] != 2012]  # Exclude 2012
    if len(chunk) > 0:
        chunks.append(chunk)
    print(f"  Processed {total_rows:,} rows, kept {sum(len(c) for c in chunks):,}")

df = pd.concat(chunks, ignore_index=True)
print(f"  Final filtered rows: {len(df):,}")

# ============================================================================
# STEP 2: COMPUTE AGE AT DACA IMPLEMENTATION (June 15, 2012)
# ============================================================================
print("\n[2] Computing age at DACA implementation...")

# DACA was implemented on June 15, 2012
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# For Q1 (Jan-Mar) and Q2 (Apr-Jun), they likely had their birthday by June 15
# For Q3 (Jul-Sep) and Q4 (Oct-Dec), they had not had their birthday yet

df['age_at_daca'] = np.where(
    df['BIRTHQTR'].isin([1, 2]),
    2012 - df['BIRTHYR'],
    2012 - df['BIRTHYR'] - 1
)
print(f"  Age at DACA range: {df['age_at_daca'].min()} to {df['age_at_daca'].max()}")

# ============================================================================
# STEP 3: DEFINE TREATMENT AND CONTROL GROUPS
# ============================================================================
print("\n[3] Defining treatment and control groups...")

# Filter to only include ages 26-35
df = df[(df['age_at_daca'] >= 26) & (df['age_at_daca'] <= 35)].copy()
print(f"  After age filter (26-35): {len(df):,}")

# Create treatment indicator
df['treated'] = (df['age_at_daca'] <= 30).astype(int)
print(f"  Treated group (age 26-30): {df['treated'].sum():,}")
print(f"  Control group (age 31-35): {(1 - df['treated']).sum():,}")

# ============================================================================
# STEP 4: ADDITIONAL DACA ELIGIBILITY CRITERIA
# ============================================================================
print("\n[4] Applying additional DACA eligibility criteria...")

# DACA requirement: Arrived before 16th birthday
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Filter: Arrived before age 16 (or YRIMMIG == 0 for N/A)
df = df[(df['age_at_immig'] < 16) | (df['YRIMMIG'] == 0)].copy()
print(f"  After arrival-before-16 filter: {len(df):,}")

# DACA requirement: Lived in US since June 15, 2007 (continuous residence)
df = df[(df['YRIMMIG'] <= 2007) | (df['YRIMMIG'] == 0)].copy()
print(f"  After continuous residence filter (arrived by 2007): {len(df):,}")

# Remove observations with YRIMMIG == 0 (missing)
df = df[df['YRIMMIG'] > 0].copy()
print(f"  After removing missing immigration year: {len(df):,}")

# ============================================================================
# STEP 5: CREATE OUTCOME AND TIME VARIABLES
# ============================================================================
print("\n[5] Creating outcome and time variables...")

# Outcome: Full-time employment (usually works 35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
print(f"  Full-time employment rate: {df['fulltime'].mean():.3f}")

# Post-period indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"  Pre-period observations (2006-2011): {(1 - df['post']).sum():,}")
print(f"  Post-period observations (2013-2016): {df['post'].sum():,}")

# Interaction term for DiD
df['treated_post'] = df['treated'] * df['post']

# ============================================================================
# STEP 6: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[6] Descriptive Statistics")
print("=" * 60)

# Sample sizes by group and period
print("\nSample Sizes:")
cross_tab = pd.crosstab(df['treated'], df['post'], margins=True)
cross_tab.index = ['Control (31-35)', 'Treated (26-30)', 'Total']
cross_tab.columns = ['Pre (2006-11)', 'Post (2013-16)', 'Total']
print(cross_tab)

# Full-time employment rates by group and period
print("\nFull-time Employment Rates:")
ft_rates = df.groupby(['treated', 'post'])['fulltime'].mean().unstack()
ft_rates.index = ['Control (31-35)', 'Treated (26-30)']
ft_rates.columns = ['Pre (2006-11)', 'Post (2013-16)']
print(ft_rates.round(4))

# Raw DiD calculation
pre_treated = df[(df['treated'] == 1) & (df['post'] == 0)]['fulltime'].mean()
post_treated = df[(df['treated'] == 1) & (df['post'] == 1)]['fulltime'].mean()
pre_control = df[(df['treated'] == 0) & (df['post'] == 0)]['fulltime'].mean()
post_control = df[(df['treated'] == 0) & (df['post'] == 1)]['fulltime'].mean()

did_raw = (post_treated - pre_treated) - (post_control - pre_control)
print(f"\nRaw DiD Estimate: {did_raw:.4f}")
print(f"  Treated change: {post_treated - pre_treated:.4f}")
print(f"  Control change: {post_control - pre_control:.4f}")

# Weighted rates
print("\nWeighted Full-time Employment Rates:")
for t in [0, 1]:
    for p in [0, 1]:
        subset = df[(df['treated'] == t) & (df['post'] == p)]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        group = 'Treated' if t == 1 else 'Control'
        period = 'Post' if p == 1 else 'Pre'
        print(f"  {group} - {period}: {weighted_mean:.4f}")

# ============================================================================
# STEP 7: MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# ============================================================================
print("\n[7] Main Difference-in-Differences Regression")
print("=" * 60)

# Model 1: Basic DiD (no controls, no weights)
print("\nModel 1: Basic DiD (Unweighted)")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit()
print(model1.summary().tables[1])

# Model 2: DiD with person weights
print("\nModel 2: DiD with Person Weights (PERWT)")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with covariates (weighted)
print("\nModel 3: DiD with Covariates (Weighted)")
# Create additional control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # High school or more
df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # College degree

# Years in US
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']

model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + years_in_us',
                  data=df, weights=df['PERWT']).fit()
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects (weighted)
print("\nModel 4: DiD with Year Fixed Effects (Weighted)")
df['year_factor'] = df['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ treated + C(year_factor) + treated_post',
                  data=df, weights=df['PERWT']).fit()
print(f"DiD coefficient (treated_post): {model4.params['treated_post']:.4f}")
print(f"Standard Error: {model4.bse['treated_post']:.4f}")
print(f"t-statistic: {model4.tvalues['treated_post']:.4f}")
print(f"p-value: {model4.pvalues['treated_post']:.4f}")

# Model 5: DiD with year and state fixed effects (weighted)
print("\nModel 5: DiD with Year and State Fixed Effects (Weighted)")
df['state_factor'] = df['STATEFIP'].astype(str)
model5 = smf.wls('fulltime ~ treated + C(year_factor) + C(state_factor) + treated_post',
                  data=df, weights=df['PERWT']).fit()
print(f"DiD coefficient (treated_post): {model5.params['treated_post']:.4f}")
print(f"Standard Error: {model5.bse['treated_post']:.4f}")
print(f"t-statistic: {model5.tvalues['treated_post']:.4f}")
print(f"p-value: {model5.pvalues['treated_post']:.4f}")

# Model 6: Full specification with controls and fixed effects
print("\nModel 6: Full Specification (Controls + Year + State FE)")
model6 = smf.wls('fulltime ~ treated + C(year_factor) + C(state_factor) + treated_post + female + married + educ_hs + years_in_us',
                  data=df, weights=df['PERWT']).fit()
print(f"DiD coefficient (treated_post): {model6.params['treated_post']:.4f}")
print(f"Standard Error: {model6.bse['treated_post']:.4f}")
print(f"t-statistic: {model6.tvalues['treated_post']:.4f}")
print(f"p-value: {model6.pvalues['treated_post']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['treated_post', 0]:.4f}, {model6.conf_int().loc['treated_post', 1]:.4f}]")

# ============================================================================
# STEP 8: ROBUSTNESS CHECKS
# ============================================================================
print("\n[8] Robustness Checks")
print("=" * 60)

# Robustness 1: Narrower age bands (27-29 vs 32-34)
print("\nRobustness 1: Narrower Age Bands (27-29 vs 32-34)")
df_narrow = df[(df['age_at_daca'] >= 27) & (df['age_at_daca'] <= 34)].copy()
df_narrow = df_narrow[~df_narrow['age_at_daca'].isin([30, 31])].copy()
df_narrow['treated_narrow'] = (df_narrow['age_at_daca'] <= 29).astype(int)
df_narrow['treated_post_narrow'] = df_narrow['treated_narrow'] * df_narrow['post']
model_narrow = smf.wls('fulltime ~ treated_narrow + C(year_factor) + treated_post_narrow',
                        data=df_narrow, weights=df_narrow['PERWT']).fit()
print(f"DiD coefficient: {model_narrow.params['treated_post_narrow']:.4f}")
print(f"SE: {model_narrow.bse['treated_post_narrow']:.4f}")
print(f"p-value: {model_narrow.pvalues['treated_post_narrow']:.4f}")
print(f"N: {len(df_narrow):,}")

# Robustness 2: Alternative outcome - any employment
print("\nRobustness 2: Alternative Outcome - Any Employment (EMPSTAT == 1)")
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
model_emp = smf.wls('employed ~ treated + C(year_factor) + treated_post',
                     data=df, weights=df['PERWT']).fit()
print(f"DiD coefficient: {model_emp.params['treated_post']:.4f}")
print(f"SE: {model_emp.bse['treated_post']:.4f}")
print(f"p-value: {model_emp.pvalues['treated_post']:.4f}")

# Robustness 3: By gender
print("\nRobustness 3: By Gender")
df_male = df[df['female'] == 0]
df_female = df[df['female'] == 1]

model_male = smf.wls('fulltime ~ treated + C(year_factor) + treated_post',
                      data=df_male, weights=df_male['PERWT']).fit()
model_female = smf.wls('fulltime ~ treated + C(year_factor) + treated_post',
                        data=df_female, weights=df_female['PERWT']).fit()
print(f"Males:   DiD = {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f})")
print(f"Females: DiD = {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f})")

# ============================================================================
# STEP 9: EVENT STUDY / PARALLEL TRENDS
# ============================================================================
print("\n[9] Event Study Analysis (Testing Parallel Trends)")
print("=" * 60)

# Create year-by-treatment interactions
years = sorted(df['YEAR'].unique())
print(f"Years in data: {years}")

# Use 2011 as reference year (last pre-treatment year)
event_df = df.copy()
for year in years:
    if year != 2011:  # reference year
        event_df[f'treat_x_{year}'] = (event_df['treated'] * (event_df['YEAR'] == year)).astype(int)

# Build formula
event_vars = [f'treat_x_{year}' for year in years if year != 2011]
formula = 'fulltime ~ treated + C(year_factor) + ' + ' + '.join(event_vars)

model_event = smf.wls(formula, data=event_df, weights=event_df['PERWT']).fit()

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
for year in years:
    if year != 2011:
        var = f'treat_x_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        pval = model_event.pvalues[var]
        print(f"  {year}: {coef:7.4f} (SE: {se:.4f}, p: {pval:.4f})")

# ============================================================================
# STEP 10: SUMMARY OF RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print(f"""
MAIN FINDING:
The difference-in-differences estimate of DACA eligibility on full-time employment
is {model4.params['treated_post']:.4f} (SE: {model4.bse['treated_post']:.4f}).

This indicates that DACA eligibility is associated with a {abs(model4.params['treated_post'])*100:.2f} percentage
point {'increase' if model4.params['treated_post'] > 0 else 'decrease'} in the probability of full-time employment
among Hispanic-Mexican immigrants born in Mexico.

Sample Size: {len(df):,} observations
  - Treated (age 26-30): {df['treated'].sum():,}
  - Control (age 31-35): {(1-df['treated']).sum():,}
  - Pre-period (2006-2011): {(1-df['post']).sum():,}
  - Post-period (2013-2016): {df['post'].sum():,}

Statistical Significance: p-value = {model4.pvalues['treated_post']:.4f}
95% Confidence Interval: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]
""")

# ============================================================================
# STEP 11: SAVE RESULTS FOR REPORT
# ============================================================================
print("\n[11] Saving results...")

# Create results dictionary
results = {
    'sample_size': int(len(df)),
    'treated_n': int(df['treated'].sum()),
    'control_n': int((1-df['treated']).sum()),
    'pre_n': int((1-df['post']).sum()),
    'post_n': int(df['post'].sum()),
    'did_basic': float(model1.params['treated_post']),
    'did_basic_se': float(model1.bse['treated_post']),
    'did_weighted': float(model2.params['treated_post']),
    'did_weighted_se': float(model2.bse['treated_post']),
    'did_yearfe': float(model4.params['treated_post']),
    'did_yearfe_se': float(model4.bse['treated_post']),
    'did_yearfe_pval': float(model4.pvalues['treated_post']),
    'did_full': float(model6.params['treated_post']),
    'did_full_se': float(model6.bse['treated_post']),
    'did_full_pval': float(model6.pvalues['treated_post']),
    'did_full_ci_low': float(model6.conf_int().loc['treated_post', 0]),
    'did_full_ci_high': float(model6.conf_int().loc['treated_post', 1]),
    'ft_pre_treated': float(pre_treated),
    'ft_post_treated': float(post_treated),
    'ft_pre_control': float(pre_control),
    'ft_post_control': float(post_control),
    'did_narrow': float(model_narrow.params['treated_post_narrow']),
    'did_narrow_se': float(model_narrow.bse['treated_post_narrow']),
    'did_emp': float(model_emp.params['treated_post']),
    'did_emp_se': float(model_emp.bse['treated_post']),
    'did_male': float(model_male.params['treated_post']),
    'did_male_se': float(model_male.bse['treated_post']),
    'did_female': float(model_female.params['treated_post']),
    'did_female_se': float(model_female.bse['treated_post']),
}

# Add event study coefficients
for year in years:
    if year != 2011:
        var = f'treat_x_{year}'
        results[f'event_{year}'] = float(model_event.params[var])
        results[f'event_{year}_se'] = float(model_event.bse[var])

# Save to file
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  Results saved to results.json")

# Save descriptive stats
desc_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean'],
    'female': ['mean'],
    'married': ['mean'],
    'educ_hs': ['mean'],
    'years_in_us': ['mean'],
    'AGE': ['mean'],
    'PERWT': ['sum']
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("  Descriptive statistics saved to descriptive_stats.csv")

# Save model summaries
with open('model_summaries.txt', 'w') as f:
    f.write("MODEL 1: Basic DiD (Unweighted)\n")
    f.write("=" * 60 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("MODEL 2: DiD with Weights\n")
    f.write("=" * 60 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("MODEL 4: DiD with Year FE\n")
    f.write("=" * 60 + "\n")
    f.write(str(model4.summary()) + "\n\n")

    f.write("MODEL 6: Full Specification\n")
    f.write("=" * 60 + "\n")
    f.write(str(model6.summary()) + "\n\n")

print("  Model summaries saved to model_summaries.txt")

# Additional statistics for tables
print("\n[12] Creating summary tables for report...")

# Table 1: Sample characteristics
print("\nTable 1: Sample Characteristics by Treatment Group and Period")
char_vars = ['female', 'married', 'educ_hs', 'years_in_us', 'AGE', 'fulltime', 'employed']
for t in [0, 1]:
    for p in [0, 1]:
        subset = df[(df['treated'] == t) & (df['post'] == p)]
        group = 'Treated' if t == 1 else 'Control'
        period = 'Post' if p == 1 else 'Pre'
        print(f"\n{group} - {period} (N = {len(subset):,}):")
        for var in char_vars:
            wmean = np.average(subset[var], weights=subset['PERWT'])
            print(f"  {var}: {wmean:.4f}")

# Table 2: Year-by-year employment rates
print("\nTable 2: Full-time Employment Rates by Year and Group")
yearly_rates = df.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
yearly_rates.columns = ['Control', 'Treated']
print(yearly_rates.round(4))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
