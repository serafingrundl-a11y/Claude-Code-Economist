"""
DACA Replication Study - Analysis Script
Run 21: Independent Replication

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States.

Treatment: Ages 26-30 as of June 15, 2012
Control: Ages 31-35 as of June 15, 2012
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("DACA REPLICATION STUDY - RUN 21")
print("="*80)

# =============================================================================
# STEP 1: Load Data in Chunks (due to large file size)
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA (CHUNKED)")
print("="*80)

# Define the columns we need
usecols = ['YEAR', 'BIRTHYR', 'BIRTHQTR', 'HISPAN', 'BPL', 'CITIZEN',
           'YRIMMIG', 'UHRSWORK', 'EMPSTAT', 'PERWT', 'AGE', 'SEX',
           'MARST', 'EDUC', 'STATEFIP']

# Filter as we read to manage memory
print("Loading and filtering ACS data from data.csv...")

# Key filtering values
# HISPAN == 1 (Mexican)
# BPL == 200 (Mexico)
# CITIZEN == 3 (Not a citizen)

chunks = []
chunk_size = 500000
total_rows = 0

for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunk_size):
    total_rows += len(chunk)
    # Apply immediate filters
    filtered = chunk[
        (chunk['HISPAN'] == 1) &   # Mexican ethnicity
        (chunk['BPL'] == 200) &     # Born in Mexico
        (chunk['CITIZEN'] == 3) &   # Not a citizen
        (chunk['YRIMMIG'] <= 2007) & # In US since 2007
        (chunk['YEAR'] != 2012)     # Exclude 2012
    ].copy()
    if len(filtered) > 0:
        chunks.append(filtered)
    del chunk, filtered
    gc.collect()
    print(f"  Processed {total_rows:,} rows...")

print(f"Total rows in file: {total_rows:,}")
df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()
print(f"Observations after initial filtering: {len(df):,}")

# =============================================================================
# STEP 2: Define Analysis Parameters
# =============================================================================
print("\n" + "="*80)
print("STEP 2: DEFINING ANALYSIS PARAMETERS")
print("="*80)

# DACA was implemented on June 15, 2012
DACA_DATE_YEAR = 2012

# Age groups as of June 15, 2012
TREATMENT_AGE_MIN = 26
TREATMENT_AGE_MAX = 30
CONTROL_AGE_MIN = 31
CONTROL_AGE_MAX = 35

# Pre-period: before DACA (2006-2011)
# Post-period: after DACA (2013-2016)
PRE_YEARS = [2006, 2007, 2008, 2009, 2010, 2011]
POST_YEARS = [2013, 2014, 2015, 2016]

print(f"DACA implementation date: June 15, {DACA_DATE_YEAR}")
print(f"Treatment group: Ages {TREATMENT_AGE_MIN}-{TREATMENT_AGE_MAX} as of June 15, 2012")
print(f"Control group: Ages {CONTROL_AGE_MIN}-{CONTROL_AGE_MAX} as of June 15, 2012")
print(f"Pre-period years: {PRE_YEARS}")
print(f"Post-period years: {POST_YEARS}")

# =============================================================================
# STEP 3: Calculate Age as of June 15, 2012
# =============================================================================
print("\n" + "="*80)
print("STEP 3: CALCULATING AGE AS OF JUNE 15, 2012")
print("="*80)

# Calculate age as of June 15, 2012
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born in Q3 or Q4 (after mid-June), they haven't had their birthday yet

df['age_june15_2012'] = 2012 - df['BIRTHYR']
# Subtract 1 for those born after June 15 (Q3, Q4)
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june15_2012'] -= 1

print(f"Age range calculated: {df['age_june15_2012'].min()} to {df['age_june15_2012'].max()}")

# =============================================================================
# STEP 4: Apply Remaining Sample Restrictions
# =============================================================================
print("\n" + "="*80)
print("STEP 4: APPLYING REMAINING SAMPLE RESTRICTIONS")
print("="*80)

print(f"\nObservations after initial filtering: {len(df):,}")

# Restriction: Arrived before age 16
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_immigration'] < 16].copy()
print(f"After restricting to arrived before age 16: {len(df):,}")

# Restriction: Age groups for treatment and control (26-35 as of June 15, 2012)
df = df[(df['age_june15_2012'] >= TREATMENT_AGE_MIN) &
        (df['age_june15_2012'] <= CONTROL_AGE_MAX)].copy()
print(f"After restricting to ages 26-35 as of June 15, 2012: {len(df):,}")

# =============================================================================
# STEP 5: Create Analysis Variables
# =============================================================================
print("\n" + "="*80)
print("STEP 5: CREATING ANALYSIS VARIABLES")
print("="*80)

# Treatment indicator (1 if age 26-30 as of June 15, 2012)
df['treat'] = ((df['age_june15_2012'] >= TREATMENT_AGE_MIN) &
               (df['age_june15_2012'] <= TREATMENT_AGE_MAX)).astype(int)

# Post-period indicator (1 if year >= 2013)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term (DiD estimate)
df['treat_post'] = df['treat'] * df['post']

# Full-time employment outcome (working 35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create an employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

print(f"\nTreatment group (ages 26-30): {df['treat'].sum():,}")
print(f"Control group (ages 31-35): {(1 - df['treat']).sum():,}")
print(f"\nPre-period observations: {(1 - df['post']).sum():,}")
print(f"Post-period observations: {df['post'].sum():,}")

# Create current age variable
df['current_age'] = df['AGE']

# Create control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)

# =============================================================================
# STEP 6: Descriptive Statistics
# =============================================================================
print("\n" + "="*80)
print("STEP 6: DESCRIPTIVE STATISTICS")
print("="*80)

# Summary statistics by treatment and period
print("\nSample sizes by group and period:")
print(df.groupby(['treat', 'post']).size().unstack())

# Cell means for DiD calculation (weighted)
cell_means = df.groupby(['treat', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

print("\n" + "-"*50)
print("Weighted Full-Time Employment Rates:")
print("-"*50)
print(f"                      Pre-DACA    Post-DACA    Change")
print(f"Treatment (26-30):    {cell_means.loc[1, 0]:.4f}      {cell_means.loc[1, 1]:.4f}       {cell_means.loc[1, 1] - cell_means.loc[1, 0]:+.4f}")
print(f"Control (31-35):      {cell_means.loc[0, 0]:.4f}      {cell_means.loc[0, 1]:.4f}       {cell_means.loc[0, 1] - cell_means.loc[0, 0]:+.4f}")
print("-"*50)
simple_did = (cell_means.loc[1, 1] - cell_means.loc[1, 0]) - (cell_means.loc[0, 1] - cell_means.loc[0, 0])
print(f"Simple DiD Estimate:  {simple_did:+.4f}")

# Unweighted cell means
cell_means_unw = df.groupby(['treat', 'post'])['fulltime'].mean().unstack()
print("\nUnweighted Full-Time Employment Rates:")
print(f"                      Pre-DACA    Post-DACA    Change")
print(f"Treatment (26-30):    {cell_means_unw.loc[1, 0]:.4f}      {cell_means_unw.loc[1, 1]:.4f}       {cell_means_unw.loc[1, 1] - cell_means_unw.loc[1, 0]:+.4f}")
print(f"Control (31-35):      {cell_means_unw.loc[0, 0]:.4f}      {cell_means_unw.loc[0, 1]:.4f}       {cell_means_unw.loc[0, 1] - cell_means_unw.loc[0, 0]:+.4f}")

# Descriptive stats by treatment group
print("\n" + "-"*50)
print("Descriptive Statistics by Treatment Group (Weighted):")
print("-"*50)
for t in [0, 1]:
    group_name = "Treatment (26-30)" if t == 1 else "Control (31-35)"
    subset = df[df['treat'] == t]
    print(f"\n{group_name}:")
    print(f"  N = {len(subset):,}")
    print(f"  Full-time employment rate: {np.average(subset['fulltime'], weights=subset['PERWT']):.4f}")
    print(f"  Employment rate: {np.average(subset['employed'], weights=subset['PERWT']):.4f}")
    print(f"  Female proportion: {np.average(subset['female'], weights=subset['PERWT']):.4f}")
    print(f"  Married proportion: {np.average(subset['married'], weights=subset['PERWT']):.4f}")
    print(f"  HS or more education: {np.average(subset['educ_hs'], weights=subset['PERWT']):.4f}")
    print(f"  Mean age at immigration: {np.average(subset['age_at_immigration'], weights=subset['PERWT']):.1f}")

# =============================================================================
# STEP 7: Main DiD Regression Analysis
# =============================================================================
print("\n" + "="*80)
print("STEP 7: DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("="*80)

# Model 1: Basic DiD (weighted)
print("\n--- Model 1: Basic DiD (Weighted) ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df,
                  weights=df['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary())

# Store results
did_estimate_1 = results1.params['treat_post']
did_se_1 = results1.bse['treat_post']
did_ci_1 = results1.conf_int().loc['treat_post']

print(f"\n*** MAIN RESULT (Model 1) ***")
print(f"DiD Coefficient: {did_estimate_1:.6f}")
print(f"Standard Error:  {did_se_1:.6f}")
print(f"95% CI: [{did_ci_1[0]:.6f}, {did_ci_1[1]:.6f}]")
print(f"p-value: {results1.pvalues['treat_post']:.6f}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Controls (Weighted) ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + current_age',
                  data=df,
                  weights=df['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary())

did_estimate_2 = results2.params['treat_post']
did_se_2 = results2.bse['treat_post']
did_ci_2 = results2.conf_int().loc['treat_post']

# Model 3: Year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
# Create year dummies
years_in_data = sorted(df['YEAR'].unique())
for year in years_in_data:
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)

year_vars = [f'year_{y}' for y in years_in_data[1:]]  # Exclude first year as reference
formula3 = 'fulltime ~ treat + treat_post + ' + ' + '.join(year_vars)

model3 = smf.wls(formula3, data=df, weights=df['PERWT'])
results3 = model3.fit(cov_type='HC1')

did_estimate_3 = results3.params['treat_post']
did_se_3 = results3.bse['treat_post']
print(f"DiD with Year FE: {did_estimate_3:.6f} (SE: {did_se_3:.6f})")

# Model 4: Full specification with controls and year FE
print("\n--- Model 4: Full Specification ---")
formula4 = 'fulltime ~ treat + treat_post + female + married + educ_hs + current_age + ' + ' + '.join(year_vars)
model4 = smf.wls(formula4, data=df, weights=df['PERWT'])
results4 = model4.fit(cov_type='HC1')

did_estimate_4 = results4.params['treat_post']
did_se_4 = results4.bse['treat_post']
did_ci_4 = results4.conf_int().loc['treat_post']

print(f"DiD Full Specification: {did_estimate_4:.6f} (SE: {did_se_4:.6f})")
print(f"95% CI: [{did_ci_4[0]:.6f}, {did_ci_4[1]:.6f}]")

# =============================================================================
# STEP 8: Robustness Checks
# =============================================================================
print("\n" + "="*80)
print("STEP 8: ROBUSTNESS CHECKS")
print("="*80)

# Check 1: Unweighted regression
print("\n--- Robustness Check 1: Unweighted DiD ---")
model_unw = smf.ols('fulltime ~ treat + post + treat_post', data=df)
results_unw = model_unw.fit(cov_type='HC1')
print(f"Unweighted DiD: {results_unw.params['treat_post']:.6f} (SE: {results_unw.bse['treat_post']:.6f})")

# Check 2: Clustered SE by state
print("\n--- Robustness Check 2: Clustered SE by State ---")
model_clust = smf.wls('fulltime ~ treat + post + treat_post',
                       data=df,
                       weights=df['PERWT'])
results_clust = model_clust.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"Clustered DiD: {results_clust.params['treat_post']:.6f} (SE: {results_clust.bse['treat_post']:.6f})")

# Check 3: Pre-trends test
print("\n--- Robustness Check 3: Pre-Trends Test ---")
df_pre = df[df['YEAR'] <= 2011].copy()
df_pre['year_trend'] = df_pre['YEAR'] - 2006
df_pre['treat_trend'] = df_pre['treat'] * df_pre['year_trend']

model_pretrend = smf.wls('fulltime ~ treat + year_trend + treat_trend',
                          data=df_pre,
                          weights=df_pre['PERWT'])
results_pretrend = model_pretrend.fit(cov_type='HC1')
print(f"Pre-trend interaction: {results_pretrend.params['treat_trend']:.6f} (SE: {results_pretrend.bse['treat_trend']:.6f})")
print(f"p-value: {results_pretrend.pvalues['treat_trend']:.4f}")
if results_pretrend.pvalues['treat_trend'] > 0.05:
    print("No significant differential pre-trend detected (parallel trends assumption supported)")
else:
    print("WARNING: Significant differential pre-trend detected")

# =============================================================================
# STEP 9: Event Study / Dynamic Effects
# =============================================================================
print("\n" + "="*80)
print("STEP 9: EVENT STUDY / DYNAMIC EFFECTS")
print("="*80)

# Create year-specific treatment effects
# Reference year: 2011 (last pre-treatment year)
event_results = {}
for year in years_in_data:
    if year != 2011:
        df[f'treat_year_{year}'] = (df['treat'] * (df['YEAR'] == year)).astype(int)

year_interact_vars = [f'treat_year_{y}' for y in years_in_data if y != 2011]
formula_event = 'fulltime ~ treat + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(year_interact_vars)

model_event = smf.wls(formula_event, data=df, weights=df['PERWT'])
results_event = model_event.fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-"*60)
print(f"{'Year':<10} {'Coefficient':>12} {'SE':>12} {'p-value':>12} {'Sig':>6}")
print("-"*60)
for year in sorted(years_in_data):
    if year != 2011:
        var_name = f'treat_year_{year}'
        coef = results_event.params[var_name]
        se = results_event.bse[var_name]
        pval = results_event.pvalues[var_name]
        sig = "*" if pval < 0.05 else ""
        sig = "**" if pval < 0.01 else sig
        sig = "***" if pval < 0.001 else sig
        event_results[year] = {'coef': coef, 'se': se, 'pval': pval}
        print(f"{year:<10} {coef:>12.4f} {se:>12.4f} {pval:>12.4f} {sig:>6}")

# =============================================================================
# STEP 10: Heterogeneity Analysis
# =============================================================================
print("\n" + "="*80)
print("STEP 10: HETEROGENEITY ANALYSIS")
print("="*80)

# By gender
print("\n--- By Gender ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex_val]
    model_sex = smf.wls('fulltime ~ treat + post + treat_post',
                         data=df_sex,
                         weights=df_sex['PERWT'])
    results_sex = model_sex.fit(cov_type='HC1')
    print(f"{sex_name}: {results_sex.params['treat_post']:.4f} (SE: {results_sex.bse['treat_post']:.4f}), N={len(df_sex):,}")

# By education
print("\n--- By Education ---")
for educ_val, educ_name in [(0, 'Less than HS'), (1, 'HS or more')]:
    df_educ = df[df['educ_hs'] == educ_val]
    model_educ = smf.wls('fulltime ~ treat + post + treat_post',
                          data=df_educ,
                          weights=df_educ['PERWT'])
    results_educ = model_educ.fit(cov_type='HC1')
    print(f"{educ_name}: {results_educ.params['treat_post']:.4f} (SE: {results_educ.bse['treat_post']:.4f}), N={len(df_educ):,}")

# By marital status
print("\n--- By Marital Status ---")
for mar_val, mar_name in [(0, 'Not Married'), (1, 'Married')]:
    df_mar = df[df['married'] == mar_val]
    model_mar = smf.wls('fulltime ~ treat + post + treat_post',
                         data=df_mar,
                         weights=df_mar['PERWT'])
    results_mar = model_mar.fit(cov_type='HC1')
    print(f"{mar_name}: {results_mar.params['treat_post']:.4f} (SE: {results_mar.bse['treat_post']:.4f}), N={len(df_mar):,}")

# =============================================================================
# STEP 11: Summary Results
# =============================================================================
print("\n" + "="*80)
print("STEP 11: SUMMARY OF RESULTS")
print("="*80)

# Final sample size
n_total = len(df)
n_treat = df['treat'].sum()
n_control = n_total - n_treat

print(f"\n*** FINAL SAMPLE ***")
print(f"Total observations: {n_total:,}")
print(f"Treatment group (ages 26-30): {n_treat:,}")
print(f"Control group (ages 31-35): {n_control:,}")

print(f"\n*** PREFERRED ESTIMATE ***")
print(f"Model: Basic DiD (weighted, robust SE)")
print(f"DiD Coefficient: {did_estimate_1:.6f}")
print(f"Standard Error:  {did_se_1:.6f}")
print(f"95% CI: [{did_ci_1[0]:.6f}, {did_ci_1[1]:.6f}]")
print(f"p-value: {results1.pvalues['treat_post']:.6f}")

print(f"\n*** INTERPRETATION ***")
if did_estimate_1 > 0:
    print(f"DACA eligibility is associated with a {did_estimate_1*100:.2f} percentage point")
    print(f"increase in the probability of full-time employment.")
else:
    print(f"DACA eligibility is associated with a {abs(did_estimate_1)*100:.2f} percentage point")
    print(f"decrease in the probability of full-time employment.")

if results1.pvalues['treat_post'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
elif results1.pvalues['treat_post'] < 0.10:
    print("This effect is marginally significant at the 10% level.")
else:
    print("This effect is NOT statistically significant at conventional levels.")

# =============================================================================
# STEP 12: Save Results
# =============================================================================
print("\n" + "="*80)
print("STEP 12: SAVING RESULTS")
print("="*80)

# Create results dictionary
results_dict = {
    'preferred_estimate': did_estimate_1,
    'standard_error': did_se_1,
    'ci_lower': did_ci_1[0],
    'ci_upper': did_ci_1[1],
    'p_value': results1.pvalues['treat_post'],
    'sample_size': n_total,
    'n_treatment': n_treat,
    'n_control': n_control,
    'model_2_estimate': did_estimate_2,
    'model_2_se': did_se_2,
    'model_3_estimate': did_estimate_3,
    'model_3_se': did_se_3,
    'model_4_estimate': did_estimate_4,
    'model_4_se': did_se_4,
    'unweighted_estimate': results_unw.params['treat_post'],
    'unweighted_se': results_unw.bse['treat_post'],
    'clustered_estimate': results_clust.params['treat_post'],
    'clustered_se': results_clust.bse['treat_post'],
    'pre_trend_coef': results_pretrend.params['treat_trend'],
    'pre_trend_pval': results_pretrend.pvalues['treat_trend']
}

# Save to CSV
results_df = pd.DataFrame([results_dict])
results_df.to_csv('results_21.csv', index=False)
print("Results saved to results_21.csv")

# Save cell means
cell_means_flat = {
    'treat_pre': cell_means.loc[1, 0],
    'treat_post': cell_means.loc[1, 1],
    'control_pre': cell_means.loc[0, 0],
    'control_post': cell_means.loc[0, 1],
    'simple_did': simple_did
}
pd.DataFrame([cell_means_flat]).to_csv('cell_means_21.csv', index=False)
print("Cell means saved to cell_means_21.csv")

# Save event study results
event_df = pd.DataFrame(event_results).T
event_df.index.name = 'year'
event_df.to_csv('event_study_21.csv')
print("Event study results saved to event_study_21.csv")

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================
print("\n\n" + "="*80)
print("TABLE: SUMMARY OF REGRESSION RESULTS")
print("="*80)
print(f"{'Model':<35} {'Coefficient':>12} {'SE':>12} {'p-value':>12} {'N':>10}")
print("-"*80)
print(f"{'1. Basic DiD (weighted)':<35} {did_estimate_1:>12.4f} {did_se_1:>12.4f} {results1.pvalues['treat_post']:>12.4f} {n_total:>10,}")
print(f"{'2. With Controls':<35} {did_estimate_2:>12.4f} {did_se_2:>12.4f} {results2.pvalues['treat_post']:>12.4f} {n_total:>10,}")
print(f"{'3. Year FE':<35} {did_estimate_3:>12.4f} {did_se_3:>12.4f} {results3.pvalues['treat_post']:>12.4f} {n_total:>10,}")
print(f"{'4. Full Specification':<35} {did_estimate_4:>12.4f} {did_se_4:>12.4f} {results4.pvalues['treat_post']:>12.4f} {n_total:>10,}")
print(f"{'5. Unweighted':<35} {results_unw.params['treat_post']:>12.4f} {results_unw.bse['treat_post']:>12.4f} {results_unw.pvalues['treat_post']:>12.4f} {n_total:>10,}")
print(f"{'6. Clustered SE (by state)':<35} {results_clust.params['treat_post']:>12.4f} {results_clust.bse['treat_post']:>12.4f} {results_clust.pvalues['treat_post']:>12.4f} {n_total:>10,}")
print("-"*80)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
