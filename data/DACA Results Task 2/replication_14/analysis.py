"""
DACA Impact on Full-Time Employment: Difference-in-Differences Analysis
Replication Study - Session 14

Research Question:
Among ethnically Hispanic-Mexican Mexican-born people in the US, what was the
causal impact of DACA eligibility on full-time employment (35+ hours/week)?

Treatment: Ages 26-30 on June 15, 2012
Control: Ages 31-35 on June 15, 2012
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
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("="*80)
print("DACA IMPACT ON FULL-TIME EMPLOYMENT: DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

# =============================================================================
# Step 1: Load Data (Memory Efficient)
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA (Memory Efficient)")
print("="*80)

# Define columns we need
needed_cols = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR', 'MARST',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'YRSUSA1', 'EDUC',
               'UHRSWORK', 'NCHILD', 'STATEFIP']

# Define dtypes to save memory
dtypes = {
    'YEAR': 'int16',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'MARST': 'int8',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'YRSUSA1': 'int8',
    'EDUC': 'int8',
    'UHRSWORK': 'int8',
    'NCHILD': 'int8',
    'STATEFIP': 'int8'
}

print("Loading ACS data in chunks and filtering to eligible sample...")

# Process in chunks and keep only eligible observations
chunks = []
chunk_size = 500000

for chunk in pd.read_csv('data/data.csv', usecols=needed_cols, dtype=dtypes, chunksize=chunk_size):
    # Apply eligibility filters immediately to reduce memory
    # Hispanic-Mexican (HISPAN == 1) and Born in Mexico (BPL == 200)
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    # Not a citizen (CITIZEN == 3)
    chunk = chunk[chunk['CITIZEN'] == 3]
    # Valid immigration year and arrived before age 16
    chunk = chunk[chunk['YRIMMIG'] > 0]
    chunk['age_at_arrival'] = chunk['YRIMMIG'] - chunk['BIRTHYR']
    chunk = chunk[chunk['age_at_arrival'] < 16]
    # Continuous presence (arrived by 2006)
    chunk = chunk[chunk['YRIMMIG'] <= 2006]

    if len(chunk) > 0:
        chunks.append(chunk)

    gc.collect()

df_eligible = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"DACA-eligible sample loaded: {len(df_eligible):,}")
print(f"Years covered: {df_eligible['YEAR'].min()} - {df_eligible['YEAR'].max()}")

# =============================================================================
# Step 2: Define Treatment and Control Age Groups
# =============================================================================
print("\n" + "="*80)
print("STEP 2: DEFINING TREATMENT AND CONTROL GROUPS")
print("="*80)

# Calculate age on June 15, 2012
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born Q1-Q2 (Jan-Jun), had birthday by June 15
# If born Q3-Q4 (Jul-Dec), hadn't had birthday yet

df_eligible['age_june2012'] = np.where(
    df_eligible['BIRTHQTR'] <= 2,
    2012 - df_eligible['BIRTHYR'],
    2012 - df_eligible['BIRTHYR'] - 1
)

# Define treatment and control based on exact ages
# Treatment: ages 26-30 on June 15, 2012
# Control: ages 31-35 on June 15, 2012
df_eligible['treatment_group'] = ((df_eligible['age_june2012'] >= 26) &
                                   (df_eligible['age_june2012'] <= 30)).astype('int8')
df_eligible['control_group'] = ((df_eligible['age_june2012'] >= 31) &
                                 (df_eligible['age_june2012'] <= 35)).astype('int8')

# Keep only treatment or control group observations
df_sample = df_eligible[(df_eligible['treatment_group'] == 1) |
                         (df_eligible['control_group'] == 1)].copy()

del df_eligible
gc.collect()

print(f"\nSample after defining treatment/control groups: {len(df_sample):,}")
print(f"  Treatment group (ages 26-30): {df_sample['treatment_group'].sum():,}")
print(f"  Control group (ages 31-35): {(df_sample['treatment_group']==0).sum():,}")

# =============================================================================
# Step 3: Define Time Periods and Exclude 2012
# =============================================================================
print("\n" + "="*80)
print("STEP 3: DEFINING TIME PERIODS")
print("="*80)

# Pre-period: 2006-2011 (before DACA)
# Post-period: 2013-2016 (after DACA, as specified in instructions)
# Exclude 2012 because DACA implemented mid-year

df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype('int8')

print(f"Sample after excluding 2012: {len(df_sample):,}")
print(f"  Pre-period (2006-2011): {(df_sample['post'] == 0).sum():,}")
print(f"  Post-period (2013-2016): {(df_sample['post'] == 1).sum():,}")

# =============================================================================
# Step 4: Create Outcome and Covariate Variables
# =============================================================================
print("\n" + "="*80)
print("STEP 4: CREATING OUTCOME AND COVARIATES")
print("="*80)

# Full-time employment: usually working 35+ hours per week
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype('int8')

# Sex: 1=Male, 2=Female
df_sample['female'] = (df_sample['SEX'] == 2).astype('int8')

# Marital status: 1=Married spouse present
df_sample['married'] = (df_sample['MARST'] == 1).astype('int8')

# Education: HS or higher (EDUC >= 7)
df_sample['educ_hs'] = (df_sample['EDUC'] >= 7).astype('int8')

# Has children
df_sample['has_children'] = (df_sample['NCHILD'] > 0).astype('int8')

# Interaction term
df_sample['treat_x_post'] = df_sample['treatment_group'] * df_sample['post']

# Weight
df_sample['weight'] = df_sample['PERWT']

print(f"Full-time employment rate (overall): {df_sample['fulltime'].mean():.4f}")
print(f"  Treatment group: {df_sample[df_sample['treatment_group']==1]['fulltime'].mean():.4f}")
print(f"  Control group: {df_sample[df_sample['treatment_group']==0]['fulltime'].mean():.4f}")
print(f"\nCovariates:")
print(f"  Female: {df_sample['female'].mean():.4f}")
print(f"  Married: {df_sample['married'].mean():.4f}")
print(f"  HS or higher: {df_sample['educ_hs'].mean():.4f}")
print(f"  Has children: {df_sample['has_children'].mean():.4f}")

# =============================================================================
# Step 5: Descriptive Statistics
# =============================================================================
print("\n" + "="*80)
print("STEP 5: DESCRIPTIVE STATISTICS")
print("="*80)

# Sample distribution by year
print("\nSample by Year:")
year_counts = df_sample.groupby('YEAR').size()
print(year_counts)

# Full-time rates by group and period
print("\nFull-time Employment Rates by Group and Period:")
grouped = df_sample.groupby(['treatment_group', 'post']).apply(
    lambda x: pd.Series({
        'n': len(x),
        'fulltime_rate': np.average(x['fulltime'], weights=x['weight']),
        'weighted_n': x['weight'].sum()
    })
)
print(grouped)

# =============================================================================
# Step 6: Simple Difference-in-Differences (2x2 Table)
# =============================================================================
print("\n" + "="*80)
print("STEP 6: SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION")
print("="*80)

# Calculate weighted means for each cell
treat_pre = df_sample[(df_sample['treatment_group']==1) & (df_sample['post']==0)]
treat_post = df_sample[(df_sample['treatment_group']==1) & (df_sample['post']==1)]
control_pre = df_sample[(df_sample['treatment_group']==0) & (df_sample['post']==0)]
control_post = df_sample[(df_sample['treatment_group']==0) & (df_sample['post']==1)]

treat_pre_mean = np.average(treat_pre['fulltime'], weights=treat_pre['weight'])
treat_post_mean = np.average(treat_post['fulltime'], weights=treat_post['weight'])
control_pre_mean = np.average(control_pre['fulltime'], weights=control_pre['weight'])
control_post_mean = np.average(control_post['fulltime'], weights=control_post['weight'])

print("\nDifference-in-Differences Table:")
print("-" * 60)
print(f"{'':20} {'Pre-DACA':>15} {'Post-DACA':>15} {'Difference':>15}")
print("-" * 60)
print(f"{'Treatment (26-30)':20} {treat_pre_mean:>15.4f} {treat_post_mean:>15.4f} {treat_post_mean-treat_pre_mean:>15.4f}")
print(f"{'Control (31-35)':20} {control_pre_mean:>15.4f} {control_post_mean:>15.4f} {control_post_mean-control_pre_mean:>15.4f}")
print("-" * 60)

treat_diff = treat_post_mean - treat_pre_mean
control_diff = control_post_mean - control_pre_mean
did_estimate = treat_diff - control_diff

print(f"{'DiD Estimate':20} {'':<15} {'':<15} {did_estimate:>15.4f}")
print("-" * 60)

# =============================================================================
# Step 7: Regression-Based DiD Analysis
# =============================================================================
print("\n" + "="*80)
print("STEP 7: REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES")
print("="*80)

# Model 1: Basic DiD (no covariates)
print("\nModel 1: Basic DiD (no covariates)")
print("-" * 60)

model1 = smf.wls('fulltime ~ treatment_group + post + treat_x_post',
                  data=df_sample, weights=df_sample['weight']).fit(cov_type='HC1')
print(model1.summary().tables[1])
print(f"\nDiD Coefficient (treat_x_post): {model1.params['treat_x_post']:.6f}")
print(f"Standard Error: {model1.bse['treat_x_post']:.6f}")
print(f"95% CI: [{model1.conf_int().loc['treat_x_post', 0]:.6f}, {model1.conf_int().loc['treat_x_post', 1]:.6f}]")
print(f"P-value: {model1.pvalues['treat_x_post']:.6f}")

# Model 2: DiD with demographic covariates
print("\nModel 2: DiD with demographic covariates")
print("-" * 60)

model2 = smf.wls('fulltime ~ treatment_group + post + treat_x_post + female + married + educ_hs + has_children',
                  data=df_sample, weights=df_sample['weight']).fit(cov_type='HC1')
print(model2.summary().tables[1])
print(f"\nDiD Coefficient (treat_x_post): {model2.params['treat_x_post']:.6f}")
print(f"Standard Error: {model2.bse['treat_x_post']:.6f}")
print(f"95% CI: [{model2.conf_int().loc['treat_x_post', 0]:.6f}, {model2.conf_int().loc['treat_x_post', 1]:.6f}]")

# Model 3: DiD with year fixed effects
print("\nModel 3: DiD with year fixed effects")
print("-" * 60)

# Create year dummies
year_dummies = pd.get_dummies(df_sample['YEAR'], prefix='year', drop_first=True)
for col in year_dummies.columns:
    df_sample[col] = year_dummies[col].astype('int8')

year_cols = [col for col in df_sample.columns if col.startswith('year_')]
formula3 = 'fulltime ~ treatment_group + treat_x_post + female + married + educ_hs + has_children + ' + ' + '.join(year_cols)
model3 = smf.wls(formula3, data=df_sample, weights=df_sample['weight']).fit(cov_type='HC1')

print(f"DiD Coefficient (treat_x_post): {model3.params['treat_x_post']:.6f}")
print(f"Standard Error: {model3.bse['treat_x_post']:.6f}")
print(f"95% CI: [{model3.conf_int().loc['treat_x_post', 0]:.6f}, {model3.conf_int().loc['treat_x_post', 1]:.6f}]")
print(f"P-value: {model3.pvalues['treat_x_post']:.6f}")

# Model 4: DiD with state fixed effects
print("\nModel 4: DiD with state and year fixed effects")
print("-" * 60)

# Create state dummies
state_dummies = pd.get_dummies(df_sample['STATEFIP'], prefix='state', drop_first=True)
for col in state_dummies.columns:
    df_sample[col] = state_dummies[col].astype('int8')

state_cols = [col for col in df_sample.columns if col.startswith('state_')]
formula4 = 'fulltime ~ treatment_group + treat_x_post + female + married + educ_hs + has_children + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)

try:
    model4 = smf.wls(formula4, data=df_sample, weights=df_sample['weight']).fit(cov_type='HC1')
    print(f"DiD Coefficient (treat_x_post): {model4.params['treat_x_post']:.6f}")
    print(f"Standard Error: {model4.bse['treat_x_post']:.6f}")
    print(f"95% CI: [{model4.conf_int().loc['treat_x_post', 0]:.6f}, {model4.conf_int().loc['treat_x_post', 1]:.6f}]")
    print(f"P-value: {model4.pvalues['treat_x_post']:.6f}")
except Exception as e:
    print(f"Could not estimate model with state FE: {e}")
    model4 = model3

# =============================================================================
# Step 8: Robustness Checks
# =============================================================================
print("\n" + "="*80)
print("STEP 8: ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Men only
print("\nRobustness 1: Men Only")
print("-" * 60)
df_men = df_sample[df_sample['female'] == 0]
model_men = smf.wls('fulltime ~ treatment_group + post + treat_x_post',
                     data=df_men, weights=df_men['weight']).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_men.params['treat_x_post']:.6f}")
print(f"SE: {model_men.bse['treat_x_post']:.6f}")
print(f"P-value: {model_men.pvalues['treat_x_post']:.6f}")
print(f"N: {len(df_men):,}")

# Robustness 2: Women only
print("\nRobustness 2: Women Only")
print("-" * 60)
df_women = df_sample[df_sample['female'] == 1]
model_women = smf.wls('fulltime ~ treatment_group + post + treat_x_post',
                       data=df_women, weights=df_women['weight']).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_women.params['treat_x_post']:.6f}")
print(f"SE: {model_women.bse['treat_x_post']:.6f}")
print(f"P-value: {model_women.pvalues['treat_x_post']:.6f}")
print(f"N: {len(df_women):,}")

# Robustness 3: Placebo test - Pre-period only (use 2009 as fake policy year)
print("\nRobustness 3: Placebo Test (2009 as fake policy year)")
print("-" * 60)
df_placebo = df_sample[df_sample['YEAR'] <= 2011].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype('int8')
df_placebo['treat_x_post_placebo'] = df_placebo['treatment_group'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treatment_group + post_placebo + treat_x_post_placebo',
                         data=df_placebo, weights=df_placebo['weight']).fit(cov_type='HC1')
print(f"Placebo DiD Coefficient: {model_placebo.params['treat_x_post_placebo']:.6f}")
print(f"SE: {model_placebo.bse['treat_x_post_placebo']:.6f}")
print(f"P-value: {model_placebo.pvalues['treat_x_post_placebo']:.6f}")

# =============================================================================
# Step 9: Event Study / Pre-trends
# =============================================================================
print("\n" + "="*80)
print("STEP 9: EVENT STUDY ANALYSIS")
print("="*80)

# Create year-specific treatment effects (relative to 2011)
base_year = 2011

for year in sorted(df_sample['YEAR'].unique()):
    if year != base_year:
        df_sample[f'treat_x_{year}'] = (df_sample['treatment_group'] *
                                         (df_sample['YEAR'] == year)).astype('int8')

# Run event study regression
event_vars = [f'treat_x_{y}' for y in sorted(df_sample['YEAR'].unique()) if y != base_year]
formula_event = 'fulltime ~ treatment_group + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(event_vars)
model_event = smf.wls(formula_event, data=df_sample, weights=df_sample['weight']).fit(cov_type='HC1')

print("Event Study Coefficients (relative to 2011):")
print("-" * 60)
event_study_results = []
for var in event_vars:
    year = int(var.split('_')[-1])
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci = model_event.conf_int().loc[var]
    pval = model_event.pvalues[var]
    print(f"  {year}: Coef = {coef:.4f}, SE = {se:.4f}, 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}], p = {pval:.4f}")
    event_study_results.append({'year': year, 'coef': coef, 'se': se, 'ci_low': ci[0], 'ci_high': ci[1], 'pval': pval})

# Add base year
event_study_results.append({'year': 2011, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0, 'pval': 1})
event_study_df = pd.DataFrame(event_study_results).sort_values('year')

# =============================================================================
# Step 10: Final Summary
# =============================================================================
print("\n" + "="*80)
print("STEP 10: FINAL RESULTS SUMMARY")
print("="*80)

print("\n*** PREFERRED ESTIMATE ***")
print(f"Model: Difference-in-Differences with Year Fixed Effects and Covariates")
print(f"DiD Coefficient: {model3.params['treat_x_post']:.6f}")
print(f"Standard Error: {model3.bse['treat_x_post']:.6f}")
print(f"95% Confidence Interval: [{model3.conf_int().loc['treat_x_post', 0]:.6f}, {model3.conf_int().loc['treat_x_post', 1]:.6f}]")
print(f"P-value: {model3.pvalues['treat_x_post']:.6f}")
print(f"Sample Size: {len(df_sample):,}")

# Treatment effect in percentage points
effect_pct = model3.params['treat_x_post'] * 100
se_pct = model3.bse['treat_x_post'] * 100
print(f"\nEffect Size: {effect_pct:.2f} percentage points")
print(f"Standard Error: {se_pct:.2f} percentage points")

# Interpretation
if model3.pvalues['treat_x_post'] < 0.05:
    sig_text = "statistically significant at the 5% level"
else:
    sig_text = "not statistically significant at the 5% level"

print(f"\nInterpretation: The estimated effect of DACA eligibility on full-time employment is {sig_text}.")

# =============================================================================
# Step 11: Export Results for Report
# =============================================================================
print("\n" + "="*80)
print("STEP 11: EXPORTING RESULTS")
print("="*80)

# Create summary table
summary_df = pd.DataFrame({
    'Specification': ['Basic DiD', 'With Covariates', 'With Year FE', 'With State+Year FE', 'Men Only', 'Women Only', 'Placebo Test'],
    'Coefficient': [
        model1.params['treat_x_post'],
        model2.params['treat_x_post'],
        model3.params['treat_x_post'],
        model4.params['treat_x_post'],
        model_men.params['treat_x_post'],
        model_women.params['treat_x_post'],
        model_placebo.params['treat_x_post_placebo']
    ],
    'Std_Error': [
        model1.bse['treat_x_post'],
        model2.bse['treat_x_post'],
        model3.bse['treat_x_post'],
        model4.bse['treat_x_post'],
        model_men.bse['treat_x_post'],
        model_women.bse['treat_x_post'],
        model_placebo.bse['treat_x_post_placebo']
    ],
    'P_value': [
        model1.pvalues['treat_x_post'],
        model2.pvalues['treat_x_post'],
        model3.pvalues['treat_x_post'],
        model4.pvalues['treat_x_post'],
        model_men.pvalues['treat_x_post'],
        model_women.pvalues['treat_x_post'],
        model_placebo.pvalues['treat_x_post_placebo']
    ],
    'CI_Low': [
        model1.conf_int().loc['treat_x_post', 0],
        model2.conf_int().loc['treat_x_post', 0],
        model3.conf_int().loc['treat_x_post', 0],
        model4.conf_int().loc['treat_x_post', 0],
        model_men.conf_int().loc['treat_x_post', 0],
        model_women.conf_int().loc['treat_x_post', 0],
        model_placebo.conf_int().loc['treat_x_post_placebo', 0]
    ],
    'CI_High': [
        model1.conf_int().loc['treat_x_post', 1],
        model2.conf_int().loc['treat_x_post', 1],
        model3.conf_int().loc['treat_x_post', 1],
        model4.conf_int().loc['treat_x_post', 1],
        model_men.conf_int().loc['treat_x_post', 1],
        model_women.conf_int().loc['treat_x_post', 1],
        model_placebo.conf_int().loc['treat_x_post_placebo', 1]
    ]
})

print("\nSummary of All Specifications:")
print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv('regression_results.csv', index=False)
print("\nResults saved to regression_results.csv")

# Create descriptive statistics table
desc_stats = df_sample.groupby('treatment_group').agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'has_children': 'mean',
    'AGE': 'mean'
})
desc_stats.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved to descriptive_stats.csv")

# Year-by-group means for visualization
year_group_means = df_sample.groupby(['YEAR', 'treatment_group']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['weight']),
        'n': len(x),
        'weighted_n': x['weight'].sum()
    })
).reset_index()
year_group_means.to_csv('year_group_means.csv', index=False)
print("Year-by-group means saved to year_group_means.csv")

# Event study results
event_study_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save key results for LaTeX report
with open('key_results.txt', 'w') as f:
    f.write(f"SAMPLE_SIZE={len(df_sample)}\n")
    f.write(f"N_TREATMENT={df_sample['treatment_group'].sum()}\n")
    f.write(f"N_CONTROL={(df_sample['treatment_group']==0).sum()}\n")
    f.write(f"TREAT_PRE_MEAN={treat_pre_mean:.4f}\n")
    f.write(f"TREAT_POST_MEAN={treat_post_mean:.4f}\n")
    f.write(f"CONTROL_PRE_MEAN={control_pre_mean:.4f}\n")
    f.write(f"CONTROL_POST_MEAN={control_post_mean:.4f}\n")
    f.write(f"DID_SIMPLE={did_estimate:.6f}\n")
    f.write(f"DID_COEF_PREFERRED={model3.params['treat_x_post']:.6f}\n")
    f.write(f"DID_SE_PREFERRED={model3.bse['treat_x_post']:.6f}\n")
    f.write(f"DID_PVAL_PREFERRED={model3.pvalues['treat_x_post']:.6f}\n")
    f.write(f"DID_CI_LOW={model3.conf_int().loc['treat_x_post', 0]:.6f}\n")
    f.write(f"DID_CI_HIGH={model3.conf_int().loc['treat_x_post', 1]:.6f}\n")
    f.write(f"EFFECT_PCT={effect_pct:.2f}\n")
    f.write(f"SE_PCT={se_pct:.2f}\n")

print("Key results saved to key_results.txt")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
