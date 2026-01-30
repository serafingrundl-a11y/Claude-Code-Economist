"""
DACA Replication Analysis - Run 20
Difference-in-Differences Analysis of DACA Effect on Full-Time Employment

Research Question:
Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on full-time employment (35+ hours/week)?

Treatment Group: Ages 26-30 as of June 15, 2012
Control Group: Ages 31-35 as of June 15, 2012
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r'C:\Users\seraf\DACA Results Task 2\replication_20')

print("=" * 80)
print("DACA REPLICATION ANALYSIS - RUN 20")
print("=" * 80)

# =============================================================================
# STEP 1: Load and Initial Processing
# =============================================================================
print("\n[1] Loading data...")

# Read CSV in chunks due to large file size
chunks = []
chunksize = 500000

# Define dtypes for efficiency
dtype_dict = {
    'YEAR': 'int32',
    'STATEFIP': 'int32',
    'PERWT': 'float64',
    'AGE': 'int32',
    'BIRTHYR': 'int32',
    'BIRTHQTR': 'int32',
    'SEX': 'int32',
    'HISPAN': 'int32',
    'BPL': 'int32',
    'CITIZEN': 'int32',
    'YRIMMIG': 'int32',
    'EDUC': 'int32',
    'EMPSTAT': 'int32',
    'UHRSWORK': 'int32',
    'MARST': 'int32',
    'NCHILD': 'int32'
}

# Columns to keep
cols_to_keep = ['YEAR', 'STATEFIP', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX',
                'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT', 'UHRSWORK',
                'MARST', 'NCHILD']

for chunk in pd.read_csv('data/data.csv', chunksize=chunksize, usecols=cols_to_keep,
                          dtype=dtype_dict, low_memory=False):
    chunks.append(chunk)
    print(f"  Loaded {len(chunks) * chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"  Total rows loaded: {len(df):,}")

# =============================================================================
# STEP 2: Sample Selection
# =============================================================================
print("\n[2] Applying sample restrictions...")

# Initial count
print(f"  Initial sample: {len(df):,}")

# Step 2a: Hispanic-Mexican ethnicity (HISPAN == 1)
df = df[df['HISPAN'] == 1]
print(f"  After Hispanic-Mexican restriction: {len(df):,}")

# Step 2b: Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200]
print(f"  After Mexico birthplace restriction: {len(df):,}")

# Step 2c: Not a citizen (CITIZEN == 3)
# As per instructions, assume non-citizens without papers are undocumented
df = df[df['CITIZEN'] == 3]
print(f"  After non-citizen restriction: {len(df):,}")

# Step 2d: Calculate age as of June 15, 2012
# Those born Jan-Jun 1982 would be 30 by June 2012
# Those born Jul-Dec 1982 would turn 30 later in 2012
# For simplicity and following instructions, we use birth year to determine age cohort

# Treatment: Birth years 1982-1986 (ages 26-30 as of June 15, 2012)
# Control: Birth years 1977-1981 (ages 31-35 as of June 15, 2012)

# Filter to relevant birth years
df = df[(df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1986)]
print(f"  After birth year restriction (1977-1986): {len(df):,}")

# Step 2e: Arrived in US before 16th birthday
# YRIMMIG gives year of immigration
# Must have arrived before turning 16: arrival_year - birth_year < 16
# Handle YRIMMIG = 0 (native born or missing) - these should already be filtered by CITIZEN
df = df[df['YRIMMIG'] > 0]
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_arrival'] < 16]
print(f"  After arrival before age 16 restriction: {len(df):,}")

# Step 2f: Continuous presence since June 15, 2007
# Proxy: immigrated by 2007
df = df[df['YRIMMIG'] <= 2007]
print(f"  After continuous presence (arrived by 2007) restriction: {len(df):,}")

# Step 2g: Exclude 2012 (treatment timing uncertain)
df = df[df['YEAR'] != 2012]
print(f"  After excluding 2012: {len(df):,}")

# =============================================================================
# STEP 3: Create Analysis Variables
# =============================================================================
print("\n[3] Creating analysis variables...")

# Treatment indicator (birth year 1982-1986 = treatment group)
df['treat'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term (DiD estimator)
df['treat_post'] = df['treat'] * df['post']

# Full-time employment outcome (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Additional covariates
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)  # Married, spouse present or absent
df['has_children'] = (df['NCHILD'] > 0).astype(int)

# Education categories
df['less_than_hs'] = (df['EDUC'] < 6).astype(int)
df['hs_grad'] = (df['EDUC'] == 6).astype(int)
df['some_college'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)
df['college_plus'] = (df['EDUC'] >= 10).astype(int)

print(f"  Treatment group (ages 26-30): {df['treat'].sum():,}")
print(f"  Control group (ages 31-35): {(1-df['treat']).sum():,}")
print(f"  Pre-DACA observations: {(1-df['post']).sum():,}")
print(f"  Post-DACA observations: {df['post'].sum():,}")

# =============================================================================
# STEP 4: Summary Statistics
# =============================================================================
print("\n[4] Computing summary statistics...")

# Pre-period summary by treatment status
pre_treat = df[(df['post'] == 0) & (df['treat'] == 1)]
pre_control = df[(df['post'] == 0) & (df['treat'] == 0)]
post_treat = df[(df['post'] == 1) & (df['treat'] == 1)]
post_control = df[(df['post'] == 1) & (df['treat'] == 0)]

def weighted_mean(data, weights):
    return np.average(data, weights=weights)

def weighted_std(data, weights):
    average = np.average(data, weights=weights)
    variance = np.average((data - average) ** 2, weights=weights)
    return np.sqrt(variance)

print("\n  Summary Statistics (Weighted Means)")
print("  " + "-" * 70)
print(f"  {'Variable':<25} {'Pre-Control':>12} {'Pre-Treat':>12} {'Post-Control':>12} {'Post-Treat':>12}")
print("  " + "-" * 70)

variables = ['fulltime', 'female', 'married', 'has_children', 'less_than_hs',
             'hs_grad', 'some_college', 'college_plus', 'AGE']

for var in variables:
    pre_c = weighted_mean(pre_control[var], pre_control['PERWT'])
    pre_t = weighted_mean(pre_treat[var], pre_treat['PERWT'])
    post_c = weighted_mean(post_control[var], post_control['PERWT'])
    post_t = weighted_mean(post_treat[var], post_treat['PERWT'])
    print(f"  {var:<25} {pre_c:>12.3f} {pre_t:>12.3f} {post_c:>12.3f} {post_t:>12.3f}")

print("  " + "-" * 70)
print(f"  {'N (unweighted)':<25} {len(pre_control):>12,} {len(pre_treat):>12,} {len(post_control):>12,} {len(post_treat):>12,}")

# =============================================================================
# STEP 5: Main Difference-in-Differences Analysis
# =============================================================================
print("\n[5] Running Difference-in-Differences Analysis...")

# Model 1: Basic DiD (no covariates)
print("\n  Model 1: Basic DiD")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"    DiD Estimate: {model1.params['treat_post']:.4f}")
print(f"    Robust SE: {model1.bse['treat_post']:.4f}")
print(f"    t-statistic: {model1.tvalues['treat_post']:.4f}")
print(f"    p-value: {model1.pvalues['treat_post']:.4f}")
print(f"    95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")

# Model 2: DiD with demographic controls
print("\n  Model 2: DiD with Demographic Controls")
model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + has_children',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"    DiD Estimate: {model2.params['treat_post']:.4f}")
print(f"    Robust SE: {model2.bse['treat_post']:.4f}")
print(f"    t-statistic: {model2.tvalues['treat_post']:.4f}")
print(f"    p-value: {model2.pvalues['treat_post']:.4f}")
print(f"    95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")

# Model 3: DiD with demographic and education controls
print("\n  Model 3: DiD with Demographic and Education Controls")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + has_children + hs_grad + some_college + college_plus',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"    DiD Estimate: {model3.params['treat_post']:.4f}")
print(f"    Robust SE: {model3.bse['treat_post']:.4f}")
print(f"    t-statistic: {model3.tvalues['treat_post']:.4f}")
print(f"    p-value: {model3.pvalues['treat_post']:.4f}")
print(f"    95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")

# Model 4: DiD with year fixed effects
print("\n  Model 4: DiD with Year Fixed Effects")
df['year_factor'] = df['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ treat + C(year_factor) + treat_post + female + married + has_children + hs_grad + some_college + college_plus',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"    DiD Estimate: {model4.params['treat_post']:.4f}")
print(f"    Robust SE: {model4.bse['treat_post']:.4f}")
print(f"    t-statistic: {model4.tvalues['treat_post']:.4f}")
print(f"    p-value: {model4.pvalues['treat_post']:.4f}")
print(f"    95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

# Model 5: DiD with state fixed effects
print("\n  Model 5: DiD with State Fixed Effects (Preferred Specification)")
df['state_factor'] = df['STATEFIP'].astype(str)
model5 = smf.wls('fulltime ~ treat + C(year_factor) + C(state_factor) + treat_post + female + married + has_children + hs_grad + some_college + college_plus',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"    DiD Estimate: {model5.params['treat_post']:.4f}")
print(f"    Robust SE: {model5.bse['treat_post']:.4f}")
print(f"    t-statistic: {model5.tvalues['treat_post']:.4f}")
print(f"    p-value: {model5.pvalues['treat_post']:.4f}")
print(f"    95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")

# =============================================================================
# STEP 6: Year-by-Year Effects (Event Study)
# =============================================================================
print("\n[6] Event Study Analysis (Year-by-Year Effects)...")

# Create year interaction terms (reference year: 2011)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in years:
    df[f'treat_year_{year}'] = (df['treat'] * (df['YEAR'] == year)).astype(int)

# Event study regression
formula = 'fulltime ~ treat + C(year_factor) + female + married + has_children + hs_grad + some_college + college_plus + '
formula += ' + '.join([f'treat_year_{year}' for year in years])
event_study = smf.wls(formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\n  Event Study Coefficients (Reference: 2011)")
print("  " + "-" * 50)
print(f"  {'Year':<10} {'Coefficient':>15} {'Std. Error':>15} {'p-value':>12}")
print("  " + "-" * 50)
for year in years:
    coef = event_study.params[f'treat_year_{year}']
    se = event_study.bse[f'treat_year_{year}']
    pval = event_study.pvalues[f'treat_year_{year}']
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"  {year:<10} {coef:>15.4f} {se:>15.4f} {pval:>11.4f} {sig}")
print("  " + "-" * 50)
print("  Note: *** p<0.01, ** p<0.05, * p<0.1")

# =============================================================================
# STEP 7: Subgroup Analysis
# =============================================================================
print("\n[7] Subgroup Analysis...")

# By gender
print("\n  A. By Gender")
for gender, label in [(0, 'Male'), (1, 'Female')]:
    subset = df[df['female'] == gender]
    model_sub = smf.wls('fulltime ~ treat + post + treat_post',
                        data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"    {label}: DiD = {model_sub.params['treat_post']:.4f} (SE: {model_sub.bse['treat_post']:.4f}, p: {model_sub.pvalues['treat_post']:.4f})")

# By education
print("\n  B. By Education Level")
for edu_var, label in [('less_than_hs', 'Less than HS'), ('hs_grad', 'HS Graduate'),
                        ('some_college', 'Some College'), ('college_plus', 'College+')]:
    subset = df[df[edu_var] == 1]
    if len(subset) > 100:  # Only if sufficient observations
        model_sub = smf.wls('fulltime ~ treat + post + treat_post',
                            data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
        print(f"    {label}: DiD = {model_sub.params['treat_post']:.4f} (SE: {model_sub.bse['treat_post']:.4f}, p: {model_sub.pvalues['treat_post']:.4f})")

# =============================================================================
# STEP 8: Robustness Checks
# =============================================================================
print("\n[8] Robustness Checks...")

# Placebo test: Use only pre-period data
print("\n  A. Placebo Test (Pre-Period Only, 2009 as Fake Treatment)")
pre_only = df[df['YEAR'] <= 2011].copy()
pre_only['fake_post'] = (pre_only['YEAR'] >= 2009).astype(int)
pre_only['fake_treat_post'] = pre_only['treat'] * pre_only['fake_post']
placebo = smf.wls('fulltime ~ treat + fake_post + fake_treat_post',
                  data=pre_only, weights=pre_only['PERWT']).fit(cov_type='HC1')
print(f"    Placebo DiD: {placebo.params['fake_treat_post']:.4f} (SE: {placebo.bse['fake_treat_post']:.4f}, p: {placebo.pvalues['fake_treat_post']:.4f})")

# Alternative age bands (narrower)
print("\n  B. Narrower Age Bands (27-29 vs 32-34)")
narrow = df[(df['BIRTHYR'] >= 1978) & (df['BIRTHYR'] <= 1984)]
narrow['treat_narrow'] = ((narrow['BIRTHYR'] >= 1983) & (narrow['BIRTHYR'] <= 1985)).astype(int)
narrow_model = smf.wls('fulltime ~ treat_narrow + post + treat_narrow:post',
                        data=narrow, weights=narrow['PERWT']).fit(cov_type='HC1')
print(f"    Narrow Band DiD: {narrow_model.params['treat_narrow:post']:.4f} (SE: {narrow_model.bse['treat_narrow:post']:.4f})")

# =============================================================================
# STEP 9: Save Results for Report
# =============================================================================
print("\n[9] Saving results for report...")

# Save full-time employment rates by group-period
results_dict = {
    'Pre-Period Control (2006-2011)': {
        'fulltime_rate': weighted_mean(pre_control['fulltime'], pre_control['PERWT']),
        'n_unweighted': len(pre_control),
        'n_weighted': pre_control['PERWT'].sum()
    },
    'Pre-Period Treated (2006-2011)': {
        'fulltime_rate': weighted_mean(pre_treat['fulltime'], pre_treat['PERWT']),
        'n_unweighted': len(pre_treat),
        'n_weighted': pre_treat['PERWT'].sum()
    },
    'Post-Period Control (2013-2016)': {
        'fulltime_rate': weighted_mean(post_control['fulltime'], post_control['PERWT']),
        'n_unweighted': len(post_control),
        'n_weighted': post_control['PERWT'].sum()
    },
    'Post-Period Treated (2013-2016)': {
        'fulltime_rate': weighted_mean(post_treat['fulltime'], post_treat['PERWT']),
        'n_unweighted': len(post_treat),
        'n_weighted': post_treat['PERWT'].sum()
    }
}

# Calculate simple DiD
simple_did = (results_dict['Post-Period Treated (2013-2016)']['fulltime_rate'] -
              results_dict['Pre-Period Treated (2006-2011)']['fulltime_rate']) - \
             (results_dict['Post-Period Control (2013-2016)']['fulltime_rate'] -
              results_dict['Pre-Period Control (2006-2011)']['fulltime_rate'])

print(f"\n  Simple DiD Calculation:")
print(f"    Pre-period Control FT rate: {results_dict['Pre-Period Control (2006-2011)']['fulltime_rate']:.4f}")
print(f"    Pre-period Treated FT rate: {results_dict['Pre-Period Treated (2006-2011)']['fulltime_rate']:.4f}")
print(f"    Post-period Control FT rate: {results_dict['Post-Period Control (2013-2016)']['fulltime_rate']:.4f}")
print(f"    Post-period Treated FT rate: {results_dict['Post-Period Treated (2013-2016)']['fulltime_rate']:.4f}")
print(f"    Simple DiD: {simple_did:.4f}")

# Export event study results
event_study_results = pd.DataFrame({
    'Year': years,
    'Coefficient': [event_study.params[f'treat_year_{y}'] for y in years],
    'Std_Error': [event_study.bse[f'treat_year_{y}'] for y in years],
    'CI_Lower': [event_study.params[f'treat_year_{y}'] - 1.96*event_study.bse[f'treat_year_{y}'] for y in years],
    'CI_Upper': [event_study.params[f'treat_year_{y}'] + 1.96*event_study.bse[f'treat_year_{y}'] for y in years]
})
event_study_results.to_csv('event_study_results.csv', index=False)

# =============================================================================
# STEP 10: Create Summary Table for Report
# =============================================================================
print("\n[10] Creating summary table...")

# Create a summary dataframe for the main regression results
summary_table = pd.DataFrame({
    'Model': ['Basic DiD', 'With Demographics', 'With Education', 'Year FE', 'State & Year FE'],
    'DiD_Estimate': [model1.params['treat_post'], model2.params['treat_post'],
                     model3.params['treat_post'], model4.params['treat_post'],
                     model5.params['treat_post']],
    'Robust_SE': [model1.bse['treat_post'], model2.bse['treat_post'],
                  model3.bse['treat_post'], model4.bse['treat_post'],
                  model5.bse['treat_post']],
    'p_value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
                model3.pvalues['treat_post'], model4.pvalues['treat_post'],
                model5.pvalues['treat_post']],
    'N': [model1.nobs, model2.nobs, model3.nobs, model4.nobs, model5.nobs],
    'R_squared': [model1.rsquared, model2.rsquared, model3.rsquared,
                  model4.rsquared, model5.rsquared]
})

summary_table.to_csv('regression_results.csv', index=False)
print("\n  Regression results saved to regression_results.csv")

# Print preferred specification details
print("\n" + "=" * 80)
print("PREFERRED SPECIFICATION RESULTS (Model 5: State & Year FE)")
print("=" * 80)
print(f"\n  Effect Size: {model5.params['treat_post']:.4f}")
print(f"  Standard Error: {model5.bse['treat_post']:.4f}")
print(f"  95% Confidence Interval: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  p-value: {model5.pvalues['treat_post']:.4f}")
print(f"  Sample Size: {int(model5.nobs):,}")
print(f"  R-squared: {model5.rsquared:.4f}")

# =============================================================================
# STEP 11: Print Full Regression Tables
# =============================================================================
print("\n" + "=" * 80)
print("FULL REGRESSION OUTPUT - MODEL 5 (PREFERRED SPECIFICATION)")
print("=" * 80)
print(model5.summary())

print("\n" + "=" * 80)
print("FULL REGRESSION OUTPUT - MODEL 1 (BASIC DID)")
print("=" * 80)
print(model1.summary())

print("\n" + "=" * 80)
print("EVENT STUDY REGRESSION OUTPUT")
print("=" * 80)
print(event_study.summary())

# Save additional statistics for the report
final_stats = {
    'total_sample_size': len(df),
    'treatment_n': df['treat'].sum(),
    'control_n': (1 - df['treat']).sum(),
    'pre_period_n': (1 - df['post']).sum(),
    'post_period_n': df['post'].sum(),
    'preferred_estimate': model5.params['treat_post'],
    'preferred_se': model5.bse['treat_post'],
    'preferred_ci_lower': model5.conf_int().loc['treat_post', 0],
    'preferred_ci_upper': model5.conf_int().loc['treat_post', 1],
    'preferred_pvalue': model5.pvalues['treat_post']
}

# Save to file
with open('final_statistics.txt', 'w') as f:
    for key, value in final_stats.items():
        f.write(f"{key}: {value}\n")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
