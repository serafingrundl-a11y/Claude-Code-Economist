"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the United States.

Identification Strategy: Difference-in-Differences
- Treatment: Ages 26-30 as of June 15, 2012
- Control: Ages 31-35 as of June 15, 2012
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded - cannot separate before/after DACA)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DACA REPLICATION ANALYSIS")
print("=" * 70)

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n[1] Loading data...")

# Read data in chunks due to large size
chunks = []
chunksize = 1000000
cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'UHRSWORK',
               'EDUC', 'MARST', 'STATEFIP']

for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunksize)):
    chunks.append(chunk)
    if (i + 1) % 10 == 0:
        print(f"  Loaded {(i+1) * chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"  Total rows loaded: {len(df):,}")

# =============================================================================
# STEP 2: Identify DACA-Eligible Population
# =============================================================================
print("\n[2] Identifying DACA-eligible population...")

# Filter to Hispanic-Mexican born in Mexico
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
print(f"  Hispanic-Mexican born in Mexico: {len(df_mex):,}")

# Not a citizen (treating non-citizens as potentially undocumented per instructions)
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"  After filtering to non-citizens: {len(df_mex):,}")

# Calculate age at immigration
# If YRIMMIG is 0 or missing, we cannot determine arrival age
df_mex = df_mex[df_mex['YRIMMIG'] > 0].copy()
df_mex['age_at_immig'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
print(f"  After filtering to valid immigration year: {len(df_mex):,}")

# Arrived before age 16
df_mex = df_mex[df_mex['age_at_immig'] < 16].copy()
print(f"  After filtering to arrived before age 16: {len(df_mex):,}")

# Lived in US since at least June 2007 (use YRIMMIG <= 2007)
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007].copy()
print(f"  After filtering to in US since 2007: {len(df_mex):,}")

# =============================================================================
# STEP 3: Define Treatment and Control Groups
# =============================================================================
print("\n[3] Defining treatment and control groups...")

# Age on June 15, 2012:
# Treatment group: 26-30 years old -> born between June 1982 and June 1986
# Control group: 31-35 years old -> born between June 1977 and June 1981

# For simplicity, we approximate using birth year:
# - Treatment: born 1982-1986 (ages 26-30 in 2012)
# - Control: born 1977-1981 (ages 31-35 in 2012)

# Create treatment group indicator
df_mex['treat'] = ((df_mex['BIRTHYR'] >= 1982) & (df_mex['BIRTHYR'] <= 1986)).astype(int)
df_mex['control'] = ((df_mex['BIRTHYR'] >= 1977) & (df_mex['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control groups
df_analysis = df_mex[(df_mex['treat'] == 1) | (df_mex['control'] == 1)].copy()
print(f"  Observations in treatment or control: {len(df_analysis):,}")

# Create post-treatment indicator
# Pre-period: 2006-2011
# Post-period: 2013-2016
# Exclude 2012 (treatment year - cannot distinguish before/after)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
print(f"  After excluding 2012: {len(df_analysis):,}")

# Create interaction term
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

# =============================================================================
# STEP 4: Define Outcome Variable
# =============================================================================
print("\n[4] Defining outcome variable (full-time employment)...")

# Full-time employment = usually working 35+ hours per week
# UHRSWORK = 0 means N/A (not employed or not applicable)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Summary statistics
print(f"\n  Sample Summary:")
print(f"  Treatment group (ages 26-30 in 2012): {df_analysis['treat'].sum():,}")
print(f"  Control group (ages 31-35 in 2012): {df_analysis['control'].sum():,}")
print(f"  Pre-period observations: {(df_analysis['post'] == 0).sum():,}")
print(f"  Post-period observations: {(df_analysis['post'] == 1).sum():,}")

# =============================================================================
# STEP 5: Descriptive Statistics
# =============================================================================
print("\n[5] Computing descriptive statistics...")

# Save descriptive statistics
desc_stats = df_analysis.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum',
    'AGE': 'mean',
    'SEX': 'mean',  # 1=Male, 2=Female, so higher means more female
    'EDUC': 'mean'
}).round(4)
print("\n  Descriptive Statistics by Group and Period:")
print(desc_stats)

# Weighted means
def weighted_mean(df, var, weight='PERWT'):
    return np.average(df[var], weights=df[weight])

print("\n  Weighted Full-Time Employment Rates:")
for t in [0, 1]:
    for p in [0, 1]:
        subset = df_analysis[(df_analysis['treat'] == t) & (df_analysis['post'] == p)]
        if len(subset) > 0:
            wm = weighted_mean(subset, 'fulltime')
            group = "Treatment" if t == 1 else "Control"
            period = "Post" if p == 1 else "Pre"
            n = len(subset)
            print(f"    {group}, {period}: {wm:.4f} (n={n:,})")

# =============================================================================
# STEP 6: Difference-in-Differences Estimation
# =============================================================================
print("\n[6] Running Difference-in-Differences estimation...")

# Simple DiD (unweighted)
print("\n  --- Model 1: Simple DiD (OLS, no controls) ---")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit()
print(model1.summary().tables[1])

# Weighted DiD
print("\n  --- Model 2: Weighted DiD (WLS, no controls) ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(model2.summary().tables[1])

# DiD with controls
print("\n  --- Model 3: Weighted DiD with controls ---")
# Create control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # High school or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # Some college or more

model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(model3.summary().tables[1])

# DiD with year fixed effects
print("\n  --- Model 4: Weighted DiD with year FE ---")
df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ treat + C(year_factor) + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(f"  DiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"  Standard error: {model4.bse['treat_post']:.4f}")
print(f"  t-statistic: {model4.tvalues['treat_post']:.4f}")
print(f"  p-value: {model4.pvalues['treat_post']:.4f}")

# DiD with year FE and state FE
print("\n  --- Model 5: Weighted DiD with year FE and state FE ---")
df_analysis['state_factor'] = df_analysis['STATEFIP'].astype(str)
model5 = smf.wls('fulltime ~ treat + C(year_factor) + C(state_factor) + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(f"  DiD coefficient (treat_post): {model5.params['treat_post']:.4f}")
print(f"  Standard error: {model5.bse['treat_post']:.4f}")
print(f"  t-statistic: {model5.tvalues['treat_post']:.4f}")
print(f"  p-value: {model5.pvalues['treat_post']:.4f}")

# Full model with all controls and FE
print("\n  --- Model 6: Full specification (year FE, state FE, demographics) ---")
model6 = smf.wls('fulltime ~ treat + C(year_factor) + C(state_factor) + treat_post + female + married + educ_hs + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(f"  DiD coefficient (treat_post): {model6.params['treat_post']:.4f}")
print(f"  Standard error: {model6.bse['treat_post']:.4f}")
print(f"  t-statistic: {model6.tvalues['treat_post']:.4f}")
print(f"  p-value: {model6.pvalues['treat_post']:.4f}")

# Cluster-robust standard errors (by state)
print("\n  --- Model 7: Full specification with clustered SE (by state) ---")
model7 = smf.wls('fulltime ~ treat + C(year_factor) + C(state_factor) + treat_post + female + married + educ_hs + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"  DiD coefficient (treat_post): {model7.params['treat_post']:.4f}")
print(f"  Robust SE (clustered by state): {model7.bse['treat_post']:.4f}")
print(f"  t-statistic: {model7.tvalues['treat_post']:.4f}")
print(f"  p-value: {model7.pvalues['treat_post']:.4f}")
ci = model7.conf_int().loc['treat_post']
print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# =============================================================================
# STEP 7: Event Study / Pre-trends Analysis
# =============================================================================
print("\n[7] Event study analysis (pre-trends check)...")

# Create year dummies interacted with treatment
df_analysis['ref_year'] = 2011  # Reference year (last pre-treatment year)
event_vars = []
for year in df_analysis['YEAR'].unique():
    if year != 2011:  # Omit reference year
        var_name = f'treat_x_{year}'
        df_analysis[var_name] = (df_analysis['treat'] * (df_analysis['YEAR'] == year)).astype(int)
        event_vars.append(var_name)

event_formula = 'fulltime ~ treat + C(year_factor) + C(state_factor) + ' + ' + '.join(event_vars)
event_model = smf.wls(event_formula, data=df_analysis, weights=df_analysis['PERWT']).fit()

print("\n  Event Study Coefficients (relative to 2011):")
for var in sorted(event_vars):
    year = var.split('_')[-1]
    coef = event_model.params[var]
    se = event_model.bse[var]
    print(f"    {year}: {coef:.4f} (SE: {se:.4f})")

# =============================================================================
# STEP 8: Robustness Checks
# =============================================================================
print("\n[8] Robustness checks...")

# Robustness 1: Different age bandwidths
print("\n  --- Robustness 1: Narrower age bandwidth (ages 28-30 vs 31-33) ---")
df_narrow = df_mex[((df_mex['BIRTHYR'] >= 1979) & (df_mex['BIRTHYR'] <= 1984))].copy()
df_narrow = df_narrow[df_narrow['YEAR'] != 2012].copy()
df_narrow['treat'] = ((df_narrow['BIRTHYR'] >= 1982) & (df_narrow['BIRTHYR'] <= 1984)).astype(int)
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['treat_post'] = df_narrow['treat'] * df_narrow['post']
df_narrow['fulltime'] = (df_narrow['UHRSWORK'] >= 35).astype(int)

if len(df_narrow) > 100:
    model_narrow = smf.wls('fulltime ~ treat + post + treat_post',
                           data=df_narrow,
                           weights=df_narrow['PERWT']).fit()
    print(f"    DiD coefficient: {model_narrow.params['treat_post']:.4f}")
    print(f"    SE: {model_narrow.bse['treat_post']:.4f}")
    print(f"    N: {len(df_narrow):,}")

# Robustness 2: By sex
print("\n  --- Robustness 2: Separate estimates by sex ---")
for sex, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_analysis[df_analysis['SEX'] == sex].copy()
    model_sex = smf.wls('fulltime ~ treat + post + treat_post',
                        data=df_sex,
                        weights=df_sex['PERWT']).fit()
    print(f"    {sex_name}: DiD = {model_sex.params['treat_post']:.4f} (SE: {model_sex.bse['treat_post']:.4f}), N = {len(df_sex):,}")

# Robustness 3: Linear probability model vs logit comparison
print("\n  --- Robustness 3: Logit model (marginal effects) ---")
try:
    # For logit, we need to scale weights
    df_analysis['scaled_weight'] = df_analysis['PERWT'] / df_analysis['PERWT'].mean()
    logit_model = smf.logit('fulltime ~ treat + post + treat_post', data=df_analysis).fit(disp=0)
    # Marginal effect at means
    mfx = logit_model.get_margeff(at='mean')
    treat_post_idx = list(mfx.summary_frame().index).index('treat_post')
    print(f"    Marginal effect of treat_post: {mfx.margeff[treat_post_idx]:.4f}")
    print(f"    SE: {mfx.margeff_se[treat_post_idx]:.4f}")
except Exception as e:
    print(f"    Logit model failed: {e}")

# =============================================================================
# STEP 9: Save Results
# =============================================================================
print("\n[9] Saving results...")

# Preferred estimate (Model 7: Full specification with clustered SE)
preferred_coef = model7.params['treat_post']
preferred_se = model7.bse['treat_post']
preferred_ci = model7.conf_int().loc['treat_post']
sample_size = len(df_analysis)

print(f"\n  PREFERRED ESTIMATE (Model 7):")
print(f"  Effect size: {preferred_coef:.4f}")
print(f"  Standard error (clustered): {preferred_se:.4f}")
print(f"  95% CI: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"  Sample size: {sample_size:,}")

# Save key results to file
results = {
    'preferred_effect': preferred_coef,
    'preferred_se': preferred_se,
    'preferred_ci_lower': preferred_ci[0],
    'preferred_ci_upper': preferred_ci[1],
    'sample_size': sample_size,
    'model1_coef': model1.params['treat_post'],
    'model1_se': model1.bse['treat_post'],
    'model2_coef': model2.params['treat_post'],
    'model2_se': model2.bse['treat_post'],
    'model3_coef': model3.params['treat_post'],
    'model3_se': model3.bse['treat_post'],
}

# Save results DataFrame
results_df = pd.DataFrame([results])
results_df.to_csv('results.csv', index=False)
print("  Results saved to results.csv")

# =============================================================================
# STEP 10: Create Tables for LaTeX
# =============================================================================
print("\n[10] Creating LaTeX tables...")

# Table 1: Descriptive Statistics
desc_by_group = df_analysis.groupby('treat').apply(
    lambda x: pd.Series({
        'N': len(x),
        'Weighted N': x['PERWT'].sum(),
        'Full-time (%)': weighted_mean(x, 'fulltime') * 100,
        'Female (%)': (x['SEX'] == 2).mean() * 100,
        'Married (%)': (x['MARST'] == 1).mean() * 100,
        'HS or more (%)': (x['EDUC'] >= 6).mean() * 100,
        'Mean Age': x['AGE'].mean(),
    })
)
desc_by_group.to_csv('desc_stats_by_group.csv')

# Table 2: Main Results
main_results = pd.DataFrame({
    'Model': ['Simple DiD', 'Weighted DiD', 'With Controls', 'Year FE', 'Year + State FE', 'Full Model', 'Clustered SE'],
    'Coefficient': [model1.params['treat_post'], model2.params['treat_post'], model3.params['treat_post'],
                    model4.params['treat_post'], model5.params['treat_post'], model6.params['treat_post'],
                    model7.params['treat_post']],
    'SE': [model1.bse['treat_post'], model2.bse['treat_post'], model3.bse['treat_post'],
           model4.bse['treat_post'], model5.bse['treat_post'], model6.bse['treat_post'],
           model7.bse['treat_post']],
    'N': [len(df_analysis)] * 7
})
main_results.to_csv('main_results.csv', index=False)

# Table 3: Event study coefficients
event_results = []
for var in sorted(event_vars):
    year = int(var.split('_')[-1])
    event_results.append({
        'Year': year,
        'Coefficient': event_model.params[var],
        'SE': event_model.bse[var]
    })
event_df = pd.DataFrame(event_results).sort_values('Year')
event_df.to_csv('event_study.csv', index=False)

# Year-by-year full-time rates
yearly_rates = df_analysis.groupby(['YEAR', 'treat']).apply(
    lambda x: pd.Series({
        'fulltime_rate': weighted_mean(x, 'fulltime'),
        'n': len(x)
    })
).reset_index()
yearly_rates.to_csv('yearly_rates.csv', index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nKey files generated:")
print("  - results.csv (main results)")
print("  - desc_stats_by_group.csv (descriptive statistics)")
print("  - main_results.csv (regression table)")
print("  - event_study.csv (event study coefficients)")
print("  - yearly_rates.csv (year-by-year employment rates)")
