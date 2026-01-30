"""
DACA Employment Effects Replication Study
Analysis Script

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for DACA
(treatment) on full-time employment (outcome)?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DACA Employment Effects Analysis")
print("="*80)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/data.csv')
print(f"Total observations: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# SAMPLE SELECTION
# ============================================================================
print("\n[2] Sample Selection...")

# Step 1: Hispanic-Mexican ethnicity
# HISPAN = 1 indicates Mexican Hispanic origin
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican filter: {len(df_sample):,}")

# Step 2: Born in Mexico
# BPL = 200 indicates Mexico birthplace
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After Mexico-born filter: {len(df_sample):,}")

# Step 3: Non-citizens (potential undocumented)
# CITIZEN = 3 means "Not a citizen"
# We focus on non-citizens since DACA targets undocumented immigrants
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After non-citizen filter: {len(df_sample):,}")

# Step 4: Working age (16-64)
df_sample = df_sample[(df_sample['AGE'] >= 16) & (df_sample['AGE'] <= 64)].copy()
print(f"After working age (16-64) filter: {len(df_sample):,}")

# Step 5: Exclude 2012 (DACA implemented mid-year June 15)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# Step 6: Valid immigration year
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
print(f"After valid immigration year filter: {len(df_sample):,}")

# ============================================================================
# VARIABLE CONSTRUCTION
# ============================================================================
print("\n[3] Constructing Variables...")

# Age at immigration
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Full-time employment outcome (35+ hours per week)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Employed indicator
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# Post-DACA indicator (2013-2016)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# DACA Eligibility Criteria:
# 1. Arrived in US before 16th birthday
# 2. Born on or after June 16, 1981 (not yet 31 by June 15, 2012)
#    Using BIRTHYR >= 1982 as conservative cutoff
# 3. Lived continuously in US since June 15, 2007 (YRIMMIG <= 2007)
# 4. Non-citizen status (already filtered)

# Create DACA eligibility indicator
df_sample['arrived_before_16'] = (df_sample['age_at_immig'] < 16).astype(int)
df_sample['young_enough'] = (df_sample['BIRTHYR'] >= 1982).astype(int)
df_sample['long_residence'] = (df_sample['YRIMMIG'] <= 2007).astype(int)

df_sample['daca_eligible'] = (
    (df_sample['arrived_before_16'] == 1) &
    (df_sample['young_enough'] == 1) &
    (df_sample['long_residence'] == 1)
).astype(int)

print(f"DACA eligible: {df_sample['daca_eligible'].sum():,}")
print(f"Not DACA eligible: {(df_sample['daca_eligible']==0).sum():,}")

# DiD interaction term
df_sample['daca_post'] = df_sample['daca_eligible'] * df_sample['post']

# Additional control variables
df_sample['age_sq'] = df_sample['AGE'] ** 2
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Education categories
df_sample['educ_hs'] = (df_sample['EDUC'] >= 6).astype(int)  # HS or more
df_sample['educ_college'] = (df_sample['EDUC'] >= 10).astype(int)  # Some college+

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[4] Descriptive Statistics...")
print("\n--- Sample Composition ---")
print(f"Total observations in analysis sample: {len(df_sample):,}")

# By year
year_counts = df_sample.groupby('YEAR').size()
print("\nObservations by year:")
print(year_counts)

# By treatment status and period
print("\n--- Treatment vs Control Groups ---")
for treat in [0, 1]:
    for post in [0, 1]:
        subset = df_sample[(df_sample['daca_eligible']==treat) & (df_sample['post']==post)]
        label = f"{'DACA Eligible' if treat else 'Control'}, {'Post' if post else 'Pre'}"
        print(f"{label}: N={len(subset):,}, Full-time rate={subset['fulltime'].mean():.4f}")

# Summary statistics
print("\n--- Summary Statistics ---")
summary_vars = ['fulltime', 'employed', 'AGE', 'female', 'married', 'educ_hs',
                'daca_eligible', 'arrived_before_16', 'young_enough', 'long_residence']
summary_stats = df_sample[summary_vars].describe().T
print(summary_stats[['mean', 'std', 'min', 'max']])

# ============================================================================
# DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n[5] Difference-in-Differences Analysis...")

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD ---")
df_sample['const'] = 1
X1 = df_sample[['const', 'daca_eligible', 'post', 'daca_post']]
y = df_sample['fulltime']
w = df_sample['PERWT']

model1 = sm.WLS(y, X1, weights=w).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
controls = ['AGE', 'age_sq', 'female', 'married', 'educ_hs']
X2 = df_sample[['const', 'daca_eligible', 'post', 'daca_post'] + controls]
model2 = sm.WLS(y, X2, weights=w).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with Year Fixed Effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
df_sample = df_sample.reset_index(drop=True)
year_dummies = pd.get_dummies(df_sample['YEAR'], prefix='year', drop_first=True, dtype=float)
X3 = pd.concat([df_sample[['const', 'daca_eligible', 'daca_post'] + controls],
                year_dummies], axis=1)
X3 = X3.astype(float).fillna(0)
y = df_sample['fulltime']
w = df_sample['PERWT']
model3 = sm.WLS(y.values, X3.values, weights=w.values).fit(cov_type='HC1')
# Map coefficient names
coef_names_3 = list(X3.columns)
print("\nKey coefficients:")
daca_elig_idx = coef_names_3.index('daca_eligible')
daca_post_idx = coef_names_3.index('daca_post')
print(f"DACA Eligible: {model3.params[daca_elig_idx]:.4f} (SE: {model3.bse[daca_elig_idx]:.4f})")
print(f"DACA x Post (DiD): {model3.params[daca_post_idx]:.4f} (SE: {model3.bse[daca_post_idx]:.4f})")

# Model 4: DiD with State and Year Fixed Effects
print("\n--- Model 4: DiD with State and Year Fixed Effects ---")
state_dummies = pd.get_dummies(df_sample['STATEFIP'], prefix='state', drop_first=True, dtype=float)
X4 = pd.concat([df_sample[['const', 'daca_eligible', 'daca_post'] + controls],
                year_dummies, state_dummies], axis=1)
X4 = X4.astype(float).fillna(0)
model4 = sm.WLS(y.values, X4.values, weights=w.values).fit(cov_type='HC1')
# Map coefficient names
coef_names_4 = list(X4.columns)
daca_elig_idx_4 = coef_names_4.index('daca_eligible')
daca_post_idx_4 = coef_names_4.index('daca_post')
print("\nKey coefficients:")
print(f"DACA Eligible: {model4.params[daca_elig_idx_4]:.4f} (SE: {model4.bse[daca_elig_idx_4]:.4f})")
print(f"DACA x Post (DiD): {model4.params[daca_post_idx_4]:.4f} (SE: {model4.bse[daca_post_idx_4]:.4f})")

# ============================================================================
# MAIN RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("MAIN RESULTS SUMMARY")
print("="*80)

print("\n--- Difference-in-Differences Estimates ---")
print("Outcome: Full-time Employment (35+ hours/week)")
print("-" * 70)
print(f"{'Model':<40} {'DiD Estimate':<15} {'SE':<12} {'p-value':<12}")
print("-" * 70)
for name, model, idx in [("Basic DiD", model1, 'daca_post'),
                     ("DiD + Demographics", model2, 'daca_post'),
                     ("DiD + Year FE", model3, daca_post_idx),
                     ("DiD + State & Year FE (Preferred)", model4, daca_post_idx_4)]:
    if isinstance(idx, str):
        est = model.params[idx]
        se = model.bse[idx]
        pval = model.pvalues[idx]
    else:
        est = model.params[idx]
        se = model.bse[idx]
        pval = model.pvalues[idx]
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    print(f"{name:<40} {est:>10.4f}{stars:<4} {se:>10.4f} {pval:>10.4f}")
print("-" * 70)
print("Note: *** p<0.01, ** p<0.05, * p<0.1")

# ============================================================================
# ROBUSTNESS CHECKS
# ============================================================================
print("\n[6] Robustness Checks...")

# Robustness 1: Alternative outcome - any employment
print("\n--- Robustness 1: Any Employment (not just full-time) ---")
y_emp = df_sample['employed'].values
model_emp = sm.WLS(y_emp, X4.values, weights=w.values).fit(cov_type='HC1')
print(f"DiD on Employment: {model_emp.params[daca_post_idx_4]:.4f} (SE: {model_emp.bse[daca_post_idx_4]:.4f})")

# Robustness 2: Different age cutoffs for eligibility
print("\n--- Robustness 2: Restricting to younger workers (16-35) ---")
df_young = df_sample[df_sample['AGE'] <= 35].copy().reset_index(drop=True)
year_dummies_y = pd.get_dummies(df_young['YEAR'], prefix='year', drop_first=True, dtype=float)
state_dummies_y = pd.get_dummies(df_young['STATEFIP'], prefix='state', drop_first=True, dtype=float)
X_young = pd.concat([df_young[['const', 'daca_eligible', 'daca_post'] + controls].reset_index(drop=True),
                     year_dummies_y.reset_index(drop=True), state_dummies_y.reset_index(drop=True)], axis=1)
X_young = X_young.astype(float)
coef_names_y = list(X_young.columns)
daca_post_idx_y = coef_names_y.index('daca_post')
model_young = sm.WLS(df_young['fulltime'].values, X_young.values, weights=df_young['PERWT'].values).fit(cov_type='HC1')
print(f"DiD (Age 16-35): {model_young.params[daca_post_idx_y]:.4f} (SE: {model_young.bse[daca_post_idx_y]:.4f})")

# Robustness 3: By gender
print("\n--- Robustness 3: By Gender ---")
for gender, label in [(1, 'Male'), (2, 'Female')]:
    df_g = df_sample[df_sample['SEX'] == gender].copy().reset_index(drop=True)
    year_dummies_g = pd.get_dummies(df_g['YEAR'], prefix='year', drop_first=True, dtype=float)
    state_dummies_g = pd.get_dummies(df_g['STATEFIP'], prefix='state', drop_first=True, dtype=float)
    controls_g = ['AGE', 'age_sq', 'married', 'educ_hs']
    X_g = pd.concat([df_g[['const', 'daca_eligible', 'daca_post'] + controls_g].reset_index(drop=True),
                     year_dummies_g.reset_index(drop=True), state_dummies_g.reset_index(drop=True)], axis=1)
    X_g = X_g.astype(float)
    coef_names_g = list(X_g.columns)
    daca_post_idx_g = coef_names_g.index('daca_post')
    model_g = sm.WLS(df_g['fulltime'].values, X_g.values, weights=df_g['PERWT'].values).fit(cov_type='HC1')
    print(f"DiD ({label}): {model_g.params[daca_post_idx_g]:.4f} (SE: {model_g.bse[daca_post_idx_g]:.4f})")

# Robustness 4: Placebo test using pre-period data only
print("\n--- Robustness 4: Placebo Test (Pre-period 2006-2011, fake treatment at 2009) ---")
df_pre = df_sample[df_sample['YEAR'] < 2013].copy().reset_index(drop=True)
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_did'] = df_pre['daca_eligible'] * df_pre['placebo_post']
year_dummies_p = pd.get_dummies(df_pre['YEAR'], prefix='year', drop_first=True, dtype=float)
state_dummies_p = pd.get_dummies(df_pre['STATEFIP'], prefix='state', drop_first=True, dtype=float)
X_placebo = pd.concat([df_pre[['const', 'daca_eligible', 'placebo_did'] + controls].reset_index(drop=True),
                       year_dummies_p.reset_index(drop=True), state_dummies_p.reset_index(drop=True)], axis=1)
X_placebo = X_placebo.astype(float)
coef_names_p = list(X_placebo.columns)
placebo_did_idx = coef_names_p.index('placebo_did')
model_placebo = sm.WLS(df_pre['fulltime'].values, X_placebo.values, weights=df_pre['PERWT'].values).fit(cov_type='HC1')
print(f"Placebo DiD: {model_placebo.params[placebo_did_idx]:.4f} (SE: {model_placebo.bse[placebo_did_idx]:.4f})")

# ============================================================================
# EVENT STUDY / DYNAMIC EFFECTS
# ============================================================================
print("\n[7] Event Study Analysis...")

# Create year-specific treatment effects
df_sample['year_2006'] = (df_sample['YEAR'] == 2006).astype(int)
df_sample['year_2007'] = (df_sample['YEAR'] == 2007).astype(int)
df_sample['year_2008'] = (df_sample['YEAR'] == 2008).astype(int)
df_sample['year_2009'] = (df_sample['YEAR'] == 2009).astype(int)
df_sample['year_2010'] = (df_sample['YEAR'] == 2010).astype(int)
df_sample['year_2011'] = (df_sample['YEAR'] == 2011).astype(int)
df_sample['year_2013'] = (df_sample['YEAR'] == 2013).astype(int)
df_sample['year_2014'] = (df_sample['YEAR'] == 2014).astype(int)
df_sample['year_2015'] = (df_sample['YEAR'] == 2015).astype(int)
df_sample['year_2016'] = (df_sample['YEAR'] == 2016).astype(int)

# Interaction terms (2011 as reference)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_sample[f'daca_year_{yr}'] = df_sample['daca_eligible'] * df_sample[f'year_{yr}']

event_vars = [f'daca_year_{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]
X_event = pd.concat([df_sample[['const', 'daca_eligible'] + controls + event_vars],
                     year_dummies, state_dummies], axis=1)
X_event = X_event.astype(float).fillna(0)
coef_names_event = list(X_event.columns)
model_event = sm.WLS(y.values, X_event.values, weights=w.values).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Reference: 2011)")
print("-" * 50)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var_idx = coef_names_event.index(f'daca_year_{yr}')
    coef = model_event.params[var_idx]
    se = model_event.bse[var_idx]
    print(f"Year {yr}: {coef:>8.4f} (SE: {se:.4f})")

# ============================================================================
# SAVE KEY RESULTS
# ============================================================================
print("\n[8] Saving Results...")

# Preferred estimate details
preferred_estimate = model4.params[daca_post_idx_4]
preferred_se = model4.bse[daca_post_idx_4]
preferred_ci_low = preferred_estimate - 1.96 * preferred_se
preferred_ci_high = preferred_estimate + 1.96 * preferred_se
sample_size = len(df_sample)
preferred_pval = model4.pvalues[daca_post_idx_4]

print("\n" + "="*80)
print("PREFERRED ESTIMATE (Model 4: DiD with State and Year Fixed Effects)")
print("="*80)
print(f"Effect Size: {preferred_estimate:.4f}")
print(f"Standard Error: {preferred_se:.4f}")
print(f"95% CI: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"Sample Size: {sample_size:,}")
print(f"P-value: {preferred_pval:.4f}")

# Create results dictionary for export
results = {
    'preferred_estimate': preferred_estimate,
    'standard_error': preferred_se,
    'ci_lower': preferred_ci_low,
    'ci_upper': preferred_ci_high,
    'sample_size': sample_size,
    'p_value': preferred_pval,
    'n_treated_pre': len(df_sample[(df_sample['daca_eligible']==1) & (df_sample['post']==0)]),
    'n_treated_post': len(df_sample[(df_sample['daca_eligible']==1) & (df_sample['post']==1)]),
    'n_control_pre': len(df_sample[(df_sample['daca_eligible']==0) & (df_sample['post']==0)]),
    'n_control_post': len(df_sample[(df_sample['daca_eligible']==0) & (df_sample['post']==1)]),
}

# Save results to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)
print("\nResults saved to results_summary.csv")

# Save full model summaries
with open('model_summaries.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("MODEL 1: BASIC DiD\n")
    f.write("="*80 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("="*80 + "\n")
    f.write("MODEL 2: DiD WITH DEMOGRAPHIC CONTROLS\n")
    f.write("="*80 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("="*80 + "\n")
    f.write("MODEL 4: DiD WITH STATE AND YEAR FIXED EFFECTS (PREFERRED)\n")
    f.write("="*80 + "\n")
    f.write(f"DACA Eligible: {model4.params[daca_elig_idx_4]:.6f} (SE: {model4.bse[daca_elig_idx_4]:.6f})\n")
    f.write(f"DACA x Post (DiD): {model4.params[daca_post_idx_4]:.6f} (SE: {model4.bse[daca_post_idx_4]:.6f})\n")
    f.write(f"P-value: {preferred_pval:.6f}\n")
    f.write(f"N: {int(model4.nobs):,}\n")
    f.write(f"R-squared: {model4.rsquared:.4f}\n")

print("\nModel summaries saved to model_summaries.txt")

# ============================================================================
# ADDITIONAL TABLES FOR REPORT
# ============================================================================
print("\n[9] Creating Tables for Report...")

# Table 1: Sample descriptive statistics by group
print("\n--- Table 1: Descriptive Statistics by Treatment Group ---")
desc_stats = df_sample.groupby('daca_eligible').agg({
    'fulltime': ['mean', 'std'],
    'employed': ['mean', 'std'],
    'AGE': ['mean', 'std'],
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'PERWT': 'sum'
}).round(4)
print(desc_stats)

# Table 2: Means by treatment/time cells
print("\n--- Table 2: Mean Full-time Employment by Treatment/Time ---")
means_table = df_sample.groupby(['daca_eligible', 'post']).agg({
    'fulltime': 'mean',
    'PERWT': 'sum'
}).round(4)
print(means_table)

# Simple DiD calculation
ft_00 = df_sample[(df_sample['daca_eligible']==0) & (df_sample['post']==0)]['fulltime'].mean()
ft_01 = df_sample[(df_sample['daca_eligible']==0) & (df_sample['post']==1)]['fulltime'].mean()
ft_10 = df_sample[(df_sample['daca_eligible']==1) & (df_sample['post']==0)]['fulltime'].mean()
ft_11 = df_sample[(df_sample['daca_eligible']==1) & (df_sample['post']==1)]['fulltime'].mean()

simple_did = (ft_11 - ft_10) - (ft_01 - ft_00)
print(f"\nSimple DiD (unadjusted): {simple_did:.4f}")
print(f"  Treated change: {ft_11 - ft_10:.4f}")
print(f"  Control change: {ft_01 - ft_00:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
