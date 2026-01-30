"""
DACA Replication Study #86
Analysis of the effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals

Research Question: What was the causal impact of DACA eligibility on
full-time employment (35+ hours/week)?

Identification: Difference-in-differences comparing ages 26-30 (treated)
vs ages 31-35 (control) as of June 15, 2012
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
print("DACA REPLICATION STUDY #86")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# SAMPLE CONSTRUCTION
# ============================================================================
print("\n[2] Constructing analytic sample...")

# Step 1: Hispanic-Mexican ethnicity (HISPAN == 1 indicates Mexican)
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican filter: {len(df_sample):,}")

# Step 2: Born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200]
print(f"After Mexico birthplace filter: {len(df_sample):,}")

# Step 3: Non-citizen (CITIZEN == 3 indicates "Not a citizen")
# This is our proxy for undocumented status as per instructions
df_sample = df_sample[df_sample['CITIZEN'] == 3]
print(f"After non-citizen filter: {len(df_sample):,}")

# Step 4: Check year of immigration for DACA criteria
# Arrived before age 16: YRIMMIG - BIRTHYR < 16
# Note: YRIMMIG == 0 means N/A, we need to handle this
df_sample = df_sample[df_sample['YRIMMIG'] > 0]  # Valid immigration year
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_arrival'] < 16]
print(f"After arrival before age 16 filter: {len(df_sample):,}")

# Step 5: Continuous presence since June 2007 (arrived by 2007)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]
print(f"After arrival by 2007 filter: {len(df_sample):,}")

# Step 6: Define treatment and control groups based on birth year
# Age on June 15, 2012:
# - Treatment (ages 26-30): Born 1982-1986
# - Control (ages 31-35): Born 1977-1981
df_sample['treat'] = ((df_sample['BIRTHYR'] >= 1982) &
                       (df_sample['BIRTHYR'] <= 1986)).astype(int)
df_sample['control'] = ((df_sample['BIRTHYR'] >= 1977) &
                         (df_sample['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control groups
df_sample = df_sample[(df_sample['treat'] == 1) | (df_sample['control'] == 1)]
print(f"After age group filter (born 1977-1986): {len(df_sample):,}")

# Step 7: Define pre/post periods
# Pre-DACA: 2006-2011
# Post-DACA: 2013-2016
# Exclude 2012 (DACA implemented June 15, 2012)
df_sample = df_sample[df_sample['YEAR'] != 2012]
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
print(f"After excluding 2012: {len(df_sample):,}")

# ============================================================================
# OUTCOME VARIABLE
# ============================================================================
print("\n[3] Creating outcome variable...")

# Full-time employment: Usually works 35+ hours per week
# UHRSWORK == 0 typically means not employed/NA
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator for reference
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

print(f"Full-time employment rate (unweighted): {df_sample['fulltime'].mean():.3f}")
print(f"Employment rate (unweighted): {df_sample['employed'].mean():.3f}")

# ============================================================================
# SAMPLE CHARACTERISTICS
# ============================================================================
print("\n[4] Sample characteristics...")
print(f"\nFinal analytic sample: {len(df_sample):,}")
print(f"  Treatment group (ages 26-30 in 2012): {(df_sample['treat']==1).sum():,}")
print(f"  Control group (ages 31-35 in 2012): {(df_sample['control']==1).sum():,}")
print(f"  Pre-period (2006-2011): {(df_sample['post']==0).sum():,}")
print(f"  Post-period (2013-2016): {(df_sample['post']==1).sum():,}")

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[5] Descriptive Statistics by Group and Period...")

# Create age variable (age at survey)
df_sample['age'] = df_sample['YEAR'] - df_sample['BIRTHYR']

# Demographics by treatment/control and pre/post
desc_stats = df_sample.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'age': 'mean',
    'SEX': lambda x: (x == 2).mean(),  # Female proportion
    'PERWT': 'sum'
}).round(4)

print("\nSummary by Treatment Group and Period:")
print("=" * 60)
for treat in [0, 1]:
    group_name = "Treatment (26-30)" if treat == 1 else "Control (31-35)"
    print(f"\n{group_name}:")
    for post in [0, 1]:
        period_name = "Post-DACA (2013-2016)" if post == 1 else "Pre-DACA (2006-2011)"
        subset = df_sample[(df_sample['treat'] == treat) & (df_sample['post'] == post)]
        n = len(subset)
        weighted_n = subset['PERWT'].sum()
        ft_rate = np.average(subset['fulltime'], weights=subset['PERWT'])
        print(f"  {period_name}: N={n:,}, Weighted N={weighted_n:,.0f}, FT Rate={ft_rate:.4f}")

# ============================================================================
# DIFFERENCE-IN-DIFFERENCES ESTIMATION
# ============================================================================
print("\n" + "=" * 80)
print("[6] DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# Create interaction term
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

# -----------------------------
# Model 1: Simple DiD (no controls)
# -----------------------------
print("\n--- Model 1: Simple DiD (no controls) ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_sample,
                  weights=df_sample['PERWT'])
results1 = model1.fit(cov_type='HC1')  # Robust standard errors
print(results1.summary().tables[1])

# -----------------------------
# Model 2: DiD with demographic controls
# -----------------------------
print("\n--- Model 2: DiD with demographic controls ---")

# Create control variables
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Education categories
df_sample['educ_hs'] = (df_sample['EDUC'] >= 6).astype(int)  # HS or more
df_sample['educ_college'] = (df_sample['EDUC'] >= 7).astype(int)  # Some college or more

# Number of children
df_sample['has_children'] = (df_sample['NCHILD'] > 0).astype(int)

model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + has_children + age',
                  data=df_sample,
                  weights=df_sample['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])

# -----------------------------
# Model 3: DiD with year fixed effects
# -----------------------------
print("\n--- Model 3: DiD with year fixed effects ---")

# Create year dummies
year_dummies = pd.get_dummies(df_sample['YEAR'], prefix='year', drop_first=True)
df_sample = pd.concat([df_sample, year_dummies], axis=1)

year_cols = [col for col in df_sample.columns if col.startswith('year_')]
formula3 = 'fulltime ~ treat + treat_post + female + married + educ_hs + has_children + ' + ' + '.join(year_cols)

model3 = smf.wls(formula3, data=df_sample, weights=df_sample['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(f"\nDiD coefficient (treat_post): {results3.params['treat_post']:.4f}")
print(f"Standard error: {results3.bse['treat_post']:.4f}")
print(f"95% CI: [{results3.conf_int().loc['treat_post', 0]:.4f}, {results3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {results3.pvalues['treat_post']:.4f}")

# -----------------------------
# Model 4: DiD with state fixed effects
# -----------------------------
print("\n--- Model 4: DiD with state fixed effects ---")

state_dummies = pd.get_dummies(df_sample['STATEFIP'], prefix='state', drop_first=True)
df_sample = pd.concat([df_sample, state_dummies], axis=1)

state_cols = [col for col in df_sample.columns if col.startswith('state_')]
formula4 = 'fulltime ~ treat + treat_post + female + married + educ_hs + has_children + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)

model4 = smf.wls(formula4, data=df_sample, weights=df_sample['PERWT'])
results4 = model4.fit(cov_type='HC1')
print(f"\nDiD coefficient (treat_post): {results4.params['treat_post']:.4f}")
print(f"Standard error: {results4.bse['treat_post']:.4f}")
print(f"95% CI: [{results4.conf_int().loc['treat_post', 0]:.4f}, {results4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {results4.pvalues['treat_post']:.4f}")

# ============================================================================
# CALCULATE SIMPLE DiD MANUALLY
# ============================================================================
print("\n" + "=" * 80)
print("[7] MANUAL DiD CALCULATION (Weighted)")
print("=" * 80)

# Calculate weighted means for each cell
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

# Treatment group
treat_pre = df_sample[(df_sample['treat'] == 1) & (df_sample['post'] == 0)]
treat_post_df = df_sample[(df_sample['treat'] == 1) & (df_sample['post'] == 1)]

# Control group
ctrl_pre = df_sample[(df_sample['treat'] == 0) & (df_sample['post'] == 0)]
ctrl_post_df = df_sample[(df_sample['treat'] == 0) & (df_sample['post'] == 1)]

# Weighted means
y_treat_pre = weighted_mean(treat_pre, 'fulltime', 'PERWT')
y_treat_post = weighted_mean(treat_post_df, 'fulltime', 'PERWT')
y_ctrl_pre = weighted_mean(ctrl_pre, 'fulltime', 'PERWT')
y_ctrl_post = weighted_mean(ctrl_post_df, 'fulltime', 'PERWT')

# DiD estimate
did_estimate = (y_treat_post - y_treat_pre) - (y_ctrl_post - y_ctrl_pre)

print(f"\nWeighted Full-Time Employment Rates:")
print(f"                    Pre-DACA    Post-DACA    Difference")
print(f"  Treatment (26-30): {y_treat_pre:.4f}      {y_treat_post:.4f}       {y_treat_post - y_treat_pre:+.4f}")
print(f"  Control (31-35):   {y_ctrl_pre:.4f}      {y_ctrl_post:.4f}       {y_ctrl_post - y_ctrl_pre:+.4f}")
print(f"\n  DiD Estimate:      {did_estimate:+.4f}")

# ============================================================================
# ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("[8] ROBUSTNESS CHECKS")
print("=" * 80)

# Robustness 1: Alternative outcome - any employment
print("\n--- Robustness 1: Any Employment (EMPSTAT==1) ---")
model_emp = smf.wls('employed ~ treat + post + treat_post + female + married + educ_hs + has_children',
                     data=df_sample,
                     weights=df_sample['PERWT'])
results_emp = model_emp.fit(cov_type='HC1')
print(f"DiD coefficient: {results_emp.params['treat_post']:.4f} (SE: {results_emp.bse['treat_post']:.4f})")

# Robustness 2: By gender
print("\n--- Robustness 2: By Gender ---")
for sex, sex_name in [(1, "Men"), (2, "Women")]:
    subset = df_sample[df_sample['SEX'] == sex]
    model_sex = smf.wls('fulltime ~ treat + post + treat_post + married + educ_hs + has_children',
                         data=subset,
                         weights=subset['PERWT'])
    results_sex = model_sex.fit(cov_type='HC1')
    print(f"  {sex_name}: DiD = {results_sex.params['treat_post']:.4f} (SE: {results_sex.bse['treat_post']:.4f}), N = {len(subset):,}")

# Robustness 3: Placebo test using pre-period only
print("\n--- Robustness 3: Placebo Test (Pre-Period 2006-2008 vs 2009-2011) ---")
pre_only = df_sample[df_sample['post'] == 0].copy()
pre_only['fake_post'] = (pre_only['YEAR'] >= 2009).astype(int)
pre_only['treat_fake_post'] = pre_only['treat'] * pre_only['fake_post']

model_placebo = smf.wls('fulltime ~ treat + fake_post + treat_fake_post + female + married + educ_hs + has_children',
                         data=pre_only,
                         weights=pre_only['PERWT'])
results_placebo = model_placebo.fit(cov_type='HC1')
print(f"Placebo DiD coefficient: {results_placebo.params['treat_fake_post']:.4f} (SE: {results_placebo.bse['treat_fake_post']:.4f})")
print(f"p-value: {results_placebo.pvalues['treat_fake_post']:.4f}")

# ============================================================================
# YEARLY TRENDS
# ============================================================================
print("\n" + "=" * 80)
print("[9] YEARLY TRENDS IN FULL-TIME EMPLOYMENT")
print("=" * 80)

yearly_trends = df_sample.groupby(['YEAR', 'treat']).apply(
    lambda x: pd.Series({
        'ft_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()

print("\nYear    Treatment FT Rate    Control FT Rate    Difference")
print("-" * 60)
for year in sorted(df_sample['YEAR'].unique()):
    treat_rate = yearly_trends[(yearly_trends['YEAR'] == year) & (yearly_trends['treat'] == 1)]['ft_rate'].values[0]
    ctrl_rate = yearly_trends[(yearly_trends['YEAR'] == year) & (yearly_trends['treat'] == 0)]['ft_rate'].values[0]
    diff = treat_rate - ctrl_rate
    print(f"{year}    {treat_rate:.4f}              {ctrl_rate:.4f}              {diff:+.4f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("[10] FINAL RESULTS SUMMARY")
print("=" * 80)

print("\n*** PREFERRED ESTIMATE (Model 4 with Year and State FE) ***")
print(f"Effect Size (DiD coefficient): {results4.params['treat_post']:.4f}")
print(f"Standard Error: {results4.bse['treat_post']:.4f}")
print(f"95% Confidence Interval: [{results4.conf_int().loc['treat_post', 0]:.4f}, {results4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"t-statistic: {results4.tvalues['treat_post']:.4f}")
print(f"p-value: {results4.pvalues['treat_post']:.4f}")
print(f"Sample Size: {len(df_sample):,}")

print("\n*** INTERPRETATION ***")
if results4.params['treat_post'] > 0:
    print(f"DACA eligibility is associated with a {results4.params['treat_post']*100:.2f} percentage point")
    print(f"INCREASE in the probability of full-time employment.")
else:
    print(f"DACA eligibility is associated with a {abs(results4.params['treat_post'])*100:.2f} percentage point")
    print(f"DECREASE in the probability of full-time employment.")

if results4.pvalues['treat_post'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
elif results4.pvalues['treat_post'] < 0.10:
    print("This effect is statistically significant at the 10% level.")
else:
    print("This effect is NOT statistically significant at conventional levels.")

# ============================================================================
# SAVE RESULTS FOR REPORT
# ============================================================================
print("\n[11] Saving results...")

# Save summary statistics
summary_table = pd.DataFrame({
    'Model': ['Simple DiD', 'With Controls', 'Year FE', 'Year + State FE'],
    'Coefficient': [results1.params['treat_post'], results2.params['treat_post'],
                    results3.params['treat_post'], results4.params['treat_post']],
    'Std_Error': [results1.bse['treat_post'], results2.bse['treat_post'],
                  results3.bse['treat_post'], results4.bse['treat_post']],
    'CI_Lower': [results1.conf_int().loc['treat_post', 0], results2.conf_int().loc['treat_post', 0],
                 results3.conf_int().loc['treat_post', 0], results4.conf_int().loc['treat_post', 0]],
    'CI_Upper': [results1.conf_int().loc['treat_post', 1], results2.conf_int().loc['treat_post', 1],
                 results3.conf_int().loc['treat_post', 1], results4.conf_int().loc['treat_post', 1]],
    'P_Value': [results1.pvalues['treat_post'], results2.pvalues['treat_post'],
                results3.pvalues['treat_post'], results4.pvalues['treat_post']]
})
summary_table.to_csv('results_summary_86.csv', index=False)
print("Results saved to results_summary_86.csv")

# Save descriptive statistics
desc_output = df_sample.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'age': 'mean',
    'PERWT': 'sum'
}).round(4)
desc_output.to_csv('descriptive_stats_86.csv')
print("Descriptive statistics saved to descriptive_stats_86.csv")

# Save yearly trends
yearly_trends.to_csv('yearly_trends_86.csv', index=False)
print("Yearly trends saved to yearly_trends_86.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
