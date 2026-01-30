"""
DACA Replication Study: Effect on Full-Time Employment
Analysis Script

Research Question: Among ethnically Hispanic-Mexican Mexican-born people
living in the United States, what was the causal impact of eligibility
for DACA on the probability of full-time employment (>=35 hours/week)?

Identification Strategy: Difference-in-Differences
- Treatment: DACA eligibility
- Control: Similar non-citizen Mexican-born Hispanics who don't meet DACA criteria
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded due to mid-year implementation)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import json
import os
warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r'C:\Users\seraf\DACA Results Task 1\replication_98')

print("="*70)
print("DACA REPLICATION STUDY: EFFECT ON FULL-TIME EMPLOYMENT")
print("="*70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[1] Loading ACS data...")

# Read CSV in chunks due to large file size
chunks = []
chunksize = 500000

for chunk in pd.read_csv('data/data.csv', chunksize=chunksize, low_memory=False):
    chunks.append(chunk)
    print(f"  Loaded chunk: {len(chunk):,} rows")

df = pd.concat(chunks, ignore_index=True)
print(f"  Total observations loaded: {len(df):,}")

# Show years available
print(f"  Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: SAMPLE RESTRICTION
# =============================================================================
print("\n[2] Restricting sample...")

# Initial count
initial_n = len(df)
print(f"  Initial observations: {initial_n:,}")

# Restrict to Hispanic-Mexican (HISPAN == 1)
df = df[df['HISPAN'] == 1].copy()
print(f"  After Hispanic-Mexican restriction: {len(df):,}")

# Restrict to born in Mexico (BPL == 200)
df = df[df['BPL'] == 200].copy()
print(f"  After Mexico birthplace restriction: {len(df):,}")

# Restrict to non-citizens (CITIZEN == 3)
# Per instructions: "Assume that anyone who is not a citizen and who has not
# received immigration papers is undocumented for DACA purposes"
df = df[df['CITIZEN'] == 3].copy()
print(f"  After non-citizen restriction: {len(df):,}")

# Exclude 2012 (DACA implemented mid-year)
df = df[df['YEAR'] != 2012].copy()
print(f"  After excluding 2012: {len(df):,}")

# =============================================================================
# STEP 3: CONSTRUCT DACA ELIGIBILITY VARIABLE
# =============================================================================
print("\n[3] Constructing DACA eligibility indicator...")

# DACA eligibility criteria:
# 1. Arrived unlawfully in the US before their 16th birthday
# 2. Had not yet had their 31st birthday as of June 15, 2012
# 3. Lived continuously in the US since June 15, 2007
# 4. Were present in the US on June 15, 2012

# Key date: June 15, 2012
# Birth year cutoffs:
# - Must be born after June 15, 1981 (under 31 on June 15, 2012)
# - For arrival before 16th birthday: arrival_year - birth_year < 16

# Create age at arrival (approximation using YRIMMIG and BIRTHYR)
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Age as of June 15, 2012
df['age_june2012'] = 2012 - df['BIRTHYR']

# Criterion 1: Arrived before 16th birthday
df['arrived_before_16'] = df['age_at_arrival'] < 16

# Criterion 2: Under 31 on June 15, 2012 (born after June 15, 1981)
# Being conservative: born in 1982 or later guarantees under 31
# Born in 1981 with BIRTHQTR > 2 (Jul-Dec) also qualifies
df['under_31_june2012'] = (df['BIRTHYR'] >= 1982) | \
                          ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))

# Criterion 3: In US since June 15, 2007 (arrived 2007 or earlier)
# Being conservative: YRIMMIG <= 2007
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007

# Criterion 4: Present in US on June 15, 2012 (and non-citizen - already filtered)
# We assume continuous presence for those who immigrated by 2007

# Additional criterion from instructions: must also be at least 15 to apply
# (as of June 15, 2012). Born on or before June 15, 1997.
# Conservative: born 1996 or earlier, or born in 1997 with BIRTHQTR 1-2
df['at_least_15_june2012'] = (df['BIRTHYR'] <= 1996) | \
                              ((df['BIRTHYR'] == 1997) & (df['BIRTHQTR'] <= 2))

# DACA eligible if ALL criteria met
df['daca_eligible'] = (df['arrived_before_16'] &
                       df['under_31_june2012'] &
                       df['in_us_since_2007'] &
                       df['at_least_15_june2012'])

# Handle missing values in YRIMMIG (code 0 means N/A or not applicable)
df.loc[df['YRIMMIG'] == 0, 'daca_eligible'] = False
df.loc[df['YRIMMIG'] == 0, 'arrived_before_16'] = False
df.loc[df['YRIMMIG'] == 0, 'in_us_since_2007'] = False

print(f"  Observations with valid YRIMMIG: {(df['YRIMMIG'] > 0).sum():,}")
print(f"  DACA eligible: {df['daca_eligible'].sum():,}")
print(f"  DACA ineligible: {(~df['daca_eligible']).sum():,}")

# =============================================================================
# STEP 4: CONSTRUCT OUTCOME AND TREATMENT VARIABLES
# =============================================================================
print("\n[4] Constructing outcome and treatment variables...")

# Outcome: Full-time employment (UHRSWORK >= 35)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employment indicator (EMPSTAT == 1 is employed)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Post-DACA period indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Treatment indicator
df['treated'] = df['daca_eligible'].astype(int)

# DiD interaction term
df['did'] = df['treated'] * df['post']

print(f"  Full-time employment rate (overall): {df['fulltime'].mean()*100:.1f}%")
print(f"  Employment rate (overall): {df['employed'].mean()*100:.1f}%")

# =============================================================================
# STEP 5: RESTRICT TO WORKING-AGE POPULATION
# =============================================================================
print("\n[5] Restricting to working-age population...")

# Focus on working-age individuals (ages 16-64)
# This also ensures we're looking at people who could reasonably be employed
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
print(f"  After age 16-64 restriction: {len(df):,}")

# Remove observations with YRIMMIG == 0 (missing immigration year)
# These can't be properly classified for DACA eligibility
df = df[df['YRIMMIG'] > 0].copy()
print(f"  After removing missing YRIMMIG: {len(df):,}")

# Final sample counts
print(f"\n  Final sample size: {len(df):,}")
print(f"  DACA eligible: {df['daca_eligible'].sum():,}")
print(f"  DACA ineligible: {(~df['daca_eligible']).sum():,}")

# =============================================================================
# STEP 6: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[6] Generating descriptive statistics...")

# Create a results dictionary to store all results
results = {}

# Summary by treatment group and period
summary_cols = ['fulltime', 'employed', 'AGE', 'SEX', 'EDUC', 'MARST']

print("\n  Sample sizes by group and period:")
for period_name, period_mask in [('Pre-DACA (2006-2011)', df['post']==0),
                                  ('Post-DACA (2013-2016)', df['post']==1)]:
    for group_name, group_mask in [('DACA Eligible', df['treated']==1),
                                    ('DACA Ineligible', df['treated']==0)]:
        n = (period_mask & group_mask).sum()
        ft_rate = df.loc[period_mask & group_mask, 'fulltime'].mean() * 100
        print(f"    {period_name}, {group_name}: N={n:,}, FT rate={ft_rate:.1f}%")

# Store key descriptive stats
desc_stats = df.groupby(['post', 'treated']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(4)

# Convert descriptive stats to serializable format
results['descriptive_stats'] = desc_stats.to_string()

# =============================================================================
# STEP 7: DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n[7] Running Difference-in-Differences analysis...")

# --- 7a: Simple DiD (no controls) ---
print("\n  [7a] Simple DiD (no controls):")

# Calculate means for 2x2 DiD
means = df.groupby(['post', 'treated'])['fulltime'].mean().unstack()
print(f"\n  Full-time employment rates:")
print(f"                    Ineligible    Eligible")
print(f"  Pre-DACA          {means.loc[0, 0]*100:.2f}%        {means.loc[0, 1]*100:.2f}%")
print(f"  Post-DACA         {means.loc[1, 0]*100:.2f}%        {means.loc[1, 1]*100:.2f}%")

# DiD estimate
diff_eligible = means.loc[1, 1] - means.loc[0, 1]
diff_ineligible = means.loc[1, 0] - means.loc[0, 0]
did_simple = diff_eligible - diff_ineligible

print(f"\n  Change for Eligible:   {diff_eligible*100:+.2f} pp")
print(f"  Change for Ineligible: {diff_ineligible*100:+.2f} pp")
print(f"  DiD Estimate:          {did_simple*100:+.2f} pp")

# --- 7b: DiD with OLS regression (no controls) ---
print("\n  [7b] DiD regression (no controls):")

model_simple = smf.ols('fulltime ~ treated + post + did', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"  DiD coefficient: {model_simple.params['did']:.4f}")
print(f"  Standard error:  {model_simple.bse['did']:.4f}")
print(f"  t-statistic:     {model_simple.tvalues['did']:.3f}")
print(f"  p-value:         {model_simple.pvalues['did']:.4f}")

results['simple_did'] = {
    'coefficient': float(model_simple.params['did']),
    'se': float(model_simple.bse['did']),
    'tstat': float(model_simple.tvalues['did']),
    'pvalue': float(model_simple.pvalues['did']),
    'n': int(len(df))
}

# --- 7c: DiD with controls ---
print("\n  [7c] DiD regression with controls:")

# Create control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # High school or more
df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # 4+ years college

# Age squared for quadratic
df['age_sq'] = df['AGE'] ** 2

# Year fixed effects
df['year_fe'] = df['YEAR'].astype(str)

# Model with individual controls
model_controls = smf.ols(
    'fulltime ~ treated + post + did + AGE + age_sq + female + married + educ_hs',
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"  DiD coefficient: {model_controls.params['did']:.4f}")
print(f"  Standard error:  {model_controls.bse['did']:.4f}")
print(f"  t-statistic:     {model_controls.tvalues['did']:.3f}")
print(f"  p-value:         {model_controls.pvalues['did']:.4f}")

results['controls_did'] = {
    'coefficient': float(model_controls.params['did']),
    'se': float(model_controls.bse['did']),
    'tstat': float(model_controls.tvalues['did']),
    'pvalue': float(model_controls.pvalues['did']),
    'n': int(model_controls.nobs)
}

# --- 7d: DiD with controls and year fixed effects ---
print("\n  [7d] DiD regression with controls and year FE:")

model_full = smf.ols(
    'fulltime ~ treated + did + AGE + age_sq + female + married + educ_hs + C(YEAR)',
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"  DiD coefficient: {model_full.params['did']:.4f}")
print(f"  Standard error:  {model_full.bse['did']:.4f}")
print(f"  t-statistic:     {model_full.tvalues['did']:.3f}")
print(f"  p-value:         {model_full.pvalues['did']:.4f}")

results['full_did'] = {
    'coefficient': float(model_full.params['did']),
    'se': float(model_full.bse['did']),
    'tstat': float(model_full.tvalues['did']),
    'pvalue': float(model_full.pvalues['did']),
    'n': int(model_full.nobs),
    'r2': float(model_full.rsquared)
}

# --- 7e: DiD with state fixed effects ---
print("\n  [7e] DiD regression with controls and state + year FE:")

model_state_fe = smf.ols(
    'fulltime ~ treated + did + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"  DiD coefficient: {model_state_fe.params['did']:.4f}")
print(f"  Standard error:  {model_state_fe.bse['did']:.4f}")
print(f"  t-statistic:     {model_state_fe.tvalues['did']:.3f}")
print(f"  p-value:         {model_state_fe.pvalues['did']:.4f}")

results['state_fe_did'] = {
    'coefficient': float(model_state_fe.params['did']),
    'se': float(model_state_fe.bse['did']),
    'tstat': float(model_state_fe.tvalues['did']),
    'pvalue': float(model_state_fe.pvalues['did']),
    'n': int(model_state_fe.nobs),
    'r2': float(model_state_fe.rsquared)
}

# =============================================================================
# STEP 8: WEIGHTED ANALYSIS
# =============================================================================
print("\n[8] Running weighted DiD analysis...")

# Using survey weights (PERWT) with WLS via smf
# Create a numeric year variable for dummies
df['YEAR_num'] = df['YEAR'].astype(int)

# Prepare weighted regression using formula API with freq_weights
# statsmodels formula API doesn't directly support freq_weights in ols, so use WLS manually
from statsmodels.regression.linear_model import WLS

# Prepare numeric X matrix
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
X_weighted = df[['treated', 'post', 'did', 'AGE', 'age_sq', 'female', 'married', 'educ_hs']].copy()
X_weighted = pd.concat([X_weighted, year_dummies], axis=1)
X_weighted = X_weighted.astype(float)
X_weighted = sm.add_constant(X_weighted)

y_weighted = df['fulltime'].astype(float).values
weights = df['PERWT'].astype(float).values

# Run weighted OLS
model_weighted = WLS(y_weighted, X_weighted.values, weights=weights).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP'].values})

# Get the index of 'did' column
did_idx = list(X_weighted.columns).index('did')
print(f"  Weighted DiD coefficient: {model_weighted.params[did_idx]:.4f}")
print(f"  Standard error:           {model_weighted.bse[did_idx]:.4f}")
print(f"  t-statistic:              {model_weighted.tvalues[did_idx]:.3f}")
print(f"  p-value:                  {model_weighted.pvalues[did_idx]:.4f}")

results['weighted_did'] = {
    'coefficient': float(model_weighted.params[did_idx]),
    'se': float(model_weighted.bse[did_idx]),
    'tstat': float(model_weighted.tvalues[did_idx]),
    'pvalue': float(model_weighted.pvalues[did_idx]),
    'n': int(model_weighted.nobs)
}

# =============================================================================
# STEP 9: ROBUSTNESS CHECKS
# =============================================================================
print("\n[9] Running robustness checks...")

# --- 9a: Alternative outcome - any employment ---
print("\n  [9a] Alternative outcome - Employment (any hours):")

model_emp = smf.ols(
    'employed ~ treated + did + AGE + age_sq + female + married + educ_hs + C(YEAR)',
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"  DiD coefficient: {model_emp.params['did']:.4f}")
print(f"  Standard error:  {model_emp.bse['did']:.4f}")
print(f"  p-value:         {model_emp.pvalues['did']:.4f}")

results['employment_did'] = {
    'coefficient': float(model_emp.params['did']),
    'se': float(model_emp.bse['did']),
    'pvalue': float(model_emp.pvalues['did']),
    'n': int(model_emp.nobs)
}

# --- 9b: Placebo test - Pre-trends (using 2009 as fake treatment) ---
print("\n  [9b] Placebo test (pre-trends, fake treatment in 2009):")

df_pre = df[df['YEAR'] <= 2011].copy()
df_pre['post_placebo'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['did_placebo'] = df_pre['treated'] * df_pre['post_placebo']

model_placebo = smf.ols(
    'fulltime ~ treated + post_placebo + did_placebo + AGE + age_sq + female + married + educ_hs',
    data=df_pre
).fit(cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})

print(f"  Placebo DiD coefficient: {model_placebo.params['did_placebo']:.4f}")
print(f"  Standard error:          {model_placebo.bse['did_placebo']:.4f}")
print(f"  p-value:                 {model_placebo.pvalues['did_placebo']:.4f}")

results['placebo_did'] = {
    'coefficient': float(model_placebo.params['did_placebo']),
    'se': float(model_placebo.bse['did_placebo']),
    'pvalue': float(model_placebo.pvalues['did_placebo']),
    'n': int(model_placebo.nobs)
}

# --- 9c: Event study / Dynamic DiD ---
print("\n  [9c] Event study (year-by-year effects):")

# Create year dummies interacted with treatment
# Use 2011 as reference year
df['year_2006'] = (df['YEAR'] == 2006).astype(int)
df['year_2007'] = (df['YEAR'] == 2007).astype(int)
df['year_2008'] = (df['YEAR'] == 2008).astype(int)
df['year_2009'] = (df['YEAR'] == 2009).astype(int)
df['year_2010'] = (df['YEAR'] == 2010).astype(int)
# 2011 is reference
df['year_2013'] = (df['YEAR'] == 2013).astype(int)
df['year_2014'] = (df['YEAR'] == 2014).astype(int)
df['year_2015'] = (df['YEAR'] == 2015).astype(int)
df['year_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions with treatment
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'treat_x_{yr}'] = df['treated'] * df[f'year_{yr}']

model_event = smf.ols(
    '''fulltime ~ treated + treat_x_2006 + treat_x_2007 + treat_x_2008 +
       treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 +
       treat_x_2015 + treat_x_2016 + AGE + age_sq + female + married +
       educ_hs + C(YEAR)''',
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"\n  Year-by-year treatment effects (relative to 2011):")
event_study_results = {}
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treat_x_{yr}']
    se = model_event.bse[f'treat_x_{yr}']
    pval = model_event.pvalues[f'treat_x_{yr}']
    sig = '*' if pval < 0.05 else ''
    print(f"    {yr}: {coef:+.4f} (SE: {se:.4f}){sig}")
    event_study_results[str(yr)] = {
        'coefficient': float(coef),
        'se': float(se),
        'pvalue': float(pval)
    }

results['event_study'] = event_study_results

# --- 9d: Subgroup analysis by gender ---
print("\n  [9d] Subgroup analysis by gender:")

for gender, gender_mask in [('Male', df['female']==0), ('Female', df['female']==1)]:
    model_gender = smf.ols(
        'fulltime ~ treated + did + AGE + age_sq + married + educ_hs + C(YEAR)',
        data=df[gender_mask]
    ).fit(cov_type='cluster', cov_kwds={'groups': df.loc[gender_mask, 'STATEFIP']})

    print(f"    {gender}: DiD = {model_gender.params['did']:.4f} "
          f"(SE: {model_gender.bse['did']:.4f}, p: {model_gender.pvalues['did']:.4f})")

    results[f'gender_{gender.lower()}_did'] = {
        'coefficient': float(model_gender.params['did']),
        'se': float(model_gender.bse['did']),
        'pvalue': float(model_gender.pvalues['did']),
        'n': int(model_gender.nobs)
    }

# =============================================================================
# STEP 10: SUMMARY AND OUTPUT
# =============================================================================
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)

print("\n[PREFERRED ESTIMATE - Model with Controls and Year FE]")
print(f"  Effect of DACA eligibility on full-time employment:")
print(f"  Coefficient:        {results['full_did']['coefficient']:.4f}")
print(f"  Standard Error:     {results['full_did']['se']:.4f}")
print(f"  95% CI:             [{results['full_did']['coefficient'] - 1.96*results['full_did']['se']:.4f}, "
      f"{results['full_did']['coefficient'] + 1.96*results['full_did']['se']:.4f}]")
print(f"  p-value:            {results['full_did']['pvalue']:.4f}")
print(f"  Sample Size:        {results['full_did']['n']:,}")

# Interpret
if results['full_did']['coefficient'] > 0:
    direction = "increased"
else:
    direction = "decreased"

sig_level = "not statistically significant"
if results['full_did']['pvalue'] < 0.01:
    sig_level = "statistically significant at the 1% level"
elif results['full_did']['pvalue'] < 0.05:
    sig_level = "statistically significant at the 5% level"
elif results['full_did']['pvalue'] < 0.10:
    sig_level = "marginally significant at the 10% level"

print(f"\n  Interpretation: DACA eligibility {direction} full-time employment by")
print(f"  {abs(results['full_did']['coefficient'])*100:.2f} percentage points, {sig_level}.")

# Save results to JSON
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n  Results saved to analysis_results.json")

# =============================================================================
# STEP 11: GENERATE TABLES FOR LATEX
# =============================================================================
print("\n[11] Generating LaTeX tables...")

# Table 1: Summary Statistics
print("\n  Generating summary statistics table...")

# Calculate detailed stats
summary_data = []
for period_name, post_val in [('Pre-DACA (2006-2011)', 0), ('Post-DACA (2013-2016)', 1)]:
    for group_name, treat_val in [('DACA Eligible', 1), ('DACA Ineligible', 0)]:
        mask = (df['post'] == post_val) & (df['treated'] == treat_val)
        subset = df[mask]
        summary_data.append({
            'Period': period_name,
            'Group': group_name,
            'N': len(subset),
            'Full-time Rate': subset['fulltime'].mean(),
            'Employment Rate': subset['employed'].mean(),
            'Mean Age': subset['AGE'].mean(),
            'Female Share': subset['female'].mean(),
            'Married Share': subset['married'].mean(),
            'HS+ Share': subset['educ_hs'].mean()
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('summary_statistics.csv', index=False)
print("  Saved summary_statistics.csv")

# Table 2: Regression Results
print("  Generating regression results table...")

reg_results = {
    'Model': ['Simple DiD', 'With Controls', 'Controls + Year FE', 'Controls + State & Year FE', 'Weighted'],
    'DiD Coefficient': [
        results['simple_did']['coefficient'],
        results['controls_did']['coefficient'],
        results['full_did']['coefficient'],
        results['state_fe_did']['coefficient'],
        results['weighted_did']['coefficient']
    ],
    'Std Error': [
        results['simple_did']['se'],
        results['controls_did']['se'],
        results['full_did']['se'],
        results['state_fe_did']['se'],
        results['weighted_did']['se']
    ],
    'p-value': [
        results['simple_did']['pvalue'],
        results['controls_did']['pvalue'],
        results['full_did']['pvalue'],
        results['state_fe_did']['pvalue'],
        results['weighted_did']['pvalue']
    ],
    'N': [
        results['simple_did']['n'],
        results['controls_did']['n'],
        results['full_did']['n'],
        results['state_fe_did']['n'],
        results['weighted_did']['n']
    ]
}

reg_df = pd.DataFrame(reg_results)
reg_df.to_csv('regression_results.csv', index=False)
print("  Saved regression_results.csv")

# Table 3: Event study results
event_df = pd.DataFrame(event_study_results).T
event_df.index.name = 'Year'
event_df.to_csv('event_study_results.csv')
print("  Saved event_study_results.csv")

# =============================================================================
# GENERATE FULL MODEL SUMMARY
# =============================================================================
print("\n[12] Full regression output for preferred model:")
print(model_full.summary())

# Save full model summary
with open('preferred_model_summary.txt', 'w') as f:
    f.write(str(model_full.summary()))
print("\n  Saved preferred_model_summary.txt")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
