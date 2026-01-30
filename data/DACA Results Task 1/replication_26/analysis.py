"""
DACA Replication Analysis - Replication 26
Effect of DACA eligibility on full-time employment among Mexican-born Hispanic immigrants

Research Question:
Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on the probability of full-time employment?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DACA REPLICATION ANALYSIS - REPLICATION 26")
print("="*80)
print()

# =============================================================================
# 1. LOAD DATA IN CHUNKS (due to large file size)
# =============================================================================
print("1. LOADING DATA (in chunks to manage memory)...")
print("-"*40)

# Only load needed columns to reduce memory
use_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'MARST',
            'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'UHRSWORK']

# Read in chunks and filter immediately
chunks = []
total_rows = 0
chunk_size = 500000

for chunk in pd.read_csv('data/data.csv', usecols=use_cols, chunksize=chunk_size):
    total_rows += len(chunk)

    # Apply initial filters to reduce size
    # Hispanic-Mexican (HISPAN=1), Born in Mexico (BPL=200), Non-citizen (CITIZEN=3)
    chunk_filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3)
    ]
    if len(chunk_filtered) > 0:
        chunks.append(chunk_filtered)

    print(f"  Processed {total_rows:,} rows, kept {sum(len(c) for c in chunks):,} so far...")

df = pd.concat(chunks, ignore_index=True)
del chunks

print(f"\nTotal observations in raw data: {total_rows:,}")
print(f"After initial filtering (Hispanic-Mexican, Mexico-born, Non-citizen): {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print()

# =============================================================================
# 2. FURTHER SAMPLE SELECTION
# =============================================================================
print("2. FURTHER SAMPLE SELECTION...")
print("-"*40)

# Working age (16-65)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 65)]
print(f"After selecting working age 16-65: {len(df):,}")

# Exclude 2012 (DACA implemented mid-year)
df = df[df['YEAR'] != 2012]
print(f"After excluding 2012: {len(df):,}")

# Valid immigration year (needed for eligibility determination)
df = df[df['YRIMMIG'] > 0]
print(f"After requiring valid YRIMMIG: {len(df):,}")

print()

# =============================================================================
# 3. CONSTRUCT VARIABLES
# =============================================================================
print("3. CONSTRUCTING VARIABLES...")
print("-"*40)

# Calculate age at immigration
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA eligibility criteria:
# 1. Arrived before age 16
# 2. Born after June 15, 1981 (under 31 on June 15, 2012) - use BIRTHYR >= 1982
# 3. Continuous residence since 2007 (YRIMMIG <= 2007)

df['arrived_under_16'] = (df['age_at_immigration'] < 16).astype(int)
df['young_enough'] = (df['BIRTHYR'] >= 1982).astype(int)
df['resident_since_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# DACA eligible = meets all three criteria
df['daca_eligible'] = (
    (df['arrived_under_16'] == 1) &
    (df['young_enough'] == 1) &
    (df['resident_since_2007'] == 1)
).astype(int)

print(f"DACA Eligibility Components:")
print(f"  - Arrived under age 16: {df['arrived_under_16'].sum():,} ({100*df['arrived_under_16'].mean():.1f}%)")
print(f"  - Young enough (born 1982+): {df['young_enough'].sum():,} ({100*df['young_enough'].mean():.1f}%)")
print(f"  - Resident since 2007: {df['resident_since_2007'].sum():,} ({100*df['resident_since_2007'].mean():.1f}%)")
print(f"  - DACA eligible (all criteria): {df['daca_eligible'].sum():,} ({100*df['daca_eligible'].mean():.1f}%)")
print()

# Post-DACA indicator (2013 onwards)
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"Pre-period (2006-2011) observations: {(df['post']==0).sum():,}")
print(f"Post-period (2013-2016) observations: {(df['post']==1).sum():,}")
print()

# Outcome: Full-time employment (35+ hours/week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
print(f"Full-time employed (35+ hours): {df['fulltime'].sum():,} ({100*df['fulltime'].mean():.1f}%)")
print()

# DiD interaction term
df['eligible_post'] = df['daca_eligible'] * df['post']

# Control variables
# Education categories
df['educ_lt_hs'] = (df['EDUC'] < 6).astype(int)
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_some_college'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)
df['educ_college_plus'] = (df['EDUC'] >= 10).astype(int)

# Gender (SEX: 1=male, 2=female)
df['female'] = (df['SEX'] == 2).astype(int)

# Marital status (married vs not)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Age squared
df['age_sq'] = df['AGE'] ** 2

print("Control variables constructed:")
print(f"  - Female: {100*df['female'].mean():.1f}%")
print(f"  - Married: {100*df['married'].mean():.1f}%")
print(f"  - Education < HS: {100*df['educ_lt_hs'].mean():.1f}%")
print(f"  - Education HS: {100*df['educ_hs'].mean():.1f}%")
print(f"  - Education Some College: {100*df['educ_some_college'].mean():.1f}%")
print(f"  - Education College+: {100*df['educ_college_plus'].mean():.1f}%")
print()

# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================
print("4. DESCRIPTIVE STATISTICS...")
print("-"*40)

# Summary by eligibility status
print("\n=== Summary Statistics by DACA Eligibility ===")
for eligible in [0, 1]:
    subset = df[df['daca_eligible'] == eligible]
    label = "DACA Eligible" if eligible else "Not Eligible"
    print(f"\n{label} (N={len(subset):,}):")
    print(f"  Mean age: {subset['AGE'].mean():.1f}")
    print(f"  Female: {100*subset['female'].mean():.1f}%")
    print(f"  Married: {100*subset['married'].mean():.1f}%")
    print(f"  Full-time employed: {100*subset['fulltime'].mean():.1f}%")
    print(f"  Mean usual hours worked: {subset['UHRSWORK'].mean():.1f}")

# Pre/post comparison by eligibility
print("\n=== Full-Time Employment by Period and Eligibility ===")
print("\n                           Pre-DACA    Post-DACA    Difference")
print("-"*60)

for eligible in [0, 1]:
    label = "DACA Eligible    " if eligible else "Not Eligible     "
    pre = df[(df['daca_eligible'] == eligible) & (df['post'] == 0)]
    post = df[(df['daca_eligible'] == eligible) & (df['post'] == 1)]
    pre_mean = pre['fulltime'].mean()
    post_mean = post['fulltime'].mean()
    diff = post_mean - pre_mean
    print(f"{label}  {100*pre_mean:.2f}%       {100*post_mean:.2f}%       {100*diff:+.2f}pp")

# Calculate raw DiD
pre_eligible = df[(df['daca_eligible']==1) & (df['post']==0)]['fulltime'].mean()
post_eligible = df[(df['daca_eligible']==1) & (df['post']==1)]['fulltime'].mean()
pre_inelig = df[(df['daca_eligible']==0) & (df['post']==0)]['fulltime'].mean()
post_inelig = df[(df['daca_eligible']==0) & (df['post']==1)]['fulltime'].mean()

raw_did = (post_eligible - pre_eligible) - (post_inelig - pre_inelig)
print(f"\nRaw Difference-in-Differences: {100*raw_did:+.2f} percentage points")
print()

# =============================================================================
# 5. REGRESSION ANALYSIS
# =============================================================================
print("5. REGRESSION ANALYSIS...")
print("-"*40)

# Prepare data for regression
df_reg = df.dropna(subset=['fulltime', 'daca_eligible', 'post', 'AGE', 'female', 'married',
                            'EDUC', 'STATEFIP', 'PERWT'])
print(f"Observations for regression: {len(df_reg):,}")

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.ols('fulltime ~ daca_eligible + post + eligible_post', data=df_reg).fit()
print(f"DiD Coefficient: {model1.params['eligible_post']:.4f}")
print(f"Std Error: {model1.bse['eligible_post']:.4f}")
print(f"t-statistic: {model1.tvalues['eligible_post']:.4f}")
print(f"p-value: {model1.pvalues['eligible_post']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['eligible_post', 0]:.4f}, {model1.conf_int().loc['eligible_post', 1]:.4f}]")
print(f"R-squared: {model1.rsquared:.4f}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
model2 = smf.ols('fulltime ~ daca_eligible + post + eligible_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                 data=df_reg).fit()
print(f"DiD Coefficient: {model2.params['eligible_post']:.4f}")
print(f"Std Error: {model2.bse['eligible_post']:.4f}")
print(f"t-statistic: {model2.tvalues['eligible_post']:.4f}")
print(f"p-value: {model2.pvalues['eligible_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['eligible_post', 0]:.4f}, {model2.conf_int().loc['eligible_post', 1]:.4f}]")
print(f"R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with state and year fixed effects
print("\n--- Model 3: DiD with demographic controls + State + Year FE ---")
df_reg['state_fe'] = df_reg['STATEFIP'].astype(str)
df_reg['year_fe'] = df_reg['YEAR'].astype(str)

model3 = smf.ols('fulltime ~ daca_eligible + eligible_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(state_fe) + C(year_fe)',
                 data=df_reg).fit()
print(f"DiD Coefficient: {model3.params['eligible_post']:.4f}")
print(f"Std Error: {model3.bse['eligible_post']:.4f}")
print(f"t-statistic: {model3.tvalues['eligible_post']:.4f}")
print(f"p-value: {model3.pvalues['eligible_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['eligible_post', 0]:.4f}, {model3.conf_int().loc['eligible_post', 1]:.4f}]")
print(f"R-squared: {model3.rsquared:.4f}")

# Model 4: Weighted regression with PERWT
print("\n--- Model 4: Weighted DiD with all controls (PREFERRED SPECIFICATION) ---")

# Prepare weighted regression
y = df_reg['fulltime'].values.astype(float)
X = df_reg[['daca_eligible', 'eligible_post', 'AGE', 'age_sq', 'female', 'married',
            'educ_hs', 'educ_some_college', 'educ_college_plus']].values.astype(float)

# Add state and year dummies
state_dummies = pd.get_dummies(df_reg['state_fe'], prefix='state', drop_first=True).values.astype(float)
year_dummies = pd.get_dummies(df_reg['year_fe'], prefix='year', drop_first=True).values.astype(float)
X = np.hstack([X, state_dummies, year_dummies])
X = sm.add_constant(X)

# Get column names for later reference
base_cols = ['const', 'daca_eligible', 'eligible_post', 'AGE', 'age_sq', 'female', 'married',
             'educ_hs', 'educ_some_college', 'educ_college_plus']
state_cols = list(pd.get_dummies(df_reg['state_fe'], prefix='state', drop_first=True).columns)
year_cols = list(pd.get_dummies(df_reg['year_fe'], prefix='year', drop_first=True).columns)
all_cols = base_cols + state_cols + year_cols

# Weighted least squares
weights = df_reg['PERWT'].values.astype(float)
model4 = sm.WLS(y, X, weights=weights).fit()

# Index for eligible_post coefficient (position 2 in the matrix)
idx_eligible_post = 2  # eligible_post is at index 2 (after const=0, daca_eligible=1)

print(f"DiD Coefficient: {model4.params[idx_eligible_post]:.4f}")
print(f"Std Error: {model4.bse[idx_eligible_post]:.4f}")
print(f"t-statistic: {model4.tvalues[idx_eligible_post]:.4f}")
print(f"p-value: {model4.pvalues[idx_eligible_post]:.4f}")
print(f"95% CI: [{model4.conf_int()[idx_eligible_post, 0]:.4f}, {model4.conf_int()[idx_eligible_post, 1]:.4f}]")
print(f"R-squared: {model4.rsquared:.4f}")

# =============================================================================
# 6. ROBUST STANDARD ERRORS
# =============================================================================
print("\n6. ROBUST STANDARD ERRORS...")
print("-"*40)

# Model with heteroskedasticity-robust standard errors (HC1)
print("\n--- Preferred Model with Robust (HC1) Standard Errors ---")
model4_robust = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')
print(f"DiD Coefficient: {model4_robust.params[idx_eligible_post]:.4f}")
print(f"Robust Std Error: {model4_robust.bse[idx_eligible_post]:.4f}")
print(f"t-statistic: {model4_robust.tvalues[idx_eligible_post]:.4f}")
print(f"p-value: {model4_robust.pvalues[idx_eligible_post]:.4f}")
print(f"95% CI: [{model4_robust.conf_int()[idx_eligible_post, 0]:.4f}, {model4_robust.conf_int()[idx_eligible_post, 1]:.4f}]")

# =============================================================================
# 7. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

effect = model4_robust.params[idx_eligible_post]
se = model4_robust.bse[idx_eligible_post]
ci_low = model4_robust.conf_int()[idx_eligible_post, 0]
ci_high = model4_robust.conf_int()[idx_eligible_post, 1]
tstat = model4_robust.tvalues[idx_eligible_post]
pval = model4_robust.pvalues[idx_eligible_post]
n = len(df_reg)
rsq = model4_robust.rsquared

print(f"""
PREFERRED ESTIMATE (Model 4 with robust SE):
============================================
Effect Size: {effect:.4f} ({100*effect:.2f} percentage points)
Standard Error: {se:.4f}
95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]
t-statistic: {tstat:.4f}
p-value: {pval:.4f}
Sample Size: {n:,}
R-squared: {rsq:.4f}

Interpretation:
DACA eligibility is associated with a {100*effect:.2f} percentage point
{"increase" if effect > 0 else "decrease"} in the probability of full-time employment
among Mexican-born Hispanic non-citizens, relative to the control group.
This effect is {"statistically significant" if pval < 0.05 else "not statistically significant"} at the 5% level.
""")

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
print("\n8. SAVING RESULTS...")
print("-"*40)

# Save key results to file
results = {
    'Model': ['Basic DiD', 'With Demographics', 'With State/Year FE', 'Weighted + Robust SE (Preferred)'],
    'DiD_Coefficient': [model1.params['eligible_post'], model2.params['eligible_post'],
                        model3.params['eligible_post'], model4_robust.params[idx_eligible_post]],
    'Std_Error': [model1.bse['eligible_post'], model2.bse['eligible_post'],
                  model3.bse['eligible_post'], model4_robust.bse[idx_eligible_post]],
    'p_value': [model1.pvalues['eligible_post'], model2.pvalues['eligible_post'],
                model3.pvalues['eligible_post'], model4_robust.pvalues[idx_eligible_post]],
    'R_squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4_robust.rsquared]
}
results_df = pd.DataFrame(results)
results_df.to_csv('regression_results.csv', index=False)
print("Saved regression results to: regression_results.csv")

# Save summary statistics
summary_stats = {
    'Statistic': ['Total Sample Size', 'DACA Eligible', 'Not Eligible',
                  'Pre-Period Obs', 'Post-Period Obs',
                  'Mean Full-Time Rate', 'Mean Age', 'Female Pct'],
    'Value': [n, df_reg['daca_eligible'].sum(), n - df_reg['daca_eligible'].sum(),
              (df_reg['post']==0).sum(), (df_reg['post']==1).sum(),
              df_reg['fulltime'].mean(), df_reg['AGE'].mean(), df_reg['female'].mean()]
}
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_statistics.csv', index=False)
print("Saved summary statistics to: summary_statistics.csv")

# Save detailed model output for preferred specification
with open('preferred_model_output.txt', 'w') as f:
    f.write("PREFERRED MODEL OUTPUT (Weighted DiD with Robust SE)\n")
    f.write("="*70 + "\n\n")
    f.write(f"DiD Coefficient (eligible_post): {effect:.6f}\n")
    f.write(f"Robust Standard Error: {se:.6f}\n")
    f.write(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]\n")
    f.write(f"t-statistic: {tstat:.4f}\n")
    f.write(f"p-value: {pval:.6f}\n")
    f.write(f"Sample Size: {n:,}\n")
    f.write(f"R-squared: {rsq:.4f}\n\n")
    f.write("Full model coefficients (main vars only):\n")
    main_vars = ['const', 'daca_eligible', 'eligible_post', 'AGE', 'age_sq', 'female', 'married',
                'educ_hs', 'educ_some_college', 'educ_college_plus']
    for i, var in enumerate(main_vars):
        if i < len(model4_robust.params):
            f.write(f"  {var}: {model4_robust.params[i]:.6f} (SE: {model4_robust.bse[i]:.6f})\n")
print("Saved detailed model output to: preferred_model_output.txt")

# =============================================================================
# 9. YEARLY TRENDS
# =============================================================================
print("\n9. ADDITIONAL ANALYSES...")
print("-"*40)

# Yearly trends
print("\n--- Full-Time Employment by Year and Eligibility ---")
yearly = df.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
yearly.columns = ['Not Eligible', 'DACA Eligible']
print(yearly.round(4))
yearly.to_csv('yearly_trends.csv')
print("Saved yearly trends to: yearly_trends.csv")

# =============================================================================
# 10. SUBGROUP ANALYSIS
# =============================================================================
print("\n10. SUBGROUP ANALYSIS...")
print("-"*40)

# By gender
print("\n--- By Gender ---")
subgroup_results = []

for sex in [0, 1]:
    sex_label = "Female" if sex == 1 else "Male"
    df_sub = df_reg[df_reg['female'] == sex].copy()

    y_sub = df_sub['fulltime'].values.astype(float)
    X_sub = df_sub[['daca_eligible', 'eligible_post', 'AGE', 'age_sq', 'married',
                    'educ_hs', 'educ_some_college', 'educ_college_plus']].values.astype(float)

    state_dummies_sub = pd.get_dummies(df_sub['state_fe'], prefix='state', drop_first=True).values.astype(float)
    year_dummies_sub = pd.get_dummies(df_sub['year_fe'], prefix='year', drop_first=True).values.astype(float)
    X_sub = np.hstack([X_sub, state_dummies_sub, year_dummies_sub])
    X_sub = sm.add_constant(X_sub)
    weights_sub = df_sub['PERWT'].values.astype(float)

    model_sub = sm.WLS(y_sub, X_sub, weights=weights_sub).fit(cov_type='HC1')
    # eligible_post is at index 2 (const=0, daca_eligible=1, eligible_post=2)
    coef = model_sub.params[2]
    se_sub = model_sub.bse[2]
    pval_sub = model_sub.pvalues[2]
    print(f"{sex_label}: DiD = {coef:.4f} (SE: {se_sub:.4f}, p={pval_sub:.4f})")
    subgroup_results.append({'Group': sex_label, 'DiD': coef, 'SE': se_sub, 'p_value': pval_sub})

# By age group
print("\n--- By Age Group ---")
df_reg['age_group'] = pd.cut(df_reg['AGE'], bins=[15, 25, 35, 45, 65], labels=['16-25', '26-35', '36-45', '46-65'])

for age_grp in ['16-25', '26-35', '36-45', '46-65']:
    df_sub = df_reg[df_reg['age_group'] == age_grp].copy()
    if len(df_sub) < 100:
        continue

    y_sub = df_sub['fulltime'].values.astype(float)
    X_sub = df_sub[['daca_eligible', 'eligible_post', 'AGE', 'age_sq', 'female', 'married',
                    'educ_hs', 'educ_some_college', 'educ_college_plus']].values.astype(float)

    state_dummies_sub = pd.get_dummies(df_sub['state_fe'], prefix='state', drop_first=True).values.astype(float)
    year_dummies_sub = pd.get_dummies(df_sub['year_fe'], prefix='year', drop_first=True).values.astype(float)
    X_sub = np.hstack([X_sub, state_dummies_sub, year_dummies_sub])
    X_sub = sm.add_constant(X_sub)
    weights_sub = df_sub['PERWT'].values.astype(float)

    try:
        model_sub = sm.WLS(y_sub, X_sub, weights=weights_sub).fit(cov_type='HC1')
        # eligible_post is at index 2
        coef = model_sub.params[2]
        se_sub = model_sub.bse[2]
        pval_sub = model_sub.pvalues[2]
        print(f"Age {age_grp}: DiD = {coef:.4f} (SE: {se_sub:.4f}, p={pval_sub:.4f})")
        subgroup_results.append({'Group': f'Age {age_grp}', 'DiD': coef, 'SE': se_sub, 'p_value': pval_sub})
    except:
        print(f"Age {age_grp}: Could not estimate (insufficient data)")

# Save subgroup results
subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv('subgroup_results.csv', index=False)
print("\nSaved subgroup results to: subgroup_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
