"""
DACA Replication Study: Impact on Full-Time Employment
Analysis Script

Research Question: Among ethnically Hispanic-Mexican Mexican-born people in the US,
what was the causal impact of DACA eligibility on full-time employment (35+ hrs/week)?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DACA REPLICATION STUDY: IMPACT ON FULL-TIME EMPLOYMENT")
print("="*70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[1] Loading data...")
data_path = "data/data.csv"

# Load data in chunks to manage memory
chunks = []
chunksize = 1000000
for chunk in pd.read_csv(data_path, chunksize=chunksize, low_memory=False):
    # Initial filter: Hispanic-Mexican (HISPAN==1), Born in Mexico (BPL==200)
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f"After Hispanic-Mexican + Mexico-born filter: {len(df):,} observations")

# =============================================================================
# STEP 2: APPLY SAMPLE RESTRICTIONS
# =============================================================================
print("\n[2] Applying sample restrictions...")

# Working age (16-64)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)]
print(f"After age 16-64 filter: {len(df):,}")

# Non-citizens only (CITIZEN == 3)
df = df[df['CITIZEN'] == 3]
print(f"After non-citizen filter: {len(df):,}")

# Exclude 2012 (transitional year)
df = df[df['YEAR'] != 2012]
print(f"After excluding 2012: {len(df):,}")

# =============================================================================
# STEP 3: CREATE ANALYSIS VARIABLES
# =============================================================================
print("\n[3] Creating analysis variables...")

# Calculate age at immigration
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA Eligibility Criteria:
# 1. Arrived before age 16: age_at_immigration < 16
# 2. Born after June 15, 1981 (under 31 as of June 2012): BIRTHYR >= 1982 (conservative)
# 3. In US since June 2007: YRIMMIG <= 2007
# 4. Non-citizen (already filtered)

df['daca_eligible'] = (
    (df['age_at_immigration'] < 16) &
    (df['age_at_immigration'] >= 0) &  # Valid immigration age
    (df['BIRTHYR'] >= 1982) &
    (df['YRIMMIG'] <= 2007) &
    (df['YRIMMIG'] > 0)  # Valid immigration year
).astype(int)

# Post-treatment indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term
df['daca_post'] = df['daca_eligible'] * df['post']

# Full-time employment outcome (35+ hours per week)
# UHRSWORK: usual hours worked per week, 0 means N/A or not working
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Any employment (for robustness)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

print(f"DACA eligible: {df['daca_eligible'].sum():,} ({df['daca_eligible'].mean()*100:.1f}%)")
print(f"Post-treatment observations: {df['post'].sum():,} ({df['post'].mean()*100:.1f}%)")
print(f"Full-time employment rate: {df['fulltime'].mean()*100:.1f}%")

# =============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*70)
print("[4] DESCRIPTIVE STATISTICS")
print("="*70)

# Summary by eligibility and period
summary = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'SEX': lambda x: (x == 2).mean(),  # Female proportion
    'PERWT': 'sum'
}).round(3)

print("\nSummary Statistics by Eligibility and Period:")
print(summary)

# Simple 2x2 DiD calculation
ft_elig_pre = df[(df['daca_eligible']==1) & (df['post']==0)]['fulltime'].mean()
ft_elig_post = df[(df['daca_eligible']==1) & (df['post']==1)]['fulltime'].mean()
ft_ctrl_pre = df[(df['daca_eligible']==0) & (df['post']==0)]['fulltime'].mean()
ft_ctrl_post = df[(df['daca_eligible']==0) & (df['post']==1)]['fulltime'].mean()

print("\n" + "-"*50)
print("2x2 Difference-in-Differences Table:")
print("-"*50)
print(f"                    Pre-DACA    Post-DACA    Diff")
print(f"DACA Eligible:      {ft_elig_pre:.4f}      {ft_elig_post:.4f}       {ft_elig_post-ft_elig_pre:+.4f}")
print(f"Control Group:      {ft_ctrl_pre:.4f}      {ft_ctrl_post:.4f}       {ft_ctrl_post-ft_ctrl_pre:+.4f}")
print(f"-"*50)
did_simple = (ft_elig_post - ft_elig_pre) - (ft_ctrl_post - ft_ctrl_pre)
print(f"DiD Estimate:       {did_simple:+.4f}")
print("-"*50)

# Sample sizes
print("\nSample Sizes:")
print(f"  DACA Eligible, Pre:  {len(df[(df['daca_eligible']==1) & (df['post']==0)]):,}")
print(f"  DACA Eligible, Post: {len(df[(df['daca_eligible']==1) & (df['post']==1)]):,}")
print(f"  Control, Pre:        {len(df[(df['daca_eligible']==0) & (df['post']==0)]):,}")
print(f"  Control, Post:       {len(df[(df['daca_eligible']==0) & (df['post']==1)]):,}")
print(f"  Total:               {len(df):,}")

# =============================================================================
# STEP 5: REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("[5] REGRESSION ANALYSIS")
print("="*70)

# Prepare control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_cat'] = pd.cut(df['EDUC'],
                        bins=[-1, 2, 6, 7, 10, 11],
                        labels=['less_hs', 'some_hs', 'hs_grad', 'some_college', 'college_plus'])
df = pd.get_dummies(df, columns=['educ_cat'], drop_first=True, dtype=int)

# Age squared for non-linearity
df['age_sq'] = df['AGE'] ** 2

# Year dummies
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

# State dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True, dtype=int)
df = pd.concat([df, state_dummies], axis=1)

# ---- Model 1: Basic DiD ----
print("\n--- Model 1: Basic DiD (No Controls) ---")
X1 = df[['daca_eligible', 'post', 'daca_post']]
X1 = sm.add_constant(X1)
y = df['fulltime']

model1 = sm.OLS(y, X1).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model1.summary().tables[1])

# ---- Model 2: With Demographic Controls ----
print("\n--- Model 2: With Demographic Controls ---")
control_vars = ['daca_eligible', 'post', 'daca_post', 'AGE', 'age_sq', 'female', 'married']
educ_cols = [c for c in df.columns if c.startswith('educ_cat_')]
X2 = df[control_vars + educ_cols]
X2 = sm.add_constant(X2)

model2 = sm.OLS(y, X2).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model2.summary().tables[1])

# ---- Model 3: With Year Fixed Effects ----
print("\n--- Model 3: With Year Fixed Effects ---")
year_cols = [c for c in df.columns if c.startswith('year_')]
# Remove 'post' since year FE capture it
control_vars3 = ['daca_eligible', 'daca_post', 'AGE', 'age_sq', 'female', 'married']
X3 = df[control_vars3 + educ_cols + year_cols]
X3 = sm.add_constant(X3)

model3 = sm.OLS(y, X3).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print("\nKey coefficients:")
print(f"  daca_eligible: {model3.params['daca_eligible']:.4f} (SE: {model3.bse['daca_eligible']:.4f})")
print(f"  daca_post (DiD): {model3.params['daca_post']:.4f} (SE: {model3.bse['daca_post']:.4f})")

# ---- Model 4: Full Model with State FE ----
print("\n--- Model 4: Full Model (Year + State FE) ---")
state_cols = [c for c in df.columns if c.startswith('state_')]
X4 = df[control_vars3 + educ_cols + year_cols + state_cols]
X4 = sm.add_constant(X4)

model4 = sm.OLS(y, X4).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print("\nKey coefficients:")
print(f"  daca_eligible: {model4.params['daca_eligible']:.4f} (SE: {model4.bse['daca_eligible']:.4f})")
print(f"  daca_post (DiD): {model4.params['daca_post']:.4f} (SE: {model4.bse['daca_post']:.4f})")

# =============================================================================
# STEP 6: PREFERRED SPECIFICATION DETAILS
# =============================================================================
print("\n" + "="*70)
print("[6] PREFERRED SPECIFICATION (Model 4)")
print("="*70)

preferred_coef = model4.params['daca_post']
preferred_se = model4.bse['daca_post']
preferred_ci_low = model4.conf_int().loc['daca_post', 0]
preferred_ci_high = model4.conf_int().loc['daca_post', 1]
preferred_pval = model4.pvalues['daca_post']
n_obs = int(model4.nobs)

print(f"\nPreferred DiD Estimate:")
print(f"  Effect Size: {preferred_coef:.4f}")
print(f"  Standard Error: {preferred_se:.4f}")
print(f"  95% CI: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"  P-value: {preferred_pval:.4f}")
print(f"  Sample Size: {n_obs:,}")
print(f"  R-squared: {model4.rsquared:.4f}")

# Interpretation
print(f"\nInterpretation:")
if preferred_pval < 0.05:
    direction = "increase" if preferred_coef > 0 else "decrease"
    print(f"  DACA eligibility is associated with a statistically significant")
    print(f"  {abs(preferred_coef)*100:.2f} percentage point {direction} in the")
    print(f"  probability of full-time employment.")
else:
    print(f"  The effect of DACA eligibility on full-time employment is")
    print(f"  not statistically significant at the 5% level.")

# =============================================================================
# STEP 7: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*70)
print("[7] ROBUSTNESS CHECKS")
print("="*70)

# ---- 7.1: Alternative outcome - Any Employment ----
print("\n--- 7.1: Alternative Outcome (Any Employment) ---")
y_emp = df['employed']
model_emp = sm.OLS(y_emp, X4).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  daca_post effect on employment: {model_emp.params['daca_post']:.4f}")
print(f"  SE: {model_emp.bse['daca_post']:.4f}, p-value: {model_emp.pvalues['daca_post']:.4f}")

# ---- 7.2: Subgroup - Males Only ----
print("\n--- 7.2: Subgroup Analysis (Males Only) ---")
df_male = df[df['female'] == 0]
X_male = df_male[control_vars3 + educ_cols + year_cols + state_cols]
X_male = sm.add_constant(X_male)
y_male = df_male['fulltime']
model_male = sm.OLS(y_male, X_male).fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
print(f"  daca_post effect (males): {model_male.params['daca_post']:.4f}")
print(f"  SE: {model_male.bse['daca_post']:.4f}, p-value: {model_male.pvalues['daca_post']:.4f}")

# ---- 7.3: Subgroup - Females Only ----
print("\n--- 7.3: Subgroup Analysis (Females Only) ---")
df_female = df[df['female'] == 1]
X_female = df_female[control_vars3 + educ_cols + year_cols + state_cols]
X_female = sm.add_constant(X_female)
y_female = df_female['fulltime']
model_female = sm.OLS(y_female, X_female).fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})
print(f"  daca_post effect (females): {model_female.params['daca_post']:.4f}")
print(f"  SE: {model_female.bse['daca_post']:.4f}, p-value: {model_female.pvalues['daca_post']:.4f}")

# ---- 7.4: Placebo Test (Pre-treatment trends) ----
print("\n--- 7.4: Placebo Test (Pre-period only: 2006-2011) ---")
df_pre = df[df['post'] == 0].copy()
# Create fake post for 2010-2011 (as if treatment happened in 2010)
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['placebo_interaction'] = df_pre['daca_eligible'] * df_pre['placebo_post']

placebo_vars = ['daca_eligible', 'placebo_post', 'placebo_interaction', 'AGE', 'age_sq', 'female', 'married']
year_cols_pre = [c for c in df_pre.columns if c.startswith('year_') and int(c.split('_')[1]) < 2013]
state_cols_pre = [c for c in df_pre.columns if c.startswith('state_')]
X_placebo = df_pre[placebo_vars + educ_cols + state_cols_pre]
X_placebo = sm.add_constant(X_placebo)
y_placebo = df_pre['fulltime']

model_placebo = sm.OLS(y_placebo, X_placebo).fit(cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})
print(f"  Placebo interaction effect: {model_placebo.params['placebo_interaction']:.4f}")
print(f"  SE: {model_placebo.bse['placebo_interaction']:.4f}, p-value: {model_placebo.pvalues['placebo_interaction']:.4f}")
if model_placebo.pvalues['placebo_interaction'] > 0.1:
    print("  [Supports parallel trends assumption - no significant pre-trend]")
else:
    print("  [Warning: Potential pre-trend detected]")

# ---- 7.5: Alternative age cutoff ----
print("\n--- 7.5: Robustness - Birth Year >= 1983 (stricter age criterion) ---")
df_strict = df[df['BIRTHYR'] >= 1983].copy()
df_strict_ctrl = df[(df['BIRTHYR'] >= 1983) | (df['daca_eligible'] == 0)].copy()

# Recalculate eligibility with stricter criterion
df['daca_eligible_strict'] = (
    (df['age_at_immigration'] < 16) &
    (df['age_at_immigration'] >= 0) &
    (df['BIRTHYR'] >= 1983) &
    (df['YRIMMIG'] <= 2007) &
    (df['YRIMMIG'] > 0)
).astype(int)
df['daca_post_strict'] = df['daca_eligible_strict'] * df['post']

control_vars_strict = ['daca_eligible_strict', 'daca_post_strict', 'AGE', 'age_sq', 'female', 'married']
X_strict = df[control_vars_strict + educ_cols + year_cols + state_cols]
X_strict = sm.add_constant(X_strict)
model_strict = sm.OLS(y, X_strict).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  daca_post effect (strict): {model_strict.params['daca_post_strict']:.4f}")
print(f"  SE: {model_strict.bse['daca_post_strict']:.4f}, p-value: {model_strict.pvalues['daca_post_strict']:.4f}")

# =============================================================================
# STEP 8: YEAR-BY-YEAR EFFECTS (EVENT STUDY)
# =============================================================================
print("\n" + "="*70)
print("[8] EVENT STUDY: Year-by-Year Effects")
print("="*70)

# Create year interactions (relative to 2011, the last pre-period year)
years_to_interact = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for yr in years_to_interact:
    df[f'elig_x_{yr}'] = df['daca_eligible'] * (df['YEAR'] == yr).astype(int)

event_cols = [f'elig_x_{yr}' for yr in years_to_interact]
control_vars_event = ['daca_eligible', 'AGE', 'age_sq', 'female', 'married']
X_event = df[control_vars_event + educ_cols + year_cols + state_cols + event_cols]
X_event = sm.add_constant(X_event)

model_event = sm.OLS(y, X_event).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("\nYear-by-Year Interaction Coefficients (ref: 2011):")
print("-"*50)
for yr in years_to_interact:
    col = f'elig_x_{yr}'
    coef = model_event.params[col]
    se = model_event.bse[col]
    pval = model_event.pvalues[col]
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
    print(f"  {yr}: {coef:+.4f} (SE: {se:.4f}) {sig}")

# =============================================================================
# STEP 9: SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("[9] SAVING RESULTS")
print("="*70)

# Save key results for LaTeX report
results_dict = {
    'did_simple': did_simple,
    'n_total': len(df),
    'n_eligible': df['daca_eligible'].sum(),
    'n_control': len(df) - df['daca_eligible'].sum(),
    'ft_elig_pre': ft_elig_pre,
    'ft_elig_post': ft_elig_post,
    'ft_ctrl_pre': ft_ctrl_pre,
    'ft_ctrl_post': ft_ctrl_post,
    'preferred_coef': preferred_coef,
    'preferred_se': preferred_se,
    'preferred_ci_low': preferred_ci_low,
    'preferred_ci_high': preferred_ci_high,
    'preferred_pval': preferred_pval,
    'preferred_n': n_obs,
    'preferred_rsq': model4.rsquared,
    'model1_coef': model1.params['daca_post'],
    'model1_se': model1.bse['daca_post'],
    'model2_coef': model2.params['daca_post'],
    'model2_se': model2.bse['daca_post'],
    'model3_coef': model3.params['daca_post'],
    'model3_se': model3.bse['daca_post'],
    'emp_coef': model_emp.params['daca_post'],
    'emp_se': model_emp.bse['daca_post'],
    'male_coef': model_male.params['daca_post'],
    'male_se': model_male.bse['daca_post'],
    'female_coef': model_female.params['daca_post'],
    'female_se': model_female.bse['daca_post'],
    'placebo_coef': model_placebo.params['placebo_interaction'],
    'placebo_pval': model_placebo.pvalues['placebo_interaction'],
}

# Event study results
for yr in years_to_interact:
    col = f'elig_x_{yr}'
    results_dict[f'event_{yr}_coef'] = model_event.params[col]
    results_dict[f'event_{yr}_se'] = model_event.bse[col]

# Save to file
pd.DataFrame([results_dict]).to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Summary statistics table
print("\n--- Summary Statistics for Report ---")
summary_stats = df.groupby('daca_eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'fulltime': 'mean',
    'employed': 'mean',
    'PERWT': 'sum'
}).round(3)
summary_stats.columns = ['Mean Age', 'Female Share', 'Married Share', 'FT Employment Rate', 'Employment Rate', 'Weighted N']
print(summary_stats)
summary_stats.to_csv('summary_statistics.csv')

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nPREFERRED ESTIMATE:")
print(f"  Effect: {preferred_coef:.4f} ({preferred_coef*100:.2f} percentage points)")
print(f"  SE: {preferred_se:.4f}")
print(f"  95% CI: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"  Sample Size: {n_obs:,}")
