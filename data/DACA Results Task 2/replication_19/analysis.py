"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican individuals born in Mexico
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

# Load data in chunks and filter
print("\n[1] Loading and filtering data in chunks...")

# Define columns we need
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'UHRSWORK']

chunks = []
chunk_size = 500000

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size):
    # Filter to Hispanic-Mexican born in Mexico, non-citizen
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3)]
    chunks.append(filtered)
    print(f"  Processed chunk, kept {len(filtered):,} rows")

df_mex = pd.concat(chunks, ignore_index=True)
print(f"\nTotal Hispanic-Mexican, Mexico-born, non-citizen: {len(df_mex):,}")

# ============================================================================
# STEP 1: Define Treatment and Control Groups
# ============================================================================
print("\n" + "=" * 70)
print("[2] DEFINING TREATMENT AND CONTROL GROUPS")
print("=" * 70)

# DACA implemented June 15, 2012
# Treatment: Ages 26-30 on June 15, 2012 -> born 1982-1986
# Control: Ages 31-35 on June 15, 2012 -> born 1977-1981

print(f"\nBirth year distribution in our sample:")
print(df_mex['BIRTHYR'].value_counts().sort_index())

# Filter to birth years 1977-1986
df_analysis = df_mex[(df_mex['BIRTHYR'] >= 1977) & (df_mex['BIRTHYR'] <= 1986)].copy()
print(f"\nAfter filtering to birth years 1977-1986: {len(df_analysis):,}")

# Create treatment indicator
df_analysis['treated'] = (df_analysis['BIRTHYR'] >= 1982).astype(int)
print(f"\nTreatment group (born 1982-1986): {df_analysis['treated'].sum():,}")
print(f"Control group (born 1977-1981): {(df_analysis['treated'] == 0).sum():,}")

# ============================================================================
# STEP 2: Define Pre/Post Period
# ============================================================================
print("\n" + "=" * 70)
print("[3] DEFINING PRE/POST PERIODS")
print("=" * 70)

# Exclude 2012 (treatment year)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"\nAfter excluding 2012: {len(df_analysis):,}")

df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
print(f"Pre-period (2006-2011): {(df_analysis['post'] == 0).sum():,}")
print(f"Post-period (2013-2016): {(df_analysis['post'] == 1).sum():,}")

# ============================================================================
# STEP 3: Additional DACA eligibility criteria
# ============================================================================
print("\n" + "=" * 70)
print("[4] ADDITIONAL DACA ELIGIBILITY FILTERING")
print("=" * 70)

# Must have arrived before age 16
df_analysis['age_at_immig'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']

# Filter to valid YRIMMIG (not 0)
df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0].copy()
print(f"After filtering to valid YRIMMIG: {len(df_analysis):,}")

# Arrived before 16th birthday
df_analysis = df_analysis[df_analysis['age_at_immig'] < 16].copy()
print(f"After filtering to arrived before age 16: {len(df_analysis):,}")

# Must have been present in US since June 15, 2007
# Approximation: YRIMMIG <= 2007
df_analysis = df_analysis[df_analysis['YRIMMIG'] <= 2007].copy()
print(f"After filtering to arrived by 2007: {len(df_analysis):,}")

# ============================================================================
# STEP 4: Define Full-Time Employment Outcome
# ============================================================================
print("\n" + "=" * 70)
print("[5] DEFINING FULL-TIME EMPLOYMENT OUTCOME")
print("=" * 70)

print(f"\nUHRSWORK distribution:")
print(df_analysis['UHRSWORK'].describe())

# Full-time employment: usually working 35+ hours per week
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)
print(f"\nFull-time employment rate: {df_analysis['fulltime'].mean():.4f}")

# ============================================================================
# STEP 5: Summary Statistics
# ============================================================================
print("\n" + "=" * 70)
print("[6] SUMMARY STATISTICS")
print("=" * 70)

print("\nSample by year and treatment group:")
print(df_analysis.groupby(['YEAR', 'treated']).size().unstack())

print("\nFull-time employment rates by year and treatment group:")
ft_rates = df_analysis.groupby(['YEAR', 'treated'])['fulltime'].mean().unstack()
print(ft_rates)

# Weighted means by year
print("\nWeighted full-time employment rates by year and treatment group:")
def weighted_mean(x, w):
    return np.average(x, weights=w)

for year in sorted(df_analysis['YEAR'].unique()):
    for treated in [0, 1]:
        subset = df_analysis[(df_analysis['YEAR'] == year) & (df_analysis['treated'] == treated)]
        if len(subset) > 0:
            wm = weighted_mean(subset['fulltime'], subset['PERWT'])
            print(f"Year {year}, Treated={treated}: {wm:.4f} (n={len(subset):,})")

# Create DiD table
print("\n" + "-" * 50)
print("DIFFERENCE-IN-DIFFERENCES TABLE (Unweighted)")
print("-" * 50)

# Calculate means
pre_control = df_analysis[(df_analysis['post'] == 0) & (df_analysis['treated'] == 0)]['fulltime'].mean()
pre_treat = df_analysis[(df_analysis['post'] == 0) & (df_analysis['treated'] == 1)]['fulltime'].mean()
post_control = df_analysis[(df_analysis['post'] == 1) & (df_analysis['treated'] == 0)]['fulltime'].mean()
post_treat = df_analysis[(df_analysis['post'] == 1) & (df_analysis['treated'] == 1)]['fulltime'].mean()

print(f"\n                      Control (31-35)    Treatment (26-30)    Difference")
print(f"Pre (2006-2011)       {pre_control:.4f}             {pre_treat:.4f}               {pre_treat - pre_control:.4f}")
print(f"Post (2013-2016)      {post_control:.4f}             {post_treat:.4f}               {post_treat - post_control:.4f}")
print(f"Difference            {post_control - pre_control:.4f}             {post_treat - pre_treat:.4f}               {(post_treat - pre_treat) - (post_control - pre_control):.4f}")

did_estimate = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\nDifference-in-Differences estimate (unweighted): {did_estimate:.4f}")

# Weighted DiD table
print("\n" + "-" * 50)
print("DIFFERENCE-IN-DIFFERENCES TABLE (Weighted)")
print("-" * 50)

def calc_weighted_mean(df, post_val, treat_val):
    subset = df[(df['post'] == post_val) & (df['treated'] == treat_val)]
    return np.average(subset['fulltime'], weights=subset['PERWT'])

pre_control_w = calc_weighted_mean(df_analysis, 0, 0)
pre_treat_w = calc_weighted_mean(df_analysis, 0, 1)
post_control_w = calc_weighted_mean(df_analysis, 1, 0)
post_treat_w = calc_weighted_mean(df_analysis, 1, 1)

print(f"\n                      Control (31-35)    Treatment (26-30)    Difference")
print(f"Pre (2006-2011)       {pre_control_w:.4f}             {pre_treat_w:.4f}               {pre_treat_w - pre_control_w:.4f}")
print(f"Post (2013-2016)      {post_control_w:.4f}             {post_treat_w:.4f}               {post_treat_w - post_control_w:.4f}")
print(f"Difference            {post_control_w - pre_control_w:.4f}             {post_treat_w - pre_treat_w:.4f}               {(post_treat_w - pre_treat_w) - (post_control_w - pre_control_w):.4f}")

did_estimate_w = (post_treat_w - pre_treat_w) - (post_control_w - pre_control_w)
print(f"\nDifference-in-Differences estimate (weighted): {did_estimate_w:.4f}")

# ============================================================================
# STEP 6: Regression Analysis
# ============================================================================
print("\n" + "=" * 70)
print("[7] REGRESSION ANALYSIS")
print("=" * 70)

# Create interaction term
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

# Basic DiD regression (unweighted)
print("\n[Model 1] Basic DiD (unweighted)")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_analysis).fit()
print(model1.summary().tables[1])

# Weighted DiD regression
print("\n[Model 2] DiD with person weights")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(model2.summary().tables[1])

# Robust standard errors
print("\n[Model 2b] DiD with person weights and robust SE")
model2b = smf.wls('fulltime ~ treated + post + treated_post',
                   data=df_analysis,
                   weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"Treated x Post: {model2b.params['treated_post']:.4f} (Robust SE: {model2b.bse['treated_post']:.4f})")

# Add covariates
print("\n[Model 3] DiD with weights and covariates")
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)
df_analysis['educ_somecol'] = (df_analysis['EDUC'] >= 7).astype(int)
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)

model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(model3.summary().tables[1])

# Add year fixed effects
print("\n[Model 4] DiD with weights, covariates, and year FE")
model4 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(f"Treated x Post: {model4.params['treated_post']:.4f} (SE: {model4.bse['treated_post']:.4f})")

# Add state fixed effects
print("\n[Model 5] Full model: weights, covariates, year FE, state FE")
model5 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(f"Treated x Post: {model5.params['treated_post']:.4f} (SE: {model5.bse['treated_post']:.4f})")
print(f"95% CI: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")
print(f"p-value: {model5.pvalues['treated_post']:.4f}")

# ============================================================================
# STEP 7: Robustness Checks
# ============================================================================
print("\n" + "=" * 70)
print("[8] ROBUSTNESS CHECKS")
print("=" * 70)

# Check 1: By gender
print("\n[Check 1] Heterogeneity by gender")
for sex in [1, 2]:
    sex_label = "Male" if sex == 1 else "Female"
    df_sex = df_analysis[df_analysis['SEX'] == sex].copy()
    model_sex = smf.wls('fulltime ~ treated + post + treated_post',
                         data=df_sex,
                         weights=df_sex['PERWT']).fit()
    print(f"  {sex_label}: {model_sex.params['treated_post']:.4f} (SE: {model_sex.bse['treated_post']:.4f})")

# Check 2: Placebo test (pre-period only)
print("\n[Check 2] Placebo test (2006-2008 vs 2009-2011)")
df_placebo = df_analysis[df_analysis['post'] == 0].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treated_post_placebo'] = df_placebo['treated'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treated + post_placebo + treated_post_placebo',
                         data=df_placebo,
                         weights=df_placebo['PERWT']).fit()
print(f"Placebo DiD estimate: {model_placebo.params['treated_post_placebo']:.4f} (SE: {model_placebo.bse['treated_post_placebo']:.4f})")
print(f"p-value: {model_placebo.pvalues['treated_post_placebo']:.4f}")

# Check 3: Event study
print("\n[Check 3] Event study coefficients (relative to 2011)")
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in years:
    df_analysis[f'treated_{year}'] = (df_analysis['treated'] * (df_analysis['YEAR'] == year)).astype(int)

formula = 'fulltime ~ treated + C(YEAR) + ' + ' + '.join([f'treated_{y}' for y in years])
model_event = smf.wls(formula, data=df_analysis, weights=df_analysis['PERWT']).fit()

print("\nYear | Coefficient | SE | p-value")
print("-" * 50)
event_results = []
for year in years:
    coef = model_event.params[f'treated_{year}']
    se = model_event.bse[f'treated_{year}']
    pval = model_event.pvalues[f'treated_{year}']
    print(f"{year} | {coef:10.4f} | {se:.4f} | {pval:.4f}")
    event_results.append({'year': year, 'coef': coef, 'se': se, 'pval': pval})

# Add 2011 reference year
event_results.append({'year': 2011, 'coef': 0, 'se': 0, 'pval': 1})
event_df = pd.DataFrame(event_results).sort_values('year')
event_df.to_csv('event_study_results.csv', index=False)

# Check 4: Alternative outcome - any employment
print("\n[Check 4] Alternative outcome: Any employment")
df_analysis['employed'] = (df_analysis['UHRSWORK'] > 0).astype(int)
model_emp = smf.wls('employed ~ treated + post + treated_post',
                     data=df_analysis,
                     weights=df_analysis['PERWT']).fit()
print(f"DiD on any employment: {model_emp.params['treated_post']:.4f} (SE: {model_emp.bse['treated_post']:.4f})")

# Check 5: Intensive margin - hours worked conditional on employment
print("\n[Check 5] Intensive margin: Hours among employed")
df_employed = df_analysis[df_analysis['UHRSWORK'] > 0].copy()
model_hours = smf.wls('UHRSWORK ~ treated + post + treated_post',
                       data=df_employed,
                       weights=df_employed['PERWT']).fit()
print(f"DiD on hours (employed only): {model_hours.params['treated_post']:.4f} (SE: {model_hours.bse['treated_post']:.4f})")

# ============================================================================
# STEP 8: Final Results Summary
# ============================================================================
print("\n" + "=" * 70)
print("[9] FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"\nPreferred Model: DiD with person weights (Model 2)")
print(f"Sample size: {len(df_analysis):,}")
print(f"\nDifference-in-Differences Estimate: {model2.params['treated_post']:.4f}")
print(f"Standard Error: {model2.bse['treated_post']:.4f}")
print(f"95% Confidence Interval: [{model2.conf_int().loc['treated_post', 0]:.4f}, {model2.conf_int().loc['treated_post', 1]:.4f}]")
print(f"t-statistic: {model2.tvalues['treated_post']:.4f}")
print(f"p-value: {model2.pvalues['treated_post']:.4f}")

# Sample sizes
n_by_group = df_analysis.groupby(['post', 'treated']).size()
print(f"\nSample sizes by group:")
print(f"  Pre-period, Control: {n_by_group[(0, 0)]:,}")
print(f"  Pre-period, Treatment: {n_by_group[(0, 1)]:,}")
print(f"  Post-period, Control: {n_by_group[(1, 0)]:,}")
print(f"  Post-period, Treatment: {n_by_group[(1, 1)]:,}")

# Save summary stats
summary_stats = df_analysis.groupby(['post', 'treated']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
}).round(4)
summary_stats.to_csv('summary_stats.csv')

# Save key results to a file
with open('key_results.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - KEY RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Sample size: {len(df_analysis):,}\n")
    f.write(f"DiD Estimate: {model2.params['treated_post']:.4f}\n")
    f.write(f"Standard Error: {model2.bse['treated_post']:.4f}\n")
    f.write(f"95% CI: [{model2.conf_int().loc['treated_post', 0]:.4f}, {model2.conf_int().loc['treated_post', 1]:.4f}]\n")
    f.write(f"t-statistic: {model2.tvalues['treated_post']:.4f}\n")
    f.write(f"p-value: {model2.pvalues['treated_post']:.4f}\n")
    f.write(f"\nDiD Table (Weighted):\n")
    f.write(f"Pre-Control: {pre_control_w:.4f}\n")
    f.write(f"Pre-Treatment: {pre_treat_w:.4f}\n")
    f.write(f"Post-Control: {post_control_w:.4f}\n")
    f.write(f"Post-Treatment: {post_treat_w:.4f}\n")
    f.write(f"\nModel with covariates:\n")
    f.write(f"DiD Estimate: {model3.params['treated_post']:.4f}\n")
    f.write(f"SE: {model3.bse['treated_post']:.4f}\n")
    f.write(f"\nFull model (year + state FE):\n")
    f.write(f"DiD Estimate: {model5.params['treated_post']:.4f}\n")
    f.write(f"SE: {model5.bse['treated_post']:.4f}\n")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# Print full model summaries
print("\n\n" + "=" * 70)
print("FULL MODEL SUMMARIES")
print("=" * 70)

print("\nMODEL 1: Basic DiD (unweighted)")
print(model1.summary())

print("\n" + "=" * 70)
print("MODEL 2: Weighted DiD (PREFERRED)")
print(model2.summary())

print("\n" + "=" * 70)
print("MODEL 3: With Covariates")
print(model3.summary())

# Covariate balance table
print("\n" + "=" * 70)
print("COVARIATE BALANCE TABLE")
print("=" * 70)

covariates = ['AGE', 'female', 'married', 'educ_hs', 'UHRSWORK']
print("\nPre-period means:")
print(df_analysis[df_analysis['post']==0].groupby('treated')[covariates].mean())

print("\nPost-period means:")
print(df_analysis[df_analysis['post']==1].groupby('treated')[covariates].mean())
