"""
DACA and Full-Time Employment Analysis
Replication 43

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for DACA on the
probability of full-time employment (35+ hours/week)?

Analysis Period: Pre-DACA (2006-2011) vs Post-DACA (2013-2016)
Note: 2012 excluded due to mid-year DACA implementation
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("DACA Full-Time Employment Analysis - Replication 43")
print("="*80)

# =============================================================================
# 1. Load Data
# =============================================================================
print("\n[1] Loading data...")
df = pd.read_csv('data/data.csv')
print(f"Total observations in full dataset: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. Filter to Target Population: Hispanic-Mexican + Mexican-Born
# =============================================================================
print("\n[2] Filtering to Hispanic-Mexican, Mexican-born population...")

# HISPAN = 1 (Mexican)
# BPL = 200 (Mexico)
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
print(f"Observations after Hispanic-Mexican + Mexico birthplace filter: {len(df_mex):,}")

# =============================================================================
# 3. Create Key Variables
# =============================================================================
print("\n[3] Creating analysis variables...")

# Age at immigration (using calendar years)
# YRIMMIG = 0 means N/A (native-born or missing)
df_mex['age_at_immig'] = np.where(
    df_mex['YRIMMIG'] > 0,
    df_mex['YRIMMIG'] - df_mex['BIRTHYR'],
    np.nan
)

# Age as of June 15, 2012
df_mex['age_june_2012'] = 2012 - df_mex['BIRTHYR']

# Years in US by 2012
df_mex['years_in_us_2012'] = np.where(
    df_mex['YRIMMIG'] > 0,
    2012 - df_mex['YRIMMIG'],
    np.nan
)

# =============================================================================
# 4. Define DACA Eligibility
# =============================================================================
print("\n[4] Defining DACA eligibility...")

"""
DACA Eligibility Criteria:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012 (born >= 1982)
3. Lived continuously in the US since June 15, 2007 (immigrated <= 2007)
4. Non-citizen (CITIZEN == 3)

Note: We cannot observe lawful/unlawful status directly. Following instructions,
we assume non-citizens without naturalization are potentially undocumented.
"""

df_mex['daca_eligible'] = (
    (df_mex['YRIMMIG'] > 0) &                    # Has valid immigration year
    (df_mex['age_at_immig'] >= 0) &              # Positive age at immigration
    (df_mex['age_at_immig'] < 16) &              # Arrived before 16th birthday
    (df_mex['BIRTHYR'] >= 1982) &                # Under 31 as of June 2012
    (df_mex['YRIMMIG'] <= 2007) &                # In US since at least 2007
    (df_mex['CITIZEN'] == 3)                     # Not a citizen
).astype(int)

# Control group: Similar non-citizens who don't meet DACA age criteria
# Either: arrived at 16+ OR turned 31 before June 2012 (born before 1982)
df_mex['control_group'] = (
    (df_mex['YRIMMIG'] > 0) &                    # Has valid immigration year
    (df_mex['CITIZEN'] == 3) &                   # Not a citizen
    (df_mex['YRIMMIG'] <= 2007) &                # In US since at least 2007 (similar tenure)
    (
        (df_mex['age_at_immig'] >= 16) |         # Arrived at 16 or older OR
        (df_mex['BIRTHYR'] < 1982)               # Too old (31+ in June 2012)
    )
).astype(int)

print(f"DACA-eligible observations: {df_mex['daca_eligible'].sum():,}")
print(f"Control group observations: {df_mex['control_group'].sum():,}")

# =============================================================================
# 5. Define Outcome Variable
# =============================================================================
print("\n[5] Creating outcome variable...")

# Full-time employment: UHRSWORK >= 35
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator
df_mex['employed'] = (df_mex['EMPSTAT'] == 1).astype(int)

print(f"Full-time employment rate (all): {df_mex['fulltime'].mean():.3f}")
print(f"Employment rate (all): {df_mex['employed'].mean():.3f}")

# =============================================================================
# 6. Create Analysis Sample
# =============================================================================
print("\n[6] Creating analysis sample...")

# Restrict to treatment or control group
df_mex['in_sample'] = (df_mex['daca_eligible'] == 1) | (df_mex['control_group'] == 1)

# Exclude 2012 (ambiguous treatment year)
df_mex['not_2012'] = df_mex['YEAR'] != 2012

# Working age (18-64)
df_mex['working_age'] = (df_mex['AGE'] >= 18) & (df_mex['AGE'] <= 64)

# Create analysis sample
df_analysis = df_mex[
    df_mex['in_sample'] &
    df_mex['not_2012'] &
    df_mex['working_age']
].copy()

print(f"Analysis sample size: {len(df_analysis):,}")
print(f"DACA-eligible in analysis sample: {df_analysis['daca_eligible'].sum():,}")
print(f"Control in analysis sample: {df_analysis['control_group'].sum():,}")

# =============================================================================
# 7. Create DiD Variables
# =============================================================================
print("\n[7] Creating difference-in-differences variables...")

# Post-DACA indicator (2013-2016)
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Treatment group indicator (for clarity)
df_analysis['treat'] = df_analysis['daca_eligible']

# DiD interaction term
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

# Year dummies
df_analysis['year'] = df_analysis['YEAR'].astype('category')

# =============================================================================
# 8. Summary Statistics
# =============================================================================
print("\n[8] Summary Statistics")
print("="*80)

# By treatment status
print("\n--- Pre-period (2006-2011) characteristics ---")
pre = df_analysis[df_analysis['post'] == 0]

treat_pre = pre[pre['treat'] == 1]
ctrl_pre = pre[pre['treat'] == 0]

print(f"\nDACA-Eligible (Treatment) - Pre-period:")
print(f"  N: {len(treat_pre):,}")
print(f"  Mean Age: {treat_pre['AGE'].mean():.1f}")
print(f"  % Female: {(treat_pre['SEX'] == 2).mean()*100:.1f}%")
print(f"  % Married: {(treat_pre['MARST'] == 1).mean()*100:.1f}%")
print(f"  Full-time rate: {treat_pre['fulltime'].mean()*100:.1f}%")
print(f"  Employment rate: {treat_pre['employed'].mean()*100:.1f}%")

print(f"\nControl Group - Pre-period:")
print(f"  N: {len(ctrl_pre):,}")
print(f"  Mean Age: {ctrl_pre['AGE'].mean():.1f}")
print(f"  % Female: {(ctrl_pre['SEX'] == 2).mean()*100:.1f}%")
print(f"  % Married: {(ctrl_pre['MARST'] == 1).mean()*100:.1f}%")
print(f"  Full-time rate: {ctrl_pre['fulltime'].mean()*100:.1f}%")
print(f"  Employment rate: {ctrl_pre['employed'].mean()*100:.1f}%")

# Post-period
print("\n--- Post-period (2013-2016) characteristics ---")
post = df_analysis[df_analysis['post'] == 1]

treat_post = post[post['treat'] == 1]
ctrl_post = post[post['treat'] == 0]

print(f"\nDACA-Eligible (Treatment) - Post-period:")
print(f"  N: {len(treat_post):,}")
print(f"  Mean Age: {treat_post['AGE'].mean():.1f}")
print(f"  Full-time rate: {treat_post['fulltime'].mean()*100:.1f}%")
print(f"  Employment rate: {treat_post['employed'].mean()*100:.1f}%")

print(f"\nControl Group - Post-period:")
print(f"  N: {len(ctrl_post):,}")
print(f"  Mean Age: {ctrl_post['AGE'].mean():.1f}")
print(f"  Full-time rate: {ctrl_post['fulltime'].mean()*100:.1f}%")
print(f"  Employment rate: {ctrl_post['employed'].mean()*100:.1f}%")

# =============================================================================
# 9. Simple DiD Calculation
# =============================================================================
print("\n[9] Simple Difference-in-Differences (Unweighted)")
print("="*80)

# Calculate means
y_t1 = treat_post['fulltime'].mean()
y_t0 = treat_pre['fulltime'].mean()
y_c1 = ctrl_post['fulltime'].mean()
y_c0 = ctrl_pre['fulltime'].mean()

did_simple = (y_t1 - y_t0) - (y_c1 - y_c0)

print(f"\nTreatment group:")
print(f"  Post:  {y_t1:.4f}")
print(f"  Pre:   {y_t0:.4f}")
print(f"  Diff:  {y_t1 - y_t0:.4f}")

print(f"\nControl group:")
print(f"  Post:  {y_c1:.4f}")
print(f"  Pre:   {y_c0:.4f}")
print(f"  Diff:  {y_c1 - y_c0:.4f}")

print(f"\nDifference-in-Differences: {did_simple:.4f}")
print(f"  (Percentage points: {did_simple*100:.2f} pp)")

# =============================================================================
# 10. Regression-based DiD
# =============================================================================
print("\n[10] Regression-based Difference-in-Differences")
print("="*80)

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (no controls, unweighted) ---")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit()
print(f"DiD coefficient (treat_post): {model1.params['treat_post']:.4f}")
print(f"Standard Error: {model1.bse['treat_post']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model1.pvalues['treat_post']:.4f}")
print(f"N: {int(model1.nobs):,}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls (unweighted) ---")

# Create control dummies
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

model2 = smf.ols('fulltime ~ treat + post + treat_post + AGE + age_sq + female + married',
                  data=df_analysis).fit()
print(f"DiD coefficient (treat_post): {model2.params['treat_post']:.4f}")
print(f"Standard Error: {model2.bse['treat_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model2.pvalues['treat_post']:.4f}")
print(f"N: {int(model2.nobs):,}")

# Model 3: DiD with demographic controls + education
print("\n--- Model 3: DiD with demographics + education (unweighted) ---")

# Education categories
df_analysis['educ_cat'] = pd.cut(df_analysis['EDUC'],
                                  bins=[-1, 2, 6, 10, 11],
                                  labels=['less_hs', 'hs', 'some_college', 'college'])

model3 = smf.ols('fulltime ~ treat + post + treat_post + AGE + age_sq + female + married + C(educ_cat)',
                  data=df_analysis).fit()
print(f"DiD coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"Standard Error: {model3.bse['treat_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model3.pvalues['treat_post']:.4f}")
print(f"N: {int(model3.nobs):,}")

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with year fixed effects + demographics (unweighted) ---")

model4 = smf.ols('fulltime ~ treat + C(YEAR) + treat_post + AGE + age_sq + female + married + C(educ_cat)',
                  data=df_analysis).fit()
print(f"DiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"Standard Error: {model4.bse['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model4.pvalues['treat_post']:.4f}")
print(f"N: {int(model4.nobs):,}")

# =============================================================================
# 11. Weighted Regressions (using PERWT)
# =============================================================================
print("\n[11] Weighted Difference-in-Differences")
print("="*80)

# Model 5: Basic DiD weighted
print("\n--- Model 5: Basic DiD (weighted by PERWT) ---")
model5 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(f"DiD coefficient (treat_post): {model5.params['treat_post']:.4f}")
print(f"Standard Error: {model5.bse['treat_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model5.pvalues['treat_post']:.4f}")

# Model 6: DiD with controls weighted
print("\n--- Model 6: DiD with controls (weighted by PERWT) ---")
model6 = smf.wls('fulltime ~ treat + post + treat_post + AGE + age_sq + female + married + C(educ_cat)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(f"DiD coefficient (treat_post): {model6.params['treat_post']:.4f}")
print(f"Standard Error: {model6.bse['treat_post']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['treat_post', 0]:.4f}, {model6.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model6.pvalues['treat_post']:.4f}")

# Model 7: Full specification weighted with year FE
print("\n--- Model 7: Full specification with year FE (weighted by PERWT) ---")
model7 = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + AGE + age_sq + female + married + C(educ_cat)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(f"DiD coefficient (treat_post): {model7.params['treat_post']:.4f}")
print(f"Standard Error: {model7.bse['treat_post']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['treat_post', 0]:.4f}, {model7.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model7.pvalues['treat_post']:.4f}")

# =============================================================================
# 12. Robust Standard Errors
# =============================================================================
print("\n[12] Preferred Specification with Robust Standard Errors")
print("="*80)

# Robust standard errors (HC1)
model_robust = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + AGE + age_sq + female + married + C(educ_cat)',
                        data=df_analysis,
                        weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"\nPreferred Model (Weighted, Year FE, Robust SE):")
print(f"DiD coefficient (treat_post): {model_robust.params['treat_post']:.4f}")
print(f"Robust Standard Error: {model_robust.bse['treat_post']:.4f}")
print(f"95% CI: [{model_robust.conf_int().loc['treat_post', 0]:.4f}, {model_robust.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model_robust.pvalues['treat_post']:.4f}")
print(f"N: {int(model_robust.nobs):,}")

# =============================================================================
# 13. Employment as Alternative Outcome
# =============================================================================
print("\n[13] Alternative Outcome: Any Employment")
print("="*80)

model_emp = smf.wls('employed ~ treat + C(YEAR) + treat_post + AGE + age_sq + female + married + C(educ_cat)',
                     data=df_analysis,
                     weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient on Employment: {model_emp.params['treat_post']:.4f}")
print(f"Robust SE: {model_emp.bse['treat_post']:.4f}")
print(f"95% CI: [{model_emp.conf_int().loc['treat_post', 0]:.4f}, {model_emp.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model_emp.pvalues['treat_post']:.4f}")

# =============================================================================
# 14. Event Study (Year-by-Year Effects)
# =============================================================================
print("\n[14] Event Study: Year-by-Year Effects")
print("="*80)

# Create year-treatment interactions (reference: 2011)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'treat_y{yr}'] = df_analysis['treat'] * (df_analysis['YEAR'] == yr).astype(int)

event_formula = 'fulltime ~ treat + C(YEAR) + treat_y2006 + treat_y2007 + treat_y2008 + treat_y2009 + treat_y2010 + treat_y2013 + treat_y2014 + treat_y2015 + treat_y2016 + AGE + age_sq + female + married + C(educ_cat)'
model_event = smf.wls(event_formula, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nYear-specific treatment effects (reference: 2011):")
print(f"  2006: {model_event.params['treat_y2006']:.4f} (SE: {model_event.bse['treat_y2006']:.4f})")
print(f"  2007: {model_event.params['treat_y2007']:.4f} (SE: {model_event.bse['treat_y2007']:.4f})")
print(f"  2008: {model_event.params['treat_y2008']:.4f} (SE: {model_event.bse['treat_y2008']:.4f})")
print(f"  2009: {model_event.params['treat_y2009']:.4f} (SE: {model_event.bse['treat_y2009']:.4f})")
print(f"  2010: {model_event.params['treat_y2010']:.4f} (SE: {model_event.bse['treat_y2010']:.4f})")
print(f"  2011: Reference year (0)")
print(f"  2013: {model_event.params['treat_y2013']:.4f} (SE: {model_event.bse['treat_y2013']:.4f})")
print(f"  2014: {model_event.params['treat_y2014']:.4f} (SE: {model_event.bse['treat_y2014']:.4f})")
print(f"  2015: {model_event.params['treat_y2015']:.4f} (SE: {model_event.bse['treat_y2015']:.4f})")
print(f"  2016: {model_event.params['treat_y2016']:.4f} (SE: {model_event.bse['treat_y2016']:.4f})")

# =============================================================================
# 15. Heterogeneity Analysis
# =============================================================================
print("\n[15] Heterogeneity Analysis")
print("="*80)

# By sex
print("\n--- By Sex ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex]
    model_sub = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + AGE + age_sq + married + C(educ_cat)',
                         data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_sub.params['treat_post']:.4f} (SE: {model_sub.bse['treat_post']:.4f}), N = {int(model_sub.nobs):,}")

# By education
print("\n--- By Education ---")
for educ, label in [('less_hs', 'Less than HS'), ('hs', 'High School'), ('some_college', 'Some College+')]:
    if educ == 'some_college':
        df_sub = df_analysis[df_analysis['educ_cat'].isin(['some_college', 'college'])]
    else:
        df_sub = df_analysis[df_analysis['educ_cat'] == educ]
    if len(df_sub) > 100:
        model_sub = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + AGE + age_sq + female + married',
                             data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
        print(f"{label}: DiD = {model_sub.params['treat_post']:.4f} (SE: {model_sub.bse['treat_post']:.4f}), N = {int(model_sub.nobs):,}")

# By age group
print("\n--- By Age Group (as of survey) ---")
for age_lo, age_hi, label in [(18, 24, '18-24'), (25, 34, '25-34'), (35, 64, '35-64')]:
    df_sub = df_analysis[(df_analysis['AGE'] >= age_lo) & (df_analysis['AGE'] <= age_hi)]
    if len(df_sub) > 100:
        model_sub = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + AGE + age_sq + female + married + C(educ_cat)',
                             data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
        print(f"{label}: DiD = {model_sub.params['treat_post']:.4f} (SE: {model_sub.bse['treat_post']:.4f}), N = {int(model_sub.nobs):,}")

# =============================================================================
# 16. Placebo Test (Pre-treatment trends)
# =============================================================================
print("\n[16] Placebo Test: Pre-treatment Trend")
print("="*80)

# Test for differential pre-trends using only pre-period data
df_pre = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_pre['time_trend'] = df_pre['YEAR'] - 2006
df_pre['treat_trend'] = df_pre['treat'] * df_pre['time_trend']

model_placebo = smf.wls('fulltime ~ treat + time_trend + treat_trend + AGE + age_sq + female + married + C(educ_cat)',
                         data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"Differential pre-trend (treat_trend): {model_placebo.params['treat_trend']:.4f}")
print(f"SE: {model_placebo.bse['treat_trend']:.4f}")
print(f"P-value: {model_placebo.pvalues['treat_trend']:.4f}")

if model_placebo.pvalues['treat_trend'] > 0.05:
    print("Result: No significant differential pre-trend (parallel trends assumption supported)")
else:
    print("Result: Evidence of differential pre-trend (parallel trends assumption may be violated)")

# =============================================================================
# 17. Final Summary
# =============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"""
PREFERRED ESTIMATE:
-------------------
Effect Size: {model_robust.params['treat_post']:.4f} ({model_robust.params['treat_post']*100:.2f} percentage points)
Standard Error (Robust): {model_robust.bse['treat_post']:.4f}
95% Confidence Interval: [{model_robust.conf_int().loc['treat_post', 0]:.4f}, {model_robust.conf_int().loc['treat_post', 1]:.4f}]
P-value: {model_robust.pvalues['treat_post']:.4f}
Sample Size: {int(model_robust.nobs):,}

INTERPRETATION:
DACA eligibility is associated with a {model_robust.params['treat_post']*100:.2f} percentage point
{'increase' if model_robust.params['treat_post'] > 0 else 'decrease'} in the probability of full-time employment
(35+ hours per week) among Hispanic-Mexican, Mexican-born non-citizens.

This effect is {'statistically significant' if model_robust.pvalues['treat_post'] < 0.05 else 'not statistically significant'}
at the 5% level.

ALTERNATIVE OUTCOME (Any Employment):
Effect: {model_emp.params['treat_post']:.4f} ({model_emp.params['treat_post']*100:.2f} pp)
SE: {model_emp.bse['treat_post']:.4f}
""")

# =============================================================================
# 18. Export Results for LaTeX
# =============================================================================
print("\n[18] Exporting results for LaTeX tables...")

# Create results dictionary for export
results = {
    'Model 1 (Basic)': {
        'coef': model1.params['treat_post'],
        'se': model1.bse['treat_post'],
        'ci_low': model1.conf_int().loc['treat_post', 0],
        'ci_high': model1.conf_int().loc['treat_post', 1],
        'pval': model1.pvalues['treat_post'],
        'n': int(model1.nobs),
        'weighted': 'No',
        'controls': 'No'
    },
    'Model 2 (Demographics)': {
        'coef': model2.params['treat_post'],
        'se': model2.bse['treat_post'],
        'ci_low': model2.conf_int().loc['treat_post', 0],
        'ci_high': model2.conf_int().loc['treat_post', 1],
        'pval': model2.pvalues['treat_post'],
        'n': int(model2.nobs),
        'weighted': 'No',
        'controls': 'Age, Sex, Married'
    },
    'Model 3 (+ Education)': {
        'coef': model3.params['treat_post'],
        'se': model3.bse['treat_post'],
        'ci_low': model3.conf_int().loc['treat_post', 0],
        'ci_high': model3.conf_int().loc['treat_post', 1],
        'pval': model3.pvalues['treat_post'],
        'n': int(model3.nobs),
        'weighted': 'No',
        'controls': '+ Education'
    },
    'Model 4 (+ Year FE)': {
        'coef': model4.params['treat_post'],
        'se': model4.bse['treat_post'],
        'ci_low': model4.conf_int().loc['treat_post', 0],
        'ci_high': model4.conf_int().loc['treat_post', 1],
        'pval': model4.pvalues['treat_post'],
        'n': int(model4.nobs),
        'weighted': 'No',
        'controls': '+ Year FE'
    },
    'Model 5 (Weighted Basic)': {
        'coef': model5.params['treat_post'],
        'se': model5.bse['treat_post'],
        'ci_low': model5.conf_int().loc['treat_post', 0],
        'ci_high': model5.conf_int().loc['treat_post', 1],
        'pval': model5.pvalues['treat_post'],
        'n': int(model5.nobs),
        'weighted': 'Yes',
        'controls': 'No'
    },
    'Model 6 (Weighted + Controls)': {
        'coef': model6.params['treat_post'],
        'se': model6.bse['treat_post'],
        'ci_low': model6.conf_int().loc['treat_post', 0],
        'ci_high': model6.conf_int().loc['treat_post', 1],
        'pval': model6.pvalues['treat_post'],
        'n': int(model6.nobs),
        'weighted': 'Yes',
        'controls': 'Full'
    },
    'Model 7 (Preferred)': {
        'coef': model_robust.params['treat_post'],
        'se': model_robust.bse['treat_post'],
        'ci_low': model_robust.conf_int().loc['treat_post', 0],
        'ci_high': model_robust.conf_int().loc['treat_post', 1],
        'pval': model_robust.pvalues['treat_post'],
        'n': int(model_robust.nobs),
        'weighted': 'Yes',
        'controls': 'Full + Robust SE'
    }
}

# Save to CSV
results_df = pd.DataFrame(results).T
results_df.to_csv('results_summary.csv')
print("Results saved to results_summary.csv")

# Summary statistics
summary_stats = {
    'treat_pre_n': len(treat_pre),
    'treat_pre_fulltime': treat_pre['fulltime'].mean(),
    'treat_post_n': len(treat_post),
    'treat_post_fulltime': treat_post['fulltime'].mean(),
    'ctrl_pre_n': len(ctrl_pre),
    'ctrl_pre_fulltime': ctrl_pre['fulltime'].mean(),
    'ctrl_post_n': len(ctrl_post),
    'ctrl_post_fulltime': ctrl_post['fulltime'].mean(),
    'did_simple': did_simple,
    'total_n': len(df_analysis),
    'treat_n': df_analysis['treat'].sum(),
    'ctrl_n': (df_analysis['treat'] == 0).sum()
}

pd.Series(summary_stats).to_csv('summary_stats.csv')
print("Summary statistics saved to summary_stats.csv")

# Event study results
event_results = {}
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    event_results[yr] = {
        'coef': model_event.params[f'treat_y{yr}'],
        'se': model_event.bse[f'treat_y{yr}'],
        'ci_low': model_event.conf_int().loc[f'treat_y{yr}', 0],
        'ci_high': model_event.conf_int().loc[f'treat_y{yr}', 1]
    }
event_results[2011] = {'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0}

pd.DataFrame(event_results).T.to_csv('event_study_results.csv')
print("Event study results saved to event_study_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
