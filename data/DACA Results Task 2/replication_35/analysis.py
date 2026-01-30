"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the US.

Design: Difference-in-Differences
- Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
- Control: Ages 31-35 at DACA implementation
- Outcome: Full-time employment (35+ hours/week)
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

print("=" * 70)
print("DACA REPLICATION STUDY - DATA ANALYSIS")
print("=" * 70)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/data.csv')
print(f"    Total observations: {len(df):,}")
print(f"    Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# STEP 1: Define Sample Selection Criteria
# ============================================================================
print("\n[2] Applying sample selection criteria...")

# 2a. Hispanic-Mexican ethnicity (HISPAN = 1)
print(f"    Initial observations: {len(df):,}")
df_sample = df[df['HISPAN'] == 1].copy()
print(f"    After Hispanic-Mexican filter (HISPAN=1): {len(df_sample):,}")

# 2b. Born in Mexico (BPL = 200)
df_sample = df_sample[df_sample['BPL'] == 200]
print(f"    After Mexico birthplace filter (BPL=200): {len(df_sample):,}")

# 2c. Not a citizen (CITIZEN = 3)
# This proxies for undocumented status as we cannot distinguish documented from undocumented
df_sample = df_sample[df_sample['CITIZEN'] == 3]
print(f"    After non-citizen filter (CITIZEN=3): {len(df_sample):,}")

# ============================================================================
# STEP 2: Calculate Age at DACA Implementation (June 15, 2012)
# ============================================================================
print("\n[3] Calculating age at DACA implementation (June 15, 2012)...")

# Age at DACA = 2012 - BIRTHYR
# But we need to be careful about birth quarter
# DACA cutoff: Had not yet had 31st birthday as of June 15, 2012
# Birth quarters: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

# Calculate age on June 15, 2012
# If born in Q1 (Jan-Mar) or Q2 (Apr-Jun before June 15): already had birthday in 2012
# If born in Q3 (Jul-Sep) or Q4 (Oct-Dec): hadn't had birthday yet

df_sample['age_at_daca'] = 2012 - df_sample['BIRTHYR']
# Adjust for those who hadn't had birthday yet (born after June 15)
# Q3 and Q4 births: subtract 1 year
df_sample.loc[df_sample['BIRTHQTR'].isin([3, 4]), 'age_at_daca'] = \
    df_sample.loc[df_sample['BIRTHQTR'].isin([3, 4]), 'age_at_daca'] - 1

print(f"    Age at DACA distribution:")
print(df_sample['age_at_daca'].describe())

# ============================================================================
# STEP 3: Define Treatment and Control Groups
# ============================================================================
print("\n[4] Defining treatment and control groups...")

# Treatment: Ages 26-30 at DACA implementation (eligible)
# Control: Ages 31-35 at DACA implementation (too old to be eligible)

df_sample['treated'] = ((df_sample['age_at_daca'] >= 26) &
                        (df_sample['age_at_daca'] <= 30)).astype(int)
df_sample['control'] = ((df_sample['age_at_daca'] >= 31) &
                        (df_sample['age_at_daca'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df_sample[(df_sample['treated'] == 1) | (df_sample['control'] == 1)].copy()
print(f"    Observations in treatment/control groups: {len(df_analysis):,}")
print(f"    Treatment group (ages 26-30): {df_analysis['treated'].sum():,}")
print(f"    Control group (ages 31-35): {(df_analysis['control'] == 1).sum():,}")

# ============================================================================
# STEP 4: Additional Eligibility Criteria
# ============================================================================
print("\n[5] Applying additional DACA eligibility criteria...")

# Arrived before age 16
# Age at arrival = YRIMMIG - BIRTHYR (approximately)
# If YRIMMIG is 0, it's N/A - we'll need to handle this

# First, let's see the YRIMMIG distribution
print(f"    YRIMMIG distribution (top values):")
print(df_analysis['YRIMMIG'].value_counts().head(10))

# Filter: YRIMMIG > 0 (valid immigration year)
df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0]
print(f"    After valid YRIMMIG filter: {len(df_analysis):,}")

# Calculate age at immigration
df_analysis['age_at_immigration'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']

# Filter: Arrived before age 16
df_analysis = df_analysis[df_analysis['age_at_immigration'] < 16]
print(f"    After arrived before age 16 filter: {len(df_analysis):,}")

# Filter: Arrived by June 15, 2007 (continuous presence requirement)
# We'll use YRIMMIG <= 2007
df_analysis = df_analysis[df_analysis['YRIMMIG'] <= 2007]
print(f"    After arrived by 2007 filter: {len(df_analysis):,}")

print(f"\n    Final sample by group:")
print(f"    Treatment group (ages 26-30): {df_analysis['treated'].sum():,}")
print(f"    Control group (ages 31-35): {(df_analysis['treated'] == 0).sum():,}")

# ============================================================================
# STEP 5: Define Outcome Variable
# ============================================================================
print("\n[6] Defining outcome variable (full-time employment)...")

# Full-time = usually working 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A or not working)

df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"    Full-time employment rate overall: {df_analysis['fulltime'].mean():.3f}")

# ============================================================================
# STEP 6: Define Pre/Post Treatment Period
# ============================================================================
print("\n[7] Defining pre/post treatment periods...")

# DACA was implemented in 2012
# Post-treatment: 2013-2016
# Pre-treatment: 2006-2011
# We exclude 2012 because we can't distinguish before/after DACA within that year

df_analysis = df_analysis[df_analysis['YEAR'] != 2012]
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

print(f"    Pre-treatment years (2006-2011): {len(df_analysis[df_analysis['post'] == 0]):,}")
print(f"    Post-treatment years (2013-2016): {len(df_analysis[df_analysis['post'] == 1]):,}")

print(f"\n    Observations by year:")
print(df_analysis.groupby('YEAR').size())

# ============================================================================
# STEP 7: Create DiD Interaction Term
# ============================================================================
print("\n[8] Creating DiD interaction term...")

df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

# ============================================================================
# STEP 8: Summary Statistics
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

# Create summary by treatment status and period
summary_stats = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'SEX': lambda x: (x == 2).mean(),  # Proportion female
    'EDUC': 'mean',
    'PERWT': 'sum'
}).round(3)

print("\nSummary statistics by group and period:")
print(summary_stats)

# Calculate simple DiD
pre_treat = df_analysis[(df_analysis['treated'] == 1) & (df_analysis['post'] == 0)]['fulltime'].mean()
post_treat = df_analysis[(df_analysis['treated'] == 1) & (df_analysis['post'] == 1)]['fulltime'].mean()
pre_control = df_analysis[(df_analysis['treated'] == 0) & (df_analysis['post'] == 0)]['fulltime'].mean()
post_control = df_analysis[(df_analysis['treated'] == 0) & (df_analysis['post'] == 1)]['fulltime'].mean()

print(f"\n    Treatment group pre-DACA:  {pre_treat:.4f}")
print(f"    Treatment group post-DACA: {post_treat:.4f}")
print(f"    Treatment change:          {post_treat - pre_treat:.4f}")
print(f"\n    Control group pre-DACA:    {pre_control:.4f}")
print(f"    Control group post-DACA:   {post_control:.4f}")
print(f"    Control change:            {post_control - pre_control:.4f}")
print(f"\n    Simple DiD estimate:       {(post_treat - pre_treat) - (post_control - pre_control):.4f}")

# ============================================================================
# STEP 9: Difference-in-Differences Regression
# ============================================================================
print("\n" + "=" * 70)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION RESULTS")
print("=" * 70)

# Model 1: Basic DiD (no weights)
print("\n--- Model 1: Basic DiD (unweighted) ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_analysis).fit()
print(model1.summary())

# Model 2: DiD with survey weights (using WLS)
print("\n--- Model 2: DiD with survey weights ---")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit()
print(model2.summary())

# Model 3: DiD with controls
print("\n--- Model 3: DiD with demographic controls ---")

# Create control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # High school or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # Some college or more

# State fixed effects - using STATEFIP
df_analysis['state'] = df_analysis['STATEFIP'].astype(str)

model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_college',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit()
print(model3.summary())

# Model 4: DiD with year and state fixed effects
print("\n--- Model 4: DiD with year fixed effects ---")
df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)

model4 = smf.wls('fulltime ~ treated + C(year_factor) + treated_post + female + married + educ_hs + educ_college',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit()
print(model4.summary())

# Model 5: With robust/clustered standard errors
print("\n--- Model 5: DiD with clustered standard errors (by state) ---")
model5 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_college',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit(cov_type='cluster',
                                                    cov_kwds={'groups': df_analysis['STATEFIP']})
print(model5.summary())

# ============================================================================
# STEP 10: Store Key Results
# ============================================================================
print("\n" + "=" * 70)
print("PREFERRED ESTIMATE (Model 5 - with controls and clustered SEs)")
print("=" * 70)

preferred_coef = model5.params['treated_post']
preferred_se = model5.bse['treated_post']
preferred_ci = model5.conf_int().loc['treated_post']
n_obs = len(df_analysis)

print(f"\n    Effect size (coefficient):     {preferred_coef:.4f}")
print(f"    Standard error:                {preferred_se:.4f}")
print(f"    95% Confidence interval:       [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"    t-statistic:                   {model5.tvalues['treated_post']:.4f}")
print(f"    p-value:                       {model5.pvalues['treated_post']:.4f}")
print(f"    Sample size:                   {n_obs:,}")

# ============================================================================
# STEP 11: Robustness Checks
# ============================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECKS")
print("=" * 70)

# Robustness 1: Linear probability model vs Logit
print("\n--- Robustness 1: Logit model (marginal effects) ---")
try:
    logit_model = smf.logit('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_college',
                            data=df_analysis).fit(disp=0)
    # Get marginal effect at mean
    margeff = logit_model.get_margeff(at='mean')
    print(f"    Logit marginal effect of treated_post: {margeff.margeff[2]:.4f}")
    print(f"    Logit SE: {margeff.margeff_se[2]:.4f}")
except Exception as e:
    print(f"    Logit model could not be estimated: {e}")

# Robustness 2: Different age bandwidths
print("\n--- Robustness 2: Narrower age bandwidth (ages 27-29 vs 32-34) ---")
df_narrow = df_analysis[((df_analysis['age_at_daca'] >= 27) & (df_analysis['age_at_daca'] <= 29)) |
                        ((df_analysis['age_at_daca'] >= 32) & (df_analysis['age_at_daca'] <= 34))].copy()
df_narrow['treated'] = ((df_narrow['age_at_daca'] >= 27) & (df_narrow['age_at_daca'] <= 29)).astype(int)
df_narrow['treated_post'] = df_narrow['treated'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_college',
                       data=df_narrow,
                       weights=df_narrow['PERWT']).fit(cov_type='cluster',
                                                       cov_kwds={'groups': df_narrow['STATEFIP']})
print(f"    Narrower bandwidth estimate: {model_narrow.params['treated_post']:.4f} (SE: {model_narrow.bse['treated_post']:.4f})")
print(f"    Sample size: {len(df_narrow):,}")

# Robustness 3: Placebo test (fake treatment in pre-period)
print("\n--- Robustness 3: Placebo test (pseudo-treatment in 2009) ---")
df_placebo = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treated_post_placebo'] = df_placebo['treated'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treated + post_placebo + treated_post_placebo + female + married + educ_hs + educ_college',
                        data=df_placebo,
                        weights=df_placebo['PERWT']).fit(cov_type='cluster',
                                                         cov_kwds={'groups': df_placebo['STATEFIP']})
print(f"    Placebo estimate: {model_placebo.params['treated_post_placebo']:.4f} (SE: {model_placebo.bse['treated_post_placebo']:.4f})")
print(f"    p-value: {model_placebo.pvalues['treated_post_placebo']:.4f}")

# Robustness 4: By gender
print("\n--- Robustness 4: Effects by gender ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_gender = df_analysis[df_analysis['SEX'] == sex].copy()
    model_gender = smf.wls('fulltime ~ treated + post + treated_post + married + educ_hs + educ_college',
                           data=df_gender,
                           weights=df_gender['PERWT']).fit(cov_type='cluster',
                                                           cov_kwds={'groups': df_gender['STATEFIP']})
    print(f"    {label}: {model_gender.params['treated_post']:.4f} (SE: {model_gender.bse['treated_post']:.4f}), n={len(df_gender):,}")

# ============================================================================
# STEP 12: Event Study / Dynamic Effects
# ============================================================================
print("\n" + "=" * 70)
print("EVENT STUDY ANALYSIS")
print("=" * 70)

# Create year dummies interacted with treatment
df_analysis['year_2006'] = (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['year_2007'] = (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['year_2008'] = (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['year_2009'] = (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['year_2010'] = (df_analysis['YEAR'] == 2010).astype(int)
df_analysis['year_2011'] = (df_analysis['YEAR'] == 2011).astype(int)
df_analysis['year_2013'] = (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['year_2014'] = (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['year_2015'] = (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['year_2016'] = (df_analysis['YEAR'] == 2016).astype(int)

# Interactions (2011 is reference year)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'treat_x_{year}'] = df_analysis['treated'] * df_analysis[f'year_{year}']

event_formula = 'fulltime ~ treated + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + ' + \
                'treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + ' + \
                'female + married + educ_hs + educ_college'

model_event = smf.wls(event_formula,
                      data=df_analysis,
                      weights=df_analysis['PERWT']).fit(cov_type='cluster',
                                                        cov_kwds={'groups': df_analysis['STATEFIP']})

print("\nEvent study coefficients (treatment x year interactions):")
print("Reference year: 2011")
event_coefs = {}
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treat_x_{year}']
    se = model_event.bse[f'treat_x_{year}']
    pval = model_event.pvalues[f'treat_x_{year}']
    event_coefs[year] = {'coef': coef, 'se': se, 'pval': pval}
    sig = '*' if pval < 0.1 else ''
    sig = '**' if pval < 0.05 else sig
    sig = '***' if pval < 0.01 else sig
    print(f"    {year}: {coef:7.4f} ({se:.4f}) {sig}")

# ============================================================================
# STEP 13: Save Results for Report
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save key results to a file
results = {
    'preferred_estimate': {
        'coefficient': preferred_coef,
        'se': preferred_se,
        'ci_lower': preferred_ci[0],
        'ci_upper': preferred_ci[1],
        't_stat': model5.tvalues['treated_post'],
        'p_value': model5.pvalues['treated_post'],
        'n_obs': n_obs
    },
    'simple_did': (post_treat - pre_treat) - (post_control - pre_control),
    'group_means': {
        'pre_treat': pre_treat,
        'post_treat': post_treat,
        'pre_control': pre_control,
        'post_control': post_control
    },
    'event_study': event_coefs
}

# Print final summary table
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"""
DIFFERENCE-IN-DIFFERENCES ESTIMATE OF DACA EFFECT ON FULL-TIME EMPLOYMENT

Population: Hispanic-Mexican, Mexican-born, non-citizens who arrived
            before age 16 and by 2007

Treatment group:  Ages 26-30 at DACA implementation (n = {df_analysis['treated'].sum():,})
Control group:    Ages 31-35 at DACA implementation (n = {(df_analysis['treated'] == 0).sum():,})

Pre-treatment period:  2006-2011
Post-treatment period: 2013-2016

MAIN RESULTS:
-------------
Simple DiD (no controls):        {(post_treat - pre_treat) - (post_control - pre_control):.4f}
DiD with controls & weights:     {preferred_coef:.4f}
Standard error (clustered):      {preferred_se:.4f}
95% confidence interval:         [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]
p-value:                         {model5.pvalues['treated_post']:.4f}

INTERPRETATION:
DACA eligibility is associated with a {preferred_coef:.1%} {"increase" if preferred_coef > 0 else "decrease"} in the probability
of full-time employment (working 35+ hours per week) among DACA-eligible individuals
relative to the control group, {"though this effect is not statistically significant at conventional levels" if model5.pvalues['treated_post'] > 0.05 else "and this effect is statistically significant at the 5% level"}.
""")

# Save detailed results to CSV
results_df = pd.DataFrame({
    'Model': ['Unweighted DiD', 'Weighted DiD', 'With Controls', 'Year FE', 'Clustered SE (Preferred)'],
    'Coefficient': [model1.params['treated_post'], model2.params['treated_post'],
                   model3.params['treated_post'], model4.params['treated_post'],
                   model5.params['treated_post']],
    'SE': [model1.bse['treated_post'], model2.bse['treated_post'],
           model3.bse['treated_post'], model4.bse['treated_post'],
           model5.bse['treated_post']],
    'p_value': [model1.pvalues['treated_post'], model2.pvalues['treated_post'],
                model3.pvalues['treated_post'], model4.pvalues['treated_post'],
                model5.pvalues['treated_post']]
})
results_df.to_csv('regression_results.csv', index=False)
print("Regression results saved to regression_results.csv")

# Save summary statistics
summary_table = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': 'mean',
    'PERWT': 'sum',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'educ_college': 'mean'
}).reset_index()
summary_table.columns = ['Treatment', 'Post', 'Fulltime_Rate', 'Weighted_N', 'Mean_Age',
                         'Pct_Female', 'Pct_Married', 'Pct_HS', 'Pct_College']
summary_table.to_csv('summary_stats.csv', index=False)
print("Summary statistics saved to summary_stats.csv")

# Save event study coefficients
event_df = pd.DataFrame(event_coefs).T
event_df.index.name = 'Year'
event_df.to_csv('event_study.csv')
print("Event study results saved to event_study.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
