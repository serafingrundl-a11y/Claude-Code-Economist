"""
DACA Replication Study - Analysis Script
=========================================
Research Question: Among Hispanic-Mexican Mexican-born people living in the US,
what was the causal impact of DACA eligibility on full-time employment (35+ hrs/week)?

Treatment Group: Ages 26-30 at the time of DACA implementation (June 15, 2012)
Control Group: Ages 31-35 at the time of DACA implementation (June 15, 2012)
Pre-period: 2006-2011
Post-period: 2013-2016 (excluding 2012 as it's ambiguous)

Difference-in-Differences Design
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 2\replication_05")

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# STEP 1: LOAD AND FILTER DATA
# =============================================================================
print("\n[STEP 1] Loading data...")

# Read data in chunks due to large file size
# We'll filter as we read to reduce memory usage

# Define the columns we need
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK', 'MARST']

# Read data
print("Reading full dataset...")
df = pd.read_csv('data/data.csv', usecols=cols_needed)
print(f"Total observations loaded: {len(df):,}")

# =============================================================================
# STEP 2: APPLY SAMPLE RESTRICTIONS
# =============================================================================
print("\n[STEP 2] Applying sample restrictions...")

# Restriction 1: Hispanic-Mexican ethnicity (HISPAN == 1)
# Based on data dictionary: HISPAN 1 = Mexican
df = df[df['HISPAN'] == 1]
print(f"After Hispanic-Mexican filter: {len(df):,}")

# Restriction 2: Born in Mexico (BPL == 200)
# Based on data dictionary: BPL 200 = Mexico
df = df[df['BPL'] == 200]
print(f"After Mexico birthplace filter: {len(df):,}")

# Restriction 3: Non-citizen (to proxy for undocumented status)
# CITIZEN == 3 means "Not a citizen"
# Note: We cannot distinguish documented vs undocumented, so we use non-citizen as proxy
df = df[df['CITIZEN'] == 3]
print(f"After non-citizen filter: {len(df):,}")

# Restriction 4: Years 2006-2016 (excluding 2012 as transition year)
df = df[(df['YEAR'] >= 2006) & (df['YEAR'] <= 2016) & (df['YEAR'] != 2012)]
print(f"After year filter (2006-2011, 2013-2016): {len(df):,}")

# =============================================================================
# STEP 3: DETERMINE DACA ELIGIBILITY AND AGE GROUPS
# =============================================================================
print("\n[STEP 3] Creating treatment and control groups...")

# DACA was implemented on June 15, 2012
# Eligibility requires:
# - Arrived before 16th birthday
# - Not yet 31st birthday as of June 15, 2012
# - Lived continuously in US since June 15, 2007
# - Present in US on June 15, 2012

# Calculate age as of June 15, 2012
# Age on June 15, 2012 = 2012 - BIRTHYR (adjusted for birth quarter)
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born in Q3 or Q4, they hadn't reached their birthday by June 15

df['age_june2012'] = 2012 - df['BIRTHYR']
# Adjust for those who hadn't had birthday yet by June 15
# If BIRTHQTR is 3 (Jul-Sep) or 4 (Oct-Dec), subtract 1
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june2012'] -= 1

# For those born in Q2 (Apr-Jun), it's ambiguous - we'll assume they've had their birthday
# This is a conservative assumption

# Treatment group: Ages 26-30 as of June 15, 2012
# Control group: Ages 31-35 as of June 15, 2012
df['treated'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)
df['control'] = ((df['age_june2012'] >= 31) & (df['age_june2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df = df[(df['treated'] == 1) | (df['control'] == 1)]
print(f"After age group filter (26-35 as of June 2012): {len(df):,}")

# Additional DACA eligibility criteria
# 1. Arrived before 16th birthday
# We need to check: year of immigration - birth year < 16
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
# Keep those who arrived before age 16 (and YRIMMIG is not 0 = N/A)
df = df[(df['YRIMMIG'] > 0) & (df['age_at_immigration'] < 16)]
print(f"After arrived-before-16 filter: {len(df):,}")

# 2. Lived continuously in US since June 15, 2007
# This means they must have arrived by 2007
df = df[df['YRIMMIG'] <= 2007]
print(f"After continuous residence filter (arrived by 2007): {len(df):,}")

# =============================================================================
# STEP 4: CREATE OUTCOME VARIABLE AND POST-PERIOD INDICATOR
# =============================================================================
print("\n[STEP 4] Creating outcome variable and post-period indicator...")

# Outcome: Full-time employment (usually working 35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Post-period indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term for DiD
df['treated_post'] = df['treated'] * df['post']

print(f"Final sample size: {len(df):,}")
print(f"\nSample composition:")
print(f"  Treatment group (26-30): {df['treated'].sum():,}")
print(f"  Control group (31-35): {(1-df['treated']).sum():,}")
print(f"  Pre-period observations: {(1-df['post']).sum():,}")
print(f"  Post-period observations: {df['post'].sum():,}")

# =============================================================================
# STEP 5: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[STEP 5] Descriptive Statistics")
print("="*80)

# Create summary statistics by group and period
def weighted_mean(data, values, weights):
    return np.average(data[values], weights=data[weights])

def weighted_std(data, values, weights):
    average = np.average(data[values], weights=data[weights])
    variance = np.average((data[values] - average)**2, weights=data[weights])
    return np.sqrt(variance)

# Summary by treatment status and period
print("\n--- Full-time Employment Rates by Group and Period ---")
for treat_val, treat_label in [(1, 'Treatment (26-30)'), (0, 'Control (31-35)')]:
    for post_val, post_label in [(0, 'Pre-DACA (2006-2011)'), (1, 'Post-DACA (2013-2016)')]:
        subset = df[(df['treated'] == treat_val) & (df['post'] == post_val)]
        if len(subset) > 0:
            mean_ft = weighted_mean(subset, 'fulltime', 'PERWT')
            n = len(subset)
            print(f"{treat_label}, {post_label}: {mean_ft:.4f} (n={n:,})")

# Calculate simple DiD estimate
pre_treat = df[(df['treated'] == 1) & (df['post'] == 0)]
post_treat = df[(df['treated'] == 1) & (df['post'] == 1)]
pre_control = df[(df['treated'] == 0) & (df['post'] == 0)]
post_control = df[(df['treated'] == 0) & (df['post'] == 1)]

mean_pre_treat = weighted_mean(pre_treat, 'fulltime', 'PERWT')
mean_post_treat = weighted_mean(post_treat, 'fulltime', 'PERWT')
mean_pre_control = weighted_mean(pre_control, 'fulltime', 'PERWT')
mean_post_control = weighted_mean(post_control, 'fulltime', 'PERWT')

simple_did = (mean_post_treat - mean_pre_treat) - (mean_post_control - mean_pre_control)

print(f"\n--- Simple Difference-in-Differences Calculation ---")
print(f"Treatment group change: {mean_post_treat:.4f} - {mean_pre_treat:.4f} = {mean_post_treat - mean_pre_treat:.4f}")
print(f"Control group change: {mean_post_control:.4f} - {mean_pre_control:.4f} = {mean_post_control - mean_pre_control:.4f}")
print(f"DiD estimate: {simple_did:.4f}")

# =============================================================================
# STEP 6: REGRESSION ANALYSIS
# =============================================================================
print("\n[STEP 6] Regression Analysis")
print("="*80)

# Prepare control variables
# Education categories
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # High school or more
df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # Some college or more

# Sex (1 = Male, 2 = Female)
df['female'] = (df['SEX'] == 2).astype(int)

# Marital status (1 = Married spouse present)
df['married'] = (df['MARST'] == 1).astype(int)

# Age in current year (for age fixed effects within treatment/control groups)
df['age_current'] = df['AGE']

# Model 1: Basic DiD (no controls, no weights)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit()
print(model1.summary().tables[1])

# Model 2: DiD with survey weights (WLS)
print("\n--- Model 2: DiD with Survey Weights ---")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with covariates and survey weights
print("\n--- Model 3: DiD with Covariates ---")
model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + age_current',
                  data=df, weights=df['PERWT']).fit()
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with Year Fixed Effects ---")
df['year_str'] = df['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ treated + C(year_str) + treated_post + female + married + educ_hs + age_current',
                  data=df, weights=df['PERWT']).fit()
print("\nDiD coefficient (treated_post):")
print(f"  Estimate: {model4.params['treated_post']:.6f}")
print(f"  Std Error: {model4.bse['treated_post']:.6f}")
print(f"  t-stat: {model4.tvalues['treated_post']:.4f}")
print(f"  p-value: {model4.pvalues['treated_post']:.6f}")

# Model 5: DiD with state fixed effects
print("\n--- Model 5: DiD with State and Year Fixed Effects ---")
df['state_str'] = df['STATEFIP'].astype(str)
model5 = smf.wls('fulltime ~ treated + C(year_str) + C(state_str) + treated_post + female + married + educ_hs + age_current',
                  data=df, weights=df['PERWT']).fit()
print("\nDiD coefficient (treated_post):")
print(f"  Estimate: {model5.params['treated_post']:.6f}")
print(f"  Std Error: {model5.bse['treated_post']:.6f}")
print(f"  t-stat: {model5.tvalues['treated_post']:.4f}")
print(f"  p-value: {model5.pvalues['treated_post']:.6f}")

# =============================================================================
# STEP 7: ROBUST STANDARD ERRORS
# =============================================================================
print("\n[STEP 7] Robust Standard Errors")
print("="*80)

# Model with clustered standard errors at the state level
print("\n--- Model with State-Clustered Standard Errors ---")
model_robust = smf.wls('fulltime ~ treated + C(year_str) + C(state_str) + treated_post + female + married + educ_hs + age_current',
                        data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                          cov_kwds={'groups': df['STATEFIP']})
print("\nDiD coefficient (treated_post) with clustered SE:")
print(f"  Estimate: {model_robust.params['treated_post']:.6f}")
print(f"  Std Error: {model_robust.bse['treated_post']:.6f}")
print(f"  t-stat: {model_robust.tvalues['treated_post']:.4f}")
print(f"  p-value: {model_robust.pvalues['treated_post']:.6f}")
ci_low = model_robust.params['treated_post'] - 1.96 * model_robust.bse['treated_post']
ci_high = model_robust.params['treated_post'] + 1.96 * model_robust.bse['treated_post']
print(f"  95% CI: [{ci_low:.6f}, {ci_high:.6f}]")

# =============================================================================
# STEP 8: HETEROGENEITY ANALYSIS
# =============================================================================
print("\n[STEP 8] Heterogeneity Analysis")
print("="*80)

# By sex
print("\n--- By Sex ---")
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    model_sub = smf.wls('fulltime ~ treated + C(year_str) + treated_post',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"{sex_label}: DiD = {model_sub.params['treated_post']:.4f} (SE = {model_sub.bse['treated_post']:.4f})")

# By education
print("\n--- By Education ---")
for educ_val, educ_label in [(0, 'Less than HS'), (1, 'HS or more')]:
    df_sub = df[df['educ_hs'] == educ_val]
    if len(df_sub) > 100:
        model_sub = smf.wls('fulltime ~ treated + C(year_str) + treated_post',
                            data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
        print(f"{educ_label}: DiD = {model_sub.params['treated_post']:.4f} (SE = {model_sub.bse['treated_post']:.4f})")

# =============================================================================
# STEP 9: PRE-TRENDS ANALYSIS
# =============================================================================
print("\n[STEP 9] Pre-Trends Analysis (Event Study)")
print("="*80)

# Create year dummies and interactions with treatment
df['year_2006'] = (df['YEAR'] == 2006).astype(int)
df['year_2007'] = (df['YEAR'] == 2007).astype(int)
df['year_2008'] = (df['YEAR'] == 2008).astype(int)
df['year_2009'] = (df['YEAR'] == 2009).astype(int)
df['year_2010'] = (df['YEAR'] == 2010).astype(int)
df['year_2011'] = (df['YEAR'] == 2011).astype(int)
df['year_2013'] = (df['YEAR'] == 2013).astype(int)
df['year_2014'] = (df['YEAR'] == 2014).astype(int)
df['year_2015'] = (df['YEAR'] == 2015).astype(int)
df['year_2016'] = (df['YEAR'] == 2016).astype(int)

# Create interaction terms (omitting 2011 as reference year)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'treat_x_{year}'] = df['treated'] * df[f'year_{year}']

# Event study regression
event_formula = 'fulltime ~ treated + C(year_str) + treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + female + married + educ_hs'
model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 60)
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in event_years:
    coef = model_event.params[f'treat_x_{year}']
    se = model_event.bse[f'treat_x_{year}']
    pval = model_event.pvalues[f'treat_x_{year}']
    sig = '*' if pval < 0.05 else ''
    print(f"  {year}: {coef:8.4f} ({se:.4f}){sig}")

# =============================================================================
# STEP 10: ALTERNATIVE SPECIFICATIONS
# =============================================================================
print("\n[STEP 10] Alternative Specifications")
print("="*80)

# Alternative 1: Include labor force participants only
print("\n--- Alternative: Labor Force Participants Only ---")
df_lf = df[df['EMPSTAT'].isin([1, 2])]  # Employed or unemployed (in labor force)
if len(df_lf) > 100:
    model_lf = smf.wls('fulltime ~ treated + C(year_str) + treated_post + female + married + educ_hs',
                        data=df_lf, weights=df_lf['PERWT']).fit(cov_type='HC1')
    print(f"DiD estimate: {model_lf.params['treated_post']:.4f} (SE = {model_lf.bse['treated_post']:.4f})")
    print(f"Sample size: {len(df_lf):,}")

# Alternative 2: Different age bandwidth (narrower)
print("\n--- Alternative: Narrower Age Bandwidth (27-29 vs 32-34) ---")
df_narrow = df[(df['age_june2012'] >= 27) & (df['age_june2012'] <= 34)]
df_narrow['treated_narrow'] = ((df_narrow['age_june2012'] >= 27) & (df_narrow['age_june2012'] <= 29)).astype(int)
df_narrow['treated_post_narrow'] = df_narrow['treated_narrow'] * df_narrow['post']
if len(df_narrow) > 100:
    model_narrow = smf.wls('fulltime ~ treated_narrow + C(year_str) + treated_post_narrow + female + married + educ_hs',
                           data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
    print(f"DiD estimate: {model_narrow.params['treated_post_narrow']:.4f} (SE = {model_narrow.bse['treated_post_narrow']:.4f})")
    print(f"Sample size: {len(df_narrow):,}")

# =============================================================================
# STEP 11: SAVE RESULTS
# =============================================================================
print("\n[STEP 11] Saving Results")
print("="*80)

# Create results dictionary
results = {
    'model1_basic': {
        'estimate': model1.params['treated_post'],
        'se': model1.bse['treated_post'],
        'pvalue': model1.pvalues['treated_post'],
        'n': model1.nobs
    },
    'model2_weighted': {
        'estimate': model2.params['treated_post'],
        'se': model2.bse['treated_post'],
        'pvalue': model2.pvalues['treated_post'],
        'n': model2.nobs
    },
    'model3_covariates': {
        'estimate': model3.params['treated_post'],
        'se': model3.bse['treated_post'],
        'pvalue': model3.pvalues['treated_post'],
        'n': model3.nobs
    },
    'model4_yearfe': {
        'estimate': model4.params['treated_post'],
        'se': model4.bse['treated_post'],
        'pvalue': model4.pvalues['treated_post'],
        'n': model4.nobs
    },
    'model5_statefe': {
        'estimate': model5.params['treated_post'],
        'se': model5.bse['treated_post'],
        'pvalue': model5.pvalues['treated_post'],
        'n': model5.nobs
    },
    'model_robust': {
        'estimate': model_robust.params['treated_post'],
        'se': model_robust.bse['treated_post'],
        'pvalue': model_robust.pvalues['treated_post'],
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n': model_robust.nobs
    }
}

# Save to CSV
results_df = pd.DataFrame(results).T
results_df.to_csv('results/regression_results.csv')

# Save descriptive statistics
desc_stats = {
    'mean_pre_treat': mean_pre_treat,
    'mean_post_treat': mean_post_treat,
    'mean_pre_control': mean_pre_control,
    'mean_post_control': mean_post_control,
    'simple_did': simple_did,
    'n_treatment': int(df['treated'].sum()),
    'n_control': int((1-df['treated']).sum()),
    'n_pre': int((1-df['post']).sum()),
    'n_post': int(df['post'].sum()),
    'n_total': len(df)
}

desc_df = pd.DataFrame([desc_stats])
desc_df.to_csv('results/descriptive_stats.csv', index=False)

# Save event study coefficients
event_results = []
for year in event_years:
    event_results.append({
        'year': year,
        'coefficient': model_event.params[f'treat_x_{year}'],
        'se': model_event.bse[f'treat_x_{year}'],
        'pvalue': model_event.pvalues[f'treat_x_{year}']
    })
event_df = pd.DataFrame(event_results)
event_df.to_csv('results/event_study.csv', index=False)

# =============================================================================
# STEP 12: SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print(f"""
Research Question: Effect of DACA eligibility on full-time employment

Sample:
  - Hispanic-Mexican individuals born in Mexico
  - Non-citizens who arrived in US before age 16 and by 2007
  - Treatment group: Ages 26-30 as of June 15, 2012 (N = {df['treated'].sum():,})
  - Control group: Ages 31-35 as of June 15, 2012 (N = {(1-df['treated']).sum():,})
  - Years: 2006-2011 (pre), 2013-2016 (post)
  - Total observations: {len(df):,}

Descriptive Statistics:
  - Treatment pre-DACA full-time rate: {mean_pre_treat:.3f}
  - Treatment post-DACA full-time rate: {mean_post_treat:.3f}
  - Control pre-DACA full-time rate: {mean_pre_control:.3f}
  - Control post-DACA full-time rate: {mean_post_control:.3f}
  - Simple DiD estimate: {simple_did:.4f}

PREFERRED ESTIMATE (Model with state-clustered SE):
  - DiD coefficient: {model_robust.params['treated_post']:.4f}
  - Standard error: {model_robust.bse['treated_post']:.4f}
  - 95% CI: [{ci_low:.4f}, {ci_high:.4f}]
  - p-value: {model_robust.pvalues['treated_post']:.4f}

Interpretation:
  DACA eligibility is associated with a {abs(model_robust.params['treated_post'])*100:.2f} percentage point
  {'increase' if model_robust.params['treated_post'] > 0 else 'decrease'} in full-time employment
  for the treatment group relative to the control group.
  {'This effect is statistically significant at the 5% level.' if model_robust.pvalues['treated_post'] < 0.05 else 'This effect is not statistically significant at the 5% level.'}
""")

print("\nAnalysis complete. Results saved to results/ folder.")
