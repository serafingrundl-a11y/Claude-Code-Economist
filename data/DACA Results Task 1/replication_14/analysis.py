"""
DACA Replication Study - Analysis Script
Research Question: Impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals

Processing in chunks due to large file size
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# STEP 1: LOAD DATA IN CHUNKS AND FILTER
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA (Processing in chunks)")
print("="*80)

# Define the columns we need
needed_cols = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT',
               'MARST', 'UHRSWORK']

# Process in chunks and filter immediately
print("Loading and filtering ACS data from data.csv...")
chunks = []
chunk_size = 500000
total_rows = 0
kept_rows = 0

for chunk in pd.read_csv('data/data.csv', chunksize=chunk_size, usecols=needed_cols, low_memory=False):
    total_rows += len(chunk)

    # Filter to Hispanic-Mexican (HISPAN == 1), Mexican-born (BPL == 200), Non-citizen (CITIZEN == 3)
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3)].copy()
    kept_rows += len(filtered)

    if len(filtered) > 0:
        chunks.append(filtered)

    print(f"  Processed {total_rows:,} rows, kept {kept_rows:,} rows so far...")
    gc.collect()

# Combine all filtered chunks
print("\nCombining filtered data...")
df_sample = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"\nTotal rows in original data: {total_rows:,}")
print(f"Rows after filtering (Hispanic-Mexican, Mexican-born, Non-citizen): {len(df_sample):,}")

# =============================================================================
# STEP 2: CREATE AGE-RELATED VARIABLES
# =============================================================================
print("\n" + "="*80)
print("STEP 2: CREATE AGE-RELATED VARIABLES")
print("="*80)

# Check YRIMMIG values
print(f"YRIMMIG range: {df_sample['YRIMMIG'].min()} to {df_sample['YRIMMIG'].max()}")

# Filter out missing/invalid YRIMMIG (0 is N/A per data dictionary)
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
print(f"After removing YRIMMIG==0 (missing): {len(df_sample):,}")

# Calculate age at immigration
df_sample['age_at_immigration'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
print(f"Age at immigration range: {df_sample['age_at_immigration'].min()} to {df_sample['age_at_immigration'].max()}")

# =============================================================================
# STEP 3: DEFINE DACA ELIGIBILITY CRITERIA
# =============================================================================
print("\n" + "="*80)
print("STEP 3: DEFINE DACA ELIGIBILITY")
print("="*80)

"""
DACA Eligibility Criteria:
1. Arrived in US before 16th birthday
2. Under 31 years old as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (immigrated by 2007)
4. Not a citizen (already filtered)
"""

# Criterion 1: Arrived before age 16
df_sample['arrived_before_16'] = df_sample['age_at_immigration'] < 16

# Criterion 2: Under 31 on June 15, 2012
# Born after June 15, 1981 means BIRTHYR >= 1982, OR BIRTHYR == 1981 with BIRTHQTR >= 3
df_sample['under_31_june2012'] = (
    (df_sample['BIRTHYR'] >= 1982) |
    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))
)

# Criterion 3: Continuous presence since June 15, 2007 (immigrated by 2007)
df_sample['present_since_2007'] = df_sample['YRIMMIG'] <= 2007

# Combined DACA eligibility
df_sample['daca_eligible'] = (
    df_sample['arrived_before_16'] &
    df_sample['under_31_june2012'] &
    df_sample['present_since_2007']
)

print(f"\nDACA Eligibility Components:")
print(f"  Arrived before age 16: {df_sample['arrived_before_16'].sum():,} ({100*df_sample['arrived_before_16'].mean():.1f}%)")
print(f"  Under 31 on June 15, 2012: {df_sample['under_31_june2012'].sum():,} ({100*df_sample['under_31_june2012'].mean():.1f}%)")
print(f"  Present since 2007: {df_sample['present_since_2007'].sum():,} ({100*df_sample['present_since_2007'].mean():.1f}%)")
print(f"  DACA Eligible (all criteria): {df_sample['daca_eligible'].sum():,} ({100*df_sample['daca_eligible'].mean():.1f}%)")

# =============================================================================
# STEP 4: DEFINE TIME PERIODS AND POST-DACA INDICATOR
# =============================================================================
print("\n" + "="*80)
print("STEP 4: DEFINE TIME PERIODS")
print("="*80)

print(f"\nYear distribution:")
print(df_sample['YEAR'].value_counts().sort_index())

# Create post-DACA indicator (2013-2016)
df_sample['post_daca'] = df_sample['YEAR'] >= 2013
df_sample['year_2012'] = df_sample['YEAR'] == 2012

print(f"\nPost-DACA (2013-2016): {df_sample['post_daca'].sum():,}")
print(f"Pre-DACA (2006-2012): {(~df_sample['post_daca']).sum():,}")

# =============================================================================
# STEP 5: DEFINE OUTCOME VARIABLE - FULL-TIME EMPLOYMENT
# =============================================================================
print("\n" + "="*80)
print("STEP 5: DEFINE OUTCOME VARIABLE")
print("="*80)

print(f"\nUHRSWORK distribution:")
print(df_sample['UHRSWORK'].describe())

print(f"\nEMPSTAT distribution:")
print(df_sample['EMPSTAT'].value_counts().sort_index())

# Full-time employment
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)
df_sample['fulltime_unconditional'] = ((df_sample['EMPSTAT'] == 1) & (df_sample['UHRSWORK'] >= 35)).astype(int)

print(f"\nFull-time (35+ hours | employed): {df_sample[df_sample['employed']==1]['fulltime'].mean():.3f}")
print(f"Full-time (35+ hours | all): {df_sample['fulltime_unconditional'].mean():.3f}")
print(f"Employed: {df_sample['employed'].mean():.3f}")

# =============================================================================
# STEP 6: PREPARE ANALYSIS SAMPLE
# =============================================================================
print("\n" + "="*80)
print("STEP 6: PREPARE ANALYSIS SAMPLE")
print("="*80)

# Restrict to working-age population (16-65)
df_sample = df_sample[(df_sample['AGE'] >= 16) & (df_sample['AGE'] <= 65)].copy()
print(f"After restricting to ages 16-65: {len(df_sample):,}")

# Exclude 2012 for main analysis (implementation year)
df_analysis = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

# Create numeric variables
df_analysis['eligible'] = df_analysis['daca_eligible'].astype(int)
df_analysis['post'] = df_analysis['post_daca'].astype(int)
df_analysis['eligible_x_post'] = df_analysis['eligible'] * df_analysis['post']

print(f"\nAnalysis sample breakdown:")
print(f"  DACA-eligible, Pre-period: {((df_analysis['daca_eligible']) & (~df_analysis['post_daca'])).sum():,}")
print(f"  DACA-eligible, Post-period: {((df_analysis['daca_eligible']) & (df_analysis['post_daca'])).sum():,}")
print(f"  Non-eligible, Pre-period: {((~df_analysis['daca_eligible']) & (~df_analysis['post_daca'])).sum():,}")
print(f"  Non-eligible, Post-period: {((~df_analysis['daca_eligible']) & (df_analysis['post_daca'])).sum():,}")

# =============================================================================
# STEP 7: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("STEP 7: DESCRIPTIVE STATISTICS")
print("="*80)

# Summary by eligibility and period
print("\n--- Full-Time Employment Rates ---")
for eligible in [True, False]:
    for post in [False, True]:
        subset = df_analysis[(df_analysis['daca_eligible']==eligible) & (df_analysis['post_daca']==post)]
        elig_str = "Eligible" if eligible else "Non-Eligible"
        period_str = "Post-DACA" if post else "Pre-DACA"

        # Weighted mean
        ft_rate = np.average(subset['fulltime_unconditional'], weights=subset['PERWT'])
        emp_rate = np.average(subset['employed'], weights=subset['PERWT'])
        n = len(subset)
        print(f"{elig_str:12} | {period_str:10} | N={n:,} | Employed={emp_rate:.3f} | Full-time={ft_rate:.3f}")

# Simple DiD
def weighted_mean(x, w):
    return np.average(x, weights=w)

pre_treat = weighted_mean(
    df_analysis[(df_analysis['daca_eligible']) & (~df_analysis['post_daca'])]['fulltime_unconditional'],
    df_analysis[(df_analysis['daca_eligible']) & (~df_analysis['post_daca'])]['PERWT']
)
post_treat = weighted_mean(
    df_analysis[(df_analysis['daca_eligible']) & (df_analysis['post_daca'])]['fulltime_unconditional'],
    df_analysis[(df_analysis['daca_eligible']) & (df_analysis['post_daca'])]['PERWT']
)
pre_control = weighted_mean(
    df_analysis[(~df_analysis['daca_eligible']) & (~df_analysis['post_daca'])]['fulltime_unconditional'],
    df_analysis[(~df_analysis['daca_eligible']) & (~df_analysis['post_daca'])]['PERWT']
)
post_control = weighted_mean(
    df_analysis[(~df_analysis['daca_eligible']) & (df_analysis['post_daca'])]['fulltime_unconditional'],
    df_analysis[(~df_analysis['daca_eligible']) & (df_analysis['post_daca'])]['PERWT']
)

simple_did = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n--- Simple Difference-in-Differences (Weighted) ---")
print(f"Treatment group change: {post_treat:.4f} - {pre_treat:.4f} = {post_treat - pre_treat:.4f}")
print(f"Control group change: {post_control:.4f} - {pre_control:.4f} = {post_control - pre_control:.4f}")
print(f"DiD estimate: {simple_did:.4f}")

# =============================================================================
# STEP 8: CREATE CONTROL VARIABLES
# =============================================================================
print("\n" + "="*80)
print("STEP 8: CREATE CONTROL VARIABLES")
print("="*80)

df_analysis['age_sq'] = df_analysis['AGE'] ** 2
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)

print("Control variables created: age_sq, female, married, educ_hs")

# =============================================================================
# STEP 9: MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("STEP 9: DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("="*80)

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.wls('fulltime_unconditional ~ eligible + post + eligible_x_post',
                  data=df_analysis, weights=df_analysis['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(f"DiD Coefficient: {results1.params['eligible_x_post']:.4f}")
print(f"SE: {results1.bse['eligible_x_post']:.4f}")
print(f"p-value: {results1.pvalues['eligible_x_post']:.4f}")
print(f"N: {int(results1.nobs):,}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls('fulltime_unconditional ~ eligible + post + eligible_x_post + AGE + age_sq + female + married + educ_hs',
                  data=df_analysis, weights=df_analysis['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(f"DiD Coefficient: {results2.params['eligible_x_post']:.4f}")
print(f"SE: {results2.bse['eligible_x_post']:.4f}")
print(f"p-value: {results2.pvalues['eligible_x_post']:.4f}")

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
model3 = smf.wls('fulltime_unconditional ~ eligible + eligible_x_post + C(YEAR) + AGE + age_sq + female + married + educ_hs',
                  data=df_analysis, weights=df_analysis['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(f"DiD Coefficient: {results3.params['eligible_x_post']:.4f}")
print(f"SE: {results3.bse['eligible_x_post']:.4f}")
print(f"p-value: {results3.pvalues['eligible_x_post']:.4f}")

# Model 4: DiD with state and year fixed effects
print("\n--- Model 4: DiD with State and Year Fixed Effects ---")
model4 = smf.wls('fulltime_unconditional ~ eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs',
                  data=df_analysis, weights=df_analysis['PERWT'])
results4 = model4.fit(cov_type='HC1')

coef = results4.params['eligible_x_post']
se = results4.bse['eligible_x_post']
ci_low = coef - 1.96*se
ci_high = coef + 1.96*se
pval = results4.pvalues['eligible_x_post']

print(f"\n*** PREFERRED ESTIMATE (Model 4) ***")
print(f"DiD Coefficient: {coef:.4f}")
print(f"Std. Error: {se:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"p-value: {pval:.4f}")

# =============================================================================
# STEP 10: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("STEP 10: ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Alternative control group - only those who arrived after age 16
print("\n--- Robustness 1: Control = Arrived After Age 16 ---")
df_robust1 = df_analysis[
    (df_analysis['daca_eligible']) |
    ((~df_analysis['arrived_before_16']) & (df_analysis['under_31_june2012']) & (df_analysis['present_since_2007']))
].copy()
df_robust1['eligible'] = df_robust1['daca_eligible'].astype(int)
df_robust1['eligible_x_post'] = df_robust1['eligible'] * df_robust1['post']

model_r1 = smf.wls('fulltime_unconditional ~ eligible + eligible_x_post + C(YEAR) + AGE + age_sq + female + married + educ_hs',
                    data=df_robust1, weights=df_robust1['PERWT'])
results_r1 = model_r1.fit(cov_type='HC1')
print(f"DiD Coefficient: {results_r1.params['eligible_x_post']:.4f} (SE: {results_r1.bse['eligible_x_post']:.4f})")
print(f"N = {len(df_robust1):,}")

# Robustness 2: Alternative control group - those born before 1981
print("\n--- Robustness 2: Control = Born Before 1981 (Too Old for DACA) ---")
df_robust2 = df_analysis[
    (df_analysis['daca_eligible']) |
    ((df_analysis['arrived_before_16']) & (~df_analysis['under_31_june2012']) & (df_analysis['present_since_2007']))
].copy()
if len(df_robust2) > 0:
    df_robust2['eligible'] = df_robust2['daca_eligible'].astype(int)
    df_robust2['eligible_x_post'] = df_robust2['eligible'] * df_robust2['post']

    model_r2 = smf.wls('fulltime_unconditional ~ eligible + eligible_x_post + C(YEAR) + AGE + age_sq + female + married + educ_hs',
                        data=df_robust2, weights=df_robust2['PERWT'])
    results_r2 = model_r2.fit(cov_type='HC1')
    print(f"DiD Coefficient: {results_r2.params['eligible_x_post']:.4f} (SE: {results_r2.bse['eligible_x_post']:.4f})")
    print(f"N = {len(df_robust2):,}")
else:
    print("Insufficient sample size for this comparison")

# Robustness 3: Employed (extensive margin) as outcome
print("\n--- Robustness 3: Employment (Extensive Margin) as Outcome ---")
model_r3 = smf.wls('employed ~ eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs',
                    data=df_analysis, weights=df_analysis['PERWT'])
results_r3 = model_r3.fit(cov_type='HC1')
print(f"DiD Coefficient: {results_r3.params['eligible_x_post']:.4f} (SE: {results_r3.bse['eligible_x_post']:.4f})")

# Robustness 4: Full-time conditional on employment
print("\n--- Robustness 4: Full-time Conditional on Employment ---")
df_employed = df_analysis[df_analysis['employed']==1].copy()
model_r4 = smf.wls('fulltime ~ eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs',
                    data=df_employed, weights=df_employed['PERWT'])
results_r4 = model_r4.fit(cov_type='HC1')
print(f"DiD Coefficient: {results_r4.params['eligible_x_post']:.4f} (SE: {results_r4.bse['eligible_x_post']:.4f})")
print(f"N (employed only) = {len(df_employed):,}")

# =============================================================================
# STEP 11: PLACEBO TEST
# =============================================================================
print("\n" + "="*80)
print("STEP 11: PLACEBO TEST")
print("="*80)

# Placebo: Use only pre-period data, set fake treatment at 2010
df_pre = df_analysis[df_analysis['YEAR'] < 2012].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['eligible_x_placebo'] = df_pre['eligible'] * df_pre['placebo_post']

model_placebo = smf.wls('fulltime_unconditional ~ eligible + placebo_post + eligible_x_placebo + AGE + age_sq + female + married + educ_hs',
                         data=df_pre, weights=df_pre['PERWT'])
results_placebo = model_placebo.fit(cov_type='HC1')
print(f"Placebo DiD (fake treatment at 2010): {results_placebo.params['eligible_x_placebo']:.4f}")
print(f"SE: {results_placebo.bse['eligible_x_placebo']:.4f}")
print(f"p-value: {results_placebo.pvalues['eligible_x_placebo']:.4f}")

# =============================================================================
# STEP 12: EVENT STUDY
# =============================================================================
print("\n" + "="*80)
print("STEP 12: EVENT STUDY")
print("="*80)

# Create year-specific treatment effects (relative to 2011)
base_year = 2011
years = sorted(df_analysis['YEAR'].unique())

for year in years:
    if year != base_year:
        df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype(int)
        df_analysis[f'eligible_x_year_{year}'] = df_analysis['eligible'] * df_analysis[f'year_{year}']

# Create the formula for event study
year_dummies = ' + '.join([f'year_{y}' for y in years if y != base_year])
interactions = ' + '.join([f'eligible_x_year_{y}' for y in years if y != base_year])
formula = f'fulltime_unconditional ~ eligible + {year_dummies} + {interactions} + AGE + age_sq + female + married + educ_hs'

model_event = smf.wls(formula, data=df_analysis, weights=df_analysis['PERWT'])
results_event = model_event.fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 60)
event_data = []
for year in years:
    if year == base_year:
        print(f"{year}: 0.0000 (reference)")
        event_data.append({'year': year, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
    else:
        coef_ev = results_event.params[f'eligible_x_year_{year}']
        se_ev = results_event.bse[f'eligible_x_year_{year}']
        print(f"{year}: {coef_ev:.4f} (SE: {se_ev:.4f})")
        event_data.append({
            'year': year,
            'coef': coef_ev,
            'se': se_ev,
            'ci_low': coef_ev - 1.96*se_ev,
            'ci_high': coef_ev + 1.96*se_ev
        })

# =============================================================================
# STEP 13: HETEROGENEITY ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("STEP 13: HETEROGENEITY ANALYSIS")
print("="*80)

# By gender
print("\n--- By Gender ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex].copy()
    model = smf.wls('fulltime_unconditional ~ eligible + eligible_x_post + C(YEAR) + AGE + age_sq + married + educ_hs',
                    data=df_sub, weights=df_sub['PERWT'])
    results = model.fit(cov_type='HC1')
    print(f"{label}: DiD = {results.params['eligible_x_post']:.4f} (SE: {results.bse['eligible_x_post']:.4f}), N = {len(df_sub):,}")

# By education
print("\n--- By Education Level ---")
for educ_level, label in [(0, 'Less than HS'), (1, 'HS or Higher')]:
    df_sub = df_analysis[df_analysis['educ_hs'] == educ_level].copy()
    if len(df_sub) > 100:
        model = smf.wls('fulltime_unconditional ~ eligible + eligible_x_post + C(YEAR) + AGE + age_sq + female + married',
                        data=df_sub, weights=df_sub['PERWT'])
        results = model.fit(cov_type='HC1')
        print(f"{label}: DiD = {results.params['eligible_x_post']:.4f} (SE: {results.bse['eligible_x_post']:.4f}), N = {len(df_sub):,}")

# =============================================================================
# STEP 14: SAVE KEY RESULTS
# =============================================================================
print("\n" + "="*80)
print("STEP 14: SAVE RESULTS")
print("="*80)

# Save summary table
summary_data = {
    'Model': ['Basic DiD', 'With Demographics', 'Year FE', 'State + Year FE'],
    'Coefficient': [
        results1.params['eligible_x_post'],
        results2.params['eligible_x_post'],
        results3.params['eligible_x_post'],
        results4.params['eligible_x_post']
    ],
    'SE': [
        results1.bse['eligible_x_post'],
        results2.bse['eligible_x_post'],
        results3.bse['eligible_x_post'],
        results4.bse['eligible_x_post']
    ],
    'pvalue': [
        results1.pvalues['eligible_x_post'],
        results2.pvalues['eligible_x_post'],
        results3.pvalues['eligible_x_post'],
        results4.pvalues['eligible_x_post']
    ],
    'N': [int(results1.nobs), int(results2.nobs), int(results3.nobs), int(results4.nobs)]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save trends data
trends = df_analysis.groupby(['YEAR', 'eligible']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime_unconditional'], weights=x['PERWT']),
        'employment_rate': np.average(x['employed'], weights=x['PERWT']),
        'n': len(x)
    })
).reset_index()
trends.to_csv('trends_data.csv', index=False)
print("Trends data saved to trends_data.csv")

# Save event study data
event_df = pd.DataFrame(event_data)
event_df.to_csv('event_study_data.csv', index=False)
print("Event study data saved to event_study_data.csv")

# Save robustness results
robustness_data = {
    'Check': ['Alt Control: Arrived After 16', 'Alt Control: Born Before 1981',
              'Outcome: Employment', 'Outcome: FT|Employed', 'Placebo (2010)'],
    'Coefficient': [
        results_r1.params['eligible_x_post'],
        results_r2.params['eligible_x_post'] if len(df_robust2) > 0 else np.nan,
        results_r3.params['eligible_x_post'],
        results_r4.params['eligible_x_post'],
        results_placebo.params['eligible_x_placebo']
    ],
    'SE': [
        results_r1.bse['eligible_x_post'],
        results_r2.bse['eligible_x_post'] if len(df_robust2) > 0 else np.nan,
        results_r3.bse['eligible_x_post'],
        results_r4.bse['eligible_x_post'],
        results_placebo.bse['eligible_x_placebo']
    ]
}
robustness_df = pd.DataFrame(robustness_data)
robustness_df.to_csv('robustness_results.csv', index=False)
print("Robustness results saved to robustness_results.csv")

# Save descriptive stats
desc_data = []
for eligible in [True, False]:
    for post in [False, True]:
        subset = df_analysis[(df_analysis['daca_eligible']==eligible) & (df_analysis['post_daca']==post)]
        elig_str = "Eligible" if eligible else "Non-Eligible"
        period_str = "Post" if post else "Pre"
        desc_data.append({
            'Group': elig_str,
            'Period': period_str,
            'N': len(subset),
            'Fulltime_Rate': np.average(subset['fulltime_unconditional'], weights=subset['PERWT']),
            'Employment_Rate': np.average(subset['employed'], weights=subset['PERWT']),
            'Mean_Age': np.average(subset['AGE'], weights=subset['PERWT']),
            'Pct_Female': np.average(subset['female'], weights=subset['PERWT']),
            'Pct_Married': np.average(subset['married'], weights=subset['PERWT']),
            'Pct_HS': np.average(subset['educ_hs'], weights=subset['PERWT'])
        })
desc_df = pd.DataFrame(desc_data)
desc_df.to_csv('descriptive_stats.csv', index=False)
print("Descriptive stats saved to descriptive_stats.csv")

# Save heterogeneity results
het_results = []

# By gender
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex].copy()
    model = smf.wls('fulltime_unconditional ~ eligible + eligible_x_post + C(YEAR) + AGE + age_sq + married + educ_hs',
                    data=df_sub, weights=df_sub['PERWT'])
    results = model.fit(cov_type='HC1')
    het_results.append({
        'Subgroup': f'Gender: {label}',
        'Coefficient': results.params['eligible_x_post'],
        'SE': results.bse['eligible_x_post'],
        'N': len(df_sub)
    })

# By education
for educ_level, label in [(0, 'Less than HS'), (1, 'HS or Higher')]:
    df_sub = df_analysis[df_analysis['educ_hs'] == educ_level].copy()
    if len(df_sub) > 100:
        model = smf.wls('fulltime_unconditional ~ eligible + eligible_x_post + C(YEAR) + AGE + age_sq + female + married',
                        data=df_sub, weights=df_sub['PERWT'])
        results = model.fit(cov_type='HC1')
        het_results.append({
            'Subgroup': f'Education: {label}',
            'Coefficient': results.params['eligible_x_post'],
            'SE': results.bse['eligible_x_post'],
            'N': len(df_sub)
        })

het_df = pd.DataFrame(het_results)
het_df.to_csv('heterogeneity_results.csv', index=False)
print("Heterogeneity results saved to heterogeneity_results.csv")

print("\n" + "="*80)
print("FINAL SUMMARY - PREFERRED ESTIMATE")
print("="*80)
print(f"\nEffect of DACA Eligibility on Full-Time Employment:")
print(f"  DiD Coefficient: {results4.params['eligible_x_post']:.4f}")
print(f"  Standard Error: {results4.bse['eligible_x_post']:.4f}")
print(f"  95% CI: [{results4.params['eligible_x_post'] - 1.96*results4.bse['eligible_x_post']:.4f}, {results4.params['eligible_x_post'] + 1.96*results4.bse['eligible_x_post']:.4f}]")
print(f"  p-value: {results4.pvalues['eligible_x_post']:.4f}")
print(f"  Sample Size: {int(results4.nobs):,}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
