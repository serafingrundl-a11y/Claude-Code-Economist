"""
DACA Replication Study - Analysis Script
Research Question: Impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US.
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

print("="*80)
print("DACA REPLICATION STUDY - DATA ANALYSIS")
print("="*80)

# Load data in chunks and filter as we go
print("\n[1] Loading and filtering data in chunks...")
data_path = "data/data.csv"

# Only keep needed columns and filter during load
# HISPAN == 1 (Mexican) and BPL == 200 (Mexico)
chunks = []
chunk_size = 1000000
for chunk in pd.read_csv(data_path, chunksize=chunk_size,
                          usecols=['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR',
                                  'MARST', 'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
                                  'EDUC', 'EMPSTAT', 'UHRSWORK']):
    # Filter to Hispanic-Mexican, Mexican-born
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"  Processed chunk, kept {len(filtered):,} rows")

df_mex = pd.concat(chunks, ignore_index=True)
print(f"\nTotal Hispanic-Mexican, Mexican-born: {len(df_mex):,}")

# Clear memory
del chunks

# Step 2: Examine citizenship status
print("\n[2] Citizenship Status (CITIZEN) among Mexican-born:")
print(df_mex['CITIZEN'].value_counts())
# CITIZEN: 0=N/A, 1=Born abroad of American parents, 2=Naturalized, 3=Not a citizen

# Focus on non-citizens (proxy for potentially undocumented)
df_noncit = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"\nAfter filtering to non-citizens: {len(df_noncit):,}")

# Clear memory
del df_mex

# Step 3: Examine year of immigration
print("\n[3] Year of Immigration (YRIMMIG) distribution:")
print(df_noncit['YRIMMIG'].describe())

# Step 4: Create DACA eligibility criteria
print("\n[4] Creating DACA Eligibility Indicator:")

# Calculate age at arrival
df_noncit['age_at_arrival'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']

# Filter valid immigration years (not 0 which means N/A)
df_valid = df_noncit[df_noncit['YRIMMIG'] > 0].copy()
print(f"After filtering valid YRIMMIG: {len(df_valid):,}")

# Clear memory
del df_noncit

# Create eligibility indicator
# DACA Eligibility Criteria:
# 1. Arrived unlawfully before 16th birthday: (YRIMMIG - BIRTHYR) < 16
# 2. Age < 31 as of June 15, 2012: BIRTHYR > 1981 (more precisely, born after June 15, 1981)
# 3. Continuous presence since June 15, 2007: YRIMMIG <= 2007
# 4. Not a citizen (already filtered)

# Criterion 1: Arrived before 16th birthday
df_valid['arrived_before_16'] = (df_valid['age_at_arrival'] < 16).astype(int)

# Criterion 2: Under 31 as of June 15, 2012 (born after June 15, 1981)
# Being conservative: born in 1982 or later definitely eligible
df_valid['under_31_in_2012'] = (df_valid['BIRTHYR'] >= 1982).astype(int)

# Criterion 3: In US since June 15, 2007 (arrived 2007 or earlier)
df_valid['in_us_since_2007'] = (df_valid['YRIMMIG'] <= 2007).astype(int)

# Combined eligibility
df_valid['daca_eligible'] = ((df_valid['arrived_before_16'] == 1) &
                              (df_valid['under_31_in_2012'] == 1) &
                              (df_valid['in_us_since_2007'] == 1)).astype(int)

print(f"DACA eligibility breakdown:")
print(f"  Arrived before 16: {df_valid['arrived_before_16'].sum():,}")
print(f"  Under 31 in 2012: {df_valid['under_31_in_2012'].sum():,}")
print(f"  In US since 2007: {df_valid['in_us_since_2007'].sum():,}")
print(f"  DACA eligible: {df_valid['daca_eligible'].sum():,}")

# Step 5: Create outcome variable - Full-time employment
print("\n[5] Creating Full-Time Employment Indicator:")
print(f"UHRSWORK distribution:")
print(df_valid['UHRSWORK'].describe())

# Full-time = 35+ hours per week
df_valid['fulltime'] = (df_valid['UHRSWORK'] >= 35).astype(int)
print(f"Full-time employed (35+ hrs): {df_valid['fulltime'].sum():,} ({df_valid['fulltime'].mean()*100:.1f}%)")

# Step 6: Create treatment period indicator
print("\n[6] Creating Treatment Period Indicator:")
# Exclude 2012 (implementation year)
df_analysis = df_valid[df_valid['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

# Clear memory
del df_valid

# Post = 1 for 2013-2016, 0 for 2006-2011
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
print(f"Pre-period (2006-2011): {(df_analysis['post'] == 0).sum():,}")
print(f"Post-period (2013-2016): {(df_analysis['post'] == 1).sum():,}")

# Step 7: Create control variables
print("\n[7] Creating Control Variables:")

# Sex (SEX: 1=Male, 2=Female)
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Marital status (MARST: 1=Married spouse present, 6=Never married, etc.)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Education (EDUC: simplified categories)
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # High school or higher
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # Some college or higher

# Age squared
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

print(f"Female: {df_analysis['female'].mean()*100:.1f}%")
print(f"Married: {df_analysis['married'].mean()*100:.1f}%")
print(f"High school+: {df_analysis['educ_hs'].mean()*100:.1f}%")
print(f"Mean age: {df_analysis['AGE'].mean():.1f}")

# Step 8: Sample restrictions for working-age population
print("\n[8] Restricting to Working-Age Population (16-64):")
df_working = df_analysis[(df_analysis['AGE'] >= 16) & (df_analysis['AGE'] <= 64)].copy()
print(f"After restricting to ages 16-64: {len(df_working):,}")

# Clear memory
del df_analysis

# Final analysis sample
print("\n[9] Final Analysis Sample Summary:")
print(f"Total observations: {len(df_working):,}")
print(f"DACA eligible: {df_working['daca_eligible'].sum():,}")
print(f"DACA ineligible: {(df_working['daca_eligible'] == 0).sum():,}")

# Summary by eligibility and period
print("\n[10] Sample by Eligibility and Period:")
summary = df_working.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'PERWT': 'sum'
}).round(3)
print(summary)

# Unweighted means
print("\n[11] Full-time Employment Rates (Unweighted):")
for elig in [0, 1]:
    for post in [0, 1]:
        subset = df_working[(df_working['daca_eligible'] == elig) & (df_working['post'] == post)]
        rate = subset['fulltime'].mean()
        n = len(subset)
        elig_label = "Eligible" if elig == 1 else "Ineligible"
        period_label = "Post" if post == 1 else "Pre"
        print(f"  {elig_label}, {period_label}: {rate*100:.2f}% (n={n:,})")

# Calculate simple DID
print("\n[12] Simple Difference-in-Differences Calculation:")
pre_elig = df_working[(df_working['daca_eligible'] == 1) & (df_working['post'] == 0)]['fulltime'].mean()
post_elig = df_working[(df_working['daca_eligible'] == 1) & (df_working['post'] == 1)]['fulltime'].mean()
pre_inelig = df_working[(df_working['daca_eligible'] == 0) & (df_working['post'] == 0)]['fulltime'].mean()
post_inelig = df_working[(df_working['daca_eligible'] == 0) & (df_working['post'] == 1)]['fulltime'].mean()

diff_elig = post_elig - pre_elig
diff_inelig = post_inelig - pre_inelig
did_simple = diff_elig - diff_inelig

print(f"Eligible: {pre_elig*100:.2f}% -> {post_elig*100:.2f}% (change: {diff_elig*100:.2f}pp)")
print(f"Ineligible: {pre_inelig*100:.2f}% -> {post_inelig*100:.2f}% (change: {diff_inelig*100:.2f}pp)")
print(f"Simple DID estimate: {did_simple*100:.2f} percentage points")

# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("REGRESSION ANALYSIS")
print("="*80)

# Create interaction term
df_working['post_eligible'] = df_working['post'] * df_working['daca_eligible']

# Model 1: Basic DID without controls
print("\n[13] Model 1: Basic DID (No Controls)")
model1 = smf.ols('fulltime ~ post + daca_eligible + post_eligible', data=df_working).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DID with demographic controls
print("\n[14] Model 2: DID with Demographic Controls")
model2 = smf.ols('fulltime ~ post + daca_eligible + post_eligible + AGE + age_sq + female + married + educ_hs',
                 data=df_working).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DID with year fixed effects
print("\n[15] Model 3: DID with Year Fixed Effects")
df_working['year_factor'] = pd.Categorical(df_working['YEAR'])
model3 = smf.ols('fulltime ~ daca_eligible + post_eligible + AGE + age_sq + female + married + educ_hs + C(YEAR)',
                 data=df_working).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DID with state and year fixed effects
print("\n[16] Model 4: DID with State and Year Fixed Effects")
model4 = smf.ols('fulltime ~ daca_eligible + post_eligible + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                 data=df_working).fit(cov_type='HC1')
print("\nModel 4 - Key Coefficients:")
print(f"  post_eligible (DID): {model4.params['post_eligible']:.4f} (SE: {model4.bse['post_eligible']:.4f})")
print(f"  t-stat: {model4.tvalues['post_eligible']:.3f}, p-value: {model4.pvalues['post_eligible']:.4f}")

# Weighted regression
print("\n[17] Model 5: Weighted DID with State and Year FE")
model5 = smf.wls('fulltime ~ daca_eligible + post_eligible + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                 data=df_working, weights=df_working['PERWT']).fit(cov_type='HC1')
print("\nModel 5 (Weighted) - Key Coefficients:")
print(f"  post_eligible (DID): {model5.params['post_eligible']:.4f} (SE: {model5.bse['post_eligible']:.4f})")
print(f"  t-stat: {model5.tvalues['post_eligible']:.3f}, p-value: {model5.pvalues['post_eligible']:.4f}")

# ============================================================================
# ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Alternative age restriction (18-35)
print("\n[18] Robustness 1: Restricted to ages 18-35")
df_young = df_working[(df_working['AGE'] >= 18) & (df_working['AGE'] <= 35)].copy()
df_young['post_eligible'] = df_young['post'] * df_young['daca_eligible']
model_r1 = smf.ols('fulltime ~ daca_eligible + post_eligible + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                   data=df_young).fit(cov_type='HC1')
print(f"Sample size: {len(df_young):,}")
print(f"DID estimate: {model_r1.params['post_eligible']:.4f} (SE: {model_r1.bse['post_eligible']:.4f})")

# Robustness 2: Men only
print("\n[19] Robustness 2: Men Only")
df_men = df_working[df_working['female'] == 0].copy()
model_r2 = smf.ols('fulltime ~ daca_eligible + post_eligible + AGE + age_sq + married + educ_hs + C(YEAR) + C(STATEFIP)',
                   data=df_men).fit(cov_type='HC1')
print(f"Sample size: {len(df_men):,}")
print(f"DID estimate: {model_r2.params['post_eligible']:.4f} (SE: {model_r2.bse['post_eligible']:.4f})")

# Robustness 3: Women only
print("\n[20] Robustness 3: Women Only")
df_women = df_working[df_working['female'] == 1].copy()
model_r3 = smf.ols('fulltime ~ daca_eligible + post_eligible + AGE + age_sq + married + educ_hs + C(YEAR) + C(STATEFIP)',
                   data=df_women).fit(cov_type='HC1')
print(f"Sample size: {len(df_women):,}")
print(f"DID estimate: {model_r3.params['post_eligible']:.4f} (SE: {model_r3.bse['post_eligible']:.4f})")

# Robustness 4: Placebo test - pre-period only (2006-2008 vs 2009-2011)
print("\n[21] Robustness 4: Placebo Test (Pre-period: 2006-2008 vs 2009-2011)")
df_pre = df_working[df_working['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_interaction'] = df_pre['placebo_post'] * df_pre['daca_eligible']
model_r4 = smf.ols('fulltime ~ daca_eligible + placebo_interaction + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                   data=df_pre).fit(cov_type='HC1')
print(f"Sample size: {len(df_pre):,}")
print(f"Placebo DID estimate: {model_r4.params['placebo_interaction']:.4f} (SE: {model_r4.bse['placebo_interaction']:.4f})")
print(f"p-value: {model_r4.pvalues['placebo_interaction']:.4f}")

# Robustness 5: Employment (any hours) instead of full-time
print("\n[22] Robustness 5: Any Employment (UHRSWORK > 0)")
df_working['employed'] = (df_working['UHRSWORK'] > 0).astype(int)
model_r5 = smf.ols('employed ~ daca_eligible + post_eligible + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                   data=df_working).fit(cov_type='HC1')
print(f"DID estimate: {model_r5.params['post_eligible']:.4f} (SE: {model_r5.bse['post_eligible']:.4f})")

# Event study
print("\n[23] Event Study - Year-by-Year Effects")
df_working['eligible_x_2006'] = (df_working['YEAR'] == 2006) * df_working['daca_eligible']
df_working['eligible_x_2007'] = (df_working['YEAR'] == 2007) * df_working['daca_eligible']
df_working['eligible_x_2008'] = (df_working['YEAR'] == 2008) * df_working['daca_eligible']
df_working['eligible_x_2009'] = (df_working['YEAR'] == 2009) * df_working['daca_eligible']
df_working['eligible_x_2010'] = (df_working['YEAR'] == 2010) * df_working['daca_eligible']
# 2011 is reference year
df_working['eligible_x_2013'] = (df_working['YEAR'] == 2013) * df_working['daca_eligible']
df_working['eligible_x_2014'] = (df_working['YEAR'] == 2014) * df_working['daca_eligible']
df_working['eligible_x_2015'] = (df_working['YEAR'] == 2015) * df_working['daca_eligible']
df_working['eligible_x_2016'] = (df_working['YEAR'] == 2016) * df_working['daca_eligible']

model_event = smf.ols('''fulltime ~ daca_eligible +
                         eligible_x_2006 + eligible_x_2007 + eligible_x_2008 + eligible_x_2009 + eligible_x_2010 +
                         eligible_x_2013 + eligible_x_2014 + eligible_x_2015 + eligible_x_2016 +
                         AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)''',
                      data=df_working).fit(cov_type='HC1')

print("Event Study Coefficients:")
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'eligible_x_{year}']
    se = model_event.bse[f'eligible_x_{year}']
    print(f"  {year}: {coef:.4f} (SE: {se:.4f})")

# ============================================================================
# SAVE RESULTS FOR REPORT
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF KEY RESULTS")
print("="*80)

print("\n[PREFERRED ESTIMATE - Model 4: DID with State and Year FE]")
print(f"Effect Size: {model4.params['post_eligible']:.4f}")
print(f"Standard Error: {model4.bse['post_eligible']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['post_eligible', 0]:.4f}, {model4.conf_int().loc['post_eligible', 1]:.4f}]")
print(f"t-statistic: {model4.tvalues['post_eligible']:.3f}")
print(f"p-value: {model4.pvalues['post_eligible']:.4f}")
print(f"Sample Size: {int(model4.nobs):,}")
print(f"R-squared: {model4.rsquared:.4f}")

# Save results to file
results_summary = {
    'Model': ['Basic DID', 'With Demographics', 'Year FE', 'Year + State FE', 'Weighted'],
    'Coefficient': [model1.params['post_eligible'], model2.params['post_eligible'],
                    model3.params['post_eligible'], model4.params['post_eligible'],
                    model5.params['post_eligible']],
    'SE': [model1.bse['post_eligible'], model2.bse['post_eligible'],
           model3.bse['post_eligible'], model4.bse['post_eligible'],
           model5.bse['post_eligible']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)]
}
results_df = pd.DataFrame(results_summary)
results_df.to_csv('regression_results.csv', index=False)
print("\nResults saved to regression_results.csv")

# Save event study results
event_results = []
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'eligible_x_{year}']
    se = model_event.bse[f'eligible_x_{year}']
    event_results.append({'Year': year, 'Coefficient': coef, 'SE': se})
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# ============================================================================
# ADDITIONAL DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

print("\n[24] Sample Characteristics by DACA Eligibility:")
for elig in [0, 1]:
    subset = df_working[df_working['daca_eligible'] == elig]
    elig_label = "DACA Eligible" if elig == 1 else "DACA Ineligible"
    print(f"\n{elig_label} (N = {len(subset):,}):")
    print(f"  Mean age: {subset['AGE'].mean():.1f}")
    print(f"  Female: {subset['female'].mean()*100:.1f}%")
    print(f"  Married: {subset['married'].mean()*100:.1f}%")
    print(f"  High school+: {subset['educ_hs'].mean()*100:.1f}%")
    print(f"  Full-time employed: {subset['fulltime'].mean()*100:.1f}%")
    print(f"  Mean hours worked: {subset['UHRSWORK'].mean():.1f}")

# Save descriptive stats
desc_stats = df_working.groupby('daca_eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'fulltime': 'mean',
    'UHRSWORK': 'mean',
    'PERWT': 'sum'
}).round(3)
desc_stats.columns = ['Mean Age', 'Female', 'Married', 'HS+', 'Full-time', 'Mean Hours', 'Pop Weight']
desc_stats.to_csv('descriptive_stats.csv')
print("\nDescriptive statistics saved to descriptive_stats.csv")

# Year by year sample sizes
print("\n[25] Sample Size by Year and Eligibility:")
year_counts = df_working.groupby(['YEAR', 'daca_eligible']).size().unstack()
year_counts.columns = ['Ineligible', 'Eligible']
print(year_counts)
year_counts.to_csv('sample_by_year.csv')

# Additional results for robustness
robustness_results = {
    'Model': ['Ages 18-35', 'Men Only', 'Women Only', 'Placebo', 'Any Employment'],
    'Coefficient': [model_r1.params['post_eligible'], model_r2.params['post_eligible'],
                    model_r3.params['post_eligible'], model_r4.params['placebo_interaction'],
                    model_r5.params['post_eligible']],
    'SE': [model_r1.bse['post_eligible'], model_r2.bse['post_eligible'],
           model_r3.bse['post_eligible'], model_r4.bse['placebo_interaction'],
           model_r5.bse['post_eligible']],
    'N': [int(model_r1.nobs), int(model_r2.nobs), int(model_r3.nobs), int(model_r4.nobs), int(model_r5.nobs)]
}
robustness_df = pd.DataFrame(robustness_results)
robustness_df.to_csv('robustness_results.csv', index=False)
print("\nRobustness results saved to robustness_results.csv")

# Full model summary for Model 4
with open('model4_summary.txt', 'w') as f:
    f.write(model4.summary().as_text())
print("Full Model 4 summary saved to model4_summary.txt")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
