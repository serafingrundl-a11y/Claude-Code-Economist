"""
DACA Impact on Full-Time Employment - Replication 97
Difference-in-Differences Analysis

Research Question: Among ethnically Hispanic-Mexican, Mexican-born people living in the US,
what was the causal impact of DACA eligibility on full-time employment (>=35 hrs/week)?

Treatment: Ages 26-30 as of June 15, 2012 (DACA eligible)
Control: Ages 31-35 as of June 15, 2012 (just over age cutoff)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import json
import os

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 2\replication_97")

print("="*80)
print("DACA REPLICATION STUDY - Analysis 97")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND FILTER DATA
# ============================================================================
print("\n[1] Loading data...")

# Read relevant columns only to manage memory
cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'MARST',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'LABFORCE',
               'UHRSWORK', 'STATEFIP', 'METRO']

# Load data in chunks to manage memory
chunks = []
chunksize = 500000

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunksize):
    # Filter to Hispanic-Mexican only (HISPAN == 1)
    chunk = chunk[chunk['HISPAN'] == 1]
    # Filter to born in Mexico (BPL == 200)
    chunk = chunk[chunk['BPL'] == 200]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f"   Initial sample (Hispanic-Mexican, born in Mexico): {len(df):,}")

# ============================================================================
# STEP 2: APPLY DACA ELIGIBILITY FILTERS
# ============================================================================
print("\n[2] Applying eligibility filters...")

# Filter to non-citizens only (proxy for undocumented status)
# CITIZEN=3 means "Not a citizen"
df = df[df['CITIZEN'] == 3]
print(f"   After non-citizen filter: {len(df):,}")

# Calculate age as of June 15, 2012
# Since we don't have exact birth month, we use BIRTHYR
# Age on June 15, 2012 = 2012 - BIRTHYR (approximate, could be off by ~1 year)
df['age_2012'] = 2012 - df['BIRTHYR']

# DACA requirement: Arrived in US before 16th birthday
# age_at_arrival = YRIMMIG - BIRTHYR
# Require: age_at_arrival < 16
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_arrival'] < 16]
print(f"   After arrived-before-16 filter: {len(df):,}")

# DACA requirement: Lived continuously in US since June 15, 2007
# Proxy: YRIMMIG <= 2007
df = df[df['YRIMMIG'] <= 2007]
print(f"   After continuous residence (YRIMMIG<=2007) filter: {len(df):,}")

# ============================================================================
# STEP 3: DEFINE TREATMENT AND CONTROL GROUPS
# ============================================================================
print("\n[3] Defining treatment/control groups...")

# Treatment group: Ages 26-30 as of June 15, 2012 (DACA eligible)
# Control group: Ages 31-35 as of June 15, 2012 (just missed cutoff)

# Create treatment indicator
df['treated'] = ((df['age_2012'] >= 26) & (df['age_2012'] <= 30)).astype(int)
df['control'] = ((df['age_2012'] >= 31) & (df['age_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df = df[(df['treated'] == 1) | (df['control'] == 1)]
print(f"   Sample with treatment/control groups: {len(df):,}")

# ============================================================================
# STEP 4: DEFINE PRE AND POST PERIODS
# ============================================================================
print("\n[4] Defining time periods...")

# Pre-period: 2006-2011 (before DACA)
# We exclude 2012 because DACA was implemented mid-year
# Post-period: 2013-2016 (after DACA fully in effect)

df['post'] = (df['YEAR'] >= 2013).astype(int)
df = df[(df['YEAR'] <= 2011) | (df['YEAR'] >= 2013)]  # Exclude 2012
print(f"   After excluding 2012: {len(df):,}")

# ============================================================================
# STEP 5: CREATE OUTCOME VARIABLE
# ============================================================================
print("\n[5] Creating outcome variable...")

# Full-time employment = Usually working 35+ hours per week
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Alternative: Employed (EMPSTAT==1) AND working 35+ hours
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
df['fulltime_employed'] = ((df['EMPSTAT'] == 1) & (df['UHRSWORK'] >= 35)).astype(int)

# ============================================================================
# STEP 6: CREATE ADDITIONAL COVARIATES
# ============================================================================
print("\n[6] Creating covariates...")

# Age in survey year
df['age_survey'] = df['YEAR'] - df['BIRTHYR']

# Female indicator
df['female'] = (df['SEX'] == 2).astype(int)

# Married indicator
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_less_hs'] = (df['EDUCD'] < 62).astype(int)  # Less than high school
df['educ_hs'] = ((df['EDUCD'] >= 62) & (df['EDUCD'] <= 64)).astype(int)  # HS diploma
df['educ_some_college'] = ((df['EDUCD'] >= 65) & (df['EDUCD'] < 101)).astype(int)
df['educ_college'] = (df['EDUCD'] >= 101).astype(int)

# Metro area indicator
df['metro'] = (df['METRO'].isin([2, 3, 4])).astype(int)

# ============================================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================================
print("\n[7] Summary Statistics")
print("="*80)

# Sample sizes by group and period
print("\nSample sizes (unweighted):")
cross = pd.crosstab(df['treated'], df['post'], margins=True)
cross.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
cross.columns = ['Pre (2006-2011)', 'Post (2013-2016)', 'Total']
print(cross)

# Weighted sample sizes
print("\nWeighted sample sizes:")
weighted_cross = df.groupby(['treated', 'post'])['PERWT'].sum().unstack()
weighted_cross.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_cross.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(weighted_cross.round(0))

# Full-time employment rates by group and period
print("\nFull-time employment rates (weighted):")
ft_rates = df.groupby(['treated', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print((ft_rates * 100).round(2))

# Calculate simple DiD
pre_diff = ft_rates.loc['Treatment (26-30)', 'Pre (2006-2011)'] - ft_rates.loc['Control (31-35)', 'Pre (2006-2011)']
post_diff = ft_rates.loc['Treatment (26-30)', 'Post (2013-2016)'] - ft_rates.loc['Control (31-35)', 'Post (2013-2016)']
simple_did = post_diff - pre_diff
print(f"\nSimple DiD estimate: {simple_did*100:.2f} percentage points")

# ============================================================================
# STEP 8: DIFFERENCE-IN-DIFFERENCES REGRESSION
# ============================================================================
print("\n[8] Difference-in-Differences Regression Analysis")
print("="*80)

# Create interaction term
df['treated_post'] = df['treated'] * df['post']

# Create year dummies
df['year_dummies'] = pd.Categorical(df['YEAR'])

# Basic DiD model (no covariates)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df, weights=df['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary().tables[1])

# DiD with year fixed effects
print("\n--- Model 2: DiD with Year Fixed Effects ---")
model2 = smf.wls('fulltime ~ treated + treated_post + C(YEAR)',
                  data=df, weights=df['PERWT'])
results2 = model2.fit(cov_type='HC1')
# Print only key coefficients
print(f"treated_post coefficient: {results2.params['treated_post']:.4f}")
print(f"Std. Error: {results2.bse['treated_post']:.4f}")
print(f"t-statistic: {results2.tvalues['treated_post']:.4f}")
print(f"p-value: {results2.pvalues['treated_post']:.4f}")
print(f"95% CI: [{results2.conf_int().loc['treated_post', 0]:.4f}, {results2.conf_int().loc['treated_post', 1]:.4f}]")

# DiD with covariates
print("\n--- Model 3: DiD with Covariates ---")
model3 = smf.wls('fulltime ~ treated + treated_post + C(YEAR) + female + married + educ_hs + educ_some_college + educ_college + metro',
                  data=df, weights=df['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(f"treated_post coefficient: {results3.params['treated_post']:.4f}")
print(f"Std. Error: {results3.bse['treated_post']:.4f}")
print(f"t-statistic: {results3.tvalues['treated_post']:.4f}")
print(f"p-value: {results3.pvalues['treated_post']:.4f}")
print(f"95% CI: [{results3.conf_int().loc['treated_post', 0]:.4f}, {results3.conf_int().loc['treated_post', 1]:.4f}]")

# DiD with state fixed effects
print("\n--- Model 4: DiD with State Fixed Effects ---")
model4 = smf.wls('fulltime ~ treated + treated_post + C(YEAR) + C(STATEFIP) + female + married + educ_hs + educ_some_college + educ_college',
                  data=df, weights=df['PERWT'])
results4 = model4.fit(cov_type='HC1')
print(f"treated_post coefficient: {results4.params['treated_post']:.4f}")
print(f"Std. Error: {results4.bse['treated_post']:.4f}")
print(f"t-statistic: {results4.tvalues['treated_post']:.4f}")
print(f"p-value: {results4.pvalues['treated_post']:.4f}")
print(f"95% CI: [{results4.conf_int().loc['treated_post', 0]:.4f}, {results4.conf_int().loc['treated_post', 1]:.4f}]")

# ============================================================================
# STEP 9: PREFERRED SPECIFICATION
# ============================================================================
print("\n[9] PREFERRED SPECIFICATION - Model with Year and State FE + Covariates")
print("="*80)

# Full results for preferred model
preferred = results4
print(f"\nPreferred Estimate:")
print(f"   Effect on Full-Time Employment: {preferred.params['treated_post']:.4f}")
print(f"   Standard Error: {preferred.bse['treated_post']:.4f}")
print(f"   95% Confidence Interval: [{preferred.conf_int().loc['treated_post', 0]:.4f}, {preferred.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   p-value: {preferred.pvalues['treated_post']:.4f}")
print(f"   Sample Size: {int(preferred.nobs):,}")

# ============================================================================
# STEP 10: ROBUSTNESS CHECKS
# ============================================================================
print("\n[10] Robustness Checks")
print("="*80)

# Check 1: Employment (not just full-time)
print("\n--- Robustness Check 1: Any Employment ---")
model_emp = smf.wls('employed ~ treated + treated_post + C(YEAR) + C(STATEFIP) + female + married + educ_hs + educ_some_college + educ_college',
                     data=df, weights=df['PERWT'])
results_emp = model_emp.fit(cov_type='HC1')
print(f"treated_post coefficient: {results_emp.params['treated_post']:.4f} (SE: {results_emp.bse['treated_post']:.4f})")

# Check 2: Men only
print("\n--- Robustness Check 2: Men Only ---")
df_men = df[df['female'] == 0]
model_men = smf.wls('fulltime ~ treated + treated_post + C(YEAR) + C(STATEFIP) + married + educ_hs + educ_some_college + educ_college',
                     data=df_men, weights=df_men['PERWT'])
results_men = model_men.fit(cov_type='HC1')
print(f"treated_post coefficient: {results_men.params['treated_post']:.4f} (SE: {results_men.bse['treated_post']:.4f})")
print(f"Sample size: {int(results_men.nobs):,}")

# Check 3: Women only
print("\n--- Robustness Check 3: Women Only ---")
df_women = df[df['female'] == 1]
model_women = smf.wls('fulltime ~ treated + treated_post + C(YEAR) + C(STATEFIP) + married + educ_hs + educ_some_college + educ_college',
                       data=df_women, weights=df_women['PERWT'])
results_women = model_women.fit(cov_type='HC1')
print(f"treated_post coefficient: {results_women.params['treated_post']:.4f} (SE: {results_women.bse['treated_post']:.4f})")
print(f"Sample size: {int(results_women.nobs):,}")

# Check 4: Alternative age bands (24-28 vs 33-37)
print("\n--- Robustness Check 4: Narrower Age Bands (27-29 vs 32-34) ---")
df_narrow = df[(df['age_2012'].isin([27, 28, 29])) | (df['age_2012'].isin([32, 33, 34]))]
df_narrow['treated'] = df_narrow['age_2012'].isin([27, 28, 29]).astype(int)
df_narrow['treated_post'] = df_narrow['treated'] * df_narrow['post']
model_narrow = smf.wls('fulltime ~ treated + treated_post + C(YEAR) + C(STATEFIP) + female + married + educ_hs + educ_some_college + educ_college',
                        data=df_narrow, weights=df_narrow['PERWT'])
results_narrow = model_narrow.fit(cov_type='HC1')
print(f"treated_post coefficient: {results_narrow.params['treated_post']:.4f} (SE: {results_narrow.bse['treated_post']:.4f})")
print(f"Sample size: {int(results_narrow.nobs):,}")

# ============================================================================
# STEP 11: PARALLEL TRENDS CHECK
# ============================================================================
print("\n[11] Parallel Trends Analysis")
print("="*80)

# Calculate full-time employment by year and group
trends = df.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
trends.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\nFull-time employment rates by year:")
print((trends * 100).round(2))

# Pre-trend test: interact treatment with year in pre-period
print("\n--- Pre-Trend Test: Event Study ---")
df_pre = df[df['post'] == 0].copy()
df_pre['year_2011'] = (df_pre['YEAR'] == 2011).astype(int)
df_pre['year_2010'] = (df_pre['YEAR'] == 2010).astype(int)
df_pre['year_2009'] = (df_pre['YEAR'] == 2009).astype(int)
df_pre['year_2008'] = (df_pre['YEAR'] == 2008).astype(int)
df_pre['year_2007'] = (df_pre['YEAR'] == 2007).astype(int)
# 2006 is reference

df_pre['treat_2011'] = df_pre['treated'] * df_pre['year_2011']
df_pre['treat_2010'] = df_pre['treated'] * df_pre['year_2010']
df_pre['treat_2009'] = df_pre['treated'] * df_pre['year_2009']
df_pre['treat_2008'] = df_pre['treated'] * df_pre['year_2008']
df_pre['treat_2007'] = df_pre['treated'] * df_pre['year_2007']

pretrend_model = smf.wls('fulltime ~ treated + C(YEAR) + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2011',
                          data=df_pre, weights=df_pre['PERWT'])
pretrend_results = pretrend_model.fit(cov_type='HC1')

print("Pre-trend interaction coefficients:")
for var in ['treat_2007', 'treat_2008', 'treat_2009', 'treat_2010', 'treat_2011']:
    print(f"  {var}: {pretrend_results.params[var]:.4f} (SE: {pretrend_results.bse[var]:.4f}, p: {pretrend_results.pvalues[var]:.4f})")

# Joint F-test for pre-trends
from scipy.stats import f as f_dist
r_matrix = np.zeros((5, len(pretrend_results.params)))
for i, var in enumerate(['treat_2007', 'treat_2008', 'treat_2009', 'treat_2010', 'treat_2011']):
    r_matrix[i, list(pretrend_results.params.index).index(var)] = 1
f_test = pretrend_results.f_test(r_matrix)
try:
    fval = f_test.fvalue[0][0] if hasattr(f_test.fvalue, '__getitem__') else f_test.fvalue
    print(f"\nJoint F-test for pre-trends: F={fval:.4f}, p-value={f_test.pvalue:.4f}")
except:
    print(f"\nJoint F-test for pre-trends: F={f_test.fvalue}, p-value={f_test.pvalue}")

# ============================================================================
# STEP 12: EVENT STUDY / DYNAMIC EFFECTS
# ============================================================================
print("\n[12] Event Study - Dynamic Effects")
print("="*80)

# Create event study with year-specific treatment effects
df['treat_2007'] = df['treated'] * (df['YEAR'] == 2007)
df['treat_2008'] = df['treated'] * (df['YEAR'] == 2008)
df['treat_2009'] = df['treated'] * (df['YEAR'] == 2009)
df['treat_2010'] = df['treated'] * (df['YEAR'] == 2010)
df['treat_2011'] = df['treated'] * (df['YEAR'] == 2011)
# 2012 excluded
df['treat_2013'] = df['treated'] * (df['YEAR'] == 2013)
df['treat_2014'] = df['treated'] * (df['YEAR'] == 2014)
df['treat_2015'] = df['treated'] * (df['YEAR'] == 2015)
df['treat_2016'] = df['treated'] * (df['YEAR'] == 2016)

event_model = smf.wls('fulltime ~ treated + C(YEAR) + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2011 + treat_2013 + treat_2014 + treat_2015 + treat_2016 + female + married + educ_hs + educ_some_college + educ_college',
                       data=df, weights=df['PERWT'])
event_results = event_model.fit(cov_type='HC1')

print("Event study coefficients (2006 is reference):")
event_years = ['treat_2007', 'treat_2008', 'treat_2009', 'treat_2010', 'treat_2011',
               'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
for var in event_years:
    print(f"  {var}: {event_results.params[var]:.4f} (SE: {event_results.bse[var]:.4f})")

# ============================================================================
# STEP 13: SAVE RESULTS AND CREATE FIGURES
# ============================================================================
print("\n[13] Creating Figures")
print("="*80)

# Figure 1: Parallel trends plot
fig1, ax1 = plt.subplots(figsize=(10, 6))
years = trends.index.tolist()
ax1.plot(years, trends['Treatment (26-30)'] * 100, 'b-o', label='Treatment (26-30)', linewidth=2, markersize=8)
ax1.plot(years, trends['Control (31-35)'] * 100, 'r--s', label='Control (31-35)', linewidth=2, markersize=8)
ax1.axvline(x=2012, color='gray', linestyle='--', label='DACA Implementation')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate (%)', fontsize=12)
ax1.set_title('Full-Time Employment Rates by Treatment Status', fontsize=14)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(years)
plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: figure1_parallel_trends.png")

# Figure 2: Event study plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
event_coefs = [event_results.params[var] for var in event_years]
event_ses = [event_results.bse[var] for var in event_years]
event_years_num = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]

# Add zero for 2006 (reference) and missing 2012
plot_years = [2006] + event_years_num[:5] + [2012] + event_years_num[5:]
plot_coefs = [0] + event_coefs[:5] + [np.nan] + event_coefs[5:]
plot_ses = [0] + event_ses[:5] + [0] + event_ses[5:]

ax2.errorbar(plot_years, [c*100 for c in plot_coefs],
             yerr=[1.96*s*100 for s in plot_ses],
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax2.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation', linewidth=2)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Effect on Full-Time Employment (pp)', fontsize=12)
ax2.set_title('Event Study: Dynamic Treatment Effects', fontsize=14)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(plot_years)
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: figure2_event_study.png")

# ============================================================================
# STEP 14: SAVE SUMMARY FOR REPORT
# ============================================================================
print("\n[14] Saving Results Summary")
print("="*80)

results_summary = {
    'preferred_estimate': {
        'coefficient': float(preferred.params['treated_post']),
        'se': float(preferred.bse['treated_post']),
        'ci_lower': float(preferred.conf_int().loc['treated_post', 0]),
        'ci_upper': float(preferred.conf_int().loc['treated_post', 1]),
        'pvalue': float(preferred.pvalues['treated_post']),
        'n': int(preferred.nobs)
    },
    'simple_did': float(simple_did),
    'model1_basic': {
        'coefficient': float(results1.params['treated_post']),
        'se': float(results1.bse['treated_post']),
        'n': int(results1.nobs)
    },
    'model2_year_fe': {
        'coefficient': float(results2.params['treated_post']),
        'se': float(results2.bse['treated_post']),
        'n': int(results2.nobs)
    },
    'model3_covariates': {
        'coefficient': float(results3.params['treated_post']),
        'se': float(results3.bse['treated_post']),
        'n': int(results3.nobs)
    },
    'robustness': {
        'employment': {
            'coefficient': float(results_emp.params['treated_post']),
            'se': float(results_emp.bse['treated_post'])
        },
        'men_only': {
            'coefficient': float(results_men.params['treated_post']),
            'se': float(results_men.bse['treated_post']),
            'n': int(results_men.nobs)
        },
        'women_only': {
            'coefficient': float(results_women.params['treated_post']),
            'se': float(results_women.bse['treated_post']),
            'n': int(results_women.nobs)
        },
        'narrow_ages': {
            'coefficient': float(results_narrow.params['treated_post']),
            'se': float(results_narrow.bse['treated_post']),
            'n': int(results_narrow.nobs)
        }
    },
    'sample_sizes': {
        'treatment_pre': int(len(df[(df['treated']==1) & (df['post']==0)])),
        'treatment_post': int(len(df[(df['treated']==1) & (df['post']==1)])),
        'control_pre': int(len(df[(df['treated']==0) & (df['post']==0)])),
        'control_post': int(len(df[(df['treated']==0) & (df['post']==1)]))
    },
    'ft_rates': {
        'treatment_pre': float(ft_rates.loc['Treatment (26-30)', 'Pre (2006-2011)']),
        'treatment_post': float(ft_rates.loc['Treatment (26-30)', 'Post (2013-2016)']),
        'control_pre': float(ft_rates.loc['Control (31-35)', 'Pre (2006-2011)']),
        'control_post': float(ft_rates.loc['Control (31-35)', 'Post (2013-2016)'])
    }
}

with open('results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print("   Saved: results_summary.json")

# Save covariate balance table
print("\n[15] Covariate Balance Table")
print("="*80)

balance_vars = ['female', 'married', 'educ_less_hs', 'educ_hs', 'educ_some_college', 'educ_college', 'age_at_arrival']
balance_data = []

for var in balance_vars:
    treat_mean = np.average(df[df['treated']==1][var], weights=df[df['treated']==1]['PERWT'])
    control_mean = np.average(df[df['treated']==0][var], weights=df[df['treated']==0]['PERWT'])
    diff = treat_mean - control_mean
    balance_data.append({
        'Variable': var,
        'Treatment': treat_mean,
        'Control': control_mean,
        'Difference': diff
    })

balance_df = pd.DataFrame(balance_data)
print(balance_df.to_string(index=False))

# ============================================================================
# FINAL OUTPUT
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"\nPreferred Specification: DiD with Year FE, State FE, and Covariates")
print(f"\nEffect of DACA Eligibility on Full-Time Employment:")
print(f"   Coefficient: {preferred.params['treated_post']:.4f} ({preferred.params['treated_post']*100:.2f} percentage points)")
print(f"   Standard Error: {preferred.bse['treated_post']:.4f}")
print(f"   95% CI: [{preferred.conf_int().loc['treated_post', 0]:.4f}, {preferred.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   p-value: {preferred.pvalues['treated_post']:.4f}")
print(f"   Sample Size: {int(preferred.nobs):,}")
print(f"\nInterpretation: DACA eligibility {'increased' if preferred.params['treated_post'] > 0 else 'decreased'} the probability")
print(f"of full-time employment by {abs(preferred.params['treated_post']*100):.2f} percentage points.")

if preferred.pvalues['treated_post'] < 0.05:
    print(f"This effect is statistically significant at the 5% level.")
else:
    print(f"This effect is NOT statistically significant at the 5% level.")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
