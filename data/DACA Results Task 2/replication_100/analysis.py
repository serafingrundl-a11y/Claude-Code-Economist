"""
DACA Replication Analysis - Replication 100
Effect of DACA eligibility on full-time employment among Mexican-born Hispanic individuals

Research Design: Difference-in-Differences
Treatment: DACA-eligible individuals aged 26-30 as of June 15, 2012
Control: Individuals aged 31-35 as of June 15, 2012 (would be eligible but for age)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DACA REPLICATION ANALYSIS - Replication 100")
print("=" * 80)

# =============================================================================
# STEP 1: Load and Explore Data
# =============================================================================
print("\n[STEP 1] Loading data...")

# Read the data
df = pd.read_csv('data/data.csv')
print(f"Total observations in raw data: {len(df):,}")
print(f"Years available: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: Sample Selection - Identify DACA-eligible population
# =============================================================================
print("\n[STEP 2] Applying sample selection criteria...")

# Criterion 1: Hispanic-Mexican ethnicity
# HISPAN = 1 indicates Mexican
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican filter: {len(df_sample):,}")

# Criterion 2: Born in Mexico (BPL = 200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After Mexico birthplace filter: {len(df_sample):,}")

# Criterion 3: Not a citizen (CITIZEN = 3)
# We assume non-citizens who are not naturalized are undocumented
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After non-citizen filter: {len(df_sample):,}")

# Criterion 4: Arrived before age 16
# Calculate age at immigration
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_immig'] < 16].copy()
print(f"After arrived before age 16 filter: {len(df_sample):,}")

# Criterion 5: Lived in US since June 15, 2007
# YRIMMIG <= 2007
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
print(f"After continuous presence since 2007 filter: {len(df_sample):,}")

# =============================================================================
# STEP 3: Define Treatment and Control Groups
# =============================================================================
print("\n[STEP 3] Defining treatment and control groups...")

# Treatment: Ages 26-30 as of June 15, 2012
# This means birthyear 1982-1986 (born between June 16, 1981 and June 15, 1986 but simplify to birth years)
# Control: Ages 31-35 as of June 15, 2012
# This means birthyear 1977-1981

# Calculate age as of June 15, 2012
# For simplicity, use birthyear to define cohorts
# Treatment: Born 1982-1986 (would be 26-30 on June 15, 2012)
# Control: Born 1977-1981 (would be 31-35 on June 15, 2012)

df_sample['treat_cohort'] = ((df_sample['BIRTHYR'] >= 1982) & (df_sample['BIRTHYR'] <= 1986)).astype(int)
df_sample['control_cohort'] = ((df_sample['BIRTHYR'] >= 1977) & (df_sample['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control cohorts
df_sample = df_sample[(df_sample['treat_cohort'] == 1) | (df_sample['control_cohort'] == 1)].copy()
print(f"After keeping treatment/control cohorts: {len(df_sample):,}")

# Create treatment indicator (1 for treatment group, 0 for control)
df_sample['treated'] = df_sample['treat_cohort']

# =============================================================================
# STEP 4: Define Time Periods
# =============================================================================
print("\n[STEP 4] Defining time periods...")

# Exclude 2012 due to ambiguity about pre/post status
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# Post period: 2013-2016
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

print(f"\nPre-period years: {sorted(df_sample[df_sample['post']==0]['YEAR'].unique())}")
print(f"Post-period years: {sorted(df_sample[df_sample['post']==1]['YEAR'].unique())}")

# =============================================================================
# STEP 5: Create Outcome Variable
# =============================================================================
print("\n[STEP 5] Creating outcome variable...")

# Full-time employment: Usually working 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Alternative: Employment status
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

print(f"Full-time employment rate: {df_sample['fulltime'].mean():.4f}")
print(f"Employment rate: {df_sample['employed'].mean():.4f}")

# =============================================================================
# STEP 6: Create Additional Controls
# =============================================================================
print("\n[STEP 6] Creating control variables...")

# Age in survey year
df_sample['age'] = df_sample['AGE']

# Female indicator
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)

# Married indicator
df_sample['married'] = (df_sample['MARST'] <= 2).astype(int)

# Education categories
df_sample['educ_lesshs'] = (df_sample['EDUC'] < 6).astype(int)
df_sample['educ_hs'] = (df_sample['EDUC'] == 6).astype(int)
df_sample['educ_somecoll'] = ((df_sample['EDUC'] >= 7) & (df_sample['EDUC'] <= 9)).astype(int)
df_sample['educ_coll'] = (df_sample['EDUC'] >= 10).astype(int)

# Years in US (continuous)
df_sample['years_in_us'] = df_sample['YRSUSA1']

# State fixed effects - convert to categorical
df_sample['state'] = df_sample['STATEFIP'].astype(str)

# DiD interaction term
df_sample['did'] = df_sample['treated'] * df_sample['post']

print(f"\nSample by group:")
print(f"Treatment (age 26-30): {(df_sample['treated']==1).sum():,}")
print(f"Control (age 31-35): {(df_sample['treated']==0).sum():,}")
print(f"\nBy period:")
print(f"Pre-period: {(df_sample['post']==0).sum():,}")
print(f"Post-period: {(df_sample['post']==1).sum():,}")

# =============================================================================
# STEP 7: Summary Statistics
# =============================================================================
print("\n[STEP 7] Generating summary statistics...")

summary_vars = ['fulltime', 'employed', 'age', 'female', 'married',
                'educ_lesshs', 'educ_hs', 'educ_somecoll', 'educ_coll', 'UHRSWORK']

# Pre-period summary by treatment status
pre_data = df_sample[df_sample['post'] == 0]
pre_treat = pre_data[pre_data['treated'] == 1][summary_vars].describe()
pre_ctrl = pre_data[pre_data['treated'] == 0][summary_vars].describe()

print("\n--- Pre-Period Summary Statistics ---")
print("\nTreatment Group (Ages 26-30 in 2012):")
print(pre_treat.loc[['count', 'mean', 'std']])
print("\nControl Group (Ages 31-35 in 2012):")
print(pre_ctrl.loc[['count', 'mean', 'std']])

# Full sample summary
print("\n--- Full Sample Summary ---")
for var in ['fulltime', 'employed', 'female', 'married']:
    mean_val = df_sample[var].mean()
    print(f"{var}: {mean_val:.4f}")

# =============================================================================
# STEP 8: Difference-in-Differences Analysis
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 8] Difference-in-Differences Regression Analysis")
print("=" * 80)

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (OLS, unweighted) ---")
model1 = smf.ols('fulltime ~ treated + post + did', data=df_sample).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.ols('fulltime ~ treated + post + did + female + married + educ_hs + educ_somecoll + educ_coll',
                 data=df_sample).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
model3 = smf.ols('fulltime ~ treated + did + female + married + educ_hs + educ_somecoll + educ_coll + C(YEAR)',
                 data=df_sample).fit(cov_type='HC1')
print(f"DiD coefficient: {model3.params['did']:.6f}")
print(f"SE (robust): {model3.bse['did']:.6f}")
print(f"t-stat: {model3.tvalues['did']:.4f}")
print(f"p-value: {model3.pvalues['did']:.6f}")

# Model 4: DiD with state and year fixed effects
print("\n--- Model 4: DiD with State and Year Fixed Effects ---")
model4 = smf.ols('fulltime ~ treated + did + female + married + educ_hs + educ_somecoll + educ_coll + C(YEAR) + C(STATEFIP)',
                 data=df_sample).fit(cov_type='HC1')
print(f"DiD coefficient: {model4.params['did']:.6f}")
print(f"SE (robust): {model4.bse['did']:.6f}")
print(f"t-stat: {model4.tvalues['did']:.4f}")
print(f"p-value: {model4.pvalues['did']:.6f}")

# Model 5: Weighted regression using person weights
print("\n--- Model 5: Weighted DiD with Controls ---")
import statsmodels.api as sm

# Prepare data for weighted regression
X = df_sample[['treated', 'post', 'did', 'female', 'married', 'educ_hs', 'educ_somecoll', 'educ_coll']].copy()
X = sm.add_constant(X)
y = df_sample['fulltime']
weights = df_sample['PERWT']

model5 = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')
print(model5.summary().tables[1])

# =============================================================================
# STEP 9: Robustness Checks
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 9] Robustness Checks")
print("=" * 80)

# Check 1: Placebo test - fake treatment in 2009 (pre-period only)
print("\n--- Robustness Check 1: Placebo Test (Fake Treatment in 2009) ---")
pre_only = df_sample[df_sample['post'] == 0].copy()
pre_only['fake_post'] = (pre_only['YEAR'] >= 2009).astype(int)
pre_only['fake_did'] = pre_only['treated'] * pre_only['fake_post']

placebo_model = smf.ols('fulltime ~ treated + fake_post + fake_did + female + married + educ_hs + educ_somecoll + educ_coll',
                        data=pre_only).fit(cov_type='HC1')
print(f"Placebo DiD coefficient: {placebo_model.params['fake_did']:.6f}")
print(f"SE (robust): {placebo_model.bse['fake_did']:.6f}")
print(f"p-value: {placebo_model.pvalues['fake_did']:.6f}")

# Check 2: Event study
print("\n--- Robustness Check 2: Event Study Coefficients ---")
# Create year interactions
df_sample['year_2006'] = (df_sample['YEAR'] == 2006).astype(int)
df_sample['year_2007'] = (df_sample['YEAR'] == 2007).astype(int)
df_sample['year_2008'] = (df_sample['YEAR'] == 2008).astype(int)
df_sample['year_2009'] = (df_sample['YEAR'] == 2009).astype(int)
df_sample['year_2010'] = (df_sample['YEAR'] == 2010).astype(int)
df_sample['year_2011'] = (df_sample['YEAR'] == 2011).astype(int)
df_sample['year_2013'] = (df_sample['YEAR'] == 2013).astype(int)
df_sample['year_2014'] = (df_sample['YEAR'] == 2014).astype(int)
df_sample['year_2015'] = (df_sample['YEAR'] == 2015).astype(int)
df_sample['year_2016'] = (df_sample['YEAR'] == 2016).astype(int)

# Interactions with treatment (omit 2011 as reference)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_sample[f'treat_x_{yr}'] = df_sample['treated'] * df_sample[f'year_{yr}']

event_formula = 'fulltime ~ treated + treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + C(YEAR) + female + married + educ_hs + educ_somecoll + educ_coll'
event_model = smf.ols(event_formula, data=df_sample).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_results = []
for yr in event_years:
    coef = event_model.params[f'treat_x_{yr}']
    se = event_model.bse[f'treat_x_{yr}']
    pval = event_model.pvalues[f'treat_x_{yr}']
    event_results.append({'Year': yr, 'Coefficient': coef, 'SE': se, 'p-value': pval})
    print(f"  {yr}: {coef:.6f} (SE: {se:.6f}, p={pval:.4f})")

event_df = pd.DataFrame(event_results)

# =============================================================================
# STEP 10: 2x2 DiD Table
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 10] 2x2 Difference-in-Differences Table")
print("=" * 80)

# Calculate group means
pre_treat_mean = df_sample[(df_sample['treated']==1) & (df_sample['post']==0)]['fulltime'].mean()
pre_ctrl_mean = df_sample[(df_sample['treated']==0) & (df_sample['post']==0)]['fulltime'].mean()
post_treat_mean = df_sample[(df_sample['treated']==1) & (df_sample['post']==1)]['fulltime'].mean()
post_ctrl_mean = df_sample[(df_sample['treated']==0) & (df_sample['post']==1)]['fulltime'].mean()

# Group sizes
n_pre_treat = len(df_sample[(df_sample['treated']==1) & (df_sample['post']==0)])
n_pre_ctrl = len(df_sample[(df_sample['treated']==0) & (df_sample['post']==0)])
n_post_treat = len(df_sample[(df_sample['treated']==1) & (df_sample['post']==1)])
n_post_ctrl = len(df_sample[(df_sample['treated']==0) & (df_sample['post']==1)])

print("\nFull-Time Employment Rates (UHRSWORK >= 35):")
print("-" * 60)
print(f"{'':20} {'Treatment':>15} {'Control':>15} {'Difference':>15}")
print(f"{'':20} {'(Age 26-30)':>15} {'(Age 31-35)':>15}")
print("-" * 60)
print(f"{'Pre-DACA':20} {pre_treat_mean:>15.4f} {pre_ctrl_mean:>15.4f} {pre_treat_mean - pre_ctrl_mean:>15.4f}")
print(f"{'(2006-2011)':20} {'(n='+str(n_pre_treat)+')':>15} {'(n='+str(n_pre_ctrl)+')':>15}")
print(f"{'Post-DACA':20} {post_treat_mean:>15.4f} {post_ctrl_mean:>15.4f} {post_treat_mean - post_ctrl_mean:>15.4f}")
print(f"{'(2013-2016)':20} {'(n='+str(n_post_treat)+')':>15} {'(n='+str(n_post_ctrl)+')':>15}")
print("-" * 60)
print(f"{'Change':20} {post_treat_mean - pre_treat_mean:>15.4f} {post_ctrl_mean - pre_ctrl_mean:>15.4f}")
print("-" * 60)
did_estimate = (post_treat_mean - pre_treat_mean) - (post_ctrl_mean - pre_ctrl_mean)
print(f"{'DiD Estimate':20} {did_estimate:>15.4f}")
print("-" * 60)

# =============================================================================
# STEP 11: Save Results for Report
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 11] Saving Results")
print("=" * 80)

# Create results dictionary
results = {
    'did_estimate_basic': model1.params['did'],
    'did_se_basic': model1.bse['did'],
    'did_pvalue_basic': model1.pvalues['did'],
    'did_estimate_controls': model2.params['did'],
    'did_se_controls': model2.bse['did'],
    'did_pvalue_controls': model2.pvalues['did'],
    'did_estimate_yearfe': model3.params['did'],
    'did_se_yearfe': model3.bse['did'],
    'did_estimate_stateyearfe': model4.params['did'],
    'did_se_stateyearfe': model4.bse['did'],
    'did_estimate_weighted': model5.params['did'],
    'did_se_weighted': model5.bse['did'],
    'n_total': len(df_sample),
    'n_treatment': (df_sample['treated']==1).sum(),
    'n_control': (df_sample['treated']==0).sum(),
    'pre_treat_mean': pre_treat_mean,
    'pre_ctrl_mean': pre_ctrl_mean,
    'post_treat_mean': post_treat_mean,
    'post_ctrl_mean': post_ctrl_mean,
    'did_2x2': did_estimate
}

# Save detailed results
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)

# Save event study results
event_df.to_csv('event_study_results.csv', index=False)

# =============================================================================
# STEP 12: Generate Figures
# =============================================================================
print("\n[STEP 12] Generating figures...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Figure 1: Parallel Trends / Event Study
fig1, ax1 = plt.subplots(figsize=(10, 6))

years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [0]  # 2011 is reference
ses = [0]

for yr in [2006, 2007, 2008, 2009, 2010]:
    coefs.insert(years.index(yr), event_model.params[f'treat_x_{yr}'])
    ses.insert(years.index(yr), event_model.bse[f'treat_x_{yr}'])

# Insert 2011 reference
coefs.insert(5, 0)
ses.insert(5, 0)

for yr in [2013, 2014, 2015, 2016]:
    idx = years.index(yr)
    coefs.insert(idx, event_model.params[f'treat_x_{yr}'])
    ses.insert(idx, event_model.bse[f'treat_x_{yr}'])

# Fix indexing
coefs_plot = []
ses_plot = []
for yr in years:
    if yr == 2011:
        coefs_plot.append(0)
        ses_plot.append(0)
    else:
        coefs_plot.append(event_model.params[f'treat_x_{yr}'])
        ses_plot.append(event_model.bse[f'treat_x_{yr}'])

ax1.errorbar(years, coefs_plot, yerr=[1.96*s for s in ses_plot], fmt='o-', capsize=4, color='navy')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
ax1.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation (2012)')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Treatment Effect (relative to 2011)', fontsize=12)
ax1.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax1.legend()
ax1.set_xticks(years)
plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Mean Full-Time Employment by Group Over Time
fig2, ax2 = plt.subplots(figsize=(10, 6))

years_all = sorted(df_sample['YEAR'].unique())
treat_means = []
ctrl_means = []

for yr in years_all:
    treat_means.append(df_sample[(df_sample['YEAR']==yr) & (df_sample['treated']==1)]['fulltime'].mean())
    ctrl_means.append(df_sample[(df_sample['YEAR']==yr) & (df_sample['treated']==0)]['fulltime'].mean())

ax2.plot(years_all, treat_means, 'o-', label='Treatment (Age 26-30 in 2012)', color='blue', linewidth=2)
ax2.plot(years_all, ctrl_means, 's-', label='Control (Age 31-35 in 2012)', color='green', linewidth=2)
ax2.axvline(x=2012.5, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax2.set_title('Full-Time Employment Trends by Treatment Status', fontsize=14)
ax2.legend()
ax2.set_xticks(years_all)
plt.tight_layout()
plt.savefig('figure2_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: DiD Visualization
fig3, ax3 = plt.subplots(figsize=(8, 6))

periods = ['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)']
treat_vals = [pre_treat_mean, post_treat_mean]
ctrl_vals = [pre_ctrl_mean, post_ctrl_mean]

x = np.arange(len(periods))
width = 0.35

bars1 = ax3.bar(x - width/2, treat_vals, width, label='Treatment (Age 26-30)', color='steelblue')
bars2 = ax3.bar(x + width/2, ctrl_vals, width, label='Control (Age 31-35)', color='darkorange')

ax3.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences: DACA Effect on Full-Time Employment', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(periods)
ax3.legend()
ax3.set_ylim(0, max(treat_vals + ctrl_vals) * 1.15)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nFigures saved:")
print("  - figure1_event_study.png")
print("  - figure2_parallel_trends.png")
print("  - figure3_did_bars.png")

# =============================================================================
# STEP 13: Final Summary
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\nSample Size: {len(df_sample):,}")
print(f"  Treatment group: {results['n_treatment']:,}")
print(f"  Control group: {results['n_control']:,}")

print(f"\nPreferred Estimate (Model 4 with State/Year FE):")
print(f"  DiD Effect: {results['did_estimate_stateyearfe']:.6f}")
print(f"  Standard Error: {results['did_se_stateyearfe']:.6f}")
print(f"  95% CI: [{results['did_estimate_stateyearfe'] - 1.96*results['did_se_stateyearfe']:.6f}, {results['did_estimate_stateyearfe'] + 1.96*results['did_se_stateyearfe']:.6f}]")

print(f"\nInterpretation:")
if results['did_estimate_stateyearfe'] > 0:
    print(f"  DACA eligibility is associated with a {results['did_estimate_stateyearfe']*100:.2f} percentage point")
    print(f"  increase in full-time employment probability.")
else:
    print(f"  DACA eligibility is associated with a {abs(results['did_estimate_stateyearfe'])*100:.2f} percentage point")
    print(f"  decrease in full-time employment probability.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Save model summaries
with open('model_summaries.txt', 'w') as f:
    f.write("MODEL SUMMARIES - DACA REPLICATION\n")
    f.write("=" * 80 + "\n\n")
    f.write("Model 1: Basic DiD (OLS, unweighted)\n")
    f.write(str(model1.summary()) + "\n\n")
    f.write("=" * 80 + "\n\n")
    f.write("Model 2: DiD with Demographic Controls\n")
    f.write(str(model2.summary()) + "\n\n")
    f.write("=" * 80 + "\n\n")
    f.write("Model 4: DiD with State and Year FE\n")
    f.write(str(model4.summary()) + "\n\n")
    f.write("=" * 80 + "\n\n")
    f.write("Model 5: Weighted DiD\n")
    f.write(str(model5.summary()) + "\n\n")

print("\nModel summaries saved to model_summaries.txt")
