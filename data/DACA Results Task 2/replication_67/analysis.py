"""
DACA Replication Analysis
Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals

Research Design: Difference-in-Differences
- Treatment group: Ages 26-30 as of June 15, 2012 (DACA eligible by age)
- Control group: Ages 31-35 as of June 15, 2012 (would be eligible but for age)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (DACA implemented June 2012)
- Outcome: Full-time employment (UHRSWORK >= 35)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# STEP 1: LOAD AND FILTER DATA
# =============================================================================
print("\n[1] Loading data...")

# Define columns needed
cols_needed = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX', 'MARST',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'UHRSWORK', 'EMPSTAT', 'STATEFIP']

# Read data in chunks for memory efficiency
chunks = []
chunksize = 1000000

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunksize):
    # Filter early to reduce memory usage
    # HISPAN == 1 means Mexican Hispanic origin
    # BPL == 200 means born in Mexico
    mask = (chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)
    chunks.append(chunk[mask])

df = pd.concat(chunks, ignore_index=True)
print(f"Records after filtering Hispanic-Mexican born in Mexico: {len(df):,}")

# =============================================================================
# STEP 2: CONSTRUCT DACA ELIGIBILITY CRITERIA
# =============================================================================
print("\n[2] Constructing DACA eligibility criteria...")

# DACA eligibility requirements (as of June 15, 2012):
# 1. Arrived before 16th birthday
# 2. Under 31 years old as of June 15, 2012
# 3. Lived continuously in US since June 15, 2007
# 4. Not a citizen (CITIZEN == 3 means "Not a citizen")
# 5. Born in Mexico (already filtered)
# 6. Hispanic-Mexican ethnicity (already filtered)

# For our age-based analysis:
# Treatment: Ages 26-30 as of June 15, 2012 (born 1982-1986)
# Control: Ages 31-35 as of June 15, 2012 (born 1977-1981)

# Compute age as of June 15, 2012
# If BIRTHQTR is 1 or 2 (Jan-June), they've had their birthday by June 15
# If BIRTHQTR is 3 or 4 (July-Dec), they haven't had their birthday yet
df['age_june2012'] = 2012 - df['BIRTHYR']
# Adjust for those born Jul-Dec (haven't had birthday yet by June 15)
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june2012'] -= 1

print(f"Age at June 2012 distribution in sample:")
print(df['age_june2012'].value_counts().sort_index().head(20))

# Filter for non-citizens only (CITIZEN == 3)
# Note: We cannot distinguish documented vs undocumented, so assume non-citizens
# without naturalization are undocumented per instructions
df = df[df['CITIZEN'] == 3]
print(f"Records after filtering to non-citizens: {len(df):,}")

# Check year of immigration - must have arrived by June 15, 2007 (5 years continuous)
# For simplicity, YRIMMIG <= 2007
df = df[df['YRIMMIG'] <= 2007]
print(f"Records after filtering YRIMMIG <= 2007: {len(df):,}")

# Check arrival before 16th birthday
# Age at arrival = YRIMMIG - BIRTHYR (approximately)
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_arrival'] < 16]
print(f"Records after filtering arrived before age 16: {len(df):,}")

# =============================================================================
# STEP 3: DEFINE TREATMENT AND CONTROL GROUPS
# =============================================================================
print("\n[3] Defining treatment and control groups...")

# Treatment group: Ages 26-30 as of June 15, 2012 (born 1982-1986)
# Control group: Ages 31-35 as of June 15, 2012 (born 1977-1981)

df['treat_group'] = np.where(
    (df['age_june2012'] >= 26) & (df['age_june2012'] <= 30), 1,
    np.where((df['age_june2012'] >= 31) & (df['age_june2012'] <= 35), 0, np.nan)
)

# Keep only treatment and control groups
df = df[df['treat_group'].notna()].copy()
print(f"Records in treatment/control age range: {len(df):,}")

print("\nGroup sizes by treatment status:")
print(df.groupby('treat_group').size())

# =============================================================================
# STEP 4: DEFINE PRE AND POST PERIODS
# =============================================================================
print("\n[4] Defining pre and post periods...")

# Pre: 2006-2011
# Post: 2013-2016
# Exclude 2012 (implementation year - cannot separate pre/post)

df = df[df['YEAR'] != 2012]
df['post'] = (df['YEAR'] >= 2013).astype(int)

print(f"Records after excluding 2012: {len(df):,}")
print("\nYear distribution:")
print(df['YEAR'].value_counts().sort_index())

# =============================================================================
# STEP 5: CONSTRUCT OUTCOME VARIABLE
# =============================================================================
print("\n[5] Constructing outcome variable...")

# Full-time employment: UHRSWORK >= 35 hours per week
# UHRSWORK == 0 typically means not working
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

print(f"Full-time employment rate overall: {df['fulltime'].mean():.4f}")
print(f"Full-time rate (weighted): {np.average(df['fulltime'], weights=df['PERWT']):.4f}")

# =============================================================================
# STEP 6: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[6] Descriptive statistics...")

# Summary by group and period
summary = df.groupby(['treat_group', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum',
    'AGE': 'mean',
    'SEX': lambda x: (x==2).mean(),  # Female proportion
    'EDUC': 'mean'
}).round(4)
print("\nSummary by Treatment Group and Period:")
print(summary)

# Weighted means
def weighted_mean(group, var, weight='PERWT'):
    return np.average(group[var], weights=group[weight])

weighted_ft = df.groupby(['treat_group', 'post']).apply(
    lambda g: weighted_mean(g, 'fulltime')
).unstack()
print("\nWeighted Full-time Employment Rates:")
print(weighted_ft)

# =============================================================================
# STEP 7: SIMPLE DIFFERENCE-IN-DIFFERENCES
# =============================================================================
print("\n[7] Simple Difference-in-Differences calculation...")

# Calculate group means (unweighted)
means = df.groupby(['treat_group', 'post'])['fulltime'].mean().unstack()
print("\nMean full-time employment by group and period:")
print(means)

# DiD calculation
diff_treat = means.loc[1, 1] - means.loc[1, 0]  # Treatment: post - pre
diff_control = means.loc[0, 1] - means.loc[0, 0]  # Control: post - pre
did_simple = diff_treat - diff_control

print(f"\nChange in treatment group (post - pre): {diff_treat:.4f}")
print(f"Change in control group (post - pre): {diff_control:.4f}")
print(f"Difference-in-Differences estimate: {did_simple:.4f}")

# =============================================================================
# STEP 8: REGRESSION-BASED DiD
# =============================================================================
print("\n[8] Regression-based Difference-in-Differences...")

# Create interaction term
df['treat_post'] = df['treat_group'] * df['post']

# Basic DiD regression (unweighted)
print("\n--- Model 1: Basic DiD (unweighted) ---")
model1 = smf.ols('fulltime ~ treat_group + post + treat_post', data=df).fit()
print(model1.summary().tables[1])

# Weighted DiD regression
print("\n--- Model 2: Basic DiD (weighted) ---")
model2 = smf.wls('fulltime ~ treat_group + post + treat_post',
                  data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# DiD with covariates
print("\n--- Model 3: DiD with covariates (weighted) ---")
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)  # Married spouse present or absent

model3 = smf.wls('fulltime ~ treat_group + post + treat_post + female + married + C(EDUC)',
                  data=df, weights=df['PERWT']).fit()
print(model3.summary().tables[1])

# DiD with state fixed effects
print("\n--- Model 4: DiD with state fixed effects (weighted) ---")
model4 = smf.wls('fulltime ~ treat_group + post + treat_post + female + married + C(STATEFIP)',
                  data=df, weights=df['PERWT']).fit()
# Just show key coefficients
print("Key coefficients:")
print(f"  treat_group: {model4.params['treat_group']:.4f} (SE: {model4.bse['treat_group']:.4f})")
print(f"  post: {model4.params['post']:.4f} (SE: {model4.bse['post']:.4f})")
print(f"  treat_post (DiD): {model4.params['treat_post']:.4f} (SE: {model4.bse['treat_post']:.4f})")

# =============================================================================
# STEP 9: ROBUST STANDARD ERRORS
# =============================================================================
print("\n[9] DiD with heteroskedasticity-robust standard errors...")

# Re-estimate with robust SEs (HC1)
model_robust = smf.wls('fulltime ~ treat_group + post + treat_post + female + married',
                        data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model_robust.summary().tables[1])

# =============================================================================
# STEP 10: YEAR-BY-YEAR ANALYSIS (EVENT STUDY)
# =============================================================================
print("\n[10] Year-by-year analysis (event study style)...")

# Create year dummies interacted with treatment
year_effects = []
for year in sorted(df['YEAR'].unique()):
    if year == 2011:  # Base year
        continue
    df[f'treat_year{year}'] = (df['treat_group'] == 1) & (df['YEAR'] == year)
    df[f'treat_year{year}'] = df[f'treat_year{year}'].astype(int)

# Build formula
year_vars = [f'treat_year{y}' for y in sorted(df['YEAR'].unique()) if y != 2011]
formula = 'fulltime ~ treat_group + C(YEAR) + ' + ' + '.join(year_vars)

model_event = smf.wls(formula, data=df, weights=df['PERWT']).fit()

print("\nEvent study coefficients (relative to 2011):")
for var in year_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    print(f"  {var}: {coef:.4f} (SE: {se:.4f})")

# =============================================================================
# STEP 11: SAVE RESULTS FOR REPORT
# =============================================================================
print("\n[11] Saving results...")

# Save summary statistics
summary_stats = {
    'n_total': len(df),
    'n_treatment': (df['treat_group'] == 1).sum(),
    'n_control': (df['treat_group'] == 0).sum(),
    'did_estimate': model2.params['treat_post'],
    'did_se': model2.bse['treat_post'],
    'did_pvalue': model2.pvalues['treat_post'],
    'did_ci_low': model2.conf_int().loc['treat_post', 0],
    'did_ci_high': model2.conf_int().loc['treat_post', 1],
}

# Weighted means for table
means_table = df.groupby(['treat_group', 'post']).apply(
    lambda g: pd.Series({
        'fulltime_mean': np.average(g['fulltime'], weights=g['PERWT']),
        'n': len(g),
        'sum_weights': g['PERWT'].sum()
    })
).unstack()

print("\n" + "="*80)
print("MAIN RESULTS")
print("="*80)
print(f"\nSample size: {len(df):,}")
print(f"  Treatment group (ages 26-30): {(df['treat_group']==1).sum():,}")
print(f"  Control group (ages 31-35): {(df['treat_group']==0).sum():,}")
print(f"\nDifference-in-Differences Estimate (Weighted):")
print(f"  Coefficient: {model2.params['treat_post']:.4f}")
print(f"  Standard Error: {model2.bse['treat_post']:.4f}")
print(f"  95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  p-value: {model2.pvalues['treat_post']:.4f}")

# Store results for LaTeX
results = {
    'model2': model2,
    'model3': model3,
    'model4': model4,
    'model_robust': model_robust,
    'model_event': model_event,
    'means_table': means_table,
    'summary_stats': summary_stats,
    'df': df
}

# =============================================================================
# STEP 12: CREATE FIGURES
# =============================================================================
print("\n[12] Creating figures...")

# Figure 1: Trends in full-time employment by group
fig1, ax1 = plt.subplots(figsize=(10, 6))

yearly_means = df.groupby(['YEAR', 'treat_group']).apply(
    lambda g: np.average(g['fulltime'], weights=g['PERWT'])
).unstack()

ax1.plot(yearly_means.index, yearly_means[1], 'b-o', label='Treatment (Ages 26-30)', linewidth=2)
ax1.plot(yearly_means.index, yearly_means[0], 'r--s', label='Control (Ages 31-35)', linewidth=2)
ax1.axvline(x=2012, color='gray', linestyle=':', label='DACA Implementation')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-time Employment Rate', fontsize=12)
ax1.set_title('Full-time Employment Trends by Treatment Status', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(2006, 2017))

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Event study plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

years_es = [y for y in sorted(df['YEAR'].unique()) if y != 2011]
coefs = [model_event.params[f'treat_year{y}'] for y in years_es]
ses = [model_event.bse[f'treat_year{y}'] for y in years_es]

# Add 2011 as reference (0)
years_plot = [2011] + years_es
coefs_plot = [0] + coefs
ses_plot = [0] + ses

ax2.errorbar(years_plot, coefs_plot, yerr=[1.96*s for s in ses_plot],
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
ax2.axhline(y=0, color='gray', linestyle='--')
ax2.axvline(x=2012, color='red', linestyle=':', label='DACA Implementation')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Treatment Effect by Year', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(2006, 2017))

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: DiD illustration
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Get pre and post means
pre_treat = yearly_means[yearly_means.index <= 2011][1].mean()
post_treat = yearly_means[yearly_means.index >= 2013][1].mean()
pre_ctrl = yearly_means[yearly_means.index <= 2011][0].mean()
post_ctrl = yearly_means[yearly_means.index >= 2013][0].mean()

# Counterfactual
counterfactual = pre_treat + (post_ctrl - pre_ctrl)

ax3.plot([0, 1], [pre_treat, post_treat], 'b-o', markersize=10, linewidth=2, label='Treatment (actual)')
ax3.plot([0, 1], [pre_ctrl, post_ctrl], 'r-s', markersize=10, linewidth=2, label='Control')
ax3.plot([0, 1], [pre_treat, counterfactual], 'b--', linewidth=2, label='Treatment (counterfactual)')

# DiD arrow
ax3.annotate('', xy=(1, post_treat), xytext=(1, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.05, (post_treat + counterfactual)/2, f'DiD = {post_treat - counterfactual:.3f}',
         fontsize=12, color='green')

ax3.set_xlim(-0.2, 1.5)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax3.set_ylabel('Full-time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences Illustration', fontsize=14)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figures saved: figure1_trends.png, figure2_eventstudy.png, figure3_did.png")

# =============================================================================
# STEP 13: ADDITIONAL ROBUSTNESS CHECKS
# =============================================================================
print("\n[13] Additional robustness checks...")

# Subgroup analysis by sex
print("\n--- Subgroup by Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    model_sub = smf.wls('fulltime ~ treat_group + post + treat_post',
                         data=df_sub, weights=df_sub['PERWT']).fit()
    print(f"{sex_name}: DiD = {model_sub.params['treat_post']:.4f} (SE: {model_sub.bse['treat_post']:.4f})")

# Alternative age windows
print("\n--- Alternative age windows ---")
# Narrower window: 27-29 vs 32-34
df_narrow = df[(df['age_june2012'] >= 27) & (df['age_june2012'] <= 34)]
df_narrow['treat_narrow'] = (df_narrow['age_june2012'] <= 29).astype(int)
model_narrow = smf.wls('fulltime ~ treat_narrow + post + treat_narrow:post',
                        data=df_narrow, weights=df_narrow['PERWT']).fit()
print(f"Narrower window (27-29 vs 32-34): DiD = {model_narrow.params['treat_narrow:post']:.4f}")

# Placebo test: Pre-period only (2006-2008 vs 2009-2011)
print("\n--- Placebo test (pre-period only) ---")
df_pre = df[df['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_treat_post'] = df_pre['treat_group'] * df_pre['placebo_post']
model_placebo = smf.wls('fulltime ~ treat_group + placebo_post + placebo_treat_post',
                         data=df_pre, weights=df_pre['PERWT']).fit()
print(f"Placebo DiD (2009-2011 vs 2006-2008): {model_placebo.params['placebo_treat_post']:.4f} (SE: {model_placebo.bse['placebo_treat_post']:.4f})")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Export key results to file for LaTeX
with open('results_summary.txt', 'w') as f:
    f.write("DACA Replication Study - Results Summary\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total sample size: {len(df):,}\n")
    f.write(f"Treatment group (ages 26-30 as of June 2012): {(df['treat_group']==1).sum():,}\n")
    f.write(f"Control group (ages 31-35 as of June 2012): {(df['treat_group']==0).sum():,}\n")
    f.write(f"\nPre-period years: 2006-2011\n")
    f.write(f"Post-period years: 2013-2016\n")
    f.write(f"\nMain DiD Estimate (weighted):\n")
    f.write(f"  Coefficient: {model2.params['treat_post']:.4f}\n")
    f.write(f"  Standard Error: {model2.bse['treat_post']:.4f}\n")
    f.write(f"  95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]\n")
    f.write(f"  t-statistic: {model2.tvalues['treat_post']:.4f}\n")
    f.write(f"  p-value: {model2.pvalues['treat_post']:.4f}\n")
    f.write(f"\nWith covariates (female, married, education):\n")
    f.write(f"  Coefficient: {model3.params['treat_post']:.4f}\n")
    f.write(f"  Standard Error: {model3.bse['treat_post']:.4f}\n")
    f.write(f"\nWith state fixed effects:\n")
    f.write(f"  Coefficient: {model4.params['treat_post']:.4f}\n")
    f.write(f"  Standard Error: {model4.bse['treat_post']:.4f}\n")

    f.write(f"\n\nYearly Full-time Employment Rates (Weighted):\n")
    f.write(yearly_means.to_string())

    f.write(f"\n\nEvent Study Coefficients (relative to 2011):\n")
    for var in year_vars:
        f.write(f"  {var}: {model_event.params[var]:.4f} (SE: {model_event.bse[var]:.4f})\n")

    f.write(f"\n\nRobustness Checks:\n")
    f.write(f"Narrower age window (27-29 vs 32-34): {model_narrow.params['treat_narrow:post']:.4f}\n")
    f.write(f"Placebo (2009-2011 vs 2006-2008): {model_placebo.params['placebo_treat_post']:.4f}\n")

print("\nResults saved to results_summary.txt")
