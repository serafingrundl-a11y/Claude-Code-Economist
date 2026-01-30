"""
DACA Replication Study: Effect of DACA Eligibility on Full-Time Employment
Among Hispanic-Mexican Mexican-born Individuals

This script performs a difference-in-differences analysis to estimate the causal
impact of DACA eligibility on full-time employment (35+ hours/week).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("DACA REPLICATION STUDY - DATA LOADING AND ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n1. Loading ACS data...")
data_path = "data/data.csv"

# Read the CSV file
df = pd.read_csv(data_path)
print(f"   Initial data shape: {df.shape}")
print(f"   Years available: {sorted(df['YEAR'].unique())}")

# ============================================================================
# STEP 2: IDENTIFY DACA-ELIGIBLE POPULATION
# ============================================================================
print("\n2. Identifying DACA-eligible population...")

# DACA requirements (from instructions):
# - Hispanic-Mexican ethnicity (HISPAN == 1)
# - Born in Mexico (BPL == 200)
# - Not a citizen (CITIZEN == 3 for "Not a citizen")
# - Arrived before age 16: YEAR - YRIMMIG gives years in US at time of survey
#   If they arrived before age 16, then: AGE - (YEAR - YRIMMIG) < 16
#   Or equivalently: YRIMMIG > BIRTHYR + 16 - 1 (arrived when younger than 16)
# - Lived continuously in US since June 15, 2007 (arrived by 2007)
# - Were in US on June 15, 2012 (arrived by 2012)

# First, filter to Hispanic-Mexican, Mexican-born, non-citizens
print("\n   Filtering criteria:")
print(f"   - Total observations: {len(df)}")

# Hispanic-Mexican (HISPAN == 1 means Mexican)
df_mex = df[df['HISPAN'] == 1].copy()
print(f"   - After Hispanic-Mexican filter: {len(df_mex)}")

# Born in Mexico (BPL == 200)
df_mex = df_mex[df_mex['BPL'] == 200].copy()
print(f"   - After Mexican birthplace filter: {len(df_mex)}")

# Not a citizen (CITIZEN == 3)
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"   - After non-citizen filter: {len(df_mex)}")

# Valid immigration year (non-zero)
df_mex = df_mex[df_mex['YRIMMIG'] > 0].copy()
print(f"   - After valid immigration year filter: {len(df_mex)}")

# Arrived before age 16 (to meet DACA requirement)
# Age at arrival = YRIMMIG - BIRTHYR
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
df_mex = df_mex[df_mex['age_at_arrival'] < 16].copy()
print(f"   - After arrived before age 16 filter: {len(df_mex)}")

# Lived in US since June 15, 2007 (arrived by 2007)
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007].copy()
print(f"   - After continuous residence since 2007 filter: {len(df_mex)}")

# ============================================================================
# STEP 3: CREATE TREATMENT AND CONTROL GROUPS
# ============================================================================
print("\n3. Creating treatment and control groups...")

# DACA was implemented on June 15, 2012
# - Treatment group: ages 26-30 at time of policy (born 1982-1986)
#   These are people who turned 31 by June 15, 2012 or later, so were eligible
# - Control group: ages 31-35 at time of policy (born 1977-1981)
#   These are people who had already turned 31 by June 15, 2012, so were NOT eligible

# Age on June 15, 2012:
# If born in Q1 (Jan-Mar): would have had birthday by June 15, so age = 2012 - BIRTHYR
# If born in Q2 (Apr-Jun): might or might not have had birthday, assume had birthday for Q1-Q2
# If born in Q3-Q4 (Jul-Dec): would not have had birthday yet, so age = 2012 - BIRTHYR - 1

# For simplicity, calculate approximate age at June 15, 2012
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
df_mex['age_june2012'] = 2012 - df_mex['BIRTHYR']
# Adjust for those born after June (Q3 and Q4)
df_mex.loc[df_mex['BIRTHQTR'].isin([3, 4]), 'age_june2012'] -= 1

# Treatment group: ages 26-30 on June 15, 2012 (DACA eligible)
# Control group: ages 31-35 on June 15, 2012 (too old for DACA)

df_mex['treatment'] = np.where(
    (df_mex['age_june2012'] >= 26) & (df_mex['age_june2012'] <= 30), 1,
    np.where((df_mex['age_june2012'] >= 31) & (df_mex['age_june2012'] <= 35), 0, np.nan)
)

# Keep only treatment and control groups
df_analysis = df_mex[df_mex['treatment'].notna()].copy()
df_analysis['treatment'] = df_analysis['treatment'].astype(int)

print(f"   Treatment group (ages 26-30): {(df_analysis['treatment']==1).sum()}")
print(f"   Control group (ages 31-35): {(df_analysis['treatment']==0).sum()}")
print(f"   Total analysis sample: {len(df_analysis)}")

# ============================================================================
# STEP 4: CREATE OUTCOME VARIABLE AND TIME PERIOD
# ============================================================================
print("\n4. Creating outcome variable and time period indicators...")

# Full-time employment: UHRSWORK >= 35 (usually works 35+ hours per week)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Create pre/post indicator
# DACA implemented June 15, 2012
# Pre-period: 2006-2011 (before DACA)
# 2012 is ambiguous (policy implemented mid-year), so we exclude it
# Post-period: 2013-2016 (after DACA, per instructions)

df_analysis['post'] = np.where(
    df_analysis['YEAR'] <= 2011, 0,
    np.where(df_analysis['YEAR'] >= 2013, 1, np.nan)
)

# Remove 2012 observations (ambiguous period)
df_analysis = df_analysis[df_analysis['post'].notna()].copy()
df_analysis['post'] = df_analysis['post'].astype(int)

print(f"   Pre-period (2006-2011) observations: {(df_analysis['post']==0).sum()}")
print(f"   Post-period (2013-2016) observations: {(df_analysis['post']==1).sum()}")

# ============================================================================
# STEP 5: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n5. Calculating descriptive statistics...")

# Save summary statistics
desc_stats = df_analysis.groupby(['treatment', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': ['mean', 'std'],
    'SEX': lambda x: (x == 1).mean(),  # Proportion male
    'EDUC': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\n   Summary by Treatment and Period:")
print(desc_stats)

# Calculate weighted means
print("\n   Weighted Full-Time Employment Rates:")
for treat in [0, 1]:
    for period in [0, 1]:
        subset = df_analysis[(df_analysis['treatment']==treat) & (df_analysis['post']==period)]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        n = len(subset)
        label_treat = "Treatment" if treat == 1 else "Control"
        label_period = "Post" if period == 1 else "Pre"
        print(f"   {label_treat}, {label_period}: {weighted_mean:.4f} (n={n})")

# ============================================================================
# STEP 6: DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n6. Running Difference-in-Differences Analysis...")

# Create interaction term
df_analysis['treat_post'] = df_analysis['treatment'] * df_analysis['post']

# Model 1: Basic DiD (unweighted)
print("\n   Model 1: Basic DiD (OLS, unweighted)")
model1 = smf.ols('fulltime ~ treatment + post + treat_post', data=df_analysis).fit()
print(model1.summary().tables[1])

# Model 2: Basic DiD (weighted)
print("\n   Model 2: Basic DiD (WLS, using person weights)")
model2 = smf.wls('fulltime ~ treatment + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: With covariates (weighted)
print("\n   Model 3: DiD with covariates (WLS)")
# Add age (centered), sex, education, marital status
df_analysis['age_centered'] = df_analysis['AGE'] - df_analysis['AGE'].mean()
df_analysis['male'] = (df_analysis['SEX'] == 1).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

model3 = smf.wls('fulltime ~ treatment + post + treat_post + age_centered + male + married + C(EDUC)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(f"\n   DiD coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"   Standard Error: {model3.bse['treat_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   P-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: With year fixed effects (weighted)
print("\n   Model 4: DiD with year fixed effects (WLS)")
model4 = smf.wls('fulltime ~ treatment + C(YEAR) + treat_post + age_centered + male + married + C(EDUC)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(f"\n   DiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"   Standard Error: {model4.bse['treat_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   P-value: {model4.pvalues['treat_post']:.4f}")

# Model 5: With robust/clustered standard errors
print("\n   Model 5: DiD with robust standard errors (WLS)")
model5 = smf.wls('fulltime ~ treatment + post + treat_post + age_centered + male + married + C(EDUC)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"\n   DiD coefficient (treat_post): {model5.params['treat_post']:.4f}")
print(f"   Robust SE: {model5.bse['treat_post']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   P-value: {model5.pvalues['treat_post']:.4f}")

# ============================================================================
# STEP 7: CALCULATE SIMPLE DiD MANUALLY FOR VERIFICATION
# ============================================================================
print("\n7. Manual DiD Calculation (Weighted)...")

def weighted_mean(data, var, weight):
    return np.average(data[var], weights=data[weight])

# Calculate cell means
mean_treat_post = weighted_mean(df_analysis[(df_analysis['treatment']==1) & (df_analysis['post']==1)], 'fulltime', 'PERWT')
mean_treat_pre = weighted_mean(df_analysis[(df_analysis['treatment']==1) & (df_analysis['post']==0)], 'fulltime', 'PERWT')
mean_ctrl_post = weighted_mean(df_analysis[(df_analysis['treatment']==0) & (df_analysis['post']==1)], 'fulltime', 'PERWT')
mean_ctrl_pre = weighted_mean(df_analysis[(df_analysis['treatment']==0) & (df_analysis['post']==0)], 'fulltime', 'PERWT')

print(f"   Treatment, Post (2013-2016): {mean_treat_post:.4f}")
print(f"   Treatment, Pre (2006-2011):  {mean_treat_pre:.4f}")
print(f"   Control, Post (2013-2016):   {mean_ctrl_post:.4f}")
print(f"   Control, Pre (2006-2011):    {mean_ctrl_pre:.4f}")

diff_treat = mean_treat_post - mean_treat_pre
diff_ctrl = mean_ctrl_post - mean_ctrl_pre
did_estimate = diff_treat - diff_ctrl

print(f"\n   Treatment group change: {diff_treat:.4f}")
print(f"   Control group change:   {diff_ctrl:.4f}")
print(f"   DiD estimate:           {did_estimate:.4f}")

# ============================================================================
# STEP 8: YEARLY TRENDS FOR PARALLEL TRENDS CHECK
# ============================================================================
print("\n8. Calculating yearly trends for parallel trends analysis...")

yearly_means = df_analysis.groupby(['YEAR', 'treatment']).apply(
    lambda x: pd.Series({
        'fulltime_mean': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()

print("\n   Yearly Full-Time Employment Rates:")
print(yearly_means.pivot(index='YEAR', columns='treatment', values='fulltime_mean').round(4))

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("\n9. Saving results...")

# Save model summaries
with open('model_results.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - REGRESSION RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("MODEL 1: Basic DiD (OLS, unweighted)\n")
    f.write("-"*40 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("MODEL 2: Basic DiD (WLS, person weights)\n")
    f.write("-"*40 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("MODEL 3: DiD with covariates (WLS)\n")
    f.write("-"*40 + "\n")
    f.write(str(model3.summary()) + "\n\n")

    f.write("MODEL 4: DiD with year fixed effects (WLS)\n")
    f.write("-"*40 + "\n")
    f.write(str(model4.summary()) + "\n\n")

    f.write("MODEL 5: DiD with robust standard errors (WLS)\n")
    f.write("-"*40 + "\n")
    f.write(str(model5.summary()) + "\n\n")

# Save yearly means for plotting
yearly_means.to_csv('yearly_means.csv', index=False)

# Save descriptive statistics
desc_full = df_analysis.groupby(['treatment', 'post']).apply(
    lambda x: pd.Series({
        'fulltime_weighted': np.average(x['fulltime'], weights=x['PERWT']),
        'age_weighted': np.average(x['AGE'], weights=x['PERWT']),
        'male_weighted': np.average(x['male'], weights=x['PERWT']),
        'married_weighted': np.average(x['married'], weights=x['PERWT']),
        'n_unweighted': len(x),
        'n_weighted': x['PERWT'].sum()
    })
).reset_index()
desc_full.to_csv('descriptive_stats.csv', index=False)

print("   Results saved to:")
print("   - model_results.txt")
print("   - yearly_means.csv")
print("   - descriptive_stats.csv")

# ============================================================================
# STEP 10: CREATE FIGURES
# ============================================================================
print("\n10. Creating figures...")

# Figure 1: Parallel Trends
fig, ax = plt.subplots(figsize=(10, 6))
years_treatment = yearly_means[yearly_means['treatment']==1]['YEAR']
years_control = yearly_means[yearly_means['treatment']==0]['YEAR']
ft_treatment = yearly_means[yearly_means['treatment']==1]['fulltime_mean']
ft_control = yearly_means[yearly_means['treatment']==0]['fulltime_mean']

ax.plot(years_treatment, ft_treatment, 'b-o', label='Treatment (Ages 26-30 in 2012)', linewidth=2, markersize=8)
ax.plot(years_control, ft_control, 'r--s', label='Control (Ages 31-35 in 2012)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by Treatment Group Over Time', fontsize=14)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(2005.5, 2016.5)
plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: DiD Visualization
fig, ax = plt.subplots(figsize=(8, 6))
pre_post = ['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)']
treatment_means = [mean_treat_pre, mean_treat_post]
control_means = [mean_ctrl_pre, mean_ctrl_post]

x = np.arange(len(pre_post))
width = 0.35

bars1 = ax.bar(x - width/2, treatment_means, width, label='Treatment (Ages 26-30)', color='steelblue')
bars2 = ax.bar(x + width/2, control_means, width, label='Control (Ages 31-35)', color='indianred')

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: Full-Time Employment', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(pre_post, fontsize=11)
ax.legend(loc='best', fontsize=10)
ax.set_ylim(0, max(max(treatment_means), max(control_means)) * 1.15)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figure2_did_bars.png', dpi=300, bbox_inches='tight')
plt.close()

print("   Figures saved to:")
print("   - figure1_parallel_trends.png")
print("   - figure2_did_bars.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"\nPreferred Estimate (Model 5: DiD with covariates and robust SEs):")
print(f"   Effect of DACA eligibility on full-time employment: {model5.params['treat_post']:.4f}")
print(f"   Robust Standard Error: {model5.bse['treat_post']:.4f}")
print(f"   95% Confidence Interval: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   P-value: {model5.pvalues['treat_post']:.4f}")
print(f"\nSample Size: {len(df_analysis)}")
print(f"   Treatment group: {(df_analysis['treatment']==1).sum()}")
print(f"   Control group: {(df_analysis['treatment']==0).sum()}")

# Save key results for report
results_summary = {
    'did_estimate': float(model5.params['treat_post']),
    'se': float(model5.bse['treat_post']),
    'ci_lower': float(model5.conf_int().loc['treat_post', 0]),
    'ci_upper': float(model5.conf_int().loc['treat_post', 1]),
    'pvalue': float(model5.pvalues['treat_post']),
    'n_total': int(len(df_analysis)),
    'n_treatment': int((df_analysis['treatment']==1).sum()),
    'n_control': int((df_analysis['treatment']==0).sum()),
    'mean_treat_pre': float(mean_treat_pre),
    'mean_treat_post': float(mean_treat_post),
    'mean_ctrl_pre': float(mean_ctrl_pre),
    'mean_ctrl_post': float(mean_ctrl_post),
    'diff_treat': float(diff_treat),
    'diff_ctrl': float(diff_ctrl),
    'did_manual': float(did_estimate)
}

import json
with open('results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\nAnalysis complete!")
