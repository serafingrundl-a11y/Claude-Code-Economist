"""
DACA Replication Study - Analysis Script
Replication ID: 61

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican individuals born in Mexico.

Design: Difference-in-Differences
- Treatment: Ages 26-30 as of June 15, 2012 (birth years 1982-1986)
- Control: Ages 31-35 as of June 15, 2012 (birth years 1977-1981)
- Pre-period: 2006-2011
- Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Set up output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('tables', exist_ok=True)

print("="*70)
print("DACA REPLICATION STUDY - ANALYSIS SCRIPT")
print("="*70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[STEP 1] Loading data...")

df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: CONSTRUCT ANALYSIS SAMPLE
# =============================================================================
print("\n[STEP 2] Constructing analysis sample...")

# Initial counts
print(f"\nInitial sample size: {len(df):,}")

# Step 2.1: Hispanic-Mexican (HISPAN == 1)
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After HISPAN==1 (Mexican Hispanic): {len(df_sample):,}")

# Step 2.2: Born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200]
print(f"After BPL==200 (Born in Mexico): {len(df_sample):,}")

# Step 2.3: Not a citizen (CITIZEN == 3) - proxy for undocumented
df_sample = df_sample[df_sample['CITIZEN'] == 3]
print(f"After CITIZEN==3 (Not a citizen): {len(df_sample):,}")

# Step 2.4: Define birth year cohorts
# Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)
# Control: Born 1977-1981 (ages 31-35 on June 15, 2012)
df_sample = df_sample[
    ((df_sample['BIRTHYR'] >= 1982) & (df_sample['BIRTHYR'] <= 1986)) |
    ((df_sample['BIRTHYR'] >= 1977) & (df_sample['BIRTHYR'] <= 1981))
]
print(f"After birth year filter (1977-1986): {len(df_sample):,}")

# Step 2.5: Arrived before age 16
# Calculate age at immigration
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
# Keep only those who arrived before turning 16
df_sample = df_sample[(df_sample['age_at_immig'] >= 0) & (df_sample['age_at_immig'] < 16)]
print(f"After age_at_immig < 16: {len(df_sample):,}")

# Step 2.6: In US since at least 2007
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]
print(f"After YRIMMIG <= 2007: {len(df_sample):,}")

# Step 2.7: Exclude 2012 (DACA implementation year)
df_sample = df_sample[df_sample['YEAR'] != 2012]
print(f"After excluding 2012: {len(df_sample):,}")

# =============================================================================
# STEP 3: CREATE ANALYSIS VARIABLES
# =============================================================================
print("\n[STEP 3] Creating analysis variables...")

# Treatment indicator (younger cohort)
df_sample['treat'] = ((df_sample['BIRTHYR'] >= 1982) & (df_sample['BIRTHYR'] <= 1986)).astype(int)

# Post-DACA indicator
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Interaction term
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

# Full-time employment outcome (35+ hours per week AND employed)
# EMPSTAT: 1=Employed, 2=Unemployed, 3=Not in labor force
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)
df_sample['fulltime'] = ((df_sample['UHRSWORK'] >= 35) & (df_sample['EMPSTAT'] == 1)).astype(int)

# Labor force participation
df_sample['in_lf'] = (df_sample['LABFORCE'] == 2).astype(int)

# Control variables
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Education categories
df_sample['educ_cat'] = pd.cut(
    df_sample['EDUC'],
    bins=[-1, 5, 6, 7, 10, 11],
    labels=['less_hs', 'hs', 'some_college', 'bachelors', 'graduate']
)

# Age at survey (for descriptives)
df_sample['age'] = df_sample['YEAR'] - df_sample['BIRTHYR']

print(f"Treatment group (young): {df_sample['treat'].sum():,}")
print(f"Control group (old): {(1-df_sample['treat']).sum():,}")
print(f"Pre-period observations: {(1-df_sample['post']).sum():,}")
print(f"Post-period observations: {df_sample['post'].sum():,}")

# =============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[STEP 4] Generating descriptive statistics...")

# Sample by year and treatment status
year_treat_counts = df_sample.groupby(['YEAR', 'treat']).agg({
    'PERWT': 'sum',
    'fulltime': lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']),
    'employed': lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']),
}).reset_index()

# Weighted statistics function
def weighted_stats(group, var, weight):
    w = group[weight]
    v = group[var]
    mean = np.average(v, weights=w)
    return mean

# Summary statistics by treatment and period
summary_stats = df_sample.groupby(['treat', 'post']).apply(
    lambda x: pd.Series({
        'N': len(x),
        'N_weighted': x['PERWT'].sum(),
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'employed_rate': np.average(x['employed'], weights=x['PERWT']),
        'in_lf_rate': np.average(x['in_lf'], weights=x['PERWT']),
        'female_share': np.average(x['female'], weights=x['PERWT']),
        'married_share': np.average(x['married'], weights=x['PERWT']),
        'mean_age': np.average(x['age'], weights=x['PERWT']),
    })
).reset_index()

print("\nSummary Statistics by Treatment and Period:")
print(summary_stats.to_string(index=False))

# Save summary stats
summary_stats.to_csv('tables/summary_stats.csv', index=False)

# Compute simple DiD manually
pre_treat = summary_stats[(summary_stats['treat']==1) & (summary_stats['post']==0)]['fulltime_rate'].values[0]
post_treat = summary_stats[(summary_stats['treat']==1) & (summary_stats['post']==1)]['fulltime_rate'].values[0]
pre_control = summary_stats[(summary_stats['treat']==0) & (summary_stats['post']==0)]['fulltime_rate'].values[0]
post_control = summary_stats[(summary_stats['treat']==0) & (summary_stats['post']==1)]['fulltime_rate'].values[0]

diff_treat = post_treat - pre_treat
diff_control = post_control - pre_control
did_simple = diff_treat - diff_control

print(f"\n--- Simple DiD Calculation ---")
print(f"Treatment group (young, 26-30):")
print(f"  Pre-DACA fulltime rate:  {pre_treat:.4f}")
print(f"  Post-DACA fulltime rate: {post_treat:.4f}")
print(f"  Difference:              {diff_treat:.4f}")
print(f"\nControl group (old, 31-35):")
print(f"  Pre-DACA fulltime rate:  {pre_control:.4f}")
print(f"  Post-DACA fulltime rate: {post_control:.4f}")
print(f"  Difference:              {diff_control:.4f}")
print(f"\nDifference-in-Differences: {did_simple:.4f}")

# =============================================================================
# STEP 5: MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n[STEP 5] Running main regression analysis...")

# Model 1: Basic DiD (no controls)
model1 = smf.wls(
    'fulltime ~ treat + post + treat_post',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

print("\nModel 1: Basic DiD")
print(f"DiD coefficient (treat_post): {model1.params['treat_post']:.4f}")
print(f"Standard Error: {model1.bse['treat_post']:.4f}")
print(f"p-value: {model1.pvalues['treat_post']:.4f}")

# Model 2: Add demographic controls
model2 = smf.wls(
    'fulltime ~ treat + post + treat_post + female + married',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

print("\nModel 2: With Demographic Controls")
print(f"DiD coefficient (treat_post): {model2.params['treat_post']:.4f}")
print(f"Standard Error: {model2.bse['treat_post']:.4f}")

# Model 3: Add state fixed effects
df_sample['state_fe'] = pd.Categorical(df_sample['STATEFIP'])
model3 = smf.wls(
    'fulltime ~ treat + post + treat_post + female + married + C(STATEFIP)',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

print("\nModel 3: With State Fixed Effects")
print(f"DiD coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"Standard Error: {model3.bse['treat_post']:.4f}")

# Model 4: Add year fixed effects (replace post with year dummies)
model4 = smf.wls(
    'fulltime ~ treat + treat_post + female + married + C(STATEFIP) + C(YEAR)',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

print("\nModel 4: With Year Fixed Effects")
print(f"DiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"Standard Error: {model4.bse['treat_post']:.4f}")

# Store preferred estimate
preferred_estimate = model4.params['treat_post']
preferred_se = model4.bse['treat_post']
preferred_pval = model4.pvalues['treat_post']
n_obs = len(df_sample)

print("\n" + "="*50)
print("PREFERRED ESTIMATE (Model 4 - Full Specification)")
print("="*50)
print(f"Effect Size: {preferred_estimate:.4f}")
print(f"Standard Error: {preferred_se:.4f}")
print(f"95% CI: [{preferred_estimate - 1.96*preferred_se:.4f}, {preferred_estimate + 1.96*preferred_se:.4f}]")
print(f"p-value: {preferred_pval:.4f}")
print(f"Sample Size: {n_obs:,}")

# =============================================================================
# STEP 6: EVENT STUDY
# =============================================================================
print("\n[STEP 6] Running event study analysis...")

# Create year-specific treatment effects
df_sample['year_factor'] = pd.Categorical(df_sample['YEAR'])
years = sorted(df_sample['YEAR'].unique())

# Create interaction terms for each year (excluding 2011 as reference)
for year in years:
    if year != 2011:  # 2011 is reference year (last pre-treatment year)
        df_sample[f'treat_x_{year}'] = (df_sample['treat'] * (df_sample['YEAR'] == year)).astype(int)

# Event study regression
event_vars = [f'treat_x_{year}' for year in years if year != 2011]
event_formula = 'fulltime ~ treat + ' + ' + '.join(event_vars) + ' + female + married + C(STATEFIP) + C(YEAR)'

model_event = smf.wls(
    event_formula,
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

# Extract event study coefficients
event_coefs = []
event_ses = []
event_years = []
for year in years:
    if year == 2011:
        event_coefs.append(0)
        event_ses.append(0)
    else:
        event_coefs.append(model_event.params[f'treat_x_{year}'])
        event_ses.append(model_event.bse[f'treat_x_{year}'])
    event_years.append(year)

event_df = pd.DataFrame({
    'year': event_years,
    'coef': event_coefs,
    'se': event_ses
})
event_df['ci_lower'] = event_df['coef'] - 1.96 * event_df['se']
event_df['ci_upper'] = event_df['coef'] + 1.96 * event_df['se']

print("\nEvent Study Coefficients:")
print(event_df.to_string(index=False))
event_df.to_csv('tables/event_study.csv', index=False)

# =============================================================================
# STEP 7: ROBUSTNESS CHECKS
# =============================================================================
print("\n[STEP 7] Running robustness checks...")

# Robustness 1: Alternative outcome - employed (extensive margin)
model_emp = smf.wls(
    'employed ~ treat + treat_post + female + married + C(STATEFIP) + C(YEAR)',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

print("\nRobustness 1: Extensive margin (employment)")
print(f"DiD coefficient: {model_emp.params['treat_post']:.4f}")
print(f"Standard Error: {model_emp.bse['treat_post']:.4f}")

# Robustness 2: Labor force participation
model_lf = smf.wls(
    'in_lf ~ treat + treat_post + female + married + C(STATEFIP) + C(YEAR)',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

print("\nRobustness 2: Labor force participation")
print(f"DiD coefficient: {model_lf.params['treat_post']:.4f}")
print(f"Standard Error: {model_lf.bse['treat_post']:.4f}")

# Robustness 3: Heterogeneity by sex
model_male = smf.wls(
    'fulltime ~ treat + treat_post + married + C(STATEFIP) + C(YEAR)',
    data=df_sample[df_sample['female'] == 0],
    weights=df_sample[df_sample['female'] == 0]['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_sample[df_sample['female'] == 0]['STATEFIP']})

model_female = smf.wls(
    'fulltime ~ treat + treat_post + married + C(STATEFIP) + C(YEAR)',
    data=df_sample[df_sample['female'] == 1],
    weights=df_sample[df_sample['female'] == 1]['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_sample[df_sample['female'] == 1]['STATEFIP']})

print("\nRobustness 3: Heterogeneity by sex")
print(f"Males - DiD coefficient: {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f})")
print(f"Females - DiD coefficient: {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f})")

# Robustness 4: Placebo test (use 2009 as fake treatment year)
df_pre = df_sample[df_sample['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['treat_placebo'] = df_pre['treat'] * df_pre['placebo_post']

model_placebo = smf.wls(
    'fulltime ~ treat + placebo_post + treat_placebo + female + married + C(STATEFIP) + C(YEAR)',
    data=df_pre,
    weights=df_pre['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})

print("\nRobustness 4: Placebo test (fake treatment in 2009)")
print(f"Placebo DiD coefficient: {model_placebo.params['treat_placebo']:.4f}")
print(f"Standard Error: {model_placebo.bse['treat_placebo']:.4f}")
print(f"p-value: {model_placebo.pvalues['treat_placebo']:.4f}")

# =============================================================================
# STEP 8: CREATE FIGURES
# =============================================================================
print("\n[STEP 8] Creating figures...")

# Figure 1: Trends in full-time employment by treatment status
fig1, ax1 = plt.subplots(figsize=(10, 6))

trends = df_sample.groupby(['YEAR', 'treat']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).reset_index()
trends.columns = ['YEAR', 'treat', 'fulltime_rate']

for treat_val, label, color in [(1, 'Treatment (Ages 26-30)', 'blue'), (0, 'Control (Ages 31-35)', 'red')]:
    data = trends[trends['treat'] == treat_val]
    ax1.plot(data['YEAR'], data['fulltime_rate'], 'o-', label=label, color=color, linewidth=2, markersize=8)

ax1.axvline(x=2012, color='black', linestyle='--', alpha=0.7, label='DACA Implementation')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-time Employment Rate', fontsize=12)
ax1.set_title('Full-time Employment Trends by Treatment Status', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(2006, 2017))
plt.tight_layout()
plt.savefig('figures/fig1_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: figures/fig1_trends.png")

# Figure 2: Event study plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.errorbar(event_df['year'], event_df['coef'], yerr=1.96*event_df['se'],
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8, color='navy')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax2.fill_between(event_df['year'], event_df['ci_lower'], event_df['ci_upper'], alpha=0.2, color='navy')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Effect of DACA Eligibility on Full-time Employment', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(2006, 2017))
plt.tight_layout()
plt.savefig('figures/fig2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: figures/fig2_event_study.png")

# Figure 3: DiD visualization
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Pre-post means
means = df_sample.groupby(['treat', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).reset_index()
means.columns = ['treat', 'post', 'fulltime_rate']

# Create bar positions
x = np.array([0, 1])
width = 0.35

treat_means = means[means['treat'] == 1]['fulltime_rate'].values
control_means = means[means['treat'] == 0]['fulltime_rate'].values

bars1 = ax3.bar(x - width/2, treat_means, width, label='Treatment (Ages 26-30)', color='steelblue', edgecolor='black')
bars2 = ax3.bar(x + width/2, control_means, width, label='Control (Ages 31-35)', color='coral', edgecolor='black')

ax3.set_xlabel('Period', fontsize=12)
ax3.set_ylabel('Full-time Employment Rate', fontsize=12)
ax3.set_title('Full-time Employment: Pre vs Post DACA', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig3_did_bars.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: figures/fig3_did_bars.png")

# =============================================================================
# STEP 9: SAVE RESULTS FOR LATEX
# =============================================================================
print("\n[STEP 9] Saving results for LaTeX report...")

# Main results table
results_dict = {
    'Model': ['(1) Basic', '(2) Demographics', '(3) State FE', '(4) Year FE', '(5) Employment', '(6) LFP'],
    'DiD_Estimate': [
        model1.params['treat_post'],
        model2.params['treat_post'],
        model3.params['treat_post'],
        model4.params['treat_post'],
        model_emp.params['treat_post'],
        model_lf.params['treat_post']
    ],
    'SE': [
        model1.bse['treat_post'],
        model2.bse['treat_post'],
        model3.bse['treat_post'],
        model4.bse['treat_post'],
        model_emp.bse['treat_post'],
        model_lf.bse['treat_post']
    ],
    'p_value': [
        model1.pvalues['treat_post'],
        model2.pvalues['treat_post'],
        model3.pvalues['treat_post'],
        model4.pvalues['treat_post'],
        model_emp.pvalues['treat_post'],
        model_lf.pvalues['treat_post']
    ]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('tables/main_results.csv', index=False)

# Heterogeneity results
hetero_dict = {
    'Subgroup': ['Full Sample', 'Males', 'Females'],
    'DiD_Estimate': [
        model4.params['treat_post'],
        model_male.params['treat_post'],
        model_female.params['treat_post']
    ],
    'SE': [
        model4.bse['treat_post'],
        model_male.bse['treat_post'],
        model_female.bse['treat_post']
    ],
    'N': [
        len(df_sample),
        len(df_sample[df_sample['female'] == 0]),
        len(df_sample[df_sample['female'] == 1])
    ]
}

hetero_df = pd.DataFrame(hetero_dict)
hetero_df.to_csv('tables/heterogeneity.csv', index=False)

# Save key numbers for LaTeX
with open('tables/key_numbers.txt', 'w') as f:
    f.write(f"preferred_estimate,{preferred_estimate:.4f}\n")
    f.write(f"preferred_se,{preferred_se:.4f}\n")
    f.write(f"preferred_ci_lower,{preferred_estimate - 1.96*preferred_se:.4f}\n")
    f.write(f"preferred_ci_upper,{preferred_estimate + 1.96*preferred_se:.4f}\n")
    f.write(f"preferred_pval,{preferred_pval:.4f}\n")
    f.write(f"n_obs,{n_obs}\n")
    f.write(f"n_treat,{df_sample['treat'].sum()}\n")
    f.write(f"n_control,{(1-df_sample['treat']).sum()}\n")
    f.write(f"pre_treat_rate,{pre_treat:.4f}\n")
    f.write(f"post_treat_rate,{post_treat:.4f}\n")
    f.write(f"pre_control_rate,{pre_control:.4f}\n")
    f.write(f"post_control_rate,{post_control:.4f}\n")
    f.write(f"did_simple,{did_simple:.4f}\n")
    f.write(f"male_estimate,{model_male.params['treat_post']:.4f}\n")
    f.write(f"male_se,{model_male.bse['treat_post']:.4f}\n")
    f.write(f"female_estimate,{model_female.params['treat_post']:.4f}\n")
    f.write(f"female_se,{model_female.bse['treat_post']:.4f}\n")
    f.write(f"placebo_estimate,{model_placebo.params['treat_placebo']:.4f}\n")
    f.write(f"placebo_se,{model_placebo.bse['treat_placebo']:.4f}\n")
    f.write(f"placebo_pval,{model_placebo.pvalues['treat_placebo']:.4f}\n")
    f.write(f"emp_estimate,{model_emp.params['treat_post']:.4f}\n")
    f.write(f"emp_se,{model_emp.bse['treat_post']:.4f}\n")
    f.write(f"lf_estimate,{model_lf.params['treat_post']:.4f}\n")
    f.write(f"lf_se,{model_lf.bse['treat_post']:.4f}\n")

print("Results saved to tables/")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nFinal Sample Size: {n_obs:,}")
print(f"Treatment Group: {df_sample['treat'].sum():,} observations")
print(f"Control Group: {(1-df_sample['treat']).sum():,} observations")
print(f"\nPreferred Estimate (Model 4):")
print(f"  Effect of DACA on Full-time Employment: {preferred_estimate:.4f}")
print(f"  Standard Error: {preferred_se:.4f}")
print(f"  95% Confidence Interval: [{preferred_estimate - 1.96*preferred_se:.4f}, {preferred_estimate + 1.96*preferred_se:.4f}]")
print(f"  p-value: {preferred_pval:.4f}")
print("\nFiles generated:")
print("  - tables/summary_stats.csv")
print("  - tables/main_results.csv")
print("  - tables/event_study.csv")
print("  - tables/heterogeneity.csv")
print("  - tables/key_numbers.txt")
print("  - figures/fig1_trends.png")
print("  - figures/fig2_event_study.png")
print("  - figures/fig3_did_bars.png")
print("="*70)
