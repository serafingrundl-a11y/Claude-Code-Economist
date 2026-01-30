"""
DACA Replication Analysis
=========================
Examining the effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexico-born individuals.

Research Design: Difference-in-Differences
- Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
- Control: Ages 31-35 at DACA implementation
- Pre-period: 2006-2011
- Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND FILTER DATA (EFFICIENT APPROACH)
# =============================================================================

print("Loading data with initial filters...")

# Define columns we need
usecols = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR', 'MARST',
           'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'UHRSWORK', 'STATEFIP']

# Read in chunks and filter to reduce memory usage
chunks = []
chunksize = 1000000

print("Reading and filtering data in chunks...")
for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunksize)):
    # Apply initial filters
    filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Hispanic-Mexican
        (chunk['BPL'] == 200) &   # Born in Mexico
        (chunk['CITIZEN'] == 3) & # Non-citizen
        (chunk['YEAR'] != 2012)   # Exclude transition year
    ]
    if len(filtered) > 0:
        chunks.append(filtered)
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"After initial filters: {len(df):,} rows")

# =============================================================================
# 2. COMPLETE SAMPLE SELECTION
# =============================================================================

print("\n" + "="*60)
print("SAMPLE SELECTION")
print("="*60)

# Filter for immigration year <= 2007 (continuous residence since June 15, 2007)
df = df[(df['YRIMMIG'] > 0) & (df['YRIMMIG'] <= 2007)].copy()
print(f"After immigration year filter (<=2007): {len(df):,}")

# Calculate age at immigration and filter for arrived before 16th birthday
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_immig'] < 16].copy()
print(f"After arrived before age 16 filter: {len(df):,}")

# Define age groups based on age at June 15, 2012
# Treatment: ages 26-30 at June 2012 -> born 1982-1986
# Control: ages 31-35 at June 2012 -> born 1977-1981
df['age_at_daca'] = 2012 - df['BIRTHYR']

# Adjust for birth quarter (Q3/Q4 haven't had birthday by June 15)
df['age_at_daca_adjusted'] = df['age_at_daca'].copy()
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_at_daca_adjusted'] -= 1

# Create treatment indicator
df['treatment'] = ((df['age_at_daca_adjusted'] >= 26) &
                   (df['age_at_daca_adjusted'] <= 30)).astype(int)
df['control'] = ((df['age_at_daca_adjusted'] >= 31) &
                 (df['age_at_daca_adjusted'] <= 35)).astype(int)

# Keep only treatment and control groups
df = df[(df['treatment'] == 1) | (df['control'] == 1)].copy()
print(f"After age group filter (26-35 at DACA): {len(df):,}")

# =============================================================================
# 3. CREATE ANALYSIS VARIABLES
# =============================================================================

print("\n" + "="*60)
print("CREATING ANALYSIS VARIABLES")
print("="*60)

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Full-time employment outcome (35+ hours/week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# DiD interaction term
df['treat_post'] = df['treatment'] * df['post']

# Covariates
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = df['MARST'].isin([1, 2]).astype(int)

# Education categories
df['educ_cat'] = pd.cut(df['EDUC'],
                         bins=[-1, 2, 6, 10, 11],
                         labels=['less_hs', 'high_school', 'some_college', 'college_plus'])

print(f"Treatment group (ages 26-30): {df['treatment'].sum():,}")
print(f"Control group (ages 31-35): {(df['treatment']==0).sum():,}")
print(f"Pre-period observations: {(df['post']==0).sum():,}")
print(f"Post-period observations: {(df['post']==1).sum():,}")

# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

def weighted_mean(data, value_col, weight_col):
    if len(data) == 0:
        return np.nan
    return np.average(data[value_col], weights=data[weight_col])

print("\nWeighted Full-Time Employment Rates:")
print("-"*60)

for treat in [0, 1]:
    for post in [0, 1]:
        subset = df[(df['treatment'] == treat) & (df['post'] == post)]
        wt_mean = weighted_mean(subset, 'fulltime', 'PERWT')
        group = "Treatment" if treat else "Control"
        period = "Post" if post else "Pre"
        print(f"{group}, {period}: {wt_mean:.4f} (N={len(subset):,})")

# Calculate simple DiD
pre_treat = df[(df['treatment']==1) & (df['post']==0)]
post_treat = df[(df['treatment']==1) & (df['post']==1)]
pre_ctrl = df[(df['treatment']==0) & (df['post']==0)]
post_ctrl = df[(df['treatment']==0) & (df['post']==1)]

wt_pre_treat = weighted_mean(pre_treat, 'fulltime', 'PERWT')
wt_post_treat = weighted_mean(post_treat, 'fulltime', 'PERWT')
wt_pre_ctrl = weighted_mean(pre_ctrl, 'fulltime', 'PERWT')
wt_post_ctrl = weighted_mean(post_ctrl, 'fulltime', 'PERWT')

simple_did = (wt_post_treat - wt_pre_treat) - (wt_post_ctrl - wt_pre_ctrl)
print(f"\nSimple Weighted DiD Estimate: {simple_did:.4f}")
print(f"  Treatment change: {wt_post_treat - wt_pre_treat:.4f}")
print(f"  Control change: {wt_post_ctrl - wt_pre_ctrl:.4f}")

# Summary statistics table
summary_stats = []
for treat, group_name in [(1, 'Treatment'), (0, 'Control')]:
    for post, period_name in [(0, 'Pre'), (1, 'Post')]:
        subset = df[(df['treatment'] == treat) & (df['post'] == post)]
        stats_row = {
            'Group': group_name,
            'Period': period_name,
            'N': len(subset),
            'Full-time Rate': weighted_mean(subset, 'fulltime', 'PERWT'),
            'Female Pct': weighted_mean(subset, 'female', 'PERWT'),
            'Married Pct': weighted_mean(subset, 'married', 'PERWT'),
            'Mean Age': weighted_mean(subset, 'AGE', 'PERWT'),
            'Mean Education': weighted_mean(subset, 'EDUC', 'PERWT')
        }
        summary_stats.append(stats_row)

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_statistics.csv', index=False)
print("\nSaved summary_statistics.csv")

# =============================================================================
# 5. REGRESSION ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("REGRESSION ANALYSIS")
print("="*60)

# Model 1: Basic DiD (OLS, no controls)
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls('fulltime ~ treatment + post + treat_post',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster',
                                            cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient (treat_post): {model1.params['treat_post']:.6f}")
print(f"Standard Error: {model1.bse['treat_post']:.6f}")
print(f"95% CI: [{model1.conf_int().loc['treat_post', 0]:.6f}, {model1.conf_int().loc['treat_post', 1]:.6f}]")
print(f"p-value: {model1.pvalues['treat_post']:.6f}")

# Model 2: With demographic controls
print("\n--- Model 2: With demographic controls ---")
model2 = smf.wls('fulltime ~ treatment + post + treat_post + female + married + C(educ_cat)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster',
                                            cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient (treat_post): {model2.params['treat_post']:.6f}")
print(f"Standard Error: {model2.bse['treat_post']:.6f}")
print(f"95% CI: [{model2.conf_int().loc['treat_post', 0]:.6f}, {model2.conf_int().loc['treat_post', 1]:.6f}]")

# Model 3: With state fixed effects
print("\n--- Model 3: With state fixed effects ---")
model3 = smf.wls('fulltime ~ treatment + post + treat_post + female + married + C(educ_cat) + C(STATEFIP)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster',
                                            cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient (treat_post): {model3.params['treat_post']:.6f}")
print(f"Standard Error: {model3.bse['treat_post']:.6f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.6f}, {model3.conf_int().loc['treat_post', 1]:.6f}]")

# Model 4: With year and state fixed effects (PREFERRED MODEL)
print("\n--- Model 4: With year and state fixed effects (PREFERRED) ---")
model4 = smf.wls('fulltime ~ treatment + treat_post + female + married + C(educ_cat) + C(STATEFIP) + C(YEAR)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster',
                                            cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient (treat_post): {model4.params['treat_post']:.6f}")
print(f"Standard Error: {model4.bse['treat_post']:.6f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.6f}, {model4.conf_int().loc['treat_post', 1]:.6f}]")
print(f"p-value: {model4.pvalues['treat_post']:.6f}")

# Model 5: With age fixed effects
print("\n--- Model 5: With age fixed effects ---")
model5 = smf.wls('fulltime ~ treatment + treat_post + female + married + C(educ_cat) + C(STATEFIP) + C(YEAR) + C(AGE)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster',
                                            cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient (treat_post): {model5.params['treat_post']:.6f}")
print(f"Standard Error: {model5.bse['treat_post']:.6f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.6f}, {model5.conf_int().loc['treat_post', 1]:.6f}]")

# =============================================================================
# 6. HETEROGENEITY ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("HETEROGENEITY ANALYSIS")
print("="*60)

het_results = []

# By gender
print("\n--- By Gender ---")
for gender, name in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == gender]
    model_het = smf.wls('fulltime ~ treatment + post + treat_post',
                         data=subset,
                         weights=subset['PERWT']).fit(cov_type='cluster',
                                                       cov_kwds={'groups': subset['STATEFIP']})
    print(f"{name}: DiD = {model_het.params['treat_post']:.4f} (SE = {model_het.bse['treat_post']:.4f})")
    het_results.append({'Subgroup': name, 'Coefficient': model_het.params['treat_post'],
                        'SE': model_het.bse['treat_post'], 'N': len(subset)})

# By education
print("\n--- By Education Level ---")
for educ_level in ['less_hs', 'high_school', 'some_college']:
    subset = df[df['educ_cat'] == educ_level]
    if len(subset) > 500:
        model_het = smf.wls('fulltime ~ treatment + post + treat_post',
                             data=subset,
                             weights=subset['PERWT']).fit(cov_type='cluster',
                                                           cov_kwds={'groups': subset['STATEFIP']})
        print(f"{educ_level}: DiD = {model_het.params['treat_post']:.4f} (SE = {model_het.bse['treat_post']:.4f})")
        het_results.append({'Subgroup': educ_level, 'Coefficient': model_het.params['treat_post'],
                            'SE': model_het.bse['treat_post'], 'N': len(subset)})

pd.DataFrame(het_results).to_csv('heterogeneity_results.csv', index=False)
print("\nSaved heterogeneity_results.csv")

# =============================================================================
# 7. EVENT STUDY / PARALLEL TRENDS
# =============================================================================

print("\n" + "="*60)
print("EVENT STUDY ANALYSIS")
print("="*60)

# Create year-specific treatment effects (2011 as reference)
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for yr in event_years:
    df[f'treat_year_{yr}'] = df['treatment'] * (df['YEAR'] == yr).astype(int)

# Event study regression
event_formula = 'fulltime ~ treatment + ' + ' + '.join([f'treat_year_{yr}' for yr in event_years]) + ' + C(YEAR) + C(STATEFIP)'
model_event = smf.wls(event_formula,
                       data=df,
                       weights=df['PERWT']).fit(cov_type='cluster',
                                                 cov_kwds={'groups': df['STATEFIP']})

print("\nEvent Study Coefficients (relative to 2011):")
event_results = []
for yr in event_years:
    coef = model_event.params[f'treat_year_{yr}']
    se = model_event.bse[f'treat_year_{yr}']
    ci_low, ci_high = model_event.conf_int().loc[f'treat_year_{yr}']
    print(f"  {yr}: {coef:.4f} (SE = {se:.4f}), 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    event_results.append({'year': yr, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})

# Add reference year
event_results.append({'year': 2011, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
event_df = pd.DataFrame(event_results).sort_values('year')
event_df.to_csv('event_study_results.csv', index=False)
print("\nSaved event_study_results.csv")

# =============================================================================
# 8. SAVE REGRESSION RESULTS
# =============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_df = pd.DataFrame({
    'Model': ['Basic DiD', 'With Controls', 'State FE', 'Year+State FE (Preferred)', 'With Age FE'],
    'Coefficient': [model1.params['treat_post'], model2.params['treat_post'],
                    model3.params['treat_post'], model4.params['treat_post'],
                    model5.params['treat_post']],
    'SE': [model1.bse['treat_post'], model2.bse['treat_post'],
           model3.bse['treat_post'], model4.bse['treat_post'],
           model5.bse['treat_post']],
    'CI_Lower': [model1.conf_int().loc['treat_post', 0], model2.conf_int().loc['treat_post', 0],
                 model3.conf_int().loc['treat_post', 0], model4.conf_int().loc['treat_post', 0],
                 model5.conf_int().loc['treat_post', 0]],
    'CI_Upper': [model1.conf_int().loc['treat_post', 1], model2.conf_int().loc['treat_post', 1],
                 model3.conf_int().loc['treat_post', 1], model4.conf_int().loc['treat_post', 1],
                 model5.conf_int().loc['treat_post', 1]],
    'p_value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
                model3.pvalues['treat_post'], model4.pvalues['treat_post'],
                model5.pvalues['treat_post']]
})
results_df.to_csv('regression_results.csv', index=False)
print("Saved regression_results.csv")

# =============================================================================
# 9. CREATE FIGURES
# =============================================================================

print("\n" + "="*60)
print("CREATING FIGURES")
print("="*60)

# Figure 1: Parallel Trends
fig, ax = plt.subplots(figsize=(10, 6))

years_all = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
treat_means = []
ctrl_means = []

for yr in years_all:
    t_subset = df[(df['YEAR'] == yr) & (df['treatment'] == 1)]
    c_subset = df[(df['YEAR'] == yr) & (df['treatment'] == 0)]
    treat_means.append(weighted_mean(t_subset, 'fulltime', 'PERWT'))
    ctrl_means.append(weighted_mean(c_subset, 'fulltime', 'PERWT'))

ax.plot(years_all, treat_means, 'b-o', label='Treatment (Ages 26-30)', linewidth=2, markersize=8)
ax.plot(years_all, ctrl_means, 'r-s', label='Control (Ages 31-35)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA (2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by Treatment Status', fontsize=14)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(years_all)
plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("Saved figure1_parallel_trends.png/pdf")

# Figure 2: Event Study
fig, ax = plt.subplots(figsize=(10, 6))

event_plot_df = event_df.sort_values('year')
years_plot = event_plot_df['year'].values
coefs_plot = event_plot_df['coef'].values
ci_low_plot = event_plot_df['ci_low'].values
ci_high_plot = event_plot_df['ci_high'].values

ax.errorbar(years_plot, coefs_plot,
            yerr=[coefs_plot - ci_low_plot, ci_high_plot - coefs_plot],
            fmt='o', color='blue', ecolor='blue', capsize=5, capthick=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA (2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Effect on Full-Time Employment', fontsize=12)
ax.set_title('Event Study: Treatment Effects by Year (Reference: 2011)', fontsize=14)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(years_plot)
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("Saved figure2_event_study.png/pdf")

# Figure 3: DiD Visualization
fig, ax = plt.subplots(figsize=(8, 6))

periods = ['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)']
treat_vals = [wt_pre_treat, wt_post_treat]
ctrl_vals = [wt_pre_ctrl, wt_post_ctrl]

x = np.arange(len(periods))
width = 0.35

bars1 = ax.bar(x - width/2, treat_vals, width, label='Treatment (Ages 26-30)', color='steelblue')
bars2 = ax.bar(x + width/2, ctrl_vals, width, label='Control (Ages 31-35)', color='coral')

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment by Group and Period', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(periods, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, max(max(treat_vals), max(ctrl_vals)) * 1.2)

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
plt.savefig('figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_bars.pdf', bbox_inches='tight')
plt.close()
print("Saved figure3_did_bars.png/pdf")

# Figure 4: Coefficient Plot
fig, ax = plt.subplots(figsize=(8, 5))

models = ['Basic DiD', 'With Controls', 'State FE', 'Year+State FE', 'With Age FE']
coefs = results_df['Coefficient'].values
ci_lows = results_df['CI_Lower'].values
ci_highs = results_df['CI_Upper'].values

y_pos = np.arange(len(models))
ax.errorbar(coefs, y_pos, xerr=[coefs - ci_lows, ci_highs - coefs],
            fmt='o', color='darkblue', ecolor='darkblue', capsize=5, capthick=2, markersize=8)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)', fontsize=12)
ax.set_title('DACA Effect Estimates Across Model Specifications', fontsize=14)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figure4_coefficient_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_coefficient_plot.pdf', bbox_inches='tight')
plt.close()
print("Saved figure4_coefficient_plot.png/pdf")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

# Print final summary
print(f"\n*** PREFERRED ESTIMATE (Model 4: Year and State FE) ***")
print(f"Effect of DACA Eligibility on Full-Time Employment: {model4.params['treat_post']:.4f}")
print(f"Standard Error: {model4.bse['treat_post']:.4f}")
print(f"95% Confidence Interval: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {model4.pvalues['treat_post']:.4f}")
print(f"\nSample Size: {len(df):,}")
print(f"  Treatment Group: {df['treatment'].sum():,}")
print(f"  Control Group: {(df['treatment']==0).sum():,}")

# Save final summary
with open('analysis_summary.txt', 'w') as f:
    f.write("DACA REPLICATION ANALYSIS SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write("Research Question: Effect of DACA eligibility on full-time employment\n\n")
    f.write("Sample Selection:\n")
    f.write("- Hispanic-Mexican ethnicity (HISPAN=1)\n")
    f.write("- Born in Mexico (BPL=200)\n")
    f.write("- Non-citizen (CITIZEN=3)\n")
    f.write("- Immigrated before age 16\n")
    f.write("- Immigration year <= 2007\n")
    f.write("- Treatment: Ages 26-30 at DACA (June 2012)\n")
    f.write("- Control: Ages 31-35 at DACA (June 2012)\n\n")
    f.write(f"Final Sample Size: {len(df):,}\n")
    f.write(f"  Treatment: {df['treatment'].sum():,}\n")
    f.write(f"  Control: {(df['treatment']==0).sum():,}\n\n")
    f.write("PREFERRED ESTIMATE (Model 4: Year and State FE):\n")
    f.write(f"  Coefficient: {model4.params['treat_post']:.4f}\n")
    f.write(f"  Standard Error: {model4.bse['treat_post']:.4f}\n")
    f.write(f"  95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]\n")
    f.write(f"  p-value: {model4.pvalues['treat_post']:.4f}\n\n")
    f.write("Weighted Full-Time Employment Rates:\n")
    f.write(f"  Treatment, Pre: {wt_pre_treat:.4f}\n")
    f.write(f"  Treatment, Post: {wt_post_treat:.4f}\n")
    f.write(f"  Control, Pre: {wt_pre_ctrl:.4f}\n")
    f.write(f"  Control, Post: {wt_post_ctrl:.4f}\n")
    f.write(f"  Simple DiD: {simple_did:.4f}\n")

print("\nSaved analysis_summary.txt")
print("\nAll outputs saved successfully!")
