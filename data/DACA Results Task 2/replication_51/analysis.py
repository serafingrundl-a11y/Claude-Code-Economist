"""
DACA Replication Study - Difference-in-Differences Analysis
Effect of DACA eligibility on full-time employment among Hispanic-Mexican immigrants
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(51)

print("=" * 70)
print("DACA REPLICATION STUDY - Analysis Script")
print("=" * 70)

# ============================================================================
# STEP 1: Load and Initial Data Exploration
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in dataset: {sorted(df['YEAR'].unique())}")

# ============================================================================
# STEP 2: Define Sample Selection Criteria
# ============================================================================
print("\n[2] Applying sample selection criteria...")

# Keep only relevant years (exclude 2012 due to ambiguity about timing)
df = df[df['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])]
print(f"After excluding 2012: {len(df):,}")

# Hispanic-Mexican ethnicity (HISPAN == 1 means Mexican)
df = df[df['HISPAN'] == 1]
print(f"After Hispanic-Mexican filter: {len(df):,}")

# Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200]
print(f"After born in Mexico filter: {len(df):,}")

# Non-citizens (CITIZEN == 3 means "Not a citizen")
# We assume non-naturalized immigrants without papers are undocumented
df = df[df['CITIZEN'] == 3]
print(f"After non-citizen filter: {len(df):,}")

# Filter based on immigration year - must have arrived by 2007 to meet continuous presence
# and this must be in the data (not 0 which is N/A)
df = df[(df['YRIMMIG'] > 0) & (df['YRIMMIG'] <= 2007)]
print(f"After YRIMMIG <= 2007 filter: {len(df):,}")

# Age requirement: must have arrived before turning 16
# Calculate age at immigration
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_immigration'] < 16]
print(f"After arrived before age 16 filter: {len(df):,}")

# ============================================================================
# STEP 3: Define Treatment and Control Groups Based on Age as of June 15, 2012
# ============================================================================
print("\n[3] Defining treatment and control groups...")

# Age on June 15, 2012
# If born in Q1/Q2 (Jan-Jun), they had their birthday by June 15
# If born in Q3/Q4 (Jul-Dec), they hadn't had their 2012 birthday yet by June 15

def age_on_june15_2012(birthyr, birthqtr):
    """Calculate age as of June 15, 2012"""
    base_age = 2012 - birthyr
    # Q1 (Jan-Mar) and Q2 (Apr-Jun): already had birthday by June 15
    # Q3 (Jul-Sep) and Q4 (Oct-Dec): haven't had 2012 birthday yet
    if birthqtr in [1, 2]:
        return base_age
    else:
        return base_age - 1

df['age_june2012'] = df.apply(lambda row: age_on_june15_2012(row['BIRTHYR'], row['BIRTHQTR']), axis=1)

# Treatment group: ages 26-30 on June 15, 2012 (DACA eligible by age)
# Control group: ages 31-35 on June 15, 2012 (too old for DACA)
df['treated'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)
df['control'] = ((df['age_june2012'] >= 31) & (df['age_june2012'] <= 35)).astype(int)

# Keep only treatment and control observations
df = df[(df['treated'] == 1) | (df['control'] == 1)]
print(f"After age-based treatment/control selection: {len(df):,}")

# ============================================================================
# STEP 4: Define Outcome Variable and Time Period
# ============================================================================
print("\n[4] Defining outcome variable and time indicators...")

# Full-time employment: UHRSWORK >= 35
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Post-treatment indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term for DiD
df['treated_post'] = df['treated'] * df['post']

# ============================================================================
# STEP 5: Summary Statistics
# ============================================================================
print("\n[5] Summary Statistics")
print("-" * 70)

# Overall sample characteristics
print(f"\nTotal analysis sample: {len(df):,}")
print(f"Treatment group (age 26-30): {df['treated'].sum():,}")
print(f"Control group (age 31-35): {df['control'].sum():,}")
print(f"Pre-period observations: {(df['post']==0).sum():,}")
print(f"Post-period observations: {(df['post']==1).sum():,}")

# Full-time employment by group and period
print("\n--- Full-time Employment Rates ---")
for treated_val, group_name in [(1, "Treatment (26-30)"), (0, "Control (31-35)")]:
    for post_val, period_name in [(0, "Pre (2006-2011)"), (1, "Post (2013-2016)")]:
        subset = df[(df['treated']==treated_val) & (df['post']==post_val)]
        rate = subset['fulltime'].mean()
        n = len(subset)
        print(f"{group_name}, {period_name}: {rate:.4f} (n={n:,})")

# Calculate simple DiD manually
pre_treat = df[(df['treated']==1) & (df['post']==0)]['fulltime'].mean()
post_treat = df[(df['treated']==1) & (df['post']==1)]['fulltime'].mean()
pre_control = df[(df['treated']==0) & (df['post']==0)]['fulltime'].mean()
post_control = df[(df['treated']==0) & (df['post']==1)]['fulltime'].mean()

did_manual = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\nSimple DiD estimate: {did_manual:.4f}")
print(f"  Treatment change: {post_treat - pre_treat:.4f}")
print(f"  Control change: {post_control - pre_control:.4f}")

# ============================================================================
# STEP 6: Main DiD Regression Analysis (Unweighted)
# ============================================================================
print("\n[6] Difference-in-Differences Regression Analysis")
print("-" * 70)

# Model 1: Basic DiD without covariates
print("\n--- Model 1: Basic DiD (No Covariates) ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])

# ============================================================================
# STEP 7: DiD with Covariates
# ============================================================================
print("\n--- Model 2: DiD with Demographic Covariates ---")

# Create categorical variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education dummies
df['educ_hs'] = ((df['EDUCD'] >= 62) & (df['EDUCD'] <= 64)).astype(int)  # High school
df['educ_somecoll'] = ((df['EDUCD'] >= 65) & (df['EDUCD'] <= 100)).astype(int)  # Some college
df['educ_ba'] = (df['EDUCD'] >= 101).astype(int)  # Bachelor's or more

# Current age in survey year
df['age_current'] = df['AGE']
df['age_current_sq'] = df['age_current'] ** 2

model2 = smf.ols('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecoll + educ_ba + age_current + age_current_sq',
                  data=df).fit(cov_type='HC1')
print(model2.summary().tables[1])

# ============================================================================
# STEP 8: DiD with Year Fixed Effects
# ============================================================================
print("\n--- Model 3: DiD with Year Fixed Effects ---")

# Create year dummies (exclude base year 2006)
for year in [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)

year_vars = ' + '.join([f'year_{y}' for y in [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]])

model3 = smf.ols(f'fulltime ~ treated + treated_post + female + married + educ_hs + educ_somecoll + educ_ba + age_current + age_current_sq + {year_vars}',
                  data=df).fit(cov_type='HC1')
print(model3.summary().tables[1])

# ============================================================================
# STEP 9: Weighted DiD Analysis (using PERWT)
# ============================================================================
print("\n--- Model 4: Weighted DiD with Covariates ---")
model4 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecoll + educ_ba + age_current + age_current_sq',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary().tables[1])

# ============================================================================
# STEP 10: State Fixed Effects (Preferred Specification)
# ============================================================================
print("\n--- Model 5: Weighted DiD with State Fixed Effects (Preferred) ---")

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
df_with_states = pd.concat([df, state_dummies], axis=1)

state_cols = [col for col in state_dummies.columns]
state_vars = ' + '.join(state_cols)

formula_preferred = f'fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecoll + educ_ba + age_current + age_current_sq + {year_vars} + {state_vars}'

model5 = smf.wls(formula_preferred, data=df_with_states, weights=df_with_states['PERWT']).fit(cov_type='HC1')

# Print key coefficients
print("\nKey Coefficients from Preferred Model:")
print(f"  treated_post (DiD estimate): {model5.params['treated_post']:.5f} (SE: {model5.bse['treated_post']:.5f})")
print(f"  95% CI: [{model5.conf_int().loc['treated_post', 0]:.5f}, {model5.conf_int().loc['treated_post', 1]:.5f}]")
print(f"  p-value: {model5.pvalues['treated_post']:.5f}")

# ============================================================================
# STEP 11: Year-by-Year Treatment Effects (Event Study)
# ============================================================================
print("\n[7] Event Study: Year-by-Year Treatment Effects")
print("-" * 70)

# Create interaction of treated with each year
for year in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df_with_states[f'treated_year_{year}'] = (df_with_states['treated'] * (df_with_states['YEAR'] == year)).astype(int)

# Use 2011 as reference year (last pre-treatment year)
event_year_vars = ' + '.join([f'treated_year_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])
formula_event = f'fulltime ~ treated + {event_year_vars} + female + married + educ_hs + educ_somecoll + educ_ba + age_current + age_current_sq + {year_vars} + {state_vars}'

model_event = smf.wls(formula_event, data=df_with_states, weights=df_with_states['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Reference: 2011):")
event_results = []
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treated_year_{year}']
    se = model_event.bse[f'treated_year_{year}']
    ci_low = model_event.conf_int().loc[f'treated_year_{year}', 0]
    ci_high = model_event.conf_int().loc[f'treated_year_{year}', 1]
    pval = model_event.pvalues[f'treated_year_{year}']
    print(f"  {year}: {coef:.5f} (SE: {se:.5f}, p={pval:.4f})")
    event_results.append({'year': year, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})

# Add reference year
event_results.append({'year': 2011, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
event_df = pd.DataFrame(event_results).sort_values('year')

# ============================================================================
# STEP 12: Create Visualizations
# ============================================================================
print("\n[8] Creating Visualizations...")

# Figure 1: Trends in Full-time Employment by Group
fig1, ax1 = plt.subplots(figsize=(10, 6))

years_order = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
treat_means = [df[(df['treated']==1) & (df['YEAR']==y)]['fulltime'].mean() for y in years_order]
control_means = [df[(df['treated']==0) & (df['YEAR']==y)]['fulltime'].mean() for y in years_order]

ax1.plot(years_order, treat_means, 'b-o', label='Treatment (Age 26-30 in 2012)', linewidth=2, markersize=8)
ax1.plot(years_order, control_means, 'r--s', label='Control (Age 31-35 in 2012)', linewidth=2, markersize=8)
ax1.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-time Employment Rate', fontsize=12)
ax1.set_title('Full-time Employment Trends by Treatment Status', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(years_order)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure1_trends.png")

# Figure 2: Event Study Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.errorbar(event_df['year'], event_df['coef'],
             yerr=[event_df['coef'] - event_df['ci_low'], event_df['ci_high'] - event_df['coef']],
             fmt='o', capsize=5, capthick=2, markersize=8, color='navy', ecolor='navy')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Treatment Effect by Year', fontsize=14)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure2_eventstudy.png")

# Figure 3: Coefficient Comparison Across Models
fig3, ax3 = plt.subplots(figsize=(10, 6))

models_data = {
    'Model 1\n(Basic DiD)': (model1.params['treated_post'], model1.bse['treated_post']),
    'Model 2\n(+ Covariates)': (model2.params['treated_post'], model2.bse['treated_post']),
    'Model 3\n(+ Year FE)': (model3.params['treated_post'], model3.bse['treated_post']),
    'Model 4\n(Weighted)': (model4.params['treated_post'], model4.bse['treated_post']),
    'Model 5\n(+ State FE)': (model5.params['treated_post'], model5.bse['treated_post']),
}

model_names = list(models_data.keys())
coefs = [models_data[m][0] for m in model_names]
ses = [models_data[m][1] for m in model_names]

x_pos = range(len(model_names))
ax3.bar(x_pos, coefs, yerr=[1.96*s for s in ses], capsize=5, color='steelblue', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_names, fontsize=10)
ax3.set_ylabel('DiD Coefficient', fontsize=12)
ax3.set_title('Treatment Effect Estimates Across Model Specifications', fontsize=14)
ax3.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figure3_models.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure3_models.png")

# ============================================================================
# STEP 13: Robustness Checks
# ============================================================================
print("\n[9] Robustness Checks")
print("-" * 70)

# Robustness 1: Narrow age bandwidth (27-29 vs 32-34)
print("\n--- Robustness 1: Narrow Age Bandwidth (27-29 vs 32-34) ---")
df_narrow = df[(df['age_june2012'].isin([27, 28, 29])) | (df['age_june2012'].isin([32, 33, 34]))].copy()
df_narrow['treated_narrow'] = df_narrow['age_june2012'].isin([27, 28, 29]).astype(int)
df_narrow['treated_post_narrow'] = df_narrow['treated_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treated_narrow + post + treated_post_narrow + female + married + educ_hs + educ_somecoll + educ_ba + age_current + age_current_sq',
                       data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (narrow band): {model_narrow.params['treated_post_narrow']:.5f} (SE: {model_narrow.bse['treated_post_narrow']:.5f})")

# Robustness 2: Men only
print("\n--- Robustness 2: Men Only ---")
df_men = df[df['female'] == 0].copy()
model_men = smf.wls('fulltime ~ treated + post + treated_post + married + educ_hs + educ_somecoll + educ_ba + age_current + age_current_sq',
                    data=df_men, weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (men only): {model_men.params['treated_post']:.5f} (SE: {model_men.bse['treated_post']:.5f})")

# Robustness 3: Women only
print("\n--- Robustness 3: Women Only ---")
df_women = df[df['female'] == 1].copy()
model_women = smf.wls('fulltime ~ treated + post + treated_post + married + educ_hs + educ_somecoll + educ_ba + age_current + age_current_sq',
                      data=df_women, weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (women only): {model_women.params['treated_post']:.5f} (SE: {model_women.bse['treated_post']:.5f})")

# Robustness 4: Placebo test using 2008 as fake treatment year
print("\n--- Robustness 4: Placebo Test (Fake Treatment in 2008) ---")
df_placebo = df[df['YEAR'].isin([2006, 2007, 2009, 2010, 2011])].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treated_post_placebo'] = df_placebo['treated'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treated + post_placebo + treated_post_placebo + female + married + educ_hs + educ_somecoll + educ_ba + age_current + age_current_sq',
                        data=df_placebo, weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD estimate: {model_placebo.params['treated_post_placebo']:.5f} (SE: {model_placebo.bse['treated_post_placebo']:.5f}, p={model_placebo.pvalues['treated_post_placebo']:.4f})")

# ============================================================================
# STEP 14: Heterogeneity Analysis
# ============================================================================
print("\n[10] Heterogeneity Analysis")
print("-" * 70)

# By education level
print("\n--- By Education Level ---")

# Low education (less than high school)
df_low_ed = df[df['educ_hs'] + df['educ_somecoll'] + df['educ_ba'] == 0].copy()
model_low_ed = smf.wls('fulltime ~ treated + post + treated_post + female + married + age_current + age_current_sq',
                       data=df_low_ed, weights=df_low_ed['PERWT']).fit(cov_type='HC1')
print(f"Less than HS: {model_low_ed.params['treated_post']:.5f} (SE: {model_low_ed.bse['treated_post']:.5f})")

# High school or more
df_high_ed = df[df['educ_hs'] + df['educ_somecoll'] + df['educ_ba'] > 0].copy()
model_high_ed = smf.wls('fulltime ~ treated + post + treated_post + female + married + age_current + age_current_sq',
                        data=df_high_ed, weights=df_high_ed['PERWT']).fit(cov_type='HC1')
print(f"HS or more: {model_high_ed.params['treated_post']:.5f} (SE: {model_high_ed.bse['treated_post']:.5f})")

# ============================================================================
# STEP 15: Final Summary and Export Results
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

# Preferred estimate (Model 5)
preferred_coef = model5.params['treated_post']
preferred_se = model5.bse['treated_post']
preferred_ci_low = model5.conf_int().loc['treated_post', 0]
preferred_ci_high = model5.conf_int().loc['treated_post', 1]
preferred_pval = model5.pvalues['treated_post']
sample_size = len(df)

print(f"\nPreferred Estimate (Model 5: Weighted DiD with State and Year FE):")
print(f"  Effect size: {preferred_coef:.5f}")
print(f"  Standard error: {preferred_se:.5f}")
print(f"  95% Confidence Interval: [{preferred_ci_low:.5f}, {preferred_ci_high:.5f}]")
print(f"  p-value: {preferred_pval:.5f}")
print(f"  Sample size: {sample_size:,}")

# Save results to file
results_dict = {
    'Preferred Estimate': preferred_coef,
    'Standard Error': preferred_se,
    'CI Lower': preferred_ci_low,
    'CI Upper': preferred_ci_high,
    'P-value': preferred_pval,
    'Sample Size': sample_size,
    'N Treatment Pre': len(df[(df['treated']==1) & (df['post']==0)]),
    'N Treatment Post': len(df[(df['treated']==1) & (df['post']==1)]),
    'N Control Pre': len(df[(df['treated']==0) & (df['post']==0)]),
    'N Control Post': len(df[(df['treated']==0) & (df['post']==1)]),
}

# Save summary statistics
summary_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'age_current': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\n--- Summary Statistics by Group ---")
print(summary_stats)

# Export key tables
summary_stats.to_csv('summary_statistics.csv')
print("\nSaved: summary_statistics.csv")

# Create comprehensive results table
results_table = pd.DataFrame({
    'Model': ['Model 1 (Basic)', 'Model 2 (+ Covariates)', 'Model 3 (+ Year FE)',
              'Model 4 (Weighted)', 'Model 5 (Preferred)'],
    'Coefficient': [model1.params['treated_post'], model2.params['treated_post'],
                   model3.params['treated_post'], model4.params['treated_post'],
                   model5.params['treated_post']],
    'Std. Error': [model1.bse['treated_post'], model2.bse['treated_post'],
                  model3.bse['treated_post'], model4.bse['treated_post'],
                  model5.bse['treated_post']],
    'P-value': [model1.pvalues['treated_post'], model2.pvalues['treated_post'],
               model3.pvalues['treated_post'], model4.pvalues['treated_post'],
               model5.pvalues['treated_post']],
})
results_table.to_csv('regression_results.csv', index=False)
print("Saved: regression_results.csv")

# Save event study results
event_df.to_csv('event_study_results.csv', index=False)
print("Saved: event_study_results.csv")

print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)
