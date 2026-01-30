"""
DACA Replication Study: Effect on Full-Time Employment
Analysis Script

This script performs a difference-in-differences analysis to estimate the causal
effect of DACA eligibility on full-time employment among Hispanic-Mexican,
Mexican-born non-citizens.
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
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)

print("=" * 80)
print("DACA REPLICATION STUDY: EFFECT ON FULL-TIME EMPLOYMENT")
print("=" * 80)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations loaded: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")
print(f"   Columns: {list(df.columns)}")

# Save initial summary
initial_summary = {
    'total_obs': len(df),
    'years': sorted(df['YEAR'].unique().tolist())
}

# ============================================================================
# STEP 2: DEFINE SAMPLE RESTRICTIONS
# ============================================================================
print("\n2. APPLYING SAMPLE RESTRICTIONS...")

# We need Hispanic-Mexican (HISPAN == 1), born in Mexico (BPL == 200), non-citizens (CITIZEN == 3)
# Additional DACA eligibility: arrived before age 16, in US since 2007

# First, filter to years we need (2006-2011 pre, 2013-2016 post; exclude 2012)
df = df[df['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df):,}")

# Filter to Hispanic-Mexican ethnicity (HISPAN == 1)
df = df[df['HISPAN'] == 1].copy()
print(f"   After filtering to Hispanic-Mexican (HISPAN=1): {len(df):,}")

# Filter to born in Mexico (BPL == 200)
df = df[df['BPL'] == 200].copy()
print(f"   After filtering to born in Mexico (BPL=200): {len(df):,}")

# Filter to non-citizens (CITIZEN == 3)
df = df[df['CITIZEN'] == 3].copy()
print(f"   After filtering to non-citizens (CITIZEN=3): {len(df):,}")

# Calculate age as of June 15, 2012
# DACA was implemented on June 15, 2012
# We need to determine who was 26-30 vs 31-35 on that date
# Age on June 15, 2012 = 2012 - BIRTHYR (approximately)
# We'll use BIRTHQTR to be more precise when possible

# For someone to be 26 on June 15, 2012:
# - If born Jan-June (BIRTHQTR 1 or 2): birth year should be 1986 (2012 - 26 = 1986)
# - If born July-Dec (BIRTHQTR 3 or 4): birth year should be 1985 (they'd turn 27 later in year)

# We'll calculate age_on_daca_date more precisely
# But simpler approach: use the midpoint of year
def calculate_age_on_june15_2012(row):
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']

    # June 15 is in Q2
    # If birthqtr is 1 or 2 (Jan-Jun), they've already had birthday by June 15
    # If birthqtr is 3 or 4 (Jul-Dec), they haven't had birthday yet

    if birth_qtr in [1, 2]:
        age = 2012 - birth_year
    else:  # birth_qtr in [3, 4] or missing
        age = 2012 - birth_year - 1

    return age

# Handle missing BIRTHQTR (code 0 or 9)
# For these, we'll use a simple approximation: 2012 - BIRTHYR
df['age_on_daca'] = df.apply(
    lambda row: 2012 - row['BIRTHYR'] if row['BIRTHQTR'] in [0, 9]
    else calculate_age_on_june15_2012(row),
    axis=1
)

print(f"\n   Age distribution on June 15, 2012:")
print(df['age_on_daca'].describe())

# DACA eligibility: arrived before age 16
# We need year of immigration to calculate age at arrival
# Age at arrival = YRIMMIG - BIRTHYR
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Filter to those who arrived before age 16
# Handle cases where YRIMMIG is 0 (not available) - these can't be verified
df_eligible = df[(df['YRIMMIG'] > 0) & (df['age_at_arrival'] < 16)].copy()
print(f"\n   After filtering to arrived before age 16: {len(df_eligible):,}")

# Additional check: must have been in US since June 15, 2007 (YRIMMIG <= 2007)
df_eligible = df_eligible[df_eligible['YRIMMIG'] <= 2007].copy()
print(f"   After filtering to in US since 2007 (YRIMMIG<=2007): {len(df_eligible):,}")

# Now filter to our age groups of interest
# Treatment group: ages 26-30 on June 15, 2012
# Control group: ages 31-35 on June 15, 2012
df_analysis = df_eligible[(df_eligible['age_on_daca'] >= 26) & (df_eligible['age_on_daca'] <= 35)].copy()
print(f"\n   After filtering to ages 26-35 on DACA date: {len(df_analysis):,}")

# Create treatment indicator (1 if in treatment group, 0 if control)
df_analysis['treat'] = (df_analysis['age_on_daca'] <= 30).astype(int)

print(f"\n   Treatment group (ages 26-30): {(df_analysis['treat']==1).sum():,}")
print(f"   Control group (ages 31-35): {(df_analysis['treat']==0).sum():,}")

# Create post indicator (1 if after DACA, 0 if before)
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

print(f"\n   Pre-period (2006-2011): {(df_analysis['post']==0).sum():,}")
print(f"   Post-period (2013-2016): {(df_analysis['post']==1).sum():,}")

# ============================================================================
# STEP 3: DEFINE OUTCOME VARIABLE
# ============================================================================
print("\n3. DEFINING OUTCOME VARIABLE...")

# Full-time employment: usually working 35+ hours per week (UHRSWORK >= 35)
# Note: UHRSWORK = 0 typically means not employed or N/A
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"   Full-time employment rate: {df_analysis['fulltime'].mean():.4f}")
print(f"   N working full-time: {df_analysis['fulltime'].sum():,}")
print(f"   N not working full-time: {(df_analysis['fulltime']==0).sum():,}")

# ============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n4. DESCRIPTIVE STATISTICS...")

# Summary by treatment and period
summary_stats = df_analysis.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'SEX': lambda x: (x == 2).mean(),  # proportion female
    'EDUC': 'mean',
    'MARST': lambda x: (x == 1).mean(),  # proportion married with spouse present
    'PERWT': 'sum'
}).round(4)

print("\n   Summary Statistics by Treatment Group and Period:")
print(summary_stats)

# Calculate weighted means
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

print("\n   Weighted Full-Time Employment Rates:")
for treat in [0, 1]:
    for post in [0, 1]:
        subset = df_analysis[(df_analysis['treat']==treat) & (df_analysis['post']==post)]
        if len(subset) > 0:
            wt_mean = weighted_mean(subset, 'fulltime', 'PERWT')
            label_treat = "Treatment (26-30)" if treat == 1 else "Control (31-35)"
            label_post = "Post (2013-2016)" if post == 1 else "Pre (2006-2011)"
            print(f"   {label_treat}, {label_post}: {wt_mean:.4f} (n={len(subset):,})")

# ============================================================================
# STEP 5: SIMPLE DIFFERENCE-IN-DIFFERENCES
# ============================================================================
print("\n5. SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION...")

# Calculate means for 2x2 table
means = df_analysis.groupby(['treat', 'post'])['fulltime'].mean().unstack()
print("\n   Full-time Employment Rates (unweighted):")
print(means)

# DiD calculation
# (Treatment Post - Treatment Pre) - (Control Post - Control Pre)
treat_diff = means.loc[1, 1] - means.loc[1, 0]
control_diff = means.loc[0, 1] - means.loc[0, 0]
did_estimate = treat_diff - control_diff

print(f"\n   Treatment group change: {treat_diff:.4f}")
print(f"   Control group change: {control_diff:.4f}")
print(f"   Difference-in-Differences estimate: {did_estimate:.4f}")

# Weighted DiD calculation
def calc_weighted_mean(df, treat_val, post_val):
    subset = df[(df['treat']==treat_val) & (df['post']==post_val)]
    return np.average(subset['fulltime'], weights=subset['PERWT'])

wt_means = {}
for t in [0, 1]:
    for p in [0, 1]:
        wt_means[(t, p)] = calc_weighted_mean(df_analysis, t, p)

wt_treat_diff = wt_means[(1,1)] - wt_means[(1,0)]
wt_control_diff = wt_means[(0,1)] - wt_means[(0,0)]
wt_did_estimate = wt_treat_diff - wt_control_diff

print(f"\n   Weighted DiD Estimate: {wt_did_estimate:.4f}")

# ============================================================================
# STEP 6: REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES
# ============================================================================
print("\n6. REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES...")

# Create interaction term
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

# Model 1: Basic DiD without controls
print("\n   Model 1: Basic DiD (no controls)")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model1.summary())

# Save key results
did_coef = model1.params['treat_post']
did_se = model1.bse['treat_post']
did_pval = model1.pvalues['treat_post']
did_ci_low = model1.conf_int().loc['treat_post', 0]
did_ci_high = model1.conf_int().loc['treat_post', 1]
n_obs_model1 = int(model1.nobs)

print(f"\n   DiD Coefficient: {did_coef:.4f}")
print(f"   Standard Error: {did_se:.4f}")
print(f"   95% CI: [{did_ci_low:.4f}, {did_ci_high:.4f}]")
print(f"   P-value: {did_pval:.4f}")

# Model 2: DiD with demographic controls
print("\n   Model 2: DiD with demographic controls")

# Create control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Create education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # HS or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # Some college or more

model2 = smf.wls('fulltime ~ treat + post + treat_post + AGE + age_sq + female + married + educ_hs',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model2.summary())

print(f"\n   DiD Coefficient (with controls): {model2.params['treat_post']:.4f}")
print(f"   Standard Error: {model2.bse['treat_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")

# Model 3: DiD with year and state fixed effects
print("\n   Model 3: DiD with year fixed effects")

# Create year dummies
df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)

model3 = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + AGE + age_sq + female + married + educ_hs',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

print(f"\n   DiD Coefficient (with year FE): {model3.params['treat_post']:.4f}")
print(f"   Standard Error: {model3.bse['treat_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")

# Model 4: DiD with state fixed effects
print("\n   Model 4: DiD with year and state fixed effects")
model4 = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + AGE + age_sq + female + married + educ_hs',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

print(f"\n   DiD Coefficient (with year & state FE): {model4.params['treat_post']:.4f}")
print(f"   Standard Error: {model4.bse['treat_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

# ============================================================================
# STEP 7: ROBUSTNESS CHECKS
# ============================================================================
print("\n7. ROBUSTNESS CHECKS...")

# A. Placebo test using pre-period data only (2006-2008 vs 2009-2011)
print("\n   A. Placebo Test (Pre-Period Only):")
df_pre = df_analysis[df_analysis['post'] == 0].copy()
df_pre['fake_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['treat_fake_post'] = df_pre['treat'] * df_pre['fake_post']

model_placebo = smf.wls('fulltime ~ treat + fake_post + treat_fake_post',
                         data=df_pre,
                         weights=df_pre['PERWT']).fit(cov_type='HC1')

print(f"   Placebo DiD Coefficient: {model_placebo.params['treat_fake_post']:.4f}")
print(f"   Standard Error: {model_placebo.bse['treat_fake_post']:.4f}")
print(f"   P-value: {model_placebo.pvalues['treat_fake_post']:.4f}")

# B. By gender
print("\n   B. Heterogeneity by Gender:")
for sex, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_analysis[df_analysis['SEX'] == sex]
    model_sex = smf.wls('fulltime ~ treat + post + treat_post',
                        data=df_sex,
                        weights=df_sex['PERWT']).fit(cov_type='HC1')
    print(f"   {sex_label}: DiD = {model_sex.params['treat_post']:.4f} (SE = {model_sex.bse['treat_post']:.4f}), n = {len(df_sex):,}")

# C. By education level
print("\n   C. Heterogeneity by Education:")
for educ_val, educ_label in [(0, 'Less than HS'), (1, 'HS or more')]:
    df_educ = df_analysis[df_analysis['educ_hs'] == educ_val]
    if len(df_educ) > 100:
        model_educ = smf.wls('fulltime ~ treat + post + treat_post',
                             data=df_educ,
                             weights=df_educ['PERWT']).fit(cov_type='HC1')
        print(f"   {educ_label}: DiD = {model_educ.params['treat_post']:.4f} (SE = {model_educ.bse['treat_post']:.4f}), n = {len(df_educ):,}")

# ============================================================================
# STEP 8: EVENT STUDY / DYNAMIC EFFECTS
# ============================================================================
print("\n8. EVENT STUDY ANALYSIS...")

# Create year-by-treatment interactions
years = sorted(df_analysis['YEAR'].unique())
print(f"   Years in analysis: {years}")

# Use 2011 as reference year (last pre-treatment year)
df_analysis['ref_year'] = (df_analysis['YEAR'] == 2011).astype(int)

event_study_results = []
for year in years:
    if year != 2011:  # Skip reference year
        df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype(int)
        df_analysis[f'treat_year_{year}'] = df_analysis['treat'] * df_analysis[f'year_{year}']

# Build formula for event study
year_terms = ' + '.join([f'year_{y}' for y in years if y != 2011])
interact_terms = ' + '.join([f'treat_year_{y}' for y in years if y != 2011])
formula = f'fulltime ~ treat + {year_terms} + {interact_terms}'

model_event = smf.wls(formula, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\n   Event Study Coefficients (relative to 2011):")
for year in years:
    if year != 2011:
        coef = model_event.params[f'treat_year_{year}']
        se = model_event.bse[f'treat_year_{year}']
        ci_low = model_event.conf_int().loc[f'treat_year_{year}', 0]
        ci_high = model_event.conf_int().loc[f'treat_year_{year}', 1]
        print(f"   {year}: {coef:.4f} (SE: {se:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
        event_study_results.append({
            'year': year,
            'coef': coef,
            'se': se,
            'ci_low': ci_low,
            'ci_high': ci_high
        })

# Add reference year (2011) with 0 values
event_study_results.append({'year': 2011, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
event_study_df = pd.DataFrame(event_study_results).sort_values('year')

# ============================================================================
# STEP 9: CREATE VISUALIZATIONS
# ============================================================================
print("\n9. CREATING VISUALIZATIONS...")

# Figure 1: Parallel trends plot
fig, ax = plt.subplots(figsize=(10, 6))

yearly_means = df_analysis.groupby(['YEAR', 'treat']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

ax.plot(yearly_means.index, yearly_means[0], 'b-o', label='Control (ages 31-35)', linewidth=2)
ax.plot(yearly_means.index, yearly_means[1], 'r-s', label='Treatment (ages 26-30)', linewidth=2)
ax.axvline(x=2012.5, color='gray', linestyle='--', label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by Treatment Group Over Time', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print("   Saved: figure1_parallel_trends.png")

# Figure 2: Event study plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(event_study_df['year'], event_study_df['coef'],
            yerr=[event_study_df['coef'] - event_study_df['ci_low'],
                  event_study_df['ci_high'] - event_study_df['coef']],
            fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()

print("   Saved: figure2_event_study.png")

# Figure 3: Sample composition by year
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# By treatment group
yearly_counts = df_analysis.groupby(['YEAR', 'treat']).size().unstack()
yearly_counts.plot(kind='bar', ax=axes[0])
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Sample Size', fontsize=12)
axes[0].set_title('Sample Size by Year and Treatment Group', fontsize=12)
axes[0].legend(['Control (31-35)', 'Treatment (26-30)'])
axes[0].tick_params(axis='x', rotation=45)

# Full-time rate histogram
axes[1].hist([df_analysis[df_analysis['treat']==0]['fulltime'],
              df_analysis[df_analysis['treat']==1]['fulltime']],
             label=['Control', 'Treatment'], alpha=0.7)
axes[1].set_xlabel('Full-Time Employment', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of Full-Time Employment', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig('figure3_sample_composition.png', dpi=300, bbox_inches='tight')
plt.close()

print("   Saved: figure3_sample_composition.png")

# ============================================================================
# STEP 10: SAVE RESULTS TABLES
# ============================================================================
print("\n10. SAVING RESULTS TABLES...")

# Create results summary for export
results_dict = {
    'Specification': ['Basic DiD', 'With Demographics', 'With Year FE', 'With Year & State FE', 'Placebo (Pre-period)'],
    'DiD Estimate': [
        model1.params['treat_post'],
        model2.params['treat_post'],
        model3.params['treat_post'],
        model4.params['treat_post'],
        model_placebo.params['treat_fake_post']
    ],
    'Std Error': [
        model1.bse['treat_post'],
        model2.bse['treat_post'],
        model3.bse['treat_post'],
        model4.bse['treat_post'],
        model_placebo.bse['treat_fake_post']
    ],
    '95% CI Lower': [
        model1.conf_int().loc['treat_post', 0],
        model2.conf_int().loc['treat_post', 0],
        model3.conf_int().loc['treat_post', 0],
        model4.conf_int().loc['treat_post', 0],
        model_placebo.conf_int().loc['treat_fake_post', 0]
    ],
    '95% CI Upper': [
        model1.conf_int().loc['treat_post', 1],
        model2.conf_int().loc['treat_post', 1],
        model3.conf_int().loc['treat_post', 1],
        model4.conf_int().loc['treat_post', 1],
        model_placebo.conf_int().loc['treat_fake_post', 1]
    ],
    'P-value': [
        model1.pvalues['treat_post'],
        model2.pvalues['treat_post'],
        model3.pvalues['treat_post'],
        model4.pvalues['treat_post'],
        model_placebo.pvalues['treat_fake_post']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model_placebo.nobs)
    ]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('results_table.csv', index=False)
print("   Saved: results_table.csv")

# Save event study results
event_study_df.to_csv('event_study_results.csv', index=False)
print("   Saved: event_study_results.csv")

# Summary statistics table
print("\n   Creating summary statistics table...")
summary_by_group = df_analysis.groupby('treat').agg({
    'fulltime': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'PERWT': 'sum'
}).round(4)
summary_by_group.columns = ['Full-Time Rate', 'Mean Age', 'Prop. Female', 'Prop. Married', 'Prop. HS+', 'Total Weight']
summary_by_group.index = ['Control (31-35)', 'Treatment (26-30)']
summary_by_group.to_csv('summary_statistics.csv')
print("   Saved: summary_statistics.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

print(f"""
Research Question: Effect of DACA eligibility on full-time employment

Sample:
  - Hispanic-Mexican, Mexican-born non-citizens
  - Arrived before age 16, in US since 2007
  - Treatment: ages 26-30 on June 15, 2012
  - Control: ages 31-35 on June 15, 2012
  - N = {len(df_analysis):,}

PREFERRED ESTIMATE (Basic DiD with robust standard errors):
  - DiD Coefficient: {did_coef:.4f}
  - Standard Error: {did_se:.4f}
  - 95% Confidence Interval: [{did_ci_low:.4f}, {did_ci_high:.4f}]
  - P-value: {did_pval:.4f}

Interpretation:
  DACA eligibility is associated with a {abs(did_coef)*100:.2f} percentage point
  {"increase" if did_coef > 0 else "decrease"} in the probability of full-time employment.
  This effect is {"statistically significant" if did_pval < 0.05 else "not statistically significant"} at the 5% level.

Robustness:
  - Results are {"robust" if abs(model2.params['treat_post'] - did_coef) < 0.02 else "sensitive"} to including demographic controls
  - Placebo test (pre-period): coef = {model_placebo.params['treat_fake_post']:.4f} (p = {model_placebo.pvalues['treat_fake_post']:.4f})
""")

# Save the final summary to a text file
with open('analysis_summary.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - ANALYSIS SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Sample Size: {len(df_analysis):,}\n")
    f.write(f"Treatment Group (ages 26-30): {(df_analysis['treat']==1).sum():,}\n")
    f.write(f"Control Group (ages 31-35): {(df_analysis['treat']==0).sum():,}\n")
    f.write(f"Pre-period observations: {(df_analysis['post']==0).sum():,}\n")
    f.write(f"Post-period observations: {(df_analysis['post']==1).sum():,}\n\n")
    f.write("MAIN RESULTS\n")
    f.write("-" * 40 + "\n")
    f.write(f"DiD Estimate: {did_coef:.4f}\n")
    f.write(f"Standard Error: {did_se:.4f}\n")
    f.write(f"95% CI: [{did_ci_low:.4f}, {did_ci_high:.4f}]\n")
    f.write(f"P-value: {did_pval:.4f}\n\n")
    f.write("FULL-TIME EMPLOYMENT RATES\n")
    f.write("-" * 40 + "\n")
    f.write(f"Treatment, Pre: {wt_means[(1,0)]:.4f}\n")
    f.write(f"Treatment, Post: {wt_means[(1,1)]:.4f}\n")
    f.write(f"Control, Pre: {wt_means[(0,0)]:.4f}\n")
    f.write(f"Control, Post: {wt_means[(0,1)]:.4f}\n")

print("\n   Saved: analysis_summary.txt")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
