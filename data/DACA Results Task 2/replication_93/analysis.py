"""
DACA Effect on Full-Time Employment Analysis
Replication Study using Difference-in-Differences

Research Question:
Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA (treatment) on the probability
of being employed full-time (>=35 hours/week)?

Treatment Group: Ages 26-30 at DACA implementation (June 15, 2012)
Control Group: Ages 31-35 at DACA implementation (who would have been eligible if not for age)

Pre-period: 2006-2011
Post-period: 2013-2016 (excluding 2012 as partial treatment year)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("DACA EFFECT ON FULL-TIME EMPLOYMENT - REPLICATION ANALYSIS")
print("=" * 70)

# =============================================================================
# STEP 1: Load and Filter Data
# =============================================================================
print("\n[1] Loading data...")

# Read data in chunks to manage memory for the large file
data_path = "data/data.csv"

# Define columns we need
cols_needed = [
    'YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'LABFORCE',
    'UHRSWORK', 'MARST', 'STATEFIP', 'METRO', 'NCHILD', 'FAMSIZE'
]

# Read data
print("  Reading full dataset...")
df = pd.read_csv(data_path, usecols=cols_needed)
print(f"  Initial observations: {len(df):,}")

# =============================================================================
# STEP 2: Define Sample Restrictions
# =============================================================================
print("\n[2] Applying sample restrictions...")

# Filter 1: Hispanic-Mexican ethnicity (HISPAN == 1 for Mexican)
df = df[df['HISPAN'] == 1]
print(f"  After Hispanic-Mexican filter: {len(df):,}")

# Filter 2: Born in Mexico (BPL == 200 for Mexico)
df = df[df['BPL'] == 200]
print(f"  After Mexico-born filter: {len(df):,}")

# Filter 3: Not a citizen (CITIZEN == 3 means not a citizen)
# This is our proxy for undocumented status as we cannot distinguish documented vs undocumented
df = df[df['CITIZEN'] == 3]
print(f"  After non-citizen filter: {len(df):,}")

# Filter 4: Arrived in US before age 16 (DACA requirement)
# Calculate age at arrival: YRIMMIG - BIRTHYR
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
# Keep those who arrived before turning 16
df = df[df['age_at_immigration'] < 16]
print(f"  After arrived-before-16 filter: {len(df):,}")

# Filter 5: Continuous residence since June 2007 - we approximate this by requiring
# immigration year <= 2007
df = df[df['YRIMMIG'] <= 2007]
print(f"  After continuous residence filter (immigrated by 2007): {len(df):,}")

# Filter 6: Define treatment and control groups based on age as of June 15, 2012
# Treatment: ages 26-30 on June 15, 2012 (born between mid-1981 and mid-1986)
# Control: ages 31-35 on June 15, 2012 (born between mid-1976 and mid-1981)

# Calculate age as of June 15, 2012
# For those born in Q1-Q2 (Jan-Jun), they have already had their birthday by June 15
# For those born in Q3-Q4 (Jul-Dec), they haven't had their birthday yet by June 15

def age_on_june_15_2012(birthyr, birthqtr):
    """Calculate age as of June 15, 2012"""
    base_age = 2012 - birthyr
    # If born in first half of year (Q1 or Q2), they've had their birthday
    # If born in second half (Q3 or Q4), they haven't
    if birthqtr in [1, 2]:
        return base_age
    else:
        return base_age - 1

df['age_june_2012'] = df.apply(lambda x: age_on_june_15_2012(x['BIRTHYR'], x['BIRTHQTR']), axis=1)

# Define treatment and control groups
# Treatment: 26-30 years old on June 15, 2012
# Control: 31-35 years old on June 15, 2012
df['treated'] = ((df['age_june_2012'] >= 26) & (df['age_june_2012'] <= 30)).astype(int)
df['control'] = ((df['age_june_2012'] >= 31) & (df['age_june_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df = df[(df['treated'] == 1) | (df['control'] == 1)]
print(f"  After age group filter (26-30 or 31-35 on June 2012): {len(df):,}")

# Filter 7: Exclude 2012 (partial treatment year)
df = df[df['YEAR'] != 2012]
print(f"  After excluding 2012: {len(df):,}")

# =============================================================================
# STEP 3: Create Key Variables
# =============================================================================
print("\n[3] Creating analysis variables...")

# Define post-treatment period (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Define full-time employment (UHRSWORK >= 35)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Interaction term for DiD
df['treated_post'] = df['treated'] * df['post']

# Age at time of survey
df['age_survey'] = df['AGE']

# Education categories
df['educ_lessHS'] = (df['EDUCD'] < 62).astype(int)  # Less than high school
df['educ_HS'] = ((df['EDUCD'] >= 62) & (df['EDUCD'] <= 64)).astype(int)  # High school
df['educ_someCollege'] = ((df['EDUCD'] > 64) & (df['EDUCD'] < 101)).astype(int)  # Some college
df['educ_BA'] = (df['EDUCD'] >= 101).astype(int)  # Bachelor's or higher

# Female indicator
df['female'] = (df['SEX'] == 2).astype(int)

# Married indicator
df['married'] = (df['MARST'] <= 2).astype(int)  # Married, spouse present or absent

# Metro area indicator
df['metro'] = (df['METRO'] >= 2).astype(int)

# Has children
df['has_children'] = (df['NCHILD'] > 0).astype(int)

print(f"  Final sample size: {len(df):,}")

# =============================================================================
# STEP 4: Summary Statistics
# =============================================================================
print("\n[4] Generating summary statistics...")

# Summary by treatment and period
summary_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'age_survey': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_HS': 'mean',
    'educ_someCollege': 'mean',
    'educ_BA': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary Statistics by Treatment Group and Period:")
print(summary_stats)

# Calculate simple DiD estimate
treated_pre = df[(df['treated']==1) & (df['post']==0)]['fulltime'].mean()
treated_post = df[(df['treated']==1) & (df['post']==1)]['fulltime'].mean()
control_pre = df[(df['treated']==0) & (df['post']==0)]['fulltime'].mean()
control_post = df[(df['treated']==0) & (df['post']==1)]['fulltime'].mean()

simple_did = (treated_post - treated_pre) - (control_post - control_pre)

print(f"\nSimple DiD Calculation:")
print(f"  Treatment group pre-period full-time rate:  {treated_pre:.4f}")
print(f"  Treatment group post-period full-time rate: {treated_post:.4f}")
print(f"  Treatment group change: {treated_post - treated_pre:.4f}")
print(f"  Control group pre-period full-time rate:    {control_pre:.4f}")
print(f"  Control group post-period full-time rate:   {control_post:.4f}")
print(f"  Control group change: {control_post - control_pre:.4f}")
print(f"  Simple DiD estimate: {simple_did:.4f}")

# =============================================================================
# STEP 5: Regression Analysis
# =============================================================================
print("\n[5] Running regression analyses...")

# Model 1: Basic DiD
print("\n  Model 1: Basic DiD (no controls, no weights)")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit()
print(f"    DiD coefficient: {model1.params['treated_post']:.4f}")
print(f"    Standard error:  {model1.bse['treated_post']:.4f}")
print(f"    P-value:         {model1.pvalues['treated_post']:.4f}")

# Model 2: DiD with demographic controls
print("\n  Model 2: DiD with demographic controls")
model2 = smf.ols('fulltime ~ treated + post + treated_post + female + married + has_children + metro + educ_HS + educ_someCollege + educ_BA', data=df).fit()
print(f"    DiD coefficient: {model2.params['treated_post']:.4f}")
print(f"    Standard error:  {model2.bse['treated_post']:.4f}")
print(f"    P-value:         {model2.pvalues['treated_post']:.4f}")

# Model 3: DiD with year fixed effects
print("\n  Model 3: DiD with year fixed effects and controls")
df['year_factor'] = pd.Categorical(df['YEAR'])
model3 = smf.ols('fulltime ~ treated + C(YEAR) + treated_post + female + married + has_children + metro + educ_HS + educ_someCollege + educ_BA', data=df).fit()
print(f"    DiD coefficient: {model3.params['treated_post']:.4f}")
print(f"    Standard error:  {model3.bse['treated_post']:.4f}")
print(f"    P-value:         {model3.pvalues['treated_post']:.4f}")

# Model 4: DiD with state fixed effects
print("\n  Model 4: DiD with year and state fixed effects and controls")
model4 = smf.ols('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + has_children + metro + educ_HS + educ_someCollege + educ_BA', data=df).fit()
print(f"    DiD coefficient: {model4.params['treated_post']:.4f}")
print(f"    Standard error:  {model4.bse['treated_post']:.4f}")
print(f"    P-value:         {model4.pvalues['treated_post']:.4f}")

# Model 5: PREFERRED SPECIFICATION - DiD with weights, year FE, state FE, and controls
print("\n  Model 5: PREFERRED - DiD with weights, year FE, state FE, and controls")
model5 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + has_children + metro + educ_HS + educ_someCollege + educ_BA',
                  data=df, weights=df['PERWT']).fit()
print(f"    DiD coefficient: {model5.params['treated_post']:.4f}")
print(f"    Standard error:  {model5.bse['treated_post']:.4f}")
print(f"    P-value:         {model5.pvalues['treated_post']:.4f}")
print(f"    95% CI:          [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")

# =============================================================================
# STEP 6: Robustness Checks
# =============================================================================
print("\n[6] Robustness checks...")

# Check 1: Robust standard errors (HC3)
print("\n  Robustness Check 1: Heteroskedasticity-robust standard errors (HC3)")
model5_robust = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + has_children + metro + educ_HS + educ_someCollege + educ_BA',
                        data=df, weights=df['PERWT']).fit(cov_type='HC3')
print(f"    DiD coefficient: {model5_robust.params['treated_post']:.4f}")
print(f"    Robust SE:       {model5_robust.bse['treated_post']:.4f}")
print(f"    P-value:         {model5_robust.pvalues['treated_post']:.4f}")
print(f"    95% CI:          [{model5_robust.conf_int().loc['treated_post', 0]:.4f}, {model5_robust.conf_int().loc['treated_post', 1]:.4f}]")

# Check 2: Separate by gender
print("\n  Robustness Check 2: Effects by gender")
model_male = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + married + has_children + metro + educ_HS + educ_someCollege + educ_BA',
                      data=df[df['female']==0], weights=df[df['female']==0]['PERWT']).fit(cov_type='HC3')
model_female = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + married + has_children + metro + educ_HS + educ_someCollege + educ_BA',
                        data=df[df['female']==1], weights=df[df['female']==1]['PERWT']).fit(cov_type='HC3')
print(f"    Male DiD:   {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f})")
print(f"    Female DiD: {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f})")

# Check 3: Pre-trends test (treatment effect in pre-period should be 0)
print("\n  Robustness Check 3: Pre-trends test")
df_pre = df[df['post'] == 0].copy()
df_pre['year_from_2011'] = df_pre['YEAR'] - 2011
df_pre['treated_trend'] = df_pre['treated'] * df_pre['year_from_2011']
model_pretrend = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_trend + female + married + has_children + metro + educ_HS + educ_someCollege + educ_BA',
                          data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC3')
print(f"    Pre-trend coefficient: {model_pretrend.params['treated_trend']:.4f}")
print(f"    Pre-trend SE:          {model_pretrend.bse['treated_trend']:.4f}")
print(f"    Pre-trend P-value:     {model_pretrend.pvalues['treated_trend']:.4f}")

# =============================================================================
# STEP 7: Create Visualization
# =============================================================================
print("\n[7] Creating visualizations...")

# Calculate weighted means by year and treatment status
yearly_means = df.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
yearly_means.columns = ['Control (31-35)', 'Treatment (26-30)']

# Plot parallel trends
plt.figure(figsize=(10, 6))
plt.plot(yearly_means.index, yearly_means['Treatment (26-30)'], 'b-o', linewidth=2, markersize=8, label='Treatment (26-30)')
plt.plot(yearly_means.index, yearly_means['Control (31-35)'], 'r--s', linewidth=2, markersize=8, label='Control (31-35)')
plt.axvline(x=2012.5, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Full-Time Employment Rates by Age Group\n(DACA-Eligible Hispanic-Mexican Non-Citizens)', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: parallel_trends.png")

# =============================================================================
# STEP 8: Save Results
# =============================================================================
print("\n[8] Saving results...")

# Save main results to file
with open('regression_results.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DACA EFFECT ON FULL-TIME EMPLOYMENT - REGRESSION RESULTS\n")
    f.write("=" * 80 + "\n\n")

    f.write("SAMPLE DESCRIPTION:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total observations: {len(df):,}\n")
    f.write(f"Treatment group (ages 26-30 on June 2012): {df['treated'].sum():,}\n")
    f.write(f"Control group (ages 31-35 on June 2012): {(1-df['treated']).sum():,}\n")
    f.write(f"Pre-period observations: {(df['post']==0).sum():,}\n")
    f.write(f"Post-period observations: {(df['post']==1).sum():,}\n\n")

    f.write("SIMPLE DiD ESTIMATE:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Treatment pre:  {treated_pre:.4f}\n")
    f.write(f"Treatment post: {treated_post:.4f}\n")
    f.write(f"Control pre:    {control_pre:.4f}\n")
    f.write(f"Control post:   {control_post:.4f}\n")
    f.write(f"DiD estimate:   {simple_did:.4f}\n\n")

    f.write("PREFERRED SPECIFICATION (Model 5):\n")
    f.write("-" * 40 + "\n")
    f.write(f"DiD Coefficient (treated_post): {model5_robust.params['treated_post']:.4f}\n")
    f.write(f"Robust Standard Error: {model5_robust.bse['treated_post']:.4f}\n")
    f.write(f"95% Confidence Interval: [{model5_robust.conf_int().loc['treated_post', 0]:.4f}, {model5_robust.conf_int().loc['treated_post', 1]:.4f}]\n")
    f.write(f"P-value: {model5_robust.pvalues['treated_post']:.4f}\n")
    f.write(f"N: {int(model5_robust.nobs):,}\n\n")

    f.write("FULL MODEL 5 RESULTS:\n")
    f.write("-" * 40 + "\n")
    f.write(model5_robust.summary().as_text())

print("  Saved: regression_results.txt")

# Save summary statistics
summary_df = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'age_survey': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_HS': 'mean',
    'educ_someCollege': 'mean',
    'educ_BA': 'mean',
    'PERWT': 'sum'
}).round(4)
summary_df.to_csv('summary_statistics.csv')
print("  Saved: summary_statistics.csv")

# =============================================================================
# STEP 9: Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"\nPreferred Estimate (DiD with weights, year FE, state FE, controls):")
print(f"  Effect size:    {model5_robust.params['treated_post']:.4f}")
print(f"  Standard error: {model5_robust.bse['treated_post']:.4f}")
print(f"  95% CI:         [{model5_robust.conf_int().loc['treated_post', 0]:.4f}, {model5_robust.conf_int().loc['treated_post', 1]:.4f}]")
print(f"  P-value:        {model5_robust.pvalues['treated_post']:.4f}")
print(f"  Sample size:    {int(model5_robust.nobs):,}")
print(f"\nInterpretation: DACA eligibility is associated with a ")
if model5_robust.params['treated_post'] > 0:
    print(f"  {model5_robust.params['treated_post']*100:.2f} percentage point INCREASE")
else:
    print(f"  {abs(model5_robust.params['treated_post'])*100:.2f} percentage point DECREASE")
print(f"  in the probability of full-time employment for the treatment group")
print(f"  relative to the control group.")
print("=" * 70)

# Store key results for report
results = {
    'did_coef': model5_robust.params['treated_post'],
    'did_se': model5_robust.bse['treated_post'],
    'did_ci_low': model5_robust.conf_int().loc['treated_post', 0],
    'did_ci_high': model5_robust.conf_int().loc['treated_post', 1],
    'did_pvalue': model5_robust.pvalues['treated_post'],
    'n_obs': int(model5_robust.nobs),
    'treated_pre': treated_pre,
    'treated_post': treated_post,
    'control_pre': control_pre,
    'control_post': control_post,
    'simple_did': simple_did,
    'n_treatment': int(df['treated'].sum()),
    'n_control': int((1-df['treated']).sum()),
    'male_did': model_male.params['treated_post'],
    'male_se': model_male.bse['treated_post'],
    'female_did': model_female.params['treated_post'],
    'female_se': model_female.bse['treated_post'],
    'pretrend_coef': model_pretrend.params['treated_trend'],
    'pretrend_se': model_pretrend.bse['treated_trend'],
    'pretrend_pvalue': model_pretrend.pvalues['treated_trend'],
    'model1_coef': model1.params['treated_post'],
    'model1_se': model1.bse['treated_post'],
    'model2_coef': model2.params['treated_post'],
    'model2_se': model2.bse['treated_post'],
    'model3_coef': model3.params['treated_post'],
    'model3_se': model3.bse['treated_post'],
    'model4_coef': model4.params['treated_post'],
    'model4_se': model4.bse['treated_post'],
}

# Save results for LaTeX report
import json
with open('results_for_report.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results_for_report.json")
