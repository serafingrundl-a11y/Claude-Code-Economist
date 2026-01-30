"""
DACA Replication Analysis - Task 83
Independent replication examining the effect of DACA eligibility on full-time employment
among Hispanic-Mexican individuals born in Mexico.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 80)
print("DACA REPLICATION ANALYSIS - TASK 83")
print("=" * 80)

data_path = "data/data.csv"
df = pd.read_csv(data_path)

print(f"\nTotal observations in raw data: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Variables: {df.columns.tolist()}")

# =============================================================================
# 2. Sample Restrictions
# =============================================================================
print("\n" + "=" * 80)
print("SAMPLE RESTRICTIONS")
print("=" * 80)

# Keep only relevant years (exclude 2012 since we can't distinguish pre/post)
df = df[df['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df):,}")

# Hispanic-Mexican (HISPAN == 1)
df = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican filter (HISPAN=1): {len(df):,}")

# Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200].copy()
print(f"After born in Mexico filter (BPL=200): {len(df):,}")

# Not a citizen (CITIZEN == 3)
# Per instructions: "anyone who is not a citizen and who has not received
# immigration papers is undocumented for DACA purposes"
df = df[df['CITIZEN'] == 3].copy()
print(f"After non-citizen filter (CITIZEN=3): {len(df):,}")

# =============================================================================
# 3. Construct Age at DACA Implementation
# =============================================================================
print("\n" + "=" * 80)
print("AGE GROUP CONSTRUCTION")
print("=" * 80)

# DACA was implemented June 15, 2012
# Treatment group: ages 26-30 as of June 15, 2012
# Control group: ages 31-35 as of June 15, 2012

# Age as of June 15, 2012 depends on birth year and quarter
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born in Q1 or Q2, had birthday by June 15
# If born in Q3 or Q4, hadn't had birthday yet by June 15

# Age at June 15, 2012 calculation:
# - If birthqtr <= 2: age = 2012 - birthyr
# - If birthqtr >= 3: age = 2012 - birthyr - 1 (birthday not yet occurred)

df['age_at_daca'] = np.where(
    df['BIRTHQTR'] <= 2,
    2012 - df['BIRTHYR'],
    2012 - df['BIRTHYR'] - 1
)

# Define treatment and control groups based on age at DACA
df['treated'] = ((df['age_at_daca'] >= 26) & (df['age_at_daca'] <= 30)).astype(int)
df['control'] = ((df['age_at_daca'] >= 31) & (df['age_at_daca'] <= 35)).astype(int)

# Keep only treatment or control group observations
df = df[(df['treated'] == 1) | (df['control'] == 1)].copy()
print(f"After age group filter (26-30 or 31-35 at DACA): {len(df):,}")

# =============================================================================
# 4. Additional DACA Eligibility Criteria
# =============================================================================
print("\n" + "=" * 80)
print("DACA ELIGIBILITY CRITERIA")
print("=" * 80)

# DACA requires: arrived in US before 16th birthday
# Calculate age at arrival
# YRIMMIG is year of immigration (0 means N/A or born in US)
# We filter to those who immigrated

df = df[df['YRIMMIG'] > 0].copy()
print(f"After valid YRIMMIG filter: {len(df):,}")

# Age at arrival = YRIMMIG - BIRTHYR (approximate)
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Keep only those who arrived before age 16
df = df[df['age_at_arrival'] < 16].copy()
print(f"After arrived before age 16 filter: {len(df):,}")

# DACA requires: lived continuously in US since June 15, 2007
# We approximate this by requiring YRIMMIG <= 2007
df = df[df['YRIMMIG'] <= 2007].copy()
print(f"After arrived by 2007 filter: {len(df):,}")

# =============================================================================
# 5. Outcome Variable: Full-Time Employment
# =============================================================================
print("\n" + "=" * 80)
print("OUTCOME VARIABLE")
print("=" * 80)

# Full-time employment = usually working 35+ hours per week
# UHRSWORK: usual hours worked per week (0 = N/A or not working)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

print(f"Full-time employment rate in sample: {df['fulltime'].mean():.3f}")

# =============================================================================
# 6. Define Pre/Post Period
# =============================================================================
# Pre-period: 2006-2011
# Post-period: 2013-2016

df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"\nPre-period years: {sorted(df[df['post']==0]['YEAR'].unique())}")
print(f"Post-period years: {sorted(df[df['post']==1]['YEAR'].unique())}")

# =============================================================================
# 7. Summary Statistics
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\nFinal analytic sample size: {len(df):,}")
print(f"Treatment group (age 26-30): {df['treated'].sum():,}")
print(f"Control group (age 31-35): {(1-df['treated']).sum():,}")
print(f"Pre-period observations: {(df['post']==0).sum():,}")
print(f"Post-period observations: {(df['post']==1).sum():,}")

# Summary by group and period
print("\n--- Full-Time Employment Rates by Group and Period ---")
summary = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
summary.columns = ['FT_Mean', 'FT_Std', 'N', 'Weighted_N']
print(summary)

# Weighted summary
print("\n--- Weighted Full-Time Employment Rates ---")
for t in [0, 1]:
    for p in [0, 1]:
        subset = df[(df['treated'] == t) & (df['post'] == p)]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        group_label = "Treatment" if t == 1 else "Control"
        period_label = "Post" if p == 1 else "Pre"
        print(f"{group_label}, {period_label}: {weighted_mean:.4f} (N={len(subset):,})")

# =============================================================================
# 8. Difference-in-Differences Estimation
# =============================================================================
print("\n" + "=" * 80)
print("DIFFERENCE-IN-DIFFERENCES ESTIMATION")
print("=" * 80)

# Create interaction term
df['treat_post'] = df['treated'] * df['post']

# --- Model 1: Basic DID (unweighted) ---
print("\n--- Model 1: Basic DID (Unweighted OLS) ---")
model1 = smf.ols('fulltime ~ treated + post + treat_post', data=df).fit(cov_type='HC1')
print(model1.summary())

# --- Model 2: Weighted DID ---
print("\n--- Model 2: Weighted DID ---")
model2 = smf.wls('fulltime ~ treated + post + treat_post', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# --- Model 3: DID with Covariates (Weighted) ---
print("\n--- Model 3: Weighted DID with Covariates ---")
# Add covariates: sex, education, marital status
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)  # 1 or 2 = married
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # High school or more
df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # Some college or more

model3 = smf.wls('fulltime ~ treated + post + treat_post + female + married + educ_hs + educ_college',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# --- Model 4: DID with Year Fixed Effects (Weighted) ---
print("\n--- Model 4: Weighted DID with Year Fixed Effects ---")
df['year_factor'] = pd.Categorical(df['YEAR'])
model4 = smf.wls('fulltime ~ treated + treat_post + C(YEAR)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# --- Model 5: Full Model with Year FE and Covariates ---
print("\n--- Model 5: Full Weighted DID with Year FE and Covariates ---")
model5 = smf.wls('fulltime ~ treated + treat_post + C(YEAR) + female + married + educ_hs + educ_college',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary())

# =============================================================================
# 9. Results Summary
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print("\nDifference-in-Differences Estimates (treat_post coefficient):")
print("-" * 60)
print(f"Model 1 (Basic, Unweighted):     {model1.params['treat_post']:.4f} (SE: {model1.bse['treat_post']:.4f})")
print(f"Model 2 (Weighted):              {model2.params['treat_post']:.4f} (SE: {model2.bse['treat_post']:.4f})")
print(f"Model 3 (Weighted + Covariates): {model3.params['treat_post']:.4f} (SE: {model3.bse['treat_post']:.4f})")
print(f"Model 4 (Weighted + Year FE):    {model4.params['treat_post']:.4f} (SE: {model4.bse['treat_post']:.4f})")
print(f"Model 5 (Full Model):            {model5.params['treat_post']:.4f} (SE: {model5.bse['treat_post']:.4f})")

# Preferred estimate: Model 5 (full model with year FE and covariates)
preferred = model5
print(f"\n*** PREFERRED ESTIMATE (Model 5) ***")
print(f"Effect Size: {preferred.params['treat_post']:.4f}")
print(f"Standard Error: {preferred.bse['treat_post']:.4f}")
ci = preferred.conf_int().loc['treat_post']
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"P-value: {preferred.pvalues['treat_post']:.4f}")
print(f"Sample Size: {len(df):,}")

# =============================================================================
# 10. Simple DID Calculation (Manual Verification)
# =============================================================================
print("\n" + "=" * 80)
print("MANUAL DID CALCULATION (VERIFICATION)")
print("=" * 80)

# Weighted means
def weighted_mean(x, w):
    return np.average(x, weights=w)

treated_pre = df[(df['treated']==1) & (df['post']==0)]
treated_post = df[(df['treated']==1) & (df['post']==1)]
control_pre = df[(df['treated']==0) & (df['post']==0)]
control_post = df[(df['treated']==0) & (df['post']==1)]

# Weighted means
ft_treated_pre = weighted_mean(treated_pre['fulltime'], treated_pre['PERWT'])
ft_treated_post = weighted_mean(treated_post['fulltime'], treated_post['PERWT'])
ft_control_pre = weighted_mean(control_pre['fulltime'], control_pre['PERWT'])
ft_control_post = weighted_mean(control_post['fulltime'], control_post['PERWT'])

print(f"Treated, Pre:  {ft_treated_pre:.4f}")
print(f"Treated, Post: {ft_treated_post:.4f}")
print(f"Control, Pre:  {ft_control_pre:.4f}")
print(f"Control, Post: {ft_control_post:.4f}")

did_estimate = (ft_treated_post - ft_treated_pre) - (ft_control_post - ft_control_pre)
print(f"\nDID = ({ft_treated_post:.4f} - {ft_treated_pre:.4f}) - ({ft_control_post:.4f} - {ft_control_pre:.4f})")
print(f"DID = {ft_treated_post - ft_treated_pre:.4f} - {ft_control_post - ft_control_pre:.4f}")
print(f"DID = {did_estimate:.4f}")

# =============================================================================
# 11. Year-by-Year Analysis for Parallel Trends
# =============================================================================
print("\n" + "=" * 80)
print("YEAR-BY-YEAR FULL-TIME EMPLOYMENT RATES")
print("=" * 80)

yearly_rates = []
for year in sorted(df['YEAR'].unique()):
    for group in [0, 1]:
        subset = df[(df['YEAR'] == year) & (df['treated'] == group)]
        if len(subset) > 0:
            rate = weighted_mean(subset['fulltime'], subset['PERWT'])
            yearly_rates.append({
                'Year': year,
                'Group': 'Treatment (26-30)' if group == 1 else 'Control (31-35)',
                'FT_Rate': rate,
                'N': len(subset)
            })

yearly_df = pd.DataFrame(yearly_rates)
print(yearly_df.pivot(index='Year', columns='Group', values='FT_Rate').round(4))

# =============================================================================
# 12. Subgroup Analysis by Sex
# =============================================================================
print("\n" + "=" * 80)
print("SUBGROUP ANALYSIS BY SEX")
print("=" * 80)

# Male subgroup
df_male = df[df['female'] == 0].copy()
model_male = smf.wls('fulltime ~ treated + treat_post + C(YEAR) + married + educ_hs + educ_college',
                      data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
print(f"\nMale Subgroup (N={len(df_male):,}):")
print(f"  DID Estimate: {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f})")

# Female subgroup
df_female = df[df['female'] == 1].copy()
model_female = smf.wls('fulltime ~ treated + treat_post + C(YEAR) + married + educ_hs + educ_college',
                        data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"\nFemale Subgroup (N={len(df_female):,}):")
print(f"  DID Estimate: {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f})")

# =============================================================================
# 13. Save Results for Report
# =============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Export key results to CSV for LaTeX
results_dict = {
    'Model': ['Basic (Unweighted)', 'Weighted', 'Weighted + Covariates',
              'Weighted + Year FE', 'Full Model (Preferred)'],
    'DID_Estimate': [model1.params['treat_post'], model2.params['treat_post'],
                     model3.params['treat_post'], model4.params['treat_post'],
                     model5.params['treat_post']],
    'Std_Error': [model1.bse['treat_post'], model2.bse['treat_post'],
                  model3.bse['treat_post'], model4.bse['treat_post'],
                  model5.bse['treat_post']],
    'P_Value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
                model3.pvalues['treat_post'], model4.pvalues['treat_post'],
                model5.pvalues['treat_post']]
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv('did_results.csv', index=False)
print("Results saved to did_results.csv")

# Export yearly trends
yearly_pivot = yearly_df.pivot(index='Year', columns='Group', values='FT_Rate')
yearly_pivot.to_csv('yearly_trends.csv')
print("Yearly trends saved to yearly_trends.csv")

# Export summary statistics
summary_stats = {
    'Statistic': ['Total Sample Size', 'Treatment Group (26-30)', 'Control Group (31-35)',
                  'Pre-Period Obs', 'Post-Period Obs',
                  'FT Rate - Treated Pre', 'FT Rate - Treated Post',
                  'FT Rate - Control Pre', 'FT Rate - Control Post',
                  'DID Estimate', 'Standard Error', '95% CI Lower', '95% CI Upper'],
    'Value': [len(df), df['treated'].sum(), (1-df['treated']).sum(),
              (df['post']==0).sum(), (df['post']==1).sum(),
              ft_treated_pre, ft_treated_post, ft_control_pre, ft_control_post,
              preferred.params['treat_post'], preferred.bse['treat_post'],
              ci[0], ci[1]]
}
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_stats.csv', index=False)
print("Summary statistics saved to summary_stats.csv")

# Full model results for table
model5_table = pd.DataFrame({
    'Variable': model5.params.index,
    'Coefficient': model5.params.values,
    'Std_Error': model5.bse.values,
    'P_Value': model5.pvalues.values
})
model5_table.to_csv('model5_full_results.csv', index=False)
print("Full model results saved to model5_full_results.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
