"""
DACA Replication Analysis
=========================

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the
United States, what was the causal impact of eligibility for DACA on the probability
that the eligible person is employed full-time (usually working 35 hours per week or more)?

DACA Eligibility Criteria (as of June 15, 2012):
1. Arrived unlawfully in US before 16th birthday
2. Had not yet turned 31 as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (arrived by 2007)
4. Were present in US on June 15, 2012 and did not have lawful status (non-citizen)

Method: Difference-in-Differences
- Treatment group: DACA-eligible non-citizens
- Control group: Non-DACA-eligible non-citizens (similar demographics)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (excluding 2012 due to implementation timing ambiguity)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[STEP 1] Loading data...")

# Load the main data file
df = pd.read_csv('data/data.csv')
print(f"Total observations in raw data: {len(df):,}")
print(f"Years available: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: DEFINE SAMPLE - Hispanic-Mexican, Mexican-born
# =============================================================================
print("\n[STEP 2] Defining sample: Hispanic-Mexican, Mexican-born individuals...")

# HISPAN == 1 indicates Mexican Hispanic origin
# BPL == 200 indicates Mexico birthplace
# CITIZEN == 3 indicates non-citizen (likely undocumented for DACA purposes)

df_sample = df.copy()

# Filter to Hispanic-Mexican ethnicity (HISPAN == 1: Mexican)
df_sample = df_sample[df_sample['HISPAN'] == 1]
print(f"After filtering Hispanic-Mexican (HISPAN=1): {len(df_sample):,}")

# Filter to Mexican-born (BPL == 200: Mexico)
df_sample = df_sample[df_sample['BPL'] == 200]
print(f"After filtering Mexican-born (BPL=200): {len(df_sample):,}")

# Keep only non-citizens (CITIZEN == 3) - likely undocumented
# Per instructions: "Assume that anyone who is not a citizen and who has not
# received immigration papers is undocumented for DACA purposes"
df_sample = df_sample[df_sample['CITIZEN'] == 3]
print(f"After filtering non-citizens (CITIZEN=3): {len(df_sample):,}")

# Exclude 2012 due to implementation timing ambiguity
df_sample = df_sample[df_sample['YEAR'] != 2012]
print(f"After excluding 2012: {len(df_sample):,}")

# =============================================================================
# STEP 3: DEFINE DACA ELIGIBILITY
# =============================================================================
print("\n[STEP 3] Defining DACA eligibility criteria...")

# DACA eligibility requires:
# 1. Arrived before 16th birthday: (YEAR - AGE) >= YRIMMIG means they were born
#    before or during their immigration year. For someone who arrived before age 16,
#    we need: AGE at immigration < 16
#    Age at immigration = YRIMMIG - BIRTHYR
#    So: YRIMMIG - BIRTHYR < 16

# 2. Not yet 31 as of June 15, 2012: BIRTHYR >= 1981
#    (more precisely, born after June 15, 1981, but we approximate with BIRTHYR >= 1982
#     to be conservative, or >= 1981 to be inclusive)

# 3. Lived continuously since June 15, 2007: YRIMMIG <= 2007

# 4. Present in US on June 15, 2012 and non-citizen: already filtered by CITIZEN == 3

def calculate_daca_eligibility(row):
    """
    Determine if an individual meets DACA eligibility criteria
    """
    # Age at immigration
    if pd.isna(row['YRIMMIG']) or row['YRIMMIG'] == 0:
        return 0  # Missing immigration year

    age_at_immigration = row['YRIMMIG'] - row['BIRTHYR']

    # Criterion 1: Arrived before 16th birthday
    arrived_before_16 = age_at_immigration < 16

    # Criterion 2: Born after June 15, 1981 (not yet 31 as of June 15, 2012)
    # We use BIRTHYR >= 1982 to be conservative (definitely born after June 15, 1981)
    # Or BIRTHYR >= 1981 and use BIRTHQTR to be more precise
    born_after_1981 = row['BIRTHYR'] >= 1981

    # Criterion 3: Lived continuously since June 15, 2007 (arrived by 2007)
    arrived_by_2007 = row['YRIMMIG'] <= 2007

    # All criteria must be met
    return int(arrived_before_16 and born_after_1981 and arrived_by_2007)

# Apply eligibility function
df_sample['DACA_ELIGIBLE'] = df_sample.apply(calculate_daca_eligibility, axis=1)

print(f"\nDACA eligibility distribution:")
print(df_sample['DACA_ELIGIBLE'].value_counts())

# =============================================================================
# STEP 4: DEFINE OUTCOME - FULL-TIME EMPLOYMENT
# =============================================================================
print("\n[STEP 4] Defining outcome: Full-time employment (UHRSWORK >= 35)...")

# Full-time employment defined as usually working 35+ hours per week
# UHRSWORK == 0 typically indicates not employed or not applicable
df_sample['FULLTIME_EMPLOYED'] = (df_sample['UHRSWORK'] >= 35).astype(int)

print(f"\nFull-time employment distribution:")
print(df_sample['FULLTIME_EMPLOYED'].value_counts())
print(f"\nFull-time employment rate: {df_sample['FULLTIME_EMPLOYED'].mean():.4f}")

# =============================================================================
# STEP 5: DEFINE PRE/POST PERIOD
# =============================================================================
print("\n[STEP 5] Defining pre/post DACA period...")

# Pre-DACA: 2006-2011
# Post-DACA: 2013-2016
df_sample['POST_DACA'] = (df_sample['YEAR'] >= 2013).astype(int)

print(f"\nPre/Post DACA distribution:")
print(df_sample['POST_DACA'].value_counts())

# Create interaction term for DiD
df_sample['DACA_X_POST'] = df_sample['DACA_ELIGIBLE'] * df_sample['POST_DACA']

# =============================================================================
# STEP 6: RESTRICT TO WORKING-AGE POPULATION
# =============================================================================
print("\n[STEP 6] Restricting to working-age population (18-64)...")

df_analysis = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 64)]
print(f"After age restriction (18-64): {len(df_analysis):,}")

# Further restrict to those in the labor force for employment analysis
# Or we can include everyone and interpret the outcome as probability of full-time work

# =============================================================================
# STEP 7: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[STEP 7] Descriptive Statistics...")
print("="*80)

# Sample sizes by treatment and period
print("\n--- Sample Sizes by Treatment Group and Period ---")
crosstab = pd.crosstab(df_analysis['DACA_ELIGIBLE'], df_analysis['POST_DACA'],
                       margins=True, margins_name='Total')
crosstab.index = ['Not Eligible', 'DACA Eligible', 'Total']
crosstab.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)', 'Total']
print(crosstab)

# Full-time employment rates by group and period
print("\n--- Full-Time Employment Rates by Group and Period ---")
ft_rates = df_analysis.groupby(['DACA_ELIGIBLE', 'POST_DACA'])['FULLTIME_EMPLOYED'].agg(['mean', 'count', 'std'])
ft_rates.columns = ['Mean', 'N', 'Std']
print(ft_rates)

# Calculate simple DiD estimate
pre_treat = df_analysis[(df_analysis['DACA_ELIGIBLE']==1) & (df_analysis['POST_DACA']==0)]['FULLTIME_EMPLOYED'].mean()
post_treat = df_analysis[(df_analysis['DACA_ELIGIBLE']==1) & (df_analysis['POST_DACA']==1)]['FULLTIME_EMPLOYED'].mean()
pre_control = df_analysis[(df_analysis['DACA_ELIGIBLE']==0) & (df_analysis['POST_DACA']==0)]['FULLTIME_EMPLOYED'].mean()
post_control = df_analysis[(df_analysis['DACA_ELIGIBLE']==0) & (df_analysis['POST_DACA']==1)]['FULLTIME_EMPLOYED'].mean()

print(f"\n--- Simple Difference-in-Differences Calculation ---")
print(f"Treatment group (DACA eligible):")
print(f"  Pre-DACA:  {pre_treat:.4f}")
print(f"  Post-DACA: {post_treat:.4f}")
print(f"  Change:    {post_treat - pre_treat:.4f}")

print(f"\nControl group (Not eligible):")
print(f"  Pre-DACA:  {pre_control:.4f}")
print(f"  Post-DACA: {post_control:.4f}")
print(f"  Change:    {post_control - pre_control:.4f}")

simple_did = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\nSimple DiD Estimate: {simple_did:.4f}")

# =============================================================================
# STEP 8: MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n[STEP 8] Main Regression Analysis...")
print("="*80)

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + POST_DACA + DACA_X_POST',
                 data=df_analysis).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")

# Create control variables
df_analysis['FEMALE'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['MARRIED'] = (df_analysis['MARST'].isin([1, 2])).astype(int)
df_analysis['AGE_SQ'] = df_analysis['AGE'] ** 2

# Education categories
df_analysis['EDUC_HS'] = (df_analysis['EDUC'] >= 6).astype(int)  # High school or more
df_analysis['EDUC_COLLEGE'] = (df_analysis['EDUC'] >= 10).astype(int)  # Some college or more

model2 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + POST_DACA + DACA_X_POST + '
                 'AGE + AGE_SQ + FEMALE + MARRIED + EDUC_HS + EDUC_COLLEGE',
                 data=df_analysis).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
df_analysis['YEAR_CAT'] = pd.Categorical(df_analysis['YEAR'])
model3 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_X_POST + C(YEAR) + '
                 'AGE + AGE_SQ + FEMALE + MARRIED + EDUC_HS + EDUC_COLLEGE',
                 data=df_analysis).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with Year and State Fixed Effects ---")
model4 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_X_POST + C(YEAR) + C(STATEFIP) + '
                 'AGE + AGE_SQ + FEMALE + MARRIED + EDUC_HS + EDUC_COLLEGE',
                 data=df_analysis).fit(cov_type='HC1')

# Extract key coefficients
print("\nKey Results from Model 4:")
print(f"DiD Coefficient (DACA_X_POST): {model4.params['DACA_X_POST']:.6f}")
print(f"Standard Error: {model4.bse['DACA_X_POST']:.6f}")
print(f"95% CI: [{model4.conf_int().loc['DACA_X_POST', 0]:.6f}, {model4.conf_int().loc['DACA_X_POST', 1]:.6f}]")
print(f"P-value: {model4.pvalues['DACA_X_POST']:.6f}")
print(f"Number of observations: {int(model4.nobs):,}")

# =============================================================================
# STEP 9: ROBUSTNESS CHECKS
# =============================================================================
print("\n[STEP 9] Robustness Checks...")
print("="*80)

# Robustness 1: Linear Probability Model with cluster-robust standard errors by state
print("\n--- Robustness Check 1: Clustered Standard Errors by State ---")
model_cluster = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_X_POST + C(YEAR) + C(STATEFIP) + '
                        'AGE + AGE_SQ + FEMALE + MARRIED + EDUC_HS + EDUC_COLLEGE',
                        data=df_analysis).fit(cov_type='cluster',
                                              cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"DiD Coefficient: {model_cluster.params['DACA_X_POST']:.6f}")
print(f"Clustered SE: {model_cluster.bse['DACA_X_POST']:.6f}")
print(f"95% CI: [{model_cluster.conf_int().loc['DACA_X_POST', 0]:.6f}, {model_cluster.conf_int().loc['DACA_X_POST', 1]:.6f}]")
print(f"P-value: {model_cluster.pvalues['DACA_X_POST']:.6f}")

# Robustness 2: Restrict to ages 18-35 (more similar to DACA-eligible population)
print("\n--- Robustness Check 2: Ages 18-35 Only ---")
df_young = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 35)]
model_young = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_X_POST + C(YEAR) + '
                      'AGE + AGE_SQ + FEMALE + MARRIED + EDUC_HS + EDUC_COLLEGE',
                      data=df_young).fit(cov_type='HC1')
print(f"Sample size: {int(model_young.nobs):,}")
print(f"DiD Coefficient: {model_young.params['DACA_X_POST']:.6f}")
print(f"Standard Error: {model_young.bse['DACA_X_POST']:.6f}")
print(f"95% CI: [{model_young.conf_int().loc['DACA_X_POST', 0]:.6f}, {model_young.conf_int().loc['DACA_X_POST', 1]:.6f}]")
print(f"P-value: {model_young.pvalues['DACA_X_POST']:.6f}")

# Robustness 3: Exclude early arrivals (arrived before 1990)
print("\n--- Robustness Check 3: Arrived 1990 or later ---")
df_recent = df_analysis[df_analysis['YRIMMIG'] >= 1990]
model_recent = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_X_POST + C(YEAR) + '
                       'AGE + AGE_SQ + FEMALE + MARRIED + EDUC_HS + EDUC_COLLEGE',
                       data=df_recent).fit(cov_type='HC1')
print(f"Sample size: {int(model_recent.nobs):,}")
print(f"DiD Coefficient: {model_recent.params['DACA_X_POST']:.6f}")
print(f"Standard Error: {model_recent.bse['DACA_X_POST']:.6f}")
print(f"95% CI: [{model_recent.conf_int().loc['DACA_X_POST', 0]:.6f}, {model_recent.conf_int().loc['DACA_X_POST', 1]:.6f}]")
print(f"P-value: {model_recent.pvalues['DACA_X_POST']:.6f}")

# Robustness 4: Alternative outcome - Employment (any work)
print("\n--- Robustness Check 4: Any Employment (EMPSTAT=1) ---")
df_analysis['EMPLOYED'] = (df_analysis['EMPSTAT'] == 1).astype(int)
model_any_emp = smf.ols('EMPLOYED ~ DACA_ELIGIBLE + DACA_X_POST + C(YEAR) + '
                        'AGE + AGE_SQ + FEMALE + MARRIED + EDUC_HS + EDUC_COLLEGE',
                        data=df_analysis).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_any_emp.params['DACA_X_POST']:.6f}")
print(f"Standard Error: {model_any_emp.bse['DACA_X_POST']:.6f}")
print(f"95% CI: [{model_any_emp.conf_int().loc['DACA_X_POST', 0]:.6f}, {model_any_emp.conf_int().loc['DACA_X_POST', 1]:.6f}]")
print(f"P-value: {model_any_emp.pvalues['DACA_X_POST']:.6f}")

# =============================================================================
# STEP 10: EVENT STUDY / DYNAMIC EFFECTS
# =============================================================================
print("\n[STEP 10] Event Study Analysis...")
print("="*80)

# Create year-specific treatment effects (relative to 2011)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in years:
    df_analysis[f'DACA_X_YEAR_{year}'] = (df_analysis['DACA_ELIGIBLE'] *
                                           (df_analysis['YEAR'] == year)).astype(int)

# Event study regression
formula_es = 'FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + ' + \
             ' + '.join([f'DACA_X_YEAR_{y}' for y in years]) + \
             ' + C(YEAR) + AGE + AGE_SQ + FEMALE + MARRIED + EDUC_HS + EDUC_COLLEGE'
model_es = smf.ols(formula_es, data=df_analysis).fit(cov_type='HC1')

print("\nEvent Study Results (Reference year: 2011):")
print("-" * 60)
print(f"{'Year':<10}{'Coefficient':<15}{'Std Error':<15}{'P-value':<15}")
print("-" * 60)
for year in years:
    coef_name = f'DACA_X_YEAR_{year}'
    print(f"{year:<10}{model_es.params[coef_name]:<15.6f}{model_es.bse[coef_name]:<15.6f}{model_es.pvalues[coef_name]:<15.6f}")

# =============================================================================
# STEP 11: HETEROGENEITY ANALYSIS
# =============================================================================
print("\n[STEP 11] Heterogeneity Analysis...")
print("="*80)

# By gender
print("\n--- Effect by Gender ---")
for sex, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_analysis[df_analysis['SEX'] == sex]
    model_sex = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_X_POST + C(YEAR) + '
                        'AGE + AGE_SQ + MARRIED + EDUC_HS + EDUC_COLLEGE',
                        data=df_sex).fit(cov_type='HC1')
    print(f"{sex_name}: DiD = {model_sex.params['DACA_X_POST']:.6f} (SE = {model_sex.bse['DACA_X_POST']:.6f}), N = {int(model_sex.nobs):,}")

# By education
print("\n--- Effect by Education Level ---")
df_low_ed = df_analysis[df_analysis['EDUC'] < 6]  # Less than high school
df_high_ed = df_analysis[df_analysis['EDUC'] >= 6]  # High school or more

model_low = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_X_POST + C(YEAR) + '
                    'AGE + AGE_SQ + FEMALE + MARRIED',
                    data=df_low_ed).fit(cov_type='HC1')
model_high = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_X_POST + C(YEAR) + '
                     'AGE + AGE_SQ + FEMALE + MARRIED',
                     data=df_high_ed).fit(cov_type='HC1')
print(f"Less than HS: DiD = {model_low.params['DACA_X_POST']:.6f} (SE = {model_low.bse['DACA_X_POST']:.6f}), N = {int(model_low.nobs):,}")
print(f"HS or more:   DiD = {model_high.params['DACA_X_POST']:.6f} (SE = {model_high.bse['DACA_X_POST']:.6f}), N = {int(model_high.nobs):,}")

# By age group
print("\n--- Effect by Age Group ---")
for age_lo, age_hi, name in [(18, 24, '18-24'), (25, 34, '25-34'), (35, 64, '35-64')]:
    df_age = df_analysis[(df_analysis['AGE'] >= age_lo) & (df_analysis['AGE'] <= age_hi)]
    model_age = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_X_POST + C(YEAR) + '
                        'AGE + FEMALE + MARRIED + EDUC_HS + EDUC_COLLEGE',
                        data=df_age).fit(cov_type='HC1')
    print(f"Ages {name}: DiD = {model_age.params['DACA_X_POST']:.6f} (SE = {model_age.bse['DACA_X_POST']:.6f}), N = {int(model_age.nobs):,}")

# =============================================================================
# STEP 12: SUMMARY OF RESULTS
# =============================================================================
print("\n" + "="*80)
print("SUMMARY OF MAIN RESULTS")
print("="*80)

print(f"""
PREFERRED SPECIFICATION: Model 4 (Year and State Fixed Effects with Controls)

Effect Size (DiD Estimate): {model4.params['DACA_X_POST']:.6f}
Standard Error (Robust):    {model4.bse['DACA_X_POST']:.6f}
95% Confidence Interval:    [{model4.conf_int().loc['DACA_X_POST', 0]:.6f}, {model4.conf_int().loc['DACA_X_POST', 1]:.6f}]
P-value:                    {model4.pvalues['DACA_X_POST']:.6f}
Sample Size:                {int(model4.nobs):,}
R-squared:                  {model4.rsquared:.4f}

INTERPRETATION:
The difference-in-differences estimate suggests that DACA eligibility
{"increased" if model4.params['DACA_X_POST'] > 0 else "decreased"} the probability of full-time employment
by {abs(model4.params['DACA_X_POST'])*100:.2f} percentage points among DACA-eligible
Mexican-born non-citizens relative to non-eligible Mexican-born non-citizens.

This effect is {"statistically significant" if model4.pvalues['DACA_X_POST'] < 0.05 else "not statistically significant"} at the 5% level
(p = {model4.pvalues['DACA_X_POST']:.4f}).
""")

# =============================================================================
# STEP 13: SAVE RESULTS FOR TABLES
# =============================================================================
print("\n[STEP 13] Saving results...")

# Create results summary dataframe
results_summary = pd.DataFrame({
    'Model': ['(1) Basic DiD', '(2) + Demographics', '(3) + Year FE',
              '(4) + State FE', '(5) Clustered SE', '(6) Ages 18-35'],
    'DiD_Coefficient': [model1.params['DACA_X_POST'], model2.params['DACA_X_POST'],
                        model3.params['DACA_X_POST'], model4.params['DACA_X_POST'],
                        model_cluster.params['DACA_X_POST'], model_young.params['DACA_X_POST']],
    'Std_Error': [model1.bse['DACA_X_POST'], model2.bse['DACA_X_POST'],
                  model3.bse['DACA_X_POST'], model4.bse['DACA_X_POST'],
                  model_cluster.bse['DACA_X_POST'], model_young.bse['DACA_X_POST']],
    'P_Value': [model1.pvalues['DACA_X_POST'], model2.pvalues['DACA_X_POST'],
                model3.pvalues['DACA_X_POST'], model4.pvalues['DACA_X_POST'],
                model_cluster.pvalues['DACA_X_POST'], model_young.pvalues['DACA_X_POST']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model_cluster.nobs), int(model_young.nobs)],
    'R_Squared': [model1.rsquared, model2.rsquared, model3.rsquared,
                  model4.rsquared, model_cluster.rsquared, model_young.rsquared]
})
results_summary.to_csv('results_summary.csv', index=False)

# Save event study results
event_study_results = pd.DataFrame({
    'Year': years,
    'Coefficient': [model_es.params[f'DACA_X_YEAR_{y}'] for y in years],
    'Std_Error': [model_es.bse[f'DACA_X_YEAR_{y}'] for y in years],
    'CI_Lower': [model_es.conf_int().loc[f'DACA_X_YEAR_{y}', 0] for y in years],
    'CI_Upper': [model_es.conf_int().loc[f'DACA_X_YEAR_{y}', 1] for y in years]
})
event_study_results.to_csv('event_study_results.csv', index=False)

# Save descriptive statistics
desc_stats = df_analysis.groupby(['DACA_ELIGIBLE', 'POST_DACA']).agg({
    'FULLTIME_EMPLOYED': ['mean', 'count', 'std'],
    'AGE': 'mean',
    'FEMALE': 'mean',
    'MARRIED': 'mean',
    'EDUC_HS': 'mean'
}).reset_index()
desc_stats.to_csv('descriptive_stats.csv', index=False)

print("\nResults saved to:")
print("  - results_summary.csv")
print("  - event_study_results.csv")
print("  - descriptive_stats.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
