"""
DACA Replication Analysis
========================
This script analyzes the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexico-born individuals.

Research Design:
- Treatment group: Ages 26-30 as of June 15, 2012 (birth years 1982-1986)
- Control group: Ages 31-35 as of June 15, 2012 (birth years 1977-1981)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded as implementation year)
- Outcome: Full-time employment (UHRSWORK >= 35)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")
df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")

# =============================================================================
# STEP 1: IDENTIFY DACA-ELIGIBLE SAMPLE
# =============================================================================
print("\n" + "="*60)
print("STEP 1: Identifying DACA-eligible sample")
print("="*60)

# Initial data summary
print(f"\nYear distribution in data:")
print(df['YEAR'].value_counts().sort_index())

# Filter criteria for DACA-like eligibility:
# 1. Hispanic-Mexican ethnicity (HISPAN == 1)
# 2. Born in Mexico (BPL == 200)
# 3. Not a citizen (CITIZEN == 3)
# 4. Immigrated before age 16 (need to calculate)
# 5. Lived continuously in US since June 15, 2007 (approximate with YRIMMIG <= 2007)

print("\n--- Filtering for Hispanic-Mexican, Mexico-born, non-citizens ---")

# Hispanic-Mexican (HISPAN == 1)
print(f"HISPAN value counts:")
print(df['HISPAN'].value_counts())

# Birthplace Mexico (BPL == 200)
print(f"\nBPL == 200 (Mexico): {(df['BPL'] == 200).sum():,}")

# Citizenship status
print(f"\nCITIZEN value counts:")
print(df['CITIZEN'].value_counts())

# Apply basic filters
df_eligible = df[
    (df['HISPAN'] == 1) &  # Hispanic-Mexican
    (df['BPL'] == 200) &   # Born in Mexico
    (df['CITIZEN'] == 3)   # Not a citizen
].copy()

print(f"\nAfter Hispanic-Mexican, Mexico-born, non-citizen filter: {len(df_eligible):,}")

# =============================================================================
# STEP 2: APPLY AGE AND IMMIGRATION CRITERIA FOR DACA ELIGIBILITY
# =============================================================================
print("\n" + "="*60)
print("STEP 2: Applying age and immigration criteria")
print("="*60)

# Calculate age at immigration
# Age at time of survey = AGE
# Year of survey = YEAR
# Year of immigration = YRIMMIG
# Birth year = BIRTHYR
# Age at immigration = YRIMMIG - BIRTHYR

df_eligible['age_at_immig'] = df_eligible['YRIMMIG'] - df_eligible['BIRTHYR']

# DACA requirement: Arrived before 16th birthday
# Also: must have immigrated by June 15, 2007 (we use YRIMMIG <= 2007)
# Also: must have been in US on June 15, 2012

print(f"\nAge at immigration distribution (before filtering):")
print(df_eligible['age_at_immig'].describe())

# Filter for arrived before age 16
df_eligible = df_eligible[df_eligible['age_at_immig'] < 16].copy()
print(f"After arrived before age 16: {len(df_eligible):,}")

# Filter for immigrated by 2007 (continuous residence requirement)
df_eligible = df_eligible[df_eligible['YRIMMIG'] <= 2007].copy()
print(f"After YRIMMIG <= 2007 (continuous residence): {len(df_eligible):,}")

# =============================================================================
# STEP 3: DEFINE TREATMENT AND CONTROL GROUPS
# =============================================================================
print("\n" + "="*60)
print("STEP 3: Defining treatment and control groups")
print("="*60)

# Treatment group: Ages 26-30 as of June 15, 2012
# This means birth years 1982-1986 (June 1982 to June 1986 to be precise)
# Control group: Ages 31-35 as of June 15, 2012
# This means birth years 1977-1981

# Age as of June 15, 2012 can be approximated using BIRTHYR
# If born in 1982, turned 30 in 2012
# If born in 1986, turned 26 in 2012
# If born in 1977, turned 35 in 2012
# If born in 1981, turned 31 in 2012

# Using BIRTHQTR to be more precise about age cutoffs
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# June 15, 2012 as reference date

# For treatment (26-30 as of June 15, 2012):
# - Born Jul 1981 - Jun 1986 would be ages 26-30.9
# - Simplify: BIRTHYR 1982-1986 (conservative)

# For control (31-35 as of June 15, 2012):
# - Born Jul 1976 - Jun 1981 would be ages 31-35.9
# - Simplify: BIRTHYR 1977-1981 (conservative)

# Define treatment indicator based on birth year
# We need to identify who would be in treated vs control group based on their
# age as of June 2012, but we observe them at different years

# Calculate age as of June 15, 2012 for all observations
# If BIRTHQTR <= 2 (Jan-Jun), they had their birthday by June 15
# If BIRTHQTR >= 3 (Jul-Dec), they hadn't had their birthday by June 15

df_eligible['age_june_2012'] = 2012 - df_eligible['BIRTHYR']
# Adjust for those born in second half of year (hadn't turned that age yet)
df_eligible.loc[df_eligible['BIRTHQTR'] >= 3, 'age_june_2012'] -= 1

print(f"\nAge as of June 15, 2012 distribution:")
print(df_eligible['age_june_2012'].value_counts().sort_index())

# Define treatment (26-30) and control (31-35) groups
df_sample = df_eligible[
    ((df_eligible['age_june_2012'] >= 26) & (df_eligible['age_june_2012'] <= 30)) |
    ((df_eligible['age_june_2012'] >= 31) & (df_eligible['age_june_2012'] <= 35))
].copy()

df_sample['treated'] = ((df_sample['age_june_2012'] >= 26) &
                        (df_sample['age_june_2012'] <= 30)).astype(int)

print(f"\nSample with treatment/control groups: {len(df_sample):,}")
print(f"Treatment group (ages 26-30): {df_sample['treated'].sum():,}")
print(f"Control group (ages 31-35): {(df_sample['treated'] == 0).sum():,}")

# =============================================================================
# STEP 4: DEFINE PRE/POST PERIODS AND OUTCOME
# =============================================================================
print("\n" + "="*60)
print("STEP 4: Defining pre/post periods and outcome variable")
print("="*60)

# Exclude 2012 (implementation year - can't distinguish pre/post within year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# Post period: 2013-2016
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

print(f"\nPre-period (2006-2011): {(df_sample['post'] == 0).sum():,}")
print(f"Post-period (2013-2016): {(df_sample['post'] == 1).sum():,}")

# Define full-time employment outcome (UHRSWORK >= 35)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

print(f"\nFull-time employment (UHRSWORK >= 35):")
print(f"Full-time: {df_sample['fulltime'].sum():,}")
print(f"Not full-time: {(df_sample['fulltime'] == 0).sum():,}")
print(f"Overall rate: {df_sample['fulltime'].mean():.4f}")

# UHRSWORK distribution
print(f"\nUHRSWORK distribution:")
print(df_sample['UHRSWORK'].describe())

# =============================================================================
# STEP 5: SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*60)
print("STEP 5: Summary Statistics")
print("="*60)

# By treatment status
print("\n--- By Treatment Status ---")
summary_by_treated = df_sample.groupby('treated').agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': ['mean', 'std'],
    'SEX': 'mean',  # 1=male, 2=female
    'EDUC': 'mean',
    'UHRSWORK': 'mean'
}).round(4)
print(summary_by_treated)

# By period
print("\n--- By Period ---")
summary_by_period = df_sample.groupby('post').agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': ['mean', 'std']
}).round(4)
print(summary_by_period)

# 2x2 table: Treatment x Period
print("\n--- 2x2 Full-time Employment Rates ---")
crosstab = df_sample.groupby(['treated', 'post'])['fulltime'].agg(['mean', 'count']).unstack()
print(crosstab)

# Calculate simple DiD
ft_treat_pre = df_sample[(df_sample['treated']==1) & (df_sample['post']==0)]['fulltime'].mean()
ft_treat_post = df_sample[(df_sample['treated']==1) & (df_sample['post']==1)]['fulltime'].mean()
ft_ctrl_pre = df_sample[(df_sample['treated']==0) & (df_sample['post']==0)]['fulltime'].mean()
ft_ctrl_post = df_sample[(df_sample['treated']==0) & (df_sample['post']==1)]['fulltime'].mean()

print(f"\nSimple DiD Calculation:")
print(f"Treatment Pre:  {ft_treat_pre:.4f}")
print(f"Treatment Post: {ft_treat_post:.4f}")
print(f"Control Pre:    {ft_ctrl_pre:.4f}")
print(f"Control Post:   {ft_ctrl_post:.4f}")
print(f"\nTreatment change: {ft_treat_post - ft_treat_pre:.4f}")
print(f"Control change:   {ft_ctrl_post - ft_ctrl_pre:.4f}")
print(f"DiD estimate:     {(ft_treat_post - ft_treat_pre) - (ft_ctrl_post - ft_ctrl_pre):.4f}")

# =============================================================================
# STEP 6: DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "="*60)
print("STEP 6: Difference-in-Differences Regression")
print("="*60)

# Create interaction term
df_sample['treated_post'] = df_sample['treated'] * df_sample['post']

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_sample).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
# Add covariates: SEX, AGE (at time of survey), education, marital status
df_sample['male'] = (df_sample['SEX'] == 1).astype(int)
df_sample['married'] = (df_sample['MARST'] <= 2).astype(int)  # Married with spouse present/absent

model2 = smf.ols('fulltime ~ treated + post + treated_post + male + AGE + EDUC + married',
                 data=df_sample).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with year fixed effects ---")
df_sample['year_factor'] = pd.Categorical(df_sample['YEAR'])
model3 = smf.ols('fulltime ~ treated + treated_post + C(YEAR)', data=df_sample).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with year FE and demographic controls
print("\n--- Model 4: DiD with year FE and demographic controls ---")
model4 = smf.ols('fulltime ~ treated + treated_post + male + AGE + EDUC + married + C(YEAR)',
                 data=df_sample).fit(cov_type='HC1')
print(model4.summary())

# Model 5: Add state fixed effects
print("\n--- Model 5: DiD with year and state fixed effects ---")
model5 = smf.ols('fulltime ~ treated + treated_post + male + AGE + EDUC + married + C(YEAR) + C(STATEFIP)',
                 data=df_sample).fit(cov_type='HC1')
print(f"DiD Coefficient (treated_post): {model5.params['treated_post']:.4f}")
print(f"Std Error: {model5.bse['treated_post']:.4f}")
print(f"t-stat: {model5.tvalues['treated_post']:.4f}")
print(f"p-value: {model5.pvalues['treated_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {int(model5.nobs)}")

# =============================================================================
# STEP 7: WEIGHTED ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("STEP 7: Weighted Analysis (using PERWT)")
print("="*60)

# Model 6: Weighted DiD with controls
print("\n--- Model 6: Weighted DiD with year FE and controls ---")
model6 = smf.wls('fulltime ~ treated + treated_post + male + AGE + EDUC + married + C(YEAR)',
                 data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient (treated_post): {model6.params['treated_post']:.4f}")
print(f"Std Error: {model6.bse['treated_post']:.4f}")
print(f"t-stat: {model6.tvalues['treated_post']:.4f}")
print(f"p-value: {model6.pvalues['treated_post']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['treated_post', 0]:.4f}, {model6.conf_int().loc['treated_post', 1]:.4f}]")
print(f"R-squared: {model6.rsquared:.4f}")
print(f"N: {int(model6.nobs)}")

# =============================================================================
# STEP 8: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*60)
print("STEP 8: Robustness Checks")
print("="*60)

# Robustness 1: By sex
print("\n--- Robustness: By Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_sample[df_sample['SEX'] == sex_val]
    model_sex = smf.ols('fulltime ~ treated + treated_post + AGE + EDUC + married + C(YEAR)',
                        data=df_sex).fit(cov_type='HC1')
    print(f"{sex_name}: DiD = {model_sex.params['treated_post']:.4f} (SE: {model_sex.bse['treated_post']:.4f}), N = {int(model_sex.nobs)}")

# Robustness 2: Alternative age bandwidths
print("\n--- Robustness: Alternative Age Bandwidths ---")
for bandwidth in [(27, 29, 32, 34), (25, 30, 31, 36)]:
    t_low, t_high, c_low, c_high = bandwidth
    df_bw = df_eligible[
        (df_eligible['YEAR'] != 2012) &
        (((df_eligible['age_june_2012'] >= t_low) & (df_eligible['age_june_2012'] <= t_high)) |
         ((df_eligible['age_june_2012'] >= c_low) & (df_eligible['age_june_2012'] <= c_high)))
    ].copy()
    df_bw['treated'] = ((df_bw['age_june_2012'] >= t_low) & (df_bw['age_june_2012'] <= t_high)).astype(int)
    df_bw['post'] = (df_bw['YEAR'] >= 2013).astype(int)
    df_bw['treated_post'] = df_bw['treated'] * df_bw['post']
    df_bw['fulltime'] = (df_bw['UHRSWORK'] >= 35).astype(int)
    df_bw['male'] = (df_bw['SEX'] == 1).astype(int)
    df_bw['married'] = (df_bw['MARST'] <= 2).astype(int)

    model_bw = smf.ols('fulltime ~ treated + treated_post + male + AGE + EDUC + married + C(YEAR)',
                       data=df_bw).fit(cov_type='HC1')
    print(f"Treated {t_low}-{t_high} vs Control {c_low}-{c_high}: DiD = {model_bw.params['treated_post']:.4f} (SE: {model_bw.bse['treated_post']:.4f}), N = {int(model_bw.nobs)}")

# Robustness 3: Placebo test (pre-period only: 2006-2008 vs 2009-2011)
print("\n--- Robustness: Placebo Test (Pre-period only) ---")
df_placebo = df_sample[df_sample['post'] == 0].copy()
df_placebo['placebo_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treated_placebo_post'] = df_placebo['treated'] * df_placebo['placebo_post']

model_placebo = smf.ols('fulltime ~ treated + placebo_post + treated_placebo_post + male + AGE + EDUC + married + C(YEAR)',
                        data=df_placebo).fit(cov_type='HC1')
print(f"Placebo DiD = {model_placebo.params['treated_placebo_post']:.4f} (SE: {model_placebo.bse['treated_placebo_post']:.4f})")
print(f"p-value: {model_placebo.pvalues['treated_placebo_post']:.4f}")

# =============================================================================
# STEP 9: EVENT STUDY / YEAR-BY-YEAR ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("STEP 9: Event Study Analysis")
print("="*60)

# Create year-specific treatment effects (relative to 2011)
years = sorted(df_sample['YEAR'].unique())
print(f"Years in sample: {years}")

# Create year dummies interacted with treatment
for year in years:
    if year != 2011:  # 2011 is reference year
        df_sample[f'treated_year_{year}'] = df_sample['treated'] * (df_sample['YEAR'] == year).astype(int)

year_vars = [f'treated_year_{year}' for year in years if year != 2011]
formula = 'fulltime ~ treated + ' + ' + '.join(year_vars) + ' + male + AGE + EDUC + married + C(YEAR)'
model_event = smf.ols(formula, data=df_sample).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
for year in years:
    if year != 2011:
        var = f'treated_year_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        pval = model_event.pvalues[var]
        print(f"Year {year}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# =============================================================================
# STEP 10: SAVE RESULTS
# =============================================================================
print("\n" + "="*60)
print("STEP 10: Saving Results")
print("="*60)

# Save detailed results
results = {
    'simple_did': (ft_treat_post - ft_treat_pre) - (ft_ctrl_post - ft_ctrl_pre),
    'model1_did': model1.params['treated_post'],
    'model1_se': model1.bse['treated_post'],
    'model4_did': model4.params['treated_post'],
    'model4_se': model4.bse['treated_post'],
    'model4_pval': model4.pvalues['treated_post'],
    'model4_ci_low': model4.conf_int().loc['treated_post', 0],
    'model4_ci_high': model4.conf_int().loc['treated_post', 1],
    'n_total': len(df_sample),
    'n_treated': df_sample['treated'].sum(),
    'n_control': (df_sample['treated'] == 0).sum(),
    'n_pre': (df_sample['post'] == 0).sum(),
    'n_post': (df_sample['post'] == 1).sum(),
    'ft_rate_overall': df_sample['fulltime'].mean(),
    'ft_treat_pre': ft_treat_pre,
    'ft_treat_post': ft_treat_post,
    'ft_ctrl_pre': ft_ctrl_pre,
    'ft_ctrl_post': ft_ctrl_post
}

# Save to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save event study coefficients
event_study_results = []
for year in years:
    if year != 2011:
        var = f'treated_year_{year}'
        event_study_results.append({
            'year': year,
            'coefficient': model_event.params[var],
            'se': model_event.bse[var],
            'pvalue': model_event.pvalues[var],
            'ci_low': model_event.conf_int().loc[var, 0],
            'ci_high': model_event.conf_int().loc[var, 1]
        })
event_study_df = pd.DataFrame(event_study_results)
event_study_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"""
Research Question: Effect of DACA eligibility on full-time employment

Sample:
- Hispanic-Mexican, Mexico-born, non-citizens
- Arrived in US before age 16
- Immigrated by 2007 (continuous residence)
- Treatment: Ages 26-30 as of June 2012 (DACA eligible)
- Control: Ages 31-35 as of June 2012 (too old for DACA)

Sample Size: {len(df_sample):,}
- Treatment group: {df_sample['treated'].sum():,}
- Control group: {(df_sample['treated'] == 0).sum():,}
- Pre-period (2006-2011): {(df_sample['post'] == 0).sum():,}
- Post-period (2013-2016): {(df_sample['post'] == 1).sum():,}

Full-time Employment Rates:
- Treatment Pre:  {ft_treat_pre:.4f} ({ft_treat_pre*100:.2f}%)
- Treatment Post: {ft_treat_post:.4f} ({ft_treat_post*100:.2f}%)
- Control Pre:    {ft_ctrl_pre:.4f} ({ft_ctrl_pre*100:.2f}%)
- Control Post:   {ft_ctrl_post:.4f} ({ft_ctrl_post*100:.2f}%)

PREFERRED ESTIMATE (Model 4: DiD with year FE and controls):
- DiD Coefficient: {model4.params['treated_post']:.4f}
- Standard Error: {model4.bse['treated_post']:.4f}
- 95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]
- p-value: {model4.pvalues['treated_post']:.4f}
- N: {int(model4.nobs)}

Interpretation: DACA eligibility is associated with a {model4.params['treated_post']*100:.2f} percentage point
{'increase' if model4.params['treated_post'] > 0 else 'decrease'} in full-time employment
(p = {model4.pvalues['treated_post']:.4f}).
""")

# Save sample statistics for the report
sample_stats = df_sample.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'male': 'mean',
    'married': 'mean',
    'EDUC': 'mean',
    'PERWT': 'sum'
}).round(4)
sample_stats.to_csv('sample_statistics.csv')
print("\nSample statistics saved to sample_statistics.csv")

# Year-by-year full-time rates by treatment status
yearly_rates = df_sample.groupby(['YEAR', 'treated'])['fulltime'].mean().unstack()
yearly_rates.columns = ['Control', 'Treatment']
yearly_rates.to_csv('yearly_fulltime_rates.csv')
print("Yearly rates saved to yearly_fulltime_rates.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
