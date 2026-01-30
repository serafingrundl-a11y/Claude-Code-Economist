"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n1. Loading data...")

# Read in chunks due to large file size
chunks = []
for chunk in pd.read_csv('data/data.csv', chunksize=500000):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

print(f"   Total observations loaded: {len(df):,}")
print(f"   Years available: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. SAMPLE SELECTION - Hispanic-Mexican Mexican-born
# =============================================================================
print("\n2. Sample selection...")

# Hispanic-Mexican ethnicity (HISPAN == 1 for Mexican)
# Born in Mexico (BPL == 200 for Mexico)
df_sample = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
print(f"   After restricting to Hispanic-Mexican Mexican-born: {len(df_sample):,}")

# =============================================================================
# 3. CONSTRUCT DACA ELIGIBILITY VARIABLES
# =============================================================================
print("\n3. Constructing DACA eligibility variables...")

# DACA was implemented June 15, 2012
# Eligibility criteria:
# 1. Arrived in US before 16th birthday
# 2. Had not yet turned 31 as of June 15, 2012
# 3. Lived continuously in US since June 15, 2007
# 4. Not a citizen

# Calculate age as of June 15, 2012
# BIRTHYR is available
df_sample['age_june2012'] = 2012 - df_sample['BIRTHYR']

# Check arrival age: arrived before 16th birthday
# Year of immigration is YRIMMIG
# Age at arrival = YRIMMIG - BIRTHYR
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Arrived before 16th birthday
df_sample['arrived_before_16'] = (df_sample['age_at_arrival'] < 16) & (df_sample['YRIMMIG'] > 0)

# Lived in US since June 15, 2007 (arrived by 2007)
df_sample['in_us_since_2007'] = (df_sample['YRIMMIG'] <= 2007) & (df_sample['YRIMMIG'] > 0)

# Not a citizen (CITIZEN == 3 means "Not a citizen")
# Also include those who have not received papers
# CITIZEN = 3 (Not a citizen) or CITIZEN = 4 (Not a citizen, but has received first papers)
# Per instructions: treat those without citizenship as potentially undocumented
df_sample['not_citizen'] = df_sample['CITIZEN'].isin([3, 4, 5])

# DACA eligible (excluding age criterion to define treatment/control)
df_sample['daca_eligible_base'] = (
    df_sample['arrived_before_16'] &
    df_sample['in_us_since_2007'] &
    df_sample['not_citizen']
)

print(f"   Arrived before age 16: {df_sample['arrived_before_16'].sum():,}")
print(f"   In US since 2007: {df_sample['in_us_since_2007'].sum():,}")
print(f"   Not citizen: {df_sample['not_citizen'].sum():,}")
print(f"   DACA eligible (base criteria): {df_sample['daca_eligible_base'].sum():,}")

# =============================================================================
# 4. DEFINE TREATMENT AND CONTROL GROUPS
# =============================================================================
print("\n4. Defining treatment and control groups...")

# Treatment group: ages 26-30 as of June 15, 2012
# These individuals were eligible for DACA (under 31 when policy implemented)
# Birth years: 1982-1986 (turning 26-30 in 2012)

# Control group: ages 31-35 as of June 15, 2012
# Too old to be eligible for DACA
# Birth years: 1977-1981 (turning 31-35 in 2012)

# Treatment: born 1982-1986 (would be 26-30 in 2012)
df_sample['treatment_group'] = df_sample['BIRTHYR'].between(1982, 1986)

# Control: born 1977-1981 (would be 31-35 in 2012)
df_sample['control_group'] = df_sample['BIRTHYR'].between(1977, 1981)

# Create treatment indicator (1 = treatment, 0 = control)
df_sample['treated'] = np.where(df_sample['treatment_group'], 1,
                                np.where(df_sample['control_group'], 0, np.nan))

# =============================================================================
# 5. DEFINE PRE/POST PERIODS
# =============================================================================
print("\n5. Defining pre/post treatment periods...")

# Pre-treatment: 2006-2011 (DACA not yet implemented)
# 2012 is excluded because DACA was implemented mid-year
# Post-treatment: 2013-2016

df_sample['post'] = np.where(df_sample['YEAR'] >= 2013, 1,
                             np.where(df_sample['YEAR'] <= 2011, 0, np.nan))

print(f"   Pre-period years (2006-2011)")
print(f"   Post-period years (2013-2016)")
print(f"   Excluding 2012 due to mid-year policy implementation")

# =============================================================================
# 6. CREATE OUTCOME VARIABLE - Full-time Employment
# =============================================================================
print("\n6. Creating outcome variable...")

# Full-time employment: usually working 35+ hours per week
# UHRSWORK: usual hours worked per week
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create alternative outcome: employed at all
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# =============================================================================
# 7. FINAL ANALYSIS SAMPLE
# =============================================================================
print("\n7. Creating final analysis sample...")

# Keep only those in treatment or control groups
# Keep only pre or post periods
# Keep only those meeting base DACA eligibility criteria

analysis_df = df_sample[
    (df_sample['treated'].notna()) &
    (df_sample['post'].notna()) &
    (df_sample['daca_eligible_base'])
].copy()

print(f"   Final analysis sample size: {len(analysis_df):,}")
print(f"\n   Sample by group and period:")
print(analysis_df.groupby(['treated', 'post']).size().unstack())

# =============================================================================
# 8. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n8. Descriptive Statistics")
print("="*60)

# Demographics by treatment group
print("\nSample characteristics by treatment status:")
print("-"*60)

# Create some additional demographic variables for controls
analysis_df['male'] = (analysis_df['SEX'] == 1).astype(int)
analysis_df['married'] = analysis_df['MARST'].isin([1, 2]).astype(int)

# Education categories
analysis_df['less_than_hs'] = (analysis_df['EDUC'] < 6).astype(int)
analysis_df['hs_grad'] = (analysis_df['EDUC'] == 6).astype(int)
analysis_df['some_college'] = (analysis_df['EDUC'].between(7, 9)).astype(int)
analysis_df['college_plus'] = (analysis_df['EDUC'] >= 10).astype(int)

# Age at survey
analysis_df['age'] = analysis_df['AGE']

for var in ['fulltime', 'employed', 'male', 'married', 'less_than_hs', 'hs_grad',
            'some_college', 'college_plus', 'age']:
    treated_mean = analysis_df.loc[analysis_df['treated'] == 1, var].mean()
    control_mean = analysis_df.loc[analysis_df['treated'] == 0, var].mean()
    print(f"   {var:15s}: Treatment={treated_mean:.3f}  Control={control_mean:.3f}")

# =============================================================================
# 9. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n9. Difference-in-Differences Analysis")
print("="*60)

# Create interaction term
analysis_df['treated_post'] = analysis_df['treated'] * analysis_df['post']

# Model 1: Basic DiD
print("\nModel 1: Basic Difference-in-Differences")
print("-"*60)

model1 = smf.ols('fulltime ~ treated + post + treated_post', data=analysis_df).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n\nModel 2: DiD with Demographic Controls")
print("-"*60)

model2 = smf.ols('fulltime ~ treated + post + treated_post + male + married + age + I(age**2)',
                 data=analysis_df).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with demographic and education controls
print("\n\nModel 3: DiD with Demographics and Education Controls")
print("-"*60)

model3 = smf.ols('fulltime ~ treated + post + treated_post + male + married + age + I(age**2) + hs_grad + some_college + college_plus',
                 data=analysis_df).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: DiD with state and year fixed effects
print("\n\nModel 4: DiD with State and Year Fixed Effects")
print("-"*60)

# Add year and state dummies
analysis_df['year_factor'] = pd.Categorical(analysis_df['YEAR'])
analysis_df['state_factor'] = pd.Categorical(analysis_df['STATEFIP'])

model4 = smf.ols('fulltime ~ treated + treated_post + male + married + age + I(age**2) + hs_grad + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                 data=analysis_df).fit(cov_type='HC1')

# Extract the coefficient of interest
print(f"   Coefficient on treated_post: {model4.params['treated_post']:.4f}")
print(f"   Standard Error: {model4.bse['treated_post']:.4f}")
print(f"   t-statistic: {model4.tvalues['treated_post']:.4f}")
print(f"   p-value: {model4.pvalues['treated_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")

# =============================================================================
# 10. ROBUSTNESS CHECKS
# =============================================================================
print("\n\n10. Robustness Checks")
print("="*60)

# 10a. Alternative outcome: Any employment
print("\nRobustness Check 1: Any Employment (instead of full-time)")
print("-"*60)
model_emp = smf.ols('employed ~ treated + treated_post + male + married + age + I(age**2) + hs_grad + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                    data=analysis_df).fit(cov_type='HC1')
print(f"   Coefficient on treated_post: {model_emp.params['treated_post']:.4f}")
print(f"   Standard Error: {model_emp.bse['treated_post']:.4f}")
print(f"   p-value: {model_emp.pvalues['treated_post']:.4f}")

# 10b. By gender
print("\nRobustness Check 2: Effects by Gender")
print("-"*60)

# Males only
male_df = analysis_df[analysis_df['male'] == 1]
model_male = smf.ols('fulltime ~ treated + treated_post + married + age + I(age**2) + hs_grad + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                     data=male_df).fit(cov_type='HC1')
print(f"   Males - Coefficient on treated_post: {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f})")

# Females only
female_df = analysis_df[analysis_df['male'] == 0]
model_female = smf.ols('fulltime ~ treated + treated_post + married + age + I(age**2) + hs_grad + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                       data=female_df).fit(cov_type='HC1')
print(f"   Females - Coefficient on treated_post: {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f})")

# 10c. Placebo test with different birth year cutoffs
print("\nRobustness Check 3: Placebo Test (older cohorts)")
print("-"*60)

# Placebo: Compare ages 31-35 (born 1977-1981) to ages 36-40 (born 1972-1976)
df_placebo = df_sample[
    (df_sample['BIRTHYR'].between(1972, 1981)) &
    (df_sample['post'].notna()) &
    (df_sample['daca_eligible_base'])
].copy()

df_placebo['placebo_treated'] = df_placebo['BIRTHYR'].between(1977, 1981).astype(int)
df_placebo['placebo_treated_post'] = df_placebo['placebo_treated'] * df_placebo['post']
df_placebo['male'] = (df_placebo['SEX'] == 1).astype(int)
df_placebo['married'] = df_placebo['MARST'].isin([1, 2]).astype(int)
df_placebo['hs_grad'] = (df_placebo['EDUC'] == 6).astype(int)
df_placebo['some_college'] = (df_placebo['EDUC'].between(7, 9)).astype(int)
df_placebo['college_plus'] = (df_placebo['EDUC'] >= 10).astype(int)
df_placebo['age'] = df_placebo['AGE']

if len(df_placebo) > 100:
    model_placebo = smf.ols('fulltime ~ placebo_treated + placebo_treated_post + male + married + age + I(age**2) + hs_grad + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                            data=df_placebo).fit(cov_type='HC1')
    print(f"   Placebo Coefficient on treated_post: {model_placebo.params['placebo_treated_post']:.4f}")
    print(f"   Standard Error: {model_placebo.bse['placebo_treated_post']:.4f}")
    print(f"   p-value: {model_placebo.pvalues['placebo_treated_post']:.4f}")

# =============================================================================
# 11. EVENT STUDY / DYNAMIC EFFECTS
# =============================================================================
print("\n\n11. Event Study Analysis")
print("="*60)

# Create year-specific treatment effects
analysis_df['year_2006'] = (analysis_df['YEAR'] == 2006).astype(int)
analysis_df['year_2007'] = (analysis_df['YEAR'] == 2007).astype(int)
analysis_df['year_2008'] = (analysis_df['YEAR'] == 2008).astype(int)
analysis_df['year_2009'] = (analysis_df['YEAR'] == 2009).astype(int)
analysis_df['year_2010'] = (analysis_df['YEAR'] == 2010).astype(int)
# 2011 is reference year
analysis_df['year_2013'] = (analysis_df['YEAR'] == 2013).astype(int)
analysis_df['year_2014'] = (analysis_df['YEAR'] == 2014).astype(int)
analysis_df['year_2015'] = (analysis_df['YEAR'] == 2015).astype(int)
analysis_df['year_2016'] = (analysis_df['YEAR'] == 2016).astype(int)

# Interactions
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    analysis_df[f'treated_year_{yr}'] = analysis_df['treated'] * analysis_df[f'year_{yr}']

event_formula = 'fulltime ~ treated + treated_year_2006 + treated_year_2007 + treated_year_2008 + treated_year_2009 + treated_year_2010 + treated_year_2013 + treated_year_2014 + treated_year_2015 + treated_year_2016 + male + married + age + I(age**2) + hs_grad + some_college + college_plus + C(YEAR) + C(STATEFIP)'

model_event = smf.ols(event_formula, data=analysis_df).fit(cov_type='HC1')

print("\nYear-by-year treatment effects (relative to 2011):")
print("-"*60)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treated_year_{yr}']
    se = model_event.bse[f'treated_year_{yr}']
    pval = model_event.pvalues[f'treated_year_{yr}']
    print(f"   {yr}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")

# =============================================================================
# 12. SUMMARY OF MAIN RESULTS
# =============================================================================
print("\n\n" + "="*80)
print("SUMMARY OF MAIN RESULTS")
print("="*80)

print("\n" + "-"*60)
print("PREFERRED SPECIFICATION: Model 4 (Full controls + FE)")
print("-"*60)
print(f"   Effect Size (DiD estimate): {model4.params['treated_post']:.4f}")
print(f"   Standard Error: {model4.bse['treated_post']:.4f}")
print(f"   95% Confidence Interval: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   t-statistic: {model4.tvalues['treated_post']:.4f}")
print(f"   p-value: {model4.pvalues['treated_post']:.4f}")
print(f"   Sample Size: {int(model4.nobs):,}")

print("\n" + "-"*60)
print("INTERPRETATION")
print("-"*60)
effect = model4.params['treated_post']
if effect > 0:
    print(f"   DACA eligibility is associated with a {effect*100:.2f} percentage point")
    print(f"   {'increase' if effect > 0 else 'decrease'} in the probability of full-time employment")
    print(f"   for the treatment group relative to the control group.")
else:
    print(f"   DACA eligibility is associated with a {abs(effect)*100:.2f} percentage point")
    print(f"   decrease in the probability of full-time employment")
    print(f"   for the treatment group relative to the control group.")

if model4.pvalues['treated_post'] < 0.05:
    print(f"\n   This effect is statistically significant at the 5% level.")
elif model4.pvalues['treated_post'] < 0.10:
    print(f"\n   This effect is statistically significant at the 10% level.")
else:
    print(f"\n   This effect is not statistically significant at conventional levels.")

# =============================================================================
# 13. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n\n13. Saving results...")

# Save key statistics to a file for the LaTeX report
results = {
    'n_total_obs': len(df),
    'n_mexican_born': len(df_sample),
    'n_analysis_sample': len(analysis_df),
    'n_treatment': int((analysis_df['treated'] == 1).sum()),
    'n_control': int((analysis_df['treated'] == 0).sum()),
    'n_pre': int((analysis_df['post'] == 0).sum()),
    'n_post': int((analysis_df['post'] == 1).sum()),
    'mean_fulltime_treat_pre': analysis_df[(analysis_df['treated']==1) & (analysis_df['post']==0)]['fulltime'].mean(),
    'mean_fulltime_treat_post': analysis_df[(analysis_df['treated']==1) & (analysis_df['post']==1)]['fulltime'].mean(),
    'mean_fulltime_control_pre': analysis_df[(analysis_df['treated']==0) & (analysis_df['post']==0)]['fulltime'].mean(),
    'mean_fulltime_control_post': analysis_df[(analysis_df['treated']==0) & (analysis_df['post']==1)]['fulltime'].mean(),
    'model1_coef': model1.params['treated_post'],
    'model1_se': model1.bse['treated_post'],
    'model1_pval': model1.pvalues['treated_post'],
    'model2_coef': model2.params['treated_post'],
    'model2_se': model2.bse['treated_post'],
    'model2_pval': model2.pvalues['treated_post'],
    'model3_coef': model3.params['treated_post'],
    'model3_se': model3.bse['treated_post'],
    'model3_pval': model3.pvalues['treated_post'],
    'model4_coef': model4.params['treated_post'],
    'model4_se': model4.bse['treated_post'],
    'model4_pval': model4.pvalues['treated_post'],
    'model4_ci_low': model4.conf_int().loc['treated_post', 0],
    'model4_ci_high': model4.conf_int().loc['treated_post', 1],
    'model4_nobs': int(model4.nobs),
    'model_emp_coef': model_emp.params['treated_post'],
    'model_emp_se': model_emp.bse['treated_post'],
    'model_male_coef': model_male.params['treated_post'],
    'model_male_se': model_male.bse['treated_post'],
    'model_female_coef': model_female.params['treated_post'],
    'model_female_se': model_female.bse['treated_post'],
}

# Save event study coefficients
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    results[f'event_{yr}_coef'] = model_event.params[f'treated_year_{yr}']
    results[f'event_{yr}_se'] = model_event.bse[f'treated_year_{yr}']

# Export to file
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("   Results saved to analysis_results.json")

# Also create summary statistics table
print("\n\nCreating summary statistics...")
summary_stats = analysis_df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean'],
    'male': ['mean'],
    'married': ['mean'],
    'age': ['mean'],
    'less_than_hs': ['mean'],
    'hs_grad': ['mean'],
    'some_college': ['mean'],
    'college_plus': ['mean']
}).round(4)

summary_stats.to_csv('summary_statistics.csv')
print("   Summary statistics saved to summary_statistics.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
