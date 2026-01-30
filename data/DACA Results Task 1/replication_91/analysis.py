"""
DACA Replication Study Analysis
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the United States.

Author: Anonymous (Replication Study 91)
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS SCRIPT")
print("=" * 80)

# =============================================================================
# STEP 1: Load Data in Chunks and Filter
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING AND FILTERING DATA")
print("=" * 80)

# Read in chunks and filter to Hispanic-Mexican only
chunks = []
chunk_size = 1000000  # 1 million rows at a time

print("Loading data in chunks...")
for i, chunk in enumerate(pd.read_csv('data/data.csv', chunksize=chunk_size)):
    # Filter to Hispanic-Mexican (HISPAN == 1) and Mexican-born (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

df = pd.concat(chunks, ignore_index=True)
del chunks  # Free memory

print(f"\nTotal Hispanic-Mexican, Mexican-born observations: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: Define DACA Eligibility
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: DEFINING DACA ELIGIBILITY")
print("=" * 80)

# Check citizenship status
print("\nCITIZEN value counts:")
print(df['CITIZEN'].value_counts())

# Filter to non-citizens (CITIZEN == 3) as proxy for undocumented
df = df[df['CITIZEN'] == 3].copy()
print(f"\nAfter filtering to non-citizens (CITIZEN=3): {len(df):,}")

# Calculate age at arrival for each person
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
print("\nAge at arrival distribution:")
print(df['age_at_arrival'].describe())

# Calculate age as of June 15, 2012
# If born in Q1-Q2 (Jan-Jun), they would have had their birthday by June 15
# If born in Q3-Q4 (Jul-Dec), they would not have had their birthday by June 15
df['age_june2012'] = 2012 - df['BIRTHYR']
# Adjust for those born in second half of year (Q3-Q4)
df.loc[df['BIRTHQTR'] >= 3, 'age_june2012'] -= 1

print("\nAge as of June 2012 distribution:")
print(df['age_june2012'].describe())

# =============================================================================
# STEP 3: Create Treatment Groups
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: CREATING TREATMENT GROUPS")
print("=" * 80)

# DACA Eligibility Criteria:
# 1. Arrived before 16th birthday (age_at_arrival < 16)
# 2. Age <= 30 as of June 15, 2012 (born in 1982 or later; age_june2012 <= 30)
# 3. Arrived by 2007 (continuous presence since June 15, 2007)

# Create eligibility flags
df['arrived_before_16'] = df['age_at_arrival'] < 16
df['age_eligible_2012'] = df['age_june2012'] <= 30
df['arrived_by_2007'] = df['YRIMMIG'] <= 2007

print("\nEligibility criteria breakdown:")
print(f"Arrived before age 16: {df['arrived_before_16'].sum():,} ({df['arrived_before_16'].mean()*100:.1f}%)")
print(f"Age <= 30 as of June 2012: {df['age_eligible_2012'].sum():,} ({df['age_eligible_2012'].mean()*100:.1f}%)")
print(f"Arrived by 2007: {df['arrived_by_2007'].sum():,} ({df['arrived_by_2007'].mean()*100:.1f}%)")

# DACA eligible = all three conditions
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['age_eligible_2012'] &
    df['arrived_by_2007']
)

print(f"\nDACA eligible (all criteria): {df['daca_eligible'].sum():,} ({df['daca_eligible'].mean()*100:.1f}%)")

# =============================================================================
# STEP 4: Create Outcome Variable and Time Period
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: CREATING OUTCOME VARIABLE")
print("=" * 80)

# Outcome: Full-time employment (usually working 35+ hours per week)
print("\nEMPSTAT value counts:")
print(df['EMPSTAT'].value_counts())

print("\nUHRSWORK summary:")
print(df['UHRSWORK'].describe())

# Full-time = employed AND usual hours >= 35
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
df['fulltime'] = ((df['EMPSTAT'] == 1) & (df['UHRSWORK'] >= 35)).astype(int)

print(f"\nEmployed: {df['employed'].sum():,} ({df['employed'].mean()*100:.1f}%)")
print(f"Full-time employed: {df['fulltime'].sum():,} ({df['fulltime'].mean()*100:.1f}%)")

# Time period
# Pre-DACA: 2006-2011
# Post-DACA: 2013-2016 (exclude 2012 due to mid-year implementation)
df['post'] = df['YEAR'] >= 2013
df['exclude_2012'] = df['YEAR'] != 2012

print(f"\nYear distribution:")
print(df.groupby('YEAR').size())

# =============================================================================
# STEP 5: Create Analysis Sample
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: CREATING ANALYSIS SAMPLE")
print("=" * 80)

# Restrict to working-age population (18-64) at time of survey
df['working_age'] = (df['AGE'] >= 18) & (df['AGE'] <= 64)

# Analysis sample: working age, exclude 2012
df_analysis = df[
    df['working_age'] &
    df['exclude_2012']
].copy()

print(f"Analysis sample size: {len(df_analysis):,}")
print(f"DACA eligible in sample: {df_analysis['daca_eligible'].sum():,}")
print(f"Not DACA eligible in sample: {(~df_analysis['daca_eligible']).sum():,}")

# For DiD, we need a control group that is similar but not eligible
# Control: Mexican-born, non-citizen, arrived before 16, arrived by 2007,
# BUT too old (31-45 as of June 2012)

df_analysis['control_group'] = (
    df_analysis['arrived_before_16'] &
    df_analysis['arrived_by_2007'] &
    (df_analysis['age_june2012'] > 30) &
    (df_analysis['age_june2012'] <= 45)
)

print(f"\nControl group size: {df_analysis['control_group'].sum():,}")

# DiD analysis sample: either treatment or control
df_did = df_analysis[
    df_analysis['daca_eligible'] | df_analysis['control_group']
].copy()

print(f"\nDiD sample size: {len(df_did):,}")
print(f"  Treatment (DACA eligible): {df_did['daca_eligible'].sum():,}")
print(f"  Control: {df_did['control_group'].sum():,}")

# Create treatment indicator
df_did['treated'] = df_did['daca_eligible'].astype(int)

# Free memory
del df, df_analysis

# =============================================================================
# STEP 6: Descriptive Statistics
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: DESCRIPTIVE STATISTICS")
print("=" * 80)

# Summary by treatment status and time period
summary_stats = df_did.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary statistics by treatment status and period:")
print(summary_stats)

# Full-time employment rates
print("\n\nFull-time employment rates (weighted):")
print("-" * 60)
ft_rates = df_did.groupby(['treated', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
ft_rates.index = ['Control (Age 31-45)', 'Treatment (DACA eligible)']
ft_rates.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
print(ft_rates)

# Calculate DiD manually
print("\n\nManual DiD calculation:")
pre_treat = ft_rates.loc['Treatment (DACA eligible)', 'Pre-DACA (2006-2011)']
post_treat = ft_rates.loc['Treatment (DACA eligible)', 'Post-DACA (2013-2016)']
pre_control = ft_rates.loc['Control (Age 31-45)', 'Pre-DACA (2006-2011)']
post_control = ft_rates.loc['Control (Age 31-45)', 'Post-DACA (2013-2016)']

diff_treat = post_treat - pre_treat
diff_control = post_control - pre_control
did_effect = diff_treat - diff_control

print(f"Treatment group change: {diff_treat:.4f}")
print(f"Control group change: {diff_control:.4f}")
print(f"Difference-in-Differences: {did_effect:.4f}")

# =============================================================================
# STEP 7: Regression Analysis
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: REGRESSION ANALYSIS")
print("=" * 80)

# Create interaction term
df_did['post_int'] = df_did['post'].astype(int)
df_did['treat_post'] = df_did['treated'] * df_did['post_int']

# Add control variables
df_did['female'] = (df_did['SEX'] == 2).astype(int)
df_did['age_sq'] = df_did['AGE'] ** 2
df_did['married'] = (df_did['MARST'] == 1).astype(int)
df_did['educ_hs'] = (df_did['EDUC'] >= 6).astype(int)  # High school or more

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ treated + post_int + treat_post',
                  data=df_did,
                  weights=df_did['PERWT']).fit(cov_type='HC1')
print(f"DiD effect: {model1.params['treat_post']:.6f}")
print(f"  SE: {model1.bse['treat_post']:.6f}")
print(f"  p-value: {model1.pvalues['treat_post']:.6f}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls('fulltime ~ treated + post_int + treat_post + AGE + age_sq + female + married + educ_hs',
                  data=df_did,
                  weights=df_did['PERWT']).fit(cov_type='HC1')
print(f"DiD effect: {model2.params['treat_post']:.6f}")
print(f"  SE: {model2.bse['treat_post']:.6f}")
print(f"  p-value: {model2.pvalues['treat_post']:.6f}")

# Model 3: DiD with state and year fixed effects
print("\n--- Model 3: DiD with State and Year Fixed Effects ---")
model3 = smf.wls('fulltime ~ treated + treat_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                  data=df_did,
                  weights=df_did['PERWT']).fit(cov_type='HC1')

print(f"\nKey coefficients from Model 3:")
print(f"treat_post (DiD effect): {model3.params['treat_post']:.6f}")
print(f"  Standard Error: {model3.bse['treat_post']:.6f}")
print(f"  t-statistic: {model3.tvalues['treat_post']:.4f}")
print(f"  p-value: {model3.pvalues['treat_post']:.6f}")
ci = model3.conf_int().loc['treat_post']
print(f"  95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")

# =============================================================================
# STEP 8: Robustness Checks
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: ROBUSTNESS CHECKS")
print("=" * 80)

# Robustness 1: Alternative outcome - any employment
print("\n--- Robustness 1: Any Employment ---")
model_emp = smf.wls('employed ~ treated + post_int + treat_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                     data=df_did,
                     weights=df_did['PERWT']).fit(cov_type='HC1')
print(f"DiD effect on employment: {model_emp.params['treat_post']:.6f}")
print(f"  SE: {model_emp.bse['treat_post']:.6f}")
print(f"  p-value: {model_emp.pvalues['treat_post']:.6f}")

# Robustness 2: Men only
print("\n--- Robustness 2: Men Only ---")
df_men = df_did[df_did['female'] == 0]
model_men = smf.wls('fulltime ~ treated + post_int + treat_post + AGE + age_sq + married + educ_hs + C(YEAR) + C(STATEFIP)',
                     data=df_men,
                     weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"DiD effect (men): {model_men.params['treat_post']:.6f}")
print(f"  SE: {model_men.bse['treat_post']:.6f}")
print(f"  p-value: {model_men.pvalues['treat_post']:.6f}")

# Robustness 3: Women only
print("\n--- Robustness 3: Women Only ---")
df_women = df_did[df_did['female'] == 1]
model_women = smf.wls('fulltime ~ treated + post_int + treat_post + AGE + age_sq + married + educ_hs + C(YEAR) + C(STATEFIP)',
                       data=df_women,
                       weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"DiD effect (women): {model_women.params['treat_post']:.6f}")
print(f"  SE: {model_women.bse['treat_post']:.6f}")
print(f"  p-value: {model_women.pvalues['treat_post']:.6f}")

# Robustness 4: Narrower control group (ages 31-40 instead of 31-45)
print("\n--- Robustness 4: Narrower Control Group (Age 31-40) ---")
df_narrow = df_did[
    df_did['daca_eligible'] |
    ((df_did['control_group']) & (df_did['age_june2012'] <= 40))
].copy()
model_narrow = smf.wls('fulltime ~ treated + post_int + treat_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                        data=df_narrow,
                        weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"DiD effect (narrow control): {model_narrow.params['treat_post']:.6f}")
print(f"  SE: {model_narrow.bse['treat_post']:.6f}")
print(f"  p-value: {model_narrow.pvalues['treat_post']:.6f}")

# =============================================================================
# STEP 9: Placebo Test
# =============================================================================
print("\n" + "=" * 80)
print("STEP 9: PLACEBO TEST")
print("=" * 80)

# Placebo: Use pre-period only and pretend DACA was implemented in 2009
df_pre = df_did[df_did['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_treat_post'] = df_pre['treated'] * df_pre['placebo_post']

model_placebo = smf.wls('fulltime ~ treated + placebo_post + placebo_treat_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                         data=df_pre,
                         weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD effect (2009): {model_placebo.params['placebo_treat_post']:.6f}")
print(f"  SE: {model_placebo.bse['placebo_treat_post']:.6f}")
print(f"  p-value: {model_placebo.pvalues['placebo_treat_post']:.6f}")

# =============================================================================
# STEP 10: Event Study
# =============================================================================
print("\n" + "=" * 80)
print("STEP 10: EVENT STUDY")
print("=" * 80)

# Create year-specific treatment effects
# Reference year: 2011 (last pre-treatment year)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_did[f'year_{year}'] = (df_did['YEAR'] == year).astype(int)
    df_did[f'treat_year_{year}'] = df_did['treated'] * df_did[f'year_{year}']

event_formula = 'fulltime ~ treated + ' + \
                'treat_year_2006 + treat_year_2007 + treat_year_2008 + treat_year_2009 + treat_year_2010 + ' + \
                'treat_year_2013 + treat_year_2014 + treat_year_2015 + treat_year_2016 + ' + \
                'AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)'

model_event = smf.wls(event_formula,
                       data=df_did,
                       weights=df_did['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 60)
event_results = []
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_year_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    ci_low, ci_high = model_event.conf_int().loc[var]
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"  {year}: {coef:8.5f} ({se:.5f}){sig}")
    event_results.append({'year': year, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})

# =============================================================================
# STEP 11: Summary Statistics for Tables
# =============================================================================
print("\n" + "=" * 80)
print("STEP 11: DETAILED SUMMARY STATISTICS FOR TABLES")
print("=" * 80)

# Sample sizes by year
print("\nSample sizes by year and treatment status:")
sample_counts = df_did.groupby(['YEAR', 'treated']).agg({
    'fulltime': 'count',
    'PERWT': 'sum'
}).round(0)
sample_counts.columns = ['Unweighted N', 'Weighted N']
print(sample_counts)

# Pre-post comparison
print("\n\nPre-Post Full-time Employment Rates (weighted):")
for treat_val, label in [(1, 'Treatment (DACA eligible)'), (0, 'Control (Age 31-45)')]:
    df_sub = df_did[df_did['treated'] == treat_val]
    pre_rate = np.average(df_sub[df_sub['post'] == False]['fulltime'],
                          weights=df_sub[df_sub['post'] == False]['PERWT'])
    post_rate = np.average(df_sub[df_sub['post'] == True]['fulltime'],
                           weights=df_sub[df_sub['post'] == True]['PERWT'])
    print(f"\n{label}:")
    print(f"  Pre-DACA:  {pre_rate:.4f} ({pre_rate*100:.2f}%)")
    print(f"  Post-DACA: {post_rate:.4f} ({post_rate*100:.2f}%)")
    print(f"  Change:    {post_rate - pre_rate:.4f} ({(post_rate-pre_rate)*100:.2f} pp)")

# Pre-treatment balance table
print("\n\nPre-treatment means (Table 1):")
df_pre_data = df_did[df_did['post'] == False]
balance_vars = ['fulltime', 'employed', 'AGE', 'female', 'married', 'educ_hs']
balance_results = []
for var in balance_vars:
    treat_mean = np.average(df_pre_data[df_pre_data['treated'] == 1][var],
                            weights=df_pre_data[df_pre_data['treated'] == 1]['PERWT'])
    control_mean = np.average(df_pre_data[df_pre_data['treated'] == 0][var],
                              weights=df_pre_data[df_pre_data['treated'] == 0]['PERWT'])
    print(f"{var}: Treatment={treat_mean:.4f}, Control={control_mean:.4f}, Diff={treat_mean-control_mean:.4f}")
    balance_results.append({
        'variable': var,
        'treatment_mean': treat_mean,
        'control_mean': control_mean,
        'difference': treat_mean - control_mean
    })

# =============================================================================
# STEP 12: Save Results
# =============================================================================
print("\n" + "=" * 80)
print("STEP 12: SAVING RESULTS")
print("=" * 80)

# Create results dictionary
results = {
    'main_effect': model3.params['treat_post'],
    'main_se': model3.bse['treat_post'],
    'main_pvalue': model3.pvalues['treat_post'],
    'main_ci_low': model3.conf_int().loc['treat_post', 0],
    'main_ci_high': model3.conf_int().loc['treat_post', 1],
    'sample_size': len(df_did),
    'n_treated': df_did['treated'].sum(),
    'n_control': (1 - df_did['treated']).sum(),
    'model1_effect': model1.params['treat_post'],
    'model1_se': model1.bse['treat_post'],
    'model2_effect': model2.params['treat_post'],
    'model2_se': model2.bse['treat_post'],
    'employment_effect': model_emp.params['treat_post'],
    'employment_se': model_emp.bse['treat_post'],
    'men_effect': model_men.params['treat_post'],
    'men_se': model_men.bse['treat_post'],
    'women_effect': model_women.params['treat_post'],
    'women_se': model_women.bse['treat_post'],
    'narrow_effect': model_narrow.params['treat_post'],
    'narrow_se': model_narrow.bse['treat_post'],
    'placebo_effect': model_placebo.params['placebo_treat_post'],
    'placebo_se': model_placebo.bse['placebo_treat_post'],
    'placebo_pvalue': model_placebo.pvalues['placebo_treat_post'],
    'pre_treat_ft': pre_treat,
    'post_treat_ft': post_treat,
    'pre_control_ft': pre_control,
    'post_control_ft': post_control,
    'manual_did': did_effect
}

# Event study results
for er in event_results:
    results[f'event_{er["year"]}_coef'] = er['coef']
    results[f'event_{er["year"]}_se'] = er['se']

# Save to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save balance table
balance_df = pd.DataFrame(balance_results)
balance_df.to_csv('balance_table.csv', index=False)
print("Balance table saved to balance_table.csv")

# Save event study results
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study.csv', index=False)
print("Event study results saved to event_study.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

print(f"\nPreferred Estimate (Model 3 with State and Year FE):")
print(f"  DiD Effect on Full-time Employment: {results['main_effect']:.5f}")
print(f"  Standard Error: {results['main_se']:.5f}")
print(f"  95% CI: [{results['main_ci_low']:.5f}, {results['main_ci_high']:.5f}]")
print(f"  p-value: {results['main_pvalue']:.4f}")
print(f"\n  Sample Size: {results['sample_size']:,}")
print(f"  Treatment Group: {int(results['n_treated']):,}")
print(f"  Control Group: {int(results['n_control']):,}")

# Interpretation
if results['main_effect'] > 0:
    direction = "increase"
else:
    direction = "decrease"

print(f"\n\nINTERPRETATION:")
print(f"DACA eligibility is associated with a {abs(results['main_effect'])*100:.2f} percentage point")
print(f"{direction} in full-time employment probability among Hispanic-Mexican,")
print(f"Mexican-born non-citizens.")
if results['main_pvalue'] < 0.05:
    print(f"This effect is statistically significant at the 5% level (p={results['main_pvalue']:.4f}).")
elif results['main_pvalue'] < 0.1:
    print(f"This effect is statistically significant at the 10% level (p={results['main_pvalue']:.4f}).")
else:
    print(f"This effect is NOT statistically significant at conventional levels (p={results['main_pvalue']:.4f}).")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
