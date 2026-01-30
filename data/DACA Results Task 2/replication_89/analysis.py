"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States.

Treatment Group: Ages 26-30 as of June 15, 2012 (born 1982-1986)
Control Group: Ages 31-35 as of June 15, 2012 (born 1977-1981)
Outcome: Full-time employment (UHRSWORK >= 35 hours per week)
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD DATA IN CHUNKS AND FILTER
# =============================================================================
print("\n1. LOADING AND FILTERING DATA IN CHUNKS...")
data_path = "data/data.csv"

# Columns we need
needed_cols = ['YEAR', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EMPSTAT',
               'LABFORCE', 'UHRSWORK', 'EDUC', 'MARST', 'PERWT']

# Process in chunks
chunk_size = 500000
filtered_chunks = []

for i, chunk in enumerate(pd.read_csv(data_path, usecols=needed_cols, chunksize=chunk_size)):
    # Filter to Hispanic-Mexican, born in Mexico, non-citizen
    chunk_filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Hispanic-Mexican
        (chunk['BPL'] == 200) &   # Born in Mexico
        (chunk['CITIZEN'] == 3)   # Non-citizen
    ].copy()

    if len(chunk_filtered) > 0:
        # Calculate age at immigration and filter
        chunk_filtered['age_at_immig'] = chunk_filtered['YRIMMIG'] - chunk_filtered['BIRTHYR']
        chunk_filtered = chunk_filtered[
            (chunk_filtered['age_at_immig'] < 16) &  # Arrived before 16
            (chunk_filtered['YRIMMIG'] <= 2007) &    # Immigrated by 2007
            (chunk_filtered['BIRTHYR'] >= 1977) &    # Birth year range
            (chunk_filtered['BIRTHYR'] <= 1986) &
            (chunk_filtered['YEAR'] != 2012)         # Exclude 2012
        ]

        if len(chunk_filtered) > 0:
            filtered_chunks.append(chunk_filtered)

    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

    # Clear memory
    del chunk
    gc.collect()

# Combine filtered chunks
print(f"\nCombining filtered data...")
df_sample = pd.concat(filtered_chunks, ignore_index=True)
del filtered_chunks
gc.collect()

print(f"Filtered sample size: {len(df_sample):,}")
print(f"Years in sample: {sorted(df_sample['YEAR'].unique())}")

# =============================================================================
# 2. CREATE ANALYSIS VARIABLES
# =============================================================================
print("\n2. CREATING ANALYSIS VARIABLES...")

# Treatment indicator: born 1982-1986 (ages 26-30 as of June 2012)
df_sample['treated'] = (df_sample['BIRTHYR'] >= 1982).astype(int)
print(f"Treatment group (born 1982-1986): {df_sample['treated'].sum():,}")
print(f"Control group (born 1977-1981): {(df_sample['treated']==0).sum():,}")

# Post-treatment indicator
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
print(f"Pre-period observations (2006-2011): {(df_sample['post']==0).sum():,}")
print(f"Post-period observations (2013-2016): {(df_sample['post']==1).sum():,}")

# Outcome: Full-time employment (35+ hours per week)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# Covariates
df_sample['male'] = (df_sample['SEX'] == 1).astype(int)
df_sample['age'] = df_sample['YEAR'] - df_sample['BIRTHYR']
df_sample['married'] = df_sample['MARST'].isin([1, 2]).astype(int)
df_sample['yrs_in_us'] = df_sample['YEAR'] - df_sample['YRIMMIG']

# Education categories
df_sample['less_than_hs'] = (df_sample['EDUC'] < 6).astype(int)
df_sample['hs_grad'] = (df_sample['EDUC'].isin([6, 7, 8, 9])).astype(int)
df_sample['some_college'] = (df_sample['EDUC'].isin([10, 11])).astype(int)
df_sample['college_plus'] = (df_sample['EDUC'] >= 12).astype(int)

# Interaction
df_sample['treated_post'] = df_sample['treated'] * df_sample['post']

print(f"\nFull-time employment rate (overall): {df_sample['fulltime'].mean():.4f}")
print(f"Employment rate (overall): {df_sample['employed'].mean():.4f}")
print(f"Male proportion: {df_sample['male'].mean():.4f}")
print(f"Married proportion: {df_sample['married'].mean():.4f}")
print(f"Mean age: {df_sample['age'].mean():.2f}")

# =============================================================================
# 3. SUMMARY STATISTICS
# =============================================================================
print("\n3. SUMMARY STATISTICS...")

# By treatment status and period
summary_groups = df_sample.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'male': 'mean',
    'age': 'mean',
    'married': 'mean',
    'yrs_in_us': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary by Treatment Status and Period:")
print(summary_groups)

# Raw difference-in-differences
pre_treat = df_sample[(df_sample['treated']==1) & (df_sample['post']==0)]['fulltime'].mean()
post_treat = df_sample[(df_sample['treated']==1) & (df_sample['post']==1)]['fulltime'].mean()
pre_ctrl = df_sample[(df_sample['treated']==0) & (df_sample['post']==0)]['fulltime'].mean()
post_ctrl = df_sample[(df_sample['treated']==0) & (df_sample['post']==1)]['fulltime'].mean()

raw_did = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)
print(f"\nRaw Difference-in-Differences:")
print(f"Treatment pre:  {pre_treat:.4f}")
print(f"Treatment post: {post_treat:.4f}")
print(f"Treatment change: {post_treat - pre_treat:.4f}")
print(f"Control pre:    {pre_ctrl:.4f}")
print(f"Control post:   {post_ctrl:.4f}")
print(f"Control change: {post_ctrl - pre_ctrl:.4f}")
print(f"Raw DiD:        {raw_did:.4f}")

# =============================================================================
# 4. DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n4. DIFFERENCE-IN-DIFFERENCES REGRESSION...")

# Model 1: Basic DiD without covariates
print("\n--- Model 1: Basic DiD (no covariates) ---")
model1 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model1.params['treated_post']:.4f}")
print(f"Std Error: {model1.bse['treated_post']:.4f}")
print(f"P-value: {model1.pvalues['treated_post']:.4f}")
print(f"R-squared: {model1.rsquared:.4f}")

# Model 2: DiD with demographic covariates
print("\n--- Model 2: DiD with demographic covariates ---")
model2 = smf.wls('fulltime ~ treated + post + treated_post + male + married + age + I(age**2)',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model2.params['treated_post']:.4f}")
print(f"Std Error: {model2.bse['treated_post']:.4f}")
print(f"P-value: {model2.pvalues['treated_post']:.4f}")
print(f"R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with full covariates
print("\n--- Model 3: DiD with full covariates ---")
model3 = smf.wls('fulltime ~ treated + post + treated_post + male + married + age + I(age**2) + hs_grad + some_college + college_plus + yrs_in_us',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model3.params['treated_post']:.4f}")
print(f"Std Error: {model3.bse['treated_post']:.4f}")
print(f"P-value: {model3.pvalues['treated_post']:.4f}")
print(f"R-squared: {model3.rsquared:.4f}")

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with year fixed effects ---")
df_sample['year_factor'] = df_sample['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ treated + C(year_factor) + treated_post + male + married + age + I(age**2) + hs_grad + some_college + college_plus + yrs_in_us',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model4.params['treated_post']:.4f}")
print(f"Std Error: {model4.bse['treated_post']:.4f}")
print(f"P-value: {model4.pvalues['treated_post']:.4f}")
print(f"R-squared: {model4.rsquared:.4f}")

# Model 5: DiD with state and year fixed effects
print("\n--- Model 5: DiD with state and year fixed effects ---")
df_sample['state_factor'] = df_sample['STATEFIP'].astype(str)
model5 = smf.wls('fulltime ~ treated + C(year_factor) + C(state_factor) + treated_post + male + married + age + I(age**2) + hs_grad + some_college + college_plus + yrs_in_us',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model5.params['treated_post']:.4f}")
print(f"Std Error: {model5.bse['treated_post']:.4f}")
print(f"P-value: {model5.pvalues['treated_post']:.4f}")
print(f"R-squared: {model5.rsquared:.4f}")

# =============================================================================
# 5. ROBUSTNESS CHECKS
# =============================================================================
print("\n5. ROBUSTNESS CHECKS...")

# By gender
print("\n--- By Gender ---")
df_male = df_sample[df_sample['male']==1]
df_female = df_sample[df_sample['male']==0]

model_male = smf.wls('fulltime ~ treated + post + treated_post + married + age + I(age**2)',
                      data=df_male,
                      weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ treated + post + treated_post + married + age + I(age**2)',
                        data=df_female,
                        weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"Males - DiD: {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f})")
print(f"Females - DiD: {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f})")

# Employment as alternative outcome
print("\n--- Employment (any) as outcome ---")
model_emp = smf.wls('employed ~ treated + post + treated_post + male + married + age + I(age**2)',
                     data=df_sample,
                     weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"Employment DiD: {model_emp.params['treated_post']:.4f} (SE: {model_emp.bse['treated_post']:.4f})")

# Narrower age bands
print("\n--- Narrower age bands (28-30 vs 31-33) ---")
df_narrow = df_sample[(df_sample['BIRTHYR'] >= 1979) & (df_sample['BIRTHYR'] <= 1984)].copy()
df_narrow['treated_narrow'] = (df_narrow['BIRTHYR'] >= 1982).astype(int)
df_narrow['treated_post_narrow'] = df_narrow['treated_narrow'] * df_narrow['post']
model_narrow = smf.wls('fulltime ~ treated_narrow + post + treated_post_narrow + male + married + age + I(age**2)',
                        data=df_narrow,
                        weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"Narrow bands DiD: {model_narrow.params['treated_post_narrow']:.4f} (SE: {model_narrow.bse['treated_post_narrow']:.4f})")

# =============================================================================
# 6. EVENT STUDY
# =============================================================================
print("\n6. EVENT STUDY ANALYSIS...")

# Create year dummies and interactions
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_sample[f'year_{year}'] = (df_sample['YEAR'] == year).astype(int)
    df_sample[f'treat_{year}'] = df_sample['treated'] * df_sample[f'year_{year}']

# Event study regression (2011 is reference)
model_event = smf.wls('''fulltime ~ treated + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 +
                        year_2013 + year_2014 + year_2015 + year_2016 +
                        treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 +
                        treat_2013 + treat_2014 + treat_2015 + treat_2016 +
                        male + married + age + I(age**2)''',
                       data=df_sample,
                       weights=df_sample['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
event_vars = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
              'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    print(f"{var}: {coef:.4f} (SE: {se:.4f})")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================
print("\n7. SAVING RESULTS...")

# Results summary table
results_summary = pd.DataFrame({
    'Model': ['Basic DiD', 'With Demographics', 'Full Covariates', 'Year FE', 'State+Year FE'],
    'DiD_Coefficient': [model1.params['treated_post'], model2.params['treated_post'],
                        model3.params['treated_post'], model4.params['treated_post'],
                        model5.params['treated_post']],
    'Std_Error': [model1.bse['treated_post'], model2.bse['treated_post'],
                  model3.bse['treated_post'], model4.bse['treated_post'],
                  model5.bse['treated_post']],
    'P_value': [model1.pvalues['treated_post'], model2.pvalues['treated_post'],
                model3.pvalues['treated_post'], model4.pvalues['treated_post'],
                model5.pvalues['treated_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)],
    'R_squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared, model5.rsquared]
})
results_summary.to_csv('results_summary.csv', index=False)

# Yearly means
yearly_means = df_sample.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
yearly_means.columns = ['Control', 'Treatment']
yearly_means.to_csv('yearly_fulltime_means.csv')

# Event study results
event_results = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [model_event.params['treat_2006'], model_event.params['treat_2007'],
                   model_event.params['treat_2008'], model_event.params['treat_2009'],
                   model_event.params['treat_2010'], 0,
                   model_event.params['treat_2013'], model_event.params['treat_2014'],
                   model_event.params['treat_2015'], model_event.params['treat_2016']],
    'SE': [model_event.bse['treat_2006'], model_event.bse['treat_2007'],
           model_event.bse['treat_2008'], model_event.bse['treat_2009'],
           model_event.bse['treat_2010'], 0,
           model_event.bse['treat_2013'], model_event.bse['treat_2014'],
           model_event.bse['treat_2015'], model_event.bse['treat_2016']]
})
event_results['CI_low'] = event_results['Coefficient'] - 1.96 * event_results['SE']
event_results['CI_high'] = event_results['Coefficient'] + 1.96 * event_results['SE']
event_results.to_csv('event_study_results.csv', index=False)

# Summary statistics by group
summary_stats = df_sample.groupby('treated').agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'male': 'mean',
    'age': 'mean',
    'married': 'mean',
    'yrs_in_us': 'mean',
    'less_than_hs': 'mean',
    'hs_grad': 'mean',
    'some_college': 'mean',
    'college_plus': 'mean',
    'YEAR': 'count'
}).rename(columns={'YEAR': 'N'})
summary_stats.index = ['Control', 'Treatment']
summary_stats.to_csv('summary_statistics.csv')

# Preferred estimate
preferred_coef = model5.params['treated_post']
preferred_se = model5.bse['treated_post']
preferred_ci_low = preferred_coef - 1.96 * preferred_se
preferred_ci_high = preferred_coef + 1.96 * preferred_se

print("\n" + "="*80)
print("PREFERRED ESTIMATE (Model 5: State + Year Fixed Effects)")
print("="*80)
print(f"Effect Size: {preferred_coef:.4f}")
print(f"Standard Error: {preferred_se:.4f}")
print(f"95% CI: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"P-value: {model5.pvalues['treated_post']:.4f}")
print(f"Sample Size: {int(model5.nobs):,}")
print("="*80)

# Save detailed output
with open('analysis_output.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - DETAILED OUTPUT\n")
    f.write("="*80 + "\n\n")

    f.write("SAMPLE CONSTRUCTION\n")
    f.write("-"*40 + "\n")
    f.write(f"Final analysis sample: {len(df_sample):,}\n")
    f.write(f"Treatment group: {df_sample['treated'].sum():,}\n")
    f.write(f"Control group: {(df_sample['treated']==0).sum():,}\n\n")

    f.write("RAW DIFFERENCE-IN-DIFFERENCES\n")
    f.write("-"*40 + "\n")
    f.write(f"Treatment pre:  {pre_treat:.4f}\n")
    f.write(f"Treatment post: {post_treat:.4f}\n")
    f.write(f"Control pre:    {pre_ctrl:.4f}\n")
    f.write(f"Control post:   {post_ctrl:.4f}\n")
    f.write(f"Raw DiD:        {raw_did:.4f}\n\n")

    f.write("MODEL RESULTS\n")
    f.write("-"*40 + "\n")
    f.write(results_summary.to_string(index=False))
    f.write("\n\n")

    f.write("PREFERRED ESTIMATE\n")
    f.write("-"*40 + "\n")
    f.write(f"DiD Coefficient: {preferred_coef:.4f}\n")
    f.write(f"Standard Error:  {preferred_se:.4f}\n")
    f.write(f"95% CI:          [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]\n")
    f.write(f"P-value:         {model5.pvalues['treated_post']:.4f}\n")
    f.write(f"Sample Size:     {int(model5.nobs):,}\n\n")

    f.write("ROBUSTNESS CHECKS\n")
    f.write("-"*40 + "\n")
    f.write(f"Males: {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f})\n")
    f.write(f"Females: {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f})\n")
    f.write(f"Employment: {model_emp.params['treated_post']:.4f} (SE: {model_emp.bse['treated_post']:.4f})\n")
    f.write(f"Narrow bands: {model_narrow.params['treated_post_narrow']:.4f} (SE: {model_narrow.bse['treated_post_narrow']:.4f})\n\n")

    f.write("EVENT STUDY\n")
    f.write("-"*40 + "\n")
    f.write(event_results.to_string(index=False))

print("\nAll results saved successfully!")
print("\nAnalysis complete.")
