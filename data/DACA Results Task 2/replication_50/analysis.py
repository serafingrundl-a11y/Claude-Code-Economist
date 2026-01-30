"""
DACA Replication Study - Difference-in-Differences Analysis
Research Question: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# File paths
DATA_PATH = "data/data.csv"
OUTPUT_PATH = "."

print("=" * 60)
print("DACA Replication Study - Analysis")
print("=" * 60)

# Define relevant columns to load
cols_to_use = [
    'YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR', 'MARST',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK', 'STATEFIP'
]

print("\nStep 1: Loading and filtering data...")
print("Reading data in chunks due to large file size...")

# Process in chunks to handle large file
chunk_size = 500000
chunks = []
total_rows = 0

for chunk in pd.read_csv(DATA_PATH, usecols=cols_to_use, chunksize=chunk_size):
    total_rows += len(chunk)

    # Initial filter: Hispanic-Mexican and born in Mexico
    # HISPAN == 1 means Mexican
    # BPL == 200 means Mexico
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]

    if len(filtered) > 0:
        chunks.append(filtered)

    if total_rows % 5000000 == 0:
        print(f"  Processed {total_rows:,} rows...")

print(f"  Total rows in original data: {total_rows:,}")

# Combine all filtered chunks
df = pd.concat(chunks, ignore_index=True)
print(f"  Rows after Hispanic-Mexican & Mexico-born filter: {len(df):,}")

# Save intermediate count
n_hispanic_mexican = len(df)

print("\nStep 2: Applying DACA eligibility criteria...")

# Filter for non-citizens (CITIZEN == 3 means "Not a citizen")
# Per instructions: assume anyone who is not a citizen and has not received
# immigration papers is undocumented
df = df[df['CITIZEN'] == 3]
print(f"  Rows after non-citizen filter: {len(df):,}")
n_non_citizen = len(df)

# Filter for those who immigrated
df = df[df['YRIMMIG'] > 0]  # 0 means N/A
print(f"  Rows after valid immigration year filter: {len(df):,}")

# Calculate age at arrival
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA requirement: arrived before 16th birthday
df = df[df['age_at_arrival'] < 16]
print(f"  Rows after 'arrived before age 16' filter: {len(df):,}")
n_arrived_before_16 = len(df)

# DACA requirement: lived continuously in US since June 15, 2007
# This means they must have arrived by 2007 at the latest
df = df[df['YRIMMIG'] <= 2007]
print(f"  Rows after 'in US since 2007' filter: {len(df):,}")
n_in_us_since_2007 = len(df)

# Calculate age as of June 15, 2012 for treatment/control assignment
# Age on June 15, 2012 depends on birth quarter
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born Q1-Q2, they've already had birthday by June 15
# If born Q3-Q4, they haven't had birthday yet by June 15

df['age_june_2012'] = 2012 - df['BIRTHYR']
# Adjust for those who haven't had birthday yet by June 15
# Q3 (Jul-Sep) and Q4 (Oct-Dec) births haven't had birthday
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june_2012'] -= 1

# Treatment group: ages 26-30 as of June 15, 2012
# Control group: ages 31-35 as of June 15, 2012
df['treat'] = ((df['age_june_2012'] >= 26) & (df['age_june_2012'] <= 30)).astype(int)
df['control'] = ((df['age_june_2012'] >= 31) & (df['age_june_2012'] <= 35)).astype(int)

# Keep only treatment or control groups
df = df[(df['treat'] == 1) | (df['control'] == 1)]
print(f"  Rows after age group filter (26-35 as of June 2012): {len(df):,}")
n_age_filtered = len(df)

# Exclude 2012 (DACA implemented mid-year, cannot distinguish pre/post)
df = df[df['YEAR'] != 2012]
print(f"  Rows after excluding 2012: {len(df):,}")

# Define post-treatment period (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Define outcome: full-time employment (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Create interaction term
df['treat_post'] = df['treat'] * df['post']

print(f"\nFinal analysis sample: {len(df):,} observations")

# Summary statistics
print("\n" + "=" * 60)
print("Step 3: Summary Statistics")
print("=" * 60)

# Sample by group and period
print("\nSample sizes by group and period:")
print(df.groupby(['treat', 'post']).agg({
    'PERWT': ['count', 'sum'],
    'fulltime': 'mean'
}).round(4))

# Detailed summary
summary_stats = df.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std'],
    'UHRSWORK': ['mean', 'std'],
    'AGE': 'mean',
    'SEX': lambda x: (x == 2).mean(),  # Female proportion
    'PERWT': 'sum'
}).round(4)
print("\nDetailed summary by group and period:")
print(summary_stats)

# Calculate raw difference-in-differences
print("\n" + "=" * 60)
print("Step 4: Raw Difference-in-Differences")
print("=" * 60)

# Mean full-time employment by group and period
means = df.groupby(['treat', 'post'])['fulltime'].apply(
    lambda x: np.average(x, weights=df.loc[x.index, 'PERWT'])
)
print("\nWeighted mean full-time employment rates:")
print(means.round(4))

# Treatment group changes
treat_pre = means[(1, 0)]
treat_post = means[(1, 1)]
control_pre = means[(0, 0)]
control_post = means[(0, 1)]

treat_diff = treat_post - treat_pre
control_diff = control_post - control_pre
raw_did = treat_diff - control_diff

print(f"\nTreatment group (26-30):")
print(f"  Pre-DACA: {treat_pre:.4f}")
print(f"  Post-DACA: {treat_post:.4f}")
print(f"  Difference: {treat_diff:.4f}")

print(f"\nControl group (31-35):")
print(f"  Pre-DACA: {control_pre:.4f}")
print(f"  Post-DACA: {control_post:.4f}")
print(f"  Difference: {control_diff:.4f}")

print(f"\nRaw Difference-in-Differences: {raw_did:.4f}")

# Regression Analysis
print("\n" + "=" * 60)
print("Step 5: Regression Analysis")
print("=" * 60)

# Model 1: Basic DiD (no controls)
print("\nModel 1: Basic Difference-in-Differences")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df, weights=df['PERWT'])
results1 = model1.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(results1.summary().tables[1])

# Store key results
did_coef1 = results1.params['treat_post']
did_se1 = results1.bse['treat_post']
did_pval1 = results1.pvalues['treat_post']
did_ci1 = results1.conf_int().loc['treat_post']

print(f"\nDiD Coefficient: {did_coef1:.4f}")
print(f"Standard Error: {did_se1:.4f}")
print(f"95% CI: [{did_ci1[0]:.4f}, {did_ci1[1]:.4f}]")
print(f"P-value: {did_pval1:.4f}")

# Model 2: DiD with demographic controls
print("\n" + "-" * 40)
print("Model 2: DiD with Demographic Controls")

# Create control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] == 1).astype(int)

# Education categories
df['educ_hs'] = (df['EDUCD'] >= 62).astype(int)  # High school or more
df['educ_somecoll'] = (df['EDUCD'] >= 65).astype(int)  # Some college or more
df['educ_ba'] = (df['EDUCD'] >= 101).astype(int)  # Bachelor's or more

model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + educ_somecoll',
                  data=df, weights=df['PERWT'])
results2 = model2.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(results2.summary().tables[1])

did_coef2 = results2.params['treat_post']
did_se2 = results2.bse['treat_post']
did_pval2 = results2.pvalues['treat_post']
did_ci2 = results2.conf_int().loc['treat_post']

print(f"\nDiD Coefficient: {did_coef2:.4f}")
print(f"Standard Error: {did_se2:.4f}")
print(f"95% CI: [{did_ci2[0]:.4f}, {did_ci2[1]:.4f}]")
print(f"P-value: {did_pval2:.4f}")

# Model 3: DiD with year fixed effects
print("\n" + "-" * 40)
print("Model 3: DiD with Year Fixed Effects")

# Create year dummies
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
df_with_years = pd.concat([df, year_dummies], axis=1)

year_cols = [col for col in year_dummies.columns]
formula3 = 'fulltime ~ treat + treat_post + female + married + educ_hs + educ_somecoll + ' + ' + '.join(year_cols)

model3 = smf.wls(formula3, data=df_with_years, weights=df_with_years['PERWT'])
results3 = model3.fit(cov_type='cluster', cov_kwds={'groups': df_with_years['STATEFIP']})
print(results3.summary().tables[1])

did_coef3 = results3.params['treat_post']
did_se3 = results3.bse['treat_post']
did_pval3 = results3.pvalues['treat_post']
did_ci3 = results3.conf_int().loc['treat_post']

print(f"\nDiD Coefficient: {did_coef3:.4f}")
print(f"Standard Error: {did_se3:.4f}")
print(f"95% CI: [{did_ci3[0]:.4f}, {did_ci3[1]:.4f}]")
print(f"P-value: {did_pval3:.4f}")

# Model 4: DiD with state fixed effects
print("\n" + "-" * 40)
print("Model 4: DiD with State Fixed Effects")

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
df_with_state = pd.concat([df, state_dummies, year_dummies], axis=1)

state_cols = [col for col in state_dummies.columns]
formula4 = 'fulltime ~ treat + treat_post + female + married + educ_hs + educ_somecoll + ' + \
           ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)

model4 = smf.wls(formula4, data=df_with_state, weights=df_with_state['PERWT'])
results4 = model4.fit(cov_type='cluster', cov_kwds={'groups': df_with_state['STATEFIP']})

# Print only main coefficients
main_vars = ['Intercept', 'treat', 'treat_post', 'female', 'married', 'educ_hs', 'educ_somecoll']
print("\nMain Coefficients (State and Year FE included but not shown):")
for var in main_vars:
    if var in results4.params:
        print(f"  {var:20s}: {results4.params[var]:8.4f} (SE: {results4.bse[var]:.4f})")

did_coef4 = results4.params['treat_post']
did_se4 = results4.bse['treat_post']
did_pval4 = results4.pvalues['treat_post']
did_ci4 = results4.conf_int().loc['treat_post']

print(f"\nDiD Coefficient: {did_coef4:.4f}")
print(f"Standard Error: {did_se4:.4f}")
print(f"95% CI: [{did_ci4[0]:.4f}, {did_ci4[1]:.4f}]")
print(f"P-value: {did_pval4:.4f}")

# Summary of results
print("\n" + "=" * 60)
print("Step 6: Summary of Results")
print("=" * 60)

print("\nDifference-in-Differences Estimates Across Specifications:")
print("-" * 70)
print(f"{'Model':<35} {'Coef':>10} {'SE':>10} {'95% CI':>20}")
print("-" * 70)
print(f"{'1. Basic DiD':<35} {did_coef1:>10.4f} {did_se1:>10.4f} [{did_ci1[0]:.4f}, {did_ci1[1]:.4f}]")
print(f"{'2. + Demographics':<35} {did_coef2:>10.4f} {did_se2:>10.4f} [{did_ci2[0]:.4f}, {did_ci2[1]:.4f}]")
print(f"{'3. + Year FE':<35} {did_coef3:>10.4f} {did_se3:>10.4f} [{did_ci3[0]:.4f}, {did_ci3[1]:.4f}]")
print(f"{'4. + State FE (Preferred)':<35} {did_coef4:>10.4f} {did_se4:>10.4f} [{did_ci4[0]:.4f}, {did_ci4[1]:.4f}]")
print("-" * 70)

# Pre-trend analysis
print("\n" + "=" * 60)
print("Step 7: Pre-Trend Analysis")
print("=" * 60)

# Calculate trends in pre-period
pre_data = df[df['post'] == 0]
pre_means = pre_data.groupby(['YEAR', 'treat']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
pre_means.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\nPre-treatment full-time employment rates by year:")
print(pre_means.round(4))

# Calculate year-by-year differences
pre_means['Difference'] = pre_means['Treatment (26-30)'] - pre_means['Control (31-35)']
print("\nDifference (Treatment - Control) by pre-treatment year:")
print(pre_means['Difference'].round(4))

# Event study analysis
print("\n" + "=" * 60)
print("Step 8: Event Study Analysis")
print("=" * 60)

# Create year interactions with treatment
df['year_relative'] = df['YEAR'] - 2012  # Relative to DACA implementation

# Exclude 2012 (already done) and use 2011 as reference year
event_study_data = df.copy()

# Create year x treatment interactions
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    event_study_data[f'treat_year_{year}'] = (
        (event_study_data['YEAR'] == year) & (event_study_data['treat'] == 1)
    ).astype(int)

# Create year dummies
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    event_study_data[f'year_{year}'] = (event_study_data['YEAR'] == year).astype(int)

# Event study regression
year_terms = ' + '.join([f'year_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])
treat_year_terms = ' + '.join([f'treat_year_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])

formula_es = f'fulltime ~ treat + {year_terms} + {treat_year_terms} + female + married + educ_hs + educ_somecoll'

model_es = smf.wls(formula_es, data=event_study_data, weights=event_study_data['PERWT'])
results_es = model_es.fit(cov_type='cluster', cov_kwds={'groups': event_study_data['STATEFIP']})

print("\nEvent Study Coefficients (Reference year: 2011):")
print("-" * 50)
print(f"{'Year':<10} {'Coef':>10} {'SE':>10} {'P-value':>10}")
print("-" * 50)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_year_{year}'
    coef = results_es.params[var]
    se = results_es.bse[var]
    pval = results_es.pvalues[var]
    print(f"{year:<10} {coef:>10.4f} {se:>10.4f} {pval:>10.4f}")
print("-" * 50)

# Robustness checks
print("\n" + "=" * 60)
print("Step 9: Robustness Checks")
print("=" * 60)

# Robustness 1: Narrower age bands (27-29 vs 32-34)
print("\nRobustness 1: Narrower age bands (27-29 vs 32-34)")
df_narrow = df.copy()
df_narrow['treat_narrow'] = ((df_narrow['age_june_2012'] >= 27) & (df_narrow['age_june_2012'] <= 29)).astype(int)
df_narrow['control_narrow'] = ((df_narrow['age_june_2012'] >= 32) & (df_narrow['age_june_2012'] <= 34)).astype(int)
df_narrow = df_narrow[(df_narrow['treat_narrow'] == 1) | (df_narrow['control_narrow'] == 1)]
df_narrow['treat_post_narrow'] = df_narrow['treat_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treat_narrow + post + treat_post_narrow + female + married + educ_hs',
                        data=df_narrow, weights=df_narrow['PERWT'])
results_narrow = model_narrow.fit(cov_type='cluster', cov_kwds={'groups': df_narrow['STATEFIP']})
print(f"DiD Coefficient: {results_narrow.params['treat_post_narrow']:.4f} (SE: {results_narrow.bse['treat_post_narrow']:.4f})")
print(f"Sample size: {len(df_narrow):,}")

# Robustness 2: By gender
print("\nRobustness 2: Heterogeneity by Gender")

# Males
df_male = df[df['SEX'] == 1]
model_male = smf.wls('fulltime ~ treat + post + treat_post + married + educ_hs + educ_somecoll',
                      data=df_male, weights=df_male['PERWT'])
results_male = model_male.fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
print(f"Males - DiD: {results_male.params['treat_post']:.4f} (SE: {results_male.bse['treat_post']:.4f}), N={len(df_male):,}")

# Females
df_female = df[df['SEX'] == 2]
model_female = smf.wls('fulltime ~ treat + post + treat_post + married + educ_hs + educ_somecoll',
                        data=df_female, weights=df_female['PERWT'])
results_female = model_female.fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})
print(f"Females - DiD: {results_female.params['treat_post']:.4f} (SE: {results_female.bse['treat_post']:.4f}), N={len(df_female):,}")

# Robustness 3: Alternative outcome - any employment
print("\nRobustness 3: Alternative outcome - Any Employment")
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
model_emp = smf.wls('employed ~ treat + post + treat_post + female + married + educ_hs + educ_somecoll',
                     data=df, weights=df['PERWT'])
results_emp = model_emp.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {results_emp.params['treat_post']:.4f} (SE: {results_emp.bse['treat_post']:.4f})")

# Robustness 4: Donut analysis (exclude those very close to cutoff)
print("\nRobustness 4: Donut analysis (exclude ages 30-31)")
df_donut = df[(df['age_june_2012'] != 30) & (df['age_june_2012'] != 31)]
model_donut = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + educ_somecoll',
                       data=df_donut, weights=df_donut['PERWT'])
results_donut = model_donut.fit(cov_type='cluster', cov_kwds={'groups': df_donut['STATEFIP']})
print(f"DiD Coefficient: {results_donut.params['treat_post']:.4f} (SE: {results_donut.bse['treat_post']:.4f})")
print(f"Sample size: {len(df_donut):,}")

# Save results for report
print("\n" + "=" * 60)
print("Saving results for LaTeX report...")
print("=" * 60)

# Prepare results dataframe
results_df = pd.DataFrame({
    'Model': ['Basic DiD', '+ Demographics', '+ Year FE', '+ State & Year FE (Preferred)',
              'Narrow Age Bands', 'Males Only', 'Females Only', 'Any Employment', 'Donut (excl. 30-31)'],
    'Coefficient': [did_coef1, did_coef2, did_coef3, did_coef4,
                   results_narrow.params['treat_post_narrow'],
                   results_male.params['treat_post'],
                   results_female.params['treat_post'],
                   results_emp.params['treat_post'],
                   results_donut.params['treat_post']],
    'Std_Error': [did_se1, did_se2, did_se3, did_se4,
                  results_narrow.bse['treat_post_narrow'],
                  results_male.bse['treat_post'],
                  results_female.bse['treat_post'],
                  results_emp.bse['treat_post'],
                  results_donut.bse['treat_post']],
    'CI_Lower': [did_ci1[0], did_ci2[0], did_ci3[0], did_ci4[0],
                 results_narrow.conf_int().loc['treat_post_narrow'][0],
                 results_male.conf_int().loc['treat_post'][0],
                 results_female.conf_int().loc['treat_post'][0],
                 results_emp.conf_int().loc['treat_post'][0],
                 results_donut.conf_int().loc['treat_post'][0]],
    'CI_Upper': [did_ci1[1], did_ci2[1], did_ci3[1], did_ci4[1],
                 results_narrow.conf_int().loc['treat_post_narrow'][1],
                 results_male.conf_int().loc['treat_post'][1],
                 results_female.conf_int().loc['treat_post'][1],
                 results_emp.conf_int().loc['treat_post'][1],
                 results_donut.conf_int().loc['treat_post'][1]]
})

results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save event study results
es_results = []
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_year_{year}'
    es_results.append({
        'Year': year,
        'Relative_Year': year - 2012,
        'Coefficient': results_es.params[var],
        'Std_Error': results_es.bse[var],
        'CI_Lower': results_es.conf_int().loc[var][0],
        'CI_Upper': results_es.conf_int().loc[var][1],
        'P_Value': results_es.pvalues[var]
    })

es_df = pd.DataFrame(es_results)
es_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save descriptive statistics
desc_stats = df.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'UHRSWORK': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'PERWT': 'sum'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved to descriptive_stats.csv")

# Pre-trend data
pre_means.to_csv('pre_trends.csv')
print("Pre-trend data saved to pre_trends.csv")

# Final summary
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\nPreferred Specification (Model 4 with State and Year FE):")
print(f"  DiD Coefficient: {did_coef4:.4f}")
print(f"  Standard Error: {did_se4:.4f}")
print(f"  95% Confidence Interval: [{did_ci4[0]:.4f}, {did_ci4[1]:.4f}]")
print(f"  P-value: {did_pval4:.4f}")
print(f"\n  Interpretation: DACA eligibility is associated with a")
print(f"  {did_coef4*100:.2f} percentage point change in the probability")
print(f"  of full-time employment among eligible individuals.")
print(f"\n  Sample size: {len(df):,} observations")
print(f"  Weighted population: {df['PERWT'].sum():,.0f}")

print("\n" + "=" * 60)
print("Analysis Complete")
print("=" * 60)
