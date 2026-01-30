"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among Hispanic-Mexican
individuals born in Mexico.

Treatment Group: Ages 26-30 as of June 15, 2012
Control Group: Ages 31-35 as of June 15, 2012

Pre-period: 2006-2011
Post-period: 2013-2016 (excluding 2012 as ambiguous)

Full-time employment: UHRSWORK >= 35
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Read data in chunks due to large file size
print("Loading data...")
data_path = "data/data.csv"

# Define dtypes to reduce memory
dtypes = {
    'YEAR': 'int16',
    'SAMPLE': 'int32',
    'SERIAL': 'int32',
    'STATEFIP': 'int8',
    'PUMA': 'int32',
    'METRO': 'int8',
    'GQ': 'int8',
    'PERNUM': 'int16',
    'PERWT': 'float32',
    'FAMSIZE': 'int8',
    'NCHILD': 'int8',
    'RELATE': 'int8',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'BIRTHYR': 'int16',
    'RACE': 'int8',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'YRSUSA1': 'int8',
    'YRSUSA2': 'int8',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'EMPSTAT': 'int8',
    'EMPSTATD': 'int8',
    'LABFORCE': 'int8',
    'CLASSWKR': 'int8',
    'UHRSWORK': 'int8',
    'INCTOT': 'int32',
    'INCWAGE': 'int32',
    'POVERTY': 'int16'
}

# Columns to use
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
           'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EDUCD',
           'EMPSTAT', 'LABFORCE', 'UHRSWORK', 'NCHILD', 'FAMSIZE']

# Load data
df = pd.read_csv(data_path, usecols=usecols, dtype={k:v for k,v in dtypes.items() if k in usecols})
print(f"Total observations loaded: {len(df):,}")

# Step 1: Filter for Hispanic-Mexican ethnicity
# HISPAN = 1 (Mexican) according to data dictionary
df = df[df['HISPAN'] == 1]
print(f"After filtering for Hispanic-Mexican (HISPAN=1): {len(df):,}")

# Step 2: Filter for born in Mexico
# BPL = 200 (Mexico) according to data dictionary
df = df[df['BPL'] == 200]
print(f"After filtering for born in Mexico (BPL=200): {len(df):,}")

# Step 3: Filter for non-citizens
# CITIZEN = 3 (Not a citizen) - per instructions, assume non-citizens without papers are undocumented
# We exclude naturalized citizens (2) and those born abroad of American parents (1)
df = df[df['CITIZEN'] == 3]
print(f"After filtering for non-citizens (CITIZEN=3): {len(df):,}")

# Step 4: Calculate age as of June 15, 2012
# For simplicity, use birth year to calculate age
# Age as of 2012: 2012 - BIRTHYR
# For those born in Q1-Q2, they would have already had their birthday by June 15
# For those born in Q3-Q4, they would not yet have had their birthday by June 15
# However, to be conservative and match the age groups cleanly, we'll use the simpler approach

# Calculate age as of 2012 using birth year
df['age_2012'] = 2012 - df['BIRTHYR']

# Adjust for birth quarter: If born in Q3 or Q4, subtract 1 from age as of June 15
# Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec
# As of June 15, those born Jul-Dec haven't had their birthday yet
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_2012'] = df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_2012'] - 1

print(f"\nAge distribution as of June 15, 2012:")
print(df['age_2012'].describe())

# Step 5: Define treatment and control groups
# Treatment: ages 26-30 as of June 15, 2012
# Control: ages 31-35 as of June 15, 2012

# Filter to relevant age groups
df = df[(df['age_2012'] >= 26) & (df['age_2012'] <= 35)]
print(f"\nAfter filtering for ages 26-35: {len(df):,}")

# Define treatment indicator
df['treated'] = (df['age_2012'] <= 30).astype(int)

# Step 6: Additional DACA eligibility criteria
# - Arrived before 16th birthday
# - Continuous residence since June 15, 2007 (approximately 5 years in US by 2012)
# - We'll use YRIMMIG to check arrival before 16th birthday

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Filter: arrived before 16th birthday (DACA requirement)
df = df[df['age_at_immig'] < 16]
print(f"After filtering for arrived before age 16: {len(df):,}")

# Step 7: Continuous residence since June 15, 2007
# This is hard to verify in cross-sectional data
# We'll require immigration year <= 2007 as a proxy for this requirement
df = df[df['YRIMMIG'] <= 2007]
print(f"After filtering for YRIMMIG <= 2007: {len(df):,}")

# Step 8: Define pre and post periods
# Pre: 2006-2011
# Post: 2013-2016
# Exclude 2012 as ambiguous
df = df[df['YEAR'] != 2012]
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"\nAfter excluding 2012: {len(df):,}")

# Step 9: Define outcome variable
# Full-time employment: usually working 35+ hours per week
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Summary statistics
print("\n" + "="*60)
print("SAMPLE COMPOSITION")
print("="*60)
print(f"\nTotal observations in analysis sample: {len(df):,}")
print(f"\nBy treatment status:")
print(df.groupby('treated')['PERWT'].agg(['count', 'sum']))
print(f"\nBy period:")
print(df.groupby('post')['PERWT'].agg(['count', 'sum']))
print(f"\nBy treatment status and period:")
print(df.groupby(['treated', 'post'])['PERWT'].agg(['count', 'sum']))

# Step 10: Create year-by-year summary for figures
print("\n" + "="*60)
print("YEAR-BY-YEAR FULL-TIME EMPLOYMENT RATES (WEIGHTED)")
print("="*60)

yearly_stats = df.groupby(['YEAR', 'treated']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n_obs': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()

print("\nTreatment Group (Ages 26-30):")
print(yearly_stats[yearly_stats['treated'] == 1][['YEAR', 'fulltime_rate', 'n_obs', 'weighted_n']].to_string(index=False))

print("\nControl Group (Ages 31-35):")
print(yearly_stats[yearly_stats['treated'] == 0][['YEAR', 'fulltime_rate', 'n_obs', 'weighted_n']].to_string(index=False))

# Step 11: Basic DiD calculation (unweighted)
print("\n" + "="*60)
print("BASIC DIFFERENCE-IN-DIFFERENCES (UNWEIGHTED)")
print("="*60)

# Calculate means by group and period
means = df.groupby(['treated', 'post'])['fulltime'].mean().unstack()
print("\nMean full-time employment rates:")
print(means)

# DiD calculation
did_simple = (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])
print(f"\nDiD Estimate (unweighted): {did_simple:.4f}")

# Step 12: DiD regression analysis
print("\n" + "="*60)
print("REGRESSION ANALYSIS")
print("="*60)

# Create interaction term
df['treated_post'] = df['treated'] * df['post']

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ treated + post + treated_post', data=df, weights=df['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Controls ---")
# Add controls: sex, marital status, education, number of children
df['male'] = (df['SEX'] == 1).astype(int)
df['married'] = (df['MARST'] == 1).astype(int)  # Married spouse present
df['has_children'] = (df['NCHILD'] > 0).astype(int)

# Education categories
df['less_than_hs'] = (df['EDUC'] < 6).astype(int)
df['hs_degree'] = (df['EDUC'] == 6).astype(int)
df['some_college'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)
df['college_plus'] = (df['EDUC'] >= 10).astype(int)

model2 = smf.wls('fulltime ~ treated + post + treated_post + male + married + has_children + hs_degree + some_college + college_plus',
                  data=df, weights=df['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary())

# Model 3: DiD with state fixed effects
print("\n--- Model 3: DiD with State Fixed Effects ---")
# Add state dummies
df['state'] = df['STATEFIP'].astype(str)
model3 = smf.wls('fulltime ~ treated + post + treated_post + male + married + has_children + hs_degree + some_college + college_plus + C(state)',
                  data=df, weights=df['PERWT'])
results3 = model3.fit(cov_type='HC1')
# Print only key coefficients
print("\nKey coefficients from Model 3:")
print(f"treated_post (DiD estimate): {results3.params['treated_post']:.4f} (SE: {results3.bse['treated_post']:.4f})")
print(f"t-statistic: {results3.tvalues['treated_post']:.3f}")
print(f"p-value: {results3.pvalues['treated_post']:.4f}")
print(f"95% CI: [{results3.conf_int().loc['treated_post', 0]:.4f}, {results3.conf_int().loc['treated_post', 1]:.4f}]")

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with Year and State Fixed Effects ---")
df['year_fe'] = df['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ treated + treated_post + male + married + has_children + hs_degree + some_college + college_plus + C(state) + C(year_fe)',
                  data=df, weights=df['PERWT'])
results4 = model4.fit(cov_type='HC1')
print("\nKey coefficients from Model 4:")
print(f"treated_post (DiD estimate): {results4.params['treated_post']:.4f} (SE: {results4.bse['treated_post']:.4f})")
print(f"t-statistic: {results4.tvalues['treated_post']:.3f}")
print(f"p-value: {results4.pvalues['treated_post']:.4f}")
print(f"95% CI: [{results4.conf_int().loc['treated_post', 0]:.4f}, {results4.conf_int().loc['treated_post', 1]:.4f}]")

# Step 13: Pre-trend analysis
print("\n" + "="*60)
print("PRE-TREND ANALYSIS")
print("="*60)

# Event study: year-by-year effects
df_pre = df[df['post'] == 0].copy()
df_pre['year_2011'] = (df_pre['YEAR'] == 2011).astype(int)
df_pre['year_2010'] = (df_pre['YEAR'] == 2010).astype(int)
df_pre['year_2009'] = (df_pre['YEAR'] == 2009).astype(int)
df_pre['year_2008'] = (df_pre['YEAR'] == 2008).astype(int)
df_pre['year_2007'] = (df_pre['YEAR'] == 2007).astype(int)
# 2006 is reference

model_pre = smf.wls('fulltime ~ treated + treated:year_2007 + treated:year_2008 + treated:year_2009 + treated:year_2010 + treated:year_2011 + year_2007 + year_2008 + year_2009 + year_2010 + year_2011',
                     data=df_pre, weights=df_pre['PERWT'])
results_pre = model_pre.fit(cov_type='HC1')
print("\nPre-period interaction coefficients (relative to 2006):")
for var in ['treated:year_2007', 'treated:year_2008', 'treated:year_2009', 'treated:year_2010', 'treated:year_2011']:
    coef = results_pre.params[var]
    se = results_pre.bse[var]
    pval = results_pre.pvalues[var]
    print(f"{var}: {coef:.4f} (SE: {se:.4f}, p={pval:.3f})")

# Joint test of pre-trends
from scipy.stats import f as f_dist
pre_coefs = ['treated:year_2007', 'treated:year_2008', 'treated:year_2009', 'treated:year_2010', 'treated:year_2011']
r_matrix = np.zeros((len(pre_coefs), len(results_pre.params)))
for i, var in enumerate(pre_coefs):
    idx = list(results_pre.params.index).index(var)
    r_matrix[i, idx] = 1
f_test = results_pre.f_test(r_matrix)
fval = f_test.fvalue if isinstance(f_test.fvalue, float) else f_test.fvalue[0][0]
pval_f = f_test.pvalue if isinstance(f_test.pvalue, float) else f_test.pvalue[0][0]
print(f"\nJoint F-test of pre-trends: F={fval:.3f}, p-value={pval_f:.4f}")

# Step 14: Event study (full)
print("\n" + "="*60)
print("EVENT STUDY ANALYSIS")
print("="*60)

# Create year dummies and interactions
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'treated_year_{year}'] = df['treated'] * df[f'year_{year}']

# Reference year is 2011 (last pre-treatment year)
event_formula = 'fulltime ~ treated + ' + ' + '.join([f'treated_year_{y}' for y in years if y != 2011]) + ' + ' + ' + '.join([f'year_{y}' for y in years if y != 2011])
model_event = smf.wls(event_formula, data=df, weights=df['PERWT'])
results_event = model_event.fit(cov_type='HC1')

print("\nEvent study coefficients (relative to 2011):")
event_results = []
for year in years:
    if year == 2011:
        coef, se, pval = 0, 0, np.nan
    else:
        var = f'treated_year_{year}'
        coef = results_event.params[var]
        se = results_event.bse[var]
        pval = results_event.pvalues[var]
    event_results.append({'year': year, 'coefficient': coef, 'se': se, 'pvalue': pval})
    print(f"{year}: {coef:.4f} (SE: {se:.4f})")

event_df = pd.DataFrame(event_results)

# Step 15: Robustness checks
print("\n" + "="*60)
print("ROBUSTNESS CHECKS")
print("="*60)

# Alternative age windows
print("\n--- Robustness: Different Age Windows ---")

# Ages 25-30 vs 31-36
df_robust1 = df.copy()
df_robust1 = df_robust1[(df_robust1['age_2012'] >= 25) & (df_robust1['age_2012'] <= 36)]
df_robust1['treated'] = (df_robust1['age_2012'] <= 30).astype(int)
df_robust1['treated_post'] = df_robust1['treated'] * df_robust1['post']

model_robust1 = smf.wls('fulltime ~ treated + post + treated_post + male + married + has_children',
                         data=df_robust1, weights=df_robust1['PERWT'])
results_robust1 = model_robust1.fit(cov_type='HC1')
print(f"Ages 25-30 vs 31-36: DiD = {results_robust1.params['treated_post']:.4f} (SE: {results_robust1.bse['treated_post']:.4f})")

# Step 16: Heterogeneity analysis
print("\n" + "="*60)
print("HETEROGENEITY ANALYSIS")
print("="*60)

# By gender
print("\n--- By Gender ---")
df_male = df[df['male'] == 1]
df_female = df[df['male'] == 0]

model_male = smf.wls('fulltime ~ treated + post + treated_post', data=df_male, weights=df_male['PERWT'])
results_male = model_male.fit(cov_type='HC1')

model_female = smf.wls('fulltime ~ treated + post + treated_post', data=df_female, weights=df_female['PERWT'])
results_female = model_female.fit(cov_type='HC1')

print(f"Male: DiD = {results_male.params['treated_post']:.4f} (SE: {results_male.bse['treated_post']:.4f})")
print(f"Female: DiD = {results_female.params['treated_post']:.4f} (SE: {results_female.bse['treated_post']:.4f})")

# By education
print("\n--- By Education Level ---")
df_loweduc = df[df['EDUC'] < 6]
df_higheduc = df[df['EDUC'] >= 6]

model_loweduc = smf.wls('fulltime ~ treated + post + treated_post', data=df_loweduc, weights=df_loweduc['PERWT'])
results_loweduc = model_loweduc.fit(cov_type='HC1')

model_higheduc = smf.wls('fulltime ~ treated + post + treated_post', data=df_higheduc, weights=df_higheduc['PERWT'])
results_higheduc = model_higheduc.fit(cov_type='HC1')

print(f"Less than HS: DiD = {results_loweduc.params['treated_post']:.4f} (SE: {results_loweduc.bse['treated_post']:.4f})")
print(f"HS or more: DiD = {results_higheduc.params['treated_post']:.4f} (SE: {results_higheduc.bse['treated_post']:.4f})")

# Step 17: Save results for report
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

# Main estimate (Model 4 with full controls)
main_estimate = results4.params['treated_post']
main_se = results4.bse['treated_post']
main_ci_low = results4.conf_int().loc['treated_post', 0]
main_ci_high = results4.conf_int().loc['treated_post', 1]
main_pvalue = results4.pvalues['treated_post']
sample_size = len(df)

print(f"\n*** PREFERRED ESTIMATE ***")
print(f"Effect Size: {main_estimate:.4f}")
print(f"Standard Error: {main_se:.4f}")
print(f"95% Confidence Interval: [{main_ci_low:.4f}, {main_ci_high:.4f}]")
print(f"p-value: {main_pvalue:.4f}")
print(f"Sample Size: {sample_size:,}")

# Save yearly stats for plotting
yearly_stats.to_csv('yearly_employment_rates.csv', index=False)
event_df.to_csv('event_study_coefficients.csv', index=False)

# Save summary statistics
summary_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum',
    'male': 'mean',
    'married': 'mean',
    'AGE': 'mean',
    'EDUC': 'mean'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')

# Save regression results
results_summary = pd.DataFrame({
    'Model': ['Basic DiD', 'With Controls', 'State FE', 'Year + State FE'],
    'Estimate': [results1.params['treated_post'], results2.params['treated_post'],
                 results3.params['treated_post'], results4.params['treated_post']],
    'SE': [results1.bse['treated_post'], results2.bse['treated_post'],
           results3.bse['treated_post'], results4.bse['treated_post']],
    'p_value': [results1.pvalues['treated_post'], results2.pvalues['treated_post'],
                results3.pvalues['treated_post'], results4.pvalues['treated_post']],
    'N': [results1.nobs, results2.nobs, results3.nobs, results4.nobs]
})
results_summary.to_csv('regression_results.csv', index=False)

print("\nResults saved to CSV files.")
print("\nAnalysis complete!")
