"""
DACA Replication Analysis Script - Replication 76
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born non-citizens in the US

Author: [Anonymous for replication]
Date: 2026-01-25
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
print("DACA REPLICATION ANALYSIS - Replication 76")
print("="*80)

# =============================================================================
# STEP 1: LOAD DATA (CHUNKED READING DUE TO FILE SIZE)
# =============================================================================
print("\n[Step 1] Loading data in chunks...")

# Load only needed columns to reduce memory usage
cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
               'EMPSTAT', 'UHRSWORK', 'EDUC', 'MARST', 'STATEFIP', 'METRO']

# Read data in chunks and filter immediately to reduce memory
chunk_size = 1000000
chunks = []
total_raw = 0

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size,
                         dtype={'YEAR': 'int16', 'SEX': 'int8', 'AGE': 'int8',
                                'HISPAN': 'int8', 'BPL': 'int16', 'CITIZEN': 'int8',
                                'EMPSTAT': 'int8', 'UHRSWORK': 'int8', 'EDUC': 'int8',
                                'MARST': 'int8', 'METRO': 'int8', 'STATEFIP': 'int8'}):
    total_raw += len(chunk)
    # Apply filters immediately to reduce memory
    # Hispanic-Mexican (HISPAN == 1) AND Born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)
    del chunk
    gc.collect()

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"  Raw data scanned: {total_raw:,} observations")
print(f"  After Hispanic-Mexican + Mexico filter: {len(df):,} observations")

# =============================================================================
# STEP 2: SAMPLE RESTRICTIONS
# =============================================================================
print("\n[Step 2] Applying additional sample restrictions...")

# 2.1 Non-citizen (CITIZEN == 3) - proxy for undocumented
df_sample = df[df['CITIZEN'] == 3].copy()
print(f"  After non-citizen filter: {len(df_sample):,} observations")
del df
gc.collect()

# 2.2 Working-age population (16-64)
df_sample = df_sample[(df_sample['AGE'] >= 16) & (df_sample['AGE'] <= 64)]
print(f"  After age 16-64 filter: {len(df_sample):,} observations")

# 2.3 Valid year of immigration
df_sample = df_sample[(df_sample['YRIMMIG'] > 0) & (df_sample['YRIMMIG'] <= df_sample['YEAR'])]
print(f"  After valid immigration year filter: {len(df_sample):,} observations")

# =============================================================================
# STEP 3: CREATE ANALYSIS VARIABLES
# =============================================================================
print("\n[Step 3] Creating analysis variables...")

# 3.1 Full-time employment outcome (35+ hours per week)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype('int8')
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype('int8')

# 3.2 Age at immigration
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# 3.3 DACA eligibility criteria (as of June 15, 2012)
# Criterion 1: Arrived before 16th birthday
df_sample['arrived_before_16'] = (df_sample['age_at_immig'] < 16).astype('int8')

# Criterion 2: Not yet 31 as of June 15, 2012 (born after June 15, 1981)
df_sample['under_31_in_2012'] = ((df_sample['BIRTHYR'] > 1981) |
                                  ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))).astype('int8')

# Criterion 3: Lived continuously in US since June 15, 2007 (immigrated by 2007)
df_sample['in_us_by_2007'] = (df_sample['YRIMMIG'] <= 2007).astype('int8')

# Combine criteria for DACA eligibility
df_sample['daca_eligible'] = ((df_sample['arrived_before_16'] == 1) &
                               (df_sample['under_31_in_2012'] == 1) &
                               (df_sample['in_us_by_2007'] == 1)).astype('int8')

# 3.4 Post-DACA indicator (2013 onwards; exclude 2012 as transitional)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype('int8')

# 3.5 DiD interaction term
df_sample['daca_x_post'] = df_sample['daca_eligible'] * df_sample['post']

# 3.6 Control variables
df_sample['female'] = (df_sample['SEX'] == 2).astype('int8')
df_sample['married'] = (df_sample['MARST'] <= 2).astype('int8')

# Education categories
df_sample['less_than_hs'] = (df_sample['EDUC'] < 6).astype('int8')
df_sample['high_school'] = ((df_sample['EDUC'] >= 6) & (df_sample['EDUC'] <= 7)).astype('int8')
df_sample['some_college'] = ((df_sample['EDUC'] >= 7) & (df_sample['EDUC'] <= 9)).astype('int8')
df_sample['college_plus'] = (df_sample['EDUC'] >= 10).astype('int8')

# Metro area
df_sample['metro'] = (df_sample['METRO'] >= 2).astype('int8')

print(f"  DACA eligible: {df_sample['daca_eligible'].sum():,} ({100*df_sample['daca_eligible'].mean():.1f}%)")
print(f"  Post-DACA period: {df_sample['post'].sum():,} ({100*df_sample['post'].mean():.1f}%)")

# =============================================================================
# STEP 4: EXCLUDE 2012 (TRANSITION YEAR)
# =============================================================================
print("\n[Step 4] Excluding 2012 (transition year)...")
df_analysis = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"  Final sample for analysis: {len(df_analysis):,} observations")

# =============================================================================
# STEP 5: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[Step 5] Descriptive Statistics")
print("-"*60)

# Summary by treatment group
print("\n5.1 Sample composition by DACA eligibility:")
summary_by_eligible = df_analysis.groupby('daca_eligible').agg({
    'YEAR': 'count',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'fulltime': 'mean',
    'employed': 'mean'
}).rename(columns={'YEAR': 'N'})
print(summary_by_eligible)

# Summary by period
print("\n5.2 Sample composition by period:")
summary_by_period = df_analysis.groupby('post').agg({
    'YEAR': 'count',
    'fulltime': 'mean',
    'employed': 'mean'
}).rename(columns={'YEAR': 'N'})
print(summary_by_period)

# Full-time employment rate by group and period
print("\n5.3 Full-time employment rate by group and period (weighted):")
ft_rates = {}
for eligible in [0, 1]:
    for period in [0, 1]:
        subset = df_analysis[(df_analysis['daca_eligible'] == eligible) & (df_analysis['post'] == period)]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        n = len(subset)
        group_name = "DACA-eligible" if eligible == 1 else "Not eligible"
        period_name = "Post (2013-16)" if period == 1 else "Pre (2006-11)"
        ft_rates[(eligible, period)] = weighted_mean
        print(f"  {group_name}, {period_name}: {weighted_mean:.4f} (N={n:,})")

# Simple DiD calculation
did_simple = (ft_rates[(1,1)] - ft_rates[(1,0)]) - (ft_rates[(0,1)] - ft_rates[(0,0)])
print(f"\n  Simple DiD estimate: {did_simple:.4f}")

# =============================================================================
# STEP 6: DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("[Step 6] DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD (no controls)
print("\n6.1 Model 1: Basic DiD (no controls)")
model1 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post',
                  data=df_analysis, weights=df_analysis['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n6.2 Model 2: DiD with demographic controls")
model2 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + I(AGE**2) + female + married',
                  data=df_analysis, weights=df_analysis['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])

# Model 3: DiD with demographic + education controls
print("\n6.3 Model 3: DiD with demographic + education controls")
model3 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + I(AGE**2) + female + married + less_than_hs + some_college + college_plus',
                  data=df_analysis, weights=df_analysis['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(results3.summary().tables[1])

# Model 4: Full model with state fixed effects
print("\n6.4 Model 4: Full model with state fixed effects")
model4 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + I(AGE**2) + female + married + less_than_hs + some_college + college_plus + metro + C(STATEFIP)',
                  data=df_analysis, weights=df_analysis['PERWT'])
results4 = model4.fit(cov_type='HC1')

# Print only the key coefficients
print("\n  Key coefficients from Model 4:")
key_vars = ['Intercept', 'daca_eligible', 'post', 'daca_x_post', 'AGE', 'I(AGE ** 2)', 'female', 'married']
for var in key_vars:
    if var in results4.params.index:
        coef = results4.params[var]
        se = results4.bse[var]
        pval = results4.pvalues[var]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {var:20s}: {coef:10.6f} ({se:.6f}){stars}")

# Model 5: Year fixed effects instead of simple post dummy
print("\n6.5 Model 5: With year fixed effects (preferred specification)")
model5 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + I(AGE**2) + female + married + less_than_hs + some_college + college_plus + metro + C(YEAR) + C(STATEFIP)',
                  data=df_analysis, weights=df_analysis['PERWT'])
results5 = model5.fit(cov_type='HC1')

print("\n  Key coefficients from Model 5 (Preferred):")
key_vars = ['daca_eligible', 'daca_x_post', 'AGE', 'I(AGE ** 2)', 'female', 'married']
for var in key_vars:
    if var in results5.params.index:
        coef = results5.params[var]
        se = results5.bse[var]
        pval = results5.pvalues[var]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {var:20s}: {coef:10.6f} ({se:.6f}){stars}")

# =============================================================================
# STEP 7: PREFERRED ESTIMATE SUMMARY
# =============================================================================
print("\n" + "="*80)
print("[Step 7] PREFERRED ESTIMATE SUMMARY")
print("="*80)

preferred_coef = results5.params['daca_x_post']
preferred_se = results5.bse['daca_x_post']
preferred_ci_low = preferred_coef - 1.96 * preferred_se
preferred_ci_high = preferred_coef + 1.96 * preferred_se
preferred_pval = results5.pvalues['daca_x_post']

print(f"\n  Treatment effect (DACA eligibility x Post):")
print(f"    Coefficient: {preferred_coef:.6f}")
print(f"    Standard Error: {preferred_se:.6f}")
print(f"    95% CI: [{preferred_ci_low:.6f}, {preferred_ci_high:.6f}]")
print(f"    P-value: {preferred_pval:.6f}")
print(f"    Sample size: {len(df_analysis):,}")

# Interpretation
print(f"\n  Interpretation:")
if preferred_pval < 0.05:
    direction = "increase" if preferred_coef > 0 else "decrease"
    print(f"    DACA eligibility is associated with a statistically significant")
    print(f"    {direction} of {abs(preferred_coef)*100:.2f} percentage points in the")
    print(f"    probability of full-time employment.")
else:
    print(f"    The effect of DACA eligibility on full-time employment is not")
    print(f"    statistically significant at the 5% level.")

# =============================================================================
# STEP 8: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("[Step 8] ROBUSTNESS CHECKS")
print("="*80)

# 8.1 Conditional on employment
print("\n8.1 Conditional on being employed (intensive margin):")
df_employed = df_analysis[df_analysis['employed'] == 1].copy()
model_employed = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + I(AGE**2) + female + married + less_than_hs + some_college + college_plus + metro + C(YEAR) + C(STATEFIP)',
                          data=df_employed, weights=df_employed['PERWT'])
results_employed = model_employed.fit(cov_type='HC1')
print(f"  DiD estimate (employed only): {results_employed.params['daca_x_post']:.6f} ({results_employed.bse['daca_x_post']:.6f})")
print(f"  Sample size: {len(df_employed):,}")

# 8.2 Event study / pre-trends check
print("\n8.2 Event study analysis (testing parallel trends):")
df_analysis['year_2006'] = (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['year_2007'] = (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['year_2008'] = (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['year_2009'] = (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['year_2010'] = (df_analysis['YEAR'] == 2010).astype(int)
df_analysis['year_2011'] = (df_analysis['YEAR'] == 2011).astype(int)  # reference year
df_analysis['year_2013'] = (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['year_2014'] = (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['year_2015'] = (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['year_2016'] = (df_analysis['YEAR'] == 2016).astype(int)

# Create interactions with treatment
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'daca_x_{year}'] = df_analysis['daca_eligible'] * df_analysis[f'year_{year}']

model_event = smf.wls('fulltime ~ daca_eligible + daca_x_2006 + daca_x_2007 + daca_x_2008 + daca_x_2009 + daca_x_2010 + daca_x_2013 + daca_x_2014 + daca_x_2015 + daca_x_2016 + AGE + I(AGE**2) + female + married + C(YEAR) + C(STATEFIP)',
                       data=df_analysis, weights=df_analysis['PERWT'])
results_event = model_event.fit(cov_type='HC1')

print("\n  Year-specific treatment effects (reference: 2011):")
event_vars = ['daca_x_2006', 'daca_x_2007', 'daca_x_2008', 'daca_x_2009', 'daca_x_2010',
              'daca_x_2013', 'daca_x_2014', 'daca_x_2015', 'daca_x_2016']
for var in event_vars:
    coef = results_event.params[var]
    se = results_event.bse[var]
    pval = results_event.pvalues[var]
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    print(f"  {var}: {coef:10.6f} ({se:.6f}){stars}")

# 8.3 By gender
print("\n8.3 Heterogeneity by gender:")
for gender, gender_name in [(1, 'Male'), (2, 'Female')]:
    df_gender = df_analysis[df_analysis['SEX'] == gender]
    model_gender = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + I(AGE**2) + married + less_than_hs + some_college + college_plus + metro + C(YEAR) + C(STATEFIP)',
                            data=df_gender, weights=df_gender['PERWT'])
    results_gender = model_gender.fit(cov_type='HC1')
    print(f"  {gender_name}: DiD = {results_gender.params['daca_x_post']:.6f} ({results_gender.bse['daca_x_post']:.6f}), N = {len(df_gender):,}")

# =============================================================================
# STEP 9: EXPORT RESULTS FOR LATEX
# =============================================================================
print("\n" + "="*80)
print("[Step 9] EXPORTING RESULTS")
print("="*80)

# Create summary statistics table
summary_stats = df_analysis.groupby(['daca_eligible', 'post']).apply(
    lambda x: pd.Series({
        'N': len(x),
        'Mean Age': np.average(x['AGE'], weights=x['PERWT']),
        'Female (%)': np.average(x['female'], weights=x['PERWT']) * 100,
        'Married (%)': np.average(x['married'], weights=x['PERWT']) * 100,
        'Full-time (%)': np.average(x['fulltime'], weights=x['PERWT']) * 100,
        'Employed (%)': np.average(x['employed'], weights=x['PERWT']) * 100
    })
).reset_index()
summary_stats.columns = ['DACA Eligible', 'Post Period', 'N', 'Mean Age', 'Female (%)',
                         'Married (%)', 'Full-time (%)', 'Employed (%)']

print("\nSummary Statistics Table:")
print(summary_stats.to_string(index=False))

# Save results to CSV
summary_stats.to_csv('summary_stats.csv', index=False)

# Create regression results table
reg_results = pd.DataFrame({
    'Model': ['(1) Basic', '(2) Demographics', '(3) + Education', '(4) + State FE', '(5) Year FE (Preferred)'],
    'DiD Estimate': [results1.params['daca_x_post'], results2.params['daca_x_post'],
                     results3.params['daca_x_post'], results4.params['daca_x_post'],
                     results5.params['daca_x_post']],
    'Std. Error': [results1.bse['daca_x_post'], results2.bse['daca_x_post'],
                   results3.bse['daca_x_post'], results4.bse['daca_x_post'],
                   results5.bse['daca_x_post']],
    'P-value': [results1.pvalues['daca_x_post'], results2.pvalues['daca_x_post'],
                results3.pvalues['daca_x_post'], results4.pvalues['daca_x_post'],
                results5.pvalues['daca_x_post']],
    'N': [len(df_analysis)] * 5
})
reg_results.to_csv('regression_results.csv', index=False)

print("\nRegression Results Summary:")
print(reg_results.to_string(index=False))

# Save event study results
event_results = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [results_event.params.get(f'daca_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]],
    'Std_Error': [results_event.bse.get(f'daca_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]]
})
event_results.loc[event_results['Year'] == 2011, ['Coefficient', 'Std_Error']] = [0, 0]  # Reference year
event_results.to_csv('event_study_results.csv', index=False)

print("\nEvent Study Results:")
print(event_results.to_string(index=False))

# Save additional results for heterogeneity
het_results = []
for gender, gender_name in [(1, 'Male'), (2, 'Female')]:
    df_gender = df_analysis[df_analysis['SEX'] == gender]
    model_gender = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + I(AGE**2) + married + C(YEAR) + C(STATEFIP)',
                            data=df_gender, weights=df_gender['PERWT'])
    results_gender = model_gender.fit(cov_type='HC1')
    het_results.append({
        'Subgroup': gender_name,
        'DiD_Estimate': results_gender.params['daca_x_post'],
        'Std_Error': results_gender.bse['daca_x_post'],
        'N': len(df_gender)
    })

het_df = pd.DataFrame(het_results)
het_df.to_csv('heterogeneity_results.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nPreferred estimate (Model 5 with year and state FE):")
print(f"  Effect size: {preferred_coef:.6f}")
print(f"  Standard error: {preferred_se:.6f}")
print(f"  95% CI: [{preferred_ci_low:.6f}, {preferred_ci_high:.6f}]")
print(f"  Sample size: {len(df_analysis):,}")

# Save final summary
with open('analysis_summary.txt', 'w') as f:
    f.write("DACA Replication Analysis Summary - Replication 76\n")
    f.write("="*60 + "\n\n")
    f.write(f"Preferred Estimate (Model 5 with Year and State FE):\n")
    f.write(f"  Treatment Effect (DACA x Post): {preferred_coef:.6f}\n")
    f.write(f"  Standard Error: {preferred_se:.6f}\n")
    f.write(f"  95% CI: [{preferred_ci_low:.6f}, {preferred_ci_high:.6f}]\n")
    f.write(f"  P-value: {preferred_pval:.6f}\n")
    f.write(f"  Sample Size: {len(df_analysis):,}\n")
