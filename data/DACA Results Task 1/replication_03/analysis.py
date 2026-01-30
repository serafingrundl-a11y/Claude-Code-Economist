"""
DACA Replication Analysis - Replication 03
Research Question: Effect of DACA eligibility on full-time employment among
Mexican-born Hispanic non-citizens

Uses chunked reading for memory efficiency with large dataset.
Uses heteroskedasticity-robust standard errors for efficiency.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

print("="*70)
print("DACA REPLICATION ANALYSIS - REPLICATION 03")
print("="*70)

# Define columns we need to reduce memory footprint
needed_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'MARST',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT', 'UHRSWORK']

# Load and filter data in chunks
print("\n1. Loading and filtering data in chunks...")
chunks = []
chunksize = 1000000  # 1 million rows at a time

for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=needed_cols, chunksize=chunksize)):
    # Filter to target population in each chunk
    chunk = chunk[chunk['BPL'] == 200]
    chunk = chunk[chunk['HISPAN'] == 1]
    chunk = chunk[chunk['CITIZEN'] == 3]
    chunk = chunk[(chunk['AGE'] >= 18) & (chunk['AGE'] <= 64)]

    if len(chunk) > 0:
        chunks.append(chunk)
    print(f"   Processed chunk {i+1}...", end='\r')

df_mex = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()
print(f"\n   Loaded and filtered data: {len(df_mex):,} observations")

# Step 3: Define DACA eligibility
print("\n2. Defining DACA eligibility...")
print("   Criteria:")
print("   - Arrived before 16th birthday")
print("   - Born after June 15, 1981 (not yet 31 on June 15, 2012)")
print("   - Arrived by 2007 (continuous residence since June 2007)")

# Calculate age at arrival
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# DACA eligibility conditions
cond_arrival_age = df_mex['age_at_arrival'] < 16
cond_birth_year = df_mex['BIRTHYR'] > 1981
cond_arrival_year = df_mex['YRIMMIG'] <= 2007

# Combined eligibility
df_mex['daca_eligible'] = (cond_arrival_age & cond_birth_year & cond_arrival_year).astype(np.int8)

print(f"\n   DACA eligible: {df_mex['daca_eligible'].sum():,} ({df_mex['daca_eligible'].mean()*100:.1f}%)")
print(f"   Not DACA eligible: {(~df_mex['daca_eligible'].astype(bool)).sum():,}")

# Step 4: Define treatment period
print("\n3. Defining treatment periods...")
df_mex = df_mex[df_mex['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_mex):,}")

df_mex['post'] = (df_mex['YEAR'] >= 2013).astype(np.int8)
print(f"   Pre-period (2006-2011): {(df_mex['post']==0).sum():,}")
print(f"   Post-period (2013-2016): {(df_mex['post']==1).sum():,}")

# Step 5: Define outcome variable
print("\n4. Defining outcome variable...")
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(np.int8)
print(f"   Full-time employed (35+ hrs/week): {df_mex['fulltime'].mean()*100:.1f}%")

df_mex['employed'] = (df_mex['EMPSTAT'] == 1).astype(np.int8)
print(f"   Employed (any hours): {df_mex['employed'].mean()*100:.1f}%")

# Step 6: Create interaction term
df_mex['daca_post'] = (df_mex['daca_eligible'] * df_mex['post']).astype(np.int8)

# Step 7: Descriptive statistics
print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

print("\n--- Full-time Employment Rates by Group and Period ---")
summary = df_mex.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'sum', 'count'],
    'PERWT': 'sum'
}).round(4)
print(summary)

print("\n--- Weighted Full-time Employment Rates ---")
means = {}
for eligible in [0, 1]:
    for post in [0, 1]:
        mask = (df_mex['daca_eligible'] == eligible) & (df_mex['post'] == post)
        subset = df_mex[mask]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        means[(eligible, post)] = weighted_mean
        period = "Post (2013-16)" if post else "Pre (2006-11)"
        group = "DACA Eligible" if eligible else "Not Eligible"
        print(f"   {group}, {period}: {weighted_mean*100:.2f}%")

print("\n--- Simple Difference-in-Differences ---")
did = (means[(1,1)] - means[(1,0)]) - (means[(0,1)] - means[(0,0)])
print(f"   Treatment effect (DiD): {did*100:.2f} percentage points")
print(f"   = [{means[(1,1)]*100:.2f}% - {means[(1,0)]*100:.2f}%] - [{means[(0,1)]*100:.2f}% - {means[(0,0)]*100:.2f}%]")

# Step 8: Regression analysis
print("\n" + "="*70)
print("REGRESSION ANALYSIS")
print("="*70)

# Prepare control variables
df_mex['educ_hs'] = (df_mex['EDUC'] == 6).astype(np.int8)
df_mex['educ_somecol'] = (df_mex['EDUC'].isin([7, 8, 9])).astype(np.int8)
df_mex['educ_college'] = (df_mex['EDUC'] >= 10).astype(np.int8)
df_mex['female'] = (df_mex['SEX'] == 2).astype(np.int8)
df_mex['married'] = (df_mex['MARST'] == 1).astype(np.int8)
df_mex['age_sq'] = df_mex['AGE'] ** 2

# Create year dummies
for yr in sorted(df_mex['YEAR'].unique()):
    if yr != 2006:
        df_mex[f'year_{yr}'] = (df_mex['YEAR'] == yr).astype(np.int8)

year_cols = [col for col in df_mex.columns if col.startswith('year_')]
year_vars = ' + '.join(year_cols)

print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls('fulltime ~ daca_eligible + post + daca_post',
                  data=df_mex, weights=df_mex['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary().tables[1])
print(f"\nDiD Coefficient (daca_post): {results1.params['daca_post']:.4f}")
print(f"Standard Error (robust): {results1.bse['daca_post']:.4f}")
print(f"95% CI: [{results1.conf_int().loc['daca_post', 0]:.4f}, {results1.conf_int().loc['daca_post', 1]:.4f}]")
print(f"p-value: {results1.pvalues['daca_post']:.4f}")

print("\n--- Model 2: DiD with demographic controls ---")
model2 = smf.wls('fulltime ~ daca_eligible + post + daca_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_mex, weights=df_mex['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])
print(f"\nDiD Coefficient (daca_post): {results2.params['daca_post']:.4f}")
print(f"Standard Error (robust): {results2.bse['daca_post']:.4f}")
print(f"95% CI: [{results2.conf_int().loc['daca_post', 0]:.4f}, {results2.conf_int().loc['daca_post', 1]:.4f}]")
print(f"p-value: {results2.pvalues['daca_post']:.4f}")

print("\n--- Model 3: DiD with year fixed effects ---")
formula3 = f'fulltime ~ daca_eligible + daca_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + {year_vars}'
model3 = smf.wls(formula3, data=df_mex, weights=df_mex['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(f"\nDiD Coefficient (daca_post): {results3.params['daca_post']:.4f}")
print(f"Standard Error (robust): {results3.bse['daca_post']:.4f}")
print(f"95% CI: [{results3.conf_int().loc['daca_post', 0]:.4f}, {results3.conf_int().loc['daca_post', 1]:.4f}]")
print(f"p-value: {results3.pvalues['daca_post']:.4f}")

print("\n--- Model 4: DiD with state and year fixed effects ---")
# Create state dummies
state_min = df_mex['STATEFIP'].min()
for st in sorted(df_mex['STATEFIP'].unique()):
    if st != state_min:
        df_mex[f'state_{st}'] = (df_mex['STATEFIP'] == st).astype(np.int8)

state_cols = [col for col in df_mex.columns if col.startswith('state_')]
state_vars = ' + '.join(state_cols)

formula4 = f'fulltime ~ daca_eligible + daca_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + {year_vars} + {state_vars}'
model4 = smf.wls(formula4, data=df_mex, weights=df_mex['PERWT'])
results4 = model4.fit(cov_type='HC1')
print(f"\nDiD Coefficient (daca_post): {results4.params['daca_post']:.4f}")
print(f"Standard Error (robust): {results4.bse['daca_post']:.4f}")
print(f"95% CI: [{results4.conf_int().loc['daca_post', 0]:.4f}, {results4.conf_int().loc['daca_post', 1]:.4f}]")
print(f"p-value: {results4.pvalues['daca_post']:.4f}")

# Step 9: Robustness checks
print("\n" + "="*70)
print("ROBUSTNESS CHECKS")
print("="*70)

# Alternative outcome: Any employment
print("\n--- Robustness 1: Any Employment as Outcome ---")
model_emp = smf.wls(f'employed ~ daca_eligible + daca_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + {year_vars}',
                     data=df_mex, weights=df_mex['PERWT'])
results_emp = model_emp.fit(cov_type='HC1')
print(f"DiD Coefficient (daca_post): {results_emp.params['daca_post']:.4f}")
print(f"Standard Error (robust): {results_emp.bse['daca_post']:.4f}")
print(f"95% CI: [{results_emp.conf_int().loc['daca_post', 0]:.4f}, {results_emp.conf_int().loc['daca_post', 1]:.4f}]")

# Males only
print("\n--- Robustness 2: Males Only ---")
df_male = df_mex[df_mex['female'] == 0]
model_male = smf.wls(f'fulltime ~ daca_eligible + daca_post + AGE + age_sq + married + educ_hs + educ_somecol + educ_college + {year_vars}',
                      data=df_male, weights=df_male['PERWT'])
results_male = model_male.fit(cov_type='HC1')
print(f"DiD Coefficient (daca_post): {results_male.params['daca_post']:.4f}")
print(f"Standard Error (robust): {results_male.bse['daca_post']:.4f}")
print(f"95% CI: [{results_male.conf_int().loc['daca_post', 0]:.4f}, {results_male.conf_int().loc['daca_post', 1]:.4f}]")

# Females only
print("\n--- Robustness 3: Females Only ---")
df_female = df_mex[df_mex['female'] == 1]
model_female = smf.wls(f'fulltime ~ daca_eligible + daca_post + AGE + age_sq + married + educ_hs + educ_somecol + educ_college + {year_vars}',
                        data=df_female, weights=df_female['PERWT'])
results_female = model_female.fit(cov_type='HC1')
print(f"DiD Coefficient (daca_post): {results_female.params['daca_post']:.4f}")
print(f"Standard Error (robust): {results_female.bse['daca_post']:.4f}")
print(f"95% CI: [{results_female.conf_int().loc['daca_post', 0]:.4f}, {results_female.conf_int().loc['daca_post', 1]:.4f}]")

# Placebo test: Use 2009 as fake treatment year (pre-DACA period)
print("\n--- Robustness 4: Placebo Test (fake treatment in 2009) ---")
df_placebo = df_mex[df_mex['YEAR'] <= 2011].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(np.int8)
df_placebo['daca_post_placebo'] = df_placebo['daca_eligible'] * df_placebo['post_placebo']

# Create year dummies for placebo period
year_cols_placebo = [f'year_{yr}' for yr in [2007, 2008, 2009, 2010, 2011]]
year_vars_placebo = ' + '.join(year_cols_placebo)

model_placebo = smf.wls(f'fulltime ~ daca_eligible + daca_post_placebo + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + {year_vars_placebo}',
                         data=df_placebo, weights=df_placebo['PERWT'])
results_placebo = model_placebo.fit(cov_type='HC1')
print(f"DiD Coefficient (placebo): {results_placebo.params['daca_post_placebo']:.4f}")
print(f"Standard Error (robust): {results_placebo.bse['daca_post_placebo']:.4f}")
print(f"p-value: {results_placebo.pvalues['daca_post_placebo']:.4f}")
print("(We expect this to be statistically insignificant or small)")

del df_placebo
gc.collect()

# Event study / Pre-trends analysis
print("\n--- Robustness 5: Event Study (Pre-trends Analysis) ---")
# Create interaction for each year
df_mex['daca_2006'] = (df_mex['daca_eligible'] * (df_mex['YEAR'] == 2006)).astype(np.int8)
df_mex['daca_2007'] = (df_mex['daca_eligible'] * (df_mex['YEAR'] == 2007)).astype(np.int8)
df_mex['daca_2008'] = (df_mex['daca_eligible'] * (df_mex['YEAR'] == 2008)).astype(np.int8)
df_mex['daca_2009'] = (df_mex['daca_eligible'] * (df_mex['YEAR'] == 2009)).astype(np.int8)
df_mex['daca_2010'] = (df_mex['daca_eligible'] * (df_mex['YEAR'] == 2010)).astype(np.int8)
# 2011 is reference year
df_mex['daca_2013'] = (df_mex['daca_eligible'] * (df_mex['YEAR'] == 2013)).astype(np.int8)
df_mex['daca_2014'] = (df_mex['daca_eligible'] * (df_mex['YEAR'] == 2014)).astype(np.int8)
df_mex['daca_2015'] = (df_mex['daca_eligible'] * (df_mex['YEAR'] == 2015)).astype(np.int8)
df_mex['daca_2016'] = (df_mex['daca_eligible'] * (df_mex['YEAR'] == 2016)).astype(np.int8)

event_formula = f'fulltime ~ daca_eligible + daca_2006 + daca_2007 + daca_2008 + daca_2009 + daca_2010 + daca_2013 + daca_2014 + daca_2015 + daca_2016 + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + {year_vars}'
model_event = smf.wls(event_formula, data=df_mex, weights=df_mex['PERWT'])
results_event = model_event.fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
event_vars = ['daca_2006', 'daca_2007', 'daca_2008', 'daca_2009', 'daca_2010', 'daca_2013', 'daca_2014', 'daca_2015', 'daca_2016']
for var in event_vars:
    coef = results_event.params[var]
    se = results_event.bse[var]
    pval = results_event.pvalues[var]
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"   {var}: {coef:.4f} ({se:.4f}){sig}")

# Summary statistics table
print("\n" + "="*70)
print("SUMMARY STATISTICS FOR REPORT")
print("="*70)

print("\n--- Sample Characteristics ---")
print(f"Total observations (analytic sample): {len(df_mex):,}")
print(f"DACA-eligible observations: {df_mex['daca_eligible'].sum():,}")
print(f"Non-eligible observations: {(~df_mex['daca_eligible'].astype(bool)).sum():,}")
print(f"\nMean age: {df_mex['AGE'].mean():.1f}")
print(f"Proportion female: {df_mex['female'].mean()*100:.1f}%")
print(f"Proportion married: {df_mex['married'].mean()*100:.1f}%")

# Additional summary stats by eligibility group
print("\n--- Summary Statistics by DACA Eligibility ---")
for elig in [0, 1]:
    subset = df_mex[df_mex['daca_eligible'] == elig]
    group_name = "DACA Eligible" if elig else "Not Eligible"
    print(f"\n{group_name} (N={len(subset):,}):")
    print(f"   Mean age: {subset['AGE'].mean():.1f}")
    print(f"   % Female: {subset['female'].mean()*100:.1f}%")
    print(f"   % Married: {subset['married'].mean()*100:.1f}%")
    print(f"   % HS diploma: {subset['educ_hs'].mean()*100:.1f}%")
    print(f"   % Some college: {subset['educ_somecol'].mean()*100:.1f}%")
    print(f"   % College+: {subset['educ_college'].mean()*100:.1f}%")
    print(f"   % Full-time employed: {subset['fulltime'].mean()*100:.1f}%")

# Final results summary
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"\nPreferred Estimate (Model 3 - Year FE, demographic controls):")
print(f"   DiD Coefficient: {results3.params['daca_post']:.4f}")
print(f"   Standard Error: {results3.bse['daca_post']:.4f}")
print(f"   95% CI: [{results3.conf_int().loc['daca_post', 0]:.4f}, {results3.conf_int().loc['daca_post', 1]:.4f}]")
print(f"   p-value: {results3.pvalues['daca_post']:.4f}")
print(f"   Sample Size: {len(df_mex):,}")
print(f"\nInterpretation: DACA eligibility is associated with a {results3.params['daca_post']*100:.2f} percentage point")
if results3.params['daca_post'] > 0:
    print(f"increase in the probability of full-time employment.")
else:
    print(f"change in the probability of full-time employment.")

# Save results for LaTeX report
results_dict = {
    'model1_coef': float(results1.params['daca_post']),
    'model1_se': float(results1.bse['daca_post']),
    'model1_ci_low': float(results1.conf_int().loc['daca_post', 0]),
    'model1_ci_high': float(results1.conf_int().loc['daca_post', 1]),
    'model1_pval': float(results1.pvalues['daca_post']),
    'model2_coef': float(results2.params['daca_post']),
    'model2_se': float(results2.bse['daca_post']),
    'model2_ci_low': float(results2.conf_int().loc['daca_post', 0]),
    'model2_ci_high': float(results2.conf_int().loc['daca_post', 1]),
    'model2_pval': float(results2.pvalues['daca_post']),
    'model3_coef': float(results3.params['daca_post']),
    'model3_se': float(results3.bse['daca_post']),
    'model3_ci_low': float(results3.conf_int().loc['daca_post', 0]),
    'model3_ci_high': float(results3.conf_int().loc['daca_post', 1]),
    'model3_pval': float(results3.pvalues['daca_post']),
    'model4_coef': float(results4.params['daca_post']),
    'model4_se': float(results4.bse['daca_post']),
    'model4_ci_low': float(results4.conf_int().loc['daca_post', 0]),
    'model4_ci_high': float(results4.conf_int().loc['daca_post', 1]),
    'model4_pval': float(results4.pvalues['daca_post']),
    'sample_size': int(len(df_mex)),
    'n_eligible': int(df_mex['daca_eligible'].sum()),
    'n_not_eligible': int((~df_mex['daca_eligible'].astype(bool)).sum()),
    'mean_fulltime_eligible_pre': float(means[(1,0)]),
    'mean_fulltime_eligible_post': float(means[(1,1)]),
    'mean_fulltime_noteligible_pre': float(means[(0,0)]),
    'mean_fulltime_noteligible_post': float(means[(0,1)]),
    'simple_did': float(did),
    'robust_employment_coef': float(results_emp.params['daca_post']),
    'robust_employment_se': float(results_emp.bse['daca_post']),
    'robust_male_coef': float(results_male.params['daca_post']),
    'robust_male_se': float(results_male.bse['daca_post']),
    'robust_female_coef': float(results_female.params['daca_post']),
    'robust_female_se': float(results_female.bse['daca_post']),
    'placebo_coef': float(results_placebo.params['daca_post_placebo']),
    'placebo_se': float(results_placebo.bse['daca_post_placebo']),
    'placebo_pval': float(results_placebo.pvalues['daca_post_placebo']),
    'mean_age': float(df_mex['AGE'].mean()),
    'pct_female': float(df_mex['female'].mean()*100),
    'pct_married': float(df_mex['married'].mean()*100),
    'n_male': int(len(df_male)),
    'n_female': int(len(df_female)),
}

# Save event study results
for var in event_vars:
    results_dict[f'event_{var}_coef'] = float(results_event.params[var])
    results_dict[f'event_{var}_se'] = float(results_event.bse[var])
    results_dict[f'event_{var}_pval'] = float(results_event.pvalues[var])

# Summary stats by group
for elig in [0, 1]:
    subset = df_mex[df_mex['daca_eligible'] == elig]
    grp = 'eligible' if elig else 'noteligible'
    results_dict[f'{grp}_n'] = int(len(subset))
    results_dict[f'{grp}_mean_age'] = float(subset['AGE'].mean())
    results_dict[f'{grp}_pct_female'] = float(subset['female'].mean()*100)
    results_dict[f'{grp}_pct_married'] = float(subset['married'].mean()*100)
    results_dict[f'{grp}_pct_hs'] = float(subset['educ_hs'].mean()*100)
    results_dict[f'{grp}_pct_somecol'] = float(subset['educ_somecol'].mean()*100)
    results_dict[f'{grp}_pct_college'] = float(subset['educ_college'].mean()*100)
    results_dict[f'{grp}_pct_fulltime'] = float(subset['fulltime'].mean()*100)

# Save to file
import json
with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n\nResults saved to results.json")
print("="*70)
print("Analysis complete!")
print("="*70)
