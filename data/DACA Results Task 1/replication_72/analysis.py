"""
DACA Replication Study: Effect of DACA Eligibility on Full-Time Employment
Among Hispanic-Mexican, Mexican-born individuals in the United States

This script performs a difference-in-differences analysis to estimate the causal
impact of DACA eligibility on full-time employment (working 35+ hours per week).
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
print("DACA REPLICATION STUDY: EFFECT ON FULL-TIME EMPLOYMENT")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND FILTER DATA
# ============================================================================
print("\n[STEP 1] Loading data...")

# Define columns to load (only what we need to save memory)
usecols = ['YEAR', 'PERWT', 'STATEFIP', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX',
           'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
           'EMPSTAT', 'UHRSWORK', 'EDUC', 'MARST']

# Read data in chunks to filter efficiently
chunk_size = 1000000
chunks = []

print("Reading and filtering data in chunks...")
for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunk_size)):
    # Filter to Hispanic-Mexican, Mexican-born individuals
    # HISPAN == 1 means Mexican
    # BPL == 200 means born in Mexico
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"\nFiltered to Hispanic-Mexican, Mexican-born individuals: {len(df):,} observations")

# ============================================================================
# STEP 2: DEFINE DACA ELIGIBILITY
# ============================================================================
print("\n[STEP 2] Defining DACA eligibility criteria...")

# DACA Eligibility Requirements:
# 1. Arrived in US before 16th birthday
# 2. Under 31 as of June 15, 2012 (born on or after June 15, 1981)
# 3. Continuously in US since June 15, 2007 (we proxy by YRIMMIG <= 2007)
# 4. Not a citizen (CITIZEN == 3)

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Define DACA eligibility based on criteria we can observe
# Note: We cannot perfectly observe continuous presence, so we use immigration year
df['daca_eligible'] = (
    (df['CITIZEN'] == 3) &                    # Not a citizen
    (df['YRIMMIG'] > 0) &                     # Has valid immigration year
    (df['age_at_immig'] < 16) &               # Arrived before age 16
    (df['BIRTHYR'] >= 1981) &                 # Under 31 as of June 2012
    (df['YRIMMIG'] <= 2007)                   # In US since at least 2007
).astype(int)

print(f"DACA eligible individuals: {df['daca_eligible'].sum():,}")
print(f"Non-eligible individuals: {(df['daca_eligible'] == 0).sum():,}")

# ============================================================================
# STEP 3: DEFINE OUTCOME AND TREATMENT PERIOD
# ============================================================================
print("\n[STEP 3] Defining outcome and treatment variables...")

# Outcome: Full-time employment (35+ hours per week)
# Only consider those who are employed (EMPSTAT == 1) or in labor force
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)

# Post-treatment indicator: 2013-2016 (after DACA implementation)
# Pre-treatment: 2006-2011
# Exclude 2012 due to mid-year implementation ambiguity
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term for DID
df['daca_x_post'] = df['daca_eligible'] * df['post']

# Exclude 2012 from main analysis
df_analysis = df[df['YEAR'] != 2012].copy()
print(f"Analysis sample (excluding 2012): {len(df_analysis):,} observations")

# ============================================================================
# STEP 4: SAMPLE RESTRICTIONS
# ============================================================================
print("\n[STEP 4] Applying sample restrictions...")

# Restrict to working-age population (16-64)
df_analysis = df_analysis[(df_analysis['AGE'] >= 16) & (df_analysis['AGE'] <= 64)]
print(f"After age restriction (16-64): {len(df_analysis):,}")

# For the main analysis comparing eligible vs non-eligible non-citizens:
# Control group: Non-citizens who are NOT DACA-eligible (e.g., arrived after age 16, or arrived after 2007)
df_analysis = df_analysis[df_analysis['CITIZEN'] == 3]  # All non-citizens
print(f"After restricting to non-citizens: {len(df_analysis):,}")

# Further restrict to those with valid immigration info
df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0]
print(f"After requiring valid immigration year: {len(df_analysis):,}")

# ============================================================================
# STEP 5: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[STEP 5] Computing descriptive statistics...")

# Summary by DACA eligibility
print("\n--- Summary Statistics by DACA Eligibility ---")
summary_stats = df_analysis.groupby('daca_eligible').agg({
    'AGE': ['mean', 'std'],
    'employed': 'mean',
    'fulltime': 'mean',
    'SEX': lambda x: (x == 1).mean(),  # Proportion male
    'PERWT': 'sum'
}).round(3)
print(summary_stats)

# Sample sizes by year and treatment status
print("\n--- Sample Sizes by Year and DACA Eligibility ---")
sample_by_year = df_analysis.groupby(['YEAR', 'daca_eligible']).size().unstack(fill_value=0)
print(sample_by_year)

# Full-time employment rates by year and eligibility
print("\n--- Full-Time Employment Rates by Year and DACA Eligibility ---")
ft_rates = df_analysis.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
ft_rates.columns = ['Non-Eligible', 'DACA-Eligible']
print(ft_rates.round(4))

# ============================================================================
# STEP 6: MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n[STEP 6] Running Difference-in-Differences Analysis...")

# Model 1: Basic DID without controls
print("\n--- Model 1: Basic DID (No Controls) ---")
model1 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post',
                  data=df_analysis, weights=df_analysis['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary().tables[1])

# Model 2: DID with demographic controls
print("\n--- Model 2: DID with Demographic Controls ---")
df_analysis['age_sq'] = df_analysis['AGE'] ** 2
df_analysis['male'] = (df_analysis['SEX'] == 1).astype(int)
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)

# Create education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # High school or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # Some college or more

model2 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + male + married + educ_hs',
                  data=df_analysis, weights=df_analysis['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])

# Model 3: DID with state and year fixed effects
print("\n--- Model 3: DID with State and Year Fixed Effects ---")
df_analysis['state_fe'] = pd.Categorical(df_analysis['STATEFIP'])
df_analysis['year_fe'] = pd.Categorical(df_analysis['YEAR'])

model3 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + male + married + educ_hs + C(STATEFIP) + C(YEAR)',
                  data=df_analysis, weights=df_analysis['PERWT'])
results3 = model3.fit(cov_type='HC1')

# Extract key coefficients
print("\nKey coefficients from Model 3:")
key_vars = ['Intercept', 'daca_eligible', 'daca_x_post', 'AGE', 'age_sq', 'male', 'married', 'educ_hs']
for var in key_vars:
    if var in results3.params.index:
        coef = results3.params[var]
        se = results3.bse[var]
        pval = results3.pvalues[var]
        print(f"{var:20s}: {coef:10.5f} (SE: {se:.5f}, p: {pval:.4f})")

# ============================================================================
# STEP 7: ROBUSTNESS CHECKS
# ============================================================================
print("\n[STEP 7] Robustness Checks...")

# Robustness 1: Alternative control group - arrived between ages 16-25
print("\n--- Robustness: Alternative Control Group (arrived age 16-25) ---")
df_robust1 = df_analysis[
    (df_analysis['daca_eligible'] == 1) |
    ((df_analysis['age_at_immig'] >= 16) & (df_analysis['age_at_immig'] <= 25))
].copy()

model_r1 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + male + married + educ_hs + C(STATEFIP) + C(YEAR)',
                    data=df_robust1, weights=df_robust1['PERWT'])
results_r1 = model_r1.fit(cov_type='HC1')
print(f"DID coefficient: {results_r1.params['daca_x_post']:.5f} (SE: {results_r1.bse['daca_x_post']:.5f})")

# Robustness 2: Include 2012 as post-period
print("\n--- Robustness: Including 2012 in Post Period ---")
df_with_2012 = df[(df['CITIZEN'] == 3) & (df['YRIMMIG'] > 0) &
                   (df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
df_with_2012['post_2012'] = (df_with_2012['YEAR'] >= 2012).astype(int)
df_with_2012['daca_x_post_2012'] = df_with_2012['daca_eligible'] * df_with_2012['post_2012']
df_with_2012['age_sq'] = df_with_2012['AGE'] ** 2
df_with_2012['male'] = (df_with_2012['SEX'] == 1).astype(int)
df_with_2012['married'] = (df_with_2012['MARST'] <= 2).astype(int)
df_with_2012['educ_hs'] = (df_with_2012['EDUC'] >= 6).astype(int)

model_r2 = smf.wls('fulltime ~ daca_eligible + daca_x_post_2012 + AGE + age_sq + male + married + educ_hs + C(STATEFIP) + C(YEAR)',
                    data=df_with_2012, weights=df_with_2012['PERWT'])
results_r2 = model_r2.fit(cov_type='HC1')
print(f"DID coefficient: {results_r2.params['daca_x_post_2012']:.5f} (SE: {results_r2.bse['daca_x_post_2012']:.5f})")

# Robustness 3: Broader employment outcome (any employment)
print("\n--- Robustness: Any Employment (not just full-time) ---")
model_r3 = smf.wls('employed ~ daca_eligible + daca_x_post + AGE + age_sq + male + married + educ_hs + C(STATEFIP) + C(YEAR)',
                    data=df_analysis, weights=df_analysis['PERWT'])
results_r3 = model_r3.fit(cov_type='HC1')
print(f"DID coefficient: {results_r3.params['daca_x_post']:.5f} (SE: {results_r3.bse['daca_x_post']:.5f})")

# ============================================================================
# STEP 8: EVENT STUDY / DYNAMIC EFFECTS
# ============================================================================
print("\n[STEP 8] Event Study Analysis...")

# Create year indicators for event study
years = sorted(df_analysis['YEAR'].unique())
base_year = 2011  # Last pre-treatment year

df_analysis['daca_x_2006'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['daca_x_2007'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['daca_x_2008'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['daca_x_2009'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['daca_x_2010'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2010).astype(int)
# 2011 is the reference year
df_analysis['daca_x_2013'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['daca_x_2014'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['daca_x_2015'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['daca_x_2016'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2016).astype(int)

event_study_formula = '''fulltime ~ daca_eligible +
    daca_x_2006 + daca_x_2007 + daca_x_2008 + daca_x_2009 + daca_x_2010 +
    daca_x_2013 + daca_x_2014 + daca_x_2015 + daca_x_2016 +
    AGE + age_sq + male + married + educ_hs + C(STATEFIP) + C(YEAR)'''

model_event = smf.wls(event_study_formula, data=df_analysis, weights=df_analysis['PERWT'])
results_event = model_event.fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
event_vars = ['daca_x_2006', 'daca_x_2007', 'daca_x_2008', 'daca_x_2009', 'daca_x_2010',
              'daca_x_2013', 'daca_x_2014', 'daca_x_2015', 'daca_x_2016']
for var in event_vars:
    coef = results_event.params[var]
    se = results_event.bse[var]
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    print(f"{var}: {coef:8.5f} (SE: {se:.5f}, 95% CI: [{ci_low:.5f}, {ci_high:.5f}])")

# ============================================================================
# STEP 9: HETEROGENEITY ANALYSIS
# ============================================================================
print("\n[STEP 9] Heterogeneity Analysis...")

# By gender
print("\n--- Heterogeneity by Gender ---")
for gender, label in [(1, 'Male'), (2, 'Female')]:
    df_gender = df_analysis[df_analysis['SEX'] == gender]
    model_g = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + married + educ_hs + C(STATEFIP) + C(YEAR)',
                      data=df_gender, weights=df_gender['PERWT'])
    results_g = model_g.fit(cov_type='HC1')
    print(f"{label}: DID = {results_g.params['daca_x_post']:.5f} (SE: {results_g.bse['daca_x_post']:.5f})")

# By education
print("\n--- Heterogeneity by Education ---")
for educ, label in [(0, 'Less than HS'), (1, 'HS or more')]:
    df_educ = df_analysis[df_analysis['educ_hs'] == educ]
    model_e = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + male + married + C(STATEFIP) + C(YEAR)',
                      data=df_educ, weights=df_educ['PERWT'])
    results_e = model_e.fit(cov_type='HC1')
    print(f"{label}: DID = {results_e.params['daca_x_post']:.5f} (SE: {results_e.bse['daca_x_post']:.5f})")

# ============================================================================
# STEP 10: SAVE RESULTS FOR REPORT
# ============================================================================
print("\n[STEP 10] Saving results...")

# Save key results to file
results_dict = {
    'main_estimate': results3.params['daca_x_post'],
    'main_se': results3.bse['daca_x_post'],
    'main_pvalue': results3.pvalues['daca_x_post'],
    'main_ci_low': results3.params['daca_x_post'] - 1.96 * results3.bse['daca_x_post'],
    'main_ci_high': results3.params['daca_x_post'] + 1.96 * results3.bse['daca_x_post'],
    'sample_size': len(df_analysis),
    'n_eligible': df_analysis['daca_eligible'].sum(),
    'n_control': (df_analysis['daca_eligible'] == 0).sum(),
    'r_squared': results3.rsquared
}

# Print final summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
print(f"\nPreferred Estimate (Model 3 - Full controls with state and year FE):")
print(f"  DID Coefficient (DACA effect on full-time employment): {results_dict['main_estimate']:.5f}")
print(f"  Standard Error: {results_dict['main_se']:.5f}")
print(f"  95% CI: [{results_dict['main_ci_low']:.5f}, {results_dict['main_ci_high']:.5f}]")
print(f"  P-value: {results_dict['main_pvalue']:.4f}")
print(f"\nSample Size: {results_dict['sample_size']:,}")
print(f"  DACA-eligible: {results_dict['n_eligible']:,}")
print(f"  Control group: {results_dict['n_control']:,}")
print(f"\nR-squared: {results_dict['r_squared']:.4f}")

# Save descriptive stats table
desc_stats = df_analysis.groupby('daca_eligible').agg({
    'AGE': 'mean',
    'male': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')

# Save full-time rates by year
ft_rates.to_csv('fulltime_rates_by_year.csv')

# Save event study coefficients
event_coefs = pd.DataFrame({
    'year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'coefficient': [results_event.params.get(f'daca_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
                   [results_event.params.get(f'daca_x_{y}', 0) for y in [2013, 2014, 2015, 2016]],
    'se': [results_event.bse.get(f'daca_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
          [results_event.bse.get(f'daca_x_{y}', 0) for y in [2013, 2014, 2015, 2016]]
})
event_coefs.to_csv('event_study_coefficients.csv', index=False)

print("\nResults saved to CSV files.")
print("\nAnalysis complete!")
