"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the United States.

Identification Strategy: Difference-in-Differences
- Treatment: DACA-eligible individuals
- Control: Non-eligible non-citizen Mexican-born Hispanics
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded due to mid-year implementation)
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

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("Effect of DACA Eligibility on Full-Time Employment")
print("="*80)

# ============================================================================
# STEP 1: Load and Filter Data
# ============================================================================
print("\n[1] Loading data...")

# Read only necessary columns to reduce memory usage
cols_needed = [
    'YEAR', 'STATEFIP', 'PERWT', 'AGE', 'SEX', 'BIRTHYR', 'BIRTHQTR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK', 'MARST'
]

# Load data in chunks to handle large file
chunks = []
chunksize = 1000000

print("Loading data in chunks...")
for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunksize)):
    # Filter to Hispanic-Mexican born in Mexico who are non-citizens
    # HISPAN == 1 is Mexican
    # BPL == 200 is Mexico
    # CITIZEN == 3 is not a citizen
    filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3)
    ].copy()

    if len(filtered) > 0:
        chunks.append(filtered)

    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"  Total observations after initial filter: {len(df):,}")

# ============================================================================
# STEP 2: Apply Sample Restrictions
# ============================================================================
print("\n[2] Applying sample restrictions...")

# Restrict to working-age population (18-64)
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)].copy()
print(f"  After age restriction (18-64): {len(df):,}")

# Exclude 2012 (mid-year DACA implementation)
df = df[df['YEAR'] != 2012].copy()
print(f"  After excluding 2012: {len(df):,}")

# Require valid immigration year for eligibility determination
df = df[df['YRIMMIG'] > 0].copy()
print(f"  After requiring valid YRIMMIG: {len(df):,}")

# ============================================================================
# STEP 3: Create DACA Eligibility Indicator
# ============================================================================
print("\n[3] Creating DACA eligibility indicator...")

# DACA Eligibility Criteria:
# 1. Arrived in US before 16th birthday
# 2. Under 31 as of June 15, 2012 (born after June 15, 1981)
# 3. In US since June 15, 2007 (arrived by 2007)
# 4. Not a citizen (already filtered)

# Calculate age at arrival
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Criterion 1: Arrived before age 16
df['arrived_before_16'] = df['age_at_arrival'] < 16

# Criterion 2: Under 31 on June 15, 2012
# Conservative: BIRTHYR >= 1982 ensures under 31
# More precise: Consider BIRTHQTR for those born in 1981
df['under_31_in_2012'] = (df['BIRTHYR'] >= 1982) | \
                          ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))

# Criterion 3: In US since 2007
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007

# Combined DACA eligibility
df['daca_eligible'] = (df['arrived_before_16'] &
                       df['under_31_in_2012'] &
                       df['in_us_since_2007']).astype(int)

print(f"  DACA eligible: {df['daca_eligible'].sum():,} ({df['daca_eligible'].mean()*100:.1f}%)")
print(f"  Not eligible: {(1-df['daca_eligible']).sum():,} ({(1-df['daca_eligible']).mean()*100:.1f}%)")

# ============================================================================
# STEP 4: Create Outcome and Control Variables
# ============================================================================
print("\n[4] Creating outcome and control variables...")

# Outcome: Full-time employment (35+ hours/week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Post-DACA period indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term for DiD
df['daca_post'] = df['daca_eligible'] * df['post']

# Control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_lesshs'] = (df['EDUCD'] < 62).astype(int)  # Less than high school
df['educ_hs'] = ((df['EDUCD'] >= 62) & (df['EDUCD'] < 65)).astype(int)  # High school
df['educ_somecoll'] = ((df['EDUCD'] >= 65) & (df['EDUCD'] < 101)).astype(int)  # Some college
df['educ_coll'] = (df['EDUCD'] >= 101).astype(int)  # College+

# Age squared for flexibility
df['age_sq'] = df['AGE'] ** 2

print(f"  Full-time employed: {df['fulltime'].mean()*100:.1f}%")
print(f"  Pre-period observations: {(df['post']==0).sum():,}")
print(f"  Post-period observations: {(df['post']==1).sum():,}")

# ============================================================================
# STEP 5: Summary Statistics
# ============================================================================
print("\n[5] Summary Statistics")
print("="*80)

# By eligibility status and period
summary_stats = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_coll': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary by DACA Eligibility and Period:")
print(summary_stats)

# Weighted summary
print("\n" + "-"*60)
print("Weighted Full-Time Employment Rates:")
for eligible in [0, 1]:
    for post in [0, 1]:
        subset = df[(df['daca_eligible'] == eligible) & (df['post'] == post)]
        if len(subset) > 0:
            weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
            n = len(subset)
            period = "Post" if post else "Pre"
            elig = "Eligible" if eligible else "Not Eligible"
            print(f"  {elig}, {period}: {weighted_mean*100:.2f}% (N={n:,})")

# ============================================================================
# STEP 6: Main Difference-in-Differences Regression
# ============================================================================
print("\n[6] Difference-in-Differences Analysis")
print("="*80)

# Simple DiD (no controls)
print("\n--- Model 1: Simple DiD ---")
X_simple = df[['daca_eligible', 'post', 'daca_post']]
X_simple = sm.add_constant(X_simple)
y = df['fulltime']
weights = df['PERWT']

model_simple = sm.WLS(y, X_simple, weights=weights).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['STATEFIP']}
)
print(model_simple.summary().tables[1])

# DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
X_controls = df[['daca_eligible', 'post', 'daca_post',
                 'AGE', 'age_sq', 'female', 'married',
                 'educ_hs', 'educ_somecoll', 'educ_coll']]
X_controls = sm.add_constant(X_controls)

model_controls = sm.WLS(y, X_controls, weights=weights).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['STATEFIP']}
)
print(model_controls.summary().tables[1])

# DiD with year and state fixed effects
print("\n--- Model 3: DiD with Year and State Fixed Effects ---")

# Create year dummies (excluding 2006 as reference)
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True).astype(float)
# Create state dummies (excluding first as reference)
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True).astype(float)

X_fe = pd.concat([
    df[['daca_eligible', 'daca_post', 'AGE', 'age_sq', 'female', 'married',
        'educ_hs', 'educ_somecoll', 'educ_coll']].astype(float),
    year_dummies,
    state_dummies
], axis=1)
X_fe = sm.add_constant(X_fe)

model_fe = sm.WLS(y.astype(float), X_fe.astype(float), weights=weights.astype(float)).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['STATEFIP']}
)

# Print only key coefficients
print("\nKey Coefficients (Full FE Model):")
key_vars = ['const', 'daca_eligible', 'daca_post', 'AGE', 'age_sq',
            'female', 'married', 'educ_hs', 'educ_somecoll', 'educ_coll']
for var in key_vars:
    if var in model_fe.params.index:
        coef = model_fe.params[var]
        se = model_fe.bse[var]
        pval = model_fe.pvalues[var]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {var:20s}: {coef:8.4f} ({se:.4f}){stars}")

# ============================================================================
# STEP 7: Robustness Checks
# ============================================================================
print("\n[7] Robustness Checks")
print("="*80)

# Robustness 1: Restrict to younger sample (more likely DACA target)
print("\n--- Robustness 1: Ages 18-35 Only ---")
df_young = df[(df['AGE'] >= 18) & (df['AGE'] <= 35)].copy()
X_young = df_young[['daca_eligible', 'post', 'daca_post',
                    'AGE', 'age_sq', 'female', 'married',
                    'educ_hs', 'educ_somecoll', 'educ_coll']]
X_young = sm.add_constant(X_young)
y_young = df_young['fulltime']
weights_young = df_young['PERWT']

model_young = sm.WLS(y_young, X_young, weights=weights_young).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_young['STATEFIP']}
)
print(f"  DiD coefficient (daca_post): {model_young.params['daca_post']:.4f}")
print(f"  Std. Error: {model_young.bse['daca_post']:.4f}")
print(f"  95% CI: [{model_young.conf_int().loc['daca_post', 0]:.4f}, {model_young.conf_int().loc['daca_post', 1]:.4f}]")
print(f"  N = {len(df_young):,}")

# Robustness 2: Male only
print("\n--- Robustness 2: Males Only ---")
df_male = df[df['female'] == 0].copy()
X_male = df_male[['daca_eligible', 'post', 'daca_post',
                  'AGE', 'age_sq', 'married',
                  'educ_hs', 'educ_somecoll', 'educ_coll']]
X_male = sm.add_constant(X_male)
y_male = df_male['fulltime']
weights_male = df_male['PERWT']

model_male = sm.WLS(y_male, X_male, weights=weights_male).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_male['STATEFIP']}
)
print(f"  DiD coefficient (daca_post): {model_male.params['daca_post']:.4f}")
print(f"  Std. Error: {model_male.bse['daca_post']:.4f}")
print(f"  N = {len(df_male):,}")

# Robustness 3: Female only
print("\n--- Robustness 3: Females Only ---")
df_female = df[df['female'] == 1].copy()
X_female = df_female[['daca_eligible', 'post', 'daca_post',
                      'AGE', 'age_sq', 'married',
                      'educ_hs', 'educ_somecoll', 'educ_coll']]
X_female = sm.add_constant(X_female)
y_female = df_female['fulltime']
weights_female = df_female['PERWT']

model_female = sm.WLS(y_female, X_female, weights=weights_female).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_female['STATEFIP']}
)
print(f"  DiD coefficient (daca_post): {model_female.params['daca_post']:.4f}")
print(f"  Std. Error: {model_female.bse['daca_post']:.4f}")
print(f"  N = {len(df_female):,}")

# ============================================================================
# STEP 8: Event Study / Pre-Trends Check
# ============================================================================
print("\n[8] Event Study - Pre-Trends Check")
print("="*80)

# Create year-specific treatment effects
df['year_2006'] = (df['YEAR'] == 2006).astype(int) * df['daca_eligible']
df['year_2007'] = (df['YEAR'] == 2007).astype(int) * df['daca_eligible']
df['year_2008'] = (df['YEAR'] == 2008).astype(int) * df['daca_eligible']
df['year_2009'] = (df['YEAR'] == 2009).astype(int) * df['daca_eligible']
df['year_2010'] = (df['YEAR'] == 2010).astype(int) * df['daca_eligible']
# 2011 is reference year (omitted)
df['year_2013'] = (df['YEAR'] == 2013).astype(int) * df['daca_eligible']
df['year_2014'] = (df['YEAR'] == 2014).astype(int) * df['daca_eligible']
df['year_2015'] = (df['YEAR'] == 2015).astype(int) * df['daca_eligible']
df['year_2016'] = (df['YEAR'] == 2016).astype(int) * df['daca_eligible']

X_event = df[['daca_eligible',
              'year_2006', 'year_2007', 'year_2008', 'year_2009', 'year_2010',
              'year_2013', 'year_2014', 'year_2015', 'year_2016',
              'AGE', 'age_sq', 'female', 'married',
              'educ_hs', 'educ_somecoll', 'educ_coll']].astype(float)

# Add year dummies
year_dummies_event = pd.get_dummies(df['YEAR'], prefix='yr', drop_first=True).astype(float)
X_event = pd.concat([X_event, year_dummies_event], axis=1)
X_event = sm.add_constant(X_event)

model_event = sm.WLS(y.astype(float), X_event.astype(float), weights=weights.astype(float)).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['STATEFIP']}
)

print("\nEvent Study Coefficients (relative to 2011):")
event_vars = ['year_2006', 'year_2007', 'year_2008', 'year_2009', 'year_2010',
              'year_2013', 'year_2014', 'year_2015', 'year_2016']
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_low = model_event.conf_int().loc[var, 0]
    ci_high = model_event.conf_int().loc[var, 1]
    pval = model_event.pvalues[var]
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    year = var.replace('year_', '')
    print(f"  {year}: {coef:7.4f} ({se:.4f}) [{ci_low:.4f}, {ci_high:.4f}]{stars}")

# ============================================================================
# STEP 9: Final Results Summary
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

# Preferred specification: Model with controls (Model 2)
preferred_coef = model_controls.params['daca_post']
preferred_se = model_controls.bse['daca_post']
preferred_ci = model_controls.conf_int().loc['daca_post']
preferred_pval = model_controls.pvalues['daca_post']
n_obs = len(df)

print(f"\nPreferred Estimate (Model 2 - DiD with Demographic Controls):")
print(f"  Effect of DACA eligibility on full-time employment: {preferred_coef:.4f}")
print(f"  Standard Error (clustered at state): {preferred_se:.4f}")
print(f"  95% Confidence Interval: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"  p-value: {preferred_pval:.4f}")
print(f"  Sample Size: {n_obs:,}")

print(f"\nInterpretation:")
print(f"  DACA eligibility is associated with a {preferred_coef*100:.2f} percentage point")
if preferred_coef > 0:
    print(f"  INCREASE in the probability of full-time employment.")
else:
    print(f"  DECREASE in the probability of full-time employment.")

if preferred_pval < 0.05:
    print(f"  This effect is statistically significant at the 5% level.")
else:
    print(f"  This effect is NOT statistically significant at the 5% level.")

# ============================================================================
# STEP 10: Save Results for LaTeX Report
# ============================================================================
print("\n[10] Saving results for report...")

# Create results dictionary
results = {
    'model1_simple': {
        'coef': model_simple.params['daca_post'],
        'se': model_simple.bse['daca_post'],
        'pval': model_simple.pvalues['daca_post'],
        'n': len(df),
        'r2': model_simple.rsquared
    },
    'model2_controls': {
        'coef': model_controls.params['daca_post'],
        'se': model_controls.bse['daca_post'],
        'pval': model_controls.pvalues['daca_post'],
        'ci_low': model_controls.conf_int().loc['daca_post', 0],
        'ci_high': model_controls.conf_int().loc['daca_post', 1],
        'n': len(df),
        'r2': model_controls.rsquared
    },
    'model3_fe': {
        'coef': model_fe.params['daca_post'],
        'se': model_fe.bse['daca_post'],
        'pval': model_fe.pvalues['daca_post'],
        'n': len(df),
        'r2': model_fe.rsquared
    }
}

# Save summary statistics
summary_data = {
    'Group': ['Non-Eligible Pre', 'Non-Eligible Post', 'Eligible Pre', 'Eligible Post'],
    'N': [],
    'FT_Rate': [],
    'Mean_Age': [],
    'Pct_Female': [],
    'Pct_Married': []
}

for eligible in [0, 1]:
    for post in [0, 1]:
        subset = df[(df['daca_eligible'] == eligible) & (df['post'] == post)]
        summary_data['N'].append(len(subset))
        summary_data['FT_Rate'].append(np.average(subset['fulltime'], weights=subset['PERWT']))
        summary_data['Mean_Age'].append(np.average(subset['AGE'], weights=subset['PERWT']))
        summary_data['Pct_Female'].append(np.average(subset['female'], weights=subset['PERWT']))
        summary_data['Pct_Married'].append(np.average(subset['married'], weights=subset['PERWT']))

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('summary_statistics.csv', index=False)
print("  Saved summary_statistics.csv")

# Save regression results
reg_results = pd.DataFrame({
    'Model': ['Simple DiD', 'With Controls', 'With FE'],
    'Coefficient': [results['model1_simple']['coef'],
                   results['model2_controls']['coef'],
                   results['model3_fe']['coef']],
    'Std_Error': [results['model1_simple']['se'],
                  results['model2_controls']['se'],
                  results['model3_fe']['se']],
    'P_Value': [results['model1_simple']['pval'],
                results['model2_controls']['pval'],
                results['model3_fe']['pval']],
    'N': [results['model1_simple']['n'],
          results['model2_controls']['n'],
          results['model3_fe']['n']],
    'R_Squared': [results['model1_simple']['r2'],
                  results['model2_controls']['r2'],
                  results['model3_fe']['r2']]
})
reg_results.to_csv('regression_results.csv', index=False)
print("  Saved regression_results.csv")

# Save event study coefficients
event_data = {
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [],
    'Std_Error': [],
    'CI_Low': [],
    'CI_High': []
}

for year in [2006, 2007, 2008, 2009, 2010]:
    var = f'year_{year}'
    event_data['Coefficient'].append(model_event.params[var])
    event_data['Std_Error'].append(model_event.bse[var])
    event_data['CI_Low'].append(model_event.conf_int().loc[var, 0])
    event_data['CI_High'].append(model_event.conf_int().loc[var, 1])

# 2011 is reference (0)
event_data['Coefficient'].append(0)
event_data['Std_Error'].append(0)
event_data['CI_Low'].append(0)
event_data['CI_High'].append(0)

for year in [2013, 2014, 2015, 2016]:
    var = f'year_{year}'
    event_data['Coefficient'].append(model_event.params[var])
    event_data['Std_Error'].append(model_event.bse[var])
    event_data['CI_Low'].append(model_event.conf_int().loc[var, 0])
    event_data['CI_High'].append(model_event.conf_int().loc[var, 1])

event_df = pd.DataFrame(event_data)
event_df.to_csv('event_study_results.csv', index=False)
print("  Saved event_study_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
