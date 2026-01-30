"""
DACA Employment Effects Replication Analysis
============================================
Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals?

Author: Replication Task 89
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
print("DACA Employment Effects Replication Analysis")
print("=" * 80)

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n[Step 1] Loading data...")

# Load data in chunks for memory efficiency, selecting only needed columns
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'MARST', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EDUCD',
               'EMPSTAT', 'UHRSWORK']

df = pd.read_csv('data/data.csv', usecols=cols_needed)
print(f"Total observations loaded: {len(df):,}")

# =============================================================================
# STEP 2: Sample Selection
# =============================================================================
print("\n[Step 2] Applying sample restrictions...")

# Initial counts
print(f"  Initial observations: {len(df):,}")

# 2a. Restrict to Hispanic-Mexican ethnicity (HISPAN == 1)
df = df[df['HISPAN'] == 1].copy()
print(f"  After Hispanic-Mexican restriction: {len(df):,}")

# 2b. Restrict to Mexican-born (BPL == 200)
df = df[df['BPL'] == 200].copy()
print(f"  After Mexican-born restriction: {len(df):,}")

# 2c. Restrict to non-citizens (CITIZEN == 3)
# Per instructions: "Assume anyone who is not a citizen and has not received
# immigration papers is undocumented for DACA purposes"
df = df[df['CITIZEN'] == 3].copy()
print(f"  After non-citizen restriction: {len(df):,}")

# 2d. Restrict to working-age population (16-64)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
print(f"  After working-age (16-64) restriction: {len(df):,}")

# 2e. Exclude year 2012 (DACA implemented June 15, 2012 - can't distinguish pre/post)
df = df[df['YEAR'] != 2012].copy()
print(f"  After excluding 2012: {len(df):,}")

# 2f. Drop observations with missing key variables
df = df.dropna(subset=['YRIMMIG', 'BIRTHYR', 'UHRSWORK', 'EMPSTAT'])
df = df[df['YRIMMIG'] > 0]  # 0 = N/A
print(f"  After dropping missing values: {len(df):,}")

# =============================================================================
# STEP 3: Create Analysis Variables
# =============================================================================
print("\n[Step 3] Creating analysis variables...")

# 3a. Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# 3b. Calculate age at immigration
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']

# 3c. DACA Eligibility Criteria
# Criterion 1: Arrived before 16th birthday
df['arrived_before_16'] = (df['age_at_immigration'] < 16).astype(int)

# Criterion 2: Under 31 as of June 15, 2012 (born 1981 or later)
# More precisely: if born in Q3 or Q4 of 1981, they would be 30 on June 15, 2012
# Conservatively: born after June 15, 1981 -> born in 1982 or later, OR Q3/Q4 of 1981
# For simplicity, use BIRTHYR >= 1981
df['under_31_june2012'] = (df['BIRTHYR'] >= 1981).astype(int)

# Criterion 3: Continuous presence since June 15, 2007 (arrived by 2007)
df['arrived_by_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# Combined DACA eligibility
df['daca_eligible'] = ((df['arrived_before_16'] == 1) &
                        (df['under_31_june2012'] == 1) &
                        (df['arrived_by_2007'] == 1)).astype(int)

# 3d. Outcome: Full-time employment (>=35 hours/week, among those employed)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
df['fulltime'] = ((df['UHRSWORK'] >= 35)).astype(int)

# Also create full-time conditional on being employed
df['fulltime_if_employed'] = np.where(df['employed'] == 1,
                                       (df['UHRSWORK'] >= 35).astype(int),
                                       np.nan)

# 3e. DiD interaction term
df['daca_x_post'] = df['daca_eligible'] * df['post']

# 3f. Additional control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_lessHS'] = (df['EDUCD'] < 62).astype(int)  # Less than HS
df['educ_HS'] = ((df['EDUCD'] >= 62) & (df['EDUCD'] <= 64)).astype(int)  # HS diploma
df['educ_someCol'] = ((df['EDUCD'] > 64) & (df['EDUCD'] < 101)).astype(int)  # Some college
df['educ_BA'] = (df['EDUCD'] >= 101).astype(int)  # BA or higher

# Age squared for non-linear age effects
df['age_sq'] = df['AGE'] ** 2

print(f"  DACA-eligible observations: {df['daca_eligible'].sum():,}")
print(f"  Non-DACA-eligible observations: {(df['daca_eligible']==0).sum():,}")
print(f"  Post-DACA period observations: {df['post'].sum():,}")
print(f"  Pre-DACA period observations: {(df['post']==0).sum():,}")

# =============================================================================
# STEP 4: Descriptive Statistics
# =============================================================================
print("\n[Step 4] Generating descriptive statistics...")

# Summary statistics by treatment group
desc_stats = df.groupby('daca_eligible').agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean', 'std'],
    'AGE': ['mean', 'std'],
    'female': 'mean',
    'married': 'mean',
    'educ_lessHS': 'mean',
    'educ_HS': 'mean',
    'educ_someCol': 'mean',
    'educ_BA': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nDescriptive Statistics by DACA Eligibility:")
print(desc_stats)

# Pre/Post comparison
print("\nFull-time Employment Rates by Group and Period:")
ft_rates = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'PERWT': 'sum'
}).round(4)
print(ft_rates)

# =============================================================================
# STEP 5: Main DiD Analysis
# =============================================================================
print("\n[Step 5] Running main difference-in-differences analysis...")

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post',
                  data=df, weights=df['PERWT'])
results1 = model1.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(results1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
model2 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married',
                  data=df, weights=df['PERWT'])
results2 = model2.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(results2.summary().tables[1])

# Model 3: DiD with demographic + education controls
print("\n--- Model 3: DiD with demographic + education controls ---")
model3 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_HS + educ_someCol + educ_BA',
                  data=df, weights=df['PERWT'])
results3 = model3.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(results3.summary().tables[1])

# Model 4: DiD with state and year fixed effects
print("\n--- Model 4: DiD with state and year fixed effects ---")
# Create dummies for states and years
df['state_fe'] = df['STATEFIP'].astype(str)
df['year_fe'] = df['YEAR'].astype(str)

model4 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_HS + educ_someCol + educ_BA + C(state_fe) + C(year_fe)',
                  data=df, weights=df['PERWT'])
results4 = model4.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

# Print only key coefficients
print("\nKey coefficients from Model 4:")
key_vars = ['Intercept', 'daca_eligible', 'daca_x_post', 'AGE', 'age_sq', 'female', 'married',
            'educ_HS', 'educ_someCol', 'educ_BA']
for var in key_vars:
    if var in results4.params.index:
        coef = results4.params[var]
        se = results4.bse[var]
        pval = results4.pvalues[var]
        print(f"  {var:20s}: {coef:10.6f} (SE: {se:.6f}, p={pval:.4f})")

# =============================================================================
# STEP 6: Main Result Summary
# =============================================================================
print("\n" + "=" * 80)
print("MAIN RESULTS SUMMARY")
print("=" * 80)

# Preferred specification is Model 4 (with FE)
main_effect = results4.params['daca_x_post']
main_se = results4.bse['daca_x_post']
main_pval = results4.pvalues['daca_x_post']
ci_lower = main_effect - 1.96 * main_se
ci_upper = main_effect + 1.96 * main_se

print(f"\nPreferred Estimate (Model 4 - with state and year FE):")
print(f"  DiD Effect (daca_x_post): {main_effect:.6f}")
print(f"  Standard Error:           {main_se:.6f}")
print(f"  p-value:                  {main_pval:.4f}")
print(f"  95% CI:                   [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"  Sample Size:              {len(df):,}")

# Convert to percentage points
print(f"\nInterpretation:")
print(f"  DACA eligibility is associated with a {main_effect*100:.2f} percentage point")
print(f"  change in the probability of full-time employment.")

# =============================================================================
# STEP 7: Robustness Checks
# =============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# 7a. Alternative age restriction for control group
print("\n--- Robustness 1: Restrict to ages 18-40 ---")
df_robust1 = df[(df['AGE'] >= 18) & (df['AGE'] <= 40)].copy()
model_r1 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_HS + educ_someCol + educ_BA + C(state_fe) + C(year_fe)',
                    data=df_robust1, weights=df_robust1['PERWT'])
results_r1 = model_r1.fit(cov_type='cluster', cov_kwds={'groups': df_robust1['STATEFIP']})
print(f"  DiD Effect: {results_r1.params['daca_x_post']:.6f} (SE: {results_r1.bse['daca_x_post']:.6f})")
print(f"  N = {len(df_robust1):,}")

# 7b. Men only
print("\n--- Robustness 2: Men only ---")
df_men = df[df['female'] == 0].copy()
model_men = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + married + educ_HS + educ_someCol + educ_BA + C(state_fe) + C(year_fe)',
                     data=df_men, weights=df_men['PERWT'])
results_men = model_men.fit(cov_type='cluster', cov_kwds={'groups': df_men['STATEFIP']})
print(f"  DiD Effect: {results_men.params['daca_x_post']:.6f} (SE: {results_men.bse['daca_x_post']:.6f})")
print(f"  N = {len(df_men):,}")

# 7c. Women only
print("\n--- Robustness 3: Women only ---")
df_women = df[df['female'] == 1].copy()
model_women = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + married + educ_HS + educ_someCol + educ_BA + C(state_fe) + C(year_fe)',
                       data=df_women, weights=df_women['PERWT'])
results_women = model_women.fit(cov_type='cluster', cov_kwds={'groups': df_women['STATEFIP']})
print(f"  DiD Effect: {results_women.params['daca_x_post']:.6f} (SE: {results_women.bse['daca_x_post']:.6f})")
print(f"  N = {len(df_women):,}")

# 7d. Employment (extensive margin) instead of full-time
print("\n--- Robustness 4: Employment (extensive margin) as outcome ---")
model_emp = smf.wls('employed ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_HS + educ_someCol + educ_BA + C(state_fe) + C(year_fe)',
                     data=df, weights=df['PERWT'])
results_emp = model_emp.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD Effect: {results_emp.params['daca_x_post']:.6f} (SE: {results_emp.bse['daca_x_post']:.6f})")

# 7e. Alternative control group - age 31+ at June 2012 but otherwise similar
print("\n--- Robustness 5: Narrower bandwidth around age 31 cutoff ---")
# DACA-eligible: ages 26-30 in 2012 (born 1982-1986)
# Control: ages 32-36 in 2012 (born 1976-1980)
df_narrow = df[((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)) |
               ((df['BIRTHYR'] >= 1976) & (df['BIRTHYR'] <= 1980))].copy()
df_narrow['daca_eligible_narrow'] = (df_narrow['BIRTHYR'] >= 1982).astype(int) * df_narrow['arrived_before_16'] * df_narrow['arrived_by_2007']
df_narrow['daca_x_post_narrow'] = df_narrow['daca_eligible_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ daca_eligible_narrow + daca_x_post_narrow + AGE + age_sq + female + married + educ_HS + educ_someCol + educ_BA + C(state_fe) + C(year_fe)',
                        data=df_narrow, weights=df_narrow['PERWT'])
results_narrow = model_narrow.fit(cov_type='cluster', cov_kwds={'groups': df_narrow['STATEFIP']})
print(f"  DiD Effect: {results_narrow.params['daca_x_post_narrow']:.6f} (SE: {results_narrow.bse['daca_x_post_narrow']:.6f})")
print(f"  N = {len(df_narrow):,}")

# =============================================================================
# STEP 8: Event Study / Pre-trends Check
# =============================================================================
print("\n" + "=" * 80)
print("EVENT STUDY / PRE-TRENDS ANALYSIS")
print("=" * 80)

# Create year dummies interacted with treatment
df['year_2006'] = (df['YEAR'] == 2006).astype(int)
df['year_2007'] = (df['YEAR'] == 2007).astype(int)
df['year_2008'] = (df['YEAR'] == 2008).astype(int)
df['year_2009'] = (df['YEAR'] == 2009).astype(int)
df['year_2010'] = (df['YEAR'] == 2010).astype(int)
df['year_2011'] = (df['YEAR'] == 2011).astype(int)
# 2012 excluded
df['year_2013'] = (df['YEAR'] == 2013).astype(int)
df['year_2014'] = (df['YEAR'] == 2014).astype(int)
df['year_2015'] = (df['YEAR'] == 2015).astype(int)
df['year_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions (2011 as reference year)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'daca_x_{yr}'] = df['daca_eligible'] * df[f'year_{yr}']

event_formula = 'fulltime ~ daca_eligible + daca_x_2006 + daca_x_2007 + daca_x_2008 + daca_x_2009 + daca_x_2010 + daca_x_2013 + daca_x_2014 + daca_x_2015 + daca_x_2016 + AGE + age_sq + female + married + educ_HS + educ_someCol + educ_BA + C(state_fe) + C(year_fe)'

model_event = smf.wls(event_formula, data=df, weights=df['PERWT'])
results_event = model_event.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("\nEvent Study Coefficients (reference year: 2011):")
event_vars = ['daca_x_2006', 'daca_x_2007', 'daca_x_2008', 'daca_x_2009', 'daca_x_2010',
              'daca_x_2013', 'daca_x_2014', 'daca_x_2015', 'daca_x_2016']
for var in event_vars:
    coef = results_event.params[var]
    se = results_event.bse[var]
    pval = results_event.pvalues[var]
    sig = '*' if pval < 0.05 else ''
    print(f"  {var}: {coef:10.6f} (SE: {se:.6f}){sig}")

# =============================================================================
# STEP 9: Save Results for Report
# =============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Create results dictionary
results_dict = {
    'main_effect': main_effect,
    'main_se': main_se,
    'main_pval': main_pval,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper,
    'sample_size': len(df),
    'n_daca_eligible': df['daca_eligible'].sum(),
    'n_control': (df['daca_eligible']==0).sum(),
    'baseline_ft_rate': df[df['daca_eligible']==1]['fulltime'].mean(),
}

# Save detailed results
results_summary = pd.DataFrame({
    'Model': ['Basic DiD', 'Demographic Controls', 'Full Controls', 'State/Year FE'],
    'Effect': [results1.params['daca_x_post'], results2.params['daca_x_post'],
               results3.params['daca_x_post'], results4.params['daca_x_post']],
    'SE': [results1.bse['daca_x_post'], results2.bse['daca_x_post'],
           results3.bse['daca_x_post'], results4.bse['daca_x_post']],
    'p-value': [results1.pvalues['daca_x_post'], results2.pvalues['daca_x_post'],
                results3.pvalues['daca_x_post'], results4.pvalues['daca_x_post']]
})
results_summary.to_csv('results_main.csv', index=False)
print("Saved main results to results_main.csv")

# Robustness results
robustness_summary = pd.DataFrame({
    'Specification': ['Ages 18-40', 'Men Only', 'Women Only', 'Employment Outcome', 'Narrow Bandwidth'],
    'Effect': [results_r1.params['daca_x_post'], results_men.params['daca_x_post'],
               results_women.params['daca_x_post'], results_emp.params['daca_x_post'],
               results_narrow.params['daca_x_post_narrow']],
    'SE': [results_r1.bse['daca_x_post'], results_men.bse['daca_x_post'],
           results_women.bse['daca_x_post'], results_emp.bse['daca_x_post'],
           results_narrow.bse['daca_x_post_narrow']]
})
robustness_summary.to_csv('results_robustness.csv', index=False)
print("Saved robustness results to results_robustness.csv")

# Event study results
event_results = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [results_event.params.get(f'daca_x_{yr}', 0) for yr in [2006, 2007, 2008, 2009, 2010]] + [0] +
                   [results_event.params.get(f'daca_x_{yr}', 0) for yr in [2013, 2014, 2015, 2016]],
    'SE': [results_event.bse.get(f'daca_x_{yr}', 0) for yr in [2006, 2007, 2008, 2009, 2010]] + [0] +
          [results_event.bse.get(f'daca_x_{yr}', 0) for yr in [2013, 2014, 2015, 2016]]
})
event_results.to_csv('results_event_study.csv', index=False)
print("Saved event study results to results_event_study.csv")

# Descriptive statistics
desc_by_group = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)
desc_by_group.to_csv('descriptive_stats.csv')
print("Saved descriptive statistics to descriptive_stats.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
