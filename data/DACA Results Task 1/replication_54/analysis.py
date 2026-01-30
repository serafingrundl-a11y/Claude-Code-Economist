"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the United States.
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

# ============================================================================
# STEP 1: Load and Initial Data Exploration
# ============================================================================
print("\n1. LOADING DATA...")

# Load data in chunks due to size
dtypes = {
    'YEAR': 'int16',
    'SAMPLE': 'int32',
    'STATEFIP': 'int8',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'YRSUSA1': 'int8',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8',
    'EDUC': 'int8',
    'MARST': 'int8',
    'PERWT': 'float64'
}

# Read only necessary columns for efficiency
cols_needed = ['YEAR', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'YRSUSA1', 'EMPSTAT', 'UHRSWORK', 'EDUC', 'MARST', 'PERWT']

print("Reading data (this may take a while for ~34M rows)...")
df = pd.read_csv('data/data.csv', usecols=cols_needed, dtype=dtypes)
print(f"Total observations loaded: {len(df):,}")

# ============================================================================
# STEP 2: Sample Restriction - Hispanic-Mexican, Born in Mexico
# ============================================================================
print("\n2. SAMPLE RESTRICTION...")

# Check years in data
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# HISPAN = 1 for Mexican (per data dictionary)
# BPL = 200 for Mexico (per data dictionary)
print(f"\nUnique HISPAN values: {sorted(df['HISPAN'].unique())}")
print(f"Unique BPL values (sample): {sorted(df['BPL'].unique())[:20]}")

# Restrict to Hispanic-Mexican ethnicity (HISPAN == 1) AND born in Mexico (BPL == 200)
df_sample = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
print(f"After Hispanic-Mexican & Mexico-born restriction: {len(df_sample):,}")

# Further restrict to non-citizens (CITIZEN == 3)
# Per instructions: "Assume that anyone who is not a citizen and who has
# not received immigration papers is undocumented for DACA purposes"
# CITIZEN = 3 is "Not a citizen"
print(f"\nUnique CITIZEN values: {sorted(df_sample['CITIZEN'].unique())}")
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After non-citizen restriction: {len(df_sample):,}")

# ============================================================================
# STEP 3: Define DACA Eligibility
# ============================================================================
print("\n3. DEFINING DACA ELIGIBILITY...")

# DACA eligibility criteria (as of June 15, 2012):
# 1. Arrived in US before age 16
# 2. Born after June 15, 1981 (age < 31 on June 15, 2012)
# 3. Continuously in US since June 15, 2007 (at least 5 years by 2012)
# 4. Present in US on June 15, 2012 without lawful status

# Calculate age at arrival in US
# YRIMMIG = year of immigration
# BIRTHYR = birth year
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Criteria 1: Arrived before age 16
df_sample['arrived_before_16'] = df_sample['age_at_arrival'] < 16

# Criteria 2: Born after June 15, 1981 (use BIRTHYR > 1981 as approximation)
# More precisely: age < 31 on June 15, 2012
# For those born in 1981, only Q3/Q4 would be under 31
# Conservative approach: BIRTHYR >= 1982
df_sample['born_after_1981'] = df_sample['BIRTHYR'] >= 1982

# Criteria 3: In US since June 15, 2007 (YRIMMIG <= 2007)
df_sample['in_us_since_2007'] = df_sample['YRIMMIG'] <= 2007

# Combined eligibility indicator
df_sample['daca_eligible'] = (df_sample['arrived_before_16'] &
                               df_sample['born_after_1981'] &
                               df_sample['in_us_since_2007'])

print(f"Observations with valid YRIMMIG: {(df_sample['YRIMMIG'] > 0).sum():,}")

# Filter to those with valid immigration year data
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
print(f"After requiring valid YRIMMIG: {len(df_sample):,}")

print(f"\nDACA Eligibility Distribution:")
print(df_sample['daca_eligible'].value_counts())
print(f"Eligible: {df_sample['daca_eligible'].sum():,} ({df_sample['daca_eligible'].mean()*100:.1f}%)")

# ============================================================================
# STEP 4: Define Outcome - Full-Time Employment
# ============================================================================
print("\n4. DEFINING OUTCOME VARIABLE...")

# Full-time employment: UHRSWORK >= 35
# UHRSWORK = usual hours worked per week
# EMPSTAT = 1 for employed

print(f"EMPSTAT distribution:")
print(df_sample['EMPSTAT'].value_counts().sort_index())

# Define full-time employment (35+ hours per week)
df_sample['fulltime'] = ((df_sample['UHRSWORK'] >= 35)).astype(int)

print(f"\nFull-time employment rate: {df_sample['fulltime'].mean()*100:.1f}%")

# ============================================================================
# STEP 5: Define Treatment Period
# ============================================================================
print("\n5. DEFINING TREATMENT PERIOD...")

# DACA was implemented June 15, 2012
# Post-treatment period: 2013-2016 (2012 is ambiguous)
# Pre-treatment period: 2006-2011

df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
print(f"\nPre-period (2006-2011): {(df_sample['post'] == 0).sum():,}")
print(f"Post-period (2013-2016): {(df_sample['post'] == 1).sum():,}")

# Exclude 2012 due to ambiguity (DACA implemented mid-year)
df_analysis = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

# ============================================================================
# STEP 6: Restrict to Working-Age Population
# ============================================================================
print("\n6. RESTRICTING TO WORKING-AGE POPULATION...")

# Typical working age: 16-64
# For DACA eligible: need to be old enough to work but young enough to be eligible
# Age restriction: 18-40 (reasonable working-age range that includes DACA-eligible)

df_analysis = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 50)].copy()
print(f"After age restriction (18-50): {len(df_analysis):,}")

# ============================================================================
# STEP 7: Create Control Variables
# ============================================================================
print("\n7. CREATING CONTROL VARIABLES...")

# Education categories
df_analysis['educ_cat'] = pd.cut(df_analysis['EDUC'],
                                  bins=[-1, 2, 5, 6, 11],
                                  labels=['less_hs', 'some_hs', 'hs_grad', 'some_college_plus'])

# Age squared
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Sex indicator (female = 1)
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Married indicator
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)

print(f"Control variables created.")

# ============================================================================
# STEP 8: Descriptive Statistics
# ============================================================================
print("\n8. DESCRIPTIVE STATISTICS...")

print("\n--- Summary by DACA Eligibility and Period ---")
summary = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'PERWT': 'sum'
}).round(3)
print(summary)

# Weighted means
print("\n--- Weighted Full-Time Employment Rates ---")
for eligible in [True, False]:
    for post in [0, 1]:
        subset = df_analysis[(df_analysis['daca_eligible'] == eligible) &
                            (df_analysis['post'] == post)]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        n = len(subset)
        print(f"Eligible={eligible}, Post={post}: {weighted_mean:.4f} (n={n:,})")

# ============================================================================
# STEP 9: Difference-in-Differences Analysis
# ============================================================================
print("\n9. DIFFERENCE-IN-DIFFERENCES ANALYSIS...")

# Create interaction term
df_analysis['daca_eligible_int'] = df_analysis['daca_eligible'].astype(int)
df_analysis['did'] = df_analysis['daca_eligible_int'] * df_analysis['post']

# Simple DiD without controls
print("\n--- Simple DiD (No Controls) ---")
model1 = smf.wls('fulltime ~ daca_eligible_int + post + did',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary().tables[1])

# DiD with demographic controls
print("\n--- DiD with Demographic Controls ---")
model2 = smf.wls('fulltime ~ daca_eligible_int + post + did + AGE + age_sq + female + married',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])

# DiD with demographic + education controls
print("\n--- DiD with Demographic + Education Controls ---")
df_analysis_edu = df_analysis.dropna(subset=['educ_cat'])
model3 = smf.wls('fulltime ~ daca_eligible_int + post + did + AGE + age_sq + female + married + C(educ_cat)',
                  data=df_analysis_edu,
                  weights=df_analysis_edu['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(results3.summary().tables[1])

# DiD with Year Fixed Effects
print("\n--- DiD with Year Fixed Effects ---")
model4 = smf.wls('fulltime ~ daca_eligible_int + did + AGE + age_sq + female + married + C(YEAR)',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results4 = model4.fit(cov_type='HC1')
print(results4.summary().tables[1])

# DiD with Year and State Fixed Effects
print("\n--- DiD with Year and State Fixed Effects ---")
model5 = smf.wls('fulltime ~ daca_eligible_int + did + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results5 = model5.fit(cov_type='HC1')
print("\nKey coefficients:")
print(f"DACA Eligible: {results5.params['daca_eligible_int']:.4f} (SE: {results5.bse['daca_eligible_int']:.4f})")
print(f"DiD (Treatment Effect): {results5.params['did']:.4f} (SE: {results5.bse['did']:.4f})")

# ============================================================================
# STEP 10: Preferred Specification - Clustered Standard Errors by State
# ============================================================================
print("\n10. PREFERRED SPECIFICATION WITH CLUSTERED SEs...")

# Full model with state-clustered standard errors
model_pref = smf.wls('fulltime ~ daca_eligible_int + did + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                      data=df_analysis,
                      weights=df_analysis['PERWT'])
results_pref = model_pref.fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

print("\n=== PREFERRED SPECIFICATION RESULTS ===")
print(f"Dependent Variable: Full-Time Employment (35+ hrs/week)")
print(f"Method: Weighted Least Squares with State-Clustered SEs")
print(f"Sample: Hispanic-Mexican, Mexico-born, Non-citizens, Age 18-50")
print(f"Observations: {len(df_analysis):,}")
print(f"\nKey Results:")
print(f"DiD Treatment Effect: {results_pref.params['did']:.4f}")
print(f"Standard Error: {results_pref.bse['did']:.4f}")
print(f"95% CI: [{results_pref.conf_int().loc['did', 0]:.4f}, {results_pref.conf_int().loc['did', 1]:.4f}]")
print(f"t-statistic: {results_pref.tvalues['did']:.3f}")
print(f"p-value: {results_pref.pvalues['did']:.4f}")

# ============================================================================
# STEP 11: Robustness Checks
# ============================================================================
print("\n11. ROBUSTNESS CHECKS...")

# Check 1: Alternative age restriction (16-35, more relevant for DACA)
print("\n--- Robustness: Age 16-35 ---")
df_robust1 = df_sample[(df_sample['YEAR'] != 2012) &
                        (df_sample['AGE'] >= 16) &
                        (df_sample['AGE'] <= 35)].copy()
df_robust1['daca_eligible_int'] = df_robust1['daca_eligible'].astype(int)
df_robust1['post'] = (df_robust1['YEAR'] >= 2013).astype(int)
df_robust1['did'] = df_robust1['daca_eligible_int'] * df_robust1['post']
df_robust1['age_sq'] = df_robust1['AGE'] ** 2
df_robust1['female'] = (df_robust1['SEX'] == 2).astype(int)
df_robust1['married'] = (df_robust1['MARST'] <= 2).astype(int)

model_r1 = smf.wls('fulltime ~ daca_eligible_int + did + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                    data=df_robust1,
                    weights=df_robust1['PERWT'])
results_r1 = model_r1.fit(cov_type='cluster', cov_kwds={'groups': df_robust1['STATEFIP']})
print(f"DiD Effect: {results_r1.params['did']:.4f} (SE: {results_r1.bse['did']:.4f})")
print(f"N = {len(df_robust1):,}")

# Check 2: Include 2012 as post-treatment
print("\n--- Robustness: Include 2012 as Post-Treatment ---")
df_robust2 = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 50)].copy()
df_robust2['daca_eligible_int'] = df_robust2['daca_eligible'].astype(int)
df_robust2['post'] = (df_robust2['YEAR'] >= 2012).astype(int)
df_robust2['did'] = df_robust2['daca_eligible_int'] * df_robust2['post']
df_robust2['age_sq'] = df_robust2['AGE'] ** 2
df_robust2['female'] = (df_robust2['SEX'] == 2).astype(int)
df_robust2['married'] = (df_robust2['MARST'] <= 2).astype(int)

model_r2 = smf.wls('fulltime ~ daca_eligible_int + did + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                    data=df_robust2,
                    weights=df_robust2['PERWT'])
results_r2 = model_r2.fit(cov_type='cluster', cov_kwds={'groups': df_robust2['STATEFIP']})
print(f"DiD Effect: {results_r2.params['did']:.4f} (SE: {results_r2.bse['did']:.4f})")
print(f"N = {len(df_robust2):,}")

# Check 3: Unweighted regression
print("\n--- Robustness: Unweighted Regression ---")
model_r3 = smf.ols('fulltime ~ daca_eligible_int + did + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                    data=df_analysis)
results_r3 = model_r3.fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"DiD Effect: {results_r3.params['did']:.4f} (SE: {results_r3.bse['did']:.4f})")

# ============================================================================
# STEP 12: Event Study / Pre-Trends Analysis
# ============================================================================
print("\n12. EVENT STUDY / PRE-TRENDS ANALYSIS...")

# Create year-specific treatment effects
df_analysis['year_dummy'] = df_analysis['YEAR'].astype(str)
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]

# Create interaction terms for each year (reference: 2011)
for yr in years:
    if yr != 2011:  # 2011 is reference year
        df_analysis[f'eligible_x_{yr}'] = ((df_analysis['YEAR'] == yr) &
                                            df_analysis['daca_eligible']).astype(int)

# Event study regression
formula_es = 'fulltime ~ daca_eligible_int + ' + ' + '.join([f'eligible_x_{yr}' for yr in years if yr != 2011]) + \
             ' + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)'
model_es = smf.wls(formula_es, data=df_analysis, weights=df_analysis['PERWT'])
results_es = model_es.fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

print("\nEvent Study Coefficients (Reference: 2011):")
for yr in years:
    if yr != 2011:
        coef = results_es.params[f'eligible_x_{yr}']
        se = results_es.bse[f'eligible_x_{yr}']
        print(f"Year {yr}: {coef:.4f} (SE: {se:.4f})")

# ============================================================================
# STEP 13: Summary Statistics Table
# ============================================================================
print("\n13. GENERATING SUMMARY STATISTICS TABLE...")

summary_stats = []
for eligible in [True, False]:
    for period in ['Pre (2006-2011)', 'Post (2013-2016)']:
        post_val = 1 if 'Post' in period else 0
        subset = df_analysis[(df_analysis['daca_eligible'] == eligible) &
                            (df_analysis['post'] == post_val)]

        stats_dict = {
            'Group': 'DACA Eligible' if eligible else 'Non-Eligible',
            'Period': period,
            'N': len(subset),
            'Full-Time Emp. Rate': subset['fulltime'].mean(),
            'Mean Age': subset['AGE'].mean(),
            'Female %': subset['female'].mean() * 100,
            'Married %': subset['married'].mean() * 100
        }
        summary_stats.append(stats_dict)

summary_df = pd.DataFrame(summary_stats)
print(summary_df.to_string(index=False))

# ============================================================================
# STEP 14: Save Results
# ============================================================================
print("\n14. SAVING RESULTS...")

# Save key results for report
results_dict = {
    'preferred_effect': results_pref.params['did'],
    'preferred_se': results_pref.bse['did'],
    'preferred_ci_low': results_pref.conf_int().loc['did', 0],
    'preferred_ci_high': results_pref.conf_int().loc['did', 1],
    'preferred_pvalue': results_pref.pvalues['did'],
    'n_obs': len(df_analysis),
    'n_eligible': df_analysis['daca_eligible'].sum(),
    'n_not_eligible': (~df_analysis['daca_eligible']).sum()
}

# Save to file
with open('results_summary.txt', 'w') as f:
    f.write("DACA REPLICATION - KEY RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write("PREFERRED SPECIFICATION:\n")
    f.write(f"Treatment Effect (DiD): {results_dict['preferred_effect']:.4f}\n")
    f.write(f"Standard Error: {results_dict['preferred_se']:.4f}\n")
    f.write(f"95% CI: [{results_dict['preferred_ci_low']:.4f}, {results_dict['preferred_ci_high']:.4f}]\n")
    f.write(f"P-value: {results_dict['preferred_pvalue']:.4f}\n\n")
    f.write(f"Sample Size: {results_dict['n_obs']:,}\n")
    f.write(f"DACA Eligible: {results_dict['n_eligible']:,}\n")
    f.write(f"Not Eligible: {results_dict['n_not_eligible']:,}\n")

print("Results saved to results_summary.txt")

# ============================================================================
# STEP 15: Export Regression Tables for LaTeX
# ============================================================================
print("\n15. EXPORTING REGRESSION TABLES...")

# Create a summary table for all specifications
reg_table = pd.DataFrame({
    'Model': ['(1) Simple DiD', '(2) Demographics', '(3) Education',
              '(4) Year FE', '(5) Year+State FE'],
    'DiD Effect': [results1.params['did'], results2.params['did'],
                   results3.params['did'], results4.params['did'],
                   results_pref.params['did']],
    'Std. Error': [results1.bse['did'], results2.bse['did'],
                   results3.bse['did'], results4.bse['did'],
                   results_pref.bse['did']],
    'N': [len(df_analysis), len(df_analysis), len(df_analysis_edu),
          len(df_analysis), len(df_analysis)]
})
reg_table['t-stat'] = reg_table['DiD Effect'] / reg_table['Std. Error']

print("\nREGRESSION SUMMARY TABLE:")
print(reg_table.to_string(index=False))

# Save for LaTeX
reg_table.to_csv('regression_table.csv', index=False)
print("\nRegression table saved to regression_table.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
