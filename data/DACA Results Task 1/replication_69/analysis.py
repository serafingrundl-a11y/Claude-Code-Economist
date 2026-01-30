"""
DACA Replication Study - Analysis Script
=========================================
Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States?

Author: Anonymous (Replication ID: 69)
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

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# =============================================================================
# STEP 1: Load Data in Chunks with Early Filtering
# =============================================================================
print("\n[1] Loading and filtering data in chunks...")

# Read only necessary columns to save memory
cols_needed = [
    'YEAR', 'PERWT', 'STATEFIP', 'AGE', 'SEX', 'BIRTHYR', 'BIRTHQTR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'UHRSWORK', 'MARST'
]

# Process in chunks and filter immediately
chunks = []
chunk_size = 1000000

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size, low_memory=False):
    # Apply early filters to reduce memory
    # Keep only relevant years (exclude 2012)
    chunk = chunk[chunk['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])]

    # Hispanic-Mexican only
    chunk = chunk[chunk['HISPAN'] == 1]

    # Born in Mexico only
    chunk = chunk[chunk['BPL'] == 200]

    # Non-citizens only
    chunk = chunk[chunk['CITIZEN'] == 3]

    # Valid immigration year (arrived by 2007)
    chunk = chunk[(chunk['YRIMMIG'] > 0) & (chunk['YRIMMIG'] <= 2007)]

    if len(chunk) > 0:
        chunks.append(chunk)

    gc.collect()

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"Filtered observations loaded: {len(df):,}")

# =============================================================================
# STEP 2: Additional Sample Restrictions
# =============================================================================
print("\n[2] Applying additional restrictions...")

# Calculate age at arrival
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Restriction: Arrived before age 16 (childhood arrivals only)
df = df[df['age_at_arrival'] < 16]
print(f"After arrived before age 16 restriction: {len(df):,}")

# =============================================================================
# STEP 3: Construct DACA Eligibility Variables
# =============================================================================
print("\n[3] Constructing DACA eligibility indicators...")

# Under 31 as of June 15, 2012 means born after June 15, 1981
# Conservative: BIRTHYR >= 1982 guarantees under 31
# For 1981: Q3 (July-Sept) and Q4 (Oct-Dec) born were under 31 in June
df['under_31_june2012'] = (df['BIRTHYR'] >= 1982) | \
                          ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'].isin([3, 4])))

# DACA eligible: under 31 as of June 2012
df['daca_eligible'] = df['under_31_june2012'].astype(int)

print(f"DACA eligible (treatment): {df['daca_eligible'].sum():,}")
print(f"Not DACA eligible (control): {(1 - df['daca_eligible']).sum():,}")

# =============================================================================
# STEP 4: Create Analysis Variables
# =============================================================================
print("\n[4] Creating analysis variables...")

# Post-DACA indicator (years 2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term: Treatment x Post (DiD estimator)
df['treat_post'] = df['daca_eligible'] * df['post']

# Outcome: Full-time employment (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# =============================================================================
# STEP 5: Restrict to Working-Age Population
# =============================================================================
print("\n[5] Restricting to working-age population...")

# Restrict to working age (18-50) at survey time for better overlap
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 50)]
print(f"After working-age restriction (18-50): {len(df):,}")

# =============================================================================
# STEP 6: Create Control Variables
# =============================================================================
print("\n[6] Creating control variables...")

# Education categories
df['educ_lesshs'] = (df['EDUCD'] < 62).astype(int)  # Less than high school
df['educ_hs'] = ((df['EDUCD'] >= 62) & (df['EDUCD'] <= 64)).astype(int)  # HS diploma
df['educ_somecoll'] = ((df['EDUCD'] >= 65) & (df['EDUCD'] < 101)).astype(int)  # Some college
df['educ_coll'] = (df['EDUCD'] >= 101).astype(int)  # Bachelor's or higher

# Gender
df['female'] = (df['SEX'] == 2).astype(int)

# Marital status
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Age polynomials
df['age_sq'] = df['AGE'] ** 2

# Years since arrival
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']

# =============================================================================
# STEP 7: Summary Statistics
# =============================================================================
print("\n[7] Summary Statistics")
print("=" * 80)

# Overall sample
print(f"\nFinal analysis sample size: {len(df):,}")
print(f"Weighted population: {df['PERWT'].sum():,.0f}")

# By treatment status
print("\nBy Treatment Status:")
for treat in [0, 1]:
    subset = df[df['daca_eligible'] == treat]
    label = "DACA Eligible (Treatment)" if treat == 1 else "Not Eligible (Control)"
    print(f"\n{label}:")
    print(f"  N: {len(subset):,}")
    print(f"  Weighted N: {subset['PERWT'].sum():,.0f}")
    print(f"  Mean Age: {np.average(subset['AGE'], weights=subset['PERWT']):.1f}")
    print(f"  Mean Full-time Rate: {np.average(subset['fulltime'], weights=subset['PERWT']):.3f}")
    print(f"  Mean Employment Rate: {np.average(subset['employed'], weights=subset['PERWT']):.3f}")

# By period
print("\nBy Period:")
for post in [0, 1]:
    subset = df[df['post'] == post]
    label = "Post-DACA (2013-2016)" if post == 1 else "Pre-DACA (2006-2011)"
    print(f"\n{label}:")
    print(f"  N: {len(subset):,}")
    print(f"  Mean Full-time Rate: {np.average(subset['fulltime'], weights=subset['PERWT']):.3f}")

# 2x2 table for DiD
print("\n" + "=" * 80)
print("DIFFERENCE-IN-DIFFERENCES TABLE: Full-Time Employment Rate")
print("=" * 80)

did_table = pd.DataFrame(index=['Pre-DACA', 'Post-DACA', 'Difference'],
                         columns=['Control', 'Treatment', 'Diff (T-C)'])

for treat, treat_label in [(0, 'Control'), (1, 'Treatment')]:
    for post, post_label in [(0, 'Pre-DACA'), (1, 'Post-DACA')]:
        subset = df[(df['daca_eligible'] == treat) & (df['post'] == post)]
        rate = np.average(subset['fulltime'], weights=subset['PERWT'])
        did_table.loc[post_label, treat_label] = rate

# Calculate differences
did_table.loc['Pre-DACA', 'Diff (T-C)'] = did_table.loc['Pre-DACA', 'Treatment'] - did_table.loc['Pre-DACA', 'Control']
did_table.loc['Post-DACA', 'Diff (T-C)'] = did_table.loc['Post-DACA', 'Treatment'] - did_table.loc['Post-DACA', 'Control']
did_table.loc['Difference', 'Control'] = did_table.loc['Post-DACA', 'Control'] - did_table.loc['Pre-DACA', 'Control']
did_table.loc['Difference', 'Treatment'] = did_table.loc['Post-DACA', 'Treatment'] - did_table.loc['Pre-DACA', 'Treatment']
did_table.loc['Difference', 'Diff (T-C)'] = did_table.loc['Difference', 'Treatment'] - did_table.loc['Difference', 'Control']

print(did_table.to_string())
print(f"\nDifference-in-Differences Estimate: {did_table.loc['Difference', 'Diff (T-C)']:.4f}")

# =============================================================================
# STEP 8: Main Regression Analysis
# =============================================================================
print("\n" + "=" * 80)
print("REGRESSION ANALYSIS")
print("=" * 80)

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic Difference-in-Differences ---")
model1 = smf.wls('fulltime ~ daca_eligible + post + treat_post',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls('fulltime ~ daca_eligible + post + treat_post + AGE + age_sq + female + married',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with demographic + education controls
print("\n--- Model 3: DiD with Demographics and Education ---")
model3 = smf.wls('fulltime ~ daca_eligible + post + treat_post + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_coll',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with Year Fixed Effects ---")
df['year_fe'] = df['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_coll + C(YEAR)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Extract treat_post coefficient
print(f"\ntreat_post coefficient: {model4.params['treat_post']:.5f}")
print(f"Standard Error: {model4.bse['treat_post']:.5f}")
print(f"t-statistic: {model4.tvalues['treat_post']:.3f}")
print(f"p-value: {model4.pvalues['treat_post']:.4f}")
ci = model4.conf_int().loc['treat_post']
print(f"95% CI: [{ci[0]:.5f}, {ci[1]:.5f}]")

# Model 5: Full model with state fixed effects
print("\n--- Model 5: DiD with Year and State Fixed Effects (PREFERRED MODEL) ---")
model5 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_coll + C(YEAR) + C(STATEFIP)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

print(f"\nPREFERRED ESTIMATE (Model 5):")
print(f"treat_post coefficient: {model5.params['treat_post']:.5f}")
print(f"Standard Error: {model5.bse['treat_post']:.5f}")
print(f"t-statistic: {model5.tvalues['treat_post']:.3f}")
print(f"p-value: {model5.pvalues['treat_post']:.4f}")
ci5 = model5.conf_int().loc['treat_post']
print(f"95% CI: [{ci5[0]:.5f}, {ci5[1]:.5f}]")

# =============================================================================
# STEP 9: Robustness Checks
# =============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# Robustness 1: Alternative age restriction (18-45)
print("\n--- Robustness 1: Restricted Age Range (18-45) ---")
df_robust1 = df[(df['AGE'] >= 18) & (df['AGE'] <= 45)]
model_r1 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_coll + C(YEAR) + C(STATEFIP)',
                    data=df_robust1, weights=df_robust1['PERWT']).fit(cov_type='HC1')
print(f"treat_post coefficient: {model_r1.params['treat_post']:.5f} (SE: {model_r1.bse['treat_post']:.5f})")
print(f"N = {len(df_robust1):,}")

# Robustness 2: Men only
print("\n--- Robustness 2: Men Only ---")
df_men = df[df['female'] == 0]
model_r2 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + married + educ_hs + educ_somecoll + educ_coll + C(YEAR) + C(STATEFIP)',
                    data=df_men, weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"treat_post coefficient: {model_r2.params['treat_post']:.5f} (SE: {model_r2.bse['treat_post']:.5f})")
print(f"N = {len(df_men):,}")

# Robustness 3: Women only
print("\n--- Robustness 3: Women Only ---")
df_women = df[df['female'] == 1]
model_r3 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + married + educ_hs + educ_somecoll + educ_coll + C(YEAR) + C(STATEFIP)',
                    data=df_women, weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"treat_post coefficient: {model_r3.params['treat_post']:.5f} (SE: {model_r3.bse['treat_post']:.5f})")
print(f"N = {len(df_women):,}")

# Robustness 4: Employment (any) instead of full-time
print("\n--- Robustness 4: Any Employment as Outcome ---")
model_r4 = smf.wls('employed ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_coll + C(YEAR) + C(STATEFIP)',
                    data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"treat_post coefficient: {model_r4.params['treat_post']:.5f} (SE: {model_r4.bse['treat_post']:.5f})")

# Robustness 5: Linear Probability Model with clustering at state level
print("\n--- Robustness 5: Clustered Standard Errors at State Level ---")
model_r5 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_coll + C(YEAR) + C(STATEFIP)',
                    data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"treat_post coefficient: {model_r5.params['treat_post']:.5f} (SE: {model_r5.bse['treat_post']:.5f})")
ci_r5 = model_r5.conf_int().loc['treat_post']
print(f"95% CI with clustering: [{ci_r5[0]:.5f}, {ci_r5[1]:.5f}]")

# =============================================================================
# STEP 10: Pre-Trends Check (Event Study)
# =============================================================================
print("\n" + "=" * 80)
print("EVENT STUDY / PRE-TRENDS CHECK")
print("=" * 80)

# Create year dummies interacted with treatment
for year in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'treat_year_{year}'] = (df['daca_eligible'] == 1) & (df['YEAR'] == year)
    df[f'treat_year_{year}'] = df[f'treat_year_{year}'].astype(int)

# Reference year: 2011 (year before DACA)
event_vars = ['treat_year_2006', 'treat_year_2007', 'treat_year_2008',
              'treat_year_2009', 'treat_year_2010',
              'treat_year_2013', 'treat_year_2014', 'treat_year_2015', 'treat_year_2016']

formula_event = 'fulltime ~ daca_eligible + ' + ' + '.join(event_vars) + ' + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_coll + C(YEAR) + C(STATEFIP)'

model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Reference: 2011):")
print("-" * 50)
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    print(f"{var}: {coef:8.5f} (SE: {se:.5f}, p: {pval:.3f})")

# =============================================================================
# STEP 11: Save Results for LaTeX Report
# =============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Create results summary
results_summary = {
    'preferred_effect': model5.params['treat_post'],
    'preferred_se': model5.bse['treat_post'],
    'preferred_ci_low': ci5[0],
    'preferred_ci_high': ci5[1],
    'preferred_pval': model5.pvalues['treat_post'],
    'n_obs': len(df),
    'n_treatment': df['daca_eligible'].sum(),
    'n_control': (1 - df['daca_eligible']).sum(),
    'weighted_n': df['PERWT'].sum()
}

# Save to file
with open('results_summary.txt', 'w') as f:
    f.write("DACA Replication Study - Results Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write("PREFERRED ESTIMATE (Model 5: Full DiD with Year and State FE)\n")
    f.write("-" * 50 + "\n")
    f.write(f"Effect of DACA eligibility on full-time employment: {results_summary['preferred_effect']:.5f}\n")
    f.write(f"Standard Error (robust): {results_summary['preferred_se']:.5f}\n")
    f.write(f"95% Confidence Interval: [{results_summary['preferred_ci_low']:.5f}, {results_summary['preferred_ci_high']:.5f}]\n")
    f.write(f"p-value: {results_summary['preferred_pval']:.4f}\n")
    f.write(f"\nSample Size: {results_summary['n_obs']:,}\n")
    f.write(f"Treatment Group (DACA eligible): {results_summary['n_treatment']:,}\n")
    f.write(f"Control Group (Not eligible): {results_summary['n_control']:,}\n")
    f.write(f"Weighted Population: {results_summary['weighted_n']:,.0f}\n")

print("\nResults saved to results_summary.txt")

# Create regression table for LaTeX
print("\n" + "=" * 80)
print("REGRESSION TABLE FOR LATEX")
print("=" * 80)

# Summary comparison table
print("\nModel Comparison:")
print("-" * 70)
print(f"{'Model':<40} {'Coefficient':>12} {'SE':>10} {'p-value':>10}")
print("-" * 70)
print(f"{'(1) Basic DiD':<40} {model1.params['treat_post']:>12.5f} {model1.bse['treat_post']:>10.5f} {model1.pvalues['treat_post']:>10.4f}")
print(f"{'(2) + Demographics':<40} {model2.params['treat_post']:>12.5f} {model2.bse['treat_post']:>10.5f} {model2.pvalues['treat_post']:>10.4f}")
print(f"{'(3) + Education':<40} {model3.params['treat_post']:>12.5f} {model3.bse['treat_post']:>10.5f} {model3.pvalues['treat_post']:>10.4f}")
print(f"{'(4) + Year FE':<40} {model4.params['treat_post']:>12.5f} {model4.bse['treat_post']:>10.5f} {model4.pvalues['treat_post']:>10.4f}")
print(f"{'(5) + State FE [PREFERRED]':<40} {model5.params['treat_post']:>12.5f} {model5.bse['treat_post']:>10.5f} {model5.pvalues['treat_post']:>10.4f}")
print("-" * 70)

# Robustness comparison
print("\nRobustness Checks:")
print("-" * 70)
print(f"{'Check':<40} {'Coefficient':>12} {'SE':>10} {'N':>12}")
print("-" * 70)
print(f"{'Main (Model 5)':<40} {model5.params['treat_post']:>12.5f} {model5.bse['treat_post']:>10.5f} {len(df):>12,}")
print(f"{'Age 18-45':<40} {model_r1.params['treat_post']:>12.5f} {model_r1.bse['treat_post']:>10.5f} {len(df_robust1):>12,}")
print(f"{'Men Only':<40} {model_r2.params['treat_post']:>12.5f} {model_r2.bse['treat_post']:>10.5f} {len(df_men):>12,}")
print(f"{'Women Only':<40} {model_r3.params['treat_post']:>12.5f} {model_r3.bse['treat_post']:>10.5f} {len(df_women):>12,}")
print(f"{'Any Employment Outcome':<40} {model_r4.params['treat_post']:>12.5f} {model_r4.bse['treat_post']:>10.5f} {len(df):>12,}")
print(f"{'Clustered SE (State)':<40} {model_r5.params['treat_post']:>12.5f} {model_r5.bse['treat_post']:>10.5f} {len(df):>12,}")
print("-" * 70)

# Save detailed statistics for LaTeX tables
stats_for_latex = {
    'model1': {'coef': model1.params['treat_post'], 'se': model1.bse['treat_post'], 'pval': model1.pvalues['treat_post'], 'n': len(df)},
    'model2': {'coef': model2.params['treat_post'], 'se': model2.bse['treat_post'], 'pval': model2.pvalues['treat_post'], 'n': len(df)},
    'model3': {'coef': model3.params['treat_post'], 'se': model3.bse['treat_post'], 'pval': model3.pvalues['treat_post'], 'n': len(df)},
    'model4': {'coef': model4.params['treat_post'], 'se': model4.bse['treat_post'], 'pval': model4.pvalues['treat_post'], 'n': len(df)},
    'model5': {'coef': model5.params['treat_post'], 'se': model5.bse['treat_post'], 'pval': model5.pvalues['treat_post'], 'n': len(df)},
}

# DiD table values
did_values = {
    'pre_control': did_table.loc['Pre-DACA', 'Control'],
    'pre_treat': did_table.loc['Pre-DACA', 'Treatment'],
    'post_control': did_table.loc['Post-DACA', 'Control'],
    'post_treat': did_table.loc['Post-DACA', 'Treatment'],
    'did_estimate': did_table.loc['Difference', 'Diff (T-C)']
}

# Event study coefficients
event_coefs = {}
for var in event_vars:
    event_coefs[var] = {
        'coef': model_event.params[var],
        'se': model_event.bse[var],
        'ci_low': model_event.conf_int().loc[var, 0],
        'ci_high': model_event.conf_int().loc[var, 1]
    }

# Save all for LaTeX generation
import json
with open('latex_data.json', 'w') as f:
    json.dump({
        'stats': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in stats_for_latex.items()},
        'did': {k: float(v) for k, v in did_values.items()},
        'event': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in event_coefs.items()},
        'robustness': {
            'age_restricted': {'coef': float(model_r1.params['treat_post']), 'se': float(model_r1.bse['treat_post']), 'n': len(df_robust1)},
            'men': {'coef': float(model_r2.params['treat_post']), 'se': float(model_r2.bse['treat_post']), 'n': len(df_men)},
            'women': {'coef': float(model_r3.params['treat_post']), 'se': float(model_r3.bse['treat_post']), 'n': len(df_women)},
            'employed': {'coef': float(model_r4.params['treat_post']), 'se': float(model_r4.bse['treat_post']), 'n': len(df)},
            'clustered': {'coef': float(model_r5.params['treat_post']), 'se': float(model_r5.bse['treat_post']), 'n': len(df),
                         'ci_low': float(ci_r5[0]), 'ci_high': float(ci_r5[1])}
        },
        'sample_info': {
            'n_total': len(df),
            'n_treat': int(df['daca_eligible'].sum()),
            'n_control': int((1 - df['daca_eligible']).sum()),
            'weighted_n': float(df['PERWT'].sum())
        }
    }, f, indent=2)

print("\nLatex data saved to latex_data.json")

# Additional descriptive statistics
print("\n" + "=" * 80)
print("DETAILED DESCRIPTIVE STATISTICS")
print("=" * 80)

desc_vars = ['AGE', 'female', 'married', 'educ_lesshs', 'educ_hs', 'educ_somecoll', 'educ_coll',
             'fulltime', 'employed', 'years_in_us']

print("\n{:<20} {:>12} {:>12} {:>12} {:>12}".format(
    'Variable', 'Control Mean', 'Treat Mean', 'Diff', 'p-value'))
print("-" * 70)

desc_stats = {}
for var in desc_vars:
    control = df[df['daca_eligible'] == 0]
    treat = df[df['daca_eligible'] == 1]

    mean_c = np.average(control[var], weights=control['PERWT'])
    mean_t = np.average(treat[var], weights=treat['PERWT'])
    diff = mean_t - mean_c

    # Simple t-test for difference
    t_stat, p_val = stats.ttest_ind(control[var], treat[var])

    print(f"{var:<20} {mean_c:>12.3f} {mean_t:>12.3f} {diff:>12.3f} {p_val:>12.4f}")
    desc_stats[var] = {'control': mean_c, 'treat': mean_t, 'diff': diff, 'pval': p_val}

# Save descriptive stats
with open('descriptive_stats.json', 'w') as f:
    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in desc_stats.items()}, f, indent=2)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
