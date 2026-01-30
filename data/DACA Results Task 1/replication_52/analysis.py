"""
DACA Replication Study: Effect of DACA Eligibility on Full-Time Employment
Among Hispanic-Mexican, Mexican-Born Individuals

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

import os
import gc

print("=" * 70)
print("DACA REPLICATION STUDY: Full-Time Employment Analysis")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA IN CHUNKS (due to large file size)
# =============================================================================
print("\n[1] Loading data in chunks...")

data_path = "data/data.csv"

# Define columns we need
cols_needed = [
    'YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'LABFORCE', 'UHRSWORK',
    'MARST', 'NCHILD', 'FAMSIZE'
]

# Process in chunks
print("Reading CSV file in chunks...")
chunks = []
chunk_size = 500000

for i, chunk in enumerate(pd.read_csv(data_path, usecols=cols_needed, chunksize=chunk_size, low_memory=False)):
    # Filter immediately to reduce memory
    # Keep only Hispanic-Mexican (HISPAN == 1), born in Mexico (BPL == 200), non-citizens (CITIZEN == 3)
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200) & (chunk['CITIZEN'] == 3)]
    if len(chunk) > 0:
        chunks.append(chunk)
    if (i + 1) % 20 == 0:
        print(f"  Processed {(i+1)*chunk_size:,} rows...")
    gc.collect()

print("Concatenating filtered chunks...")
df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"Total observations after initial filters: {len(df):,}")

# =============================================================================
# 2. ADDITIONAL SAMPLE RESTRICTIONS
# =============================================================================
print("\n[2] Applying additional sample restrictions...")

# Keep working-age individuals (16-40 years old)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 40)].copy()
print(f"After restricting to ages 16-40: {len(df):,}")

# Exclude 2012 (DACA implemented mid-year)
df = df[df['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df):,}")

# =============================================================================
# 3. DEFINE DACA ELIGIBILITY (TREATMENT)
# =============================================================================
print("\n[3] Defining DACA eligibility criteria...")

# Calculate age at arrival
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# For age cutoff as of June 15, 2012:
# Born after June 15, 1981 means under 31 on that date
def is_under_31_on_june15_2012(row):
    """Check if person was under 31 on June 15, 2012"""
    if row['BIRTHYR'] > 1981:
        return True
    elif row['BIRTHYR'] == 1981:
        return row['BIRTHQTR'] >= 3
    else:
        return False

# Criteria 1: Arrived before age 16
df['arrived_before_16'] = df['age_at_arrival'] < 16

# Criteria 2: Under 31 as of June 15, 2012
df['under_31_on_daca_date'] = df.apply(is_under_31_on_june15_2012, axis=1)

# Criteria 3: Arrived by 2007
df['arrived_by_2007'] = df['YRIMMIG'] <= 2007

# Valid immigration year
df['valid_yrimmig'] = df['YRIMMIG'] > 0

# DACA Eligible: meets all criteria
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['under_31_on_daca_date'] &
    df['arrived_by_2007'] &
    df['valid_yrimmig']
).astype(int)

print(f"\nDACA Eligibility Summary:")
print(f"  Arrived before age 16: {df['arrived_before_16'].sum():,}")
print(f"  Under 31 on June 15, 2012: {df['under_31_on_daca_date'].sum():,}")
print(f"  Arrived by 2007: {df['arrived_by_2007'].sum():,}")
print(f"  Valid immigration year: {df['valid_yrimmig'].sum():,}")
print(f"  DACA Eligible (all criteria): {df['daca_eligible'].sum():,}")

# =============================================================================
# 4. DEFINE OUTCOME VARIABLE
# =============================================================================
print("\n[4] Defining outcome variable...")

# Full-time employment: Usually works 35+ hours per week
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employment indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

print(f"Outcome variable (full-time employment):")
print(f"  Full-time (35+ hours): {df['fulltime'].mean()*100:.1f}%")
print(f"  Employed: {df['employed'].mean()*100:.1f}%")

# =============================================================================
# 5. DEFINE TIME PERIODS
# =============================================================================
print("\n[5] Defining time periods...")

# Pre-DACA: 2006-2011
# Post-DACA: 2013-2016
df['post'] = (df['YEAR'] >= 2013).astype(int)

print(f"Time period distribution:")
print(df.groupby(['YEAR', 'post']).size())

# =============================================================================
# 6. CREATE CONTROL VARIABLES
# =============================================================================
print("\n[6] Creating control variables...")

df['age_sq'] = df['AGE'] ** 2
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)
df['educ_college'] = (df['EDUC'] >= 7).astype(int)
df['has_children'] = (df['NCHILD'] > 0).astype(int)
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']
df.loc[df['years_in_us'] < 0, 'years_in_us'] = np.nan

print("Control variables created.")

# =============================================================================
# 7. FINAL SAMPLE PREPARATION
# =============================================================================
print("\n[7] Final sample preparation...")

df_analysis = df.dropna(subset=['fulltime', 'daca_eligible', 'post',
                                 'AGE', 'female', 'married', 'PERWT']).copy()

print(f"Final analysis sample: {len(df_analysis):,}")

# =============================================================================
# 8. SUMMARY STATISTICS
# =============================================================================
print("\n[8] Generating summary statistics...")

def weighted_mean(data, weights):
    return np.average(data, weights=weights)

summary_stats = []
for treat in [0, 1]:
    for period in [0, 1]:
        subset = df_analysis[(df_analysis['daca_eligible'] == treat) &
                             (df_analysis['post'] == period)]
        if len(subset) > 0:
            weights = subset['PERWT']
            stats_row = {
                'DACA_Eligible': treat,
                'Post_DACA': period,
                'N': len(subset),
                'Weighted_N': weights.sum(),
                'Fulltime_Pct': weighted_mean(subset['fulltime'], weights) * 100,
                'Employed_Pct': weighted_mean(subset['employed'], weights) * 100,
                'Age_Mean': weighted_mean(subset['AGE'], weights),
                'Female_Pct': weighted_mean(subset['female'], weights) * 100,
                'Married_Pct': weighted_mean(subset['married'], weights) * 100,
                'HS_Plus_Pct': weighted_mean(subset['educ_hs'], weights) * 100,
                'Has_Children_Pct': weighted_mean(subset['has_children'], weights) * 100,
            }
            summary_stats.append(stats_row)

summary_df = pd.DataFrame(summary_stats)
print("\nSummary Statistics by Group and Period:")
print(summary_df.to_string(index=False))
summary_df.to_csv('summary_statistics.csv', index=False)

# =============================================================================
# 9. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n[9] Running Difference-in-Differences Analysis...")

df_analysis['treat_post'] = df_analysis['daca_eligible'] * df_analysis['post']

# Model 1: Basic DiD
print("\nModel 1: Basic DiD")
model1 = smf.wls('fulltime ~ daca_eligible + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\nModel 2: DiD with Demographic Controls")
model2 = smf.wls('fulltime ~ daca_eligible + post + treat_post + AGE + age_sq + female + married',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])

# Model 3: DiD with full controls
print("\nModel 3: DiD with Full Controls")
model3 = smf.wls('fulltime ~ daca_eligible + post + treat_post + AGE + age_sq + female + married + educ_hs + has_children',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(results3.summary().tables[1])

# Model 4: DiD with year fixed effects
print("\nModel 4: DiD with Year Fixed Effects")
model4 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + has_children + C(YEAR)',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results4 = model4.fit(cov_type='HC1')

print("\nKey coefficients from Model 4:")
for var in ['daca_eligible', 'treat_post']:
    if var in results4.params.index:
        coef = results4.params[var]
        se = results4.bse[var]
        pval = results4.pvalues[var]
        print(f"  {var}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# Model 5: Full model with state and year FE (PREFERRED)
print("\nModel 5: DiD with State and Year Fixed Effects (PREFERRED)")
model5 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + has_children + C(STATEFIP) + C(YEAR)',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results5 = model5.fit(cov_type='HC1')

print("\nKey coefficients from Model 5 (PREFERRED SPECIFICATION):")
for var in ['daca_eligible', 'treat_post']:
    if var in results5.params.index:
        coef = results5.params[var]
        se = results5.bse[var]
        pval = results5.pvalues[var]
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        print(f"  {var}: {coef:.4f} (SE: {se:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}], p={pval:.4f})")

# =============================================================================
# 10. HETEROGENEITY ANALYSIS
# =============================================================================
print("\n[10] Heterogeneity Analysis...")

print("\nBy Gender:")
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    subset = df_analysis[df_analysis['SEX'] == sex_val]
    if len(subset) > 100:
        model = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + married + educ_hs + C(YEAR)',
                         data=subset,
                         weights=subset['PERWT'])
        res = model.fit(cov_type='HC1')
        coef = res.params['treat_post']
        se = res.bse['treat_post']
        print(f"  {sex_label}: DiD = {coef:.4f} (SE: {se:.4f}), N = {len(subset):,}")

print("\nBy Age Group:")
for age_min, age_max, label in [(16, 24, '16-24'), (25, 32, '25-32'), (33, 40, '33-40')]:
    subset = df_analysis[(df_analysis['AGE'] >= age_min) & (df_analysis['AGE'] <= age_max)]
    if len(subset) > 100:
        model = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + female + married + educ_hs + C(YEAR)',
                         data=subset,
                         weights=subset['PERWT'])
        res = model.fit(cov_type='HC1')
        if 'treat_post' in res.params.index:
            coef = res.params['treat_post']
            se = res.bse['treat_post']
            print(f"  Ages {label}: DiD = {coef:.4f} (SE: {se:.4f}), N = {len(subset):,}")

# =============================================================================
# 11. ROBUSTNESS CHECKS
# =============================================================================
print("\n[11] Robustness Checks...")

# Robustness 1: Any employment
print("\nRobustness 1: Any Employment")
model_r1 = smf.wls('employed ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + has_children + C(YEAR)',
                    data=df_analysis,
                    weights=df_analysis['PERWT'])
res_r1 = model_r1.fit(cov_type='HC1')
print(f"  DiD (Any Employment): {res_r1.params['treat_post']:.4f} (SE: {res_r1.bse['treat_post']:.4f})")

# Robustness 2: Including 2012
print("\nRobustness 2: Including 2012")
df_with_2012 = df.dropna(subset=['fulltime', 'daca_eligible', 'AGE', 'female', 'married', 'PERWT']).copy()
df_with_2012['post'] = (df_with_2012['YEAR'] >= 2013).astype(int)
df_with_2012['treat_post'] = df_with_2012['daca_eligible'] * df_with_2012['post']

model_r2 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + has_children + C(YEAR)',
                    data=df_with_2012,
                    weights=df_with_2012['PERWT'])
res_r2 = model_r2.fit(cov_type='HC1')
print(f"  DiD (including 2012): {res_r2.params['treat_post']:.4f} (SE: {res_r2.bse['treat_post']:.4f})")

# Robustness 3: Alternative age range
print("\nRobustness 3: Alternative Age Range (18-35)")
df_alt_age = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 35)].copy()
model_r3 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + has_children + C(YEAR)',
                    data=df_alt_age,
                    weights=df_alt_age['PERWT'])
res_r3 = model_r3.fit(cov_type='HC1')
print(f"  DiD (ages 18-35): {res_r3.params['treat_post']:.4f} (SE: {res_r3.bse['treat_post']:.4f})")

# Robustness 4: Unweighted
print("\nRobustness 4: Unweighted Regression")
model_r4 = smf.ols('fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + has_children + C(YEAR)',
                    data=df_analysis)
res_r4 = model_r4.fit(cov_type='HC1')
print(f"  DiD (unweighted): {res_r4.params['treat_post']:.4f} (SE: {res_r4.bse['treat_post']:.4f})")

# =============================================================================
# 12. EVENT STUDY / PARALLEL TRENDS
# =============================================================================
print("\n[12] Event Study Analysis...")

years = sorted(df_analysis['YEAR'].unique())
for year in years:
    if year != 2011:
        df_analysis[f'treat_x_{year}'] = (df_analysis['daca_eligible'] * (df_analysis['YEAR'] == year)).astype(int)

year_interactions = ' + '.join([f'treat_x_{year}' for year in years if year != 2011])
formula_es = f'fulltime ~ daca_eligible + {year_interactions} + AGE + age_sq + female + married + C(YEAR)'

model_es = smf.wls(formula_es, data=df_analysis, weights=df_analysis['PERWT'])
res_es = model_es.fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
event_study_results = []
for year in years:
    if year != 2011:
        var = f'treat_x_{year}'
        if var in res_es.params.index:
            coef = res_es.params[var]
            se = res_es.bse[var]
            ci_low = coef - 1.96 * se
            ci_high = coef + 1.96 * se
            print(f"  {year}: {coef:.4f} (SE: {se:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
            event_study_results.append({'Year': year, 'Coefficient': coef, 'SE': se, 'CI_Low': ci_low, 'CI_High': ci_high})
    else:
        print(f"  {year}: 0.0000 (reference)")
        event_study_results.append({'Year': year, 'Coefficient': 0, 'SE': 0, 'CI_Low': 0, 'CI_High': 0})

event_study_df = pd.DataFrame(event_study_results)
event_study_df.to_csv('event_study_results.csv', index=False)

# =============================================================================
# 13. SAVE RESULTS
# =============================================================================
print("\n[13] Saving results...")

results_dict = {
    'Model 1 (Basic DiD)': {
        'treat_post': (results1.params['treat_post'], results1.bse['treat_post'], results1.pvalues['treat_post']),
        'N': int(results1.nobs),
        'R2': results1.rsquared
    },
    'Model 2 (Demo Controls)': {
        'treat_post': (results2.params['treat_post'], results2.bse['treat_post'], results2.pvalues['treat_post']),
        'N': int(results2.nobs),
        'R2': results2.rsquared
    },
    'Model 3 (Full Controls)': {
        'treat_post': (results3.params['treat_post'], results3.bse['treat_post'], results3.pvalues['treat_post']),
        'N': int(results3.nobs),
        'R2': results3.rsquared
    },
    'Model 4 (Year FE)': {
        'treat_post': (results4.params['treat_post'], results4.bse['treat_post'], results4.pvalues['treat_post']),
        'N': int(results4.nobs),
        'R2': results4.rsquared
    },
    'Model 5 (State + Year FE)': {
        'treat_post': (results5.params['treat_post'], results5.bse['treat_post'], results5.pvalues['treat_post']),
        'N': int(results5.nobs),
        'R2': results5.rsquared
    }
}

main_results = []
for model_name, vals in results_dict.items():
    coef, se, pval = vals['treat_post']
    main_results.append({
        'Model': model_name,
        'DiD_Coefficient': coef,
        'Std_Error': se,
        'P_Value': pval,
        'N': vals['N'],
        'R_Squared': vals['R2']
    })

main_results_df = pd.DataFrame(main_results)
main_results_df.to_csv('main_regression_results.csv', index=False)

# Also save robustness results
robustness_results = [
    {'Test': 'Any Employment', 'Coefficient': res_r1.params['treat_post'], 'SE': res_r1.bse['treat_post']},
    {'Test': 'Including 2012', 'Coefficient': res_r2.params['treat_post'], 'SE': res_r2.bse['treat_post']},
    {'Test': 'Ages 18-35', 'Coefficient': res_r3.params['treat_post'], 'SE': res_r3.bse['treat_post']},
    {'Test': 'Unweighted', 'Coefficient': res_r4.params['treat_post'], 'SE': res_r4.bse['treat_post']},
]
robustness_df = pd.DataFrame(robustness_results)
robustness_df.to_csv('robustness_results.csv', index=False)

# =============================================================================
# 14. PREFERRED ESTIMATE
# =============================================================================
print("\n" + "=" * 70)
print("PREFERRED ESTIMATE (Model 5: State and Year Fixed Effects)")
print("=" * 70)

preferred_coef = results5.params['treat_post']
preferred_se = results5.bse['treat_post']
preferred_pval = results5.pvalues['treat_post']
preferred_ci_low = preferred_coef - 1.96 * preferred_se
preferred_ci_high = preferred_coef + 1.96 * preferred_se
preferred_n = int(results5.nobs)

print(f"\nEffect Size: {preferred_coef:.4f}")
print(f"Standard Error: {preferred_se:.4f}")
print(f"95% Confidence Interval: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"P-value: {preferred_pval:.4f}")
print(f"Sample Size: {preferred_n:,}")

print(f"\nInterpretation: DACA eligibility is associated with a {preferred_coef*100:.2f} percentage point")
print(f"change in the probability of full-time employment.")

if preferred_pval < 0.05:
    print(f"This effect is statistically significant at the 5% level.")
else:
    print(f"This effect is NOT statistically significant at the 5% level.")

with open('preferred_estimate.txt', 'w') as f:
    f.write("PREFERRED ESTIMATE SUMMARY\n")
    f.write("=" * 50 + "\n")
    f.write(f"Effect Size (Coefficient): {preferred_coef:.6f}\n")
    f.write(f"Standard Error: {preferred_se:.6f}\n")
    f.write(f"95% CI Lower: {preferred_ci_low:.6f}\n")
    f.write(f"95% CI Upper: {preferred_ci_high:.6f}\n")
    f.write(f"P-value: {preferred_pval:.6f}\n")
    f.write(f"Sample Size: {preferred_n}\n")
    f.write(f"\nModel: OLS with DACA eligible x Post interaction,\n")
    f.write(f"       demographic controls, state and year fixed effects,\n")
    f.write(f"       weighted by PERWT, robust standard errors.\n")

# Save detailed summary for heterogeneity
hetero_results = []
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    subset = df_analysis[df_analysis['SEX'] == sex_val]
    if len(subset) > 100:
        model = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + married + educ_hs + C(YEAR)',
                         data=subset, weights=subset['PERWT'])
        res = model.fit(cov_type='HC1')
        hetero_results.append({'Subgroup': sex_label, 'Coefficient': res.params['treat_post'],
                               'SE': res.bse['treat_post'], 'N': len(subset)})

for age_min, age_max, label in [(16, 24, '16-24'), (25, 32, '25-32'), (33, 40, '33-40')]:
    subset = df_analysis[(df_analysis['AGE'] >= age_min) & (df_analysis['AGE'] <= age_max)]
    if len(subset) > 100:
        model = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + female + married + educ_hs + C(YEAR)',
                         data=subset, weights=subset['PERWT'])
        res = model.fit(cov_type='HC1')
        if 'treat_post' in res.params.index:
            hetero_results.append({'Subgroup': f'Ages {label}', 'Coefficient': res.params['treat_post'],
                                   'SE': res.bse['treat_post'], 'N': len(subset)})

hetero_df = pd.DataFrame(hetero_results)
hetero_df.to_csv('heterogeneity_results.csv', index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files created:")
print("  - summary_statistics.csv")
print("  - main_regression_results.csv")
print("  - event_study_results.csv")
print("  - robustness_results.csv")
print("  - heterogeneity_results.csv")
print("  - preferred_estimate.txt")
