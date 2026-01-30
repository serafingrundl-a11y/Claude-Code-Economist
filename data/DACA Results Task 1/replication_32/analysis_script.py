"""
DACA Impact on Full-Time Employment - Replication Analysis
Replication ID: 32

Research Question:
Among ethnically Hispanic-Mexican, Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on the probability of full-time employment?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load Data in Chunks and Filter
# ============================================================================
print("="*80)
print("STEP 1: Loading and Filtering Data (Chunked)")
print("="*80)

data_path = "data/data.csv"
print(f"Loading data from {data_path} in chunks...")

# Define the columns we need
needed_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EMPSTAT', 'UHRSWORK', 'EDUC']

# Read and filter in chunks
chunk_size = 1000000  # 1 million rows at a time
filtered_chunks = []

for i, chunk in enumerate(pd.read_csv(data_path, usecols=needed_cols, chunksize=chunk_size)):
    # Filter to Hispanic-Mexican (HISPAN == 1), born in Mexico (BPL == 200), non-citizen (CITIZEN == 3)
    filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3)
    ].copy()

    if len(filtered) > 0:
        filtered_chunks.append(filtered)

    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

print(f"  Finished processing all chunks.")

# Combine filtered chunks
df = pd.concat(filtered_chunks, ignore_index=True)
print(f"\nTotal observations after filtering: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# STEP 2: Define DACA Eligibility
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Defining DACA Eligibility")
print("="*80)

# DACA eligibility criteria:
# 1. Arrived before age 16
# 2. Under age 31 as of June 15, 2012 (born after June 15, 1981)
# 3. In US continuously since June 15, 2007 (immigrated by 2007)

# Calculate age at immigration
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']

# For age < 31 on June 15, 2012, need to be born after June 15, 1981
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# June 15 falls in Q2, so those born in Q3 or Q4 of 1981 are eligible
df['born_after_cutoff'] = (
    (df['BIRTHYR'] > 1981) |
    ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))
)

# DACA eligible if:
# - Arrived before age 16
# - Born after June 15, 1981 (under 31 on June 15, 2012)
# - Immigrated by 2007
df['daca_eligible'] = (
    (df['age_at_immigration'] < 16) &
    (df['age_at_immigration'] >= 0) &  # Valid age at immigration
    (df['born_after_cutoff']) &
    (df['YRIMMIG'] <= 2007) &
    (df['YRIMMIG'] > 0)
)

print(f"\nDACA eligibility breakdown:")
print(f"  Total non-citizen Mexican-born Hispanics: {len(df):,}")
print(f"  DACA eligible: {df['daca_eligible'].sum():,}")
print(f"  Not DACA eligible: {(~df['daca_eligible']).sum():,}")

# ============================================================================
# STEP 3: Define Sample and Outcome
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Defining Sample and Outcome")
print("="*80)

# Restrict to working-age population (18-64)
df_working = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)].copy()
print(f"\nAfter restricting to ages 18-64: {len(df_working):,}")

# Full-time employment: Usually working 35+ hours per week
df_working['fulltime'] = (df_working['UHRSWORK'] >= 35).astype(int)

# Employment status
df_working['employed'] = (df_working['EMPSTAT'] == 1).astype(int)

print(f"\nEmployment status distribution:")
print(df_working['EMPSTAT'].value_counts())

print(f"\nFull-time employment rate: {df_working['fulltime'].mean():.4f}")
print(f"Employment rate: {df_working['employed'].mean():.4f}")

# ============================================================================
# STEP 4: Create DiD Variables
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Creating DiD Variables")
print("="*80)

# Post-DACA indicator (2013-2016)
# Excluding 2012 as transition year
df_working['post'] = (df_working['YEAR'] >= 2013).astype(int)

# Treatment indicator
df_working['treated'] = df_working['daca_eligible'].astype(int)

# Interaction term
df_working['did'] = df_working['post'] * df_working['treated']

# Demographic controls
df_working['female'] = (df_working['SEX'] == 2).astype(int)
df_working['age_sq'] = df_working['AGE'] ** 2

print(f"\nYear distribution:")
print(df_working.groupby('YEAR').size())

print(f"\nDACA eligible by year:")
print(df_working.groupby(['YEAR', 'treated']).size().unstack())

# ============================================================================
# STEP 5: Descriptive Statistics
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Descriptive Statistics")
print("="*80)

def weighted_mean(df, var, weight='PERWT'):
    return np.average(df[var], weights=df[weight])

def weighted_std(df, var, weight='PERWT'):
    avg = weighted_mean(df, var, weight)
    return np.sqrt(np.average((df[var] - avg)**2, weights=df[weight]))

# Summary by treatment and period
print("\nWeighted descriptive statistics:")
print("\n" + "-"*70)
print(f"{'Group':<30} {'FT Rate':>10} {'Age':>8} {'Female':>10} {'N':>12}")
print("-"*70)

for treated in [0, 1]:
    for post in [0, 1]:
        subset = df_working[(df_working['treated'] == treated) & (df_working['post'] == post)]
        ft_rate = weighted_mean(subset, 'fulltime')
        avg_age = weighted_mean(subset, 'AGE')
        pct_female = weighted_mean(subset, 'female')
        n = len(subset)

        label_treat = "DACA eligible" if treated == 1 else "Not eligible"
        label_post = "Post (2013-16)" if post == 1 else "Pre (2006-11)"
        label = f"{label_treat}, {label_post}"
        print(f"{label:<30} {ft_rate:>10.4f} {avg_age:>8.1f} {pct_female:>10.4f} {n:>12,}")

print("-"*70)

# ============================================================================
# STEP 6: Main DiD Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Main DiD Analysis")
print("="*80)

# Exclude 2012 from analysis
df_analysis = df_working[df_working['YEAR'] != 2012].copy()
print(f"\nAnalysis sample (excluding 2012): {len(df_analysis):,}")

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls('fulltime ~ treated + post + did',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model1.params['did']:.4f}")
print(f"Standard error: {model1.bse['did']:.4f}")
print(f"t-statistic: {model1.tvalues['did']:.4f}")
print(f"p-value: {model1.pvalues['did']:.4f}")
print(f"N: {int(model1.nobs):,}")
print(f"R-squared: {model1.rsquared:.4f}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographics ---")
model2 = smf.wls('fulltime ~ treated + post + did + AGE + age_sq + female',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model2.params['did']:.4f}")
print(f"Standard error: {model2.bse['did']:.4f}")
print(f"t-statistic: {model2.tvalues['did']:.4f}")
print(f"p-value: {model2.pvalues['did']:.4f}")
print(f"N: {int(model2.nobs):,}")
print(f"R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year FE + Demographics ---")
model3 = smf.wls('fulltime ~ treated + did + C(YEAR) + AGE + age_sq + female',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model3.params['did']:.4f}")
print(f"Standard error: {model3.bse['did']:.4f}")
print(f"t-statistic: {model3.tvalues['did']:.4f}")
print(f"p-value: {model3.pvalues['did']:.4f}")
print(f"N: {int(model3.nobs):,}")
print(f"R-squared: {model3.rsquared:.4f}")

# Model 4: DiD with state and year fixed effects
print("\n--- Model 4: DiD with State + Year FE + Demographics ---")
model4 = smf.wls('fulltime ~ treated + did + C(YEAR) + C(STATEFIP) + AGE + age_sq + female',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model4.params['did']:.4f}")
print(f"Standard error: {model4.bse['did']:.4f}")
print(f"t-statistic: {model4.tvalues['did']:.4f}")
print(f"p-value: {model4.pvalues['did']:.4f}")
print(f"N: {int(model4.nobs):,}")
print(f"R-squared: {model4.rsquared:.4f}")

# ============================================================================
# STEP 7: Robustness Checks
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Robustness Checks")
print("="*80)

# Robustness 1: Alternative age range (16-35)
print("\n--- Robustness 1: Younger Sample (Ages 16-35) ---")
df_young = df[(df['AGE'] >= 16) & (df['AGE'] <= 35)].copy()
df_young['fulltime'] = (df_young['UHRSWORK'] >= 35).astype(int)
df_young['post'] = (df_young['YEAR'] >= 2013).astype(int)
df_young['treated'] = df_young['daca_eligible'].astype(int)
df_young['did'] = df_young['post'] * df_young['treated']
df_young['female'] = (df_young['SEX'] == 2).astype(int)
df_young['age_sq'] = df_young['AGE'] ** 2
df_young = df_young[df_young['YEAR'] != 2012]

robust1 = smf.wls('fulltime ~ treated + did + C(YEAR) + AGE + age_sq + female',
                   data=df_young,
                   weights=df_young['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {robust1.params['did']:.4f} (SE: {robust1.bse['did']:.4f})")
print(f"N: {int(robust1.nobs):,}")

# Robustness 2: Any employment as outcome
print("\n--- Robustness 2: Any Employment as Outcome ---")
robust2 = smf.wls('employed ~ treated + did + C(YEAR) + AGE + age_sq + female',
                   data=df_analysis,
                   weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {robust2.params['did']:.4f} (SE: {robust2.bse['did']:.4f})")
print(f"N: {int(robust2.nobs):,}")

# Robustness 3: Hours worked (continuous outcome)
print("\n--- Robustness 3: Usual Hours Worked (Continuous) ---")
robust3 = smf.wls('UHRSWORK ~ treated + did + C(YEAR) + AGE + age_sq + female',
                   data=df_analysis,
                   weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {robust3.params['did']:.4f} (SE: {robust3.bse['did']:.4f})")
print(f"N: {int(robust3.nobs):,}")

# Robustness 4: Shorter pre-period (2009-2011)
print("\n--- Robustness 4: Shorter Pre-Period (2009-2011) ---")
df_short = df_analysis[(df_analysis['YEAR'] >= 2009) | (df_analysis['post'] == 1)].copy()
robust4 = smf.wls('fulltime ~ treated + did + C(YEAR) + AGE + age_sq + female',
                   data=df_short,
                   weights=df_short['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {robust4.params['did']:.4f} (SE: {robust4.bse['did']:.4f})")
print(f"N: {int(robust4.nobs):,}")

# ============================================================================
# STEP 8: Heterogeneity Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Heterogeneity Analysis")
print("="*80)

# By gender
print("\n--- By Gender ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_analysis[df_analysis['SEX'] == sex]
    model_sex = smf.wls('fulltime ~ treated + did + C(YEAR) + AGE + age_sq',
                        data=df_sex,
                        weights=df_sex['PERWT']).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_sex.params['did']:.4f} (SE: {model_sex.bse['did']:.4f}), N = {int(model_sex.nobs):,}")

# By education
print("\n--- By Education Level ---")
# EDUC: 0-5 = less than high school, 6 = high school, 7+ = some college or more
df_analysis['educ_cat'] = pd.cut(df_analysis['EDUC'], bins=[-1, 5, 6, 20], labels=['<HS', 'HS', '>HS'])
for educ in ['<HS', 'HS', '>HS']:
    df_educ = df_analysis[df_analysis['educ_cat'] == educ]
    if len(df_educ) > 1000:
        model_educ = smf.wls('fulltime ~ treated + did + C(YEAR) + AGE + age_sq + female',
                             data=df_educ,
                             weights=df_educ['PERWT']).fit(cov_type='HC1')
        print(f"{educ}: DiD = {model_educ.params['did']:.4f} (SE: {model_educ.bse['did']:.4f}), N = {int(model_educ.nobs):,}")

# ============================================================================
# STEP 9: Event Study
# ============================================================================
print("\n" + "="*80)
print("STEP 9: Event Study Analysis")
print("="*80)

years = sorted(df_analysis['YEAR'].unique())
ref_year = 2011

# Create interaction terms for each year
for year in years:
    if year != ref_year:
        df_analysis[f'treated_x_{year}'] = df_analysis['treated'] * (df_analysis['YEAR'] == year).astype(int)

# Run event study regression
year_vars = [f'treated_x_{y}' for y in years if y != ref_year]
formula = 'fulltime ~ treated + ' + ' + '.join(year_vars) + ' + C(YEAR) + AGE + age_sq + female'
event_study = smf.wls(formula, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (reference year: 2011):")
print("-"*60)
for year in sorted(years):
    if year != ref_year:
        var = f'treated_x_{year}'
        coef = event_study.params[var]
        se = event_study.bse[var]
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        sig = "*" if abs(coef/se) > 1.96 else ""
        print(f"  {year}: {coef:>8.4f} (SE: {se:.4f}) [{ci_low:>7.4f}, {ci_high:>7.4f}] {sig}")
    else:
        print(f"  {year}: {0:>8.4f} (reference year)")
print("-"*60)

# Save event study results
event_results = []
for year in years:
    if year != ref_year:
        var = f'treated_x_{year}'
        event_results.append({
            'year': year,
            'coef': event_study.params[var],
            'se': event_study.bse[var],
            'ci_low': event_study.params[var] - 1.96 * event_study.bse[var],
            'ci_high': event_study.params[var] + 1.96 * event_study.bse[var]
        })
    else:
        event_results.append({
            'year': year,
            'coef': 0,
            'se': 0,
            'ci_low': 0,
            'ci_high': 0
        })

event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

# ============================================================================
# STEP 10: Final Results Summary
# ============================================================================
print("\n" + "="*80)
print("STEP 10: Final Results Summary")
print("="*80)

# Compile main results
results_dict = {
    'Model': ['(1) Basic', '(2) Demographics', '(3) Year FE', '(4) State+Year FE'],
    'DiD_Coef': [model1.params['did'], model2.params['did'], model3.params['did'], model4.params['did']],
    'SE': [model1.bse['did'], model2.bse['did'], model3.bse['did'], model4.bse['did']],
    't_stat': [model1.tvalues['did'], model2.tvalues['did'], model3.tvalues['did'], model4.tvalues['did']],
    'p_value': [model1.pvalues['did'], model2.pvalues['did'], model3.pvalues['did'], model4.pvalues['did']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs)],
    'R2': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared]
}

results_df = pd.DataFrame(results_dict)
print("\nMain Results Table:")
print(results_df.to_string(index=False))
results_df.to_csv('results_table.csv', index=False)

# Simple 2x2 DiD calculation for verification
print("\n\n" + "="*60)
print("Simple 2x2 DiD Calculation (Weighted)")
print("="*60)

pre_treat = df_analysis[(df_analysis['post']==0) & (df_analysis['treated']==1)]
pre_ctrl = df_analysis[(df_analysis['post']==0) & (df_analysis['treated']==0)]
post_treat = df_analysis[(df_analysis['post']==1) & (df_analysis['treated']==1)]
post_ctrl = df_analysis[(df_analysis['post']==1) & (df_analysis['treated']==0)]

y00 = weighted_mean(pre_ctrl, 'fulltime')
y01 = weighted_mean(pre_treat, 'fulltime')
y10 = weighted_mean(post_ctrl, 'fulltime')
y11 = weighted_mean(post_treat, 'fulltime')

print(f"\n                     Control      Treatment")
print(f"Pre (2006-11)       {y00:.4f}        {y01:.4f}")
print(f"Post (2013-16)      {y10:.4f}        {y11:.4f}")
print(f"\nTreatment effect:   {y11 - y01:.4f} - {y10 - y00:.4f} = {(y11 - y01) - (y10 - y00):.4f}")

# Preferred specification summary
print("\n\n" + "="*60)
print("PREFERRED ESTIMATE (Model 3: Year FE + Demographics)")
print("="*60)
print(f"Effect of DACA eligibility on full-time employment:")
print(f"  Coefficient: {model3.params['did']:.4f}")
print(f"  Standard Error: {model3.bse['did']:.4f}")
print(f"  95% CI: [{model3.params['did'] - 1.96*model3.bse['did']:.4f}, {model3.params['did'] + 1.96*model3.bse['did']:.4f}]")
print(f"  t-statistic: {model3.tvalues['did']:.4f}")
print(f"  p-value: {model3.pvalues['did']:.4f}")
print(f"  Sample Size: {int(model3.nobs):,}")

# Additional statistics for report
print("\n\n" + "="*60)
print("Additional Statistics for Report")
print("="*60)

# Counts
n_treated_pre = len(pre_treat)
n_treated_post = len(post_treat)
n_control_pre = len(pre_ctrl)
n_control_post = len(post_ctrl)

print(f"\nSample sizes:")
print(f"  Pre-DACA treatment: {n_treated_pre:,}")
print(f"  Post-DACA treatment: {n_treated_post:,}")
print(f"  Pre-DACA control: {n_control_pre:,}")
print(f"  Post-DACA control: {n_control_post:,}")
print(f"  Total: {len(df_analysis):,}")

# Demographics summary
print(f"\nDemographics (treatment group):")
treat_group = df_analysis[df_analysis['treated']==1]
print(f"  Mean age: {weighted_mean(treat_group, 'AGE'):.1f}")
print(f"  Percent female: {weighted_mean(treat_group, 'female')*100:.1f}%")

print(f"\nDemographics (control group):")
ctrl_group = df_analysis[df_analysis['treated']==0]
print(f"  Mean age: {weighted_mean(ctrl_group, 'AGE'):.1f}")
print(f"  Percent female: {weighted_mean(ctrl_group, 'female')*100:.1f}%")

print("\n\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save full model summaries
with open('model_summaries.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("MODEL 1: Basic DiD\n")
    f.write("="*80 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("="*80 + "\n")
    f.write("MODEL 2: DiD with Demographics\n")
    f.write("="*80 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("="*80 + "\n")
    f.write("MODEL 3: DiD with Year FE + Demographics\n")
    f.write("="*80 + "\n")
    f.write(str(model3.summary()) + "\n\n")

    f.write("="*80 + "\n")
    f.write("MODEL 4: DiD with State + Year FE + Demographics\n")
    f.write("="*80 + "\n")
    f.write(str(model4.summary()) + "\n\n")

print("\nModel summaries saved to model_summaries.txt")
