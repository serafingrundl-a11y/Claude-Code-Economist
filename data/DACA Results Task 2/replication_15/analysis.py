"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the United States
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA - Only necessary columns
# ============================================================================
print("\n1. Loading data...")

# Select only columns we need to reduce memory
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
           'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
           'EMPSTAT', 'UHRSWORK']

# Define dtypes to reduce memory
dtypes = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8'
}

# Read in chunks and filter immediately
chunks = []
chunksize = 100000
total_rows = 0

for chunk in pd.read_csv('data/data.csv', usecols=usecols, dtype=dtypes,
                          chunksize=chunksize):
    total_rows += len(chunk)
    # Apply initial filters immediately
    # Hispanic-Mexican (HISPAN = 1) AND Born in Mexico (BPL = 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)].copy()
    if len(filtered) > 0:
        chunks.append(filtered)
    del chunk
    gc.collect()
    if total_rows % 1000000 == 0:
        print(f"   Processed {total_rows:,} rows...")

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"   Total rows processed: {total_rows:,}")
print(f"   Hispanic-Mexican, Mexican-born: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# 2. FURTHER SAMPLE SELECTION
# ============================================================================
print("\n2. Applying further sample selection criteria...")

# Non-citizen (CITIZEN = 3) - proxy for undocumented
df_sample = df[df['CITIZEN'] == 3].copy()
del df
gc.collect()
print(f"   After non-citizen filter: {len(df_sample):,}")

# ============================================================================
# 3. CALCULATE AGE AT DACA IMPLEMENTATION
# ============================================================================
print("\n3. Calculating age at DACA implementation (June 15, 2012)...")

# DACA implemented June 15, 2012
# Age at DACA = 2012 - BIRTHYR, adjusted for birth quarter
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born Jul-Dec, hadn't had birthday by June 15

df_sample['age_at_daca_raw'] = 2012 - df_sample['BIRTHYR']
df_sample['age_at_daca'] = df_sample['age_at_daca_raw'].copy()

# Adjust for birth quarter (if born Jul-Dec, subtract 1)
mask_late_birth = df_sample['BIRTHQTR'].isin([3, 4])
df_sample.loc[mask_late_birth, 'age_at_daca'] = \
    df_sample.loc[mask_late_birth, 'age_at_daca_raw'] - 1

print(f"   Age at DACA range: {df_sample['age_at_daca'].min()} to {df_sample['age_at_daca'].max()}")

# ============================================================================
# 4. DEFINE TREATMENT AND CONTROL GROUPS
# ============================================================================
print("\n4. Defining treatment and control groups...")

# Treatment: Ages 26-30 at DACA (June 2012)
# Control: Ages 31-35 at DACA (June 2012)
df_sample = df_sample[(df_sample['age_at_daca'] >= 26) &
                       (df_sample['age_at_daca'] <= 35)].copy()
print(f"   After age restriction (26-35): {len(df_sample):,}")

# Create treatment indicator
df_sample['treat'] = (df_sample['age_at_daca'] <= 30).astype('int8')

print(f"   Treatment group (26-30): {df_sample['treat'].sum():,}")
print(f"   Control group (31-35): {(1-df_sample['treat']).sum():,}")

# ============================================================================
# 5. ADDITIONAL ELIGIBILITY CRITERIA
# ============================================================================
print("\n5. Applying additional DACA eligibility criteria...")

# Arrived in US before 16th birthday
# YRIMMIG must be <= BIRTHYR + 15 (before turning 16)
df_sample['year_turn_16'] = df_sample['BIRTHYR'] + 16
df_sample = df_sample[df_sample['YRIMMIG'] <= df_sample['year_turn_16']].copy()
print(f"   After arrived before age 16 filter: {len(df_sample):,}")

# Must have been in US since June 2007 (5 years continuous)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
print(f"   After continuous residence since 2007 filter: {len(df_sample):,}")

print(f"\n   Final treatment group (26-30): {df_sample['treat'].sum():,}")
print(f"   Final control group (31-35): {(1-df_sample['treat']).sum():,}")

# ============================================================================
# 6. DEFINE TIME PERIODS
# ============================================================================
print("\n6. Defining time periods...")

# Pre-period: 2006-2011
# Post-period: 2013-2016 (excluding 2012 due to timing ambiguity)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()

df_sample['post'] = (df_sample['YEAR'] >= 2013).astype('int8')
print(f"   Pre-period (2006-2011): {(df_sample['post']==0).sum():,}")
print(f"   Post-period (2013-2016): {(df_sample['post']==1).sum():,}")
print(f"   Years used: {sorted(df_sample['YEAR'].unique())}")

# ============================================================================
# 7. DEFINE OUTCOME VARIABLE
# ============================================================================
print("\n7. Defining outcome variable...")

# Full-time employment: Usually works 35+ hours per week
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype('int8')

print(f"   Full-time employment rate: {df_sample['fulltime'].mean():.4f}")
print(f"   Mean usual hours worked: {df_sample['UHRSWORK'].mean():.2f}")

# ============================================================================
# 8. CREATE ANALYSIS VARIABLES
# ============================================================================
print("\n8. Creating analysis variables...")

# Create interaction term
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

# Covariates
df_sample['female'] = (df_sample['SEX'] == 2).astype('int8')
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype('int8')

# Education categories
df_sample['educ_lesshs'] = (df_sample['EDUC'] < 6).astype('int8')
df_sample['educ_hs'] = (df_sample['EDUC'] == 6).astype('int8')
df_sample['educ_somecol'] = (df_sample['EDUC'].isin([7, 8, 9])).astype('int8')
df_sample['educ_college'] = (df_sample['EDUC'] >= 10).astype('int8')

# Summary
print("\n   Sample sizes by group and period:")
summary = df_sample.groupby(['treat', 'post']).agg({
    'fulltime': ['count', 'mean'],
    'PERWT': 'sum'
}).round(4)
summary.columns = ['N', 'FT_rate', 'Pop_weight']
print(summary)

# ============================================================================
# 9. DIFFERENCE-IN-DIFFERENCES ESTIMATION
# ============================================================================
print("\n" + "=" * 70)
print("9. DIFFERENCE-IN-DIFFERENCES ESTIMATION")
print("=" * 70)

# 9.1 Simple DiD (unweighted)
print("\n9.1 Simple DiD (unweighted, no covariates):")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_sample).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']}
)
print(model1.summary().tables[1])

# 9.2 DiD with survey weights
print("\n9.2 DiD with survey weights (PERWT):")
model2 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']}
)
print(model2.summary().tables[1])

# 9.3 DiD with covariates
print("\n9.3 DiD with covariates (weighted):")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + '
                  'educ_hs + educ_somecol + educ_college + C(STATEFIP)',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']}
)
print("\n   Key coefficients:")
print(f"   treat_post (DiD estimate): {model3.params['treat_post']:.4f}")
print(f"   Std. Error: {model3.bse['treat_post']:.4f}")
print(f"   t-statistic: {model3.tvalues['treat_post']:.4f}")
print(f"   p-value: {model3.pvalues['treat_post']:.4f}")
ci = model3.conf_int().loc['treat_post']
print(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# ============================================================================
# 10. EVENT STUDY ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("10. EVENT STUDY ANALYSIS")
print("=" * 70)

# Create year dummies interacted with treatment
years = sorted(df_sample['YEAR'].unique())
ref_year = 2011  # Reference year

# Create interaction terms
for y in years:
    if y != ref_year:
        df_sample[f'treat_year_{y}'] = ((df_sample['treat'] == 1) &
                                         (df_sample['YEAR'] == y)).astype('int8')

# Build formula
event_formula = 'fulltime ~ treat + C(YEAR)'
for y in years:
    if y != ref_year:
        event_formula += f' + treat_year_{y}'

model_event = smf.wls(event_formula, data=df_sample,
                       weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']}
)

print("\nEvent study coefficients (relative to 2011):")
print("-" * 60)
print(f"{'Year':<8} {'Coefficient':>12} {'Std.Err':>10} {'95% CI Low':>12} {'95% CI High':>12}")
print("-" * 60)

event_results = []
for y in years:
    if y == ref_year:
        event_results.append({
            'Year': y, 'Coefficient': 0.0, 'SE': 0.0,
            'CI_low': 0.0, 'CI_high': 0.0, 'pvalue': np.nan
        })
        print(f"{y:<8} {0:>12.4f} {'(ref)':>10} {'-':>12} {'-':>12}")
    else:
        var = f'treat_year_{y}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        ci_low = model_event.conf_int().loc[var, 0]
        ci_high = model_event.conf_int().loc[var, 1]
        pval = model_event.pvalues[var]
        event_results.append({
            'Year': y, 'Coefficient': coef, 'SE': se,
            'CI_low': ci_low, 'CI_high': ci_high, 'pvalue': pval
        })
        print(f"{y:<8} {coef:>12.4f} {se:>10.4f} {ci_low:>12.4f} {ci_high:>12.4f}")
print("-" * 60)

event_df = pd.DataFrame(event_results)

# ============================================================================
# 11. ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 70)
print("11. ROBUSTNESS CHECKS")
print("=" * 70)

# 11.1 Different age bandwidths
print("\n11.1 Different age bandwidths around cutoff (age 30/31):")
print("-" * 65)
print(f"{'Bandwidth':<12} {'Coefficient':>12} {'Std.Err':>10} {'p-value':>10} {'N':>12}")
print("-" * 65)

bandwidth_results = []
for bandwidth in [3, 4, 5]:
    lower_bound = 31 - bandwidth
    upper_bound = 30 + bandwidth
    df_bw = df_sample[(df_sample['age_at_daca'] >= lower_bound) &
                       (df_sample['age_at_daca'] <= upper_bound)].copy()
    df_bw['treat_bw'] = (df_bw['age_at_daca'] <= 30).astype('int8')
    df_bw['treat_post_bw'] = df_bw['treat_bw'] * df_bw['post']

    model_bw = smf.wls('fulltime ~ treat_bw + post + treat_post_bw',
                        data=df_bw,
                        weights=df_bw['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': df_bw['STATEFIP']}
    )
    coef = model_bw.params['treat_post_bw']
    se = model_bw.bse['treat_post_bw']
    pval = model_bw.pvalues['treat_post_bw']
    n = len(df_bw)
    bandwidth_results.append({
        'bandwidth': bandwidth, 'coef': coef, 'se': se, 'pval': pval, 'n': n
    })
    print(f"+-{bandwidth} years   {coef:>12.4f} {se:>10.4f} {pval:>10.4f} {n:>12,}")
print("-" * 65)

# 11.2 By sex
print("\n11.2 Heterogeneity by sex:")
print("-" * 65)
print(f"{'Group':<12} {'Coefficient':>12} {'Std.Err':>10} {'p-value':>10} {'N':>12}")
print("-" * 65)

sex_results = []
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_sample[df_sample['SEX'] == sex_val].copy()
    model_sex = smf.wls('fulltime ~ treat + post + treat_post',
                         data=df_sex,
                         weights=df_sex['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': df_sex['STATEFIP']}
    )
    coef = model_sex.params['treat_post']
    se = model_sex.bse['treat_post']
    pval = model_sex.pvalues['treat_post']
    n = len(df_sex)
    sex_results.append({
        'group': sex_name, 'coef': coef, 'se': se, 'pval': pval, 'n': n
    })
    print(f"{sex_name:<12} {coef:>12.4f} {se:>10.4f} {pval:>10.4f} {n:>12,}")
print("-" * 65)

# 11.3 Placebo test
print("\n11.3 Placebo test (pre-period only, 2009 as fake treatment):")
df_pre = df_sample[df_sample['YEAR'] <= 2011].copy()
df_pre['post_placebo'] = (df_pre['YEAR'] >= 2009).astype('int8')
df_pre['treat_post_placebo'] = df_pre['treat'] * df_pre['post_placebo']

model_placebo = smf.wls('fulltime ~ treat + post_placebo + treat_post_placebo',
                         data=df_pre,
                         weights=df_pre['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']}
)
print(f"   Placebo DiD coefficient: {model_placebo.params['treat_post_placebo']:.4f}")
print(f"   Standard error: {model_placebo.bse['treat_post_placebo']:.4f}")
print(f"   p-value: {model_placebo.pvalues['treat_post_placebo']:.4f}")

# 11.4 Alternative outcome: Employment (any hours)
print("\n11.4 Alternative outcome: Any employment (UHRSWORK > 0):")
df_sample['employed'] = (df_sample['UHRSWORK'] > 0).astype('int8')
model_emp = smf.wls('employed ~ treat + post + treat_post',
                     data=df_sample,
                     weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']}
)
print(f"   DiD coefficient: {model_emp.params['treat_post']:.4f}")
print(f"   Standard error: {model_emp.bse['treat_post']:.4f}")
print(f"   p-value: {model_emp.pvalues['treat_post']:.4f}")

# ============================================================================
# 12. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("12. SUMMARY STATISTICS")
print("=" * 70)

print("\n12.1 Sample characteristics by treatment group:")
print("-" * 50)
summary_stats = df_sample.groupby('treat').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_lesshs': 'mean',
    'educ_hs': 'mean',
    'educ_somecol': 'mean',
    'educ_college': 'mean',
    'fulltime': 'mean',
    'UHRSWORK': 'mean'
}).round(4)
summary_stats.index = ['Control (31-35)', 'Treatment (26-30)']
summary_stats.columns = ['Age', 'Female', 'Married', 'Less_HS', 'HS',
                          'Some_Col', 'College+', 'FT_Rate', 'Hrs_Work']
print(summary_stats.T)

print("\n12.2 Full-time employment by group and period:")
print("-" * 50)
ft_means = df_sample.groupby(['treat', 'post'])['fulltime'].mean().unstack()
ft_means.index = ['Control (31-35)', 'Treatment (26-30)']
ft_means.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(ft_means.round(4))

# Calculate simple DiD from means
did_simple = (ft_means.loc['Treatment (26-30)', 'Post (2013-2016)'] -
              ft_means.loc['Treatment (26-30)', 'Pre (2006-2011)']) - \
             (ft_means.loc['Control (31-35)', 'Post (2013-2016)'] -
              ft_means.loc['Control (31-35)', 'Pre (2006-2011)'])
print(f"\n   Simple DiD from means: {did_simple:.4f}")

# Sample counts
print("\n12.3 Sample counts:")
print(df_sample.groupby(['treat', 'post']).size().unstack())

# ============================================================================
# 13. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("13. SAVING RESULTS")
print("=" * 70)

# Main results
results = {
    'main_estimate': model2.params['treat_post'],
    'main_se': model2.bse['treat_post'],
    'main_pvalue': model2.pvalues['treat_post'],
    'main_ci_low': model2.conf_int().loc['treat_post', 0],
    'main_ci_high': model2.conf_int().loc['treat_post', 1],
    'cov_estimate': model3.params['treat_post'],
    'cov_se': model3.bse['treat_post'],
    'cov_pvalue': model3.pvalues['treat_post'],
    'cov_ci_low': model3.conf_int().loc['treat_post', 0],
    'cov_ci_high': model3.conf_int().loc['treat_post', 1],
    'n_total': len(df_sample),
    'n_treat': int(df_sample['treat'].sum()),
    'n_control': int((1-df_sample['treat']).sum()),
    'n_pre': int((df_sample['post']==0).sum()),
    'n_post': int((df_sample['post']==1).sum())
}

results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)
print("   Results saved to results_summary.csv")

event_df.to_csv('event_study_results.csv', index=False)
print("   Event study results saved to event_study_results.csv")

summary_stats.T.to_csv('summary_statistics.csv')
print("   Summary statistics saved to summary_statistics.csv")

# Save full regression output
with open('regression_output.txt', 'w') as f:
    f.write("DACA REPLICATION - FULL REGRESSION OUTPUT\n")
    f.write("=" * 70 + "\n\n")
    f.write("Model 1: Simple DiD (unweighted)\n")
    f.write(str(model1.summary()) + "\n\n")
    f.write("Model 2: DiD with weights\n")
    f.write(str(model2.summary()) + "\n\n")
    f.write("Model 3: DiD with weights and covariates\n")
    f.write(str(model3.summary()) + "\n\n")
    f.write("Model 4: Event Study\n")
    f.write(str(model_event.summary()) + "\n")
print("   Full regression output saved to regression_output.txt")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# Print final preferred estimate
print(f"\n*** PREFERRED ESTIMATE (Weighted DiD) ***")
print(f"Effect on full-time employment: {results['main_estimate']:.4f}")
print(f"Standard Error (clustered by state): {results['main_se']:.4f}")
print(f"95% Confidence Interval: [{results['main_ci_low']:.4f}, {results['main_ci_high']:.4f}]")
print(f"p-value: {results['main_pvalue']:.4f}")
print(f"Total sample size: {results['n_total']:,}")
print(f"  - Treatment (ages 26-30 at DACA): {results['n_treat']:,}")
print(f"  - Control (ages 31-35 at DACA): {results['n_control']:,}")
