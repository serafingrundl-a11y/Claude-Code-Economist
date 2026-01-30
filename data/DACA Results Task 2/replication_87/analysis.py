"""
DACA Replication Study - Analysis Script
Estimating the effect of DACA eligibility on full-time employment
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DACA Replication Study - Analysis")
print("=" * 60)

# Load data - filter on read for efficiency
print("\nLoading data...")

# Define the columns we need
cols_needed = ['YEAR', 'BIRTHYR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD',
               'CITIZEN', 'YRIMMIG', 'UHRSWORK', 'EMPSTAT', 'PERWT',
               'SEX', 'EDUC', 'MARST', 'AGE', 'STATEFIP']

# Read data in chunks for efficiency
chunk_size = 500000
chunks = []

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size):
    # Filter to Mexican-born, Hispanic-Mexican
    chunk_filtered = chunk[
        (chunk['BPL'] == 200) &  # Born in Mexico
        (chunk['HISPAN'] == 1) &  # Hispanic-Mexican
        (chunk['CITIZEN'] == 3) &  # Not a citizen (proxy for undocumented)
        (chunk['YEAR'] != 2012)  # Exclude 2012 (policy year)
    ]
    chunks.append(chunk_filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"Initial filtered sample: {len(df):,} observations")

# Create age as of June 15, 2012
# Age on June 15, 2012 = 2012 - BIRTHYR (approximate, ignoring birth quarter)
df['age_2012'] = 2012 - df['BIRTHYR']

# Filter to relevant age groups (26-35 as of June 2012)
df = df[(df['age_2012'] >= 26) & (df['age_2012'] <= 35)]
print(f"After age filter (26-35 in 2012): {len(df):,} observations")

# Check for arrival before age 16 (DACA requirement)
# Approximating: YRIMMIG should be <= BIRTHYR + 15
df['arrived_before_16'] = df['YRIMMIG'] <= (df['BIRTHYR'] + 15)

# Filter for continuous residence since 2007 (approximate)
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007

# Apply remaining DACA eligibility criteria
df = df[df['arrived_before_16'] & df['in_us_since_2007']]
print(f"After DACA eligibility filters: {len(df):,} observations")

# Define treatment group (ages 26-30 in 2012) vs control (ages 31-35)
df['treated'] = (df['age_2012'] >= 26) & (df['age_2012'] <= 30)
df['treated'] = df['treated'].astype(int)

# Define post-treatment period (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Create outcome variable: Full-time employment (35+ hours/week)
# UHRSWORK = usual hours worked per week
# EMPSTAT = 1 means employed
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)

# Also create an employment indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

print("\n" + "=" * 60)
print("Sample Composition")
print("=" * 60)

print(f"\nTotal observations: {len(df):,}")
print(f"\nBy treatment status:")
print(f"  Treatment (ages 26-30): {df['treated'].sum():,}")
print(f"  Control (ages 31-35): {(1-df['treated']).sum():,}")

print(f"\nBy time period:")
print(f"  Pre-treatment (2006-2011): {(df['post'] == 0).sum():,}")
print(f"  Post-treatment (2013-2016): {(df['post'] == 1).sum():,}")

# Summary statistics by group and period
print("\n" + "=" * 60)
print("Summary Statistics: Full-Time Employment Rates")
print("=" * 60)

# Weighted means
def weighted_mean(df_group, var, weight):
    return np.average(df_group[var], weights=df_group[weight])

def weighted_se(df_group, var, weight):
    n = len(df_group)
    if n < 2:
        return np.nan
    w = df_group[weight].values
    x = df_group[var].values
    mean = np.average(x, weights=w)
    variance = np.average((x - mean)**2, weights=w) * n / (n - 1)
    return np.sqrt(variance / n)

# Group means
results_table = []
for treated_val in [0, 1]:
    for post_val in [0, 1]:
        subset = df[(df['treated'] == treated_val) & (df['post'] == post_val)]
        n = len(subset)
        ft_rate = weighted_mean(subset, 'fulltime', 'PERWT')
        emp_rate = weighted_mean(subset, 'employed', 'PERWT')
        se_ft = weighted_se(subset, 'fulltime', 'PERWT')

        group = "Treatment" if treated_val == 1 else "Control"
        period = "Post" if post_val == 1 else "Pre"
        results_table.append({
            'Group': group,
            'Period': period,
            'N': n,
            'Full-Time Rate': ft_rate,
            'SE': se_ft,
            'Employment Rate': emp_rate
        })

results_df = pd.DataFrame(results_table)
print("\n" + results_df.to_string(index=False))

# DiD Calculation - Simple
print("\n" + "=" * 60)
print("Difference-in-Differences Calculation")
print("=" * 60)

# Get the four cell means
pre_treat = results_df[(results_df['Group'] == 'Treatment') & (results_df['Period'] == 'Pre')]['Full-Time Rate'].values[0]
post_treat = results_df[(results_df['Group'] == 'Treatment') & (results_df['Period'] == 'Post')]['Full-Time Rate'].values[0]
pre_ctrl = results_df[(results_df['Group'] == 'Control') & (results_df['Period'] == 'Pre')]['Full-Time Rate'].values[0]
post_ctrl = results_df[(results_df['Group'] == 'Control') & (results_df['Period'] == 'Post')]['Full-Time Rate'].values[0]

# Within-group changes
treat_change = post_treat - pre_treat
ctrl_change = post_ctrl - pre_ctrl

# DiD estimate
did_simple = treat_change - ctrl_change

print(f"\nTreatment group (ages 26-30):")
print(f"  Pre-DACA:  {pre_treat:.4f}")
print(f"  Post-DACA: {post_treat:.4f}")
print(f"  Change:    {treat_change:+.4f}")

print(f"\nControl group (ages 31-35):")
print(f"  Pre-DACA:  {pre_ctrl:.4f}")
print(f"  Post-DACA: {post_ctrl:.4f}")
print(f"  Change:    {ctrl_change:+.4f}")

print(f"\nDifference-in-Differences estimate: {did_simple:+.4f}")
print(f"  (i.e., {did_simple*100:+.2f} percentage points)")

# Regression-based DiD with controls
print("\n" + "=" * 60)
print("Regression Analysis")
print("=" * 60)

# Create interaction term
df['did'] = df['treated'] * df['post']

# Model 1: Basic DiD (no controls)
print("\nModel 1: Basic DiD (no controls)")
model1 = smf.wls('fulltime ~ treated + post + did', data=df, weights=df['PERWT'])
results1 = model1.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD coefficient: {results1.params['did']:.4f}")
print(f"  Std Error (clustered by state): {results1.bse['did']:.4f}")
print(f"  95% CI: [{results1.conf_int().loc['did', 0]:.4f}, {results1.conf_int().loc['did', 1]:.4f}]")
print(f"  t-statistic: {results1.tvalues['did']:.3f}")
print(f"  p-value: {results1.pvalues['did']:.4f}")

# Model 2: With demographic controls
print("\nModel 2: DiD with demographic controls")

# Create control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education dummies
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # High school or more
df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # 4+ years college

model2 = smf.wls('fulltime ~ treated + post + did + female + married + educ_hs + educ_college',
                 data=df, weights=df['PERWT'])
results2 = model2.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD coefficient: {results2.params['did']:.4f}")
print(f"  Std Error (clustered by state): {results2.bse['did']:.4f}")
print(f"  95% CI: [{results2.conf_int().loc['did', 0]:.4f}, {results2.conf_int().loc['did', 1]:.4f}]")
print(f"  t-statistic: {results2.tvalues['did']:.3f}")
print(f"  p-value: {results2.pvalues['did']:.4f}")

# Model 3: With year and state fixed effects
print("\nModel 3: DiD with year and state fixed effects")

# Create year dummies
for year in df['YEAR'].unique():
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)

# Create state dummies - only include if we have enough variation
state_counts = df['STATEFIP'].value_counts()
major_states = state_counts[state_counts >= 100].index.tolist()
df['in_major_state'] = df['STATEFIP'].isin(major_states)

# Use formula with fixed effects
year_dummies = ' + '.join([f'C(YEAR, Treatment({df["YEAR"].min()}))'])
state_fe = 'C(STATEFIP)'

formula3 = f'fulltime ~ treated + did + female + married + educ_hs + educ_college + C(YEAR) + C(STATEFIP)'
model3 = smf.wls(formula3, data=df, weights=df['PERWT'])
results3 = model3.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD coefficient: {results3.params['did']:.4f}")
print(f"  Std Error (clustered by state): {results3.bse['did']:.4f}")
print(f"  95% CI: [{results3.conf_int().loc['did', 0]:.4f}, {results3.conf_int().loc['did', 1]:.4f}]")
print(f"  t-statistic: {results3.tvalues['did']:.3f}")
print(f"  p-value: {results3.pvalues['did']:.4f}")

# Preferred specification summary
print("\n" + "=" * 60)
print("PREFERRED ESTIMATE (Model 3 - Full specification)")
print("=" * 60)
print(f"\nEffect of DACA eligibility on full-time employment:")
print(f"  Point estimate: {results3.params['did']:.4f} ({results3.params['did']*100:.2f} percentage points)")
print(f"  Standard error: {results3.bse['did']:.4f}")
print(f"  95% CI: [{results3.conf_int().loc['did', 0]:.4f}, {results3.conf_int().loc['did', 1]:.4f}]")
print(f"  Sample size: {len(df):,}")

# Year-by-year analysis for event study
print("\n" + "=" * 60)
print("Event Study Analysis")
print("=" * 60)

# Create year-treatment interactions
years_analysis = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_results = []

for year in years_analysis:
    subset = df[df['YEAR'] == year]
    n_treat = (subset['treated'] == 1).sum()
    n_ctrl = (subset['treated'] == 0).sum()

    ft_treat = weighted_mean(subset[subset['treated'] == 1], 'fulltime', 'PERWT')
    ft_ctrl = weighted_mean(subset[subset['treated'] == 0], 'fulltime', 'PERWT')
    gap = ft_treat - ft_ctrl

    event_results.append({
        'Year': year,
        'N_Treatment': n_treat,
        'N_Control': n_ctrl,
        'FT_Treatment': ft_treat,
        'FT_Control': ft_ctrl,
        'Gap': gap
    })

event_df = pd.DataFrame(event_results)
print("\nYear-by-year treatment-control gap in full-time employment:")
print(event_df.to_string(index=False))

# Pre-trend test
print("\n" + "=" * 60)
print("Pre-Trend Analysis")
print("=" * 60)

pre_gaps = event_df[event_df['Year'] <= 2011]['Gap'].values
years_pre = event_df[event_df['Year'] <= 2011]['Year'].values

# Simple linear regression on pre-period gaps
slope, intercept, r_value, p_value, std_err = stats.linregress(years_pre, pre_gaps)
print(f"\nLinear trend in pre-period gaps:")
print(f"  Slope: {slope:.6f} per year")
print(f"  Std Error: {std_err:.6f}")
print(f"  p-value for slope = 0: {p_value:.4f}")

if p_value > 0.05:
    print("  --> No significant pre-trend detected (parallel trends assumption supported)")
else:
    print("  --> WARNING: Significant pre-trend detected")

# Robustness: Different age bandwidths
print("\n" + "=" * 60)
print("Robustness: Different Age Bandwidths")
print("=" * 60)

# Reload for different bandwidths
bandwidths = [(27, 29, 32, 34), (26, 30, 31, 35)]  # narrow, baseline
bandwidth_results = []

for t_low, t_high, c_low, c_high in bandwidths:
    df_bw = df.copy()
    df_bw = df_bw[(df_bw['age_2012'] >= t_low) & (df_bw['age_2012'] <= c_high)]
    df_bw['treated_bw'] = ((df_bw['age_2012'] >= t_low) & (df_bw['age_2012'] <= t_high)).astype(int)
    df_bw['did_bw'] = df_bw['treated_bw'] * df_bw['post']

    formula_bw = 'fulltime ~ treated_bw + did_bw + female + married + educ_hs + educ_college + C(YEAR) + C(STATEFIP)'
    model_bw = smf.wls(formula_bw, data=df_bw, weights=df_bw['PERWT'])
    results_bw = model_bw.fit(cov_type='cluster', cov_kwds={'groups': df_bw['STATEFIP']})

    bandwidth_results.append({
        'Treatment Ages': f'{t_low}-{t_high}',
        'Control Ages': f'{c_low}-{c_high}',
        'N': len(df_bw),
        'DiD Estimate': results_bw.params['did_bw'],
        'SE': results_bw.bse['did_bw']
    })

bw_df = pd.DataFrame(bandwidth_results)
print("\n" + bw_df.to_string(index=False))

# Subgroup analysis by gender
print("\n" + "=" * 60)
print("Subgroup Analysis by Gender")
print("=" * 60)

for sex, label in [(1, 'Men'), (2, 'Women')]:
    df_sex = df[df['SEX'] == sex]

    formula_sex = 'fulltime ~ treated + did + married + educ_hs + educ_college + C(YEAR) + C(STATEFIP)'
    model_sex = smf.wls(formula_sex, data=df_sex, weights=df_sex['PERWT'])
    results_sex = model_sex.fit(cov_type='cluster', cov_kwds={'groups': df_sex['STATEFIP']})

    print(f"\n{label}:")
    print(f"  N = {len(df_sex):,}")
    print(f"  DiD estimate: {results_sex.params['did']:.4f} (SE: {results_sex.bse['did']:.4f})")
    print(f"  95% CI: [{results_sex.conf_int().loc['did', 0]:.4f}, {results_sex.conf_int().loc['did', 1]:.4f}]")

# Save key results for the report
print("\n" + "=" * 60)
print("Saving Results for Report")
print("=" * 60)

# Create results summary
results_summary = {
    'preferred_estimate': results3.params['did'],
    'preferred_se': results3.bse['did'],
    'preferred_ci_low': results3.conf_int().loc['did', 0],
    'preferred_ci_high': results3.conf_int().loc['did', 1],
    'sample_size': len(df),
    'n_treatment': df['treated'].sum(),
    'n_control': (1-df['treated']).sum(),
}

# Save event study data for plotting
event_df.to_csv('event_study_data.csv', index=False)
results_df.to_csv('summary_stats.csv', index=False)

print("\nResults saved to CSV files.")

# Full regression table for Model 3
print("\n" + "=" * 60)
print("Full Regression Output (Model 3)")
print("=" * 60)

# Print only key coefficients
key_vars = ['Intercept', 'treated', 'did', 'female', 'married', 'educ_hs', 'educ_college']
print("\nVariable             Coef      SE       t-stat    p-value")
print("-" * 60)
for var in key_vars:
    if var in results3.params.index:
        coef = results3.params[var]
        se = results3.bse[var]
        tstat = results3.tvalues[var]
        pval = results3.pvalues[var]
        print(f"{var:20s} {coef:8.4f}  {se:7.4f}  {tstat:7.3f}  {pval:7.4f}")

print(f"\nR-squared: {results3.rsquared:.4f}")
print(f"Number of observations: {results3.nobs:,.0f}")
print(f"Number of state clusters: {df['STATEFIP'].nunique()}")

print("\n" + "=" * 60)
print("Analysis Complete")
print("=" * 60)
