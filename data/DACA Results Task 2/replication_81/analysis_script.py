"""
DACA Replication Study - Analysis Script
Research ID: 81

Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the United States.

Identification Strategy: Difference-in-differences
- Treatment: Ages 26-30 on June 15, 2012 (birth years 1982-1986)
- Control: Ages 31-35 on June 15, 2012 (birth years 1977-1981)
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
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n1. LOADING DATA...")
df = pd.read_csv(r"C:\Users\seraf\DACA Results Task 2\replication_81\data\data.csv")
print(f"   Total observations loaded: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. SAMPLE CONSTRUCTION
# =============================================================================
print("\n2. CONSTRUCTING ANALYSIS SAMPLE...")

# Step 2a: Filter to Hispanic-Mexican (HISPAN = 1)
df_hispanic = df[df['HISPAN'] == 1].copy()
print(f"   After Hispanic-Mexican filter: {len(df_hispanic):,}")

# Step 2b: Filter to born in Mexico (BPL = 200)
df_mexico = df_hispanic[df_hispanic['BPL'] == 200].copy()
print(f"   After Mexico birthplace filter: {len(df_mexico):,}")

# Step 2c: Filter to non-citizens (CITIZEN = 3)
# This is our proxy for undocumented status
df_noncit = df_mexico[df_mexico['CITIZEN'] == 3].copy()
print(f"   After non-citizen filter: {len(df_noncit):,}")

# Step 2d: Define treatment and control groups based on birth year
# DACA cutoff: Must not have had 31st birthday as of June 15, 2012
# Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)
# Control: Born 1977-1981 (ages 31-35 on June 15, 2012)

df_noncit['treat'] = np.where(
    (df_noncit['BIRTHYR'] >= 1982) & (df_noncit['BIRTHYR'] <= 1986), 1,
    np.where((df_noncit['BIRTHYR'] >= 1977) & (df_noncit['BIRTHYR'] <= 1981), 0, np.nan)
)

# Keep only treatment and control groups
df_sample = df_noncit[df_noncit['treat'].notna()].copy()
print(f"   After age group filter (birth years 1977-1986): {len(df_sample):,}")

# Step 2e: Exclude 2012 (cannot distinguish pre/post within year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_sample):,}")

# Step 2f: Define post-treatment indicator
# Pre: 2006-2011, Post: 2013-2016
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Step 2g: Define outcome - full-time employment (35+ hours/week)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator (EMPSTAT == 1)
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# Interaction term
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

print(f"\n   Final analysis sample: {len(df_sample):,}")

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n3. DESCRIPTIVE STATISTICS")
print("-"*80)

# By group and period
for period_name, period_val in [('Pre-treatment (2006-2011)', 0), ('Post-treatment (2013-2016)', 1)]:
    print(f"\n{period_name}:")
    for group_name, group_val in [('Control (31-35)', 0), ('Treatment (26-30)', 1)]:
        subset = df_sample[(df_sample['post'] == period_val) & (df_sample['treat'] == group_val)]
        n_unweighted = len(subset)
        n_weighted = subset['PERWT'].sum()
        ft_rate = np.average(subset['fulltime'], weights=subset['PERWT'])
        emp_rate = np.average(subset['employed'], weights=subset['PERWT'])
        print(f"   {group_name}: N={n_unweighted:,} (weighted: {n_weighted:,.0f}), "
              f"FT rate={ft_rate:.4f}, Emp rate={emp_rate:.4f}")

# Summary statistics for key variables
print("\n\nSummary Statistics (Full Sample):")
print("-"*60)
summary_vars = ['YEAR', 'AGE', 'BIRTHYR', 'SEX', 'UHRSWORK', 'fulltime', 'employed', 'treat', 'post']
for var in summary_vars:
    if var in df_sample.columns:
        print(f"   {var:12s}: mean={df_sample[var].mean():8.3f}, "
              f"sd={df_sample[var].std():8.3f}, "
              f"min={df_sample[var].min():8.0f}, max={df_sample[var].max():8.0f}")

# =============================================================================
# 4. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n\n4. DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

# 4a. Simple 2x2 DiD (unweighted means)
print("\n4a. Simple 2x2 Difference-in-Differences Table (Weighted Means):")
print("-"*60)

# Calculate cell means
cells = {}
for post in [0, 1]:
    for treat in [0, 1]:
        subset = df_sample[(df_sample['post'] == post) & (df_sample['treat'] == treat)]
        cells[(post, treat)] = np.average(subset['fulltime'], weights=subset['PERWT'])

print(f"{'':20s} {'Control (31-35)':>15s} {'Treatment (26-30)':>17s} {'Difference':>12s}")
print(f"{'Pre (2006-2011)':20s} {cells[(0,0)]:15.4f} {cells[(0,1)]:17.4f} {cells[(0,1)]-cells[(0,0)]:12.4f}")
print(f"{'Post (2013-2016)':20s} {cells[(1,0)]:15.4f} {cells[(1,1)]:17.4f} {cells[(1,1)]-cells[(1,0)]:12.4f}")
print(f"{'Change':20s} {cells[(1,0)]-cells[(0,0)]:15.4f} {cells[(1,1)]-cells[(0,1)]:17.4f} ", end="")
did_simple = (cells[(1,1)] - cells[(0,1)]) - (cells[(1,0)] - cells[(0,0)])
print(f"{did_simple:12.4f}")
print(f"\nDifference-in-Differences Estimate: {did_simple:.4f}")

# 4b. Regression-based DiD (with weights)
print("\n\n4b. Regression-based DiD (OLS with survey weights):")
print("-"*60)

# Model 1: Basic DiD
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                 data=df_sample,
                 weights=df_sample['PERWT'])
results1 = model1.fit()
print("\nModel 1: Basic DiD (fulltime ~ treat + post + treat*post)")
print(results1.summary().tables[1])

# Model 2: DiD with year fixed effects
df_sample['year_fe'] = df_sample['YEAR'].astype(str)
model2 = smf.wls('fulltime ~ treat + treat_post + C(YEAR)',
                 data=df_sample,
                 weights=df_sample['PERWT'])
results2 = model2.fit()
print("\n\nModel 2: DiD with year fixed effects")
print(f"   treat_post coefficient: {results2.params['treat_post']:.4f}")
print(f"   Standard error: {results2.bse['treat_post']:.4f}")
print(f"   t-statistic: {results2.tvalues['treat_post']:.4f}")
print(f"   p-value: {results2.pvalues['treat_post']:.4f}")

# Model 3: DiD with covariates
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = df_sample['MARST'].isin([1, 2]).astype(int)

model3 = smf.wls('fulltime ~ treat + treat_post + female + married + C(YEAR)',
                 data=df_sample,
                 weights=df_sample['PERWT'])
results3 = model3.fit()
print("\n\nModel 3: DiD with year FE and demographics (female, married)")
print(f"   treat_post coefficient: {results3.params['treat_post']:.4f}")
print(f"   Standard error: {results3.bse['treat_post']:.4f}")
print(f"   t-statistic: {results3.tvalues['treat_post']:.4f}")
print(f"   p-value: {results3.pvalues['treat_post']:.4f}")

# Model 4: DiD with state fixed effects (if available)
print("\n\nModel 4: DiD with year and state fixed effects")
model4 = smf.wls('fulltime ~ treat + treat_post + C(YEAR) + C(STATEFIP)',
                 data=df_sample,
                 weights=df_sample['PERWT'])
results4 = model4.fit()
print(f"   treat_post coefficient: {results4.params['treat_post']:.4f}")
print(f"   Standard error: {results4.bse['treat_post']:.4f}")
print(f"   t-statistic: {results4.tvalues['treat_post']:.4f}")
print(f"   p-value: {results4.pvalues['treat_post']:.4f}")

# Model 5: Full specification
print("\n\nModel 5: Full specification (year FE + state FE + demographics)")
model5 = smf.wls('fulltime ~ treat + treat_post + female + married + C(YEAR) + C(STATEFIP)',
                 data=df_sample,
                 weights=df_sample['PERWT'])
results5 = model5.fit()
print(f"   treat_post coefficient: {results5.params['treat_post']:.4f}")
print(f"   Standard error: {results5.bse['treat_post']:.4f}")
print(f"   t-statistic: {results5.tvalues['treat_post']:.4f}")
print(f"   p-value: {results5.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{results5.conf_int().loc['treat_post', 0]:.4f}, {results5.conf_int().loc['treat_post', 1]:.4f}]")

# =============================================================================
# 5. ROBUSTNESS CHECKS
# =============================================================================
print("\n\n5. ROBUSTNESS CHECKS")
print("="*80)

# 5a. Alternative outcome: Employment (any hours)
print("\n5a. Alternative outcome: Employment (any hours > 0)")
model_emp = smf.wls('employed ~ treat + treat_post + C(YEAR) + C(STATEFIP)',
                    data=df_sample,
                    weights=df_sample['PERWT'])
results_emp = model_emp.fit()
print(f"   treat_post coefficient: {results_emp.params['treat_post']:.4f}")
print(f"   Standard error: {results_emp.bse['treat_post']:.4f}")
print(f"   p-value: {results_emp.pvalues['treat_post']:.4f}")

# 5b. By sex
print("\n5b. Heterogeneity by sex:")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    subset = df_sample[df_sample['SEX'] == sex_val]
    model_sex = smf.wls('fulltime ~ treat + treat_post + C(YEAR)',
                        data=subset,
                        weights=subset['PERWT'])
    results_sex = model_sex.fit()
    print(f"   {sex_name}: coef={results_sex.params['treat_post']:.4f}, "
          f"se={results_sex.bse['treat_post']:.4f}, "
          f"p={results_sex.pvalues['treat_post']:.4f}")

# 5c. Event study
print("\n5c. Event Study (Year-by-year effects):")
df_sample['treat_year'] = df_sample['treat'] * df_sample['YEAR']
# Create year dummies interacted with treatment
for year in df_sample['YEAR'].unique():
    df_sample[f'treat_x_{year}'] = df_sample['treat'] * (df_sample['YEAR'] == year).astype(int)

# Use 2011 as reference year
year_vars = [f'treat_x_{year}' for year in sorted(df_sample['YEAR'].unique()) if year != 2011]
formula_event = 'fulltime ~ treat + ' + ' + '.join(year_vars) + ' + C(YEAR)'
model_event = smf.wls(formula_event, data=df_sample, weights=df_sample['PERWT'])
results_event = model_event.fit()

print(f"   Reference year: 2011")
for year in sorted(df_sample['YEAR'].unique()):
    if year != 2011:
        varname = f'treat_x_{year}'
        print(f"   {year}: coef={results_event.params[varname]:.4f}, "
              f"se={results_event.bse[varname]:.4f}, "
              f"p={results_event.pvalues[varname]:.4f}")

# =============================================================================
# 6. SAMPLE SIZE AND COUNTS
# =============================================================================
print("\n\n6. SAMPLE SIZE SUMMARY")
print("="*80)

print(f"\nTotal observations in analysis sample: {len(df_sample):,}")
print(f"Weighted population: {df_sample['PERWT'].sum():,.0f}")

print("\nBy treatment status:")
for treat_val, treat_name in [(0, 'Control (31-35)'), (1, 'Treatment (26-30)')]:
    subset = df_sample[df_sample['treat'] == treat_val]
    print(f"   {treat_name}: {len(subset):,} (weighted: {subset['PERWT'].sum():,.0f})")

print("\nBy time period:")
for post_val, period_name in [(0, 'Pre-treatment (2006-2011)'), (1, 'Post-treatment (2013-2016)')]:
    subset = df_sample[df_sample['post'] == post_val]
    print(f"   {period_name}: {len(subset):,} (weighted: {subset['PERWT'].sum():,.0f})")

print("\nBy year:")
for year in sorted(df_sample['YEAR'].unique()):
    subset = df_sample[df_sample['YEAR'] == year]
    print(f"   {year}: {len(subset):,} (weighted: {subset['PERWT'].sum():,.0f})")

# =============================================================================
# 7. PREFERRED ESTIMATE
# =============================================================================
print("\n\n" + "="*80)
print("PREFERRED ESTIMATE SUMMARY")
print("="*80)

# Use Model 5 (full specification) as preferred
preferred = results5
print(f"\nPreferred Model: DiD with year and state fixed effects + demographics")
print(f"\nEffect of DACA eligibility on full-time employment:")
print(f"   Point estimate: {preferred.params['treat_post']:.4f}")
print(f"   Standard error: {preferred.bse['treat_post']:.4f}")
print(f"   95% CI: [{preferred.conf_int().loc['treat_post', 0]:.4f}, {preferred.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   t-statistic: {preferred.tvalues['treat_post']:.4f}")
print(f"   p-value: {preferred.pvalues['treat_post']:.4f}")
print(f"\nSample size: {len(df_sample):,}")
print(f"Weighted N: {df_sample['PERWT'].sum():,.0f}")

# Interpretation
direction = "increase" if preferred.params['treat_post'] > 0 else "decrease"
magnitude = abs(preferred.params['treat_post']) * 100
sig_level = "statistically significant at 5% level" if preferred.pvalues['treat_post'] < 0.05 else "not statistically significant at 5% level"
print(f"\nInterpretation: DACA eligibility is associated with a {magnitude:.2f} percentage point "
      f"{direction} in the probability of full-time employment. This effect is {sig_level}.")

# =============================================================================
# 8. SAVE RESULTS FOR LATEX
# =============================================================================
print("\n\nSaving results for LaTeX report...")

# Create summary table
results_dict = {
    'Model': ['Basic DiD', 'Year FE', 'Year FE + Demo', 'Year + State FE', 'Full Spec'],
    'Coefficient': [results1.params['treat_post'], results2.params['treat_post'],
                    results3.params['treat_post'], results4.params['treat_post'],
                    results5.params['treat_post']],
    'Std_Error': [results1.bse['treat_post'], results2.bse['treat_post'],
                  results3.bse['treat_post'], results4.bse['treat_post'],
                  results5.bse['treat_post']],
    'p_value': [results1.pvalues['treat_post'], results2.pvalues['treat_post'],
                results3.pvalues['treat_post'], results4.pvalues['treat_post'],
                results5.pvalues['treat_post']]
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv(r"C:\Users\seraf\DACA Results Task 2\replication_81\results_table.csv", index=False)

# Event study results
event_results = []
for year in sorted(df_sample['YEAR'].unique()):
    if year != 2011:
        varname = f'treat_x_{year}'
        event_results.append({
            'Year': year,
            'Coefficient': results_event.params[varname],
            'Std_Error': results_event.bse[varname],
            'CI_lower': results_event.conf_int().loc[varname, 0],
            'CI_upper': results_event.conf_int().loc[varname, 1]
        })
    else:
        event_results.append({
            'Year': 2011,
            'Coefficient': 0,
            'Std_Error': 0,
            'CI_lower': 0,
            'CI_upper': 0
        })
event_df = pd.DataFrame(event_results).sort_values('Year')
event_df.to_csv(r"C:\Users\seraf\DACA Results Task 2\replication_81\event_study.csv", index=False)

# Descriptive statistics by group
desc_stats = []
for post in [0, 1]:
    for treat in [0, 1]:
        subset = df_sample[(df_sample['post'] == post) & (df_sample['treat'] == treat)]
        desc_stats.append({
            'Period': 'Pre' if post == 0 else 'Post',
            'Group': 'Control' if treat == 0 else 'Treatment',
            'N': len(subset),
            'N_weighted': subset['PERWT'].sum(),
            'FT_rate': np.average(subset['fulltime'], weights=subset['PERWT']),
            'Emp_rate': np.average(subset['employed'], weights=subset['PERWT']),
            'Pct_female': np.average(subset['female'], weights=subset['PERWT']),
            'Pct_married': np.average(subset['married'], weights=subset['PERWT'])
        })
desc_df = pd.DataFrame(desc_stats)
desc_df.to_csv(r"C:\Users\seraf\DACA Results Task 2\replication_81\descriptive_stats.csv", index=False)

print("Results saved to CSV files.")
print("\nAnalysis complete!")
