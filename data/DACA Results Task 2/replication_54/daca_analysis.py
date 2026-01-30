"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the US.

Author: Replication Study ID 54
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. Load and Filter Data
# ============================================================================
print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

print("\n[1] Loading data...")
df = pd.read_csv('data/data.csv')
print(f"    Total observations loaded: {len(df):,}")

# ============================================================================
# 2. Define Sample Restrictions
# ============================================================================
print("\n[2] Applying sample restrictions...")

# Step 2a: Hispanic-Mexican ethnicity (HISPAN = 1)
df_sample = df[df['HISPAN'] == 1].copy()
print(f"    After Hispanic-Mexican filter: {len(df_sample):,}")

# Step 2b: Born in Mexico (BPL = 200)
df_sample = df_sample[df_sample['BPL'] == 200]
print(f"    After Mexico birthplace filter: {len(df_sample):,}")

# Step 2c: Not a citizen (CITIZEN = 3) - proxy for undocumented
df_sample = df_sample[df_sample['CITIZEN'] == 3]
print(f"    After non-citizen filter: {len(df_sample):,}")

# Step 2d: Filter valid immigration year data (YRIMMIG > 0)
df_sample = df_sample[df_sample['YRIMMIG'] > 0]
print(f"    After valid YRIMMIG filter: {len(df_sample):,}")

# Step 2e: Arrived before 16th birthday (DACA requirement)
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_arrival'] < 16]
print(f"    After arrived before age 16 filter: {len(df_sample):,}")

# Step 2f: Lived continuously in US since June 15, 2007 (YRIMMIG <= 2007)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]
print(f"    After continuous residence (YRIMMIG <= 2007) filter: {len(df_sample):,}")

# Step 2g: Treatment group (ages 26-30 in June 2012) - BIRTHYR 1982-1986
# Control group (ages 31-35 in June 2012) - BIRTHYR 1977-1981
df_sample['treat_group'] = np.where(df_sample['BIRTHYR'].between(1982, 1986), 1,
                           np.where(df_sample['BIRTHYR'].between(1977, 1981), 0, np.nan))
df_sample = df_sample[df_sample['treat_group'].notna()]
print(f"    After age group filter (26-30 or 31-35 in 2012): {len(df_sample):,}")

# Step 2h: Exclude 2012 (cannot distinguish pre/post DACA within year)
df_sample = df_sample[df_sample['YEAR'] != 2012]
print(f"    After excluding 2012: {len(df_sample):,}")

# ============================================================================
# 3. Create Analysis Variables
# ============================================================================
print("\n[3] Creating analysis variables...")

# Post-treatment indicator (2013-2016)
df_sample['post'] = np.where(df_sample['YEAR'] >= 2013, 1, 0)

# Full-time employment indicator (≥35 hours/week)
df_sample['fulltime'] = np.where(df_sample['UHRSWORK'] >= 35, 1, 0)

# DiD interaction term
df_sample['treat_post'] = df_sample['treat_group'] * df_sample['post']

# Age at time of survey
df_sample['age'] = df_sample['YEAR'] - df_sample['BIRTHYR']

# Create categorical variables for fixed effects
df_sample['year_factor'] = df_sample['YEAR'].astype(str)
df_sample['state_factor'] = df_sample['STATEFIP'].astype(str)
df_sample['birthyr_factor'] = df_sample['BIRTHYR'].astype(str)

print(f"    Variables created successfully")

# ============================================================================
# 4. Descriptive Statistics
# ============================================================================
print("\n[4] Generating descriptive statistics...")

print("\n    4.1 Sample size by group and period:")
print("    " + "=" * 50)
sample_counts = df_sample.groupby(['treat_group', 'post']).size().unstack()
sample_counts.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
sample_counts.index = ['Control (ages 31-35)', 'Treatment (ages 26-30)']
print(sample_counts.to_string())

print("\n    4.2 Weighted sample size by group and period:")
print("    " + "=" * 50)
weighted_counts = df_sample.groupby(['treat_group', 'post'])['PERWT'].sum().unstack()
weighted_counts.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
weighted_counts.index = ['Control (ages 31-35)', 'Treatment (ages 26-30)']
print(weighted_counts.astype(int).to_string())

print("\n    4.3 Full-time employment rate by group and period:")
print("    " + "=" * 50)

# Weighted means
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

ft_rates = df_sample.groupby(['treat_group', 'post']).apply(weighted_mean).unstack()
ft_rates.columns = ['Pre-DACA', 'Post-DACA']
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
print((ft_rates * 100).round(2).to_string())

# Calculate simple DiD estimate from means
did_simple = (ft_rates.loc['Treatment (26-30)', 'Post-DACA'] -
              ft_rates.loc['Treatment (26-30)', 'Pre-DACA']) - \
             (ft_rates.loc['Control (31-35)', 'Post-DACA'] -
              ft_rates.loc['Control (31-35)', 'Pre-DACA'])
print(f"\n    Simple DiD estimate: {did_simple*100:.2f} percentage points")

# ============================================================================
# 5. Summary Statistics Table
# ============================================================================
print("\n    4.4 Summary statistics for key variables:")
print("    " + "=" * 50)

summary_vars = ['fulltime', 'UHRSWORK', 'age', 'SEX', 'EDUC', 'NCHILD']
summary_stats = df_sample[summary_vars].describe().T
print(summary_stats[['mean', 'std', 'min', 'max']].round(3).to_string())

# ============================================================================
# 6. Regression Analysis
# ============================================================================
print("\n[5] Running regression analysis...")

# Model 1: Basic DiD (no controls)
print("\n    Model 1: Basic Difference-in-Differences")
print("    " + "-" * 50)

# Using WLS with person weights
model1 = smf.wls('fulltime ~ treat_group + post + treat_post',
                  data=df_sample, weights=df_sample['PERWT'])
results1 = model1.fit(cov_type='HC1')  # Robust standard errors
print(results1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n    Model 2: DiD with Demographic Controls (Age, Sex, Education)")
print("    " + "-" * 50)

df_sample['female'] = np.where(df_sample['SEX'] == 2, 1, 0)
model2 = smf.wls('fulltime ~ treat_group + post + treat_post + age + female + EDUC',
                  data=df_sample, weights=df_sample['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n    Model 3: DiD with Year Fixed Effects")
print("    " + "-" * 50)

model3 = smf.wls('fulltime ~ treat_group + treat_post + C(YEAR)',
                  data=df_sample, weights=df_sample['PERWT'])
results3 = model3.fit(cov_type='HC1')

# Print only key coefficients
print(f"    treat_group:    coef = {results3.params['treat_group']:.4f}, "
      f"se = {results3.bse['treat_group']:.4f}, "
      f"p = {results3.pvalues['treat_group']:.4f}")
print(f"    treat_post:     coef = {results3.params['treat_post']:.4f}, "
      f"se = {results3.bse['treat_post']:.4f}, "
      f"p = {results3.pvalues['treat_post']:.4f}")

# Model 4: DiD with year FE and controls (Preferred Model)
print("\n    Model 4: DiD with Year FE and Controls (PREFERRED)")
print("    " + "-" * 50)

model4 = smf.wls('fulltime ~ treat_group + treat_post + C(YEAR) + female + EDUC + NCHILD + C(MARST)',
                  data=df_sample, weights=df_sample['PERWT'])
results4 = model4.fit(cov_type='HC1')

print(f"    treat_group:    coef = {results4.params['treat_group']:.4f}, "
      f"se = {results4.bse['treat_group']:.4f}, "
      f"p = {results4.pvalues['treat_group']:.4f}")
print(f"    treat_post:     coef = {results4.params['treat_post']:.4f}, "
      f"se = {results4.bse['treat_post']:.4f}, "
      f"p = {results4.pvalues['treat_post']:.4f}")
print(f"    R-squared:      {results4.rsquared:.4f}")

# Model 5: DiD with state FE
print("\n    Model 5: DiD with Year and State Fixed Effects")
print("    " + "-" * 50)

model5 = smf.wls('fulltime ~ treat_group + treat_post + C(YEAR) + C(STATEFIP) + female + EDUC',
                  data=df_sample, weights=df_sample['PERWT'])
results5 = model5.fit(cov_type='HC1')

print(f"    treat_group:    coef = {results5.params['treat_group']:.4f}, "
      f"se = {results5.bse['treat_group']:.4f}, "
      f"p = {results5.pvalues['treat_group']:.4f}")
print(f"    treat_post:     coef = {results5.params['treat_post']:.4f}, "
      f"se = {results5.bse['treat_post']:.4f}, "
      f"p = {results5.pvalues['treat_post']:.4f}")
print(f"    R-squared:      {results5.rsquared:.4f}")

# ============================================================================
# 7. Robustness Checks
# ============================================================================
print("\n[6] Robustness checks...")

# 6.1 Placebo test using pre-treatment years only
print("\n    6.1 Placebo Test: Pre-treatment periods only (2006-2008 vs 2009-2011)")
print("    " + "-" * 50)

df_pre = df_sample[df_sample['YEAR'] <= 2011].copy()
df_pre['fake_post'] = np.where(df_pre['YEAR'] >= 2009, 1, 0)
df_pre['fake_treat_post'] = df_pre['treat_group'] * df_pre['fake_post']

placebo_model = smf.wls('fulltime ~ treat_group + fake_post + fake_treat_post',
                        data=df_pre, weights=df_pre['PERWT'])
placebo_results = placebo_model.fit(cov_type='HC1')

print(f"    fake_treat_post: coef = {placebo_results.params['fake_treat_post']:.4f}, "
      f"se = {placebo_results.bse['fake_treat_post']:.4f}, "
      f"p = {placebo_results.pvalues['fake_treat_post']:.4f}")

# 6.2 By sex
print("\n    6.2 Heterogeneous Effects by Sex")
print("    " + "-" * 50)

for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_sample[df_sample['SEX'] == sex_val]
    model_sex = smf.wls('fulltime ~ treat_group + post + treat_post',
                        data=df_sex, weights=df_sex['PERWT'])
    results_sex = model_sex.fit(cov_type='HC1')
    print(f"    {sex_name}: treat_post = {results_sex.params['treat_post']:.4f} "
          f"(SE = {results_sex.bse['treat_post']:.4f}), p = {results_sex.pvalues['treat_post']:.4f}")

# 6.3 Alternative age bandwidth
print("\n    6.3 Alternative Age Bandwidth (24-28 vs 33-37 in 2012)")
print("    " + "-" * 50)

df_alt = df[(df['HISPAN'] == 1) & (df['BPL'] == 200) & (df['CITIZEN'] == 3)].copy()
df_alt = df_alt[df_alt['YRIMMIG'] > 0]
df_alt['age_at_arrival'] = df_alt['YRIMMIG'] - df_alt['BIRTHYR']
df_alt = df_alt[(df_alt['age_at_arrival'] < 16) & (df_alt['YRIMMIG'] <= 2007)]
df_alt['treat_alt'] = np.where(df_alt['BIRTHYR'].between(1984, 1988), 1,
                      np.where(df_alt['BIRTHYR'].between(1975, 1979), 0, np.nan))
df_alt = df_alt[df_alt['treat_alt'].notna()]
df_alt = df_alt[df_alt['YEAR'] != 2012]
df_alt['post'] = np.where(df_alt['YEAR'] >= 2013, 1, 0)
df_alt['fulltime'] = np.where(df_alt['UHRSWORK'] >= 35, 1, 0)
df_alt['treat_post_alt'] = df_alt['treat_alt'] * df_alt['post']

model_alt = smf.wls('fulltime ~ treat_alt + post + treat_post_alt',
                    data=df_alt, weights=df_alt['PERWT'])
results_alt = model_alt.fit(cov_type='HC1')
print(f"    treat_post (alt bandwidth): {results_alt.params['treat_post_alt']:.4f} "
      f"(SE = {results_alt.bse['treat_post_alt']:.4f}), p = {results_alt.pvalues['treat_post_alt']:.4f}")

# ============================================================================
# 8. Event Study / Parallel Trends Check
# ============================================================================
print("\n[7] Event study analysis (parallel trends check)...")
print("    " + "-" * 50)

# Create year-specific treatment effects relative to 2011
df_sample['year_2006'] = np.where(df_sample['YEAR'] == 2006, 1, 0)
df_sample['year_2007'] = np.where(df_sample['YEAR'] == 2007, 1, 0)
df_sample['year_2008'] = np.where(df_sample['YEAR'] == 2008, 1, 0)
df_sample['year_2009'] = np.where(df_sample['YEAR'] == 2009, 1, 0)
df_sample['year_2010'] = np.where(df_sample['YEAR'] == 2010, 1, 0)
# 2011 is reference year
df_sample['year_2013'] = np.where(df_sample['YEAR'] == 2013, 1, 0)
df_sample['year_2014'] = np.where(df_sample['YEAR'] == 2014, 1, 0)
df_sample['year_2015'] = np.where(df_sample['YEAR'] == 2015, 1, 0)
df_sample['year_2016'] = np.where(df_sample['YEAR'] == 2016, 1, 0)

# Interactions
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_sample[f'treat_yr_{yr}'] = df_sample['treat_group'] * df_sample[f'year_{yr}']

event_formula = ('fulltime ~ treat_group + year_2006 + year_2007 + year_2008 + year_2009 + '
                 'year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + '
                 'treat_yr_2006 + treat_yr_2007 + treat_yr_2008 + treat_yr_2009 + treat_yr_2010 + '
                 'treat_yr_2013 + treat_yr_2014 + treat_yr_2015 + treat_yr_2016')

event_model = smf.wls(event_formula, data=df_sample, weights=df_sample['PERWT'])
event_results = event_model.fit(cov_type='HC1')

print("    Event study coefficients (relative to 2011):")
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for yr in event_years:
    coef = event_results.params[f'treat_yr_{yr}']
    se = event_results.bse[f'treat_yr_{yr}']
    pval = event_results.pvalues[f'treat_yr_{yr}']
    sig = "*" if pval < 0.05 else ""
    print(f"    {yr}: {coef:>8.4f} ({se:.4f}) {sig}")

# ============================================================================
# 9. Save Results
# ============================================================================
print("\n[8] Saving results...")

# Save event study coefficients
event_df = pd.DataFrame({
    'year': event_years,
    'coefficient': [event_results.params[f'treat_yr_{yr}'] for yr in event_years],
    'std_error': [event_results.bse[f'treat_yr_{yr}'] for yr in event_years],
    'p_value': [event_results.pvalues[f'treat_yr_{yr}'] for yr in event_years]
})
event_df.to_csv('event_study_results.csv', index=False)

# Create full-time employment rates by year and group for plotting
yearly_rates = df_sample.groupby(['treat_group', 'YEAR']).apply(
    lambda x: pd.Series({
        'ft_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()
yearly_rates.to_csv('yearly_ft_rates.csv', index=False)

# Save main regression results
results_summary = pd.DataFrame({
    'Model': ['Basic DiD', 'With Controls', 'Year FE', 'Year FE + Controls', 'Year + State FE'],
    'Coefficient': [results1.params['treat_post'], results2.params['treat_post'],
                    results3.params['treat_post'], results4.params['treat_post'],
                    results5.params['treat_post']],
    'Std_Error': [results1.bse['treat_post'], results2.bse['treat_post'],
                  results3.bse['treat_post'], results4.bse['treat_post'],
                  results5.bse['treat_post']],
    'P_Value': [results1.pvalues['treat_post'], results2.pvalues['treat_post'],
                results3.pvalues['treat_post'], results4.pvalues['treat_post'],
                results5.pvalues['treat_post']],
    'R_Squared': [results1.rsquared, results2.rsquared, results3.rsquared,
                  results4.rsquared, results5.rsquared]
})
results_summary.to_csv('regression_results.csv', index=False)

print("    Results saved to CSV files")

# ============================================================================
# 10. Final Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF MAIN FINDINGS")
print("=" * 70)

print(f"\nSample Size: {len(df_sample):,} observations (unweighted)")
print(f"Treatment Group (ages 26-30 in 2012): {int(df_sample['treat_group'].sum()):,}")
print(f"Control Group (ages 31-35 in 2012): {int((df_sample['treat_group']==0).sum()):,}")

print(f"\nPreferred Estimate (Model 4: Year FE + Controls):")
print(f"  DiD Coefficient: {results4.params['treat_post']:.4f}")
print(f"  Standard Error:  {results4.bse['treat_post']:.4f}")
print(f"  95% CI: [{results4.conf_int().loc['treat_post', 0]:.4f}, {results4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  P-value: {results4.pvalues['treat_post']:.4f}")

interpretation = "positive" if results4.params['treat_post'] > 0 else "negative"
sig_text = "statistically significant" if results4.pvalues['treat_post'] < 0.05 else "not statistically significant"
print(f"\nInterpretation: DACA eligibility is associated with a {interpretation} effect")
print(f"on full-time employment that is {sig_text} at the 5% level.")

print("\n" + "=" * 70)
print("Analysis complete.")
print("=" * 70)
