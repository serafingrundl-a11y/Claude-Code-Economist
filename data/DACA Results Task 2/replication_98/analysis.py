"""
DACA Replication Study: Effect on Full-Time Employment
Difference-in-Differences Analysis

Research Question: Among ethnically Hispanic-Mexican, Mexican-born people living in the US,
what was the causal impact of DACA eligibility on full-time employment (35+ hours/week)?

Treatment group: Ages 26-30 on June 15, 2012 (born 1982-1986)
Control group: Ages 31-35 on June 15, 2012 (born 1977-1981)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: READ DATA IN CHUNKS AND FILTER
# =============================================================================

print("Loading and filtering data in chunks...")

# Columns we need
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
               'EMPSTAT', 'UHRSWORK']

# Read in chunks and filter
chunks = []
chunk_size = 500000

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size):
    # Filter for Hispanic-Mexican (HISPAN = 1)
    chunk = chunk[chunk['HISPAN'] == 1]
    # Filter for born in Mexico (BPL = 200)
    chunk = chunk[chunk['BPL'] == 200]
    # Filter for non-citizens (CITIZEN = 3)
    chunk = chunk[chunk['CITIZEN'] == 3]
    # Filter for valid immigration year
    chunk = chunk[chunk['YRIMMIG'] > 0]
    # Filter for birth years in our range (1977-1986)
    chunk = chunk[(chunk['BIRTHYR'] >= 1977) & (chunk['BIRTHYR'] <= 1986)]

    if len(chunk) > 0:
        chunks.append(chunk)
    print(f"Processed chunk, current filtered rows: {sum(len(c) for c in chunks):,}")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal filtered observations: {len(df):,}")

# =============================================================================
# STEP 2: APPLY ELIGIBILITY CRITERIA
# =============================================================================

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Filter for arrived before age 16
df = df[df['age_at_immig'] < 16].copy()
print(f"Arrived before age 16: {len(df):,}")

# Filter for arrived by 2007 (to have lived continuously since June 15, 2007)
df = df[df['YRIMMIG'] <= 2007].copy()
print(f"Arrived by 2007: {len(df):,}")

# =============================================================================
# STEP 3: DEFINE TREATMENT/CONTROL AND TIME PERIODS
# =============================================================================

# Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)
# Control: Born 1977-1981 (ages 31-35 on June 15, 2012)
df['treat'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)
df['control'] = ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)).astype(int)

# Verify all observations are in treatment or control
assert df['treat'].sum() + df['control'].sum() == len(df), "All obs should be treat or control"

# Exclude 2012 (implementation year - timing uncertainty)
df = df[df['YEAR'] != 2012].copy()

# Define post period (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Define outcome: Full-time employment (usually works 35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Create interaction term
df['treat_post'] = df['treat'] * df['post']

# Create additional variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] == 1).astype(int)
df['year_factor'] = df['YEAR'].astype(str)

print(f"\nFinal analysis sample: {len(df):,}")

# =============================================================================
# STEP 4: SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Sample sizes by group and period
print("\nSample Sizes (Unweighted):")
sample_counts = df.groupby(['treat', 'post']).size().unstack(fill_value=0)
sample_counts.index = ['Control (31-35)', 'Treatment (26-30)']
sample_counts.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(sample_counts)
print(f"\nTotal sample size: {len(df):,}")

# Weighted sample sizes
print("\nWeighted Sample Sizes (using PERWT):")
weighted_counts = df.groupby(['treat', 'post'])['PERWT'].sum().unstack(fill_value=0)
weighted_counts.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_counts.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(weighted_counts.round(0))

# Full-time employment rates by group and period (weighted)
print("\nFull-Time Employment Rates (Weighted):")
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

emp_rates = df.groupby(['treat', 'post']).apply(weighted_mean).unstack(fill_value=0)
emp_rates.index = ['Control (31-35)', 'Treatment (26-30)']
emp_rates.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(emp_rates.round(4))

# Calculate simple difference-in-differences
did_simple = (emp_rates.iloc[1, 1] - emp_rates.iloc[1, 0]) - (emp_rates.iloc[0, 1] - emp_rates.iloc[0, 0])
print(f"\nSimple Difference-in-Differences: {did_simple:.4f}")

# Demographic summary
print("\n--- Demographics by Treatment Status ---")
for grp, name in [(0, 'Control'), (1, 'Treatment')]:
    subset = df[df['treat'] == grp]
    print(f"\n{name} Group:")
    print(f"  N = {len(subset):,}")
    print(f"  Mean Age (in 2012): {(2012 - subset['BIRTHYR']).mean():.1f}")
    print(f"  % Female: {100*np.average(subset['female'], weights=subset['PERWT']):.1f}%")
    print(f"  % Married: {100*np.average(subset['married'], weights=subset['PERWT']):.1f}%")
    print(f"  Mean Year of Immigration: {np.average(subset['YRIMMIG'], weights=subset['PERWT']):.1f}")

# Employment rates by year
print("\n--- Full-Time Employment Rate by Year ---")
for year in sorted(df['YEAR'].unique()):
    subset = df[df['YEAR'] == year]
    treat_rate = np.average(subset[subset['treat']==1]['fulltime'],
                            weights=subset[subset['treat']==1]['PERWT'])
    control_rate = np.average(subset[subset['treat']==0]['fulltime'],
                              weights=subset[subset['treat']==0]['PERWT'])
    diff = treat_rate - control_rate
    print(f"  {year}: Treatment={treat_rate:.4f}, Control={control_rate:.4f}, Diff={diff:.4f}")

# =============================================================================
# STEP 5: DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================

print("\n" + "="*80)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION RESULTS")
print("="*80)

# Model 1: Basic DiD without covariates (weighted)
print("\n--- Model 1: Basic DiD (Weighted) ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df,
                  weights=df['PERWT']).fit()

print(f"Dependent Variable: Full-Time Employment (UHRSWORK >= 35)")
print(f"N = {int(model1.nobs):,}")
print(f"\n{'Variable':<20} {'Coef':>12} {'Std Err':>12} {'t':>10} {'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}")
print("-" * 84)
for var in ['Intercept', 'treat', 'post', 'treat_post']:
    print(f"{var:<20} {model1.params[var]:>12.6f} {model1.bse[var]:>12.6f} "
          f"{model1.tvalues[var]:>10.3f} {model1.pvalues[var]:>10.4f} "
          f"{model1.conf_int().loc[var, 0]:>10.6f} {model1.conf_int().loc[var, 1]:>10.6f}")
print(f"\nR-squared: {model1.rsquared:.4f}")

# Store results
did_coef = model1.params['treat_post']
did_se = model1.bse['treat_post']
did_ci = model1.conf_int().loc['treat_post']

# Model 2: DiD with year fixed effects
print("\n--- Model 2: DiD with Year Fixed Effects (Weighted) ---")
model2 = smf.wls('fulltime ~ treat + C(year_factor) + treat_post',
                  data=df,
                  weights=df['PERWT']).fit()

did_coef2 = model2.params['treat_post']
did_se2 = model2.bse['treat_post']
did_ci2 = model2.conf_int().loc['treat_post']
print(f"DiD Estimate (treat_post): {did_coef2:.6f}")
print(f"Standard Error: {did_se2:.6f}")
print(f"95% CI: [{did_ci2[0]:.6f}, {did_ci2[1]:.6f}]")
print(f"P-value: {model2.pvalues['treat_post']:.6f}")
print(f"R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with covariates
print("\n--- Model 3: DiD with Covariates (Weighted) ---")
model3 = smf.wls('fulltime ~ treat + C(year_factor) + treat_post + female + married + C(EDUC)',
                  data=df,
                  weights=df['PERWT']).fit()

did_coef3 = model3.params['treat_post']
did_se3 = model3.bse['treat_post']
did_ci3 = model3.conf_int().loc['treat_post']
print(f"DiD Estimate (treat_post): {did_coef3:.6f}")
print(f"Standard Error: {did_se3:.6f}")
print(f"95% CI: [{did_ci3[0]:.6f}, {did_ci3[1]:.6f}]")
print(f"P-value: {model3.pvalues['treat_post']:.6f}")
print(f"R-squared: {model3.rsquared:.4f}")

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with State and Year Fixed Effects (Weighted) ---")
model4 = smf.wls('fulltime ~ treat + C(year_factor) + C(STATEFIP) + treat_post + female + married + C(EDUC)',
                  data=df,
                  weights=df['PERWT']).fit()

did_coef4 = model4.params['treat_post']
did_se4 = model4.bse['treat_post']
did_ci4 = model4.conf_int().loc['treat_post']
print(f"DiD Estimate (treat_post): {did_coef4:.6f}")
print(f"Standard Error: {did_se4:.6f}")
print(f"95% CI: [{did_ci4[0]:.6f}, {did_ci4[1]:.6f}]")
print(f"P-value: {model4.pvalues['treat_post']:.6f}")
print(f"R-squared: {model4.rsquared:.4f}")

# =============================================================================
# STEP 6: ROBUST STANDARD ERRORS
# =============================================================================

print("\n" + "="*80)
print("MODELS WITH ROBUST STANDARD ERRORS (HC1)")
print("="*80)

# Model 1 with robust SEs
model1_robust = smf.wls('fulltime ~ treat + post + treat_post',
                         data=df,
                         weights=df['PERWT']).fit(cov_type='HC1')
did_coef_r = model1_robust.params['treat_post']
did_se_r = model1_robust.bse['treat_post']
did_ci_r = model1_robust.conf_int().loc['treat_post']
print(f"\nModel 1 (Basic DiD) with Robust SE:")
print(f"  DiD Estimate: {did_coef_r:.6f}")
print(f"  Robust SE: {did_se_r:.6f}")
print(f"  95% CI: [{did_ci_r[0]:.6f}, {did_ci_r[1]:.6f}]")
print(f"  P-value: {model1_robust.pvalues['treat_post']:.6f}")

# Model 4 with robust SEs (preferred specification)
model4_robust = smf.wls('fulltime ~ treat + C(year_factor) + C(STATEFIP) + treat_post + female + married + C(EDUC)',
                         data=df,
                         weights=df['PERWT']).fit(cov_type='HC1')
did_coef4_r = model4_robust.params['treat_post']
did_se4_r = model4_robust.bse['treat_post']
did_ci4_r = model4_robust.conf_int().loc['treat_post']
print(f"\nModel 4 (Full Model) with Robust SE:")
print(f"  DiD Estimate: {did_coef4_r:.6f}")
print(f"  Robust SE: {did_se4_r:.6f}")
print(f"  95% CI: [{did_ci4_r[0]:.6f}, {did_ci4_r[1]:.6f}]")
print(f"  P-value: {model4_robust.pvalues['treat_post']:.6f}")

# =============================================================================
# STEP 7: HETEROGENEITY ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("HETEROGENEITY ANALYSIS")
print("="*80)

# Analysis by sex
print("\n--- DiD by Sex ---")
for sex, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex]
    model_sex = smf.wls('fulltime ~ treat + post + treat_post',
                         data=df_sex,
                         weights=df_sex['PERWT']).fit(cov_type='HC1')
    print(f"{sex_name}: DiD = {model_sex.params['treat_post']:.4f} (Robust SE = {model_sex.bse['treat_post']:.4f}, N = {len(df_sex):,})")

# =============================================================================
# STEP 8: PRE-TREND ANALYSIS / EVENT STUDY
# =============================================================================

print("\n" + "="*80)
print("PRE-TREND AND EVENT STUDY ANALYSIS")
print("="*80)

# Pre-trend test: Is there a differential trend before treatment?
print("\n--- Pre-Trend Test ---")
df_pre = df[df['post'] == 0].copy()
df_pre['year_centered'] = df_pre['YEAR'] - 2011
df_pre['treat_year'] = df_pre['treat'] * df_pre['year_centered']

model_pretrend = smf.wls('fulltime ~ treat + year_centered + treat_year',
                          data=df_pre,
                          weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"Pre-trend interaction (treat*year): {model_pretrend.params['treat_year']:.6f}")
print(f"Robust SE: {model_pretrend.bse['treat_year']:.6f}")
print(f"P-value: {model_pretrend.pvalues['treat_year']:.6f}")
print("(Non-significant p-value suggests parallel pre-trends)")

# Event study: Year-by-year effects (2011 as reference)
print("\n--- Event Study (Year-by-Year Effects) ---")
print("Reference year: 2011")

# Create year dummies
years = sorted(df['YEAR'].unique())
ref_year = 2011
for year in years:
    if year != ref_year:
        df[f'year_{year}'] = (df['YEAR'] == year).astype(int)
        df[f'treat_year_{year}'] = df['treat'] * df[f'year_{year}']

# Build formula
year_vars = [f'year_{y}' for y in years if y != ref_year]
interact_vars = [f'treat_year_{y}' for y in years if y != ref_year]
formula_event = 'fulltime ~ treat + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(interact_vars)

model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nTreatment * Year Coefficients:")
print(f"{'Year':<8} {'Coefficient':>12} {'Robust SE':>12} {'P-value':>10} {'Significant':>12}")
print("-" * 54)
for year in sorted(years):
    if year != ref_year:
        var = f'treat_year_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        pval = model_event.pvalues[var]
        sig = "*" if pval < 0.05 else ("+" if pval < 0.10 else "")
        print(f"{year:<8} {coef:>12.4f} {se:>12.4f} {pval:>10.4f} {sig:>12}")

# =============================================================================
# STEP 9: PLACEBO TEST
# =============================================================================

print("\n" + "="*80)
print("PLACEBO TEST")
print("="*80)

# Test: Use only pre-period data, pretend 2009 was treatment year
print("\n--- Placebo Test: Fake Treatment in 2009 ---")
df_placebo = df[df['YEAR'] <= 2011].copy()
df_placebo['fake_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treat_fake_post'] = df_placebo['treat'] * df_placebo['fake_post']

model_placebo = smf.wls('fulltime ~ treat + fake_post + treat_fake_post',
                         data=df_placebo,
                         weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD: {model_placebo.params['treat_fake_post']:.6f}")
print(f"Robust SE: {model_placebo.bse['treat_fake_post']:.6f}")
print(f"P-value: {model_placebo.pvalues['treat_fake_post']:.6f}")
print("(Non-significant result supports validity of research design)")

# =============================================================================
# STEP 10: SUMMARY TABLE
# =============================================================================

print("\n" + "="*80)
print("SUMMARY OF ALL RESULTS")
print("="*80)

print("\n--- Main Results Table ---")
print(f"{'Model':<35} {'DiD Est':>10} {'SE':>10} {'95% CI':>24} {'p-value':>10}")
print("-" * 89)
print(f"{'(1) Basic DiD':<35} {did_coef:>10.4f} {did_se:>10.4f} [{did_ci[0]:>9.4f}, {did_ci[1]:>9.4f}] {model1.pvalues['treat_post']:>10.4f}")
print(f"{'(2) + Year FE':<35} {did_coef2:>10.4f} {did_se2:>10.4f} [{did_ci2[0]:>9.4f}, {did_ci2[1]:>9.4f}] {model2.pvalues['treat_post']:>10.4f}")
print(f"{'(3) + Covariates':<35} {did_coef3:>10.4f} {did_se3:>10.4f} [{did_ci3[0]:>9.4f}, {did_ci3[1]:>9.4f}] {model3.pvalues['treat_post']:>10.4f}")
print(f"{'(4) + State FE':<35} {did_coef4:>10.4f} {did_se4:>10.4f} [{did_ci4[0]:>9.4f}, {did_ci4[1]:>9.4f}] {model4.pvalues['treat_post']:>10.4f}")

print("\n--- With Robust Standard Errors (HC1) ---")
print(f"{'Model':<35} {'DiD Est':>10} {'Rob SE':>10} {'95% CI':>24} {'p-value':>10}")
print("-" * 89)
print(f"{'(1) Basic DiD':<35} {did_coef_r:>10.4f} {did_se_r:>10.4f} [{did_ci_r[0]:>9.4f}, {did_ci_r[1]:>9.4f}] {model1_robust.pvalues['treat_post']:>10.4f}")
print(f"{'(4) Full Model':<35} {did_coef4_r:>10.4f} {did_se4_r:>10.4f} [{did_ci4_r[0]:>9.4f}, {did_ci4_r[1]:>9.4f}] {model4_robust.pvalues['treat_post']:>10.4f}")

# =============================================================================
# STEP 11: PREFERRED ESTIMATE
# =============================================================================

print("\n" + "="*80)
print("PREFERRED ESTIMATE")
print("="*80)

print(f"""
PREFERRED SPECIFICATION: Model 4 with Robust Standard Errors
- Weighted least squares using survey weights (PERWT)
- Year fixed effects
- State fixed effects
- Demographic covariates (sex, marital status, education)
- Heteroskedasticity-robust (HC1) standard errors

Sample:
  Total N = {len(df):,}
  Treatment Group (ages 26-30 on June 15, 2012): {df['treat'].sum():,}
  Control Group (ages 31-35 on June 15, 2012): {(df['treat']==0).sum():,}

Effect on Full-Time Employment (35+ hours/week):
  DiD Estimate: {did_coef4_r:.4f}
  Robust SE: {did_se4_r:.4f}
  95% CI: [{did_ci4_r[0]:.4f}, {did_ci4_r[1]:.4f}]
  P-value: {model4_robust.pvalues['treat_post']:.4f}

Interpretation: DACA eligibility is associated with a {abs(did_coef4_r)*100:.1f} percentage point
{'increase' if did_coef4_r > 0 else 'decrease'} in full-time employment among eligible individuals
aged 26-30 compared to the control group aged 31-35.
""")

# =============================================================================
# STEP 12: SAVE RESULTS
# =============================================================================

# Save key results to file for report
with open('results_summary.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - RESULTS SUMMARY\n")
    f.write("="*70 + "\n\n")

    f.write("SAMPLE CONSTRUCTION:\n")
    f.write("-"*70 + "\n")
    f.write("Population: Hispanic-Mexican, born in Mexico, non-citizens\n")
    f.write("Additional eligibility: Arrived in US before age 16, by 2007\n")
    f.write(f"Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)\n")
    f.write(f"Control: Born 1977-1981 (ages 31-35 on June 15, 2012)\n")
    f.write(f"Pre-period: 2006-2011\n")
    f.write(f"Post-period: 2013-2016\n")
    f.write(f"Excluded: 2012 (treatment timing uncertainty)\n\n")

    f.write(f"SAMPLE SIZE:\n")
    f.write("-"*70 + "\n")
    f.write(f"Total N: {len(df):,}\n")
    f.write(f"Treatment Group: {df['treat'].sum():,}\n")
    f.write(f"Control Group: {(df['treat']==0).sum():,}\n\n")

    f.write("FULL-TIME EMPLOYMENT RATES (Weighted):\n")
    f.write("-"*70 + "\n")
    f.write(emp_rates.to_string() + "\n")
    f.write(f"\nSimple DiD: {did_simple:.4f}\n\n")

    f.write("REGRESSION RESULTS:\n")
    f.write("-"*70 + "\n")
    f.write(f"Model 1 (Basic DiD):     {did_coef:.4f} (SE={did_se:.4f})\n")
    f.write(f"Model 2 (+ Year FE):     {did_coef2:.4f} (SE={did_se2:.4f})\n")
    f.write(f"Model 3 (+ Covariates):  {did_coef3:.4f} (SE={did_se3:.4f})\n")
    f.write(f"Model 4 (+ State FE):    {did_coef4:.4f} (SE={did_se4:.4f})\n\n")

    f.write("PREFERRED ESTIMATE (Model 4 with Robust SE):\n")
    f.write("-"*70 + "\n")
    f.write(f"DiD Effect: {did_coef4_r:.6f}\n")
    f.write(f"Robust SE: {did_se4_r:.6f}\n")
    f.write(f"95% CI: [{did_ci4_r[0]:.6f}, {did_ci4_r[1]:.6f}]\n")
    f.write(f"P-value: {model4_robust.pvalues['treat_post']:.6f}\n\n")

    f.write("PRE-TREND TEST:\n")
    f.write("-"*70 + "\n")
    f.write(f"Coefficient: {model_pretrend.params['treat_year']:.6f}\n")
    f.write(f"P-value: {model_pretrend.pvalues['treat_year']:.6f}\n\n")

    f.write("PLACEBO TEST:\n")
    f.write("-"*70 + "\n")
    f.write(f"Coefficient: {model_placebo.params['treat_fake_post']:.6f}\n")
    f.write(f"P-value: {model_placebo.pvalues['treat_fake_post']:.6f}\n")

print("\nResults saved to results_summary.txt")

# Save event study coefficients for plotting
event_coefs = []
for year in sorted(years):
    if year != ref_year:
        var = f'treat_year_{year}'
        event_coefs.append({
            'year': year,
            'coef': model_event.params[var],
            'se': model_event.bse[var],
            'ci_low': model_event.conf_int().loc[var, 0],
            'ci_high': model_event.conf_int().loc[var, 1]
        })
    else:
        event_coefs.append({
            'year': year,
            'coef': 0,
            'se': 0,
            'ci_low': 0,
            'ci_high': 0
        })
event_df = pd.DataFrame(event_coefs)
event_df.to_csv('event_study_coefs.csv', index=False)
print("Event study coefficients saved to event_study_coefs.csv")

# Save employment rates by year for plotting
yearly_rates = []
for year in sorted(df['YEAR'].unique()):
    subset = df[df['YEAR'] == year]
    treat_rate = np.average(subset[subset['treat']==1]['fulltime'],
                            weights=subset[subset['treat']==1]['PERWT'])
    control_rate = np.average(subset[subset['treat']==0]['fulltime'],
                              weights=subset[subset['treat']==0]['PERWT'])
    yearly_rates.append({
        'year': year,
        'treatment_rate': treat_rate,
        'control_rate': control_rate,
        'difference': treat_rate - control_rate
    })
yearly_df = pd.DataFrame(yearly_rates)
yearly_df.to_csv('yearly_employment_rates.csv', index=False)
print("Yearly employment rates saved to yearly_employment_rates.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
