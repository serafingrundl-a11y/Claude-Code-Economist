"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the US.

Author: Replication Study
Date: 2026-01-25
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
# 1. LOAD DATA (CHUNKED APPROACH DUE TO FILE SIZE)
# =============================================================================
print("\n[1] Loading ACS data (chunked approach for large file)...")

# Define columns to keep (minimize memory)
cols_to_keep = [
    'YEAR', 'STATEFIP', 'PUMA', 'METRO', 'GQ', 'PERWT',
    'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
    'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EMPSTAT', 'UHRSWORK'
]

# Read in chunks and filter early
chunk_size = 500000
chunks = []
total_rows = 0

for chunk in pd.read_csv('data/data.csv', chunksize=chunk_size, usecols=cols_to_keep):
    total_rows += len(chunk)

    # Apply early filters to reduce memory
    # Hispanic-Mexican AND born in Mexico AND non-citizen
    filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Hispanic-Mexican
        (chunk['BPL'] == 200) &    # Born in Mexico
        (chunk['CITIZEN'] == 3)    # Non-citizen
    ].copy()

    if len(filtered) > 0:
        chunks.append(filtered)

    if total_rows % 5000000 == 0:
        print(f"  Processed {total_rows:,} rows...")

print(f"\nTotal rows scanned: {total_rows:,}")

# Concatenate filtered chunks
df = pd.concat(chunks, ignore_index=True)
print(f"Rows after initial filtering (Hispanic-Mexican, Mexico-born, non-citizen): {len(df):,}")

# Clear memory
del chunks

# =============================================================================
# 2. ADDITIONAL SAMPLE RESTRICTIONS
# =============================================================================
print("\n[2] Applying additional sample restrictions...")

# Restrict to working age (18-64)
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)]
print(f"After working age (18-64) restriction: {len(df):,}")

# Exclude 2012 (DACA implemented mid-year, cannot distinguish pre/post)
df = df[df['YEAR'] != 2012]
print(f"After excluding 2012: {len(df):,}")

# Exclude group quarters (institutional population)
df = df[df['GQ'].isin([1, 2, 5])]  # Household population only
print(f"After excluding group quarters: {len(df):,}")

# Remove observations with missing key variables
df = df[df['YRIMMIG'] > 0]  # Valid year of immigration
print(f"After removing missing YRIMMIG: {len(df):,}")

df = df[df['BIRTHYR'] > 0]  # Valid birth year
print(f"After removing missing BIRTHYR: {len(df):,}")

# =============================================================================
# 3. CREATE VARIABLES
# =============================================================================
print("\n[3] Creating analysis variables...")

# Age at arrival in US
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Clean age at arrival (should be non-negative and reasonable)
df = df[(df['age_at_arrival'] >= 0) & (df['age_at_arrival'] <= df['AGE'])]
print(f"After cleaning age at arrival: {len(df):,}")

# DACA Eligibility Criteria:
# 1. Arrived before 16th birthday
# 2. Born after June 15, 1981 (not yet 31 as of June 15, 2012)
# 3. Lived continuously in US since June 15, 2007 (use YRIMMIG <= 2007)
# 4. Non-citizen (already restricted above)

# Criterion 1: Arrived before age 16
df['arrived_before_16'] = (df['age_at_arrival'] < 16).astype(int)

# Criterion 2: Born after June 15, 1981
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Born after June 15, 1981 means BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
df['born_after_june1981'] = (
    (df['BIRTHYR'] >= 1982) |
    ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))
).astype(int)

# Criterion 3: Present since June 2007 (YRIMMIG <= 2007)
df['present_since_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# DACA Eligible: meets all criteria
df['daca_eligible'] = (
    (df['arrived_before_16'] == 1) &
    (df['born_after_june1981'] == 1) &
    (df['present_since_2007'] == 1)
).astype(int)

print(f"\nDACA eligibility components:")
print(f"  Arrived before 16: {df['arrived_before_16'].sum():,} ({df['arrived_before_16'].mean()*100:.1f}%)")
print(f"  Born after June 1981: {df['born_after_june1981'].sum():,} ({df['born_after_june1981'].mean()*100:.1f}%)")
print(f"  Present since 2007: {df['present_since_2007'].sum():,} ({df['present_since_2007'].mean()*100:.1f}%)")
print(f"  DACA eligible (all criteria): {df['daca_eligible'].sum():,} ({df['daca_eligible'].mean()*100:.1f}%)")

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term
df['daca_x_post'] = df['daca_eligible'] * df['post']

# Outcome: Full-time employment (UHRSWORK >= 35)
# UHRSWORK = 0 means N/A or not working
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employed variable (EMPSTAT == 1)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Create other control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['age_sq'] = df['AGE'] ** 2
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['less_than_hs'] = (df['EDUC'] <= 5).astype(int)
df['high_school'] = (df['EDUC'] == 6).astype(int)
df['some_college'] = (df['EDUC'].isin([7, 8, 9])).astype(int)
df['college_plus'] = (df['EDUC'] >= 10).astype(int)

print(f"\nFull-time employment rate: {df['fulltime'].mean()*100:.1f}%")
print(f"Employment rate: {df['employed'].mean()*100:.1f}%")

# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[4] Descriptive Statistics")
print("="*80)

# Summary by DACA eligibility and time period
print("\n--- Sample Sizes by Group and Period ---")
summary = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['count', 'mean'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'PERWT': 'sum'
}).round(3)
print(summary)

# Full-time employment by group and period (unweighted)
print("\n--- Full-time Employment Rates ---")
ft_rates = df.groupby(['daca_eligible', 'post'])['fulltime'].mean().unstack()
ft_rates.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
ft_rates.index = ['Non-eligible', 'DACA-eligible']
print(ft_rates.round(4))

# Simple DiD calculation
print("\n--- Simple Difference-in-Differences ---")
pre_eligible = df[(df['daca_eligible']==1) & (df['post']==0)]['fulltime'].mean()
post_eligible = df[(df['daca_eligible']==1) & (df['post']==1)]['fulltime'].mean()
pre_noneligible = df[(df['daca_eligible']==0) & (df['post']==0)]['fulltime'].mean()
post_noneligible = df[(df['daca_eligible']==0) & (df['post']==1)]['fulltime'].mean()

diff_eligible = post_eligible - pre_eligible
diff_noneligible = post_noneligible - pre_noneligible
did_simple = diff_eligible - diff_noneligible

print(f"DACA-eligible:     Pre={pre_eligible:.4f}, Post={post_eligible:.4f}, Diff={diff_eligible:.4f}")
print(f"Non-eligible:      Pre={pre_noneligible:.4f}, Post={post_noneligible:.4f}, Diff={diff_noneligible:.4f}")
print(f"DiD estimate: {did_simple:.4f}")

# =============================================================================
# 5. MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n[5] Regression Analysis")
print("="*80)

# Prepare data for regression
# Create year dummies for fixed effects
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.wls(
    'fulltime ~ daca_eligible + post + daca_x_post',
    data=df,
    weights=df['PERWT']
).fit(cov_type='HC1')
print(model1.summary().tables[1])
print(f"\nDiD Coefficient: {model1.params['daca_x_post']:.4f}")
print(f"Standard Error:  {model1.bse['daca_x_post']:.4f}")
print(f"P-value:         {model1.pvalues['daca_x_post']:.4f}")
print(f"95% CI:          [{model1.conf_int().loc['daca_x_post', 0]:.4f}, {model1.conf_int().loc['daca_x_post', 1]:.4f}]")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls(
    'fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married',
    data=df,
    weights=df['PERWT']
).fit(cov_type='HC1')
print(model2.summary().tables[1])
print(f"\nDiD Coefficient: {model2.params['daca_x_post']:.4f}")
print(f"Standard Error:  {model2.bse['daca_x_post']:.4f}")
print(f"P-value:         {model2.pvalues['daca_x_post']:.4f}")
print(f"95% CI:          [{model2.conf_int().loc['daca_x_post', 0]:.4f}, {model2.conf_int().loc['daca_x_post', 1]:.4f}]")

# Model 3: DiD with demographic controls and education
print("\n--- Model 3: DiD with Demographics + Education ---")
model3 = smf.wls(
    'fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + high_school + some_college + college_plus',
    data=df,
    weights=df['PERWT']
).fit(cov_type='HC1')
print(model3.summary().tables[1])
print(f"\nDiD Coefficient: {model3.params['daca_x_post']:.4f}")
print(f"Standard Error:  {model3.bse['daca_x_post']:.4f}")
print(f"P-value:         {model3.pvalues['daca_x_post']:.4f}")
print(f"95% CI:          [{model3.conf_int().loc['daca_x_post', 0]:.4f}, {model3.conf_int().loc['daca_x_post', 1]:.4f}]")

# Model 4: Full model with year fixed effects
print("\n--- Model 4: Full Model with Year Fixed Effects ---")
df_reg = pd.concat([df, year_dummies], axis=1)
year_vars = [c for c in year_dummies.columns]

formula4 = 'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + high_school + some_college + college_plus + ' + ' + '.join(year_vars)
model4 = smf.wls(
    formula4,
    data=df_reg,
    weights=df_reg['PERWT']
).fit(cov_type='HC1')

print(f"DiD Coefficient: {model4.params['daca_x_post']:.4f}")
print(f"Standard Error:  {model4.bse['daca_x_post']:.4f}")
print(f"P-value:         {model4.pvalues['daca_x_post']:.4f}")
print(f"95% CI:          [{model4.conf_int().loc['daca_x_post', 0]:.4f}, {model4.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"R-squared:       {model4.rsquared:.4f}")
print(f"N observations:  {int(model4.nobs):,}")

# Model 5: Full model with year and state fixed effects
print("\n--- Model 5: Full Model with Year + State Fixed Effects ---")
df_reg2 = pd.concat([df_reg, state_dummies], axis=1)
state_vars = [c for c in state_dummies.columns]

formula5 = 'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + high_school + some_college + college_plus + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(state_vars)
model5 = smf.wls(
    formula5,
    data=df_reg2,
    weights=df_reg2['PERWT']
).fit(cov_type='HC1')

print(f"DiD Coefficient: {model5.params['daca_x_post']:.4f}")
print(f"Standard Error:  {model5.bse['daca_x_post']:.4f}")
print(f"P-value:         {model5.pvalues['daca_x_post']:.4f}")
print(f"95% CI:          [{model5.conf_int().loc['daca_x_post', 0]:.4f}, {model5.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"R-squared:       {model5.rsquared:.4f}")
print(f"N observations:  {int(model5.nobs):,}")

# =============================================================================
# 6. ROBUSTNESS CHECKS
# =============================================================================
print("\n[6] Robustness Checks")
print("="*80)

# Robustness 1: Alternative outcome - Employment (any)
print("\n--- Robustness 1: Employment (any work) as outcome ---")
model_r1 = smf.wls(
    'employed ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + high_school + some_college + college_plus + ' + ' + '.join(year_vars),
    data=df_reg,
    weights=df_reg['PERWT']
).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_r1.params['daca_x_post']:.4f}")
print(f"Standard Error:  {model_r1.bse['daca_x_post']:.4f}")
print(f"P-value:         {model_r1.pvalues['daca_x_post']:.4f}")

# Robustness 2: Restrict to ages 18-35 (younger workers more likely DACA eligible)
print("\n--- Robustness 2: Restricted to ages 18-35 ---")
df_young = df_reg[df_reg['AGE'] <= 35].copy()
model_r2 = smf.wls(
    'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + high_school + some_college + college_plus + ' + ' + '.join(year_vars),
    data=df_young,
    weights=df_young['PERWT']
).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_r2.params['daca_x_post']:.4f}")
print(f"Standard Error:  {model_r2.bse['daca_x_post']:.4f}")
print(f"P-value:         {model_r2.pvalues['daca_x_post']:.4f}")
print(f"N observations:  {int(model_r2.nobs):,}")

# Robustness 3: Men only
print("\n--- Robustness 3: Men only ---")
df_men = df_reg[df_reg['female'] == 0].copy()
model_r3 = smf.wls(
    'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + married + high_school + some_college + college_plus + ' + ' + '.join(year_vars),
    data=df_men,
    weights=df_men['PERWT']
).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_r3.params['daca_x_post']:.4f}")
print(f"Standard Error:  {model_r3.bse['daca_x_post']:.4f}")
print(f"P-value:         {model_r3.pvalues['daca_x_post']:.4f}")

# Robustness 4: Women only
print("\n--- Robustness 4: Women only ---")
df_women = df_reg[df_reg['female'] == 1].copy()
model_r4 = smf.wls(
    'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + married + high_school + some_college + college_plus + ' + ' + '.join(year_vars),
    data=df_women,
    weights=df_women['PERWT']
).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_r4.params['daca_x_post']:.4f}")
print(f"Standard Error:  {model_r4.bse['daca_x_post']:.4f}")
print(f"P-value:         {model_r4.pvalues['daca_x_post']:.4f}")

# Robustness 5: Different full-time threshold (40 hours)
print("\n--- Robustness 5: Full-time = 40+ hours ---")
df_reg['fulltime40'] = (df_reg['UHRSWORK'] >= 40).astype(int)
model_r5 = smf.wls(
    'fulltime40 ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + high_school + some_college + college_plus + ' + ' + '.join(year_vars),
    data=df_reg,
    weights=df_reg['PERWT']
).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_r5.params['daca_x_post']:.4f}")
print(f"Standard Error:  {model_r5.bse['daca_x_post']:.4f}")
print(f"P-value:         {model_r5.pvalues['daca_x_post']:.4f}")

# =============================================================================
# 7. EVENT STUDY / DYNAMIC EFFECTS
# =============================================================================
print("\n[7] Event Study - Year-by-Year Effects")
print("="*80)

# Create year interactions
for year in sorted(df['YEAR'].unique()):
    df_reg[f'daca_x_{year}'] = (df_reg['daca_eligible'] * (df_reg['YEAR'] == year)).astype(int)

# Reference year: 2011 (last pre-treatment year)
event_years = [y for y in sorted(df['YEAR'].unique()) if y != 2011]
event_vars = [f'daca_x_{y}' for y in event_years]

formula_event = 'fulltime ~ daca_eligible + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(event_vars) + ' + AGE + age_sq + female + married'
model_event = smf.wls(
    formula_event,
    data=df_reg,
    weights=df_reg['PERWT']
).fit(cov_type='HC1')

print("\nYear-by-Year DACA Effects (reference: 2011):")
print("-" * 50)
event_study_results = []
for year in event_years:
    var = f'daca_x_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    ci_low = model_event.conf_int().loc[var, 0]
    ci_high = model_event.conf_int().loc[var, 1]
    print(f"  {year}: coef = {coef:7.4f}  (SE = {se:.4f}, p = {pval:.3f})")
    event_study_results.append({
        'year': int(year),
        'coef': float(coef),
        'se': float(se),
        'pval': float(pval),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high)
    })

# =============================================================================
# 8. SUMMARY TABLE FOR REPORT
# =============================================================================
print("\n[8] Summary of Main Results")
print("="*80)

# Create summary table
results_summary = pd.DataFrame({
    'Model': [
        '(1) Basic DiD',
        '(2) + Demographics',
        '(3) + Education',
        '(4) + Year FE',
        '(5) + Year & State FE'
    ],
    'DiD Coefficient': [
        model1.params['daca_x_post'],
        model2.params['daca_x_post'],
        model3.params['daca_x_post'],
        model4.params['daca_x_post'],
        model5.params['daca_x_post']
    ],
    'Std. Error': [
        model1.bse['daca_x_post'],
        model2.bse['daca_x_post'],
        model3.bse['daca_x_post'],
        model4.bse['daca_x_post'],
        model5.bse['daca_x_post']
    ],
    'P-value': [
        model1.pvalues['daca_x_post'],
        model2.pvalues['daca_x_post'],
        model3.pvalues['daca_x_post'],
        model4.pvalues['daca_x_post'],
        model5.pvalues['daca_x_post']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model5.nobs)
    ]
})
print(results_summary.to_string(index=False))

# =============================================================================
# 9. SAVE KEY RESULTS
# =============================================================================
print("\n[9] Saving Results")
print("="*80)

# Save detailed results to file
with open('analysis_results.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - ANALYSIS RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("SAMPLE INFORMATION\n")
    f.write("-"*40 + "\n")
    f.write(f"Total observations: {len(df):,}\n")
    f.write(f"DACA-eligible: {df['daca_eligible'].sum():,}\n")
    f.write(f"Non-eligible: {(df['daca_eligible']==0).sum():,}\n")
    f.write(f"Pre-period obs: {(df['post']==0).sum():,}\n")
    f.write(f"Post-period obs: {(df['post']==1).sum():,}\n\n")

    f.write("PREFERRED ESTIMATE (Model 4: Year Fixed Effects)\n")
    f.write("-"*40 + "\n")
    f.write(f"Effect size (DiD coefficient): {model4.params['daca_x_post']:.4f}\n")
    f.write(f"Standard error: {model4.bse['daca_x_post']:.4f}\n")
    f.write(f"95% CI: [{model4.conf_int().loc['daca_x_post', 0]:.4f}, {model4.conf_int().loc['daca_x_post', 1]:.4f}]\n")
    f.write(f"P-value: {model4.pvalues['daca_x_post']:.4f}\n")
    f.write(f"N: {int(model4.nobs):,}\n\n")

    f.write("INTERPRETATION\n")
    f.write("-"*40 + "\n")
    effect_pct = model4.params['daca_x_post'] * 100
    f.write(f"DACA eligibility is associated with a {effect_pct:.2f} percentage point\n")
    if effect_pct > 0:
        f.write("increase in the probability of full-time employment.\n")
    else:
        f.write("change in the probability of full-time employment.\n")

    f.write("\n\nFULL MODEL RESULTS\n")
    f.write("-"*40 + "\n")
    f.write(str(model4.summary()))

print("Results saved to 'analysis_results.txt'")

# Save summary statistics
desc_stats = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['count', 'mean', 'std'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'less_than_hs': 'mean',
    'high_school': 'mean',
    'some_college': 'mean',
    'college_plus': 'mean'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved to 'descriptive_stats.csv'")

# =============================================================================
# 10. FINAL OUTPUT FOR REPORT
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY - PREFERRED ESTIMATE")
print("="*80)
print(f"\nResearch Question: Effect of DACA eligibility on full-time employment")
print(f"Sample: Hispanic-Mexican, Mexican-born, non-citizen, ages 18-64")
print(f"Method: Difference-in-Differences")
print(f"\nPreferred Estimate (Model 4 with year fixed effects):")
print(f"  Effect size: {model4.params['daca_x_post']:.4f}")
print(f"  Standard error: {model4.bse['daca_x_post']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['daca_x_post', 0]:.4f}, {model4.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"  P-value: {model4.pvalues['daca_x_post']:.4f}")
print(f"  Sample size: {int(model4.nobs):,}")
print(f"\nMean full-time employment rate: {df['fulltime'].mean():.4f}")
print("="*80)

# Export for LaTeX
latex_results = {
    'main_effect': model4.params['daca_x_post'],
    'main_se': model4.bse['daca_x_post'],
    'main_pval': model4.pvalues['daca_x_post'],
    'main_ci_low': model4.conf_int().loc['daca_x_post', 0],
    'main_ci_high': model4.conf_int().loc['daca_x_post', 1],
    'main_n': int(model4.nobs),
    'model1_effect': model1.params['daca_x_post'],
    'model1_se': model1.bse['daca_x_post'],
    'model1_pval': model1.pvalues['daca_x_post'],
    'model2_effect': model2.params['daca_x_post'],
    'model2_se': model2.bse['daca_x_post'],
    'model2_pval': model2.pvalues['daca_x_post'],
    'model3_effect': model3.params['daca_x_post'],
    'model3_se': model3.bse['daca_x_post'],
    'model3_pval': model3.pvalues['daca_x_post'],
    'model5_effect': model5.params['daca_x_post'],
    'model5_se': model5.bse['daca_x_post'],
    'model5_pval': model5.pvalues['daca_x_post'],
    'robust_emp_effect': model_r1.params['daca_x_post'],
    'robust_emp_se': model_r1.bse['daca_x_post'],
    'robust_emp_pval': model_r1.pvalues['daca_x_post'],
    'robust_young_effect': model_r2.params['daca_x_post'],
    'robust_young_se': model_r2.bse['daca_x_post'],
    'robust_young_pval': model_r2.pvalues['daca_x_post'],
    'robust_young_n': int(model_r2.nobs),
    'robust_men_effect': model_r3.params['daca_x_post'],
    'robust_men_se': model_r3.bse['daca_x_post'],
    'robust_men_pval': model_r3.pvalues['daca_x_post'],
    'robust_women_effect': model_r4.params['daca_x_post'],
    'robust_women_se': model_r4.bse['daca_x_post'],
    'robust_women_pval': model_r4.pvalues['daca_x_post'],
    'robust_ft40_effect': model_r5.params['daca_x_post'],
    'robust_ft40_se': model_r5.bse['daca_x_post'],
    'robust_ft40_pval': model_r5.pvalues['daca_x_post'],
    'mean_fulltime': df['fulltime'].mean(),
    'mean_employed': df['employed'].mean(),
    'n_daca_eligible': int(df['daca_eligible'].sum()),
    'n_non_eligible': int((df['daca_eligible']==0).sum()),
    'pre_eligible_ft': pre_eligible,
    'post_eligible_ft': post_eligible,
    'pre_nonelig_ft': pre_noneligible,
    'post_nonelig_ft': post_noneligible,
    'simple_did': did_simple,
    'event_study': event_study_results
}

# Save for LaTeX use
import json
with open('latex_results.json', 'w') as f:
    json.dump(latex_results, f, indent=2)
print("\nLaTeX-ready results saved to 'latex_results.json'")

print("\nAnalysis complete!")
