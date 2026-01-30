"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born non-citizens in the US.

Author: Independent Replication
Date: 2026
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
print("DACA REPLICATION ANALYSIS")
print("="*80)

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load the ACS data
data_path = "data/data.csv"
print(f"Loading data from: {data_path}")

# Read in chunks due to large file size
chunks = []
chunk_size = 500000
for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
    chunks.append(chunk)
    print(f"  Loaded chunk with {len(chunk):,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: Define Sample - Hispanic-Mexican, Mexican-born, Non-citizens
# =============================================================================
print("\n" + "="*80)
print("STEP 2: DEFINING SAMPLE")
print("="*80)

# Step 2a: Restrict to Hispanic-Mexican ethnicity
# HISPAN == 1 indicates Mexican
print(f"\nOriginal sample size: {len(df):,}")

df_hispanic_mexican = df[df['HISPAN'] == 1].copy()
print(f"After restricting to Hispanic-Mexican (HISPAN==1): {len(df_hispanic_mexican):,}")

# Step 2b: Restrict to born in Mexico
# BPL == 200 indicates Mexico
df_mexico_born = df_hispanic_mexican[df_hispanic_mexican['BPL'] == 200].copy()
print(f"After restricting to Mexican-born (BPL==200): {len(df_mexico_born):,}")

# Step 2c: Restrict to non-citizens
# CITIZEN == 3 indicates "Not a citizen"
df_noncitizen = df_mexico_born[df_mexico_born['CITIZEN'] == 3].copy()
print(f"After restricting to non-citizens (CITIZEN==3): {len(df_noncitizen):,}")

# Step 2d: Restrict to working-age population (16-64)
df_working_age = df_noncitizen[(df_noncitizen['AGE'] >= 16) & (df_noncitizen['AGE'] <= 64)].copy()
print(f"After restricting to working age 16-64: {len(df_working_age):,}")

# Step 2e: Restrict to non-institutional group quarters
# GQ == 1 (households) or GQ == 2 (additional households)
df_sample = df_working_age[df_working_age['GQ'].isin([1, 2, 5])].copy()
print(f"After restricting to non-institutional (GQ in 1,2,5): {len(df_sample):,}")

# Step 2f: Exclude 2012 (transitional year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# Step 2g: Require valid immigration year
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
print(f"After requiring valid YRIMMIG: {len(df_sample):,}")

# =============================================================================
# STEP 3: Define DACA Eligibility
# =============================================================================
print("\n" + "="*80)
print("STEP 3: DEFINING DACA ELIGIBILITY")
print("="*80)

# Calculate age at immigration
# Age at immigration = Year of immigration - Birth year
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# DACA Eligibility Criteria:
# 1. Arrived before age 16
# 2. Born after June 15, 1981 (under 31 as of June 15, 2012)
# 3. Immigrated by 2007 (continuous presence since June 2007)

# Criterion 1: Arrived before age 16
df_sample['arrived_before_16'] = (df_sample['age_at_immig'] < 16).astype(int)

# Criterion 2: Under 31 as of June 15, 2012
# Must be born 1982 or later (conservative; being born in 1981 Q3-Q4 could still qualify)
# For simplicity, use BIRTHYR >= 1982
df_sample['under_31_in_2012'] = (df_sample['BIRTHYR'] >= 1982).astype(int)

# Criterion 3: Present since at least June 2007 (5 years continuous presence)
df_sample['arrived_by_2007'] = (df_sample['YRIMMIG'] <= 2007).astype(int)

# DACA eligible if all three criteria met
df_sample['daca_eligible'] = ((df_sample['arrived_before_16'] == 1) &
                               (df_sample['under_31_in_2012'] == 1) &
                               (df_sample['arrived_by_2007'] == 1)).astype(int)

print("\nDACA Eligibility Criteria:")
print(f"  Arrived before age 16: {df_sample['arrived_before_16'].sum():,} ({df_sample['arrived_before_16'].mean()*100:.1f}%)")
print(f"  Under 31 in June 2012 (born 1982+): {df_sample['under_31_in_2012'].sum():,} ({df_sample['under_31_in_2012'].mean()*100:.1f}%)")
print(f"  Arrived by 2007: {df_sample['arrived_by_2007'].sum():,} ({df_sample['arrived_by_2007'].mean()*100:.1f}%)")
print(f"  DACA Eligible (all criteria): {df_sample['daca_eligible'].sum():,} ({df_sample['daca_eligible'].mean()*100:.1f}%)")

# =============================================================================
# STEP 4: Define Post-DACA Period and Outcome
# =============================================================================
print("\n" + "="*80)
print("STEP 4: DEFINING POST-PERIOD AND OUTCOME")
print("="*80)

# Post-DACA indicator (2013-2016)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
print(f"\nPost-DACA period (2013-2016): {df_sample['post'].sum():,} ({df_sample['post'].mean()*100:.1f}%)")

# Define full-time employment outcome
# Full-time = usually working 35+ hours per week
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)
print(f"Full-time employment (35+ hrs/week): {df_sample['fulltime'].sum():,} ({df_sample['fulltime'].mean()*100:.1f}%)")

# Interaction term for DID
df_sample['eligible_x_post'] = df_sample['daca_eligible'] * df_sample['post']

# =============================================================================
# STEP 5: Create Control Variables
# =============================================================================
print("\n" + "="*80)
print("STEP 5: CREATING CONTROL VARIABLES")
print("="*80)

# Age squared
df_sample['age_sq'] = df_sample['AGE'] ** 2

# Sex indicator (female = 1)
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)

# Marital status (married = 1)
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Education categories
# EDUC: 0=N/A, 1=None/preschool, 2=Grades1-4, 3=Grades5-8, 4=Grade9, 5=Grade10,
#       6=Grade11, 7=Grade12/HS diploma, 8=1yr college, 9=2yr college,
#       10=3yr college, 11=4yr/bachelor's, 12-14=5+yr/graduate
df_sample['educ_less_hs'] = (df_sample['EDUC'] < 6).astype(int)
df_sample['educ_hs'] = (df_sample['EDUC'] == 6).astype(int)
df_sample['educ_some_college'] = (df_sample['EDUC'].isin([7, 8, 9, 10])).astype(int)
df_sample['educ_college_plus'] = (df_sample['EDUC'] >= 11).astype(int)

print("Control variables created:")
print(f"  Female: {df_sample['female'].mean()*100:.1f}%")
print(f"  Married: {df_sample['married'].mean()*100:.1f}%")
print(f"  Education < HS: {df_sample['educ_less_hs'].mean()*100:.1f}%")
print(f"  Education HS: {df_sample['educ_hs'].mean()*100:.1f}%")
print(f"  Education Some College: {df_sample['educ_some_college'].mean()*100:.1f}%")
print(f"  Education College+: {df_sample['educ_college_plus'].mean()*100:.1f}%")

# =============================================================================
# STEP 6: Summary Statistics
# =============================================================================
print("\n" + "="*80)
print("STEP 6: SUMMARY STATISTICS")
print("="*80)

# Summary by eligibility status
print("\n--- Summary by DACA Eligibility Status ---")
summary_vars = ['AGE', 'female', 'married', 'educ_less_hs', 'educ_hs',
                'educ_some_college', 'educ_college_plus', 'UHRSWORK', 'fulltime']

eligible = df_sample[df_sample['daca_eligible'] == 1]
ineligible = df_sample[df_sample['daca_eligible'] == 0]

print(f"\n{'Variable':<25} {'Eligible':<15} {'Ineligible':<15} {'Difference':<15}")
print("-" * 70)
for var in summary_vars:
    elig_mean = eligible[var].mean()
    inelig_mean = ineligible[var].mean()
    diff = elig_mean - inelig_mean
    print(f"{var:<25} {elig_mean:<15.3f} {inelig_mean:<15.3f} {diff:<15.3f}")

print(f"\n{'N observations':<25} {len(eligible):<15,} {len(ineligible):<15,}")

# Summary by period
print("\n\n--- Summary by Period ---")
pre = df_sample[df_sample['post'] == 0]
post_df = df_sample[df_sample['post'] == 1]

print(f"\n{'Variable':<25} {'Pre-DACA':<15} {'Post-DACA':<15} {'Difference':<15}")
print("-" * 70)
for var in summary_vars:
    pre_mean = pre[var].mean()
    post_mean = post_df[var].mean()
    diff = post_mean - pre_mean
    print(f"{var:<25} {pre_mean:<15.3f} {post_mean:<15.3f} {diff:<15.3f}")

print(f"\n{'N observations':<25} {len(pre):<15,} {len(post_df):<15,}")

# 2x2 DID table for full-time employment
print("\n\n--- Difference-in-Differences Table: Full-Time Employment ---")
did_table = df_sample.groupby(['daca_eligible', 'post']).agg(
    fulltime_mean=('fulltime', 'mean'),
    n=('fulltime', 'count')
).round(4)

print("\nUnweighted means:")
print(did_table)

# Calculate DID estimate manually
pre_elig = df_sample[(df_sample['daca_eligible']==1) & (df_sample['post']==0)]['fulltime'].mean()
post_elig = df_sample[(df_sample['daca_eligible']==1) & (df_sample['post']==1)]['fulltime'].mean()
pre_inelig = df_sample[(df_sample['daca_eligible']==0) & (df_sample['post']==0)]['fulltime'].mean()
post_inelig = df_sample[(df_sample['daca_eligible']==0) & (df_sample['post']==1)]['fulltime'].mean()

diff_elig = post_elig - pre_elig
diff_inelig = post_inelig - pre_inelig
did_estimate = diff_elig - diff_inelig

print(f"\nDID Calculation (unweighted):")
print(f"  Eligible: Pre={pre_elig:.4f}, Post={post_elig:.4f}, Diff={diff_elig:.4f}")
print(f"  Ineligible: Pre={pre_inelig:.4f}, Post={post_inelig:.4f}, Diff={diff_inelig:.4f}")
print(f"  DID Estimate: {did_estimate:.4f}")

# =============================================================================
# STEP 7: Regression Analysis
# =============================================================================
print("\n" + "="*80)
print("STEP 7: REGRESSION ANALYSIS")
print("="*80)

# Create year dummies - ensure numeric
year_dummies = pd.get_dummies(df_sample['YEAR'], prefix='year', drop_first=True, dtype=float)
df_sample = pd.concat([df_sample.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1)

# Create state dummies - ensure numeric
state_dummies = pd.get_dummies(df_sample['STATEFIP'], prefix='state', drop_first=True, dtype=float)
df_sample = pd.concat([df_sample.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)

# Get column names for year and state dummies
year_cols = [col for col in df_sample.columns if col.startswith('year_')]
state_cols = [col for col in df_sample.columns if col.startswith('state_')]

# Ensure all variables are float
df_sample['daca_eligible'] = df_sample['daca_eligible'].astype(float)
df_sample['post'] = df_sample['post'].astype(float)
df_sample['eligible_x_post'] = df_sample['eligible_x_post'].astype(float)
df_sample['fulltime'] = df_sample['fulltime'].astype(float)
df_sample['AGE'] = df_sample['AGE'].astype(float)
df_sample['age_sq'] = df_sample['age_sq'].astype(float)
df_sample['female'] = df_sample['female'].astype(float)
df_sample['married'] = df_sample['married'].astype(float)
df_sample['educ_hs'] = df_sample['educ_hs'].astype(float)
df_sample['educ_some_college'] = df_sample['educ_some_college'].astype(float)
df_sample['educ_college_plus'] = df_sample['educ_college_plus'].astype(float)
df_sample['PERWT'] = df_sample['PERWT'].astype(float)

# Model 1: Basic DID without controls
print("\n--- Model 1: Basic DID (no controls) ---")
X1 = df_sample[['daca_eligible', 'post', 'eligible_x_post']].astype(float)
X1 = sm.add_constant(X1)
y = df_sample['fulltime'].astype(float)

model1 = sm.OLS(y, X1).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DID with demographic controls
print("\n\n--- Model 2: DID with demographic controls ---")
controls_demo = ['AGE', 'age_sq', 'female', 'married', 'educ_hs', 'educ_some_college', 'educ_college_plus']
X2 = df_sample[['daca_eligible', 'post', 'eligible_x_post'] + controls_demo].astype(float)
X2 = sm.add_constant(X2)

model2 = sm.OLS(y, X2).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DID with year fixed effects
print("\n\n--- Model 3: DID with demographic controls + year FE ---")
X3_cols = ['daca_eligible', 'eligible_x_post'] + controls_demo + year_cols
X3 = df_sample[X3_cols].astype(float)
X3 = sm.add_constant(X3)

model3 = sm.OLS(y, X3).fit(cov_type='HC1')
print("\nKey coefficients from Model 3:")
print(f"  daca_eligible:     {model3.params['daca_eligible']:.4f} (SE: {model3.bse['daca_eligible']:.4f})")
print(f"  eligible_x_post:   {model3.params['eligible_x_post']:.4f} (SE: {model3.bse['eligible_x_post']:.4f})")
print(f"  p-value for DID:   {model3.pvalues['eligible_x_post']:.4f}")

# Model 4: Full model with state and year fixed effects
print("\n\n--- Model 4: Full model with demographic controls + year FE + state FE ---")
X4_cols = ['daca_eligible', 'eligible_x_post'] + controls_demo + year_cols + state_cols
X4 = df_sample[X4_cols].astype(float)
X4 = sm.add_constant(X4)

model4 = sm.OLS(y, X4).fit(cov_type='HC1')
print("\nKey coefficients from Model 4 (PREFERRED SPECIFICATION):")
print(f"  daca_eligible:     {model4.params['daca_eligible']:.4f} (SE: {model4.bse['daca_eligible']:.4f})")
print(f"  eligible_x_post:   {model4.params['eligible_x_post']:.4f} (SE: {model4.bse['eligible_x_post']:.4f})")
print(f"  p-value for DID:   {model4.pvalues['eligible_x_post']:.4f}")
print(f"  95% CI for DID:    [{model4.conf_int().loc['eligible_x_post', 0]:.4f}, {model4.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"  R-squared:         {model4.rsquared:.4f}")
print(f"  N:                 {int(model4.nobs):,}")

# =============================================================================
# STEP 8: Weighted Regression (Using Person Weights)
# =============================================================================
print("\n" + "="*80)
print("STEP 8: WEIGHTED REGRESSION (PERWT)")
print("="*80)

# Model 5: Weighted full model
print("\n--- Model 5: Weighted full model (PREFERRED) ---")
model5 = sm.WLS(y, X4, weights=df_sample['PERWT']).fit(cov_type='HC1')
print("\nKey coefficients from Model 5 (Weighted, PREFERRED SPECIFICATION):")
print(f"  daca_eligible:     {model5.params['daca_eligible']:.4f} (SE: {model5.bse['daca_eligible']:.4f})")
print(f"  eligible_x_post:   {model5.params['eligible_x_post']:.4f} (SE: {model5.bse['eligible_x_post']:.4f})")
print(f"  p-value for DID:   {model5.pvalues['eligible_x_post']:.4f}")
print(f"  95% CI for DID:    [{model5.conf_int().loc['eligible_x_post', 0]:.4f}, {model5.conf_int().loc['eligible_x_post', 1]:.4f}]")

# =============================================================================
# STEP 9: Robustness Checks
# =============================================================================
print("\n" + "="*80)
print("STEP 9: ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Restrict to employed individuals
print("\n--- Robustness 1: Conditional on being employed (EMPSTAT==1) ---")
df_employed = df_sample[df_sample['EMPSTAT'] == 1].copy()
print(f"Sample size (employed): {len(df_employed):,}")

X_emp_cols = ['daca_eligible', 'eligible_x_post'] + controls_demo + year_cols + state_cols
X_emp = df_employed[X_emp_cols].astype(float)
X_emp = sm.add_constant(X_emp)
y_emp = df_employed['fulltime'].astype(float)

model_emp = sm.WLS(y_emp, X_emp, weights=df_employed['PERWT']).fit(cov_type='HC1')
print(f"  DID effect (employed only): {model_emp.params['eligible_x_post']:.4f} (SE: {model_emp.bse['eligible_x_post']:.4f})")
print(f"  p-value: {model_emp.pvalues['eligible_x_post']:.4f}")

# Robustness 2: Males only
print("\n--- Robustness 2: Males only ---")
df_male = df_sample[df_sample['female'] == 0].copy()
print(f"Sample size (males): {len(df_male):,}")

X_male = df_male[X4_cols].astype(float)
X_male = sm.add_constant(X_male)
y_male = df_male['fulltime'].astype(float)

model_male = sm.WLS(y_male, X_male, weights=df_male['PERWT']).fit(cov_type='HC1')
print(f"  DID effect (males): {model_male.params['eligible_x_post']:.4f} (SE: {model_male.bse['eligible_x_post']:.4f})")
print(f"  p-value: {model_male.pvalues['eligible_x_post']:.4f}")

# Robustness 3: Females only
print("\n--- Robustness 3: Females only ---")
df_female = df_sample[df_sample['female'] == 1].copy()
print(f"Sample size (females): {len(df_female):,}")

X_female = df_female[X4_cols].astype(float)
X_female = sm.add_constant(X_female)
y_female = df_female['fulltime'].astype(float)

model_female = sm.WLS(y_female, X_female, weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"  DID effect (females): {model_female.params['eligible_x_post']:.4f} (SE: {model_female.bse['eligible_x_post']:.4f})")
print(f"  p-value: {model_female.pvalues['eligible_x_post']:.4f}")

# Robustness 4: Alternative age cutoff for eligibility (born 1981+)
print("\n--- Robustness 4: Alternative age cutoff (born 1981+) ---")
df_sample['under_31_alt'] = (df_sample['BIRTHYR'] >= 1981).astype(float)
df_sample['daca_eligible_alt'] = ((df_sample['arrived_before_16'] == 1) &
                                   (df_sample['under_31_alt'] == 1) &
                                   (df_sample['arrived_by_2007'] == 1)).astype(float)
df_sample['eligible_x_post_alt'] = df_sample['daca_eligible_alt'] * df_sample['post']

print(f"DACA Eligible (alt): {df_sample['daca_eligible_alt'].sum():,.0f}")

X_alt_cols = ['daca_eligible_alt', 'eligible_x_post_alt'] + controls_demo + year_cols + state_cols
X_alt = df_sample[X_alt_cols].astype(float)
X_alt = sm.add_constant(X_alt)

model_alt = sm.WLS(y, X_alt, weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"  DID effect (alt cutoff): {model_alt.params['eligible_x_post_alt']:.4f} (SE: {model_alt.bse['eligible_x_post_alt']:.4f})")
print(f"  p-value: {model_alt.pvalues['eligible_x_post_alt']:.4f}")

# =============================================================================
# STEP 10: Event Study / Year-by-Year Effects
# =============================================================================
print("\n" + "="*80)
print("STEP 10: EVENT STUDY - YEAR-BY-YEAR EFFECTS")
print("="*80)

# Create year-specific interaction terms (omitting 2011 as reference)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for yr in years:
    df_sample[f'elig_x_{yr}'] = (df_sample['daca_eligible'] * (df_sample['YEAR'] == yr)).astype(float)

event_cols = [f'elig_x_{yr}' for yr in years]
X_event_cols = ['daca_eligible'] + event_cols + controls_demo + year_cols + state_cols
X_event = df_sample[X_event_cols].astype(float)
X_event = sm.add_constant(X_event)

model_event = sm.WLS(y, X_event, weights=df_sample['PERWT']).fit(cov_type='HC1')

print("\nYear-by-Year Treatment Effects (relative to 2011):")
print(f"{'Year':<10} {'Coefficient':<15} {'Std. Error':<15} {'p-value':<15}")
print("-" * 55)
for yr in years:
    coef = model_event.params[f'elig_x_{yr}']
    se = model_event.bse[f'elig_x_{yr}']
    pval = model_event.pvalues[f'elig_x_{yr}']
    print(f"{yr:<10} {coef:<15.4f} {se:<15.4f} {pval:<15.4f}")

# =============================================================================
# STEP 11: Final Results Summary
# =============================================================================
print("\n" + "="*80)
print("STEP 11: FINAL RESULTS SUMMARY")
print("="*80)

print("\n" + "="*80)
print("PREFERRED ESTIMATE (Model 5 - Weighted with full controls)")
print("="*80)
print(f"\nEffect of DACA eligibility on full-time employment:")
print(f"  Coefficient:   {model5.params['eligible_x_post']:.4f}")
print(f"  Std. Error:    {model5.bse['eligible_x_post']:.4f}")
print(f"  t-statistic:   {model5.tvalues['eligible_x_post']:.4f}")
print(f"  p-value:       {model5.pvalues['eligible_x_post']:.4f}")
print(f"  95% CI:        [{model5.conf_int().loc['eligible_x_post', 0]:.4f}, {model5.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"\n  Sample Size:   {int(model5.nobs):,}")
print(f"  R-squared:     {model5.rsquared:.4f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
effect_pp = model5.params['eligible_x_post'] * 100
print(f"\nDACA eligibility is associated with a {effect_pp:.2f} percentage point")
print(f"{'increase' if effect_pp > 0 else 'decrease'} in the probability of full-time employment.")
if model5.pvalues['eligible_x_post'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
elif model5.pvalues['eligible_x_post'] < 0.10:
    print("This effect is statistically significant at the 10% level.")
else:
    print("This effect is not statistically significant at conventional levels.")

# =============================================================================
# STEP 12: Save Results for Report
# =============================================================================
print("\n" + "="*80)
print("STEP 12: SAVING RESULTS")
print("="*80)

# Create results dictionary for export
results = {
    'preferred_estimate': model5.params['eligible_x_post'],
    'preferred_se': model5.bse['eligible_x_post'],
    'preferred_ci_low': model5.conf_int().loc['eligible_x_post', 0],
    'preferred_ci_high': model5.conf_int().loc['eligible_x_post', 1],
    'preferred_pvalue': model5.pvalues['eligible_x_post'],
    'sample_size': int(model5.nobs),
    'n_eligible': int(df_sample['daca_eligible'].sum()),
    'n_ineligible': int((1 - df_sample['daca_eligible']).sum()),
    'n_pre': int((1 - df_sample['post']).sum()),
    'n_post': int(df_sample['post'].sum()),
}

# Save summary statistics to CSV
summary_stats = df_sample[['AGE', 'female', 'married', 'educ_less_hs', 'educ_hs',
                           'educ_some_college', 'educ_college_plus', 'UHRSWORK',
                           'fulltime', 'daca_eligible', 'post', 'PERWT']].describe()
summary_stats.to_csv('summary_stats_90.csv')
print("Summary statistics saved to summary_stats_90.csv")

# Save regression results
reg_results = pd.DataFrame({
    'Model': ['Basic DID', 'With Demographics', 'With Year FE', 'Full (Unweighted)', 'Full (Weighted)'],
    'DID_Estimate': [model1.params['eligible_x_post'], model2.params['eligible_x_post'],
                     model3.params['eligible_x_post'], model4.params['eligible_x_post'],
                     model5.params['eligible_x_post']],
    'Std_Error': [model1.bse['eligible_x_post'], model2.bse['eligible_x_post'],
                  model3.bse['eligible_x_post'], model4.bse['eligible_x_post'],
                  model5.bse['eligible_x_post']],
    'P_Value': [model1.pvalues['eligible_x_post'], model2.pvalues['eligible_x_post'],
                model3.pvalues['eligible_x_post'], model4.pvalues['eligible_x_post'],
                model5.pvalues['eligible_x_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)],
    'R_squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared, model5.rsquared]
})
reg_results.to_csv('regression_results_90.csv', index=False)
print("Regression results saved to regression_results_90.csv")

# Save event study results
event_results = pd.DataFrame({
    'Year': years,
    'Coefficient': [model_event.params[f'elig_x_{yr}'] for yr in years],
    'Std_Error': [model_event.bse[f'elig_x_{yr}'] for yr in years],
    'P_Value': [model_event.pvalues[f'elig_x_{yr}'] for yr in years]
})
event_results.to_csv('event_study_90.csv', index=False)
print("Event study results saved to event_study_90.csv")

# Save DID table
did_summary = pd.DataFrame({
    'Group': ['Eligible', 'Eligible', 'Ineligible', 'Ineligible'],
    'Period': ['Pre', 'Post', 'Pre', 'Post'],
    'Mean_Fulltime': [pre_elig, post_elig, pre_inelig, post_inelig],
    'N': [len(df_sample[(df_sample['daca_eligible']==1) & (df_sample['post']==0)]),
          len(df_sample[(df_sample['daca_eligible']==1) & (df_sample['post']==1)]),
          len(df_sample[(df_sample['daca_eligible']==0) & (df_sample['post']==0)]),
          len(df_sample[(df_sample['daca_eligible']==0) & (df_sample['post']==1)])]
})
did_summary.to_csv('did_table_90.csv', index=False)
print("DID table saved to did_table_90.csv")

# Save robustness results
robustness_results = pd.DataFrame({
    'Specification': ['Employed Only', 'Males Only', 'Females Only', 'Alt Age Cutoff (1981+)'],
    'DID_Estimate': [model_emp.params['eligible_x_post'], model_male.params['eligible_x_post'],
                     model_female.params['eligible_x_post'], model_alt.params['eligible_x_post_alt']],
    'Std_Error': [model_emp.bse['eligible_x_post'], model_male.bse['eligible_x_post'],
                  model_female.bse['eligible_x_post'], model_alt.bse['eligible_x_post_alt']],
    'P_Value': [model_emp.pvalues['eligible_x_post'], model_male.pvalues['eligible_x_post'],
                model_female.pvalues['eligible_x_post'], model_alt.pvalues['eligible_x_post_alt']],
    'N': [int(model_emp.nobs), int(model_male.nobs), int(model_female.nobs), int(model_alt.nobs)]
})
robustness_results.to_csv('robustness_results_90.csv', index=False)
print("Robustness results saved to robustness_results_90.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
