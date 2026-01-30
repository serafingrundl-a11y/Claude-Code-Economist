"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among Hispanic-Mexican
Mexican-born non-citizens in the US, 2013-2016.

Identification Strategy: Difference-in-Differences
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

print("=" * 80)
print("DACA REPLICATION STUDY - DATA LOADING AND ANALYSIS")
print("=" * 80)

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n[Step 1] Loading ACS data...")

# Load in chunks due to large file size
chunks = []
chunk_size = 500000

# Columns needed for analysis
cols_needed = [
    'YEAR', 'PERWT', 'STATEFIP', 'AGE', 'SEX', 'BIRTHYR', 'BIRTHQTR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
    'EDUC', 'EDUCD', 'MARST', 'EMPSTAT', 'UHRSWORK', 'LABFORCE'
]

# Read data
for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size):
    chunks.append(chunk)
    print(f"  Loaded {len(chunks) * chunk_size:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal rows loaded: {len(df):,}")

# =============================================================================
# STEP 2: Filter to Target Population
# =============================================================================
print("\n[Step 2] Filtering to target population...")

# Filter to Hispanic-Mexican ethnicity
# HISPAN=1 (Mexican) or HISPAND in 100-107 (detailed Mexican codes)
df_hisp = df[df['HISPAN'] == 1].copy()
print(f"  After Hispanic-Mexican filter: {len(df_hisp):,}")

# Filter to born in Mexico (BPL=200)
df_mex = df_hisp[df_hisp['BPL'] == 200].copy()
print(f"  After Mexico birthplace filter: {len(df_mex):,}")

# Filter to non-citizens (CITIZEN=3)
df_noncit = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"  After non-citizen filter: {len(df_noncit):,}")

# Exclude 2012 (DACA implementation year - timing unclear)
df_sample = df_noncit[df_noncit['YEAR'] != 2012].copy()
print(f"  After excluding 2012: {len(df_sample):,}")

# =============================================================================
# STEP 3: Define DACA Eligibility (Treatment Variable)
# =============================================================================
print("\n[Step 3] Constructing DACA eligibility variable...")

"""
DACA Eligibility Requirements:
1. Arrived in US before 16th birthday
2. Under 31 as of June 15, 2012 (born on or after June 15, 1981)
3. Present in US continuously since June 15, 2007
4. Not a citizen (already filtered)

For operationalization:
- Age at immigration = YRIMMIG - BIRTHYR (must be < 16)
- Born 1981 or later (using June 15 cutoff with BIRTHQTR)
- YRIMMIG <= 2007 (continuous presence)
"""

# Calculate age at immigration
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Criterion 1: Arrived before age 16
df_sample['arrived_before_16'] = (df_sample['age_at_immig'] < 16) & (df_sample['age_at_immig'] >= 0)

# Criterion 2: Under 31 as of June 15, 2012
# Born June 15, 1981 or later
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# June 15 falls in Q2
# If born in 1981 Q1 or Q2 before June 15: could be 31 by June 15, 2012
# Conservative approach: born 1982 or later, OR born 1981 Q3-Q4
df_sample['under_31_june2012'] = (
    (df_sample['BIRTHYR'] >= 1982) |
    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))
)

# Criterion 3: In US since at least June 2007
# YRIMMIG <= 2007 suggests arrival by that year
df_sample['in_us_since_2007'] = df_sample['YRIMMIG'] <= 2007

# Combine criteria for DACA eligibility
df_sample['daca_eligible'] = (
    df_sample['arrived_before_16'] &
    df_sample['under_31_june2012'] &
    df_sample['in_us_since_2007']
).astype(int)

# Create ineligible control group (non-citizens who don't meet DACA criteria)
# Focus on those who are ineligible due to age (too old) - cleaner comparison
df_sample['too_old_for_daca'] = (
    (df_sample['BIRTHYR'] <= 1980) &  # Definitely over 31 by June 2012
    df_sample['arrived_before_16'] &   # Still arrived young
    df_sample['in_us_since_2007']      # Still long-term resident
).astype(int)

print(f"  DACA eligible: {df_sample['daca_eligible'].sum():,}")
print(f"  Control (too old): {df_sample['too_old_for_daca'].sum():,}")

# =============================================================================
# STEP 4: Define Outcome Variable
# =============================================================================
print("\n[Step 4] Constructing outcome variable...")

# Full-time employment: usually works 35+ hours per week
# UHRSWORK: usual hours worked per week (0 = N/A or not working)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

print(f"  Full-time employment rate: {df_sample['fulltime'].mean():.1%}")
print(f"  Employment rate: {df_sample['employed'].mean():.1%}")

# =============================================================================
# STEP 5: Define Time Periods
# =============================================================================
print("\n[Step 5] Defining time periods...")

# Post-DACA indicator (2013-2016)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

print(f"  Pre-DACA (2006-2011) observations: {(df_sample['post']==0).sum():,}")
print(f"  Post-DACA (2013-2016) observations: {(df_sample['post']==1).sum():,}")

# =============================================================================
# STEP 6: Restrict to Analysis Sample
# =============================================================================
print("\n[Step 6] Creating analysis sample...")

# Use DACA eligible + too-old control group for primary analysis
df_analysis = df_sample[
    (df_sample['daca_eligible'] == 1) | (df_sample['too_old_for_daca'] == 1)
].copy()

# Restrict to working-age population (18-64)
df_analysis = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 64)]

print(f"  Analysis sample size: {len(df_analysis):,}")
print(f"  Treatment group (DACA eligible): {df_analysis['daca_eligible'].sum():,}")
print(f"  Control group (too old): {df_analysis['too_old_for_daca'].sum():,}")

# =============================================================================
# STEP 7: Create Control Variables
# =============================================================================
print("\n[Step 7] Creating control variables...")

# Age squared
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Female indicator
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUCD'] >= 62).astype(int)  # HS or more
df_analysis['educ_college'] = (df_analysis['EDUCD'] >= 101).astype(int)  # BA or more

# Married indicator
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)

# DID interaction
df_analysis['daca_x_post'] = df_analysis['daca_eligible'] * df_analysis['post']

print("  Control variables created: age, age_sq, female, educ_hs, educ_college, married")

# =============================================================================
# STEP 8: Descriptive Statistics
# =============================================================================
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

# Summary by treatment and time period
print("\n[Full-time employment rates by group and period]")
summary = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
print(summary)

# Weighted means
print("\n[Weighted full-time employment rates]")
for eligible in [0, 1]:
    for post in [0, 1]:
        mask = (df_analysis['daca_eligible'] == eligible) & (df_analysis['post'] == post)
        subset = df_analysis[mask]
        wt_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        elig_label = "Eligible" if eligible else "Control"
        time_label = "Post" if post else "Pre"
        print(f"  {elig_label}, {time_label}: {wt_mean:.4f}")

# =============================================================================
# STEP 9: Difference-in-Differences Estimation
# =============================================================================
print("\n" + "=" * 80)
print("DIFFERENCE-IN-DIFFERENCES ESTIMATION")
print("=" * 80)

# Model 1: Basic DID (no controls)
print("\n[Model 1: Basic DID - No Controls]")
model1 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post', data=df_analysis).fit()
print(model1.summary().tables[1])

# Model 2: DID with demographic controls
print("\n[Model 2: DID with Demographic Controls]")
model2 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married',
                 data=df_analysis).fit()
print(model2.summary().tables[1])

# Model 3: DID with demographics + education
print("\n[Model 3: DID with Demographics + Education]")
model3 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_college',
                 data=df_analysis).fit()
print(model3.summary().tables[1])

# Model 4: DID with state fixed effects
print("\n[Model 4: DID with State Fixed Effects]")
df_analysis['state'] = df_analysis['STATEFIP'].astype('category')
model4 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_college + C(state)',
                 data=df_analysis).fit()
# Print only key coefficients
print("  DID coefficient (daca_x_post):", f"{model4.params['daca_x_post']:.4f}")
print("  Std Error:", f"{model4.bse['daca_x_post']:.4f}")
print("  t-stat:", f"{model4.tvalues['daca_x_post']:.4f}")
print("  p-value:", f"{model4.pvalues['daca_x_post']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['daca_x_post', 0]:.4f}, {model4.conf_int().loc['daca_x_post', 1]:.4f}]")

# Model 5: DID with year fixed effects
print("\n[Model 5: DID with Year and State Fixed Effects]")
model5 = smf.ols('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_college + C(state) + C(YEAR)',
                 data=df_analysis).fit()
print("  DID coefficient (daca_x_post):", f"{model5.params['daca_x_post']:.4f}")
print("  Std Error:", f"{model5.bse['daca_x_post']:.4f}")
print("  t-stat:", f"{model5.tvalues['daca_x_post']:.4f}")
print("  p-value:", f"{model5.pvalues['daca_x_post']:.4f}")
print(f"  95% CI: [{model5.conf_int().loc['daca_x_post', 0]:.4f}, {model5.conf_int().loc['daca_x_post', 1]:.4f}]")

# =============================================================================
# STEP 10: Weighted Estimation (Preferred Specification)
# =============================================================================
print("\n" + "=" * 80)
print("WEIGHTED ESTIMATION (PREFERRED SPECIFICATION)")
print("=" * 80)

# Model 6: Weighted DID with full controls
print("\n[Model 6: Weighted DID - Full Controls with Year and State FE]")
model6 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_college + C(state) + C(YEAR)',
                 data=df_analysis, weights=df_analysis['PERWT']).fit()
print("  DID coefficient (daca_x_post):", f"{model6.params['daca_x_post']:.4f}")
print("  Std Error:", f"{model6.bse['daca_x_post']:.4f}")
print("  t-stat:", f"{model6.tvalues['daca_x_post']:.4f}")
print("  p-value:", f"{model6.pvalues['daca_x_post']:.4f}")
print(f"  95% CI: [{model6.conf_int().loc['daca_x_post', 0]:.4f}, {model6.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"  N: {int(model6.nobs):,}")
print(f"  R-squared: {model6.rsquared:.4f}")

# =============================================================================
# STEP 11: Robustness Checks
# =============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# Robustness 1: Alternative control group - recently arrived
print("\n[Robustness 1: Alternative Control - Recent Arrivals]")
df_sample['recent_arrival'] = (
    (df_sample['YRIMMIG'] > 2007) &  # Arrived after 2007
    (df_sample['BPL'] == 200) &       # Born in Mexico
    (df_sample['CITIZEN'] == 3)       # Non-citizen
).astype(int)

df_robust1 = df_sample[
    (df_sample['daca_eligible'] == 1) | (df_sample['recent_arrival'] == 1)
].copy()
df_robust1 = df_robust1[(df_robust1['AGE'] >= 18) & (df_robust1['AGE'] <= 64)]
df_robust1['age_sq'] = df_robust1['AGE'] ** 2
df_robust1['female'] = (df_robust1['SEX'] == 2).astype(int)
df_robust1['married'] = (df_robust1['MARST'] <= 2).astype(int)
df_robust1['educ_hs'] = (df_robust1['EDUCD'] >= 62).astype(int)
df_robust1['educ_college'] = (df_robust1['EDUCD'] >= 101).astype(int)
df_robust1['daca_x_post'] = df_robust1['daca_eligible'] * df_robust1['post']
df_robust1['state'] = df_robust1['STATEFIP'].astype('category')

if len(df_robust1) > 100:
    model_r1 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_college + C(state) + C(YEAR)',
                       data=df_robust1, weights=df_robust1['PERWT']).fit()
    print("  DID coefficient:", f"{model_r1.params['daca_x_post']:.4f}")
    print("  Std Error:", f"{model_r1.bse['daca_x_post']:.4f}")
    print(f"  95% CI: [{model_r1.conf_int().loc['daca_x_post', 0]:.4f}, {model_r1.conf_int().loc['daca_x_post', 1]:.4f}]")
    print(f"  N: {int(model_r1.nobs):,}")

# Robustness 2: Men only
print("\n[Robustness 2: Men Only]")
df_men = df_analysis[df_analysis['female'] == 0].copy()
model_men = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + married + educ_hs + educ_college + C(state) + C(YEAR)',
                    data=df_men, weights=df_men['PERWT']).fit()
print("  DID coefficient:", f"{model_men.params['daca_x_post']:.4f}")
print("  Std Error:", f"{model_men.bse['daca_x_post']:.4f}")
print(f"  95% CI: [{model_men.conf_int().loc['daca_x_post', 0]:.4f}, {model_men.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"  N: {int(model_men.nobs):,}")

# Robustness 3: Women only
print("\n[Robustness 3: Women Only]")
df_women = df_analysis[df_analysis['female'] == 1].copy()
model_women = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + married + educ_hs + educ_college + C(state) + C(YEAR)',
                      data=df_women, weights=df_women['PERWT']).fit()
print("  DID coefficient:", f"{model_women.params['daca_x_post']:.4f}")
print("  Std Error:", f"{model_women.bse['daca_x_post']:.4f}")
print(f"  95% CI: [{model_women.conf_int().loc['daca_x_post', 0]:.4f}, {model_women.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"  N: {int(model_women.nobs):,}")

# Robustness 4: Alternative outcome - any employment
print("\n[Robustness 4: Any Employment (not just full-time)]")
model_emp = smf.wls('employed ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_college + C(state) + C(YEAR)',
                    data=df_analysis, weights=df_analysis['PERWT']).fit()
print("  DID coefficient:", f"{model_emp.params['daca_x_post']:.4f}")
print("  Std Error:", f"{model_emp.bse['daca_x_post']:.4f}")
print(f"  95% CI: [{model_emp.conf_int().loc['daca_x_post', 0]:.4f}, {model_emp.conf_int().loc['daca_x_post', 1]:.4f}]")

# =============================================================================
# STEP 12: Event Study / Parallel Trends Check
# =============================================================================
print("\n" + "=" * 80)
print("EVENT STUDY - PARALLEL TRENDS CHECK")
print("=" * 80)

# Create year dummies interacted with treatment
df_analysis['year_dummy'] = df_analysis['YEAR']
for year in df_analysis['YEAR'].unique():
    df_analysis[f'daca_x_{year}'] = (df_analysis['daca_eligible'] * (df_analysis['YEAR'] == year)).astype(int)

# Reference year: 2011 (last pre-treatment year)
year_vars = [f'daca_x_{y}' for y in sorted(df_analysis['YEAR'].unique()) if y != 2011]
formula = 'fulltime ~ daca_eligible + ' + ' + '.join(year_vars) + ' + AGE + age_sq + female + married + educ_hs + educ_college + C(state) + C(YEAR)'

model_es = smf.wls(formula, data=df_analysis, weights=df_analysis['PERWT']).fit()

print("\n[Event Study Coefficients - Reference Year: 2011]")
print("  Year | Coefficient |   Std Err  |    95% CI")
print("  -----|-------------|------------|------------------")
for year in sorted(df_analysis['YEAR'].unique()):
    if year == 2011:
        print(f"  {year} |    0.0000   |     --     | [Reference Year]")
    else:
        var = f'daca_x_{year}'
        coef = model_es.params[var]
        se = model_es.bse[var]
        ci_low = model_es.conf_int().loc[var, 0]
        ci_high = model_es.conf_int().loc[var, 1]
        print(f"  {year} |   {coef:7.4f}   |   {se:.4f}  | [{ci_low:.4f}, {ci_high:.4f}]")

# =============================================================================
# STEP 13: Save Results for Report
# =============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Create results dictionary
results = {
    'preferred_estimate': {
        'coefficient': model6.params['daca_x_post'],
        'std_error': model6.bse['daca_x_post'],
        'ci_low': model6.conf_int().loc['daca_x_post', 0],
        'ci_high': model6.conf_int().loc['daca_x_post', 1],
        'pvalue': model6.pvalues['daca_x_post'],
        'n_obs': int(model6.nobs),
        'r_squared': model6.rsquared
    },
    'sample_sizes': {
        'total': len(df_analysis),
        'treatment': df_analysis['daca_eligible'].sum(),
        'control': len(df_analysis) - df_analysis['daca_eligible'].sum()
    }
}

# Save summary statistics
summary_stats = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'educ_hs': 'mean',
    'educ_college': 'mean',
    'married': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)
summary_stats.to_csv('summary_statistics.csv')
print("  Saved: summary_statistics.csv")

# Save regression coefficients for all models
coef_table = pd.DataFrame({
    'Model 1 (Basic)': [model1.params['daca_x_post'], model1.bse['daca_x_post'], model1.nobs, model1.rsquared],
    'Model 2 (Demog)': [model2.params['daca_x_post'], model2.bse['daca_x_post'], model2.nobs, model2.rsquared],
    'Model 3 (Educ)': [model3.params['daca_x_post'], model3.bse['daca_x_post'], model3.nobs, model3.rsquared],
    'Model 4 (State FE)': [model4.params['daca_x_post'], model4.bse['daca_x_post'], model4.nobs, model4.rsquared],
    'Model 5 (Year FE)': [model5.params['daca_x_post'], model5.bse['daca_x_post'], model5.nobs, model5.rsquared],
    'Model 6 (Weighted)': [model6.params['daca_x_post'], model6.bse['daca_x_post'], model6.nobs, model6.rsquared],
}, index=['DID Coefficient', 'Std Error', 'N', 'R-squared'])
coef_table.to_csv('regression_results.csv')
print("  Saved: regression_results.csv")

# Save event study coefficients
es_coefs = []
for year in sorted(df_analysis['YEAR'].unique()):
    if year == 2011:
        es_coefs.append({'Year': year, 'Coefficient': 0, 'StdError': np.nan, 'CI_Low': np.nan, 'CI_High': np.nan})
    else:
        var = f'daca_x_{year}'
        es_coefs.append({
            'Year': year,
            'Coefficient': model_es.params[var],
            'StdError': model_es.bse[var],
            'CI_Low': model_es.conf_int().loc[var, 0],
            'CI_High': model_es.conf_int().loc[var, 1]
        })
es_df = pd.DataFrame(es_coefs)
es_df.to_csv('event_study_results.csv', index=False)
print("  Saved: event_study_results.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Print final preferred estimate summary
print("\n*** PREFERRED ESTIMATE SUMMARY ***")
print(f"Effect of DACA eligibility on full-time employment: {results['preferred_estimate']['coefficient']:.4f}")
print(f"Standard Error: {results['preferred_estimate']['std_error']:.4f}")
print(f"95% Confidence Interval: [{results['preferred_estimate']['ci_low']:.4f}, {results['preferred_estimate']['ci_high']:.4f}]")
print(f"p-value: {results['preferred_estimate']['pvalue']:.4f}")
print(f"Sample Size: {results['preferred_estimate']['n_obs']:,}")
print(f"R-squared: {results['preferred_estimate']['r_squared']:.4f}")

# Interpretation
effect_pct = results['preferred_estimate']['coefficient'] * 100
print(f"\nInterpretation: DACA eligibility is associated with a {effect_pct:.1f} percentage point")
if effect_pct > 0:
    print("increase in the probability of full-time employment (>=35 hours/week).")
else:
    print("decrease in the probability of full-time employment (>=35 hours/week).")
