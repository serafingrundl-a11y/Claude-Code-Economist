"""
DACA Replication Study: Effect of DACA Eligibility on Full-Time Employment
Among Hispanic-Mexican, Mexico-Born Population in the United States

This script performs a difference-in-differences analysis examining the causal
impact of DACA eligibility on full-time employment (35+ hours/week).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING AND INITIAL FILTERING
# =============================================================================

print("=" * 80)
print("DACA REPLICATION STUDY: DATA PREPARATION")
print("=" * 80)

# Load the filtered sample (Hispanic-Mexican, Mexico-born)
df = pd.read_csv('data/filtered_sample.csv')
print(f"\nLoaded filtered sample: {len(df):,} observations")
print(f"Years covered: {df['YEAR'].min()} - {df['YEAR'].max()}")

# =============================================================================
# 2. DEFINE DACA ELIGIBILITY CRITERIA
# =============================================================================

print("\n" + "=" * 80)
print("DEFINING DACA ELIGIBILITY CRITERIA")
print("=" * 80)

"""
DACA Eligibility Requirements (as of June 15, 2012):
1. Arrived in the US before their 16th birthday
2. Not yet 31 years old as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007
4. Present in the US on June 15, 2012
5. Not a citizen (non-citizen status)

For this analysis:
- Treatment group: DACA-eligible non-citizens
- Control group: Non-eligible non-citizens (similar characteristics but don't meet age/arrival criteria)
- Pre-period: 2006-2011 (before DACA)
- Post-period: 2013-2016 (after DACA implementation)
- We exclude 2012 since DACA was implemented mid-year (June 15, 2012)
"""

# Restrict to non-citizens only (CITIZEN == 3 means "Not a citizen")
df_noncit = df[df['CITIZEN'] == 3].copy()
print(f"\nNon-citizens only: {len(df_noncit):,} observations")

# Exclude 2012 (implementation year - cannot distinguish pre/post)
df_analysis = df_noncit[df_noncit['YEAR'] != 2012].copy()
print(f"Excluding 2012: {len(df_analysis):,} observations")

# Create POST indicator (1 for 2013-2016, 0 for 2006-2011)
df_analysis['POST'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Calculate age at June 15, 2012 for eligibility determination
# Using BIRTHYR and BIRTHQTR to estimate
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

def calc_age_at_daca(row):
    """Calculate age as of June 15, 2012"""
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']

    # Estimate if birthday passed by June 15, 2012
    # Q1 (Jan-Mar) and Q2 (Apr-Jun early) would have had birthday
    if birth_qtr in [1, 2]:
        return 2012 - birth_year
    else:
        return 2012 - birth_year - 1

df_analysis['AGE_AT_DACA'] = df_analysis.apply(calc_age_at_daca, axis=1)

# Calculate arrival age (age when immigrated)
# YRIMMIG is the year of immigration
df_analysis['ARRIVAL_AGE'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']

# Create DACA eligibility indicator (at the time of DACA implementation - June 2012)
# Criteria:
# 1. Age at DACA < 31 (born after June 1981, so age < 31 as of June 2012)
# 2. Arrived before age 16
# 3. In US since at least 2007 (YRIMMIG <= 2007)
# 4. Age at DACA >= some minimum (we'll use 15 to focus on working-age population)

def is_daca_eligible(row):
    """Determine DACA eligibility based on criteria"""
    age_at_daca = row['AGE_AT_DACA']
    arrival_age = row['ARRIVAL_AGE']
    yrimmig = row['YRIMMIG']

    # Age under 31 as of June 15, 2012
    age_under_31 = age_at_daca < 31

    # Arrived before 16th birthday
    arrived_before_16 = arrival_age < 16

    # In US since 2007 (lived continuously)
    in_us_since_2007 = yrimmig <= 2007

    return int(age_under_31 and arrived_before_16 and in_us_since_2007)

df_analysis['DACA_ELIGIBLE'] = df_analysis.apply(is_daca_eligible, axis=1)

print(f"\nDACA eligibility distribution:")
print(df_analysis['DACA_ELIGIBLE'].value_counts())
print(f"\nEligibility rate: {df_analysis['DACA_ELIGIBLE'].mean():.1%}")

# =============================================================================
# 3. DEFINE OUTCOME VARIABLE: FULL-TIME EMPLOYMENT
# =============================================================================

print("\n" + "=" * 80)
print("DEFINING OUTCOME VARIABLE")
print("=" * 80)

# Full-time employment: Usually works 35+ hours per week AND currently employed
# EMPSTAT: 1 = Employed
# UHRSWORK: Usual hours worked per week (35+ = full-time)

df_analysis['FULLTIME_EMPLOYED'] = (
    (df_analysis['EMPSTAT'] == 1) &
    (df_analysis['UHRSWORK'] >= 35)
).astype(int)

print(f"\nFull-time employment rate: {df_analysis['FULLTIME_EMPLOYED'].mean():.1%}")

# =============================================================================
# 4. RESTRICT TO WORKING-AGE POPULATION
# =============================================================================

print("\n" + "=" * 80)
print("RESTRICTING TO WORKING-AGE POPULATION")
print("=" * 80)

# Focus on working-age population (18-64)
# We use the AGE variable which is age at time of survey
df_working_age = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 64)].copy()
print(f"\nWorking-age (18-64): {len(df_working_age):,} observations")

# Further restrict to those who could potentially be DACA-eligible based on age
# and create a comparison group of older non-citizens
# Treatment: DACA-eligible (met age/arrival criteria)
# Control: Non-eligible due to age > 30 at DACA but similar otherwise

print(f"\nFinal sample: {len(df_working_age):,} observations")
print(f"DACA eligible: {df_working_age['DACA_ELIGIBLE'].sum():,}")
print(f"Not eligible: {(~df_working_age['DACA_ELIGIBLE'].astype(bool)).sum():,}")

# =============================================================================
# 5. DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

# By eligibility status
print("\n--- Summary by DACA Eligibility Status ---")
for eligible in [0, 1]:
    subset = df_working_age[df_working_age['DACA_ELIGIBLE'] == eligible]
    label = "DACA Eligible" if eligible == 1 else "Not Eligible"
    print(f"\n{label} (N = {len(subset):,}):")
    print(f"  Mean age: {subset['AGE'].mean():.1f}")
    print(f"  Mean years in US: {(subset['YEAR'] - subset['YRIMMIG']).mean():.1f}")
    print(f"  Female share: {(subset['SEX'] == 2).mean():.1%}")
    print(f"  Full-time employment rate: {subset['FULLTIME_EMPLOYED'].mean():.1%}")

# By period
print("\n--- Summary by Period ---")
for post in [0, 1]:
    subset = df_working_age[df_working_age['POST'] == post]
    label = "Post-DACA (2013-2016)" if post == 1 else "Pre-DACA (2006-2011)"
    print(f"\n{label} (N = {len(subset):,}):")
    print(f"  Full-time employment rate: {subset['FULLTIME_EMPLOYED'].mean():.1%}")

# 2x2 table
print("\n--- 2x2 Table: Full-Time Employment Rates ---")
table = df_working_age.groupby(['DACA_ELIGIBLE', 'POST'])['FULLTIME_EMPLOYED'].mean().unstack()
table.index = ['Not Eligible', 'DACA Eligible']
table.columns = ['Pre-DACA', 'Post-DACA']
print(table.round(4))

# Calculate simple DiD
pre_elig = df_working_age[(df_working_age['DACA_ELIGIBLE'] == 1) & (df_working_age['POST'] == 0)]['FULLTIME_EMPLOYED'].mean()
post_elig = df_working_age[(df_working_age['DACA_ELIGIBLE'] == 1) & (df_working_age['POST'] == 1)]['FULLTIME_EMPLOYED'].mean()
pre_inelig = df_working_age[(df_working_age['DACA_ELIGIBLE'] == 0) & (df_working_age['POST'] == 0)]['FULLTIME_EMPLOYED'].mean()
post_inelig = df_working_age[(df_working_age['DACA_ELIGIBLE'] == 0) & (df_working_age['POST'] == 1)]['FULLTIME_EMPLOYED'].mean()

did_simple = (post_elig - pre_elig) - (post_inelig - pre_inelig)
print(f"\nSimple DiD estimate: {did_simple:.4f} ({did_simple*100:.2f} percentage points)")

# =============================================================================
# 6. MAIN REGRESSION ANALYSIS: DIFFERENCE-IN-DIFFERENCES
# =============================================================================

print("\n" + "=" * 80)
print("MAIN REGRESSION ANALYSIS")
print("=" * 80)

# Create interaction term
df_working_age['DACA_x_POST'] = df_working_age['DACA_ELIGIBLE'] * df_working_age['POST']

# Control variables
df_working_age['FEMALE'] = (df_working_age['SEX'] == 2).astype(int)
df_working_age['MARRIED'] = (df_working_age['MARST'].isin([1, 2])).astype(int)
df_working_age['AGE_SQ'] = df_working_age['AGE'] ** 2

# Education categories
df_working_age['HS_OR_LESS'] = (df_working_age['EDUCD'] <= 62).astype(int)  # HS diploma or less
df_working_age['SOME_COLLEGE'] = ((df_working_age['EDUCD'] > 62) & (df_working_age['EDUCD'] < 101)).astype(int)

# Years in US
df_working_age['YEARS_IN_US'] = df_working_age['YEAR'] - df_working_age['YRIMMIG']
df_working_age['YEARS_IN_US'] = df_working_age['YEARS_IN_US'].clip(lower=0)

# Model 1: Basic DiD
print("\n--- Model 1: Basic Difference-in-Differences ---")
model1 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + POST + DACA_x_POST',
                 data=df_working_age).fit(cov_type='HC1')
print(model1.summary().tables[1])
print(f"\nDiD Coefficient: {model1.params['DACA_x_POST']:.4f}")
print(f"Standard Error: {model1.bse['DACA_x_POST']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['DACA_x_POST', 0]:.4f}, {model1.conf_int().loc['DACA_x_POST', 1]:.4f}]")

# Model 2: With demographic controls
print("\n--- Model 2: With Demographic Controls ---")
model2 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + POST + DACA_x_POST + AGE + AGE_SQ + FEMALE + MARRIED',
                 data=df_working_age).fit(cov_type='HC1')
print(model2.summary().tables[1])
print(f"\nDiD Coefficient: {model2.params['DACA_x_POST']:.4f}")
print(f"Standard Error: {model2.bse['DACA_x_POST']:.4f}")

# Model 3: With education and years in US
print("\n--- Model 3: With Education and Years in US ---")
model3 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + POST + DACA_x_POST + AGE + AGE_SQ + FEMALE + MARRIED + HS_OR_LESS + SOME_COLLEGE + YEARS_IN_US',
                 data=df_working_age).fit(cov_type='HC1')
print(model3.summary().tables[1])
print(f"\nDiD Coefficient: {model3.params['DACA_x_POST']:.4f}")
print(f"Standard Error: {model3.bse['DACA_x_POST']:.4f}")

# Model 4: With year fixed effects
print("\n--- Model 4: With Year Fixed Effects ---")
df_working_age['YEAR_FE'] = df_working_age['YEAR'].astype(str)
model4 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_x_POST + AGE + AGE_SQ + FEMALE + MARRIED + HS_OR_LESS + SOME_COLLEGE + YEARS_IN_US + C(YEAR_FE)',
                 data=df_working_age).fit(cov_type='HC1')
print(f"\nDiD Coefficient: {model4.params['DACA_x_POST']:.4f}")
print(f"Standard Error: {model4.bse['DACA_x_POST']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['DACA_x_POST', 0]:.4f}, {model4.conf_int().loc['DACA_x_POST', 1]:.4f}]")

# Model 5: With state fixed effects
print("\n--- Model 5: With Year and State Fixed Effects ---")
df_working_age['STATE_FE'] = df_working_age['STATEFIP'].astype(str)
model5 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_x_POST + AGE + AGE_SQ + FEMALE + MARRIED + HS_OR_LESS + SOME_COLLEGE + YEARS_IN_US + C(YEAR_FE) + C(STATE_FE)',
                 data=df_working_age).fit(cov_type='HC1')
print(f"\nDiD Coefficient: {model5.params['DACA_x_POST']:.4f}")
print(f"Standard Error: {model5.bse['DACA_x_POST']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['DACA_x_POST', 0]:.4f}, {model5.conf_int().loc['DACA_x_POST', 1]:.4f}]")

# =============================================================================
# 7. ROBUSTNESS CHECKS
# =============================================================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# Robustness 1: Alternative age restriction (25-55)
print("\n--- Robustness 1: Restricted Age Range (25-55) ---")
df_robust1 = df_working_age[(df_working_age['AGE'] >= 25) & (df_working_age['AGE'] <= 55)]
model_r1 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_x_POST + AGE + AGE_SQ + FEMALE + MARRIED + C(YEAR_FE)',
                   data=df_robust1).fit(cov_type='HC1')
print(f"Sample size: {len(df_robust1):,}")
print(f"DiD Coefficient: {model_r1.params['DACA_x_POST']:.4f}")
print(f"Standard Error: {model_r1.bse['DACA_x_POST']:.4f}")

# Robustness 2: Men only
print("\n--- Robustness 2: Men Only ---")
df_men = df_working_age[df_working_age['FEMALE'] == 0]
model_men = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_x_POST + AGE + AGE_SQ + MARRIED + C(YEAR_FE)',
                    data=df_men).fit(cov_type='HC1')
print(f"Sample size: {len(df_men):,}")
print(f"DiD Coefficient: {model_men.params['DACA_x_POST']:.4f}")
print(f"Standard Error: {model_men.bse['DACA_x_POST']:.4f}")

# Robustness 3: Women only
print("\n--- Robustness 3: Women Only ---")
df_women = df_working_age[df_working_age['FEMALE'] == 1]
model_women = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_x_POST + AGE + AGE_SQ + MARRIED + C(YEAR_FE)',
                      data=df_women).fit(cov_type='HC1')
print(f"Sample size: {len(df_women):,}")
print(f"DiD Coefficient: {model_women.params['DACA_x_POST']:.4f}")
print(f"Standard Error: {model_women.bse['DACA_x_POST']:.4f}")

# Robustness 4: Including 2012 as post period
print("\n--- Robustness 4: Including 2012 as Post Period ---")
df_incl_2012 = df_noncit[(df_noncit['AGE'] >= 18) & (df_noncit['AGE'] <= 64)].copy()
df_incl_2012['POST'] = (df_incl_2012['YEAR'] >= 2012).astype(int)
df_incl_2012['AGE_AT_DACA'] = df_incl_2012.apply(calc_age_at_daca, axis=1)
df_incl_2012['ARRIVAL_AGE'] = df_incl_2012['YRIMMIG'] - df_incl_2012['BIRTHYR']
df_incl_2012['DACA_ELIGIBLE'] = df_incl_2012.apply(is_daca_eligible, axis=1)
df_incl_2012['DACA_x_POST'] = df_incl_2012['DACA_ELIGIBLE'] * df_incl_2012['POST']
df_incl_2012['FULLTIME_EMPLOYED'] = ((df_incl_2012['EMPSTAT'] == 1) & (df_incl_2012['UHRSWORK'] >= 35)).astype(int)
df_incl_2012['AGE_SQ'] = df_incl_2012['AGE'] ** 2
df_incl_2012['FEMALE'] = (df_incl_2012['SEX'] == 2).astype(int)
df_incl_2012['MARRIED'] = (df_incl_2012['MARST'].isin([1, 2])).astype(int)
df_incl_2012['YEAR_FE'] = df_incl_2012['YEAR'].astype(str)
model_r4 = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + DACA_x_POST + AGE + AGE_SQ + FEMALE + MARRIED + C(YEAR_FE)',
                   data=df_incl_2012).fit(cov_type='HC1')
print(f"Sample size: {len(df_incl_2012):,}")
print(f"DiD Coefficient: {model_r4.params['DACA_x_POST']:.4f}")
print(f"Standard Error: {model_r4.bse['DACA_x_POST']:.4f}")

# =============================================================================
# 8. EVENT STUDY / PARALLEL TRENDS
# =============================================================================

print("\n" + "=" * 80)
print("EVENT STUDY ANALYSIS")
print("=" * 80)

# Create year dummies interacted with eligibility
years = sorted(df_working_age['YEAR'].unique())
base_year = 2011  # Reference year (last pre-treatment year)

event_study_results = []
for year in years:
    if year == base_year:
        event_study_results.append({'Year': year, 'Coef': 0, 'SE': 0, 'CI_low': 0, 'CI_high': 0})
        continue

    df_temp = df_working_age[df_working_age['YEAR'].isin([base_year, year])].copy()
    df_temp['YEAR_IND'] = (df_temp['YEAR'] == year).astype(int)
    df_temp['DACA_x_YEAR'] = df_temp['DACA_ELIGIBLE'] * df_temp['YEAR_IND']

    model_es = smf.ols('FULLTIME_EMPLOYED ~ DACA_ELIGIBLE + YEAR_IND + DACA_x_YEAR + AGE + AGE_SQ + FEMALE + MARRIED',
                       data=df_temp).fit(cov_type='HC1')

    event_study_results.append({
        'Year': year,
        'Coef': model_es.params['DACA_x_YEAR'],
        'SE': model_es.bse['DACA_x_YEAR'],
        'CI_low': model_es.conf_int().loc['DACA_x_YEAR', 0],
        'CI_high': model_es.conf_int().loc['DACA_x_YEAR', 1]
    })

event_study_df = pd.DataFrame(event_study_results)
print("\nEvent Study Coefficients (relative to 2011):")
print(event_study_df.to_string(index=False))

# =============================================================================
# 9. WEIGHTED ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("WEIGHTED ANALYSIS")
print("=" * 80)

# Using person weights (PERWT)
import statsmodels.api as sm

# Prepare data for weighted regression
X_weighted = df_working_age[['DACA_ELIGIBLE', 'POST', 'DACA_x_POST', 'AGE', 'AGE_SQ', 'FEMALE', 'MARRIED']].copy()
X_weighted = sm.add_constant(X_weighted)
y_weighted = df_working_age['FULLTIME_EMPLOYED']
weights = df_working_age['PERWT']

model_weighted = sm.WLS(y_weighted, X_weighted, weights=weights).fit(cov_type='HC1')
print(f"\nWeighted DiD Coefficient: {model_weighted.params['DACA_x_POST']:.4f}")
print(f"Standard Error: {model_weighted.bse['DACA_x_POST']:.4f}")
print(f"95% CI: [{model_weighted.conf_int().loc['DACA_x_POST', 0]:.4f}, {model_weighted.conf_int().loc['DACA_x_POST', 1]:.4f}]")

# =============================================================================
# 10. SAVE RESULTS FOR REPORT
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results_summary = {
    'Model': ['Basic DiD', 'With Demographics', 'With Education', 'Year FE', 'Year + State FE', 'Weighted'],
    'Coefficient': [
        model1.params['DACA_x_POST'],
        model2.params['DACA_x_POST'],
        model3.params['DACA_x_POST'],
        model4.params['DACA_x_POST'],
        model5.params['DACA_x_POST'],
        model_weighted.params['DACA_x_POST']
    ],
    'SE': [
        model1.bse['DACA_x_POST'],
        model2.bse['DACA_x_POST'],
        model3.bse['DACA_x_POST'],
        model4.bse['DACA_x_POST'],
        model5.bse['DACA_x_POST'],
        model_weighted.bse['DACA_x_POST']
    ],
    'N': [
        model1.nobs,
        model2.nobs,
        model3.nobs,
        model4.nobs,
        model5.nobs,
        model_weighted.nobs
    ]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results_summary.csv', index=False)
print("Saved results_summary.csv")

# Save event study results
event_study_df.to_csv('event_study_results.csv', index=False)
print("Saved event_study_results.csv")

# Save descriptive statistics
desc_stats = df_working_age.groupby(['DACA_ELIGIBLE', 'POST']).agg({
    'FULLTIME_EMPLOYED': ['mean', 'count'],
    'AGE': 'mean',
    'FEMALE': 'mean',
    'MARRIED': 'mean',
    'PERWT': 'sum'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("Saved descriptive_stats.csv")

# =============================================================================
# 11. FINAL PREFERRED ESTIMATE
# =============================================================================

print("\n" + "=" * 80)
print("PREFERRED ESTIMATE (Model 4: Year Fixed Effects)")
print("=" * 80)

print(f"""
Effect of DACA Eligibility on Full-Time Employment
===================================================
Coefficient: {model4.params['DACA_x_POST']:.4f}
Standard Error: {model4.bse['DACA_x_POST']:.4f}
t-statistic: {model4.tvalues['DACA_x_POST']:.2f}
p-value: {model4.pvalues['DACA_x_POST']:.4f}
95% CI: [{model4.conf_int().loc['DACA_x_POST', 0]:.4f}, {model4.conf_int().loc['DACA_x_POST', 1]:.4f}]
Sample Size: {int(model4.nobs):,}
R-squared: {model4.rsquared:.4f}

Interpretation:
DACA eligibility is associated with a {model4.params['DACA_x_POST']*100:.2f} percentage point
{'increase' if model4.params['DACA_x_POST'] > 0 else 'decrease'} in the probability of full-time employment.
This effect is {'statistically significant' if model4.pvalues['DACA_x_POST'] < 0.05 else 'not statistically significant'} at the 5% level.
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
