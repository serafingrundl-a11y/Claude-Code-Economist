"""
DACA Replication Study - Analysis Script
Replication 40

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the
United States, what was the causal impact of eligibility for DACA on the probability
of full-time employment (35+ hours/week)?

Identification Strategy: Difference-in-Differences comparing DACA-eligible individuals
to similar non-eligible individuals before and after DACA implementation.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n1. LOADING DATA...")

# Define columns to use (to save memory)
usecols = [
    'YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'LABFORCE', 'UHRSWORK'
]

# Load data
df = pd.read_csv('data/data.csv', usecols=usecols, low_memory=False)
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. SAMPLE SELECTION
# =============================================================================
print("\n2. SAMPLE SELECTION...")

# Step 2a: Keep only Hispanic-Mexican individuals
# HISPAN == 1 is Mexican (general version)
df_mex = df[df['HISPAN'] == 1].copy()
print(f"After selecting Hispanic-Mexican (HISPAN=1): {len(df_mex):,}")

# Step 2b: Keep only those born in Mexico
# BPL == 200 is Mexico
df_mex = df_mex[df_mex['BPL'] == 200].copy()
print(f"After selecting born in Mexico (BPL=200): {len(df_mex):,}")

# Step 2c: Keep only non-citizens (proxy for undocumented)
# CITIZEN == 3 is "Not a citizen"
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"After selecting non-citizens (CITIZEN=3): {len(df_mex):,}")

# Step 2d: Exclude 2012 due to mid-year DACA implementation
df_mex = df_mex[df_mex['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_mex):,}")

# Step 2e: Working age population (16-64)
df_mex = df_mex[(df_mex['AGE'] >= 16) & (df_mex['AGE'] <= 64)].copy()
print(f"After selecting working age (16-64): {len(df_mex):,}")

# =============================================================================
# 3. VARIABLE CONSTRUCTION
# =============================================================================
print("\n3. CONSTRUCTING VARIABLES...")

# Post-DACA indicator
df_mex['post'] = (df_mex['YEAR'] >= 2013).astype(int)
print(f"Post-DACA periods (2013-2016): {df_mex['post'].sum():,} observations")

# Calculate age on June 15, 2012
# Using mid-year approximation: If BIRTHQTR <= 2, they had birthday by June 15
# Since BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# For simplicity, calculate age in 2012 based on birth year
df_mex['age_2012'] = 2012 - df_mex['BIRTHYR']

# Age at immigration
df_mex['age_at_immig'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# =============================================================================
# 4. DACA ELIGIBILITY DETERMINATION
# =============================================================================
print("\n4. DETERMINING DACA ELIGIBILITY...")

"""
DACA Eligibility Criteria:
1. Arrived before age 16: age_at_immig < 16
2. Age < 31 on June 15, 2012: age_2012 < 31 (born after June 15, 1981)
3. In US since June 15, 2007: YRIMMIG <= 2007
4. Present in US on June 15, 2012: YRIMMIG <= 2012 (implicitly satisfied by criterion 3)

For criterion 2, being more precise:
- If born in Q3 or Q4 of 1981, they would be under 31 on June 15, 2012
- If born in Q1 or Q2 of 1981, they would have turned 31 by June 15, 2012
- For simplicity, use BIRTHYR > 1981 as the cutoff (conservative)
"""

# Valid immigration year (not 0, which means N/A or born in US)
df_mex['valid_yrimmig'] = (df_mex['YRIMMIG'] > 0) & (df_mex['YRIMMIG'] <= 2012)

# Criterion 1: Arrived before age 16
df_mex['arrived_before_16'] = df_mex['age_at_immig'] < 16

# Criterion 2: Under 31 on June 15, 2012 (born after 1981)
df_mex['under_31_2012'] = df_mex['BIRTHYR'] > 1981

# Criterion 3: In US since 2007
df_mex['in_us_since_2007'] = df_mex['YRIMMIG'] <= 2007

# DACA eligible (meet all criteria)
df_mex['daca_eligible'] = (
    df_mex['valid_yrimmig'] &
    df_mex['arrived_before_16'] &
    df_mex['under_31_2012'] &
    df_mex['in_us_since_2007']
).astype(int)

print(f"DACA eligible observations: {df_mex['daca_eligible'].sum():,}")
print(f"Not DACA eligible: {(df_mex['daca_eligible'] == 0).sum():,}")

# =============================================================================
# 5. OUTCOME VARIABLE
# =============================================================================
print("\n5. DEFINING OUTCOME VARIABLE...")

# Full-time employment: UHRSWORK >= 35
# UHRSWORK = 0 means N/A (not in labor force or not working)
# We define outcome as 1 if working 35+ hours, 0 otherwise (including not working)

df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

print(f"Full-time employed: {df_mex['fulltime'].sum():,}")
print(f"Not full-time: {(df_mex['fulltime'] == 0).sum():,}")

# Also create employed variable (EMPSTAT == 1)
df_mex['employed'] = (df_mex['EMPSTAT'] == 1).astype(int)

# =============================================================================
# 6. CONTROL VARIABLES
# =============================================================================
print("\n6. CREATING CONTROL VARIABLES...")

# Female indicator
df_mex['female'] = (df_mex['SEX'] == 2).astype(int)

# Married indicator (MARST 1 or 2 = married)
df_mex['married'] = df_mex['MARST'].isin([1, 2]).astype(int)

# Education categories
# EDUC codes: 0=N/A, 1=No school, 2=Some school, 3-5=Grades, 6=High school,
# 7=Some college, 8=College degree, 9-11=Higher
df_mex['educ_hs'] = (df_mex['EDUC'] >= 6).astype(int)  # At least HS
df_mex['educ_college'] = (df_mex['EDUC'] >= 8).astype(int)  # College degree+

# Age squared
df_mex['age_sq'] = df_mex['AGE'] ** 2

# State fixed effects (will use in regression)

# =============================================================================
# 7. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n7. DESCRIPTIVE STATISTICS...")

# Create analysis sample: those with valid YRIMMIG
analysis_df = df_mex[df_mex['valid_yrimmig']].copy()
print(f"\nAnalysis sample size: {len(analysis_df):,}")

# Descriptive stats by treatment status
print("\n--- Summary Statistics by DACA Eligibility ---")
vars_desc = ['AGE', 'female', 'married', 'educ_hs', 'educ_college', 'employed', 'fulltime', 'UHRSWORK']

print("\nDACA Eligible:")
print(analysis_df[analysis_df['daca_eligible'] == 1][vars_desc].describe())

print("\nNot DACA Eligible:")
print(analysis_df[analysis_df['daca_eligible'] == 0][vars_desc].describe())

# Weighted means
def weighted_mean(data, var, weight='PERWT'):
    return np.average(data[var], weights=data[weight])

print("\n--- Weighted Means by Group and Period ---")
for eligible in [0, 1]:
    for post in [0, 1]:
        subset = analysis_df[(analysis_df['daca_eligible'] == eligible) & (analysis_df['post'] == post)]
        if len(subset) > 0:
            ft_rate = weighted_mean(subset, 'fulltime')
            emp_rate = weighted_mean(subset, 'employed')
            n = len(subset)
            print(f"Eligible={eligible}, Post={post}: N={n:,}, FT Rate={ft_rate:.4f}, Emp Rate={emp_rate:.4f}")

# =============================================================================
# 8. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("8. DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# Treatment variable: interaction of DACA-eligible and post-DACA
analysis_df['treat'] = analysis_df['daca_eligible'] * analysis_df['post']

# 8.1 Simple 2x2 DiD
print("\n8.1 Simple 2x2 Difference-in-Differences")

# Calculate group means
groups = analysis_df.groupby(['daca_eligible', 'post']).apply(
    lambda x: pd.Series({
        'fulltime': weighted_mean(x, 'fulltime'),
        'n': len(x),
        'sum_weight': x['PERWT'].sum()
    })
)
print("\nGroup Means (Weighted):")
print(groups)

# Calculate DiD estimate
if len(groups) == 4:
    pre_control = groups.loc[(0, 0), 'fulltime']
    post_control = groups.loc[(0, 1), 'fulltime']
    pre_treat = groups.loc[(1, 0), 'fulltime']
    post_treat = groups.loc[(1, 1), 'fulltime']

    did_estimate = (post_treat - pre_treat) - (post_control - pre_control)

    print(f"\nPre-DACA Control: {pre_control:.4f}")
    print(f"Post-DACA Control: {post_control:.4f}")
    print(f"Pre-DACA Treated: {pre_treat:.4f}")
    print(f"Post-DACA Treated: {post_treat:.4f}")
    print(f"\nDiD Estimate: {did_estimate:.4f}")

# 8.2 Regression DiD (without controls)
print("\n8.2 Regression DiD (No Controls)")

model1 = smf.wls('fulltime ~ daca_eligible + post + treat',
                  data=analysis_df, weights=analysis_df['PERWT']).fit(cov_type='HC1')
print(model1.summary())

# 8.3 Regression DiD with controls
print("\n8.3 Regression DiD with Individual Controls")

model2 = smf.wls('fulltime ~ daca_eligible + post + treat + female + married + AGE + age_sq + educ_hs + educ_college',
                  data=analysis_df, weights=analysis_df['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# 8.4 With year fixed effects
print("\n8.4 Regression DiD with Year Fixed Effects")

analysis_df['year_factor'] = pd.Categorical(analysis_df['YEAR'])
model3 = smf.wls('fulltime ~ daca_eligible + treat + female + married + AGE + age_sq + educ_hs + educ_college + C(YEAR)',
                  data=analysis_df, weights=analysis_df['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# 8.5 With state fixed effects
print("\n8.5 Regression DiD with State and Year Fixed Effects")

model4 = smf.wls('fulltime ~ daca_eligible + treat + female + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)',
                  data=analysis_df, weights=analysis_df['PERWT']).fit(cov_type='HC1')
# Only print key results due to many state dummies
print(f"\nDiD Coefficient (treat): {model4.params['treat']:.5f}")
print(f"Standard Error: {model4.bse['treat']:.5f}")
print(f"t-statistic: {model4.tvalues['treat']:.3f}")
print(f"p-value: {model4.pvalues['treat']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat', 0]:.5f}, {model4.conf_int().loc['treat', 1]:.5f}]")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {model4.nobs:,.0f}")

# =============================================================================
# 9. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("9. ROBUSTNESS CHECKS")
print("=" * 80)

# 9.1 Employment (any) as outcome
print("\n9.1 Alternative Outcome: Any Employment")
model_emp = smf.wls('employed ~ daca_eligible + treat + female + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)',
                     data=analysis_df, weights=analysis_df['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient (treat): {model_emp.params['treat']:.5f}")
print(f"Standard Error: {model_emp.bse['treat']:.5f}")
print(f"p-value: {model_emp.pvalues['treat']:.4f}")

# 9.2 Restrict to labor force participants
print("\n9.2 Restrict to Labor Force Participants")
labor_force = analysis_df[analysis_df['LABFORCE'] == 2].copy()
print(f"Labor force sample size: {len(labor_force):,}")

model_lf = smf.wls('fulltime ~ daca_eligible + treat + female + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)',
                    data=labor_force, weights=labor_force['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient (treat): {model_lf.params['treat']:.5f}")
print(f"Standard Error: {model_lf.bse['treat']:.5f}")
print(f"p-value: {model_lf.pvalues['treat']:.4f}")

# 9.3 Different age cutoff (use 1982 instead of 1981)
print("\n9.3 Alternative Age Cutoff (BIRTHYR >= 1982)")
analysis_df['daca_eligible_alt'] = (
    analysis_df['valid_yrimmig'] &
    analysis_df['arrived_before_16'] &
    (analysis_df['BIRTHYR'] >= 1982) &
    analysis_df['in_us_since_2007']
).astype(int)
analysis_df['treat_alt'] = analysis_df['daca_eligible_alt'] * analysis_df['post']

model_alt = smf.wls('fulltime ~ daca_eligible_alt + treat_alt + female + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)',
                     data=analysis_df, weights=analysis_df['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient (treat): {model_alt.params['treat_alt']:.5f}")
print(f"Standard Error: {model_alt.bse['treat_alt']:.5f}")
print(f"p-value: {model_alt.pvalues['treat_alt']:.4f}")

# 9.4 Younger sample (under 40 in survey year)
print("\n9.4 Younger Sample (Age < 40)")
younger = analysis_df[analysis_df['AGE'] < 40].copy()
print(f"Younger sample size: {len(younger):,}")

model_young = smf.wls('fulltime ~ daca_eligible + treat + female + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)',
                       data=younger, weights=younger['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient (treat): {model_young.params['treat']:.5f}")
print(f"Standard Error: {model_young.bse['treat']:.5f}")
print(f"p-value: {model_young.pvalues['treat']:.4f}")

# 9.5 By gender
print("\n9.5 Heterogeneity by Gender")
for gender, name in [(0, 'Male'), (1, 'Female')]:
    subset = analysis_df[analysis_df['female'] == gender]
    model_g = smf.wls('fulltime ~ daca_eligible + treat + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)',
                       data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"{name}: DiD = {model_g.params['treat']:.5f} (SE = {model_g.bse['treat']:.5f}, p = {model_g.pvalues['treat']:.4f})")

# =============================================================================
# 10. EVENT STUDY / PARALLEL TRENDS
# =============================================================================
print("\n" + "=" * 80)
print("10. EVENT STUDY ANALYSIS")
print("=" * 80)

# Create year interactions with treatment
years = sorted(analysis_df['YEAR'].unique())
base_year = 2011  # Last pre-treatment year

analysis_df['year_rel'] = analysis_df['YEAR']
for yr in years:
    if yr != base_year:
        analysis_df[f'treat_{yr}'] = (analysis_df['daca_eligible'] * (analysis_df['YEAR'] == yr)).astype(int)

# Event study regression
year_vars = ' + '.join([f'treat_{yr}' for yr in years if yr != base_year])
formula = f'fulltime ~ daca_eligible + {year_vars} + female + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)'

model_event = smf.wls(formula, data=analysis_df, weights=analysis_df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (base year = 2011):")
event_results = []
for yr in years:
    if yr != base_year:
        coef = model_event.params[f'treat_{yr}']
        se = model_event.bse[f'treat_{yr}']
        ci_low = model_event.conf_int().loc[f'treat_{yr}', 0]
        ci_high = model_event.conf_int().loc[f'treat_{yr}', 1]
        print(f"Year {yr}: {coef:.5f} (SE: {se:.5f}), 95% CI: [{ci_low:.5f}, {ci_high:.5f}]")
        event_results.append({'year': yr, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})

# =============================================================================
# 11. SAVE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("11. SAVING RESULTS")
print("=" * 80)

# Create results summary
results_summary = {
    'Model': ['Basic DiD', 'With Controls', 'Year FE', 'Year+State FE',
              'Employment Outcome', 'Labor Force Only', 'Alt Age Cutoff',
              'Age<40', 'Males', 'Females'],
    'Coefficient': [
        model1.params['treat'],
        model2.params['treat'],
        model3.params['treat'],
        model4.params['treat'],
        model_emp.params['treat'],
        model_lf.params['treat'],
        model_alt.params['treat_alt'],
        model_young.params['treat'],
        smf.wls('fulltime ~ daca_eligible + treat + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)',
                data=analysis_df[analysis_df['female']==0], weights=analysis_df[analysis_df['female']==0]['PERWT']).fit(cov_type='HC1').params['treat'],
        smf.wls('fulltime ~ daca_eligible + treat + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)',
                data=analysis_df[analysis_df['female']==1], weights=analysis_df[analysis_df['female']==1]['PERWT']).fit(cov_type='HC1').params['treat']
    ],
    'SE': [
        model1.bse['treat'],
        model2.bse['treat'],
        model3.bse['treat'],
        model4.bse['treat'],
        model_emp.bse['treat'],
        model_lf.bse['treat'],
        model_alt.bse['treat_alt'],
        model_young.bse['treat'],
        smf.wls('fulltime ~ daca_eligible + treat + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)',
                data=analysis_df[analysis_df['female']==0], weights=analysis_df[analysis_df['female']==0]['PERWT']).fit(cov_type='HC1').bse['treat'],
        smf.wls('fulltime ~ daca_eligible + treat + married + AGE + age_sq + educ_hs + educ_college + C(YEAR) + C(STATEFIP)',
                data=analysis_df[analysis_df['female']==1], weights=analysis_df[analysis_df['female']==1]['PERWT']).fit(cov_type='HC1').bse['treat']
    ]
}

results_df = pd.DataFrame(results_summary)
results_df['t_stat'] = results_df['Coefficient'] / results_df['SE']
results_df['p_value'] = 2 * (1 - stats.t.cdf(abs(results_df['t_stat']), df=model4.df_resid))
results_df['CI_low'] = results_df['Coefficient'] - 1.96 * results_df['SE']
results_df['CI_high'] = results_df['Coefficient'] + 1.96 * results_df['SE']

print("\n--- RESULTS SUMMARY ---")
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('results_summary.csv', index=False)
print("\nResults saved to results_summary.csv")

# Save event study results
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# =============================================================================
# 12. PREFERRED ESTIMATE
# =============================================================================
print("\n" + "=" * 80)
print("12. PREFERRED ESTIMATE")
print("=" * 80)

print(f"""
PREFERRED MODEL: Difference-in-Differences with Year and State Fixed Effects

Outcome: Full-time employment (UHRSWORK >= 35 hours/week)
Treatment: DACA eligibility (arrival before age 16, born after 1981, in US since 2007)
Control Group: Mexican-born non-citizen Hispanics not meeting DACA criteria

Sample: Hispanic-Mexican, Mexican-born, non-citizen individuals aged 16-64
Years: 2006-2011 (pre-DACA), 2013-2016 (post-DACA), excluding 2012

RESULTS:
- DiD Coefficient: {model4.params['treat']:.5f}
- Robust Standard Error: {model4.bse['treat']:.5f}
- 95% Confidence Interval: [{model4.conf_int().loc['treat', 0]:.5f}, {model4.conf_int().loc['treat', 1]:.5f}]
- t-statistic: {model4.tvalues['treat']:.3f}
- p-value: {model4.pvalues['treat']:.4f}
- Sample Size: {model4.nobs:,.0f}
- R-squared: {model4.rsquared:.4f}

INTERPRETATION:
DACA eligibility is associated with a {model4.params['treat']*100:.2f} percentage point
{'increase' if model4.params['treat'] > 0 else 'decrease'} in the probability of full-time employment.
This effect is {'statistically significant' if model4.pvalues['treat'] < 0.05 else 'not statistically significant'} at the 5% level.
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
