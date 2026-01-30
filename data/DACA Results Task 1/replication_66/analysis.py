"""
DACA Replication Study - Analysis Script
Research Question: Impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US

Author: Replication 66
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
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: SAMPLE CONSTRUCTION
# =============================================================================
print("\n" + "="*80)
print("STEP 2: SAMPLE CONSTRUCTION")
print("="*80)

# Initial counts
print(f"\nInitial sample size: {len(df):,}")

# Step 2a: Hispanic-Mexican ethnicity (HISPAN = 1)
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican filter (HISPAN=1): {len(df_sample):,}")

# Step 2b: Born in Mexico (BPL = 200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After Mexican-born filter (BPL=200): {len(df_sample):,}")

# Step 2c: Non-citizens (CITIZEN = 3)
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After non-citizen filter (CITIZEN=3): {len(df_sample):,}")

# Step 2d: Valid immigration year (YRIMMIG > 0)
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
print(f"After valid immigration year filter: {len(df_sample):,}")

# Step 2e: Working age (16-64)
df_sample = df_sample[(df_sample['AGE'] >= 16) & (df_sample['AGE'] <= 64)].copy()
print(f"After working age filter (16-64): {len(df_sample):,}")

# Step 2f: Exclude 2012 (transition year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# =============================================================================
# STEP 3: CREATE DACA ELIGIBILITY INDICATOR
# =============================================================================
print("\n" + "="*80)
print("STEP 3: CREATE DACA ELIGIBILITY INDICATOR")
print("="*80)

# Calculate age at arrival to US
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Criterion 1: Arrived before age 16
df_sample['arrived_before_16'] = (df_sample['age_at_arrival'] < 16).astype(int)

# Criterion 2: Born after June 15, 1981
df_sample['born_after_june1981'] = (
    (df_sample['BIRTHYR'] > 1981) |
    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))
).astype(int)

# Criterion 3: Present in US since June 2007
df_sample['in_us_since_2007'] = (df_sample['YRIMMIG'] <= 2007).astype(int)

# Combined DACA eligibility
df_sample['daca_eligible'] = (
    (df_sample['arrived_before_16'] == 1) &
    (df_sample['born_after_june1981'] == 1) &
    (df_sample['in_us_since_2007'] == 1)
).astype(int)

print(f"\nDACA Eligibility Breakdown:")
print(f"  Arrived before age 16: {df_sample['arrived_before_16'].sum():,} ({df_sample['arrived_before_16'].mean()*100:.1f}%)")
print(f"  Born after June 1981: {df_sample['born_after_june1981'].sum():,} ({df_sample['born_after_june1981'].mean()*100:.1f}%)")
print(f"  In US since 2007: {df_sample['in_us_since_2007'].sum():,} ({df_sample['in_us_since_2007'].mean()*100:.1f}%)")
print(f"  DACA Eligible (all criteria): {df_sample['daca_eligible'].sum():,} ({df_sample['daca_eligible'].mean()*100:.1f}%)")

# =============================================================================
# STEP 4: CREATE OUTCOME AND TREATMENT VARIABLES
# =============================================================================
print("\n" + "="*80)
print("STEP 4: CREATE OUTCOME AND TREATMENT VARIABLES")
print("="*80)

# Outcome: Full-time employment (35+ hours per week)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Post-DACA indicator (2013 onwards)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Interaction term (DiD estimator)
df_sample['eligible_post'] = df_sample['daca_eligible'] * df_sample['post']

# Employment indicator
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# Labor force participation
df_sample['in_labor_force'] = (df_sample['LABFORCE'] == 2).astype(int)

print(f"\nOutcome Variable Summary:")
print(f"  Full-time employment rate: {df_sample['fulltime'].mean()*100:.1f}%")
print(f"  Employment rate: {df_sample['employed'].mean()*100:.1f}%")
print(f"  Labor force participation: {df_sample['in_labor_force'].mean()*100:.1f}%")

print(f"\nTreatment Variables:")
print(f"  Pre-DACA observations (2006-2011): {(df_sample['post'] == 0).sum():,}")
print(f"  Post-DACA observations (2013-2016): {(df_sample['post'] == 1).sum():,}")

# =============================================================================
# STEP 5: CREATE CONTROL VARIABLES
# =============================================================================
print("\n" + "="*80)
print("STEP 5: CREATE CONTROL VARIABLES")
print("="*80)

# Age squared
df_sample['age_sq'] = df_sample['AGE'] ** 2

# Sex indicator (female = 1)
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)

# Marital status (married = 1)
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Education categories
df_sample['educ_hs'] = (df_sample['EDUC'] >= 6).astype(int)
df_sample['educ_college'] = (df_sample['EDUC'] >= 10).astype(int)

# Years in US
df_sample['years_in_us'] = df_sample['YEAR'] - df_sample['YRIMMIG']

# State fixed effects (use numeric codes directly)
df_sample['state'] = df_sample['STATEFIP'].astype(int)

print(f"\nControl Variable Summary:")
print(f"  Female: {df_sample['female'].mean()*100:.1f}%")
print(f"  Married: {df_sample['married'].mean()*100:.1f}%")
print(f"  High school+: {df_sample['educ_hs'].mean()*100:.1f}%")
print(f"  Some college+: {df_sample['educ_college'].mean()*100:.1f}%")
print(f"  Mean age: {df_sample['AGE'].mean():.1f}")
print(f"  Mean years in US: {df_sample['years_in_us'].mean():.1f}")

# =============================================================================
# STEP 6: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("STEP 6: DESCRIPTIVE STATISTICS")
print("="*80)

# Summary by eligibility and period
print("\n--- Full-time Employment Rates by Group and Period ---")
summary = df_sample.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'in_labor_force': 'mean',
    'PERWT': 'sum'
}).round(4)
print(summary)

# Weighted means
print("\n--- Weighted Full-time Employment Rates ---")
for elig in [0, 1]:
    for period in [0, 1]:
        subset = df_sample[(df_sample['daca_eligible'] == elig) & (df_sample['post'] == period)]
        if len(subset) > 0:
            weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
            elig_label = "Eligible" if elig == 1 else "Not Eligible"
            period_label = "Post (2013-16)" if period == 1 else "Pre (2006-11)"
            print(f"  {elig_label}, {period_label}: {weighted_mean*100:.2f}%")

# Simple DiD calculation
print("\n--- Simple Difference-in-Differences ---")
pre_elig = np.average(df_sample[(df_sample['daca_eligible'] == 1) & (df_sample['post'] == 0)]['fulltime'],
                       weights=df_sample[(df_sample['daca_eligible'] == 1) & (df_sample['post'] == 0)]['PERWT'])
post_elig = np.average(df_sample[(df_sample['daca_eligible'] == 1) & (df_sample['post'] == 1)]['fulltime'],
                        weights=df_sample[(df_sample['daca_eligible'] == 1) & (df_sample['post'] == 1)]['PERWT'])
pre_ctrl = np.average(df_sample[(df_sample['daca_eligible'] == 0) & (df_sample['post'] == 0)]['fulltime'],
                       weights=df_sample[(df_sample['daca_eligible'] == 0) & (df_sample['post'] == 0)]['PERWT'])
post_ctrl = np.average(df_sample[(df_sample['daca_eligible'] == 0) & (df_sample['post'] == 1)]['fulltime'],
                        weights=df_sample[(df_sample['daca_eligible'] == 0) & (df_sample['post'] == 1)]['PERWT'])

diff_elig = post_elig - pre_elig
diff_ctrl = post_ctrl - pre_ctrl
did_simple = diff_elig - diff_ctrl

print(f"  Eligible: Pre={pre_elig*100:.2f}%, Post={post_elig*100:.2f}%, Diff={diff_elig*100:.2f}pp")
print(f"  Control:  Pre={pre_ctrl*100:.2f}%, Post={post_ctrl*100:.2f}%, Diff={diff_ctrl*100:.2f}pp")
print(f"  DiD Estimate: {did_simple*100:.2f} percentage points")

# =============================================================================
# STEP 7: REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("STEP 7: REGRESSION ANALYSIS")
print("="*80)

# Convert all variables to float for regression
df_reg = df_sample[['fulltime', 'employed', 'in_labor_force', 'daca_eligible', 'post', 'eligible_post',
                    'AGE', 'age_sq', 'female', 'married', 'educ_hs', 'educ_college', 'years_in_us',
                    'YEAR', 'state', 'PERWT']].copy()
df_reg = df_reg.astype(float)

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (no controls) ---")
X1 = sm.add_constant(df_reg[['daca_eligible', 'post', 'eligible_post']])
y = df_reg['fulltime']
weights = df_reg['PERWT']

model1 = sm.WLS(y, X1, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': df_reg['state']})
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
controls = ['AGE', 'age_sq', 'female', 'married', 'educ_hs', 'educ_college', 'years_in_us']
X2 = sm.add_constant(df_reg[['daca_eligible', 'post', 'eligible_post'] + controls])
model2 = sm.WLS(y, X2, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': df_reg['state']})
print(model2.summary().tables[1])

# Model 3: DiD with demographic controls + year FE
print("\n--- Model 3: DiD with Demographics + Year FE ---")
year_dummies = pd.get_dummies(df_reg['YEAR'], prefix='year', drop_first=True, dtype=float)
df_with_year = pd.concat([df_reg[['daca_eligible', 'eligible_post'] + controls], year_dummies], axis=1)
X3 = sm.add_constant(df_with_year)
model3 = sm.WLS(y, X3, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': df_reg['state']})
print("\nKey coefficients:")
print(f"  daca_eligible: {model3.params['daca_eligible']:.4f} (SE: {model3.bse['daca_eligible']:.4f})")
print(f"  eligible_post (DiD): {model3.params['eligible_post']:.4f} (SE: {model3.bse['eligible_post']:.4f})")

# Model 4: Full model with state and year FE
print("\n--- Model 4: Full Model with State and Year FE ---")
state_dummies = pd.get_dummies(df_reg['state'], prefix='state', drop_first=True, dtype=float)
df_full = pd.concat([df_reg[['daca_eligible', 'eligible_post'] + controls], year_dummies, state_dummies], axis=1)
X4 = sm.add_constant(df_full)
model4 = sm.WLS(y, X4, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': df_reg['state']})
print("\nKey coefficients:")
print(f"  daca_eligible: {model4.params['daca_eligible']:.4f} (SE: {model4.bse['daca_eligible']:.4f})")
print(f"  eligible_post (DiD): {model4.params['eligible_post']:.4f} (SE: {model4.bse['eligible_post']:.4f})")
print(f"  P-value: {model4.pvalues['eligible_post']:.4f}")

# =============================================================================
# STEP 8: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("STEP 8: ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Employment as outcome
print("\n--- Robustness 1: Employment as Outcome ---")
y_emp = df_reg['employed']
model_emp = sm.WLS(y_emp, X4, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': df_reg['state']})
print(f"  eligible_post (DiD): {model_emp.params['eligible_post']:.4f} (SE: {model_emp.bse['eligible_post']:.4f})")

# Robustness 2: Labor force participation as outcome
print("\n--- Robustness 2: Labor Force Participation as Outcome ---")
y_lfp = df_reg['in_labor_force']
model_lfp = sm.WLS(y_lfp, X4, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': df_reg['state']})
print(f"  eligible_post (DiD): {model_lfp.params['eligible_post']:.4f} (SE: {model_lfp.bse['eligible_post']:.4f})")

# Robustness 3: Males only
print("\n--- Robustness 3: Males Only ---")
df_male = df_reg[df_reg['female'] == 0].copy()
y_male = df_male['fulltime']
w_male = df_male['PERWT']

year_dum_m = pd.get_dummies(df_male['YEAR'], prefix='year', drop_first=True, dtype=float)
state_dum_m = pd.get_dummies(df_male['state'], prefix='state', drop_first=True, dtype=float)
controls_male = ['AGE', 'age_sq', 'married', 'educ_hs', 'educ_college', 'years_in_us']
df_male_full = pd.concat([df_male[['daca_eligible', 'eligible_post'] + controls_male].reset_index(drop=True),
                          year_dum_m.reset_index(drop=True),
                          state_dum_m.reset_index(drop=True)], axis=1)
X_male = sm.add_constant(df_male_full)
model_male = sm.WLS(y_male.reset_index(drop=True), X_male, weights=w_male.reset_index(drop=True)).fit(
    cov_type='cluster', cov_kwds={'groups': df_male['state'].reset_index(drop=True)})
print(f"  eligible_post (DiD): {model_male.params['eligible_post']:.4f} (SE: {model_male.bse['eligible_post']:.4f})")

# Robustness 4: Females only
print("\n--- Robustness 4: Females Only ---")
df_fem = df_reg[df_reg['female'] == 1].copy()
y_fem = df_fem['fulltime']
w_fem = df_fem['PERWT']

year_dum_f = pd.get_dummies(df_fem['YEAR'], prefix='year', drop_first=True, dtype=float)
state_dum_f = pd.get_dummies(df_fem['state'], prefix='state', drop_first=True, dtype=float)
df_fem_full = pd.concat([df_fem[['daca_eligible', 'eligible_post'] + controls_male].reset_index(drop=True),
                         year_dum_f.reset_index(drop=True),
                         state_dum_f.reset_index(drop=True)], axis=1)
X_fem = sm.add_constant(df_fem_full)
model_fem = sm.WLS(y_fem.reset_index(drop=True), X_fem, weights=w_fem.reset_index(drop=True)).fit(
    cov_type='cluster', cov_kwds={'groups': df_fem['state'].reset_index(drop=True)})
print(f"  eligible_post (DiD): {model_fem.params['eligible_post']:.4f} (SE: {model_fem.bse['eligible_post']:.4f})")

# =============================================================================
# STEP 9: EVENT STUDY / PARALLEL TRENDS
# =============================================================================
print("\n" + "="*80)
print("STEP 9: EVENT STUDY ANALYSIS")
print("="*80)

# Create year-specific treatment effects
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
df_event = df_reg.copy()
for yr in years:
    df_event[f'elig_year_{yr}'] = (df_event['daca_eligible'] * (df_event['YEAR'] == yr)).astype(float)

# Exclude 2011 as reference year
event_vars = [f'elig_year_{yr}' for yr in years if yr != 2011]
df_event_full = pd.concat([df_event[['daca_eligible'] + event_vars + controls],
                           year_dummies, state_dummies], axis=1)
X_event = sm.add_constant(df_event_full)
model_event = sm.WLS(y, X_event, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': df_reg['state']})

print("\nEvent Study Coefficients (Reference: 2011):")
print("-" * 60)
for yr in years:
    if yr != 2011:
        coef = model_event.params[f'elig_year_{yr}']
        se = model_event.bse[f'elig_year_{yr}']
        pval = model_event.pvalues[f'elig_year_{yr}']
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {yr}: {coef:7.4f} (SE: {se:.4f}) [{ci_low:7.4f}, {ci_high:7.4f}] {sig}")
    else:
        print(f"  {yr}: {0:7.4f} (Reference)")

# =============================================================================
# STEP 10: SAVE RESULTS
# =============================================================================
print("\n" + "="*80)
print("STEP 10: SAVE RESULTS")
print("="*80)

# Create results summary
results = {
    'model': ['Basic DiD', 'With Controls', 'Controls + Year FE', 'Full Model (State + Year FE)',
              'Employment', 'LFP', 'Males Only', 'Females Only'],
    'coefficient': [
        model1.params['eligible_post'],
        model2.params['eligible_post'],
        model3.params['eligible_post'],
        model4.params['eligible_post'],
        model_emp.params['eligible_post'],
        model_lfp.params['eligible_post'],
        model_male.params['eligible_post'],
        model_fem.params['eligible_post']
    ],
    'std_error': [
        model1.bse['eligible_post'],
        model2.bse['eligible_post'],
        model3.bse['eligible_post'],
        model4.bse['eligible_post'],
        model_emp.bse['eligible_post'],
        model_lfp.bse['eligible_post'],
        model_male.bse['eligible_post'],
        model_fem.bse['eligible_post']
    ],
    'pvalue': [
        model1.pvalues['eligible_post'],
        model2.pvalues['eligible_post'],
        model3.pvalues['eligible_post'],
        model4.pvalues['eligible_post'],
        model_emp.pvalues['eligible_post'],
        model_lfp.pvalues['eligible_post'],
        model_male.pvalues['eligible_post'],
        model_fem.pvalues['eligible_post']
    ]
}

results_df = pd.DataFrame(results)
results_df['ci_lower'] = results_df['coefficient'] - 1.96 * results_df['std_error']
results_df['ci_upper'] = results_df['coefficient'] + 1.96 * results_df['std_error']
results_df.to_csv('regression_results.csv', index=False)

print("\nMain Results Summary:")
print(results_df.to_string(index=False))

# Save event study results
event_results = []
for yr in years:
    if yr != 2011:
        event_results.append({
            'year': yr,
            'coefficient': model_event.params[f'elig_year_{yr}'],
            'std_error': model_event.bse[f'elig_year_{yr}'],
            'pvalue': model_event.pvalues[f'elig_year_{yr}']
        })
    else:
        event_results.append({
            'year': yr,
            'coefficient': 0,
            'std_error': 0,
            'pvalue': 1
        })

event_df = pd.DataFrame(event_results)
event_df['ci_lower'] = event_df['coefficient'] - 1.96 * event_df['std_error']
event_df['ci_upper'] = event_df['coefficient'] + 1.96 * event_df['std_error']
event_df.to_csv('event_study_results.csv', index=False)

# Save descriptive statistics
desc_stats = df_sample.groupby(['daca_eligible', 'post']).apply(
    lambda x: pd.Series({
        'n': len(x),
        'n_weighted': x['PERWT'].sum(),
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'employed_rate': np.average(x['employed'], weights=x['PERWT']),
        'lfp_rate': np.average(x['in_labor_force'], weights=x['PERWT']),
        'mean_age': np.average(x['AGE'], weights=x['PERWT']),
        'female_share': np.average(x['female'], weights=x['PERWT']),
        'married_share': np.average(x['married'], weights=x['PERWT']),
        'hs_share': np.average(x['educ_hs'], weights=x['PERWT'])
    })
).reset_index()
desc_stats.to_csv('descriptive_stats.csv', index=False)

# =============================================================================
# STEP 11: FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nSample Size: {len(df_sample):,}")
print(f"DACA Eligible: {df_sample['daca_eligible'].sum():,} ({df_sample['daca_eligible'].mean()*100:.1f}%)")
print(f"Control Group: {(1-df_sample['daca_eligible']).sum():,} ({(1-df_sample['daca_eligible'].mean())*100:.1f}%)")

print(f"\n*** PREFERRED ESTIMATE (Full Model with State and Year FE) ***")
print(f"DiD Coefficient: {model4.params['eligible_post']:.4f}")
print(f"Standard Error: {model4.bse['eligible_post']:.4f}")
ci_low = model4.params['eligible_post'] - 1.96*model4.bse['eligible_post']
ci_high = model4.params['eligible_post'] + 1.96*model4.bse['eligible_post']
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"P-value: {model4.pvalues['eligible_post']:.4f}")

# Interpretation
print(f"\nInterpretation:")
print(f"  DACA eligibility increased the probability of full-time employment")
print(f"  by approximately {model4.params['eligible_post']*100:.1f} percentage points")
print(f"  among Hispanic-Mexican, Mexican-born non-citizens.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
