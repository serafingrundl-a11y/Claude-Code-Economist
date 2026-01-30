"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("="*70)
print("STEP 1: Loading Data")
print("="*70)

df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Columns: {list(df.columns)}")

# =============================================================================
# STEP 2: Sample Construction
# =============================================================================
print("\n" + "="*70)
print("STEP 2: Sample Construction")
print("="*70)

# Initial count
print(f"\nInitial sample: {len(df):,}")

# Filter to Hispanic-Mexican (HISPAN == 1 means Mexican)
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican filter: {len(df_sample):,}")

# Filter to Mexican-born (BPL == 200 for Mexico)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After Mexican-born filter: {len(df_sample):,}")

# Filter to non-citizens (CITIZEN == 3)
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After non-citizen filter: {len(df_sample):,}")

# Exclude year 2012 (DACA implemented mid-year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# Restrict to working-age (18-45)
df_sample = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 45)].copy()
print(f"After age 18-45 restriction: {len(df_sample):,}")

# =============================================================================
# STEP 3: Define DACA Eligibility
# =============================================================================
print("\n" + "="*70)
print("STEP 3: Define DACA Eligibility")
print("="*70)

# Calculate age at arrival
# YRIMMIG gives year of immigration, BIRTHYR gives birth year
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Age on June 15, 2012
df_sample['age_june_2012'] = 2012 - df_sample['BIRTHYR']
# Adjust for birth quarter: if born Q3 or Q4 (July onwards), they're slightly younger
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# For simplicity, we use birth year directly (conservative)

# DACA eligibility criteria:
# 1. Arrived before 16th birthday
arrived_before_16 = df_sample['age_at_arrival'] < 16

# 2. Under 31 on June 15, 2012 (born after June 15, 1981)
# Using BIRTHYR >= 1982 as conservative (could be June 1981 with Q3/Q4 birth)
under_31_in_2012 = df_sample['BIRTHYR'] >= 1982

# 3. In US since June 15, 2007 (arrived by 2007)
in_us_since_2007 = df_sample['YRIMMIG'] <= 2007

# 4. YRIMMIG > 0 (has valid immigration year)
valid_immig = df_sample['YRIMMIG'] > 0

# Combined DACA eligibility
df_sample['daca_eligible'] = (arrived_before_16 & under_31_in_2012 &
                               in_us_since_2007 & valid_immig).astype(int)

print(f"\nDACA Eligibility Breakdown:")
print(f"  Arrived before 16: {arrived_before_16.sum():,}")
print(f"  Under 31 in 2012: {under_31_in_2012.sum():,}")
print(f"  In US since 2007: {in_us_since_2007.sum():,}")
print(f"  Valid immigration year: {valid_immig.sum():,}")
print(f"  DACA eligible (all criteria): {df_sample['daca_eligible'].sum():,}")
print(f"  DACA ineligible: {(df_sample['daca_eligible']==0).sum():,}")

# =============================================================================
# STEP 4: Create Analysis Variables
# =============================================================================
print("\n" + "="*70)
print("STEP 4: Create Analysis Variables")
print("="*70)

# Outcome: Full-time employment (35+ hours per week)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Post-DACA indicator (2013 and after)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Interaction term
df_sample['treat_post'] = df_sample['daca_eligible'] * df_sample['post']

# Control variables
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'] == 1).astype(int)
df_sample['age_sq'] = df_sample['AGE'] ** 2

# Education categories (using EDUC general version)
# EDUC: 0=N/A, 1=None, 2=Grade 1-4, 3=Grade 5-8, 4=Grade 9, 5=Grade 10,
#       6=Grade 11, 7=High school, 8=1 yr college, 9=2 yrs college,
#       10=3 yrs college, 11=4+ yrs college
df_sample['educ_hs'] = (df_sample['EDUC'] == 7).astype(int)  # HS grad
df_sample['educ_somecoll'] = ((df_sample['EDUC'] >= 8) & (df_sample['EDUC'] <= 10)).astype(int)
df_sample['educ_college'] = (df_sample['EDUC'] >= 11).astype(int)

# Employment status
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)
df_sample['in_labor_force'] = (df_sample['LABFORCE'] == 2).astype(int)

print(f"\nOutcome Variable - Full-time Employment:")
print(f"  Full-time workers: {df_sample['fulltime'].sum():,}")
print(f"  Not full-time: {(df_sample['fulltime']==0).sum():,}")
print(f"  Full-time rate: {df_sample['fulltime'].mean():.3f}")

print(f"\nTreatment and Post:")
print(f"  Pre-DACA (2006-2011): {(df_sample['post']==0).sum():,}")
print(f"  Post-DACA (2013-2016): {(df_sample['post']==1).sum():,}")

# =============================================================================
# STEP 5: Summary Statistics
# =============================================================================
print("\n" + "="*70)
print("STEP 5: Summary Statistics")
print("="*70)

def weighted_stats(df, var, weight='PERWT'):
    """Calculate weighted mean and std"""
    w = df[weight]
    mean = np.average(df[var], weights=w)
    var_val = np.average((df[var] - mean)**2, weights=w)
    std = np.sqrt(var_val)
    return mean, std

# Split by treatment group
treated = df_sample[df_sample['daca_eligible'] == 1]
control = df_sample[df_sample['daca_eligible'] == 0]

print("\nSample sizes by group:")
print(f"  DACA-eligible: {len(treated):,}")
print(f"  DACA-ineligible: {len(control):,}")

print("\n--- DACA-Eligible (Treatment) ---")
print(f"  Mean age: {treated['AGE'].mean():.1f}")
print(f"  Female share: {treated['female'].mean():.3f}")
print(f"  Married share: {treated['married'].mean():.3f}")
print(f"  Full-time rate: {treated['fulltime'].mean():.3f}")
print(f"  Employment rate: {treated['employed'].mean():.3f}")

print("\n--- DACA-Ineligible (Control) ---")
print(f"  Mean age: {control['AGE'].mean():.1f}")
print(f"  Female share: {control['female'].mean():.3f}")
print(f"  Married share: {control['married'].mean():.3f}")
print(f"  Full-time rate: {control['fulltime'].mean():.3f}")
print(f"  Employment rate: {control['employed'].mean():.3f}")

# Pre/Post breakdown
print("\n--- Full-time rates by group and period ---")
for eligible in [1, 0]:
    group_name = "DACA-eligible" if eligible == 1 else "DACA-ineligible"
    for post in [0, 1]:
        period = "Post" if post == 1 else "Pre"
        subset = df_sample[(df_sample['daca_eligible']==eligible) & (df_sample['post']==post)]
        if len(subset) > 0:
            ft_rate = np.average(subset['fulltime'], weights=subset['PERWT'])
            print(f"  {group_name}, {period}: {ft_rate:.4f} (n={len(subset):,})")

# =============================================================================
# STEP 6: Difference-in-Differences Analysis
# =============================================================================
print("\n" + "="*70)
print("STEP 6: Difference-in-Differences Analysis")
print("="*70)

# Simple 2x2 DiD calculation first
print("\n--- Simple 2x2 DiD Calculation ---")
cells = {}
for eligible in [0, 1]:
    for post in [0, 1]:
        subset = df_sample[(df_sample['daca_eligible']==eligible) & (df_sample['post']==post)]
        cells[(eligible, post)] = np.average(subset['fulltime'], weights=subset['PERWT'])

print(f"DACA-ineligible, Pre:  {cells[(0,0)]:.4f}")
print(f"DACA-ineligible, Post: {cells[(0,1)]:.4f}")
print(f"DACA-eligible, Pre:    {cells[(1,0)]:.4f}")
print(f"DACA-eligible, Post:   {cells[(1,1)]:.4f}")

did_simple = (cells[(1,1)] - cells[(1,0)]) - (cells[(0,1)] - cells[(0,0)])
print(f"\nSimple DiD estimate: {did_simple:.4f}")
print(f"  Treatment change: {cells[(1,1)] - cells[(1,0)]:.4f}")
print(f"  Control change: {cells[(0,1)] - cells[(0,0)]:.4f}")

# =============================================================================
# STEP 7: Regression Analysis
# =============================================================================
print("\n" + "="*70)
print("STEP 7: Regression Analysis")
print("="*70)

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD (no controls) ---")
X1 = sm.add_constant(df_sample[['daca_eligible', 'post', 'treat_post']])
y = df_sample['fulltime']
weights = df_sample['PERWT']

model1 = sm.WLS(y, X1, weights=weights).fit(cov_type='cluster',
                                             cov_kwds={'groups': df_sample['STATEFIP']})
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
X2_vars = ['daca_eligible', 'post', 'treat_post', 'AGE', 'age_sq', 'female', 'married']
X2 = sm.add_constant(df_sample[X2_vars])
model2 = sm.WLS(y, X2, weights=weights).fit(cov_type='cluster',
                                             cov_kwds={'groups': df_sample['STATEFIP']})
print(f"DiD coefficient (treat_post): {model2.params['treat_post']:.4f}")
print(f"Std Error: {model2.bse['treat_post']:.4f}")
print(f"t-stat: {model2.tvalues['treat_post']:.3f}")
print(f"p-value: {model2.pvalues['treat_post']:.4f}")

# Model 3: DiD with demographic and education controls
print("\n--- Model 3: DiD with demographic and education controls ---")
X3_vars = ['daca_eligible', 'post', 'treat_post', 'AGE', 'age_sq', 'female',
           'married', 'educ_hs', 'educ_somecoll', 'educ_college']
X3 = sm.add_constant(df_sample[X3_vars])
model3 = sm.WLS(y, X3, weights=weights).fit(cov_type='cluster',
                                             cov_kwds={'groups': df_sample['STATEFIP']})
print(f"DiD coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"Std Error: {model3.bse['treat_post']:.4f}")
print(f"t-stat: {model3.tvalues['treat_post']:.3f}")
print(f"p-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: DiD with state and year fixed effects
print("\n--- Model 4: DiD with State and Year Fixed Effects ---")
# Create dummies for states and years
state_dummies = pd.get_dummies(df_sample['STATEFIP'], prefix='state', drop_first=True)
year_dummies = pd.get_dummies(df_sample['YEAR'], prefix='year', drop_first=True)

X4_base = df_sample[['daca_eligible', 'treat_post', 'AGE', 'age_sq', 'female',
                     'married', 'educ_hs', 'educ_somecoll', 'educ_college']].reset_index(drop=True)
state_dummies = state_dummies.reset_index(drop=True)
year_dummies = year_dummies.reset_index(drop=True)
X4 = pd.concat([X4_base, state_dummies, year_dummies], axis=1)
X4 = sm.add_constant(X4)

# Ensure all columns are numeric
X4 = X4.astype(float)
y_reset = y.reset_index(drop=True)
weights_reset = weights.reset_index(drop=True)
groups_reset = df_sample['STATEFIP'].reset_index(drop=True)

model4 = sm.WLS(y_reset, X4, weights=weights_reset).fit(cov_type='cluster',
                                             cov_kwds={'groups': groups_reset})
print(f"DiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"Std Error: {model4.bse['treat_post']:.4f}")
print(f"t-stat: {model4.tvalues['treat_post']:.3f}")
print(f"p-value: {model4.pvalues['treat_post']:.4f}")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {int(model4.nobs):,}")

# =============================================================================
# STEP 8: Robustness Checks
# =============================================================================
print("\n" + "="*70)
print("STEP 8: Robustness Checks")
print("="*70)

# Robustness 1: Different age ranges
print("\n--- Robustness 1: Different Age Ranges ---")
for age_min, age_max in [(18, 35), (20, 40), (18, 50)]:
    df_robust = df_sample[(df_sample['AGE'] >= age_min) & (df_sample['AGE'] <= age_max)]
    if len(df_robust) > 100:
        X_r = sm.add_constant(df_robust[['daca_eligible', 'post', 'treat_post',
                                          'AGE', 'age_sq', 'female', 'married']])
        y_r = df_robust['fulltime']
        w_r = df_robust['PERWT']
        model_r = sm.WLS(y_r, X_r, weights=w_r).fit(cov_type='cluster',
                                                    cov_kwds={'groups': df_robust['STATEFIP']})
        print(f"Ages {age_min}-{age_max}: coef={model_r.params['treat_post']:.4f}, "
              f"se={model_r.bse['treat_post']:.4f}, n={len(df_robust):,}")

# Robustness 2: By gender
print("\n--- Robustness 2: By Gender ---")
for sex, sex_name in [(0, 'Male'), (1, 'Female')]:
    df_sex = df_sample[df_sample['female'] == sex]
    X_s = sm.add_constant(df_sex[['daca_eligible', 'post', 'treat_post',
                                   'AGE', 'age_sq', 'married']])
    y_s = df_sex['fulltime']
    w_s = df_sex['PERWT']
    model_s = sm.WLS(y_s, X_s, weights=w_s).fit(cov_type='cluster',
                                                cov_kwds={'groups': df_sex['STATEFIP']})
    print(f"{sex_name}: coef={model_s.params['treat_post']:.4f}, "
          f"se={model_s.bse['treat_post']:.4f}, n={len(df_sex):,}")

# Robustness 3: Employment (any) as outcome
print("\n--- Robustness 3: Employment (any) as Outcome ---")
X_emp = sm.add_constant(df_sample[['daca_eligible', 'post', 'treat_post',
                                    'AGE', 'age_sq', 'female', 'married']])
y_emp = df_sample['employed']
model_emp = sm.WLS(y_emp, X_emp, weights=weights).fit(cov_type='cluster',
                                                       cov_kwds={'groups': df_sample['STATEFIP']})
print(f"Employment (any): coef={model_emp.params['treat_post']:.4f}, "
      f"se={model_emp.bse['treat_post']:.4f}")

# Robustness 4: Labor force participation as outcome
print("\n--- Robustness 4: Labor Force Participation as Outcome ---")
y_lf = df_sample['in_labor_force']
model_lf = sm.WLS(y_lf, X_emp, weights=weights).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df_sample['STATEFIP']})
print(f"Labor force participation: coef={model_lf.params['treat_post']:.4f}, "
      f"se={model_lf.bse['treat_post']:.4f}")

# =============================================================================
# STEP 9: Parallel Trends Analysis
# =============================================================================
print("\n" + "="*70)
print("STEP 9: Parallel Trends Analysis")
print("="*70)

# Event study / year-by-year effects
print("\n--- Year-by-Year Full-time Rates ---")
yearly_rates = df_sample.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
yearly_rates.columns = ['Control', 'Treatment']
print(yearly_rates)

# Calculate year-specific treatment effects
print("\n--- Event Study Coefficients ---")
df_sample['year_2006'] = (df_sample['YEAR'] == 2006).astype(int)
df_sample['year_2007'] = (df_sample['YEAR'] == 2007).astype(int)
df_sample['year_2008'] = (df_sample['YEAR'] == 2008).astype(int)
df_sample['year_2009'] = (df_sample['YEAR'] == 2009).astype(int)
df_sample['year_2010'] = (df_sample['YEAR'] == 2010).astype(int)
df_sample['year_2011'] = (df_sample['YEAR'] == 2011).astype(int)
df_sample['year_2013'] = (df_sample['YEAR'] == 2013).astype(int)
df_sample['year_2014'] = (df_sample['YEAR'] == 2014).astype(int)
df_sample['year_2015'] = (df_sample['YEAR'] == 2015).astype(int)
df_sample['year_2016'] = (df_sample['YEAR'] == 2016).astype(int)

# Interaction with treatment (omit 2011 as reference)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_sample[f'treat_x_{yr}'] = df_sample['daca_eligible'] * df_sample[f'year_{yr}']

event_vars = ['daca_eligible', 'treat_x_2006', 'treat_x_2007', 'treat_x_2008',
              'treat_x_2009', 'treat_x_2010', 'treat_x_2013', 'treat_x_2014',
              'treat_x_2015', 'treat_x_2016',
              'year_2006', 'year_2007', 'year_2008', 'year_2009', 'year_2010',
              'year_2013', 'year_2014', 'year_2015', 'year_2016',
              'AGE', 'age_sq', 'female', 'married']

X_event = sm.add_constant(df_sample[event_vars])
model_event = sm.WLS(y, X_event, weights=weights).fit(cov_type='cluster',
                                                       cov_kwds={'groups': df_sample['STATEFIP']})

print("Event Study Coefficients (treatment x year, base=2011):")
event_coefs = []
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treat_x_{yr}']
    se = model_event.bse[f'treat_x_{yr}']
    print(f"  {yr}: {coef:.4f} ({se:.4f})")
    event_coefs.append({'year': yr, 'coef': coef, 'se': se})

event_df = pd.DataFrame(event_coefs)

# =============================================================================
# STEP 10: Create Output Tables
# =============================================================================
print("\n" + "="*70)
print("STEP 10: Creating Output for Report")
print("="*70)

# Save key results for the report
results = {
    'simple_did': did_simple,
    'model1_coef': model1.params['treat_post'],
    'model1_se': model1.bse['treat_post'],
    'model2_coef': model2.params['treat_post'],
    'model2_se': model2.bse['treat_post'],
    'model3_coef': model3.params['treat_post'],
    'model3_se': model3.bse['treat_post'],
    'model4_coef': model4.params['treat_post'],
    'model4_se': model4.bse['treat_post'],
    'model4_pval': model4.pvalues['treat_post'],
    'model4_r2': model4.rsquared,
    'n_total': len(df_sample),
    'n_treated': len(treated),
    'n_control': len(control),
    'pre_treat_mean': cells[(1,0)],
    'post_treat_mean': cells[(1,1)],
    'pre_control_mean': cells[(0,0)],
    'post_control_mean': cells[(0,1)],
}

print("\n=== SUMMARY OF KEY RESULTS ===")
print(f"Sample size: {results['n_total']:,}")
print(f"Treatment group: {results['n_treated']:,}")
print(f"Control group: {results['n_control']:,}")
print(f"\nSimple DiD estimate: {results['simple_did']:.4f}")
print(f"\nPreferred estimate (Model 4 with state/year FE):")
print(f"  Coefficient: {results['model4_coef']:.4f}")
print(f"  Standard Error: {results['model4_se']:.4f}")
print(f"  95% CI: [{results['model4_coef'] - 1.96*results['model4_se']:.4f}, "
      f"{results['model4_coef'] + 1.96*results['model4_se']:.4f}]")
print(f"  P-value: {results['model4_pval']:.4f}")

# Save yearly rates for plotting
yearly_rates.to_csv('yearly_rates.csv')
event_df.to_csv('event_study.csv', index=False)

# Save full summary statistics
summary_stats = pd.DataFrame({
    'Variable': ['Age', 'Female', 'Married', 'HS Graduate', 'Some College',
                 'College+', 'Full-time', 'Employed', 'In Labor Force'],
    'Treatment Mean': [treated['AGE'].mean(), treated['female'].mean(),
                      treated['married'].mean(), treated['educ_hs'].mean(),
                      treated['educ_somecoll'].mean(), treated['educ_college'].mean(),
                      treated['fulltime'].mean(), treated['employed'].mean(),
                      treated['in_labor_force'].mean()],
    'Control Mean': [control['AGE'].mean(), control['female'].mean(),
                    control['married'].mean(), control['educ_hs'].mean(),
                    control['educ_somecoll'].mean(), control['educ_college'].mean(),
                    control['fulltime'].mean(), control['employed'].mean(),
                    control['in_labor_force'].mean()]
})
summary_stats.to_csv('summary_stats.csv', index=False)

print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)

# Print results formatted for LaTeX
print("\n=== LaTeX Table Ready Values ===")
print(f"Model 1 (Basic): {results['model1_coef']:.4f} ({results['model1_se']:.4f})")
print(f"Model 2 (Demographics): {results['model2_coef']:.4f} ({results['model2_se']:.4f})")
print(f"Model 3 (+ Education): {results['model3_coef']:.4f} ({results['model3_se']:.4f})")
print(f"Model 4 (+ State/Year FE): {results['model4_coef']:.4f} ({results['model4_se']:.4f})")
