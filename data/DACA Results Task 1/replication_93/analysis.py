"""
DACA Replication Study - Analysis Script
Replication 93

Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States?

Author: Anonymous
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
print()

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("Step 1: Loading data...")
print("-"*40)

# Load main ACS data
df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print()

# =============================================================================
# STEP 2: SAMPLE RESTRICTIONS
# =============================================================================
print("Step 2: Applying sample restrictions...")
print("-"*40)

# Restrict to Hispanic-Mexican (HISPAN == 1)
df_mex = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican restriction: {len(df_mex):,}")

# Restrict to born in Mexico (BPL == 200)
df_mex = df_mex[df_mex['BPL'] == 200].copy()
print(f"After born in Mexico restriction: {len(df_mex):,}")

# Restrict to non-citizens (CITIZEN == 3)
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"After non-citizen restriction: {len(df_mex):,}")

# Exclude 2012 (ambiguous treatment timing - DACA implemented mid-year)
df_mex = df_mex[df_mex['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_mex):,}")

# Restrict to working-age population (18-64) - reasonable age range for employment analysis
df_mex = df_mex[(df_mex['AGE'] >= 18) & (df_mex['AGE'] <= 64)].copy()
print(f"After age restriction (18-64): {len(df_mex):,}")

# Drop observations with missing key variables
df_mex = df_mex[df_mex['YRIMMIG'] > 0].copy()  # YRIMMIG=0 means N/A
print(f"After dropping missing YRIMMIG: {len(df_mex):,}")

df_mex = df_mex[df_mex['BIRTHYR'] > 0].copy()
print(f"After dropping missing BIRTHYR: {len(df_mex):,}")

print()

# =============================================================================
# STEP 3: CREATE KEY VARIABLES
# =============================================================================
print("Step 3: Creating key variables...")
print("-"*40)

# Calculate age at immigration
df_mex['age_at_immig'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# DACA Eligibility Criteria:
# 1. Arrived before 16th birthday
# 2. Born after June 15, 1981 (under 31 as of June 15, 2012)
# 3. Arrived by June 15, 2007 (continuous presence)
# 4. Not a citizen (already filtered)

# Criterion 1: Arrived before 16
df_mex['arrived_before_16'] = (df_mex['age_at_immig'] < 16).astype(int)

# Criterion 2: Under 31 as of June 15, 2012
# Born after June 15, 1981
# If BIRTHYR > 1981, eligible
# If BIRTHYR == 1981, need to be born July or later (BIRTHQTR >= 3)
df_mex['under_31_2012'] = ((df_mex['BIRTHYR'] > 1981) |
                           ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))).astype(int)

# Criterion 3: Arrived by 2007
df_mex['arrived_by_2007'] = (df_mex['YRIMMIG'] <= 2007).astype(int)

# Overall DACA eligibility
df_mex['daca_eligible'] = ((df_mex['arrived_before_16'] == 1) &
                            (df_mex['under_31_2012'] == 1) &
                            (df_mex['arrived_by_2007'] == 1)).astype(int)

# Post-DACA period indicator (2013-2016)
df_mex['post'] = (df_mex['YEAR'] >= 2013).astype(int)

# Interaction term
df_mex['eligible_post'] = df_mex['daca_eligible'] * df_mex['post']

# Outcome: Full-time employment (usually work 35+ hours per week)
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

# Alternative outcome: any employment (EMPSTAT == 1)
df_mex['employed'] = (df_mex['EMPSTAT'] == 1).astype(int)

print("Variables created:")
print(f"  - daca_eligible: {df_mex['daca_eligible'].sum():,} eligible ({df_mex['daca_eligible'].mean()*100:.1f}%)")
print(f"  - post: {df_mex['post'].sum():,} post-period obs ({df_mex['post'].mean()*100:.1f}%)")
print(f"  - fulltime: {df_mex['fulltime'].mean()*100:.1f}% work full-time")
print(f"  - employed: {df_mex['employed'].mean()*100:.1f}% employed")
print()

# Create additional control variables
df_mex['female'] = (df_mex['SEX'] == 2).astype(int)
df_mex['married'] = (df_mex['MARST'] <= 2).astype(int)  # Married spouse present or absent
df_mex['age_sq'] = df_mex['AGE'] ** 2

# Education categories
df_mex['educ_lesshs'] = (df_mex['EDUC'] <= 5).astype(int)  # Less than HS
df_mex['educ_hs'] = (df_mex['EDUC'] == 6).astype(int)  # High school
df_mex['educ_somecol'] = ((df_mex['EDUC'] >= 7) & (df_mex['EDUC'] <= 9)).astype(int)  # Some college
df_mex['educ_college'] = (df_mex['EDUC'] >= 10).astype(int)  # College+

print(f"Control variables created.")
print()

# =============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# =============================================================================
print("Step 4: Descriptive Statistics")
print("-"*40)
print()

# Summary by eligibility and period
print("Table 1: Sample Composition")
print("-"*40)
print(df_mex.groupby(['daca_eligible', 'post']).size().unstack())
print()

# Means by group
print("Table 2: Mean Characteristics by DACA Eligibility")
print("-"*40)
summary_vars = ['AGE', 'female', 'married', 'educ_lesshs', 'educ_hs', 'educ_somecol',
                'educ_college', 'fulltime', 'employed', 'UHRSWORK']

summary_stats = df_mex.groupby('daca_eligible')[summary_vars].mean()
summary_stats.index = ['Not Eligible', 'Eligible']
print(summary_stats.round(3).T)
print()

# Pre-post means
print("Table 3: Full-time Employment Rates by Group and Period")
print("-"*40)
ft_means = df_mex.groupby(['daca_eligible', 'post'])['fulltime'].mean().unstack()
ft_means.index = ['Not Eligible', 'Eligible']
ft_means.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(ft_means.round(4))
print()

# Calculate raw DiD estimate
pre_eligible = df_mex[(df_mex['daca_eligible']==1) & (df_mex['post']==0)]['fulltime'].mean()
post_eligible = df_mex[(df_mex['daca_eligible']==1) & (df_mex['post']==1)]['fulltime'].mean()
pre_nonelig = df_mex[(df_mex['daca_eligible']==0) & (df_mex['post']==0)]['fulltime'].mean()
post_nonelig = df_mex[(df_mex['daca_eligible']==0) & (df_mex['post']==1)]['fulltime'].mean()

raw_did = (post_eligible - pre_eligible) - (post_nonelig - pre_nonelig)
print(f"Raw DiD Estimate: {raw_did:.4f}")
print(f"  Eligible change: {post_eligible - pre_eligible:.4f}")
print(f"  Non-eligible change: {post_nonelig - pre_nonelig:.4f}")
print()

# =============================================================================
# STEP 5: MAIN REGRESSION ANALYSIS
# =============================================================================
print("Step 5: Main Regression Analysis")
print("="*80)
print()

# Model 1: Basic DiD without controls
print("Model 1: Basic Difference-in-Differences (no controls)")
print("-"*40)
model1 = smf.ols('fulltime ~ daca_eligible + post + eligible_post', data=df_mex).fit(
    cov_type='cluster', cov_kwds={'groups': df_mex['STATEFIP']})
print(model1.summary().tables[1])
print()

# Model 2: DiD with demographic controls
print("Model 2: DiD with Demographic Controls")
print("-"*40)
model2 = smf.ols('fulltime ~ daca_eligible + post + eligible_post + AGE + age_sq + female + married',
                  data=df_mex).fit(cov_type='cluster', cov_kwds={'groups': df_mex['STATEFIP']})
print(model2.summary().tables[1])
print()

# Model 3: DiD with demographic + education controls
print("Model 3: DiD with Demographic + Education Controls")
print("-"*40)
model3 = smf.ols('fulltime ~ daca_eligible + post + eligible_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_mex).fit(cov_type='cluster', cov_kwds={'groups': df_mex['STATEFIP']})
print(model3.summary().tables[1])
print()

# Model 4: DiD with controls + state fixed effects
print("Model 4: DiD with Controls + State Fixed Effects")
print("-"*40)
model4 = smf.ols('fulltime ~ daca_eligible + post + eligible_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + C(STATEFIP)',
                  data=df_mex).fit(cov_type='cluster', cov_kwds={'groups': df_mex['STATEFIP']})
# Print only key coefficients
print("Key Coefficients:")
print(f"  eligible_post: {model4.params['eligible_post']:.4f} (SE: {model4.bse['eligible_post']:.4f})")
print(f"  daca_eligible: {model4.params['daca_eligible']:.4f} (SE: {model4.bse['daca_eligible']:.4f})")
print(f"  post:          {model4.params['post']:.4f} (SE: {model4.bse['post']:.4f})")
print()

# Model 5: DiD with controls + state and year fixed effects (PREFERRED SPECIFICATION)
print("Model 5: DiD with Controls + State & Year Fixed Effects (PREFERRED)")
print("-"*40)
model5 = smf.ols('fulltime ~ daca_eligible + eligible_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + C(STATEFIP) + C(YEAR)',
                  data=df_mex).fit(cov_type='cluster', cov_kwds={'groups': df_mex['STATEFIP']})
print("Key Coefficients:")
print(f"  eligible_post (DiD): {model5.params['eligible_post']:.4f} (SE: {model5.bse['eligible_post']:.4f})")
print(f"  daca_eligible:       {model5.params['daca_eligible']:.4f} (SE: {model5.bse['daca_eligible']:.4f})")
print(f"  N: {int(model5.nobs):,}")
print(f"  R-squared: {model5.rsquared:.4f}")
# Calculate p-value and CI
t_stat = model5.params['eligible_post'] / model5.bse['eligible_post']
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), model5.df_resid))
ci_low = model5.params['eligible_post'] - 1.96 * model5.bse['eligible_post']
ci_high = model5.params['eligible_post'] + 1.96 * model5.bse['eligible_post']
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print()

# =============================================================================
# STEP 6: ROBUSTNESS CHECKS
# =============================================================================
print("Step 6: Robustness Checks")
print("="*80)
print()

# Robustness 1: Using employment (any work) as outcome
print("Robustness 1: Employment (any work) as outcome")
print("-"*40)
model_emp = smf.ols('employed ~ daca_eligible + eligible_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + C(STATEFIP) + C(YEAR)',
                     data=df_mex).fit(cov_type='cluster', cov_kwds={'groups': df_mex['STATEFIP']})
print(f"  eligible_post (DiD): {model_emp.params['eligible_post']:.4f} (SE: {model_emp.bse['eligible_post']:.4f})")
print()

# Robustness 2: Restrict to labor force participants
print("Robustness 2: Conditional on labor force participation")
print("-"*40)
df_lf = df_mex[df_mex['LABFORCE'] == 2].copy()  # In labor force
model_lf = smf.ols('fulltime ~ daca_eligible + eligible_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + C(STATEFIP) + C(YEAR)',
                    data=df_lf).fit(cov_type='cluster', cov_kwds={'groups': df_lf['STATEFIP']})
print(f"  eligible_post (DiD): {model_lf.params['eligible_post']:.4f} (SE: {model_lf.bse['eligible_post']:.4f})")
print(f"  N: {int(model_lf.nobs):,}")
print()

# Robustness 3: Males only
print("Robustness 3: Males only")
print("-"*40)
df_male = df_mex[df_mex['female'] == 0].copy()
model_male = smf.ols('fulltime ~ daca_eligible + eligible_post + AGE + age_sq + married + educ_hs + educ_somecol + educ_college + C(STATEFIP) + C(YEAR)',
                      data=df_male).fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
print(f"  eligible_post (DiD): {model_male.params['eligible_post']:.4f} (SE: {model_male.bse['eligible_post']:.4f})")
print(f"  N: {int(model_male.nobs):,}")
print()

# Robustness 4: Females only
print("Robustness 4: Females only")
print("-"*40)
df_female = df_mex[df_mex['female'] == 1].copy()
model_female = smf.ols('fulltime ~ daca_eligible + eligible_post + AGE + age_sq + married + educ_hs + educ_somecol + educ_college + C(STATEFIP) + C(YEAR)',
                        data=df_female).fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})
print(f"  eligible_post (DiD): {model_female.params['eligible_post']:.4f} (SE: {model_female.bse['eligible_post']:.4f})")
print(f"  N: {int(model_female.nobs):,}")
print()

# Robustness 5: Alternative age restriction (16-35 to focus on young adults)
print("Robustness 5: Young adults (16-35)")
print("-"*40)
df_young = df_mex[(df_mex['AGE'] >= 16) & (df_mex['AGE'] <= 35)].copy()
model_young = smf.ols('fulltime ~ daca_eligible + eligible_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + C(STATEFIP) + C(YEAR)',
                       data=df_young).fit(cov_type='cluster', cov_kwds={'groups': df_young['STATEFIP']})
print(f"  eligible_post (DiD): {model_young.params['eligible_post']:.4f} (SE: {model_young.bse['eligible_post']:.4f})")
print(f"  N: {int(model_young.nobs):,}")
print()

# Robustness 6: Using person weights
print("Robustness 6: Weighted regression (using PERWT)")
print("-"*40)
model_wt = smf.wls('fulltime ~ daca_eligible + eligible_post + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + C(STATEFIP) + C(YEAR)',
                    data=df_mex, weights=df_mex['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_mex['STATEFIP']})
print(f"  eligible_post (DiD): {model_wt.params['eligible_post']:.4f} (SE: {model_wt.bse['eligible_post']:.4f})")
print()

# =============================================================================
# STEP 7: EVENT STUDY / PARALLEL TRENDS CHECK
# =============================================================================
print("Step 7: Event Study Analysis (Parallel Trends Check)")
print("="*80)
print()

# Create year dummies interacted with eligibility
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    df_mex[f'year_{year}'] = (df_mex['YEAR'] == year).astype(int)
    df_mex[f'elig_year_{year}'] = df_mex['daca_eligible'] * df_mex[f'year_{year}']

# Reference year: 2011 (last pre-treatment year)
event_formula = 'fulltime ~ daca_eligible + ' + ' + '.join([f'elig_year_{y}' for y in years if y != 2011]) + ' + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + C(STATEFIP) + C(YEAR)'
model_event = smf.ols(event_formula, data=df_mex).fit(cov_type='cluster', cov_kwds={'groups': df_mex['STATEFIP']})

print("Event Study Coefficients (relative to 2011):")
print("-"*40)
for year in years:
    if year != 2011:
        coef = model_event.params[f'elig_year_{year}']
        se = model_event.bse[f'elig_year_{year}']
        print(f"  {year}: {coef:.4f} (SE: {se:.4f})")
print()

# =============================================================================
# STEP 8: EXPORT RESULTS
# =============================================================================
print("Step 8: Exporting Results")
print("="*80)
print()

# Create results summary
results_summary = {
    'Model': ['Basic DiD', 'With Demographics', 'With Education', 'State FE', 'State+Year FE (Preferred)',
              'Employment Outcome', 'Labor Force Only', 'Males Only', 'Females Only', 'Young Adults', 'Weighted'],
    'DiD Coefficient': [model1.params['eligible_post'], model2.params['eligible_post'],
                        model3.params['eligible_post'], model4.params['eligible_post'],
                        model5.params['eligible_post'], model_emp.params['eligible_post'],
                        model_lf.params['eligible_post'], model_male.params['eligible_post'],
                        model_female.params['eligible_post'], model_young.params['eligible_post'],
                        model_wt.params['eligible_post']],
    'Std Error': [model1.bse['eligible_post'], model2.bse['eligible_post'],
                  model3.bse['eligible_post'], model4.bse['eligible_post'],
                  model5.bse['eligible_post'], model_emp.bse['eligible_post'],
                  model_lf.bse['eligible_post'], model_male.bse['eligible_post'],
                  model_female.bse['eligible_post'], model_young.bse['eligible_post'],
                  model_wt.bse['eligible_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs),
          int(model5.nobs), int(model_emp.nobs), int(model_lf.nobs), int(model_male.nobs),
          int(model_female.nobs), int(model_young.nobs), int(model_wt.nobs)]
}

results_df = pd.DataFrame(results_summary)
results_df['t-stat'] = results_df['DiD Coefficient'] / results_df['Std Error']
results_df['p-value'] = 2 * (1 - stats.t.cdf(abs(results_df['t-stat']), 50))  # Approx df
results_df['CI_low'] = results_df['DiD Coefficient'] - 1.96 * results_df['Std Error']
results_df['CI_high'] = results_df['DiD Coefficient'] + 1.96 * results_df['Std Error']

print("Summary of All Models:")
print(results_df.to_string(index=False))
print()

# Save results to CSV
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# =============================================================================
# STEP 9: FINAL SUMMARY
# =============================================================================
print()
print("="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print()
print("PREFERRED ESTIMATE (Model 5: State + Year Fixed Effects):")
print("-"*40)
print(f"Effect of DACA eligibility on full-time employment:")
print(f"  Coefficient: {model5.params['eligible_post']:.4f}")
print(f"  Standard Error: {model5.bse['eligible_post']:.4f}")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  p-value: {p_value:.4f}")
print(f"  Sample Size: {int(model5.nobs):,}")
print()
print("INTERPRETATION:")
if model5.params['eligible_post'] > 0:
    print(f"DACA eligibility is associated with a {model5.params['eligible_post']*100:.2f} percentage point")
    print("INCREASE in the probability of full-time employment.")
else:
    print(f"DACA eligibility is associated with a {abs(model5.params['eligible_post'])*100:.2f} percentage point")
    print("DECREASE in the probability of full-time employment.")
if p_value < 0.05:
    print("This effect is statistically significant at the 5% level.")
elif p_value < 0.10:
    print("This effect is statistically significant at the 10% level.")
else:
    print("This effect is NOT statistically significant at conventional levels.")
print()

# Save key statistics for the report
with open('key_results.txt', 'w') as f:
    f.write("DACA Replication Study - Key Results\n")
    f.write("="*50 + "\n\n")
    f.write("Preferred Estimate (Model 5):\n")
    f.write(f"  DiD Coefficient: {model5.params['eligible_post']:.4f}\n")
    f.write(f"  Standard Error: {model5.bse['eligible_post']:.4f}\n")
    f.write(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n")
    f.write(f"  t-statistic: {t_stat:.3f}\n")
    f.write(f"  p-value: {p_value:.4f}\n")
    f.write(f"  N: {int(model5.nobs):,}\n")
    f.write(f"  R-squared: {model5.rsquared:.4f}\n")
    f.write("\n")
    f.write("Sample Composition:\n")
    f.write(f"  Total observations (post-restrictions): {len(df_mex):,}\n")
    f.write(f"  DACA eligible: {df_mex['daca_eligible'].sum():,}\n")
    f.write(f"  Non-eligible: {(df_mex['daca_eligible']==0).sum():,}\n")
    f.write(f"  Pre-period obs: {(df_mex['post']==0).sum():,}\n")
    f.write(f"  Post-period obs: {(df_mex['post']==1).sum():,}\n")
    f.write("\n")
    f.write("Descriptive Statistics:\n")
    f.write(f"  Pre-period eligible FT rate: {pre_eligible:.4f}\n")
    f.write(f"  Post-period eligible FT rate: {post_eligible:.4f}\n")
    f.write(f"  Pre-period non-eligible FT rate: {pre_nonelig:.4f}\n")
    f.write(f"  Post-period non-eligible FT rate: {post_nonelig:.4f}\n")
    f.write(f"  Raw DiD: {raw_did:.4f}\n")

print("Key results saved to key_results.txt")
print()
print("Analysis complete!")
