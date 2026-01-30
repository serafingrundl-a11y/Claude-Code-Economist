"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US
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

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load the data
print("Loading data.csv...")
df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# STEP 2: SAMPLE SELECTION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: SAMPLE SELECTION")
print("="*80)

# Initial sample size
n_initial = len(df)
print(f"Initial sample size: {n_initial:,}")

# 2a. Restrict to Hispanic-Mexican ethnicity
# HISPAN == 1 indicates Mexican
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After restricting to Hispanic-Mexican (HISPAN==1): {len(df_sample):,}")

# 2b. Restrict to born in Mexico
# BPL == 200 indicates Mexico
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After restricting to born in Mexico (BPL==200): {len(df_sample):,}")

# 2c. Restrict to non-citizens (proxy for undocumented)
# CITIZEN == 3 indicates "Not a citizen"
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After restricting to non-citizens (CITIZEN==3): {len(df_sample):,}")

# 2d. Exclude 2012 (policy implemented mid-year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# 2e. Restrict to working-age population (16-64)
df_sample = df_sample[(df_sample['AGE'] >= 16) & (df_sample['AGE'] <= 64)].copy()
print(f"After restricting to ages 16-64: {len(df_sample):,}")

# ============================================================================
# STEP 3: VARIABLE CONSTRUCTION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: VARIABLE CONSTRUCTION")
print("="*80)

# 3a. Calculate age at arrival in US
# Age at arrival = YRIMMIG - BIRTHYR (approximate)
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
print(f"Age at arrival range: {df_sample['age_at_arrival'].min()} to {df_sample['age_at_arrival'].max()}")

# 3b. Create DACA eligibility indicator
# Criteria:
# 1. Arrived before age 16: age_at_arrival < 16
# 2. Under 31 as of June 15, 2012:
#    - Born 1982 or later, OR
#    - Born in 1981 Q3 or Q4 (July-Dec)
# 3. In US since 2007: YRIMMIG <= 2007
# 4. Non-citizen: Already filtered above

# Criterion 1: Arrived before 16
arrived_before_16 = df_sample['age_at_arrival'] < 16

# Criterion 2: Under 31 on June 15, 2012
# Born after June 15, 1981
under_31_june2012 = (df_sample['BIRTHYR'] >= 1982) | \
                    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))

# Criterion 3: In US since 2007
in_us_since_2007 = df_sample['YRIMMIG'] <= 2007

# Additional: Must have been in US by 2012 (present on June 15, 2012)
# This is implied by YRIMMIG <= 2007

# Combine all criteria for DACA eligibility
df_sample['daca_eligible'] = (arrived_before_16 & under_31_june2012 & in_us_since_2007).astype(int)

print(f"\nDACA Eligibility Criteria Applied:")
print(f"  - Arrived before age 16: {arrived_before_16.sum():,}")
print(f"  - Under 31 on June 2012: {under_31_june2012.sum():,}")
print(f"  - In US since 2007: {in_us_since_2007.sum():,}")
print(f"  - DACA eligible (all criteria): {df_sample['daca_eligible'].sum():,}")
print(f"  - Not DACA eligible: {(df_sample['daca_eligible']==0).sum():,}")

# 3c. Create post-DACA indicator
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
print(f"\nPost-DACA (2013-2016): {df_sample['post'].sum():,}")
print(f"Pre-DACA (2006-2011): {(df_sample['post']==0).sum():,}")

# 3d. Create interaction term
df_sample['daca_post'] = df_sample['daca_eligible'] * df_sample['post']

# 3e. Create full-time employment outcome
# Full-time = usually works 35+ hours per week
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)
print(f"\nFull-time employed (35+ hrs/week): {df_sample['fulltime'].sum():,}")
print(f"Not full-time: {(df_sample['fulltime']==0).sum():,}")

# 3f. Create any employment outcome (secondary)
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)
print(f"\nEmployed: {df_sample['employed'].sum():,}")

# 3g. Create control variables
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'] <= 2).astype(int)  # Married with spouse present or absent
df_sample['age_sq'] = df_sample['AGE'] ** 2

# Education categories
df_sample['educ_less_hs'] = (df_sample['EDUC'] < 6).astype(int)  # Less than high school
df_sample['educ_hs'] = (df_sample['EDUC'] == 6).astype(int)  # High school
df_sample['educ_some_college'] = ((df_sample['EDUC'] >= 7) & (df_sample['EDUC'] <= 9)).astype(int)
df_sample['educ_college_plus'] = (df_sample['EDUC'] >= 10).astype(int)  # Bachelor's or more

print("\nControl Variables Created:")
print(f"  Female: {df_sample['female'].mean():.3f}")
print(f"  Married: {df_sample['married'].mean():.3f}")
print(f"  Less than HS: {df_sample['educ_less_hs'].mean():.3f}")
print(f"  High school: {df_sample['educ_hs'].mean():.3f}")
print(f"  Some college: {df_sample['educ_some_college'].mean():.3f}")
print(f"  College+: {df_sample['educ_college_plus'].mean():.3f}")

# ============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: DESCRIPTIVE STATISTICS")
print("="*80)

# Summary by treatment status and period
print("\n--- Sample Size by Group ---")
print(df_sample.groupby(['daca_eligible', 'post']).size().unstack(fill_value=0))

# Mean outcomes by group
print("\n--- Mean Full-Time Employment Rate by Group ---")
ft_means = df_sample.groupby(['daca_eligible', 'post'])['fulltime'].mean().unstack()
ft_means.index = ['Not Eligible', 'DACA Eligible']
ft_means.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
print(ft_means.round(4))

# Calculate simple DiD
did_simple = (ft_means.iloc[1,1] - ft_means.iloc[1,0]) - (ft_means.iloc[0,1] - ft_means.iloc[0,0])
print(f"\nSimple DiD estimate: {did_simple:.4f}")

# Weighted means
print("\n--- Weighted Mean Full-Time Employment Rate by Group ---")
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

ft_weighted = df_sample.groupby(['daca_eligible', 'post']).apply(weighted_mean).unstack()
ft_weighted.index = ['Not Eligible', 'DACA Eligible']
ft_weighted.columns = ['Pre-DACA', 'Post-DACA']
print(ft_weighted.round(4))

did_weighted = (ft_weighted.iloc[1,1] - ft_weighted.iloc[1,0]) - (ft_weighted.iloc[0,1] - ft_weighted.iloc[0,0])
print(f"\nWeighted DiD estimate: {did_weighted:.4f}")

# Demographics by group
print("\n--- Demographics by DACA Eligibility ---")
demo_vars = ['AGE', 'female', 'married', 'educ_less_hs', 'educ_hs', 'educ_some_college', 'educ_college_plus']
demo_stats = df_sample.groupby('daca_eligible')[demo_vars].mean()
demo_stats.index = ['Not Eligible', 'DACA Eligible']
print(demo_stats.round(3))

# Time trends
print("\n--- Full-Time Employment Rate by Year and Eligibility ---")
yearly_ft = df_sample.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
yearly_ft.columns = ['Not Eligible', 'DACA Eligible']
print(yearly_ft.round(4))

# ============================================================================
# STEP 5: MAIN REGRESSION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: MAIN REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.ols('fulltime ~ daca_eligible + post + daca_post', data=df_sample).fit()
print(model1.summary().tables[1])

# Model 2: DiD with individual controls
print("\n--- Model 2: DiD with Individual Controls ---")
model2 = smf.ols('fulltime ~ daca_eligible + post + daca_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                 data=df_sample).fit()
print(model2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
model3 = smf.ols('fulltime ~ daca_eligible + daca_post + C(YEAR) + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                 data=df_sample).fit()
# Extract relevant coefficients
print(f"daca_eligible: {model3.params['daca_eligible']:.5f} (SE: {model3.bse['daca_eligible']:.5f})")
print(f"daca_post: {model3.params['daca_post']:.5f} (SE: {model3.bse['daca_post']:.5f})")

# Model 4: DiD with state and year fixed effects
print("\n--- Model 4: DiD with State and Year Fixed Effects ---")
model4 = smf.ols('fulltime ~ daca_eligible + daca_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                 data=df_sample).fit()
print(f"daca_eligible: {model4.params['daca_eligible']:.5f} (SE: {model4.bse['daca_eligible']:.5f})")
print(f"daca_post: {model4.params['daca_post']:.5f} (SE: {model4.bse['daca_post']:.5f})")

# Model 5: With clustered standard errors at state level
print("\n--- Model 5: DiD with Clustered SEs (by State) ---")
model5 = smf.ols('fulltime ~ daca_eligible + daca_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                 data=df_sample).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"daca_eligible: {model5.params['daca_eligible']:.5f} (SE: {model5.bse['daca_eligible']:.5f})")
print(f"daca_post: {model5.params['daca_post']:.5f} (SE: {model5.bse['daca_post']:.5f})")
print(f"p-value for daca_post: {model5.pvalues['daca_post']:.5f}")
print(f"95% CI: [{model5.conf_int().loc['daca_post', 0]:.5f}, {model5.conf_int().loc['daca_post', 1]:.5f}]")

# ============================================================================
# STEP 6: WEIGHTED REGRESSION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: WEIGHTED REGRESSION (PREFERRED SPECIFICATION)")
print("="*80)

# Weighted regression with person weights
from statsmodels.regression.linear_model import WLS

# Create design matrices
formula = 'fulltime ~ daca_eligible + daca_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus'
y, X = smf.ols(formula, data=df_sample).endog, smf.ols(formula, data=df_sample).exog

# Get variable names
model_temp = smf.ols(formula, data=df_sample)
X_df = pd.DataFrame(X, columns=model_temp.exog_names)

# Run weighted regression
model6_wls = sm.WLS(y, X, weights=df_sample['PERWT'].values).fit()

# Get clustered standard errors for weighted model
# Using OLS with frequency weights approximation
print("\n--- Model 6: Weighted DiD with State and Year FEs ---")
model6 = smf.wls('fulltime ~ daca_eligible + daca_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                 data=df_sample, weights=df_sample['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

print(f"daca_eligible: {model6.params['daca_eligible']:.5f} (SE: {model6.bse['daca_eligible']:.5f})")
print(f"daca_post (DiD estimate): {model6.params['daca_post']:.5f} (SE: {model6.bse['daca_post']:.5f})")
print(f"t-statistic: {model6.tvalues['daca_post']:.3f}")
print(f"p-value: {model6.pvalues['daca_post']:.5f}")
print(f"95% CI: [{model6.conf_int().loc['daca_post', 0]:.5f}, {model6.conf_int().loc['daca_post', 1]:.5f}]")
print(f"N: {model6.nobs:.0f}")
print(f"R-squared: {model6.rsquared:.4f}")

# ============================================================================
# STEP 7: ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: ROBUSTNESS CHECKS")
print("="*80)

# 7a. Alternative age restrictions (18-35)
print("\n--- Robustness 1: Restricted Age (18-35) ---")
df_robust1 = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 35)].copy()
model_r1 = smf.wls('fulltime ~ daca_eligible + daca_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                   data=df_robust1, weights=df_robust1['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_robust1['STATEFIP']})
print(f"DiD estimate: {model_r1.params['daca_post']:.5f} (SE: {model_r1.bse['daca_post']:.5f})")
print(f"N: {model_r1.nobs:.0f}")

# 7b. Men only
print("\n--- Robustness 2: Men Only ---")
df_robust2 = df_sample[df_sample['female'] == 0].copy()
model_r2 = smf.wls('fulltime ~ daca_eligible + daca_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + married + educ_hs + educ_some_college + educ_college_plus',
                   data=df_robust2, weights=df_robust2['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_robust2['STATEFIP']})
print(f"DiD estimate: {model_r2.params['daca_post']:.5f} (SE: {model_r2.bse['daca_post']:.5f})")
print(f"N: {model_r2.nobs:.0f}")

# 7c. Women only
print("\n--- Robustness 3: Women Only ---")
df_robust3 = df_sample[df_sample['female'] == 1].copy()
model_r3 = smf.wls('fulltime ~ daca_eligible + daca_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + married + educ_hs + educ_some_college + educ_college_plus',
                   data=df_robust3, weights=df_robust3['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_robust3['STATEFIP']})
print(f"DiD estimate: {model_r3.params['daca_post']:.5f} (SE: {model_r3.bse['daca_post']:.5f})")
print(f"N: {model_r3.nobs:.0f}")

# 7d. Any employment as outcome
print("\n--- Robustness 4: Any Employment as Outcome ---")
model_r4 = smf.wls('employed ~ daca_eligible + daca_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                   data=df_sample, weights=df_sample['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"DiD estimate: {model_r4.params['daca_post']:.5f} (SE: {model_r4.bse['daca_post']:.5f})")
print(f"N: {model_r4.nobs:.0f}")

# 7e. Placebo test - using 2009 as fake treatment year
print("\n--- Robustness 5: Placebo Test (2009 as Fake Treatment) ---")
df_placebo = df_sample[(df_sample['YEAR'] <= 2011)].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['daca_post_placebo'] = df_placebo['daca_eligible'] * df_placebo['post_placebo']
model_r5 = smf.wls('fulltime ~ daca_eligible + daca_post_placebo + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                   data=df_placebo, weights=df_placebo['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_placebo['STATEFIP']})
print(f"Placebo DiD estimate: {model_r5.params['daca_post_placebo']:.5f} (SE: {model_r5.bse['daca_post_placebo']:.5f})")
print(f"N: {model_r5.nobs:.0f}")

# ============================================================================
# STEP 8: EVENT STUDY
# ============================================================================
print("\n" + "="*80)
print("STEP 8: EVENT STUDY ANALYSIS")
print("="*80)

# Create year-specific treatment effects (relative to 2011)
for year in df_sample['YEAR'].unique():
    df_sample[f'daca_year_{year}'] = ((df_sample['YEAR'] == year) & (df_sample['daca_eligible'] == 1)).astype(int)

# Drop 2011 as reference year
year_vars = [f'daca_year_{y}' for y in sorted(df_sample['YEAR'].unique()) if y != 2011]
formula_es = 'fulltime ~ daca_eligible + ' + ' + '.join(year_vars) + ' + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus'

model_es = smf.wls(formula_es, data=df_sample, weights=df_sample['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

print("\nEvent Study Coefficients (Reference Year: 2011):")
print("-" * 50)
for year in sorted(df_sample['YEAR'].unique()):
    if year != 2011:
        var_name = f'daca_year_{year}'
        coef = model_es.params[var_name]
        se = model_es.bse[var_name]
        print(f"Year {year}: {coef:.5f} (SE: {se:.5f})")

# ============================================================================
# STEP 9: SUMMARY OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 9: SUMMARY OF KEY RESULTS")
print("="*80)

print("\n" + "="*50)
print("PREFERRED SPECIFICATION (Model 6)")
print("="*50)
print(f"Outcome: Full-time employment (35+ hours/week)")
print(f"Sample: Hispanic-Mexican, born in Mexico, non-citizens, ages 16-64")
print(f"Method: Difference-in-Differences with state and year fixed effects")
print(f"Weights: Person weights (PERWT)")
print(f"Standard Errors: Clustered by state")
print(f"\nDiD Estimate (DACA effect): {model6.params['daca_post']:.5f}")
print(f"Standard Error: {model6.bse['daca_post']:.5f}")
print(f"t-statistic: {model6.tvalues['daca_post']:.3f}")
print(f"p-value: {model6.pvalues['daca_post']:.5f}")
print(f"95% Confidence Interval: [{model6.conf_int().loc['daca_post', 0]:.5f}, {model6.conf_int().loc['daca_post', 1]:.5f}]")
print(f"Sample Size: {model6.nobs:.0f}")
print(f"R-squared: {model6.rsquared:.4f}")

# Interpret
effect_pct = model6.params['daca_post'] * 100
print(f"\nInterpretation: DACA eligibility is associated with a {effect_pct:.2f} percentage point")
if model6.params['daca_post'] > 0:
    print("increase in the probability of full-time employment.")
else:
    print("decrease in the probability of full-time employment.")

if model6.pvalues['daca_post'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
elif model6.pvalues['daca_post'] < 0.10:
    print("This effect is statistically significant at the 10% level.")
else:
    print("This effect is NOT statistically significant at conventional levels.")

# ============================================================================
# STEP 10: SAVE RESULTS FOR REPORT
# ============================================================================
print("\n" + "="*80)
print("STEP 10: SAVING RESULTS")
print("="*80)

# Save key results to a file
results = {
    'preferred_estimate': model6.params['daca_post'],
    'standard_error': model6.bse['daca_post'],
    'ci_lower': model6.conf_int().loc['daca_post', 0],
    'ci_upper': model6.conf_int().loc['daca_post', 1],
    'p_value': model6.pvalues['daca_post'],
    't_stat': model6.tvalues['daca_post'],
    'n_obs': int(model6.nobs),
    'r_squared': model6.rsquared
}

# Save yearly means for plotting
yearly_means = df_sample.groupby(['YEAR', 'daca_eligible']).agg({
    'fulltime': 'mean',
    'PERWT': 'sum'
}).reset_index()
yearly_means.to_csv('yearly_means.csv', index=False)
print("Saved yearly_means.csv")

# Save event study coefficients
es_results = []
for year in sorted(df_sample['YEAR'].unique()):
    if year != 2011:
        var_name = f'daca_year_{year}'
        es_results.append({
            'year': year,
            'coefficient': model_es.params[var_name],
            'se': model_es.bse[var_name],
            'ci_lower': model_es.conf_int().loc[var_name, 0],
            'ci_upper': model_es.conf_int().loc[var_name, 1]
        })
    else:
        es_results.append({
            'year': year,
            'coefficient': 0,
            'se': 0,
            'ci_lower': 0,
            'ci_upper': 0
        })

es_df = pd.DataFrame(es_results)
es_df.to_csv('event_study_results.csv', index=False)
print("Saved event_study_results.csv")

# Summary statistics
summary_stats = df_sample.groupby('daca_eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_less_hs': 'mean',
    'educ_hs': 'mean',
    'educ_some_college': 'mean',
    'educ_college_plus': 'mean',
    'fulltime': 'mean',
    'employed': 'mean',
    'PERWT': 'count'
}).round(4)
summary_stats.columns = ['Mean Age', 'Female', 'Married', 'Less than HS', 'High School',
                         'Some College', 'College+', 'Full-time', 'Employed', 'N']
summary_stats.to_csv('summary_statistics.csv')
print("Saved summary_statistics.csv")

# Model comparison table
model_comparison = pd.DataFrame({
    'Model': ['Basic DiD', 'With Controls', 'Year FE', 'State+Year FE (OLS)', 'State+Year FE (Clustered)', 'Weighted (Preferred)'],
    'DiD Estimate': [model1.params['daca_post'], model2.params['daca_post'], model3.params['daca_post'],
                     model4.params['daca_post'], model5.params['daca_post'], model6.params['daca_post']],
    'SE': [model1.bse['daca_post'], model2.bse['daca_post'], model3.bse['daca_post'],
           model4.bse['daca_post'], model5.bse['daca_post'], model6.bse['daca_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs), int(model6.nobs)]
})
model_comparison.to_csv('model_comparison.csv', index=False)
print("Saved model_comparison.csv")

# Robustness results
robustness = pd.DataFrame({
    'Specification': ['Age 18-35', 'Men Only', 'Women Only', 'Any Employment', 'Placebo (2009)'],
    'DiD Estimate': [model_r1.params['daca_post'], model_r2.params['daca_post'], model_r3.params['daca_post'],
                     model_r4.params['daca_post'], model_r5.params['daca_post_placebo']],
    'SE': [model_r1.bse['daca_post'], model_r2.bse['daca_post'], model_r3.bse['daca_post'],
           model_r4.bse['daca_post'], model_r5.bse['daca_post_placebo']],
    'N': [int(model_r1.nobs), int(model_r2.nobs), int(model_r3.nobs),
          int(model_r4.nobs), int(model_r5.nobs)]
})
robustness.to_csv('robustness_results.csv', index=False)
print("Saved robustness_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
