"""
DACA Replication Study - Analysis Script
Participant 91

Research Question: Causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US.

Design: Difference-in-Differences
- Treatment: Ages 26-30 as of June 15, 2012
- Control: Ages 31-35 as of June 15, 2012
- Pre-period: 2006-2011
- Post-period: 2013-2016
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
# STEP 1: Load Data - Only necessary columns to save memory
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Only load columns we need
cols_to_use = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
               'EMPSTAT', 'UHRSWORK']

# Use low_memory=False and specify dtypes
dtypes = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8'
}

print("Loading data with memory optimization...")
df = pd.read_csv('data/data.csv', usecols=cols_to_use, dtype=dtypes)
print(f"Total observations in raw data: {len(df):,}")
print(f"Years covered: {df['YEAR'].min()} - {df['YEAR'].max()}")

# =============================================================================
# STEP 2: Sample Selection
# =============================================================================
print("\n" + "="*80)
print("STEP 2: SAMPLE SELECTION")
print("="*80)

# Initial count
n_initial = len(df)
print(f"Initial observations: {n_initial:,}")

# Step 2a: Keep only Hispanic-Mexican (HISPAN == 1)
df = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican restriction (HISPAN==1): {len(df):,}")

# Step 2b: Keep only born in Mexico (BPL == 200)
df = df[df['BPL'] == 200].copy()
print(f"After Mexico birthplace restriction (BPL==200): {len(df):,}")

# Step 2c: Keep only non-citizens
# CITIZEN: 3 = Not a citizen, 4 = Has first papers (not naturalized)
df = df[df['CITIZEN'].isin([3, 4])].copy()
print(f"After non-citizen restriction (CITIZEN in [3,4]): {len(df):,}")

# Step 2d: Calculate age as of June 15, 2012
df['age_2012'] = 2012 - df['BIRTHYR']

# Step 2e: Keep only those who immigrated before age 16
# Age at immigration = YRIMMIG - BIRTHYR
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['YRIMMIG'] > 0].copy()  # Filter out missing immigration year
df = df[df['age_at_immig'] < 16].copy()
print(f"After arrived before age 16 restriction: {len(df):,}")

# Step 2f: Keep only those present since June 2007 (YRIMMIG <= 2007)
df = df[df['YRIMMIG'] <= 2007].copy()
print(f"After continuous presence since 2007 (YRIMMIG<=2007): {len(df):,}")

# Step 2g: Define treatment and control groups based on age in 2012
# Treatment: 26-30 in 2012 (born 1982-1986)
# Control: 31-35 in 2012 (born 1977-1981)
df['treatment_group'] = df['age_2012'].between(26, 30).astype('int8')
df['control_group'] = df['age_2012'].between(31, 35).astype('int8')

# Keep only treatment or control group members
df_analysis = df[(df['treatment_group'] == 1) | (df['control_group'] == 1)].copy()
print(f"After age group restriction (26-30 or 31-35 in 2012): {len(df_analysis):,}")

# Step 2h: Exclude 2012 (policy implementation year)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

# Clear memory
del df
import gc
gc.collect()

# =============================================================================
# STEP 3: Create Analysis Variables
# =============================================================================
print("\n" + "="*80)
print("STEP 3: CREATE ANALYSIS VARIABLES")
print("="*80)

# Treatment indicator (treated = age 26-30 in 2012)
df_analysis['treated'] = df_analysis['treatment_group'].astype('int8')

# Post-policy indicator
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype('int8')

# Interaction term (DiD estimator)
df_analysis['treated_post'] = (df_analysis['treated'] * df_analysis['post']).astype('int8')

# Outcome: Full-time employment (35+ hours/week)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype('int8')

# Covariates
df_analysis['female'] = (df_analysis['SEX'] == 2).astype('int8')
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype('int8')

# Education categories
df_analysis['educ_lesshs'] = (df_analysis['EDUC'] < 6).astype('int8')
df_analysis['educ_hs'] = (df_analysis['EDUC'] == 6).astype('int8')
df_analysis['educ_somecol'] = (df_analysis['EDUC'].isin([7, 8, 9])).astype('int8')
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype('int8')

print(f"\nFinal analysis sample size: {len(df_analysis):,}")
print(f"Treatment group (ages 26-30 in 2012): {df_analysis['treated'].sum():,}")
print(f"Control group (ages 31-35 in 2012): {(1-df_analysis['treated']).sum():,}")

# =============================================================================
# STEP 4: Descriptive Statistics
# =============================================================================
print("\n" + "="*80)
print("STEP 4: DESCRIPTIVE STATISTICS")
print("="*80)

# Summary by group and period
print("\nMean Full-Time Employment Rate by Group and Period:")
summary = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
summary.columns = ['Mean FT', 'Std Dev', 'N', 'Sum Weights']
print(summary)

# Weighted means
print("\nWeighted Mean Full-Time Employment Rate by Group and Period:")
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

weighted_summary = df_analysis.groupby(['treated', 'post']).apply(weighted_mean)
print(weighted_summary)

# Simple DiD calculation
pre_treat = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==0)]['fulltime'].mean()
post_treat = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==1)]['fulltime'].mean()
pre_control = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==0)]['fulltime'].mean()
post_control = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==1)]['fulltime'].mean()

did_simple = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\nSimple DiD Calculation (Unweighted):")
print(f"Treatment group pre-period mean: {pre_treat:.4f}")
print(f"Treatment group post-period mean: {post_treat:.4f}")
print(f"Control group pre-period mean: {pre_control:.4f}")
print(f"Control group post-period mean: {post_control:.4f}")
print(f"DiD estimate: ({post_treat:.4f} - {pre_treat:.4f}) - ({post_control:.4f} - {pre_control:.4f}) = {did_simple:.4f}")

# =============================================================================
# STEP 5: Main Regression Analysis
# =============================================================================
print("\n" + "="*80)
print("STEP 5: MAIN REGRESSION ANALYSIS")
print("="*80)

# Convert to float for regression
df_analysis['fulltime'] = df_analysis['fulltime'].astype(float)
df_analysis['treated'] = df_analysis['treated'].astype(float)
df_analysis['post'] = df_analysis['post'].astype(float)
df_analysis['treated_post'] = df_analysis['treated_post'].astype(float)
df_analysis['PERWT'] = df_analysis['PERWT'].astype(float)

# Model 1: Basic DiD (no controls, no weights)
print("\n--- Model 1: Basic DiD (OLS, no weights) ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_analysis).fit(cov_type='HC1')
print(f"Intercept: {model1.params['Intercept']:.6f}")
print(f"treated: {model1.params['treated']:.6f} (SE: {model1.bse['treated']:.6f})")
print(f"post: {model1.params['post']:.6f} (SE: {model1.bse['post']:.6f})")
print(f"treated_post: {model1.params['treated_post']:.6f} (SE: {model1.bse['treated_post']:.6f})")
print(f"95% CI for treated_post: [{model1.conf_int().loc['treated_post', 0]:.6f}, {model1.conf_int().loc['treated_post', 1]:.6f}]")
print(f"p-value: {model1.pvalues['treated_post']:.4f}")
print(f"R-squared: {model1.rsquared:.4f}")
print(f"N: {int(model1.nobs):,}")

# Model 2: Basic DiD with survey weights
print("\n--- Model 2: Basic DiD (WLS with PERWT) ---")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treated_post: {model2.params['treated_post']:.6f} (SE: {model2.bse['treated_post']:.6f})")
print(f"95% CI for treated_post: [{model2.conf_int().loc['treated_post', 0]:.6f}, {model2.conf_int().loc['treated_post', 1]:.6f}]")
print(f"p-value: {model2.pvalues['treated_post']:.4f}")
print(f"R-squared: {model2.rsquared:.4f}")
print(f"N: {int(model2.nobs):,}")

# Model 3: DiD with year fixed effects (weighted)
print("\n--- Model 3: DiD with Year Fixed Effects (WLS) ---")
model3 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treated_post: {model3.params['treated_post']:.6f} (SE: {model3.bse['treated_post']:.6f})")
print(f"95% CI: [{model3.conf_int().loc['treated_post', 0]:.6f}, {model3.conf_int().loc['treated_post', 1]:.6f}]")
print(f"p-value: {model3.pvalues['treated_post']:.4f}")
print(f"R-squared: {model3.rsquared:.4f}")
print(f"N: {int(model3.nobs):,}")

# Model 4: DiD with year fixed effects and covariates (weighted)
print("\n--- Model 4: DiD with Year FE and Covariates (WLS) ---")
model4 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treated_post: {model4.params['treated_post']:.6f} (SE: {model4.bse['treated_post']:.6f})")
print(f"95% CI: [{model4.conf_int().loc['treated_post', 0]:.6f}, {model4.conf_int().loc['treated_post', 1]:.6f}]")
print(f"p-value: {model4.pvalues['treated_post']:.4f}")
print(f"female: {model4.params['female']:.6f}")
print(f"married: {model4.params['married']:.6f}")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {int(model4.nobs):,}")

# Model 5: DiD with year and state fixed effects (weighted)
print("\n--- Model 5: DiD with Year and State Fixed Effects (WLS) ---")
model5 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treated_post coefficient: {model5.params['treated_post']:.6f}")
print(f"treated_post std error: {model5.bse['treated_post']:.6f}")
print(f"95% CI: [{model5.conf_int().loc['treated_post', 0]:.6f}, {model5.conf_int().loc['treated_post', 1]:.6f}]")
print(f"treated_post p-value: {model5.pvalues['treated_post']:.4f}")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {int(model5.nobs):,}")

# Model 6: Full specification with covariates and fixed effects
print("\n--- Model 6: Full Specification (Year FE + State FE + Covariates, WLS) ---")
model6 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treated_post coefficient: {model6.params['treated_post']:.6f}")
print(f"treated_post std error: {model6.bse['treated_post']:.6f}")
print(f"95% CI: [{model6.conf_int().loc['treated_post', 0]:.6f}, {model6.conf_int().loc['treated_post', 1]:.6f}]")
print(f"treated_post p-value: {model6.pvalues['treated_post']:.4f}")
print(f"R-squared: {model6.rsquared:.4f}")
print(f"N: {int(model6.nobs):,}")

# =============================================================================
# STEP 6: Event Study for Parallel Trends Check
# =============================================================================
print("\n" + "="*80)
print("STEP 6: EVENT STUDY ANALYSIS")
print("="*80)

# Create treatment x year interactions (omitting 2011 as reference)
years = sorted(df_analysis['YEAR'].unique())
ref_year = 2011

for year in years:
    df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype(float)
    if year != ref_year:
        df_analysis[f'treated_x_{year}'] = df_analysis['treated'] * df_analysis[f'year_{year}']

# Event study regression
interaction_terms = ' + '.join([f'treated_x_{y}' for y in years if y != ref_year])
event_formula = f'fulltime ~ treated + C(YEAR) + {interaction_terms}'

model_event = smf.wls(event_formula,
                       data=df_analysis,
                       weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Reference Year: 2011):")
print("-" * 60)
event_results = []
for year in years:
    if year != ref_year:
        coef = model_event.params[f'treated_x_{year}']
        se = model_event.bse[f'treated_x_{year}']
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        pval = model_event.pvalues[f'treated_x_{year}']
        event_results.append({
            'Year': year,
            'Coefficient': coef,
            'Std Error': se,
            'CI Lower': ci_low,
            'CI Upper': ci_high,
            'P-value': pval
        })
        print(f"Year {year}: Coef = {coef:.6f}, SE = {se:.6f}, 95% CI = [{ci_low:.6f}, {ci_high:.6f}], p = {pval:.4f}")

event_df = pd.DataFrame(event_results)

# =============================================================================
# STEP 7: Robustness Checks
# =============================================================================
print("\n" + "="*80)
print("STEP 7: ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Placebo test using years 2006-2008 vs 2009-2011
print("\n--- Robustness 1: Placebo Test (Pre-DACA: 2006-2008 vs 2009-2011) ---")
df_placebo = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(float)
df_placebo['treated_post_placebo'] = df_placebo['treated'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treated + C(YEAR) + treated_post_placebo',
                         data=df_placebo,
                         weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"treated_post_placebo coefficient: {model_placebo.params['treated_post_placebo']:.6f}")
print(f"treated_post_placebo std error: {model_placebo.bse['treated_post_placebo']:.6f}")
print(f"95% CI: [{model_placebo.conf_int().loc['treated_post_placebo', 0]:.6f}, {model_placebo.conf_int().loc['treated_post_placebo', 1]:.6f}]")
print(f"treated_post_placebo p-value: {model_placebo.pvalues['treated_post_placebo']:.4f}")
print(f"N: {int(model_placebo.nobs):,}")

# Robustness 2: By gender
print("\n--- Robustness 2: Subgroup Analysis by Gender ---")
# Men
df_men = df_analysis[df_analysis['female'] == 0].copy()
model_men = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                     data=df_men,
                     weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"Men - treated_post coefficient: {model_men.params['treated_post']:.6f}")
print(f"Men - SE: {model_men.bse['treated_post']:.6f}")
print(f"Men - 95% CI: [{model_men.conf_int().loc['treated_post', 0]:.6f}, {model_men.conf_int().loc['treated_post', 1]:.6f}]")
print(f"Men - N: {int(model_men.nobs):,}")

# Women
df_women = df_analysis[df_analysis['female'] == 1].copy()
model_women = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                       data=df_women,
                       weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"Women - treated_post coefficient: {model_women.params['treated_post']:.6f}")
print(f"Women - SE: {model_women.bse['treated_post']:.6f}")
print(f"Women - 95% CI: [{model_women.conf_int().loc['treated_post', 0]:.6f}, {model_women.conf_int().loc['treated_post', 1]:.6f}]")
print(f"Women - N: {int(model_women.nobs):,}")

# Robustness 3: By marital status
print("\n--- Robustness 3: Subgroup Analysis by Marital Status ---")
# Married
df_married = df_analysis[df_analysis['married'] == 1].copy()
model_married = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                         data=df_married,
                         weights=df_married['PERWT']).fit(cov_type='HC1')
print(f"Married - treated_post coefficient: {model_married.params['treated_post']:.6f}")
print(f"Married - SE: {model_married.bse['treated_post']:.6f}")
print(f"Married - N: {int(model_married.nobs):,}")

# Not married
df_notmarried = df_analysis[df_analysis['married'] == 0].copy()
model_notmarried = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                            data=df_notmarried,
                            weights=df_notmarried['PERWT']).fit(cov_type='HC1')
print(f"Not married - treated_post coefficient: {model_notmarried.params['treated_post']:.6f}")
print(f"Not married - SE: {model_notmarried.bse['treated_post']:.6f}")
print(f"Not married - N: {int(model_notmarried.nobs):,}")

# =============================================================================
# STEP 8: Summary of Results
# =============================================================================
print("\n" + "="*80)
print("STEP 8: SUMMARY OF RESULTS")
print("="*80)

print("\n" + "="*60)
print("PREFERRED ESTIMATE (Model 4 - Year FE + Covariates)")
print("="*60)
preferred_coef = model4.params['treated_post']
preferred_se = model4.bse['treated_post']
preferred_ci = model4.conf_int().loc['treated_post']
preferred_pval = model4.pvalues['treated_post']
preferred_n = int(model4.nobs)

print(f"Effect Size (DiD Estimate): {preferred_coef:.6f}")
print(f"Standard Error: {preferred_se:.6f}")
print(f"95% Confidence Interval: [{preferred_ci[0]:.6f}, {preferred_ci[1]:.6f}]")
print(f"P-value: {preferred_pval:.4f}")
print(f"Sample Size: {preferred_n:,}")

# Convert to percentage points
print(f"\nInterpretation: DACA eligibility is associated with a {preferred_coef*100:.2f} percentage point")
print(f"{'increase' if preferred_coef > 0 else 'decrease'} in the probability of full-time employment.")
if preferred_pval < 0.05:
    print("This effect is statistically significant at the 5% level.")
else:
    print("This effect is NOT statistically significant at the 5% level.")

# =============================================================================
# STEP 9: Save Results for Report
# =============================================================================
print("\n" + "="*80)
print("STEP 9: SAVING RESULTS")
print("="*80)

# Create results summary dictionary
results_summary = {
    'Model': ['Basic DiD (OLS)', 'Basic DiD (WLS)', 'Year FE', 'Year FE + Covariates',
              'Year + State FE', 'Full Specification'],
    'Coefficient': [model1.params['treated_post'], model2.params['treated_post'],
                    model3.params['treated_post'], model4.params['treated_post'],
                    model5.params['treated_post'], model6.params['treated_post']],
    'Std_Error': [model1.bse['treated_post'], model2.bse['treated_post'],
                  model3.bse['treated_post'], model4.bse['treated_post'],
                  model5.bse['treated_post'], model6.bse['treated_post']],
    'CI_Lower': [model1.conf_int().loc['treated_post', 0], model2.conf_int().loc['treated_post', 0],
                 model3.conf_int().loc['treated_post', 0], model4.conf_int().loc['treated_post', 0],
                 model5.conf_int().loc['treated_post', 0], model6.conf_int().loc['treated_post', 0]],
    'CI_Upper': [model1.conf_int().loc['treated_post', 1], model2.conf_int().loc['treated_post', 1],
                 model3.conf_int().loc['treated_post', 1], model4.conf_int().loc['treated_post', 1],
                 model5.conf_int().loc['treated_post', 1], model6.conf_int().loc['treated_post', 1]],
    'P_value': [model1.pvalues['treated_post'], model2.pvalues['treated_post'],
                model3.pvalues['treated_post'], model4.pvalues['treated_post'],
                model5.pvalues['treated_post'], model6.pvalues['treated_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs), int(model6.nobs)],
    'R_squared': [model1.rsquared, model2.rsquared, model3.rsquared,
                  model4.rsquared, model5.rsquared, model6.rsquared]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results_summary_91.csv', index=False)
print("Results summary saved to results_summary_91.csv")

# Save event study results
event_df.to_csv('event_study_91.csv', index=False)
print("Event study results saved to event_study_91.csv")

# Save descriptive statistics
desc_stats = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'AGE': 'mean',
    'educ_college': 'mean',
    'PERWT': 'sum'
}).round(4)
desc_stats.to_csv('descriptive_stats_91.csv')
print("Descriptive statistics saved to descriptive_stats_91.csv")

# Additional summary for subgroups
subgroup_results = {
    'Subgroup': ['Men', 'Women', 'Married', 'Not Married'],
    'Coefficient': [model_men.params['treated_post'], model_women.params['treated_post'],
                    model_married.params['treated_post'], model_notmarried.params['treated_post']],
    'Std_Error': [model_men.bse['treated_post'], model_women.bse['treated_post'],
                  model_married.bse['treated_post'], model_notmarried.bse['treated_post']],
    'N': [int(model_men.nobs), int(model_women.nobs),
          int(model_married.nobs), int(model_notmarried.nobs)]
}
subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv('subgroup_results_91.csv', index=False)
print("Subgroup results saved to subgroup_results_91.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
