"""
DACA Replication Analysis
=========================
This script analyzes the effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals using a difference-in-differences design.

Research Question:
What was the causal impact of DACA eligibility on the probability of full-time employment?

Treatment Group: Ages 26-30 as of June 15, 2012 (born 1982-1986)
Control Group: Ages 31-35 as of June 15, 2012 (born 1977-1981)
Outcome: Full-time employment (usually working 35+ hours per week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os
import gc
warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r'C:\Users\seraf\DACA Results Task 2\replication_13')

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD DATA (using chunked reading due to large file size)
# =============================================================================
print("\n[1] Loading data with filtering...")

# Define the columns we need
needed_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
               'EDUC', 'UHRSWORK']

# Read in chunks and filter as we go
chunks = []
chunk_size = 500000

for chunk in pd.read_csv('data/data.csv', usecols=needed_cols, chunksize=chunk_size):
    # Apply filters immediately to reduce memory
    # Year filter: 2006-2011 and 2013-2016 (excluding 2012)
    chunk = chunk[chunk['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])]

    # Hispanic-Mexican ethnicity
    chunk = chunk[chunk['HISPAN'] == 1]

    # Born in Mexico
    chunk = chunk[chunk['BPL'] == 200]

    # Non-citizen
    chunk = chunk[chunk['CITIZEN'] == 3]

    # Birth years for treatment/control (1977-1986)
    chunk = chunk[chunk['BIRTHYR'].isin(range(1977, 1987))]

    if len(chunk) > 0:
        chunks.append(chunk)

    gc.collect()

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"   Observations after initial filtering: {len(df):,}")

# =============================================================================
# 2. APPLY ADDITIONAL DACA ELIGIBILITY CRITERIA
# =============================================================================
print("\n[2] Applying additional DACA eligibility criteria...")

# Must have arrived before age 16
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_immigration'] < 16]
print(f"   After filtering arrived before age 16: {len(df):,}")

# Must have lived continuously in the US since June 15, 2007
df = df[df['YRIMMIG'] <= 2007]
print(f"   After filtering arrived by 2007: {len(df):,}")

# =============================================================================
# 3. DEFINE TREATMENT AND CONTROL GROUPS
# =============================================================================
print("\n[3] Defining treatment and control groups...")

# Treatment: Ages 26-30 as of June 15, 2012 -> born 1982-1986
# Control: Ages 31-35 as of June 15, 2012 -> born 1977-1981
treatment_births = [1982, 1983, 1984, 1985, 1986]
control_births = [1977, 1978, 1979, 1980, 1981]

df['treated'] = df['BIRTHYR'].isin(treatment_births).astype(int)

print(f"   Treatment group (born 1982-1986): {df['treated'].sum():,}")
print(f"   Control group (born 1977-1981): {(1-df['treated']).sum():,}")

# =============================================================================
# 4. CREATE OUTCOME AND CONTROL VARIABLES
# =============================================================================
print("\n[4] Creating outcome and control variables...")

# Full-time employment: usually working 35+ hours per week
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
print(f"   Full-time employment rate: {df['fulltime'].mean()*100:.2f}%")

# Post period indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"   Pre-period observations (2006-2011): {(1-df['post']).sum():,}")
print(f"   Post-period observations (2013-2016): {df['post'].sum():,}")

# DiD interaction
df['treated_post'] = df['treated'] * df['post']

# Control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = df['MARST'].isin([1, 2]).astype(int)

# Education categories
df['educ_lesshs'] = (df['EDUC'] < 6).astype(int)
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_somecol'] = (df['EDUC'].isin([7, 8, 9])).astype(int)
df['educ_college'] = (df['EDUC'] >= 10).astype(int)

# Years in US
df['years_in_us'] = df['YRSUSA1']

# =============================================================================
# 5. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[5] Descriptive Statistics")
print("="*80)

print("\nFull-time Employment Rates by Group and Period:")
print("-"*60)
ft_summary = df.groupby(['treated', 'post'])['fulltime'].agg(['mean', 'count'])
ft_summary.columns = ['Mean', 'N']
print(ft_summary)

# Weighted means
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

weighted_ft = df.groupby(['treated', 'post']).apply(weighted_mean)
print("\nWeighted Full-time Employment Rates:")
print(weighted_ft)

# Calculate raw DiD (unweighted)
pre_treat = df[(df['treated']==1) & (df['post']==0)]['fulltime'].mean()
post_treat = df[(df['treated']==1) & (df['post']==1)]['fulltime'].mean()
pre_control = df[(df['treated']==0) & (df['post']==0)]['fulltime'].mean()
post_control = df[(df['treated']==0) & (df['post']==1)]['fulltime'].mean()

raw_did = (post_treat - pre_treat) - (post_control - pre_control)

print(f"\nRaw Difference-in-Differences Calculation (unweighted):")
print(f"   Treatment Pre-period mean:  {pre_treat:.4f}")
print(f"   Treatment Post-period mean: {post_treat:.4f}")
print(f"   Control Pre-period mean:    {pre_control:.4f}")
print(f"   Control Post-period mean:   {post_control:.4f}")
print(f"   Treatment change: {post_treat - pre_treat:.4f}")
print(f"   Control change:   {post_control - pre_control:.4f}")
print(f"   Raw DiD estimate: {raw_did:.4f}")

# Weighted DiD
w_pre_treat = np.average(df[(df['treated']==1) & (df['post']==0)]['fulltime'],
                         weights=df[(df['treated']==1) & (df['post']==0)]['PERWT'])
w_post_treat = np.average(df[(df['treated']==1) & (df['post']==1)]['fulltime'],
                          weights=df[(df['treated']==1) & (df['post']==1)]['PERWT'])
w_pre_control = np.average(df[(df['treated']==0) & (df['post']==0)]['fulltime'],
                           weights=df[(df['treated']==0) & (df['post']==0)]['PERWT'])
w_post_control = np.average(df[(df['treated']==0) & (df['post']==1)]['fulltime'],
                            weights=df[(df['treated']==0) & (df['post']==1)]['PERWT'])

w_raw_did = (w_post_treat - w_pre_treat) - (w_post_control - w_pre_control)

print(f"\nWeighted Difference-in-Differences Calculation:")
print(f"   Treatment Pre-period mean:  {w_pre_treat:.4f}")
print(f"   Treatment Post-period mean: {w_post_treat:.4f}")
print(f"   Control Pre-period mean:    {w_pre_control:.4f}")
print(f"   Control Post-period mean:   {w_post_control:.4f}")
print(f"   Treatment change: {w_post_treat - w_pre_treat:.4f}")
print(f"   Control change:   {w_post_control - w_pre_control:.4f}")
print(f"   Weighted DiD estimate: {w_raw_did:.4f}")

# Summary statistics table
print("\n\nSample Characteristics by Group:")
print("-"*70)
print(f"{'Variable':<25} {'Treatment Pre':<12} {'Treatment Post':<14} {'Control Pre':<12} {'Control Post':<12}")
print("-"*70)

groups = [(1,0), (1,1), (0,0), (0,1)]
for var, label in [('female', 'Female (%)'), ('married', 'Married (%)'),
                   ('educ_hs', 'High School (%)'), ('educ_college', 'College+ (%)'),
                   ('years_in_us', 'Years in US'), ('AGE', 'Age')]:
    vals = []
    for t, p in groups:
        subset = df[(df['treated']==t) & (df['post']==p)]
        if var in ['female', 'married', 'educ_hs', 'educ_college']:
            vals.append(f"{subset[var].mean()*100:.1f}")
        else:
            vals.append(f"{subset[var].mean():.1f}")
    print(f"{label:<25} {vals[0]:<12} {vals[1]:<14} {vals[2]:<12} {vals[3]:<12}")

# =============================================================================
# 6. MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n\n[6] Main Regression Analysis")
print("="*80)

# Model 1: Basic DiD (no controls, no weights)
print("\n--- Model 1: Basic DiD (OLS, no weights) ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit()
print(model1.summary().tables[1])

# Model 2: DiD with survey weights
print("\n--- Model 2: DiD with Survey Weights (WLS) ---")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df,
                  weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with controls
print("\n--- Model 3: DiD with Controls (WLS) ---")
model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecol + educ_college + years_in_us',
                  data=df,
                  weights=df['PERWT']).fit()
print(model3.summary().tables[1])

# Model 4: DiD with controls and state fixed effects
print("\n--- Model 4: DiD with Controls and State Fixed Effects (WLS) ---")
model4 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecol + educ_college + years_in_us + C(STATEFIP)',
                  data=df,
                  weights=df['PERWT']).fit()
print(f"treated_post coefficient: {model4.params['treated_post']:.6f}")
print(f"Standard error: {model4.bse['treated_post']:.6f}")
print(f"t-statistic: {model4.tvalues['treated_post']:.4f}")
print(f"p-value: {model4.pvalues['treated_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treated_post', 0]:.6f}, {model4.conf_int().loc['treated_post', 1]:.6f}]")

# Model 5: DiD with year fixed effects (PREFERRED)
print("\n--- Model 5: DiD with Year and State Fixed Effects (Preferred) ---")
model5 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + educ_hs + educ_somecol + educ_college + years_in_us + C(STATEFIP)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

print(f"\nPREFERRED ESTIMATE (Model 5):")
print(f"   DiD coefficient (treated_post): {model5.params['treated_post']:.6f}")
print(f"   Robust standard error: {model5.bse['treated_post']:.6f}")
print(f"   t-statistic: {model5.tvalues['treated_post']:.4f}")
print(f"   p-value: {model5.pvalues['treated_post']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['treated_post', 0]:.6f}, {model5.conf_int().loc['treated_post', 1]:.6f}]")

# =============================================================================
# 7. ROBUSTNESS CHECKS
# =============================================================================
print("\n[7] Robustness Checks")
print("="*80)

# Robustness 1: Clustered standard errors by state
print("\n--- Robustness 1: Clustered Standard Errors by State ---")
model_cluster = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + educ_hs + educ_somecol + educ_college + years_in_us + C(STATEFIP)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"   DiD coefficient: {model_cluster.params['treated_post']:.6f}")
print(f"   Clustered SE: {model_cluster.bse['treated_post']:.6f}")
print(f"   95% CI: [{model_cluster.conf_int().loc['treated_post', 0]:.6f}, {model_cluster.conf_int().loc['treated_post', 1]:.6f}]")

# Robustness 2: Placebo test
print("\n--- Robustness 2: Placebo Test (fake treatment in 2010) ---")
df_placebo = df[df['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011])].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2010).astype(int)
df_placebo['treated_post_placebo'] = df_placebo['treated'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treated + post_placebo + treated_post_placebo + female + married + educ_hs + educ_somecol + educ_college + years_in_us',
                  data=df_placebo,
                  weights=df_placebo['PERWT']).fit()
print(f"   Placebo DiD coefficient: {model_placebo.params['treated_post_placebo']:.6f}")
print(f"   Standard error: {model_placebo.bse['treated_post_placebo']:.6f}")
print(f"   p-value: {model_placebo.pvalues['treated_post_placebo']:.4f}")

# Robustness 3: By gender
print("\n--- Robustness 3: Heterogeneity by Gender ---")
df_male = df[df['female'] == 0]
df_female = df[df['female'] == 1]

model_male = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + married + educ_hs + educ_somecol + educ_college + years_in_us',
                  data=df_male,
                  weights=df_male['PERWT']).fit()
model_female = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + married + educ_hs + educ_somecol + educ_college + years_in_us',
                  data=df_female,
                  weights=df_female['PERWT']).fit()

print(f"   Male DiD coefficient: {model_male.params['treated_post']:.6f} (SE: {model_male.bse['treated_post']:.6f})")
print(f"   Female DiD coefficient: {model_female.params['treated_post']:.6f} (SE: {model_female.bse['treated_post']:.6f})")

# =============================================================================
# 8. EVENT STUDY / PARALLEL TRENDS
# =============================================================================
print("\n[8] Event Study Analysis (Parallel Trends)")
print("="*80)

# Create year dummies interacted with treatment (2011 is reference year)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'treat_x_{year}'] = df['treated'] * (df['YEAR'] == year).astype(int)

event_formula = 'fulltime ~ treated + C(YEAR) + treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + female + married + educ_hs + educ_somecol + educ_college + years_in_us'
model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit()

print("\nEvent Study Coefficients (relative to 2011):")
print("-"*50)
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_coefs = []
event_ses = []
for year in event_years:
    coef = model_event.params[f'treat_x_{year}']
    se = model_event.bse[f'treat_x_{year}']
    pval = model_event.pvalues[f'treat_x_{year}']
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
    print(f"   {year}: {coef:8.4f} (SE: {se:.4f}){sig}")
    event_coefs.append(coef)
    event_ses.append(se)

# =============================================================================
# 9. FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
Sample Size: {len(df):,}
   - Treatment group (ages 26-30 in 2012): {df['treated'].sum():,}
   - Control group (ages 31-35 in 2012): {(1-df['treated']).sum():,}

Sample Restrictions Applied:
   1. Years 2006-2011 and 2013-2016 (excluding 2012)
   2. Hispanic-Mexican ethnicity (HISPAN == 1)
   3. Born in Mexico (BPL == 200)
   4. Non-citizen (CITIZEN == 3)
   5. Arrived before age 16 (age_at_immigration < 16)
   6. Arrived by 2007 (YRIMMIG <= 2007)
   7. Birth years 1977-1986 (ages 26-35 in 2012)

Preferred Estimate (Model 5 - Year and State FE, Robust SE):
   DiD Coefficient: {model5.params['treated_post']:.4f}
   Robust Standard Error: {model5.bse['treated_post']:.4f}
   95% Confidence Interval: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]
   p-value: {model5.pvalues['treated_post']:.4f}

Interpretation:
   DACA eligibility is associated with a {model5.params['treated_post']*100:.2f} percentage point
   {"increase" if model5.params['treated_post'] > 0 else "decrease"} in the probability of full-time employment.
   This effect is {"statistically significant" if model5.pvalues['treated_post'] < 0.05 else "not statistically significant"} at the 5% level.
""")

# =============================================================================
# 10. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n[10] Saving results...")

# Save summary statistics
summary_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'AGE': 'mean',
    'years_in_us': 'mean'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')

# Save model results
results_dict = {
    'Model': ['Model 1 (Basic)', 'Model 2 (Weighted)', 'Model 3 (Controls)',
              'Model 4 (State FE)', 'Model 5 (Year+State FE)', 'Clustered SE'],
    'DiD_Coefficient': [model1.params['treated_post'], model2.params['treated_post'],
                        model3.params['treated_post'], model4.params['treated_post'],
                        model5.params['treated_post'], model_cluster.params['treated_post']],
    'Std_Error': [model1.bse['treated_post'], model2.bse['treated_post'],
                  model3.bse['treated_post'], model4.bse['treated_post'],
                  model5.bse['treated_post'], model_cluster.bse['treated_post']],
    'p_value': [model1.pvalues['treated_post'], model2.pvalues['treated_post'],
                model3.pvalues['treated_post'], model4.pvalues['treated_post'],
                model5.pvalues['treated_post'], model_cluster.pvalues['treated_post']],
    'CI_lower': [model1.conf_int().loc['treated_post', 0], model2.conf_int().loc['treated_post', 0],
                 model3.conf_int().loc['treated_post', 0], model4.conf_int().loc['treated_post', 0],
                 model5.conf_int().loc['treated_post', 0], model_cluster.conf_int().loc['treated_post', 0]],
    'CI_upper': [model1.conf_int().loc['treated_post', 1], model2.conf_int().loc['treated_post', 1],
                 model3.conf_int().loc['treated_post', 1], model4.conf_int().loc['treated_post', 1],
                 model5.conf_int().loc['treated_post', 1], model_cluster.conf_int().loc['treated_post', 1]]
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv('regression_results.csv', index=False)

# Save event study coefficients
event_results = {
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': event_coefs[:5] + [0] + event_coefs[5:],
    'Std_Error': event_ses[:5] + [0] + event_ses[5:]
}
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

# Full-time employment trends by group
trends = df.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
trends.columns = ['Control', 'Treatment']
trends.to_csv('trends_data.csv')

# Save heterogeneity results
hetero_dict = {
    'Subgroup': ['Male', 'Female'],
    'DiD_Coefficient': [model_male.params['treated_post'], model_female.params['treated_post']],
    'Std_Error': [model_male.bse['treated_post'], model_female.bse['treated_post']],
    'N': [len(df_male), len(df_female)]
}
hetero_df = pd.DataFrame(hetero_dict)
hetero_df.to_csv('heterogeneity_results.csv', index=False)

print("   Saved: summary_statistics.csv")
print("   Saved: regression_results.csv")
print("   Saved: event_study_results.csv")
print("   Saved: trends_data.csv")
print("   Saved: heterogeneity_results.csv")

print("\nAnalysis complete!")
print("="*80)

# Store key results for report
key_results = {
    'sample_size': len(df),
    'treatment_n': df['treated'].sum(),
    'control_n': (1-df['treated']).sum(),
    'did_coef': model5.params['treated_post'],
    'did_se': model5.bse['treated_post'],
    'did_pval': model5.pvalues['treated_post'],
    'did_ci_low': model5.conf_int().loc['treated_post', 0],
    'did_ci_high': model5.conf_int().loc['treated_post', 1]
}

import json
with open('key_results.json', 'w') as f:
    json.dump(key_results, f, indent=2)
print("   Saved: key_results.json")
