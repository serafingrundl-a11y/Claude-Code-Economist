"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals

Treatment: Ages 26-30 as of June 15, 2012 (DACA-eligible)
Control: Ages 31-35 as of June 15, 2012 (would be eligible if not for age cutoff)
Outcome: Full-time employment (usual hours worked >= 35 per week)
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DACA REPLICATION STUDY - FULL ANALYSIS")
print("="*80)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations: {len(df):,}")

# =============================================================================
# STEP 1: Define DACA eligibility criteria
# =============================================================================
print("\n2. DEFINING SAMPLE SELECTION CRITERIA...")

# Hispanic-Mexican ethnicity AND born in Mexico
# HISPAN = 1 (Mexican) and BPL = 200 (Mexico)
df['hispanic_mexican'] = (df['HISPAN'] == 1) & (df['BPL'] == 200)
print(f"   Hispanic-Mexican born in Mexico: {df['hispanic_mexican'].sum():,}")

# Non-citizen (proxy for undocumented status)
# CITIZEN = 3 (Not a citizen)
df['non_citizen'] = df['CITIZEN'] == 3
print(f"   Non-citizens: {df['non_citizen'].sum():,}")

# Calculate age as of June 15, 2012
# DACA was announced June 15, 2012
# We need to determine age as of mid-2012
# Using BIRTHYR and approximating based on when data was collected

# For year Y, the person's age at mid-2012 would be:
# If Y < 2012: age_2012 = AGE + (2012 - Y)
# If Y = 2012: age_2012 = AGE (approximately)
# If Y > 2012: age_2012 = AGE - (Y - 2012)

df['age_in_2012'] = df['AGE'] - (df['YEAR'] - 2012)

# Treatment group: ages 26-30 as of June 2012 (born 1982-1986)
# Control group: ages 31-35 as of June 2012 (born 1977-1981)
df['treat'] = (df['age_in_2012'] >= 26) & (df['age_in_2012'] <= 30)
df['control'] = (df['age_in_2012'] >= 31) & (df['age_in_2012'] <= 35)

print(f"   Treatment group (age 26-30 in 2012): {df['treat'].sum():,}")
print(f"   Control group (age 31-35 in 2012): {df['control'].sum():,}")

# DACA eligibility requirements:
# 1. Arrived in US before 16th birthday
# 2. In US continuously since June 15, 2007 (at least 5 years)
# 3. Not yet 31 as of June 15, 2012 (treatment group)

# Calculate age at immigration
# YRIMMIG is year of immigration
# Person arrived before age 16 if: BIRTHYR + 16 > YRIMMIG
df['arrived_before_16'] = np.where(
    df['YRIMMIG'] > 0,
    (df['YRIMMIG'] - df['BIRTHYR']) < 16,
    False
)

# In US since at least 2007
# YRSUSA1 is years in USA, or we can use YRIMMIG <= 2007
df['in_us_since_2007'] = np.where(
    df['YRIMMIG'] > 0,
    df['YRIMMIG'] <= 2007,
    False
)

print(f"   Arrived before age 16: {df['arrived_before_16'].sum():,}")
print(f"   In US since 2007: {df['in_us_since_2007'].sum():,}")

# =============================================================================
# STEP 2: Create analysis sample
# =============================================================================
print("\n3. CREATING ANALYSIS SAMPLE...")

# Base sample: Hispanic-Mexican, born in Mexico, non-citizen
base_sample = df['hispanic_mexican'] & df['non_citizen']
print(f"   Base sample (Hispanic-Mexican, Mexican-born, non-citizen): {base_sample.sum():,}")

# Add DACA eligibility criteria (arrived before 16, in US since 2007)
eligible_sample = base_sample & df['arrived_before_16'] & df['in_us_since_2007']
print(f"   With DACA eligibility criteria: {eligible_sample.sum():,}")

# Restrict to treatment or control groups
analysis_sample = eligible_sample & (df['treat'] | df['control'])
print(f"   In treatment or control age groups: {analysis_sample.sum():,}")

# Create analysis dataframe
dfa = df[analysis_sample].copy()
print(f"\n   Final analysis sample: {len(dfa):,}")

# =============================================================================
# STEP 3: Define outcome and key variables
# =============================================================================
print("\n4. DEFINING OUTCOME VARIABLE...")

# Full-time employment: UHRSWORK >= 35
# First, filter to those in labor force or employed
# EMPSTAT: 1 = Employed, 2 = Unemployed, 3 = Not in labor force

# Primary outcome: Full-time employment among all (including not in labor force)
dfa['fulltime'] = (dfa['UHRSWORK'] >= 35).astype(int)

# Alternative: Among employed only
dfa['employed'] = (dfa['EMPSTAT'] == 1).astype(int)
dfa['fulltime_if_employed'] = np.where(
    dfa['employed'] == 1,
    (dfa['UHRSWORK'] >= 35).astype(int),
    np.nan
)

print(f"   Full-time employed (all): {dfa['fulltime'].mean()*100:.2f}%")
print(f"   Employed: {dfa['employed'].mean()*100:.2f}%")

# =============================================================================
# STEP 4: Create DiD variables
# =============================================================================
print("\n5. CREATING DIFFERENCE-IN-DIFFERENCES VARIABLES...")

# Post period: 2013-2016 (after DACA implementation)
# Pre period: 2006-2011 (before DACA)
# Note: 2012 is ambiguous (DACA announced June 15), so we exclude it

dfa['post'] = (dfa['YEAR'] >= 2013).astype(int)
dfa['pre'] = (dfa['YEAR'] <= 2011).astype(int)

# Create the treatment indicator (1 if ages 26-30 in 2012, 0 if ages 31-35)
dfa['treated'] = dfa['treat'].astype(int)

# Interaction term for DiD
dfa['treat_post'] = dfa['treated'] * dfa['post']

# Exclude 2012 for cleaner identification
dfa_clean = dfa[dfa['YEAR'] != 2012].copy()
print(f"   Sample excluding 2012: {len(dfa_clean):,}")

print(f"\n   Pre-period observations (2006-2011): {dfa_clean['pre'].sum():,}")
print(f"   Post-period observations (2013-2016): {dfa_clean['post'].sum():,}")
print(f"   Treatment group: {dfa_clean['treated'].sum():,}")
print(f"   Control group: {(1-dfa_clean['treated']).sum():,}")

# =============================================================================
# STEP 5: Summary Statistics
# =============================================================================
print("\n6. SUMMARY STATISTICS...")

print("\n   By Treatment Status and Period:")
summary = dfa_clean.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean'],
    'AGE': ['mean'],
    'SEX': ['mean'],  # 1=Male, 2=Female
    'PERWT': ['sum']
}).round(4)
print(summary)

# Weighted statistics
print("\n   Weighted Full-Time Employment Rates:")
for treat_val in [0, 1]:
    for post_val in [0, 1]:
        mask = (dfa_clean['treated'] == treat_val) & (dfa_clean['post'] == post_val)
        subset = dfa_clean[mask]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        treat_label = "Treatment" if treat_val == 1 else "Control"
        period_label = "Post" if post_val == 1 else "Pre"
        print(f"   {treat_label}, {period_label}: {weighted_mean*100:.2f}%")

# =============================================================================
# STEP 6: Simple DiD calculation
# =============================================================================
print("\n7. SIMPLE DIFFERENCE-IN-DIFFERENCES...")

# Calculate means for each group-period
means = dfa_clean.groupby(['treated', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

print("\n   Full-time employment rates (weighted):")
print(f"   Treatment, Pre:  {means.loc[1, 0]*100:.2f}%")
print(f"   Treatment, Post: {means.loc[1, 1]*100:.2f}%")
print(f"   Control, Pre:    {means.loc[0, 0]*100:.2f}%")
print(f"   Control, Post:   {means.loc[0, 1]*100:.2f}%")

# DiD estimate
treat_diff = means.loc[1, 1] - means.loc[1, 0]
control_diff = means.loc[0, 1] - means.loc[0, 0]
did_estimate = treat_diff - control_diff

print(f"\n   Treatment group change: {treat_diff*100:.2f} pp")
print(f"   Control group change:   {control_diff*100:.2f} pp")
print(f"   DiD Estimate:           {did_estimate*100:.2f} pp")

# =============================================================================
# STEP 7: Regression Analysis
# =============================================================================
print("\n8. REGRESSION ANALYSIS...")

# Model 1: Basic DiD (unweighted)
print("\n   Model 1: Basic DiD (Unweighted OLS)")
model1 = smf.ols('fulltime ~ treated + post + treat_post', data=dfa_clean).fit()
print(f"   DiD coefficient: {model1.params['treat_post']:.4f}")
print(f"   Std. Error: {model1.bse['treat_post']:.4f}")
print(f"   t-statistic: {model1.tvalues['treat_post']:.4f}")
print(f"   p-value: {model1.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")

# Model 2: DiD with sample weights (WLS)
print("\n   Model 2: DiD with Survey Weights (WLS)")
model2 = smf.wls('fulltime ~ treated + post + treat_post',
                  data=dfa_clean,
                  weights=dfa_clean['PERWT']).fit()
print(f"   DiD coefficient: {model2.params['treat_post']:.4f}")
print(f"   Std. Error: {model2.bse['treat_post']:.4f}")
print(f"   t-statistic: {model2.tvalues['treat_post']:.4f}")
print(f"   p-value: {model2.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")

# Model 3: DiD with year fixed effects
print("\n   Model 3: DiD with Year Fixed Effects (WLS)")
dfa_clean['year_factor'] = dfa_clean['YEAR'].astype(str)
model3 = smf.wls('fulltime ~ treated + C(year_factor) + treat_post',
                  data=dfa_clean,
                  weights=dfa_clean['PERWT']).fit()
print(f"   DiD coefficient: {model3.params['treat_post']:.4f}")
print(f"   Std. Error: {model3.bse['treat_post']:.4f}")
print(f"   t-statistic: {model3.tvalues['treat_post']:.4f}")
print(f"   p-value: {model3.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")

# Model 4: DiD with covariates
print("\n   Model 4: DiD with Covariates (WLS)")
# Add control variables
dfa_clean['female'] = (dfa_clean['SEX'] == 2).astype(int)
dfa_clean['married'] = (dfa_clean['MARST'] == 1).astype(int)
dfa_clean['has_children'] = (dfa_clean['NCHILD'] > 0).astype(int)

# Education categories
dfa_clean['less_than_hs'] = (dfa_clean['EDUC'] < 6).astype(int)
dfa_clean['hs_grad'] = (dfa_clean['EDUC'] == 6).astype(int)
dfa_clean['some_college'] = ((dfa_clean['EDUC'] >= 7) & (dfa_clean['EDUC'] <= 9)).astype(int)
dfa_clean['college_plus'] = (dfa_clean['EDUC'] >= 10).astype(int)

model4 = smf.wls('fulltime ~ treated + C(year_factor) + treat_post + female + married + has_children + hs_grad + some_college + college_plus',
                  data=dfa_clean,
                  weights=dfa_clean['PERWT']).fit()
print(f"   DiD coefficient: {model4.params['treat_post']:.4f}")
print(f"   Std. Error: {model4.bse['treat_post']:.4f}")
print(f"   t-statistic: {model4.tvalues['treat_post']:.4f}")
print(f"   p-value: {model4.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

# Model 5: DiD with state fixed effects
print("\n   Model 5: DiD with State and Year Fixed Effects (WLS)")
model5 = smf.wls('fulltime ~ treated + C(year_factor) + C(STATEFIP) + treat_post + female + married + has_children + hs_grad + some_college + college_plus',
                  data=dfa_clean,
                  weights=dfa_clean['PERWT']).fit()
print(f"   DiD coefficient: {model5.params['treat_post']:.4f}")
print(f"   Std. Error: {model5.bse['treat_post']:.4f}")
print(f"   t-statistic: {model5.tvalues['treat_post']:.4f}")
print(f"   p-value: {model5.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")

# =============================================================================
# STEP 8: Robust Standard Errors
# =============================================================================
print("\n9. ROBUST STANDARD ERRORS...")

# Model with heteroskedasticity-robust standard errors
print("\n   Model 6: DiD with Robust Standard Errors (HC1)")
model6 = smf.wls('fulltime ~ treated + C(year_factor) + C(STATEFIP) + treat_post + female + married + has_children + hs_grad + some_college + college_plus',
                  data=dfa_clean,
                  weights=dfa_clean['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient: {model6.params['treat_post']:.4f}")
print(f"   Robust Std. Error: {model6.bse['treat_post']:.4f}")
print(f"   t-statistic: {model6.tvalues['treat_post']:.4f}")
print(f"   p-value: {model6.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model6.conf_int().loc['treat_post', 0]:.4f}, {model6.conf_int().loc['treat_post', 1]:.4f}]")

# =============================================================================
# STEP 9: Event Study / Parallel Trends
# =============================================================================
print("\n10. EVENT STUDY / PARALLEL TRENDS CHECK...")

# Create year-treatment interactions
years = sorted(dfa_clean['YEAR'].unique())
reference_year = 2011  # Last pre-period year

dfa_clean['year_int'] = dfa_clean['YEAR'].astype(int)

# Create interactions for each year except reference
for year in years:
    if year != reference_year:
        dfa_clean[f'treat_x_{year}'] = dfa_clean['treated'] * (dfa_clean['YEAR'] == year).astype(int)

# Build formula
year_interactions = ' + '.join([f'treat_x_{year}' for year in years if year != reference_year])
formula = f'fulltime ~ treated + C(year_factor) + {year_interactions}'

model_es = smf.wls(formula, data=dfa_clean, weights=dfa_clean['PERWT']).fit()

print("\n   Year-by-Treatment Interactions (relative to 2011):")
for year in years:
    if year != reference_year:
        coef = model_es.params[f'treat_x_{year}']
        se = model_es.bse[f'treat_x_{year}']
        pval = model_es.pvalues[f'treat_x_{year}']
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"   {year}: {coef:.4f} ({se:.4f}) {sig}")

# =============================================================================
# STEP 10: Heterogeneity Analysis
# =============================================================================
print("\n11. HETEROGENEITY ANALYSIS...")

# By gender
print("\n   By Gender:")
for gender, label in [(1, 'Male'), (2, 'Female')]:
    subset = dfa_clean[dfa_clean['SEX'] == gender]
    model = smf.wls('fulltime ~ treated + C(year_factor) + treat_post',
                    data=subset, weights=subset['PERWT']).fit()
    print(f"   {label}: DiD = {model.params['treat_post']:.4f} (SE: {model.bse['treat_post']:.4f})")

# By education
print("\n   By Education:")
dfa_clean['low_educ'] = (dfa_clean['EDUC'] < 6).astype(int)
for educ, label in [(1, 'Less than HS'), (0, 'HS or more')]:
    subset = dfa_clean[dfa_clean['low_educ'] == educ]
    if len(subset) > 100:
        model = smf.wls('fulltime ~ treated + C(year_factor) + treat_post',
                        data=subset, weights=subset['PERWT']).fit()
        print(f"   {label}: DiD = {model.params['treat_post']:.4f} (SE: {model.bse['treat_post']:.4f})")

# =============================================================================
# STEP 11: Alternative Outcomes
# =============================================================================
print("\n12. ALTERNATIVE OUTCOMES...")

# Employment (any)
print("\n   Employment (any job):")
model_emp = smf.wls('employed ~ treated + C(year_factor) + treat_post',
                    data=dfa_clean, weights=dfa_clean['PERWT']).fit()
print(f"   DiD coefficient: {model_emp.params['treat_post']:.4f}")
print(f"   Std. Error: {model_emp.bse['treat_post']:.4f}")
print(f"   p-value: {model_emp.pvalues['treat_post']:.4f}")

# Labor force participation
dfa_clean['in_labor_force'] = (dfa_clean['LABFORCE'] == 2).astype(int)
print("\n   Labor Force Participation:")
model_lf = smf.wls('in_labor_force ~ treated + C(year_factor) + treat_post',
                   data=dfa_clean, weights=dfa_clean['PERWT']).fit()
print(f"   DiD coefficient: {model_lf.params['treat_post']:.4f}")
print(f"   Std. Error: {model_lf.bse['treat_post']:.4f}")
print(f"   p-value: {model_lf.pvalues['treat_post']:.4f}")

# =============================================================================
# STEP 12: Sample Size Summary
# =============================================================================
print("\n13. SAMPLE SIZE SUMMARY...")

print(f"\n   Total analysis sample (excl. 2012): {len(dfa_clean):,}")
print(f"   Treatment group: {dfa_clean['treated'].sum():,}")
print(f"   Control group: {(1-dfa_clean['treated']).sum():,}")
print(f"   Pre-period: {(dfa_clean['YEAR'] <= 2011).sum():,}")
print(f"   Post-period: {(dfa_clean['YEAR'] >= 2013).sum():,}")

# Weighted population
print(f"\n   Weighted population: {dfa_clean['PERWT'].sum():,.0f}")

# =============================================================================
# STEP 13: Save Results
# =============================================================================
print("\n14. SAVING RESULTS...")

# Save key results for the report
results = {
    'sample_size': len(dfa_clean),
    'treatment_n': int(dfa_clean['treated'].sum()),
    'control_n': int((1-dfa_clean['treated']).sum()),
    'did_coef': model6.params['treat_post'],
    'did_se': model6.bse['treat_post'],
    'did_pval': model6.pvalues['treat_post'],
    'did_ci_low': model6.conf_int().loc['treat_post', 0],
    'did_ci_high': model6.conf_int().loc['treat_post', 1],
    'pre_treat_rate': means.loc[1, 0],
    'post_treat_rate': means.loc[1, 1],
    'pre_control_rate': means.loc[0, 0],
    'post_control_rate': means.loc[0, 1],
}

# Save to file
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("   Results saved to results.json")

# =============================================================================
# Print full model summary for preferred specification
# =============================================================================
print("\n" + "="*80)
print("PREFERRED MODEL SUMMARY (Model 6)")
print("="*80)
print(model6.summary())

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
