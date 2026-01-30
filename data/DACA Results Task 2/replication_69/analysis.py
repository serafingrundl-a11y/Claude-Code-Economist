"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born non-citizens

Treatment group: Ages 26-30 as of June 15, 2012
Control group: Ages 31-35 as of June 15, 2012
Pre-period: 2006-2011
Post-period: 2013-2016

Outcome: Full-time employment (usually working 35+ hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("DACA REPLICATION ANALYSIS")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations loaded: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# 2. SAMPLE SELECTION
# ============================================================================
print("\n2. Sample Selection...")

# Step 2a: Keep only Hispanic-Mexican ethnicity
# HISPAN == 1 indicates Mexican
print(f"   Initial sample: {len(df):,}")
df_sample = df[df['HISPAN'] == 1].copy()
print(f"   After keeping HISPAN == 1 (Mexican): {len(df_sample):,}")

# Step 2b: Keep only those born in Mexico
# BPL == 200 indicates Mexico
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"   After keeping BPL == 200 (Mexico): {len(df_sample):,}")

# Step 2c: Keep only non-citizens (proxy for undocumented)
# CITIZEN == 3 indicates "Not a citizen"
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"   After keeping CITIZEN == 3 (Non-citizen): {len(df_sample):,}")

# ============================================================================
# 3. CREATE AGE AT POLICY IMPLEMENTATION (June 15, 2012)
# ============================================================================
print("\n3. Calculating age at DACA implementation (June 15, 2012)...")

# Age at June 15, 2012 depends on birth year and quarter
# BIRTHQTR: 1 = Jan-Mar, 2 = Apr-Jun, 3 = Jul-Sep, 4 = Oct-Dec
# For June 15, 2012:
# - If born Q1 (Jan-Mar) of year Y: age = 2012 - Y (already had birthday)
# - If born Q2 (Apr-Jun) of year Y: age = 2012 - Y - 0.5 (approximately, some had birthday, some not)
# - If born Q3 (Jul-Sep) of year Y: age = 2012 - Y - 1 (not yet had birthday in 2012)
# - If born Q4 (Oct-Dec) of year Y: age = 2012 - Y - 1 (not yet had birthday)

# Simpler approach: age at June 2012 = 2012 - BIRTHYR, then adjust by birth quarter
# For conservative estimates, use year-based calculation

df_sample['age_at_daca'] = 2012 - df_sample['BIRTHYR']

# For more precision, adjust based on birth quarter
# Those born Jul-Dec (Q3, Q4) would not have had their 2012 birthday by June 15
df_sample.loc[df_sample['BIRTHQTR'].isin([3, 4]), 'age_at_daca'] = df_sample.loc[df_sample['BIRTHQTR'].isin([3, 4]), 'age_at_daca'] - 1

print(f"   Age at DACA range: {df_sample['age_at_daca'].min()} to {df_sample['age_at_daca'].max()}")

# ============================================================================
# 4. DEFINE TREATMENT AND CONTROL GROUPS
# ============================================================================
print("\n4. Defining treatment and control groups...")

# Treatment group: 26-30 years old at June 15, 2012 (DACA eligible by age)
# Control group: 31-35 years old at June 15, 2012 (too old for DACA)

df_sample['treatment'] = ((df_sample['age_at_daca'] >= 26) & (df_sample['age_at_daca'] <= 30)).astype(int)
df_sample['control'] = ((df_sample['age_at_daca'] >= 31) & (df_sample['age_at_daca'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df_sample[(df_sample['treatment'] == 1) | (df_sample['control'] == 1)].copy()
print(f"   Treatment group (ages 26-30): {(df_analysis['treatment'] == 1).sum():,}")
print(f"   Control group (ages 31-35): {(df_analysis['control'] == 1).sum():,}")
print(f"   Total in analysis sample: {len(df_analysis):,}")

# ============================================================================
# 5. APPLY DACA ELIGIBILITY CRITERIA
# ============================================================================
print("\n5. Applying DACA eligibility criteria...")

# DACA eligibility requirements:
# 1. Arrived unlawfully before 16th birthday
# 2. Under 31 as of June 15, 2012 (treatment group by definition)
# 3. Lived continuously in US since June 15, 2007
# 4. Present in US on June 15, 2012 (assumed for ACS respondents)

# We can approximate criterion 1 (arrived before 16th birthday):
# YRIMMIG - BIRTHYR < 16
df_analysis['age_at_immigration'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']

# Filter: arrived before 16th birthday
# Note: YRIMMIG == 0 means N/A - these are likely US-born or missing
df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0].copy()
print(f"   After removing YRIMMIG == 0: {len(df_analysis):,}")

df_analysis = df_analysis[df_analysis['age_at_immigration'] < 16].copy()
print(f"   After keeping arrived before age 16: {len(df_analysis):,}")

# Criterion 3: Lived in US since June 15, 2007
# YRIMMIG <= 2007
df_analysis = df_analysis[df_analysis['YRIMMIG'] <= 2007].copy()
print(f"   After keeping YRIMMIG <= 2007: {len(df_analysis):,}")

# ============================================================================
# 6. DEFINE TIME PERIODS
# ============================================================================
print("\n6. Defining time periods...")

# Pre-period: 2006-2011 (before DACA)
# Post-period: 2013-2016 (after DACA)
# Exclude 2012: DACA implemented June 15, 2012 - can't distinguish pre/post

df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

print(f"   Pre-period (2006-2011): {(df_analysis['post'] == 0).sum():,}")
print(f"   Post-period (2013-2016): {(df_analysis['post'] == 1).sum():,}")

# ============================================================================
# 7. CREATE OUTCOME VARIABLE
# ============================================================================
print("\n7. Creating outcome variable...")

# Full-time employment: UHRSWORK >= 35
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"   Full-time employment rate (overall): {df_analysis['fulltime'].mean():.3f}")
print(f"   Treatment group - Pre: {df_analysis[(df_analysis['treatment']==1) & (df_analysis['post']==0)]['fulltime'].mean():.3f}")
print(f"   Treatment group - Post: {df_analysis[(df_analysis['treatment']==1) & (df_analysis['post']==1)]['fulltime'].mean():.3f}")
print(f"   Control group - Pre: {df_analysis[(df_analysis['treatment']==0) & (df_analysis['post']==0)]['fulltime'].mean():.3f}")
print(f"   Control group - Post: {df_analysis[(df_analysis['treatment']==0) & (df_analysis['post']==1)]['fulltime'].mean():.3f}")

# ============================================================================
# 8. CREATE INTERACTION TERM
# ============================================================================
print("\n8. Creating DiD interaction term...")

df_analysis['treat_post'] = df_analysis['treatment'] * df_analysis['post']

# ============================================================================
# 9. SUMMARY STATISTICS
# ============================================================================
print("\n9. Summary Statistics...")
print("-"*70)

# Create summary statistics table
summary_vars = ['fulltime', 'AGE', 'SEX', 'EDUC', 'UHRSWORK']

print("\nSample Size by Group and Period:")
print("-"*50)
for t in [0, 1]:
    for p in [0, 1]:
        n = len(df_analysis[(df_analysis['treatment']==t) & (df_analysis['post']==p)])
        group = "Treatment" if t else "Control"
        period = "Post" if p else "Pre"
        print(f"   {group}, {period}: {n:,}")

print("\nMean Full-time Employment Rate:")
print("-"*50)
treatment_pre = df_analysis[(df_analysis['treatment']==1) & (df_analysis['post']==0)]['fulltime'].mean()
treatment_post = df_analysis[(df_analysis['treatment']==1) & (df_analysis['post']==1)]['fulltime'].mean()
control_pre = df_analysis[(df_analysis['treatment']==0) & (df_analysis['post']==0)]['fulltime'].mean()
control_post = df_analysis[(df_analysis['treatment']==0) & (df_analysis['post']==1)]['fulltime'].mean()

print(f"   Treatment Pre:  {treatment_pre:.4f}")
print(f"   Treatment Post: {treatment_post:.4f}")
print(f"   Control Pre:    {control_pre:.4f}")
print(f"   Control Post:   {control_post:.4f}")

# Simple DiD estimate
did_simple = (treatment_post - treatment_pre) - (control_post - control_pre)
print(f"\n   Simple DiD estimate: {did_simple:.4f}")

# ============================================================================
# 10. MAIN REGRESSION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("10. DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("="*70)

# Model 1: Basic DiD without covariates
print("\nModel 1: Basic DiD (no covariates)")
print("-"*70)
model1 = smf.ols('fulltime ~ treatment + post + treat_post', data=df_analysis).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with year fixed effects
print("\n\nModel 2: DiD with Year Fixed Effects")
print("-"*70)
df_analysis['year_factor'] = pd.Categorical(df_analysis['YEAR'])
model2 = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post', data=df_analysis).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with covariates
print("\n\nModel 3: DiD with Covariates (sex, education, marital status)")
print("-"*70)
# Create education categories
df_analysis['educ_cat'] = pd.cut(df_analysis['EDUC'], bins=[-1, 2, 6, 10, 11], labels=['less_hs', 'hs', 'some_college', 'college_plus'])
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

model3 = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post + female + C(educ_cat) + married',
                 data=df_analysis).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with state fixed effects
print("\n\nModel 4: DiD with Year and State Fixed Effects")
print("-"*70)
model4 = smf.ols('fulltime ~ treatment + C(YEAR) + C(STATEFIP) + treat_post',
                 data=df_analysis).fit(cov_type='HC1')
print(f"DiD Coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"Standard Error: {model4.bse['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model4.pvalues['treat_post']:.4f}")
print(f"N: {int(model4.nobs):,}")
print(f"R-squared: {model4.rsquared:.4f}")

# Model 5: Full model with covariates and fixed effects
print("\n\nModel 5: Full Model (Year FE + State FE + Covariates)")
print("-"*70)
model5 = smf.ols('fulltime ~ treatment + C(YEAR) + C(STATEFIP) + treat_post + female + C(educ_cat) + married',
                 data=df_analysis).fit(cov_type='HC1')
print(f"DiD Coefficient (treat_post): {model5.params['treat_post']:.4f}")
print(f"Standard Error: {model5.bse['treat_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model5.pvalues['treat_post']:.4f}")
print(f"N: {int(model5.nobs):,}")
print(f"R-squared: {model5.rsquared:.4f}")

# ============================================================================
# 11. ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*70)
print("11. ROBUSTNESS CHECKS")
print("="*70)

# 11a. Weighted regression using person weights
print("\nRobustness Check 1: Weighted Regression (using PERWT)")
print("-"*70)
model_weighted = smf.wls('fulltime ~ treatment + C(YEAR) + treat_post + female + C(educ_cat) + married',
                         data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient (treat_post): {model_weighted.params['treat_post']:.4f}")
print(f"Standard Error: {model_weighted.bse['treat_post']:.4f}")
print(f"95% CI: [{model_weighted.conf_int().loc['treat_post', 0]:.4f}, {model_weighted.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model_weighted.pvalues['treat_post']:.4f}")

# 11b. Check by gender
print("\nRobustness Check 2: Effects by Gender")
print("-"*70)
model_male = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post',
                     data=df_analysis[df_analysis['female']==0]).fit(cov_type='HC1')
model_female = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post',
                       data=df_analysis[df_analysis['female']==1]).fit(cov_type='HC1')
print(f"Male - DiD Coefficient: {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f})")
print(f"Female - DiD Coefficient: {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f})")

# 11c. Placebo test: use pre-period only with fake treatment year
print("\nRobustness Check 3: Placebo Test (Pre-period 2006-2008 vs 2009-2011)")
print("-"*70)
df_placebo = df_analysis[df_analysis['post'] == 0].copy()
df_placebo['fake_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['fake_treat_post'] = df_placebo['treatment'] * df_placebo['fake_post']
model_placebo = smf.ols('fulltime ~ treatment + C(YEAR) + fake_treat_post',
                        data=df_placebo).fit(cov_type='HC1')
print(f"Placebo DiD Coefficient: {model_placebo.params['fake_treat_post']:.4f}")
print(f"Standard Error: {model_placebo.bse['fake_treat_post']:.4f}")
print(f"P-value: {model_placebo.pvalues['fake_treat_post']:.4f}")

# ============================================================================
# 12. EVENT STUDY ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("12. EVENT STUDY / DYNAMIC EFFECTS")
print("="*70)

# Create year-specific treatment effects (relative to 2011)
years = sorted(df_analysis['YEAR'].unique())
df_analysis['year_2006'] = ((df_analysis['YEAR'] == 2006) & (df_analysis['treatment'] == 1)).astype(int)
df_analysis['year_2007'] = ((df_analysis['YEAR'] == 2007) & (df_analysis['treatment'] == 1)).astype(int)
df_analysis['year_2008'] = ((df_analysis['YEAR'] == 2008) & (df_analysis['treatment'] == 1)).astype(int)
df_analysis['year_2009'] = ((df_analysis['YEAR'] == 2009) & (df_analysis['treatment'] == 1)).astype(int)
df_analysis['year_2010'] = ((df_analysis['YEAR'] == 2010) & (df_analysis['treatment'] == 1)).astype(int)
# 2011 is reference year (omitted)
df_analysis['year_2013'] = ((df_analysis['YEAR'] == 2013) & (df_analysis['treatment'] == 1)).astype(int)
df_analysis['year_2014'] = ((df_analysis['YEAR'] == 2014) & (df_analysis['treatment'] == 1)).astype(int)
df_analysis['year_2015'] = ((df_analysis['YEAR'] == 2015) & (df_analysis['treatment'] == 1)).astype(int)
df_analysis['year_2016'] = ((df_analysis['YEAR'] == 2016) & (df_analysis['treatment'] == 1)).astype(int)

model_event = smf.ols('fulltime ~ treatment + C(YEAR) + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016',
                      data=df_analysis).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-"*50)
event_vars = ['year_2006', 'year_2007', 'year_2008', 'year_2009', 'year_2010',
              'year_2013', 'year_2014', 'year_2015', 'year_2016']
for var in event_vars:
    year = var.replace('year_', '')
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    print(f"   {year}: {coef:7.4f} (SE: {se:.4f}, p={pval:.4f})")

# ============================================================================
# 13. SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("13. SAVING RESULTS")
print("="*70)

# Save analysis dataset
df_analysis.to_csv('analysis_data.csv', index=False)
print("   Saved: analysis_data.csv")

# Create results summary
results_summary = {
    'Model': ['Basic DiD', 'Year FE', 'Year FE + Covariates', 'Year + State FE', 'Full Model', 'Weighted'],
    'Coefficient': [
        model1.params['treat_post'],
        model2.params['treat_post'],
        model3.params['treat_post'],
        model4.params['treat_post'],
        model5.params['treat_post'],
        model_weighted.params['treat_post']
    ],
    'Std_Error': [
        model1.bse['treat_post'],
        model2.bse['treat_post'],
        model3.bse['treat_post'],
        model4.bse['treat_post'],
        model5.bse['treat_post'],
        model_weighted.bse['treat_post']
    ],
    'P_value': [
        model1.pvalues['treat_post'],
        model2.pvalues['treat_post'],
        model3.pvalues['treat_post'],
        model4.pvalues['treat_post'],
        model5.pvalues['treat_post'],
        model_weighted.pvalues['treat_post']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model5.nobs),
        int(model_weighted.nobs)
    ]
}
results_df = pd.DataFrame(results_summary)
results_df.to_csv('regression_results.csv', index=False)
print("   Saved: regression_results.csv")

# ============================================================================
# 14. PREFERRED ESTIMATE SUMMARY
# ============================================================================
print("\n" + "="*70)
print("14. PREFERRED ESTIMATE (Model 3: Year FE + Covariates)")
print("="*70)

preferred_model = model3
print(f"\n   Effect Size: {preferred_model.params['treat_post']:.4f}")
print(f"   Standard Error: {preferred_model.bse['treat_post']:.4f}")
print(f"   95% Confidence Interval: [{preferred_model.conf_int().loc['treat_post', 0]:.4f}, {preferred_model.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   T-statistic: {preferred_model.tvalues['treat_post']:.4f}")
print(f"   P-value: {preferred_model.pvalues['treat_post']:.4f}")
print(f"   Sample Size: {int(preferred_model.nobs):,}")
print(f"   R-squared: {preferred_model.rsquared:.4f}")

# Store key results for LaTeX
with open('key_results.txt', 'w') as f:
    f.write("KEY RESULTS FOR LATEX REPORT\n")
    f.write("="*50 + "\n\n")
    f.write(f"Sample Size: {int(preferred_model.nobs):,}\n")
    f.write(f"Treatment Group N: {(df_analysis['treatment']==1).sum():,}\n")
    f.write(f"Control Group N: {(df_analysis['treatment']==0).sum():,}\n\n")
    f.write(f"Preferred Estimate (DiD coefficient): {preferred_model.params['treat_post']:.4f}\n")
    f.write(f"Standard Error: {preferred_model.bse['treat_post']:.4f}\n")
    f.write(f"95% CI Lower: {preferred_model.conf_int().loc['treat_post', 0]:.4f}\n")
    f.write(f"95% CI Upper: {preferred_model.conf_int().loc['treat_post', 1]:.4f}\n")
    f.write(f"P-value: {preferred_model.pvalues['treat_post']:.4f}\n\n")
    f.write(f"Mean FT Employment - Treatment Pre: {treatment_pre:.4f}\n")
    f.write(f"Mean FT Employment - Treatment Post: {treatment_post:.4f}\n")
    f.write(f"Mean FT Employment - Control Pre: {control_pre:.4f}\n")
    f.write(f"Mean FT Employment - Control Post: {control_post:.4f}\n")
    f.write(f"Simple DiD: {did_simple:.4f}\n")
print("   Saved: key_results.txt")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
