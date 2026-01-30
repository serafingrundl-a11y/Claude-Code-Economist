"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
for Hispanic-Mexican, Mexican-born individuals in the US

Treatment: DACA-eligible individuals aged 26-30 at policy implementation (June 2012)
Control: Individuals aged 31-35 at policy implementation who would otherwise be eligible

Outcome: Full-time employment (usually working 35+ hours per week)
Method: Difference-in-Differences
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
print("DACA REPLICATION ANALYSIS")
print("="*80)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv('data/data.csv')
print(f"Total observations in raw data: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# Examine key variables
print("\n2. EXAMINING KEY VARIABLES...")
print(f"\nHISPAN (Hispanic origin): {dict(df['HISPAN'].value_counts().sort_index())}")
print(f"\nBPL (Birthplace) - Mexico is 200: {(df['BPL'] == 200).sum():,} born in Mexico")
print(f"\nCITIZEN: {dict(df['CITIZEN'].value_counts().sort_index())}")

# DACA Eligibility Criteria from instructions:
# 1. Hispanic-Mexican ethnicity (HISPAN == 1)
# 2. Born in Mexico (BPL == 200)
# 3. Not a citizen (CITIZEN == 3 means "not a citizen")
# 4. Arrived in US before age 16 (need to calculate)
# 5. Lived continuously in US since June 15, 2007 (approximated by YRIMMIG <= 2007)
# 6. Present in US on June 15, 2012

# Treatment group: Age 26-30 on June 15, 2012 (born 1982-1986)
# Control group: Age 31-35 on June 15, 2012 (born 1977-1981)

print("\n3. CONSTRUCTING DACA ELIGIBILITY SAMPLE...")

# Filter for Hispanic-Mexican ethnicity (HISPAN == 1)
df_mex = df[df['HISPAN'] == 1].copy()
print(f"After filtering for Hispanic-Mexican: {len(df_mex):,}")

# Filter for born in Mexico (BPL == 200)
df_mex = df_mex[df_mex['BPL'] == 200].copy()
print(f"After filtering for Mexico-born: {len(df_mex):,}")

# Filter for non-citizens (CITIZEN == 3)
# CITIZEN codes: 0=N/A, 1=Born abroad of American parents, 2=Naturalized, 3=Not a citizen
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"After filtering for non-citizens: {len(df_mex):,}")

# Calculate age at June 15, 2012
# Using BIRTHYR to determine age on June 15, 2012
# Someone born in 1982 would be 30 in 2012 (if birthday before June 15) or 29 (if after)
# Someone born in 1986 would be 26 in 2012 (if birthday before June 15) or 25 (if after)
# We'll use birth year to approximate - treat everyone as if mid-year birth

df_mex['age_at_daca'] = 2012 - df_mex['BIRTHYR']
print(f"\nAge at DACA implementation (2012) distribution:")
print(df_mex['age_at_daca'].describe())

# Create age groups
# Treatment: 26-30 at DACA (born 1982-1986)
# Control: 31-35 at DACA (born 1977-1981)

# Filter to ages 26-35 at DACA implementation
df_analysis = df_mex[(df_mex['age_at_daca'] >= 26) & (df_mex['age_at_daca'] <= 35)].copy()
print(f"\nAfter filtering for ages 26-35 at DACA: {len(df_analysis):,}")

# Arrived before age 16
# Age at arrival = YRIMMIG - BIRTHYR (approximately)
# Need to have arrived before 16th birthday
df_analysis['age_at_arrival'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']
df_analysis = df_analysis[df_analysis['age_at_arrival'] < 16].copy()
print(f"After filtering for arrived before age 16: {len(df_analysis):,}")

# Lived in US since June 2007 (approximated by immigration year <= 2007)
df_analysis = df_analysis[df_analysis['YRIMMIG'] <= 2007].copy()
print(f"After filtering for in US since 2007: {len(df_analysis):,}")

# Create treatment indicator (1 if age 26-30 at DACA, 0 if age 31-35)
df_analysis['treated'] = (df_analysis['age_at_daca'] <= 30).astype(int)
print(f"\nTreatment group (26-30): {(df_analysis['treated'] == 1).sum():,}")
print(f"Control group (31-35): {(df_analysis['treated'] == 0).sum():,}")

# Create post indicator (1 if year >= 2013, 0 if year <= 2011)
# Exclude 2012 as it's the transition year
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
print(f"\nPre-period (2006-2011): {(df_analysis['post'] == 0).sum():,}")
print(f"Post-period (2013-2016): {(df_analysis['post'] == 1).sum():,}")

# Create outcome: Full-time employment (UHRSWORK >= 35)
# UHRSWORK is usual hours worked per week
# 0 = N/A (not in labor force or not working)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)
print(f"\nFull-time employed: {df_analysis['fulltime'].mean():.3f} ({df_analysis['fulltime'].sum():,})")

# Create DiD interaction
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

print("\n" + "="*80)
print("4. SAMPLE SUMMARY STATISTICS")
print("="*80)

# Summary by treatment and period
print("\nFull-time employment rates by group and period:")
summary = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'PERWT': 'sum'
}).round(4)
summary.columns = ['Mean FT', 'N', 'Sum Weights']
print(summary)

# Calculate simple DiD
treat_pre = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==0)]['fulltime'].mean()
treat_post = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==1)]['fulltime'].mean()
ctrl_pre = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==0)]['fulltime'].mean()
ctrl_post = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==1)]['fulltime'].mean()

print(f"\nSimple DiD calculation:")
print(f"Treatment group pre:  {treat_pre:.4f}")
print(f"Treatment group post: {treat_post:.4f}")
print(f"Treatment change:     {treat_post - treat_pre:.4f}")
print(f"Control group pre:    {ctrl_pre:.4f}")
print(f"Control group post:   {ctrl_post:.4f}")
print(f"Control change:       {ctrl_post - ctrl_pre:.4f}")
print(f"DiD estimate:         {(treat_post - treat_pre) - (ctrl_post - ctrl_pre):.4f}")

print("\n" + "="*80)
print("5. DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("="*80)

# Basic DiD regression (unweighted)
print("\n5.1 Basic DiD (OLS, unweighted):")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_analysis).fit()
print(model1.summary().tables[1])

# DiD with robust standard errors
print("\n5.2 DiD with robust (HC1) standard errors:")
model2 = smf.ols('fulltime ~ treated + post + treated_post', data=df_analysis).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Weighted DiD regression
print("\n5.3 Weighted DiD (using PERWT):")
model3 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(model3.summary().tables[1])

# Weighted with robust SE
print("\n5.4 Weighted DiD with robust SE:")
model4 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model4.summary().tables[1])

print("\n" + "="*80)
print("6. REGRESSION WITH COVARIATES")
print("="*80)

# Add covariates
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = df_analysis['MARST'].isin([1, 2]).astype(int)

# Education categories
df_analysis['educ_less_hs'] = (df_analysis['EDUC'] < 6).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] == 6).astype(int)
df_analysis['educ_some_college'] = df_analysis['EDUC'].isin([7, 8, 9]).astype(int)
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)

# Current age in survey year
df_analysis['age_survey'] = df_analysis['AGE']
df_analysis['age_sq'] = df_analysis['age_survey'] ** 2

print("\n6.1 DiD with demographic controls:")
model5 = smf.wls('fulltime ~ treated + post + treated_post + female + married + age_survey + age_sq',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model5.summary().tables[1])

print("\n6.2 DiD with demographic and education controls:")
model6 = smf.wls('fulltime ~ treated + post + treated_post + female + married + age_survey + age_sq + educ_hs + educ_some_college + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model6.summary().tables[1])

# Year fixed effects
df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)
print("\n6.3 DiD with year fixed effects:")
model7 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + age_survey + age_sq + educ_hs + educ_some_college + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
# Extract key coefficients
print(f"treated_post coefficient: {model7.params['treated_post']:.4f}")
print(f"treated_post SE: {model7.bse['treated_post']:.4f}")
print(f"treated_post 95% CI: [{model7.conf_int().loc['treated_post', 0]:.4f}, {model7.conf_int().loc['treated_post', 1]:.4f}]")

print("\n" + "="*80)
print("7. ROBUSTNESS CHECKS")
print("="*80)

# State fixed effects
print("\n7.1 DiD with state fixed effects:")
model8 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + age_survey + age_sq + educ_hs + educ_some_college + educ_college + C(STATEFIP)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treated_post coefficient: {model8.params['treated_post']:.4f}")
print(f"treated_post SE: {model8.bse['treated_post']:.4f}")
print(f"treated_post 95% CI: [{model8.conf_int().loc['treated_post', 0]:.4f}, {model8.conf_int().loc['treated_post', 1]:.4f}]")

# Clustered standard errors at state level
print("\n7.2 DiD with state-clustered standard errors:")
model9 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + age_survey + age_sq + educ_hs + educ_some_college + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"treated_post coefficient: {model9.params['treated_post']:.4f}")
print(f"treated_post SE: {model9.bse['treated_post']:.4f}")
print(f"treated_post 95% CI: [{model9.conf_int().loc['treated_post', 0]:.4f}, {model9.conf_int().loc['treated_post', 1]:.4f}]")

print("\n" + "="*80)
print("8. HETEROGENEITY ANALYSIS")
print("="*80)

# By gender
print("\n8.1 Effects by gender:")
for sex, name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex]
    model_sub = smf.wls('fulltime ~ treated + post + treated_post',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"{name}: DiD = {model_sub.params['treated_post']:.4f} (SE: {model_sub.bse['treated_post']:.4f})")

# By education
print("\n8.2 Effects by education:")
df_analysis['high_educ'] = (df_analysis['EDUC'] >= 6).astype(int)
for educ, name in [(0, 'Less than HS'), (1, 'HS or more')]:
    df_sub = df_analysis[df_analysis['high_educ'] == educ]
    if len(df_sub) > 100:
        model_sub = smf.wls('fulltime ~ treated + post + treated_post',
                            data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
        print(f"{name}: DiD = {model_sub.params['treated_post']:.4f} (SE: {model_sub.bse['treated_post']:.4f})")

print("\n" + "="*80)
print("9. EVENT STUDY / PARALLEL TRENDS")
print("="*80)

# Create year-specific treatment effects
years = sorted(df_analysis['YEAR'].unique())
ref_year = 2011  # Reference year (last pre-treatment year)

df_analysis['treated_2006'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['treated_2007'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['treated_2008'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['treated_2009'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['treated_2010'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2010).astype(int)
# 2011 is reference year
df_analysis['treated_2013'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['treated_2014'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['treated_2015'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['treated_2016'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2016).astype(int)

print("Event study estimates (reference year: 2011):")
model_event = smf.wls('''fulltime ~ treated + C(YEAR) +
                         treated_2006 + treated_2007 + treated_2008 + treated_2009 + treated_2010 +
                         treated_2013 + treated_2014 + treated_2015 + treated_2016 +
                         female + married + age_survey + age_sq''',
                      data=df_analysis,
                      weights=df_analysis['PERWT']).fit(cov_type='HC1')

event_vars = ['treated_2006', 'treated_2007', 'treated_2008', 'treated_2009', 'treated_2010',
              'treated_2013', 'treated_2014', 'treated_2015', 'treated_2016']
print(f"\n{'Year':<12} {'Coef':>10} {'SE':>10} {'95% CI':>24}")
print("-"*60)
for var in event_vars:
    year = var.split('_')[1]
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_low, ci_high = model_event.conf_int().loc[var]
    print(f"{year:<12} {coef:>10.4f} {se:>10.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}]")

print("\n" + "="*80)
print("10. FINAL RESULTS SUMMARY")
print("="*80)

# Preferred specification: Weighted DiD with demographic controls and year FE, clustered SE
print("\nPREFERRED SPECIFICATION: Weighted DiD with controls, year FE, state-clustered SE")
print("-"*70)
print(f"DiD Estimate (DACA effect on full-time employment): {model9.params['treated_post']:.4f}")
print(f"Standard Error: {model9.bse['treated_post']:.4f}")
ci = model9.conf_int().loc['treated_post']
print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"p-value: {model9.pvalues['treated_post']:.4f}")
print(f"Sample size: {len(df_analysis):,}")

# Save key results to file
results_dict = {
    'estimate': model9.params['treated_post'],
    'se': model9.bse['treated_post'],
    'ci_low': ci[0],
    'ci_high': ci[1],
    'pvalue': model9.pvalues['treated_post'],
    'n': len(df_analysis),
    'n_treated_pre': len(df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==0)]),
    'n_treated_post': len(df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==1)]),
    'n_control_pre': len(df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==0)]),
    'n_control_post': len(df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==1)])
}

print("\n" + "="*80)
print("SAMPLE SIZES BY GROUP")
print("="*80)
print(f"Treatment group, pre-period:  {results_dict['n_treated_pre']:,}")
print(f"Treatment group, post-period: {results_dict['n_treated_post']:,}")
print(f"Control group, pre-period:    {results_dict['n_control_pre']:,}")
print(f"Control group, post-period:   {results_dict['n_control_post']:,}")
print(f"Total sample size:            {results_dict['n']:,}")

# Additional descriptive statistics for report
print("\n" + "="*80)
print("11. DESCRIPTIVE STATISTICS FOR REPORT")
print("="*80)

desc_vars = ['fulltime', 'female', 'married', 'age_survey', 'educ_less_hs', 'educ_hs', 'educ_some_college', 'educ_college']
print("\nMean values by treatment status (pre-period only):")
pre_data = df_analysis[df_analysis['post'] == 0]
print(f"\n{'Variable':<20} {'Treatment':>12} {'Control':>12} {'Diff':>12}")
print("-"*60)
for var in desc_vars:
    t_mean = pre_data[pre_data['treated']==1][var].mean()
    c_mean = pre_data[pre_data['treated']==0][var].mean()
    print(f"{var:<20} {t_mean:>12.3f} {c_mean:>12.3f} {t_mean-c_mean:>12.3f}")

# Save analysis results for LaTeX
print("\n\nSaving results for LaTeX tables...")

# Export key tables
results_for_latex = {
    'basic_did': {
        'coef': model1.params['treated_post'],
        'se': model1.bse['treated_post'],
        'n': len(df_analysis)
    },
    'weighted_did': {
        'coef': model4.params['treated_post'],
        'se': model4.bse['treated_post'],
        'n': len(df_analysis)
    },
    'with_controls': {
        'coef': model6.params['treated_post'],
        'se': model6.bse['treated_post'],
        'n': len(df_analysis)
    },
    'year_fe': {
        'coef': model7.params['treated_post'],
        'se': model7.bse['treated_post'],
        'n': len(df_analysis)
    },
    'state_fe': {
        'coef': model8.params['treated_post'],
        'se': model8.bse['treated_post'],
        'n': len(df_analysis)
    },
    'preferred': {
        'coef': model9.params['treated_post'],
        'se': model9.bse['treated_post'],
        'ci_low': ci[0],
        'ci_high': ci[1],
        'n': len(df_analysis)
    }
}

# Print for LaTeX
print("\n" + "="*80)
print("LATEX TABLE DATA")
print("="*80)
print("\nMain Results Table:")
print(f"{'Specification':<30} {'Coefficient':>12} {'SE':>12}")
print("-"*55)
for spec, vals in results_for_latex.items():
    print(f"{spec:<30} {vals['coef']:>12.4f} {vals['se']:>12.4f}")

print("\n\nAnalysis complete!")
