"""
DACA Replication Analysis
Independent Replication - Clean Room Approach

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the
United States, what was the causal impact of eligibility for the Deferred Action for
Childhood Arrivals (DACA) program on the probability that the eligible person is
employed full-time (usually working 35 hours per week or more)?

DACA was implemented on June 15, 2012. We examine effects on full-time employment
in 2013-2016.
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
print("DACA REPLICATION STUDY - INDEPENDENT ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n1. LOADING DATA...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations in raw data: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. SAMPLE RESTRICTIONS
# =============================================================================
print("\n2. APPLYING SAMPLE RESTRICTIONS...")

# 2a. Restrict to Hispanic-Mexican ethnicity (HISPAN == 1)
# HISPAN: 0=Not Hispanic, 1=Mexican, 2=Puerto Rican, 3=Cuban, 4=Other
df_hisp = df[df['HISPAN'] == 1].copy()
print(f"   After restricting to Hispanic-Mexican: {len(df_hisp):,}")

# 2b. Restrict to Mexican-born (BPL == 200)
# BPL 200 = Mexico
df_mex = df_hisp[df_hisp['BPL'] == 200].copy()
print(f"   After restricting to Mexican-born: {len(df_mex):,}")

# 2c. Restrict to non-citizens (potential undocumented)
# CITIZEN: 0=N/A, 1=Born abroad of American parents, 2=Naturalized,
#          3=Not a citizen, 4=Not citizen but first papers, 5=Foreign born unknown
# We use CITIZEN == 3 (not a citizen) as proxy for potentially undocumented
df_noncit = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"   After restricting to non-citizens: {len(df_noncit):,}")

# =============================================================================
# 3. DEFINE DACA ELIGIBILITY
# =============================================================================
print("\n3. DEFINING DACA ELIGIBILITY CRITERIA...")

"""
DACA Eligibility Requirements (as of June 15, 2012):
1. Arrived in US before 16th birthday
2. Under 31 years old as of June 15, 2012 (born after June 15, 1981)
3. Continuously resided in US since June 15, 2007 (arrived by 2007 or earlier)
4. Present in US on June 15, 2012 (non-citizen, not naturalized)

We will define:
- Treatment group: DACA-eligible (meet all criteria)
- Control group: Similar demographics but NOT eligible (age cutoff)
"""

# Calculate age as of June 15, 2012
# BIRTHYR contains birth year
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# We approximate: if BIRTHQTR <= 2, assume born in first half of year

df_noncit['age_june2012'] = 2012 - df_noncit['BIRTHYR']
# Adjust for those born in Q3-Q4 (they would be younger by 0.5 years on average in June)
# Those born in Q3-Q4 haven't had birthday yet by June 15
df_noncit.loc[df_noncit['BIRTHQTR'].isin([3, 4]), 'age_june2012'] = \
    df_noncit.loc[df_noncit['BIRTHQTR'].isin([3, 4]), 'age_june2012'] - 1

# DACA eligibility criterion 1: Under 31 as of June 15, 2012
# (born after June 15, 1981)
df_noncit['under_31_2012'] = df_noncit['age_june2012'] < 31

# DACA eligibility criterion 2: Arrived before 16th birthday
# Calculate age at immigration
df_noncit['age_at_immig'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']
df_noncit['arrived_before_16'] = df_noncit['age_at_immig'] < 16

# DACA eligibility criterion 3: In US continuously since June 15, 2007
# Approximation: arrived by 2007 or earlier
df_noncit['in_us_since_2007'] = df_noncit['YRIMMIG'] <= 2007

# Define DACA eligible (meeting all criteria)
df_noncit['daca_eligible'] = (
    df_noncit['under_31_2012'] &
    df_noncit['arrived_before_16'] &
    df_noncit['in_us_since_2007']
)

print(f"   DACA eligible: {df_noncit['daca_eligible'].sum():,}")
print(f"   Not DACA eligible: {(~df_noncit['daca_eligible']).sum():,}")

# =============================================================================
# 4. CREATE ANALYSIS SAMPLE
# =============================================================================
print("\n4. CREATING ANALYSIS SAMPLE...")

# Focus on working-age population (16-64)
df_noncit['current_age'] = df_noncit['YEAR'] - df_noncit['BIRTHYR']
df_work = df_noncit[(df_noncit['current_age'] >= 16) & (df_noncit['current_age'] <= 64)].copy()
print(f"   After restricting to ages 16-64: {len(df_work):,}")

# Exclude those with missing key variables
df_work = df_work[df_work['YRIMMIG'] > 0].copy()  # Valid immigration year
df_work = df_work[df_work['BIRTHYR'] > 0].copy()
print(f"   After excluding missing immigration year: {len(df_work):,}")

# =============================================================================
# 5. DEFINE OUTCOME VARIABLE
# =============================================================================
print("\n5. DEFINING OUTCOME VARIABLE...")

# Full-time employment: UHRSWORK >= 35 (usual hours worked per week)
# Also must be employed (EMPSTAT == 1)
df_work['employed'] = (df_work['EMPSTAT'] == 1).astype(int)
df_work['fulltime'] = ((df_work['UHRSWORK'] >= 35) & (df_work['EMPSTAT'] == 1)).astype(int)

print(f"   Employment rate: {df_work['employed'].mean():.3f}")
print(f"   Full-time employment rate: {df_work['fulltime'].mean():.3f}")

# =============================================================================
# 6. CREATE TREATMENT INDICATORS
# =============================================================================
print("\n6. CREATING TREATMENT INDICATORS...")

# Post-DACA period: 2013-2016 (DACA implemented June 2012)
# Pre-DACA period: 2006-2011 (2012 excluded due to ambiguity)
df_work['post_daca'] = (df_work['YEAR'] >= 2013).astype(int)

# Treatment indicator
df_work['treated'] = df_work['daca_eligible'].astype(int)

# Difference-in-differences interaction
df_work['did'] = df_work['treated'] * df_work['post_daca']

# Exclude 2012 from analysis (ambiguous treatment status)
df_analysis = df_work[df_work['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_analysis):,}")

print(f"\n   Sample by year:")
print(df_analysis.groupby('YEAR').size())

print(f"\n   Treatment group sizes:")
print(f"   Treated (DACA-eligible): {df_analysis['treated'].sum():,}")
print(f"   Control (Not eligible): {(1-df_analysis['treated']).sum():,}")

# =============================================================================
# 7. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("7. DESCRIPTIVE STATISTICS")
print("="*80)

# Create additional control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)
df_analysis['age_sq'] = df_analysis['current_age'] ** 2

# Education categories
df_analysis['less_than_hs'] = (df_analysis['EDUC'] < 6).astype(int)
df_analysis['hs_diploma'] = (df_analysis['EDUC'] == 6).astype(int)
df_analysis['some_college'] = (df_analysis['EDUC'].isin([7, 8, 9])).astype(int)
df_analysis['college_plus'] = (df_analysis['EDUC'] >= 10).astype(int)

# Summary statistics by treatment status
print("\nSummary Statistics by DACA Eligibility (Pre-DACA Period):")
pre_daca = df_analysis[df_analysis['post_daca'] == 0]

stats_vars = ['fulltime', 'employed', 'current_age', 'female', 'married',
              'less_than_hs', 'hs_diploma', 'some_college', 'college_plus']

summary_stats = pre_daca.groupby('treated')[stats_vars].mean()
summary_stats = summary_stats.T
summary_stats.columns = ['Control', 'Treated']
summary_stats['Difference'] = summary_stats['Treated'] - summary_stats['Control']
print(summary_stats.round(3))

# Sample sizes
print("\n\nSample Sizes by Treatment Status and Period:")
sample_size = df_analysis.groupby(['treated', 'post_daca']).size().unstack()
sample_size.columns = ['Pre-DACA', 'Post-DACA']
sample_size.index = ['Control', 'Treated']
print(sample_size)

# =============================================================================
# 8. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("8. DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

# 8a. Simple DiD (2x2 comparison)
print("\n8a. Simple Difference-in-Differences (Unweighted):")

# Calculate group means
means = df_analysis.groupby(['treated', 'post_daca'])['fulltime'].mean().unstack()
means.columns = ['Pre', 'Post']
means.index = ['Control', 'Treated']
print("\nGroup Means (Full-time Employment Rate):")
print(means.round(4))

# Calculate DiD
dd_control = means.loc['Control', 'Post'] - means.loc['Control', 'Pre']
dd_treated = means.loc['Treated', 'Post'] - means.loc['Treated', 'Pre']
did_estimate = dd_treated - dd_control

print(f"\nChange for Control: {dd_control:.4f}")
print(f"Change for Treated: {dd_treated:.4f}")
print(f"Difference-in-Differences Estimate: {did_estimate:.4f}")

# 8b. DiD Regression (no controls)
print("\n8b. DiD Regression (No Controls):")
model1 = smf.ols('fulltime ~ treated + post_daca + did', data=df_analysis).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(model1.summary2().tables[1].to_string())

# 8c. DiD Regression with controls
print("\n8c. DiD Regression (With Controls):")
model2 = smf.ols('fulltime ~ treated + post_daca + did + current_age + age_sq + female + married + C(EDUC) + C(STATEFIP)',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

# Extract key coefficients
print("\nKey Coefficients:")
print(f"{'Variable':<20} {'Coef':<12} {'Std Err':<12} {'t':<10} {'P>|t|':<10}")
print("-"*64)
for var in ['treated', 'post_daca', 'did']:
    print(f"{var:<20} {model2.params[var]:<12.4f} {model2.bse[var]:<12.4f} {model2.tvalues[var]:<10.2f} {model2.pvalues[var]:<10.4f}")

# 8d. DiD Regression with year fixed effects
print("\n8d. DiD Regression (Year Fixed Effects):")
model3 = smf.ols('fulltime ~ treated + did + current_age + age_sq + female + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

print("\nKey Coefficients:")
print(f"{'Variable':<20} {'Coef':<12} {'Std Err':<12} {'t':<10} {'P>|t|':<10}")
print("-"*64)
for var in ['treated', 'did']:
    print(f"{var:<20} {model3.params[var]:<12.4f} {model3.bse[var]:<12.4f} {model3.tvalues[var]:<10.2f} {model3.pvalues[var]:<10.4f}")

# 8e. Weighted analysis
print("\n8e. DiD Regression (Weighted, With Controls):")
model4 = smf.wls('fulltime ~ treated + post_daca + did + current_age + age_sq + female + married + C(EDUC) + C(STATEFIP)',
                 data=df_analysis, weights=df_analysis['PERWT']).fit(
                     cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

print("\nKey Coefficients:")
print(f"{'Variable':<20} {'Coef':<12} {'Std Err':<12} {'t':<10} {'P>|t|':<10}")
print("-"*64)
for var in ['treated', 'post_daca', 'did']:
    print(f"{var:<20} {model4.params[var]:<12.4f} {model4.bse[var]:<12.4f} {model4.tvalues[var]:<10.2f} {model4.pvalues[var]:<10.4f}")

# =============================================================================
# 9. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("9. ROBUSTNESS CHECKS")
print("="*80)

# 9a. Alternative control group - use slightly older non-eligible cohort
print("\n9a. Robustness: Restricted Age Range (20-40):")
df_restricted = df_analysis[(df_analysis['current_age'] >= 20) & (df_analysis['current_age'] <= 40)].copy()
model5 = smf.ols('fulltime ~ treated + post_daca + did + current_age + age_sq + female + married + C(EDUC) + C(STATEFIP)',
                 data=df_restricted).fit(cov_type='cluster', cov_kwds={'groups': df_restricted['STATEFIP']})

print(f"Sample size: {len(df_restricted):,}")
print(f"DiD coefficient: {model5.params['did']:.4f} (SE: {model5.bse['did']:.4f})")
print(f"95% CI: [{model5.conf_int().loc['did', 0]:.4f}, {model5.conf_int().loc['did', 1]:.4f}]")

# 9b. Employment (any hours) as outcome
print("\n9b. Robustness: Employment (Any Hours) as Outcome:")
model6 = smf.ols('employed ~ treated + post_daca + did + current_age + age_sq + female + married + C(EDUC) + C(STATEFIP)',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

print(f"DiD coefficient: {model6.params['did']:.4f} (SE: {model6.bse['did']:.4f})")
print(f"95% CI: [{model6.conf_int().loc['did', 0]:.4f}, {model6.conf_int().loc['did', 1]:.4f}]")

# 9c. By gender
print("\n9c. Robustness: By Gender:")
for gender, label in [(0, 'Male'), (1, 'Female')]:
    df_gender = df_analysis[df_analysis['female'] == gender]
    model_g = smf.ols('fulltime ~ treated + post_daca + did + current_age + age_sq + married + C(EDUC) + C(STATEFIP)',
                      data=df_gender).fit(cov_type='cluster', cov_kwds={'groups': df_gender['STATEFIP']})
    print(f"{label}: DiD = {model_g.params['did']:.4f} (SE: {model_g.bse['did']:.4f}), n = {len(df_gender):,}")

# 9d. Placebo test - use 2008 as fake treatment year
print("\n9d. Placebo Test (Fake Treatment in 2008):")
df_placebo = df_analysis[df_analysis['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011])].copy()
df_placebo['post_fake'] = (df_placebo['YEAR'] >= 2008).astype(int)
df_placebo['did_fake'] = df_placebo['treated'] * df_placebo['post_fake']

model_placebo = smf.ols('fulltime ~ treated + post_fake + did_fake + current_age + age_sq + female + married + C(EDUC) + C(STATEFIP)',
                        data=df_placebo).fit(cov_type='cluster', cov_kwds={'groups': df_placebo['STATEFIP']})
print(f"Placebo DiD coefficient: {model_placebo.params['did_fake']:.4f} (SE: {model_placebo.bse['did_fake']:.4f})")
print(f"P-value: {model_placebo.pvalues['did_fake']:.4f}")

# =============================================================================
# 10. EVENT STUDY
# =============================================================================
print("\n" + "="*80)
print("10. EVENT STUDY ANALYSIS")
print("="*80)

# Create year dummies interacted with treatment
df_analysis['year_2006'] = (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['year_2007'] = (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['year_2008'] = (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['year_2009'] = (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['year_2010'] = (df_analysis['YEAR'] == 2010).astype(int)
df_analysis['year_2011'] = (df_analysis['YEAR'] == 2011).astype(int)
df_analysis['year_2013'] = (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['year_2014'] = (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['year_2015'] = (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['year_2016'] = (df_analysis['YEAR'] == 2016).astype(int)

# Interactions (omit 2011 as reference)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'treat_x_{year}'] = df_analysis['treated'] * df_analysis[f'year_{year}']

event_formula = ('fulltime ~ treated + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + '
                 'year_2013 + year_2014 + year_2015 + year_2016 + '
                 'treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + '
                 'treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + '
                 'current_age + age_sq + female + married + C(EDUC) + C(STATEFIP)')

model_event = smf.ols(event_formula, data=df_analysis).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

print("\nEvent Study Coefficients (Reference Year: 2011):")
print(f"{'Year':<10} {'Coef':<12} {'Std Err':<12} {'95% CI':<25}")
print("-"*60)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_x_{year}'
    ci = model_event.conf_int().loc[var]
    print(f"{year:<10} {model_event.params[var]:<12.4f} {model_event.bse[var]:<12.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

# =============================================================================
# 11. PREFERRED ESTIMATE SUMMARY
# =============================================================================
print("\n" + "="*80)
print("11. PREFERRED ESTIMATE SUMMARY")
print("="*80)

# Preferred model: Model 4 (weighted with controls)
print("\nPREFERRED SPECIFICATION: Weighted DiD with Controls")
print(f"Effect Size (DiD coefficient): {model4.params['did']:.4f}")
print(f"Standard Error: {model4.bse['did']:.4f}")
print(f"95% Confidence Interval: [{model4.conf_int().loc['did', 0]:.4f}, {model4.conf_int().loc['did', 1]:.4f}]")
print(f"t-statistic: {model4.tvalues['did']:.2f}")
print(f"p-value: {model4.pvalues['did']:.4f}")
print(f"Sample Size: {len(df_analysis):,}")
print(f"Number of Clusters (States): {df_analysis['STATEFIP'].nunique()}")

# =============================================================================
# 12. SAVE RESULTS
# =============================================================================
print("\n" + "="*80)
print("12. SAVING RESULTS")
print("="*80)

# Save summary statistics
summary_stats.to_csv('summary_statistics.csv')
print("Saved: summary_statistics.csv")

# Save regression results
results_dict = {
    'Model': ['Simple DiD', 'DiD + Controls', 'DiD + Year FE', 'Weighted DiD',
              'Restricted Age', 'Employment Outcome', 'Placebo Test'],
    'Coefficient': [model1.params['did'], model2.params['did'], model3.params['did'],
                    model4.params['did'], model5.params['did'], model6.params['did'],
                    model_placebo.params['did_fake']],
    'Std_Error': [model1.bse['did'], model2.bse['did'], model3.bse['did'],
                  model4.bse['did'], model5.bse['did'], model6.bse['did'],
                  model_placebo.bse['did_fake']],
    'P_Value': [model1.pvalues['did'], model2.pvalues['did'], model3.pvalues['did'],
                model4.pvalues['did'], model5.pvalues['did'], model6.pvalues['did'],
                model_placebo.pvalues['did_fake']],
    'N': [len(df_analysis), len(df_analysis), len(df_analysis),
          len(df_analysis), len(df_restricted), len(df_analysis),
          len(df_placebo)]
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv('regression_results.csv', index=False)
print("Saved: regression_results.csv")

# Save event study results
event_results = []
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_x_{year}'
    ci = model_event.conf_int().loc[var]
    event_results.append({
        'Year': year,
        'Coefficient': model_event.params[var],
        'Std_Error': model_event.bse[var],
        'CI_Lower': ci[0],
        'CI_Upper': ci[1]
    })
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Saved: event_study_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
