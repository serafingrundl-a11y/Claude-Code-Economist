"""
DACA Replication Study - Analysis Script
Participant ID: 92

Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals?

Treatment group: Ages 26-30 as of June 15, 2012
Control group: Ages 31-35 as of June 15, 2012
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
# 1. LOAD DATA
# =============================================================================
print("\n1. Loading data...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. SAMPLE SELECTION
# =============================================================================
print("\n2. Applying sample selection criteria...")

# 2.1 Hispanic-Mexican ethnicity (HISPAN == 1)
df_sample = df[df['HISPAN'] == 1].copy()
print(f"   After Hispanic-Mexican filter: {len(df_sample):,}")

# 2.2 Born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"   After Mexico birthplace filter: {len(df_sample):,}")

# 2.3 Not a citizen (CITIZEN == 3 or 4 or 5, treating as potentially undocumented)
# Per instructions: "Assume anyone who is not a citizen and who has not received
# immigration papers is undocumented"
# CITIZEN: 3 = Not a citizen, 4 = Not a citizen but received first papers
# We use CITIZEN == 3 (clearly not a citizen with no papers)
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"   After non-citizen filter: {len(df_sample):,}")

# 2.4 Exclude 2012 (DACA implemented mid-year, cannot distinguish pre/post)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_sample):,}")

# =============================================================================
# 3. CALCULATE AGE AS OF JUNE 15, 2012
# =============================================================================
print("\n3. Calculating age as of June 15, 2012...")

# Age as of June 15, 2012 depends on birth year and quarter
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born Q1 or Q2 (Jan-June), they've had their birthday by June 15
# If born Q3 or Q4 (Jul-Dec), they haven't had their birthday yet

def calc_age_june_2012(row):
    """Calculate age as of June 15, 2012"""
    birthyr = row['BIRTHYR']
    birthqtr = row['BIRTHQTR']

    if birthqtr in [1, 2]:  # Born Jan-June, birthday already passed
        return 2012 - birthyr
    else:  # Born Jul-Dec, birthday not yet reached
        return 2012 - birthyr - 1

df_sample['age_june_2012'] = df_sample.apply(calc_age_june_2012, axis=1)

print(f"   Age range in sample: {df_sample['age_june_2012'].min()} to {df_sample['age_june_2012'].max()}")

# =============================================================================
# 4. DEFINE TREATMENT AND CONTROL GROUPS
# =============================================================================
print("\n4. Defining treatment and control groups...")

# Treatment: Ages 26-30 as of June 15, 2012
# Control: Ages 31-35 as of June 15, 2012

df_sample['treated'] = ((df_sample['age_june_2012'] >= 26) &
                        (df_sample['age_june_2012'] <= 30)).astype(int)
df_sample['control'] = ((df_sample['age_june_2012'] >= 31) &
                        (df_sample['age_june_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df_sample[(df_sample['treated'] == 1) | (df_sample['control'] == 1)].copy()
print(f"   Treatment group (26-30): {df_analysis['treated'].sum():,}")
print(f"   Control group (31-35): {(df_analysis['control'] == 1).sum():,}")
print(f"   Total analysis sample: {len(df_analysis):,}")

# =============================================================================
# 5. APPLY ADDITIONAL DACA ELIGIBILITY CRITERIA
# =============================================================================
print("\n5. Applying additional DACA eligibility criteria...")

# 5.1 Must have arrived in US before age 16
# Calculate age at arrival: YRSUSA1 is years in USA as of survey year
# Age at arrival = AGE (at survey) - YRSUSA1
# But we need to check if they arrived before turning 16

# For YRIMMIG, we can calculate age at immigration
df_analysis['age_at_immigration'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']

# Filter: arrived before age 16
df_analysis = df_analysis[df_analysis['age_at_immigration'] < 16].copy()
print(f"   After arrived before age 16 filter: {len(df_analysis):,}")

# 5.2 Must have been in US since June 15, 2007 (continuous presence)
# YRIMMIG <= 2007 (we use 2007 as cutoff since we can't verify month)
df_analysis = df_analysis[df_analysis['YRIMMIG'] <= 2007].copy()
print(f"   After continuous presence (immigrated by 2007) filter: {len(df_analysis):,}")

# Check sample sizes
print(f"\n   Final treatment group (26-30): {df_analysis['treated'].sum():,}")
print(f"   Final control group (31-35): {(df_analysis['treated'] == 0).sum():,}")

# =============================================================================
# 6. CREATE OUTCOME AND ANALYSIS VARIABLES
# =============================================================================
print("\n6. Creating outcome and analysis variables...")

# Outcome: Full-time employment (UHRSWORK >= 35)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Post-treatment indicator (2013-2016)
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Pre-treatment years: 2006-2011
# Post-treatment years: 2013-2016

print(f"   Full-time employment rate: {df_analysis['fulltime'].mean():.3f}")
print(f"   Pre-period observations: {(df_analysis['post'] == 0).sum():,}")
print(f"   Post-period observations: {(df_analysis['post'] == 1).sum():,}")

# =============================================================================
# 7. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("7. DESCRIPTIVE STATISTICS")
print("="*80)

# Summary by group and period
summary = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum',
    'AGE': 'mean',
    'SEX': lambda x: (x == 1).mean(),  # Male proportion
    'EDUC': 'mean'
}).round(4)

print("\nSummary by Treatment Group and Period:")
print(summary)

# Weighted means
print("\n\nWeighted Full-Time Employment Rates:")
for treated in [0, 1]:
    for post in [0, 1]:
        mask = (df_analysis['treated'] == treated) & (df_analysis['post'] == post)
        subset = df_analysis[mask]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        group = "Treatment" if treated else "Control"
        period = "Post" if post else "Pre"
        print(f"   {group}, {period}: {weighted_mean:.4f} (N={len(subset):,})")

# =============================================================================
# 8. SIMPLE DIFFERENCE-IN-DIFFERENCES
# =============================================================================
print("\n" + "="*80)
print("8. SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("="*80)

# Calculate means for each cell
means = df_analysis.groupby(['treated', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

print("\n2x2 Table of Weighted Full-Time Employment Rates:")
print(f"                    Pre         Post        Difference")
print(f"Treatment (26-30):  {means.loc[1, 0]:.4f}      {means.loc[1, 1]:.4f}      {means.loc[1, 1] - means.loc[1, 0]:.4f}")
print(f"Control (31-35):    {means.loc[0, 0]:.4f}      {means.loc[0, 1]:.4f}      {means.loc[0, 1] - means.loc[0, 0]:.4f}")

# DiD estimate
did_estimate = (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])
print(f"\nDiD Estimate: {did_estimate:.4f}")

# =============================================================================
# 9. REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES
# =============================================================================
print("\n" + "="*80)
print("9. REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES")
print("="*80)

# 9.1 Basic DiD regression (unweighted)
print("\n9.1 Basic DiD Regression (Unweighted):")
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_analysis).fit()
print(model1.summary().tables[1])

# 9.2 Weighted DiD regression
print("\n9.2 Weighted DiD Regression:")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_analysis, weights=df_analysis['PERWT']).fit()
print(model2.summary().tables[1])

# 9.3 DiD with covariates
print("\n9.3 Weighted DiD with Covariates:")
# Add covariates: sex, education, marital status
df_analysis['male'] = (df_analysis['SEX'] == 1).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

model3 = smf.wls('fulltime ~ treated + post + treated_post + male + married + C(EDUC)',
                  data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"\nDiD Coefficient (treated_post): {model3.params['treated_post']:.4f}")
print(f"Standard Error: {model3.bse['treated_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")

# 9.4 DiD with year fixed effects
print("\n9.4 Weighted DiD with Year Fixed Effects:")
model4 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + male + married + C(EDUC)',
                  data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"\nDiD Coefficient (treated_post): {model4.params['treated_post']:.4f}")
print(f"Standard Error: {model4.bse['treated_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")

# 9.5 DiD with state fixed effects
print("\n9.5 Weighted DiD with State and Year Fixed Effects:")
model5 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + male + married + C(EDUC)',
                  data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"\nDiD Coefficient (treated_post): {model5.params['treated_post']:.4f}")
print(f"Standard Error: {model5.bse['treated_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")

# =============================================================================
# 10. ROBUST STANDARD ERRORS
# =============================================================================
print("\n" + "="*80)
print("10. ROBUST STANDARD ERRORS")
print("="*80)

# Cluster at state level
print("\n10.1 DiD with Robust (HC1) Standard Errors:")
model_robust = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + male + married + C(EDUC)',
                        data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"\nDiD Coefficient (treated_post): {model_robust.params['treated_post']:.4f}")
print(f"Robust Standard Error: {model_robust.bse['treated_post']:.4f}")
print(f"95% CI: [{model_robust.conf_int().loc['treated_post', 0]:.4f}, {model_robust.conf_int().loc['treated_post', 1]:.4f}]")

# =============================================================================
# 11. PRE-TRENDS ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("11. PRE-TRENDS ANALYSIS")
print("="*80)

# Create year-specific treatment effects (event study)
df_analysis['year_treated'] = df_analysis['YEAR'].astype(str) + '_' + df_analysis['treated'].astype(str)

# Calculate yearly means by group
yearly_means = df_analysis.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

print("\nYearly Full-Time Employment Rates by Group:")
print(yearly_means.round(4))

# Calculate difference between treatment and control each year
yearly_diff = yearly_means[1] - yearly_means[0]
print("\nYearly Difference (Treatment - Control):")
print(yearly_diff.round(4))

# =============================================================================
# 12. EVENT STUDY SPECIFICATION
# =============================================================================
print("\n" + "="*80)
print("12. EVENT STUDY SPECIFICATION")
print("="*80)

# Create interaction terms for each year with treatment
# Reference year: 2011 (last pre-treatment year)
df_analysis['year_2006'] = (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['year_2007'] = (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['year_2008'] = (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['year_2009'] = (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['year_2010'] = (df_analysis['YEAR'] == 2010).astype(int)
# 2011 is reference
df_analysis['year_2013'] = (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['year_2014'] = (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['year_2015'] = (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['year_2016'] = (df_analysis['YEAR'] == 2016).astype(int)

# Create interaction terms
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'treat_x_{year}'] = df_analysis['treated'] * df_analysis[f'year_{year}']

event_study = smf.wls('''fulltime ~ treated + C(YEAR) +
                         treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 +
                         treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 +
                         male + married + C(EDUC)''',
                       data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
event_vars = ['treat_x_2006', 'treat_x_2007', 'treat_x_2008', 'treat_x_2009', 'treat_x_2010',
              'treat_x_2013', 'treat_x_2014', 'treat_x_2015', 'treat_x_2016']

print(f"{'Year':<12} {'Coef':>10} {'SE':>10} {'95% CI':>24}")
print("-" * 60)
for var in event_vars:
    year = var.split('_')[-1]
    coef = event_study.params[var]
    se = event_study.bse[var]
    ci_low = event_study.conf_int().loc[var, 0]
    ci_high = event_study.conf_int().loc[var, 1]
    print(f"{year:<12} {coef:>10.4f} {se:>10.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}]")

# =============================================================================
# 13. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("13. ROBUSTNESS CHECKS")
print("="*80)

# 13.1 Alternative age bandwidths
print("\n13.1 Alternative Age Bandwidths:")

for bandwidth in [3, 4, 5]:
    df_robust = df_sample.copy()
    age_min_treat = 26
    age_max_treat = 26 + bandwidth - 1
    age_min_control = 31
    age_max_control = 31 + bandwidth - 1

    df_robust['treated'] = ((df_robust['age_june_2012'] >= age_min_treat) &
                            (df_robust['age_june_2012'] <= age_max_treat)).astype(int)
    df_robust['control'] = ((df_robust['age_june_2012'] >= age_min_control) &
                            (df_robust['age_june_2012'] <= age_max_control)).astype(int)

    df_robust = df_robust[(df_robust['treated'] == 1) | (df_robust['control'] == 1)].copy()

    # Apply other filters
    df_robust['age_at_immigration'] = df_robust['YRIMMIG'] - df_robust['BIRTHYR']
    df_robust = df_robust[df_robust['age_at_immigration'] < 16].copy()
    df_robust = df_robust[df_robust['YRIMMIG'] <= 2007].copy()

    df_robust['fulltime'] = (df_robust['UHRSWORK'] >= 35).astype(int)
    df_robust['post'] = (df_robust['YEAR'] >= 2013).astype(int)
    df_robust['treated_post'] = df_robust['treated'] * df_robust['post']
    df_robust['male'] = (df_robust['SEX'] == 1).astype(int)
    df_robust['married'] = (df_robust['MARST'] == 1).astype(int)

    try:
        model_robust_bw = smf.wls('fulltime ~ treated + post + treated_post + male + married + C(EDUC)',
                                   data=df_robust, weights=df_robust['PERWT']).fit()
        print(f"   Bandwidth {bandwidth} years: DiD = {model_robust_bw.params['treated_post']:.4f} "
              f"(SE = {model_robust_bw.bse['treated_post']:.4f}), N = {len(df_robust):,}")
    except Exception as e:
        print(f"   Bandwidth {bandwidth}: Error - {e}")

# 13.2 Subgroup analysis by sex
print("\n13.2 Subgroup Analysis by Sex:")
for sex, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_analysis[df_analysis['SEX'] == sex].copy()
    model_sex = smf.wls('fulltime ~ treated + post + treated_post + married + C(EDUC)',
                         data=df_sex, weights=df_sex['PERWT']).fit()
    print(f"   {sex_label}: DiD = {model_sex.params['treated_post']:.4f} "
          f"(SE = {model_sex.bse['treated_post']:.4f}), N = {len(df_sex):,}")

# 13.3 Placebo test: use 2009 as fake treatment year
print("\n13.3 Placebo Test (Fake Treatment Year 2009):")
df_placebo = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treated_post_placebo'] = df_placebo['treated'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treated + post_placebo + treated_post_placebo + male + married + C(EDUC)',
                         data=df_placebo, weights=df_placebo['PERWT']).fit()
print(f"   Placebo DiD (2009): {model_placebo.params['treated_post_placebo']:.4f} "
      f"(SE = {model_placebo.bse['treated_post_placebo']:.4f})")

# =============================================================================
# 14. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n" + "="*80)
print("14. SUMMARY OF MAIN RESULTS")
print("="*80)

print(f"""
PREFERRED SPECIFICATION: Weighted DiD with Year FE and Covariates
-----------------------------------------------------------------
Effect Size (DiD Estimate): {model4.params['treated_post']:.4f}
Standard Error: {model4.bse['treated_post']:.4f}
95% Confidence Interval: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]
Sample Size: {len(df_analysis):,}
Treatment Group N: {df_analysis['treated'].sum():,}
Control Group N: {(df_analysis['treated'] == 0).sum():,}

Interpretation:
DACA eligibility is associated with a {model4.params['treated_post']*100:.2f} percentage point
{'increase' if model4.params['treated_post'] > 0 else 'decrease'} in full-time employment
among the eligible age group (26-30) compared to the slightly older ineligible group (31-35).
""")

# Save key results to file for LaTeX report
results_dict = {
    'did_estimate': model4.params['treated_post'],
    'did_se': model4.bse['treated_post'],
    'did_ci_low': model4.conf_int().loc['treated_post', 0],
    'did_ci_high': model4.conf_int().loc['treated_post', 1],
    'sample_size': len(df_analysis),
    'treatment_n': df_analysis['treated'].sum(),
    'control_n': (df_analysis['treated'] == 0).sum(),
    'simple_did': did_estimate,
}

# Save results
pd.DataFrame([results_dict]).to_csv('results_summary.csv', index=False)
print("\nResults saved to results_summary.csv")

# =============================================================================
# 15. GENERATE DATA FOR FIGURES
# =============================================================================
print("\n" + "="*80)
print("15. GENERATING DATA FOR FIGURES")
print("="*80)

# Figure 1: Parallel trends
fig1_data = yearly_means.reset_index()
fig1_data.columns = ['YEAR', 'Control', 'Treatment']
fig1_data.to_csv('figure1_parallel_trends.csv', index=False)
print("Saved: figure1_parallel_trends.csv")

# Figure 2: Event study coefficients
fig2_data = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [event_study.params.get(f'treat_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
                   [event_study.params.get(f'treat_x_{y}', 0) for y in [2013, 2014, 2015, 2016]],
    'SE': [event_study.bse.get(f'treat_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
          [event_study.bse.get(f'treat_x_{y}', 0) for y in [2013, 2014, 2015, 2016]]
})
fig2_data.to_csv('figure2_event_study.csv', index=False)
print("Saved: figure2_event_study.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
