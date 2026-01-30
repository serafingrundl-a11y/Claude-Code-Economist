"""
DACA Effect on Full-Time Employment Replication Analysis
============================================================
Research Question: Effect of DACA eligibility on full-time employment
(35+ hours per week) among Hispanic-Mexican, Mexican-born individuals

Treatment: Eligible individuals aged 26-30 as of June 15, 2012
Control: Individuals aged 31-35 as of June 15, 2012 (otherwise eligible)

Method: Difference-in-Differences
Pre-period: 2006-2011
Post-period: 2013-2016 (DACA implemented in 2012)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DACA REPLICATION ANALYSIS - FULL-TIME EMPLOYMENT")
print("=" * 70)

# Load data
print("\n1. LOADING DATA...")
print("-" * 50)

# Read data in chunks due to large size
chunks = []
chunksize = 500000
for chunk in pd.read_csv('data/data.csv', chunksize=chunksize, low_memory=False):
    chunks.append(chunk)
    print(f"  Loaded {len(chunks)} chunks ({len(chunks) * chunksize:,} rows processed)...")

df = pd.concat(chunks, ignore_index=True)
print(f"\n  Total observations loaded: {len(df):,}")
print(f"  Years in data: {sorted(df['YEAR'].unique())}")
print(f"  Columns: {df.columns.tolist()}")

# =============================================================================
# 2. DEFINE DACA ELIGIBILITY CRITERIA
# =============================================================================
print("\n2. DEFINING DACA ELIGIBILITY CRITERIA...")
print("-" * 50)

"""
DACA Eligibility Requirements (as of June 15, 2012):
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Not a citizen
5. Not a legal resident

We identify potential DACA-eligible individuals as:
- Hispanic-Mexican ethnicity (HISPAN = 1)
- Born in Mexico (BPL = 200)
- Non-citizen (CITIZEN = 3 - "Not a citizen")
- Immigrated before age 16
- Lived in US since 2007 (YRIMMIG <= 2007)
"""

# Create working copy
print("  Creating working copy of data...")
data = df.copy()

# Filter for Hispanic-Mexican ethnicity
print(f"  Filtering for Hispanic-Mexican (HISPAN=1)...")
data = data[data['HISPAN'] == 1]
print(f"    Remaining observations: {len(data):,}")

# Filter for born in Mexico
print(f"  Filtering for born in Mexico (BPL=200)...")
data = data[data['BPL'] == 200]
print(f"    Remaining observations: {len(data):,}")

# Filter for non-citizens
print(f"  Filtering for non-citizens (CITIZEN=3)...")
data = data[data['CITIZEN'] == 3]
print(f"    Remaining observations: {len(data):,}")

# Exclude 2012 (cannot distinguish pre/post DACA implementation)
print(f"  Excluding 2012 (implementation year)...")
data = data[data['YEAR'] != 2012]
print(f"    Remaining observations: {len(data):,}")

# =============================================================================
# 3. CALCULATE AGE AS OF JUNE 15, 2012
# =============================================================================
print("\n3. CALCULATING AGE AS OF JUNE 15, 2012...")
print("-" * 50)

"""
To determine which age group an individual belongs to (treatment or control),
we need their age as of June 15, 2012.

June 15, 2012 falls in Q2 (April-May-June).
- If someone turns X in 2012 and BIRTHQTR <= 2, they were X on June 15, 2012
- If BIRTHQTR > 2 (July onwards), they were X-1 on June 15, 2012
"""

# Calculate age as of June 15, 2012
# This is the age they would have been on the specific date June 15, 2012
data['age_june2012'] = 2012 - data['BIRTHYR']

# Adjust for birth quarter: if born after June (Q3 or Q4), subtract 1
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# June 15 falls in Q2, so those born in Q3 or Q4 haven't had their birthday yet
data.loc[data['BIRTHQTR'] > 2, 'age_june2012'] = data['age_june2012'] - 1

print(f"  Age distribution as of June 15, 2012:")
print(data['age_june2012'].describe())

# =============================================================================
# 4. FILTER FOR ARRIVAL BEFORE AGE 16 & CONTINUOUS RESIDENCE
# =============================================================================
print("\n4. APPLYING ARRIVAL AND RESIDENCE CRITERIA...")
print("-" * 50)

# Calculate age at immigration
# YRIMMIG is year of immigration, BIRTHYR is birth year
data['age_at_immig'] = data['YRIMMIG'] - data['BIRTHYR']

# Filter: arrived before age 16
print(f"  Filtering for arrived before age 16...")
data = data[data['age_at_immig'] < 16]
print(f"    Remaining observations: {len(data):,}")

# Filter: in US since 2007 (YRIMMIG <= 2007)
print(f"  Filtering for continuous US residence since 2007 (YRIMMIG <= 2007)...")
data = data[data['YRIMMIG'] <= 2007]
print(f"    Remaining observations: {len(data):,}")

# =============================================================================
# 5. DEFINE TREATMENT AND CONTROL GROUPS
# =============================================================================
print("\n5. DEFINING TREATMENT AND CONTROL GROUPS...")
print("-" * 50)

"""
Treatment group: Age 26-30 as of June 15, 2012 (DACA-eligible by age)
Control group: Age 31-35 as of June 15, 2012 (would be eligible except for age)

Note: The 31st birthday cutoff means those aged 31+ were NOT eligible.
"""

# Filter to relevant age groups
print(f"  Filtering for ages 26-35 as of June 2012...")
data = data[(data['age_june2012'] >= 26) & (data['age_june2012'] <= 35)]
print(f"    Remaining observations: {len(data):,}")

# Create treatment indicator (1 = treatment, 0 = control)
data['treated'] = (data['age_june2012'] <= 30).astype(int)

print(f"\n  Treatment group (age 26-30): {(data['treated']==1).sum():,} observations")
print(f"  Control group (age 31-35): {(data['treated']==0).sum():,} observations")

# =============================================================================
# 6. DEFINE PRE/POST PERIODS
# =============================================================================
print("\n6. DEFINING PRE/POST PERIODS...")
print("-" * 50)

"""
Pre-period: 2006-2011 (before DACA)
Post-period: 2013-2016 (after DACA implementation)
2012 is excluded because DACA was implemented mid-year
"""

data['post'] = (data['YEAR'] >= 2013).astype(int)

print(f"  Pre-period (2006-2011): {(data['post']==0).sum():,} observations")
print(f"  Post-period (2013-2016): {(data['post']==1).sum():,} observations")

# =============================================================================
# 7. CREATE OUTCOME VARIABLE: FULL-TIME EMPLOYMENT
# =============================================================================
print("\n7. CREATING OUTCOME VARIABLE...")
print("-" * 50)

"""
Outcome: Full-time employment
Defined as usually working 35 hours per week or more (UHRSWORK >= 35)
"""

# Create full-time employment indicator
data['fulltime'] = (data['UHRSWORK'] >= 35).astype(int)

# Summary statistics
print(f"  Full-time employment (UHRSWORK >= 35):")
print(f"    Overall rate: {data['fulltime'].mean():.4f} ({data['fulltime'].mean()*100:.2f}%)")
print(f"    N working full-time: {data['fulltime'].sum():,}")
print(f"    N not full-time: {(data['fulltime']==0).sum():,}")

# =============================================================================
# 8. CREATE COVARIATES
# =============================================================================
print("\n8. CREATING COVARIATES...")
print("-" * 50)

# Sex indicator (female = 1)
data['female'] = (data['SEX'] == 2).astype(int)

# Age in the current survey year
data['age_current'] = data['AGE']

# Marital status (married = 1)
data['married'] = data['MARST'].isin([1, 2]).astype(int)

# Education categories
data['edu_less_hs'] = (data['EDUC'] < 6).astype(int)
data['edu_hs'] = (data['EDUC'] == 6).astype(int)
data['edu_some_college'] = ((data['EDUC'] > 6) & (data['EDUC'] < 10)).astype(int)
data['edu_college_plus'] = (data['EDUC'] >= 10).astype(int)

# State fixed effects
data['state'] = data['STATEFIP']

# Years in USA
data['years_in_usa'] = data['YRSUSA1']

# Family size
data['famsize'] = data['FAMSIZE']

# Number of children
data['nchild'] = data['NCHILD']

print("  Covariates created: female, age_current, married, education dummies, state")

# =============================================================================
# 9. SUMMARY STATISTICS
# =============================================================================
print("\n9. SUMMARY STATISTICS...")
print("-" * 50)

# By treatment status and period
summary_groups = data.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'age_current': 'mean',
    'married': 'mean',
    'edu_less_hs': 'mean',
    'edu_hs': 'mean',
    'edu_some_college': 'mean',
    'edu_college_plus': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary by Treatment Status and Period:")
print(summary_groups)

# Save summary statistics
summary_stats = pd.DataFrame({
    'Group': ['Treatment Pre', 'Treatment Post', 'Control Pre', 'Control Post'],
    'N': [
        len(data[(data['treated']==1) & (data['post']==0)]),
        len(data[(data['treated']==1) & (data['post']==1)]),
        len(data[(data['treated']==0) & (data['post']==0)]),
        len(data[(data['treated']==0) & (data['post']==1)])
    ],
    'Fulltime_Mean': [
        data[(data['treated']==1) & (data['post']==0)]['fulltime'].mean(),
        data[(data['treated']==1) & (data['post']==1)]['fulltime'].mean(),
        data[(data['treated']==0) & (data['post']==0)]['fulltime'].mean(),
        data[(data['treated']==0) & (data['post']==1)]['fulltime'].mean()
    ],
    'Female_Mean': [
        data[(data['treated']==1) & (data['post']==0)]['female'].mean(),
        data[(data['treated']==1) & (data['post']==1)]['female'].mean(),
        data[(data['treated']==0) & (data['post']==0)]['female'].mean(),
        data[(data['treated']==0) & (data['post']==1)]['female'].mean()
    ]
})
summary_stats.to_csv('summary_statistics.csv', index=False)
print("\n  Summary statistics saved to summary_statistics.csv")

# =============================================================================
# 10. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n10. DIFFERENCE-IN-DIFFERENCES ANALYSIS...")
print("-" * 50)

# Create interaction term
data['treated_post'] = data['treated'] * data['post']

# Basic 2x2 DiD calculation
y_treat_pre = data[(data['treated']==1) & (data['post']==0)]['fulltime'].mean()
y_treat_post = data[(data['treated']==1) & (data['post']==1)]['fulltime'].mean()
y_control_pre = data[(data['treated']==0) & (data['post']==0)]['fulltime'].mean()
y_control_post = data[(data['treated']==0) & (data['post']==1)]['fulltime'].mean()

did_estimate = (y_treat_post - y_treat_pre) - (y_control_post - y_control_pre)

print("\n  Simple 2x2 Difference-in-Differences:")
print(f"    Treatment Pre:  {y_treat_pre:.4f}")
print(f"    Treatment Post: {y_treat_post:.4f}")
print(f"    Treatment Diff: {y_treat_post - y_treat_pre:.4f}")
print(f"    Control Pre:    {y_control_pre:.4f}")
print(f"    Control Post:   {y_control_post:.4f}")
print(f"    Control Diff:   {y_control_post - y_control_pre:.4f}")
print(f"    DiD Estimate:   {did_estimate:.4f}")

# =============================================================================
# 11. REGRESSION MODELS
# =============================================================================
print("\n11. REGRESSION MODELS...")
print("-" * 50)

# Model 1: Basic DiD without controls
print("\n  Model 1: Basic DiD (OLS)")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=data).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n  Model 2: DiD with demographic controls")
model2 = smf.ols('fulltime ~ treated + post + treated_post + female + age_current + married',
                 data=data).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with demographic controls and education
print("\n  Model 3: DiD with demographics and education controls")
model3 = smf.ols('fulltime ~ treated + post + treated_post + female + age_current + married + edu_hs + edu_some_college + edu_college_plus',
                 data=data).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects
print("\n  Model 4: DiD with year fixed effects")
data['year_factor'] = pd.Categorical(data['YEAR'])
model4 = smf.ols('fulltime ~ treated + treated_post + C(YEAR) + female + age_current + married + edu_hs + edu_some_college + edu_college_plus',
                 data=data).fit(cov_type='HC1')
print(f"    DiD Coefficient (treated_post): {model4.params['treated_post']:.4f}")
print(f"    Std Error: {model4.bse['treated_post']:.4f}")
print(f"    t-stat: {model4.tvalues['treated_post']:.4f}")
print(f"    p-value: {model4.pvalues['treated_post']:.4f}")

# Model 5: DiD with year and state fixed effects (preferred specification)
print("\n  Model 5: DiD with year and state fixed effects (PREFERRED)")
model5 = smf.ols('fulltime ~ treated + treated_post + C(YEAR) + C(state) + female + age_current + married + edu_hs + edu_some_college + edu_college_plus',
                 data=data).fit(cov_type='HC1')
print(f"    DiD Coefficient (treated_post): {model5.params['treated_post']:.4f}")
print(f"    Std Error: {model5.bse['treated_post']:.4f}")
print(f"    t-stat: {model5.tvalues['treated_post']:.4f}")
print(f"    p-value: {model5.pvalues['treated_post']:.4f}")
print(f"    95% CI: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")

# =============================================================================
# 12. WEIGHTED REGRESSION USING PERSON WEIGHTS
# =============================================================================
print("\n12. WEIGHTED REGRESSION ANALYSIS...")
print("-" * 50)

# Model 6: Weighted DiD with full controls (using survey weights)
print("\n  Model 6: Weighted DiD with year and state fixed effects")
model6 = smf.wls('fulltime ~ treated + treated_post + C(YEAR) + C(state) + female + age_current + married + edu_hs + edu_some_college + edu_college_plus',
                 data=data, weights=data['PERWT']).fit(cov_type='HC1')
print(f"    DiD Coefficient (treated_post): {model6.params['treated_post']:.4f}")
print(f"    Std Error: {model6.bse['treated_post']:.4f}")
print(f"    t-stat: {model6.tvalues['treated_post']:.4f}")
print(f"    p-value: {model6.pvalues['treated_post']:.4f}")
print(f"    95% CI: [{model6.conf_int().loc['treated_post', 0]:.4f}, {model6.conf_int().loc['treated_post', 1]:.4f}]")

# =============================================================================
# 13. ROBUSTNESS CHECKS
# =============================================================================
print("\n13. ROBUSTNESS CHECKS...")
print("-" * 50)

# Check 1: Narrower age bands (27-29 vs 32-34)
print("\n  Robustness Check 1: Narrower age bands (27-29 vs 32-34)")
data_narrow = data[(data['age_june2012'] >= 27) & (data['age_june2012'] <= 34)]
data_narrow = data_narrow[(data_narrow['age_june2012'] <= 29) | (data_narrow['age_june2012'] >= 32)]
data_narrow['treated_narrow'] = (data_narrow['age_june2012'] <= 29).astype(int)
data_narrow['treated_post_narrow'] = data_narrow['treated_narrow'] * data_narrow['post']

model_narrow = smf.ols('fulltime ~ treated_narrow + treated_post_narrow + C(YEAR) + C(state) + female + age_current + married',
                       data=data_narrow).fit(cov_type='HC1')
print(f"    DiD Coefficient: {model_narrow.params['treated_post_narrow']:.4f}")
print(f"    Std Error: {model_narrow.bse['treated_post_narrow']:.4f}")
print(f"    p-value: {model_narrow.pvalues['treated_post_narrow']:.4f}")

# Check 2: Different definition of full-time (40+ hours)
print("\n  Robustness Check 2: Full-time defined as 40+ hours")
data['fulltime_40'] = (data['UHRSWORK'] >= 40).astype(int)
model_40hr = smf.ols('fulltime_40 ~ treated + treated_post + C(YEAR) + C(state) + female + age_current + married',
                     data=data).fit(cov_type='HC1')
print(f"    DiD Coefficient: {model_40hr.params['treated_post']:.4f}")
print(f"    Std Error: {model_40hr.bse['treated_post']:.4f}")
print(f"    p-value: {model_40hr.pvalues['treated_post']:.4f}")

# Check 3: Pre-trend test (placebo test using 2009 as fake treatment)
print("\n  Robustness Check 3: Pre-trend test (placebo 2009)")
data_pretrend = data[data['YEAR'] <= 2011]
data_pretrend['post_placebo'] = (data_pretrend['YEAR'] >= 2009).astype(int)
data_pretrend['treated_post_placebo'] = data_pretrend['treated'] * data_pretrend['post_placebo']
model_pretrend = smf.ols('fulltime ~ treated + treated_post_placebo + C(YEAR) + female + age_current + married',
                         data=data_pretrend).fit(cov_type='HC1')
print(f"    Placebo DiD Coefficient: {model_pretrend.params['treated_post_placebo']:.4f}")
print(f"    Std Error: {model_pretrend.bse['treated_post_placebo']:.4f}")
print(f"    p-value: {model_pretrend.pvalues['treated_post_placebo']:.4f}")

# =============================================================================
# 14. EVENT STUDY ANALYSIS
# =============================================================================
print("\n14. EVENT STUDY ANALYSIS...")
print("-" * 50)

# Create year dummies interacted with treatment
years = sorted(data['YEAR'].unique())
data['year_str'] = data['YEAR'].astype(str)

# Use 2011 as reference year
for year in years:
    if year != 2011:  # 2011 is reference
        data[f'treat_year_{year}'] = ((data['treated'] == 1) & (data['YEAR'] == year)).astype(int)

# Event study regression
event_study_vars = ' + '.join([f'treat_year_{y}' for y in years if y != 2011])
event_study_formula = f'fulltime ~ treated + C(YEAR) + {event_study_vars} + female + age_current + married'
model_event = smf.ols(event_study_formula, data=data).fit(cov_type='HC1')

print("\n  Event Study Coefficients (relative to 2011):")
print("  Year    Coefficient    Std Error      p-value")
print("  " + "-"*50)
for year in sorted(years):
    if year != 2011:
        var = f'treat_year_{year}'
        print(f"  {year}    {model_event.params[var]:10.4f}    {model_event.bse[var]:10.4f}    {model_event.pvalues[var]:10.4f}")

# =============================================================================
# 15. HETEROGENEITY ANALYSIS
# =============================================================================
print("\n15. HETEROGENEITY ANALYSIS...")
print("-" * 50)

# By gender
print("\n  Heterogeneity by Gender:")
for sex_val, sex_label in [(0, 'Male'), (1, 'Female')]:
    data_sub = data[data['female'] == sex_val]
    model_sub = smf.ols('fulltime ~ treated + treated_post + C(YEAR) + C(state) + age_current + married',
                        data=data_sub).fit(cov_type='HC1')
    print(f"    {sex_label}: DiD = {model_sub.params['treated_post']:.4f} (SE: {model_sub.bse['treated_post']:.4f}, p: {model_sub.pvalues['treated_post']:.4f})")

# By education level
print("\n  Heterogeneity by Education:")
data['edu_level'] = 'Less than HS'
data.loc[data['edu_hs'] == 1, 'edu_level'] = 'High School'
data.loc[data['edu_some_college'] == 1, 'edu_level'] = 'Some College'
data.loc[data['edu_college_plus'] == 1, 'edu_level'] = 'College+'

for edu in ['Less than HS', 'High School', 'Some College', 'College+']:
    data_sub = data[data['edu_level'] == edu]
    if len(data_sub) > 100:
        model_sub = smf.ols('fulltime ~ treated + treated_post + C(YEAR) + female + age_current + married',
                            data=data_sub).fit(cov_type='HC1')
        print(f"    {edu}: DiD = {model_sub.params['treated_post']:.4f} (SE: {model_sub.bse['treated_post']:.4f}, p: {model_sub.pvalues['treated_post']:.4f}, N: {len(data_sub):,})")

# =============================================================================
# 16. SAVE RESULTS
# =============================================================================
print("\n16. SAVING RESULTS...")
print("-" * 50)

# Save main results
results_dict = {
    'Model': ['Basic DiD', 'With Demographics', 'With Education', 'Year FE', 'Year+State FE (Preferred)', 'Weighted Year+State FE'],
    'DiD_Estimate': [
        model1.params['treated_post'],
        model2.params['treated_post'],
        model3.params['treated_post'],
        model4.params['treated_post'],
        model5.params['treated_post'],
        model6.params['treated_post']
    ],
    'Std_Error': [
        model1.bse['treated_post'],
        model2.bse['treated_post'],
        model3.bse['treated_post'],
        model4.bse['treated_post'],
        model5.bse['treated_post'],
        model6.bse['treated_post']
    ],
    'p_value': [
        model1.pvalues['treated_post'],
        model2.pvalues['treated_post'],
        model3.pvalues['treated_post'],
        model4.pvalues['treated_post'],
        model5.pvalues['treated_post'],
        model6.pvalues['treated_post']
    ],
    'N': [model1.nobs, model2.nobs, model3.nobs, model4.nobs, model5.nobs, model6.nobs],
    'R_squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared, model5.rsquared, model6.rsquared]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('regression_results.csv', index=False)
print("  Regression results saved to regression_results.csv")

# Save event study results
event_results = []
for year in sorted(years):
    if year != 2011:
        var = f'treat_year_{year}'
        event_results.append({
            'Year': year,
            'Coefficient': model_event.params[var],
            'Std_Error': model_event.bse[var],
            'CI_Lower': model_event.conf_int().loc[var, 0],
            'CI_Upper': model_event.conf_int().loc[var, 1],
            'p_value': model_event.pvalues[var]
        })
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("  Event study results saved to event_study_results.csv")

# Save full model summaries
with open('model_summaries.txt', 'w') as f:
    f.write("DACA REPLICATION ANALYSIS - FULL MODEL SUMMARIES\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODEL 1: Basic DiD\n")
    f.write("-" * 50 + "\n")
    f.write(model1.summary().as_text())
    f.write("\n\n")

    f.write("MODEL 5: DiD with Year and State Fixed Effects (PREFERRED)\n")
    f.write("-" * 50 + "\n")
    f.write(model5.summary().as_text())
    f.write("\n\n")

    f.write("MODEL 6: Weighted DiD with Year and State Fixed Effects\n")
    f.write("-" * 50 + "\n")
    f.write(model6.summary().as_text())

print("  Model summaries saved to model_summaries.txt")

# =============================================================================
# 17. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY - PREFERRED ESTIMATES")
print("=" * 70)

print(f"\n  Sample Size: {len(data):,}")
print(f"  Treatment Group (age 26-30): {(data['treated']==1).sum():,}")
print(f"  Control Group (age 31-35): {(data['treated']==0).sum():,}")
print(f"  Pre-period years: 2006-2011")
print(f"  Post-period years: 2013-2016")

print(f"\n  Outcome: Full-time employment (35+ hours/week)")
print(f"  Overall full-time rate: {data['fulltime'].mean()*100:.2f}%")

print(f"\n  PREFERRED ESTIMATE (Model 5 - Unweighted with Year+State FE):")
print(f"    Effect Size: {model5.params['treated_post']:.4f}")
print(f"    Standard Error: {model5.bse['treated_post']:.4f}")
print(f"    95% CI: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model5.pvalues['treated_post']:.4f}")
print(f"    N: {int(model5.nobs):,}")

print(f"\n  WEIGHTED ESTIMATE (Model 6 - With Survey Weights):")
print(f"    Effect Size: {model6.params['treated_post']:.4f}")
print(f"    Standard Error: {model6.bse['treated_post']:.4f}")
print(f"    95% CI: [{model6.conf_int().loc['treated_post', 0]:.4f}, {model6.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model6.pvalues['treated_post']:.4f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
