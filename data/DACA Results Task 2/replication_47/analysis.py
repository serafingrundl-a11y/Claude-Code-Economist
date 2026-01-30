"""
DACA Replication Analysis - Task 47
Difference-in-Differences analysis of DACA's impact on full-time employment
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DACA Replication Analysis - Task 47")
print("=" * 60)

# Load data
print("\n1. Loading ACS data...")
dtype_dict = {
    'YEAR': 'int32',
    'SAMPLE': 'int32',
    'SERIAL': 'int64',
    'HHWT': 'float64',
    'REGION': 'int16',
    'STATEFIP': 'int16',
    'METRO': 'int16',
    'GQ': 'int16',
    'PERNUM': 'int16',
    'PERWT': 'float64',
    'FAMSIZE': 'int16',
    'NCHILD': 'int16',
    'SEX': 'int16',
    'AGE': 'int16',
    'BIRTHQTR': 'int16',
    'MARST': 'int16',
    'BIRTHYR': 'int32',
    'HISPAN': 'int16',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int16',
    'YRIMMIG': 'int32',
    'YRSUSA1': 'int16',
    'EDUC': 'int16',
    'EDUCD': 'int16',
    'EMPSTAT': 'int16',
    'LABFORCE': 'int16',
    'UHRSWORK': 'int16',
}

# Read data
df = pd.read_csv('data/data.csv', dtype=dtype_dict, low_memory=False)
print(f"Total observations loaded: {len(df):,}")

# Check years available
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")

# Step 2: Filter to relevant population
print("\n2. Filtering to DACA-relevant population...")

# Hispanic-Mexican (HISPAN = 1 indicates Mexican)
df_mex = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican filter: {len(df_mex):,}")

# Born in Mexico (BPL = 200)
df_mex = df_mex[df_mex['BPL'] == 200]
print(f"After born in Mexico filter: {len(df_mex):,}")

# Not a citizen (CITIZEN = 3)
df_mex = df_mex[df_mex['CITIZEN'] == 3]
print(f"After non-citizen filter: {len(df_mex):,}")

# Step 3: Define age groups based on age at DACA implementation (June 15, 2012)
print("\n3. Defining treatment and control groups...")

# Treatment group: Ages 26-30 on June 15, 2012
# Birth years: 1982-1986 (approximately)
# Control group: Ages 31-35 on June 15, 2012
# Birth years: 1977-1981 (approximately)

# Calculate age on June 15, 2012 using birth year
# For simplicity, using birth year to determine treatment status
# Ages 26-30 in 2012: born 1982-1986
# Ages 31-35 in 2012: born 1977-1981

treatment_birthyears = [1982, 1983, 1984, 1985, 1986]
control_birthyears = [1977, 1978, 1979, 1980, 1981]

# Filter to these birth cohorts
df_sample = df_mex[df_mex['BIRTHYR'].isin(treatment_birthyears + control_birthyears)].copy()
print(f"After birth year filter (1977-1986): {len(df_sample):,}")

# Create treatment indicator (1 if in treatment cohort, 0 if control)
df_sample['treated'] = df_sample['BIRTHYR'].isin(treatment_birthyears).astype(int)
print(f"Treatment group size: {df_sample['treated'].sum():,}")
print(f"Control group size: {(df_sample['treated'] == 0).sum():,}")

# Step 4: Check arrival before age 16 (proxy for DACA eligibility)
print("\n4. Applying DACA eligibility restrictions...")

# Calculate age at immigration
df_sample['age_at_immigration'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Filter: must have arrived before age 16
# Handle cases where YRIMMIG might be 0 (N/A) - keep only valid immigration years
df_sample = df_sample[df_sample['YRIMMIG'] > 0]
df_sample = df_sample[df_sample['age_at_immigration'] < 16]
print(f"After arrived before age 16 filter: {len(df_sample):,}")

# Must have arrived by 2007 (continuous residence since June 15, 2007)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]
print(f"After arrived by 2007 filter: {len(df_sample):,}")

# Step 5: Define pre/post periods
print("\n5. Defining pre/post periods...")

# Pre-period: 2006-2011
# Post-period: 2013-2016
# Exclude 2012 (DACA implemented mid-year)

df_sample = df_sample[df_sample['YEAR'] != 2012]
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
print(f"Pre-period observations (2006-2011): {(df_sample['post'] == 0).sum():,}")
print(f"Post-period observations (2013-2016): {(df_sample['post'] == 1).sum():,}")

# Step 6: Define outcome variable
print("\n6. Defining outcome variable...")

# Full-time employment: usually working 35 hours or more per week
# UHRSWORK = usual hours worked per week
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create employed variable for reference
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

print(f"Full-time employment rate (overall): {df_sample['fulltime'].mean():.3f}")
print(f"Employment rate (overall): {df_sample['employed'].mean():.3f}")

# Step 7: Summary statistics
print("\n7. Summary Statistics by Group")
print("=" * 60)

# Summary by treatment status and period
summary = df_sample.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'UHRSWORK': 'mean',
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(3)

print("\nMeans by Treatment Status and Period:")
print("-" * 60)

for treated in [0, 1]:
    for post in [0, 1]:
        group = df_sample[(df_sample['treated'] == treated) & (df_sample['post'] == post)]
        group_name = f"{'Treatment' if treated else 'Control'} x {'Post' if post else 'Pre'}"
        print(f"\n{group_name}:")
        print(f"  N (unweighted): {len(group):,}")
        print(f"  N (weighted): {group['PERWT'].sum():,.0f}")
        print(f"  Full-time rate: {group['fulltime'].mean():.4f}")
        print(f"  Employment rate: {group['employed'].mean():.4f}")
        print(f"  Mean hours worked: {group['UHRSWORK'].mean():.2f}")
        print(f"  Mean age: {group['AGE'].mean():.2f}")

# Step 8: Simple Difference-in-Differences
print("\n\n8. Difference-in-Differences Estimation")
print("=" * 60)

# Calculate group means
treat_pre = df_sample[(df_sample['treated'] == 1) & (df_sample['post'] == 0)]['fulltime'].mean()
treat_post = df_sample[(df_sample['treated'] == 1) & (df_sample['post'] == 1)]['fulltime'].mean()
control_pre = df_sample[(df_sample['treated'] == 0) & (df_sample['post'] == 0)]['fulltime'].mean()
control_post = df_sample[(df_sample['treated'] == 0) & (df_sample['post'] == 1)]['fulltime'].mean()

print(f"\nGroup Means (Full-time Employment Rate):")
print(f"  Treatment Pre:  {treat_pre:.4f}")
print(f"  Treatment Post: {treat_post:.4f}")
print(f"  Control Pre:    {control_pre:.4f}")
print(f"  Control Post:   {control_post:.4f}")

# Simple DiD estimate
did_simple = (treat_post - treat_pre) - (control_post - control_pre)
print(f"\nSimple DiD Estimate: {did_simple:.4f}")
print(f"  Treatment change: {treat_post - treat_pre:.4f}")
print(f"  Control change:   {control_post - control_pre:.4f}")

# Step 9: Regression-based DiD (unweighted)
print("\n\n9. Regression-based DiD (Unweighted)")
print("=" * 60)

# Create interaction term
df_sample['treated_post'] = df_sample['treated'] * df_sample['post']

# Basic DiD regression
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_sample).fit(cov_type='HC1')
print("\nModel 1: Basic DiD (OLS, Robust SE)")
print(model1.summary().tables[1])

# Step 10: Regression-based DiD (weighted)
print("\n\n10. Regression-based DiD (Weighted)")
print("=" * 60)

model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print("\nModel 2: Basic DiD (WLS, Robust SE)")
print(model2.summary().tables[1])

# Step 11: DiD with covariates
print("\n\n11. DiD with Covariates")
print("=" * 60)

# Prepare covariates
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'] == 1).astype(int)

# Education categories
df_sample['educ_hs'] = ((df_sample['EDUCD'] >= 62) & (df_sample['EDUCD'] < 65)).astype(int)  # HS diploma
df_sample['educ_somecol'] = ((df_sample['EDUCD'] >= 65) & (df_sample['EDUCD'] < 101)).astype(int)  # Some college
df_sample['educ_college'] = (df_sample['EDUCD'] >= 101).astype(int)  # Bachelor's or higher

# Create age at survey variable for controlling
df_sample['age_sq'] = df_sample['AGE'] ** 2

# Model with covariates
model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecol + educ_college + AGE + age_sq',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print("\nModel 3: DiD with Demographics (WLS, Robust SE)")
print(model3.summary().tables[1])

# Step 12: DiD with year and state fixed effects
print("\n\n12. DiD with Fixed Effects")
print("=" * 60)

# Create year dummies (excluding reference year)
for year in df_sample['YEAR'].unique():
    df_sample[f'year_{year}'] = (df_sample['YEAR'] == year).astype(int)

# Create state dummies (subset for model)
df_sample['state'] = df_sample['STATEFIP']

# Model with year fixed effects
year_vars = [f'year_{y}' for y in sorted(df_sample['YEAR'].unique())[1:]]  # Exclude first year as reference
formula_ye = f"fulltime ~ treated + treated_post + female + married + educ_hs + educ_somecol + educ_college + AGE + age_sq + {' + '.join(year_vars)}"

model4 = smf.wls(formula_ye, data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print("\nModel 4: DiD with Year Fixed Effects (WLS, Robust SE)")
print(f"DiD Coefficient (treated_post): {model4.params['treated_post']:.4f}")
print(f"Standard Error: {model4.bse['treated_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"P-value: {model4.pvalues['treated_post']:.4f}")

# Step 13: Event study / dynamic effects
print("\n\n13. Event Study Analysis")
print("=" * 60)

# Create year-specific treatment effects
# Reference year: 2011 (last pre-treatment year)
for year in df_sample['YEAR'].unique():
    if year != 2011:  # Reference year
        df_sample[f'treat_year_{year}'] = (df_sample['treated'] * (df_sample['YEAR'] == year)).astype(int)

treat_year_vars = [f'treat_year_{y}' for y in sorted(df_sample['YEAR'].unique()) if y != 2011]
formula_es = f"fulltime ~ treated + {' + '.join(year_vars)} + {' + '.join(treat_year_vars)} + female + married + educ_hs + educ_somecol + educ_college + AGE + age_sq"

model_es = smf.wls(formula_es, data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Treatment x Year interactions):")
print("-" * 40)
for var in treat_year_vars:
    year = var.split('_')[-1]
    coef = model_es.params[var]
    se = model_es.bse[var]
    pval = model_es.pvalues[var]
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}, p={pval:.3f})")

# Step 14: Heterogeneity analysis by gender
print("\n\n14. Heterogeneity Analysis by Gender")
print("=" * 60)

# Males
df_males = df_sample[df_sample['female'] == 0]
model_males = smf.wls('fulltime ~ treated + post + treated_post + married + educ_hs + educ_somecol + educ_college + AGE + age_sq',
                       data=df_males, weights=df_males['PERWT']).fit(cov_type='HC1')

# Females
df_females = df_sample[df_sample['female'] == 1]
model_females = smf.wls('fulltime ~ treated + post + treated_post + married + educ_hs + educ_somecol + educ_college + AGE + age_sq',
                         data=df_females, weights=df_females['PERWT']).fit(cov_type='HC1')

print(f"\nMales:")
print(f"  N: {len(df_males):,}")
print(f"  DiD Coefficient: {model_males.params['treated_post']:.4f}")
print(f"  SE: {model_males.bse['treated_post']:.4f}")
print(f"  95% CI: [{model_males.conf_int().loc['treated_post', 0]:.4f}, {model_males.conf_int().loc['treated_post', 1]:.4f}]")

print(f"\nFemales:")
print(f"  N: {len(df_females):,}")
print(f"  DiD Coefficient: {model_females.params['treated_post']:.4f}")
print(f"  SE: {model_females.bse['treated_post']:.4f}")
print(f"  95% CI: [{model_females.conf_int().loc['treated_post', 0]:.4f}, {model_females.conf_int().loc['treated_post', 1]:.4f}]")

# Step 15: Robustness checks
print("\n\n15. Robustness Checks")
print("=" * 60)

# Robustness 1: Different age bandwidth (24-28 vs 33-37)
print("\nRobustness 1: Narrower age bandwidth")
df_narrow = df_mex[df_mex['BIRTHYR'].isin([1984, 1985, 1986, 1978, 1979, 1980])].copy()
df_narrow = df_narrow[df_narrow['YRIMMIG'] > 0]
df_narrow = df_narrow[(df_narrow['YRIMMIG'] - df_narrow['BIRTHYR']) < 16]
df_narrow = df_narrow[df_narrow['YRIMMIG'] <= 2007]
df_narrow = df_narrow[df_narrow['YEAR'] != 2012]
df_narrow['treated'] = df_narrow['BIRTHYR'].isin([1984, 1985, 1986]).astype(int)
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['treated_post'] = df_narrow['treated'] * df_narrow['post']
df_narrow['fulltime'] = (df_narrow['UHRSWORK'] >= 35).astype(int)
df_narrow['female'] = (df_narrow['SEX'] == 2).astype(int)
df_narrow['married'] = (df_narrow['MARST'] == 1).astype(int)
df_narrow['age_sq'] = df_narrow['AGE'] ** 2

model_narrow = smf.wls('fulltime ~ treated + post + treated_post + female + married + AGE + age_sq',
                        data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"  DiD Coefficient: {model_narrow.params['treated_post']:.4f} (SE: {model_narrow.bse['treated_post']:.4f})")

# Robustness 2: Employment (any) instead of full-time
print("\nRobustness 2: Any Employment as Outcome")
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)
model_emp = smf.wls('employed ~ treated + post + treated_post + female + married + educ_hs + educ_somecol + educ_college + AGE + age_sq',
                     data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"  DiD Coefficient: {model_emp.params['treated_post']:.4f} (SE: {model_emp.bse['treated_post']:.4f})")

# Robustness 3: Hours worked (continuous)
print("\nRobustness 3: Usual Hours Worked (continuous)")
model_hrs = smf.wls('UHRSWORK ~ treated + post + treated_post + female + married + educ_hs + educ_somecol + educ_college + AGE + age_sq',
                     data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"  DiD Coefficient: {model_hrs.params['treated_post']:.4f} hours (SE: {model_hrs.bse['treated_post']:.4f})")

# Step 16: Final Summary
print("\n\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print("\n** Preferred Estimate (Model 3 - DiD with Demographics) **")
print(f"Effect Size: {model3.params['treated_post']:.4f}")
print(f"Standard Error: {model3.bse['treated_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")
print(f"P-value: {model3.pvalues['treated_post']:.4f}")
print(f"Sample Size: {len(df_sample):,}")
print(f"Weighted Sample Size: {df_sample['PERWT'].sum():,.0f}")

print("\n\nInterpretation:")
coef = model3.params['treated_post']
if coef > 0:
    print(f"DACA eligibility is associated with a {abs(coef)*100:.2f} percentage point")
    print("INCREASE in the probability of full-time employment.")
else:
    print(f"DACA eligibility is associated with a {abs(coef)*100:.2f} percentage point")
    print("DECREASE in the probability of full-time employment.")

if model3.pvalues['treated_post'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
else:
    print("This effect is NOT statistically significant at the 5% level.")

# Save key results to file
results_dict = {
    'preferred_estimate': model3.params['treated_post'],
    'standard_error': model3.bse['treated_post'],
    'ci_lower': model3.conf_int().loc['treated_post', 0],
    'ci_upper': model3.conf_int().loc['treated_post', 1],
    'p_value': model3.pvalues['treated_post'],
    'sample_size': len(df_sample),
    'weighted_n': df_sample['PERWT'].sum()
}

# Save full model summaries
with open('model_results.txt', 'w') as f:
    f.write("DACA Replication Analysis - Full Model Results\n")
    f.write("=" * 70 + "\n\n")

    f.write("Model 1: Basic DiD (Unweighted, Robust SE)\n")
    f.write("-" * 70 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("Model 2: Basic DiD (Weighted, Robust SE)\n")
    f.write("-" * 70 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("Model 3: DiD with Demographics (Weighted, Robust SE) - PREFERRED\n")
    f.write("-" * 70 + "\n")
    f.write(str(model3.summary()) + "\n\n")

    f.write("Model 4: DiD with Year Fixed Effects (Weighted, Robust SE)\n")
    f.write("-" * 70 + "\n")
    f.write(str(model4.summary()) + "\n\n")

print("\n\nResults saved to model_results.txt")

# Create summary tables for LaTeX
print("\n\nGenerating LaTeX tables...")

# Table 1: Summary Statistics
summary_stats = []
for treated in [0, 1]:
    for post in [0, 1]:
        group = df_sample[(df_sample['treated'] == treated) & (df_sample['post'] == post)]
        summary_stats.append({
            'Group': 'Treatment' if treated else 'Control',
            'Period': 'Post' if post else 'Pre',
            'N': len(group),
            'Weighted_N': group['PERWT'].sum(),
            'Fulltime_Rate': group['fulltime'].mean(),
            'Employment_Rate': group['employed'].mean(),
            'Mean_Hours': group['UHRSWORK'].mean(),
            'Mean_Age': group['AGE'].mean(),
            'Female_Share': group['female'].mean(),
            'Married_Share': group['married'].mean()
        })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_statistics.csv', index=False)
print("Summary statistics saved to summary_statistics.csv")

# Table 2: Main Results
main_results = []
models = [
    ('Basic DiD (Unweighted)', model1),
    ('Basic DiD (Weighted)', model2),
    ('DiD with Covariates', model3),
    ('DiD with Year FE', model4)
]

for name, model in models:
    main_results.append({
        'Model': name,
        'Coefficient': model.params['treated_post'],
        'SE': model.bse['treated_post'],
        'CI_Lower': model.conf_int().loc['treated_post', 0],
        'CI_Upper': model.conf_int().loc['treated_post', 1],
        'P_Value': model.pvalues['treated_post'],
        'N': int(model.nobs),
        'R_Squared': model.rsquared
    })

results_df = pd.DataFrame(main_results)
results_df.to_csv('main_results.csv', index=False)
print("Main results saved to main_results.csv")

# Table 3: Event Study Results
event_study_results = []
for var in treat_year_vars:
    year = int(var.split('_')[-1])
    event_study_results.append({
        'Year': year,
        'Coefficient': model_es.params[var],
        'SE': model_es.bse[var],
        'CI_Lower': model_es.conf_int().loc[var, 0],
        'CI_Upper': model_es.conf_int().loc[var, 1],
        'P_Value': model_es.pvalues[var]
    })

event_df = pd.DataFrame(event_study_results).sort_values('Year')
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Table 4: Heterogeneity Results
het_results = [
    {'Group': 'Males',
     'N': len(df_males),
     'Coefficient': model_males.params['treated_post'],
     'SE': model_males.bse['treated_post'],
     'CI_Lower': model_males.conf_int().loc['treated_post', 0],
     'CI_Upper': model_males.conf_int().loc['treated_post', 1],
     'P_Value': model_males.pvalues['treated_post']},
    {'Group': 'Females',
     'N': len(df_females),
     'Coefficient': model_females.params['treated_post'],
     'SE': model_females.bse['treated_post'],
     'CI_Lower': model_females.conf_int().loc['treated_post', 0],
     'CI_Upper': model_females.conf_int().loc['treated_post', 1],
     'P_Value': model_females.pvalues['treated_post']}
]

het_df = pd.DataFrame(het_results)
het_df.to_csv('heterogeneity_results.csv', index=False)
print("Heterogeneity results saved to heterogeneity_results.csv")

# Table 5: Robustness Results
robust_results = [
    {'Specification': 'Main (Full-time)',
     'Coefficient': model3.params['treated_post'],
     'SE': model3.bse['treated_post']},
    {'Specification': 'Narrow Bandwidth',
     'Coefficient': model_narrow.params['treated_post'],
     'SE': model_narrow.bse['treated_post']},
    {'Specification': 'Any Employment',
     'Coefficient': model_emp.params['treated_post'],
     'SE': model_emp.bse['treated_post']},
    {'Specification': 'Hours Worked',
     'Coefficient': model_hrs.params['treated_post'],
     'SE': model_hrs.bse['treated_post']}
]

robust_df = pd.DataFrame(robust_results)
robust_df.to_csv('robustness_results.csv', index=False)
print("Robustness results saved to robustness_results.csv")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
