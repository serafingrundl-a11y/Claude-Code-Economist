"""
DACA Replication Analysis - Study 34
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals

Author: Anonymous
Date: 2024
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
print("DACA REPLICATION ANALYSIS - STUDY 34")
print("="*80)

#############################################################################
# STEP 1: Load Data in Chunks and Filter
#############################################################################
print("\n" + "="*80)
print("STEP 1: Loading Data (Chunked Processing)")
print("="*80)

# Define columns we need
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
           'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
           'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'UHRSWORK']

# Read in chunks and filter immediately
chunks = []
chunksize = 1000000

for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunksize):
    # Filter to Hispanic-Mexican (HISPAN=1) and born in Mexico (BPL=200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"Processed chunk, kept {len(filtered):,} rows")

# Combine filtered chunks
df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal observations after filtering: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

#############################################################################
# STEP 2: Further Sample Restrictions
#############################################################################
print("\n" + "="*80)
print("STEP 2: Sample Restrictions")
print("="*80)

# Exclude 2012 (DACA implemented mid-year)
sample = df[df['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(sample):,}")

# Create POST indicator (1 if year >= 2013)
sample['POST'] = (sample['YEAR'] >= 2013).astype(int)

# Restrict to working age population (16-64)
sample = sample[(sample['AGE'] >= 16) & (sample['AGE'] <= 64)].copy()
print(f"After restricting to ages 16-64: {len(sample):,}")

print(f"\nYear distribution:")
print(sample['YEAR'].value_counts().sort_index())

print(f"\nCitizenship distribution:")
print(sample['CITIZEN'].value_counts().sort_index())

#############################################################################
# STEP 3: Define DACA Eligibility
#############################################################################
print("\n" + "="*80)
print("STEP 3: Define DACA Eligibility")
print("="*80)

"""
DACA Eligibility Criteria (as of June 15, 2012):
1. Arrived in US before 16th birthday
2. Born after June 15, 1981 (under 31 on June 15, 2012)
3. Lived continuously in US since June 15, 2007 (immigrated by 2007)
4. Not a citizen (CITIZEN=3)
"""

# Calculate age at immigration
sample['age_at_immig'] = sample['YRIMMIG'] - sample['BIRTHYR']

# Check YRIMMIG values
print(f"\nYRIMMIG distribution:")
print(f"  YRIMMIG=0 (N/A): {(sample['YRIMMIG']==0).sum():,}")
print(f"  YRIMMIG>0: {(sample['YRIMMIG']>0).sum():,}")

# Filter to those with valid immigration year
sample = sample[sample['YRIMMIG'] > 0].copy()
print(f"\nSample with valid YRIMMIG: {len(sample):,}")

# Define eligibility components
# 1. Arrived before age 16
sample['arrived_young'] = (sample['age_at_immig'] < 16).astype(int)

# 2. Born after June 15, 1981 (for age <31 on June 15, 2012)
# Born in 1982+ definitely qualifies
# Born in 1981: Q3-Q4 (Jul-Dec) qualifies (born after June 15)
sample['under_31'] = ((sample['BIRTHYR'] >= 1982) |
                       ((sample['BIRTHYR'] == 1981) &
                        (sample['BIRTHQTR'] >= 3))).astype(int)

# 3. Immigrated by 2007 (continuous residence since June 15, 2007)
sample['early_immig'] = (sample['YRIMMIG'] <= 2007).astype(int)

# 4. Not a citizen
sample['non_citizen'] = (sample['CITIZEN'] == 3).astype(int)

# DACA eligible = all criteria met
sample['DACA_ELIGIBLE'] = (
    (sample['arrived_young'] == 1) &
    (sample['under_31'] == 1) &
    (sample['early_immig'] == 1) &
    (sample['non_citizen'] == 1)
).astype(int)

print(f"\nDACA eligibility components:")
print(f"  Arrived before age 16: {sample['arrived_young'].sum():,}")
print(f"  Under 31 on June 15, 2012: {sample['under_31'].sum():,}")
print(f"  Immigrated by 2007: {sample['early_immig'].sum():,}")
print(f"  Non-citizen: {sample['non_citizen'].sum():,}")
print(f"  DACA eligible (all criteria): {sample['DACA_ELIGIBLE'].sum():,}")

#############################################################################
# STEP 4: Create Outcome Variable
#############################################################################
print("\n" + "="*80)
print("STEP 4: Create Outcome Variable")
print("="*80)

# Full-time employment: UHRSWORK >= 35
sample['fulltime'] = (sample['UHRSWORK'] >= 35).astype(int)

print(f"\nUHRSWORK summary:")
print(sample['UHRSWORK'].describe())

# Employment status
sample['employed'] = (sample['EMPSTAT'] == 1).astype(int)
print(f"\nEmployment rate: {sample['employed'].mean():.4f}")
print(f"Full-time employment rate: {sample['fulltime'].mean():.4f}")

#############################################################################
# STEP 5: Create Comparison Groups and Analysis Sample
#############################################################################
print("\n" + "="*80)
print("STEP 5: Create Comparison Groups")
print("="*80)

"""
For DiD, we need a control group that is similar to DACA-eligible but ineligible.
Control: Non-citizens who arrived young, immigrated early, but are too old (born before 1981)
"""

# Focus on non-citizens
analysis_sample = sample[sample['non_citizen'] == 1].copy()
print(f"\nAnalysis sample (non-citizens): {len(analysis_sample):,}")

# Control group: arrived young, immigrated early, but too old
analysis_sample['control_group'] = (
    (analysis_sample['arrived_young'] == 1) &
    (analysis_sample['early_immig'] == 1) &
    (analysis_sample['under_31'] == 0)
).astype(int)

print(f"Treatment (DACA eligible): {analysis_sample['DACA_ELIGIBLE'].sum():,}")
print(f"Control (similar but too old): {analysis_sample['control_group'].sum():,}")

# Main analysis sample: treatment + control
main_sample = analysis_sample[
    (analysis_sample['DACA_ELIGIBLE'] == 1) | (analysis_sample['control_group'] == 1)
].copy()
print(f"\nMain DiD sample: {len(main_sample):,}")

#############################################################################
# STEP 6: Summary Statistics
#############################################################################
print("\n" + "="*80)
print("STEP 6: Summary Statistics")
print("="*80)

# Create demographic variables
main_sample['female'] = (main_sample['SEX'] == 2).astype(int)
main_sample['married'] = (main_sample['MARST'] == 1).astype(int)
main_sample['educ_hs'] = (main_sample['EDUC'] >= 6).astype(int)
main_sample['educ_college'] = (main_sample['EDUC'] >= 10).astype(int)

# Summary by period and treatment
print("\n--- Pre-Period (2006-2011) Characteristics ---")
pre = main_sample[main_sample['POST'] == 0]
post = main_sample[main_sample['POST'] == 1]

for var in ['AGE', 'fulltime', 'employed', 'female', 'married', 'educ_hs']:
    treat_pre = pre[pre['DACA_ELIGIBLE']==1][var].mean()
    ctrl_pre = pre[pre['DACA_ELIGIBLE']==0][var].mean()
    treat_post = post[post['DACA_ELIGIBLE']==1][var].mean()
    ctrl_post = post[post['DACA_ELIGIBLE']==0][var].mean()
    print(f"{var:12s}: Pre(T={treat_pre:.3f}, C={ctrl_pre:.3f}), Post(T={treat_post:.3f}, C={ctrl_post:.3f})")

print(f"\nSample by year and treatment:")
print(main_sample.groupby(['YEAR', 'DACA_ELIGIBLE']).size().unstack().fillna(0).astype(int))

#############################################################################
# STEP 7: Difference-in-Differences Analysis
#############################################################################
print("\n" + "="*80)
print("STEP 7: Difference-in-Differences Analysis")
print("="*80)

# Create interaction
main_sample['DACA_x_POST'] = main_sample['DACA_ELIGIBLE'] * main_sample['POST']

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ DACA_ELIGIBLE + POST + DACA_x_POST',
                  data=main_sample, weights=main_sample['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary().tables[1])

# Model 2: With demographic controls
print("\n--- Model 2: With Demographic Controls ---")
model2 = smf.wls('fulltime ~ DACA_ELIGIBLE + POST + DACA_x_POST + AGE + I(AGE**2) + female + married + educ_hs',
                  data=main_sample, weights=main_sample['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])

# Model 3: With year fixed effects
print("\n--- Model 3: Year Fixed Effects ---")
model3 = smf.wls('fulltime ~ DACA_ELIGIBLE + C(YEAR) + DACA_x_POST + AGE + I(AGE**2) + female + married + educ_hs',
                  data=main_sample, weights=main_sample['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(f"\nDACA_x_POST: {results3.params['DACA_x_POST']:.4f} (SE: {results3.bse['DACA_x_POST']:.4f}, p: {results3.pvalues['DACA_x_POST']:.4f})")

# Model 4: With state fixed effects
print("\n--- Model 4: State + Year Fixed Effects ---")
model4 = smf.wls('fulltime ~ DACA_ELIGIBLE + C(YEAR) + DACA_x_POST + AGE + I(AGE**2) + female + married + educ_hs + C(STATEFIP)',
                  data=main_sample, weights=main_sample['PERWT'])
results4 = model4.fit(cov_type='HC1')

print(f"\nPreferred Model (State + Year FE):")
print(f"  DACA_x_POST: {results4.params['DACA_x_POST']:.4f}")
print(f"  Std Error: {results4.bse['DACA_x_POST']:.4f}")
print(f"  t-stat: {results4.tvalues['DACA_x_POST']:.3f}")
print(f"  p-value: {results4.pvalues['DACA_x_POST']:.4f}")
ci4 = results4.conf_int().loc['DACA_x_POST']
print(f"  95% CI: [{ci4[0]:.4f}, {ci4[1]:.4f}]")

#############################################################################
# STEP 8: Robustness Checks
#############################################################################
print("\n" + "="*80)
print("STEP 8: Robustness Checks")
print("="*80)

# R1: Any employment outcome
print("\n--- Robustness 1: Any Employment ---")
model_r1 = smf.wls('employed ~ DACA_ELIGIBLE + C(YEAR) + DACA_x_POST + AGE + I(AGE**2) + female + married + educ_hs + C(STATEFIP)',
                    data=main_sample, weights=main_sample['PERWT'])
results_r1 = model_r1.fit(cov_type='HC1')
print(f"DACA_x_POST: {results_r1.params['DACA_x_POST']:.4f} (SE: {results_r1.bse['DACA_x_POST']:.4f})")

# R2: Males only
print("\n--- Robustness 2: Males Only ---")
male_sample = main_sample[main_sample['female'] == 0]
model_r2 = smf.wls('fulltime ~ DACA_ELIGIBLE + C(YEAR) + DACA_x_POST + AGE + I(AGE**2) + married + educ_hs + C(STATEFIP)',
                    data=male_sample, weights=male_sample['PERWT'])
results_r2 = model_r2.fit(cov_type='HC1')
print(f"DACA_x_POST: {results_r2.params['DACA_x_POST']:.4f} (SE: {results_r2.bse['DACA_x_POST']:.4f})")

# R3: Females only
print("\n--- Robustness 3: Females Only ---")
female_sample = main_sample[main_sample['female'] == 1]
model_r3 = smf.wls('fulltime ~ DACA_ELIGIBLE + C(YEAR) + DACA_x_POST + AGE + I(AGE**2) + married + educ_hs + C(STATEFIP)',
                    data=female_sample, weights=female_sample['PERWT'])
results_r3 = model_r3.fit(cov_type='HC1')
print(f"DACA_x_POST: {results_r3.params['DACA_x_POST']:.4f} (SE: {results_r3.bse['DACA_x_POST']:.4f})")

# R4: Alternative control - all non-eligible non-citizens
print("\n--- Robustness 4: Broader Control Group ---")
alt_sample = analysis_sample.copy()
alt_sample['female'] = (alt_sample['SEX'] == 2).astype(int)
alt_sample['married'] = (alt_sample['MARST'] == 1).astype(int)
alt_sample['educ_hs'] = (alt_sample['EDUC'] >= 6).astype(int)
alt_sample['DACA_x_POST'] = alt_sample['DACA_ELIGIBLE'] * alt_sample['POST']
model_r4 = smf.wls('fulltime ~ DACA_ELIGIBLE + C(YEAR) + DACA_x_POST + AGE + I(AGE**2) + female + married + educ_hs + C(STATEFIP)',
                    data=alt_sample, weights=alt_sample['PERWT'])
results_r4 = model_r4.fit(cov_type='HC1')
print(f"DACA_x_POST: {results_r4.params['DACA_x_POST']:.4f} (SE: {results_r4.bse['DACA_x_POST']:.4f})")

#############################################################################
# STEP 9: Event Study
#############################################################################
print("\n" + "="*80)
print("STEP 9: Event Study Analysis")
print("="*80)

years = sorted(main_sample['YEAR'].unique())
ref_year = 2011

for yr in years:
    main_sample[f'DACA_x_{yr}'] = main_sample['DACA_ELIGIBLE'] * (main_sample['YEAR'] == yr).astype(int)

year_interactions = ' + '.join([f'DACA_x_{yr}' for yr in years if yr != ref_year])
formula_es = f'fulltime ~ DACA_ELIGIBLE + C(YEAR) + {year_interactions} + AGE + I(AGE**2) + female + married + educ_hs + C(STATEFIP)'

model_es = smf.wls(formula_es, data=main_sample, weights=main_sample['PERWT'])
results_es = model_es.fit(cov_type='HC1')

print(f"\nEvent Study Coefficients (Reference: {ref_year}):")
print("-" * 50)
event_study_data = []
for yr in years:
    if yr != ref_year:
        coef = results_es.params[f'DACA_x_{yr}']
        se = results_es.bse[f'DACA_x_{yr}']
        ci = results_es.conf_int().loc[f'DACA_x_{yr}']
        print(f"  Year {yr}: {coef:7.4f} (SE: {se:.4f}) [{ci[0]:.4f}, {ci[1]:.4f}]")
        event_study_data.append({'Year': yr, 'Coefficient': coef, 'SE': se, 'CI_lower': ci[0], 'CI_upper': ci[1]})
    else:
        event_study_data.append({'Year': yr, 'Coefficient': 0, 'SE': 0, 'CI_lower': 0, 'CI_upper': 0})

event_study_df = pd.DataFrame(event_study_data).sort_values('Year')

#############################################################################
# STEP 10: Export Results
#############################################################################
print("\n" + "="*80)
print("STEP 10: Export Results")
print("="*80)

# Summary statistics table
summary_stats = []
for period_name, period_data in [('Pre-Period', pre), ('Post-Period', post)]:
    for group_name, group_val in [('DACA Eligible', 1), ('Control', 0)]:
        grp = period_data[period_data['DACA_ELIGIBLE'] == group_val]
        row = {
            'Period': period_name,
            'Group': group_name,
            'N': len(grp),
            'Full-time': grp['fulltime'].mean(),
            'Employed': grp['employed'].mean(),
            'Age': grp['AGE'].mean(),
            'Female': grp['female'].mean(),
            'Married': grp['married'].mean(),
            'HS+': grp['educ_hs'].mean()
        }
        summary_stats.append(row)

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_statistics.csv', index=False)
print("\nSummary Statistics:")
print(summary_df.to_string(index=False))

# Regression results
reg_results = pd.DataFrame({
    'Model': ['Basic DiD', 'Demographics', 'Year FE', 'State+Year FE'],
    'Coefficient': [results1.params['DACA_x_POST'], results2.params['DACA_x_POST'],
                    results3.params['DACA_x_POST'], results4.params['DACA_x_POST']],
    'SE': [results1.bse['DACA_x_POST'], results2.bse['DACA_x_POST'],
           results3.bse['DACA_x_POST'], results4.bse['DACA_x_POST']],
    'p_value': [results1.pvalues['DACA_x_POST'], results2.pvalues['DACA_x_POST'],
                results3.pvalues['DACA_x_POST'], results4.pvalues['DACA_x_POST']],
    'CI_lower': [results1.conf_int().loc['DACA_x_POST', 0], results2.conf_int().loc['DACA_x_POST', 0],
                 results3.conf_int().loc['DACA_x_POST', 0], results4.conf_int().loc['DACA_x_POST', 0]],
    'CI_upper': [results1.conf_int().loc['DACA_x_POST', 1], results2.conf_int().loc['DACA_x_POST', 1],
                 results3.conf_int().loc['DACA_x_POST', 1], results4.conf_int().loc['DACA_x_POST', 1]],
    'N': [int(results1.nobs), int(results2.nobs), int(results3.nobs), int(results4.nobs)]
})
reg_results.to_csv('regression_results.csv', index=False)
print("\nRegression Results:")
print(reg_results.to_string(index=False))

# Robustness results
rob_results = pd.DataFrame({
    'Specification': ['Any Employment', 'Males Only', 'Females Only', 'Broader Control'],
    'Coefficient': [results_r1.params['DACA_x_POST'], results_r2.params['DACA_x_POST'],
                    results_r3.params['DACA_x_POST'], results_r4.params['DACA_x_POST']],
    'SE': [results_r1.bse['DACA_x_POST'], results_r2.bse['DACA_x_POST'],
           results_r3.bse['DACA_x_POST'], results_r4.bse['DACA_x_POST']],
    'N': [int(results_r1.nobs), int(results_r2.nobs), int(results_r3.nobs), int(results_r4.nobs)]
})
rob_results.to_csv('robustness_results.csv', index=False)
print("\nRobustness Results:")
print(rob_results.to_string(index=False))

# Event study
event_study_df.to_csv('event_study_results.csv', index=False)

#############################################################################
# FINAL SUMMARY
#############################################################################
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
PREFERRED ESTIMATE (Model 4 - State + Year Fixed Effects):

Outcome: Full-time employment (UHRSWORK >= 35)
Treatment: DACA eligibility
Sample: Hispanic-Mexican, Mexican-born, non-citizens, ages 16-64

Effect Size: {results4.params['DACA_x_POST']:.4f}
Standard Error: {results4.bse['DACA_x_POST']:.4f}
95% CI: [{ci4[0]:.4f}, {ci4[1]:.4f}]
p-value: {results4.pvalues['DACA_x_POST']:.4f}
Sample Size: {int(results4.nobs):,}

Interpretation: DACA eligibility is associated with a {abs(results4.params['DACA_x_POST']*100):.2f} percentage point
{'increase' if results4.params['DACA_x_POST'] > 0 else 'decrease'} in the probability of full-time employment.
""")

# Save final results
with open('final_results.txt', 'w') as f:
    f.write("DACA REPLICATION ANALYSIS - FINAL RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Effect Size: {results4.params['DACA_x_POST']:.4f}\n")
    f.write(f"Standard Error: {results4.bse['DACA_x_POST']:.4f}\n")
    f.write(f"95% CI: [{ci4[0]:.4f}, {ci4[1]:.4f}]\n")
    f.write(f"p-value: {results4.pvalues['DACA_x_POST']:.4f}\n")
    f.write(f"Sample Size: {int(results4.nobs):,}\n")

print("\nAnalysis complete. Results exported to CSV files.")
