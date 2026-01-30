"""
DACA Replication Study - Analysis Script
Replication 76: Effect of DACA on Full-Time Employment

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for DACA on the
probability of full-time employment (35+ hours/week)?

Treatment: Ages 26-30 on June 15, 2012 (eligible for DACA)
Control: Ages 31-35 on June 15, 2012 (ineligible due to age, but otherwise would qualify)
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n1. LOADING DATA...")

# Define columns to load
cols = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'HISPAN', 'HISPAND',
        'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'UHRSWORK', 'EMPSTAT', 'EMPSTATD',
        'SEX', 'EDUC', 'EDUCD', 'MARST', 'STATEFIP', 'LABFORCE']

# Load data
df = pd.read_csv('data/data.csv', usecols=cols, low_memory=False)
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. SAMPLE CONSTRUCTION
# =============================================================================
print("\n2. CONSTRUCTING ANALYSIS SAMPLE...")

# Step 1: Hispanic-Mexican ethnicity (HISPAN == 1 for Mexican)
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican filter: {len(df_sample):,}")

# Step 2: Born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After Mexico birthplace filter: {len(df_sample):,}")

# Step 3: Not a citizen (CITIZEN == 3)
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After non-citizen filter: {len(df_sample):,}")

# Step 4: Valid immigration year (not 0 and <= 2007 for continuous residence)
df_sample = df_sample[(df_sample['YRIMMIG'] > 0) & (df_sample['YRIMMIG'] <= 2007)].copy()
print(f"After immigration year filter (<=2007): {len(df_sample):,}")

# Step 5: Arrived before age 16
df_sample['age_at_immigration'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_immigration'] < 16].copy()
print(f"After arrived before age 16 filter: {len(df_sample):,}")

# Step 6: Define treatment and control groups based on age at June 15, 2012
# Treatment: born 1982-1986 (ages 26-30 on June 15, 2012)
# Control: born 1977-1981 (ages 31-35 on June 15, 2012)

# Note: For more precision with birth quarter:
# - Someone born in Q3/Q4 of 1981 would be 30 in June 2012
# - Someone born in Q1/Q2 of 1982 would be 30 in June 2012
# However, for simplicity and consistency with instructions, use birth year cutoffs

df_sample['treat'] = ((df_sample['BIRTHYR'] >= 1982) & (df_sample['BIRTHYR'] <= 1986)).astype(int)
df_sample['control'] = ((df_sample['BIRTHYR'] >= 1977) & (df_sample['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control groups
df_sample = df_sample[(df_sample['treat'] == 1) | (df_sample['control'] == 1)].copy()
print(f"After treatment/control group filter: {len(df_sample):,}")

# Step 7: Exclude 2012 (DACA implemented mid-year, cannot distinguish pre/post)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# =============================================================================
# 3. VARIABLE CONSTRUCTION
# =============================================================================
print("\n3. CONSTRUCTING ANALYSIS VARIABLES...")

# Post-treatment indicator (2013-2016)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Full-time employment outcome (35+ hours/week)
# UHRSWORK == 0 is N/A (not employed or not applicable)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Employed indicator (EMPSTAT == 1 is employed)
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# Full-time conditional on being in labor force
# For robustness, also consider employment among all

# Interaction term
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

# Female indicator
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)

# Married indicator
df_sample['married'] = (df_sample['MARST'] <= 2).astype(int)

# Education categories
def categorize_education(educ):
    if educ <= 5:  # Less than high school
        return 0
    elif educ == 6:  # High school
        return 1
    elif educ <= 9:  # Some college
        return 2
    else:  # College or higher
        return 3

df_sample['educ_cat'] = df_sample['EDUC'].apply(categorize_education)

# Create year dummies for event study
for year in df_sample['YEAR'].unique():
    df_sample[f'year_{year}'] = (df_sample['YEAR'] == year).astype(int)

# Print sample summary
print(f"\nFinal sample size: {len(df_sample):,}")
print(f"Treatment group: {len(df_sample[df_sample['treat']==1]):,}")
print(f"Control group: {len(df_sample[df_sample['control']==1]):,}")
print(f"Pre-period (2006-2011): {len(df_sample[df_sample['post']==0]):,}")
print(f"Post-period (2013-2016): {len(df_sample[df_sample['post']==1]):,}")

# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n4. DESCRIPTIVE STATISTICS")
print("=" * 80)

# Summary statistics by treatment/control and pre/post
groups = df_sample.groupby(['treat', 'post'])

print("\nSample sizes by group:")
print(groups.size().unstack())

print("\n\nMean full-time employment rate by group:")
ft_means = groups['fulltime'].apply(lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']))
print(ft_means.unstack())

print("\n\nMean employment rate by group:")
emp_means = groups['employed'].apply(lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']))
print(emp_means.unstack())

# Demographic characteristics
print("\n\nDemographic characteristics (weighted):")
for var in ['female', 'married', 'AGE']:
    treat_mean = np.average(df_sample[df_sample['treat']==1][var],
                           weights=df_sample[df_sample['treat']==1]['PERWT'])
    control_mean = np.average(df_sample[df_sample['control']==1][var],
                             weights=df_sample[df_sample['control']==1]['PERWT'])
    print(f"{var}: Treatment={treat_mean:.3f}, Control={control_mean:.3f}")

# =============================================================================
# 5. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n5. DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (no covariates, unweighted) ---")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_sample).fit()
print(f"DiD Estimate (treat_post): {model1.params['treat_post']:.4f}")
print(f"Standard Error: {model1.bse['treat_post']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model1.pvalues['treat_post']:.4f}")
print(f"N: {model1.nobs:.0f}")

# Model 2: Basic DiD (weighted)
print("\n--- Model 2: Basic DiD (no covariates, weighted) ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post', data=df_sample, weights=df_sample['PERWT']).fit()
print(f"DiD Estimate (treat_post): {model2.params['treat_post']:.4f}")
print(f"Standard Error: {model2.bse['treat_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model2.pvalues['treat_post']:.4f}")
print(f"N: {model2.nobs:.0f}")

# Model 3: DiD with demographic covariates
print("\n--- Model 3: DiD with demographic covariates (weighted) ---")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + C(educ_cat)',
                 data=df_sample, weights=df_sample['PERWT']).fit()
print(f"DiD Estimate (treat_post): {model3.params['treat_post']:.4f}")
print(f"Standard Error: {model3.bse['treat_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model3.pvalues['treat_post']:.4f}")
print(f"N: {model3.nobs:.0f}")

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with state fixed effects (weighted) ---")
model4 = smf.wls('fulltime ~ treat + post + treat_post + female + married + C(educ_cat) + C(STATEFIP)',
                 data=df_sample, weights=df_sample['PERWT']).fit()
print(f"DiD Estimate (treat_post): {model4.params['treat_post']:.4f}")
print(f"Standard Error: {model4.bse['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model4.pvalues['treat_post']:.4f}")
print(f"N: {model4.nobs:.0f}")

# Model 5: DiD with year fixed effects (preferred specification)
print("\n--- Model 5: DiD with year and state FE (PREFERRED, weighted) ---")
model5 = smf.wls('fulltime ~ treat + treat_post + female + married + C(educ_cat) + C(STATEFIP) + C(YEAR)',
                 data=df_sample, weights=df_sample['PERWT']).fit()
print(f"DiD Estimate (treat_post): {model5.params['treat_post']:.4f}")
print(f"Standard Error: {model5.bse['treat_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model5.pvalues['treat_post']:.4f}")
print(f"N: {model5.nobs:.0f}")

# Model 6: Robust standard errors (preferred with clustering)
print("\n--- Model 6: DiD with robust standard errors (weighted) ---")
model6 = smf.wls('fulltime ~ treat + treat_post + female + married + C(educ_cat) + C(STATEFIP) + C(YEAR)',
                 data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate (treat_post): {model6.params['treat_post']:.4f}")
print(f"Robust Standard Error: {model6.bse['treat_post']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['treat_post', 0]:.4f}, {model6.conf_int().loc['treat_post', 1]:.4f}]")
print(f"P-value: {model6.pvalues['treat_post']:.4f}")
print(f"N: {model6.nobs:.0f}")

# =============================================================================
# 6. ROBUSTNESS CHECKS
# =============================================================================
print("\n6. ROBUSTNESS CHECKS")
print("=" * 80)

# Check 1: Employment as outcome (instead of full-time)
print("\n--- Robustness 1: Employment as outcome ---")
model_emp = smf.wls('employed ~ treat + treat_post + female + married + C(educ_cat) + C(STATEFIP) + C(YEAR)',
                    data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate: {model_emp.params['treat_post']:.4f}")
print(f"SE: {model_emp.bse['treat_post']:.4f}")
print(f"P-value: {model_emp.pvalues['treat_post']:.4f}")

# Check 2: Placebo test - different age cutoffs
print("\n--- Robustness 2: Placebo test (ages 36-40 vs 41-45) ---")
df_placebo = df[(df['HISPAN'] == 1) & (df['BPL'] == 200) & (df['CITIZEN'] == 3)].copy()
df_placebo = df_placebo[(df_placebo['YRIMMIG'] > 0) & (df_placebo['YRIMMIG'] <= 2007)]
df_placebo['age_at_immigration'] = df_placebo['YRIMMIG'] - df_placebo['BIRTHYR']
df_placebo = df_placebo[df_placebo['age_at_immigration'] < 16]

# Placebo groups: 36-40 (born 1972-1976) vs 41-45 (born 1967-1971)
df_placebo['treat'] = ((df_placebo['BIRTHYR'] >= 1972) & (df_placebo['BIRTHYR'] <= 1976)).astype(int)
df_placebo['control'] = ((df_placebo['BIRTHYR'] >= 1967) & (df_placebo['BIRTHYR'] <= 1971)).astype(int)
df_placebo = df_placebo[(df_placebo['treat'] == 1) | (df_placebo['control'] == 1)]
df_placebo = df_placebo[df_placebo['YEAR'] != 2012]
df_placebo['post'] = (df_placebo['YEAR'] >= 2013).astype(int)
df_placebo['treat_post'] = df_placebo['treat'] * df_placebo['post']
df_placebo['fulltime'] = (df_placebo['UHRSWORK'] >= 35).astype(int)
df_placebo['female'] = (df_placebo['SEX'] == 2).astype(int)
df_placebo['married'] = (df_placebo['MARST'] <= 2).astype(int)
df_placebo['educ_cat'] = df_placebo['EDUC'].apply(categorize_education)

model_placebo = smf.wls('fulltime ~ treat + treat_post + female + married + C(educ_cat) + C(STATEFIP) + C(YEAR)',
                        data=df_placebo, weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD Estimate: {model_placebo.params['treat_post']:.4f}")
print(f"SE: {model_placebo.bse['treat_post']:.4f}")
print(f"P-value: {model_placebo.pvalues['treat_post']:.4f}")

# Check 3: Men only
print("\n--- Robustness 3: Men only ---")
df_men = df_sample[df_sample['female'] == 0]
model_men = smf.wls('fulltime ~ treat + treat_post + married + C(educ_cat) + C(STATEFIP) + C(YEAR)',
                    data=df_men, weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate (Men): {model_men.params['treat_post']:.4f}")
print(f"SE: {model_men.bse['treat_post']:.4f}")
print(f"P-value: {model_men.pvalues['treat_post']:.4f}")

# Check 4: Women only
print("\n--- Robustness 4: Women only ---")
df_women = df_sample[df_sample['female'] == 1]
model_women = smf.wls('fulltime ~ treat + treat_post + married + C(educ_cat) + C(STATEFIP) + C(YEAR)',
                      data=df_women, weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate (Women): {model_women.params['treat_post']:.4f}")
print(f"SE: {model_women.bse['treat_post']:.4f}")
print(f"P-value: {model_women.pvalues['treat_post']:.4f}")

# =============================================================================
# 7. EVENT STUDY ANALYSIS
# =============================================================================
print("\n7. EVENT STUDY ANALYSIS")
print("=" * 80)

# Create interaction terms for each year
years = sorted(df_sample['YEAR'].unique())
base_year = 2011  # Last pre-treatment year as reference

for year in years:
    df_sample[f'treat_year_{year}'] = df_sample['treat'] * (df_sample['YEAR'] == year).astype(int)

# Remove the base year interaction
year_interactions = [f'treat_year_{y}' for y in years if y != base_year]
formula_event = 'fulltime ~ treat + ' + ' + '.join(year_interactions) + ' + female + married + C(educ_cat) + C(STATEFIP) + C(YEAR)'

model_event = smf.wls(formula_event, data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
for year in years:
    if year != base_year:
        coef = model_event.params[f'treat_year_{year}']
        se = model_event.bse[f'treat_year_{year}']
        pval = model_event.pvalues[f'treat_year_{year}']
        print(f"Year {year}: Coef={coef:.4f}, SE={se:.4f}, p={pval:.4f}")

# =============================================================================
# 8. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n8. SAVING RESULTS")
print("=" * 80)

# Create results dictionary for export
results = {
    'preferred_estimate': model6.params['treat_post'],
    'preferred_se': model6.bse['treat_post'],
    'preferred_ci_lower': model6.conf_int().loc['treat_post', 0],
    'preferred_ci_upper': model6.conf_int().loc['treat_post', 1],
    'preferred_pvalue': model6.pvalues['treat_post'],
    'sample_size': int(model6.nobs),
    'n_treatment': int(df_sample['treat'].sum()),
    'n_control': int((1 - df_sample['treat']).sum()),
}

# Save results to file
with open('results_summary.txt', 'w') as f:
    f.write("DACA Replication Study - Results Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write("PREFERRED ESTIMATE (Model 6: DiD with year/state FE, robust SE)\n")
    f.write("-" * 60 + "\n")
    f.write(f"Effect estimate: {results['preferred_estimate']:.4f}\n")
    f.write(f"Standard error: {results['preferred_se']:.4f}\n")
    f.write(f"95% CI: [{results['preferred_ci_lower']:.4f}, {results['preferred_ci_upper']:.4f}]\n")
    f.write(f"P-value: {results['preferred_pvalue']:.4f}\n")
    f.write(f"Sample size: {results['sample_size']:,}\n")
    f.write(f"Treatment group N: {results['n_treatment']:,}\n")
    f.write(f"Control group N: {results['n_control']:,}\n")

print("Results saved to results_summary.txt")

# =============================================================================
# 9. GENERATE TABLES AND FIGURES DATA
# =============================================================================
print("\n9. GENERATING DATA FOR TABLES AND FIGURES")
print("=" * 80)

# Table 1: Summary statistics
print("\nGenerating summary statistics table data...")

# Calculate weighted means
def weighted_mean(group, var, weight='PERWT'):
    return np.average(group[var], weights=group[weight])

summary_stats = []
for (treat, post), group in df_sample.groupby(['treat', 'post']):
    stats_dict = {
        'Treatment': 'Treated (26-30)' if treat == 1 else 'Control (31-35)',
        'Period': 'Post (2013-2016)' if post == 1 else 'Pre (2006-2011)',
        'N': len(group),
        'Full-time (%)': weighted_mean(group, 'fulltime') * 100,
        'Employed (%)': weighted_mean(group, 'employed') * 100,
        'Female (%)': weighted_mean(group, 'female') * 100,
        'Married (%)': weighted_mean(group, 'married') * 100,
        'Age': weighted_mean(group, 'AGE'),
    }
    summary_stats.append(stats_dict)

summary_df = pd.DataFrame(summary_stats)
print(summary_df.to_string(index=False))

# Table 2: Regression results
print("\n\nRegression results summary:")
print("-" * 80)
models_summary = [
    ('Model 1: Basic DiD', model1),
    ('Model 2: Weighted', model2),
    ('Model 3: + Demographics', model3),
    ('Model 4: + State FE', model4),
    ('Model 5: + Year FE', model5),
    ('Model 6: Robust SE', model6),
]

for name, model in models_summary:
    print(f"{name}: coef={model.params['treat_post']:.4f}, se={model.bse['treat_post']:.4f}, p={model.pvalues['treat_post']:.4f}")

# Event study coefficients for figure
print("\n\nEvent study coefficients for figure:")
event_study_data = []
for year in years:
    if year == base_year:
        event_study_data.append({'year': year, 'coef': 0, 'se': 0, 'ci_lower': 0, 'ci_upper': 0})
    else:
        coef = model_event.params[f'treat_year_{year}']
        se = model_event.bse[f'treat_year_{year}']
        ci = model_event.conf_int().loc[f'treat_year_{year}']
        event_study_data.append({'year': year, 'coef': coef, 'se': se, 'ci_lower': ci[0], 'ci_upper': ci[1]})

event_df = pd.DataFrame(event_study_data)
print(event_df.to_string(index=False))

# Save for later use
summary_df.to_csv('summary_statistics.csv', index=False)
event_df.to_csv('event_study_coefficients.csv', index=False)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
