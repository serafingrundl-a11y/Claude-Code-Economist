"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
ethnically Hispanic-Mexican Mexican-born people.

Treatment group: Ages 26-30 at time of policy (June 15, 2012)
Control group: Ages 31-35 at time of policy (June 15, 2012)
Outcome: Full-time employment (usually working 35+ hours per week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")

# Read in chunks due to large file size
chunks = []
chunksize = 1000000
for chunk in pd.read_csv('data/data.csv', chunksize=chunksize, low_memory=False):
    # Filter to Hispanic-Mexican and Mexico-born
    # HISPAN == 1 is Mexican
    # BPL == 200 is Mexico
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk)
    print(f"Processed chunk, kept {len(chunk)} rows")

df = pd.concat(chunks, ignore_index=True)
print(f"Total Hispanic-Mexican Mexico-born observations: {len(df)}")

# Save filtered data for faster reprocessing if needed
df.to_csv('data/filtered_data.csv', index=False)
print("Filtered data saved to data/filtered_data.csv")

# Basic data exploration
print("\n=== Data Summary ===")
print(f"Years available: {sorted(df['YEAR'].unique())}")
print(f"Total observations: {len(df)}")
print(f"\nCITIZEN distribution:")
print(df['CITIZEN'].value_counts().sort_index())

# Create analysis sample
print("\n=== Creating Analysis Sample ===")

# DACA was implemented June 15, 2012
# Born June 15, 1981 or later = under 31 (eligible by age)
# Treatment: Age 26-30 at June 15, 2012 (born June 15, 1982 to June 15, 1986)
# Control: Age 31-35 at June 15, 2012 (born June 15, 1977 to June 15, 1981)

# For simplicity, use BIRTHYR to approximate:
# Treatment: Born 1982-1986 (ages 26-30 in 2012)
# Control: Born 1977-1981 (ages 31-35 in 2012)

# DACA eligibility requirements:
# 1. Not a citizen (CITIZEN == 3)
# 2. Arrived before age 16
# 3. Continuously present since June 15, 2007
# 4. Born in Mexico (already filtered)
# 5. Hispanic-Mexican (already filtered)

# Calculate age at arrival
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Filter for non-citizens only (undocumented proxy)
# CITIZEN == 3 means "Not a citizen"
df_noncit = df[df['CITIZEN'] == 3].copy()
print(f"Non-citizen observations: {len(df_noncit)}")

# Filter for those who arrived before age 16
df_noncit = df_noncit[df_noncit['age_at_arrival'] < 16].copy()
print(f"After filtering arrived before age 16: {len(df_noncit)}")

# Filter for continuous presence since June 15, 2007
# YRIMMIG must be <= 2007
df_noncit = df_noncit[df_noncit['YRIMMIG'] <= 2007].copy()
print(f"After filtering arrived by 2007: {len(df_noncit)}")

# Remove YRIMMIG == 0 (N/A)
df_noncit = df_noncit[df_noncit['YRIMMIG'] > 0].copy()
print(f"After removing missing YRIMMIG: {len(df_noncit)}")

# Create treatment and control groups based on birth year
# Treatment: Born 1982-1986 (ages 26-30 in 2012)
# Control: Born 1977-1981 (ages 31-35 in 2012)
df_noncit['treatment_group'] = ((df_noncit['BIRTHYR'] >= 1982) &
                                 (df_noncit['BIRTHYR'] <= 1986)).astype(int)
df_noncit['control_group'] = ((df_noncit['BIRTHYR'] >= 1977) &
                               (df_noncit['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control groups
df_analysis = df_noncit[(df_noncit['treatment_group'] == 1) |
                         (df_noncit['control_group'] == 1)].copy()
print(f"Analysis sample (treatment + control): {len(df_analysis)}")

# Create treated indicator (treatment group = 1, control = 0)
df_analysis['treated'] = df_analysis['treatment_group']

# Create post-treatment indicator
# Pre-treatment: 2006-2011
# Post-treatment: 2013-2016 (2012 excluded due to mid-year implementation)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
print(f"After excluding 2012: {len(df_analysis)}")

# Create outcome: full-time employment (UHRSWORK >= 35)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Create interaction term
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

# Summary statistics
print("\n=== Sample Breakdown ===")
print(f"Pre-treatment observations: {len(df_analysis[df_analysis['post']==0])}")
print(f"Post-treatment observations: {len(df_analysis[df_analysis['post']==1])}")
print(f"Treatment group (ages 26-30): {len(df_analysis[df_analysis['treated']==1])}")
print(f"Control group (ages 31-35): {len(df_analysis[df_analysis['treated']==0])}")

# Check by year
print("\n=== Observations by Year ===")
print(df_analysis.groupby('YEAR').size())

print("\n=== Full-time Employment Rates ===")
summary = df_analysis.groupby(['treated', 'post'])['fulltime'].agg(['mean', 'count'])
print(summary)

# Save analysis dataset
df_analysis.to_csv('analysis_sample.csv', index=False)
print("\nAnalysis sample saved to analysis_sample.csv")

# Run Difference-in-Differences regression
print("\n" + "="*60)
print("MAIN RESULTS: DIFFERENCE-IN-DIFFERENCES")
print("="*60)

# Basic DiD without covariates
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_analysis).fit()
print("\n--- Model 1: Basic DiD ---")
print(model1.summary())

# DiD with clustered standard errors by state
model1_clustered = smf.ols('fulltime ~ treated + post + treated_post',
                           data=df_analysis).fit(cov_type='cluster',
                                                  cov_kwds={'groups': df_analysis['STATEFIP']})
print("\n--- Model 1 with State-Clustered SE ---")
print(model1_clustered.summary())

# DiD with demographic controls
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # High school or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # Some college or more

model2 = smf.ols('fulltime ~ treated + post + treated_post + female + married + educ_hs',
                 data=df_analysis).fit(cov_type='cluster',
                                       cov_kwds={'groups': df_analysis['STATEFIP']})
print("\n--- Model 2: DiD with Demographics ---")
print(model2.summary())

# DiD with year fixed effects
year_dummies = pd.get_dummies(df_analysis['YEAR'], prefix='year', drop_first=True)
df_analysis = pd.concat([df_analysis, year_dummies], axis=1)

model3_formula = 'fulltime ~ treated + treated_post + female + married + educ_hs + ' + ' + '.join(year_dummies.columns)
model3 = smf.ols(model3_formula, data=df_analysis).fit(cov_type='cluster',
                                                        cov_kwds={'groups': df_analysis['STATEFIP']})
print("\n--- Model 3: DiD with Year FE and Demographics ---")
print(model3.summary())

# DiD with state fixed effects
state_dummies = pd.get_dummies(df_analysis['STATEFIP'], prefix='state', drop_first=True)
df_analysis = pd.concat([df_analysis, state_dummies], axis=1)

model4_formula = 'fulltime ~ treated + treated_post + female + married + educ_hs + ' + \
                 ' + '.join(year_dummies.columns) + ' + ' + ' + '.join(state_dummies.columns)
model4 = smf.ols(model4_formula, data=df_analysis).fit(cov_type='cluster',
                                                        cov_kwds={'groups': df_analysis['STATEFIP']})
print("\n--- Model 4: DiD with Year and State FE ---")
print(model4.summary())

# Extract key results
print("\n" + "="*60)
print("SUMMARY OF DiD ESTIMATES")
print("="*60)
results_table = pd.DataFrame({
    'Model': ['Basic DiD', 'DiD (Clustered SE)', 'DiD + Demographics', 'DiD + Year FE', 'DiD + Year & State FE'],
    'DiD Estimate': [model1.params['treated_post'],
                     model1_clustered.params['treated_post'],
                     model2.params['treated_post'],
                     model3.params['treated_post'],
                     model4.params['treated_post']],
    'Std Error': [model1.bse['treated_post'],
                  model1_clustered.bse['treated_post'],
                  model2.bse['treated_post'],
                  model3.bse['treated_post'],
                  model4.bse['treated_post']],
    'p-value': [model1.pvalues['treated_post'],
                model1_clustered.pvalues['treated_post'],
                model2.pvalues['treated_post'],
                model3.pvalues['treated_post'],
                model4.pvalues['treated_post']]
})
results_table['95% CI Lower'] = results_table['DiD Estimate'] - 1.96 * results_table['Std Error']
results_table['95% CI Upper'] = results_table['DiD Estimate'] + 1.96 * results_table['Std Error']
print(results_table.to_string(index=False))

# Save results table
results_table.to_csv('did_results.csv', index=False)

# Calculate means for DiD manual check
print("\n=== Manual DiD Calculation ===")
means = df_analysis.groupby(['treated', 'post'])['fulltime'].mean()
print(means)

# Treatment group change
treat_change = means[1, 1] - means[1, 0]
# Control group change
control_change = means[0, 1] - means[0, 0]
# DiD
did_manual = treat_change - control_change

print(f"\nTreatment group change: {treat_change:.4f}")
print(f"Control group change: {control_change:.4f}")
print(f"DiD (manual): {did_manual:.4f}")

# Event study / Parallel trends check
print("\n" + "="*60)
print("EVENT STUDY ANALYSIS")
print("="*60)

# Create year-specific treatment effects (relative to 2011)
df_analysis['year_treat'] = df_analysis['YEAR'].astype(str) + '_' + df_analysis['treated'].astype(str)
df_analysis['year_num'] = df_analysis['YEAR'] - 2012  # Center at 2012

# Create interaction terms for each year
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for y in years:
    df_analysis[f'treat_y{y}'] = ((df_analysis['YEAR'] == y) & (df_analysis['treated'] == 1)).astype(int)

# Event study regression (omit 2011 as reference)
event_formula = 'fulltime ~ treated + ' + ' + '.join([f'treat_y{y}' for y in years if y != 2011]) + \
                ' + ' + ' + '.join(year_dummies.columns)
event_model = smf.ols(event_formula, data=df_analysis).fit(cov_type='cluster',
                                                            cov_kwds={'groups': df_analysis['STATEFIP']})
print("\n--- Event Study Results ---")
print(event_model.summary())

# Extract event study coefficients
event_coefs = []
for y in years:
    if y == 2011:
        event_coefs.append({'Year': y, 'Coefficient': 0, 'SE': 0, 'p-value': np.nan})
    else:
        coef = event_model.params[f'treat_y{y}']
        se = event_model.bse[f'treat_y{y}']
        pval = event_model.pvalues[f'treat_y{y}']
        event_coefs.append({'Year': y, 'Coefficient': coef, 'SE': se, 'p-value': pval})

event_df = pd.DataFrame(event_coefs)
event_df['CI_lower'] = event_df['Coefficient'] - 1.96 * event_df['SE']
event_df['CI_upper'] = event_df['Coefficient'] + 1.96 * event_df['SE']
print("\n=== Event Study Coefficients ===")
print(event_df.to_string(index=False))
event_df.to_csv('event_study_results.csv', index=False)

# Robustness: Alternative outcome - any employment
print("\n" + "="*60)
print("ROBUSTNESS: ANY EMPLOYMENT")
print("="*60)

df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)
model_emp = smf.ols('employed ~ treated + post + treated_post + female + married + educ_hs',
                    data=df_analysis).fit(cov_type='cluster',
                                          cov_kwds={'groups': df_analysis['STATEFIP']})
print(model_emp.summary())

# Robustness: By sex
print("\n" + "="*60)
print("HETEROGENEITY BY SEX")
print("="*60)

# Males
df_male = df_analysis[df_analysis['SEX'] == 1]
model_male = smf.ols('fulltime ~ treated + post + treated_post + married + educ_hs',
                     data=df_male).fit(cov_type='cluster',
                                       cov_kwds={'groups': df_male['STATEFIP']})
print("\n--- Males ---")
print(f"DiD Estimate: {model_male.params['treated_post']:.4f}")
print(f"SE: {model_male.bse['treated_post']:.4f}")
print(f"p-value: {model_male.pvalues['treated_post']:.4f}")
print(f"N: {len(df_male)}")

# Females
df_female = df_analysis[df_analysis['SEX'] == 2]
model_female = smf.ols('fulltime ~ treated + post + treated_post + married + educ_hs',
                       data=df_female).fit(cov_type='cluster',
                                           cov_kwds={'groups': df_female['STATEFIP']})
print("\n--- Females ---")
print(f"DiD Estimate: {model_female.params['treated_post']:.4f}")
print(f"SE: {model_female.bse['treated_post']:.4f}")
print(f"p-value: {model_female.pvalues['treated_post']:.4f}")
print(f"N: {len(df_female)}")

# Summary statistics for paper
print("\n" + "="*60)
print("SUMMARY STATISTICS FOR PAPER")
print("="*60)

# Demographics
print("\n=== Sample Demographics ===")
print(f"Total N: {len(df_analysis)}")
print(f"Female: {df_analysis['female'].mean():.3f}")
print(f"Married: {df_analysis['married'].mean():.3f}")
print(f"High school+: {df_analysis['educ_hs'].mean():.3f}")
print(f"Mean age: {df_analysis['AGE'].mean():.1f}")
print(f"Mean UHRSWORK: {df_analysis['UHRSWORK'].mean():.1f}")
print(f"Full-time rate: {df_analysis['fulltime'].mean():.3f}")
print(f"Employment rate: {df_analysis['employed'].mean():.3f}")

# By treatment/control and pre/post
print("\n=== Summary by Group and Period ===")
summary_stats = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'UHRSWORK': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean'
}).round(3)
print(summary_stats)
summary_stats.to_csv('summary_stats.csv')

# Preferred estimate
print("\n" + "="*60)
print("PREFERRED ESTIMATE (Model 4: DiD with Year and State FE)")
print("="*60)
print(f"Effect size: {model4.params['treated_post']:.4f}")
print(f"Standard error: {model4.bse['treated_post']:.4f}")
print(f"95% CI: [{model4.params['treated_post'] - 1.96*model4.bse['treated_post']:.4f}, "
      f"{model4.params['treated_post'] + 1.96*model4.bse['treated_post']:.4f}]")
print(f"p-value: {model4.pvalues['treated_post']:.4f}")
print(f"Sample size: {len(df_analysis)}")

# Save preferred results
with open('preferred_estimate.txt', 'w') as f:
    f.write("PREFERRED ESTIMATE\n")
    f.write("="*50 + "\n")
    f.write(f"Effect size: {model4.params['treated_post']:.6f}\n")
    f.write(f"Standard error: {model4.bse['treated_post']:.6f}\n")
    f.write(f"95% CI lower: {model4.params['treated_post'] - 1.96*model4.bse['treated_post']:.6f}\n")
    f.write(f"95% CI upper: {model4.params['treated_post'] + 1.96*model4.bse['treated_post']:.6f}\n")
    f.write(f"p-value: {model4.pvalues['treated_post']:.6f}\n")
    f.write(f"Sample size: {len(df_analysis)}\n")

print("\nAnalysis complete!")
