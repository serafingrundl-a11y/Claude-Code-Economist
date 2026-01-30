"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US (2013-2016)

Identification Strategy: Difference-in-Differences
- Treatment: DACA-eligible individuals
- Control: Similar Mexican-born non-citizens who are NOT DACA-eligible
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'data/data.csv'
OUTPUT_DIR = '.'

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

# =============================================================================
# STEP 1: Load Data with filtering during read
# =============================================================================
print("\n[1] Loading ACS data (chunked approach)...")

# We'll read in chunks and filter to only keep the relevant subset
# This drastically reduces memory usage

# Only keep columns we need
usecols = ['YEAR', 'PERWT', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'BIRTHYR',
           'BIRTHQTR', 'AGE', 'SEX', 'EDUC', 'MARST', 'EMPSTAT', 'UHRSWORK',
           'STATEFIP']

chunks = []
chunk_size = 1000000

for chunk in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=chunk_size):
    # Filter immediately to reduce memory
    # Hispanic-Mexican (HISPAN == 1) born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)
    del chunk
    gc.collect()

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"    Hispanic-Mexican, Mexican-born observations: {len(df):,}")
print(f"    Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: Define Sample Restrictions
# =============================================================================
print("\n[2] Applying sample restrictions...")

# Restrict to non-citizens (CITIZEN == 3) - proxy for undocumented
df = df[df['CITIZEN'] == 3].copy()
print(f"    After non-citizen restriction: {len(df):,}")

# Exclude 2012 (can't distinguish pre/post DACA within survey year)
df = df[df['YEAR'] != 2012].copy()
print(f"    After excluding 2012: {len(df):,}")

# Working-age sample (18-45 for core analysis)
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 45)].copy()
print(f"    After age restriction (18-45): {len(df):,}")

# Must have valid immigration year for eligibility determination
df = df[df['YRIMMIG'] > 0].copy()
print(f"    After valid immigration year: {len(df):,}")

# Rename for clarity
df_mex = df
del df
gc.collect()

# =============================================================================
# STEP 3: Create DACA Eligibility Variables
# =============================================================================
print("\n[3] Creating DACA eligibility variables...")

# DACA Eligibility Criteria:
# 1. Arrived before age 16
# 2. Born after June 15, 1981 (under 31 as of June 15, 2012)
# 3. In US since at least June 15, 2007 (arrived before 2008 to be safe)
# 4. Non-citizen (already restricted)

# Calculate age at arrival
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# Criterion 1: Arrived before age 16
df_mex['arrived_before_16'] = (df_mex['age_at_arrival'] < 16).astype(int)

# Criterion 2: Born after June 15, 1981
# Be conservative: count as eligible if born in 1982 or later
# Or if born in 1981 Q3 or Q4 (July-Dec)
df_mex['born_after_june1981'] = (
    (df_mex['BIRTHYR'] >= 1982) |
    ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))
).astype(int)

# Criterion 3: In US since at least 2007
df_mex['in_us_since_2007'] = (df_mex['YRIMMIG'] <= 2007).astype(int)

# DACA Eligible: meets all criteria
df_mex['daca_eligible'] = (
    (df_mex['arrived_before_16'] == 1) &
    (df_mex['born_after_june1981'] == 1) &
    (df_mex['in_us_since_2007'] == 1)
).astype(int)

print(f"    DACA eligible: {df_mex['daca_eligible'].sum():,} ({100*df_mex['daca_eligible'].mean():.1f}%)")
print(f"    Non-eligible: {(df_mex['daca_eligible']==0).sum():,} ({100*(1-df_mex['daca_eligible'].mean()):.1f}%)")

# =============================================================================
# STEP 4: Create Outcome Variables
# =============================================================================
print("\n[4] Creating outcome variables...")

# Full-time employment: UHRSWORK >= 35 (among those employed)
# First, identify employed individuals
df_mex['employed'] = (df_mex['EMPSTAT'] == 1).astype(int)

# Full-time: works 35+ hours per week
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

# Combined: employed full-time
df_mex['employed_fulltime'] = (
    (df_mex['employed'] == 1) & (df_mex['fulltime'] == 1)
).astype(int)

print(f"    Employment rate: {100*df_mex['employed'].mean():.1f}%")
print(f"    Full-time employment rate: {100*df_mex['employed_fulltime'].mean():.1f}%")

# =============================================================================
# STEP 5: Create Treatment/Time Variables for DiD
# =============================================================================
print("\n[5] Creating DiD variables...")

# Post-DACA indicator (2013 and later)
df_mex['post'] = (df_mex['YEAR'] >= 2013).astype(int)

# DiD interaction term
df_mex['daca_post'] = df_mex['daca_eligible'] * df_mex['post']

print(f"    Pre-period (2006-2011): {(df_mex['post']==0).sum():,}")
print(f"    Post-period (2013-2016): {(df_mex['post']==1).sum():,}")

# =============================================================================
# STEP 6: Create Control Variables
# =============================================================================
print("\n[6] Creating control variables...")

# Sex dummy
df_mex['female'] = (df_mex['SEX'] == 2).astype(int)

# Age and age squared
df_mex['age_sq'] = df_mex['AGE'] ** 2

# Education categories
df_mex['educ_hs'] = (df_mex['EDUC'] >= 6).astype(int)  # High school or more
df_mex['educ_college'] = (df_mex['EDUC'] >= 10).astype(int)  # Some college or more

# Married
df_mex['married'] = (df_mex['MARST'] <= 2).astype(int)

# Create year dummies for fixed effects
for year in df_mex['YEAR'].unique():
    df_mex[f'year_{year}'] = (df_mex['YEAR'] == year).astype(int)

print("    Control variables created: female, age, age_sq, educ_hs, educ_college, married")

# =============================================================================
# STEP 7: Descriptive Statistics
# =============================================================================
print("\n[7] Generating descriptive statistics...")

# Summary by treatment status and period
desc_stats = df_mex.groupby(['daca_eligible', 'post']).agg({
    'employed_fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'educ_hs': 'mean',
    'married': 'mean',
    'PERWT': 'sum'
}).round(3)

print("\n    Summary Statistics by Group and Period:")
print(desc_stats)

# Save descriptive stats
desc_output = df_mex.groupby(['daca_eligible', 'post']).agg({
    'employed_fulltime': 'mean',
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'educ_hs': 'mean',
    'married': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)
desc_output.to_csv(f'{OUTPUT_DIR}/descriptive_stats.csv')

# =============================================================================
# STEP 8: Main DiD Regression
# =============================================================================
print("\n[8] Running main DiD regressions...")

# Model 1: Basic DiD (no controls)
print("\n    Model 1: Basic DiD")
model1 = smf.wls('employed_fulltime ~ daca_eligible + post + daca_post',
                  data=df_mex, weights=df_mex['PERWT']).fit(cov_type='cluster',
                  cov_kwds={'groups': df_mex['STATEFIP']})
print(f"    DiD Coefficient: {model1.params['daca_post']:.4f}")
print(f"    Std Error (clustered): {model1.bse['daca_post']:.4f}")
print(f"    P-value: {model1.pvalues['daca_post']:.4f}")

# Model 2: DiD with demographic controls
print("\n    Model 2: DiD with demographic controls")
model2 = smf.wls('employed_fulltime ~ daca_eligible + post + daca_post + female + AGE + age_sq + educ_hs + married',
                  data=df_mex, weights=df_mex['PERWT']).fit(cov_type='cluster',
                  cov_kwds={'groups': df_mex['STATEFIP']})
print(f"    DiD Coefficient: {model2.params['daca_post']:.4f}")
print(f"    Std Error (clustered): {model2.bse['daca_post']:.4f}")
print(f"    P-value: {model2.pvalues['daca_post']:.4f}")

# Model 3: DiD with year fixed effects
print("\n    Model 3: DiD with year fixed effects")
year_dummies = ' + '.join([f'year_{y}' for y in sorted(df_mex['YEAR'].unique())[1:]])  # drop one for identification
model3_formula = f'employed_fulltime ~ daca_eligible + daca_post + female + AGE + age_sq + educ_hs + married + {year_dummies}'
model3 = smf.wls(model3_formula, data=df_mex, weights=df_mex['PERWT']).fit(cov_type='cluster',
                  cov_kwds={'groups': df_mex['STATEFIP']})
print(f"    DiD Coefficient: {model3.params['daca_post']:.4f}")
print(f"    Std Error (clustered): {model3.bse['daca_post']:.4f}")
print(f"    P-value: {model3.pvalues['daca_post']:.4f}")

# Model 4: Full model with state fixed effects (preferred specification)
print("\n    Model 4: Full model with state fixed effects (PREFERRED)")
# Create state dummies
state_dummies = pd.get_dummies(df_mex['STATEFIP'], prefix='state', drop_first=True)
df_reg = pd.concat([df_mex.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)

# Get state dummy column names
state_cols = [c for c in df_reg.columns if c.startswith('state_')]
state_formula = ' + '.join(state_cols)

model4_formula = f'employed_fulltime ~ daca_eligible + daca_post + female + AGE + age_sq + educ_hs + married + {year_dummies} + {state_formula}'
model4 = smf.wls(model4_formula, data=df_reg, weights=df_reg['PERWT']).fit(cov_type='HC1')
print(f"    DiD Coefficient: {model4.params['daca_post']:.4f}")
print(f"    Std Error (robust): {model4.bse['daca_post']:.4f}")
print(f"    P-value: {model4.pvalues['daca_post']:.4f}")
print(f"    95% CI: [{model4.conf_int().loc['daca_post', 0]:.4f}, {model4.conf_int().loc['daca_post', 1]:.4f}]")

# =============================================================================
# STEP 9: Robustness Checks
# =============================================================================
print("\n[9] Running robustness checks...")

# Robustness 1: Alternative control group - arrived at 16-20
print("\n    Robustness 1: Alternative control (arrival age 16-20)")
df_robust1 = df_mex[(df_mex['arrived_before_16'] == 1) |
                    ((df_mex['age_at_arrival'] >= 16) & (df_mex['age_at_arrival'] <= 20))].copy()
robust1 = smf.wls('employed_fulltime ~ daca_eligible + post + daca_post + female + AGE + age_sq + educ_hs + married',
                   data=df_robust1, weights=df_robust1['PERWT']).fit(cov_type='cluster',
                   cov_kwds={'groups': df_robust1['STATEFIP']})
print(f"    DiD Coefficient: {robust1.params['daca_post']:.4f} (SE: {robust1.bse['daca_post']:.4f})")

# Robustness 2: Employment (any) instead of full-time
print("\n    Robustness 2: Any employment as outcome")
robust2 = smf.wls('employed ~ daca_eligible + post + daca_post + female + AGE + age_sq + educ_hs + married',
                   data=df_mex, weights=df_mex['PERWT']).fit(cov_type='cluster',
                   cov_kwds={'groups': df_mex['STATEFIP']})
print(f"    DiD Coefficient: {robust2.params['daca_post']:.4f} (SE: {robust2.bse['daca_post']:.4f})")

# Robustness 3: Narrow age band (18-35)
print("\n    Robustness 3: Narrower age band (18-35)")
df_robust3 = df_mex[(df_mex['AGE'] >= 18) & (df_mex['AGE'] <= 35)].copy()
robust3 = smf.wls('employed_fulltime ~ daca_eligible + post + daca_post + female + AGE + age_sq + educ_hs + married',
                   data=df_robust3, weights=df_robust3['PERWT']).fit(cov_type='cluster',
                   cov_kwds={'groups': df_robust3['STATEFIP']})
print(f"    DiD Coefficient: {robust3.params['daca_post']:.4f} (SE: {robust3.bse['daca_post']:.4f})")

# Robustness 4: By sex
print("\n    Robustness 4: By sex")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_mex[df_mex['SEX'] == sex].copy()
    robust_sex = smf.wls('employed_fulltime ~ daca_eligible + post + daca_post + AGE + age_sq + educ_hs + married',
                          data=df_sex, weights=df_sex['PERWT']).fit(cov_type='cluster',
                          cov_kwds={'groups': df_sex['STATEFIP']})
    print(f"    {label}: DiD = {robust_sex.params['daca_post']:.4f} (SE: {robust_sex.bse['daca_post']:.4f})")

# =============================================================================
# STEP 10: Event Study / Dynamic Effects
# =============================================================================
print("\n[10] Running event study analysis...")

# Create year-specific treatment effects
df_event = df_mex.copy()
for year in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df_event[f'daca_year_{year}'] = df_event['daca_eligible'] * (df_event['YEAR'] == year).astype(int)

event_vars = ' + '.join([f'daca_year_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])  # 2011 as reference
event_formula = f'employed_fulltime ~ daca_eligible + {event_vars} + female + AGE + age_sq + educ_hs + married + {year_dummies}'
event_model = smf.wls(event_formula, data=df_event, weights=df_event['PERWT']).fit(cov_type='cluster',
                       cov_kwds={'groups': df_event['STATEFIP']})

print("\n    Year-specific treatment effects (relative to 2011):")
event_results = []
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'daca_year_{year}'
    if var in event_model.params:
        coef = event_model.params[var]
        se = event_model.bse[var]
        event_results.append({'year': year, 'coefficient': coef, 'std_error': se})
        print(f"    {year}: {coef:.4f} (SE: {se:.4f})")

event_df = pd.DataFrame(event_results)
event_df.to_csv(f'{OUTPUT_DIR}/event_study_results.csv', index=False)

# =============================================================================
# STEP 11: Compute Simple DiD Manually for Validation
# =============================================================================
print("\n[11] Manual DiD calculation (validation)...")

# Calculate weighted means
def weighted_mean(df, var, weight='PERWT'):
    return np.average(df[var], weights=df[weight])

# 2x2 means
treat_pre = weighted_mean(df_mex[(df_mex['daca_eligible']==1) & (df_mex['post']==0)], 'employed_fulltime')
treat_post = weighted_mean(df_mex[(df_mex['daca_eligible']==1) & (df_mex['post']==1)], 'employed_fulltime')
ctrl_pre = weighted_mean(df_mex[(df_mex['daca_eligible']==0) & (df_mex['post']==0)], 'employed_fulltime')
ctrl_post = weighted_mean(df_mex[(df_mex['daca_eligible']==0) & (df_mex['post']==1)], 'employed_fulltime')

did_manual = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)

print(f"\n    Treatment group (DACA eligible):")
print(f"        Pre:  {treat_pre:.4f}")
print(f"        Post: {treat_post:.4f}")
print(f"        Change: {treat_post - treat_pre:.4f}")
print(f"\n    Control group (Not DACA eligible):")
print(f"        Pre:  {ctrl_pre:.4f}")
print(f"        Post: {ctrl_post:.4f}")
print(f"        Change: {ctrl_post - ctrl_pre:.4f}")
print(f"\n    DiD Estimate: {did_manual:.4f}")

# =============================================================================
# STEP 12: Save Results
# =============================================================================
print("\n[12] Saving results...")

# Main regression results
results_summary = {
    'Model': ['Basic DiD', 'With Controls', 'Year FE', 'State + Year FE'],
    'DiD_Coefficient': [model1.params['daca_post'], model2.params['daca_post'],
                        model3.params['daca_post'], model4.params['daca_post']],
    'Std_Error': [model1.bse['daca_post'], model2.bse['daca_post'],
                  model3.bse['daca_post'], model4.bse['daca_post']],
    'P_Value': [model1.pvalues['daca_post'], model2.pvalues['daca_post'],
                model3.pvalues['daca_post'], model4.pvalues['daca_post']],
    'N': [model1.nobs, model2.nobs, model3.nobs, model4.nobs]
}
results_df = pd.DataFrame(results_summary)
results_df.to_csv(f'{OUTPUT_DIR}/main_results.csv', index=False)
print(f"    Saved main_results.csv")

# Robustness results
robustness_summary = {
    'Specification': ['Alternative Control (age 16-20)', 'Any Employment Outcome',
                      'Age 18-35', 'Males Only', 'Females Only'],
    'DiD_Coefficient': [robust1.params['daca_post'], robust2.params['daca_post'],
                        robust3.params['daca_post'],
                        smf.wls('employed_fulltime ~ daca_eligible + post + daca_post + AGE + age_sq + educ_hs + married',
                                data=df_mex[df_mex['SEX']==1], weights=df_mex[df_mex['SEX']==1]['PERWT']).fit().params['daca_post'],
                        smf.wls('employed_fulltime ~ daca_eligible + post + daca_post + AGE + age_sq + educ_hs + married',
                                data=df_mex[df_mex['SEX']==2], weights=df_mex[df_mex['SEX']==2]['PERWT']).fit().params['daca_post']],
    'Std_Error': [robust1.bse['daca_post'], robust2.bse['daca_post'], robust3.bse['daca_post'], None, None]
}
robustness_df = pd.DataFrame(robustness_summary)
robustness_df.to_csv(f'{OUTPUT_DIR}/robustness_results.csv', index=False)
print(f"    Saved robustness_results.csv")

# Full model summary
with open(f'{OUTPUT_DIR}/preferred_model_summary.txt', 'w') as f:
    f.write("PREFERRED SPECIFICATION: Model 2 (DiD with Demographic Controls)\n")
    f.write("=" * 70 + "\n\n")
    f.write(str(model2.summary()))
    f.write("\n\n" + "=" * 70 + "\n")
    f.write("FULL MODEL WITH STATE FE (Model 4)\n")
    f.write("=" * 70 + "\n\n")
    # Print only key coefficients for model4
    f.write("Key Coefficients:\n")
    for var in ['daca_eligible', 'daca_post', 'female', 'AGE', 'age_sq', 'educ_hs', 'married']:
        if var in model4.params:
            f.write(f"  {var}: {model4.params[var]:.4f} (SE: {model4.bse[var]:.4f})\n")
print(f"    Saved preferred_model_summary.txt")

# Sample sizes by group
sample_sizes = df_mex.groupby(['daca_eligible', 'post']).agg({
    'PERWT': ['count', 'sum']
}).reset_index()
sample_sizes.columns = ['daca_eligible', 'post', 'unweighted_n', 'weighted_n']
sample_sizes.to_csv(f'{OUTPUT_DIR}/sample_sizes.csv', index=False)
print(f"    Saved sample_sizes.csv")

# =============================================================================
# STEP 13: Generate LaTeX Tables
# =============================================================================
print("\n[13] Generating LaTeX tables...")

# Table 1: Summary Statistics
latex_desc = """
\\begin{table}[htbp]
\\centering
\\caption{Summary Statistics by DACA Eligibility and Time Period}
\\label{tab:summary}
\\begin{tabular}{lcccc}
\\hline\\hline
 & \\multicolumn{2}{c}{DACA Eligible} & \\multicolumn{2}{c}{Not Eligible} \\\\
 & Pre-2012 & Post-2012 & Pre-2012 & Post-2012 \\\\
\\hline
"""

# Calculate stats
for var, label in [('employed_fulltime', 'Full-time Employed'),
                   ('employed', 'Employed'),
                   ('AGE', 'Age'),
                   ('female', 'Female'),
                   ('educ_hs', 'High School+'),
                   ('married', 'Married')]:
    row = f"{label} "
    for elig in [1, 0]:
        for post in [0, 1]:
            subset = df_mex[(df_mex['daca_eligible']==elig) & (df_mex['post']==post)]
            mean = weighted_mean(subset, var) if len(subset) > 0 else 0
            row += f"& {mean:.3f} "
    row += "\\\\\n"
    latex_desc += row

# Add sample sizes
row = "N (unweighted) "
for elig in [1, 0]:
    for post in [0, 1]:
        n = len(df_mex[(df_mex['daca_eligible']==elig) & (df_mex['post']==post)])
        row += f"& {n:,} "
row += "\\\\\n"
latex_desc += row

latex_desc += """\\hline\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Notes: Sample includes Hispanic-Mexican, Mexican-born non-citizens aged 18-45.
Pre-period: 2006-2011. Post-period: 2013-2016. Year 2012 excluded.
All statistics are weighted using ACS person weights.
\\end{tablenotes}
\\end{table}
"""

with open(f'{OUTPUT_DIR}/table_summary.tex', 'w') as f:
    f.write(latex_desc)
print(f"    Saved table_summary.tex")

# Table 2: Main Results
latex_main = """
\\begin{table}[htbp]
\\centering
\\caption{Effect of DACA Eligibility on Full-time Employment: Difference-in-Differences Estimates}
\\label{tab:main}
\\begin{tabular}{lcccc}
\\hline\\hline
 & (1) & (2) & (3) & (4) \\\\
 & Basic & Controls & Year FE & State+Year FE \\\\
\\hline
"""

latex_main += f"DACA Eligible $\\times$ Post & {model1.params['daca_post']:.4f} & {model2.params['daca_post']:.4f} & {model3.params['daca_post']:.4f} & {model4.params['daca_post']:.4f} \\\\\n"
latex_main += f" & ({model1.bse['daca_post']:.4f}) & ({model2.bse['daca_post']:.4f}) & ({model3.bse['daca_post']:.4f}) & ({model4.bse['daca_post']:.4f}) \\\\\n"
latex_main += f"DACA Eligible & {model1.params['daca_eligible']:.4f} & {model2.params['daca_eligible']:.4f} & {model3.params['daca_eligible']:.4f} & {model4.params['daca_eligible']:.4f} \\\\\n"
latex_main += f" & ({model1.bse['daca_eligible']:.4f}) & ({model2.bse['daca_eligible']:.4f}) & ({model3.bse['daca_eligible']:.4f}) & ({model4.bse['daca_eligible']:.4f}) \\\\\n"
latex_main += "\\hline\n"
latex_main += "Demographic Controls & No & Yes & Yes & Yes \\\\\n"
latex_main += "Year Fixed Effects & No & No & Yes & Yes \\\\\n"
latex_main += "State Fixed Effects & No & No & No & Yes \\\\\n"
latex_main += f"N & {int(model1.nobs):,} & {int(model2.nobs):,} & {int(model3.nobs):,} & {int(model4.nobs):,} \\\\\n"
latex_main += f"R-squared & {model1.rsquared:.3f} & {model2.rsquared:.3f} & {model3.rsquared:.3f} & {model4.rsquared:.3f} \\\\\n"

latex_main += """\\hline\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Notes: Robust standard errors in parentheses, clustered at the state level in columns (1)-(3).
Column (4) uses heteroskedasticity-robust standard errors.
Demographic controls include female, age, age squared, high school education indicator, and married indicator.
* p<0.10, ** p<0.05, *** p<0.01
\\end{tablenotes}
\\end{table}
"""

with open(f'{OUTPUT_DIR}/table_main.tex', 'w') as f:
    f.write(latex_main)
print(f"    Saved table_main.tex")

# Table 3: Robustness
latex_robust = """
\\begin{table}[htbp]
\\centering
\\caption{Robustness Checks}
\\label{tab:robust}
\\begin{tabular}{lcc}
\\hline\\hline
Specification & DiD Coefficient & Std. Error \\\\
\\hline
"""
latex_robust += f"Main specification & {model2.params['daca_post']:.4f} & {model2.bse['daca_post']:.4f} \\\\\n"
latex_robust += f"Alternative control (arrival age 16-20) & {robust1.params['daca_post']:.4f} & {robust1.bse['daca_post']:.4f} \\\\\n"
latex_robust += f"Any employment outcome & {robust2.params['daca_post']:.4f} & {robust2.bse['daca_post']:.4f} \\\\\n"
latex_robust += f"Narrower age band (18-35) & {robust3.params['daca_post']:.4f} & {robust3.bse['daca_post']:.4f} \\\\\n"

latex_robust += """\\hline\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Notes: All specifications include demographic controls (female, age, age squared,
high school education, married). Standard errors clustered at state level.
\\end{tablenotes}
\\end{table}
"""

with open(f'{OUTPUT_DIR}/table_robust.tex', 'w') as f:
    f.write(latex_robust)
print(f"    Saved table_robust.tex")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"""
PREFERRED ESTIMATE (Model 2 with demographic controls):
  Effect of DACA on Full-time Employment: {model2.params['daca_post']:.4f}
  Standard Error (clustered by state): {model2.bse['daca_post']:.4f}
  95% Confidence Interval: [{model2.conf_int().loc['daca_post', 0]:.4f}, {model2.conf_int().loc['daca_post', 1]:.4f}]
  P-value: {model2.pvalues['daca_post']:.4f}
  Sample Size: {int(model2.nobs):,}

INTERPRETATION:
  DACA eligibility is associated with a {abs(model2.params['daca_post'])*100:.2f} percentage point
  {'increase' if model2.params['daca_post'] > 0 else 'decrease'} in the probability of full-time employment.
  {'This effect is statistically significant at the 5% level.' if model2.pvalues['daca_post'] < 0.05 else 'This effect is not statistically significant at the 5% level.'}

FILES CREATED:
  - descriptive_stats.csv
  - main_results.csv
  - robustness_results.csv
  - event_study_results.csv
  - sample_sizes.csv
  - preferred_model_summary.txt
  - table_summary.tex
  - table_main.tex
  - table_robust.tex
""")
