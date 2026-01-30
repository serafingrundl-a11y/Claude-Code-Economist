"""
DACA Replication Study - Analysis Script
Research Question: Impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US
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

# -----------------------------------------------------------------------------
# 1. LOAD DATA (with filtering during read to reduce memory)
# -----------------------------------------------------------------------------
print("\n[1] Loading data with chunked processing...")
data_path = "data/data.csv"

# Read in chunks and filter to Hispanic-Mexican born in Mexico
chunks = []
chunksize = 1000000  # 1 million rows at a time

# Define dtypes to reduce memory
dtype_dict = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'PUMA': 'int16',
    'SEX': 'int8',
    'AGE': 'int8',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8',
    'PERWT': 'float32'
}

for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize, dtype=dtype_dict,
                                       usecols=['YEAR', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR',
                                               'MARST', 'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN',
                                               'YRIMMIG', 'EDUC', 'EMPSTAT', 'UHRSWORK', 'PERWT'])):
    # Filter to Hispanic-Mexican (HISPAN == 1) born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)
    if (i + 1) % 10 == 0:
        print(f"   Processed {(i+1) * chunksize:,} rows...")

df_mex = pd.concat(chunks, ignore_index=True)
print(f"   Total Hispanic-Mexican, Mexican-born observations: {len(df_mex):,}")
print(f"   Years in data: {sorted(df_mex['YEAR'].unique())}")

# Clean up
del chunks

# -----------------------------------------------------------------------------
# 2. CREATE KEY VARIABLES
# -----------------------------------------------------------------------------
print("\n[2] Creating key variables...")

# Age at time of immigration
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# Filter to valid immigration year
df_mex = df_mex[df_mex['YRIMMIG'] > 0].copy()
print(f"   After filtering to valid YRIMMIG: {len(df_mex):,}")

# Non-citizen indicator (CITIZEN == 3)
df_mex['non_citizen'] = (df_mex['CITIZEN'] == 3).astype('int8')
print(f"   Non-citizens: {df_mex['non_citizen'].sum():,} ({100*df_mex['non_citizen'].mean():.1f}%)")

# Full-time employment (UHRSWORK >= 35)
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype('int8')

# Employed indicator
df_mex['employed'] = (df_mex['EMPSTAT'] == 1).astype('int8')

# -----------------------------------------------------------------------------
# 3. DEFINE DACA ELIGIBILITY
# -----------------------------------------------------------------------------
print("\n[3] Defining DACA eligibility...")

# DACA Eligibility Criteria:
# 1. Arrived before 16th birthday: age_at_arrival < 16
# 2. Under 31 on June 15, 2012:
#    - Born after June 15, 1981
#    - If born in 1982 or later: definitely under 31
#    - If born in 1981: need BIRTHQTR >= 3 (July or later)
# 3. In US since June 15, 2007: YRIMMIG <= 2007
# 4. Non-citizen (assuming undocumented)

# Criterion 1: Arrived before 16
crit_arrival = df_mex['age_at_arrival'] < 16

# Criterion 2: Under 31 on June 15, 2012
crit_under31 = (df_mex['BIRTHYR'] >= 1982) | \
               ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))

# Criterion 3: In US since June 15, 2007
crit_continuous = df_mex['YRIMMIG'] <= 2007

# Criterion 4: Non-citizen
crit_noncitizen = df_mex['CITIZEN'] == 3

# DACA eligible if all criteria met
df_mex['daca_eligible'] = (crit_arrival & crit_under31 & crit_continuous & crit_noncitizen).astype('int8')

print(f"   Criterion 1 (arrived < 16): {crit_arrival.sum():,}")
print(f"   Criterion 2 (under 31 on 6/15/12): {crit_under31.sum():,}")
print(f"   Criterion 3 (in US since 2007): {crit_continuous.sum():,}")
print(f"   Criterion 4 (non-citizen): {crit_noncitizen.sum():,}")
print(f"   DACA eligible: {df_mex['daca_eligible'].sum():,}")

# -----------------------------------------------------------------------------
# 4. DEFINE TREATMENT AND POST PERIODS
# -----------------------------------------------------------------------------
print("\n[4] Defining treatment and post periods...")

# Post-DACA period (2013-2016, excluding 2012 which is transition year)
df_mex['post'] = (df_mex['YEAR'] >= 2013).astype('int8')

# Treatment indicator (DACA eligible x Post)
df_mex['treat_x_post'] = (df_mex['daca_eligible'] * df_mex['post']).astype('int8')

# Year 2012 is ambiguous - DACA implemented June 15, 2012
# We'll exclude 2012 for cleaner identification
df_analysis = df_mex[df_mex['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_analysis):,}")

# -----------------------------------------------------------------------------
# 5. CREATE ANALYSIS SAMPLE
# -----------------------------------------------------------------------------
print("\n[5] Creating analysis sample with treatment and control groups...")

# Restrict to non-citizens only for the DiD
df_noncit = df_analysis[df_analysis['non_citizen'] == 1].copy()
print(f"   Non-citizens only: {len(df_noncit):,}")

# Restrict to working-age population (16-45)
df_noncit = df_noncit[(df_noncit['AGE'] >= 16) & (df_noncit['AGE'] <= 45)].copy()
print(f"   Working-age (16-45): {len(df_noncit):,}")

# Summary of treatment vs control
print(f"\n   Treatment (DACA eligible): {df_noncit['daca_eligible'].sum():,}")
print(f"   Control (non-eligible): {(df_noncit['daca_eligible'] == 0).sum():,}")

# Create additional control variables
df_noncit['female'] = (df_noncit['SEX'] == 2).astype('int8')
df_noncit['married'] = (df_noncit['MARST'].isin([1, 2])).astype('int8')
df_noncit['age_sq'] = (df_noncit['AGE'].astype('float32') ** 2)

# -----------------------------------------------------------------------------
# 6. DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------------
print("\n[6] Descriptive Statistics...")

# By eligibility status
desc_stats = df_noncit.groupby('daca_eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'UHRSWORK': 'mean',
    'EDUC': 'mean',
    'PERWT': 'sum'
}).round(3)
desc_stats.columns = ['Mean Age', 'Prop Female', 'Employed', 'Fulltime', 'Avg Hours', 'Education', 'Weighted N']
print("\n   Summary by DACA Eligibility:")
print(desc_stats.to_string())

# By year and eligibility
yearly_ft = df_noncit.groupby(['YEAR', 'daca_eligible']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'PERWT': 'sum'
}).round(4)
print("\n   Full-time employment rate by year and eligibility:")
print(yearly_ft.to_string())

# Save descriptive stats for report
yearly_ft.to_csv('yearly_trends.csv')

# -----------------------------------------------------------------------------
# 7. MAIN REGRESSION ANALYSIS (DiD)
# -----------------------------------------------------------------------------
print("\n[7] Main Regression Analysis (Difference-in-Differences)...")

# Model 1: Basic DiD without controls
print("\n   Model 1: Basic DiD (no controls)")
model1 = smf.ols('fulltime ~ daca_eligible + post + treat_x_post', data=df_noncit).fit()
print(f"   treat_x_post coefficient: {model1.params['treat_x_post']:.4f}")
print(f"   Standard error: {model1.bse['treat_x_post']:.4f}")
print(f"   t-statistic: {model1.tvalues['treat_x_post']:.4f}")
print(f"   p-value: {model1.pvalues['treat_x_post']:.4f}")
print(f"   N: {int(model1.nobs):,}")
print(f"   R-squared: {model1.rsquared:.4f}")

# Model 2: DiD with demographic controls
print("\n   Model 2: DiD with demographic controls")
model2 = smf.ols('fulltime ~ daca_eligible + post + treat_x_post + AGE + age_sq + female + married + C(EDUC)',
                 data=df_noncit).fit()
print(f"   treat_x_post coefficient: {model2.params['treat_x_post']:.4f}")
print(f"   Standard error: {model2.bse['treat_x_post']:.4f}")
print(f"   t-statistic: {model2.tvalues['treat_x_post']:.4f}")
print(f"   p-value: {model2.pvalues['treat_x_post']:.4f}")
print(f"   N: {int(model2.nobs):,}")
print(f"   R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with state and year fixed effects
print("\n   Model 3: DiD with state and year fixed effects")
model3 = smf.ols('fulltime ~ daca_eligible + treat_x_post + AGE + age_sq + female + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                 data=df_noncit).fit()
print(f"   treat_x_post coefficient: {model3.params['treat_x_post']:.4f}")
print(f"   Standard error: {model3.bse['treat_x_post']:.4f}")
print(f"   t-statistic: {model3.tvalues['treat_x_post']:.4f}")
print(f"   p-value: {model3.pvalues['treat_x_post']:.4f}")
print(f"   N: {int(model3.nobs):,}")
print(f"   R-squared: {model3.rsquared:.4f}")

# Model 4: Robust standard errors clustered by state
print("\n   Model 4: DiD with clustered standard errors (by state)")
model4 = smf.ols('fulltime ~ daca_eligible + treat_x_post + AGE + age_sq + female + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                 data=df_noncit).fit(cov_type='cluster', cov_kwds={'groups': df_noncit['STATEFIP']})
print(f"   treat_x_post coefficient: {model4.params['treat_x_post']:.4f}")
print(f"   Standard error (clustered): {model4.bse['treat_x_post']:.4f}")
print(f"   t-statistic: {model4.tvalues['treat_x_post']:.4f}")
print(f"   p-value: {model4.pvalues['treat_x_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treat_x_post', 0]:.4f}, {model4.conf_int().loc['treat_x_post', 1]:.4f}]")

# -----------------------------------------------------------------------------
# 8. ROBUSTNESS CHECKS
# -----------------------------------------------------------------------------
print("\n[8] Robustness Checks...")

# Robustness 1: Employment as outcome
print("\n   Robustness 1: Employment (any) as outcome")
model_emp = smf.ols('employed ~ daca_eligible + treat_x_post + AGE + age_sq + female + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                    data=df_noncit).fit(cov_type='cluster', cov_kwds={'groups': df_noncit['STATEFIP']})
print(f"   treat_x_post coefficient: {model_emp.params['treat_x_post']:.4f}")
print(f"   Standard error (clustered): {model_emp.bse['treat_x_post']:.4f}")
print(f"   p-value: {model_emp.pvalues['treat_x_post']:.4f}")

# Robustness 2: Restrict to ages 18-35
print("\n   Robustness 2: Restricted age range (18-35)")
df_restricted = df_noncit[(df_noncit['AGE'] >= 18) & (df_noncit['AGE'] <= 35)].copy()
model_age = smf.ols('fulltime ~ daca_eligible + treat_x_post + AGE + age_sq + female + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                    data=df_restricted).fit(cov_type='cluster', cov_kwds={'groups': df_restricted['STATEFIP']})
print(f"   treat_x_post coefficient: {model_age.params['treat_x_post']:.4f}")
print(f"   Standard error (clustered): {model_age.bse['treat_x_post']:.4f}")
print(f"   p-value: {model_age.pvalues['treat_x_post']:.4f}")
print(f"   N: {int(model_age.nobs):,}")

# Robustness 3: Males only
print("\n   Robustness 3: Males only")
df_males = df_noncit[df_noncit['SEX'] == 1].copy()
model_males = smf.ols('fulltime ~ daca_eligible + treat_x_post + AGE + age_sq + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                      data=df_males).fit(cov_type='cluster', cov_kwds={'groups': df_males['STATEFIP']})
print(f"   treat_x_post coefficient: {model_males.params['treat_x_post']:.4f}")
print(f"   Standard error (clustered): {model_males.bse['treat_x_post']:.4f}")
print(f"   p-value: {model_males.pvalues['treat_x_post']:.4f}")

# Robustness 4: Females only
print("\n   Robustness 4: Females only")
df_females = df_noncit[df_noncit['SEX'] == 2].copy()
model_females = smf.ols('fulltime ~ daca_eligible + treat_x_post + AGE + age_sq + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                        data=df_females).fit(cov_type='cluster', cov_kwds={'groups': df_females['STATEFIP']})
print(f"   treat_x_post coefficient: {model_females.params['treat_x_post']:.4f}")
print(f"   Standard error (clustered): {model_females.bse['treat_x_post']:.4f}")
print(f"   p-value: {model_females.pvalues['treat_x_post']:.4f}")

# Robustness 5: Include 2012 as pre-period
print("\n   Robustness 5: Include 2012 as pre-period")
df_with2012 = df_mex[df_mex['YEAR'] <= 2016].copy()
df_with2012 = df_with2012[(df_with2012['non_citizen'] == 1) &
                          (df_with2012['AGE'] >= 16) & (df_with2012['AGE'] <= 45)].copy()
df_with2012['post'] = (df_with2012['YEAR'] >= 2013).astype('int8')
df_with2012['treat_x_post'] = (df_with2012['daca_eligible'] * df_with2012['post']).astype('int8')
df_with2012['female'] = (df_with2012['SEX'] == 2).astype('int8')
df_with2012['married'] = (df_with2012['MARST'].isin([1, 2])).astype('int8')
df_with2012['age_sq'] = df_with2012['AGE'].astype('float32') ** 2

model_2012 = smf.ols('fulltime ~ daca_eligible + treat_x_post + AGE + age_sq + female + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                     data=df_with2012).fit(cov_type='cluster', cov_kwds={'groups': df_with2012['STATEFIP']})
print(f"   treat_x_post coefficient: {model_2012.params['treat_x_post']:.4f}")
print(f"   Standard error (clustered): {model_2012.bse['treat_x_post']:.4f}")
print(f"   p-value: {model_2012.pvalues['treat_x_post']:.4f}")

# -----------------------------------------------------------------------------
# 9. EVENT STUDY SPECIFICATION
# -----------------------------------------------------------------------------
print("\n[9] Event Study Specification...")

# Create year dummies interacted with treatment
years = sorted(df_noncit['YEAR'].unique())
base_year = 2011  # Last pre-treatment year

for year in years:
    df_noncit[f'year_{year}'] = (df_noncit['YEAR'] == year).astype('int8')
    df_noncit[f'treat_x_{year}'] = (df_noncit['daca_eligible'] * df_noncit[f'year_{year}']).astype('int8')

# Event study formula (omit 2011 as base year)
event_vars = [f'treat_x_{y}' for y in years if y != 2011]
formula_event = 'fulltime ~ daca_eligible + ' + ' + '.join(event_vars) + ' + AGE + age_sq + female + married + C(EDUC) + C(YEAR) + C(STATEFIP)'

model_event = smf.ols(formula_event, data=df_noncit).fit(cov_type='cluster', cov_kwds={'groups': df_noncit['STATEFIP']})

print("\n   Event Study Coefficients (Treatment x Year, base=2011):")
for year in years:
    if year != 2011:
        var = f'treat_x_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        pval = model_event.pvalues[var]
        print(f"   {year}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")

# -----------------------------------------------------------------------------
# 10. WEIGHTED ANALYSIS
# -----------------------------------------------------------------------------
print("\n[10] Weighted Analysis (using PERWT)...")

# Prepare data for weighted regression
df_wt = df_noncit.copy()
df_wt = df_wt.dropna(subset=['fulltime', 'daca_eligible', 'treat_x_post', 'AGE', 'female', 'married', 'EDUC', 'YEAR', 'STATEFIP', 'PERWT'])

# Use WLS with person weights
from patsy import dmatrices
y, X = dmatrices('fulltime ~ daca_eligible + treat_x_post + AGE + age_sq + female + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                  data=df_wt, return_type='dataframe')

wls_model = sm.WLS(y, X, weights=df_wt['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_wt['STATEFIP']})
print(f"   treat_x_post coefficient (weighted): {wls_model.params['treat_x_post']:.4f}")
print(f"   Standard error (clustered): {wls_model.bse['treat_x_post']:.4f}")
print(f"   p-value: {wls_model.pvalues['treat_x_post']:.4f}")
print(f"   95% CI: [{wls_model.conf_int().loc['treat_x_post', 0]:.4f}, {wls_model.conf_int().loc['treat_x_post', 1]:.4f}]")

# -----------------------------------------------------------------------------
# 11. SAVE RESULTS FOR REPORT
# -----------------------------------------------------------------------------
print("\n[11] Saving results...")

# Create results dictionary
results = {
    'preferred_estimate': {
        'coefficient': model4.params['treat_x_post'],
        'se': model4.bse['treat_x_post'],
        'pvalue': model4.pvalues['treat_x_post'],
        'ci_lower': model4.conf_int().loc['treat_x_post', 0],
        'ci_upper': model4.conf_int().loc['treat_x_post', 1],
        'n_obs': int(model4.nobs),
        'r_squared': model4.rsquared
    },
    'weighted_estimate': {
        'coefficient': wls_model.params['treat_x_post'],
        'se': wls_model.bse['treat_x_post'],
        'pvalue': wls_model.pvalues['treat_x_post'],
        'ci_lower': wls_model.conf_int().loc['treat_x_post', 0],
        'ci_upper': wls_model.conf_int().loc['treat_x_post', 1]
    }
}

# Print summary
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)
print(f"\nPreferred Estimate (Model 4 - DiD with controls, state/year FE, clustered SE):")
print(f"   Effect of DACA eligibility on full-time employment: {results['preferred_estimate']['coefficient']:.4f}")
print(f"   Standard Error: {results['preferred_estimate']['se']:.4f}")
print(f"   95% CI: [{results['preferred_estimate']['ci_lower']:.4f}, {results['preferred_estimate']['ci_upper']:.4f}]")
print(f"   p-value: {results['preferred_estimate']['pvalue']:.4f}")
print(f"   Sample size: {results['preferred_estimate']['n_obs']:,}")

print(f"\nWeighted Estimate (using PERWT):")
print(f"   Effect: {results['weighted_estimate']['coefficient']:.4f}")
print(f"   Standard Error: {results['weighted_estimate']['se']:.4f}")
print(f"   95% CI: [{results['weighted_estimate']['ci_lower']:.4f}, {results['weighted_estimate']['ci_upper']:.4f}]")

# -----------------------------------------------------------------------------
# 12. CREATE DATA FOR FIGURES
# -----------------------------------------------------------------------------
print("\n[12] Creating data for figures...")

# Figure 1: Full-time employment trends by eligibility
trends = df_noncit.groupby(['YEAR', 'daca_eligible']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'PERWT': 'sum'
}).reset_index()
trends.to_csv('figure_data_trends.csv', index=False)
print("   Saved trend data to figure_data_trends.csv")

# Figure 2: Event study coefficients
event_coefs = []
for year in years:
    if year != 2011:
        var = f'treat_x_{year}'
        event_coefs.append({
            'year': year,
            'coefficient': model_event.params[var],
            'se': model_event.bse[var],
            'ci_lower': model_event.conf_int().loc[var, 0],
            'ci_upper': model_event.conf_int().loc[var, 1]
        })
    else:
        event_coefs.append({
            'year': year,
            'coefficient': 0,
            'se': 0,
            'ci_lower': 0,
            'ci_upper': 0
        })
event_df = pd.DataFrame(event_coefs)
event_df.to_csv('figure_data_event_study.csv', index=False)
print("   Saved event study data to figure_data_event_study.csv")

# Summary statistics table
print("\n[13] Summary Statistics Table...")
summary_pre = df_noncit[df_noncit['post'] == 0].groupby('daca_eligible').agg({
    'AGE': ['mean', 'std'],
    'female': 'mean',
    'married': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'UHRSWORK': 'mean'
}).round(3)

summary_post = df_noncit[df_noncit['post'] == 1].groupby('daca_eligible').agg({
    'AGE': ['mean', 'std'],
    'female': 'mean',
    'married': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'UHRSWORK': 'mean'
}).round(3)

print("\nPre-DACA period (2006-2011):")
print(summary_pre)
print("\nPost-DACA period (2013-2016):")
print(summary_post)

# Save summary stats
summary_all = df_noncit.groupby(['post', 'daca_eligible']).agg({
    'AGE': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'UHRSWORK': 'mean',
    'PERWT': 'sum'
}).round(4)
summary_all.to_csv('summary_statistics.csv')

# Save full model summaries for the report
with open('model_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("DACA REPLICATION STUDY - FULL MODEL RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("MODEL 1: Basic DiD (no controls)\n")
    f.write("-"*40 + "\n")
    f.write(f"treat_x_post: {model1.params['treat_x_post']:.6f} (SE: {model1.bse['treat_x_post']:.6f})\n")
    f.write(f"N: {int(model1.nobs)}, R2: {model1.rsquared:.4f}\n\n")

    f.write("MODEL 2: DiD with demographic controls\n")
    f.write("-"*40 + "\n")
    f.write(f"treat_x_post: {model2.params['treat_x_post']:.6f} (SE: {model2.bse['treat_x_post']:.6f})\n")
    f.write(f"N: {int(model2.nobs)}, R2: {model2.rsquared:.4f}\n\n")

    f.write("MODEL 3: DiD with state and year FE\n")
    f.write("-"*40 + "\n")
    f.write(f"treat_x_post: {model3.params['treat_x_post']:.6f} (SE: {model3.bse['treat_x_post']:.6f})\n")
    f.write(f"N: {int(model3.nobs)}, R2: {model3.rsquared:.4f}\n\n")

    f.write("MODEL 4: DiD with clustered SE (PREFERRED)\n")
    f.write("-"*40 + "\n")
    f.write(f"treat_x_post: {model4.params['treat_x_post']:.6f} (SE: {model4.bse['treat_x_post']:.6f})\n")
    f.write(f"95% CI: [{model4.conf_int().loc['treat_x_post', 0]:.6f}, {model4.conf_int().loc['treat_x_post', 1]:.6f}]\n")
    f.write(f"N: {int(model4.nobs)}, R2: {model4.rsquared:.4f}\n\n")

    f.write("WEIGHTED MODEL (PERWT):\n")
    f.write("-"*40 + "\n")
    f.write(f"treat_x_post: {wls_model.params['treat_x_post']:.6f} (SE: {wls_model.bse['treat_x_post']:.6f})\n")
    f.write(f"95% CI: [{wls_model.conf_int().loc['treat_x_post', 0]:.6f}, {wls_model.conf_int().loc['treat_x_post', 1]:.6f}]\n\n")

    f.write("\nROBUSTNESS CHECKS:\n")
    f.write("-"*40 + "\n")
    f.write(f"Employment outcome: {model_emp.params['treat_x_post']:.6f} (SE: {model_emp.bse['treat_x_post']:.6f})\n")
    f.write(f"Ages 18-35: {model_age.params['treat_x_post']:.6f} (SE: {model_age.bse['treat_x_post']:.6f})\n")
    f.write(f"Males only: {model_males.params['treat_x_post']:.6f} (SE: {model_males.bse['treat_x_post']:.6f})\n")
    f.write(f"Females only: {model_females.params['treat_x_post']:.6f} (SE: {model_females.bse['treat_x_post']:.6f})\n")
    f.write(f"Include 2012: {model_2012.params['treat_x_post']:.6f} (SE: {model_2012.bse['treat_x_post']:.6f})\n\n")

    f.write("\nEVENT STUDY COEFFICIENTS:\n")
    f.write("-"*40 + "\n")
    for year in years:
        if year != 2011:
            var = f'treat_x_{year}'
            f.write(f"{year}: {model_event.params[var]:.6f} (SE: {model_event.bse[var]:.6f})\n")
        else:
            f.write(f"{year}: 0.000000 (base year)\n")

print("   Saved full model results to model_results.txt")

# Save results to JSON for later use
import json
with open('results.json', 'w') as f:
    json.dump({
        'preferred': {
            'coef': float(model4.params['treat_x_post']),
            'se': float(model4.bse['treat_x_post']),
            'pval': float(model4.pvalues['treat_x_post']),
            'ci_low': float(model4.conf_int().loc['treat_x_post', 0]),
            'ci_high': float(model4.conf_int().loc['treat_x_post', 1]),
            'n': int(model4.nobs),
            'r2': float(model4.rsquared)
        },
        'weighted': {
            'coef': float(wls_model.params['treat_x_post']),
            'se': float(wls_model.bse['treat_x_post']),
            'pval': float(wls_model.pvalues['treat_x_post']),
            'ci_low': float(wls_model.conf_int().loc['treat_x_post', 0]),
            'ci_high': float(wls_model.conf_int().loc['treat_x_post', 1])
        },
        'robustness': {
            'employment': float(model_emp.params['treat_x_post']),
            'age_restricted': float(model_age.params['treat_x_post']),
            'males': float(model_males.params['treat_x_post']),
            'females': float(model_females.params['treat_x_post']),
            'with_2012': float(model_2012.params['treat_x_post'])
        }
    }, f, indent=2)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
