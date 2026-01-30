"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the United States.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n[Step 1] Loading data...")

# Define data types to reduce memory usage
dtypes = {
    'YEAR': 'int16',
    'SAMPLE': 'int32',
    'SERIAL': 'int32',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int8',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'EMPSTAT': 'int8',
    'EMPSTATD': 'int8',
    'LABFORCE': 'int8',
    'UHRSWORK': 'int8',
    'MARST': 'int8',
    'STATEFIP': 'int8',
    'METRO': 'int8',
}

# Load only columns we need
cols_to_use = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EDUCD',
               'EMPSTAT', 'LABFORCE', 'UHRSWORK', 'MARST', 'STATEFIP', 'METRO']

df = pd.read_csv('data/data.csv', usecols=cols_to_use, dtype=dtypes)
print(f"   Total observations loaded: {len(df):,}")

# =============================================================================
# STEP 2: Sample Selection
# =============================================================================
print("\n[Step 2] Applying sample selection criteria...")

# Step 2.1: Hispanic-Mexican ethnicity
df = df[df['HISPAN'] == 1]
print(f"   After Hispanic-Mexican filter: {len(df):,}")

# Step 2.2: Born in Mexico
df = df[df['BPL'] == 200]
print(f"   After Mexico birthplace filter: {len(df):,}")

# Step 2.3: Not a citizen (proxy for undocumented)
# CITIZEN = 3 means "Not a citizen"
df = df[df['CITIZEN'] == 3]
print(f"   After non-citizen filter: {len(df):,}")

# Step 2.4: Immigration year available and <= 2007 (continuous residence since 2007)
df = df[(df['YRIMMIG'] > 0) & (df['YRIMMIG'] <= 2007)]
print(f"   After immigration year filter (<=2007): {len(df):,}")

# Step 2.5: Calculate age at arrival and filter for arrival before age 16
# Age at arrival = Year of survey - Birth year - Years since immigration
# Or more simply: Age at immigration = YRIMMIG - BIRTHYR
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_immigration'] < 16]
print(f"   After arrival before age 16 filter: {len(df):,}")

# =============================================================================
# STEP 3: Define Treatment and Control Groups
# =============================================================================
print("\n[Step 3] Defining treatment and control groups...")

# DACA implementation date: June 15, 2012
# Treatment group: Ages 26-30 on June 15, 2012
# Control group: Ages 31-35 on June 15, 2012

# Calculate approximate age on June 15, 2012
# Birth quarter: 1 = Jan-Mar, 2 = Apr-Jun, 3 = Jul-Sep, 4 = Oct-Dec
# For June 15, those born in Q1-Q2 of year Y would be (2012 - Y) years old
# Those born in Q3-Q4 of year Y would be (2012 - Y - 1) years old (haven't had birthday yet)

def calc_age_on_june_2012(birthyr, birthqtr):
    """Calculate age on June 15, 2012 based on birth year and quarter"""
    # If born in first half of year (Q1-Q2), already had birthday by June 15
    # If born in second half (Q3-Q4), haven't had birthday yet
    if birthqtr <= 2:
        return 2012 - birthyr
    else:
        return 2012 - birthyr - 1

df['age_june_2012'] = df.apply(lambda x: calc_age_on_june_2012(x['BIRTHYR'], x['BIRTHQTR']), axis=1)

# Treatment group: ages 26-30 on June 15, 2012
# Control group: ages 31-35 on June 15, 2012
df['treat'] = ((df['age_june_2012'] >= 26) & (df['age_june_2012'] <= 30)).astype(int)
df['control'] = ((df['age_june_2012'] >= 31) & (df['age_june_2012'] <= 35)).astype(int)

# Keep only treatment and control observations
df = df[(df['treat'] == 1) | (df['control'] == 1)]
print(f"   After treatment/control group filter: {len(df):,}")

# =============================================================================
# STEP 4: Define Time Periods
# =============================================================================
print("\n[Step 4] Defining time periods...")

# Exclude 2012 (implementation year with ambiguity)
df = df[df['YEAR'] != 2012]
print(f"   After excluding 2012: {len(df):,}")

# Post period: 2013-2016 (as specified in instructions)
df['post'] = (df['YEAR'] >= 2013).astype(int)

print(f"\n   Pre-period years (2006-2011): {df[df['post']==0]['YEAR'].unique()}")
print(f"   Post-period years (2013-2016): {df[df['post']==1]['YEAR'].unique()}")

# =============================================================================
# STEP 5: Define Outcome Variable
# =============================================================================
print("\n[Step 5] Defining outcome variable...")

# Full-time employment: UHRSWORK >= 35 AND EMPSTAT == 1 (employed)
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)

print(f"   Full-time employment rate: {df['fulltime'].mean()*100:.2f}%")

# =============================================================================
# STEP 6: Create Covariates
# =============================================================================
print("\n[Step 6] Creating covariates...")

# Sex (female indicator)
df['female'] = (df['SEX'] == 2).astype(int)

# Education categories
# EDUC: 0=N/A, 1=None/preschool, 2=Grade 1-4, 3=Grade 5-8, 4=Grade 9,
#       5=Grade 10, 6=Grade 11, 7=Grade 12/HS diploma, 8=1 yr college,
#       9=2 yrs college, 10=3 yrs college, 11=4+ yrs college
df['educ_lesshs'] = (df['EDUC'] < 6).astype(int)
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_somecoll'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)
df['educ_coll'] = (df['EDUC'] >= 10).astype(int)

# Marital status
df['married'] = (df['MARST'] <= 2).astype(int)  # 1=married spouse present, 2=married spouse absent

# Age (current age at time of survey as control)
df['age'] = df['AGE']

# Metropolitan area
df['metro'] = (df['METRO'] >= 2).astype(int)  # In metro area

# =============================================================================
# STEP 7: Summary Statistics
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

# Sample sizes by group and period
print("\n[Table 1] Sample Size by Group and Period")
print("-" * 50)
sample_counts = df.groupby(['treat', 'post']).apply(lambda x: pd.Series({
    'N': len(x),
    'Weighted_N': x['PERWT'].sum()
})).round(0)
print(sample_counts)

# Pre-treatment characteristics
print("\n[Table 2] Pre-Treatment Characteristics by Group")
print("-" * 60)

pre_df = df[df['post'] == 0]

def weighted_mean(data, values_col, weights_col):
    return np.average(data[values_col], weights=data[weights_col])

def weighted_std(data, values_col, weights_col):
    average = np.average(data[values_col], weights=data[weights_col])
    variance = np.average((data[values_col] - average)**2, weights=data[weights_col])
    return np.sqrt(variance)

vars_to_summarize = ['fulltime', 'female', 'married', 'age', 'educ_lesshs', 'educ_hs', 'educ_somecoll', 'educ_coll', 'metro']
var_labels = ['Full-time employed', 'Female', 'Married', 'Age', 'Less than HS', 'High School', 'Some College', 'College+', 'Metro Area']

print(f"{'Variable':<20} {'Treatment':>15} {'Control':>15} {'Diff':>10}")
print("-" * 60)

for var, label in zip(vars_to_summarize, var_labels):
    treat_mean = weighted_mean(pre_df[pre_df['treat']==1], var, 'PERWT')
    control_mean = weighted_mean(pre_df[pre_df['control']==1], var, 'PERWT')
    diff = treat_mean - control_mean
    print(f"{label:<20} {treat_mean:>15.3f} {control_mean:>15.3f} {diff:>10.3f}")

# =============================================================================
# STEP 8: Difference-in-Differences Analysis
# =============================================================================
print("\n" + "=" * 70)
print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 70)

# DiD interaction term
df['treat_post'] = df['treat'] * df['post']

# Model 1: Basic DiD (no covariates)
print("\n[Model 1] Basic Difference-in-Differences")
print("-" * 60)

# Using weighted least squares
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

print(f"DiD Estimate (treat_post): {model1.params['treat_post']:.4f}")
print(f"Standard Error (robust): {model1.bse['treat_post']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")
print(f"t-statistic: {model1.tvalues['treat_post']:.3f}")
print(f"p-value: {model1.pvalues['treat_post']:.4f}")
print(f"\nSample size: {int(model1.nobs):,}")

# Model 2: DiD with demographic covariates
print("\n[Model 2] DiD with Demographic Covariates")
print("-" * 60)

model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + age + I(age**2)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

print(f"DiD Estimate (treat_post): {model2.params['treat_post']:.4f}")
print(f"Standard Error (robust): {model2.bse['treat_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"t-statistic: {model2.tvalues['treat_post']:.3f}")
print(f"p-value: {model2.pvalues['treat_post']:.4f}")

# Model 3: DiD with full covariates
print("\n[Model 3] DiD with Full Covariates")
print("-" * 60)

model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + age + I(age**2) + educ_hs + educ_somecoll + educ_coll + metro',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

print(f"DiD Estimate (treat_post): {model3.params['treat_post']:.4f}")
print(f"Standard Error (robust): {model3.bse['treat_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"t-statistic: {model3.tvalues['treat_post']:.3f}")
print(f"p-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: With year fixed effects
print("\n[Model 4] DiD with Year Fixed Effects")
print("-" * 60)

df['year_factor'] = pd.Categorical(df['YEAR'])
model4 = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + female + married + age + I(age**2) + educ_hs + educ_somecoll + educ_coll + metro',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

print(f"DiD Estimate (treat_post): {model4.params['treat_post']:.4f}")
print(f"Standard Error (robust): {model4.bse['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"t-statistic: {model4.tvalues['treat_post']:.3f}")
print(f"p-value: {model4.pvalues['treat_post']:.4f}")

# Model 5: With state fixed effects
print("\n[Model 5] DiD with State and Year Fixed Effects (Preferred)")
print("-" * 60)

model5 = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + female + married + age + I(age**2) + educ_hs + educ_somecoll + educ_coll + metro',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

print(f"DiD Estimate (treat_post): {model5.params['treat_post']:.4f}")
print(f"Standard Error (robust): {model5.bse['treat_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"t-statistic: {model5.tvalues['treat_post']:.3f}")
print(f"p-value: {model5.pvalues['treat_post']:.4f}")

# =============================================================================
# STEP 9: Event Study Analysis (Check Pre-trends)
# =============================================================================
print("\n" + "=" * 70)
print("EVENT STUDY ANALYSIS (Pre-trends Check)")
print("=" * 70)

# Create year-specific treatment effects
years = sorted(df['YEAR'].unique())
ref_year = 2011  # Use 2011 as reference year (last pre-treatment year)

for year in years:
    df[f'treat_year_{year}'] = (df['treat'] * (df['YEAR'] == year)).astype(int)

# Remove reference year interaction
year_vars = [f'treat_year_{y}' for y in years if y != ref_year]
formula_event = 'fulltime ~ treat + C(YEAR) + ' + ' + '.join(year_vars) + ' + female + married + age + I(age**2) + educ_hs + educ_somecoll + educ_coll + metro'

model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print(f"\n{'Year':<10} {'Coefficient':>12} {'SE':>10} {'95% CI':>25}")
print("-" * 60)
for year in years:
    if year == ref_year:
        print(f"{year:<10} {'0.0000 (ref)':>12}")
    else:
        var = f'treat_year_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        ci_low, ci_high = model_event.conf_int().loc[var]
        print(f"{year:<10} {coef:>12.4f} {se:>10.4f} [{ci_low:>9.4f}, {ci_high:>8.4f}]")

# =============================================================================
# STEP 10: Subgroup Analysis
# =============================================================================
print("\n" + "=" * 70)
print("SUBGROUP ANALYSIS")
print("=" * 70)

# By sex
print("\n[Subgroup] By Sex")
print("-" * 40)

for sex, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex]
    model_sub = smf.wls('fulltime ~ treat + post + treat_post + married + age + I(age**2) + educ_hs + educ_somecoll + educ_coll + metro',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"{sex_label}: DiD = {model_sub.params['treat_post']:.4f} (SE = {model_sub.bse['treat_post']:.4f}), N = {int(model_sub.nobs):,}")

# =============================================================================
# STEP 11: Simple 2x2 DiD Table
# =============================================================================
print("\n" + "=" * 70)
print("2x2 DIFFERENCE-IN-DIFFERENCES TABLE")
print("=" * 70)

# Calculate weighted means for each cell
def weighted_group_mean(data, group_var, time_var, outcome_var, weight_var):
    result = data.groupby([group_var, time_var]).apply(
        lambda x: np.average(x[outcome_var], weights=x[weight_var])
    ).unstack()
    return result

dd_table = weighted_group_mean(df, 'treat', 'post', 'fulltime', 'PERWT')
dd_table.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
dd_table.index = ['Control (31-35)', 'Treatment (26-30)']

print("\n                        Pre (2006-2011)    Post (2013-2016)    Difference")
print("-" * 75)
for idx in dd_table.index:
    pre_val = dd_table.loc[idx, 'Pre (2006-2011)']
    post_val = dd_table.loc[idx, 'Post (2013-2016)']
    diff = post_val - pre_val
    print(f"{idx:<25} {pre_val:>14.4f} {post_val:>18.4f} {diff:>14.4f}")

# Calculate DiD
treat_diff = dd_table.loc['Treatment (26-30)', 'Post (2013-2016)'] - dd_table.loc['Treatment (26-30)', 'Pre (2006-2011)']
control_diff = dd_table.loc['Control (31-35)', 'Post (2013-2016)'] - dd_table.loc['Control (31-35)', 'Pre (2006-2011)']
did_estimate = treat_diff - control_diff

print("-" * 75)
print(f"{'Diff-in-Diff':<25} {'':<14} {'':<18} {did_estimate:>14.4f}")

# =============================================================================
# STEP 12: Save Results
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"""
PREFERRED ESTIMATE (Model 5 with State and Year FE):
----------------------------------------------------
Effect of DACA eligibility on full-time employment: {model5.params['treat_post']:.4f}
Standard Error (robust):                             {model5.bse['treat_post']:.4f}
95% Confidence Interval:                            [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]
t-statistic:                                         {model5.tvalues['treat_post']:.3f}
p-value:                                             {model5.pvalues['treat_post']:.4f}
Sample Size:                                         {int(model5.nobs):,}

INTERPRETATION:
DACA eligibility is associated with a {abs(model5.params['treat_post'])*100:.2f} percentage point
{'increase' if model5.params['treat_post'] > 0 else 'decrease'} in the probability of full-time employment
among Hispanic-Mexican Mexican-born non-citizens who arrived in the US
before age 16.
""")

# Save key results to a file
results_dict = {
    'Model': ['Basic DiD', 'With Demographics', 'With Full Covariates', 'With Year FE', 'With State+Year FE (Preferred)'],
    'DiD_Estimate': [model1.params['treat_post'], model2.params['treat_post'], model3.params['treat_post'], model4.params['treat_post'], model5.params['treat_post']],
    'SE': [model1.bse['treat_post'], model2.bse['treat_post'], model3.bse['treat_post'], model4.bse['treat_post'], model5.bse['treat_post']],
    'CI_Lower': [model1.conf_int().loc['treat_post', 0], model2.conf_int().loc['treat_post', 0], model3.conf_int().loc['treat_post', 0], model4.conf_int().loc['treat_post', 0], model5.conf_int().loc['treat_post', 0]],
    'CI_Upper': [model1.conf_int().loc['treat_post', 1], model2.conf_int().loc['treat_post', 1], model3.conf_int().loc['treat_post', 1], model4.conf_int().loc['treat_post', 1], model5.conf_int().loc['treat_post', 1]],
    'p_value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'], model3.pvalues['treat_post'], model4.pvalues['treat_post'], model5.pvalues['treat_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('results_summary.csv', index=False)
print("\nResults saved to results_summary.csv")

# Save event study coefficients
event_results = []
for year in years:
    if year == ref_year:
        event_results.append({'Year': year, 'Coefficient': 0.0, 'SE': 0.0, 'CI_Lower': 0.0, 'CI_Upper': 0.0})
    else:
        var = f'treat_year_{year}'
        event_results.append({
            'Year': year,
            'Coefficient': model_event.params[var],
            'SE': model_event.bse[var],
            'CI_Lower': model_event.conf_int().loc[var, 0],
            'CI_Upper': model_event.conf_int().loc[var, 1]
        })

event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save summary statistics
summary_stats = {
    'Group': ['Treatment (26-30)', 'Control (31-35)'],
    'Pre_FT_Rate': [
        weighted_mean(df[(df['treat']==1) & (df['post']==0)], 'fulltime', 'PERWT'),
        weighted_mean(df[(df['control']==1) & (df['post']==0)], 'fulltime', 'PERWT')
    ],
    'Post_FT_Rate': [
        weighted_mean(df[(df['treat']==1) & (df['post']==1)], 'fulltime', 'PERWT'),
        weighted_mean(df[(df['control']==1) & (df['post']==1)], 'fulltime', 'PERWT')
    ],
    'Pre_N': [
        len(df[(df['treat']==1) & (df['post']==0)]),
        len(df[(df['control']==1) & (df['post']==0)])
    ],
    'Post_N': [
        len(df[(df['treat']==1) & (df['post']==1)]),
        len(df[(df['control']==1) & (df['post']==1)])
    ]
}
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_statistics.csv', index=False)
print("Summary statistics saved to summary_statistics.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
