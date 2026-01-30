"""
DACA Replication Analysis
Difference-in-Differences estimation of DACA effects on full-time employment
among Hispanic-Mexican, Mexican-born non-citizens
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = "data/data.csv"
OUTPUT_DIR = "."

print("="*60)
print("DACA REPLICATION ANALYSIS")
print("="*60)

# Define key parameters based on research design
# DACA was implemented June 15, 2012
# Treatment group: ages 26-30 at policy time (June 2012)
# Control group: ages 31-35 at policy time (June 2012)
# These would have been eligible except for age cutoff (31st birthday by June 15, 2012)

# Pre-treatment years: 2006-2011
# Post-treatment years: 2013-2016 (instruction says examine 2013-2016)
# 2012 is ambiguous since implementation was mid-year

print("\n1. Loading and filtering data...")
print("   This may take several minutes due to file size...")

# Read data in chunks to handle large file
chunks = []
chunksize = 1000000

# Key variables needed:
# YEAR, HISPAN (1=Mexican), BPL (200=Mexico), CITIZEN (3=Not a citizen)
# YRIMMIG, BIRTHYR, BIRTHQTR, AGE, UHRSWORK, PERWT
# SEX, EDUC, MARST, STATEFIP for controls

cols_needed = ['YEAR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'BIRTHYR', 'BIRTHQTR', 'AGE', 'UHRSWORK', 'PERWT',
               'SEX', 'EDUC', 'MARST', 'STATEFIP', 'EMPSTAT', 'LABFORCE']

for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=chunksize, usecols=cols_needed)):
    # Filter early to reduce memory
    # Hispanic-Mexican: HISPAN == 1
    # Born in Mexico: BPL == 200
    # Not a citizen: CITIZEN == 3
    filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3)
    ]
    if len(filtered) > 0:
        chunks.append(filtered)
    if (i+1) % 10 == 0:
        print(f"   Processed {(i+1)*chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"\n   Filtered sample size: {len(df):,} observations")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# Save initial filtered data info
initial_n = len(df)

print("\n2. Defining treatment and control groups...")

# DACA eligibility criteria:
# 1. Arrived in US before 16th birthday
# 2. Under 31 as of June 15, 2012
# 3. Continuous residence in US since June 15, 2007
# 4. Present in US on June 15, 2012

# For treatment group (ages 26-30 on June 15, 2012):
# Born between June 16, 1981 and June 15, 1986
# For control group (ages 31-35 on June 15, 2012):
# Born between June 16, 1976 and June 15, 1981

# Calculate approximate age on June 15, 2012
# Birth quarter: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

def calc_age_june2012(row):
    """Calculate approximate age as of June 15, 2012"""
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']

    # June 15 falls in Q2
    if birth_qtr <= 2:  # Born Jan-June
        age = 2012 - birth_year
    else:  # Born July-December (birthday hasn't happened yet by June)
        age = 2012 - birth_year - 1
    return age

df['age_june2012'] = df.apply(calc_age_june2012, axis=1)

# Define treatment and control based on age as of June 15, 2012
# Treatment: 26-30 (would have been DACA-eligible based on age)
# Control: 31-35 (too old for DACA)

df['treat'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)
df['control'] = ((df['age_june2012'] >= 31) & (df['age_june2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df = df[(df['treat'] == 1) | (df['control'] == 1)]

print(f"   Sample restricted to ages 26-35: {len(df):,} observations")

print("\n3. Applying additional DACA eligibility criteria...")

# Additional eligibility: arrived before 16th birthday
# This means YRIMMIG <= BIRTHYR + 15

df['arrived_before_16'] = (df['YRIMMIG'] <= df['BIRTHYR'] + 15) & (df['YRIMMIG'] > 0)

# Also need continuous presence since 2007
# We can only proxy this by checking if they immigrated by 2007
df['in_us_by_2007'] = df['YRIMMIG'] <= 2007

# Apply eligibility filters (these apply to both groups for comparability)
df_eligible = df[df['arrived_before_16'] & df['in_us_by_2007']].copy()

print(f"   After applying arrival before 16: {len(df[df['arrived_before_16']]):,}")
print(f"   After also applying in US by 2007: {len(df_eligible):,}")

print("\n4. Creating outcome variable and time periods...")

# Outcome: Full-time employment (35+ hours per week)
df_eligible['fulltime'] = (df_eligible['UHRSWORK'] >= 35).astype(int)

# Define pre and post periods
# Pre: 2006-2011
# Post: 2013-2016
# Exclude 2012 (implementation year - ambiguous)

df_eligible['post'] = (df_eligible['YEAR'] >= 2013).astype(int)
df_eligible['pre'] = (df_eligible['YEAR'] <= 2011).astype(int)

# Exclude 2012
df_analysis = df_eligible[df_eligible['YEAR'] != 2012].copy()

print(f"   Excluding 2012 (ambiguous year): {len(df_analysis):,} observations")
print(f"   Pre-period (2006-2011): {len(df_analysis[df_analysis['pre']==1]):,}")
print(f"   Post-period (2013-2016): {len(df_analysis[df_analysis['post']==1]):,}")

# DiD interaction term
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

print("\n5. Descriptive statistics...")

# Summary statistics by group and period
desc_stats = df_analysis.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'PERWT': 'sum',
    'SEX': lambda x: (x==1).mean(),  # Proportion male
    'UHRSWORK': 'mean'
}).round(4)

print("\n   Summary by Treatment Group and Period:")
print(desc_stats)

# Save detailed descriptive stats
desc_detail = pd.DataFrame()
for (t, p), grp in df_analysis.groupby(['treat', 'post']):
    group_name = f"{'Treat' if t==1 else 'Control'}_{'Post' if p==1 else 'Pre'}"
    desc_detail[group_name] = pd.Series({
        'N (unweighted)': len(grp),
        'N (weighted)': grp['PERWT'].sum(),
        'Full-time rate': grp['fulltime'].mean(),
        'Mean hours': grp['UHRSWORK'].mean(),
        'Mean age (survey)': grp['AGE'].mean(),
        'Prop male': (grp['SEX']==1).mean(),
        'Mean years in US': (grp['YEAR'] - grp['YRIMMIG']).mean()
    })

print("\n   Detailed Summary Statistics:")
print(desc_detail.T.round(4))

print("\n6. Running difference-in-differences regressions...")

# Model 1: Basic DiD (unweighted)
print("\n   Model 1: Basic DiD (unweighted)")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit()
print(f"   DiD estimate (treat_post): {model1.params['treat_post']:.4f}")
print(f"   Standard error: {model1.bse['treat_post']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model1.pvalues['treat_post']:.4f}")
print(f"   N: {int(model1.nobs):,}")

# Model 2: Basic DiD (weighted)
print("\n   Model 2: Basic DiD (weighted)")
model2 = smf.wls('fulltime ~ treat + post + treat_post', data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"   DiD estimate (treat_post): {model2.params['treat_post']:.4f}")
print(f"   Standard error: {model2.bse['treat_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model2.pvalues['treat_post']:.4f}")

# Model 3: DiD with year fixed effects (weighted)
print("\n   Model 3: DiD with year fixed effects (weighted)")
df_analysis['year_factor'] = pd.Categorical(df_analysis['YEAR'])
model3 = smf.wls('fulltime ~ treat + C(YEAR) + treat_post', data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"   DiD estimate (treat_post): {model3.params['treat_post']:.4f}")
print(f"   Standard error: {model3.bse['treat_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: DiD with covariates (weighted)
print("\n   Model 4: DiD with covariates (weighted)")
# Create covariate dummies
df_analysis['male'] = (df_analysis['SEX'] == 1).astype(int)
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)  # 1=married spouse present, 2=married spouse absent

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # HS or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # 4+ years college

model4 = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + male + married + educ_hs + educ_college',
                  data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"   DiD estimate (treat_post): {model4.params['treat_post']:.4f}")
print(f"   Standard error: {model4.bse['treat_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model4.pvalues['treat_post']:.4f}")

# Model 5: DiD with state fixed effects (weighted)
print("\n   Model 5: DiD with year and state fixed effects (weighted)")
model5 = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + male + married + educ_hs + educ_college',
                  data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"   DiD estimate (treat_post): {model5.params['treat_post']:.4f}")
print(f"   Standard error: {model5.bse['treat_post']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model5.pvalues['treat_post']:.4f}")

# Robust standard errors for preferred model
print("\n   Model 6: Preferred specification with robust (HC1) standard errors")
model6 = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + male + married + educ_hs + educ_college',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"   DiD estimate (treat_post): {model6.params['treat_post']:.4f}")
print(f"   Robust standard error: {model6.bse['treat_post']:.4f}")
print(f"   95% CI: [{model6.conf_int().loc['treat_post', 0]:.4f}, {model6.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model6.pvalues['treat_post']:.4f}")

print("\n7. Creating year-by-year analysis for parallel trends...")

# Year-specific treatment effects (event study style)
year_effects = []
for year in sorted(df_analysis['YEAR'].unique()):
    df_year = df_analysis[df_analysis['YEAR'] == year]

    # Simple mean difference
    treat_mean = df_year[df_year['treat']==1]['fulltime'].mean()
    control_mean = df_year[df_year['treat']==0]['fulltime'].mean()
    diff = treat_mean - control_mean

    # Weighted means
    treat_weighted = np.average(df_year[df_year['treat']==1]['fulltime'],
                                weights=df_year[df_year['treat']==1]['PERWT'])
    control_weighted = np.average(df_year[df_year['treat']==0]['fulltime'],
                                  weights=df_year[df_year['treat']==0]['PERWT'])
    diff_weighted = treat_weighted - control_weighted

    year_effects.append({
        'Year': year,
        'Treat_FT_Rate': treat_mean,
        'Control_FT_Rate': control_mean,
        'Difference': diff,
        'Treat_FT_Weighted': treat_weighted,
        'Control_FT_Weighted': control_weighted,
        'Diff_Weighted': diff_weighted,
        'N_Treat': len(df_year[df_year['treat']==1]),
        'N_Control': len(df_year[df_year['treat']==0])
    })

year_df = pd.DataFrame(year_effects)
print("\n   Year-by-Year Full-Time Employment Rates:")
print(year_df.round(4).to_string(index=False))

# Event study regression
print("\n8. Event study specification...")
df_analysis['year_treat'] = df_analysis['YEAR'].astype(str) + '_' + df_analysis['treat'].astype(str)

# Create year dummies interacted with treatment
for year in sorted(df_analysis['YEAR'].unique()):
    df_analysis[f'treat_y{year}'] = ((df_analysis['YEAR'] == year) & (df_analysis['treat'] == 1)).astype(int)

# Use 2011 as reference year (last pre-treatment year)
year_vars = [f'treat_y{y}' for y in sorted(df_analysis['YEAR'].unique()) if y != 2011]
formula_event = 'fulltime ~ treat + C(YEAR) + ' + ' + '.join(year_vars)

model_event = smf.wls(formula_event, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\n   Event Study Coefficients (reference: 2011):")
event_results = []
for var in year_vars:
    year = int(var.replace('treat_y', ''))
    event_results.append({
        'Year': year,
        'Coefficient': model_event.params[var],
        'Std_Error': model_event.bse[var],
        'p_value': model_event.pvalues[var],
        'CI_low': model_event.conf_int().loc[var, 0],
        'CI_high': model_event.conf_int().loc[var, 1]
    })

event_df = pd.DataFrame(event_results)
print(event_df.round(4).to_string(index=False))

print("\n9. Robustness checks...")

# Robustness 1: Alternative age ranges
print("\n   Robustness 1: Narrower age bands (27-29 vs 32-34)")
df_narrow = df_analysis.copy()
df_narrow['treat_narrow'] = ((df_narrow['age_june2012'] >= 27) & (df_narrow['age_june2012'] <= 29)).astype(int)
df_narrow['control_narrow'] = ((df_narrow['age_june2012'] >= 32) & (df_narrow['age_june2012'] <= 34)).astype(int)
df_narrow = df_narrow[(df_narrow['treat_narrow'] == 1) | (df_narrow['control_narrow'] == 1)]
df_narrow['treat_post_narrow'] = df_narrow['treat_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treat_narrow + C(YEAR) + treat_post_narrow + male + married + educ_hs + educ_college',
                        data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"   DiD estimate: {model_narrow.params['treat_post_narrow']:.4f}")
print(f"   SE: {model_narrow.bse['treat_post_narrow']:.4f}, p={model_narrow.pvalues['treat_post_narrow']:.4f}")
print(f"   N: {int(model_narrow.nobs):,}")

# Robustness 2: Including 2012
print("\n   Robustness 2: Including 2012 as post-treatment")
df_with2012 = df_eligible.copy()
df_with2012['post2'] = (df_with2012['YEAR'] >= 2012).astype(int)
df_with2012['treat_post2'] = df_with2012['treat'] * df_with2012['post2']
df_with2012['male'] = (df_with2012['SEX'] == 1).astype(int)
df_with2012['married'] = (df_with2012['MARST'] <= 2).astype(int)
df_with2012['educ_hs'] = (df_with2012['EDUC'] >= 6).astype(int)
df_with2012['educ_college'] = (df_with2012['EDUC'] >= 10).astype(int)

model_2012 = smf.wls('fulltime ~ treat + C(YEAR) + treat_post2 + male + married + educ_hs + educ_college',
                      data=df_with2012, weights=df_with2012['PERWT']).fit(cov_type='HC1')
print(f"   DiD estimate: {model_2012.params['treat_post2']:.4f}")
print(f"   SE: {model_2012.bse['treat_post2']:.4f}, p={model_2012.pvalues['treat_post2']:.4f}")

# Robustness 3: By gender
print("\n   Robustness 3: Heterogeneity by gender")
df_male = df_analysis[df_analysis['male']==1]
df_female = df_analysis[df_analysis['male']==0]

model_male = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + married + educ_hs + educ_college',
                      data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + married + educ_hs + educ_college',
                        data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"   Males - DiD: {model_male.params['treat_post']:.4f}, SE: {model_male.bse['treat_post']:.4f}, N={int(model_male.nobs):,}")
print(f"   Females - DiD: {model_female.params['treat_post']:.4f}, SE: {model_female.bse['treat_post']:.4f}, N={int(model_female.nobs):,}")

# Robustness 4: Placebo test using pre-treatment data only
print("\n   Robustness 4: Placebo test (2006-2008 vs 2009-2011)")
df_placebo = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_placebo['placebo_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treat_placebo'] = df_placebo['treat'] * df_placebo['placebo_post']

model_placebo = smf.wls('fulltime ~ treat + C(YEAR) + treat_placebo + male + married + educ_hs + educ_college',
                         data=df_placebo, weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"   Placebo DiD: {model_placebo.params['treat_placebo']:.4f}")
print(f"   SE: {model_placebo.bse['treat_placebo']:.4f}, p={model_placebo.pvalues['treat_placebo']:.4f}")

print("\n10. Saving results...")

# Save main results
results_dict = {
    'Model': ['Basic DiD (unweighted)', 'Basic DiD (weighted)', 'Year FE (weighted)',
              'Covariates (weighted)', 'Year+State FE (weighted)', 'Preferred (robust SE)'],
    'Estimate': [model1.params['treat_post'], model2.params['treat_post'], model3.params['treat_post'],
                 model4.params['treat_post'], model5.params['treat_post'], model6.params['treat_post']],
    'SE': [model1.bse['treat_post'], model2.bse['treat_post'], model3.bse['treat_post'],
           model4.bse['treat_post'], model5.bse['treat_post'], model6.bse['treat_post']],
    'p_value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'], model3.pvalues['treat_post'],
                model4.pvalues['treat_post'], model5.pvalues['treat_post'], model6.pvalues['treat_post']],
    'CI_low': [model1.conf_int().loc['treat_post', 0], model2.conf_int().loc['treat_post', 0],
               model3.conf_int().loc['treat_post', 0], model4.conf_int().loc['treat_post', 0],
               model5.conf_int().loc['treat_post', 0], model6.conf_int().loc['treat_post', 0]],
    'CI_high': [model1.conf_int().loc['treat_post', 1], model2.conf_int().loc['treat_post', 1],
                model3.conf_int().loc['treat_post', 1], model4.conf_int().loc['treat_post', 1],
                model5.conf_int().loc['treat_post', 1], model6.conf_int().loc['treat_post', 1]],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs), int(model6.nobs)]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('main_results.csv', index=False)
print("   Saved main_results.csv")

# Save year-by-year results
year_df.to_csv('yearly_results.csv', index=False)
print("   Saved yearly_results.csv")

# Save event study results
event_df.to_csv('event_study_results.csv', index=False)
print("   Saved event_study_results.csv")

# Save descriptive statistics
desc_detail.T.to_csv('descriptive_stats.csv')
print("   Saved descriptive_stats.csv")

# Save full model summaries
with open('model_summaries.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("DACA REPLICATION: FULL MODEL SUMMARIES\n")
    f.write("="*80 + "\n\n")

    f.write("MODEL 1: Basic DiD (unweighted)\n")
    f.write("-"*40 + "\n")
    f.write(model1.summary().as_text())
    f.write("\n\n")

    f.write("MODEL 2: Basic DiD (weighted)\n")
    f.write("-"*40 + "\n")
    f.write(model2.summary().as_text())
    f.write("\n\n")

    f.write("MODEL 6: Preferred Specification (Year + State FE, Covariates, Robust SE)\n")
    f.write("-"*40 + "\n")
    f.write(model6.summary().as_text())
    f.write("\n\n")

    f.write("EVENT STUDY MODEL\n")
    f.write("-"*40 + "\n")
    f.write(model_event.summary().as_text())

print("   Saved model_summaries.txt")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

# Print key findings summary
print("\n*** KEY FINDINGS ***")
print(f"\nPreferred Estimate (Model 6):")
print(f"   Effect of DACA eligibility on full-time employment: {model6.params['treat_post']:.4f}")
print(f"   This represents a {model6.params['treat_post']*100:.2f} percentage point change")
print(f"   Robust standard error: {model6.bse['treat_post']:.4f}")
print(f"   95% Confidence Interval: [{model6.conf_int().loc['treat_post', 0]:.4f}, {model6.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   p-value: {model6.pvalues['treat_post']:.4f}")
print(f"   Sample size: {int(model6.nobs):,}")

# Calculate baseline rate for context
baseline_rate = df_analysis[(df_analysis['treat']==1) & (df_analysis['pre']==1)]['fulltime'].mean()
print(f"\n   Context: Pre-treatment full-time rate for treatment group: {baseline_rate:.4f}")
print(f"   Relative effect: {(model6.params['treat_post']/baseline_rate)*100:.2f}% change from baseline")
