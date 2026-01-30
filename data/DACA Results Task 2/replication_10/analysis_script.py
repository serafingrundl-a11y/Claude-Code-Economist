"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born non-citizens.

Treatment: Ages 26-30 as of June 15, 2012
Control: Ages 31-35 as of June 15, 2012
Method: Difference-in-Differences
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
print("DACA REPLICATION STUDY - DATA ANALYSIS")
print("="*80)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
print("\n1. Loading data...")
data_path = "data/data.csv"

# Load with appropriate dtypes to reduce memory
dtype_dict = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'MARST': 'int8',
    'UHRSWORK': 'int8',
    'EMPSTAT': 'int8',
    'PERWT': 'float64'
}

# Load only needed columns
cols_needed = ['YEAR', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'MARST',
               'UHRSWORK', 'EMPSTAT', 'PERWT', 'LABFORCE']

df = pd.read_csv(data_path, usecols=cols_needed, dtype=dtype_dict)
print(f"   Loaded {len(df):,} observations")

# -----------------------------------------------------------------------------
# 2. FILTER TO ANALYSIS SAMPLE
# -----------------------------------------------------------------------------
print("\n2. Filtering to analysis sample...")

# Step 2a: Hispanic-Mexican ethnicity (HISPAN == 1 is Mexican)
df_sample = df[df['HISPAN'] == 1].copy()
print(f"   After Hispanic-Mexican filter: {len(df_sample):,}")

# Step 2b: Born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200]
print(f"   After Mexico birthplace filter: {len(df_sample):,}")

# Step 2c: Not a citizen (CITIZEN == 3)
# Non-citizens who have not received papers - proxy for undocumented
df_sample = df_sample[df_sample['CITIZEN'] == 3]
print(f"   After non-citizen filter: {len(df_sample):,}")

# Step 2d: Exclude 2012 (ambiguous treatment timing)
df_sample = df_sample[df_sample['YEAR'] != 2012]
print(f"   After excluding 2012: {len(df_sample):,}")

# -----------------------------------------------------------------------------
# 3. DETERMINE DACA ELIGIBILITY AND AGE GROUPS
# -----------------------------------------------------------------------------
print("\n3. Determining treatment and control groups...")

# DACA was implemented June 15, 2012
# Treatment: ages 26-30 on June 15, 2012
# Control: ages 31-35 on June 15, 2012

# Calculate age as of June 15, 2012
# If birth quarter is 1 or 2 (Jan-Jun), they would have had their birthday by June 15
# If birth quarter is 3 or 4 (Jul-Dec), they would not have had their birthday yet

# Age on June 15, 2012 = 2012 - BIRTHYR - adjustment for birth quarter
# Birth quarter 1,2 -> birthday before June 15 -> age = 2012 - BIRTHYR
# Birth quarter 3,4 -> birthday after June 15 -> age = 2012 - BIRTHYR - 1

df_sample['age_june2012'] = 2012 - df_sample['BIRTHYR']
# Adjust for those whose birthday hasn't occurred yet by June 15
df_sample.loc[df_sample['BIRTHQTR'].isin([3, 4]), 'age_june2012'] -= 1

# Treatment group: 26-30 on June 15, 2012
# Control group: 31-35 on June 15, 2012
df_sample['treated'] = ((df_sample['age_june2012'] >= 26) &
                        (df_sample['age_june2012'] <= 30)).astype(int)
df_sample['control'] = ((df_sample['age_june2012'] >= 31) &
                        (df_sample['age_june2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_sample = df_sample[(df_sample['treated'] == 1) | (df_sample['control'] == 1)]
print(f"   After age group filter (26-35 as of June 2012): {len(df_sample):,}")

# -----------------------------------------------------------------------------
# 4. ADDITIONAL DACA ELIGIBILITY CRITERIA
# -----------------------------------------------------------------------------
print("\n4. Applying additional DACA eligibility criteria...")

# Must have arrived before 16th birthday
# arrival_age = YRIMMIG - BIRTHYR (approximate, since we don't have month of immigration)
df_sample['arrival_age'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Filter: arrived before age 16
df_sample = df_sample[df_sample['arrival_age'] < 16]
print(f"   After arrived before age 16 filter: {len(df_sample):,}")

# Must have been in US since June 15, 2007 (arrived by 2007)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]
print(f"   After continuous residence since 2007 filter: {len(df_sample):,}")

# Note: We cannot verify physical presence or lawful status directly in ACS

# -----------------------------------------------------------------------------
# 5. CREATE ANALYSIS VARIABLES
# -----------------------------------------------------------------------------
print("\n5. Creating analysis variables...")

# Full-time employment outcome (35+ hours per week)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Post-treatment indicator (2013-2016)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Interaction term
df_sample['treated_post'] = df_sample['treated'] * df_sample['post']

# Age at survey time (for controls)
df_sample['age_survey'] = df_sample['AGE']

# Education categories
df_sample['educ_cat'] = pd.cut(df_sample['EDUC'],
                                bins=[-1, 5, 6, 10, 11],
                                labels=['less_hs', 'hs', 'some_college', 'college_plus'])

# Female indicator
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)

# Married indicator
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

print(f"\n   Final analysis sample: {len(df_sample):,} observations")

# -----------------------------------------------------------------------------
# 6. DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("6. DESCRIPTIVE STATISTICS")
print("="*80)

# Sample sizes by group and period
print("\n6.1 Sample sizes by group and period:")
sample_counts = df_sample.groupby(['treated', 'post']).size().unstack()
sample_counts.index = ['Control (31-35)', 'Treatment (26-30)']
sample_counts.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(sample_counts)

# Weighted sample sizes
print("\n6.2 Weighted sample sizes:")
weighted_counts = df_sample.groupby(['treated', 'post'])['PERWT'].sum().unstack()
weighted_counts.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_counts.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(weighted_counts.round(0))

# Full-time employment rates by group and period
print("\n6.3 Full-time employment rates (weighted):")
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

ft_rates = df_sample.groupby(['treated', 'post']).apply(weighted_mean).unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print((ft_rates * 100).round(2))

# Calculate simple DiD estimate
print("\n6.4 Simple Difference-in-Differences:")
did_simple = (ft_rates.loc['Treatment (26-30)', 'Post (2013-2016)'] -
              ft_rates.loc['Treatment (26-30)', 'Pre (2006-2011)']) - \
             (ft_rates.loc['Control (31-35)', 'Post (2013-2016)'] -
              ft_rates.loc['Control (31-35)', 'Pre (2006-2011)'])
print(f"   Treatment change: {(ft_rates.loc['Treatment (26-30)', 'Post (2013-2016)'] - ft_rates.loc['Treatment (26-30)', 'Pre (2006-2011)'])*100:.2f} pp")
print(f"   Control change:   {(ft_rates.loc['Control (31-35)', 'Post (2013-2016)'] - ft_rates.loc['Control (31-35)', 'Pre (2006-2011)'])*100:.2f} pp")
print(f"   DiD estimate:     {did_simple*100:.2f} percentage points")

# Covariate balance
print("\n6.5 Covariate means by treatment group (pre-period, weighted):")
pre_data = df_sample[df_sample['post'] == 0]

def weighted_stats(group, var):
    return np.average(group[var], weights=group['PERWT'])

for var in ['female', 'married', 'age_survey']:
    treat_mean = weighted_stats(pre_data[pre_data['treated']==1], var)
    control_mean = weighted_stats(pre_data[pre_data['treated']==0], var)
    print(f"   {var}: Treatment={treat_mean:.3f}, Control={control_mean:.3f}")

# Education distribution
print("\n   Education distribution (pre-period):")
for group_name, group_val in [('Treatment', 1), ('Control', 0)]:
    group_data = pre_data[pre_data['treated']==group_val]
    print(f"   {group_name}:")
    for educ in ['less_hs', 'hs', 'some_college', 'college_plus']:
        mask = (group_data['educ_cat']==educ).astype(int)
        pct = np.average(mask, weights=group_data['PERWT'])
        print(f"      {educ}: {pct*100:.1f}%")

# -----------------------------------------------------------------------------
# 7. MAIN REGRESSION ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("7. REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD (unweighted)
print("\n7.1 Model 1: Basic DiD (unweighted)")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_sample).fit()
print(f"   DiD Coefficient: {model1.params['treated_post']:.4f}")
print(f"   Standard Error:  {model1.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model1.tvalues['treated_post']:.3f}")
print(f"   p-value:         {model1.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model1.conf_int().loc['treated_post', 0]:.4f}, {model1.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   N:               {int(model1.nobs):,}")

# Model 2: Basic DiD (weighted)
print("\n7.2 Model 2: Basic DiD (weighted)")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_sample, weights=df_sample['PERWT']).fit()
print(f"   DiD Coefficient: {model2.params['treated_post']:.4f}")
print(f"   Standard Error:  {model2.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model2.tvalues['treated_post']:.3f}")
print(f"   p-value:         {model2.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model2.conf_int().loc['treated_post', 0]:.4f}, {model2.conf_int().loc['treated_post', 1]:.4f}]")

# Model 3: With demographic controls (weighted)
print("\n7.3 Model 3: With demographic controls (weighted)")
model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + C(educ_cat)',
                  data=df_sample, weights=df_sample['PERWT']).fit()
print(f"   DiD Coefficient: {model3.params['treated_post']:.4f}")
print(f"   Standard Error:  {model3.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model3.tvalues['treated_post']:.3f}")
print(f"   p-value:         {model3.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")

# Model 4: With year fixed effects (weighted)
print("\n7.4 Model 4: With year fixed effects (weighted)")
model4 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + C(educ_cat)',
                  data=df_sample, weights=df_sample['PERWT']).fit()
print(f"   DiD Coefficient: {model4.params['treated_post']:.4f}")
print(f"   Standard Error:  {model4.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model4.tvalues['treated_post']:.3f}")
print(f"   p-value:         {model4.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")

# Model 5: With state fixed effects (weighted)
print("\n7.5 Model 5: With year and state fixed effects (weighted)")
model5 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + C(educ_cat)',
                  data=df_sample, weights=df_sample['PERWT']).fit()
print(f"   DiD Coefficient: {model5.params['treated_post']:.4f}")
print(f"   Standard Error:  {model5.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model5.tvalues['treated_post']:.3f}")
print(f"   p-value:         {model5.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")

# -----------------------------------------------------------------------------
# 8. ROBUST STANDARD ERRORS
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("8. MAIN MODEL WITH ROBUST STANDARD ERRORS")
print("="*80)

# Preferred specification with robust (HC1) standard errors
model_robust = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + C(educ_cat)',
                        data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print("\nPreferred Model (Year + State FE, Controls, Robust SE):")
print(f"   DiD Coefficient: {model_robust.params['treated_post']:.4f}")
print(f"   Robust SE:       {model_robust.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model_robust.tvalues['treated_post']:.3f}")
print(f"   p-value:         {model_robust.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model_robust.conf_int().loc['treated_post', 0]:.4f}, {model_robust.conf_int().loc['treated_post', 1]:.4f}]")

# -----------------------------------------------------------------------------
# 9. CLUSTERED STANDARD ERRORS (BY STATE)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("9. MAIN MODEL WITH STATE-CLUSTERED STANDARD ERRORS")
print("="*80)

model_cluster = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + C(educ_cat)',
                         data=df_sample, weights=df_sample['PERWT']).fit(cov_type='cluster',
                         cov_kwds={'groups': df_sample['STATEFIP']})
print("\nPreferred Model (State-Clustered SE):")
print(f"   DiD Coefficient: {model_cluster.params['treated_post']:.4f}")
print(f"   Clustered SE:    {model_cluster.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model_cluster.tvalues['treated_post']:.3f}")
print(f"   p-value:         {model_cluster.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model_cluster.conf_int().loc['treated_post', 0]:.4f}, {model_cluster.conf_int().loc['treated_post', 1]:.4f}]")

# -----------------------------------------------------------------------------
# 10. YEAR-BY-YEAR EFFECTS (EVENT STUDY)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("10. EVENT STUDY / YEAR-BY-YEAR EFFECTS")
print("="*80)

# Create year-specific treatment indicators
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    df_sample[f'treated_{year}'] = (df_sample['treated'] * (df_sample['YEAR'] == year)).astype(int)

# Run event study (omitting 2011 as reference)
formula_es = 'fulltime ~ treated + C(YEAR) + C(STATEFIP) + ' + \
             ' + '.join([f'treated_{y}' for y in years if y != 2011]) + \
             ' + female + married + C(educ_cat)'

model_es = smf.wls(formula_es, data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_es.params[f'treated_{year}']
    se = model_es.bse[f'treated_{year}']
    print(f"   {year}: {coef:.4f} ({se:.4f})")

# -----------------------------------------------------------------------------
# 11. HETEROGENEITY ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("11. HETEROGENEITY ANALYSIS")
print("="*80)

# By gender
print("\n11.1 By Gender:")
for sex, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_sample[df_sample['SEX'] == sex]
    model_sex = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + married + C(educ_cat)',
                         data=df_sex, weights=df_sex['PERWT']).fit(cov_type='HC1')
    print(f"   {sex_name}: DiD = {model_sex.params['treated_post']:.4f} (SE = {model_sex.bse['treated_post']:.4f}), N = {int(model_sex.nobs):,}")

# By education
print("\n11.2 By Education:")
df_sample['has_hs'] = (df_sample['EDUC'] >= 6).astype(int)
for educ, educ_name in [(0, 'Less than HS'), (1, 'HS or more')]:
    df_educ = df_sample[df_sample['has_hs'] == educ]
    model_educ = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married',
                          data=df_educ, weights=df_educ['PERWT']).fit(cov_type='HC1')
    print(f"   {educ_name}: DiD = {model_educ.params['treated_post']:.4f} (SE = {model_educ.bse['treated_post']:.4f}), N = {int(model_educ.nobs):,}")

# -----------------------------------------------------------------------------
# 12. PLACEBO TEST
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("12. PLACEBO TEST (PRE-PERIOD ONLY)")
print("="*80)

# Use only pre-period data, treat 2009-2011 as "post"
df_pre = df_sample[df_sample['post'] == 0].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_treated_post'] = df_pre['treated'] * df_pre['placebo_post']

model_placebo = smf.wls('fulltime ~ treated + C(YEAR) + placebo_treated_post + female + married + C(educ_cat)',
                         data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"\nPlacebo DiD (2006-2008 vs 2009-2011):")
print(f"   Coefficient: {model_placebo.params['placebo_treated_post']:.4f}")
print(f"   SE:          {model_placebo.bse['placebo_treated_post']:.4f}")
print(f"   p-value:     {model_placebo.pvalues['placebo_treated_post']:.4f}")

# -----------------------------------------------------------------------------
# 13. SUMMARY OF MAIN RESULTS
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("13. SUMMARY OF MAIN RESULTS")
print("="*80)

print("\n*** PREFERRED ESTIMATE ***")
print(f"Effect Size:        {model_cluster.params['treated_post']:.4f}")
print(f"                    ({model_cluster.params['treated_post']*100:.2f} percentage points)")
print(f"Standard Error:     {model_cluster.bse['treated_post']:.4f}")
print(f"95% CI:             [{model_cluster.conf_int().loc['treated_post', 0]:.4f}, {model_cluster.conf_int().loc['treated_post', 1]:.4f}]")
print(f"Sample Size:        {int(model_cluster.nobs):,}")
print(f"p-value:            {model_cluster.pvalues['treated_post']:.4f}")

# -----------------------------------------------------------------------------
# 14. SAVE RESULTS FOR REPORT
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("14. SAVING RESULTS")
print("="*80)

# Save key results to file
results_dict = {
    'preferred_estimate': model_cluster.params['treated_post'],
    'se_cluster': model_cluster.bse['treated_post'],
    'ci_lower': model_cluster.conf_int().loc['treated_post', 0],
    'ci_upper': model_cluster.conf_int().loc['treated_post', 1],
    'p_value': model_cluster.pvalues['treated_post'],
    'n_obs': int(model_cluster.nobs),
    'model1_coef': model1.params['treated_post'],
    'model1_se': model1.bse['treated_post'],
    'model2_coef': model2.params['treated_post'],
    'model2_se': model2.bse['treated_post'],
    'model3_coef': model3.params['treated_post'],
    'model3_se': model3.bse['treated_post'],
    'model4_coef': model4.params['treated_post'],
    'model4_se': model4.bse['treated_post'],
    'model5_coef': model5.params['treated_post'],
    'model5_se': model5.bse['treated_post'],
    'model_robust_coef': model_robust.params['treated_post'],
    'model_robust_se': model_robust.bse['treated_post'],
}

# Save to CSV
results_df = pd.DataFrame([results_dict])
results_df.to_csv('analysis_results.csv', index=False)
print("Results saved to analysis_results.csv")

# Full-time employment rates for table
ft_table = df_sample.groupby(['treated', 'post']).agg({
    'fulltime': lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']),
    'PERWT': 'sum'
}).reset_index()
ft_table.columns = ['treated', 'post', 'ft_rate', 'weighted_n']
ft_table.to_csv('fulltime_rates.csv', index=False)
print("Full-time rates saved to fulltime_rates.csv")

# Event study coefficients
es_results = []
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    es_results.append({
        'year': year,
        'coefficient': model_es.params[f'treated_{year}'],
        'se': model_es.bse[f'treated_{year}'],
        'ci_lower': model_es.conf_int().loc[f'treated_{year}', 0],
        'ci_upper': model_es.conf_int().loc[f'treated_{year}', 1]
    })
es_df = pd.DataFrame(es_results)
es_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Sample descriptives
desc_stats = df_sample.groupby('treated').agg({
    'female': lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']),
    'married': lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']),
    'age_survey': lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']),
    'PERWT': 'sum'
}).reset_index()
desc_stats.to_csv('descriptive_stats.csv', index=False)
print("Descriptive statistics saved to descriptive_stats.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
