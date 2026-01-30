"""
DACA Employment Replication Study - Analysis Script
Run 30

This script performs a difference-in-differences analysis examining the effect of
DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born
individuals in the United States.
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
pd.set_option('display.float_format', '{:.4f}'.format)

print("="*80)
print("DACA EMPLOYMENT REPLICATION STUDY - RUN 30")
print("="*80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "="*80)
print("SECTION 1: DATA LOADING")
print("="*80)

# Load the data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\nTotal observations loaded: {len(df):,}")
print(f"Number of variables: {len(df.columns)}")
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")
print(f"Year range: {df['YEAR'].min()} to {df['YEAR'].max()}")

# =============================================================================
# 2. VARIABLE VERIFICATION
# =============================================================================
print("\n" + "="*80)
print("SECTION 2: VARIABLE VERIFICATION")
print("="*80)

# Check key variables
print("\n--- FT (Full-time employment) ---")
print(df['FT'].value_counts().sort_index())
print(f"Mean FT: {df['FT'].mean():.4f}")

print("\n--- ELIGIBLE (Treatment group indicator) ---")
print(df['ELIGIBLE'].value_counts().sort_index())
print(f"Proportion eligible: {df['ELIGIBLE'].mean():.4f}")

print("\n--- AFTER (Post-DACA indicator) ---")
print(df['AFTER'].value_counts().sort_index())

# Year by AFTER
print("\n--- YEAR by AFTER ---")
print(df.groupby('YEAR')['AFTER'].first())

# =============================================================================
# 3. SAMPLE CHARACTERISTICS
# =============================================================================
print("\n" + "="*80)
print("SECTION 3: SAMPLE CHARACTERISTICS")
print("="*80)

# Sample sizes by group
print("\n--- Sample Sizes by Treatment and Period ---")
sample_counts = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
sample_counts.index = ['Control (31-35)', 'Treatment (26-30)']
sample_counts.columns = ['Pre-DACA (2008-11)', 'Post-DACA (2013-16)']
print(sample_counts)
print(f"\nTotal N: {len(df):,}")

# Sample by year
print("\n--- Sample Size by Year ---")
print(df.groupby('YEAR').size())

# Demographics
print("\n--- Key Demographics ---")
print(f"Mean age: {df['AGE'].mean():.2f}")
print(f"Age range: {df['AGE'].min()} to {df['AGE'].max()}")

# Sex distribution (1=Male, 2=Female in IPUMS)
print(f"\nSex distribution:")
print(df['SEX'].value_counts().sort_index())
male_pct = (df['SEX'] == 1).mean() * 100
print(f"Male: {male_pct:.1f}%, Female: {100-male_pct:.1f}%")

# Marital status
print(f"\nMarital status distribution:")
print(df['MARST'].value_counts().sort_index())

# Education
print(f"\nEducation distribution (EDUC):")
print(df['EDUC'].value_counts().sort_index().head(15))

# =============================================================================
# 4. FULL-TIME EMPLOYMENT RATES BY GROUP
# =============================================================================
print("\n" + "="*80)
print("SECTION 4: FULL-TIME EMPLOYMENT RATES")
print("="*80)

# Unweighted rates
print("\n--- Unweighted Full-Time Employment Rates ---")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre-DACA', 'Post-DACA']
print(ft_rates)

# Calculate DiD manually
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"\nDifference-in-Differences (unweighted):")
print(f"Treatment change: {post_treat - pre_treat:.4f} ({pre_treat:.4f} -> {post_treat:.4f})")
print(f"Control change: {post_control - pre_control:.4f} ({pre_control:.4f} -> {post_control:.4f})")
print(f"DiD estimate: {(post_treat - pre_treat) - (post_control - pre_control):.4f}")

# Weighted rates
print("\n--- Weighted Full-Time Employment Rates ---")
def weighted_mean(df_sub, col, weight_col='PERWT'):
    return np.average(df_sub[col], weights=df_sub[weight_col])

ft_rates_weighted = pd.DataFrame(index=['Control (31-35)', 'Treatment (26-30)'],
                                  columns=['Pre-DACA', 'Post-DACA'])

for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE']==eligible) & (df['AFTER']==after)]
        wt_mean = weighted_mean(subset, 'FT')
        row = 'Treatment (26-30)' if eligible == 1 else 'Control (31-35)'
        col = 'Post-DACA' if after == 1 else 'Pre-DACA'
        ft_rates_weighted.loc[row, col] = wt_mean

print(ft_rates_weighted)

# Weighted DiD
pre_treat_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)], 'FT')
post_treat_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)], 'FT')
pre_control_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)], 'FT')
post_control_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)], 'FT')

print(f"\nDifference-in-Differences (weighted):")
print(f"Treatment change: {post_treat_w - pre_treat_w:.4f} ({pre_treat_w:.4f} -> {post_treat_w:.4f})")
print(f"Control change: {post_control_w - pre_control_w:.4f} ({pre_control_w:.4f} -> {post_control_w:.4f})")
print(f"DiD estimate: {(post_treat_w - pre_treat_w) - (post_control_w - pre_control_w):.4f}")

# =============================================================================
# 5. FULL-TIME EMPLOYMENT BY YEAR
# =============================================================================
print("\n" + "="*80)
print("SECTION 5: FULL-TIME EMPLOYMENT TRENDS BY YEAR")
print("="*80)

# By year and group (unweighted)
print("\n--- Unweighted FT rates by year ---")
ft_by_year = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
ft_by_year.columns = ['Control (31-35)', 'Treatment (26-30)']
print(ft_by_year)
ft_by_year['Difference'] = ft_by_year['Treatment (26-30)'] - ft_by_year['Control (31-35)']
print("\nWith difference column:")
print(ft_by_year)

# Weighted by year
print("\n--- Weighted FT rates by year ---")
ft_by_year_weighted = {}
for year in sorted(df['YEAR'].unique()):
    for eligible in [0, 1]:
        subset = df[(df['YEAR']==year) & (df['ELIGIBLE']==eligible)]
        wt_mean = weighted_mean(subset, 'FT')
        ft_by_year_weighted[(year, eligible)] = wt_mean

ft_year_df = pd.DataFrame(ft_by_year_weighted, index=['FT_rate']).T
ft_year_df.index = pd.MultiIndex.from_tuples(ft_year_df.index, names=['Year', 'Eligible'])
ft_year_df = ft_year_df.unstack()
ft_year_df.columns = ['Control (31-35)', 'Treatment (26-30)']
print(ft_year_df)

# =============================================================================
# 6. MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "="*80)
print("SECTION 6: MAIN DiD REGRESSION ANALYSIS")
print("="*80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (OLS, unweighted) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary())

# Model 2: Basic DiD with robust standard errors
print("\n--- Model 2: Basic DiD (OLS, robust SE) ---")
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model2.summary())

# Model 3: Basic DiD (weighted)
print("\n--- Model 3: Basic DiD (WLS, weighted) ---")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model3.summary())

# Model 4: WLS with robust SE
print("\n--- Model 4: Basic DiD (WLS, robust SE) ---")
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# Store key results
key_results = {
    'Model 1 (OLS)': {
        'coef': model1.params['ELIGIBLE_AFTER'],
        'se': model1.bse['ELIGIBLE_AFTER'],
        'pval': model1.pvalues['ELIGIBLE_AFTER'],
        'ci_low': model1.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model1.conf_int().loc['ELIGIBLE_AFTER', 1]
    },
    'Model 2 (OLS, robust)': {
        'coef': model2.params['ELIGIBLE_AFTER'],
        'se': model2.bse['ELIGIBLE_AFTER'],
        'pval': model2.pvalues['ELIGIBLE_AFTER'],
        'ci_low': model2.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model2.conf_int().loc['ELIGIBLE_AFTER', 1]
    },
    'Model 3 (WLS)': {
        'coef': model3.params['ELIGIBLE_AFTER'],
        'se': model3.bse['ELIGIBLE_AFTER'],
        'pval': model3.pvalues['ELIGIBLE_AFTER'],
        'ci_low': model3.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model3.conf_int().loc['ELIGIBLE_AFTER', 1]
    },
    'Model 4 (WLS, robust)': {
        'coef': model4.params['ELIGIBLE_AFTER'],
        'se': model4.bse['ELIGIBLE_AFTER'],
        'pval': model4.pvalues['ELIGIBLE_AFTER'],
        'ci_low': model4.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model4.conf_int().loc['ELIGIBLE_AFTER', 1]
    }
}

print("\n--- Summary of DiD Estimates ---")
results_df = pd.DataFrame(key_results).T
print(results_df)

# =============================================================================
# 7. MODELS WITH COVARIATES
# =============================================================================
print("\n" + "="*80)
print("SECTION 7: DiD WITH COVARIATES")
print("="*80)

# Prepare covariates
# SEX: 1=Male, 2=Female
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# MARST: 1=Married spouse present, 2=Married spouse absent, 3=Separated, 4=Divorced, 5=Widowed, 6=Never married
df['MARRIED'] = (df['MARST'] <= 2).astype(int)

# Education dummies (using EDUC)
# EDUC codes: 0=N/A, 1=No schooling, 2=Nursery-4th grade, 3=Grades 5-6, 4=Grades 7-8
# 5=Grade 9, 6=Grade 10, 7=Grade 11, 8=12th grade no diploma, 9=HS graduate/GED
# 10=Some college but less than 1 yr, 11=1+ years college, no degree
df['HS_OR_MORE'] = (df['EDUC'] >= 6).astype(int)

# Create state dummies (use C() notation in formula for categorical)
# Also year dummies

# Model 5: With demographic controls
print("\n--- Model 5: DiD with demographic controls (age, sex, marital) ---")
model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary())

# Model 6: With education
print("\n--- Model 6: DiD with demographics + education ---")
model6 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + HS_OR_MORE',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model6.summary())

# Model 7: With year fixed effects
print("\n--- Model 7: DiD with year fixed effects ---")
model7 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model7.summary())

# Model 8: With year FE and demographics
print("\n--- Model 8: DiD with year FE + demographics ---")
model8 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + HS_OR_MORE',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model8.summary())

# Model 9: With state fixed effects
print("\n--- Model 9: DiD with state fixed effects ---")
model9 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model9.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model9.bse['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model9.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 10: Full model with state and year FE and demographics
print("\n--- Model 10: Full model (state FE + year FE + demographics) ---")
model10 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER + C(STATEFIP) + AGE + FEMALE + MARRIED + HS_OR_MORE',
                   data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model10.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model10.bse['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model10.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model10.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model10.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# =============================================================================
# 8. SUMMARY OF ALL MODELS
# =============================================================================
print("\n" + "="*80)
print("SECTION 8: SUMMARY OF ALL MODELS")
print("="*80)

all_models = {
    'M1: Basic OLS': model1,
    'M2: OLS robust': model2,
    'M3: WLS': model3,
    'M4: WLS robust': model4,
    'M5: Demographics': model5,
    'M6: Demo+Educ': model6,
    'M7: Year FE': model7,
    'M8: Year FE+Demo': model8,
    'M9: State FE': model9,
    'M10: Full model': model10
}

summary_data = []
for name, model in all_models.items():
    summary_data.append({
        'Model': name,
        'DiD Coef': model.params['ELIGIBLE_AFTER'],
        'Std Err': model.bse['ELIGIBLE_AFTER'],
        't-stat': model.tvalues['ELIGIBLE_AFTER'],
        'p-value': model.pvalues['ELIGIBLE_AFTER'],
        'CI Low': model.conf_int().loc['ELIGIBLE_AFTER', 0],
        'CI High': model.conf_int().loc['ELIGIBLE_AFTER', 1],
        'N': int(model.nobs),
        'R-squared': model.rsquared
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# =============================================================================
# 9. ROBUSTNESS: EVENT STUDY / YEAR-BY-YEAR EFFECTS
# =============================================================================
print("\n" + "="*80)
print("SECTION 9: EVENT STUDY ANALYSIS")
print("="*80)

# Create year dummies interacted with ELIGIBLE
# Use 2011 as reference year (last pre-treatment year)
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
# 2011 is reference
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions
df['ELIG_2008'] = df['ELIGIBLE'] * df['YEAR_2008']
df['ELIG_2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIG_2010'] = df['ELIGIBLE'] * df['YEAR_2010']
df['ELIG_2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIG_2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIG_2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIG_2016'] = df['ELIGIBLE'] * df['YEAR_2016']

# Event study model
print("\n--- Event Study Model (2011 as reference year) ---")
event_formula = ('FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + '
                 'ELIG_2008 + ELIG_2009 + ELIG_2010 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016')
event_model = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(event_model.summary())

# Extract event study coefficients
event_coefs = {}
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    if year == 2011:
        event_coefs[year] = {'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0}
    else:
        var = f'ELIG_{year}'
        event_coefs[year] = {
            'coef': event_model.params[var],
            'se': event_model.bse[var],
            'ci_low': event_model.conf_int().loc[var, 0],
            'ci_high': event_model.conf_int().loc[var, 1]
        }

print("\n--- Event Study Coefficients ---")
event_df = pd.DataFrame(event_coefs).T
event_df.index.name = 'Year'
print(event_df)

# =============================================================================
# 10. SUBGROUP ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SECTION 10: SUBGROUP ANALYSIS")
print("="*80)

# By sex
print("\n--- By Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"{sex_name}:")
    print(f"  DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f}, SE = {model_sub.bse['ELIGIBLE_AFTER']:.4f}, "
          f"p = {model_sub.pvalues['ELIGIBLE_AFTER']:.4f}, N = {int(model_sub.nobs):,}")

# By education
print("\n--- By Education Level ---")
df_hs = df[df['HS_OR_MORE'] == 1]
df_less = df[df['HS_OR_MORE'] == 0]

model_hs = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                   data=df_hs, weights=df_hs['PERWT']).fit(cov_type='HC1')
print(f"HS or more:")
print(f"  DiD = {model_hs.params['ELIGIBLE_AFTER']:.4f}, SE = {model_hs.bse['ELIGIBLE_AFTER']:.4f}, "
      f"p = {model_hs.pvalues['ELIGIBLE_AFTER']:.4f}, N = {int(model_hs.nobs):,}")

model_less = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                     data=df_less, weights=df_less['PERWT']).fit(cov_type='HC1')
print(f"Less than HS:")
print(f"  DiD = {model_less.params['ELIGIBLE_AFTER']:.4f}, SE = {model_less.bse['ELIGIBLE_AFTER']:.4f}, "
      f"p = {model_less.pvalues['ELIGIBLE_AFTER']:.4f}, N = {int(model_less.nobs):,}")

# By marital status
print("\n--- By Marital Status ---")
df_married = df[df['MARRIED'] == 1]
df_unmarried = df[df['MARRIED'] == 0]

model_mar = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                    data=df_married, weights=df_married['PERWT']).fit(cov_type='HC1')
print(f"Married:")
print(f"  DiD = {model_mar.params['ELIGIBLE_AFTER']:.4f}, SE = {model_mar.bse['ELIGIBLE_AFTER']:.4f}, "
      f"p = {model_mar.pvalues['ELIGIBLE_AFTER']:.4f}, N = {int(model_mar.nobs):,}")

model_unmar = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                      data=df_unmarried, weights=df_unmarried['PERWT']).fit(cov_type='HC1')
print(f"Unmarried:")
print(f"  DiD = {model_unmar.params['ELIGIBLE_AFTER']:.4f}, SE = {model_unmar.bse['ELIGIBLE_AFTER']:.4f}, "
      f"p = {model_unmar.pvalues['ELIGIBLE_AFTER']:.4f}, N = {int(model_unmar.nobs):,}")

# =============================================================================
# 11. PARALLEL TRENDS CHECK
# =============================================================================
print("\n" + "="*80)
print("SECTION 11: PARALLEL TRENDS DIAGNOSTIC")
print("="*80)

# Pre-treatment trends test
df_pre = df[df['AFTER'] == 0].copy()
df_pre['TIME'] = df_pre['YEAR'] - 2008  # Linear time trend

# Test if treatment group has different pre-trend
df_pre['ELIG_TIME'] = df_pre['ELIGIBLE'] * df_pre['TIME']

print("\n--- Pre-treatment Differential Trends Test ---")
trend_model = smf.wls('FT ~ ELIGIBLE + TIME + ELIG_TIME',
                      data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(trend_model.summary())
print(f"\nDifferential trend coefficient (ELIG_TIME): {trend_model.params['ELIG_TIME']:.4f}")
print(f"p-value: {trend_model.pvalues['ELIG_TIME']:.4f}")

if trend_model.pvalues['ELIG_TIME'] > 0.05:
    print("Parallel trends assumption appears satisfied (differential trend not statistically significant)")
else:
    print("WARNING: Possible violation of parallel trends assumption")

# =============================================================================
# 12. PLACEBO TEST
# =============================================================================
print("\n" + "="*80)
print("SECTION 12: PLACEBO TEST")
print("="*80)

# Use pre-treatment period only, treat 2010-2011 as "post" and 2008-2009 as "pre"
df_placebo = df[df['YEAR'].isin([2008, 2009, 2010, 2011])].copy()
df_placebo['PLACEBO_AFTER'] = (df_placebo['YEAR'] >= 2010).astype(int)
df_placebo['PLACEBO_INTERACTION'] = df_placebo['ELIGIBLE'] * df_placebo['PLACEBO_AFTER']

print("\n--- Placebo Test (2008-2009 vs 2010-2011, pre-DACA only) ---")
placebo_model = smf.wls('FT ~ ELIGIBLE + PLACEBO_AFTER + PLACEBO_INTERACTION',
                        data=df_placebo, weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD coefficient: {placebo_model.params['PLACEBO_INTERACTION']:.4f}")
print(f"SE: {placebo_model.bse['PLACEBO_INTERACTION']:.4f}")
print(f"p-value: {placebo_model.pvalues['PLACEBO_INTERACTION']:.4f}")
print(f"95% CI: [{placebo_model.conf_int().loc['PLACEBO_INTERACTION', 0]:.4f}, "
      f"{placebo_model.conf_int().loc['PLACEBO_INTERACTION', 1]:.4f}]")

# =============================================================================
# 13. STATE-LEVEL POLICY INTERACTIONS
# =============================================================================
print("\n" + "="*80)
print("SECTION 13: STATE POLICY INTERACTIONS")
print("="*80)

# Check if driver's license access affects the impact
print("\n--- Interaction with Driver's License Access ---")
df['ELIG_AFTER_DL'] = df['ELIGIBLE_AFTER'] * df['DRIVERSLICENSES']
dl_model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + DRIVERSLICENSES + ELIG_AFTER_DL',
                   data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Base DiD effect: {dl_model.params['ELIGIBLE_AFTER']:.4f}")
print(f"Driver's license interaction: {dl_model.params['ELIG_AFTER_DL']:.4f}")
print(f"Interaction p-value: {dl_model.pvalues['ELIG_AFTER_DL']:.4f}")

# E-Verify interaction
print("\n--- Interaction with E-Verify ---")
df['ELIG_AFTER_EV'] = df['ELIGIBLE_AFTER'] * df['EVERIFY']
ev_model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + EVERIFY + ELIG_AFTER_EV',
                   data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Base DiD effect: {ev_model.params['ELIGIBLE_AFTER']:.4f}")
print(f"E-Verify interaction: {ev_model.params['ELIG_AFTER_EV']:.4f}")
print(f"Interaction p-value: {ev_model.pvalues['ELIG_AFTER_EV']:.4f}")

# =============================================================================
# 14. FINAL PREFERRED ESTIMATE
# =============================================================================
print("\n" + "="*80)
print("SECTION 14: PREFERRED ESTIMATE")
print("="*80)

# The preferred specification: WLS with robust SE, including state and year FE and demographics
print("\n*** PREFERRED MODEL: WLS with state FE, year FE, and demographics ***")
print(f"Sample size: {int(model10.nobs):,}")
print(f"DiD coefficient: {model10.params['ELIGIBLE_AFTER']:.6f}")
print(f"Standard error: {model10.bse['ELIGIBLE_AFTER']:.6f}")
print(f"t-statistic: {model10.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model10.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI: [{model10.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model10.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")
print(f"R-squared: {model10.rsquared:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# =============================================================================
# 15. SAVE RESULTS FOR REPORT
# =============================================================================

# Save key results to CSV for use in report
summary_df.to_csv('model_summary.csv', index=False)

# Save event study coefficients
event_df.to_csv('event_study.csv')

# Save FT rates by year
ft_by_year.to_csv('ft_rates_by_year.csv')

print("\nResults saved to CSV files for report generation.")
