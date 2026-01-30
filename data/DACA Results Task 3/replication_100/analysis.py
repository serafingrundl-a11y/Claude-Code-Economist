"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US

Methodology: Difference-in-Differences (DiD)
Treatment: ELIGIBLE = 1 (ages 26-30 in June 2012)
Control: ELIGIBLE = 0 (ages 31-35 in June 2012)
Pre-period: 2008-2011 (AFTER=0)
Post-period: 2013-2016 (AFTER=1)
Outcome: FT (full-time employment, 35+ hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Total observations: {len(df)}")

# ============================================
# SECTION 1: DATA EXPLORATION AND SUMMARY
# ============================================
print("\n" + "="*60)
print("SECTION 1: DATA EXPLORATION")
print("="*60)

print("\nSample size by group:")
print(pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True))

print("\nYears in sample:")
print(df['YEAR'].value_counts().sort_index())

print("\nFull-time employment rate by group and period:")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
print(ft_rates)

# Weighted means
print("\nWeighted full-time employment rates:")
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE']==eligible) & (df['AFTER']==after)]
        weighted_mean = np.average(subset['FT'], weights=subset['PERWT'])
        n = len(subset)
        label = f"ELIGIBLE={eligible}, AFTER={after}"
        print(f"  {label}: {weighted_mean:.4f} (N={n})")

# ============================================
# SECTION 2: SIMPLE DIFFERENCE-IN-DIFFERENCES
# ============================================
print("\n" + "="*60)
print("SECTION 2: SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("="*60)

# Calculate simple DiD by hand
ft_00 = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
ft_01 = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
ft_10 = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
ft_11 = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

diff_control = ft_01 - ft_00
diff_treated = ft_11 - ft_10
did_simple = diff_treated - diff_control

print("\nUnweighted DiD Calculation:")
print(f"  Control pre (ELIGIBLE=0, AFTER=0):  {ft_00:.4f}")
print(f"  Control post (ELIGIBLE=0, AFTER=1): {ft_01:.4f}")
print(f"  Control change: {diff_control:.4f}")
print(f"  Treated pre (ELIGIBLE=1, AFTER=0):  {ft_10:.4f}")
print(f"  Treated post (ELIGIBLE=1, AFTER=1): {ft_11:.4f}")
print(f"  Treated change: {diff_treated:.4f}")
print(f"  DiD estimate: {did_simple:.4f}")

# Weighted DiD
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

ft_00_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)], 'FT', 'PERWT')
ft_01_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)], 'FT', 'PERWT')
ft_10_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)], 'FT', 'PERWT')
ft_11_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)], 'FT', 'PERWT')

diff_control_w = ft_01_w - ft_00_w
diff_treated_w = ft_11_w - ft_10_w
did_weighted = diff_treated_w - diff_control_w

print("\nWeighted DiD Calculation (using PERWT):")
print(f"  Control pre:  {ft_00_w:.4f}")
print(f"  Control post: {ft_01_w:.4f}")
print(f"  Control change: {diff_control_w:.4f}")
print(f"  Treated pre:  {ft_10_w:.4f}")
print(f"  Treated post: {ft_11_w:.4f}")
print(f"  Treated change: {diff_treated_w:.4f}")
print(f"  DiD estimate: {did_weighted:.4f}")

# ============================================
# SECTION 3: REGRESSION-BASED DID ANALYSIS
# ============================================
print("\n" + "="*60)
print("SECTION 3: REGRESSION-BASED DID ANALYSIS")
print("="*60)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (unweighted) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model1.summary())

# Model 2: Basic DiD (weighted)
print("\n--- Model 2: Basic DiD (weighted with PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: With year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects (weighted) ---")
df['YEAR_factor'] = df['YEAR'].astype(str)
model3 = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# ============================================
# SECTION 4: DID WITH COVARIATES
# ============================================
print("\n" + "="*60)
print("SECTION 4: DID WITH COVARIATES")
print("="*60)

# Create dummy variables for education
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_2YR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Create female dummy (SEX: 1=Male, 2=Female)
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Create married dummy (MARST: 1=Married spouse present)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Create has children dummy
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Model 4: With individual-level covariates
print("\n--- Model 4: DiD with Individual Covariates (weighted) ---")
model4 = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_SOMECOL + EDUC_2YR + EDUC_BA',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# Model 5: Add state fixed effects
print("\n--- Model 5: DiD with State Fixed Effects (weighted) ---")
df['STATE_factor'] = df['STATEFIP'].astype(str)
model5 = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_SOMECOL + EDUC_2YR + EDUC_BA + C(STATE_factor)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE (HC1): {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"N = {model5.nobs:.0f}")

# ============================================
# SECTION 5: ROBUSTNESS CHECKS
# ============================================
print("\n" + "="*60)
print("SECTION 5: ROBUSTNESS CHECKS")
print("="*60)

# 5a: Clustered standard errors at state level
print("\n--- 5a: State-Clustered Standard Errors ---")
model5_cluster = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_SOMECOL + EDUC_2YR + EDUC_BA + C(STATE_factor)',
                         data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD coefficient: {model5_cluster.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {model5_cluster.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5_cluster.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5_cluster.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model5_cluster.pvalues['ELIGIBLE_AFTER']:.4f}")

# 5b: By gender
print("\n--- 5b: Heterogeneity by Gender ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex].copy()
    model_sex = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_AFTER + MARRIED + HAS_CHILDREN + EDUC_SOMECOL + EDUC_2YR + EDUC_BA + C(STATE_factor)',
                        data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_sex.params['ELIGIBLE_AFTER']:.4f}, SE = {model_sex.bse['ELIGIBLE_AFTER']:.4f}, N = {len(subset)}")

# 5c: Year-by-year effects (event study)
print("\n--- 5c: Event Study (Year-by-Year Effects) ---")
df['ELIGIBLE_2008'] = df['ELIGIBLE'] * (df['YEAR'] == 2008).astype(int)
df['ELIGIBLE_2009'] = df['ELIGIBLE'] * (df['YEAR'] == 2009).astype(int)
df['ELIGIBLE_2010'] = df['ELIGIBLE'] * (df['YEAR'] == 2010).astype(int)
# 2011 is reference year
df['ELIGIBLE_2013'] = df['ELIGIBLE'] * (df['YEAR'] == 2013).astype(int)
df['ELIGIBLE_2014'] = df['ELIGIBLE'] * (df['YEAR'] == 2014).astype(int)
df['ELIGIBLE_2015'] = df['ELIGIBLE'] * (df['YEAR'] == 2015).astype(int)
df['ELIGIBLE_2016'] = df['ELIGIBLE'] * (df['YEAR'] == 2016).astype(int)

model_event = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_2008 + ELIGIBLE_2009 + ELIGIBLE_2010 + ELIGIBLE_2013 + ELIGIBLE_2014 + ELIGIBLE_2015 + ELIGIBLE_2016 + FEMALE + MARRIED + HAS_CHILDREN + EDUC_SOMECOL + EDUC_2YR + EDUC_BA + C(STATE_factor)',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("Event Study Coefficients (relative to 2011):")
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'ELIGIBLE_{year}'
    print(f"  {year}: {model_event.params[var]:.4f} (SE: {model_event.bse[var]:.4f})")

# ============================================
# SECTION 6: BALANCE CHECKS (PRE-TREATMENT)
# ============================================
print("\n" + "="*60)
print("SECTION 6: BALANCE CHECKS (PRE-TREATMENT)")
print("="*60)

pre_df = df[df['AFTER'] == 0].copy()

balance_vars = ['FEMALE', 'MARRIED', 'HAS_CHILDREN', 'EDUC_HS', 'EDUC_SOMECOL', 'EDUC_2YR', 'EDUC_BA', 'AGE']

print("\nPre-treatment covariate balance:")
print(f"{'Variable':<20} {'Control':<12} {'Treated':<12} {'Diff':<12} {'p-value':<10}")
print("-" * 66)

for var in balance_vars:
    control_mean = pre_df[pre_df['ELIGIBLE']==0][var].mean()
    treated_mean = pre_df[pre_df['ELIGIBLE']==1][var].mean()
    diff = treated_mean - control_mean
    # t-test
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(pre_df[pre_df['ELIGIBLE']==0][var],
                                     pre_df[pre_df['ELIGIBLE']==1][var])
    print(f"{var:<20} {control_mean:<12.4f} {treated_mean:<12.4f} {diff:<12.4f} {p_val:<10.4f}")

# ============================================
# SECTION 7: PARALLEL TRENDS TEST
# ============================================
print("\n" + "="*60)
print("SECTION 7: PARALLEL TRENDS TEST")
print("="*60)

# Test for differential pre-trends
pre_df = df[df['AFTER'] == 0].copy()
pre_df['YEAR_TREND'] = pre_df['YEAR'] - 2008
pre_df['ELIGIBLE_TREND'] = pre_df['ELIGIBLE'] * pre_df['YEAR_TREND']

model_pretrend = smf.wls('FT ~ ELIGIBLE + YEAR_TREND + ELIGIBLE_TREND + FEMALE + MARRIED + HAS_CHILDREN + EDUC_SOMECOL + EDUC_2YR + EDUC_BA',
                         data=pre_df, weights=pre_df['PERWT']).fit(cov_type='HC1')
print("\nPre-treatment Differential Trend Test:")
print(f"ELIGIBLE_TREND coefficient: {model_pretrend.params['ELIGIBLE_TREND']:.4f}")
print(f"SE: {model_pretrend.bse['ELIGIBLE_TREND']:.4f}")
print(f"p-value: {model_pretrend.pvalues['ELIGIBLE_TREND']:.4f}")
print("(Null hypothesis: parallel pre-trends)")

# ============================================
# SECTION 8: SUMMARY OF RESULTS
# ============================================
print("\n" + "="*60)
print("SECTION 8: SUMMARY OF MAIN RESULTS")
print("="*60)

print("\n*** PREFERRED SPECIFICATION (Model 5 with State Clusters) ***")
print(f"Effect of DACA eligibility on full-time employment:")
print(f"  Point estimate: {model5_cluster.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Robust SE (state-clustered): {model5_cluster.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model5_cluster.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5_cluster.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  p-value: {model5_cluster.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  Sample size: {int(model5_cluster.nobs)}")

print("\nInterpretation:")
effect = model5_cluster.params['ELIGIBLE_AFTER']
if effect > 0:
    print(f"  DACA eligibility is associated with a {effect*100:.2f} percentage point")
    print(f"  increase in the probability of full-time employment.")
else:
    print(f"  DACA eligibility is associated with a {abs(effect)*100:.2f} percentage point")
    print(f"  decrease in the probability of full-time employment.")

if model5_cluster.pvalues['ELIGIBLE_AFTER'] < 0.05:
    print("  This effect is statistically significant at the 5% level.")
elif model5_cluster.pvalues['ELIGIBLE_AFTER'] < 0.10:
    print("  This effect is marginally significant at the 10% level.")
else:
    print("  This effect is NOT statistically significant at conventional levels.")

# ============================================
# SECTION 9: EXPORT RESULTS FOR LATEX
# ============================================
print("\n" + "="*60)
print("SECTION 9: EXPORT RESULTS")
print("="*60)

# Save key results to a file
results = {
    'Model': ['Basic DiD', 'Weighted DiD', 'Year FE', 'Covariates', 'State FE (HC1)', 'State FE (Clustered)'],
    'Coefficient': [
        model1.params['ELIGIBLE_AFTER'],
        model2.params['ELIGIBLE_AFTER'],
        model3.params['ELIGIBLE_AFTER'],
        model4.params['ELIGIBLE_AFTER'],
        model5.params['ELIGIBLE_AFTER'],
        model5_cluster.params['ELIGIBLE_AFTER']
    ],
    'SE': [
        model1.bse['ELIGIBLE_AFTER'],
        model2.bse['ELIGIBLE_AFTER'],
        model3.bse['ELIGIBLE_AFTER'],
        model4.bse['ELIGIBLE_AFTER'],
        model5.bse['ELIGIBLE_AFTER'],
        model5_cluster.bse['ELIGIBLE_AFTER']
    ],
    'p_value': [
        model1.pvalues['ELIGIBLE_AFTER'],
        model2.pvalues['ELIGIBLE_AFTER'],
        model3.pvalues['ELIGIBLE_AFTER'],
        model4.pvalues['ELIGIBLE_AFTER'],
        model5.pvalues['ELIGIBLE_AFTER'],
        model5_cluster.pvalues['ELIGIBLE_AFTER']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model5.nobs),
        int(model5_cluster.nobs)
    ]
}

results_df = pd.DataFrame(results)
results_df.to_csv('did_results.csv', index=False)
print("\nResults saved to did_results.csv")

# Save event study results
event_results = {
    'Year': [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [
        model_event.params['ELIGIBLE_2008'],
        model_event.params['ELIGIBLE_2009'],
        model_event.params['ELIGIBLE_2010'],
        0,  # Reference year
        model_event.params['ELIGIBLE_2013'],
        model_event.params['ELIGIBLE_2014'],
        model_event.params['ELIGIBLE_2015'],
        model_event.params['ELIGIBLE_2016']
    ],
    'SE': [
        model_event.bse['ELIGIBLE_2008'],
        model_event.bse['ELIGIBLE_2009'],
        model_event.bse['ELIGIBLE_2010'],
        0,  # Reference year
        model_event.bse['ELIGIBLE_2013'],
        model_event.bse['ELIGIBLE_2014'],
        model_event.bse['ELIGIBLE_2015'],
        model_event.bse['ELIGIBLE_2016']
    ]
}
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save descriptive statistics
desc_stats = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved to descriptive_stats.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
