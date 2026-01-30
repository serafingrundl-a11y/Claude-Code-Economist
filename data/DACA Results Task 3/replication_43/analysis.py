"""
DACA Replication Analysis - Task 43
Difference-in-Differences estimation of DACA eligibility on full-time employment

Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States?
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
print("DACA REPLICATION ANALYSIS - TASK 43")
print("="*80)

# Load the data
print("\n--- Loading Data ---")
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations: {len(df):,}")
print(f"Number of variables: {len(df.columns)}")

# Basic data inspection
print("\n--- Data Structure ---")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"ELIGIBLE values: {df['ELIGIBLE'].unique()}")
print(f"AFTER values: {df['AFTER'].unique()}")
print(f"FT values: {df['FT'].unique()}")

# Check key variable distributions
print("\n--- Key Variable Distributions ---")
print(f"\nELIGIBLE (Treatment vs Control):")
print(df['ELIGIBLE'].value_counts().sort_index())
print(f"\nAFTER (Pre vs Post):")
print(df['AFTER'].value_counts().sort_index())
print(f"\nFT (Full-time employment):")
print(df['FT'].value_counts().sort_index())

# Check years by AFTER
print("\n--- Years by AFTER indicator ---")
print(df.groupby('AFTER')['YEAR'].value_counts().sort_index())

# Examine treatment and control group sizes by period
print("\n--- Sample Size by Group and Period ---")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (31-35)', 'Treated (26-30)', 'Total']
crosstab.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)', 'Total']
print(crosstab)

# Calculate mean FT by group and period
print("\n--- Full-Time Employment Rates by Group and Period ---")
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
ft_means.index = pd.MultiIndex.from_tuples([
    ('Control (31-35)', 'Pre'), ('Control (31-35)', 'Post'),
    ('Treated (26-30)', 'Pre'), ('Treated (26-30)', 'Post')
])
print(ft_means.round(4))

# Calculate simple DiD manually
print("\n--- Simple Difference-in-Differences Calculation ---")
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

diff_control = post_control - pre_control
diff_treat = post_treat - pre_treat
did_simple = diff_treat - diff_control

print(f"Control group (ages 31-35):")
print(f"  Pre-DACA FT rate:  {pre_control:.4f}")
print(f"  Post-DACA FT rate: {post_control:.4f}")
print(f"  Change:            {diff_control:.4f}")
print(f"\nTreatment group (ages 26-30):")
print(f"  Pre-DACA FT rate:  {pre_treat:.4f}")
print(f"  Post-DACA FT rate: {post_treat:.4f}")
print(f"  Change:            {diff_treat:.4f}")
print(f"\nDifference-in-Differences: {did_simple:.4f}")

# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("REGRESSION ANALYSIS")
print("="*80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no weights, no controls)
print("\n--- Model 1: Basic DiD (No Weights, No Controls) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(f"DiD Coefficient: {model1.params['ELIGIBLE_AFTER']:.6f}")
print(f"Std Error:       {model1.bse['ELIGIBLE_AFTER']:.6f}")
print(f"t-statistic:     {model1.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model1.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI:          [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")
print(f"N:               {int(model1.nobs):,}")
print(f"R-squared:       {model1.rsquared:.6f}")

# Model 2: Basic DiD with survey weights
print("\n--- Model 2: Basic DiD (With Survey Weights) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model2.params['ELIGIBLE_AFTER']:.6f}")
print(f"Std Error:       {model2.bse['ELIGIBLE_AFTER']:.6f}")
print(f"t-statistic:     {model2.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model2.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI:          [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")
print(f"N:               {int(model2.nobs):,}")
print(f"R-squared:       {model2.rsquared:.6f}")

# Prepare control variables
print("\n--- Preparing Control Variables ---")

# Recode SEX to 0/1 (1=Male in IPUMS, 2=Female)
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# MARST: Create married indicator (1=Married spouse present)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Education categories (using EDUC_RECODE if available)
print(f"Education categories: {df['EDUC_RECODE'].unique()}")

# Create education dummies
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_AA'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)
# Reference: Less than High School

# Number of children
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Metropolitan status
df['METRO_AREA'] = (df['METRO'] >= 2).astype(int)  # 2,3,4 are in metro areas

# Check for state policy variables
state_policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID',
                     'HIGHEREDBAN', 'EVERIFY', 'LIMITEVERIFY', 'OMNIBUS',
                     'TASK287G', 'JAIL287G', 'SECURECOMMUNITIES']
print(f"\nState policy variables available: {[v for v in state_policy_vars if v in df.columns]}")

# Model 3: DiD with demographic controls (weighted)
print("\n--- Model 3: DiD with Demographic Controls (Weighted) ---")
formula3 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + METRO_AREA'
model3 = smf.wls(formula3, data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model3.params['ELIGIBLE_AFTER']:.6f}")
print(f"Std Error:       {model3.bse['ELIGIBLE_AFTER']:.6f}")
print(f"t-statistic:     {model3.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model3.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI:          [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")
print(f"N:               {int(model3.nobs):,}")
print(f"R-squared:       {model3.rsquared:.6f}")

# Model 4: DiD with demographic controls + state fixed effects (weighted)
print("\n--- Model 4: DiD with Demographic Controls + State FE (Weighted) ---")
formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + METRO_AREA + C(STATEFIP)'
model4 = smf.wls(formula4, data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model4.params['ELIGIBLE_AFTER']:.6f}")
print(f"Std Error:       {model4.bse['ELIGIBLE_AFTER']:.6f}")
print(f"t-statistic:     {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model4.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI:          [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")
print(f"N:               {int(model4.nobs):,}")
print(f"R-squared:       {model4.rsquared:.6f}")

# Model 5: DiD with demographic controls + year fixed effects (weighted)
print("\n--- Model 5: DiD with Demographic Controls + Year FE (Weighted) ---")
formula5 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + METRO_AREA + C(YEAR)'
model5 = smf.wls(formula5, data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model5.params['ELIGIBLE_AFTER']:.6f}")
print(f"Std Error:       {model5.bse['ELIGIBLE_AFTER']:.6f}")
print(f"t-statistic:     {model5.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model5.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI:          [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")
print(f"N:               {int(model5.nobs):,}")
print(f"R-squared:       {model5.rsquared:.6f}")

# Model 6: PREFERRED SPECIFICATION - Full model with state and year FE
print("\n--- Model 6 (PREFERRED): DiD with Controls + State & Year FE (Weighted) ---")
formula6 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + METRO_AREA + C(STATEFIP) + C(YEAR)'
model6 = smf.wls(formula6, data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model6.params['ELIGIBLE_AFTER']:.6f}")
print(f"Std Error:       {model6.bse['ELIGIBLE_AFTER']:.6f}")
print(f"t-statistic:     {model6.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model6.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI:          [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")
print(f"N:               {int(model6.nobs):,}")
print(f"R-squared:       {model6.rsquared:.6f}")

# ============================================================================
# ROBUSTNESS CHECKS
# ============================================================================

print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Heteroskedasticity-robust standard errors for preferred model
print("\n--- Model 6 with Robust (HC1) Standard Errors ---")
model6_robust = model6.get_robustcov_results(cov_type='HC1')
# Find the index for ELIGIBLE_AFTER
ea_idx = list(model6.params.index).index('ELIGIBLE_AFTER')
print(f"DiD Coefficient: {model6_robust.params[ea_idx]:.6f}")
print(f"Robust SE:       {model6_robust.bse[ea_idx]:.6f}")
print(f"t-statistic:     {model6_robust.tvalues[ea_idx]:.4f}")
print(f"p-value:         {model6_robust.pvalues[ea_idx]:.6f}")
robust_ci = model6_robust.conf_int()[ea_idx]
print(f"95% CI:          [{robust_ci[0]:.6f}, {robust_ci[1]:.6f}]")

# State-clustered standard errors
print("\n--- Model 6 with State-Clustered Standard Errors ---")
model6_cluster = model6.get_robustcov_results(cov_type='cluster', groups=df['STATEFIP'])
print(f"DiD Coefficient: {model6_cluster.params[ea_idx]:.6f}")
print(f"Clustered SE:    {model6_cluster.bse[ea_idx]:.6f}")
print(f"t-statistic:     {model6_cluster.tvalues[ea_idx]:.4f}")
print(f"p-value:         {model6_cluster.pvalues[ea_idx]:.6f}")
cluster_ci = model6_cluster.conf_int()[ea_idx]
print(f"95% CI:          [{cluster_ci[0]:.6f}, {cluster_ci[1]:.6f}]")

# Subgroup analysis by sex
print("\n--- Subgroup Analysis by Sex ---")
print("\nMales only:")
df_male = df[df['SEX'] == 1]
model_male = smf.wls(formula6, data=df_male, weights=df_male['PERWT']).fit()
print(f"DiD Coefficient: {model_male.params['ELIGIBLE_AFTER']:.6f}")
print(f"Std Error:       {model_male.bse['ELIGIBLE_AFTER']:.6f}")
print(f"N:               {int(model_male.nobs):,}")

print("\nFemales only:")
df_female = df[df['SEX'] == 2]
model_female = smf.wls(formula6, data=df_female, weights=df_female['PERWT']).fit()
print(f"DiD Coefficient: {model_female.params['ELIGIBLE_AFTER']:.6f}")
print(f"Std Error:       {model_female.bse['ELIGIBLE_AFTER']:.6f}")
print(f"N:               {int(model_female.nobs):,}")

# ============================================================================
# EVENT STUDY / PARALLEL TRENDS CHECK
# ============================================================================

print("\n" + "="*80)
print("EVENT STUDY ANALYSIS (Parallel Trends Check)")
print("="*80)

# Create year-by-eligible interactions (reference: 2011)
years = sorted(df['YEAR'].unique())
print(f"Years: {years}")

for year in years:
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Event study regression (omit 2011 as reference year)
year_interactions = ' + '.join([f'ELIGIBLE_YEAR_{y}' for y in years if y != 2011])
formula_event = f'FT ~ ELIGIBLE + {year_interactions} + C(YEAR) + FEMALE + MARRIED + AGE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + METRO_AREA + C(STATEFIP)'
model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit()

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
for year in years:
    if year != 2011:
        coef = model_event.params[f'ELIGIBLE_YEAR_{year}']
        se = model_event.bse[f'ELIGIBLE_YEAR_{year}']
        pval = model_event.pvalues[f'ELIGIBLE_YEAR_{year}']
        ci_low, ci_high = model_event.conf_int().loc[f'ELIGIBLE_YEAR_{year}']
        period = "Pre-DACA" if year < 2012 else "Post-DACA"
        print(f"{year} ({period}): {coef:8.4f} (SE: {se:.4f}) [{ci_low:.4f}, {ci_high:.4f}] p={pval:.4f}")

# ============================================================================
# PLACEBO TEST
# ============================================================================

print("\n" + "="*80)
print("PLACEBO TEST (Pre-treatment periods only)")
print("="*80)

# Use only pre-treatment data, with placebo treatment at 2010
df_pre = df[df['AFTER'] == 0].copy()
df_pre['PLACEBO_POST'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['ELIGIBLE_PLACEBO'] = df_pre['ELIGIBLE'] * df_pre['PLACEBO_POST']

formula_placebo = 'FT ~ ELIGIBLE + PLACEBO_POST + ELIGIBLE_PLACEBO + FEMALE + MARRIED + AGE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + METRO_AREA + C(STATEFIP)'
model_placebo = smf.wls(formula_placebo, data=df_pre, weights=df_pre['PERWT']).fit()
print(f"Placebo DiD (2010-2011 vs 2008-2009):")
print(f"Coefficient: {model_placebo.params['ELIGIBLE_PLACEBO']:.6f}")
print(f"Std Error:   {model_placebo.bse['ELIGIBLE_PLACEBO']:.6f}")
print(f"p-value:     {model_placebo.pvalues['ELIGIBLE_PLACEBO']:.6f}")
print(f"N:           {int(model_placebo.nobs):,}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Summary statistics for key variables by treatment status
print("\n--- Summary Statistics by Treatment Group (Pre-DACA Period) ---")
df_pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]
df_pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]

summary_vars = ['FT', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD', 'METRO_AREA']
summary_data = []
for var in summary_vars:
    control_mean = df_pre_control[var].mean()
    treat_mean = df_pre_treat[var].mean()
    diff = treat_mean - control_mean
    summary_data.append([var, control_mean, treat_mean, diff])

summary_df = pd.DataFrame(summary_data, columns=['Variable', 'Control (31-35)', 'Treated (26-30)', 'Difference'])
print(summary_df.to_string(index=False))

# Education distribution
print("\n--- Education Distribution by Treatment Group (Pre-DACA) ---")
for group, label in [(0, 'Control (31-35)'), (1, 'Treated (26-30)')]:
    df_group = df[(df['ELIGIBLE']==group) & (df['AFTER']==0)]
    print(f"\n{label}:")
    print(df_group['EDUC_RECODE'].value_counts(normalize=True).round(4))

# ============================================================================
# FULL REGRESSION TABLE
# ============================================================================

print("\n" + "="*80)
print("FULL REGRESSION TABLE (Preferred Model)")
print("="*80)

# Print full model 6 results
print("\nModel 6 - Full Results:")
print("-" * 60)
main_vars = ['Intercept', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'AGE', 'NCHILD', 'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_AA', 'EDUC_BA', 'METRO_AREA']
for var in main_vars:
    if var in model6.params:
        coef = model6.params[var]
        se = model6.bse[var]
        pval = model6.pvalues[var]
        stars = ''
        if pval < 0.01:
            stars = '***'
        elif pval < 0.05:
            stars = '**'
        elif pval < 0.1:
            stars = '*'
        print(f"{var:20s}: {coef:10.6f} ({se:.6f}){stars}")

# ============================================================================
# EXPORT RESULTS FOR LATEX
# ============================================================================

print("\n" + "="*80)
print("RESULTS FOR LATEX TABLES")
print("="*80)

# Store results for LaTeX
results_dict = {
    'Model 1 (Basic)': {
        'coef': model1.params['ELIGIBLE_AFTER'],
        'se': model1.bse['ELIGIBLE_AFTER'],
        'pval': model1.pvalues['ELIGIBLE_AFTER'],
        'n': int(model1.nobs),
        'r2': model1.rsquared,
        'controls': 'No',
        'weights': 'No',
        'state_fe': 'No',
        'year_fe': 'No'
    },
    'Model 2 (Weighted)': {
        'coef': model2.params['ELIGIBLE_AFTER'],
        'se': model2.bse['ELIGIBLE_AFTER'],
        'pval': model2.pvalues['ELIGIBLE_AFTER'],
        'n': int(model2.nobs),
        'r2': model2.rsquared,
        'controls': 'No',
        'weights': 'Yes',
        'state_fe': 'No',
        'year_fe': 'No'
    },
    'Model 3 (Controls)': {
        'coef': model3.params['ELIGIBLE_AFTER'],
        'se': model3.bse['ELIGIBLE_AFTER'],
        'pval': model3.pvalues['ELIGIBLE_AFTER'],
        'n': int(model3.nobs),
        'r2': model3.rsquared,
        'controls': 'Yes',
        'weights': 'Yes',
        'state_fe': 'No',
        'year_fe': 'No'
    },
    'Model 4 (State FE)': {
        'coef': model4.params['ELIGIBLE_AFTER'],
        'se': model4.bse['ELIGIBLE_AFTER'],
        'pval': model4.pvalues['ELIGIBLE_AFTER'],
        'n': int(model4.nobs),
        'r2': model4.rsquared,
        'controls': 'Yes',
        'weights': 'Yes',
        'state_fe': 'Yes',
        'year_fe': 'No'
    },
    'Model 5 (Year FE)': {
        'coef': model5.params['ELIGIBLE_AFTER'],
        'se': model5.bse['ELIGIBLE_AFTER'],
        'pval': model5.pvalues['ELIGIBLE_AFTER'],
        'n': int(model5.nobs),
        'r2': model5.rsquared,
        'controls': 'Yes',
        'weights': 'Yes',
        'state_fe': 'No',
        'year_fe': 'Yes'
    },
    'Model 6 (Full)': {
        'coef': model6.params['ELIGIBLE_AFTER'],
        'se': model6.bse['ELIGIBLE_AFTER'],
        'pval': model6.pvalues['ELIGIBLE_AFTER'],
        'n': int(model6.nobs),
        'r2': model6.rsquared,
        'controls': 'Yes',
        'weights': 'Yes',
        'state_fe': 'Yes',
        'year_fe': 'Yes'
    }
}

# Save results to CSV for LaTeX import
results_list = []
for model_name, res in results_dict.items():
    results_list.append({
        'Model': model_name,
        'Coefficient': res['coef'],
        'Std_Error': res['se'],
        'p_value': res['pval'],
        'N': res['n'],
        'R_squared': res['r2'],
        'Controls': res['controls'],
        'Weights': res['weights'],
        'State_FE': res['state_fe'],
        'Year_FE': res['year_fe']
    })

results_df = pd.DataFrame(results_list)
results_df.to_csv('regression_results.csv', index=False)
print("\nRegression results saved to 'regression_results.csv'")

# Event study results
event_results = []
for year in years:
    if year != 2011:
        event_results.append({
            'Year': year,
            'Coefficient': model_event.params[f'ELIGIBLE_YEAR_{year}'],
            'Std_Error': model_event.bse[f'ELIGIBLE_YEAR_{year}'],
            'p_value': model_event.pvalues[f'ELIGIBLE_YEAR_{year}'],
            'CI_low': model_event.conf_int().loc[f'ELIGIBLE_YEAR_{year}', 0],
            'CI_high': model_event.conf_int().loc[f'ELIGIBLE_YEAR_{year}', 1]
        })
    else:
        event_results.append({
            'Year': year,
            'Coefficient': 0,
            'Std_Error': 0,
            'p_value': np.nan,
            'CI_low': 0,
            'CI_high': 0
        })

event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to 'event_study_results.csv'")

# Summary statistics
summary_stats = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'FEMALE': 'mean',
    'MARRIED': 'mean',
    'NCHILD': 'mean'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')
print("Summary statistics saved to 'summary_statistics.csv'")

print("\n" + "="*80)
print("PREFERRED ESTIMATE SUMMARY")
print("="*80)
print(f"\nPreferred Specification: Model 6")
print(f"  - Difference-in-Differences with demographic controls")
print(f"  - State and year fixed effects")
print(f"  - Survey weights (PERWT)")
print(f"\nEffect of DACA eligibility on full-time employment:")
print(f"  Coefficient:  {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Std Error:    {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI:       [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  p-value:      {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  Sample size:  {int(model6.nobs):,}")
print(f"\nInterpretation: DACA eligibility is associated with a")
print(f"  {model6.params['ELIGIBLE_AFTER']*100:.2f} percentage point change in the probability")
print(f"  of full-time employment (relative to the control group trend).")

# Calculate percentage change from baseline
baseline_ft = pre_treat
pct_change = (model6.params['ELIGIBLE_AFTER'] / baseline_ft) * 100
print(f"\n  This represents approximately a {pct_change:.1f}% change relative to")
print(f"  the pre-DACA treatment group mean of {baseline_ft*100:.1f}%.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
