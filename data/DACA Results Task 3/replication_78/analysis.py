"""
DACA Replication Study - Analysis Script
Research Question: Impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US.

Method: Difference-in-Differences (DiD)
Treatment: Ages 26-30 at June 15, 2012 (ELIGIBLE=1)
Control: Ages 31-35 at June 15, 2012 (ELIGIBLE=0)
Pre-period: 2008-2011
Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
import warnings
import json
warnings.filterwarnings('ignore')

# Set output path
output_path = r'C:\Users\seraf\DACA Results Task 3\replication_78'

print("="*80)
print("DACA REPLICATION STUDY - FULL ANALYSIS")
print("="*80)

# ============================================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: DATA LOADING AND INITIAL EXPLORATION")
print("="*80)

# Load data
df = pd.read_csv(f'{output_path}/data/prepared_data_numeric_version.csv')

print(f"\nDataset loaded successfully")
print(f"Total observations: {len(df):,}")
print(f"Total variables: {len(df.columns)}")

# Basic structure
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")
print(f"Year distribution:")
print(df['YEAR'].value_counts().sort_index())

# Key variables check
print(f"\n--- Key Variables ---")
print(f"FT (Full-time employment): {df['FT'].value_counts().sort_index().to_dict()}")
print(f"ELIGIBLE (Treatment indicator): {df['ELIGIBLE'].value_counts().sort_index().to_dict()}")
print(f"AFTER (Post-DACA indicator): {df['AFTER'].value_counts().sort_index().to_dict()}")

# ============================================================================
# SECTION 2: SAMPLE DESCRIPTION
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: SAMPLE DESCRIPTION")
print("="*80)

# Group sizes
print("\n--- Sample Sizes by Group ---")
group_sizes = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
group_sizes.index = ['Control (31-35)', 'Treatment (26-30)']
group_sizes.columns = ['Pre-DACA', 'Post-DACA']
print(group_sizes)
print(f"\nTotal sample: {len(df):,}")

# Weighted sample sizes (sum of person weights)
print("\n--- Weighted Population Counts ---")
weighted_pop = df.groupby(['ELIGIBLE', 'AFTER'])['PERWT'].sum().unstack()
weighted_pop.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_pop.columns = ['Pre-DACA', 'Post-DACA']
print(weighted_pop.round(0))

# ============================================================================
# SECTION 3: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: DESCRIPTIVE STATISTICS")
print("="*80)

def weighted_mean(data, weights):
    """Calculate weighted mean"""
    return np.average(data, weights=weights)

def weighted_std(data, weights):
    """Calculate weighted standard deviation"""
    avg = np.average(data, weights=weights)
    variance = np.average((data - avg)**2, weights=weights)
    return np.sqrt(variance)

# Demographics by group
print("\n--- Demographic Characteristics by Treatment Status (Pre-DACA period) ---")

pre_daca = df[df['AFTER'] == 0]
treatment_pre = pre_daca[pre_daca['ELIGIBLE'] == 1]
control_pre = pre_daca[pre_daca['ELIGIBLE'] == 0]

demographics = {}
for var in ['AGE', 'SEX', 'MARST', 'FAMSIZE', 'NCHILD']:
    if var in df.columns:
        t_mean = weighted_mean(treatment_pre[var].dropna(), treatment_pre.loc[treatment_pre[var].notna(), 'PERWT'])
        c_mean = weighted_mean(control_pre[var].dropna(), control_pre.loc[control_pre[var].notna(), 'PERWT'])
        demographics[var] = {'Treatment': t_mean, 'Control': c_mean}

demo_df = pd.DataFrame(demographics).T
demo_df.columns = ['Treatment (26-30)', 'Control (31-35)']
print(demo_df.round(3))

# Education distribution
print("\n--- Education Distribution (Pre-DACA, Weighted) ---")
if 'EDUC_RECODE' in df.columns:
    # Load labelled version for education labels
    df_lab = pd.read_csv(f'{output_path}/data/prepared_data_labelled_version.csv')
    educ_cats = df_lab['EDUC_RECODE'].unique()
    print("Education categories in data:", educ_cats)

# Full-time employment rates over time
print("\n--- Full-Time Employment Rates by Year and Group ---")
ft_by_year = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
ft_by_year.columns = ['Control (31-35)', 'Treatment (26-30)']
print((ft_by_year * 100).round(2))

# ============================================================================
# SECTION 4: MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# 4.1 Simple 2x2 DiD (unweighted)
print("\n--- 4.1 Simple 2x2 DiD Table (Unweighted Means) ---")
did_table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
did_table.index = ['Control (31-35)', 'Treatment (26-30)']
did_table.columns = ['Pre-DACA', 'Post-DACA']
did_table['Difference'] = did_table['Post-DACA'] - did_table['Pre-DACA']
print((did_table * 100).round(3))

diff_control = did_table.loc['Control (31-35)', 'Difference']
diff_treatment = did_table.loc['Treatment (26-30)', 'Difference']
simple_did = diff_treatment - diff_control
print(f"\nSimple DiD estimate: {simple_did*100:.3f} percentage points")

# 4.2 Weighted 2x2 DiD
print("\n--- 4.2 Weighted 2x2 DiD Table ---")
did_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
did_weighted.index = ['Control (31-35)', 'Treatment (26-30)']
did_weighted.columns = ['Pre-DACA', 'Post-DACA']
did_weighted['Difference'] = did_weighted['Post-DACA'] - did_weighted['Pre-DACA']
print((did_weighted * 100).round(3))

diff_control_w = did_weighted.loc['Control (31-35)', 'Difference']
diff_treatment_w = did_weighted.loc['Treatment (26-30)', 'Difference']
weighted_did = diff_treatment_w - diff_control_w
print(f"\nWeighted DiD estimate: {weighted_did*100:.3f} percentage points")

# 4.3 Regression-based DiD (OLS without covariates)
print("\n--- 4.3 Regression-Based DiD (OLS, no covariates) ---")

# Unweighted OLS
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print("\nUnweighted OLS:")
print(f"DiD coefficient (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.5f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_AFTER']:.5f}")
print(f"t-statistic: {model1.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")

# Weighted OLS
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print("\nWeighted OLS:")
print(f"DiD coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.5f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_AFTER']:.5f}")
print(f"t-statistic: {model2.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")

# 4.4 Regression with robust standard errors
print("\n--- 4.4 Weighted OLS with Heteroskedasticity-Robust Standard Errors ---")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model3.params['ELIGIBLE_AFTER']:.5f}")
print(f"Robust SE: {model3.bse['ELIGIBLE_AFTER']:.5f}")
print(f"t-statistic: {model3.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")

# ============================================================================
# SECTION 5: EXTENDED MODELS WITH COVARIATES
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: EXTENDED MODELS WITH COVARIATES")
print("="*80)

# 5.1 Model with demographic controls
print("\n--- 5.1 DiD with Demographic Controls ---")
# Create dummy variables for categorical controls
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Check which variables are available and have variation
available_controls = []
for var in ['FEMALE', 'MARRIED', 'FAMSIZE', 'NCHILD']:
    if var in df.columns and df[var].std() > 0:
        available_controls.append(var)

print(f"Available control variables: {available_controls}")

formula_demo = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD'
model4 = smf.wls(formula_demo, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"\nDiD coefficient: {model4.params['ELIGIBLE_AFTER']:.5f}")
print(f"Robust SE: {model4.bse['ELIGIBLE_AFTER']:.5f}")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"Observations: {int(model4.nobs):,}")

# 5.2 Model with state fixed effects
print("\n--- 5.2 DiD with State Fixed Effects ---")
df['STATE_FE'] = df['STATEFIP'].astype('category')

# Create state dummies manually for explicit control
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
df_with_states = pd.concat([df, state_dummies], axis=1)

formula_state = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + C(STATEFIP)'
try:
    model5 = smf.wls(formula_state, data=df, weights=df['PERWT']).fit(cov_type='HC1')
    print(f"DiD coefficient: {model5.params['ELIGIBLE_AFTER']:.5f}")
    print(f"Robust SE: {model5.bse['ELIGIBLE_AFTER']:.5f}")
    print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
    print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")
except Exception as e:
    print(f"State FE model error: {e}")

# 5.3 Model with year fixed effects
print("\n--- 5.3 DiD with Year Fixed Effects ---")
formula_year = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + C(YEAR)'
model6 = smf.wls(formula_year, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model6.params['ELIGIBLE_AFTER']:.5f}")
print(f"Robust SE: {model6.bse['ELIGIBLE_AFTER']:.5f}")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")

# 5.4 Full model with state and year FE
print("\n--- 5.4 Full Model: State FE + Year FE + Demographics ---")
formula_full = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + C(YEAR) + C(STATEFIP)'
try:
    model7 = smf.wls(formula_full, data=df, weights=df['PERWT']).fit(cov_type='HC1')
    print(f"DiD coefficient: {model7.params['ELIGIBLE_AFTER']:.5f}")
    print(f"Robust SE: {model7.bse['ELIGIBLE_AFTER']:.5f}")
    print(f"p-value: {model7.pvalues['ELIGIBLE_AFTER']:.4f}")
    print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")
    print(f"R-squared: {model7.rsquared:.4f}")
except Exception as e:
    print(f"Full model error: {e}")

# ============================================================================
# SECTION 6: ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: ROBUSTNESS CHECKS")
print("="*80)

# 6.1 Clustered standard errors by state
print("\n--- 6.1 Clustered Standard Errors (by State) ---")
try:
    model_cluster = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD',
                            data=df, weights=df['PERWT']).fit(cov_type='cluster',
                            cov_kwds={'groups': df['STATEFIP']})
    print(f"DiD coefficient: {model_cluster.params['ELIGIBLE_AFTER']:.5f}")
    print(f"Clustered SE: {model_cluster.bse['ELIGIBLE_AFTER']:.5f}")
    print(f"p-value: {model_cluster.pvalues['ELIGIBLE_AFTER']:.4f}")
    print(f"95% CI: [{model_cluster.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model_cluster.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")
except Exception as e:
    print(f"Clustering error: {e}")

# 6.2 Gender-specific effects
print("\n--- 6.2 Gender-Specific Effects ---")
df_male = df[df['FEMALE'] == 0]
df_female = df[df['FEMALE'] == 1]

model_male = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"\nMale subsample (n={len(df_male):,}):")
print(f"  DiD coefficient: {model_male.params['ELIGIBLE_AFTER']:.5f}")
print(f"  SE: {model_male.bse['ELIGIBLE_AFTER']:.5f}")
print(f"  p-value: {model_male.pvalues['ELIGIBLE_AFTER']:.4f}")

print(f"\nFemale subsample (n={len(df_female):,}):")
print(f"  DiD coefficient: {model_female.params['ELIGIBLE_AFTER']:.5f}")
print(f"  SE: {model_female.bse['ELIGIBLE_AFTER']:.5f}")
print(f"  p-value: {model_female.pvalues['ELIGIBLE_AFTER']:.4f}")

# 6.3 Effect by education level
print("\n--- 6.3 Effects by Education Level ---")
if 'EDUC_RECODE' in df.columns:
    for educ in df['EDUC_RECODE'].unique():
        df_educ = df[df['EDUC_RECODE'] == educ]
        if len(df_educ) > 100:
            try:
                model_educ = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                                     data=df_educ, weights=df_educ['PERWT']).fit(cov_type='HC1')
                print(f"\n{educ} (n={len(df_educ):,}):")
                print(f"  DiD: {model_educ.params['ELIGIBLE_AFTER']:.5f}, SE: {model_educ.bse['ELIGIBLE_AFTER']:.5f}, p={model_educ.pvalues['ELIGIBLE_AFTER']:.4f}")
            except:
                pass

# 6.4 Placebo test - pre-treatment trends
print("\n--- 6.4 Parallel Trends Check (Pre-Treatment Period) ---")
pre_data = df[df['AFTER'] == 0].copy()
pre_data['POST_2010'] = (pre_data['YEAR'] >= 2010).astype(int)
pre_data['ELIGIBLE_POST2010'] = pre_data['ELIGIBLE'] * pre_data['POST_2010']

model_placebo = smf.wls('FT ~ ELIGIBLE + POST_2010 + ELIGIBLE_POST2010',
                        data=pre_data, weights=pre_data['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD (2010-11 vs 2008-09): {model_placebo.params['ELIGIBLE_POST2010']:.5f}")
print(f"SE: {model_placebo.bse['ELIGIBLE_POST2010']:.5f}")
print(f"p-value: {model_placebo.pvalues['ELIGIBLE_POST2010']:.4f}")
print("(Null finding supports parallel trends assumption)")

# 6.5 Event study specification
print("\n--- 6.5 Event Study Coefficients ---")
# Use 2011 as reference year
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
# 2011 is reference
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

df['ELIG_2008'] = df['ELIGIBLE'] * df['YEAR_2008']
df['ELIG_2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIG_2010'] = df['ELIGIBLE'] * df['YEAR_2010']
df['ELIG_2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIG_2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIG_2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIG_2016'] = df['ELIGIBLE'] * df['YEAR_2016']

event_formula = 'FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIG_2008 + ELIG_2009 + ELIG_2010 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016'
model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nYear x Eligible interactions (relative to 2011):")
event_vars = ['ELIG_2008', 'ELIG_2009', 'ELIG_2010', 'ELIG_2013', 'ELIG_2014', 'ELIG_2015', 'ELIG_2016']
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    year = var.split('_')[1]
    print(f"  {year}: coef={coef:.5f}, SE={se:.5f}, p={pval:.4f}")

# ============================================================================
# SECTION 7: ADDITIONAL ANALYSES
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: ADDITIONAL ANALYSES")
print("="*80)

# 7.1 Effect on labor force participation
print("\n--- 7.1 Effect on Labor Force Participation ---")
df['IN_LF'] = (df['LABFORCE'] == 2).astype(int)
model_lf = smf.wls('IN_LF ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD on LFP: {model_lf.params['ELIGIBLE_AFTER']:.5f}")
print(f"SE: {model_lf.bse['ELIGIBLE_AFTER']:.5f}")
print(f"p-value: {model_lf.pvalues['ELIGIBLE_AFTER']:.4f}")

# 7.2 Effect on employment (any work)
print("\n--- 7.2 Effect on Any Employment ---")
df['EMPLOYED'] = (df['EMPSTAT'] == 1).astype(int)
model_emp = smf.wls('EMPLOYED ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD on Employment: {model_emp.params['ELIGIBLE_AFTER']:.5f}")
print(f"SE: {model_emp.bse['ELIGIBLE_AFTER']:.5f}")
print(f"p-value: {model_emp.pvalues['ELIGIBLE_AFTER']:.4f}")

# 7.3 Effect on usual hours worked
print("\n--- 7.3 Effect on Usual Hours Worked ---")
model_hrs = smf.wls('UHRSWORK ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD on Hours: {model_hrs.params['ELIGIBLE_AFTER']:.5f}")
print(f"SE: {model_hrs.bse['ELIGIBLE_AFTER']:.5f}")
print(f"p-value: {model_hrs.pvalues['ELIGIBLE_AFTER']:.4f}")

# ============================================================================
# SECTION 8: SUMMARY OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: SUMMARY OF RESULTS")
print("="*80)

print("\n=== PREFERRED SPECIFICATION: Weighted DiD with Robust SEs ===")
print(f"Effect size: {model3.params['ELIGIBLE_AFTER']:.5f}")
print(f"Standard error: {model3.bse['ELIGIBLE_AFTER']:.5f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")
print(f"Sample size: {int(model3.nobs):,}")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")

print("\n=== MODEL COMPARISON TABLE ===")
print("-" * 85)
print(f"{'Model':<40} {'Coef':>10} {'SE':>10} {'p-value':>10} {'N':>10}")
print("-" * 85)
print(f"{'1. Basic OLS (unweighted)':<40} {model1.params['ELIGIBLE_AFTER']:>10.5f} {model1.bse['ELIGIBLE_AFTER']:>10.5f} {model1.pvalues['ELIGIBLE_AFTER']:>10.4f} {int(model1.nobs):>10,}")
print(f"{'2. Weighted OLS':<40} {model2.params['ELIGIBLE_AFTER']:>10.5f} {model2.bse['ELIGIBLE_AFTER']:>10.5f} {model2.pvalues['ELIGIBLE_AFTER']:>10.4f} {int(model2.nobs):>10,}")
print(f"{'3. Weighted OLS + Robust SE':<40} {model3.params['ELIGIBLE_AFTER']:>10.5f} {model3.bse['ELIGIBLE_AFTER']:>10.5f} {model3.pvalues['ELIGIBLE_AFTER']:>10.4f} {int(model3.nobs):>10,}")
print(f"{'4. + Demographics':<40} {model4.params['ELIGIBLE_AFTER']:>10.5f} {model4.bse['ELIGIBLE_AFTER']:>10.5f} {model4.pvalues['ELIGIBLE_AFTER']:>10.4f} {int(model4.nobs):>10,}")
print(f"{'5. + Year FE':<40} {model6.params['ELIGIBLE_AFTER']:>10.5f} {model6.bse['ELIGIBLE_AFTER']:>10.5f} {model6.pvalues['ELIGIBLE_AFTER']:>10.4f} {int(model6.nobs):>10,}")
print("-" * 85)

# Save key results to JSON
results = {
    'preferred_estimate': {
        'effect_size': float(model3.params['ELIGIBLE_AFTER']),
        'effect_size_pct_points': float(model3.params['ELIGIBLE_AFTER'] * 100),
        'standard_error': float(model3.bse['ELIGIBLE_AFTER']),
        'ci_lower': float(model3.conf_int().loc['ELIGIBLE_AFTER', 0]),
        'ci_upper': float(model3.conf_int().loc['ELIGIBLE_AFTER', 1]),
        'p_value': float(model3.pvalues['ELIGIBLE_AFTER']),
        'sample_size': int(model3.nobs)
    },
    'with_covariates': {
        'effect_size': float(model4.params['ELIGIBLE_AFTER']),
        'standard_error': float(model4.bse['ELIGIBLE_AFTER']),
        'p_value': float(model4.pvalues['ELIGIBLE_AFTER'])
    },
    'with_year_fe': {
        'effect_size': float(model6.params['ELIGIBLE_AFTER']),
        'standard_error': float(model6.bse['ELIGIBLE_AFTER']),
        'p_value': float(model6.pvalues['ELIGIBLE_AFTER'])
    }
}

with open(f'{output_path}/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults saved to: {output_path}/analysis_results.json")
