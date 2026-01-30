"""
DACA Replication Study #52: Impact of DACA Eligibility on Full-Time Employment
Difference-in-Differences Analysis

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for the DACA
program (treatment) on the probability that the eligible person is employed
full-time (35+ hours/week)?

Identification Strategy:
- Treated: Ages 26-30 at DACA implementation (June 2012) - ELIGIBLE=1
- Control: Ages 31-35 at DACA implementation - ELIGIBLE=0
- Pre-period: 2008-2011 (AFTER=0)
- Post-period: 2013-2016 (AFTER=1)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("DACA REPLICATION STUDY #52")
print("=" * 80)

# Load numeric version for analysis
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"\nData loaded: {df.shape[0]:,} observations, {df.shape[1]} variables")

# =============================================================================
# DATA EXPLORATION
# =============================================================================
print("\n" + "=" * 80)
print("DATA EXPLORATION")
print("=" * 80)

# Year distribution
print("\n--- Year Distribution ---")
print(df['YEAR'].value_counts().sort_index())

# ELIGIBLE distribution
print("\n--- ELIGIBLE Distribution ---")
print(f"ELIGIBLE=1 (Treatment, ages 26-30): {(df['ELIGIBLE']==1).sum():,}")
print(f"ELIGIBLE=0 (Control, ages 31-35): {(df['ELIGIBLE']==0).sum():,}")

# AFTER distribution
print("\n--- AFTER (Pre/Post) Distribution ---")
print(f"AFTER=0 (Pre-DACA, 2008-2011): {(df['AFTER']==0).sum():,}")
print(f"AFTER=1 (Post-DACA, 2013-2016): {(df['AFTER']==1).sum():,}")

# FT distribution
print("\n--- FT (Full-Time Employment) Distribution ---")
print(f"FT=1 (Full-time, 35+ hrs): {(df['FT']==1).sum():,}")
print(f"FT=0 (Not full-time): {(df['FT']==0).sum():,}")
print(f"FT Mean (unweighted): {df['FT'].mean():.4f}")

# Cross-tabulation
print("\n--- Cross-tabulation: ELIGIBLE x AFTER ---")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (31-35)', 'Treated (26-30)', 'Total']
crosstab.columns = ['Pre (2008-11)', 'Post (2013-16)', 'Total']
print(crosstab)

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

# Key variables summary
key_vars = ['YEAR', 'FT', 'ELIGIBLE', 'AFTER', 'AGE', 'SEX', 'PERWT',
            'EDUC', 'MARST', 'NCHILD', 'UHRSWORK']

print("\n--- Summary Statistics (Key Variables) ---")
print(df[key_vars].describe().round(3))

# Weighted mean of FT by group
print("\n--- Weighted FT Rates by Group ---")

def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

def weighted_std(data, value_col, weight_col):
    avg = weighted_mean(data, value_col, weight_col)
    variance = np.average((data[value_col] - avg)**2, weights=data[weight_col])
    return np.sqrt(variance)

groups = df.groupby(['ELIGIBLE', 'AFTER'])
results = []
for (eligible, after), group in groups:
    n = len(group)
    wt_mean = weighted_mean(group, 'FT', 'PERWT')
    wt_std = weighted_std(group, 'FT', 'PERWT')
    results.append({
        'ELIGIBLE': eligible,
        'AFTER': after,
        'N': n,
        'Weighted_FT_Mean': wt_mean,
        'Weighted_FT_SD': wt_std
    })

desc_df = pd.DataFrame(results)
desc_df['Group'] = desc_df.apply(lambda x: f"{'Treated' if x['ELIGIBLE']==1 else 'Control'}, {'Post' if x['AFTER']==1 else 'Pre'}", axis=1)
print(desc_df[['Group', 'N', 'Weighted_FT_Mean', 'Weighted_FT_SD']].to_string(index=False))

# Calculate simple DiD
print("\n--- Simple Difference-in-Differences Calculation ---")
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]

ft_pre_control = weighted_mean(pre_control, 'FT', 'PERWT')
ft_post_control = weighted_mean(post_control, 'FT', 'PERWT')
ft_pre_treat = weighted_mean(pre_treat, 'FT', 'PERWT')
ft_post_treat = weighted_mean(post_treat, 'FT', 'PERWT')

print(f"\nControl Group (Ages 31-35):")
print(f"  Pre-DACA FT Rate:  {ft_pre_control:.4f}")
print(f"  Post-DACA FT Rate: {ft_post_control:.4f}")
print(f"  Change:            {ft_post_control - ft_pre_control:+.4f}")

print(f"\nTreatment Group (Ages 26-30):")
print(f"  Pre-DACA FT Rate:  {ft_pre_treat:.4f}")
print(f"  Post-DACA FT Rate: {ft_post_treat:.4f}")
print(f"  Change:            {ft_post_treat - ft_pre_treat:+.4f}")

simple_did = (ft_post_treat - ft_pre_treat) - (ft_post_control - ft_pre_control)
print(f"\nSimple DiD Estimate: {simple_did:+.4f}")

# =============================================================================
# COVARIATE BALANCE CHECK
# =============================================================================
print("\n" + "=" * 80)
print("COVARIATE BALANCE: PRE-TREATMENT PERIOD")
print("=" * 80)

pre_period = df[df['AFTER'] == 0]

# Convert SEX to female indicator (SEX=2 is female in IPUMS)
df['FEMALE'] = (df['SEX'] == 2).astype(int)
pre_period = df[df['AFTER'] == 0]

covariates = ['AGE', 'FEMALE', 'MARST', 'NCHILD', 'EDUC', 'HHINCOME', 'POVERTY']

print("\n--- Balance Table (Pre-Period Means) ---")
balance_results = []
for var in covariates:
    if var in pre_period.columns:
        try:
            treat_mean = weighted_mean(pre_period[pre_period['ELIGIBLE']==1], var, 'PERWT')
            control_mean = weighted_mean(pre_period[pre_period['ELIGIBLE']==0], var, 'PERWT')
            diff = treat_mean - control_mean
            balance_results.append({
                'Variable': var,
                'Treated Mean': treat_mean,
                'Control Mean': control_mean,
                'Difference': diff
            })
        except:
            pass

balance_df = pd.DataFrame(balance_results)
print(balance_df.to_string(index=False))

# =============================================================================
# MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("MAIN REGRESSION ANALYSIS: DIFFERENCE-IN-DIFFERENCES")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (OLS with survey weights)
print("\n--- Model 1: Basic DiD (Weighted OLS) ---")
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model1.summary().tables[1])
print(f"\nDiD Estimate: {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Std Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 2: DiD with Year Fixed Effects
print("\n--- Model 2: DiD with Year Fixed Effects ---")
df['YEAR_factor'] = pd.Categorical(df['YEAR'])
model2 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
did_coef2 = model2.params['ELIGIBLE_AFTER']
did_se2 = model2.bse['ELIGIBLE_AFTER']
did_pval2 = model2.pvalues['ELIGIBLE_AFTER']
did_ci2 = model2.conf_int().loc['ELIGIBLE_AFTER']
print(f"DiD Estimate: {did_coef2:.4f}")
print(f"Std Error: {did_se2:.4f}")
print(f"95% CI: [{did_ci2[0]:.4f}, {did_ci2[1]:.4f}]")
print(f"p-value: {did_pval2:.4f}")

# Model 3: DiD with Individual Covariates
print("\n--- Model 3: DiD with Individual Covariates ---")
# Create dummy variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Education categories (EDUC_RECODE: 1=Less than HS, 2=HS, 3=Some College, 4=2-yr, 5=BA+)
# Create dummies for education
df['EDUC_HS'] = (df['EDUC_RECODE'] == 2).astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 3).astype(int)
df['EDUC_ASSOC'] = (df['EDUC_RECODE'] == 4).astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 5).astype(int)

# Marital status: married = 1 or 2 in IPUMS
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)

model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_ASSOC + EDUC_BA',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
did_coef3 = model3.params['ELIGIBLE_AFTER']
did_se3 = model3.bse['ELIGIBLE_AFTER']
did_pval3 = model3.pvalues['ELIGIBLE_AFTER']
did_ci3 = model3.conf_int().loc['ELIGIBLE_AFTER']
print(f"DiD Estimate: {did_coef3:.4f}")
print(f"Std Error: {did_se3:.4f}")
print(f"95% CI: [{did_ci3[0]:.4f}, {did_ci3[1]:.4f}]")
print(f"p-value: {did_pval3:.4f}")
print("\nFull Model Output:")
print(model3.summary().tables[1])

# Model 4: DiD with Year FE and Individual Covariates
print("\n--- Model 4: DiD with Year FE and Covariates ---")
model4 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_ASSOC + EDUC_BA',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
did_coef4 = model4.params['ELIGIBLE_AFTER']
did_se4 = model4.bse['ELIGIBLE_AFTER']
did_pval4 = model4.pvalues['ELIGIBLE_AFTER']
did_ci4 = model4.conf_int().loc['ELIGIBLE_AFTER']
print(f"DiD Estimate: {did_coef4:.4f}")
print(f"Std Error: {did_se4:.4f}")
print(f"95% CI: [{did_ci4[0]:.4f}, {did_ci4[1]:.4f}]")
print(f"p-value: {did_pval4:.4f}")

# Model 5: DiD with Year FE, Covariates, and State FE
print("\n--- Model 5: DiD with Year FE, Covariates, and State FE ---")
model5 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_ASSOC + EDUC_BA',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
did_coef5 = model5.params['ELIGIBLE_AFTER']
did_se5 = model5.bse['ELIGIBLE_AFTER']
did_pval5 = model5.pvalues['ELIGIBLE_AFTER']
did_ci5 = model5.conf_int().loc['ELIGIBLE_AFTER']
print(f"DiD Estimate: {did_coef5:.4f}")
print(f"Std Error: {did_se5:.4f}")
print(f"95% CI: [{did_ci5[0]:.4f}, {did_ci5[1]:.4f}]")
print(f"p-value: {did_pval5:.4f}")

# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# 1. State-level clustering
print("\n--- Robustness 1: State-Clustered Standard Errors ---")
# Use cluster-robust standard errors by state
from statsmodels.stats.sandwich_covariance import cov_cluster
model5_cluster = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_ASSOC + EDUC_BA',
                         data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
did_coef_cl = model5_cluster.params['ELIGIBLE_AFTER']
did_se_cl = model5_cluster.bse['ELIGIBLE_AFTER']
did_ci_cl = model5_cluster.conf_int().loc['ELIGIBLE_AFTER']
print(f"DiD Estimate: {did_coef_cl:.4f}")
print(f"Clustered SE: {did_se_cl:.4f}")
print(f"95% CI: [{did_ci_cl[0]:.4f}, {did_ci_cl[1]:.4f}]")

# 2. Separate analysis by sex
print("\n--- Robustness 2: Analysis by Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex_val]
    model_sex = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_ASSOC + EDUC_BA',
                        data=df_sex, weights=df_sex['PERWT']).fit(cov_type='HC1')
    print(f"{sex_name}: DiD = {model_sex.params['ELIGIBLE_AFTER']:.4f} (SE = {model_sex.bse['ELIGIBLE_AFTER']:.4f})")

# 3. Pre-trends test (year-by-eligible interactions in pre-period)
print("\n--- Robustness 3: Pre-Trends Test ---")
pre_df = df[df['AFTER'] == 0]
pre_df = pre_df.copy()
pre_df['YEAR_2009'] = (pre_df['YEAR'] == 2009).astype(int)
pre_df['YEAR_2010'] = (pre_df['YEAR'] == 2010).astype(int)
pre_df['YEAR_2011'] = (pre_df['YEAR'] == 2011).astype(int)
pre_df['ELIGIBLE_2009'] = pre_df['ELIGIBLE'] * pre_df['YEAR_2009']
pre_df['ELIGIBLE_2010'] = pre_df['ELIGIBLE'] * pre_df['YEAR_2010']
pre_df['ELIGIBLE_2011'] = pre_df['ELIGIBLE'] * pre_df['YEAR_2011']

pretrend_model = smf.wls('FT ~ ELIGIBLE + YEAR_2009 + YEAR_2010 + YEAR_2011 + ELIGIBLE_2009 + ELIGIBLE_2010 + ELIGIBLE_2011',
                         data=pre_df, weights=pre_df['PERWT']).fit(cov_type='HC1')
print("Pre-period year x eligible interactions:")
for var in ['ELIGIBLE_2009', 'ELIGIBLE_2010', 'ELIGIBLE_2011']:
    print(f"  {var}: {pretrend_model.params[var]:.4f} (SE = {pretrend_model.bse[var]:.4f}, p = {pretrend_model.pvalues[var]:.4f})")

# Joint test for pre-trends
f_test = pretrend_model.f_test('ELIGIBLE_2009 = ELIGIBLE_2010 = ELIGIBLE_2011 = 0')
try:
    f_val = f_test.fvalue if isinstance(f_test.fvalue, float) else f_test.fvalue[0][0]
    p_val = f_test.pvalue if isinstance(f_test.pvalue, float) else f_test.pvalue[0][0]
    print(f"\nJoint F-test for parallel pre-trends: F = {f_val:.4f}, p = {p_val:.4f}")
except:
    print(f"\nJoint F-test for parallel pre-trends: F = {f_test.fvalue}, p = {f_test.pvalue}")

# =============================================================================
# HETEROGENEITY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("HETEROGENEITY ANALYSIS")
print("=" * 80)

# By education level
print("\n--- Heterogeneity by Education ---")
for educ_val, educ_name in [(1, 'Less than HS'), (2, 'High School'), (3, 'Some College'), (4, 'Associate'), (5, 'BA+')]:
    df_educ = df[df['EDUC_RECODE'] == educ_val]
    if len(df_educ) > 100:
        try:
            model_educ = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                                data=df_educ, weights=df_educ['PERWT']).fit(cov_type='HC1')
            print(f"{educ_name}: DiD = {model_educ.params['ELIGIBLE_AFTER']:.4f} (SE = {model_educ.bse['ELIGIBLE_AFTER']:.4f}), N = {len(df_educ)}")
        except:
            print(f"{educ_name}: Could not estimate (N = {len(df_educ)})")
    else:
        print(f"{educ_name}: Insufficient observations (N = {len(df_educ)})")

# By region
print("\n--- Heterogeneity by Census Region ---")
for region in df['CensusRegion'].unique():
    df_region = df[df['CensusRegion'] == region]
    if len(df_region) > 100:
        try:
            model_region = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                                   data=df_region, weights=df_region['PERWT']).fit(cov_type='HC1')
            print(f"{region}: DiD = {model_region.params['ELIGIBLE_AFTER']:.4f} (SE = {model_region.bse['ELIGIBLE_AFTER']:.4f}), N = {len(df_region)}")
        except:
            print(f"{region}: Could not estimate")

# =============================================================================
# DYNAMIC EFFECTS (EVENT STUDY)
# =============================================================================
print("\n" + "=" * 80)
print("DYNAMIC EFFECTS (EVENT STUDY)")
print("=" * 80)

# Create year dummies
years = sorted(df['YEAR'].unique())
base_year = 2011  # Use 2011 as the base year (last pre-treatment year)

for yr in years:
    if yr != base_year:
        df[f'YEAR_{yr}'] = (df['YEAR'] == yr).astype(int)
        df[f'ELIGIBLE_YEAR_{yr}'] = df['ELIGIBLE'] * df[f'YEAR_{yr}']

# Build formula for event study
year_terms = ' + '.join([f'YEAR_{yr}' for yr in years if yr != base_year])
interaction_terms = ' + '.join([f'ELIGIBLE_YEAR_{yr}' for yr in years if yr != base_year])
formula_es = f'FT ~ ELIGIBLE + {year_terms} + {interaction_terms}'

model_es = smf.wls(formula_es, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
event_study_results = []
for yr in years:
    if yr != base_year:
        var = f'ELIGIBLE_YEAR_{yr}'
        coef = model_es.params[var]
        se = model_es.bse[var]
        ci_low = model_es.conf_int().loc[var, 0]
        ci_high = model_es.conf_int().loc[var, 1]
        pval = model_es.pvalues[var]
        event_study_results.append({
            'Year': yr,
            'Coefficient': coef,
            'SE': se,
            'CI_Low': ci_low,
            'CI_High': ci_high,
            'p-value': pval
        })
        period = "Pre" if yr < 2012 else "Post"
        print(f"{yr} ({period}): {coef:+.4f} (SE = {se:.4f}), 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# =============================================================================
# SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF MAIN RESULTS")
print("=" * 80)

print("\n--- Main DiD Estimates ---")
print(f"Model                                          Estimate        SE               95% CI       p-value")
print("-" * 100)

# Get confidence intervals
ci1 = model1.conf_int().loc['ELIGIBLE_AFTER']
ci_str1 = f"[{ci1[0]:.4f}, {ci1[1]:.4f}]"
print(f"1. Basic DiD                                   {model1.params['ELIGIBLE_AFTER']:>8.4f}  {model1.bse['ELIGIBLE_AFTER']:>8.4f}  {ci_str1:>22}  {model1.pvalues['ELIGIBLE_AFTER']:>8.4f}")

ci_str2 = f"[{did_ci2[0]:.4f}, {did_ci2[1]:.4f}]"
print(f"2. Year FE                                     {did_coef2:>8.4f}  {did_se2:>8.4f}  {ci_str2:>22}  {did_pval2:>8.4f}")

ci_str3 = f"[{did_ci3[0]:.4f}, {did_ci3[1]:.4f}]"
print(f"3. Individual Covariates                       {did_coef3:>8.4f}  {did_se3:>8.4f}  {ci_str3:>22}  {did_pval3:>8.4f}")

ci_str4 = f"[{did_ci4[0]:.4f}, {did_ci4[1]:.4f}]"
print(f"4. Year FE + Covariates                        {did_coef4:>8.4f}  {did_se4:>8.4f}  {ci_str4:>22}  {did_pval4:>8.4f}")

ci_str5 = f"[{did_ci5[0]:.4f}, {did_ci5[1]:.4f}]"
print(f"5. Year + State FE + Covariates                {did_coef5:>8.4f}  {did_se5:>8.4f}  {ci_str5:>22}  {did_pval5:>8.4f}")

ci_str_cl = f"[{did_ci_cl[0]:.4f}, {did_ci_cl[1]:.4f}]"
pval_cl = model5_cluster.pvalues['ELIGIBLE_AFTER']
print(f"5b. With State-Clustered SE                    {did_coef_cl:>8.4f}  {did_se_cl:>8.4f}  {ci_str_cl:>22}  {pval_cl:>8.4f}")

print("\n--- Sample Size ---")
print(f"Total observations: {len(df):,}")
print(f"Treatment group (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum():,}")
print(f"Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum():,}")

# Preferred estimate
print("\n" + "=" * 80)
print("PREFERRED ESTIMATE (Model 5b - Full specification with clustered SE)")
print("=" * 80)
print(f"\nEffect of DACA eligibility on full-time employment:")
print(f"  Point estimate: {did_coef_cl:.4f} ({did_coef_cl*100:.2f} percentage points)")
print(f"  Standard error: {did_se_cl:.4f} (state-clustered)")
print(f"  95% CI: [{did_ci_cl[0]:.4f}, {did_ci_cl[1]:.4f}]")
print(f"  p-value: {model5_cluster.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  Sample size: {len(df):,}")

# Save event study results
es_df = pd.DataFrame(event_study_results)
es_df.to_csv('event_study_results.csv', index=False)

# Save main results
main_results = {
    'model': ['Basic DiD', 'Year FE', 'Covariates', 'Year FE + Covariates', 'Full (Year+State FE+Covariates)', 'Full with Clustered SE'],
    'estimate': [model1.params['ELIGIBLE_AFTER'], did_coef2, did_coef3, did_coef4, did_coef5, did_coef_cl],
    'se': [model1.bse['ELIGIBLE_AFTER'], did_se2, did_se3, did_se4, did_se5, did_se_cl],
    'ci_low': [model1.conf_int().loc['ELIGIBLE_AFTER', 0], did_ci2[0], did_ci3[0], did_ci4[0], did_ci5[0], did_ci_cl[0]],
    'ci_high': [model1.conf_int().loc['ELIGIBLE_AFTER', 1], did_ci2[1], did_ci3[1], did_ci4[1], did_ci5[1], did_ci_cl[1]],
    'pvalue': [model1.pvalues['ELIGIBLE_AFTER'], did_pval2, did_pval3, did_pval4, did_pval5, model5_cluster.pvalues['ELIGIBLE_AFTER']]
}
results_df = pd.DataFrame(main_results)
results_df.to_csv('main_results.csv', index=False)

print("\n\nResults saved to 'main_results.csv' and 'event_study_results.csv'")
print("\nAnalysis complete.")
