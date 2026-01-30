"""
DACA Replication Analysis Script
Independent Replication #38

Research Question: Effect of DACA eligibility on full-time employment
among Mexican-born Hispanic individuals in the US.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Load Data
# =============================================================================
print("=" * 80)
print("DACA REPLICATION ANALYSIS - Replication 38")
print("=" * 80)

df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"\nTotal observations loaded: {len(df)}")

# =============================================================================
# Data Exploration
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: DATA EXPLORATION")
print("=" * 80)

# Check years available
print("\n--- Years in Dataset ---")
print(df['YEAR'].value_counts().sort_index())

# Check ELIGIBLE distribution
print("\n--- ELIGIBLE Distribution ---")
print(f"Treatment (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum()}")
print(f"Control (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum()}")

# Check AFTER distribution
print("\n--- AFTER Distribution ---")
print(f"Pre-DACA (AFTER=0): {(df['AFTER']==0).sum()}")
print(f"Post-DACA (AFTER=1): {(df['AFTER']==1).sum()}")

# Check FT distribution
print("\n--- Full-Time Employment (FT) ---")
print(f"Not Full-Time (FT=0): {(df['FT']==0).sum()}")
print(f"Full-Time (FT=1): {(df['FT']==1).sum()}")
print(f"Overall FT Rate: {df['FT'].mean():.4f}")

# Age in June 2012 distribution by group
print("\n--- Age at DACA Implementation (June 2012) ---")
print("\nTreatment Group (ELIGIBLE=1):")
print(df[df['ELIGIBLE']==1]['AGE_IN_JUNE_2012'].describe())
print("\nControl Group (ELIGIBLE=0):")
print(df[df['ELIGIBLE']==0]['AGE_IN_JUNE_2012'].describe())

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: SUMMARY STATISTICS")
print("=" * 80)

# Create groups
df['group'] = df['ELIGIBLE'].map({1: 'Treatment (26-30)', 0: 'Control (31-35)'})
df['period'] = df['AFTER'].map({0: 'Pre-DACA (2008-2011)', 1: 'Post-DACA (2013-2016)'})

# Summary statistics by group and period
print("\n--- Sample Sizes by Group and Period ---")
crosstab = pd.crosstab(df['group'], df['period'])
print(crosstab)

# Full-time employment rates by group and period
print("\n--- Full-Time Employment Rates by Group and Period ---")
ft_rates = df.groupby(['group', 'period'])['FT'].mean().unstack()
print(ft_rates.round(4))

# Weighted FT rates
print("\n--- Weighted Full-Time Employment Rates by Group and Period ---")
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

weighted_ft = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean).unstack()
weighted_ft.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_ft.columns = ['Pre-DACA', 'Post-DACA']
print(weighted_ft.round(4))

# Compute simple DiD
print("\n--- Simple Difference-in-Differences (Unweighted) ---")
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"Treatment Pre:  {pre_treat:.4f}")
print(f"Treatment Post: {post_treat:.4f}")
print(f"Treatment Change: {post_treat - pre_treat:.4f}")
print(f"\nControl Pre:  {pre_control:.4f}")
print(f"Control Post: {post_control:.4f}")
print(f"Control Change: {post_control - pre_control:.4f}")
print(f"\nDiD Estimate (unweighted): {(post_treat - pre_treat) - (post_control - pre_control):.4f}")

# =============================================================================
# Demographic Characteristics
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: DEMOGRAPHIC CHARACTERISTICS")
print("=" * 80)

# Sex distribution (1=Male, 2=Female in IPUMS)
print("\n--- Sex Distribution by Group ---")
sex_by_group = df.groupby('ELIGIBLE')['SEX'].value_counts(normalize=True).unstack()
sex_by_group.index = ['Control', 'Treatment']
sex_by_group.columns = ['Male', 'Female']
print(sex_by_group.round(4))

# Age distribution
print("\n--- Mean Age by Group and Period ---")
age_stats = df.groupby(['ELIGIBLE', 'AFTER'])['AGE'].mean().unstack()
age_stats.index = ['Control', 'Treatment']
age_stats.columns = ['Pre-DACA', 'Post-DACA']
print(age_stats.round(2))

# Education distribution
print("\n--- Education Distribution by Group ---")
educ_by_group = df.groupby(['ELIGIBLE', 'EDUC_RECODE']).size().unstack(fill_value=0)
educ_by_group = educ_by_group.div(educ_by_group.sum(axis=1), axis=0)
print(educ_by_group.round(4))

# Marital status
print("\n--- Marital Status Distribution by Group ---")
marst_by_group = df.groupby('ELIGIBLE')['MARST'].value_counts(normalize=True).unstack()
marst_by_group.index = ['Control', 'Treatment']
print(marst_by_group.round(4))

# =============================================================================
# Main DiD Regression Analysis
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (Unweighted) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD Coefficient: {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 2: Basic DiD (weighted)
print("\n--- Model 2: Basic DiD (Weighted) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])
print(f"\nDiD Coefficient: {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 3: DiD with year fixed effects (weighted)
print("\n--- Model 3: DiD with Year Fixed Effects (Weighted) ---")
df['YEAR'] = df['YEAR'].astype('category')
model3 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR)', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 4: DiD with year and state fixed effects (weighted)
print("\n--- Model 4: DiD with Year and State Fixed Effects (Weighted) ---")
df['STATEFIP'] = df['STATEFIP'].astype('category')
model4 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP)', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 5: Full model with covariates
print("\n--- Model 5: Full Model with Covariates (Weighted) ---")
# Create binary variables for covariates
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Education dummies (reference: Less than High School)
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

model5 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP) + FEMALE + MARRIED + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                 data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")

# Print select coefficients
print("\nSelected Coefficients:")
for var in ['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'AGE', 'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_TWOYEAR', 'EDUC_BA']:
    if var in model5.params:
        print(f"  {var}: {model5.params[var]:.4f} (SE: {model5.bse[var]:.4f})")

# =============================================================================
# Robust Standard Errors
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: ROBUST STANDARD ERRORS")
print("=" * 80)

# Model with heteroskedasticity-robust (HC1) standard errors
print("\n--- Model with Robust (HC1) Standard Errors ---")
model_robust = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP) + FEMALE + MARRIED + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                       data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_robust.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model_robust.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_robust.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_robust.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model_robust.pvalues['ELIGIBLE_AFTER']:.4f}")

# Cluster-robust standard errors at state level
print("\n--- Model with State-Clustered Standard Errors ---")
model_cluster = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP) + FEMALE + MARRIED + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                        data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model_cluster.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {model_cluster.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_cluster.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_cluster.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model_cluster.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# Pre-Trends Analysis
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 6: PRE-TRENDS ANALYSIS")
print("=" * 80)

# Calculate FT rates by year and group
yearly_ft = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
yearly_ft.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\n--- Weighted Full-Time Employment by Year ---")
print(yearly_ft.round(4))

# Calculate year-by-year differences
yearly_ft['Difference'] = yearly_ft['Treatment (26-30)'] - yearly_ft['Control (31-35)']
print("\n--- Treatment-Control Difference by Year ---")
print(yearly_ft['Difference'].round(4))

# Event study style analysis
print("\n--- Event Study Regression ---")
df['YEAR_num'] = df['YEAR'].astype(int)
years = sorted(df['YEAR_num'].unique())
# Create year dummies and interaction with ELIGIBLE
# Use 2011 as reference year (last pre-treatment year)
for year in years:
    if year != 2011:
        df[f'YEAR_{year}'] = (df['YEAR_num'] == year).astype(int)
        df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Build formula for event study
year_terms = ' + '.join([f'YEAR_{y}' for y in years if y != 2011])
interaction_terms = ' + '.join([f'ELIGIBLE_YEAR_{y}' for y in years if y != 2011])

event_model = smf.wls(f'FT ~ ELIGIBLE + {year_terms} + {interaction_terms} + C(STATEFIP) + FEMALE + MARRIED + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Eligible x Year, base = 2011):")
for year in sorted(years):
    if year != 2011:
        coef = event_model.params[f'ELIGIBLE_YEAR_{year}']
        se = event_model.bse[f'ELIGIBLE_YEAR_{year}']
        pval = event_model.pvalues[f'ELIGIBLE_YEAR_{year}']
        print(f"  Year {year}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# =============================================================================
# Heterogeneity Analysis
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 7: HETEROGENEITY ANALYSIS")
print("=" * 80)

# By Sex
print("\n--- Heterogeneity by Sex ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex]
    model_sex = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP) + MARRIED + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                        data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_sex.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sex.bse['ELIGIBLE_AFTER']:.4f}, p={model_sex.pvalues['ELIGIBLE_AFTER']:.4f})")

# By Education
print("\n--- Heterogeneity by Education ---")
df['HIGH_EDUC'] = df['EDUC_RECODE'].isin(['Some College', 'Two-Year Degree', 'BA+']).astype(int)
for educ, label in [(0, 'HS or Less'), (1, 'Some College+')]:
    subset = df[df['HIGH_EDUC'] == educ]
    model_educ = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP) + FEMALE + MARRIED + AGE',
                         data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_educ.params['ELIGIBLE_AFTER']:.4f} (SE: {model_educ.bse['ELIGIBLE_AFTER']:.4f}, p={model_educ.pvalues['ELIGIBLE_AFTER']:.4f})")

# =============================================================================
# Placebo Tests
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 8: PLACEBO TESTS")
print("=" * 80)

# Placebo test using only pre-period data (2008-2011)
# Treat 2010-2011 as "post" and 2008-2009 as "pre"
print("\n--- Placebo Test: Fake Treatment in 2010 (Pre-Period Only) ---")
pre_data = df[df['AFTER'] == 0].copy()
pre_data['FAKE_AFTER'] = (pre_data['YEAR_num'] >= 2010).astype(int)
pre_data['ELIGIBLE_FAKE_AFTER'] = pre_data['ELIGIBLE'] * pre_data['FAKE_AFTER']

placebo_model = smf.wls('FT ~ ELIGIBLE + FAKE_AFTER + ELIGIBLE_FAKE_AFTER + C(STATEFIP) + FEMALE + MARRIED + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                        data=pre_data, weights=pre_data['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD: {placebo_model.params['ELIGIBLE_FAKE_AFTER']:.4f}")
print(f"SE: {placebo_model.bse['ELIGIBLE_FAKE_AFTER']:.4f}")
print(f"p-value: {placebo_model.pvalues['ELIGIBLE_FAKE_AFTER']:.4f}")

# =============================================================================
# Summary of Results
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 9: SUMMARY OF RESULTS")
print("=" * 80)

print("\n--- Main Results Summary ---")
print(f"{'Model':<50} {'DiD':<10} {'SE':<10} {'95% CI':<25} {'p-value':<10}")
print("-" * 105)

results = [
    ("1. Basic (Unweighted)", model1.params['ELIGIBLE_AFTER'], model1.bse['ELIGIBLE_AFTER'],
     f"[{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]",
     model1.pvalues['ELIGIBLE_AFTER']),
    ("2. Basic (Weighted)", model2.params['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'],
     f"[{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]",
     model2.pvalues['ELIGIBLE_AFTER']),
    ("3. Year FE (Weighted)", model3.params['ELIGIBLE_AFTER'], model3.bse['ELIGIBLE_AFTER'],
     f"[{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]",
     model3.pvalues['ELIGIBLE_AFTER']),
    ("4. Year + State FE (Weighted)", model4.params['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'],
     f"[{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]",
     model4.pvalues['ELIGIBLE_AFTER']),
    ("5. Full Model with Covariates", model5.params['ELIGIBLE_AFTER'], model5.bse['ELIGIBLE_AFTER'],
     f"[{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]",
     model5.pvalues['ELIGIBLE_AFTER']),
    ("6. Robust SE (HC1)", model_robust.params['ELIGIBLE_AFTER'], model_robust.bse['ELIGIBLE_AFTER'],
     f"[{model_robust.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_robust.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]",
     model_robust.pvalues['ELIGIBLE_AFTER']),
    ("7. State-Clustered SE", model_cluster.params['ELIGIBLE_AFTER'], model_cluster.bse['ELIGIBLE_AFTER'],
     f"[{model_cluster.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_cluster.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]",
     model_cluster.pvalues['ELIGIBLE_AFTER']),
]

for name, coef, se, ci, pval in results:
    print(f"{name:<50} {coef:<10.4f} {se:<10.4f} {ci:<25} {pval:<10.4f}")

# =============================================================================
# Preferred Estimate
# =============================================================================
print("\n" + "=" * 80)
print("PREFERRED ESTIMATE (Model 7: State-Clustered SE)")
print("=" * 80)
print(f"\nEffect Size: {model_cluster.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model_cluster.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model_cluster.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_cluster.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model_cluster.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample Size: {len(df)}")

# =============================================================================
# Save Results for Report
# =============================================================================
results_dict = {
    'preferred_estimate': model_cluster.params['ELIGIBLE_AFTER'],
    'preferred_se': model_cluster.bse['ELIGIBLE_AFTER'],
    'preferred_ci_lower': model_cluster.conf_int().loc['ELIGIBLE_AFTER', 0],
    'preferred_ci_upper': model_cluster.conf_int().loc['ELIGIBLE_AFTER', 1],
    'preferred_pvalue': model_cluster.pvalues['ELIGIBLE_AFTER'],
    'sample_size': len(df),
    'n_treatment': (df['ELIGIBLE']==1).sum(),
    'n_control': (df['ELIGIBLE']==0).sum(),
    'n_pre': (df['AFTER']==0).sum(),
    'n_post': (df['AFTER']==1).sum(),
}

# Save to file
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\nAnalysis complete. Results saved to analysis_results.json")
