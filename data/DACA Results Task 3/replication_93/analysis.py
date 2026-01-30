"""
DACA Replication Analysis - Session 93
Independent Replication of DACA Impact on Full-Time Employment

Research Question: What was the causal impact of DACA eligibility on the probability
of full-time employment among Hispanic-Mexican, Mexican-born individuals?

Method: Difference-in-Differences
- Treatment: DACA-eligible individuals aged 26-30 at policy implementation
- Control: Would-be eligible individuals aged 31-35 (ineligible due to age)
- Pre-period: 2008-2011
- Post-period: 2013-2016
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

print("=" * 80)
print("DACA REPLICATION ANALYSIS - SESSION 93")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

data_path = r"C:\Users\seraf\DACA Results Task 3\replication_93\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"\nDataset shape: {df.shape[0]:,} observations, {df.shape[1]} variables")
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")
print(f"\nELIGIBLE values: {df['ELIGIBLE'].value_counts().to_dict()}")
print(f"AFTER values: {df['AFTER'].value_counts().to_dict()}")
print(f"FT values: {df['FT'].value_counts().to_dict()}")

# =============================================================================
# 2. DATA VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("2. DATA VALIDATION")
print("=" * 80)

# Check key variables
print("\nChecking key variables:")
print(f"  - Missing ELIGIBLE: {df['ELIGIBLE'].isna().sum()}")
print(f"  - Missing AFTER: {df['AFTER'].isna().sum()}")
print(f"  - Missing FT: {df['FT'].isna().sum()}")
print(f"  - Missing PERWT: {df['PERWT'].isna().sum()}")

# Sample size by group
print("\n\nSample Distribution by Group and Period:")
sample_dist = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['count', 'mean'],
    'PERWT': 'sum'
}).round(4)
sample_dist.columns = ['N', 'FT_Rate', 'Weighted_N']
print(sample_dist)

# By year
print("\n\nSample Size by Year and Treatment Group:")
year_dist = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
year_dist.columns = ['Control (31-35)', 'Treatment (26-30)']
print(year_dist)

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 80)

# Full-time employment rates by group and period
print("\n\nFull-Time Employment Rates (Unweighted):")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
ft_rates.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)']
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
print(ft_rates.round(4))

# Calculate simple DiD
did_simple = (ft_rates.iloc[1, 1] - ft_rates.iloc[1, 0]) - (ft_rates.iloc[0, 1] - ft_rates.iloc[0, 0])
print(f"\nSimple DiD Estimate (Unweighted): {did_simple:.4f}")

# Weighted Full-time employment rates
def weighted_mean(group, value_col, weight_col):
    return np.average(group[value_col], weights=group[weight_col])

print("\n\nFull-Time Employment Rates (Weighted by PERWT):")
ft_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT')
).unstack()
ft_weighted.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)']
ft_weighted.index = ['Control (31-35)', 'Treatment (26-30)']
print(ft_weighted.round(4))

did_weighted = (ft_weighted.iloc[1, 1] - ft_weighted.iloc[1, 0]) - (ft_weighted.iloc[0, 1] - ft_weighted.iloc[0, 0])
print(f"\nSimple DiD Estimate (Weighted): {did_weighted:.4f}")

# Demographic comparison
print("\n\nDemographic Characteristics by Treatment Status (Pre-Period):")
pre_data = df[df['AFTER'] == 0]
demo_vars = ['AGE', 'SEX', 'FAMSIZE', 'NCHILD', 'EDUC']
for var in demo_vars:
    if var in pre_data.columns:
        treated_mean = pre_data[pre_data['ELIGIBLE'] == 1][var].mean()
        control_mean = pre_data[pre_data['ELIGIBLE'] == 0][var].mean()
        print(f"  {var}: Treatment = {treated_mean:.2f}, Control = {control_mean:.2f}")

# =============================================================================
# 4. PARALLEL TRENDS CHECK
# =============================================================================
print("\n" + "=" * 80)
print("4. PARALLEL TRENDS ANALYSIS")
print("=" * 80)

print("\nFull-Time Employment Rate by Year and Group (Weighted):")
yearly_ft = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT')
).unstack()
yearly_ft.columns = ['Control (31-35)', 'Treatment (26-30)']
yearly_ft['Difference'] = yearly_ft['Treatment (26-30)'] - yearly_ft['Control (31-35)']
print(yearly_ft.round(4))

# Pre-treatment trend test
pre_years = [2008, 2009, 2010, 2011]
pre_trends = yearly_ft.loc[pre_years, 'Difference'].values
years_numeric = np.array([0, 1, 2, 3])
slope, intercept, r_value, p_value, std_err = stats.linregress(years_numeric, pre_trends)
print(f"\nPre-Treatment Trend Test:")
print(f"  Slope of difference: {slope:.4f}")
print(f"  Standard error: {std_err:.4f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Interpretation: {'No significant pre-trend' if p_value > 0.05 else 'Warning: Significant pre-trend detected'}")

# =============================================================================
# 5. MAIN DiD REGRESSION MODELS
# =============================================================================
print("\n" + "=" * 80)
print("5. DIFFERENCE-IN-DIFFERENCES REGRESSION MODELS")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (OLS, no weights)
print("\n--- MODEL 1: Basic DiD (OLS, Unweighted) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model1.rsquared:.4f}")
print(f"N: {int(model1.nobs):,}")

# Model 2: Basic DiD (WLS, weighted)
print("\n--- MODEL 2: Basic DiD (WLS, Weighted by PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with Year Fixed Effects
print("\n--- MODEL 3: DiD with Year Fixed Effects (Weighted) ---")
df['YEAR_factor'] = pd.Categorical(df['YEAR'])
model3 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model3.rsquared:.4f}")

# Model 4: DiD with State Fixed Effects
print("\n--- MODEL 4: DiD with State Fixed Effects (Weighted) ---")
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model4.rsquared:.4f}")

# Model 5: DiD with Year and State Fixed Effects
print("\n--- MODEL 5: DiD with Year and State Fixed Effects (Weighted) ---")
model5 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model5.rsquared:.4f}")

# =============================================================================
# 6. MODEL WITH DEMOGRAPHIC COVARIATES
# =============================================================================
print("\n" + "=" * 80)
print("6. DiD MODEL WITH DEMOGRAPHIC COVARIATES")
print("=" * 80)

# Prepare covariates
# SEX: 1=Male, 2=Female in IPUMS
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# MARST: Marital status (1=married spouse present)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Check for education recode variable
if 'EDUC_RECODE' in df.columns:
    print("\nUsing EDUC_RECODE for education controls")
else:
    # Create education categories based on EDUC
    # EDUC: 0=N/A, 1=Nursery-4, 2=5-8, 3=9, 4=10, 5=11, 6=12/HS, 7=1yr college, 8=2yr college, 9=3yr college, 10=4yr/BA, 11=5+yr
    df['HS_OR_MORE'] = (df['EDUC'] >= 6).astype(int)
    df['SOME_COLLEGE'] = (df['EDUC'] >= 7).astype(int)
    df['BA_OR_MORE'] = (df['EDUC'] >= 10).astype(int)

# Metro status
df['METRO_AREA'] = (df['METRO'] >= 2).astype(int)

# Model 6: Full model with covariates
print("\n--- MODEL 6: DiD with Covariates (Weighted) ---")
try:
    model6 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + HS_OR_MORE + SOME_COLLEGE + BA_OR_MORE + METRO_AREA',
                     data=df, weights=df['PERWT']).fit()
    print(f"DiD Coefficient (ELIGIBLE_AFTER): {model6.params['ELIGIBLE_AFTER']:.4f}")
    print(f"Standard Error: {model6.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
    print(f"P-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
    print(f"R-squared: {model6.rsquared:.4f}")

    print("\nCovariate coefficients:")
    covariates = ['FEMALE', 'MARRIED', 'FAMSIZE', 'NCHILD', 'HS_OR_MORE', 'SOME_COLLEGE', 'BA_OR_MORE', 'METRO_AREA']
    for cov in covariates:
        if cov in model6.params:
            print(f"  {cov}: {model6.params[cov]:.4f} (SE: {model6.bse[cov]:.4f})")
except Exception as e:
    print(f"Error fitting model 6: {e}")

# =============================================================================
# 7. ROBUST STANDARD ERRORS
# =============================================================================
print("\n" + "=" * 80)
print("7. ROBUST AND CLUSTERED STANDARD ERRORS")
print("=" * 80)

# Model with robust (HC1) standard errors
print("\n--- DiD with Robust Standard Errors (HC1) ---")
model_robust = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER',
                       data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_robust.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model_robust.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_robust.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_robust.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model_robust.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model with state-clustered standard errors
print("\n--- DiD with State-Clustered Standard Errors ---")
model_cluster = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER',
                        data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model_cluster.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {model_cluster.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_cluster.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_cluster.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model_cluster.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# 8. HETEROGENEITY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("8. HETEROGENEITY ANALYSIS")
print("=" * 80)

# By Sex
print("\n--- Subgroup Analysis by Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    sub_df = df[df['SEX'] == sex_val]
    model_sub = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER',
                        data=sub_df, weights=sub_df['PERWT']).fit()
    print(f"{sex_name}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}), N = {len(sub_df):,}")

# By Education
print("\n--- Subgroup Analysis by Education ---")
df['LOW_EDUC'] = (df['EDUC'] < 6).astype(int)
for educ_val, educ_name in [(1, 'Less than HS'), (0, 'HS or more')]:
    sub_df = df[df['LOW_EDUC'] == educ_val]
    model_sub = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER',
                        data=sub_df, weights=sub_df['PERWT']).fit()
    print(f"{educ_name}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}), N = {len(sub_df):,}")

# =============================================================================
# 9. EVENT STUDY / DYNAMIC EFFECTS
# =============================================================================
print("\n" + "=" * 80)
print("9. EVENT STUDY ANALYSIS (Dynamic Treatment Effects)")
print("=" * 80)

# Create year-specific treatment interactions (base year = 2011)
years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    df[f'ELIGIBLE_YEAR_{year}'] = (df['ELIGIBLE'] == 1) & (df['YEAR'] == year)
    df[f'ELIGIBLE_YEAR_{year}'] = df[f'ELIGIBLE_YEAR_{year}'].astype(int)

# Event study regression (omit 2011 as reference)
formula = 'FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_YEAR_2008 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 + ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016'
model_event = smf.wls(formula, data=df, weights=df['PERWT']).fit()

print("\nEvent Study Coefficients (Reference Year: 2011):")
event_vars = ['ELIGIBLE_YEAR_2008', 'ELIGIBLE_YEAR_2009', 'ELIGIBLE_YEAR_2010',
              'ELIGIBLE_YEAR_2013', 'ELIGIBLE_YEAR_2014', 'ELIGIBLE_YEAR_2015', 'ELIGIBLE_YEAR_2016']
for var in event_vars:
    if var in model_event.params:
        coef = model_event.params[var]
        se = model_event.bse[var]
        ci_low = model_event.conf_int().loc[var, 0]
        ci_high = model_event.conf_int().loc[var, 1]
        year = var.split('_')[-1]
        print(f"  {year}: {coef:.4f} (SE: {se:.4f}) [{ci_low:.4f}, {ci_high:.4f}]")

# =============================================================================
# 10. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("10. SUMMARY OF MAIN RESULTS")
print("=" * 80)

print("\n=== PREFERRED SPECIFICATION (Model 5: Year + State FE, Weighted) ===")
print(f"DiD Estimate: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample Size: {int(model5.nobs):,}")
print(f"R-squared: {model5.rsquared:.4f}")

# Statistical significance
sig_level = 0.05
is_significant = model5.pvalues['ELIGIBLE_AFTER'] < sig_level
print(f"\nStatistically Significant at 5% level: {'Yes' if is_significant else 'No'}")

# Effect interpretation
baseline_ft = ft_weighted.iloc[1, 0]  # Treatment group pre-period FT rate
pct_change = (model5.params['ELIGIBLE_AFTER'] / baseline_ft) * 100
print(f"\nInterpretation:")
print(f"  DACA eligibility is associated with a {model5.params['ELIGIBLE_AFTER']:.4f} ({model5.params['ELIGIBLE_AFTER']*100:.2f} percentage point)")
print(f"  {'increase' if model5.params['ELIGIBLE_AFTER'] > 0 else 'decrease'} in the probability of full-time employment.")
print(f"  This represents approximately a {abs(pct_change):.1f}% change relative to the pre-DACA baseline")
print(f"  full-time employment rate of {baseline_ft:.4f} ({baseline_ft*100:.2f}%) for the treatment group.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# =============================================================================
# 11. EXPORT RESULTS FOR REPORT
# =============================================================================

# Save key results to a file for the LaTeX report
results_summary = {
    'Model 1 (Basic)': {
        'coef': model1.params['ELIGIBLE_AFTER'],
        'se': model1.bse['ELIGIBLE_AFTER'],
        'ci_low': model1.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model1.conf_int().loc['ELIGIBLE_AFTER', 1],
        'pval': model1.pvalues['ELIGIBLE_AFTER'],
        'n': int(model1.nobs),
        'r2': model1.rsquared
    },
    'Model 2 (Weighted)': {
        'coef': model2.params['ELIGIBLE_AFTER'],
        'se': model2.bse['ELIGIBLE_AFTER'],
        'ci_low': model2.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model2.conf_int().loc['ELIGIBLE_AFTER', 1],
        'pval': model2.pvalues['ELIGIBLE_AFTER'],
        'n': int(model2.nobs),
        'r2': model2.rsquared
    },
    'Model 3 (Year FE)': {
        'coef': model3.params['ELIGIBLE_AFTER'],
        'se': model3.bse['ELIGIBLE_AFTER'],
        'ci_low': model3.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model3.conf_int().loc['ELIGIBLE_AFTER', 1],
        'pval': model3.pvalues['ELIGIBLE_AFTER'],
        'n': int(model3.nobs),
        'r2': model3.rsquared
    },
    'Model 4 (State FE)': {
        'coef': model4.params['ELIGIBLE_AFTER'],
        'se': model4.bse['ELIGIBLE_AFTER'],
        'ci_low': model4.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model4.conf_int().loc['ELIGIBLE_AFTER', 1],
        'pval': model4.pvalues['ELIGIBLE_AFTER'],
        'n': int(model4.nobs),
        'r2': model4.rsquared
    },
    'Model 5 (Year+State FE)': {
        'coef': model5.params['ELIGIBLE_AFTER'],
        'se': model5.bse['ELIGIBLE_AFTER'],
        'ci_low': model5.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model5.conf_int().loc['ELIGIBLE_AFTER', 1],
        'pval': model5.pvalues['ELIGIBLE_AFTER'],
        'n': int(model5.nobs),
        'r2': model5.rsquared
    },
    'Model 5 (Clustered SE)': {
        'coef': model_cluster.params['ELIGIBLE_AFTER'],
        'se': model_cluster.bse['ELIGIBLE_AFTER'],
        'ci_low': model_cluster.conf_int().loc['ELIGIBLE_AFTER', 0],
        'ci_high': model_cluster.conf_int().loc['ELIGIBLE_AFTER', 1],
        'pval': model_cluster.pvalues['ELIGIBLE_AFTER'],
        'n': int(model_cluster.nobs),
        'r2': model_cluster.rsquared
    }
}

# Print formatted table for LaTeX
print("\n\nFORMATTED RESULTS TABLE FOR LATEX:")
print("-" * 100)
print(f"{'Model':<25} {'Coef':>10} {'SE':>10} {'95% CI':>22} {'P-value':>10} {'N':>10} {'R2':>8}")
print("-" * 100)
for model_name, res in results_summary.items():
    ci_str = f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]"
    print(f"{model_name:<25} {res['coef']:>10.4f} {res['se']:>10.4f} {ci_str:>22} {res['pval']:>10.4f} {res['n']:>10,} {res['r2']:>8.4f}")
print("-" * 100)

# Save detailed results
print("\n\nEVENT STUDY RESULTS:")
print("-" * 60)
print(f"{'Year':>6} {'Coefficient':>12} {'SE':>10} {'95% CI':>24}")
print("-" * 60)
print(f"{'2011':>6} {'(reference)':>12} {'-':>10} {'-':>24}")
for var in event_vars:
    year = var.split('_')[-1]
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_str = f"[{model_event.conf_int().loc[var, 0]:.4f}, {model_event.conf_int().loc[var, 1]:.4f}]"
    print(f"{year:>6} {coef:>12.4f} {se:>10.4f} {ci_str:>24}")
print("-" * 60)

# Save full-time rates by year
print("\n\nFULL-TIME EMPLOYMENT RATES BY YEAR (WEIGHTED):")
print(yearly_ft.round(4).to_string())
