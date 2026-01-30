"""
DACA Replication Analysis
Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born people in the US?

Treatment: ELIGIBLE = 1 (ages 26-30 at June 2012)
Control: ELIGIBLE = 0 (ages 31-35 at June 2012)
Pre-period: AFTER = 0 (2008-2011)
Post-period: AFTER = 1 (2013-2016)
Outcome: FT (full-time employment, 35+ hours/week)

Method: Difference-in-Differences (DiD)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set output directory
output_dir = r'C:\Users\seraf\DACA Results Task 3\replication_09'

# Load data
print("="*60)
print("DACA REPLICATION ANALYSIS")
print("="*60)
print("\n1. LOADING DATA")
print("-"*40)

df = pd.read_csv(f'{output_dir}/data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Total observations: {len(df):,}")
print(f"Variables: {len(df.columns)}")

# Basic data validation
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")
print(f"Note: 2012 is excluded as treatment occurred mid-year")

# =============================================================================
# 2. SAMPLE DESCRIPTION
# =============================================================================
print("\n" + "="*60)
print("2. SAMPLE DESCRIPTION")
print("="*60)

# Treatment and control groups
print("\nTreatment vs Control Groups:")
print(f"  ELIGIBLE=1 (Treated, ages 26-30 in June 2012): {(df['ELIGIBLE']==1).sum():,} obs")
print(f"  ELIGIBLE=0 (Control, ages 31-35 in June 2012): {(df['ELIGIBLE']==0).sum():,} obs")

# Pre vs Post periods
print("\nPre vs Post Treatment Periods:")
print(f"  AFTER=0 (Pre: 2008-2011): {(df['AFTER']==0).sum():,} obs")
print(f"  AFTER=1 (Post: 2013-2016): {(df['AFTER']==1).sum():,} obs")

# Cross-tabulation
print("\nCross-tabulation (ELIGIBLE x AFTER):")
ct = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
ct.index = ['Control (31-35)', 'Treated (26-30)', 'Total']
ct.columns = ['Pre (2008-11)', 'Post (2013-16)', 'Total']
print(ct)

# Outcome variable
print("\nOutcome Variable - Full-Time Employment (FT):")
print(f"  FT=1: {(df['FT']==1).sum():,} ({100*(df['FT']==1).mean():.1f}%)")
print(f"  FT=0: {(df['FT']==0).sum():,} ({100*(df['FT']==0).mean():.1f}%)")

# =============================================================================
# 3. SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*60)
print("3. SUMMARY STATISTICS")
print("="*60)

# Demographics by group
summary_vars = ['AGE', 'SEX', 'FAMSIZE', 'NCHILD', 'UHRSWORK', 'FT']

def summarize_group(data, name):
    print(f"\n{name}:")
    for var in summary_vars:
        if var in data.columns:
            print(f"  {var}: mean={data[var].mean():.3f}, sd={data[var].std():.3f}, n={data[var].notna().sum()}")

# By treatment status (pre-period for baseline comparison)
pre_treated = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]

print("\nBaseline Characteristics (Pre-Period Only):")
summarize_group(pre_treated, "Treated Group (Ages 26-30)")
summarize_group(pre_control, "Control Group (Ages 31-35)")

# Balance test
print("\nBalance Test (Pre-Period):")
for var in ['SEX', 'FAMSIZE', 'NCHILD']:
    if var in df.columns:
        t_stat, p_val = stats.ttest_ind(pre_treated[var].dropna(), pre_control[var].dropna())
        print(f"  {var}: t={t_stat:.3f}, p={p_val:.3f}")

# =============================================================================
# 4. RAW DIFFERENCE-IN-DIFFERENCES
# =============================================================================
print("\n" + "="*60)
print("4. RAW DIFFERENCE-IN-DIFFERENCES")
print("="*60)

# Calculate means by group and period
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
print("\nMean FT by Group and Period:")
print(means)

# Extract values for DiD calculation
# Control (ELIGIBLE=0)
control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
# Treated (ELIGIBLE=1)
treated_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
treated_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

# Calculate differences
diff_treated = treated_post - treated_pre
diff_control = control_post - control_pre
did_estimate = diff_treated - diff_control

print(f"\nControl Group (31-35 years old):")
print(f"  Pre-period mean:  {control_pre:.4f}")
print(f"  Post-period mean: {control_post:.4f}")
print(f"  Difference:       {diff_control:.4f}")

print(f"\nTreated Group (26-30 years old):")
print(f"  Pre-period mean:  {treated_pre:.4f}")
print(f"  Post-period mean: {treated_post:.4f}")
print(f"  Difference:       {diff_treated:.4f}")

print(f"\n*** DiD Estimate: {did_estimate:.4f} ***")
print(f"Interpretation: DACA eligibility increased full-time employment")
print(f"by {did_estimate*100:.2f} percentage points")

# =============================================================================
# 5. REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("5. REGRESSION ANALYSIS")
print("="*60)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD coefficient (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
# Create female indicator
df['FEMALE'] = (df['SEX'] == 2).astype(int)

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + FAMSIZE + NCHILD', data=df).fit()
print(model2.summary().tables[1])
print(f"\nDiD coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 3: DiD with robust standard errors
print("\n--- Model 3: DiD with Robust Standard Errors ---")
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model3.summary().tables[1])
print(f"\nDiD coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with Year Fixed Effects ---")
df['YEAR_factor'] = pd.Categorical(df['YEAR'])
model4 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR)', data=df).fit(cov_type='HC1')
print(f"\nDiD coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 5: DiD with state fixed effects
print("\n--- Model 5: DiD with State Fixed Effects ---")
model5 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)', data=df).fit(cov_type='HC1')
print(f"\nDiD coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 6: Full model with year and state FE + demographics
print("\n--- Model 6: Full Model (Year FE + State FE + Demographics) ---")
model6 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + FAMSIZE + NCHILD + C(YEAR) + C(STATEFIP)',
                 data=df).fit(cov_type='HC1')
print(f"\nDiD coefficient (ELIGIBLE_AFTER): {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# 6. WEIGHTED ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("6. WEIGHTED ANALYSIS (Using PERWT)")
print("="*60)

# Weighted DiD regression
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS

# Basic weighted model
print("\n--- Weighted DiD Model ---")
X = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']]
X = sm.add_constant(X)
y = df['FT']
weights = df['PERWT']

model_weighted = WLS(y, X, weights=weights).fit()
print(f"DiD coefficient (ELIGIBLE_AFTER): {model_weighted.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model_weighted.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_weighted.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_weighted.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model_weighted.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# 7. PARALLEL TRENDS CHECK
# =============================================================================
print("\n" + "="*60)
print("7. PARALLEL TRENDS CHECK")
print("="*60)

# Calculate yearly means by group
yearly_means = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_means.columns = ['Control (31-35)', 'Treated (26-30)']
print("\nYearly FT Rates by Group:")
print(yearly_means.round(4))

# Pre-trend analysis: regress on year trend in pre-period
pre_data = df[df['AFTER']==0].copy()
pre_data['YEAR_centered'] = pre_data['YEAR'] - 2008
pre_data['ELIGIBLE_YEAR'] = pre_data['ELIGIBLE'] * pre_data['YEAR_centered']

print("\n--- Pre-Trend Analysis ---")
trend_model = smf.ols('FT ~ ELIGIBLE + YEAR_centered + ELIGIBLE_YEAR', data=pre_data).fit()
print(f"ELIGIBLE x YEAR interaction (differential trend): {trend_model.params['ELIGIBLE_YEAR']:.4f}")
print(f"SE: {trend_model.bse['ELIGIBLE_YEAR']:.4f}")
print(f"p-value: {trend_model.pvalues['ELIGIBLE_YEAR']:.4f}")
if trend_model.pvalues['ELIGIBLE_YEAR'] > 0.05:
    print("=> Parallel trends assumption supported (no differential pre-trend)")
else:
    print("=> Warning: Some evidence of differential pre-trends")

# =============================================================================
# 8. EVENT STUDY
# =============================================================================
print("\n" + "="*60)
print("8. EVENT STUDY ANALYSIS")
print("="*60)

# Create year dummies and interactions
years = sorted(df['YEAR'].unique())
ref_year = 2011  # reference year (last pre-treatment year)

# Create event study data
for year in years:
    if year != ref_year:
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
        df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Event study regression
event_vars = [f'ELIGIBLE_YEAR_{y}' for y in years if y != ref_year]
formula = 'FT ~ ELIGIBLE + ' + ' + '.join([f'YEAR_{y}' for y in years if y != ref_year]) + ' + ' + ' + '.join(event_vars)
event_model = smf.ols(formula, data=df).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
event_results = []
for year in years:
    if year == ref_year:
        coef, se, pval = 0, 0, np.nan
    else:
        var = f'ELIGIBLE_YEAR_{year}'
        coef = event_model.params[var]
        se = event_model.bse[var]
        pval = event_model.pvalues[var]
    event_results.append({'Year': year, 'Coefficient': coef, 'SE': se, 'p-value': pval})
    print(f"  {year}: coef={coef:.4f}, SE={se:.4f}, p={pval:.4f}" if year != ref_year else f"  {year}: reference year")

event_df = pd.DataFrame(event_results)

# =============================================================================
# 9. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*60)
print("9. ROBUSTNESS CHECKS")
print("="*60)

# 9.1 By gender
print("\n--- 9.1 Heterogeneous Effects by Gender ---")
for sex, name in [(1, 'Male'), (2, 'Female')]:
    sub = df[df['SEX']==sex]
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=sub).fit(cov_type='HC1')
    print(f"{name}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE={model_sub.bse['ELIGIBLE_AFTER']:.4f}, p={model_sub.pvalues['ELIGIBLE_AFTER']:.4f})")

# 9.2 By education level (if available)
if 'EDUC_RECODE' in df.columns:
    print("\n--- 9.2 Heterogeneous Effects by Education ---")
    for edu in df['EDUC_RECODE'].dropna().unique()[:4]:
        sub = df[df['EDUC_RECODE']==edu]
        if len(sub) > 100:
            model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=sub).fit(cov_type='HC1')
            print(f"{edu}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE={model_sub.bse['ELIGIBLE_AFTER']:.4f}, n={len(sub)})")

# 9.3 Placebo test - use only pre-period data with fake treatment at 2010
print("\n--- 9.3 Placebo Test (Fake Treatment at 2010) ---")
pre_only = df[df['AFTER']==0].copy()
pre_only['PLACEBO_AFTER'] = (pre_only['YEAR'] >= 2010).astype(int)
pre_only['PLACEBO_DID'] = pre_only['ELIGIBLE'] * pre_only['PLACEBO_AFTER']
placebo_model = smf.ols('FT ~ ELIGIBLE + PLACEBO_AFTER + PLACEBO_DID', data=pre_only).fit(cov_type='HC1')
print(f"Placebo DiD: {placebo_model.params['PLACEBO_DID']:.4f}")
print(f"SE: {placebo_model.bse['PLACEBO_DID']:.4f}")
print(f"p-value: {placebo_model.pvalues['PLACEBO_DID']:.4f}")
if abs(placebo_model.pvalues['PLACEBO_DID']) > 0.05:
    print("=> Placebo test passed (no significant pre-treatment effect)")

# =============================================================================
# 10. CREATE SUMMARY TABLE FOR REPORT
# =============================================================================
print("\n" + "="*60)
print("10. SUMMARY OF RESULTS")
print("="*60)

results_summary = pd.DataFrame({
    'Model': ['Basic DiD', 'With Demographics', 'Robust SE', 'Year FE', 'State FE', 'Full Model', 'Weighted'],
    'Coefficient': [model1.params['ELIGIBLE_AFTER'], model2.params['ELIGIBLE_AFTER'],
                    model3.params['ELIGIBLE_AFTER'], model4.params['ELIGIBLE_AFTER'],
                    model5.params['ELIGIBLE_AFTER'], model6.params['ELIGIBLE_AFTER'],
                    model_weighted.params['ELIGIBLE_AFTER']],
    'SE': [model1.bse['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'],
           model3.bse['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'],
           model5.bse['ELIGIBLE_AFTER'], model6.bse['ELIGIBLE_AFTER'],
           model_weighted.bse['ELIGIBLE_AFTER']],
    'p-value': [model1.pvalues['ELIGIBLE_AFTER'], model2.pvalues['ELIGIBLE_AFTER'],
                model3.pvalues['ELIGIBLE_AFTER'], model4.pvalues['ELIGIBLE_AFTER'],
                model5.pvalues['ELIGIBLE_AFTER'], model6.pvalues['ELIGIBLE_AFTER'],
                model_weighted.pvalues['ELIGIBLE_AFTER']]
})
results_summary['95% CI Lower'] = results_summary['Coefficient'] - 1.96 * results_summary['SE']
results_summary['95% CI Upper'] = results_summary['Coefficient'] + 1.96 * results_summary['SE']
results_summary = results_summary.round(4)

print("\n*** MAIN RESULTS TABLE ***")
print(results_summary.to_string(index=False))

# =============================================================================
# 11. SAVE KEY RESULTS FOR LATEX
# =============================================================================
print("\n" + "="*60)
print("11. SAVING RESULTS")
print("="*60)

# Save summary statistics
summary_stats = {
    'total_n': len(df),
    'treated_n': (df['ELIGIBLE']==1).sum(),
    'control_n': (df['ELIGIBLE']==0).sum(),
    'pre_n': (df['AFTER']==0).sum(),
    'post_n': (df['AFTER']==1).sum(),
    'ft_mean': df['FT'].mean(),
    'treated_pre_ft': treated_pre,
    'treated_post_ft': treated_post,
    'control_pre_ft': control_pre,
    'control_post_ft': control_post,
    'did_raw': did_estimate,
    'did_basic': model1.params['ELIGIBLE_AFTER'],
    'did_basic_se': model1.bse['ELIGIBLE_AFTER'],
    'did_basic_pval': model1.pvalues['ELIGIBLE_AFTER'],
    'did_robust': model3.params['ELIGIBLE_AFTER'],
    'did_robust_se': model3.bse['ELIGIBLE_AFTER'],
    'did_robust_pval': model3.pvalues['ELIGIBLE_AFTER'],
    'did_full': model6.params['ELIGIBLE_AFTER'],
    'did_full_se': model6.bse['ELIGIBLE_AFTER'],
    'did_full_pval': model6.pvalues['ELIGIBLE_AFTER'],
}

# Save to file
results_file = f'{output_dir}/analysis_results.txt'
with open(results_file, 'w') as f:
    for key, val in summary_stats.items():
        f.write(f'{key}: {val}\n')
print(f"Results saved to {results_file}")

# Save tables for LaTeX
results_summary.to_csv(f'{output_dir}/results_table.csv', index=False)
yearly_means.to_csv(f'{output_dir}/yearly_means.csv')
event_df.to_csv(f'{output_dir}/event_study.csv', index=False)
print(f"Tables saved to CSV files")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\n*** PREFERRED ESTIMATE (Basic DiD with Robust SE) ***")
print(f"Effect: {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample size: {len(df):,}")
print(f"\nInterpretation: DACA eligibility is associated with a {model3.params['ELIGIBLE_AFTER']*100:.2f}")
print(f"percentage point increase in full-time employment probability.")
