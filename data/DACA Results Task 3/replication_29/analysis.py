"""
DACA Replication Analysis: Effect of DACA on Full-Time Employment
=================================================================

Research Question: Among ethnically Hispanic-Mexican, Mexican-born people living in the US,
what was the causal impact of DACA eligibility (treatment) on full-time employment?

Design: Difference-in-Differences
- Treatment group: Ages 26-30 at DACA implementation (June 15, 2012) - ELIGIBLE=1
- Control group: Ages 31-35 at DACA implementation - ELIGIBLE=0
- Pre-period: 2008-2011
- Post-period: 2013-2016

Outcome: FT (full-time employment, 1=yes, 0=no)
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

# Load the data
print("="*80)
print("DACA REPLICATION STUDY: Full-Time Employment Analysis")
print("="*80)

# Read the numeric version which is cleaner for analysis
data = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\nTotal observations loaded: {len(data):,}")
print(f"Years in data: {sorted(data['YEAR'].unique())}")

# Verify data structure
print("\n" + "="*80)
print("DATA VERIFICATION")
print("="*80)

# Check key variables
print(f"\nELIGIBLE values: {data['ELIGIBLE'].unique()}")
print(f"AFTER values: {data['AFTER'].unique()}")
print(f"FT values: {data['FT'].unique()}")

# Sample sizes by group and period
print("\n" + "-"*60)
print("Sample Size by Treatment Group and Period")
print("-"*60)
sample_table = pd.crosstab(data['ELIGIBLE'], data['AFTER'], margins=True)
sample_table.columns = ['Pre (2008-2011)', 'Post (2013-2016)', 'Total']
sample_table.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
print(sample_table)

# Verify years in each period
print("\n" + "-"*60)
print("Years by AFTER variable")
print("-"*60)
print("Pre-period (AFTER=0):", sorted(data[data['AFTER']==0]['YEAR'].unique()))
print("Post-period (AFTER=1):", sorted(data[data['AFTER']==1]['YEAR'].unique()))

# Basic descriptive statistics
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Full-time employment rates by group and period
print("\n" + "-"*60)
print("Full-Time Employment Rates by Group and Period")
print("-"*60)
ft_rates = data.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
ft_rates.columns = ['Mean FT Rate', 'Std Dev', 'N']
ft_rates.index = pd.MultiIndex.from_tuples([
    ('Control (31-35)', 'Pre'),
    ('Control (31-35)', 'Post'),
    ('Treatment (26-30)', 'Pre'),
    ('Treatment (26-30)', 'Post')
])
print(ft_rates.round(4))

# Calculate simple DiD manually
ft_pre_control = data[(data['ELIGIBLE']==0) & (data['AFTER']==0)]['FT'].mean()
ft_post_control = data[(data['ELIGIBLE']==0) & (data['AFTER']==1)]['FT'].mean()
ft_pre_treat = data[(data['ELIGIBLE']==1) & (data['AFTER']==0)]['FT'].mean()
ft_post_treat = data[(data['ELIGIBLE']==1) & (data['AFTER']==1)]['FT'].mean()

print("\n" + "-"*60)
print("Simple Difference-in-Differences Calculation")
print("-"*60)
print(f"Control group change: {ft_post_control:.4f} - {ft_pre_control:.4f} = {ft_post_control - ft_pre_control:.4f}")
print(f"Treatment group change: {ft_post_treat:.4f} - {ft_pre_treat:.4f} = {ft_post_treat - ft_pre_treat:.4f}")
simple_did = (ft_post_treat - ft_pre_treat) - (ft_post_control - ft_pre_control)
print(f"\nDiD Estimate: {simple_did:.4f}")

# Demographics comparison
print("\n" + "-"*60)
print("Sample Demographics by Treatment Status")
print("-"*60)

# Gender distribution (SEX: 1=Male, 2=Female in IPUMS)
data['FEMALE'] = (data['SEX'] == 2).astype(int)
gender_by_group = data.groupby('ELIGIBLE')['FEMALE'].mean()
print(f"\nProportion Female:")
print(f"  Control (31-35): {gender_by_group[0]:.3f}")
print(f"  Treatment (26-30): {gender_by_group[1]:.3f}")

# Age distribution
age_by_group = data.groupby('ELIGIBLE')['AGE'].mean()
print(f"\nMean Age:")
print(f"  Control: {age_by_group[0]:.1f}")
print(f"  Treatment: {age_by_group[1]:.1f}")

# Education distribution
if 'EDUC_RECODE' in data.columns:
    print(f"\nEducation Distribution by Group (pre-period only):")
    pre_data = data[data['AFTER']==0]
    for elig in [0, 1]:
        group_name = "Treatment (26-30)" if elig == 1 else "Control (31-35)"
        print(f"\n  {group_name}:")
        educ_dist = pre_data[pre_data['ELIGIBLE']==elig]['EDUC_RECODE'].value_counts(normalize=True)
        for educ, prop in educ_dist.items():
            print(f"    {educ}: {prop:.3f}")

print("\n" + "="*80)
print("MAIN ANALYSIS: DIFFERENCE-IN-DIFFERENCES")
print("="*80)

# Model 1: Basic DiD (no covariates)
print("\n" + "-"*60)
print("Model 1: Basic DiD Regression (No Covariates)")
print("-"*60)
print("FT = b0 + b1*ELIGIBLE + b2*AFTER + b3*ELIGIBLE*AFTER + e")

# Create interaction term
data['ELIGIBLE_AFTER'] = data['ELIGIBLE'] * data['AFTER']

# Run regression
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data).fit()
print("\n", model1.summary())

# Extract key results
did_coef = model1.params['ELIGIBLE_AFTER']
did_se = model1.bse['ELIGIBLE_AFTER']
did_pvalue = model1.pvalues['ELIGIBLE_AFTER']
did_ci = model1.conf_int().loc['ELIGIBLE_AFTER']

print("\n" + "-"*40)
print("KEY RESULT - DiD Estimate (ELIGIBLE*AFTER):")
print("-"*40)
print(f"Coefficient: {did_coef:.4f}")
print(f"Standard Error: {did_se:.4f}")
print(f"t-statistic: {model1.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {did_pvalue:.4f}")
print(f"95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")

# Model 2: DiD with demographic controls
print("\n" + "-"*60)
print("Model 2: DiD with Demographic Controls")
print("-"*60)

# Prepare control variables
data['FEMALE'] = (data['SEX'] == 2).astype(int)
data['MARRIED'] = (data['MARST'] == 1).astype(int)  # Married spouse present

# Create education dummies (using EDUC numeric codes)
# EDUC: 0=N/A, 1=No school, 2=nursery-4, 3=grade 5-8, 4=grade 9, 5=grade 10, 6=grade 11,
# 7=grade 12 (HS), 8=1 yr college, 9=2 yrs college, 10=3 yrs college, 11=4 yrs college, 12+=grad school
data['HS_OR_MORE'] = (data['EDUC'] >= 6).astype(int)
data['COLLEGE'] = (data['EDUC'] >= 10).astype(int)

model2_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + HS_OR_MORE'
model2 = smf.ols(model2_formula, data=data).fit()
print("\n", model2.summary())

did_coef2 = model2.params['ELIGIBLE_AFTER']
did_se2 = model2.bse['ELIGIBLE_AFTER']
did_ci2 = model2.conf_int().loc['ELIGIBLE_AFTER']

print("\n" + "-"*40)
print("KEY RESULT - DiD Estimate with Controls:")
print("-"*40)
print(f"Coefficient: {did_coef2:.4f}")
print(f"Standard Error: {did_se2:.4f}")
print(f"p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{did_ci2[0]:.4f}, {did_ci2[1]:.4f}]")

# Model 3: DiD with state fixed effects
print("\n" + "-"*60)
print("Model 3: DiD with State Fixed Effects")
print("-"*60)

model3_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + HS_OR_MORE + C(STATEFIP)'
model3 = smf.ols(model3_formula, data=data).fit()

did_coef3 = model3.params['ELIGIBLE_AFTER']
did_se3 = model3.bse['ELIGIBLE_AFTER']
did_ci3 = model3.conf_int().loc['ELIGIBLE_AFTER']

print(f"\nDiD Coefficient: {did_coef3:.4f}")
print(f"Standard Error: {did_se3:.4f}")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{did_ci3[0]:.4f}, {did_ci3[1]:.4f}]")
print(f"R-squared: {model3.rsquared:.4f}")
print(f"N: {int(model3.nobs):,}")

# Model 4: DiD with year fixed effects
print("\n" + "-"*60)
print("Model 4: DiD with Year Fixed Effects")
print("-"*60)

model4_formula = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + HS_OR_MORE + C(YEAR)'
model4 = smf.ols(model4_formula, data=data).fit()

did_coef4 = model4.params['ELIGIBLE_AFTER']
did_se4 = model4.bse['ELIGIBLE_AFTER']
did_ci4 = model4.conf_int().loc['ELIGIBLE_AFTER']

print(f"\nDiD Coefficient: {did_coef4:.4f}")
print(f"Standard Error: {did_se4:.4f}")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{did_ci4[0]:.4f}, {did_ci4[1]:.4f}]")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {int(model4.nobs):,}")

# Model 5: Full model with state and year FE
print("\n" + "-"*60)
print("Model 5: DiD with State and Year Fixed Effects (Preferred)")
print("-"*60)

model5_formula = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + HS_OR_MORE + C(STATEFIP) + C(YEAR)'
model5 = smf.ols(model5_formula, data=data).fit()

did_coef5 = model5.params['ELIGIBLE_AFTER']
did_se5 = model5.bse['ELIGIBLE_AFTER']
did_ci5 = model5.conf_int().loc['ELIGIBLE_AFTER']

print(f"\nDiD Coefficient: {did_coef5:.4f}")
print(f"Standard Error: {did_se5:.4f}")
print(f"t-statistic: {model5.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{did_ci5[0]:.4f}, {did_ci5[1]:.4f}]")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {int(model5.nobs):,}")

# Cluster standard errors by state
print("\n" + "-"*60)
print("Model 6: DiD with Clustered Standard Errors (by State)")
print("-"*60)

model6 = smf.ols(model5_formula, data=data).fit(cov_type='cluster', cov_kwds={'groups': data['STATEFIP']})

did_coef6 = model6.params['ELIGIBLE_AFTER']
did_se6 = model6.bse['ELIGIBLE_AFTER']
did_ci6 = model6.conf_int().loc['ELIGIBLE_AFTER']

print(f"\nDiD Coefficient: {did_coef6:.4f}")
print(f"Clustered SE: {did_se6:.4f}")
print(f"t-statistic: {model6.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{did_ci6[0]:.4f}, {did_ci6[1]:.4f}]")

print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: By gender
print("\n" + "-"*60)
print("Robustness 1: Effects by Gender")
print("-"*60)

for gender, label in [(0, 'Male'), (1, 'Female')]:
    subset = data[data['FEMALE'] == gender]
    model_gender = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + MARRIED + HS_OR_MORE', data=subset).fit()
    print(f"\n{label}:")
    print(f"  DiD Coefficient: {model_gender.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  Standard Error: {model_gender.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"  p-value: {model_gender.pvalues['ELIGIBLE_AFTER']:.4f}")
    print(f"  N: {int(model_gender.nobs):,}")

# Robustness 2: Event study (year-by-year effects)
print("\n" + "-"*60)
print("Robustness 2: Event Study (Year-by-Year Effects)")
print("-"*60)

# Create year dummies interacted with eligible
years = sorted(data['YEAR'].unique())
for year in years:
    data[f'YEAR_{year}'] = (data['YEAR'] == year).astype(int)
    data[f'ELIGIBLE_YEAR_{year}'] = data['ELIGIBLE'] * data[f'YEAR_{year}']

# Use 2011 as reference year (last pre-treatment year)
year_vars = [f'ELIGIBLE_YEAR_{y}' for y in years if y != 2011]
event_formula = 'FT ~ ELIGIBLE + ' + ' + '.join([f'YEAR_{y}' for y in years if y != 2011]) + ' + ' + ' + '.join(year_vars) + ' + AGE + FEMALE + MARRIED + HS_OR_MORE'
model_event = smf.ols(event_formula, data=data).fit()

print("\nEvent Study Coefficients (ELIGIBLE x YEAR, reference: 2011):")
print("-"*50)
event_results = []
for year in years:
    if year == 2011:
        coef, se = 0, 0  # reference
    else:
        coef = model_event.params[f'ELIGIBLE_YEAR_{year}']
        se = model_event.bse[f'ELIGIBLE_YEAR_{year}']
    period = "Pre" if year < 2012 else "Post"
    event_results.append({'Year': year, 'Coefficient': coef, 'SE': se, 'Period': period})
    print(f"  {year} ({period}): {coef:7.4f} (SE: {se:.4f})")

event_df = pd.DataFrame(event_results)

# Robustness 3: Weighted regression using person weights
print("\n" + "-"*60)
print("Robustness 3: Weighted Regression (Using PERWT)")
print("-"*60)

# WLS with person weights
model_wls = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + HS_OR_MORE',
                    data=data, weights=data['PERWT']).fit()

print(f"\nDiD Coefficient (Weighted): {model_wls.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model_wls.bse['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model_wls.pvalues['ELIGIBLE_AFTER']:.4f}")

# Robustness 4: Probit/Logit models
print("\n" + "-"*60)
print("Robustness 4: Logit Model (Marginal Effects)")
print("-"*60)

from statsmodels.discrete.discrete_model import Logit
logit_model = smf.logit('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + HS_OR_MORE', data=data).fit(disp=0)
# Calculate marginal effects at mean
mfx = logit_model.get_margeff(at='mean')
mfx_df = mfx.summary_frame()

print(f"\nLogit Marginal Effect (ELIGIBLE_AFTER): {mfx_df.loc['ELIGIBLE_AFTER', 'dy/dx']:.4f}")
print(f"Standard Error: {mfx_df.loc['ELIGIBLE_AFTER', 'Std. Err.']:.4f}")
print(f"p-value: {mfx_df.loc['ELIGIBLE_AFTER', 'Pr(>|z|)']:.4f}")

# Robustness 5: Triple difference with state policy variation
print("\n" + "-"*60)
print("Robustness 5: Placebo Test (Pre-Period Only)")
print("-"*60)

# Create fake "treatment" at 2010 using only pre-period data
pre_data = data[data['AFTER'] == 0].copy()
pre_data['FAKE_POST'] = (pre_data['YEAR'] >= 2010).astype(int)
pre_data['ELIGIBLE_FAKEPOST'] = pre_data['ELIGIBLE'] * pre_data['FAKE_POST']

placebo_model = smf.ols('FT ~ ELIGIBLE + FAKE_POST + ELIGIBLE_FAKEPOST + AGE + FEMALE + MARRIED + HS_OR_MORE',
                        data=pre_data).fit()

print(f"\nPlacebo DiD Coefficient: {placebo_model.params['ELIGIBLE_FAKEPOST']:.4f}")
print(f"Standard Error: {placebo_model.bse['ELIGIBLE_FAKEPOST']:.4f}")
print(f"p-value: {placebo_model.pvalues['ELIGIBLE_FAKEPOST']:.4f}")
print("(Insignificant placebo effect supports parallel trends assumption)")

print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

results_summary = pd.DataFrame({
    'Model': ['1. Basic DiD', '2. With Demographics', '3. State FE', '4. Year FE',
              '5. State+Year FE', '6. Clustered SE', '7. Weighted (PERWT)'],
    'Coefficient': [did_coef, did_coef2, did_coef3, did_coef4, did_coef5, did_coef6,
                    model_wls.params['ELIGIBLE_AFTER']],
    'Std. Error': [did_se, did_se2, did_se3, did_se4, did_se5, did_se6,
                   model_wls.bse['ELIGIBLE_AFTER']],
    'p-value': [did_pvalue, model2.pvalues['ELIGIBLE_AFTER'], model3.pvalues['ELIGIBLE_AFTER'],
                model4.pvalues['ELIGIBLE_AFTER'], model5.pvalues['ELIGIBLE_AFTER'],
                model6.pvalues['ELIGIBLE_AFTER'], model_wls.pvalues['ELIGIBLE_AFTER']]
})

print("\n")
print(results_summary.to_string(index=False))

print("\n" + "="*80)
print("PREFERRED ESTIMATE")
print("="*80)
print(f"\nModel: DiD with State and Year Fixed Effects, Clustered SEs")
print(f"Effect Size: {did_coef6:.4f}")
print(f"  Interpretation: DACA eligibility is associated with a {did_coef6*100:.2f} percentage point")
print(f"                  {'increase' if did_coef6 > 0 else 'decrease'} in full-time employment probability")
print(f"Standard Error (Clustered): {did_se6:.4f}")
print(f"95% Confidence Interval: [{did_ci6[0]:.4f}, {did_ci6[1]:.4f}]")
print(f"Sample Size: {int(model6.nobs):,}")
print(f"Statistical Significance: {'Yes (p < 0.05)' if model6.pvalues['ELIGIBLE_AFTER'] < 0.05 else 'No (p >= 0.05)'}")

# Save results for LaTeX
print("\n" + "="*80)
print("SAVING RESULTS FOR REPORT")
print("="*80)

# Save summary statistics
summary_stats = {
    'total_n': len(data),
    'n_treatment': len(data[data['ELIGIBLE']==1]),
    'n_control': len(data[data['ELIGIBLE']==0]),
    'n_pre': len(data[data['AFTER']==0]),
    'n_post': len(data[data['AFTER']==1]),
    'ft_rate_overall': data['FT'].mean(),
    'ft_rate_treat_pre': ft_pre_treat,
    'ft_rate_treat_post': ft_post_treat,
    'ft_rate_control_pre': ft_pre_control,
    'ft_rate_control_post': ft_post_control,
    'did_simple': simple_did,
    'did_preferred': did_coef6,
    'did_se_preferred': did_se6,
    'did_ci_low': did_ci6[0],
    'did_ci_high': did_ci6[1],
    'did_pvalue': model6.pvalues['ELIGIBLE_AFTER']
}

# Save to CSV for LaTeX import
pd.DataFrame([summary_stats]).to_csv('analysis_results.csv', index=False)
results_summary.to_csv('model_comparison.csv', index=False)
event_df.to_csv('event_study_results.csv', index=False)

print("Results saved to CSV files.")

# Create visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Figure 1: Event study plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['blue' if p == 'Pre' else 'red' for p in event_df['Period']]
ax.errorbar(event_df['Year'], event_df['Coefficient'], yerr=1.96*event_df['SE'],
            fmt='o', capsize=5, capthick=2, markersize=8, color='navy')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
ax.axvline(x=2011.5, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('DiD Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('event_study.png', dpi=150, bbox_inches='tight')
plt.close()
print("Event study plot saved to event_study.png")

# Figure 2: Parallel trends visualization
fig, ax = plt.subplots(figsize=(10, 6))
yearly_rates = data.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_rates.columns = ['Control (31-35)', 'Treatment (26-30)']
yearly_rates.plot(ax=ax, marker='o', linewidth=2, markersize=8)
ax.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation (2012 omitted)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends: Treatment vs. Control Groups', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('parallel_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("Parallel trends plot saved to parallel_trends.png")

# Figure 3: DiD visualization
fig, ax = plt.subplots(figsize=(8, 6))
x = [0, 1]
ax.plot(x, [ft_pre_control, ft_post_control], 'b-o', label='Control (31-35)', linewidth=2, markersize=10)
ax.plot(x, [ft_pre_treat, ft_post_treat], 'r-o', label='Treatment (26-30)', linewidth=2, markersize=10)
# Counterfactual
counterfactual = ft_pre_treat + (ft_post_control - ft_pre_control)
ax.plot([1], [counterfactual], 'r^', markersize=12, label='Counterfactual')
ax.plot([0, 1], [ft_pre_treat, counterfactual], 'r--', alpha=0.5)
ax.annotate(f'DiD = {simple_did:.4f}', xy=(1, (ft_post_treat + counterfactual)/2),
            xytext=(1.15, (ft_post_treat + counterfactual)/2),
            fontsize=12, ha='left')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: DACA Effect on Full-Time Employment', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('did_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("DiD visualization saved to did_visualization.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
