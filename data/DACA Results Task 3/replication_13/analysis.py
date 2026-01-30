"""
DACA Replication Analysis - Session 13
Independent replication of the causal effect of DACA eligibility on full-time employment

Research Question:
What was the causal impact of DACA eligibility on the probability of full-time employment
among ethnically Hispanic-Mexican Mexican-born people living in the United States?

Study Design:
- Treatment: DACA-eligible individuals aged 26-30 at policy implementation (June 15, 2012)
- Control: Individuals aged 31-35 at policy implementation (otherwise eligible)
- Method: Difference-in-Differences
- Pre-period: 2008-2011
- Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("DACA REPLICATION ANALYSIS - SESSION 13")
print("=" * 70)
print()

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("1. LOADING AND PREPARING DATA")
print("-" * 50)

# Load data
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_13\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"Total observations: {len(df):,}")
print(f"Variables: {len(df.columns)}")
print()

# Check key variables
print("Key Variable Summary:")
print(f"  ELIGIBLE: {df['ELIGIBLE'].value_counts().to_dict()}")
print(f"  AFTER: {df['AFTER'].value_counts().to_dict()}")
print(f"  FT: {df['FT'].value_counts().to_dict()}")
print(f"  Years: {sorted(df['YEAR'].unique())}")
print()

# =============================================================================
# 2. DESCRIPTIVE STATISTICS
# =============================================================================
print("2. DESCRIPTIVE STATISTICS")
print("-" * 50)

# Create groups
df['Treatment'] = df['ELIGIBLE'].map({1: 'Eligible (26-30)', 0: 'Control (31-35)'})
df['Period'] = df['AFTER'].map({1: 'Post-DACA (2013-16)', 0: 'Pre-DACA (2008-11)'})

# Sample sizes by group
print("\nSample Sizes by Group:")
group_counts = df.groupby(['ELIGIBLE', 'AFTER']).size().reset_index(name='N')
print(group_counts)
print()

# Weighted sample sizes
print("\nWeighted Sample Sizes by Group:")
weighted_counts = df.groupby(['ELIGIBLE', 'AFTER'])['PERWT'].sum().reset_index(name='Weighted_N')
print(weighted_counts)
print()

# Full-time employment rates by group (unweighted)
print("\nUnweighted Full-Time Employment Rates:")
ft_rates_unweighted = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().reset_index(name='FT_Rate')
print(ft_rates_unweighted)
print()

# Full-time employment rates by group (weighted)
def weighted_mean(x, weights):
    return np.average(x, weights=weights)

print("\nWeighted Full-Time Employment Rates:")
ft_rates_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: pd.Series({
        'FT_Rate': weighted_mean(x['FT'], x['PERWT']),
        'N': len(x),
        'Weighted_N': x['PERWT'].sum()
    })
).reset_index()
print(ft_rates_weighted)
print()

# =============================================================================
# 3. SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION
# =============================================================================
print("3. SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("-" * 50)

# Calculate 2x2 DiD manually (weighted)
eligible_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]
eligible_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]
control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]
control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]

# Weighted means
mean_eligible_pre = np.average(eligible_pre['FT'], weights=eligible_pre['PERWT'])
mean_eligible_post = np.average(eligible_post['FT'], weights=eligible_post['PERWT'])
mean_control_pre = np.average(control_pre['FT'], weights=control_pre['PERWT'])
mean_control_post = np.average(control_post['FT'], weights=control_post['PERWT'])

# Differences
diff_eligible = mean_eligible_post - mean_eligible_pre
diff_control = mean_control_post - mean_control_pre
did_estimate = diff_eligible - diff_control

print("Weighted Full-Time Employment Rates (2x2 Table):")
print(f"                        Pre-DACA    Post-DACA    Difference")
print(f"  Eligible (26-30):     {mean_eligible_pre:.4f}      {mean_eligible_post:.4f}       {diff_eligible:+.4f}")
print(f"  Control (31-35):      {mean_control_pre:.4f}      {mean_control_post:.4f}       {diff_control:+.4f}")
print(f"  ---------------------------------------------------------------")
print(f"  Diff-in-Diff:                                      {did_estimate:+.4f}")
print()

# =============================================================================
# 4. REGRESSION ANALYSIS - BASIC DiD
# =============================================================================
print("4. REGRESSION ANALYSIS")
print("-" * 50)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD without weights
print("\nModel 1: Basic DiD (OLS, no weights)")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(f"  ELIGIBLE coefficient:       {model1.params['ELIGIBLE']:.4f} (SE: {model1.bse['ELIGIBLE']:.4f})")
print(f"  AFTER coefficient:          {model1.params['AFTER']:.4f} (SE: {model1.bse['AFTER']:.4f})")
print(f"  DiD (ELIGIBLE_AFTER):       {model1.params['ELIGIBLE_AFTER']:.4f} (SE: {model1.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  t-statistic:                {model1.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value:                    {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  R-squared:                  {model1.rsquared:.4f}")
print(f"  N:                          {int(model1.nobs):,}")
print()

# Model 2: Weighted DiD
print("\nModel 2: Weighted DiD (WLS)")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"  ELIGIBLE coefficient:       {model2.params['ELIGIBLE']:.4f} (SE: {model2.bse['ELIGIBLE']:.4f})")
print(f"  AFTER coefficient:          {model2.params['AFTER']:.4f} (SE: {model2.bse['AFTER']:.4f})")
print(f"  DiD (ELIGIBLE_AFTER):       {model2.params['ELIGIBLE_AFTER']:.4f} (SE: {model2.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  t-statistic:                {model2.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value:                    {model2.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  R-squared:                  {model2.rsquared:.4f}")
print()

# =============================================================================
# 5. REGRESSION WITH COVARIATES
# =============================================================================
print("5. REGRESSION WITH COVARIATES")
print("-" * 50)

# Prepare covariates
# SEX: 1=Male, 2=Female in IPUMS coding
df['MALE'] = (df['SEX'] == 1).astype(int)

# Age (centered around 30 for interpretability)
df['AGE_CENTERED'] = df['AGE'] - 30

# Marital status (MARST: 1=married spouse present)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Education categories (from EDUC_RECODE)
print(f"Education categories: {df['EDUC_RECODE'].unique()}")

# Create dummy for at least high school
df['HS_DEGREE_NUM'] = df['HS_DEGREE'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
if df['HS_DEGREE_NUM'].isna().any():
    df['HS_DEGREE_NUM'] = df['HS_DEGREE'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# Model 3: DiD with demographic controls (weighted)
print("\nModel 3: DiD with Demographics (WLS)")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + AGE_CENTERED + MARRIED + HS_DEGREE_NUM + FAMSIZE',
                 data=df, weights=df['PERWT']).fit()
print(f"  DiD (ELIGIBLE_AFTER):       {model3.params['ELIGIBLE_AFTER']:.4f} (SE: {model3.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  t-statistic:                {model3.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value:                    {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  R-squared:                  {model3.rsquared:.4f}")
print()
print("  Covariate coefficients:")
for var in ['MALE', 'AGE_CENTERED', 'MARRIED', 'HS_DEGREE_NUM', 'FAMSIZE']:
    print(f"    {var}: {model3.params[var]:.4f} (SE: {model3.bse[var]:.4f})")
print()

# Model 4: DiD with state fixed effects (weighted)
print("\nModel 4: DiD with State Fixed Effects (WLS)")
df['STATE_FE'] = pd.Categorical(df['STATEFIP'])
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit()
print(f"  DiD (ELIGIBLE_AFTER):       {model4.params['ELIGIBLE_AFTER']:.4f} (SE: {model4.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  t-statistic:                {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value:                    {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  R-squared:                  {model4.rsquared:.4f}")
print()

# Model 5: Full specification with demographics and state FE
print("\nModel 5: Full Model with Demographics + State FE (WLS)")
model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + AGE_CENTERED + MARRIED + HS_DEGREE_NUM + FAMSIZE + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit()
print(f"  DiD (ELIGIBLE_AFTER):       {model5.params['ELIGIBLE_AFTER']:.4f} (SE: {model5.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  t-statistic:                {model5.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value:                    {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  R-squared:                  {model5.rsquared:.4f}")
print()

# =============================================================================
# 6. ROBUST STANDARD ERRORS (Clustered at State Level)
# =============================================================================
print("6. CLUSTERED STANDARD ERRORS")
print("-" * 50)

# Model 6: DiD with clustered standard errors at state level
print("\nModel 6: Basic DiD with State-Clustered Standard Errors")
model6 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                 data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                    cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD (ELIGIBLE_AFTER):       {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Clustered SE:               {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  t-statistic:                {model6.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value:                    {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI:                     [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print()

# Model 7: Full model with clustered SEs
print("\nModel 7: Full Model with State-Clustered Standard Errors")
model7 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + AGE_CENTERED + MARRIED + HS_DEGREE_NUM + FAMSIZE + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                    cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD (ELIGIBLE_AFTER):       {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Clustered SE:               {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  t-statistic:                {model7.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value:                    {model7.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI:                     [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print()

# =============================================================================
# 7. PARALLEL TRENDS ANALYSIS
# =============================================================================
print("7. PARALLEL TRENDS ANALYSIS")
print("-" * 50)

# Calculate year-by-year FT rates
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'FT_Rate': np.average(x['FT'], weights=x['PERWT']),
        'N': len(x)
    })
).reset_index()

print("\nYearly Full-Time Employment Rates (Weighted):")
pivot_yearly = yearly_rates.pivot(index='YEAR', columns='ELIGIBLE', values='FT_Rate')
pivot_yearly.columns = ['Control (31-35)', 'Eligible (26-30)']
print(pivot_yearly)
print()

# Event study regression - year-specific treatment effects
df['YEAR_FACTOR'] = pd.Categorical(df['YEAR'])
years = sorted(df['YEAR'].unique())

# Create year dummies interacted with ELIGIBLE (omitting 2011 as reference)
for year in years:
    if year != 2011:  # omit 2011 as reference year
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
        df[f'ELIGIBLE_X_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

year_interaction_vars = [f'ELIGIBLE_X_YEAR_{y}' for y in years if y != 2011]
year_vars = [f'YEAR_{y}' for y in years if y != 2011]

formula_event = 'FT ~ ELIGIBLE + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(year_interaction_vars)
model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit()

print("\nEvent Study Coefficients (Reference Year: 2011):")
print("Year     Coef      SE        t-stat    p-value")
print("-" * 50)
for year in [y for y in years if y != 2011]:
    var = f'ELIGIBLE_X_YEAR_{year}'
    print(f"{year}    {model_event.params[var]:+.4f}    {model_event.bse[var]:.4f}    {model_event.tvalues[var]:+.4f}    {model_event.pvalues[var]:.4f}")
print()

# Pre-trend test: joint significance of pre-treatment interactions
pre_years = [2008, 2009, 2010]  # 2011 is reference
pre_vars = [f'ELIGIBLE_X_YEAR_{y}' for y in pre_years]
# F-test for pre-trends
r_matrix = np.zeros((len(pre_vars), len(model_event.params)))
for i, var in enumerate(pre_vars):
    r_matrix[i, model_event.params.index.get_loc(var)] = 1
f_test = model_event.f_test(r_matrix)
print(f"Joint F-test for pre-trends (H0: no differential pre-trends):")
try:
    fval = float(f_test.fvalue) if np.isscalar(f_test.fvalue) else f_test.fvalue[0][0]
    pval = float(f_test.pvalue)
    print(f"  F-statistic: {fval:.4f}")
    print(f"  p-value: {pval:.4f}")
except:
    print(f"  F-statistic: {f_test.fvalue}")
    print(f"  p-value: {f_test.pvalue}")
print()

# =============================================================================
# 8. HETEROGENEITY ANALYSIS
# =============================================================================
print("8. HETEROGENEITY ANALYSIS")
print("-" * 50)

# By sex
print("\nHeterogeneity by Sex:")
for sex, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex]
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=df_sub, weights=df_sub['PERWT']).fit()
    print(f"  {sex_label}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}), N = {len(df_sub):,}")

# By education
print("\nHeterogeneity by Education:")
for edu, edu_label in [('TRUE', 'HS Degree+'), ('FALSE', 'Less than HS')]:
    # Handle both string and boolean formats
    df_sub = df[df['HS_DEGREE'].astype(str).str.upper() == edu]
    if len(df_sub) > 100:
        model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                            data=df_sub, weights=df_sub['PERWT']).fit()
        print(f"  {edu_label}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}), N = {len(df_sub):,}")

# =============================================================================
# 9. ROBUSTNESS CHECKS
# =============================================================================
print("\n9. ROBUSTNESS CHECKS")
print("-" * 50)

# Robustness 1: Linear Probability Model vs Probit marginal effects comparison
print("\nRobustness Check 1: Comparing LPM to Probit")
# Already have LPM from model2
print(f"  LPM DiD estimate: {model2.params['ELIGIBLE_AFTER']:.4f}")

# Probit model
try:
    probit_model = sm.Probit(df['FT'], sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])).fit(disp=0)
    # Approximate marginal effect at mean
    print(f"  Probit DiD coefficient: {probit_model.params['ELIGIBLE_AFTER']:.4f}")
except:
    print("  Probit model estimation failed")

# Robustness 2: Exclude extreme weight observations
print("\nRobustness Check 2: Trimmed Weights (1st-99th percentile)")
p1, p99 = df['PERWT'].quantile([0.01, 0.99])
df_trimmed = df[(df['PERWT'] >= p1) & (df['PERWT'] <= p99)]
model_trimmed = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=df_trimmed, weights=df_trimmed['PERWT']).fit()
print(f"  DiD estimate (trimmed): {model_trimmed.params['ELIGIBLE_AFTER']:.4f} (SE: {model_trimmed.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  N (trimmed): {len(df_trimmed):,}")

# Robustness 3: Unweighted analysis
print("\nRobustness Check 3: Unweighted Analysis")
print(f"  DiD estimate (unweighted): {model1.params['ELIGIBLE_AFTER']:.4f} (SE: {model1.bse['ELIGIBLE_AFTER']:.4f})")

# =============================================================================
# 10. SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF MAIN RESULTS")
print("=" * 70)

print("\n                                      DiD Estimate    SE        p-value    N")
print("-" * 75)
print(f"Model 1: Basic OLS                     {model1.params['ELIGIBLE_AFTER']:+.4f}        {model1.bse['ELIGIBLE_AFTER']:.4f}    {model1.pvalues['ELIGIBLE_AFTER']:.4f}     {int(model1.nobs):,}")
print(f"Model 2: Weighted (WLS)                {model2.params['ELIGIBLE_AFTER']:+.4f}        {model2.bse['ELIGIBLE_AFTER']:.4f}    {model2.pvalues['ELIGIBLE_AFTER']:.4f}     {int(model2.nobs):,}")
print(f"Model 3: WLS + Demographics            {model3.params['ELIGIBLE_AFTER']:+.4f}        {model3.bse['ELIGIBLE_AFTER']:.4f}    {model3.pvalues['ELIGIBLE_AFTER']:.4f}     {int(model3.nobs):,}")
print(f"Model 4: WLS + State FE                {model4.params['ELIGIBLE_AFTER']:+.4f}        {model4.bse['ELIGIBLE_AFTER']:.4f}    {model4.pvalues['ELIGIBLE_AFTER']:.4f}     {int(model4.nobs):,}")
print(f"Model 5: WLS + Demo + State FE         {model5.params['ELIGIBLE_AFTER']:+.4f}        {model5.bse['ELIGIBLE_AFTER']:.4f}    {model5.pvalues['ELIGIBLE_AFTER']:.4f}     {int(model5.nobs):,}")
print(f"Model 6: WLS + Clustered SE            {model6.params['ELIGIBLE_AFTER']:+.4f}        {model6.bse['ELIGIBLE_AFTER']:.4f}    {model6.pvalues['ELIGIBLE_AFTER']:.4f}     {int(model6.nobs):,}")
print(f"Model 7: Full + Clustered SE           {model7.params['ELIGIBLE_AFTER']:+.4f}        {model7.bse['ELIGIBLE_AFTER']:.4f}    {model7.pvalues['ELIGIBLE_AFTER']:.4f}     {int(model7.nobs):,}")
print("-" * 75)

# Preferred specification
print("\nPREFERRED SPECIFICATION: Model 7 (Full model with clustered SE)")
print(f"  Effect Size:     {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error:  {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI:          [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  t-statistic:     {model7.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value:         {model7.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  Sample Size:     {int(model7.nobs):,}")

# =============================================================================
# 11. GENERATE FIGURES
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

output_dir = r"C:\Users\seraf\DACA Results Task 3\replication_13"

# Figure 1: Parallel Trends
fig1, ax1 = plt.subplots(figsize=(10, 6))
pivot_yearly.plot(ax=ax1, marker='o', linewidth=2, markersize=8)
ax1.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Rates by DACA Eligibility Status', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])
plt.tight_layout()
fig1.savefig(f'{output_dir}/figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
print("Saved: figure1_parallel_trends.png")

# Figure 2: Event Study Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
event_years = [y for y in years if y != 2011]
event_coefs = [model_event.params[f'ELIGIBLE_X_YEAR_{y}'] for y in event_years]
event_ses = [model_event.bse[f'ELIGIBLE_X_YEAR_{y}'] for y in event_years]
event_ci_lower = [c - 1.96*s for c, s in zip(event_coefs, event_ses)]
event_ci_upper = [c + 1.96*s for c, s in zip(event_coefs, event_ses)]

ax2.errorbar(event_years, event_coefs, yerr=[(c-l) for c, l in zip(event_coefs, event_ci_lower)],
             fmt='o', capsize=5, capthick=2, markersize=8, color='blue')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.axvline(x=2011.5, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax2.plot([2011], [0], 'ko', markersize=8, label='Reference Year (2011)')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Year-Specific Treatment Effects', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
fig2.savefig(f'{output_dir}/figure2_event_study.png', dpi=300, bbox_inches='tight')
print("Saved: figure2_event_study.png")

# Figure 3: DiD Visualization (2x2)
fig3, ax3 = plt.subplots(figsize=(8, 6))
x_pre, x_post = 0, 1
ax3.plot([x_pre, x_post], [mean_control_pre, mean_control_post], 'b-o',
         linewidth=2, markersize=10, label='Control (31-35)')
ax3.plot([x_pre, x_post], [mean_eligible_pre, mean_eligible_post], 'r-o',
         linewidth=2, markersize=10, label='Eligible (26-30)')
# Counterfactual
counterfactual_post = mean_eligible_pre + diff_control
ax3.plot([x_pre, x_post], [mean_eligible_pre, counterfactual_post], 'r--',
         linewidth=2, alpha=0.5, label='Counterfactual')
# DiD arrow
ax3.annotate('', xy=(x_post + 0.05, mean_eligible_post),
             xytext=(x_post + 0.05, counterfactual_post),
             arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(x_post + 0.1, (mean_eligible_post + counterfactual_post)/2,
         f'DiD = {did_estimate:.3f}', fontsize=12, color='green')
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)'])
ax3.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences Visualization', fontsize=14)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, max(mean_control_post, mean_eligible_post) + 0.1])
plt.tight_layout()
fig3.savefig(f'{output_dir}/figure3_did_visual.png', dpi=300, bbox_inches='tight')
print("Saved: figure3_did_visual.png")

# Figure 4: Coefficient Comparison Across Models
fig4, ax4 = plt.subplots(figsize=(10, 6))
model_names = ['M1: OLS', 'M2: WLS', 'M3: +Demo', 'M4: +State FE',
               'M5: Full', 'M6: Clust SE', 'M7: Full+Clust']
models = [model1, model2, model3, model4, model5, model6, model7]
coefs = [m.params['ELIGIBLE_AFTER'] for m in models]
ses = [m.bse['ELIGIBLE_AFTER'] for m in models]
ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]

y_pos = np.arange(len(model_names))
ax4.errorbar(coefs, y_pos, xerr=[(c-l) for c, l in zip(coefs, ci_lower)],
             fmt='o', capsize=5, capthick=2, markersize=8, color='blue')
ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(model_names)
ax4.set_xlabel('DiD Coefficient (95% CI)', fontsize=12)
ax4.set_title('DiD Estimates Across Model Specifications', fontsize=14)
ax4.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
fig4.savefig(f'{output_dir}/figure4_coefficient_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: figure4_coefficient_comparison.png")

plt.close('all')

# =============================================================================
# 12. SAVE DETAILED RESULTS FOR REPORT
# =============================================================================
print("\n" + "=" * 70)
print("SAVING DETAILED RESULTS")
print("=" * 70)

# Save summary statistics
summary_stats = df.groupby(['Treatment', 'Period']).agg({
    'FT': ['mean', 'std', 'count'],
    'PERWT': 'sum',
    'MALE': 'mean',
    'AGE': 'mean',
    'MARRIED': 'mean',
    'FAMSIZE': 'mean'
}).round(4)
summary_stats.to_csv(f'{output_dir}/summary_statistics.csv')
print("Saved: summary_statistics.csv")

# Save yearly rates
pivot_yearly.to_csv(f'{output_dir}/yearly_ft_rates.csv')
print("Saved: yearly_ft_rates.csv")

# Save event study coefficients
event_study_df = pd.DataFrame({
    'Year': event_years,
    'Coefficient': event_coefs,
    'SE': event_ses,
    'CI_Lower': event_ci_lower,
    'CI_Upper': event_ci_upper
})
event_study_df.to_csv(f'{output_dir}/event_study_results.csv', index=False)
print("Saved: event_study_results.csv")

# Save full model results
with open(f'{output_dir}/model_results.txt', 'w') as f:
    f.write("DACA REPLICATION - MODEL RESULTS\n")
    f.write("=" * 70 + "\n\n")
    f.write("PREFERRED SPECIFICATION: Model 7 (Full model with clustered SE)\n\n")
    f.write(model7.summary().as_text())
print("Saved: model_results.txt")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
