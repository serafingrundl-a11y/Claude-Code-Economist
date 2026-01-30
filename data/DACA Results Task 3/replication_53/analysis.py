"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Mexican-born Hispanic individuals in the United States.

Method: Difference-in-Differences (DiD)
Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
Control: Ages 31-35 at DACA implementation
Pre-period: 2008-2011
Post-period: 2013-2016
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
import os

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 3\replication_53")

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("\n1. LOADING DATA...")

# Load the numeric version for analysis
df = pd.read_csv("data/prepared_data_numeric_version.csv", low_memory=False)
print(f"Dataset loaded: {df.shape[0]:,} observations, {df.shape[1]} variables")

# Ensure key variables are numeric
df['ELIGIBLE'] = pd.to_numeric(df['ELIGIBLE'], errors='coerce')
df['AFTER'] = pd.to_numeric(df['AFTER'], errors='coerce')
df['FT'] = pd.to_numeric(df['FT'], errors='coerce')
df['PERWT'] = pd.to_numeric(df['PERWT'], errors='coerce')
df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
df['SEX'] = pd.to_numeric(df['SEX'], errors='coerce')
df['MARST'] = pd.to_numeric(df['MARST'], errors='coerce')
df['STATEFIP'] = pd.to_numeric(df['STATEFIP'], errors='coerce')
df['EDUC'] = pd.to_numeric(df['EDUC'], errors='coerce')

# Basic info
print("\n--- Variable Types ---")
print(df.dtypes.value_counts())

print("\n--- Key Variables Summary ---")
key_vars = ['YEAR', 'FT', 'AFTER', 'ELIGIBLE', 'PERWT', 'SEX', 'AGE', 'EDUC']
for var in key_vars:
    if var in df.columns:
        print(f"\n{var}:")
        print(f"  Non-missing: {df[var].notna().sum():,}")
        print(f"  Unique values: {df[var].nunique()}")
        if df[var].nunique() <= 10:
            print(f"  Value counts:\n{df[var].value_counts().sort_index()}")
        else:
            print(f"  Range: {df[var].min()} - {df[var].max()}")
            print(f"  Mean: {df[var].mean():.3f}")

# ============================================================================
# 2. DATA VERIFICATION
# ============================================================================
print("\n" + "=" * 70)
print("2. DATA VERIFICATION")
print("=" * 70)

# Check years
print("\n--- Years in Dataset ---")
print(df['YEAR'].value_counts().sort_index())

# Verify ELIGIBLE coding
print("\n--- ELIGIBLE Distribution ---")
print(df['ELIGIBLE'].value_counts())

# Verify AFTER coding
print("\n--- AFTER Distribution ---")
print(df['AFTER'].value_counts())

# Verify FT coding
print("\n--- FT (Full-Time Employment) Distribution ---")
print(df['FT'].value_counts())

# Cross-tabulation
print("\n--- Sample Sizes by Group (ELIGIBLE x AFTER) ---")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
crosstab.columns = ['Pre (2008-11)', 'Post (2013-16)', 'Total']
print(crosstab)

# ============================================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 70)

# Full-time employment rates by group
print("\n--- Full-Time Employment Rates by Group ---")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: pd.Series({
        'FT_Rate': np.average(x['FT'], weights=x['PERWT']),
        'N': len(x),
        'Weighted_N': x['PERWT'].sum()
    })
)
print(ft_rates)

# Calculate DiD manually
ft_treat_post = ft_rates.loc[(1, 1), 'FT_Rate']
ft_treat_pre = ft_rates.loc[(1, 0), 'FT_Rate']
ft_ctrl_post = ft_rates.loc[(0, 1), 'FT_Rate']
ft_ctrl_pre = ft_rates.loc[(0, 0), 'FT_Rate']

did_simple = (ft_treat_post - ft_treat_pre) - (ft_ctrl_post - ft_ctrl_pre)
print(f"\n--- Simple DiD Calculation ---")
print(f"Treatment (26-30) Post: {ft_treat_post:.4f}")
print(f"Treatment (26-30) Pre:  {ft_treat_pre:.4f}")
print(f"Treatment Change:       {ft_treat_post - ft_treat_pre:.4f}")
print(f"Control (31-35) Post:   {ft_ctrl_post:.4f}")
print(f"Control (31-35) Pre:    {ft_ctrl_pre:.4f}")
print(f"Control Change:         {ft_ctrl_post - ft_ctrl_pre:.4f}")
print(f"DiD Estimate:           {did_simple:.4f}")

# ============================================================================
# 4. TREND ANALYSIS BY YEAR
# ============================================================================
print("\n" + "=" * 70)
print("4. TREND ANALYSIS BY YEAR")
print("=" * 70)

trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
trends.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\n--- Full-Time Employment Rate by Year and Group ---")
print(trends)

# Save trends for plotting
trends.to_csv("output_trends.csv")

# ============================================================================
# 5. MAIN REGRESSION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("5. MAIN REGRESSION ANALYSIS")
print("=" * 70)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# ----- Model 1: Basic DiD (no controls) -----
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model1.summary())

# Store results
results = []
results.append({
    'Model': 'Basic DiD',
    'Coefficient': model1.params['ELIGIBLE_AFTER'],
    'SE': model1.bse['ELIGIBLE_AFTER'],
    'CI_lower': model1.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_upper': model1.conf_int().loc['ELIGIBLE_AFTER', 1],
    'P_value': model1.pvalues['ELIGIBLE_AFTER'],
    'N': int(model1.nobs),
    'R_squared': model1.rsquared
})

# ----- Model 2: DiD with Year Fixed Effects -----
print("\n--- Model 2: DiD with Year Fixed Effects ---")
# Use matrix-based approach to avoid formula parsing issues
df_analysis = df.copy()

# SEX (1=Male, 2=Female in IPUMS coding)
df_analysis['FEMALE'] = (df_analysis['SEX'] == 2).astype(int)

# Marital status (MARST: 1=Married spouse present, 2=Married spouse absent, etc.)
if 'MARST' in df_analysis.columns:
    df_analysis['MARRIED'] = (df_analysis['MARST'] <= 2).astype(int)

# Create year dummies - use integer conversion for column names
year_dummies = pd.get_dummies(df_analysis['YEAR'].astype(int), prefix='YEAR', drop_first=True).astype(float)
year_cols = list(year_dummies.columns)

# Create education dummies with integer column names
if 'EDUC' in df_analysis.columns:
    df_analysis['EDUC_INT'] = df_analysis['EDUC'].fillna(6).astype(int)
    educ_dummies = pd.get_dummies(df_analysis['EDUC_INT'], prefix='EDUC', drop_first=True).astype(float)
    educ_cols = list(educ_dummies.columns)
else:
    educ_dummies = pd.DataFrame()
    educ_cols = []

# Create state dummies
state_dummies = pd.get_dummies(df_analysis['STATEFIP'].astype(int), prefix='STATE', drop_first=True).astype(float)
state_cols = list(state_dummies.columns)

# Combine all dummies
df_analysis = pd.concat([df_analysis, year_dummies, educ_dummies, state_dummies], axis=1)

# Build design matrix for Model 2 (Year FE)
X2_vars = ['ELIGIBLE', 'ELIGIBLE_AFTER'] + year_cols
X2 = sm.add_constant(df_analysis[X2_vars])
y = df_analysis['FT']
weights = df_analysis['PERWT']

model2 = sm.WLS(y, X2, weights=weights).fit(cov_type='HC1')
print(model2.summary())

results.append({
    'Model': 'Year FE',
    'Coefficient': model2.params['ELIGIBLE_AFTER'],
    'SE': model2.bse['ELIGIBLE_AFTER'],
    'CI_lower': model2.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_upper': model2.conf_int().loc['ELIGIBLE_AFTER', 1],
    'P_value': model2.pvalues['ELIGIBLE_AFTER'],
    'N': int(model2.nobs),
    'R_squared': model2.rsquared
})

# ----- Model 3: DiD with Demographics -----
print("\n--- Model 3: DiD with Demographic Controls ---")
demo_vars = ['FEMALE', 'MARRIED', 'AGE']
demo_vars = [v for v in demo_vars if v in df_analysis.columns]

X3_vars = ['ELIGIBLE', 'ELIGIBLE_AFTER'] + year_cols + demo_vars + educ_cols
X3 = sm.add_constant(df_analysis[X3_vars])

model3 = sm.WLS(y, X3, weights=weights).fit(cov_type='HC1')
print(model3.summary())

results.append({
    'Model': 'Demographics',
    'Coefficient': model3.params['ELIGIBLE_AFTER'],
    'SE': model3.bse['ELIGIBLE_AFTER'],
    'CI_lower': model3.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_upper': model3.conf_int().loc['ELIGIBLE_AFTER', 1],
    'P_value': model3.pvalues['ELIGIBLE_AFTER'],
    'N': int(model3.nobs),
    'R_squared': model3.rsquared
})

# ----- Model 4: DiD with State Fixed Effects -----
print("\n--- Model 4: DiD with State Fixed Effects ---")
X4_vars = ['ELIGIBLE', 'ELIGIBLE_AFTER'] + year_cols + demo_vars + educ_cols + state_cols
X4 = sm.add_constant(df_analysis[X4_vars])

model4 = sm.WLS(y, X4, weights=weights).fit(cov_type='HC1')

print(f"\nModel 4 - Coefficient on ELIGIBLE_AFTER: {model4.params['ELIGIBLE_AFTER']:.6f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")
print(f"P-value: {model4.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"R-squared: {model4.rsquared:.4f}")

results.append({
    'Model': 'State FE',
    'Coefficient': model4.params['ELIGIBLE_AFTER'],
    'SE': model4.bse['ELIGIBLE_AFTER'],
    'CI_lower': model4.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_upper': model4.conf_int().loc['ELIGIBLE_AFTER', 1],
    'P_value': model4.pvalues['ELIGIBLE_AFTER'],
    'N': int(model4.nobs),
    'R_squared': model4.rsquared
})

# ----- Model 5: Full Model with State Policies -----
print("\n--- Model 5: Full Model with State Policies ---")
policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID',
               'EVERIFY', 'SECURECOMMUNITIES', 'LFPR', 'UNEMP']
policy_vars = [v for v in policy_vars if v in df_analysis.columns]

X5_vars = ['ELIGIBLE', 'ELIGIBLE_AFTER'] + year_cols + demo_vars + educ_cols + state_cols + policy_vars
X5 = sm.add_constant(df_analysis[X5_vars])

model5 = sm.WLS(y, X5, weights=weights).fit(cov_type='HC1')

print(f"\nModel 5 - Coefficient on ELIGIBLE_AFTER: {model5.params['ELIGIBLE_AFTER']:.6f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.6f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.6f}]")
print(f"P-value: {model5.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"R-squared: {model5.rsquared:.4f}")

results.append({
    'Model': 'Full Model',
    'Coefficient': model5.params['ELIGIBLE_AFTER'],
    'SE': model5.bse['ELIGIBLE_AFTER'],
    'CI_lower': model5.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_upper': model5.conf_int().loc['ELIGIBLE_AFTER', 1],
    'P_value': model5.pvalues['ELIGIBLE_AFTER'],
    'N': int(model5.nobs),
    'R_squared': model5.rsquared
})

# ============================================================================
# 6. RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("6. RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
print("\n--- All Models Summary ---")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("output_results.csv", index=False)

# ============================================================================
# 7. ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 70)
print("7. ROBUSTNESS CHECKS")
print("=" * 70)

# ----- Robustness 1: Unweighted regression -----
print("\n--- Robustness Check 1: Unweighted Regression ---")
model_unweighted = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(f"Unweighted DiD Coefficient: {model_unweighted.params['ELIGIBLE_AFTER']:.6f}")
print(f"SE: {model_unweighted.bse['ELIGIBLE_AFTER']:.6f}")

# ----- Robustness 2: By Gender -----
print("\n--- Robustness Check 2: By Gender ---")
df_analysis['FEMALE'] = (df_analysis['SEX'] == 2).astype(int)

# Males
df_male = df_analysis[df_analysis['FEMALE'] == 0].copy()
model_male = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                     data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
print(f"Males - DiD Coefficient: {model_male.params['ELIGIBLE_AFTER']:.6f}")
print(f"        SE: {model_male.bse['ELIGIBLE_AFTER']:.6f}")
print(f"        N: {int(model_male.nobs):,}")

# Females
df_female = df_analysis[df_analysis['FEMALE'] == 1].copy()
model_female = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                       data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"Females - DiD Coefficient: {model_female.params['ELIGIBLE_AFTER']:.6f}")
print(f"          SE: {model_female.bse['ELIGIBLE_AFTER']:.6f}")
print(f"          N: {int(model_female.nobs):,}")

# ----- Robustness 3: Placebo test (pre-treatment trends) -----
print("\n--- Robustness Check 3: Pre-Treatment Parallel Trends Test ---")
df_pre = df[df['AFTER'] == 0].copy()
df_pre['YEAR_TREND'] = df_pre['YEAR'] - 2008
df_pre['ELIGIBLE_TREND'] = df_pre['ELIGIBLE'] * df_pre['YEAR_TREND']

model_pretrend = smf.wls('FT ~ ELIGIBLE + YEAR_TREND + ELIGIBLE_TREND',
                         data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"Pre-trend Interaction (ELIGIBLE*YEAR_TREND): {model_pretrend.params['ELIGIBLE_TREND']:.6f}")
print(f"SE: {model_pretrend.bse['ELIGIBLE_TREND']:.6f}")
print(f"P-value: {model_pretrend.pvalues['ELIGIBLE_TREND']:.6f}")
if model_pretrend.pvalues['ELIGIBLE_TREND'] > 0.05:
    print("-> No significant pre-treatment differential trend (supports parallel trends assumption)")
else:
    print("-> Warning: Significant pre-treatment differential trend detected")

# ----- Robustness 4: Event Study -----
print("\n--- Robustness Check 4: Event Study (Year-by-Year Effects) ---")
# Create year-specific interactions (excluding 2011 as reference)
event_study_data = df.copy()
event_study_data['ELIGIBLE'] = event_study_data['ELIGIBLE']
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    event_study_data[f'ELIG_Y{year}'] = (event_study_data['YEAR'] == year).astype(int) * event_study_data['ELIGIBLE']

event_vars = ['ELIGIBLE'] + [f'ELIG_Y{y}' for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]]
year_dummies_event = pd.get_dummies(event_study_data['YEAR'].astype(int), prefix='YEAR', drop_first=True).astype(float)
event_study_data = pd.concat([event_study_data, year_dummies_event], axis=1)

X_event = sm.add_constant(event_study_data[event_vars + list(year_dummies_event.columns)])
y_event = event_study_data['FT']
w_event = event_study_data['PERWT']

model_event = sm.WLS(y_event, X_event, weights=w_event).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'ELIG_Y{year}']
    se = model_event.bse[f'ELIG_Y{year}']
    pval = model_event.pvalues[f'ELIG_Y{year}']
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}, p={pval:.3f})")

# Save event study results
event_results = pd.DataFrame({
    'Year': [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [model_event.params[f'ELIG_Y{y}'] if y != 2011 else 0 for y in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]],
    'SE': [model_event.bse[f'ELIG_Y{y}'] if y != 2011 else 0 for y in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]]
})
event_results.to_csv("output_event_study.csv", index=False)

# ============================================================================
# 8. SAVE ADDITIONAL OUTPUTS
# ============================================================================
print("\n" + "=" * 70)
print("8. SAVING OUTPUTS")
print("=" * 70)

# Descriptive statistics table
desc_stats = df.groupby('ELIGIBLE').agg({
    'FT': ['mean', 'std'],
    'AGE': ['mean', 'std'],
    'SEX': lambda x: (x == 2).mean(),  # Proportion female
    'PERWT': 'sum'
}).round(4)
desc_stats.columns = ['FT_Mean', 'FT_SD', 'Age_Mean', 'Age_SD', 'Prop_Female', 'Weighted_N']
desc_stats.index = ['Control (31-35)', 'Treatment (26-30)']
desc_stats.to_csv("output_descriptives.csv")
print("Saved: output_descriptives.csv")

# Summary statistics by group and period
summary_by_group = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': 'mean',
    'PERWT': ['count', 'sum']
}).round(4)
summary_by_group.to_csv("output_group_summary.csv")
print("Saved: output_group_summary.csv")

# ============================================================================
# 9. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("9. CREATING VISUALIZATIONS")
print("=" * 70)

# Figure 1: Trends in Full-Time Employment
plt.figure(figsize=(10, 6))
trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
trends.columns = ['Control (31-35)', 'Treatment (26-30)']

plt.plot(trends.index, trends['Control (31-35)'], 'b-o', label='Control (31-35)', linewidth=2, markersize=8)
plt.plot(trends.index, trends['Treatment (26-30)'], 'r-s', label='Treatment (26-30)', linewidth=2, markersize=8)
plt.axvline(x=2012, color='gray', linestyle='--', label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Full-Time Employment Rates by DACA Eligibility Status', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure1_trends.png")

# Figure 2: DiD Visualization
fig, ax = plt.subplots(figsize=(8, 6))
pre_post = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
pre_post.columns = ['Pre-DACA (2008-11)', 'Post-DACA (2013-16)']
pre_post.index = ['Control (31-35)', 'Treatment (26-30)']

x = np.array([0, 1])
width = 0.35
ax.bar(x - width/2, pre_post['Pre-DACA (2008-11)'], width, label='Pre-DACA (2008-11)', color='lightblue', edgecolor='navy')
ax.bar(x + width/2, pre_post['Post-DACA (2013-16)'], width, label='Post-DACA (2013-16)', color='salmon', edgecolor='darkred')
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['Control (31-35)', 'Treatment (26-30)'], fontsize=11)
ax.legend(fontsize=10)
ax.set_title('Full-Time Employment: Pre vs Post DACA by Group', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figure2_did.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure2_did.png")

# Figure 3: Coefficient Plot
fig, ax = plt.subplots(figsize=(8, 6))
models = results_df['Model'].values
coefs = results_df['Coefficient'].values
ci_lower = results_df['CI_lower'].values
ci_upper = results_df['CI_upper'].values

y_pos = np.arange(len(models))
ax.errorbar(coefs, y_pos, xerr=[coefs - ci_lower, ci_upper - coefs],
            fmt='o', capsize=5, capthick=2, markersize=8, color='navy')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)', fontsize=12)
ax.set_title('DACA Effect Estimates Across Model Specifications', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('figure3_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure3_coefficients.png")

# Figure 4: Event Study Plot
fig, ax = plt.subplots(figsize=(10, 6))
years_plot = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs_event = []
ci_low_event = []
ci_high_event = []

for year in years_plot:
    if year == 2011:
        coefs_event.append(0)
        ci_low_event.append(0)
        ci_high_event.append(0)
    else:
        c = model_event.params[f'ELIG_Y{year}']
        se = model_event.bse[f'ELIG_Y{year}']
        coefs_event.append(c)
        ci_low_event.append(c - 1.96 * se)
        ci_high_event.append(c + 1.96 * se)

ax.errorbar(years_plot, coefs_event,
            yerr=[np.array(coefs_event) - np.array(ci_low_event),
                  np.array(ci_high_event) - np.array(coefs_event)],
            fmt='o-', capsize=5, capthick=2, markersize=8, color='navy', linewidth=2)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Effect on Full-Time Employment (relative to 2011)', fontsize=12)
ax.set_title('Event Study: DACA Effect on Full-Time Employment', fontsize=14)
ax.set_xticks(years_plot)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure4_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure4_event_study.png")

# ============================================================================
# 10. FINAL PREFERRED ESTIMATE
# ============================================================================
print("\n" + "=" * 70)
print("10. PREFERRED ESTIMATE")
print("=" * 70)

# Use Model 4 (State FE) as preferred - balances control for confounders with interpretability
preferred = results_df[results_df['Model'] == 'State FE'].iloc[0]
print(f"\nPREFERRED MODEL: DiD with Year, Demographic, and State Fixed Effects")
print(f"Effect Size (DiD Coefficient): {preferred['Coefficient']:.6f}")
print(f"Standard Error: {preferred['SE']:.6f}")
print(f"95% Confidence Interval: [{preferred['CI_lower']:.6f}, {preferred['CI_upper']:.6f}]")
print(f"P-value: {preferred['P_value']:.6f}")
print(f"Sample Size: {preferred['N']:,}")

# Also show basic model for comparison
basic = results_df[results_df['Model'] == 'Basic DiD'].iloc[0]
print(f"\nBASIC MODEL (for reference):")
print(f"Effect Size (DiD Coefficient): {basic['Coefficient']:.6f}")
print(f"Standard Error: {basic['SE']:.6f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# Save full regression output for Model 4
with open("output_model4_full.txt", "w") as f:
    f.write(str(model4.summary()))
print("Saved: output_model4_full.txt")
