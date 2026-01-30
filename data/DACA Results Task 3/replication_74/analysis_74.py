"""
DACA Replication Analysis - ID 74
Effect of DACA Eligibility on Full-Time Employment

Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US?

Identification Strategy: Difference-in-Differences
- Treated: Eligible individuals aged 26-30 when DACA enacted (June 2012)
- Control: Individuals aged 31-35 when DACA enacted (otherwise eligible)
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
np.random.seed(74)

print("="*80)
print("DACA REPLICATION ANALYSIS - ID 74")
print("="*80)

# Load data
print("\n[1] Loading Data...")
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Total observations loaded: {len(df):,}")

# Basic data inspection
print("\n[2] Data Structure Overview")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Number of unique years: {df['YEAR'].nunique()}")

# Key variables check
print("\n[3] Key Variables Summary")
print(f"\nFT (Full-Time Employment):")
print(df['FT'].value_counts().sort_index())
print(f"Mean FT: {df['FT'].mean():.4f}")

print(f"\nELIGIBLE (Treatment Group):")
print(df['ELIGIBLE'].value_counts().sort_index())

print(f"\nAFTER (Post-Treatment Period):")
print(df['AFTER'].value_counts().sort_index())

# Verify no 2012 data
print(f"\nVerifying 2012 excluded: {2012 not in df['YEAR'].values}")

# Sample sizes by group
print("\n[4] Sample Sizes by Treatment Group and Period")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (31-35)', 'Treated (26-30)', 'Total']
crosstab.columns = ['Pre (2008-2011)', 'Post (2013-2016)', 'Total']
print(crosstab)

# Calculate weighted means
print("\n[5] Weighted Full-Time Employment Rates by Group and Period")

def weighted_mean(data, value_col, weight_col):
    """Calculate weighted mean"""
    return np.average(data[value_col], weights=data[weight_col])

def weighted_std(data, value_col, weight_col):
    """Calculate weighted standard deviation"""
    avg = weighted_mean(data, value_col, weight_col)
    variance = np.average((data[value_col] - avg)**2, weights=data[weight_col])
    return np.sqrt(variance)

# Calculate means for each cell
groups = {
    'Control Pre': df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)],
    'Control Post': df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)],
    'Treated Pre': df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)],
    'Treated Post': df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]
}

print("\nWeighted FT Employment Rates:")
for name, group in groups.items():
    wmean = weighted_mean(group, 'FT', 'PERWT')
    wstd = weighted_std(group, 'FT', 'PERWT')
    n = len(group)
    print(f"  {name}: {wmean:.4f} (SD: {wstd:.4f}, N: {n:,})")

# Calculate simple DiD estimate
control_pre = weighted_mean(groups['Control Pre'], 'FT', 'PERWT')
control_post = weighted_mean(groups['Control Post'], 'FT', 'PERWT')
treated_pre = weighted_mean(groups['Treated Pre'], 'FT', 'PERWT')
treated_post = weighted_mean(groups['Treated Post'], 'FT', 'PERWT')

control_diff = control_post - control_pre
treated_diff = treated_post - treated_pre
did_simple = treated_diff - control_diff

print(f"\nDifference-in-Differences (Simple Calculation):")
print(f"  Control change: {control_post:.4f} - {control_pre:.4f} = {control_diff:.4f}")
print(f"  Treated change: {treated_post:.4f} - {treated_pre:.4f} = {treated_diff:.4f}")
print(f"  DiD Estimate: {treated_diff:.4f} - {control_diff:.4f} = {did_simple:.4f}")

# ============================================================================
# MAIN REGRESSION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD (no covariates)
print("\n[Model 1] Basic Difference-in-Differences")
print("FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER")

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# WLS regression with robust standard errors
X1 = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']]
X1 = sm.add_constant(X1)
y = df['FT']
weights = df['PERWT']

model1 = sm.WLS(y, X1, weights=weights).fit(cov_type='HC1')
print(model1.summary())

# Extract key results
did_coef = model1.params['ELIGIBLE_AFTER']
did_se = model1.bse['ELIGIBLE_AFTER']
did_pval = model1.pvalues['ELIGIBLE_AFTER']
did_ci = model1.conf_int().loc['ELIGIBLE_AFTER']

print(f"\n*** PREFERRED ESTIMATE (Model 1) ***")
print(f"DiD Coefficient: {did_coef:.6f}")
print(f"Standard Error: {did_se:.6f}")
print(f"95% CI: [{did_ci[0]:.6f}, {did_ci[1]:.6f}]")
print(f"p-value: {did_pval:.6f}")
print(f"Sample Size: {len(df):,}")

# Model 2: DiD with demographic covariates
print("\n" + "-"*80)
print("\n[Model 2] DiD with Demographic Covariates")
print("Adding: SEX, AGE, MARST, EDUC_RECODE")

# Create dummy variables for categorical covariates
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Education dummies (reference: Less than High School)
# Fill NaN with 'Unknown' to handle missing values
df['EDUC_RECODE_CLEAN'] = df['EDUC_RECODE'].fillna('Unknown')
# Create education dummies with clean column names
df['EDUC_HS'] = (df['EDUC_RECODE_CLEAN'] == 'High School Degree').astype(int)
df['EDUC_SC'] = (df['EDUC_RECODE_CLEAN'] == 'Some College').astype(int)
df['EDUC_2Y'] = (df['EDUC_RECODE_CLEAN'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE_CLEAN'] == 'BA+').astype(int)
# Less than High School is reference category

educ_cols = ['EDUC_HS', 'EDUC_SC', 'EDUC_2Y', 'EDUC_BA']

# Model 2 specification
X2_cols = ['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE', 'MARRIED'] + educ_cols
X2 = df[X2_cols].astype(float)
X2 = sm.add_constant(X2)

model2 = sm.WLS(y, X2, weights=weights).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with Year Fixed Effects
print("\n" + "-"*80)
print("\n[Model 3] DiD with Year Fixed Effects")

# Create year dummies (reference: 2008)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

year_cols = ['YEAR_2009', 'YEAR_2010', 'YEAR_2011', 'YEAR_2013', 'YEAR_2014', 'YEAR_2015', 'YEAR_2016']

X3_cols = ['ELIGIBLE', 'ELIGIBLE_AFTER'] + year_cols
X3 = df[X3_cols].astype(float)
X3 = sm.add_constant(X3)

model3 = sm.WLS(y, X3, weights=weights).fit(cov_type='HC1')
print(model3.summary())

# Model 4: Full model with demographics and year FE
print("\n" + "-"*80)
print("\n[Model 4] Full Model: Demographics + Year Fixed Effects")

X4_cols = ['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE', 'MARRIED'] + educ_cols + year_cols
X4 = df[X4_cols].astype(float)
X4 = sm.add_constant(X4)

model4 = sm.WLS(y, X4, weights=weights).fit(cov_type='HC1')
print(model4.summary())

# Model 5: Add state fixed effects
print("\n" + "-"*80)
print("\n[Model 5] Full Model with State Fixed Effects")

# Create state dummies manually
unique_states = sorted(df['STATEFIP'].unique())
state_cols = []
for state in unique_states[1:]:  # Skip first state as reference
    col_name = f'STATE_{state}'
    df[col_name] = (df['STATEFIP'] == state).astype(int)
    state_cols.append(col_name)

X5_cols = ['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE', 'MARRIED'] + educ_cols + year_cols + state_cols
X5 = df[X5_cols].astype(float)
X5 = sm.add_constant(X5)

model5 = sm.WLS(y, X5, weights=weights).fit(cov_type='HC1')

# Print only the key DiD results (full output too long)
print(f"\nModel 5 Summary (Key Results Only):")
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.6f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_AFTER']:.6f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER'][0]:.6f}, {model5.conf_int().loc['ELIGIBLE_AFTER'][1]:.6f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.6f}")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"Observations: {int(model5.nobs):,}")

# ============================================================================
# PARALLEL TRENDS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PARALLEL TRENDS ANALYSIS")
print("="*80)

# Calculate yearly FT rates by treatment group
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'FT_rate': np.average(x['FT'], weights=x['PERWT']),
        'N': len(x),
        'weighted_N': x['PERWT'].sum()
    })
).reset_index()

print("\nYearly Full-Time Employment Rates by Group:")
print(yearly_rates.pivot(index='YEAR', columns='ELIGIBLE', values='FT_rate'))

# Event study regression
print("\n[Event Study Regression]")
print("Testing for pre-trends using year-by-treatment interactions")

# Create year-treatment interactions (reference: 2011, last pre-treatment year)
df['YEAR_2008_ELIGIBLE'] = ((df['YEAR'] == 2008) & (df['ELIGIBLE'] == 1)).astype(int)
df['YEAR_2009_ELIGIBLE'] = ((df['YEAR'] == 2009) & (df['ELIGIBLE'] == 1)).astype(int)
df['YEAR_2010_ELIGIBLE'] = ((df['YEAR'] == 2010) & (df['ELIGIBLE'] == 1)).astype(int)
# 2011 is reference year
df['YEAR_2013_ELIGIBLE'] = ((df['YEAR'] == 2013) & (df['ELIGIBLE'] == 1)).astype(int)
df['YEAR_2014_ELIGIBLE'] = ((df['YEAR'] == 2014) & (df['ELIGIBLE'] == 1)).astype(int)
df['YEAR_2015_ELIGIBLE'] = ((df['YEAR'] == 2015) & (df['ELIGIBLE'] == 1)).astype(int)
df['YEAR_2016_ELIGIBLE'] = ((df['YEAR'] == 2016) & (df['ELIGIBLE'] == 1)).astype(int)

event_cols = ['YEAR_2008_ELIGIBLE', 'YEAR_2009_ELIGIBLE', 'YEAR_2010_ELIGIBLE',
              'YEAR_2013_ELIGIBLE', 'YEAR_2014_ELIGIBLE', 'YEAR_2015_ELIGIBLE', 'YEAR_2016_ELIGIBLE']

X_event = df[['ELIGIBLE'] + year_cols + event_cols].astype(float)
X_event = sm.add_constant(X_event)

model_event = sm.WLS(y, X_event, weights=weights).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
for col in event_cols:
    year = col.split('_')[1]
    coef = model_event.params[col]
    se = model_event.bse[col]
    pval = model_event.pvalues[col]
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# Test joint significance of pre-trends
pre_trend_cols = ['YEAR_2008_ELIGIBLE', 'YEAR_2009_ELIGIBLE', 'YEAR_2010_ELIGIBLE']
print("\nJoint Test of Pre-Trend Coefficients (H0: all pre-period coefficients = 0):")

# F-test for pre-trends
r_matrix = np.zeros((3, len(X_event.columns)))
for i, col in enumerate(pre_trend_cols):
    r_matrix[i, list(X_event.columns).index(col)] = 1

f_test = model_event.f_test(r_matrix)
try:
    fval = f_test.fvalue[0][0] if hasattr(f_test.fvalue, '__getitem__') else float(f_test.fvalue)
except:
    fval = float(f_test.fvalue)
print(f"F-statistic: {fval:.4f}")
print(f"p-value: {float(f_test.pvalue):.4f}")

# ============================================================================
# HETEROGENEITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("HETEROGENEITY ANALYSIS")
print("="*80)

# By gender
print("\n[Heterogeneity by Gender]")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    X_sub = df_sub[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']]
    X_sub = sm.add_constant(X_sub)
    y_sub = df_sub['FT']
    w_sub = df_sub['PERWT']
    model_sub = sm.WLS(y_sub, X_sub, weights=w_sub).fit(cov_type='HC1')
    print(f"  {sex_name}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}, N: {len(df_sub):,})")

# By education
print("\n[Heterogeneity by Education]")
for educ_val in df['EDUC_RECODE'].unique():
    df_sub = df[df['EDUC_RECODE'] == educ_val]
    if len(df_sub) > 100:  # Only if sufficient sample
        X_sub = df_sub[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']]
        X_sub = sm.add_constant(X_sub)
        y_sub = df_sub['FT']
        w_sub = df_sub['PERWT']
        model_sub = sm.WLS(y_sub, X_sub, weights=w_sub).fit(cov_type='HC1')
        print(f"  {educ_val}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}, N: {len(df_sub):,})")

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Variables to summarize
sum_vars = ['FT', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD', 'FAMSIZE']

print("\nSummary Statistics by Treatment Status (Pre-Period Only):")
pre_df = df[df['AFTER'] == 0]

for var in sum_vars:
    if var in df.columns:
        ctrl = pre_df[pre_df['ELIGIBLE'] == 0]
        treat = pre_df[pre_df['ELIGIBLE'] == 1]

        ctrl_mean = np.average(ctrl[var], weights=ctrl['PERWT'])
        treat_mean = np.average(treat[var], weights=treat['PERWT'])

        # T-test for difference
        t_stat, p_val = stats.ttest_ind(ctrl[var], treat[var])

        print(f"\n{var}:")
        print(f"  Control: {ctrl_mean:.4f}")
        print(f"  Treated: {treat_mean:.4f}")
        print(f"  Diff: {treat_mean - ctrl_mean:.4f} (p={p_val:.4f})")

# ============================================================================
# FIGURES
# ============================================================================

print("\n" + "="*80)
print("GENERATING FIGURES")
print("="*80)

# Figure 1: Parallel Trends Plot
fig1, ax1 = plt.subplots(figsize=(10, 6))

control_rates = yearly_rates[yearly_rates['ELIGIBLE'] == 0]
treated_rates = yearly_rates[yearly_rates['ELIGIBLE'] == 1]

ax1.plot(control_rates['YEAR'], control_rates['FT_rate'], 'b-o', label='Control (Ages 31-35)', linewidth=2, markersize=8)
ax1.plot(treated_rates['YEAR'], treated_rates['FT_rate'], 'r-s', label='Treated (Ages 26-30)', linewidth=2, markersize=8)

ax1.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation (2012)')
ax1.axvspan(2012, 2016.5, alpha=0.1, color='green', label='Post-Treatment Period')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Rates by DACA Eligibility Status', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
print("Saved: figure1_parallel_trends.png")

# Figure 2: Event Study Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [model_event.params.get(f'YEAR_{y}_ELIGIBLE', 0) for y in years]
ses = [model_event.bse.get(f'YEAR_{y}_ELIGIBLE', 0) for y in years]
# Reference year 2011 has 0 coefficient and 0 SE
coefs[3] = 0
ses[3] = 0

ax2.errorbar(years, coefs, yerr=[1.96*s for s in ses], fmt='ko-', capsize=5, capthick=2, linewidth=2, markersize=8)
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax2.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax2.set_xticks(years)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
print("Saved: figure2_event_study.png")

# Figure 3: DiD Estimates Across Specifications
fig3, ax3 = plt.subplots(figsize=(10, 6))

model_names = ['Basic DiD', 'Demographics', 'Year FE', 'Full Model', 'State FE']
estimates = [
    model1.params['ELIGIBLE_AFTER'],
    model2.params['ELIGIBLE_AFTER'],
    model3.params['ELIGIBLE_AFTER'],
    model4.params['ELIGIBLE_AFTER'],
    model5.params['ELIGIBLE_AFTER']
]
errors = [
    1.96 * model1.bse['ELIGIBLE_AFTER'],
    1.96 * model2.bse['ELIGIBLE_AFTER'],
    1.96 * model3.bse['ELIGIBLE_AFTER'],
    1.96 * model4.bse['ELIGIBLE_AFTER'],
    1.96 * model5.bse['ELIGIBLE_AFTER']
]

x_pos = np.arange(len(model_names))
ax3.errorbar(x_pos, estimates, yerr=errors, fmt='ko', capsize=8, capthick=2, markersize=10)
ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1)

ax3.set_ylabel('DiD Coefficient (Effect on FT Employment)', fontsize=12)
ax3.set_xlabel('Model Specification', fontsize=12)
ax3.set_title('DACA Effect Estimates Across Model Specifications', fontsize=14)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_names, fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure3_robustness.png', dpi=300, bbox_inches='tight')
print("Saved: figure3_robustness.png")

# ============================================================================
# FINAL RESULTS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print("\n*** PREFERRED ESTIMATE (Model 1: Basic DiD) ***")
print(f"Effect of DACA Eligibility on Full-Time Employment: {did_coef:.6f}")
print(f"Standard Error (Robust): {did_se:.6f}")
print(f"95% Confidence Interval: [{did_ci[0]:.6f}, {did_ci[1]:.6f}]")
print(f"p-value: {did_pval:.6f}")
print(f"Sample Size: {len(df):,}")

print("\n*** ROBUSTNESS CHECK SUMMARY ***")
print(f"Model 1 (Basic DiD):        {model1.params['ELIGIBLE_AFTER']:.4f} (SE: {model1.bse['ELIGIBLE_AFTER']:.4f})")
print(f"Model 2 (Demographics):     {model2.params['ELIGIBLE_AFTER']:.4f} (SE: {model2.bse['ELIGIBLE_AFTER']:.4f})")
print(f"Model 3 (Year FE):          {model3.params['ELIGIBLE_AFTER']:.4f} (SE: {model3.bse['ELIGIBLE_AFTER']:.4f})")
print(f"Model 4 (Full):             {model4.params['ELIGIBLE_AFTER']:.4f} (SE: {model4.bse['ELIGIBLE_AFTER']:.4f})")
print(f"Model 5 (State FE):         {model5.params['ELIGIBLE_AFTER']:.4f} (SE: {model5.bse['ELIGIBLE_AFTER']:.4f})")

print("\n*** INTERPRETATION ***")
if did_coef > 0:
    direction = "increased"
else:
    direction = "decreased"

if did_pval < 0.05:
    significance = "statistically significant at the 5% level"
elif did_pval < 0.10:
    significance = "marginally significant at the 10% level"
else:
    significance = "not statistically significant at conventional levels"

print(f"DACA eligibility {direction} full-time employment by {abs(did_coef)*100:.2f} percentage points.")
print(f"This effect is {significance} (p = {did_pval:.4f}).")

# Save results to file
results_df = pd.DataFrame({
    'Model': ['Basic DiD', 'Demographics', 'Year FE', 'Full Model', 'State FE'],
    'Coefficient': [model1.params['ELIGIBLE_AFTER'], model2.params['ELIGIBLE_AFTER'],
                   model3.params['ELIGIBLE_AFTER'], model4.params['ELIGIBLE_AFTER'],
                   model5.params['ELIGIBLE_AFTER']],
    'SE': [model1.bse['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'],
           model3.bse['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'],
           model5.bse['ELIGIBLE_AFTER']],
    'pvalue': [model1.pvalues['ELIGIBLE_AFTER'], model2.pvalues['ELIGIBLE_AFTER'],
               model3.pvalues['ELIGIBLE_AFTER'], model4.pvalues['ELIGIBLE_AFTER'],
               model5.pvalues['ELIGIBLE_AFTER']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)]
})
results_df.to_csv('regression_results.csv', index=False)
print("\nSaved: regression_results.csv")

# Save yearly rates for table
yearly_rates.to_csv('yearly_rates.csv', index=False)
print("Saved: yearly_rates.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
