"""
DACA Replication Study Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
Identification Strategy: Difference-in-Differences
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
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# Load data
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_84\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"\nTotal observations: {len(df)}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# DATA EXPLORATION AND SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("DATA EXPLORATION")
print("=" * 80)

# Check key variables
print(f"\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts().sort_index())

print(f"\nAFTER distribution:")
print(df['AFTER'].value_counts().sort_index())

print(f"\nFT (Full-time employment) distribution:")
print(df['FT'].value_counts().sort_index())

# Sample sizes by group and period
print("\n" + "-" * 60)
print("Sample sizes by Treatment Group and Period:")
print("-" * 60)
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
crosstab.columns = ['Pre-DACA (2008-11)', 'Post-DACA (2013-16)', 'Total']
print(crosstab)

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

# FT employment rates by group and period (unweighted)
print("\n" + "-" * 60)
print("Full-time Employment Rates (Unweighted):")
print("-" * 60)
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre-DACA', 'Post-DACA']
print(ft_rates.round(4))

# Calculate simple DiD
did_simple = (ft_rates.loc['Treatment (26-30)', 'Post-DACA'] - ft_rates.loc['Treatment (26-30)', 'Pre-DACA']) - \
             (ft_rates.loc['Control (31-35)', 'Post-DACA'] - ft_rates.loc['Control (31-35)', 'Pre-DACA'])
print(f"\nSimple DiD estimate (unweighted): {did_simple:.4f}")

# FT employment rates by group and period (weighted)
print("\n" + "-" * 60)
print("Full-time Employment Rates (Weighted by PERWT):")
print("-" * 60)

def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

ft_rates_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean).unstack()
ft_rates_weighted.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates_weighted.columns = ['Pre-DACA', 'Post-DACA']
print(ft_rates_weighted.round(4))

# Calculate weighted simple DiD
did_weighted = (ft_rates_weighted.loc['Treatment (26-30)', 'Post-DACA'] - ft_rates_weighted.loc['Treatment (26-30)', 'Pre-DACA']) - \
               (ft_rates_weighted.loc['Control (31-35)', 'Post-DACA'] - ft_rates_weighted.loc['Control (31-35)', 'Pre-DACA'])
print(f"\nSimple DiD estimate (weighted): {did_weighted:.4f}")

# Additional demographics
print("\n" + "-" * 60)
print("Sample Characteristics by Treatment Group:")
print("-" * 60)

for elig in [0, 1]:
    group_name = "Treatment (26-30)" if elig == 1 else "Control (31-35)"
    subset = df[df['ELIGIBLE'] == elig]
    print(f"\n{group_name}:")
    print(f"  Mean Age: {subset['AGE'].mean():.2f}")
    print(f"  Female (%): {(subset['SEX'] == 2).mean()*100:.1f}%")
    print(f"  Mean years in USA: {subset['YRSUSA1'].mean():.2f}")

    # Education distribution
    educ_dist = subset['EDUC_RECODE'].value_counts(normalize=True).sort_index()
    print(f"  Education distribution:")
    for educ, pct in educ_dist.items():
        print(f"    {educ}: {pct*100:.1f}%")

# Year-by-year FT rates
print("\n" + "-" * 60)
print("Full-time Employment Rates by Year (Unweighted):")
print("-" * 60)
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_rates.columns = ['Control (31-35)', 'Treatment (26-30)']
print(yearly_rates.round(4))

# =============================================================================
# REGRESSION ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("REGRESSION ANALYSIS")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# --- Model 1: Basic DiD (no weights) ---
print("\n" + "-" * 60)
print("Model 1: Basic Difference-in-Differences (OLS, no weights)")
print("-" * 60)

model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model1.summary())

# --- Model 2: Basic DiD with survey weights ---
print("\n" + "-" * 60)
print("Model 2: Basic DiD with Survey Weights (WLS)")
print("-" * 60)

model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# --- Model 3: DiD with demographic controls ---
print("\n" + "-" * 60)
print("Model 3: DiD with Demographic Controls (WLS)")
print("-" * 60)

# Create demographic control variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Education dummies (reference: Less than High School)
df['HS_DEGREE_BIN'] = (df['HS_DEGREE'] == True).astype(int)

model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + YRSUSA1 + HS_DEGREE_BIN',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# --- Model 4: DiD with demographic controls and state fixed effects ---
print("\n" + "-" * 60)
print("Model 4: DiD with Demographic Controls and State Fixed Effects (WLS)")
print("-" * 60)

# Create state dummies
df['STATEFIP'] = df['STATEFIP'].astype('category')

model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + YRSUSA1 + HS_DEGREE_BIN + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Print only key coefficients (not all state FEs)
print("\nKey Coefficients (State FE suppressed):")
print("-" * 40)
key_vars = ['Intercept', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'AGE', 'YRSUSA1', 'HS_DEGREE_BIN']
for var in key_vars:
    if var in model4.params.index:
        coef = model4.params[var]
        se = model4.bse[var]
        pval = model4.pvalues[var]
        ci_low = model4.conf_int().loc[var, 0]
        ci_high = model4.conf_int().loc[var, 1]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"{var:20s}: {coef:8.4f} ({se:.4f}){stars}  [{ci_low:.4f}, {ci_high:.4f}]")

print(f"\nN = {int(model4.nobs)}")
print(f"R-squared = {model4.rsquared:.4f}")

# --- Model 5: DiD with year fixed effects instead of AFTER dummy ---
print("\n" + "-" * 60)
print("Model 5: DiD with Year Fixed Effects and State Fixed Effects (WLS)")
print("-" * 60)

df['YEAR'] = df['YEAR'].astype('category')

model5 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + YRSUSA1 + HS_DEGREE_BIN + C(YEAR) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nKey Coefficients (Year FE and State FE suppressed):")
print("-" * 40)
key_vars5 = ['Intercept', 'ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'AGE', 'YRSUSA1', 'HS_DEGREE_BIN']
for var in key_vars5:
    if var in model5.params.index:
        coef = model5.params[var]
        se = model5.bse[var]
        pval = model5.pvalues[var]
        ci_low = model5.conf_int().loc[var, 0]
        ci_high = model5.conf_int().loc[var, 1]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"{var:20s}: {coef:8.4f} ({se:.4f}){stars}  [{ci_low:.4f}, {ci_high:.4f}]")

print(f"\nN = {int(model5.nobs)}")
print(f"R-squared = {model5.rsquared:.4f}")

# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# --- Robustness 1: Probit model ---
print("\n" + "-" * 60)
print("Robustness Check 1: Probit Model (unweighted)")
print("-" * 60)

try:
    probit_model = smf.probit('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + YRSUSA1 + HS_DEGREE_BIN',
                              data=df).fit(disp=0)
    print("\nProbit Coefficients:")
    print(probit_model.summary().tables[1])

    # Calculate marginal effect of ELIGIBLE_AFTER
    # Average marginal effect approximation
    mfx = probit_model.get_margeff(at='overall')
    print("\nMarginal Effects at the Mean:")
    print(mfx.summary())
except Exception as e:
    print(f"Probit estimation failed: {e}")

# --- Robustness 2: Logit model ---
print("\n" + "-" * 60)
print("Robustness Check 2: Logit Model (unweighted)")
print("-" * 60)

try:
    logit_model = smf.logit('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + YRSUSA1 + HS_DEGREE_BIN',
                            data=df).fit(disp=0)
    print("\nLogit Coefficients:")
    print(logit_model.summary().tables[1])

    # Calculate marginal effect
    mfx_logit = logit_model.get_margeff(at='overall')
    print("\nMarginal Effects at the Mean:")
    print(mfx_logit.summary())
except Exception as e:
    print(f"Logit estimation failed: {e}")

# --- Robustness 3: Check for differential pre-trends ---
print("\n" + "-" * 60)
print("Pre-trends Analysis:")
print("-" * 60)

pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_NUM'] = pre_data['YEAR'].astype(int)
pre_data['YEAR_ELIGIBLE'] = pre_data['YEAR_NUM'] * pre_data['ELIGIBLE']

pretrend_model = smf.wls('FT ~ ELIGIBLE + YEAR_NUM + YEAR_ELIGIBLE',
                         data=pre_data, weights=pre_data['PERWT']).fit(cov_type='HC1')
print("\nPre-trends test (interaction of ELIGIBLE with linear time trend):")
print(f"YEAR_ELIGIBLE coefficient: {pretrend_model.params['YEAR_ELIGIBLE']:.6f}")
print(f"Standard error: {pretrend_model.bse['YEAR_ELIGIBLE']:.6f}")
print(f"P-value: {pretrend_model.pvalues['YEAR_ELIGIBLE']:.4f}")

if pretrend_model.pvalues['YEAR_ELIGIBLE'] > 0.05:
    print("=> No significant differential pre-trend detected (p > 0.05)")
else:
    print("=> Warning: Differential pre-trend may be present (p < 0.05)")

# --- Robustness 4: Placebo test using only pre-period ---
print("\n" + "-" * 60)
print("Placebo Test (Pre-period only, 2010-2011 as 'post'):")
print("-" * 60)

placebo_data = df[df['AFTER'] == 0].copy()
# Need to restore YEAR as integer since it was converted to categorical
# Use the original year values from the categories
placebo_data['YEAR_INT'] = placebo_data['YEAR'].astype(str).astype(int)
placebo_data['PLACEBO_POST'] = (placebo_data['YEAR_INT'] >= 2010).astype(int)
placebo_data['PLACEBO_INTERACTION'] = placebo_data['ELIGIBLE'] * placebo_data['PLACEBO_POST']

placebo_model = smf.wls('FT ~ ELIGIBLE + PLACEBO_POST + PLACEBO_INTERACTION + FEMALE + MARRIED + AGE + YRSUSA1 + HS_DEGREE_BIN',
                        data=placebo_data, weights=placebo_data['PERWT']).fit(cov_type='HC1')

print(f"Placebo DiD coefficient: {placebo_model.params['PLACEBO_INTERACTION']:.4f}")
print(f"Standard error: {placebo_model.bse['PLACEBO_INTERACTION']:.4f}")
print(f"P-value: {placebo_model.pvalues['PLACEBO_INTERACTION']:.4f}")

if placebo_model.pvalues['PLACEBO_INTERACTION'] > 0.05:
    print("=> Placebo test passed: No significant effect in pre-period (p > 0.05)")
else:
    print("=> Warning: Placebo test suggests pre-existing differences")

# =============================================================================
# HETEROGENEITY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("HETEROGENEITY ANALYSIS")
print("=" * 80)

# --- By Gender ---
print("\n" + "-" * 60)
print("Effect by Gender:")
print("-" * 60)

for gender in [1, 2]:
    gender_name = "Male" if gender == 1 else "Female"
    subset = df[df['SEX'] == gender]
    model_gender = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MARRIED + AGE + YRSUSA1 + HS_DEGREE_BIN',
                           data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"\n{gender_name} (N={len(subset)}):")
    print(f"  DiD estimate: {model_gender.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  SE: {model_gender.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"  95% CI: [{model_gender.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_gender.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
    print(f"  P-value: {model_gender.pvalues['ELIGIBLE_AFTER']:.4f}")

# --- By Education ---
print("\n" + "-" * 60)
print("Effect by Education Level:")
print("-" * 60)

for hs in [0, 1]:
    educ_name = "HS Degree or Higher" if hs == 1 else "Less than HS"
    subset = df[df['HS_DEGREE_BIN'] == hs]
    if len(subset) > 100:  # Only if enough observations
        model_educ = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + YRSUSA1',
                             data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
        print(f"\n{educ_name} (N={len(subset)}):")
        print(f"  DiD estimate: {model_educ.params['ELIGIBLE_AFTER']:.4f}")
        print(f"  SE: {model_educ.bse['ELIGIBLE_AFTER']:.4f}")
        print(f"  95% CI: [{model_educ.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_educ.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
        print(f"  P-value: {model_educ.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# SUMMARY OF KEY RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF KEY RESULTS")
print("=" * 80)

print("\n" + "-" * 60)
print("Main DiD Estimates Across Specifications:")
print("-" * 60)
print(f"{'Model':<50} {'Estimate':>10} {'SE':>10} {'P-value':>10}")
print("-" * 80)
print(f"{'1. Basic OLS (no weights)':<50} {model1.params['ELIGIBLE_AFTER']:>10.4f} {model1.bse['ELIGIBLE_AFTER']:>10.4f} {model1.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'2. WLS (survey weights)':<50} {model2.params['ELIGIBLE_AFTER']:>10.4f} {model2.bse['ELIGIBLE_AFTER']:>10.4f} {model2.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'3. WLS + demographic controls':<50} {model3.params['ELIGIBLE_AFTER']:>10.4f} {model3.bse['ELIGIBLE_AFTER']:>10.4f} {model3.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'4. WLS + demographics + state FE':<50} {model4.params['ELIGIBLE_AFTER']:>10.4f} {model4.bse['ELIGIBLE_AFTER']:>10.4f} {model4.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'5. WLS + demographics + state FE + year FE':<50} {model5.params['ELIGIBLE_AFTER']:>10.4f} {model5.bse['ELIGIBLE_AFTER']:>10.4f} {model5.pvalues['ELIGIBLE_AFTER']:>10.4f}")

# PREFERRED ESTIMATE (Model 4 or 5 with controls and FE)
preferred_model = model4
print("\n" + "-" * 60)
print("PREFERRED ESTIMATE: Model 4 (WLS with demographics + state FE)")
print("-" * 60)
print(f"Effect of DACA eligibility on full-time employment: {preferred_model.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust standard error: {preferred_model.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence interval: [{preferred_model.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {preferred_model.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {preferred_model.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample size: {int(preferred_model.nobs)}")

# Save key results to a file for the report
results_summary = {
    'model1_coef': model1.params['ELIGIBLE_AFTER'],
    'model1_se': model1.bse['ELIGIBLE_AFTER'],
    'model1_pval': model1.pvalues['ELIGIBLE_AFTER'],
    'model2_coef': model2.params['ELIGIBLE_AFTER'],
    'model2_se': model2.bse['ELIGIBLE_AFTER'],
    'model2_pval': model2.pvalues['ELIGIBLE_AFTER'],
    'model3_coef': model3.params['ELIGIBLE_AFTER'],
    'model3_se': model3.bse['ELIGIBLE_AFTER'],
    'model3_pval': model3.pvalues['ELIGIBLE_AFTER'],
    'model4_coef': model4.params['ELIGIBLE_AFTER'],
    'model4_se': model4.bse['ELIGIBLE_AFTER'],
    'model4_pval': model4.pvalues['ELIGIBLE_AFTER'],
    'model4_ci_low': model4.conf_int().loc['ELIGIBLE_AFTER', 0],
    'model4_ci_high': model4.conf_int().loc['ELIGIBLE_AFTER', 1],
    'model5_coef': model5.params['ELIGIBLE_AFTER'],
    'model5_se': model5.bse['ELIGIBLE_AFTER'],
    'model5_pval': model5.pvalues['ELIGIBLE_AFTER'],
    'n_obs': int(preferred_model.nobs),
    'simple_did_unweighted': did_simple,
    'simple_did_weighted': did_weighted
}

# Save to JSON
import json
with open(r'C:\Users\seraf\DACA Results Task 3\replication_84\analysis_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("Results saved to analysis_results.json")
print("=" * 80)
