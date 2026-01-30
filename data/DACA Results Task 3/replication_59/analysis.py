"""
DACA Replication Study - Difference-in-Differences Analysis
Effect of DACA eligibility on full-time employment among Mexican-born Hispanic individuals
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 60)
print("DACA REPLICATION STUDY - ANALYSIS SCRIPT")
print("=" * 60)

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"\nData loaded: {df.shape[0]} observations, {df.shape[1]} variables")

# ============================================================
# SECTION 1: DATA EXPLORATION
# ============================================================
print("\n" + "=" * 60)
print("SECTION 1: DATA EXPLORATION")
print("=" * 60)

# Sample composition
print("\n--- Sample Composition ---")
print(f"Total observations: {len(df)}")
print(f"ELIGIBLE=1 (Treated group, ages 26-30 in June 2012): {(df['ELIGIBLE']==1).sum()}")
print(f"ELIGIBLE=0 (Control group, ages 31-35 in June 2012): {(df['ELIGIBLE']==0).sum()}")

# Time periods
print("\n--- Time Periods ---")
print(f"Pre-DACA (AFTER=0, years 2008-2011): {(df['AFTER']==0).sum()}")
print(f"Post-DACA (AFTER=1, years 2013-2016): {(df['AFTER']==1).sum()}")

# Year breakdown
print("\n--- Observations by Year ---")
print(df['YEAR'].value_counts().sort_index())

# Full-time employment rates
print("\n--- Full-Time Employment Rates ---")
print(f"Overall FT rate: {df['FT'].mean():.4f}")

# Create 2x2 table
print("\n--- 2x2 DiD Table: Mean FT Employment ---")
did_table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'std'])
print(did_table)

# Calculate simple DiD
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
print("\n--- FT Employment Rates Matrix ---")
print(ft_means)

did_estimate_simple = (ft_means.loc[1, 1] - ft_means.loc[1, 0]) - (ft_means.loc[0, 1] - ft_means.loc[0, 0])
print(f"\nSimple DiD estimate: {did_estimate_simple:.4f}")
print(f"  Treated (26-30) change: {ft_means.loc[1, 1] - ft_means.loc[1, 0]:.4f}")
print(f"  Control (31-35) change: {ft_means.loc[0, 1] - ft_means.loc[0, 0]:.4f}")

# ============================================================
# SECTION 2: MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 60)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls, no weights)
print("\n--- Model 1: Basic DiD (OLS, no controls, no weights) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary())

# Model 2: Basic DiD with survey weights
print("\n--- Model 2: Basic DiD (WLS with PERWT weights) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary())

# Model 3: DiD with robust standard errors (HC1)
print("\n--- Model 3: Basic DiD (OLS with robust SE) ---")
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model3.summary())

# ============================================================
# SECTION 3: DiD WITH COVARIATES
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: DiD WITH COVARIATES")
print("=" * 60)

# Create additional binary/categorical variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)  # Married spouse present or absent

# Education categories (create dummies)
# EDUC_RECODE values based on typical categories
df['ED_LESSTHANHS'] = (df['EDUC_RECODE'] == 'Less than High School').astype(int)
df['ED_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['ED_SOMECOLLEGE'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['ED_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['ED_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

print("\n--- Education Distribution ---")
print(df['EDUC_RECODE'].value_counts())

# Model 4: DiD with demographic controls
print("\n--- Model 4: DiD with demographic controls (age, sex, marital status) ---")
model4 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED', data=df).fit(cov_type='HC1')
print(model4.summary())

# Model 5: DiD with demographic + education controls
print("\n--- Model 5: DiD with demographic + education controls ---")
model5 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + ED_HS + ED_SOMECOLLEGE + ED_TWOYEAR + ED_BA', data=df).fit(cov_type='HC1')
print(model5.summary())

# Model 6: Full model with state fixed effects
print("\n--- Model 6: DiD with state fixed effects ---")
model6 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)', data=df).fit(cov_type='HC1')
print(f"\nDiD Coefficient (ELIGIBLE_AFTER): {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model6.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 7: Full model with state and year fixed effects
print("\n--- Model 7: DiD with state and year fixed effects ---")
model7 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(STATEFIP) + C(YEAR)', data=df).fit(cov_type='HC1')
print(f"\nDiD Coefficient (ELIGIBLE_AFTER): {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model7.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model7.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 8: Full model with all controls (PREFERRED SPECIFICATION)
print("\n--- Model 8: PREFERRED SPECIFICATION ---")
print("DiD with demographics, education, state FE, and year FE")
model8 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + ED_HS + ED_SOMECOLLEGE + ED_TWOYEAR + ED_BA + C(STATEFIP) + C(YEAR)', data=df).fit(cov_type='HC1')
print(f"\nDiD Coefficient (ELIGIBLE_AFTER): {model8.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model8.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model8.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model8.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model8.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model8.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"N: {int(model8.nobs)}")
print(f"R-squared: {model8.rsquared:.4f}")

# ============================================================
# SECTION 4: ROBUSTNESS CHECKS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: ROBUSTNESS CHECKS")
print("=" * 60)

# Robustness 1: Linear Probability Model with clustered SE by state
print("\n--- Robustness 1: Clustered standard errors by state ---")
model_clustered = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + C(STATEFIP) + C(YEAR)', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model_clustered.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {model_clustered.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_clustered.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_clustered.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Robustness 2: Weighted regression
print("\n--- Robustness 2: WLS with PERWT weights ---")
model_wls = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + C(STATEFIP) + C(YEAR)', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model_wls.params['ELIGIBLE_AFTER']:.4f}")
print(f"SE: {model_wls.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_wls.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_wls.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Robustness 3: Logit model
print("\n--- Robustness 3: Logit model (marginal effects) ---")
try:
    logit_model = smf.logit('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED', data=df).fit(disp=0)
    # Calculate marginal effect at means
    print(f"Logit coefficient on ELIGIBLE_AFTER: {logit_model.params['ELIGIBLE_AFTER']:.4f}")
    # Approximate marginal effect
    p_mean = df['FT'].mean()
    marginal_effect = logit_model.params['ELIGIBLE_AFTER'] * p_mean * (1 - p_mean)
    print(f"Approximate marginal effect: {marginal_effect:.4f}")
except Exception as e:
    print(f"Logit model estimation issue: {e}")

# Robustness 4: Placebo test - pre-trends
print("\n--- Robustness 4: Pre-trends test (2008-2011 only) ---")
df_pre = df[df['AFTER'] == 0].copy()
# Create a "fake post" period (2010-2011)
df_pre['FAKE_POST'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['ELIGIBLE_FAKEPOST'] = df_pre['ELIGIBLE'] * df_pre['FAKE_POST']
model_placebo = smf.ols('FT ~ ELIGIBLE + FAKE_POST + ELIGIBLE_FAKEPOST', data=df_pre).fit(cov_type='HC1')
print(f"Placebo DiD Coefficient: {model_placebo.params['ELIGIBLE_FAKEPOST']:.4f}")
print(f"SE: {model_placebo.bse['ELIGIBLE_FAKEPOST']:.4f}")
print(f"p-value: {model_placebo.pvalues['ELIGIBLE_FAKEPOST']:.4f}")
print("(Non-significant coefficient suggests parallel pre-trends)")

# ============================================================
# SECTION 5: HETEROGENEITY ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: HETEROGENEITY ANALYSIS")
print("=" * 60)

# By gender
print("\n--- By Gender ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex]
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE = {model_sub.bse['ELIGIBLE_AFTER']:.4f}), N = {len(df_sub)}")

# By education
print("\n--- By Education ---")
for ed_level in ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']:
    df_sub = df[df['EDUC_RECODE'] == ed_level]
    if len(df_sub) > 100:
        model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub).fit(cov_type='HC1')
        print(f"{ed_level}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE = {model_sub.bse['ELIGIBLE_AFTER']:.4f}), N = {len(df_sub)}")

# By marital status
print("\n--- By Marital Status ---")
df['MARRIED_CAT'] = df['MARST'].apply(lambda x: 'Married' if x in [1, 2] else 'Not Married')
for mar_status in ['Married', 'Not Married']:
    df_sub = df[df['MARRIED_CAT'] == mar_status]
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub).fit(cov_type='HC1')
    print(f"{mar_status}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE = {model_sub.bse['ELIGIBLE_AFTER']:.4f}), N = {len(df_sub)}")

# ============================================================
# SECTION 6: EVENT STUDY / DYNAMIC EFFECTS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6: EVENT STUDY / YEAR-BY-YEAR EFFECTS")
print("=" * 60)

# Create year dummies and interactions
print("\n--- Year-by-Year DiD Effects ---")
for year in sorted(df['YEAR'].unique()):
    df_pair = df[df['YEAR'].isin([2011, year])]
    if year != 2011:
        df_pair = df_pair.copy()
        df_pair['POST'] = (df_pair['YEAR'] == year).astype(int)
        df_pair['ELIGIBLE_POST'] = df_pair['ELIGIBLE'] * df_pair['POST']
        model_year = smf.ols('FT ~ ELIGIBLE + POST + ELIGIBLE_POST', data=df_pair).fit(cov_type='HC1')
        print(f"2011 vs {year}: DiD = {model_year.params['ELIGIBLE_POST']:.4f} (SE = {model_year.bse['ELIGIBLE_POST']:.4f})")

# ============================================================
# SECTION 7: SUMMARY OF KEY RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 7: SUMMARY OF KEY RESULTS")
print("=" * 60)

print("\n--- PREFERRED ESTIMATE ---")
print(f"Model: Linear Probability Model with DiD")
print(f"Outcome: Full-time employment (FT)")
print(f"Treatment: DACA eligibility (ages 26-30 in June 2012)")
print(f"Control: Ages 31-35 in June 2012 (would be eligible if younger)")
print(f"\nDiD Coefficient: {model8.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model8.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model8.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model8.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model8.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample Size: {int(model8.nobs)}")

# Save key results to file
results_dict = {
    'Model': ['Basic DiD', 'DiD + Demographics', 'DiD + Demographics + Education',
              'DiD + State FE', 'DiD + State + Year FE', 'PREFERRED (Full Model)',
              'Clustered SE', 'WLS Weighted'],
    'DiD_Coefficient': [
        model1.params['ELIGIBLE_AFTER'],
        model4.params['ELIGIBLE_AFTER'],
        model5.params['ELIGIBLE_AFTER'],
        model6.params['ELIGIBLE_AFTER'],
        model7.params['ELIGIBLE_AFTER'],
        model8.params['ELIGIBLE_AFTER'],
        model_clustered.params['ELIGIBLE_AFTER'],
        model_wls.params['ELIGIBLE_AFTER']
    ],
    'SE': [
        model1.bse['ELIGIBLE_AFTER'],
        model4.bse['ELIGIBLE_AFTER'],
        model5.bse['ELIGIBLE_AFTER'],
        model6.bse['ELIGIBLE_AFTER'],
        model7.bse['ELIGIBLE_AFTER'],
        model8.bse['ELIGIBLE_AFTER'],
        model_clustered.bse['ELIGIBLE_AFTER'],
        model_wls.bse['ELIGIBLE_AFTER']
    ],
    'N': [
        int(model1.nobs),
        int(model4.nobs),
        int(model5.nobs),
        int(model6.nobs),
        int(model7.nobs),
        int(model8.nobs),
        int(model_clustered.nobs),
        int(model_wls.nobs)
    ]
}

results_df = pd.DataFrame(results_dict)
results_df['CI_Lower'] = results_df['DiD_Coefficient'] - 1.96 * results_df['SE']
results_df['CI_Upper'] = results_df['DiD_Coefficient'] + 1.96 * results_df['SE']
results_df.to_csv('results_summary.csv', index=False)
print("\nResults saved to results_summary.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
