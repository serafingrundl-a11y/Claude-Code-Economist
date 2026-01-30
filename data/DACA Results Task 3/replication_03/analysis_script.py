"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among ethnically Hispanic-Mexican Mexican-born individuals in the US.

Treatment: DACA eligibility (ages 26-30 at June 2012)
Control: Ages 31-35 at June 2012 (would have been eligible but for age)
Pre-period: 2008-2011
Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 80)
print("DACA REPLICATION ANALYSIS")
print("=" * 80)

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"\nDataset loaded: {df.shape[0]:,} observations, {df.shape[1]} variables")

# =============================================================================
# 2. DATA PREPARATION
# =============================================================================
print("\n" + "=" * 80)
print("DATA PREPARATION")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Create female indicator (SEX: 1=Male, 2=Female)
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Create education dummies (using EDUC_RECODE)
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_AA'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)
# Reference: Less than High School

# Create married indicator (MARST: 1=Married spouse present)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Create year dummies for event study
for year in df['YEAR'].unique():
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

# Verify key variables
print("\nKey Variables Summary:")
print(f"ELIGIBLE: {df['ELIGIBLE'].sum():,} treated, {(1-df['ELIGIBLE']).sum():,} control")
print(f"AFTER: {df['AFTER'].sum():,} post-treatment, {(1-df['AFTER']).sum():,} pre-treatment")
print(f"FT (full-time): {df['FT'].sum():,} employed FT ({df['FT'].mean()*100:.1f}%)")

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

# Table 1: Sample characteristics by treatment status
print("\n--- Table 1: Sample Characteristics by ELIGIBLE Status ---")
vars_desc = ['FT', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD', 'FAMSIZE']
desc_stats = df.groupby('ELIGIBLE')[vars_desc].agg(['mean', 'std'])
print(desc_stats.round(3))

# Cross-tabulation
print("\n--- Cross-tabulation: ELIGIBLE x AFTER ---")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
print(crosstab)

# FT rates by group and period
print("\n--- Full-Time Employment Rates by Group and Period ---")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
ft_rates.columns = ['FT_Rate', 'Std_Dev', 'N']
print(ft_rates.round(4))

# Calculate raw DID
ft_pivot = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
did_raw = (ft_pivot.loc[1, 1] - ft_pivot.loc[1, 0]) - (ft_pivot.loc[0, 1] - ft_pivot.loc[0, 0])
print(f"\nRaw DID Estimate: {did_raw:.4f} ({did_raw*100:.2f} percentage points)")

# Education distribution by treatment status
print("\n--- Education Distribution by ELIGIBLE Status ---")
educ_dist = pd.crosstab(df['ELIGIBLE'], df['EDUC_RECODE'], normalize='index') * 100
print(educ_dist.round(1))

# Year-by-group FT rates
print("\n--- FT Rates by Year and ELIGIBLE Status ---")
yearly_ft = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_ft.columns = ['Control (31-35)', 'Treated (26-30)']
print(yearly_ft.round(4))

# =============================================================================
# 4. MAIN DID REGRESSION ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("=" * 80)

# Model 1: Basic DID (no covariates)
print("\n--- Model 1: Basic DID ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])

# Model 2: DID with demographic controls
print("\n--- Model 2: DID with Demographic Controls ---")
formula2 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + I(AGE**2) + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA'
model2 = smf.ols(formula2, data=df).fit()
print(model2.summary().tables[1])

# Model 3: DID with demographic + state policy controls
print("\n--- Model 3: DID with Demographic + State Policy Controls ---")
formula3 = '''FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + I(AGE**2) + MARRIED + NCHILD +
              EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA +
              DRIVERSLICENSES + INSTATETUITION + EVERIFY + OMNIBUS + UNEMP + LFPR'''
model3 = smf.ols(formula3, data=df).fit()
print(model3.summary().tables[1])

# Model 4: DID with state fixed effects and clustered SEs
print("\n--- Model 4: DID with State Fixed Effects ---")
formula4 = '''FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + I(AGE**2) + MARRIED + NCHILD +
              EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + C(STATEFIP)'''
model4 = smf.ols(formula4, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DID Coefficient: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 5: Weighted DID (PREFERRED SPECIFICATION)
print("\n--- Model 5: Weighted DID with State FE (PREFERRED) ---")
model5 = smf.wls(formula4, data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DID Coefficient: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model5.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# =============================================================================
# 5. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# 5.1 Event Study / Pre-trends Analysis
print("\n--- Event Study: Year-by-Year Effects ---")
# Create year interactions with ELIGIBLE (omit 2011 as reference)
df['ELIGIBLE_2008'] = df['ELIGIBLE'] * df['YEAR_2008']
df['ELIGIBLE_2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIGIBLE_2010'] = df['ELIGIBLE'] * df['YEAR_2010']
# 2011 is reference
df['ELIGIBLE_2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIGIBLE_2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIGIBLE_2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIGIBLE_2016'] = df['ELIGIBLE'] * df['YEAR_2016']

formula_event = '''FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 +
                   ELIGIBLE_2008 + ELIGIBLE_2009 + ELIGIBLE_2010 + ELIGIBLE_2013 + ELIGIBLE_2014 + ELIGIBLE_2015 + ELIGIBLE_2016 +
                   FEMALE + AGE + I(AGE**2) + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA'''
model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("Pre-treatment coefficients (should be close to 0 for parallel trends):")
for year in [2008, 2009, 2010]:
    var = f'ELIGIBLE_{year}'
    print(f"  {year}: {model_event.params[var]:.4f} (SE: {model_event.bse[var]:.4f}, p={model_event.pvalues[var]:.3f})")

print("\nPost-treatment coefficients:")
for year in [2013, 2014, 2015, 2016]:
    var = f'ELIGIBLE_{year}'
    print(f"  {year}: {model_event.params[var]:.4f} (SE: {model_event.bse[var]:.4f}, p={model_event.pvalues[var]:.3f})")

# 5.2 Heterogeneity by Gender
print("\n--- Heterogeneity Analysis: By Gender ---")
df_male = df[df['FEMALE'] == 0]
df_female = df[df['FEMALE'] == 1]

formula_het = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + I(AGE**2) + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA'
model_male = smf.wls(formula_het, data=df_male, weights=df_male['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
model_female = smf.wls(formula_het, data=df_female, weights=df_female['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})

print(f"Males:   DID = {model_male.params['ELIGIBLE_AFTER']:.4f} (SE: {model_male.bse['ELIGIBLE_AFTER']:.4f}, p={model_male.pvalues['ELIGIBLE_AFTER']:.3f})")
print(f"Females: DID = {model_female.params['ELIGIBLE_AFTER']:.4f} (SE: {model_female.bse['ELIGIBLE_AFTER']:.4f}, p={model_female.pvalues['ELIGIBLE_AFTER']:.3f})")

# 5.3 Heterogeneity by Education
print("\n--- Heterogeneity Analysis: By Education ---")
df_loweduc = df[df['EDUC_RECODE'].isin(['Less than High School', 'High School Degree'])]
df_higheduc = df[~df['EDUC_RECODE'].isin(['Less than High School', 'High School Degree'])]

formula_educ = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + I(AGE**2) + MARRIED + NCHILD'
model_loweduc = smf.wls(formula_educ, data=df_loweduc, weights=df_loweduc['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_loweduc['STATEFIP']})
model_higheduc = smf.wls(formula_educ, data=df_higheduc, weights=df_higheduc['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_higheduc['PERWT']})

print(f"HS or less:    DID = {model_loweduc.params['ELIGIBLE_AFTER']:.4f} (SE: {model_loweduc.bse['ELIGIBLE_AFTER']:.4f}, p={model_loweduc.pvalues['ELIGIBLE_AFTER']:.3f})")
print(f"Some college+: DID = {model_higheduc.params['ELIGIBLE_AFTER']:.4f} (SE: {model_higheduc.bse['ELIGIBLE_AFTER']:.4f}, p={model_higheduc.pvalues['ELIGIBLE_AFTER']:.3f})")

# 5.4 Logit Model
print("\n--- Logit Model for Comparison ---")
formula_logit = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA'
model_logit = smf.logit(formula_logit, data=df).fit(disp=0)
# Calculate marginal effect at means
print(f"Logit coefficient on ELIGIBLE_AFTER: {model_logit.params['ELIGIBLE_AFTER']:.4f}")
print(f"Odds Ratio: {np.exp(model_logit.params['ELIGIBLE_AFTER']):.4f}")

# Average marginal effect approximation
p_avg = df['FT'].mean()
marginal_effect = model_logit.params['ELIGIBLE_AFTER'] * p_avg * (1 - p_avg)
print(f"Approx. Average Marginal Effect: {marginal_effect:.4f}")

# =============================================================================
# 6. SUMMARY RESULTS TABLE
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF MAIN RESULTS")
print("=" * 80)

results_summary = pd.DataFrame({
    'Model': ['(1) Basic DID', '(2) + Demographics', '(3) + State Policy', '(4) + State FE', '(5) Weighted + State FE'],
    'DID_Coefficient': [
        model1.params['ELIGIBLE_AFTER'],
        model2.params['ELIGIBLE_AFTER'],
        model3.params['ELIGIBLE_AFTER'],
        model4.params['ELIGIBLE_AFTER'],
        model5.params['ELIGIBLE_AFTER']
    ],
    'Std_Error': [
        model1.bse['ELIGIBLE_AFTER'],
        model2.bse['ELIGIBLE_AFTER'],
        model3.bse['ELIGIBLE_AFTER'],
        model4.bse['ELIGIBLE_AFTER'],
        model5.bse['ELIGIBLE_AFTER']
    ],
    'p_value': [
        model1.pvalues['ELIGIBLE_AFTER'],
        model2.pvalues['ELIGIBLE_AFTER'],
        model3.pvalues['ELIGIBLE_AFTER'],
        model4.pvalues['ELIGIBLE_AFTER'],
        model5.pvalues['ELIGIBLE_AFTER']
    ],
    'N': [model1.nobs, model2.nobs, model3.nobs, model4.nobs, model5.nobs],
    'R_squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared, model5.rsquared]
})

print(results_summary.to_string(index=False))

# =============================================================================
# 7. SAVE KEY RESULTS FOR LATEX
# =============================================================================
print("\n" + "=" * 80)
print("KEY RESULTS FOR REPORT")
print("=" * 80)

preferred = model5
print(f"\nPREFERRED ESTIMATE (Weighted DID with State FE and Clustered SE):")
print(f"  Effect Size: {preferred.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error: {preferred.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{preferred.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {preferred.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  p-value: {preferred.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  Sample Size: {int(preferred.nobs):,}")

print("\nInterpretation:")
effect_pp = preferred.params['ELIGIBLE_AFTER'] * 100
print(f"  DACA eligibility increased full-time employment by {effect_pp:.1f} percentage points")
print(f"  among the eligible population (ages 26-30 at policy implementation).")

# Save results to CSV for LaTeX tables
results_summary.to_csv('results_summary.csv', index=False)
print("\nResults saved to results_summary.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
