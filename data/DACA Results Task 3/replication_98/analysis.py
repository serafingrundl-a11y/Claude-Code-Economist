"""
DACA Replication Analysis - ID 98
Difference-in-Differences Analysis of DACA Eligibility on Full-Time Employment

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the US,
what was the causal impact of DACA eligibility on probability of full-time employment?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND EXPLORE DATA
# =============================================================================

print("=" * 80)
print("DACA REPLICATION ANALYSIS - ID 98")
print("=" * 80)

# Load the numeric version for analysis
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\n1. DATA OVERVIEW")
print(f"   Total observations: {len(df):,}")
print(f"   Number of variables: {len(df.columns)}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. KEY VARIABLE SUMMARIES
# =============================================================================

print(f"\n2. KEY VARIABLE DISTRIBUTIONS")

# Check ELIGIBLE variable
print(f"\n   ELIGIBLE (Treatment Group Indicator):")
print(f"   {df['ELIGIBLE'].value_counts().sort_index().to_string()}")

# Check AFTER variable
print(f"\n   AFTER (Post-DACA Period):")
print(f"   {df['AFTER'].value_counts().sort_index().to_string()}")

# Check FT variable
print(f"\n   FT (Full-Time Employment Outcome):")
print(f"   {df['FT'].value_counts().sort_index().to_string()}")

# =============================================================================
# 3. SAMPLE COMPOSITION BY GROUP AND PERIOD
# =============================================================================

print(f"\n3. SAMPLE COMPOSITION")

# Cross-tabulation
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (31-35)', 'Treated (26-30)', 'Total']
crosstab.columns = ['Pre-DACA (2008-11)', 'Post-DACA (2013-16)', 'Total']
print(f"\n   Sample sizes by group and period:")
print(f"   {crosstab.to_string()}")

# =============================================================================
# 4. OUTCOME BY GROUP AND PERIOD
# =============================================================================

print(f"\n4. FULL-TIME EMPLOYMENT RATES BY GROUP AND PERIOD")

# Mean FT by group and period
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
ft_means = ft_means.round(4)
print(f"\n   {ft_means.to_string()}")

# Calculate group means for DiD
ft_00 = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()  # Control, Pre
ft_01 = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()  # Control, Post
ft_10 = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()  # Treated, Pre
ft_11 = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()  # Treated, Post

print(f"\n   Summary of FT Employment Rates:")
print(f"   Control group (31-35), Pre-DACA:  {ft_00:.4f} ({ft_00*100:.2f}%)")
print(f"   Control group (31-35), Post-DACA: {ft_01:.4f} ({ft_01*100:.2f}%)")
print(f"   Treated group (26-30), Pre-DACA:  {ft_10:.4f} ({ft_10*100:.2f}%)")
print(f"   Treated group (26-30), Post-DACA: {ft_11:.4f} ({ft_11*100:.2f}%)")

# Manual DiD calculation
did_control_change = ft_01 - ft_00
did_treated_change = ft_11 - ft_10
did_estimate = did_treated_change - did_control_change

print(f"\n   Difference-in-Differences Calculation:")
print(f"   Control group change (Post - Pre): {did_control_change:.4f} ({did_control_change*100:.2f} pp)")
print(f"   Treated group change (Post - Pre): {did_treated_change:.4f} ({did_treated_change*100:.2f} pp)")
print(f"   DiD Estimate:                      {did_estimate:.4f} ({did_estimate*100:.2f} pp)")

# =============================================================================
# 5. DIFFERENCE-IN-DIFFERENCES REGRESSION (MAIN SPECIFICATION)
# =============================================================================

print(f"\n{'='*80}")
print(f"5. MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION")
print(f"{'='*80}")

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Basic DiD regression: FT = b0 + b1*ELIGIBLE + b2*AFTER + b3*ELIGIBLE*AFTER
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')

print(f"\n   Model 1: Basic DiD (no controls)")
print(f"   FT = b0 + b1*ELIGIBLE + b2*AFTER + b3*ELIGIBLE*AFTER")
print(f"\n   {model1.summary2().tables[1].to_string()}")
print(f"\n   R-squared: {model1.rsquared:.4f}")
print(f"   N: {int(model1.nobs)}")

# =============================================================================
# 6. DiD WITH DEMOGRAPHIC CONTROLS
# =============================================================================

print(f"\n{'='*80}")
print(f"6. DiD WITH DEMOGRAPHIC CONTROLS")
print(f"{'='*80}")

# Check available control variables
print(f"\n   Available demographic variables:")
print(f"   SEX: {df['SEX'].value_counts().to_dict()}")
print(f"   MARST: {sorted(df['MARST'].unique())}")
print(f"   NCHILD: {sorted(df['NCHILD'].unique())[:10]}...")

# Create demographic control variables
# SEX: 1=Male, 2=Female
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# MARST: 1=Married spouse present, others=not married with spouse present
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Has children
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Age (continuous - centered at 30 for interpretability)
df['AGE_CENTERED'] = df['AGE'] - 30

# Model 2: DiD with basic demographic controls
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE_CENTERED',
                  data=df).fit(cov_type='HC1')

print(f"\n   Model 2: DiD with demographic controls (Sex, Marital Status, Children, Age)")
print(f"\n   {model2.summary2().tables[1].to_string()}")
print(f"\n   R-squared: {model2.rsquared:.4f}")
print(f"   N: {int(model2.nobs)}")

# =============================================================================
# 7. DiD WITH EDUCATION CONTROLS
# =============================================================================

print(f"\n{'='*80}")
print(f"7. DiD WITH EDUCATION CONTROLS")
print(f"{'='*80}")

# Check EDUC_RECODE variable
print(f"\n   EDUC_RECODE distribution:")
print(f"   {df['EDUC_RECODE'].value_counts().to_string()}")

# Model 3: Add education
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE_CENTERED + C(EDUC_RECODE)',
                  data=df).fit(cov_type='HC1')

print(f"\n   Model 3: DiD with demographics + education")
print(f"\n   Key coefficients:")
key_vars = ['Intercept', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'HAS_CHILDREN', 'AGE_CENTERED']
for var in key_vars:
    if var in model3.params.index:
        coef = model3.params[var]
        se = model3.bse[var]
        pval = model3.pvalues[var]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"   {var:25s}: {coef:8.4f} (SE: {se:.4f}){stars}")
print(f"\n   R-squared: {model3.rsquared:.4f}")
print(f"   N: {int(model3.nobs)}")

# =============================================================================
# 8. DiD WITH STATE AND YEAR FIXED EFFECTS
# =============================================================================

print(f"\n{'='*80}")
print(f"8. DiD WITH STATE AND YEAR FIXED EFFECTS")
print(f"{'='*80}")

# Model 4: Add state and year fixed effects (cannot include AFTER when using year FE)
model4 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE_CENTERED + C(EDUC_RECODE) + C(YEAR) + C(STATEFIP)',
                  data=df).fit(cov_type='HC1')

print(f"\n   Model 4: DiD with demographics, education, state FE, and year FE")
print(f"\n   Key coefficients (DiD and demographics only):")
key_vars = ['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'HAS_CHILDREN', 'AGE_CENTERED']
for var in key_vars:
    if var in model4.params.index:
        coef = model4.params[var]
        se = model4.bse[var]
        pval = model4.pvalues[var]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"   {var:25s}: {coef:8.4f} (SE: {se:.4f}){stars}")
print(f"\n   R-squared: {model4.rsquared:.4f}")
print(f"   N: {int(model4.nobs)}")

# =============================================================================
# 9. PREFERRED SPECIFICATION WITH CLUSTERED STANDARD ERRORS
# =============================================================================

print(f"\n{'='*80}")
print(f"9. PREFERRED SPECIFICATION (Clustered SE at State Level)")
print(f"{'='*80}")

# Create a subset without missing data for the model
df_model = df[['FT', 'ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'HAS_CHILDREN', 'AGE_CENTERED', 'EDUC_RECODE', 'YEAR', 'STATEFIP']].dropna()

# Model 5: Same as Model 4 but with clustered standard errors at state level
model5 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE_CENTERED + C(EDUC_RECODE) + C(YEAR) + C(STATEFIP)',
                  data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['STATEFIP']})

print(f"\n   Model 5: Full specification with state-clustered SE")
print(f"\n   Key coefficients (DiD and demographics only):")
key_vars = ['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'HAS_CHILDREN', 'AGE_CENTERED']
for var in key_vars:
    if var in model5.params.index:
        coef = model5.params[var]
        se = model5.bse[var]
        pval = model5.pvalues[var]
        ci_low, ci_high = model5.conf_int().loc[var]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"   {var:25s}: {coef:8.4f} (SE: {se:.4f}) [95% CI: {ci_low:.4f}, {ci_high:.4f}]{stars}")
print(f"\n   R-squared: {model5.rsquared:.4f}")
print(f"   N: {int(model5.nobs)}")

# =============================================================================
# 10. ROBUSTNESS CHECKS
# =============================================================================

print(f"\n{'='*80}")
print(f"10. ROBUSTNESS CHECKS")
print(f"{'='*80}")

# Check A: By gender
print(f"\n   A. Heterogeneity by Gender:")
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val].copy()
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub).fit(cov_type='HC1')
    coef = model_sub.params['ELIGIBLE_AFTER']
    se = model_sub.bse['ELIGIBLE_AFTER']
    pval = model_sub.pvalues['ELIGIBLE_AFTER']
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"      {sex_label:8s}: DiD = {coef:7.4f} (SE: {se:.4f}){stars}, N = {int(model_sub.nobs)}")

# Check B: By education level
print(f"\n   B. Heterogeneity by Education:")
for educ in df['EDUC_RECODE'].unique():
    df_sub = df[df['EDUC_RECODE'] == educ].copy()
    if len(df_sub) > 100:  # Only if enough observations
        model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub).fit(cov_type='HC1')
        coef = model_sub.params['ELIGIBLE_AFTER']
        se = model_sub.bse['ELIGIBLE_AFTER']
        pval = model_sub.pvalues['ELIGIBLE_AFTER']
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"      {str(educ)[:20]:20s}: DiD = {coef:7.4f} (SE: {se:.4f}){stars}, N = {int(model_sub.nobs)}")

# Check C: Year-by-year analysis
print(f"\n   C. Year-by-Year Analysis (Treated group FT rate by year):")
yearly_ft = df[df['ELIGIBLE']==1].groupby('YEAR')['FT'].mean()
for year, ft_rate in yearly_ft.items():
    period = "Pre" if year < 2012 else "Post"
    print(f"      {year} ({period}): {ft_rate:.4f} ({ft_rate*100:.2f}%)")

# Check D: Placebo test using only pre-period data
print(f"\n   D. Placebo Test (Pre-period only, fake treatment at 2010):")
df_pre = df[df['AFTER'] == 0].copy()
df_pre['FAKE_AFTER'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['FAKE_INTERACTION'] = df_pre['ELIGIBLE'] * df_pre['FAKE_AFTER']
model_placebo = smf.ols('FT ~ ELIGIBLE + FAKE_AFTER + FAKE_INTERACTION', data=df_pre).fit(cov_type='HC1')
coef_placebo = model_placebo.params['FAKE_INTERACTION']
se_placebo = model_placebo.bse['FAKE_INTERACTION']
pval_placebo = model_placebo.pvalues['FAKE_INTERACTION']
stars = '***' if pval_placebo < 0.01 else ('**' if pval_placebo < 0.05 else ('*' if pval_placebo < 0.1 else ''))
print(f"      Placebo DiD estimate: {coef_placebo:.4f} (SE: {se_placebo:.4f}){stars}")
print(f"      (A non-significant placebo supports parallel trends assumption)")

# =============================================================================
# 11. SUMMARY TABLE FOR REPORT
# =============================================================================

print(f"\n{'='*80}")
print(f"11. SUMMARY OF ALL MODELS")
print(f"{'='*80}")

models = [
    ('Model 1: Basic DiD', model1),
    ('Model 2: + Demographics', model2),
    ('Model 3: + Education', model3),
    ('Model 4: + State/Year FE (HC1)', model4),
    ('Model 5: + State Clustered SE', model5)
]

print(f"\n   {'Model':<35s} {'DiD Estimate':>12s} {'Std. Error':>12s} {'p-value':>10s} {'N':>8s} {'R-sq':>8s}")
print(f"   {'-'*35} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")

for name, model in models:
    coef = model.params['ELIGIBLE_AFTER']
    se = model.bse['ELIGIBLE_AFTER']
    pval = model.pvalues['ELIGIBLE_AFTER']
    n = int(model.nobs)
    rsq = model.rsquared
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"   {name:<35s} {coef:>11.4f}{stars} {se:>12.4f} {pval:>10.4f} {n:>8d} {rsq:>8.4f}")

# =============================================================================
# 12. FINAL PREFERRED ESTIMATE
# =============================================================================

print(f"\n{'='*80}")
print(f"12. PREFERRED ESTIMATE FOR REPORTING")
print(f"{'='*80}")

# Using Model 5 as preferred specification
preferred_coef = model5.params['ELIGIBLE_AFTER']
preferred_se = model5.bse['ELIGIBLE_AFTER']
preferred_ci = model5.conf_int().loc['ELIGIBLE_AFTER']
preferred_pval = model5.pvalues['ELIGIBLE_AFTER']
preferred_n = int(model5.nobs)

print(f"\n   PREFERRED SPECIFICATION: Model 5")
print(f"   (DiD with demographics, education, state/year FE, state-clustered SE)")
print(f"\n   Effect Size (DiD Estimate):     {preferred_coef:.4f}")
print(f"   Standard Error:                 {preferred_se:.4f}")
print(f"   95% Confidence Interval:        [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"   p-value:                        {preferred_pval:.4f}")
print(f"   Sample Size:                    {preferred_n:,}")
print(f"\n   Interpretation:")
print(f"   DACA eligibility is associated with a {preferred_coef*100:.2f} percentage point")
if preferred_pval < 0.05:
    print(f"   {'increase' if preferred_coef > 0 else 'decrease'} in full-time employment probability (statistically significant at 5% level).")
else:
    print(f"   change in full-time employment probability (not statistically significant at 5% level).")

# =============================================================================
# 13. ADDITIONAL DESCRIPTIVE STATISTICS FOR REPORT
# =============================================================================

print(f"\n{'='*80}")
print(f"13. DESCRIPTIVE STATISTICS FOR REPORT")
print(f"{'='*80}")

# Demographics by group
print(f"\n   A. Demographics by Treatment Group:")
for elig_val, elig_label in [(0, 'Control (31-35)'), (1, 'Treated (26-30)')]:
    df_sub = df[df['ELIGIBLE'] == elig_val]
    print(f"\n      {elig_label}:")
    print(f"         N = {len(df_sub):,}")
    print(f"         Mean Age: {df_sub['AGE'].mean():.2f}")
    print(f"         % Female: {df_sub['FEMALE'].mean()*100:.1f}%")
    print(f"         % Married (spouse present): {df_sub['MARRIED'].mean()*100:.1f}%")
    print(f"         % Has Children: {df_sub['HAS_CHILDREN'].mean()*100:.1f}%")
    print(f"         FT Employment Rate: {df_sub['FT'].mean()*100:.1f}%")

# Pre-treatment balance check
print(f"\n   B. Pre-Treatment Balance (2008-2011 only):")
df_pre = df[df['AFTER'] == 0]
for elig_val, elig_label in [(0, 'Control'), (1, 'Treated')]:
    df_sub = df_pre[df_pre['ELIGIBLE'] == elig_val]
    print(f"\n      {elig_label}:")
    print(f"         N = {len(df_sub):,}")
    print(f"         % Female: {df_sub['FEMALE'].mean()*100:.1f}%")
    print(f"         % Married: {df_sub['MARRIED'].mean()*100:.1f}%")
    print(f"         FT Rate: {df_sub['FT'].mean()*100:.1f}%")

# Year distribution
print(f"\n   C. Observations by Year:")
year_counts = df.groupby('YEAR').size()
for year, count in year_counts.items():
    print(f"      {year}: {count:,}")

# State distribution (top 10)
print(f"\n   D. Observations by State (Top 10):")
state_counts = df.groupby('statename').size().sort_values(ascending=False).head(10)
for state, count in state_counts.items():
    print(f"      {state}: {count:,} ({count/len(df)*100:.1f}%)")

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*80}")

# Save key results to file for report generation
results_summary = {
    'preferred_coef': preferred_coef,
    'preferred_se': preferred_se,
    'preferred_ci_low': preferred_ci[0],
    'preferred_ci_high': preferred_ci[1],
    'preferred_pval': preferred_pval,
    'sample_size': preferred_n,
    'manual_did': did_estimate,
    'ft_control_pre': ft_00,
    'ft_control_post': ft_01,
    'ft_treated_pre': ft_10,
    'ft_treated_post': ft_11
}

# Save results for later use
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\nResults saved to analysis_results.json")
