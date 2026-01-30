"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("="*70)
print("DACA REPLICATION ANALYSIS")
print("="*70)

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

print(f"\nDataset loaded: {df.shape[0]} observations, {df.shape[1]} variables")

# ============================================================================
# SECTION 1: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*70)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("="*70)

# Sample composition
print("\n--- Sample Composition ---")
print(f"Total observations: {len(df)}")
print(f"Treatment group (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum()}")
print(f"Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum()}")
print(f"Pre-period (AFTER=0): {(df['AFTER']==0).sum()}")
print(f"Post-period (AFTER=1): {(df['AFTER']==1).sum()}")

# Year distribution
print("\n--- Year Distribution ---")
print(df['YEAR'].value_counts().sort_index())

# Cross-tabulation of treatment and time
print("\n--- Cross-tabulation: ELIGIBLE x AFTER ---")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (0)', 'Treatment (1)', 'Total']
crosstab.columns = ['Pre (0)', 'Post (1)', 'Total']
print(crosstab)

# FT employment rates by group and time
print("\n--- Full-Time Employment Rates by Group and Time ---")
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
print(ft_means)

# Compute raw DiD
ft_treat_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
ft_treat_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
ft_ctrl_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
ft_ctrl_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

raw_did = (ft_treat_post - ft_treat_pre) - (ft_ctrl_post - ft_ctrl_pre)

print(f"\n--- Raw Difference-in-Differences Calculation ---")
print(f"Treatment Pre:  {ft_treat_pre:.4f}")
print(f"Treatment Post: {ft_treat_post:.4f}")
print(f"Treatment Change: {ft_treat_post - ft_treat_pre:.4f}")
print(f"Control Pre:    {ft_ctrl_pre:.4f}")
print(f"Control Post:   {ft_ctrl_post - ft_ctrl_pre:.4f}")
print(f"Control Change: {ft_ctrl_post - ft_ctrl_pre:.4f}")
print(f"RAW DiD ESTIMATE: {raw_did:.4f} ({raw_did*100:.2f} percentage points)")

# Demographics summary
print("\n--- Demographic Summary by Group ---")
demo_vars = ['SEX', 'AGE', 'MARST', 'EDUC', 'NCHILD']
for var in demo_vars:
    if var in df.columns:
        print(f"\n{var}:")
        print(df.groupby('ELIGIBLE')[var].describe()[['mean', 'std']])

# ============================================================================
# SECTION 2: DIFFERENCE-IN-DIFFERENCES REGRESSION
# ============================================================================
print("\n" + "="*70)
print("SECTION 2: DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("="*70)

# Create interaction term
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (Unweighted, No Covariates) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit()
print(model1.summary())

# Model 2: Basic DiD with person weights
print("\n--- Model 2: Basic DiD (Weighted, No Covariates) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                 data=df, weights=df['PERWT']).fit()
print(model2.summary())

# Model 3: Basic DiD with robust standard errors
print("\n--- Model 3: Basic DiD (Weighted, Robust SE) ---")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# ============================================================================
# SECTION 3: MODELS WITH COVARIATES
# ============================================================================
print("\n" + "="*70)
print("SECTION 3: MODELS WITH COVARIATES")
print("="*70)

# Create dummy variables for categorical controls
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

# Education recoding - check what values exist
print("\n--- Education Distribution ---")
print(df['EDUC'].value_counts().sort_index())

# Model 4: DiD with demographic controls
print("\n--- Model 4: DiD with Demographic Controls ---")
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# Model 5: DiD with demographic controls + education
print("\n--- Model 5: DiD with Demographic + Education Controls ---")
# Create education dummies based on EDUC values
df['EDUC_HS'] = (df['EDUC'] == 6).astype(int)  # High school
df['EDUC_SOMECOLL'] = df['EDUC'].isin([7, 8, 9]).astype(int)  # Some college
df['EDUC_COLLEGE'] = df['EDUC'].isin([10, 11]).astype(int)  # College+

model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_COLLEGE',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary())

# Model 6: Full model with year fixed effects
print("\n--- Model 6: DiD with Year Fixed Effects ---")
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

model6 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_COLLEGE + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model6.summary())

# ============================================================================
# SECTION 4: STATE FIXED EFFECTS AND CLUSTERED STANDARD ERRORS
# ============================================================================
print("\n" + "="*70)
print("SECTION 4: STATE FIXED EFFECTS AND CLUSTERED SE")
print("="*70)

# Model 7: State and Year Fixed Effects
print("\n--- Model 7: DiD with State and Year Fixed Effects ---")
# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)

# Get state dummy column names
state_cols = [col for col in df.columns if col.startswith('STATE_')]

# Build formula
state_formula = ' + '.join(state_cols)
year_formula = 'YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016'
full_formula = f'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_COLLEGE + {year_formula} + {state_formula}'

model7 = smf.wls(full_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Print only main coefficients
print("Main coefficients (state FE not shown):")
main_vars = ['Intercept', 'ELIGIBLE', 'ELIGIBLE_X_AFTER', 'FEMALE', 'MARRIED', 'NCHILD', 'AGE',
             'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_COLLEGE']
for var in main_vars:
    if var in model7.params.index:
        print(f"{var:20s}: {model7.params[var]:8.4f} (SE: {model7.bse[var]:.4f}, p: {model7.pvalues[var]:.4f})")

# Model 8: Clustered standard errors at state level
print("\n--- Model 8: DiD with Clustered SE at State Level ---")
# Use basic model with state clustering
model8 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_COLLEGE',
                 data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model8.summary())

# ============================================================================
# SECTION 5: PARALLEL TRENDS ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("SECTION 5: PARALLEL TRENDS ANALYSIS")
print("="*70)

# FT rates by year and group
print("\n--- Full-Time Employment by Year and Treatment Status ---")
yearly_ft = df.groupby(['YEAR', 'ELIGIBLE']).agg({
    'FT': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
print(yearly_ft)

# Event study analysis
print("\n--- Event Study Specification ---")
# Create year-treatment interactions for pre-treatment years
df['ELIG_X_2009'] = df['ELIGIBLE'] * (df['YEAR'] == 2009).astype(int)
df['ELIG_X_2010'] = df['ELIGIBLE'] * (df['YEAR'] == 2010).astype(int)
df['ELIG_X_2011'] = df['ELIGIBLE'] * (df['YEAR'] == 2011).astype(int)
# 2008 is reference year
df['ELIG_X_2013'] = df['ELIGIBLE'] * (df['YEAR'] == 2013).astype(int)
df['ELIG_X_2014'] = df['ELIGIBLE'] * (df['YEAR'] == 2014).astype(int)
df['ELIG_X_2015'] = df['ELIGIBLE'] * (df['YEAR'] == 2015).astype(int)
df['ELIG_X_2016'] = df['ELIGIBLE'] * (df['YEAR'] == 2016).astype(int)

event_formula = 'FT ~ ELIGIBLE + ELIG_X_2009 + ELIG_X_2010 + ELIG_X_2011 + ELIG_X_2013 + ELIG_X_2014 + ELIG_X_2015 + ELIG_X_2016 + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + FEMALE + MARRIED + NCHILD + AGE'

model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Treatment x Year):")
event_vars = ['ELIG_X_2009', 'ELIG_X_2010', 'ELIG_X_2011', 'ELIG_X_2013', 'ELIG_X_2014', 'ELIG_X_2015', 'ELIG_X_2016']
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    ci_low = coef - 1.96*se
    ci_high = coef + 1.96*se
    print(f"{var}: {coef:8.4f} (SE: {se:.4f}, p: {pval:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}])")

# Test for pre-trends (joint F-test)
print("\n--- Testing Pre-Treatment Parallel Trends ---")
# Restrict to pre-period
df_pre = df[df['AFTER'] == 0].copy()

# Model with year-treatment interactions in pre-period
pre_formula = 'FT ~ ELIGIBLE + ELIG_X_2009 + ELIG_X_2010 + ELIG_X_2011 + YEAR_2009 + YEAR_2010 + YEAR_2011 + FEMALE + MARRIED + NCHILD + AGE'
model_pre = smf.wls(pre_formula, data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')

# Check coefficients for pre-trend interactions
print("Pre-treatment interaction coefficients:")
pre_event_vars = ['ELIG_X_2009', 'ELIG_X_2010', 'ELIG_X_2011']
for var in pre_event_vars:
    coef = model_pre.params[var]
    se = model_pre.bse[var]
    pval = model_pre.pvalues[var]
    print(f"{var}: {coef:8.4f} (SE: {se:.4f}, p: {pval:.4f})")

# ============================================================================
# SECTION 6: HETEROGENEITY ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("SECTION 6: HETEROGENEITY ANALYSIS")
print("="*70)

# By sex
print("\n--- DiD Effect by Sex ---")
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    coef = model_sub.params['ELIGIBLE_X_AFTER']
    se = model_sub.bse['ELIGIBLE_X_AFTER']
    pval = model_sub.pvalues['ELIGIBLE_X_AFTER']
    n = len(df_sub)
    print(f"{sex_label}: DiD = {coef:.4f} (SE: {se:.4f}, p: {pval:.4f}, N = {n})")

# By education
print("\n--- DiD Effect by Education Level ---")
educ_groups = [
    ((df['EDUC'] < 6), 'Less than HS'),
    ((df['EDUC'] == 6), 'High School'),
    ((df['EDUC'].isin([7,8,9])), 'Some College'),
    ((df['EDUC'].isin([10,11])), 'College+')
]
for mask, label in educ_groups:
    df_sub = df[mask]
    if len(df_sub) > 100:
        model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                            data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
        coef = model_sub.params['ELIGIBLE_X_AFTER']
        se = model_sub.bse['ELIGIBLE_X_AFTER']
        pval = model_sub.pvalues['ELIGIBLE_X_AFTER']
        n = len(df_sub)
        print(f"{label:15s}: DiD = {coef:.4f} (SE: {se:.4f}, p: {pval:.4f}, N = {n})")

# By marital status
print("\n--- DiD Effect by Marital Status ---")
for married_val, married_label in [(1, 'Married'), (0, 'Not Married')]:
    df_sub = df[df['MARRIED'] == married_val]
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    coef = model_sub.params['ELIGIBLE_X_AFTER']
    se = model_sub.bse['ELIGIBLE_X_AFTER']
    pval = model_sub.pvalues['ELIGIBLE_X_AFTER']
    n = len(df_sub)
    print(f"{married_label:15s}: DiD = {coef:.4f} (SE: {se:.4f}, p: {pval:.4f}, N = {n})")

# ============================================================================
# SECTION 7: ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*70)
print("SECTION 7: ROBUSTNESS CHECKS")
print("="*70)

# Linear probability model vs logit
print("\n--- Logit Model for Comparison ---")
try:
    model_logit = smf.logit('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit(disp=0)
    # Calculate marginal effect at means
    print("Logit coefficients:")
    print(f"  ELIGIBLE_X_AFTER: {model_logit.params['ELIGIBLE_X_AFTER']:.4f}")
    print(f"  p-value: {model_logit.pvalues['ELIGIBLE_X_AFTER']:.4f}")
    # Average marginal effect approximation
    avg_pred = model_logit.predict(df).mean()
    marginal_effect = model_logit.params['ELIGIBLE_X_AFTER'] * avg_pred * (1 - avg_pred)
    print(f"  Approximate marginal effect: {marginal_effect:.4f}")
except:
    print("Logit model failed to converge")

# Bandwidth sensitivity
print("\n--- Bandwidth Sensitivity (Ages around 31 cutoff) ---")
for bw in [3, 4, 5]:
    # Keep only ages within bandwidth of cutoff
    df_bw = df[(df['AGE_IN_JUNE_2012'] >= 31 - bw) & (df['AGE_IN_JUNE_2012'] <= 31 + bw - 0.01)]
    if len(df_bw) > 100:
        model_bw = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                          data=df_bw, weights=df_bw['PERWT']).fit(cov_type='HC1')
        coef = model_bw.params['ELIGIBLE_X_AFTER']
        se = model_bw.bse['ELIGIBLE_X_AFTER']
        pval = model_bw.pvalues['ELIGIBLE_X_AFTER']
        n = len(df_bw)
        print(f"Bandwidth {bw} years: DiD = {coef:.4f} (SE: {se:.4f}, p: {pval:.4f}, N = {n})")

# Placebo test using pre-period only
print("\n--- Placebo Test (Pre-period only, fake treatment at 2010) ---")
df_pre = df[df['AFTER'] == 0].copy()
df_pre['FAKE_AFTER'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['FAKE_INTERACTION'] = df_pre['ELIGIBLE'] * df_pre['FAKE_AFTER']
model_placebo = smf.wls('FT ~ ELIGIBLE + FAKE_AFTER + FAKE_INTERACTION',
                        data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD: {model_placebo.params['FAKE_INTERACTION']:.4f}")
print(f"SE: {model_placebo.bse['FAKE_INTERACTION']:.4f}")
print(f"p-value: {model_placebo.pvalues['FAKE_INTERACTION']:.4f}")

# ============================================================================
# SECTION 8: SUMMARY RESULTS TABLE
# ============================================================================
print("\n" + "="*70)
print("SECTION 8: SUMMARY OF MAIN RESULTS")
print("="*70)

# Create summary table
print("\n" + "-"*90)
print(f"{'Model':<40} {'DiD Est':>10} {'SE':>10} {'p-value':>10} {'N':>10}")
print("-"*90)

results = [
    ("1. Basic DiD (unweighted)", model1.params['ELIGIBLE_X_AFTER'], model1.bse['ELIGIBLE_X_AFTER'], model1.pvalues['ELIGIBLE_X_AFTER'], model1.nobs),
    ("2. Basic DiD (weighted)", model2.params['ELIGIBLE_X_AFTER'], model2.bse['ELIGIBLE_X_AFTER'], model2.pvalues['ELIGIBLE_X_AFTER'], model2.nobs),
    ("3. Weighted + Robust SE", model3.params['ELIGIBLE_X_AFTER'], model3.bse['ELIGIBLE_X_AFTER'], model3.pvalues['ELIGIBLE_X_AFTER'], model3.nobs),
    ("4. + Demographics", model4.params['ELIGIBLE_X_AFTER'], model4.bse['ELIGIBLE_X_AFTER'], model4.pvalues['ELIGIBLE_X_AFTER'], model4.nobs),
    ("5. + Education", model5.params['ELIGIBLE_X_AFTER'], model5.bse['ELIGIBLE_X_AFTER'], model5.pvalues['ELIGIBLE_X_AFTER'], model5.nobs),
    ("6. + Year FE", model6.params['ELIGIBLE_X_AFTER'], model6.bse['ELIGIBLE_X_AFTER'], model6.pvalues['ELIGIBLE_X_AFTER'], model6.nobs),
    ("7. + State FE", model7.params['ELIGIBLE_X_AFTER'], model7.bse['ELIGIBLE_X_AFTER'], model7.pvalues['ELIGIBLE_X_AFTER'], model7.nobs),
    ("8. Clustered SE (State)", model8.params['ELIGIBLE_X_AFTER'], model8.bse['ELIGIBLE_X_AFTER'], model8.pvalues['ELIGIBLE_X_AFTER'], model8.nobs),
]

for name, coef, se, pval, n in results:
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    print(f"{name:<40} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {int(n):>10}")

print("-"*90)
print("Note: *** p<0.01, ** p<0.05, * p<0.1")

# ============================================================================
# SECTION 9: PREFERRED ESTIMATE
# ============================================================================
print("\n" + "="*70)
print("SECTION 9: PREFERRED ESTIMATE")
print("="*70)

# Preferred model: Model 5 with demographics and education, robust SE
preferred = model5
print(f"\nPreferred Specification: Model 5 (Weighted DiD with Demographics and Education)")
print(f"  DiD Estimate (ELIGIBLE x AFTER): {preferred.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"  Standard Error (Robust): {preferred.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"  95% Confidence Interval: [{preferred.params['ELIGIBLE_X_AFTER'] - 1.96*preferred.bse['ELIGIBLE_X_AFTER']:.4f}, {preferred.params['ELIGIBLE_X_AFTER'] + 1.96*preferred.bse['ELIGIBLE_X_AFTER']:.4f}]")
print(f"  t-statistic: {preferred.tvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"  p-value: {preferred.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"  Sample Size: {int(preferred.nobs)}")
print(f"  R-squared: {preferred.rsquared:.4f}")

print("\nInterpretation:")
print(f"  DACA eligibility is associated with a {preferred.params['ELIGIBLE_X_AFTER']*100:.2f} percentage point")
print(f"  increase in the probability of full-time employment.")

# Save key results for report
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
