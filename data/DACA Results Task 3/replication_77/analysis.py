"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among Mexican-born Hispanic individuals

Design: Difference-in-Differences
- Treatment: ELIGIBLE = 1 (ages 26-30 in June 2012)
- Control: ELIGIBLE = 0 (ages 31-35 in June 2012)
- Pre-period: 2008-2011 (AFTER = 0)
- Post-period: 2013-2016 (AFTER = 1)
- Outcome: FT (full-time employment, 35+ hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# Load data
print("\n[1] LOADING DATA...")
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_77\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"Total observations: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# Basic data structure
print("\n[2] DATA STRUCTURE AND KEY VARIABLES")
print("-"*50)

# Check key variables
print("\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

print("\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())

print("\nAFTER distribution:")
print(df['AFTER'].value_counts())

print("\nFT (Full-time employment) distribution:")
print(df['FT'].value_counts())

# Check for missing values in key variables
print("\nMissing values in key variables:")
for var in ['YEAR', 'ELIGIBLE', 'AFTER', 'FT', 'PERWT']:
    missing = df[var].isna().sum()
    print(f"  {var}: {missing}")

print("\n[3] SAMPLE CHARACTERISTICS")
print("-"*50)

# Sample size by treatment group and period
print("\nSample sizes by ELIGIBLE x AFTER:")
cross_tab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
cross_tab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
cross_tab.columns = ['Pre-DACA', 'Post-DACA', 'Total']
print(cross_tab)

# Weighted sample sizes
print("\nWeighted sample sizes (using PERWT):")
for elig in [0, 1]:
    for after in [0, 1]:
        mask = (df['ELIGIBLE'] == elig) & (df['AFTER'] == after)
        weighted_n = df.loc[mask, 'PERWT'].sum()
        print(f"  ELIGIBLE={elig}, AFTER={after}: {weighted_n:,.0f}")

print("\n[4] SUMMARY STATISTICS")
print("-"*50)

# Summary stats for key continuous variables
cont_vars = ['AGE', 'FAMSIZE', 'NCHILD', 'UHRSWORK', 'INCTOT']
available_vars = [v for v in cont_vars if v in df.columns]

print("\nContinuous variables (unweighted):")
print(df[available_vars].describe())

# FT rates by group and period
print("\n[5] FULL-TIME EMPLOYMENT RATES (Unweighted)")
print("-"*50)

ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
ft_rates.index = pd.MultiIndex.from_tuples(
    [(('Control', 'Treatment')[e], ('Pre', 'Post')[a]) for e, a in ft_rates.index],
    names=['Group', 'Period']
)
print(ft_rates)

# Calculate simple DiD
print("\n[6] DIFFERENCE-IN-DIFFERENCES CALCULATION (Unweighted)")
print("-"*50)

# Get means
ft_treat_pre = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]['FT'].mean()
ft_treat_post = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]['FT'].mean()
ft_control_pre = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]['FT'].mean()
ft_control_post = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]['FT'].mean()

print(f"Treatment group (ages 26-30):")
print(f"  Pre-DACA FT rate:  {ft_treat_pre:.4f} ({ft_treat_pre*100:.2f}%)")
print(f"  Post-DACA FT rate: {ft_treat_post:.4f} ({ft_treat_post*100:.2f}%)")
print(f"  Change:            {ft_treat_post - ft_treat_pre:.4f} ({(ft_treat_post - ft_treat_pre)*100:.2f} pp)")

print(f"\nControl group (ages 31-35):")
print(f"  Pre-DACA FT rate:  {ft_control_pre:.4f} ({ft_control_pre*100:.2f}%)")
print(f"  Post-DACA FT rate: {ft_control_post:.4f} ({ft_control_post*100:.2f}%)")
print(f"  Change:            {ft_control_post - ft_control_pre:.4f} ({(ft_control_post - ft_control_pre)*100:.2f} pp)")

did_estimate = (ft_treat_post - ft_treat_pre) - (ft_control_post - ft_control_pre)
print(f"\nDifference-in-Differences estimate: {did_estimate:.4f} ({did_estimate*100:.2f} percentage points)")

print("\n[7] REGRESSION ANALYSIS")
print("-"*50)

# Create interaction term
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD without controls (OLS)
print("\n--- Model 1: Basic DiD (OLS, no weights) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD coefficient: {model1.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Std error: {model1.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model1.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 2: DiD with survey weights (WLS)
print("\n--- Model 2: DiD with survey weights (WLS) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])
print(f"\nDiD coefficient: {model2.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Std error: {model2.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 3: DiD with demographic controls
print("\n--- Model 3: DiD with demographic controls (WLS) ---")

# Create dummy variables for categorical controls
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Marital status (MARST: 1=married spouse present, other = not married with spouse present)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Check education variable
print("\nEducation distribution (EDUC_RECODE):")
print(df['EDUC_RECODE'].value_counts())

# Create education dummies
df['HS_DEGREE_DUM'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['SOME_COLLEGE'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['TWO_YEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['BA_PLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Model with controls
formula3 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + HS_DEGREE_DUM + SOME_COLLEGE + TWO_YEAR + BA_PLUS'
model3 = smf.wls(formula3, data=df, weights=df['PERWT']).fit()
print("\nRegression results:")
print(model3.summary().tables[1])
print(f"\nDiD coefficient: {model3.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Std error: {model3.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with year fixed effects (WLS) ---")

# Create year dummies (excluding one year as reference)
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True)
df = pd.concat([df, year_dummies], axis=1)

year_cols = [c for c in df.columns if c.startswith('YEAR_')]
formula4 = f'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + {" + ".join(year_cols)} + FEMALE + MARRIED + NCHILD + HS_DEGREE_DUM + SOME_COLLEGE + TWO_YEAR + BA_PLUS'
model4 = smf.wls(formula4, data=df, weights=df['PERWT']).fit()
print("\nRegression results (showing key coefficients):")
print(f"ELIGIBLE:         {model4.params['ELIGIBLE']:.4f} (SE: {model4.bse['ELIGIBLE']:.4f})")
print(f"ELIGIBLE_X_AFTER: {model4.params['ELIGIBLE_X_AFTER']:.4f} (SE: {model4.bse['ELIGIBLE_X_AFTER']:.4f})")
print(f"\nDiD coefficient: {model4.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model4.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 5: DiD with state fixed effects
print("\n--- Model 5: DiD with state fixed effects (WLS) ---")

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)

state_cols = [c for c in df.columns if c.startswith('STATE_')]
formula5 = f'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + {" + ".join(year_cols)} + {" + ".join(state_cols)} + FEMALE + MARRIED + NCHILD + HS_DEGREE_DUM + SOME_COLLEGE + TWO_YEAR + BA_PLUS'
model5 = smf.wls(formula5, data=df, weights=df['PERWT']).fit()
print(f"\nDiD coefficient: {model5.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Std error: {model5.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {int(model5.nobs):,}")

print("\n[8] ROBUST STANDARD ERRORS")
print("-"*50)

# Re-run preferred model with heteroskedasticity-robust standard errors
print("\n--- Model 5 with HC1 robust standard errors ---")
model5_robust = smf.wls(formula5, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"\nDiD coefficient: {model5_robust.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Robust Std error: {model5_robust.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model5_robust.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model5_robust.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model5_robust.pvalues['ELIGIBLE_X_AFTER']:.4f}")

print("\n[9] YEAR-BY-YEAR EFFECTS (Event Study)")
print("-"*50)

# Create year-specific treatment effects
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions (2008 is reference year)
df['ELIG_2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIG_2010'] = df['ELIGIBLE'] * df['YEAR_2010']
df['ELIG_2011'] = df['ELIGIBLE'] * df['YEAR_2011']
df['ELIG_2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIG_2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIG_2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIG_2016'] = df['ELIGIBLE'] * df['YEAR_2016']

event_formula = f'FT ~ ELIGIBLE + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIG_2009 + ELIG_2010 + ELIG_2011 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016 + {" + ".join(state_cols)} + FEMALE + MARRIED + NCHILD + HS_DEGREE_DUM + SOME_COLLEGE + TWO_YEAR + BA_PLUS'

model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nYear-specific treatment effects (relative to 2008):")
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    var = f'ELIG_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    ci_low, ci_high = model_event.conf_int().loc[var]
    sig = "*" if pval < 0.05 else ("+" if pval < 0.10 else "")
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}) [{ci_low:.4f}, {ci_high:.4f}] p={pval:.3f} {sig}")

print("\n[10] SUMMARY OF RESULTS")
print("="*80)

# Summary table
print("\n                             Model Comparison")
print("-"*80)
print(f"{'Model':<45} {'DiD Coef':>10} {'SE':>10} {'p-value':>10}")
print("-"*80)
print(f"{'1. Basic DiD (no weights)':<45} {model1.params['ELIGIBLE_X_AFTER']:>10.4f} {model1.bse['ELIGIBLE_X_AFTER']:>10.4f} {model1.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'2. DiD with survey weights':<45} {model2.params['ELIGIBLE_X_AFTER']:>10.4f} {model2.bse['ELIGIBLE_X_AFTER']:>10.4f} {model2.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'3. + Demographic controls':<45} {model3.params['ELIGIBLE_X_AFTER']:>10.4f} {model3.bse['ELIGIBLE_X_AFTER']:>10.4f} {model3.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'4. + Year fixed effects':<45} {model4.params['ELIGIBLE_X_AFTER']:>10.4f} {model4.bse['ELIGIBLE_X_AFTER']:>10.4f} {model4.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'5. + State fixed effects':<45} {model5.params['ELIGIBLE_X_AFTER']:>10.4f} {model5.bse['ELIGIBLE_X_AFTER']:>10.4f} {model5.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'5b. + Robust SE':<45} {model5_robust.params['ELIGIBLE_X_AFTER']:>10.4f} {model5_robust.bse['ELIGIBLE_X_AFTER']:>10.4f} {model5_robust.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print("-"*80)

# Preferred estimate
print("\n*** PREFERRED ESTIMATE (Model 5 with robust SEs) ***")
print(f"DiD Estimate: {model5_robust.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model5_robust.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model5_robust.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model5_robust.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"Sample Size: {int(model5_robust.nobs):,}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save results for report
results = {
    'model1': {'coef': model1.params['ELIGIBLE_X_AFTER'], 'se': model1.bse['ELIGIBLE_X_AFTER'], 'pval': model1.pvalues['ELIGIBLE_X_AFTER']},
    'model2': {'coef': model2.params['ELIGIBLE_X_AFTER'], 'se': model2.bse['ELIGIBLE_X_AFTER'], 'pval': model2.pvalues['ELIGIBLE_X_AFTER']},
    'model3': {'coef': model3.params['ELIGIBLE_X_AFTER'], 'se': model3.bse['ELIGIBLE_X_AFTER'], 'pval': model3.pvalues['ELIGIBLE_X_AFTER']},
    'model4': {'coef': model4.params['ELIGIBLE_X_AFTER'], 'se': model4.bse['ELIGIBLE_X_AFTER'], 'pval': model4.pvalues['ELIGIBLE_X_AFTER']},
    'model5': {'coef': model5.params['ELIGIBLE_X_AFTER'], 'se': model5.bse['ELIGIBLE_X_AFTER'], 'pval': model5.pvalues['ELIGIBLE_X_AFTER']},
    'model5_robust': {'coef': model5_robust.params['ELIGIBLE_X_AFTER'], 'se': model5_robust.bse['ELIGIBLE_X_AFTER'], 'pval': model5_robust.pvalues['ELIGIBLE_X_AFTER']},
    'n': int(model5_robust.nobs),
    'ft_treat_pre': ft_treat_pre,
    'ft_treat_post': ft_treat_post,
    'ft_control_pre': ft_control_pre,
    'ft_control_post': ft_control_post
}

# Export key statistics for the report
print("\n\nExporting statistics for report...")
print(f"Sample size: {len(df):,}")
print(f"Treatment group (pre): {len(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]):,}")
print(f"Treatment group (post): {len(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]):,}")
print(f"Control group (pre): {len(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]):,}")
print(f"Control group (post): {len(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]):,}")
