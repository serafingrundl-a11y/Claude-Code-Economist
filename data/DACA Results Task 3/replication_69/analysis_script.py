#!/usr/bin/env python3
"""
DACA Replication Study - Analysis Script
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

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# Load the labelled version for analysis
df = pd.read_csv('data/prepared_data_labelled_version.csv')

print(f"\nTotal observations: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. DATA SUMMARY AND VALIDATION
# =============================================================================
print("\n" + "="*80)
print("DATA VALIDATION")
print("="*80)

# Check key variables
print(f"\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())

print(f"\nAFTER distribution:")
print(df['AFTER'].value_counts())

print(f"\nFT distribution:")
print(df['FT'].value_counts())

print(f"\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

# Cross-tabulation of treatment and time
print("\n" + "-"*40)
print("Treatment x Time Crosstab (Unweighted):")
print("-"*40)
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.columns = ['Pre (2008-2011)', 'Post (2013-2016)', 'Total']
crosstab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
print(crosstab)

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Function to compute weighted mean and SE
def weighted_stats(df, var, weight='PERWT'):
    """Calculate weighted mean and standard error"""
    w = df[weight]
    x = df[var]
    wmean = np.average(x, weights=w)
    # Weighted variance
    wvar = np.average((x - wmean)**2, weights=w)
    n = len(x)
    wse = np.sqrt(wvar / n)
    return wmean, wse, n

# Summary statistics by group
print("\n" + "-"*40)
print("Full-Time Employment Rate by Group (Weighted):")
print("-"*40)

groups = [
    ('Control', 'Pre', (df['ELIGIBLE']==0) & (df['AFTER']==0)),
    ('Control', 'Post', (df['ELIGIBLE']==0) & (df['AFTER']==1)),
    ('Treatment', 'Pre', (df['ELIGIBLE']==1) & (df['AFTER']==0)),
    ('Treatment', 'Post', (df['ELIGIBLE']==1) & (df['AFTER']==1))
]

ft_rates = {}
for name, period, mask in groups:
    mean, se, n = weighted_stats(df[mask], 'FT')
    ft_rates[(name, period)] = {'mean': mean, 'se': se, 'n': n}
    print(f"{name:10s} {period:5s}: {mean:.4f} (SE: {se:.4f}, N: {n:,})")

# Simple DiD calculation
did_simple = (ft_rates[('Treatment', 'Post')]['mean'] - ft_rates[('Treatment', 'Pre')]['mean']) - \
             (ft_rates[('Control', 'Post')]['mean'] - ft_rates[('Control', 'Pre')]['mean'])
print(f"\nSimple DiD Estimate (weighted means): {did_simple:.4f}")

# =============================================================================
# 4. COVARIATE BALANCE TABLE
# =============================================================================
print("\n" + "="*80)
print("COVARIATE BALANCE (Pre-Period)")
print("="*80)

# Select pre-period data
pre_df = df[df['AFTER'] == 0].copy()

# Define covariates to check
covariates = ['AGE', 'SEX', 'MARST', 'FAMSIZE', 'NCHILD']

print("\nVariable          Treatment    Control    Diff")
print("-"*50)
for var in covariates:
    if var in pre_df.columns:
        # Convert categorical SEX (1=Male, 2=Female) to binary for comparison
        if var == 'SEX':
            pre_df['SEX_FEMALE'] = (pre_df['SEX'] == 2).astype(int) if pre_df['SEX'].dtype == 'int64' else (pre_df['SEX'] == 'Female').astype(int)
            var_use = 'SEX_FEMALE'
        elif var == 'MARST':
            # Check if married
            pre_df['MARRIED'] = pre_df['MARST'].apply(lambda x: 1 if 'Married' in str(x) or x in [1,2] else 0)
            var_use = 'MARRIED'
        else:
            var_use = var

        t_mean, _, _ = weighted_stats(pre_df[pre_df['ELIGIBLE']==1], var_use)
        c_mean, _, _ = weighted_stats(pre_df[pre_df['ELIGIBLE']==0], var_use)
        diff = t_mean - c_mean
        print(f"{var:15s}  {t_mean:10.3f}  {c_mean:10.3f}  {diff:8.3f}")

# =============================================================================
# 5. MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("MAIN DiD REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD (no controls, no weights)
print("\n" + "-"*40)
print("Model 1: Basic DiD (OLS, no weights)")
print("-"*40)

df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary())

# Model 2: DiD with survey weights
print("\n" + "-"*40)
print("Model 2: DiD with Survey Weights (WLS)")
print("-"*40)

model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                  data=df, weights=df['PERWT']).fit()
print(model2.summary())

# Model 3: DiD with clustered standard errors at state level
print("\n" + "-"*40)
print("Model 3: DiD with Clustered SEs (State-level)")
print("-"*40)

model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model3.summary())

# Model 4: DiD with weights and clustered SEs
print("\n" + "-"*40)
print("Model 4: DiD with Weights and Clustered SEs (PREFERRED)")
print("-"*40)

model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                  data=df, weights=df['PERWT']).fit(
                  cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model4.summary())

# =============================================================================
# 6. DiD WITH COVARIATES
# =============================================================================
print("\n" + "="*80)
print("DiD WITH DEMOGRAPHIC CONTROLS")
print("="*80)

# Prepare covariates
# SEX: 1=Male, 2=Female -> Female dummy
df['FEMALE'] = (df['SEX'] == 2).astype(int) if df['SEX'].dtype == 'int64' else (df['SEX'] == 'Female').astype(int)

# MARST: Create married indicator
df['MARRIED'] = df['MARST'].apply(lambda x: 1 if 'Married' in str(x) or x in [1,2] else 0)

# Education: Create numeric variable from EDUC_RECODE
educ_map = {
    'Less than High School': 1,
    'High School Degree': 2,
    'Some College': 3,
    'Two-Year Degree': 4,
    'BA+': 5
}
df['EDUC_NUM'] = df['EDUC_RECODE'].map(educ_map)
# Fill missing with most common
df['EDUC_NUM'] = df['EDUC_NUM'].fillna(df['EDUC_NUM'].mode()[0])

# Create education dummies manually with clean names
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Model 5: DiD with demographic controls
print("\n" + "-"*40)
print("Model 5: DiD with Demographic Controls")
print("-"*40)

formula5 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + AGE'
model5 = smf.wls(formula5, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model5.summary())

# Model 6: DiD with demographic and education controls
print("\n" + "-"*40)
print("Model 6: DiD with Demographic + Education Controls")
print("-"*40)

formula6 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA'
model6 = smf.wls(formula6, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model6.summary())

# =============================================================================
# 7. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Model 7: Year fixed effects using C() notation
print("\n" + "-"*40)
print("Model 7: DiD with Year Fixed Effects")
print("-"*40)

formula7 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR)'
model7 = smf.wls(formula7, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model7.summary())

# Model 8: State fixed effects
print("\n" + "-"*40)
print("Model 8: DiD with State Fixed Effects")
print("-"*40)

formula8 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)'
model8 = smf.wls(formula8, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"\nModel 8 DiD Coefficient (ELIGIBLE_AFTER): {model8.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model8.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model8.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model8.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model8.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 9: Full model with year and state FEs
print("\n" + "-"*40)
print("Model 9: Full Model with Year and State FEs + Demographics")
print("-"*40)

formula9 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + C(YEAR) + C(STATEFIP)'
model9 = smf.wls(formula9, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"\nModel 9 DiD Coefficient (ELIGIBLE_AFTER): {model9.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model9.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model9.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model9.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model9.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# 8. EVENT STUDY / PARALLEL TRENDS
# =============================================================================
print("\n" + "="*80)
print("EVENT STUDY - PARALLEL TRENDS CHECK")
print("="*80)

# Create year-specific treatment effects
for year in df['YEAR'].unique():
    df[f'ELIG_Y{year}'] = (df['ELIGIBLE'] * (df['YEAR'] == year)).astype(int)

# Exclude one pre-treatment year (2011) as reference
event_years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_terms = ' + '.join([f'ELIG_Y{y}' for y in event_years])
formula_event = f'FT ~ ELIGIBLE + C(YEAR) + {event_terms}'

model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("\nEvent Study Coefficients (Reference: 2011):")
print("-"*50)
print(f"{'Year':<10} {'Coefficient':<12} {'SE':<10} {'p-value':<10}")
print("-"*50)
for year in event_years:
    coef = model_event.params[f'ELIG_Y{year}']
    se = model_event.bse[f'ELIG_Y{year}']
    pval = model_event.pvalues[f'ELIG_Y{year}']
    print(f"{year:<10} {coef:<12.4f} {se:<10.4f} {pval:<10.4f}")

# =============================================================================
# 9. HETEROGENEITY ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("HETEROGENEITY ANALYSIS")
print("="*80)

# By gender
print("\n" + "-"*40)
print("DiD by Gender:")
print("-"*40)

for gender, label in [(1, 'Male'), (2, 'Female')]:
    if df['SEX'].dtype == 'int64':
        mask = df['SEX'] == gender
    else:
        mask = df['SEX'] == label

    model_g = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                       data=df[mask], weights=df[mask]['PERWT']).fit(
                       cov_type='cluster', cov_kwds={'groups': df[mask]['STATEFIP']})

    print(f"\n{label}:")
    print(f"  DiD Estimate: {model_g.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  SE: {model_g.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"  95% CI: [{model_g.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_g.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
    print(f"  N: {mask.sum():,}")

# By education
print("\n" + "-"*40)
print("DiD by Education Level:")
print("-"*40)

for educ in df['EDUC_RECODE'].unique():
    mask = df['EDUC_RECODE'] == educ
    if mask.sum() > 100:  # Minimum sample size
        try:
            model_e = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                               data=df[mask], weights=df[mask]['PERWT']).fit(
                               cov_type='cluster', cov_kwds={'groups': df[mask]['STATEFIP']})

            print(f"\n{educ}:")
            print(f"  DiD Estimate: {model_e.params['ELIGIBLE_AFTER']:.4f}")
            print(f"  SE: {model_e.bse['ELIGIBLE_AFTER']:.4f}")
            print(f"  N: {mask.sum():,}")
        except:
            print(f"\n{educ}: Estimation failed (insufficient variation)")

# =============================================================================
# 10. ADDITIONAL DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("ADDITIONAL DESCRIPTIVE STATISTICS")
print("="*80)

# Yearly FT rates by treatment group
print("\n" + "-"*40)
print("FT Employment by Year and Treatment Status (Weighted):")
print("-"*40)
print(f"{'Year':<8} {'Treatment':<12} {'Control':<12} {'Difference':<12}")
print("-"*40)

yearly_rates = {}
for year in sorted(df['YEAR'].unique()):
    year_df = df[df['YEAR'] == year]
    t_rate, _, n_t = weighted_stats(year_df[year_df['ELIGIBLE']==1], 'FT')
    c_rate, _, n_c = weighted_stats(year_df[year_df['ELIGIBLE']==0], 'FT')
    yearly_rates[year] = {'treatment': t_rate, 'control': c_rate}
    print(f"{year:<8} {t_rate:<12.4f} {c_rate:<12.4f} {t_rate-c_rate:<12.4f}")

# Calculate pre-treatment and post-treatment averages
pre_years = [2008, 2009, 2010, 2011]
post_years = [2013, 2014, 2015, 2016]

pre_t = np.mean([yearly_rates[y]['treatment'] for y in pre_years])
pre_c = np.mean([yearly_rates[y]['control'] for y in pre_years])
post_t = np.mean([yearly_rates[y]['treatment'] for y in post_years])
post_c = np.mean([yearly_rates[y]['control'] for y in post_years])

print("-"*40)
print(f"{'Pre-avg':<8} {pre_t:<12.4f} {pre_c:<12.4f} {pre_t-pre_c:<12.4f}")
print(f"{'Post-avg':<8} {post_t:<12.4f} {post_c:<12.4f} {post_t-post_c:<12.4f}")
print("-"*40)
print(f"Change:   {post_t-pre_t:<12.4f} {post_c-pre_c:<12.4f}")
print(f"DiD:      {(post_t-pre_t)-(post_c-pre_c):.4f}")

# =============================================================================
# 11. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "="*80)
print("SUMMARY OF MAIN RESULTS")
print("="*80)

print("\n" + "-"*70)
print(f"{'Model':<35} {'DiD Est.':<12} {'SE':<10} {'p-value':<10}")
print("-"*70)

models_summary = [
    ('1. Basic OLS', model1),
    ('2. Weighted (WLS)', model2),
    ('3. Clustered SEs', model3),
    ('4. Weighted + Clustered (PREFERRED)', model4),
    ('5. With Demographics', model5),
    ('6. With Demographics + Education', model6),
    ('7. Year Fixed Effects', model7),
]

for name, model in models_summary:
    coef = model.params['ELIGIBLE_AFTER']
    se = model.bse['ELIGIBLE_AFTER']
    pval = model.pvalues['ELIGIBLE_AFTER']
    print(f"{name:<35} {coef:<12.4f} {se:<10.4f} {pval:<10.4f}")

# =============================================================================
# 12. EXPORT RESULTS
# =============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Create results dictionary
results = {
    'preferred_estimate': model4.params['ELIGIBLE_AFTER'],
    'preferred_se': model4.bse['ELIGIBLE_AFTER'],
    'preferred_ci_lower': model4.conf_int().loc['ELIGIBLE_AFTER', 0],
    'preferred_ci_upper': model4.conf_int().loc['ELIGIBLE_AFTER', 1],
    'preferred_pvalue': model4.pvalues['ELIGIBLE_AFTER'],
    'sample_size': len(df),
    'n_treatment': (df['ELIGIBLE']==1).sum(),
    'n_control': (df['ELIGIBLE']==0).sum()
}

# Save to CSV
pd.DataFrame([results]).to_csv('analysis_results.csv', index=False)
print("\nResults saved to analysis_results.csv")

# Save full regression tables
with open('regression_tables.txt', 'w') as f:
    f.write("DACA Replication Study - Regression Results\n")
    f.write("="*80 + "\n\n")

    f.write("Model 4 (Preferred): Weighted DiD with Clustered SEs\n")
    f.write("-"*80 + "\n")
    f.write(model4.summary().as_text())
    f.write("\n\n")

    f.write("Model 6: With Demographic and Education Controls\n")
    f.write("-"*80 + "\n")
    f.write(model6.summary().as_text())
    f.write("\n\n")

    f.write("Event Study Results\n")
    f.write("-"*80 + "\n")
    f.write(model_event.summary().as_text())

print("Regression tables saved to regression_tables.txt")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print(f"\n*** PREFERRED ESTIMATE (Model 4) ***")
print(f"DiD Effect of DACA Eligibility on Full-Time Employment:")
print(f"  Point Estimate: {results['preferred_estimate']:.4f}")
print(f"  Standard Error: {results['preferred_se']:.4f}")
print(f"  95% CI: [{results['preferred_ci_lower']:.4f}, {results['preferred_ci_upper']:.4f}]")
print(f"  P-value: {results['preferred_pvalue']:.4f}")
print(f"  Sample Size: {results['sample_size']:,}")

# Save for LaTeX report
with open('results_for_latex.txt', 'w') as f:
    f.write("=== DATA FOR LATEX REPORT ===\n\n")

    # Sample sizes
    f.write("SAMPLE SIZES\n")
    f.write(f"Total N: {len(df)}\n")
    f.write(f"Treatment (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum()}\n")
    f.write(f"Control (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum()}\n")
    f.write(f"Pre-period: {(df['AFTER']==0).sum()}\n")
    f.write(f"Post-period: {(df['AFTER']==1).sum()}\n\n")

    # FT rates
    f.write("FT EMPLOYMENT RATES (weighted)\n")
    for (name, period), data in ft_rates.items():
        f.write(f"{name} {period}: {data['mean']:.4f} (N={data['n']})\n")
    f.write(f"\nSimple DiD: {did_simple:.4f}\n\n")

    # Model results
    f.write("MODEL RESULTS\n")
    for name, model in models_summary:
        coef = model.params['ELIGIBLE_AFTER']
        se = model.bse['ELIGIBLE_AFTER']
        ci_lo = model.conf_int().loc['ELIGIBLE_AFTER', 0]
        ci_hi = model.conf_int().loc['ELIGIBLE_AFTER', 1]
        pval = model.pvalues['ELIGIBLE_AFTER']
        f.write(f"{name}\n")
        f.write(f"  Coef: {coef:.4f}, SE: {se:.4f}, 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}], p={pval:.4f}\n")

    # Event study
    f.write("\nEVENT STUDY (ref=2011)\n")
    for year in event_years:
        coef = model_event.params[f'ELIG_Y{year}']
        se = model_event.bse[f'ELIG_Y{year}']
        f.write(f"{year}: {coef:.4f} (SE: {se:.4f})\n")

    # Yearly rates
    f.write("\nYEARLY FT RATES\n")
    for year, rates in sorted(yearly_rates.items()):
        f.write(f"{year}: Treatment={rates['treatment']:.4f}, Control={rates['control']:.4f}\n")

print("\nData for LaTeX saved to results_for_latex.txt")
