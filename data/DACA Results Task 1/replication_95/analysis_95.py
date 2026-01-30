"""
DACA Replication Study - Analysis Script 95
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born non-citizens in the US
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

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS 95")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: SAMPLE RESTRICTIONS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: APPLYING SAMPLE RESTRICTIONS")
print("=" * 80)

# Restriction 1: Hispanic-Mexican ethnicity (HISPAN == 1)
print(f"\nHISPAN value counts (before filter):")
print(df['HISPAN'].value_counts().head(10))
df_filtered = df[df['HISPAN'] == 1].copy()
print(f"After restricting to Hispanic-Mexican (HISPAN=1): {len(df_filtered):,}")

# Restriction 2: Born in Mexico (BPL == 200)
print(f"\nBPL value counts for Hispanic-Mexican (top 10):")
print(df_filtered['BPL'].value_counts().head(10))
df_filtered = df_filtered[df_filtered['BPL'] == 200].copy()
print(f"After restricting to Mexican-born (BPL=200): {len(df_filtered):,}")

# Restriction 3: Non-citizens (CITIZEN == 3)
print(f"\nCITIZEN value counts:")
print(df_filtered['CITIZEN'].value_counts())
df_filtered = df_filtered[df_filtered['CITIZEN'] == 3].copy()
print(f"After restricting to non-citizens (CITIZEN=3): {len(df_filtered):,}")

# Restriction 4: Exclude 2012 (transition year)
df_filtered = df_filtered[df_filtered['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_filtered):,}")

# Restriction 5: Working age population (16-64)
df_filtered = df_filtered[(df_filtered['AGE'] >= 16) & (df_filtered['AGE'] <= 64)].copy()
print(f"After restricting to ages 16-64: {len(df_filtered):,}")

# =============================================================================
# STEP 3: CONSTRUCT VARIABLES
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: CONSTRUCTING VARIABLES")
print("=" * 80)

# Age at immigration
df_filtered['age_at_immig'] = df_filtered['YRIMMIG'] - df_filtered['BIRTHYR']

# Check YRIMMIG distribution
print(f"\nYRIMMIG summary statistics:")
print(df_filtered['YRIMMIG'].describe())
print(f"\nYRIMMIG == 0 count: {(df_filtered['YRIMMIG'] == 0).sum():,}")

# DACA Eligibility Criteria:
# 1. Arrived before 16th birthday: age_at_immig < 16
# 2. Under 31 as of June 15, 2012: BIRTHYR >= 1982 (conservative)
# 3. In US since June 15, 2007: YRIMMIG <= 2007
# 4. Valid immigration year: YRIMMIG > 0

df_filtered['arrived_before_16'] = (df_filtered['age_at_immig'] < 16) & (df_filtered['age_at_immig'] >= 0)
df_filtered['under_31_in_2012'] = df_filtered['BIRTHYR'] >= 1982
df_filtered['in_us_since_2007'] = df_filtered['YRIMMIG'] <= 2007
df_filtered['valid_yrimmig'] = df_filtered['YRIMMIG'] > 0

# DACA eligible if all criteria met
df_filtered['daca_eligible'] = (
    df_filtered['arrived_before_16'] &
    df_filtered['under_31_in_2012'] &
    df_filtered['in_us_since_2007'] &
    df_filtered['valid_yrimmig']
).astype(int)

print(f"\nDACA eligibility criteria breakdown:")
print(f"  Arrived before age 16: {df_filtered['arrived_before_16'].sum():,}")
print(f"  Under 31 in 2012 (born >= 1982): {df_filtered['under_31_in_2012'].sum():,}")
print(f"  In US since 2007 (YRIMMIG <= 2007): {df_filtered['in_us_since_2007'].sum():,}")
print(f"  Valid YRIMMIG (> 0): {df_filtered['valid_yrimmig'].sum():,}")
print(f"  DACA eligible (all criteria): {df_filtered['daca_eligible'].sum():,}")

# Post-treatment indicator (2013 onwards)
df_filtered['post'] = (df_filtered['YEAR'] >= 2013).astype(int)

# Interaction term
df_filtered['eligible_post'] = df_filtered['daca_eligible'] * df_filtered['post']

# Outcome: Full-time employment (usually work 35+ hours per week)
df_filtered['fulltime'] = (df_filtered['UHRSWORK'] >= 35).astype(int)

# Alternative outcome: Currently employed
df_filtered['employed'] = (df_filtered['EMPSTAT'] == 1).astype(int)

print(f"\nOutcome variable distributions:")
print(f"Full-time (UHRSWORK >= 35): {df_filtered['fulltime'].mean():.4f}")
print(f"Employed (EMPSTAT == 1): {df_filtered['employed'].mean():.4f}")

# =============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: DESCRIPTIVE STATISTICS")
print("=" * 80)

# Overall sample characteristics
print("\n--- Overall Sample Characteristics ---")
print(f"Total observations: {len(df_filtered):,}")
print(f"DACA eligible: {df_filtered['daca_eligible'].sum():,} ({df_filtered['daca_eligible'].mean()*100:.1f}%)")
print(f"Control group: {(1-df_filtered['daca_eligible']).sum():,} ({(1-df_filtered['daca_eligible']).mean()*100:.1f}%)")
print(f"Pre-treatment (2006-2011): {(df_filtered['post']==0).sum():,}")
print(f"Post-treatment (2013-2016): {(df_filtered['post']==1).sum():,}")

# By group and period
print("\n--- Full-time Employment Rates by Group and Period ---")
grouped = df_filtered.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
print(grouped)

# Weighted means
print("\n--- Weighted Full-time Employment Rates ---")
for eligible in [0, 1]:
    for post_period in [0, 1]:
        subset = df_filtered[(df_filtered['daca_eligible'] == eligible) & (df_filtered['post'] == post_period)]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        group_name = "Eligible" if eligible else "Control"
        period_name = "Post" if post_period else "Pre"
        print(f"{group_name}, {period_name}: {weighted_mean:.4f} (n={len(subset):,})")

# Simple difference-in-differences calculation
print("\n--- Simple Difference-in-Differences ---")
# Unweighted
pre_control = df_filtered[(df_filtered['daca_eligible']==0) & (df_filtered['post']==0)]['fulltime'].mean()
post_control = df_filtered[(df_filtered['daca_eligible']==0) & (df_filtered['post']==1)]['fulltime'].mean()
pre_treat = df_filtered[(df_filtered['daca_eligible']==1) & (df_filtered['post']==0)]['fulltime'].mean()
post_treat = df_filtered[(df_filtered['daca_eligible']==1) & (df_filtered['post']==1)]['fulltime'].mean()

diff_control = post_control - pre_control
diff_treat = post_treat - pre_treat
did_estimate = diff_treat - diff_control

print(f"Control group change: {post_control:.4f} - {pre_control:.4f} = {diff_control:.4f}")
print(f"Treatment group change: {post_treat:.4f} - {pre_treat:.4f} = {diff_treat:.4f}")
print(f"Difference-in-differences estimate: {did_estimate:.4f}")

# =============================================================================
# STEP 5: REGRESSION ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: REGRESSION ANALYSIS")
print("=" * 80)

# Create additional control variables
df_filtered['age_sq'] = df_filtered['AGE'] ** 2
df_filtered['female'] = (df_filtered['SEX'] == 2).astype(int)
df_filtered['married'] = df_filtered['MARST'].isin([1, 2]).astype(int)
df_filtered['years_in_us'] = df_filtered['YRSUSA1']

# Education categories
df_filtered['educ_hs'] = (df_filtered['EDUC'] >= 6).astype(int)  # HS or more
df_filtered['educ_college'] = (df_filtered['EDUC'] >= 10).astype(int)  # Some college or more

# Model 1: Basic DiD (no fixed effects)
print("\n--- Model 1: Basic Difference-in-Differences ---")
model1 = smf.wls('fulltime ~ daca_eligible + post + eligible_post',
                  data=df_filtered, weights=df_filtered['PERWT']).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: With Year Fixed Effects
print("\n--- Model 2: Year Fixed Effects ---")
model2 = smf.wls('fulltime ~ daca_eligible + eligible_post + C(YEAR)',
                  data=df_filtered, weights=df_filtered['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: Year and State Fixed Effects
print("\n--- Model 3: Year and State Fixed Effects ---")
model3 = smf.wls('fulltime ~ daca_eligible + eligible_post + C(YEAR) + C(STATEFIP)',
                  data=df_filtered, weights=df_filtered['PERWT']).fit(cov_type='HC1')
# Print key coefficients only
print(f"daca_eligible: {model3.params['daca_eligible']:.6f} (SE: {model3.bse['daca_eligible']:.6f})")
print(f"eligible_post: {model3.params['eligible_post']:.6f} (SE: {model3.bse['eligible_post']:.6f})")
print(f"R-squared: {model3.rsquared:.4f}")
print(f"N: {int(model3.nobs):,}")

# Model 4: With Individual Controls
print("\n--- Model 4: Full Model with Controls ---")
model4 = smf.wls('fulltime ~ daca_eligible + eligible_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                  data=df_filtered, weights=df_filtered['PERWT']).fit(cov_type='HC1')
print(f"daca_eligible: {model4.params['daca_eligible']:.6f} (SE: {model4.bse['daca_eligible']:.6f})")
print(f"eligible_post: {model4.params['eligible_post']:.6f} (SE: {model4.bse['eligible_post']:.6f})")
print(f"AGE: {model4.params['AGE']:.6f}")
print(f"female: {model4.params['female']:.6f}")
print(f"married: {model4.params['married']:.6f}")
print(f"educ_hs: {model4.params['educ_hs']:.6f}")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {int(model4.nobs):,}")

# =============================================================================
# STEP 6: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: ROBUSTNESS CHECKS")
print("=" * 80)

# Robustness 1: Alternative age cutoff for eligibility (born >= 1983)
print("\n--- Robustness 1: Stricter Age Cutoff (born >= 1983) ---")
df_filtered['daca_eligible_strict'] = (
    df_filtered['arrived_before_16'] &
    (df_filtered['BIRTHYR'] >= 1983) &
    df_filtered['in_us_since_2007'] &
    df_filtered['valid_yrimmig']
).astype(int)
df_filtered['eligible_post_strict'] = df_filtered['daca_eligible_strict'] * df_filtered['post']

model_r1 = smf.wls('fulltime ~ daca_eligible_strict + eligible_post_strict + C(YEAR) + C(STATEFIP)',
                    data=df_filtered, weights=df_filtered['PERWT']).fit(cov_type='HC1')
print(f"eligible_post (strict): {model_r1.params['eligible_post_strict']:.6f} (SE: {model_r1.bse['eligible_post_strict']:.6f})")

# Robustness 2: Placebo test - use 2009 as fake treatment year
print("\n--- Robustness 2: Placebo Test (Fake Treatment in 2009) ---")
df_pre = df_filtered[df_filtered['YEAR'] <= 2011].copy()
df_pre['post_placebo'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['eligible_post_placebo'] = df_pre['daca_eligible'] * df_pre['post_placebo']

model_placebo = smf.wls('fulltime ~ daca_eligible + eligible_post_placebo + C(YEAR) + C(STATEFIP)',
                         data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"eligible_post (placebo): {model_placebo.params['eligible_post_placebo']:.6f} (SE: {model_placebo.bse['eligible_post_placebo']:.6f})")
print(f"p-value: {model_placebo.pvalues['eligible_post_placebo']:.4f}")

# Robustness 3: Men only
print("\n--- Robustness 3: Men Only ---")
df_men = df_filtered[df_filtered['SEX'] == 1].copy()
model_men = smf.wls('fulltime ~ daca_eligible + eligible_post + C(YEAR) + C(STATEFIP)',
                     data=df_men, weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"eligible_post (men): {model_men.params['eligible_post']:.6f} (SE: {model_men.bse['eligible_post']:.6f})")

# Robustness 4: Women only
print("\n--- Robustness 4: Women Only ---")
df_women = df_filtered[df_filtered['SEX'] == 2].copy()
model_women = smf.wls('fulltime ~ daca_eligible + eligible_post + C(YEAR) + C(STATEFIP)',
                       data=df_women, weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"eligible_post (women): {model_women.params['eligible_post']:.6f} (SE: {model_women.bse['eligible_post']:.6f})")

# Robustness 5: Alternative outcome - any employment
print("\n--- Robustness 5: Alternative Outcome (Any Employment) ---")
model_emp = smf.wls('employed ~ daca_eligible + eligible_post + C(YEAR) + C(STATEFIP)',
                     data=df_filtered, weights=df_filtered['PERWT']).fit(cov_type='HC1')
print(f"eligible_post (employment): {model_emp.params['eligible_post']:.6f} (SE: {model_emp.bse['eligible_post']:.6f})")

# =============================================================================
# STEP 7: EVENT STUDY
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: EVENT STUDY ANALYSIS")
print("=" * 80)

# Create year-specific interactions (2011 as reference)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for yr in years:
    df_filtered[f'eligible_y{yr}'] = (df_filtered['daca_eligible'] * (df_filtered['YEAR'] == yr)).astype(int)

formula_event = 'fulltime ~ daca_eligible + ' + ' + '.join([f'eligible_y{yr}' for yr in years]) + ' + C(YEAR) + C(STATEFIP)'
model_event = smf.wls(formula_event, data=df_filtered, weights=df_filtered['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print(f"{'Year':<10} {'Coefficient':<15} {'Std Error':<15} {'95% CI':<25}")
print("-" * 65)
for yr in years:
    coef = model_event.params[f'eligible_y{yr}']
    se = model_event.bse[f'eligible_y{yr}']
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    print(f"{yr:<10} {coef:<15.6f} {se:<15.6f} [{ci_low:.6f}, {ci_high:.6f}]")

# =============================================================================
# STEP 8: SUMMARY TABLES FOR REPORT
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: SUMMARY FOR REPORT")
print("=" * 80)

# Main results table
print("\n=== TABLE 1: MAIN REGRESSION RESULTS ===")
print(f"{'Model':<30} {'DiD Estimate':<15} {'Std Error':<15} {'p-value':<12} {'N':<12}")
print("=" * 84)
print(f"{'(1) Basic DiD':<30} {model1.params['eligible_post']:<15.4f} {model1.bse['eligible_post']:<15.4f} {model1.pvalues['eligible_post']:<12.4f} {int(model1.nobs):<12,}")
print(f"{'(2) Year FE':<30} {model2.params['eligible_post']:<15.4f} {model2.bse['eligible_post']:<15.4f} {model2.pvalues['eligible_post']:<12.4f} {int(model2.nobs):<12,}")
print(f"{'(3) Year + State FE':<30} {model3.params['eligible_post']:<15.4f} {model3.bse['eligible_post']:<15.4f} {model3.pvalues['eligible_post']:<12.4f} {int(model3.nobs):<12,}")
print(f"{'(4) Full Controls':<30} {model4.params['eligible_post']:<15.4f} {model4.bse['eligible_post']:<15.4f} {model4.pvalues['eligible_post']:<12.4f} {int(model4.nobs):<12,}")

print("\n=== TABLE 2: ROBUSTNESS CHECKS ===")
print(f"{'Specification':<35} {'DiD Estimate':<15} {'Std Error':<15} {'p-value':<12}")
print("=" * 77)
print(f"{'Stricter age cutoff (born>=1983)':<35} {model_r1.params['eligible_post_strict']:<15.4f} {model_r1.bse['eligible_post_strict']:<15.4f} {model_r1.pvalues['eligible_post_strict']:<12.4f}")
print(f"{'Placebo (fake 2009 treatment)':<35} {model_placebo.params['eligible_post_placebo']:<15.4f} {model_placebo.bse['eligible_post_placebo']:<15.4f} {model_placebo.pvalues['eligible_post_placebo']:<12.4f}")
print(f"{'Men only':<35} {model_men.params['eligible_post']:<15.4f} {model_men.bse['eligible_post']:<15.4f} {model_men.pvalues['eligible_post']:<12.4f}")
print(f"{'Women only':<35} {model_women.params['eligible_post']:<15.4f} {model_women.bse['eligible_post']:<15.4f} {model_women.pvalues['eligible_post']:<12.4f}")
print(f"{'Any employment outcome':<35} {model_emp.params['eligible_post']:<15.4f} {model_emp.bse['eligible_post']:<15.4f} {model_emp.pvalues['eligible_post']:<12.4f}")

# Descriptive statistics for report
print("\n=== TABLE 3: SAMPLE CHARACTERISTICS ===")
print(f"{'Variable':<25} {'Eligible':<20} {'Control':<20}")
print("-" * 65)
for var, label in [('AGE', 'Age'), ('female', 'Female'), ('married', 'Married'),
                    ('educ_hs', 'HS or more'), ('fulltime', 'Full-time employed'),
                    ('employed', 'Employed')]:
    elig_mean = df_filtered[df_filtered['daca_eligible']==1][var].mean()
    ctrl_mean = df_filtered[df_filtered['daca_eligible']==0][var].mean()
    print(f"{label:<25} {elig_mean:<20.4f} {ctrl_mean:<20.4f}")

print(f"\n{'N (observations)':<25} {df_filtered[df_filtered['daca_eligible']==1].shape[0]:<20,} {df_filtered[df_filtered['daca_eligible']==0].shape[0]:<20,}")

# PREFERRED ESTIMATE
print("\n" + "=" * 80)
print("PREFERRED ESTIMATE (Model 3 - Year and State FE)")
print("=" * 80)
print(f"Effect Size: {model3.params['eligible_post']:.6f}")
print(f"Standard Error: {model3.bse['eligible_post']:.6f}")
print(f"95% CI: [{model3.params['eligible_post'] - 1.96*model3.bse['eligible_post']:.6f}, {model3.params['eligible_post'] + 1.96*model3.bse['eligible_post']:.6f}]")
print(f"p-value: {model3.pvalues['eligible_post']:.6f}")
print(f"Sample Size: {int(model3.nobs):,}")

# Save key results to file
results_dict = {
    'preferred_estimate': model3.params['eligible_post'],
    'preferred_se': model3.bse['eligible_post'],
    'preferred_pvalue': model3.pvalues['eligible_post'],
    'preferred_n': int(model3.nobs),
    'model1_estimate': model1.params['eligible_post'],
    'model1_se': model1.bse['eligible_post'],
    'model2_estimate': model2.params['eligible_post'],
    'model2_se': model2.bse['eligible_post'],
    'model4_estimate': model4.params['eligible_post'],
    'model4_se': model4.bse['eligible_post'],
    'placebo_estimate': model_placebo.params['eligible_post_placebo'],
    'placebo_pvalue': model_placebo.pvalues['eligible_post_placebo'],
    'men_estimate': model_men.params['eligible_post'],
    'women_estimate': model_women.params['eligible_post'],
    'employment_estimate': model_emp.params['eligible_post'],
}

# Event study results
event_results = {}
for yr in years:
    event_results[f'event_{yr}'] = model_event.params[f'eligible_y{yr}']
    event_results[f'event_{yr}_se'] = model_event.bse[f'eligible_y{yr}']

results_dict.update(event_results)

# Save to CSV for report generation
pd.DataFrame([results_dict]).to_csv('results_95.csv', index=False)
print("\nResults saved to results_95.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
