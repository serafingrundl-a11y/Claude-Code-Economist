"""
DACA Replication Study - Analysis Script
Replication ID: 56

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING - CHUNKED TO HANDLE LARGE FILE
# ============================================================================
print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

print("\n1. Loading data in chunks and filtering...")

# Define columns we need
cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR', 'MARST',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT',
               'UHRSWORK', 'STATEFIP']

# Read in chunks and filter
chunks = []
chunk_size = 500000
total_rows = 0
filtered_rows = 0

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size):
    total_rows += len(chunk)

    # Apply initial filters:
    # 1. Hispanic-Mexican (HISPAN == 1)
    # 2. Born in Mexico (BPL == 200)
    # 3. Non-citizen (CITIZEN == 3)
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3)].copy()

    if len(filtered) > 0:
        chunks.append(filtered)
        filtered_rows += len(filtered)

    if total_rows % 5000000 == 0:
        print(f"   Processed {total_rows:,} rows, kept {filtered_rows:,}")

df = pd.concat(chunks, ignore_index=True)
print(f"\n   Total observations in file: {total_rows:,}")
print(f"   After Hispanic-Mexican/Mexico-born/Non-citizen filter: {len(df):,}")

# ============================================================================
# 2. DETERMINE AGE-BASED TREATMENT AND CONTROL GROUPS
# ============================================================================
print("\n2. Defining Treatment and Control Groups...")

# DACA implementation date: June 15, 2012
# Treatment: Ages 26-30 on June 15, 2012 (born July 1, 1981 - June 14, 1986)
# Control: Ages 31-35 on June 15, 2012 (born July 1, 1976 - June 14, 1981)

# Calculate age as of June 15, 2012
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

# Age on June 15, 2012:
# If born before June 15 (Q1, Q2), use 2012 - BIRTHYR
# If born after June 15 (Q3, Q4), use 2012 - BIRTHYR - 1

df['age_june_2012'] = 2012 - df['BIRTHYR']
# Adjust for those born after June 15 (roughly Q3 and Q4)
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june_2012'] -= 1

# Define treatment group (ages 26-30 on June 15, 2012)
df['treatment'] = ((df['age_june_2012'] >= 26) &
                   (df['age_june_2012'] <= 30)).astype(int)

# Define control group (ages 31-35 on June 15, 2012)
df['control'] = ((df['age_june_2012'] >= 31) &
                 (df['age_june_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df[(df['treatment'] == 1) | (df['control'] == 1)].copy()
print(f"   Observations in treatment/control groups: {len(df_analysis):,}")

# Create single treatment indicator (1 = treatment group, 0 = control group)
df_analysis['treated'] = df_analysis['treatment']

print(f"   Treatment group (26-30): {len(df_analysis[df_analysis['treated']==1]):,}")
print(f"   Control group (31-35): {len(df_analysis[df_analysis['treated']==0]):,}")

# ============================================================================
# 3. APPLY DACA ELIGIBILITY CRITERIA
# ============================================================================
print("\n3. Applying DACA Eligibility Criteria...")

# Calculate age at immigration
df_analysis['age_at_immigration'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']

# Criterion 1: Arrived before 16th birthday
df_analysis['arrived_before_16'] = (df_analysis['age_at_immigration'] < 16).astype(int)
print(f"   Arrived before age 16: {df_analysis['arrived_before_16'].sum():,}")

# Criterion 2: Arrived by June 15, 2007 (continuous residence)
df_analysis['arrived_by_2007'] = (df_analysis['YRIMMIG'] <= 2007).astype(int)
print(f"   Arrived by 2007: {df_analysis['arrived_by_2007'].sum():,}")

# Criterion 3: Present in US on June 15, 2012
df_analysis['present_2012'] = (df_analysis['YRIMMIG'] <= 2012).astype(int)

# Apply all eligibility criteria
eligible_mask = ((df_analysis['arrived_before_16'] == 1) &
                 (df_analysis['arrived_by_2007'] == 1) &
                 (df_analysis['present_2012'] == 1) &
                 (df_analysis['YRIMMIG'] > 0))  # Valid immigration year

df_eligible = df_analysis[eligible_mask].copy()
print(f"   Observations meeting all eligibility criteria: {len(df_eligible):,}")

# ============================================================================
# 4. DEFINE TIME PERIODS AND OUTCOME
# ============================================================================
print("\n4. Defining Time Periods and Outcome...")

# Exclude 2012 (DACA implemented mid-year)
df_eligible = df_eligible[df_eligible['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_eligible):,}")

# Post-treatment indicator (2013-2016)
df_eligible['post'] = (df_eligible['YEAR'] >= 2013).astype(int)

# Full-time employment (UHRSWORK >= 35)
df_eligible['fulltime'] = (df_eligible['UHRSWORK'] >= 35).astype(int)

# ============================================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n5. Descriptive Statistics...")

# Sample sizes by group and period
print("\n   Sample Sizes (unweighted):")
print("   " + "-" * 50)
crosstab = pd.crosstab(df_eligible['treated'], df_eligible['post'],
                       margins=True, margins_name='Total')
crosstab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
crosstab.columns = ['Pre (2006-2011)', 'Post (2013-2016)', 'Total']
print(crosstab.to_string())

# Weighted sample sizes
print("\n   Weighted Population Sizes:")
print("   " + "-" * 50)
for treated in [0, 1]:
    for post in [0, 1]:
        mask = (df_eligible['treated'] == treated) & (df_eligible['post'] == post)
        weight_sum = df_eligible.loc[mask, 'PERWT'].sum()
        group = 'Treatment' if treated else 'Control'
        period = 'Post' if post else 'Pre'
        print(f"   {group} x {period}: {weight_sum:,.0f}")

# Full-time employment rates by group and period
print("\n   Full-Time Employment Rates (weighted):")
print("   " + "-" * 50)
rates = {}
for treated in [0, 1]:
    for post in [0, 1]:
        mask = (df_eligible['treated'] == treated) & (df_eligible['post'] == post)
        subset = df_eligible[mask]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        rates[(treated, post)] = weighted_mean
        group = 'Treatment' if treated else 'Control'
        period = 'Post' if post else 'Pre'
        print(f"   {group} x {period}: {weighted_mean:.4f} ({weighted_mean*100:.2f}%)")

# Calculate simple DiD
did_simple = (rates[(1, 1)] - rates[(1, 0)]) - (rates[(0, 1)] - rates[(0, 0)])
print(f"\n   Simple DiD estimate: {did_simple:.4f} ({did_simple*100:.2f} pp)")

# ============================================================================
# 6. MAIN REGRESSION ANALYSIS
# ============================================================================
print("\n6. Main Regression Analysis...")
print("=" * 70)

# Create interaction term
df_eligible['treated_post'] = df_eligible['treated'] * df_eligible['post']

# Model 1: Basic DiD (no controls)
print("\n   Model 1: Basic Difference-in-Differences")
print("   " + "-" * 50)
model1 = smf.wls('fulltime ~ treated + post + treated_post',
                 data=df_eligible,
                 weights=df_eligible['PERWT']).fit(cov_type='HC1')
print(f"   DiD Coefficient (treated_post): {model1.params['treated_post']:.4f}")
print(f"   Standard Error: {model1.bse['treated_post']:.4f}")
print(f"   t-statistic: {model1.tvalues['treated_post']:.4f}")
print(f"   p-value: {model1.pvalues['treated_post']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['treated_post', 0]:.4f}, "
      f"{model1.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   N = {int(model1.nobs):,}")

# Model 2: DiD with year fixed effects
print("\n   Model 2: DiD with Year Fixed Effects")
print("   " + "-" * 50)
model2 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                 data=df_eligible,
                 weights=df_eligible['PERWT']).fit(cov_type='HC1')
print(f"   DiD Coefficient (treated_post): {model2.params['treated_post']:.4f}")
print(f"   Standard Error: {model2.bse['treated_post']:.4f}")
print(f"   t-statistic: {model2.tvalues['treated_post']:.4f}")
print(f"   p-value: {model2.pvalues['treated_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['treated_post', 0]:.4f}, "
      f"{model2.conf_int().loc['treated_post', 1]:.4f}]")

# Model 3: DiD with covariates
print("\n   Model 3: DiD with Year FE and Individual Covariates")
print("   " + "-" * 50)

# Create covariate variables
df_eligible['female'] = (df_eligible['SEX'] == 2).astype(int)
df_eligible['married'] = (df_eligible['MARST'] == 1).astype(int)
df_eligible['educ_hs_plus'] = (df_eligible['EDUC'] >= 6).astype(int)  # HS or more

model3 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + educ_hs_plus',
                 data=df_eligible,
                 weights=df_eligible['PERWT']).fit(cov_type='HC1')
print(f"   DiD Coefficient (treated_post): {model3.params['treated_post']:.4f}")
print(f"   Standard Error: {model3.bse['treated_post']:.4f}")
print(f"   t-statistic: {model3.tvalues['treated_post']:.4f}")
print(f"   p-value: {model3.pvalues['treated_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['treated_post', 0]:.4f}, "
      f"{model3.conf_int().loc['treated_post', 1]:.4f}]")

# Model 4: Full model with state fixed effects (PREFERRED)
print("\n   Model 4: DiD with Year FE, State FE, and Covariates (PREFERRED)")
print("   " + "-" * 50)
model4 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + educ_hs_plus',
                 data=df_eligible,
                 weights=df_eligible['PERWT']).fit(cov_type='HC1')
print(f"   DiD Coefficient (treated_post): {model4.params['treated_post']:.4f}")
print(f"   Standard Error: {model4.bse['treated_post']:.4f}")
print(f"   t-statistic: {model4.tvalues['treated_post']:.4f}")
print(f"   p-value: {model4.pvalues['treated_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, "
      f"{model4.conf_int().loc['treated_post', 1]:.4f}]")

# ============================================================================
# 7. ROBUSTNESS CHECKS
# ============================================================================
print("\n7. Robustness Checks...")
print("=" * 70)

# Check 1: Pre-treatment trends (event study)
print("\n   Event Study / Pre-Trend Analysis")
print("   " + "-" * 50)

# Create year-specific treatment effects
years = sorted(df_eligible['YEAR'].unique())
for year in years:
    df_eligible[f'treated_year_{year}'] = (df_eligible['treated'] *
                                           (df_eligible['YEAR'] == year)).astype(int)

# Use 2011 as reference year
year_vars = [f'treated_year_{y}' for y in years if y != 2011]
formula = 'fulltime ~ treated + C(YEAR) + ' + ' + '.join(year_vars)
model_event = smf.wls(formula, data=df_eligible,
                      weights=df_eligible['PERWT']).fit(cov_type='HC1')

print("   Year-specific treatment effects (relative to 2011):")
for year in years:
    if year != 2011:
        var = f'treated_year_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        pval = model_event.pvalues[var]
        sig = '*' if pval < 0.05 else ''
        print(f"   {year}: {coef:7.4f} (SE: {se:.4f}) {sig}")
    else:
        print(f"   {year}:  0.0000 (reference year)")

# ============================================================================
# 8. HETEROGENEITY ANALYSIS
# ============================================================================
print("\n8. Heterogeneity Analysis...")
print("=" * 70)

# By gender
print("\n   By Gender:")
print("   " + "-" * 50)
for gender, label in [(1, 'Male'), (2, 'Female')]:
    subset = df_eligible[df_eligible['SEX'] == gender]
    model_gender = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                           data=subset,
                           weights=subset['PERWT']).fit(cov_type='HC1')
    ci = model_gender.conf_int().loc['treated_post']
    print(f"   {label}: {model_gender.params['treated_post']:.4f} "
          f"(SE: {model_gender.bse['treated_post']:.4f}), "
          f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}], N={len(subset):,}")

# By education
print("\n   By Education Level:")
print("   " + "-" * 50)
for educ_label, educ_cond in [('Less than HS', df_eligible['EDUC'] < 6),
                               ('HS or more', df_eligible['EDUC'] >= 6)]:
    subset = df_eligible[educ_cond]
    if len(subset) > 100:
        model_educ = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                             data=subset,
                             weights=subset['PERWT']).fit(cov_type='HC1')
        ci = model_educ.conf_int().loc['treated_post']
        print(f"   {educ_label}: {model_educ.params['treated_post']:.4f} "
              f"(SE: {model_educ.bse['treated_post']:.4f}), "
              f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}], N={len(subset):,}")

# ============================================================================
# 9. ADDITIONAL DESCRIPTIVE STATISTICS
# ============================================================================
print("\n9. Additional Descriptive Statistics...")
print("=" * 70)

# Demographics by treatment status
print("\n   Demographics by Treatment Status (Pre-period only):")
print("   " + "-" * 50)
pre_data = df_eligible[df_eligible['post'] == 0]
for treated, label in [(0, 'Control (31-35)'), (1, 'Treatment (26-30)')]:
    subset = pre_data[pre_data['treated'] == treated]
    pct_female = np.average(subset['female'], weights=subset['PERWT']) * 100
    pct_married = np.average(subset['married'], weights=subset['PERWT']) * 100
    pct_hs = np.average(subset['educ_hs_plus'], weights=subset['PERWT']) * 100
    mean_age = np.average(subset['AGE'], weights=subset['PERWT'])
    print(f"   {label}:")
    print(f"      Female: {pct_female:.1f}%")
    print(f"      Married: {pct_married:.1f}%")
    print(f"      HS or more: {pct_hs:.1f}%")
    print(f"      Mean age: {mean_age:.1f}")

# ============================================================================
# 10. SAVE RESULTS FOR REPORT
# ============================================================================
print("\n10. Saving Results...")
print("=" * 70)

# Create results dictionary
results = {
    'n_total': len(df_eligible),
    'n_treatment': len(df_eligible[df_eligible['treated'] == 1]),
    'n_control': len(df_eligible[df_eligible['treated'] == 0]),
    'did_simple': did_simple,
    'model1_coef': model1.params['treated_post'],
    'model1_se': model1.bse['treated_post'],
    'model1_pval': model1.pvalues['treated_post'],
    'model1_ci_low': model1.conf_int().loc['treated_post', 0],
    'model1_ci_high': model1.conf_int().loc['treated_post', 1],
    'model2_coef': model2.params['treated_post'],
    'model2_se': model2.bse['treated_post'],
    'model3_coef': model3.params['treated_post'],
    'model3_se': model3.bse['treated_post'],
    'model4_coef': model4.params['treated_post'],
    'model4_se': model4.bse['treated_post'],
    'model4_pval': model4.pvalues['treated_post'],
    'model4_ci_low': model4.conf_int().loc['treated_post', 0],
    'model4_ci_high': model4.conf_int().loc['treated_post', 1],
}

# Save rates
results['rate_treat_pre'] = rates[(1, 0)]
results['rate_treat_post'] = rates[(1, 1)]
results['rate_ctrl_pre'] = rates[(0, 0)]
results['rate_ctrl_post'] = rates[(0, 1)]

# Save to CSV
pd.DataFrame([results]).to_csv('results_summary.csv', index=False)
print("   Results saved to results_summary.csv")

# ============================================================================
# 11. CREATE DATA FOR FIGURES
# ============================================================================
print("\n11. Creating Data for Figures...")

# Yearly means by group
yearly_means = df_eligible.groupby(['YEAR', 'treated']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()

yearly_means.to_csv('yearly_means.csv', index=False)
print("   Yearly means saved to yearly_means.csv")

# Event study coefficients
event_coefs = []
for year in years:
    if year == 2011:
        event_coefs.append({'year': year, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
    else:
        var = f'treated_year_{year}'
        event_coefs.append({
            'year': year,
            'coef': model_event.params[var],
            'se': model_event.bse[var],
            'ci_low': model_event.conf_int().loc[var, 0],
            'ci_high': model_event.conf_int().loc[var, 1]
        })

pd.DataFrame(event_coefs).to_csv('event_study_coefs.csv', index=False)
print("   Event study coefficients saved to event_study_coefs.csv")

# ============================================================================
# 12. PRINT FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY - PREFERRED ESTIMATE")
print("=" * 70)
print(f"\nPreferred Model: Model 4 (Year FE + State FE + Covariates)")
print(f"Sample Size: {len(df_eligible):,}")
print(f"Treatment Effect (DiD): {model4.params['treated_post']:.4f}")
print(f"Standard Error: {model4.bse['treated_post']:.4f}")
print(f"95% Confidence Interval: [{model4.conf_int().loc['treated_post', 0]:.4f}, "
      f"{model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"p-value: {model4.pvalues['treated_post']:.4f}")

interpretation = "increase" if model4.params['treated_post'] > 0 else "decrease"
pct_points = abs(model4.params['treated_post'] * 100)
print(f"\nInterpretation: DACA eligibility is associated with a {pct_points:.2f} percentage point")
print(f"{interpretation} in the probability of full-time employment.")

if model4.pvalues['treated_post'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
else:
    print("This effect is not statistically significant at the 5% level.")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
