"""
DACA Impact on Full-Time Employment - Replication Analysis
==========================================================
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for DACA on the
probability of full-time employment (35+ hours/week)?

Author: Replication 79
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import os

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('tables', exist_ok=True)

print("=" * 70)
print("DACA IMPACT ON FULL-TIME EMPLOYMENT - REPLICATION ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")
df = pd.read_csv('data/data.csv')
print(f"    Total observations loaded: {len(df):,}")
print(f"    Years in data: {sorted(df['YEAR'].unique())}")
print(f"    Variables: {df.columns.tolist()}")

# =============================================================================
# 2. SAMPLE CONSTRUCTION
# =============================================================================
print("\n[2] Constructing analysis sample...")

# Step 2a: Filter to Hispanic-Mexican ethnicity
# HISPAN = 1 corresponds to Mexican Hispanic origin
df_sample = df[df['HISPAN'] == 1].copy()
print(f"    After Hispanic-Mexican filter: {len(df_sample):,}")

# Step 2b: Filter to born in Mexico
# BPL = 200 corresponds to Mexico
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"    After Mexican-born filter: {len(df_sample):,}")

# Step 2c: Filter to non-citizens
# CITIZEN = 3 corresponds to "Not a citizen"
# We assume non-citizens who are not naturalized are potentially undocumented
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"    After non-citizen filter: {len(df_sample):,}")

# Step 2d: Restrict to working-age population (16-64)
df_sample = df_sample[(df_sample['AGE'] >= 16) & (df_sample['AGE'] <= 64)].copy()
print(f"    After working-age (16-64) filter: {len(df_sample):,}")

# Step 2e: Exclude 2012 (ambiguous treatment year)
# DACA was announced June 15, 2012; applications began August 15, 2012
# Cannot distinguish pre/post within 2012 in ACS
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"    After excluding 2012: {len(df_sample):,}")

# =============================================================================
# 3. VARIABLE CONSTRUCTION
# =============================================================================
print("\n[3] Constructing analysis variables...")

# 3a. Calculate age at arrival
# Age at arrival = Year of immigration - Birth year
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
# Handle cases where YRIMMIG might be 0 or missing
df_sample.loc[df_sample['YRIMMIG'] == 0, 'age_at_arrival'] = np.nan

# 3b. DACA Eligibility Criteria
# 1. Arrived before 16th birthday: age_at_arrival < 16
# 2. Born after June 15, 1981 (under 31 on June 15, 2012): BIRTHYR >= 1982
# 3. Arrived by 2007 (continuous presence since June 2007): YRIMMIG <= 2007
# 4. Not a citizen (already filtered)

# Create eligibility components
df_sample['arrived_before_16'] = (df_sample['age_at_arrival'] < 16) & (df_sample['age_at_arrival'] >= 0)
df_sample['born_after_1981'] = df_sample['BIRTHYR'] >= 1982
df_sample['arrived_by_2007'] = (df_sample['YRIMMIG'] <= 2007) & (df_sample['YRIMMIG'] > 0)

# DACA eligible if ALL criteria met
df_sample['daca_eligible'] = (
    df_sample['arrived_before_16'] &
    df_sample['born_after_1981'] &
    df_sample['arrived_by_2007']
).astype(int)

print(f"    DACA eligible observations: {df_sample['daca_eligible'].sum():,}")
print(f"    Non-eligible observations: {(df_sample['daca_eligible'] == 0).sum():,}")

# 3c. Outcome variable: Full-time employment (35+ hours/week)
# UHRSWORK = Usual hours worked per week
df_sample['fulltime_emp'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator for reference
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# 3d. Post-DACA period indicator
# Pre-period: 2006-2011
# Post-period: 2013-2016
df_sample['post_daca'] = (df_sample['YEAR'] >= 2013).astype(int)

# 3e. Interaction term (DiD estimator)
df_sample['eligible_x_post'] = df_sample['daca_eligible'] * df_sample['post_daca']

# 3f. Control variables
# Age squared
df_sample['age_sq'] = df_sample['AGE'] ** 2

# Female indicator
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)

# Married indicator (married spouse present or absent)
df_sample['married'] = df_sample['MARST'].isin([1, 2]).astype(int)

# Education categories
df_sample['educ_less_hs'] = (df_sample['EDUC'] <= 5).astype(int)
df_sample['educ_hs'] = (df_sample['EDUC'] == 6).astype(int)
df_sample['educ_some_college'] = df_sample['EDUC'].isin([7, 8, 9]).astype(int)
df_sample['educ_college_plus'] = (df_sample['EDUC'] >= 10).astype(int)

print(f"    Full-time employed: {df_sample['fulltime_emp'].sum():,} ({100*df_sample['fulltime_emp'].mean():.1f}%)")

# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[4] Generating descriptive statistics...")

# 4a. Sample composition by year and eligibility
sample_by_year = df_sample.groupby(['YEAR', 'daca_eligible']).agg({
    'PERWT': 'sum',
    'fulltime_emp': lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']),
    'employed': lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']),
    'AGE': lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT'])
}).reset_index()
sample_by_year.columns = ['Year', 'DACA_Eligible', 'Pop_Weight', 'FT_Emp_Rate', 'Emp_Rate', 'Mean_Age']

print("\n    Sample by Year and Eligibility:")
print(sample_by_year.to_string(index=False))

# 4b. Summary statistics table
def weighted_stats(group, var, weight_var='PERWT'):
    """Calculate weighted mean and std"""
    weights = group[weight_var]
    values = group[var]
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean)**2, weights=weights)
    std = np.sqrt(variance)
    n = len(values)
    return pd.Series({'mean': mean, 'std': std, 'n': n})

# Pre-period statistics
pre_eligible = df_sample[(df_sample['post_daca'] == 0) & (df_sample['daca_eligible'] == 1)]
pre_control = df_sample[(df_sample['post_daca'] == 0) & (df_sample['daca_eligible'] == 0)]
post_eligible = df_sample[(df_sample['post_daca'] == 1) & (df_sample['daca_eligible'] == 1)]
post_control = df_sample[(df_sample['post_daca'] == 1) & (df_sample['daca_eligible'] == 0)]

summary_vars = ['fulltime_emp', 'employed', 'AGE', 'female', 'married', 'educ_less_hs', 'educ_hs', 'educ_some_college', 'educ_college_plus']

print("\n    Summary Statistics by Group:")
print("\n    Pre-Period (2006-2011):")
print("    " + "-"*60)
print(f"    {'Variable':<25} {'Eligible':<15} {'Control':<15}")
print("    " + "-"*60)
for var in summary_vars:
    e_mean = np.average(pre_eligible[var], weights=pre_eligible['PERWT'])
    c_mean = np.average(pre_control[var], weights=pre_control['PERWT'])
    print(f"    {var:<25} {e_mean:>12.3f}   {c_mean:>12.3f}")

print("\n    Post-Period (2013-2016):")
print("    " + "-"*60)
for var in summary_vars:
    e_mean = np.average(post_eligible[var], weights=post_eligible['PERWT'])
    c_mean = np.average(post_control[var], weights=post_control['PERWT'])
    print(f"    {var:<25} {e_mean:>12.3f}   {c_mean:>12.3f}")

# Save summary statistics to file
summary_data = []
for var in summary_vars:
    row = {
        'Variable': var,
        'Pre_Eligible_Mean': np.average(pre_eligible[var], weights=pre_eligible['PERWT']),
        'Pre_Control_Mean': np.average(pre_control[var], weights=pre_control['PERWT']),
        'Post_Eligible_Mean': np.average(post_eligible[var], weights=post_eligible['PERWT']),
        'Post_Control_Mean': np.average(post_control[var], weights=post_control['PERWT'])
    }
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('tables/summary_statistics.csv', index=False)

# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================
print("\n[5] Creating visualizations...")

# 5a. Time trends in full-time employment by eligibility group
yearly_trends = df_sample.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime_emp'], weights=x['PERWT'])
).unstack()
yearly_trends.columns = ['Non-Eligible', 'DACA-Eligible']

plt.figure(figsize=(10, 6))
plt.plot(yearly_trends.index, yearly_trends['DACA-Eligible'], 'b-o', linewidth=2, markersize=8, label='DACA-Eligible')
plt.plot(yearly_trends.index, yearly_trends['Non-Eligible'], 'r-s', linewidth=2, markersize=8, label='Non-Eligible')
plt.axvline(x=2012.5, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('figures/fulltime_emp_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fulltime_emp_trends.pdf', bbox_inches='tight')
plt.close()

print("    Saved: figures/fulltime_emp_trends.png")

# 5b. Difference plot
diff_series = yearly_trends['DACA-Eligible'] - yearly_trends['Non-Eligible']
plt.figure(figsize=(10, 6))
plt.bar(diff_series.index, diff_series.values, color=['blue' if y < 2013 else 'green' for y in diff_series.index], alpha=0.7)
plt.axvline(x=2012.5, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Difference in FT Employment Rate\n(Eligible - Non-Eligible)', fontsize=12)
plt.title('Difference in Full-Time Employment: DACA-Eligible vs Non-Eligible', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figures/diff_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/diff_trends.pdf', bbox_inches='tight')
plt.close()

print("    Saved: figures/diff_trends.png")

# Save trend data
yearly_trends.to_csv('tables/yearly_trends.csv')

# =============================================================================
# 6. MAIN DIFFERENCE-IN-DIFFERENCES ESTIMATION
# =============================================================================
print("\n[6] Running Difference-in-Differences estimation...")

# Model 1: Basic DiD (no controls)
print("\n    Model 1: Basic DiD (no controls)...")
model1 = smf.wls(
    'fulltime_emp ~ daca_eligible + post_daca + eligible_x_post',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='HC1')

print(f"    DiD Estimate (eligible_x_post): {model1.params['eligible_x_post']:.4f}")
print(f"    Robust SE: {model1.bse['eligible_x_post']:.4f}")
print(f"    t-stat: {model1.tvalues['eligible_x_post']:.3f}")
print(f"    p-value: {model1.pvalues['eligible_x_post']:.4f}")

# Model 2: DiD with demographic controls
print("\n    Model 2: DiD with demographic controls...")
model2 = smf.wls(
    'fulltime_emp ~ daca_eligible + post_daca + eligible_x_post + AGE + age_sq + female + married',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='HC1')

print(f"    DiD Estimate (eligible_x_post): {model2.params['eligible_x_post']:.4f}")
print(f"    Robust SE: {model2.bse['eligible_x_post']:.4f}")

# Model 3: DiD with demographic + education controls
print("\n    Model 3: DiD with demographic + education controls...")
model3 = smf.wls(
    'fulltime_emp ~ daca_eligible + post_daca + eligible_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='HC1')

print(f"    DiD Estimate (eligible_x_post): {model3.params['eligible_x_post']:.4f}")
print(f"    Robust SE: {model3.bse['eligible_x_post']:.4f}")

# Model 4: DiD with year fixed effects
print("\n    Model 4: DiD with year fixed effects...")
df_sample['year_factor'] = pd.Categorical(df_sample['YEAR'])
model4 = smf.wls(
    'fulltime_emp ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR)',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='HC1')

print(f"    DiD Estimate (eligible_x_post): {model4.params['eligible_x_post']:.4f}")
print(f"    Robust SE: {model4.bse['eligible_x_post']:.4f}")

# Model 5: Full model with state and year fixed effects
print("\n    Model 5: Full model with state and year fixed effects...")
model5 = smf.wls(
    'fulltime_emp ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='HC1')

print(f"    DiD Estimate (eligible_x_post): {model5.params['eligible_x_post']:.4f}")
print(f"    Robust SE: {model5.bse['eligible_x_post']:.4f}")
print(f"    95% CI: [{model5.conf_int().loc['eligible_x_post', 0]:.4f}, {model5.conf_int().loc['eligible_x_post', 1]:.4f}]")

# =============================================================================
# 7. EVENT STUDY SPECIFICATION
# =============================================================================
print("\n[7] Running Event Study specification...")

# Create year-specific interactions
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    df_sample[f'eligible_x_{year}'] = df_sample['daca_eligible'] * (df_sample['YEAR'] == year).astype(int)

# Reference year: 2011 (last pre-treatment year)
event_formula = 'fulltime_emp ~ daca_eligible + '
event_formula += ' + '.join([f'eligible_x_{y}' for y in years if y != 2011])
event_formula += ' + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)'

event_model = smf.wls(event_formula, data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')

# Extract event study coefficients
event_coefs = []
for year in years:
    if year == 2011:
        event_coefs.append({'year': year, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
    else:
        var = f'eligible_x_{year}'
        coef = event_model.params[var]
        se = event_model.bse[var]
        ci = event_model.conf_int().loc[var]
        event_coefs.append({'year': year, 'coef': coef, 'se': se, 'ci_low': ci[0], 'ci_high': ci[1]})

event_df = pd.DataFrame(event_coefs)
event_df.to_csv('tables/event_study_coefficients.csv', index=False)

print("\n    Event Study Coefficients (relative to 2011):")
print("    " + "-"*60)
for _, row in event_df.iterrows():
    sig = '*' if row['ci_low'] > 0 or row['ci_high'] < 0 else ''
    print(f"    Year {int(row['year'])}: {row['coef']:>8.4f} (SE: {row['se']:.4f}) [{row['ci_low']:.4f}, {row['ci_high']:.4f}] {sig}")

# Plot event study
plt.figure(figsize=(10, 6))
plt.errorbar(event_df['year'], event_df['coef'],
             yerr=[event_df['coef'] - event_df['ci_low'], event_df['ci_high'] - event_df['coef']],
             fmt='o-', capsize=4, capthick=2, linewidth=2, markersize=8, color='blue')
plt.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.fill_between([2012, 2016.5], -0.1, 0.1, alpha=0.1, color='green', label='Post-DACA Period')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Effect on Full-Time Employment\n(Relative to 2011)', fontsize=12)
plt.title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(years)
plt.tight_layout()
plt.savefig('figures/event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/event_study.pdf', bbox_inches='tight')
plt.close()

print("    Saved: figures/event_study.png")

# =============================================================================
# 8. ROBUSTNESS CHECKS
# =============================================================================
print("\n[8] Running robustness checks...")

# 8a. Subsample analysis by gender
print("\n    8a. Subgroup analysis by gender...")
male_sample = df_sample[df_sample['female'] == 0]
female_sample = df_sample[df_sample['female'] == 1]

model_male = smf.wls(
    'fulltime_emp ~ daca_eligible + eligible_x_post + AGE + age_sq + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)',
    data=male_sample,
    weights=male_sample['PERWT']
).fit(cov_type='HC1')

model_female = smf.wls(
    'fulltime_emp ~ daca_eligible + eligible_x_post + AGE + age_sq + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)',
    data=female_sample,
    weights=female_sample['PERWT']
).fit(cov_type='HC1')

print(f"    Males - DiD Estimate: {model_male.params['eligible_x_post']:.4f} (SE: {model_male.bse['eligible_x_post']:.4f})")
print(f"    Females - DiD Estimate: {model_female.params['eligible_x_post']:.4f} (SE: {model_female.bse['eligible_x_post']:.4f})")

# 8b. Alternative age restrictions
print("\n    8b. Alternative age restrictions (18-55)...")
df_age_alt = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 55)]
model_age_alt = smf.wls(
    'fulltime_emp ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)',
    data=df_age_alt,
    weights=df_age_alt['PERWT']
).fit(cov_type='HC1')

print(f"    Ages 18-55 - DiD Estimate: {model_age_alt.params['eligible_x_post']:.4f} (SE: {model_age_alt.bse['eligible_x_post']:.4f})")

# 8c. Any employment (not just full-time)
print("\n    8c. Any employment outcome...")
model_any_emp = smf.wls(
    'employed ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)',
    data=df_sample,
    weights=df_sample['PERWT']
).fit(cov_type='HC1')

print(f"    Any Employment - DiD Estimate: {model_any_emp.params['eligible_x_post']:.4f} (SE: {model_any_emp.bse['eligible_x_post']:.4f})")

# 8d. Placebo test: Citizens as control (if any in Mexico-born sample)
print("\n    8d. Checking naturalized citizens (placebo group)...")
df_citizens = df[(df['HISPAN'] == 1) & (df['BPL'] == 200) & (df['CITIZEN'] == 2) &
                  (df['AGE'] >= 16) & (df['AGE'] <= 64) & (df['YEAR'] != 2012)]
print(f"    Naturalized citizens in sample: {len(df_citizens):,}")

# =============================================================================
# 9. SAVE REGRESSION RESULTS
# =============================================================================
print("\n[9] Saving regression results...")

# Create summary table
results_summary = pd.DataFrame({
    'Model': ['Basic DiD', 'Demographics', 'Demographics + Education', 'Year FE', 'State + Year FE'],
    'DiD_Estimate': [
        model1.params['eligible_x_post'],
        model2.params['eligible_x_post'],
        model3.params['eligible_x_post'],
        model4.params['eligible_x_post'],
        model5.params['eligible_x_post']
    ],
    'Robust_SE': [
        model1.bse['eligible_x_post'],
        model2.bse['eligible_x_post'],
        model3.bse['eligible_x_post'],
        model4.bse['eligible_x_post'],
        model5.bse['eligible_x_post']
    ],
    't_stat': [
        model1.tvalues['eligible_x_post'],
        model2.tvalues['eligible_x_post'],
        model3.tvalues['eligible_x_post'],
        model4.tvalues['eligible_x_post'],
        model5.tvalues['eligible_x_post']
    ],
    'p_value': [
        model1.pvalues['eligible_x_post'],
        model2.pvalues['eligible_x_post'],
        model3.pvalues['eligible_x_post'],
        model4.pvalues['eligible_x_post'],
        model5.pvalues['eligible_x_post']
    ],
    'N': [model1.nobs, model2.nobs, model3.nobs, model4.nobs, model5.nobs],
    'R_squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared, model5.rsquared]
})

results_summary.to_csv('tables/regression_results.csv', index=False)

# Robustness results
robustness_summary = pd.DataFrame({
    'Specification': ['Males Only', 'Females Only', 'Ages 18-55', 'Any Employment'],
    'DiD_Estimate': [
        model_male.params['eligible_x_post'],
        model_female.params['eligible_x_post'],
        model_age_alt.params['eligible_x_post'],
        model_any_emp.params['eligible_x_post']
    ],
    'Robust_SE': [
        model_male.bse['eligible_x_post'],
        model_female.bse['eligible_x_post'],
        model_age_alt.bse['eligible_x_post'],
        model_any_emp.bse['eligible_x_post']
    ],
    'N': [model_male.nobs, model_female.nobs, model_age_alt.nobs, model_any_emp.nobs]
})

robustness_summary.to_csv('tables/robustness_results.csv', index=False)

# =============================================================================
# 10. PRINT FINAL RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print("\n--- PREFERRED ESTIMATE (Model 5: Full specification) ---")
print(f"DiD Estimate (Effect of DACA eligibility on full-time employment): {model5.params['eligible_x_post']:.4f}")
print(f"Robust Standard Error: {model5.bse['eligible_x_post']:.4f}")
print(f"t-statistic: {model5.tvalues['eligible_x_post']:.3f}")
print(f"p-value: {model5.pvalues['eligible_x_post']:.4f}")
ci = model5.conf_int().loc['eligible_x_post']
print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"Sample Size: {int(model5.nobs):,}")
print(f"R-squared: {model5.rsquared:.4f}")

print("\n--- INTERPRETATION ---")
effect_pct = model5.params['eligible_x_post'] * 100
print(f"DACA eligibility is associated with a {effect_pct:.2f} percentage point")
if model5.params['eligible_x_post'] > 0:
    print("increase in the probability of full-time employment.")
else:
    print("decrease in the probability of full-time employment.")

if model5.pvalues['eligible_x_post'] < 0.01:
    print("This effect is statistically significant at the 1% level.")
elif model5.pvalues['eligible_x_post'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
elif model5.pvalues['eligible_x_post'] < 0.10:
    print("This effect is statistically significant at the 10% level.")
else:
    print("This effect is NOT statistically significant at conventional levels.")

print("\n--- SAMPLE INFORMATION ---")
print(f"Total sample: {len(df_sample):,} person-year observations")
print(f"DACA-eligible: {df_sample['daca_eligible'].sum():,}")
print(f"Non-eligible: {(df_sample['daca_eligible'] == 0).sum():,}")
print(f"Pre-period years: 2006-2011")
print(f"Post-period years: 2013-2016")

# Pre-treatment means
pre_elig_mean = np.average(pre_eligible['fulltime_emp'], weights=pre_eligible['PERWT'])
pre_ctrl_mean = np.average(pre_control['fulltime_emp'], weights=pre_control['PERWT'])
print(f"\nPre-treatment FT employment rate (Eligible): {pre_elig_mean:.3f}")
print(f"Pre-treatment FT employment rate (Control): {pre_ctrl_mean:.3f}")

print("\n" + "=" * 70)
print("Analysis complete. Results saved to tables/ and figures/ directories.")
print("=" * 70)

# Save key results for report
with open('tables/key_results.txt', 'w') as f:
    f.write("PREFERRED ESTIMATE SUMMARY\n")
    f.write("=" * 50 + "\n")
    f.write(f"DiD Estimate: {model5.params['eligible_x_post']:.4f}\n")
    f.write(f"Robust SE: {model5.bse['eligible_x_post']:.4f}\n")
    f.write(f"t-statistic: {model5.tvalues['eligible_x_post']:.3f}\n")
    f.write(f"p-value: {model5.pvalues['eligible_x_post']:.4f}\n")
    f.write(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")
    f.write(f"Sample Size: {int(model5.nobs):,}\n")
    f.write(f"R-squared: {model5.rsquared:.4f}\n")
