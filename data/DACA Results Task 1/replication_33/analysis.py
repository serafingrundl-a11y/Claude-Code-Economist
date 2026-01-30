"""
DACA Impact on Full-Time Employment: Independent Replication Analysis
======================================================================

Research Question:
Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on the probability of full-time employment?

Author: Anonymous
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

# Set display options
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)

print("="*80)
print("DACA IMPACT ON FULL-TIME EMPLOYMENT: REPLICATION ANALYSIS")
print("="*80)

#------------------------------------------------------------------------------
# 1. LOAD DATA (CHUNKED FOR MEMORY EFFICIENCY)
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("SECTION 1: DATA LOADING")
print("="*80)

print("\nLoading ACS data (2006-2016) using chunked reading...")
data_path = 'data/data.csv'

# Define columns we need to reduce memory
use_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
            'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
            'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'LABFORCE', 'UHRSWORK']

# Read data in chunks and filter on the fly
chunks = []
chunksize = 500000
total_rows = 0

for chunk in pd.read_csv(data_path, usecols=use_cols, chunksize=chunksize, low_memory=False):
    total_rows += len(chunk)
    # Apply initial filters to reduce memory
    # Filter: Hispanic-Mexican (HISPAN=1), Born in Mexico (BPL=200), Non-citizen (CITIZEN=3)
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3)].copy()
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"  Processed {total_rows:,} rows, kept {sum(len(c) for c in chunks):,} so far...")

# Combine chunks
df = pd.concat(chunks, ignore_index=True)
del chunks  # Free memory

print(f"\nTotal rows in original data: {total_rows:,}")
print(f"Rows after initial filtering: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

#------------------------------------------------------------------------------
# 2. SAMPLE RESTRICTIONS
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("SECTION 2: SAMPLE RESTRICTIONS")
print("="*80)

# Track sample restrictions
sample_tracking = []
sample_tracking.append(('Initial sample (full ACS data)', total_rows))
sample_tracking.append(('Hispanic-Mexican, Mexican-born, non-citizen', len(df)))

# Exclude 2012 (DACA implemented mid-year, cannot distinguish pre/post)
df = df[df['YEAR'] != 2012].copy()
sample_tracking.append(('Exclude 2012 (mid-implementation year)', len(df)))
print(f"After excluding 2012: {len(df):,}")

# Restrict to working-age population (18-64)
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)].copy()
sample_tracking.append(('Working age 18-64', len(df)))
print(f"After restricting to ages 18-64: {len(df):,}")

# Print sample tracking table
print("\n" + "-"*60)
print("SAMPLE RESTRICTION SUMMARY")
print("-"*60)
for desc, n in sample_tracking:
    print(f"  {desc}: {n:,}")

# Final analytic sample
df_sample = df.copy()
del df  # Free memory

#------------------------------------------------------------------------------
# 3. CONSTRUCT VARIABLES
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("SECTION 3: VARIABLE CONSTRUCTION")
print("="*80)

# 3.1 Outcome Variable: Full-time employment (UHRSWORK >= 35)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)
print(f"\nFull-time employment rate (UHRSWORK >= 35): {df_sample['fulltime'].mean():.4f}")

# Alternative outcome: Employed (EMPSTAT == 1)
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)
print(f"Employment rate (EMPSTAT=1): {df_sample['employed'].mean():.4f}")

# 3.2 Post-DACA indicator
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
print(f"\nPost-DACA observations (2013-2016): {df_sample['post'].sum():,} ({df_sample['post'].mean():.2%})")

# 3.3 DACA Eligibility Criteria
# Criteria from instructions:
# 1. Arrived before 16th birthday
# 2. Not yet 31 as of June 15, 2012 (born after June 15, 1981)
# 3. Lived continuously in US since June 15, 2007 (arrived by 2007)
# 4. Non-citizen (already filtered)

# Calculate age at arrival
# YRIMMIG is year of immigration
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Handle cases where YRIMMIG might be 0 or missing
df_sample.loc[df_sample['YRIMMIG'] == 0, 'age_at_arrival'] = np.nan

# Criterion 1: Arrived before 16th birthday
df_sample['arrived_before_16'] = (df_sample['age_at_arrival'] < 16).astype(int)
df_sample.loc[df_sample['age_at_arrival'].isna(), 'arrived_before_16'] = 0

# Criterion 2: Born after June 15, 1981 (not yet 31 as of June 15, 2012)
# Being conservative: use BIRTHYR > 1981 (those born in 1982 or later definitely qualify)
# For those born in 1981, they might or might not qualify depending on birth month
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born in 1981 Q3 or Q4 (after June 15), they qualify
df_sample['young_enough'] = ((df_sample['BIRTHYR'] > 1981) |
                              ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))).astype(int)

# Criterion 3: In US since 2007 (YRIMMIG <= 2007)
df_sample['in_us_since_2007'] = ((df_sample['YRIMMIG'] <= 2007) & (df_sample['YRIMMIG'] > 0)).astype(int)

# Combined DACA eligibility (meeting all criteria)
df_sample['daca_eligible'] = ((df_sample['arrived_before_16'] == 1) &
                               (df_sample['young_enough'] == 1) &
                               (df_sample['in_us_since_2007'] == 1)).astype(int)

print(f"\nDACA Eligibility Components:")
print(f"  Arrived before age 16: {df_sample['arrived_before_16'].mean():.2%}")
print(f"  Born after June 1981 (not 31 by June 2012): {df_sample['young_enough'].mean():.2%}")
print(f"  In US since 2007: {df_sample['in_us_since_2007'].mean():.2%}")
print(f"  DACA eligible (all criteria): {df_sample['daca_eligible'].mean():.2%}")
print(f"  N DACA eligible: {df_sample['daca_eligible'].sum():,}")

# 3.4 Treatment variable (interaction)
df_sample['treat'] = df_sample['daca_eligible'] * df_sample['post']
print(f"\nTreatment (DACA eligible x Post): {df_sample['treat'].sum():,} observations")

# 3.5 Control Variables
# Sex (male = 1)
df_sample['male'] = (df_sample['SEX'] == 1).astype(int)

# Age polynomials
df_sample['age_sq'] = df_sample['AGE'] ** 2

# Married
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Education categories
df_sample['educ_hs'] = (df_sample['EDUC'] >= 6).astype(int)  # High school or more
df_sample['educ_college'] = (df_sample['EDUC'] >= 10).astype(int)  # Some college or more

# Years in US
df_sample['years_in_us'] = df_sample['YEAR'] - df_sample['YRIMMIG']
df_sample.loc[df_sample['YRIMMIG'] == 0, 'years_in_us'] = np.nan

print(f"\nControl Variables Summary:")
print(f"  Male: {df_sample['male'].mean():.2%}")
print(f"  Mean age: {df_sample['AGE'].mean():.1f}")
print(f"  Married: {df_sample['married'].mean():.2%}")
print(f"  HS or more: {df_sample['educ_hs'].mean():.2%}")
print(f"  Some college+: {df_sample['educ_college'].mean():.2%}")
print(f"  Mean years in US: {df_sample['years_in_us'].mean():.1f}")

#------------------------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("SECTION 4: DESCRIPTIVE STATISTICS")
print("="*80)

# 4.1 Summary statistics by DACA eligibility
print("\n" + "-"*60)
print("SUMMARY STATISTICS BY DACA ELIGIBILITY")
print("-"*60)

summary_vars = ['fulltime', 'employed', 'AGE', 'male', 'married', 'educ_hs',
                'educ_college', 'years_in_us', 'UHRSWORK']

for var in summary_vars:
    eligible_mean = df_sample[df_sample['daca_eligible']==1][var].mean()
    ineligible_mean = df_sample[df_sample['daca_eligible']==0][var].mean()
    diff = eligible_mean - ineligible_mean
    print(f"  {var:15s}: Eligible={eligible_mean:8.3f}, Ineligible={ineligible_mean:8.3f}, Diff={diff:+8.3f}")

# 4.2 Sample sizes by year and eligibility
print("\n" + "-"*60)
print("SAMPLE SIZE BY YEAR AND DACA ELIGIBILITY")
print("-"*60)
crosstab = pd.crosstab(df_sample['YEAR'], df_sample['daca_eligible'], margins=True)
crosstab.columns = ['Ineligible', 'Eligible', 'Total']
print(crosstab)

# 4.3 Full-time employment by year and eligibility
print("\n" + "-"*60)
print("FULL-TIME EMPLOYMENT RATE BY YEAR AND DACA ELIGIBILITY")
print("-"*60)
ft_by_year = df_sample.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
ft_by_year.columns = ['Ineligible', 'Eligible']
ft_by_year['Difference'] = ft_by_year['Eligible'] - ft_by_year['Ineligible']
print(ft_by_year.round(4))

#------------------------------------------------------------------------------
# 5. MAIN REGRESSION ANALYSIS: DIFFERENCE-IN-DIFFERENCES
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("SECTION 5: MAIN REGRESSION ANALYSIS (DIFFERENCE-IN-DIFFERENCES)")
print("="*80)

# Drop rows with missing values for regression
reg_vars = ['fulltime', 'employed', 'daca_eligible', 'post', 'treat', 'male', 'AGE',
            'age_sq', 'married', 'educ_hs', 'educ_college', 'years_in_us', 'YEAR', 'STATEFIP']
df_reg = df_sample.dropna(subset=reg_vars).copy()
print(f"\nObservations for regression (after dropping missing): {len(df_reg):,}")

# 5.1 Simple DiD without controls
print("\n" + "-"*60)
print("Model 1: Basic DiD (no controls)")
print("-"*60)

model1 = smf.ols('fulltime ~ daca_eligible + post + treat', data=df_reg).fit(
    cov_type='cluster', cov_kwds={'groups': df_reg['STATEFIP']})
print(model1.summary())

# 5.2 DiD with demographic controls
print("\n" + "-"*60)
print("Model 2: DiD with demographic controls")
print("-"*60)

model2 = smf.ols('fulltime ~ daca_eligible + post + treat + male + AGE + age_sq + married + educ_hs',
                  data=df_reg).fit(cov_type='cluster', cov_kwds={'groups': df_reg['STATEFIP']})
print(model2.summary())

# 5.3 DiD with year fixed effects
print("\n" + "-"*60)
print("Model 3: DiD with year fixed effects")
print("-"*60)

model3 = smf.ols('fulltime ~ daca_eligible + treat + male + AGE + age_sq + married + educ_hs + C(YEAR)',
                  data=df_reg).fit(cov_type='cluster', cov_kwds={'groups': df_reg['STATEFIP']})
print(model3.summary())

# 5.4 DiD with state and year fixed effects
print("\n" + "-"*60)
print("Model 4: DiD with state and year fixed effects")
print("-"*60)

model4 = smf.ols('fulltime ~ daca_eligible + treat + male + AGE + age_sq + married + educ_hs + C(YEAR) + C(STATEFIP)',
                  data=df_reg).fit(cov_type='cluster', cov_kwds={'groups': df_reg['STATEFIP']})
print(model4.summary())

# 5.5 Preferred specification: Full model
print("\n" + "-"*60)
print("Model 5 (PREFERRED): Full model with controls and fixed effects")
print("-"*60)

model5 = smf.ols('fulltime ~ daca_eligible + treat + male + AGE + age_sq + married + educ_hs + educ_college + years_in_us + C(YEAR) + C(STATEFIP)',
                  data=df_reg).fit(cov_type='cluster', cov_kwds={'groups': df_reg['STATEFIP']})
print(model5.summary())

# Extract key results
treat_coef = model5.params['treat']
treat_se = model5.bse['treat']
treat_pval = model5.pvalues['treat']
treat_ci_low = model5.conf_int().loc['treat', 0]
treat_ci_high = model5.conf_int().loc['treat', 1]

print("\n" + "="*60)
print("PREFERRED ESTIMATE SUMMARY")
print("="*60)
print(f"Treatment Effect (DACA eligible x Post):")
print(f"  Coefficient: {treat_coef:.6f}")
print(f"  Standard Error: {treat_se:.6f}")
print(f"  t-statistic: {treat_coef/treat_se:.3f}")
print(f"  p-value: {treat_pval:.6f}")
print(f"  95% CI: [{treat_ci_low:.6f}, {treat_ci_high:.6f}]")
print(f"  Sample Size: {int(model5.nobs):,}")

#------------------------------------------------------------------------------
# 6. ROBUSTNESS CHECKS
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("SECTION 6: ROBUSTNESS CHECKS")
print("="*80)

# 6.1 Alternative outcome: Any employment
print("\n" + "-"*60)
print("Robustness 1: Alternative outcome - Any employment")
print("-"*60)

model_employed = smf.ols('employed ~ daca_eligible + treat + male + AGE + age_sq + married + educ_hs + educ_college + years_in_us + C(YEAR) + C(STATEFIP)',
                          data=df_reg).fit(cov_type='cluster', cov_kwds={'groups': df_reg['STATEFIP']})
print(f"Treatment effect on any employment: {model_employed.params['treat']:.6f} (SE: {model_employed.bse['treat']:.6f})")

# 6.2 Restrict to males only
print("\n" + "-"*60)
print("Robustness 2: Males only")
print("-"*60)

df_males = df_reg[df_reg['male']==1]
model_males = smf.ols('fulltime ~ daca_eligible + treat + AGE + age_sq + married + educ_hs + educ_college + years_in_us + C(YEAR) + C(STATEFIP)',
                       data=df_males).fit(cov_type='cluster', cov_kwds={'groups': df_males['STATEFIP']})
print(f"Treatment effect (males only): {model_males.params['treat']:.6f} (SE: {model_males.bse['treat']:.6f})")
print(f"Sample size: {int(model_males.nobs):,}")

# 6.3 Restrict to females only
print("\n" + "-"*60)
print("Robustness 3: Females only")
print("-"*60)

df_females = df_reg[df_reg['male']==0]
model_females = smf.ols('fulltime ~ daca_eligible + treat + AGE + age_sq + married + educ_hs + educ_college + years_in_us + C(YEAR) + C(STATEFIP)',
                         data=df_females).fit(cov_type='cluster', cov_kwds={'groups': df_females['STATEFIP']})
print(f"Treatment effect (females only): {model_females.params['treat']:.6f} (SE: {model_females.bse['treat']:.6f})")
print(f"Sample size: {int(model_females.nobs):,}")

# 6.4 Alternative age restriction (25-54 prime working age)
print("\n" + "-"*60)
print("Robustness 4: Prime working age (25-54)")
print("-"*60)

df_prime = df_reg[(df_reg['AGE'] >= 25) & (df_reg['AGE'] <= 54)]
model_prime = smf.ols('fulltime ~ daca_eligible + treat + male + AGE + age_sq + married + educ_hs + educ_college + years_in_us + C(YEAR) + C(STATEFIP)',
                       data=df_prime).fit(cov_type='cluster', cov_kwds={'groups': df_prime['STATEFIP']})
print(f"Treatment effect (age 25-54): {model_prime.params['treat']:.6f} (SE: {model_prime.bse['treat']:.6f})")
print(f"Sample size: {int(model_prime.nobs):,}")

# 6.5 Placebo test: Pre-trends (using 2009 as fake treatment year)
print("\n" + "-"*60)
print("Robustness 5: Placebo test (fake 2009 treatment)")
print("-"*60)

df_pre = df_reg[df_reg['YEAR'] <= 2011].copy()
df_pre['fake_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['fake_treat'] = df_pre['daca_eligible'] * df_pre['fake_post']

model_placebo = smf.ols('fulltime ~ daca_eligible + fake_post + fake_treat + male + AGE + age_sq + married + educ_hs + C(YEAR) + C(STATEFIP)',
                         data=df_pre).fit(cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})
print(f"Placebo treatment effect: {model_placebo.params['fake_treat']:.6f} (SE: {model_placebo.bse['fake_treat']:.6f})")
print(f"p-value: {model_placebo.pvalues['fake_treat']:.4f}")

# 6.6 Event study
print("\n" + "-"*60)
print("Robustness 6: Event Study Coefficients")
print("-"*60)

# Create year-specific treatment indicators
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    df_reg[f'treat_{year}'] = ((df_reg['YEAR'] == year) & (df_reg['daca_eligible'] == 1)).astype(int)

# Run event study regression (2011 as reference year)
event_formula = 'fulltime ~ daca_eligible + ' + ' + '.join([f'treat_{y}' for y in years if y != 2011]) + ' + male + AGE + age_sq + married + educ_hs + C(YEAR) + C(STATEFIP)'
model_event = smf.ols(event_formula, data=df_reg).fit(cov_type='cluster', cov_kwds={'groups': df_reg['STATEFIP']})

print("Event Study Coefficients (relative to 2011):")
for year in years:
    if year != 2011:
        coef = model_event.params[f'treat_{year}']
        se = model_event.bse[f'treat_{year}']
        print(f"  {year}: {coef:+.6f} (SE: {se:.6f})")

#------------------------------------------------------------------------------
# 7. CREATE FIGURES
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("SECTION 7: CREATING FIGURES")
print("="*80)

# Figure 1: Full-time employment trends by DACA eligibility
fig1, ax1 = plt.subplots(figsize=(10, 6))
ft_by_year_plot = df_sample.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
ft_by_year_plot.columns = ['Ineligible', 'Eligible']
ft_by_year_plot.plot(ax=ax1, marker='o', linewidth=2)
ax1.axvline(x=2012.5, color='red', linestyle='--', label='DACA Implementation', alpha=0.7)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Trends by DACA Eligibility', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure1_trends.png")

# Figure 2: Event Study Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
event_years = [y for y in years if y != 2011]
event_coefs = [model_event.params[f'treat_{y}'] for y in event_years]
event_ses = [model_event.bse[f'treat_{y}'] for y in event_years]
event_ci_low = [c - 1.96*s for c, s in zip(event_coefs, event_ses)]
event_ci_high = [c + 1.96*s for c, s in zip(event_coefs, event_ses)]

# Add reference year (2011) with 0
plot_years = sorted(event_years + [2011])
plot_coefs = []
plot_ci_low = []
plot_ci_high = []
for y in plot_years:
    if y == 2011:
        plot_coefs.append(0)
        plot_ci_low.append(0)
        plot_ci_high.append(0)
    else:
        idx = event_years.index(y)
        plot_coefs.append(event_coefs[idx])
        plot_ci_low.append(event_ci_low[idx])
        plot_ci_high.append(event_ci_high[idx])

ax2.plot(plot_years, plot_coefs, 'b-o', linewidth=2, markersize=8)
ax2.fill_between(plot_years, plot_ci_low, plot_ci_high, alpha=0.2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axvline(x=2012.5, color='red', linestyle='--', label='DACA Implementation', alpha=0.7)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Treatment Effect (relative to 2011)', fontsize=12)
ax2.set_title('Event Study: DACA Effect on Full-Time Employment', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure2_eventstudy.png")

# Figure 3: Distribution of UHRSWORK
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pre-DACA
pre_data = df_sample[df_sample['post']==0]
axes[0].hist([pre_data[pre_data['daca_eligible']==1]['UHRSWORK'],
              pre_data[pre_data['daca_eligible']==0]['UHRSWORK']],
             bins=range(0, 100, 5), label=['Eligible', 'Ineligible'], alpha=0.7)
axes[0].axvline(x=35, color='red', linestyle='--', label='Full-time threshold')
axes[0].set_xlabel('Usual Hours Worked per Week')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Pre-DACA (2006-2011)')
axes[0].legend()

# Post-DACA
post_data = df_sample[df_sample['post']==1]
axes[1].hist([post_data[post_data['daca_eligible']==1]['UHRSWORK'],
              post_data[post_data['daca_eligible']==0]['UHRSWORK']],
             bins=range(0, 100, 5), label=['Eligible', 'Ineligible'], alpha=0.7)
axes[1].axvline(x=35, color='red', linestyle='--', label='Full-time threshold')
axes[1].set_xlabel('Usual Hours Worked per Week')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Post-DACA (2013-2016)')
axes[1].legend()

plt.tight_layout()
plt.savefig('figure3_hours_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure3_hours_distribution.png")

#------------------------------------------------------------------------------
# 8. SUMMARY TABLES FOR REPORT
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("SECTION 8: SUMMARY TABLES")
print("="*80)

# Table 1: Sample Restrictions
print("\n" + "-"*60)
print("TABLE 1: SAMPLE RESTRICTIONS")
print("-"*60)
print("{:<50} {:>15}".format("Restriction", "N"))
print("-"*65)
for desc, n in sample_tracking:
    print("{:<50} {:>15,}".format(desc, n))

# Table 2: Summary Statistics
print("\n" + "-"*60)
print("TABLE 2: SUMMARY STATISTICS BY DACA ELIGIBILITY")
print("-"*60)
print("{:<20} {:>12} {:>12} {:>12}".format("Variable", "Eligible", "Ineligible", "Difference"))
print("-"*56)
for var in ['fulltime', 'employed', 'AGE', 'male', 'married', 'educ_hs', 'years_in_us']:
    e = df_sample[df_sample['daca_eligible']==1][var].mean()
    i = df_sample[df_sample['daca_eligible']==0][var].mean()
    d = e - i
    print("{:<20} {:>12.4f} {:>12.4f} {:>+12.4f}".format(var, e, i, d))

# Table 3: Main Results
print("\n" + "-"*60)
print("TABLE 3: MAIN DIFFERENCE-IN-DIFFERENCES RESULTS")
print("-"*60)

results_table = pd.DataFrame({
    'Model': ['(1) Basic', '(2) Demographics', '(3) Year FE', '(4) State+Year FE', '(5) Full (Preferred)'],
    'Treatment Effect': [model1.params['treat'], model2.params['treat'], model3.params['treat'],
                         model4.params['treat'], model5.params['treat']],
    'Std. Error': [model1.bse['treat'], model2.bse['treat'], model3.bse['treat'],
                   model4.bse['treat'], model5.bse['treat']],
    'p-value': [model1.pvalues['treat'], model2.pvalues['treat'], model3.pvalues['treat'],
                model4.pvalues['treat'], model5.pvalues['treat']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)]
})
print(results_table.to_string(index=False))

# Table 4: Robustness Checks
print("\n" + "-"*60)
print("TABLE 4: ROBUSTNESS CHECKS")
print("-"*60)

robustness_table = pd.DataFrame({
    'Specification': ['Any Employment', 'Males Only', 'Females Only', 'Age 25-54', 'Placebo (2009)'],
    'Treatment Effect': [model_employed.params['treat'], model_males.params['treat'],
                         model_females.params['treat'], model_prime.params['treat'],
                         model_placebo.params['fake_treat']],
    'Std. Error': [model_employed.bse['treat'], model_males.bse['treat'],
                   model_females.bse['treat'], model_prime.bse['treat'],
                   model_placebo.bse['fake_treat']],
    'p-value': [model_employed.pvalues['treat'], model_males.pvalues['treat'],
                model_females.pvalues['treat'], model_prime.pvalues['treat'],
                model_placebo.pvalues['fake_treat']],
    'N': [int(model_employed.nobs), int(model_males.nobs), int(model_females.nobs),
          int(model_prime.nobs), int(model_placebo.nobs)]
})
print(robustness_table.to_string(index=False))

#------------------------------------------------------------------------------
# 9. SAVE RESULTS FOR LATEX
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("SECTION 9: SAVING RESULTS")
print("="*80)

# Save key statistics
results_dict = {
    'preferred_coef': treat_coef,
    'preferred_se': treat_se,
    'preferred_pval': treat_pval,
    'preferred_ci_low': treat_ci_low,
    'preferred_ci_high': treat_ci_high,
    'sample_size': int(model5.nobs),
    'n_eligible': int(df_sample['daca_eligible'].sum()),
    'n_ineligible': int((df_sample['daca_eligible']==0).sum()),
    'ft_rate_eligible': df_sample[df_sample['daca_eligible']==1]['fulltime'].mean(),
    'ft_rate_ineligible': df_sample[df_sample['daca_eligible']==0]['fulltime'].mean()
}

# Save to file
with open('results_summary.txt', 'w') as f:
    f.write("DACA REPLICATION RESULTS SUMMARY\n")
    f.write("="*50 + "\n\n")
    for key, val in results_dict.items():
        f.write(f"{key}: {val}\n")
    f.write("\n")
    f.write("Main Results Table:\n")
    f.write(results_table.to_string(index=False))
    f.write("\n\nRobustness Table:\n")
    f.write(robustness_table.to_string(index=False))

print("Saved: results_summary.txt")

# Save tables as CSV for LaTeX
results_table.to_csv('table_main_results.csv', index=False)
robustness_table.to_csv('table_robustness.csv', index=False)
print("Saved: table_main_results.csv")
print("Saved: table_robustness.csv")

# Save event study coefficients
event_study_df = pd.DataFrame({
    'Year': plot_years,
    'Coefficient': plot_coefs,
    'CI_Lower': plot_ci_low,
    'CI_Upper': plot_ci_high
})
event_study_df.to_csv('table_eventstudy.csv', index=False)
print("Saved: table_eventstudy.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nPREFERRED ESTIMATE:")
print(f"  Effect of DACA eligibility on full-time employment: {treat_coef:.4f}")
print(f"  Standard Error: {treat_se:.4f}")
print(f"  95% CI: [{treat_ci_low:.4f}, {treat_ci_high:.4f}]")
print(f"  p-value: {treat_pval:.4f}")
print(f"  Sample Size: {int(model5.nobs):,}")

if treat_pval < 0.05:
    print(f"\n  Interpretation: DACA eligibility is associated with a statistically significant")
    print(f"  {treat_coef*100:.2f} percentage point {'increase' if treat_coef > 0 else 'decrease'} in the probability")
    print(f"  of full-time employment among Hispanic-Mexican, Mexican-born non-citizens.")
else:
    print(f"\n  Interpretation: The effect of DACA eligibility on full-time employment is not")
    print(f"  statistically significant at the 5% level.")
