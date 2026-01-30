"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
                   among Hispanic-Mexican Mexican-born non-citizens

Design: Difference-in-Differences
Treatment: Ages 26-30 as of June 15, 2012 (DACA eligible by age)
Control: Ages 31-35 as of June 15, 2012 (ineligible due to age cutoff)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 70)
print("DACA REPLICATION STUDY - DATA ANALYSIS")
print("=" * 70)

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n[STEP 1] Loading data...")

# Read in chunks due to large file size
dtype_spec = {
    'YEAR': 'int32',
    'SAMPLE': 'int64',
    'SERIAL': 'int64',
    'PERNUM': 'int32',
    'PERWT': 'float64',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'UHRSWORK': 'int8',
    'EMPSTAT': 'int8',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'MARST': 'int8',
    'STATEFIP': 'int8',
    'FAMSIZE': 'int8',
    'NCHILD': 'int8',
    'LABFORCE': 'int8',
}

# Columns we need
cols_needed = ['YEAR', 'SAMPLE', 'SERIAL', 'PERNUM', 'PERWT', 'SEX', 'AGE',
               'BIRTHQTR', 'BIRTHYR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD',
               'CITIZEN', 'YRIMMIG', 'UHRSWORK', 'EMPSTAT', 'EDUC', 'EDUCD',
               'MARST', 'STATEFIP', 'FAMSIZE', 'NCHILD', 'LABFORCE']

df = pd.read_csv('data/data.csv', usecols=cols_needed, dtype=dtype_spec)
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: Apply Sample Restrictions
# =============================================================================
print("\n[STEP 2] Applying sample restrictions...")

# 2.1 Hispanic-Mexican ethnicity
# HISPAN == 1 means Mexican
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican filter: {len(df_sample):,}")

# 2.2 Born in Mexico
# BPL == 200 means Mexico
df_sample = df_sample[df_sample['BPL'] == 200]
print(f"After Mexico birthplace filter: {len(df_sample):,}")

# 2.3 Not a citizen (proxy for undocumented)
# CITIZEN == 3 means not a citizen
df_sample = df_sample[df_sample['CITIZEN'] == 3]
print(f"After non-citizen filter: {len(df_sample):,}")

# 2.4 Valid immigration year (needed to check DACA eligibility)
# Exclude 0 (N/A) and any invalid values
df_sample = df_sample[(df_sample['YRIMMIG'] > 0) & (df_sample['YRIMMIG'] < 9999)]
print(f"After valid immigration year filter: {len(df_sample):,}")

# 2.5 Arrived before age 16 (DACA requirement)
df_sample['arrival_age'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['arrival_age'] < 16]
print(f"After arrived before age 16 filter: {len(df_sample):,}")

# 2.6 In US since at least June 2007 (5 years continuous residence by DACA date)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]
print(f"After in US since 2007 or earlier filter: {len(df_sample):,}")

# =============================================================================
# STEP 3: Define Treatment and Control Groups
# =============================================================================
print("\n[STEP 3] Defining treatment and control groups...")

# DACA was announced June 15, 2012
# Treatment: Had not yet had 31st birthday as of June 15, 2012
#            (i.e., born after June 15, 1981)
# Control: Ages 31-35 as of June 15, 2012
#          (i.e., born between June 15, 1977 and June 15, 1981)

# Calculate approximate birth date
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# For June 15, 2012 cutoff:
# - Born Q1-Q2 of a year: by June 15 they've had their birthday that year
# - Born Q3-Q4 of a year: by June 15 they haven't had their birthday yet

# Age as of June 15, 2012
def calc_age_june_2012(birthyr, birthqtr):
    # If born Q1 or Q2 (Jan-Jun), they've had their 2012 birthday by June 15
    # If born Q3 or Q4 (Jul-Dec), they haven't had their 2012 birthday yet
    base_age = 2012 - birthyr
    # Adjust for those born after June
    adjustment = np.where(birthqtr >= 3, -1, 0)
    return base_age + adjustment

df_sample['age_june_2012'] = calc_age_june_2012(
    df_sample['BIRTHYR'].values,
    df_sample['BIRTHQTR'].values
)

# Treatment group: ages 26-30 as of June 15, 2012
# Control group: ages 31-35 as of June 15, 2012
df_sample['treated'] = ((df_sample['age_june_2012'] >= 26) &
                        (df_sample['age_june_2012'] <= 30)).astype(int)
df_sample['control'] = ((df_sample['age_june_2012'] >= 31) &
                        (df_sample['age_june_2012'] <= 35)).astype(int)

# Keep only treatment or control group members
df_analysis = df_sample[(df_sample['treated'] == 1) | (df_sample['control'] == 1)].copy()
print(f"Observations in treatment or control groups: {len(df_analysis):,}")

# =============================================================================
# STEP 4: Define Time Periods and Outcome
# =============================================================================
print("\n[STEP 4] Defining time periods and outcome variable...")

# Exclude 2012 as DACA was implemented mid-year
df_analysis = df_analysis[df_analysis['YEAR'] != 2012]
print(f"After excluding 2012: {len(df_analysis):,}")

# Post-treatment: 2013-2016
# Pre-treatment: 2006-2011
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Full-time employment: usually work 35+ hours per week
# UHRSWORK == 0 means N/A (not working or not in labor force)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# For employed analysis, also create indicator for being employed at all
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)

# =============================================================================
# STEP 5: Sample Summary Statistics
# =============================================================================
print("\n[STEP 5] Sample summary statistics...")

print(f"\nSample by year:")
print(df_analysis.groupby('YEAR').size())

print(f"\nSample by treatment status:")
print(f"Treatment group (ages 26-30): {(df_analysis['treated'] == 1).sum():,}")
print(f"Control group (ages 31-35): {(df_analysis['control'] == 1).sum():,}")

print(f"\nSample by period:")
print(f"Pre-treatment (2006-2011): {(df_analysis['post'] == 0).sum():,}")
print(f"Post-treatment (2013-2016): {(df_analysis['post'] == 1).sum():,}")

# Cross-tabulation
print(f"\nCross-tabulation (Treatment x Post):")
crosstab = pd.crosstab(df_analysis['treated'], df_analysis['post'], margins=True)
crosstab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
crosstab.columns = ['Pre (2006-2011)', 'Post (2013-2016)', 'Total']
print(crosstab)

# =============================================================================
# STEP 6: Descriptive Statistics and Balance Check
# =============================================================================
print("\n[STEP 6] Descriptive statistics...")

# Create education categories
def educ_category(educd):
    if educd <= 61:  # Less than high school
        return 0
    elif educd <= 64:  # High school/GED
        return 1
    elif educd <= 100:  # Some college
        return 2
    else:  # Bachelor's or higher
        return 3

df_analysis['educ_cat'] = df_analysis['EDUCD'].apply(educ_category)

# Create covariate dummies
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)
df_analysis['has_children'] = (df_analysis['NCHILD'] > 0).astype(int)
df_analysis['hs_or_more'] = (df_analysis['educ_cat'] >= 1).astype(int)
df_analysis['college'] = (df_analysis['educ_cat'] >= 2).astype(int)

# Descriptive stats by treatment/control in pre-period
pre_data = df_analysis[df_analysis['post'] == 0]

desc_vars = ['fulltime', 'employed', 'female', 'married', 'has_children',
             'hs_or_more', 'college', 'AGE']

print("\nPre-treatment characteristics by group:")
print("-" * 60)
print(f"{'Variable':<20} {'Treatment':>15} {'Control':>15} {'Diff':>10}")
print("-" * 60)

balance_results = []
for var in desc_vars:
    treat_mean = pre_data[pre_data['treated'] == 1][var].mean()
    ctrl_mean = pre_data[pre_data['control'] == 1][var].mean()
    diff = treat_mean - ctrl_mean
    balance_results.append({
        'Variable': var,
        'Treatment': treat_mean,
        'Control': ctrl_mean,
        'Difference': diff
    })
    print(f"{var:<20} {treat_mean:>15.3f} {ctrl_mean:>15.3f} {diff:>10.3f}")

print("-" * 60)

# =============================================================================
# STEP 7: Main DiD Analysis
# =============================================================================
print("\n[STEP 7] Difference-in-Differences Analysis...")

# Create interaction term
df_analysis['treat_post'] = df_analysis['treated'] * df_analysis['post']

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls('fulltime ~ treated + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
model2 = smf.wls('fulltime ~ treated + post + treat_post + female + married + has_children + hs_or_more + college',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with year fixed effects ---")
df_analysis['year_factor'] = df_analysis['YEAR'].astype('category')
model3 = smf.wls('fulltime ~ treated + treat_post + C(YEAR)',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(f"DiD coefficient (treat_post): {results3.params['treat_post']:.4f}")
print(f"Std. Error: {results3.bse['treat_post']:.4f}")
print(f"t-stat: {results3.tvalues['treat_post']:.4f}")
print(f"p-value: {results3.pvalues['treat_post']:.4f}")
print(f"95% CI: [{results3.conf_int().loc['treat_post', 0]:.4f}, {results3.conf_int().loc['treat_post', 1]:.4f}]")

# Model 4: Full model with year FE and controls
print("\n--- Model 4: DiD with year FE and demographic controls ---")
model4 = smf.wls('fulltime ~ treated + treat_post + female + married + has_children + hs_or_more + college + C(YEAR)',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results4 = model4.fit(cov_type='HC1')
print(f"DiD coefficient (treat_post): {results4.params['treat_post']:.4f}")
print(f"Std. Error: {results4.bse['treat_post']:.4f}")
print(f"t-stat: {results4.tvalues['treat_post']:.4f}")
print(f"p-value: {results4.pvalues['treat_post']:.4f}")
print(f"95% CI: [{results4.conf_int().loc['treat_post', 0]:.4f}, {results4.conf_int().loc['treat_post', 1]:.4f}]")

# Model 5: With state fixed effects
print("\n--- Model 5: DiD with year FE, state FE, and demographic controls ---")
model5 = smf.wls('fulltime ~ treated + treat_post + female + married + has_children + hs_or_more + college + C(YEAR) + C(STATEFIP)',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results5 = model5.fit(cov_type='HC1')
print(f"DiD coefficient (treat_post): {results5.params['treat_post']:.4f}")
print(f"Std. Error: {results5.bse['treat_post']:.4f}")
print(f"t-stat: {results5.tvalues['treat_post']:.4f}")
print(f"p-value: {results5.pvalues['treat_post']:.4f}")
print(f"95% CI: [{results5.conf_int().loc['treat_post', 0]:.4f}, {results5.conf_int().loc['treat_post', 1]:.4f}]")

# =============================================================================
# STEP 8: Alternative Outcome - Employment
# =============================================================================
print("\n[STEP 8] Alternative outcome: Employment (any work)...")

model_emp = smf.wls('employed ~ treated + treat_post + female + married + has_children + hs_or_more + college + C(YEAR) + C(STATEFIP)',
                     data=df_analysis,
                     weights=df_analysis['PERWT'])
results_emp = model_emp.fit(cov_type='HC1')
print(f"DiD coefficient (treat_post): {results_emp.params['treat_post']:.4f}")
print(f"Std. Error: {results_emp.bse['treat_post']:.4f}")
print(f"95% CI: [{results_emp.conf_int().loc['treat_post', 0]:.4f}, {results_emp.conf_int().loc['treat_post', 1]:.4f}]")

# =============================================================================
# STEP 9: Heterogeneity by Sex
# =============================================================================
print("\n[STEP 9] Heterogeneity analysis by sex...")

for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_analysis[df_analysis['SEX'] == sex]
    model_sex = smf.wls('fulltime ~ treated + treat_post + married + has_children + hs_or_more + college + C(YEAR)',
                        data=df_sex,
                        weights=df_sex['PERWT'])
    results_sex = model_sex.fit(cov_type='HC1')
    print(f"\n{label}:")
    print(f"  DiD coefficient: {results_sex.params['treat_post']:.4f}")
    print(f"  Std. Error: {results_sex.bse['treat_post']:.4f}")
    print(f"  95% CI: [{results_sex.conf_int().loc['treat_post', 0]:.4f}, {results_sex.conf_int().loc['treat_post', 1]:.4f}]")
    print(f"  N: {len(df_sex):,}")

# =============================================================================
# STEP 10: Event Study / Pre-trends Check
# =============================================================================
print("\n[STEP 10] Event study / Pre-trends analysis...")

# Create year dummies interacted with treatment
# Using 2011 as the reference year
df_analysis['year_2006'] = (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['year_2007'] = (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['year_2008'] = (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['year_2009'] = (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['year_2010'] = (df_analysis['YEAR'] == 2010).astype(int)
# 2011 is reference
df_analysis['year_2013'] = (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['year_2014'] = (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['year_2015'] = (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['year_2016'] = (df_analysis['YEAR'] == 2016).astype(int)

# Interactions
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'treat_x_{yr}'] = df_analysis['treated'] * df_analysis[f'year_{yr}']

event_formula = ('fulltime ~ treated + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + '
                 'year_2013 + year_2014 + year_2015 + year_2016 + '
                 'treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + '
                 'treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + '
                 'female + married + has_children + hs_or_more + college')

model_event = smf.wls(event_formula, data=df_analysis, weights=df_analysis['PERWT'])
results_event = model_event.fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_coeffs = []
for yr in years:
    coef = results_event.params[f'treat_x_{yr}']
    se = results_event.bse[f'treat_x_{yr}']
    ci_low = results_event.conf_int().loc[f'treat_x_{yr}', 0]
    ci_high = results_event.conf_int().loc[f'treat_x_{yr}', 1]
    pval = results_event.pvalues[f'treat_x_{yr}']
    event_coeffs.append({'Year': yr, 'Coefficient': coef, 'SE': se,
                        'CI_Low': ci_low, 'CI_High': ci_high, 'p-value': pval})
    print(f"  {yr}: {coef:>8.4f} (SE: {se:.4f}), 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# Add reference year
event_coeffs.insert(5, {'Year': 2011, 'Coefficient': 0, 'SE': 0,
                        'CI_Low': 0, 'CI_High': 0, 'p-value': np.nan})

# =============================================================================
# STEP 11: Create Visualizations
# =============================================================================
print("\n[STEP 11] Creating visualizations...")

# Figure 1: Event study plot
fig, ax = plt.subplots(figsize=(10, 6))
years_plot = [d['Year'] for d in event_coeffs]
coeffs_plot = [d['Coefficient'] for d in event_coeffs]
ci_low_plot = [d['CI_Low'] for d in event_coeffs]
ci_high_plot = [d['CI_High'] for d in event_coeffs]

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1, label='DACA Implementation')

ax.plot(years_plot, coeffs_plot, 'o-', color='blue', markersize=8)
ax.fill_between(years_plot, ci_low_plot, ci_high_plot, alpha=0.2, color='blue')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('DiD Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.set_xticks(years_plot)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure_event_study.png")

# Figure 2: Trends by treatment/control
fig, ax = plt.subplots(figsize=(10, 6))

trend_data = df_analysis.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

ax.plot(trend_data.index, trend_data[0], 'o--', color='gray',
        markersize=8, label='Control (Ages 31-35)')
ax.plot(trend_data.index, trend_data[1], 's-', color='blue',
        markersize=8, label='Treatment (Ages 26-30)')

ax.axvline(x=2012, color='red', linestyle='--', linewidth=1, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Group', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.8)

plt.tight_layout()
plt.savefig('figure_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure_trends.png")

# Figure 3: Difference trends
fig, ax = plt.subplots(figsize=(10, 6))

diff_data = trend_data[1] - trend_data[0]
ax.plot(diff_data.index, diff_data.values, 'o-', color='purple', markersize=8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Difference (Treatment - Control)', fontsize=12)
ax.set_title('Difference in Full-Time Employment Rates: Treatment vs Control', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_difference.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure_difference.png")

# =============================================================================
# STEP 12: Summary Table
# =============================================================================
print("\n[STEP 12] Creating summary regression table...")

# Extract key results
summary_table = pd.DataFrame({
    'Model': ['(1) Basic DiD', '(2) + Demographics', '(3) + Year FE',
              '(4) + Year FE + Demo', '(5) + State FE'],
    'DiD Estimate': [
        results1.params['treat_post'],
        results2.params['treat_post'],
        results3.params['treat_post'],
        results4.params['treat_post'],
        results5.params['treat_post']
    ],
    'Std. Error': [
        results1.bse['treat_post'],
        results2.bse['treat_post'],
        results3.bse['treat_post'],
        results4.bse['treat_post'],
        results5.bse['treat_post']
    ],
    'CI Lower': [
        results1.conf_int().loc['treat_post', 0],
        results2.conf_int().loc['treat_post', 0],
        results3.conf_int().loc['treat_post', 0],
        results4.conf_int().loc['treat_post', 0],
        results5.conf_int().loc['treat_post', 0]
    ],
    'CI Upper': [
        results1.conf_int().loc['treat_post', 0],
        results2.conf_int().loc['treat_post', 1],
        results3.conf_int().loc['treat_post', 1],
        results4.conf_int().loc['treat_post', 1],
        results5.conf_int().loc['treat_post', 1]
    ],
    'p-value': [
        results1.pvalues['treat_post'],
        results2.pvalues['treat_post'],
        results3.pvalues['treat_post'],
        results4.pvalues['treat_post'],
        results5.pvalues['treat_post']
    ],
    'N': [
        int(results1.nobs),
        int(results2.nobs),
        int(results3.nobs),
        int(results4.nobs),
        int(results5.nobs)
    ]
})

print("\n" + "=" * 90)
print("SUMMARY OF REGRESSION RESULTS")
print("=" * 90)
print(summary_table.to_string(index=False))
print("=" * 90)

# =============================================================================
# STEP 13: Save Results
# =============================================================================
print("\n[STEP 13] Saving results...")

# Save detailed results to file
with open('analysis_results.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - ANALYSIS RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write("RESEARCH QUESTION\n")
    f.write("-" * 70 + "\n")
    f.write("Effect of DACA eligibility on full-time employment among\n")
    f.write("Hispanic-Mexican Mexican-born non-citizens.\n\n")

    f.write("SAMPLE DESCRIPTION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total observations in analysis: {len(df_analysis):,}\n")
    f.write(f"Treatment group (ages 26-30 as of 6/15/2012): {(df_analysis['treated'] == 1).sum():,}\n")
    f.write(f"Control group (ages 31-35 as of 6/15/2012): {(df_analysis['control'] == 1).sum():,}\n")
    f.write(f"Pre-treatment years: 2006-2011\n")
    f.write(f"Post-treatment years: 2013-2016\n\n")

    f.write("PREFERRED ESTIMATE (Model 5: Full model with state and year FE)\n")
    f.write("-" * 70 + "\n")
    f.write(f"DiD Coefficient: {results5.params['treat_post']:.4f}\n")
    f.write(f"Standard Error: {results5.bse['treat_post']:.4f}\n")
    f.write(f"95% CI: [{results5.conf_int().loc['treat_post', 0]:.4f}, {results5.conf_int().loc['treat_post', 1]:.4f}]\n")
    f.write(f"p-value: {results5.pvalues['treat_post']:.4f}\n")
    f.write(f"Sample size: {int(results5.nobs):,}\n\n")

    f.write("SUMMARY TABLE\n")
    f.write("-" * 70 + "\n")
    f.write(summary_table.to_string(index=False))
    f.write("\n\n")

    f.write("EVENT STUDY COEFFICIENTS\n")
    f.write("-" * 70 + "\n")
    for d in event_coeffs:
        f.write(f"Year {d['Year']}: {d['Coefficient']:.4f} (SE: {d['SE']:.4f})\n")

    f.write("\n\nHETEROGENEITY BY SEX\n")
    f.write("-" * 70 + "\n")
    for sex, label in [(1, 'Male'), (2, 'Female')]:
        df_sex = df_analysis[df_analysis['SEX'] == sex]
        model_sex = smf.wls('fulltime ~ treated + treat_post + married + has_children + hs_or_more + college + C(YEAR)',
                            data=df_sex,
                            weights=df_sex['PERWT'])
        results_sex = model_sex.fit(cov_type='HC1')
        f.write(f"{label}: {results_sex.params['treat_post']:.4f} (SE: {results_sex.bse['treat_post']:.4f}), N={len(df_sex):,}\n")

print("Saved: analysis_results.txt")

# Save event study data for LaTeX
event_df = pd.DataFrame(event_coeffs)
event_df.to_csv('event_study_data.csv', index=False)
print("Saved: event_study_data.csv")

# Save trend data for LaTeX
trend_data.to_csv('trend_data.csv')
print("Saved: trend_data.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nPreferred Estimate (Model 5):")
print(f"  Effect Size: {results5.params['treat_post']:.4f}")
print(f"  Standard Error: {results5.bse['treat_post']:.4f}")
print(f"  95% CI: [{results5.conf_int().loc['treat_post', 0]:.4f}, {results5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  Sample Size: {int(results5.nobs):,}")
print(f"\nInterpretation: DACA eligibility {'increased' if results5.params['treat_post'] > 0 else 'decreased'}")
print(f"full-time employment by {abs(results5.params['treat_post'])*100:.2f} percentage points.")
print("=" * 70)
