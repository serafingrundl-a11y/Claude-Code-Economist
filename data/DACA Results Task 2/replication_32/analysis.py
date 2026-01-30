"""
DACA Replication Study - Analysis Script
Replication 32

Research Question: Causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the United States.

Treatment: Ages 26-30 as of June 15, 2012
Control: Ages 31-35 as of June 15, 2012
Outcome: Full-time employment (working 35+ hours/week)
Design: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
print("\n[1] Loading data...")

# Read the data - only columns we need to reduce memory
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
               'EMPSTAT', 'UHRSWORK', 'EDUC', 'MARST']

df = pd.read_csv('data/data.csv', usecols=cols_needed, low_memory=False)
print(f"Total observations loaded: {len(df):,}")

# -----------------------------------------------------------------------------
# 2. SAMPLE SELECTION
# -----------------------------------------------------------------------------
print("\n[2] Applying sample selection criteria...")

# Step 2a: Hispanic-Mexican ethnicity (HISPAN = 1)
df_hisp = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican filter: {len(df_hisp):,}")

# Step 2b: Born in Mexico (BPL = 200)
df_mex = df_hisp[df_hisp['BPL'] == 200].copy()
print(f"After Mexico birthplace filter: {len(df_mex):,}")

# Step 2c: Non-citizen (CITIZEN = 3) - proxy for undocumented
df_noncit = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"After non-citizen filter: {len(df_noncit):,}")

# Step 2d: Calculate age as of June 15, 2012
# DACA was announced June 15, 2012
# Someone born in 1982 would be 30 in 2012
# Someone born in 1986 would be 26 in 2012
# Someone born in 1977 would be 35 in 2012
# Someone born in 1981 would be 31 in 2012

# Age as of June 15, 2012 = 2012 - BIRTHYR, adjusting for birth quarter
# Q1 (Jan-Mar): by June 15, already had birthday
# Q2 (Apr-Jun): may or may not have had birthday (conservative: count as had)
# Q3 (Jul-Sep): have not had birthday yet
# Q4 (Oct-Dec): have not had birthday yet

def calc_age_june2012(row):
    """Calculate age as of June 15, 2012"""
    base_age = 2012 - row['BIRTHYR']
    # If born in Q3 or Q4, subtract 1 (haven't had birthday by June 15)
    if row['BIRTHQTR'] in [3, 4]:
        return base_age - 1
    return base_age

df_noncit['age_june2012'] = df_noncit.apply(calc_age_june2012, axis=1)

# Step 2e: Select treatment (ages 26-30) and control (ages 31-35) groups
df_sample = df_noncit[(df_noncit['age_june2012'] >= 26) &
                       (df_noncit['age_june2012'] <= 35)].copy()
print(f"After age 26-35 filter: {len(df_sample):,}")

# Step 2f: Arrived before age 16 (DACA requirement)
# Age at immigration = YRIMMIG - BIRTHYR
# Need to handle missing YRIMMIG (0 = N/A)
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()  # Remove missing immigration year
df_sample['age_at_immigration'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_immigration'] < 16].copy()
print(f"After arrived before age 16 filter: {len(df_sample):,}")

# Step 2g: Continuous residence since June 15, 2007 (YRIMMIG <= 2007)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
print(f"After continuous residence since 2007 filter: {len(df_sample):,}")

# Step 2h: Exclude 2012 (ambiguous pre/post)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# Step 2i: Keep years 2006-2016
df_sample = df_sample[(df_sample['YEAR'] >= 2006) & (df_sample['YEAR'] <= 2016)].copy()
print(f"Final sample (2006-2011, 2013-2016): {len(df_sample):,}")

# -----------------------------------------------------------------------------
# 3. CREATE ANALYSIS VARIABLES
# -----------------------------------------------------------------------------
print("\n[3] Creating analysis variables...")

# Treatment indicator: 1 if age 26-30 at June 15, 2012
df_sample['treat'] = (df_sample['age_june2012'] <= 30).astype(int)

# Post-treatment indicator: 1 if year >= 2013
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Interaction term (DiD estimator)
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

# Outcome: Full-time employment (35+ hours per week)
# UHRSWORK = 0 means N/A (not working), so treat as not full-time
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Additional covariates
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Education categories
df_sample['educ_lesshs'] = (df_sample['EDUC'] < 6).astype(int)
df_sample['educ_hs'] = (df_sample['EDUC'] == 6).astype(int)
df_sample['educ_somecol'] = (df_sample['EDUC'].isin([7, 8, 9])).astype(int)
df_sample['educ_college'] = (df_sample['EDUC'] >= 10).astype(int)

print(f"Treatment group (ages 26-30): {df_sample['treat'].sum():,}")
print(f"Control group (ages 31-35): {(1 - df_sample['treat']).sum():,}")
print(f"Pre-treatment observations: {(1 - df_sample['post']).sum():,}")
print(f"Post-treatment observations: {df_sample['post'].sum():,}")

# -----------------------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------------
print("\n[4] Generating descriptive statistics...")

# Table 1: Sample sizes by group and period
print("\n--- Table 1: Sample Distribution ---")
cross_tab = pd.crosstab(df_sample['treat'], df_sample['post'],
                        margins=True, margins_name='Total')
cross_tab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
cross_tab.columns = ['Pre (2006-2011)', 'Post (2013-2016)', 'Total']
print(cross_tab)

# Weighted sample sizes
print("\n--- Weighted Sample Sizes ---")
for t in [0, 1]:
    for p in [0, 1]:
        subset = df_sample[(df_sample['treat'] == t) & (df_sample['post'] == p)]
        group = 'Treatment' if t == 1 else 'Control'
        period = 'Post' if p == 1 else 'Pre'
        print(f"{group}, {period}: N={len(subset):,}, Weighted N={subset['PERWT'].sum():,.0f}")

# Full-time employment rates by group and period
print("\n--- Table 2: Full-time Employment Rates ---")
ft_rates = df_sample.groupby(['treat', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(ft_rates.round(4))

# Calculate raw DiD
pre_treat = ft_rates.loc['Treatment (26-30)', 'Pre (2006-2011)']
post_treat = ft_rates.loc['Treatment (26-30)', 'Post (2013-2016)']
pre_control = ft_rates.loc['Control (31-35)', 'Pre (2006-2011)']
post_control = ft_rates.loc['Control (31-35)', 'Post (2013-2016)']

diff_treat = post_treat - pre_treat
diff_control = post_control - pre_control
raw_did = diff_treat - diff_control

print(f"\nRaw DiD Calculation:")
print(f"  Treatment change: {post_treat:.4f} - {pre_treat:.4f} = {diff_treat:.4f}")
print(f"  Control change:   {post_control:.4f} - {pre_control:.4f} = {diff_control:.4f}")
print(f"  DiD estimate:     {diff_treat:.4f} - {diff_control:.4f} = {raw_did:.4f}")

# Table 3: Balance table (pre-treatment means)
print("\n--- Table 3: Balance Table (Pre-treatment Period) ---")
pre_data = df_sample[df_sample['post'] == 0]

balance_vars = ['AGE', 'female', 'married', 'educ_lesshs', 'educ_hs',
                'educ_somecol', 'educ_college', 'fulltime']
balance_labels = ['Age (survey year)', 'Female', 'Married', 'Less than HS',
                  'High School', 'Some College', 'College+', 'Full-time Employed']

balance_results = []
for var, label in zip(balance_vars, balance_labels):
    treat_mean = np.average(pre_data[pre_data['treat'] == 1][var],
                           weights=pre_data[pre_data['treat'] == 1]['PERWT'])
    control_mean = np.average(pre_data[pre_data['treat'] == 0][var],
                             weights=pre_data[pre_data['treat'] == 0]['PERWT'])
    diff = treat_mean - control_mean
    balance_results.append({
        'Variable': label,
        'Treatment': treat_mean,
        'Control': control_mean,
        'Difference': diff
    })

balance_df = pd.DataFrame(balance_results)
print(balance_df.to_string(index=False))

# -----------------------------------------------------------------------------
# 5. TRENDS OVER TIME
# -----------------------------------------------------------------------------
print("\n[5] Analyzing trends over time...")

# Full-time employment by year and group
yearly_ft = df_sample.groupby(['YEAR', 'treat']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
yearly_ft.columns = ['Control (31-35)', 'Treatment (26-30)']

print("\n--- Full-time Employment Rate by Year ---")
print(yearly_ft.round(4))

# Save for plotting
yearly_ft.to_csv('yearly_fulltime_rates.csv')

# Create trends plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(yearly_ft.index, yearly_ft['Treatment (26-30)'], 'b-o',
        label='Treatment (26-30)', linewidth=2, markersize=8)
ax.plot(yearly_ft.index, yearly_ft['Control (31-35)'], 'r--s',
        label='Control (31-35)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA (2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-time Employment Rate', fontsize=12)
ax.set_title('Full-time Employment Trends by Group', fontsize=14)
ax.legend(loc='best', fontsize=10)
ax.set_xticks(yearly_ft.index)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved trends figure to 'figure_trends.png'")

# -----------------------------------------------------------------------------
# 6. DIFFERENCE-IN-DIFFERENCES REGRESSION
# -----------------------------------------------------------------------------
print("\n[6] Running Difference-in-Differences regressions...")

# Model 1: Basic DiD (no covariates)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                 data=df_sample, weights=df_sample['PERWT']).fit(
                     cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographics ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + '
                 'educ_hs + educ_somecol + educ_college',
                 data=df_sample, weights=df_sample['PERWT']).fit(
                     cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(model2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
df_sample['year_factor'] = df_sample['YEAR'].astype(str)
model3 = smf.wls('fulltime ~ treat + treat_post + female + married + '
                 'educ_hs + educ_somecol + educ_college + C(year_factor)',
                 data=df_sample, weights=df_sample['PERWT']).fit(
                     cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

# Print only key coefficients
print(f"\nKey Coefficients (Model 3):")
print(f"  treat_post (DiD): {model3.params['treat_post']:.4f} "
      f"(SE: {model3.bse['treat_post']:.4f}, p: {model3.pvalues['treat_post']:.4f})")
print(f"  treat:            {model3.params['treat']:.4f} "
      f"(SE: {model3.bse['treat']:.4f})")

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with Year and State Fixed Effects ---")
model4 = smf.wls('fulltime ~ treat + treat_post + female + married + '
                 'educ_hs + educ_somecol + educ_college + C(year_factor) + C(STATEFIP)',
                 data=df_sample, weights=df_sample['PERWT']).fit(
                     cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

print(f"\nKey Coefficients (Model 4):")
print(f"  treat_post (DiD): {model4.params['treat_post']:.4f} "
      f"(SE: {model4.bse['treat_post']:.4f}, p: {model4.pvalues['treat_post']:.4f})")

# -----------------------------------------------------------------------------
# 7. EVENT STUDY / DYNAMIC EFFECTS
# -----------------------------------------------------------------------------
print("\n[7] Event Study Analysis...")

# Create year-specific treatment effects
df_sample['treat_2006'] = df_sample['treat'] * (df_sample['YEAR'] == 2006).astype(int)
df_sample['treat_2007'] = df_sample['treat'] * (df_sample['YEAR'] == 2007).astype(int)
df_sample['treat_2008'] = df_sample['treat'] * (df_sample['YEAR'] == 2008).astype(int)
df_sample['treat_2009'] = df_sample['treat'] * (df_sample['YEAR'] == 2009).astype(int)
df_sample['treat_2010'] = df_sample['treat'] * (df_sample['YEAR'] == 2010).astype(int)
# 2011 is reference year
df_sample['treat_2013'] = df_sample['treat'] * (df_sample['YEAR'] == 2013).astype(int)
df_sample['treat_2014'] = df_sample['treat'] * (df_sample['YEAR'] == 2014).astype(int)
df_sample['treat_2015'] = df_sample['treat'] * (df_sample['YEAR'] == 2015).astype(int)
df_sample['treat_2016'] = df_sample['treat'] * (df_sample['YEAR'] == 2016).astype(int)

event_model = smf.wls('fulltime ~ treat + treat_2006 + treat_2007 + treat_2008 + '
                      'treat_2009 + treat_2010 + treat_2013 + treat_2014 + '
                      'treat_2015 + treat_2016 + female + married + '
                      'educ_hs + educ_somecol + educ_college + C(year_factor)',
                      data=df_sample, weights=df_sample['PERWT']).fit(
                          cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

# Extract event study coefficients
event_years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs = [event_model.params.get(f'treat_{y}', 0) for y in event_years]
event_ses = [event_model.bse.get(f'treat_{y}', 0) for y in event_years]
# 2011 is reference (coefficient = 0)
event_coefs[5] = 0
event_ses[5] = 0

print("\n--- Event Study Coefficients ---")
print("Year     Coefficient    SE          95% CI")
for i, year in enumerate(event_years):
    ci_low = event_coefs[i] - 1.96 * event_ses[i]
    ci_high = event_coefs[i] + 1.96 * event_ses[i]
    print(f"{year}     {event_coefs[i]:8.4f}     {event_ses[i]:8.4f}    [{ci_low:7.4f}, {ci_high:7.4f}]")

# Event study plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(event_years, event_coefs, yerr=[1.96*se for se in event_ses],
            fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8, color='blue')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA (2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Effect on Full-time Employment', fontsize=12)
ax.set_title('Event Study: Treatment Effect by Year (Reference: 2011)', fontsize=14)
ax.set_xticks(event_years)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved event study figure to 'figure_event_study.png'")

# -----------------------------------------------------------------------------
# 8. ROBUSTNESS CHECKS
# -----------------------------------------------------------------------------
print("\n[8] Robustness Checks...")

# Robustness 1: Narrower age bandwidth (27-29 vs 32-34)
print("\n--- Robustness 1: Narrower Age Bandwidth (27-29 vs 32-34) ---")
df_narrow = df_sample[(df_sample['age_june2012'] >= 27) & (df_sample['age_june2012'] <= 34) &
                      ~((df_sample['age_june2012'] >= 30) & (df_sample['age_june2012'] <= 31))].copy()
df_narrow['treat_narrow'] = (df_narrow['age_june2012'] <= 29).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treat_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treat_narrow + post + treat_post_narrow + female + married + '
                       'educ_hs + educ_somecol + educ_college',
                       data=df_narrow, weights=df_narrow['PERWT']).fit(
                           cov_type='cluster', cov_kwds={'groups': df_narrow['STATEFIP']})
print(f"DiD (narrow): {model_narrow.params['treat_post_narrow']:.4f} "
      f"(SE: {model_narrow.bse['treat_post_narrow']:.4f})")

# Robustness 2: Placebo test - use only pre-treatment period
print("\n--- Robustness 2: Placebo Test (Pre-period only: 2006-2008 vs 2009-2011) ---")
df_placebo = df_sample[df_sample['post'] == 0].copy()
df_placebo['placebo_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treat_placebo_post'] = df_placebo['treat'] * df_placebo['placebo_post']

model_placebo = smf.wls('fulltime ~ treat + placebo_post + treat_placebo_post + female + married + '
                        'educ_hs + educ_somecol + educ_college',
                        data=df_placebo, weights=df_placebo['PERWT']).fit(
                            cov_type='cluster', cov_kwds={'groups': df_placebo['STATEFIP']})
print(f"Placebo DiD: {model_placebo.params['treat_placebo_post']:.4f} "
      f"(SE: {model_placebo.bse['treat_placebo_post']:.4f}, p: {model_placebo.pvalues['treat_placebo_post']:.4f})")

# Robustness 3: By gender
print("\n--- Robustness 3: Heterogeneity by Gender ---")
for gender, label in [(0, 'Male'), (1, 'Female')]:
    df_gender = df_sample[df_sample['female'] == gender]
    model_gender = smf.wls('fulltime ~ treat + post + treat_post + married + '
                           'educ_hs + educ_somecol + educ_college',
                           data=df_gender, weights=df_gender['PERWT']).fit(
                               cov_type='cluster', cov_kwds={'groups': df_gender['STATEFIP']})
    print(f"  {label}: DiD = {model_gender.params['treat_post']:.4f} "
          f"(SE: {model_gender.bse['treat_post']:.4f})")

# -----------------------------------------------------------------------------
# 9. SUMMARY OF RESULTS
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print(f"""
Sample: Hispanic-Mexican, Mexico-born, non-citizen individuals
        who arrived before age 16 and have been in US since 2007

Treatment group: Ages 26-30 as of June 15, 2012 (N = {df_sample['treat'].sum():,})
Control group:   Ages 31-35 as of June 15, 2012 (N = {(1-df_sample['treat']).sum():,})

Pre-treatment period:  2006-2011 (excluding 2012)
Post-treatment period: 2013-2016

Outcome: Full-time employment (working 35+ hours/week)

MAIN RESULTS (Preferred Specification: Model 2 - DiD with Demographics)
-----------------------------------------------------------------------
DiD Estimate:    {model2.params['treat_post']:.4f}
Standard Error:  {model2.bse['treat_post']:.4f}
95% CI:          [{model2.params['treat_post'] - 1.96*model2.bse['treat_post']:.4f}, {model2.params['treat_post'] + 1.96*model2.bse['treat_post']:.4f}]
P-value:         {model2.pvalues['treat_post']:.4f}
Sample Size:     {len(df_sample):,}

Interpretation: DACA eligibility is associated with a {model2.params['treat_post']:.1%} {'increase' if model2.params['treat_post'] > 0 else 'decrease'}
in the probability of full-time employment among eligible individuals,
compared to slightly older individuals who were otherwise similar but
ineligible due to their age.
""")

# Save key results for report
results_summary = {
    'main_estimate': model2.params['treat_post'],
    'main_se': model2.bse['treat_post'],
    'main_pvalue': model2.pvalues['treat_post'],
    'main_ci_low': model2.params['treat_post'] - 1.96*model2.bse['treat_post'],
    'main_ci_high': model2.params['treat_post'] + 1.96*model2.bse['treat_post'],
    'sample_size': len(df_sample),
    'n_treatment': int(df_sample['treat'].sum()),
    'n_control': int((1-df_sample['treat']).sum()),
    'model1_estimate': model1.params['treat_post'],
    'model1_se': model1.bse['treat_post'],
    'model3_estimate': model3.params['treat_post'],
    'model3_se': model3.bse['treat_post'],
    'model4_estimate': model4.params['treat_post'],
    'model4_se': model4.bse['treat_post'],
    'raw_did': raw_did,
    'ft_treat_pre': pre_treat,
    'ft_treat_post': post_treat,
    'ft_control_pre': pre_control,
    'ft_control_post': post_control,
}

# Save results to file
import json
with open('results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print("\nResults saved to 'results_summary.json'")

# Save regression tables
with open('regression_tables.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - REGRESSION RESULTS\n")
    f.write("=" * 80 + "\n\n")

    f.write("Model 1: Basic DiD (no covariates)\n")
    f.write("-" * 40 + "\n")
    f.write(model1.summary().as_text() + "\n\n")

    f.write("Model 2: DiD with Demographic Controls (PREFERRED)\n")
    f.write("-" * 40 + "\n")
    f.write(model2.summary().as_text() + "\n\n")

    f.write("Model 3: DiD with Year Fixed Effects\n")
    f.write("-" * 40 + "\n")
    f.write(model3.summary().as_text() + "\n\n")

    f.write("Model 4: DiD with Year and State Fixed Effects\n")
    f.write("-" * 40 + "\n")
    f.write(model4.summary().as_text() + "\n\n")

print("Regression tables saved to 'regression_tables.txt'")
print("\n[ANALYSIS COMPLETE]")
