"""
DACA Replication Study - Analysis Script
Replication 96

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals.

Design: Difference-in-Differences
- Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
- Control: Ages 31-35 at DACA implementation

Memory-efficient version: reads data in chunks
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DACA REPLICATION STUDY - REPLICATION 96")
print("="*70)

# =============================================================================
# STEP 1: LOAD DATA IN CHUNKS AND FILTER IMMEDIATELY
# =============================================================================
print("\n[1] Loading and filtering ACS data (chunked for memory efficiency)...")

# Define columns we need
usecols = ['YEAR', 'SAMPLE', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
           'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
           'EDUC', 'EMPSTAT', 'UHRSWORK', 'STATEFIP']

chunks = []
total_rows = 0

for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=500000):
    total_rows += len(chunk)

    # Criterion 1: Hispanic-Mexican ethnicity
    chunk['is_hispanic_mexican'] = (chunk['HISPAN'] == 1) | (chunk['HISPAND'].isin(range(100, 108)))

    # Criterion 2: Born in Mexico
    chunk['born_mexico'] = (chunk['BPL'] == 200) | (chunk['BPLD'] == 20000)

    # Criterion 3: Not a citizen
    chunk['not_citizen'] = chunk['CITIZEN'] == 3

    # Criterion 4: Arrived before age 16
    chunk['age_at_arrival'] = chunk['YRIMMIG'] - chunk['BIRTHYR']
    chunk['arrived_before_16'] = (chunk['age_at_arrival'] < 16) & (chunk['age_at_arrival'] >= 0)

    # Criterion 5: In US since 2007
    chunk['in_us_since_2007'] = chunk['YRIMMIG'] <= 2007

    # Base eligibility
    chunk['base_eligible'] = (chunk['is_hispanic_mexican'] &
                              chunk['born_mexico'] &
                              chunk['not_citizen'] &
                              chunk['arrived_before_16'] &
                              chunk['in_us_since_2007'])

    # Age as of June 15, 2012
    def calc_age(row):
        if pd.isna(row['BIRTHYR']) or pd.isna(row['BIRTHQTR']):
            return np.nan
        if row['BIRTHQTR'] in [1, 2]:
            return 2012 - row['BIRTHYR']
        else:
            return 2012 - row['BIRTHYR'] - 1

    chunk['age_june_2012'] = chunk.apply(calc_age, axis=1)

    # Treatment and control groups
    chunk['treatment'] = (chunk['age_june_2012'] >= 26) & (chunk['age_june_2012'] <= 30)
    chunk['control'] = (chunk['age_june_2012'] >= 31) & (chunk['age_june_2012'] <= 35)
    chunk['in_sample'] = chunk['base_eligible'] & (chunk['treatment'] | chunk['control'])

    # Filter to eligible sample and exclude 2012
    filtered = chunk[(chunk['in_sample']) & (chunk['YEAR'] != 2012)].copy()

    if len(filtered) > 0:
        chunks.append(filtered)

    print(f"    Processed {total_rows:,} rows, kept {sum(len(c) for c in chunks):,} eligible")

print(f"\n    Total rows processed: {total_rows:,}")

# Combine all filtered chunks
sample = pd.concat(chunks, ignore_index=True)
print(f"    Final sample size: {len(sample):,}")

# =============================================================================
# STEP 2: CREATE ANALYSIS VARIABLES
# =============================================================================
print("\n[2] Creating analysis variables...")

# Outcome: Full-time employment (35+ hours/week)
sample['fulltime'] = (sample['UHRSWORK'] >= 35).astype(int)

# Alternative: Employed (any hours)
sample['employed'] = (sample['EMPSTAT'] == 1).astype(int)

# Post-treatment indicator (2013-2016)
sample['post'] = (sample['YEAR'] >= 2013).astype(int)

# Treatment indicator
sample['treat'] = sample['treatment'].astype(int)

# Interaction term for DiD
sample['treat_post'] = sample['treat'] * sample['post']

# Covariates
sample['female'] = (sample['SEX'] == 2).astype(int)
sample['married'] = (sample['MARST'].isin([1, 2])).astype(int)
sample['educ_hs'] = (sample['EDUC'] >= 6).astype(int)

# Year dummies
for yr in range(2006, 2017):
    if yr != 2012:
        sample[f'year_{yr}'] = (sample['YEAR'] == yr).astype(int)

print(f"    Treatment group obs: {sample['treat'].sum():,}")
print(f"    Control group obs: {(1-sample['treat']).sum():,}")
print(f"    Pre-period obs: {(1-sample['post']).sum():,}")
print(f"    Post-period obs: {sample['post'].sum():,}")

# =============================================================================
# STEP 3: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[3] Descriptive Statistics")
print("-"*70)

print("\nWeighted means by group:")
for treat in [0, 1]:
    for post in [0, 1]:
        subset = sample[(sample['treat']==treat) & (sample['post']==post)]
        wt_fulltime = np.average(subset['fulltime'], weights=subset['PERWT'])
        wt_employed = np.average(subset['employed'], weights=subset['PERWT'])
        wt_female = np.average(subset['female'], weights=subset['PERWT'])
        wt_married = np.average(subset['married'], weights=subset['PERWT'])
        wt_hs = np.average(subset['educ_hs'], weights=subset['PERWT'])
        group = "Treatment" if treat else "Control"
        period = "Post" if post else "Pre"
        n = len(subset)
        print(f"  {group:10} | {period:4} | N={n:6,} | FT={wt_fulltime:.4f} | Emp={wt_employed:.4f} | Fem={wt_female:.4f} | Mar={wt_married:.4f} | HS={wt_hs:.4f}")

# =============================================================================
# STEP 4: DIFFERENCE-IN-DIFFERENCES ESTIMATION
# =============================================================================
print("\n\n[4] Difference-in-Differences Estimation")
print("="*70)

# Simple 2x2 DiD calculation
print("\n--- Simple 2x2 DiD (weighted means) ---")

def weighted_mean(data, weights):
    return np.average(data, weights=weights)

treat_pre = sample[(sample['treat']==1) & (sample['post']==0)]
treat_post = sample[(sample['treat']==1) & (sample['post']==1)]
control_pre = sample[(sample['treat']==0) & (sample['post']==0)]
control_post = sample[(sample['treat']==0) & (sample['post']==1)]

y_treat_pre = weighted_mean(treat_pre['fulltime'], treat_pre['PERWT'])
y_treat_post = weighted_mean(treat_post['fulltime'], treat_post['PERWT'])
y_control_pre = weighted_mean(control_pre['fulltime'], control_pre['PERWT'])
y_control_post = weighted_mean(control_post['fulltime'], control_post['PERWT'])

did_simple = (y_treat_post - y_treat_pre) - (y_control_post - y_control_pre)

print(f"\n  Treatment Pre:   {y_treat_pre:.4f}")
print(f"  Treatment Post:  {y_treat_post:.4f}")
print(f"  Treatment Diff:  {y_treat_post - y_treat_pre:.4f}")
print(f"\n  Control Pre:     {y_control_pre:.4f}")
print(f"  Control Post:    {y_control_post:.4f}")
print(f"  Control Diff:    {y_control_post - y_control_pre:.4f}")
print(f"\n  DiD Estimate:    {did_simple:.4f}")

# =============================================================================
# STEP 5: REGRESSION-BASED DiD
# =============================================================================
print("\n\n--- Regression-Based DiD ---")

# Model 1: Basic DiD (unweighted)
print("\nModel 1: Basic DiD (unweighted)")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=sample).fit(cov_type='HC1')
print(f"  DiD coefficient (treat_post): {model1.params['treat_post']:.4f}")
print(f"  Std. Error: {model1.bse['treat_post']:.4f}")
print(f"  95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  P-value: {model1.pvalues['treat_post']:.4f}")
print(f"  N: {int(model1.nobs):,}")

# Model 2: Basic DiD (weighted)
print("\nModel 2: Basic DiD (weighted)")
model2 = smf.wls('fulltime ~ treat + post + treat_post', data=sample, weights=sample['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient (treat_post): {model2.params['treat_post']:.4f}")
print(f"  Std. Error: {model2.bse['treat_post']:.4f}")
print(f"  95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  P-value: {model2.pvalues['treat_post']:.4f}")

# Model 3: DiD with year fixed effects (unweighted)
print("\nModel 3: DiD with Year Fixed Effects (unweighted)")
model3 = smf.ols('fulltime ~ treat + treat_post + C(YEAR)', data=sample).fit(cov_type='HC1')
print(f"  DiD coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"  Std. Error: {model3.bse['treat_post']:.4f}")
print(f"  95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  P-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: DiD with year fixed effects (weighted) - PREFERRED
print("\nModel 4: DiD with Year Fixed Effects (weighted) - PREFERRED")
model4 = smf.wls('fulltime ~ treat + treat_post + C(YEAR)', data=sample, weights=sample['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"  Std. Error: {model4.bse['treat_post']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  P-value: {model4.pvalues['treat_post']:.4f}")

# Model 5: DiD with covariates
print("\nModel 5: DiD with Year FE and Covariates (weighted)")
model5 = smf.wls('fulltime ~ treat + treat_post + C(YEAR) + female + married + educ_hs',
                  data=sample, weights=sample['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient (treat_post): {model5.params['treat_post']:.4f}")
print(f"  Std. Error: {model5.bse['treat_post']:.4f}")
print(f"  95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  P-value: {model5.pvalues['treat_post']:.4f}")

# =============================================================================
# STEP 6: EVENT STUDY
# =============================================================================
print("\n\n[5] Event Study Analysis (Pre-Trends Check)")
print("="*70)

# Create year-specific treatment effects (relative to 2011)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    sample[f'treat_{yr}'] = sample['treat'] * sample[f'year_{yr}']

event_formula = ('fulltime ~ treat + treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 + '
                 'treat_2013 + treat_2014 + treat_2015 + treat_2016 + C(YEAR)')
event_model = smf.wls(event_formula, data=sample, weights=sample['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-"*50)
event_years = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
               'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
for var in event_years:
    coef = event_model.params[var]
    se = event_model.bse[var]
    pval = event_model.pvalues[var]
    ci = event_model.conf_int().loc[var]
    year = var.split('_')[1]
    sig = "*" if pval < 0.05 else ("+" if pval < 0.10 else " ")
    print(f"  {year}: {coef:8.4f} ({se:.4f}) [{ci[0]:7.4f}, {ci[1]:7.4f}] {sig}")

# =============================================================================
# STEP 7: ROBUSTNESS CHECKS
# =============================================================================
print("\n\n[6] Robustness Checks")
print("="*70)

# Robustness 1: Alternative outcome - any employment
print("\n--- Robustness 1: Any Employment (vs Full-time) ---")
rob1 = smf.wls('employed ~ treat + treat_post + C(YEAR)', data=sample, weights=sample['PERWT']).fit(cov_type='HC1')
print(f"  DiD (any employment): {rob1.params['treat_post']:.4f} (SE: {rob1.bse['treat_post']:.4f})")
print(f"  95% CI: [{rob1.conf_int().loc['treat_post', 0]:.4f}, {rob1.conf_int().loc['treat_post', 1]:.4f}]")

# Robustness 2: Different age bandwidths
print("\n--- Robustness 2: Narrower Age Bandwidth (28-30 vs 31-33) ---")
sample_narrow = sample[(sample['age_june_2012'].between(28,30)) | (sample['age_june_2012'].between(31,33))].copy()
sample_narrow['treat'] = (sample_narrow['age_june_2012'].between(28,30)).astype(int)
sample_narrow['treat_post'] = sample_narrow['treat'] * sample_narrow['post']
rob2 = smf.wls('fulltime ~ treat + treat_post + C(YEAR)', data=sample_narrow, weights=sample_narrow['PERWT']).fit(cov_type='HC1')
print(f"  DiD (narrow bandwidth): {rob2.params['treat_post']:.4f} (SE: {rob2.bse['treat_post']:.4f}), N={int(rob2.nobs):,}")

# Robustness 3: By sex
print("\n--- Robustness 3: By Sex ---")
sample_male = sample[sample['SEX'] == 1]
sample_female = sample[sample['SEX'] == 2]
rob3m = smf.wls('fulltime ~ treat + treat_post + C(YEAR)', data=sample_male, weights=sample_male['PERWT']).fit(cov_type='HC1')
rob3f = smf.wls('fulltime ~ treat + treat_post + C(YEAR)', data=sample_female, weights=sample_female['PERWT']).fit(cov_type='HC1')
print(f"  DiD (males):   {rob3m.params['treat_post']:.4f} (SE: {rob3m.bse['treat_post']:.4f}), N={int(rob3m.nobs):,}")
print(f"  DiD (females): {rob3f.params['treat_post']:.4f} (SE: {rob3f.bse['treat_post']:.4f}), N={int(rob3f.nobs):,}")

# Robustness 4: Placebo test
print("\n--- Robustness 4: Placebo Test (Pre-DACA Only: 2006-2008 vs 2009-2011) ---")
sample_placebo = sample[sample['YEAR'] <= 2011].copy()
sample_placebo['post_placebo'] = (sample_placebo['YEAR'] >= 2009).astype(int)
sample_placebo['treat_post_placebo'] = sample_placebo['treat'] * sample_placebo['post_placebo']
rob4 = smf.wls('fulltime ~ treat + treat_post_placebo + C(YEAR)', data=sample_placebo, weights=sample_placebo['PERWT']).fit(cov_type='HC1')
print(f"  Placebo DiD: {rob4.params['treat_post_placebo']:.4f} (SE: {rob4.bse['treat_post_placebo']:.4f})")
print(f"  95% CI: [{rob4.conf_int().loc['treat_post_placebo', 0]:.4f}, {rob4.conf_int().loc['treat_post_placebo', 1]:.4f}]")

# Robustness 5: State fixed effects
print("\n--- Robustness 5: With State Fixed Effects ---")
rob5 = smf.wls('fulltime ~ treat + treat_post + C(YEAR) + C(STATEFIP)', data=sample, weights=sample['PERWT']).fit(cov_type='HC1')
print(f"  DiD (with state FE): {rob5.params['treat_post']:.4f} (SE: {rob5.bse['treat_post']:.4f})")

# =============================================================================
# STEP 8: SAVE RESULTS
# =============================================================================
print("\n\n[7] Saving Results")
print("="*70)

# Save summary statistics
summary_stats = sample.groupby(['treat', 'post']).apply(
    lambda x: pd.Series({
        'n': len(x),
        'fulltime_mean': np.average(x['fulltime'], weights=x['PERWT']),
        'employed_mean': np.average(x['employed'], weights=x['PERWT']),
        'hours_mean': np.average(x['UHRSWORK'], weights=x['PERWT']),
        'female_pct': np.average(x['female'], weights=x['PERWT']),
        'married_pct': np.average(x['married'], weights=x['PERWT']),
        'hs_pct': np.average(x['educ_hs'], weights=x['PERWT']),
    })
).round(4)
summary_stats.to_csv('summary_stats.csv')
print("  Saved summary_stats.csv")

# Save event study coefficients
event_coefs = pd.DataFrame({
    'year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'coefficient': [event_model.params.get(f'treat_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
                   [event_model.params.get(f'treat_{y}', 0) for y in [2013, 2014, 2015, 2016]],
    'se': [event_model.bse.get(f'treat_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
          [event_model.bse.get(f'treat_{y}', 0) for y in [2013, 2014, 2015, 2016]],
})
event_coefs['ci_low'] = event_coefs['coefficient'] - 1.96 * event_coefs['se']
event_coefs['ci_high'] = event_coefs['coefficient'] + 1.96 * event_coefs['se']
event_coefs.to_csv('event_study_coefs.csv', index=False)
print("  Saved event_study_coefs.csv")

# Save yearly means for plotting
yearly_means = sample.groupby(['YEAR', 'treat']).apply(
    lambda x: pd.Series({
        'fulltime_mean': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x)
    })
).reset_index()
yearly_means.to_csv('yearly_means.csv', index=False)
print("  Saved yearly_means.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"""
PREFERRED ESTIMATE (Model 4: DiD with Year FE, weighted):
---------------------------------------------------------
Effect of DACA eligibility on full-time employment:
  Coefficient: {model4.params['treat_post']:.4f}
  Std. Error:  {model4.bse['treat_post']:.4f}
  95% CI:      [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]
  P-value:     {model4.pvalues['treat_post']:.4f}

Sample Information:
  Total N:     {int(model4.nobs):,}
  Treatment:   {int(sample['treat'].sum()):,}
  Control:     {int((1-sample['treat']).sum()):,}
  Pre-period:  {int((1-sample['post']).sum()):,}
  Post-period: {int(sample['post'].sum()):,}

Interpretation:
  DACA eligibility {'increased' if model4.params['treat_post'] > 0 else 'decreased'}
  full-time employment by {abs(model4.params['treat_post'])*100:.2f} percentage points
  among eligible Hispanic-Mexican Mexican-born individuals aged 26-30.
  This effect is {'statistically significant' if model4.pvalues['treat_post'] < 0.05 else 'not statistically significant'} at the 5% level.
""")

# Print full regression output
print("\n\nFull Regression Output (Preferred Model - Model 4):")
print("-"*70)
print(model4.summary())

print("\n\nFull Regression Output (Model with Covariates - Model 5):")
print("-"*70)
print(model5.summary())

print("\n\nAnalysis complete.")
print("="*70)
