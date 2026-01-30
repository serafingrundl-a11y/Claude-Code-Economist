"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexico-born individuals in the US.

Author: Independent Replication
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# STEP 1: LOAD DATA WITH FILTERING (to handle large file)
# =============================================================================
print("\n[STEP 1] Loading data with chunked reading and filtering...")

# Define columns we need (reduced set to save memory)
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
           'MARST', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
           'EMPSTAT', 'UHRSWORK']

# Read in chunks and filter as we go
chunks = []
chunk_size = 1000000

for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunk_size)):
    # Filter to target population: Hispanic-Mexican (HISPAN==1), Born in Mexico (BPL==200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1)*chunk_size:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal Hispanic-Mexican, Mexico-born observations: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: FURTHER FILTERING
# =============================================================================
print("\n[STEP 2] Further filtering...")

# Filter: Non-citizen (CITIZEN == 3)
df = df[df['CITIZEN'] == 3].copy()
print(f"After non-citizen filter: {len(df):,}")

# Filter: Working age (18-64)
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)].copy()
print(f"After working age (18-64) filter: {len(df):,}")

# Exclude 2012 (DACA implemented mid-year)
df = df[df['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df):,}")

# =============================================================================
# STEP 3: CREATE DACA ELIGIBILITY VARIABLES
# =============================================================================
print("\n[STEP 3] Creating DACA eligibility variables...")

# Calculate age at arrival
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Born after June 15, 1981 (not yet 31 on June 15, 2012)
df['born_after_cutoff'] = (
    (df['BIRTHYR'] >= 1982) |
    ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))
)

# Arrived before 16th birthday
df['arrived_before_16'] = df['age_at_arrival'] < 16

# Lived in US since 2007
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007

# Full DACA eligibility
df['daca_eligible'] = (
    df['born_after_cutoff'] &
    df['arrived_before_16'] &
    df['in_us_since_2007']
).astype(int)

print(f"DACA eligible: {df['daca_eligible'].sum():,} ({df['daca_eligible'].mean()*100:.1f}%)")
print(f"Not eligible: {(1-df['daca_eligible']).sum():,} ({(1-df['daca_eligible'].mean())*100:.1f}%)")

# =============================================================================
# STEP 4: CREATE OUTCOME AND TREATMENT VARIABLES
# =============================================================================
print("\n[STEP 4] Creating outcome and treatment variables...")

# Outcome: Full-time employment
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)

# Post-DACA period (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term
df['daca_x_post'] = df['daca_eligible'] * df['post']

print(f"\nOutcome summary:")
print(f"Employment rate: {df['employed'].mean()*100:.1f}%")
print(f"Full-time employment rate: {df['fulltime'].mean()*100:.1f}%")

# =============================================================================
# STEP 5: CREATE CONTROL VARIABLES
# =============================================================================
print("\n[STEP 5] Creating control variables...")

df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)

# Education categories
df['educ_less_hs'] = (df['EDUC'] <= 5).astype(int)
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_some_college'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)
df['educ_college'] = (df['EDUC'] >= 10).astype(int)

df['age_sq'] = df['AGE'] ** 2
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']

print("Control variables created.")

# =============================================================================
# STEP 6: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[STEP 6] Descriptive Statistics")
print("="*60)

summary_vars = ['fulltime', 'employed', 'AGE', 'female', 'married',
                'educ_less_hs', 'educ_hs', 'educ_some_college', 'educ_college',
                'years_in_us']

print("\n--- Pre-DACA Period (2006-2011) ---")
pre_eligible = df[(df['post'] == 0) & (df['daca_eligible'] == 1)]
pre_ineligible = df[(df['post'] == 0) & (df['daca_eligible'] == 0)]

print(f"\nDACA Eligible (N = {len(pre_eligible):,}):")
for var in summary_vars:
    print(f"  {var}: {pre_eligible[var].mean():.3f}")

print(f"\nDACA Ineligible (N = {len(pre_ineligible):,}):")
for var in summary_vars:
    print(f"  {var}: {pre_ineligible[var].mean():.3f}")

print("\n--- Post-DACA Period (2013-2016) ---")
post_eligible = df[(df['post'] == 1) & (df['daca_eligible'] == 1)]
post_ineligible = df[(df['post'] == 1) & (df['daca_eligible'] == 0)]

print(f"\nDACA Eligible (N = {len(post_eligible):,}):")
for var in summary_vars:
    print(f"  {var}: {post_eligible[var].mean():.3f}")

print(f"\nDACA Ineligible (N = {len(post_ineligible):,}):")
for var in summary_vars:
    print(f"  {var}: {post_ineligible[var].mean():.3f}")

# =============================================================================
# STEP 7: MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("[STEP 7] MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

# Simple DiD
pre_treat_mean = pre_eligible['fulltime'].mean()
pre_control_mean = pre_ineligible['fulltime'].mean()
post_treat_mean = post_eligible['fulltime'].mean()
post_control_mean = post_ineligible['fulltime'].mean()

simple_did = (post_treat_mean - pre_treat_mean) - (post_control_mean - pre_control_mean)

print("\n--- Simple DiD Calculation ---")
print(f"Pre-DACA, Eligible: {pre_treat_mean:.4f}")
print(f"Pre-DACA, Ineligible: {pre_control_mean:.4f}")
print(f"Post-DACA, Eligible: {post_treat_mean:.4f}")
print(f"Post-DACA, Ineligible: {post_control_mean:.4f}")
print(f"\nSimple DiD Estimate: {simple_did:.4f}")

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post',
                  data=df, weights=df['PERWT'])
results1 = model1.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(results1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married',
                  data=df, weights=df['PERWT'])
results2 = model2.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(results2.summary().tables[1])

# Model 3: DiD with demographic and education controls
print("\n--- Model 3: DiD with Demographics + Education ---")
model3 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college',
                  data=df, weights=df['PERWT'])
results3 = model3.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(results3.summary().tables[1])

# Model 4: Full model with state and year fixed effects
print("\n--- Model 4: Full Model with State and Year FE ---")
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
df_fe = pd.concat([df, year_dummies, state_dummies], axis=1)

year_cols = [col for col in df_fe.columns if col.startswith('year_')]
state_cols = [col for col in df_fe.columns if col.startswith('state_')]

formula4 = 'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)

model4 = smf.wls(formula4, data=df_fe, weights=df_fe['PERWT'])
results4 = model4.fit(cov_type='cluster', cov_kwds={'groups': df_fe['STATEFIP']})

print(f"\ndaca_eligible: {results4.params['daca_eligible']:.4f} (SE: {results4.bse['daca_eligible']:.4f})")
print(f"daca_x_post: {results4.params['daca_x_post']:.4f} (SE: {results4.bse['daca_x_post']:.4f})")
print(f"  t-stat: {results4.tvalues['daca_x_post']:.3f}, p-value: {results4.pvalues['daca_x_post']:.4f}")

# =============================================================================
# STEP 8: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("[STEP 8] ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Alternative Control Group (Ages 31-45 only)
print("\n--- Robustness 1: Alternative Control Group (Ages 31-45 only) ---")
df_robust1 = df[(df['daca_eligible'] == 1) |
                ((df['AGE'] >= 31) & (df['AGE'] <= 45))].copy()
model_r1 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married',
                    data=df_robust1, weights=df_robust1['PERWT'])
results_r1 = model_r1.fit(cov_type='cluster', cov_kwds={'groups': df_robust1['STATEFIP']})
print(f"daca_x_post: {results_r1.params['daca_x_post']:.4f} (SE: {results_r1.bse['daca_x_post']:.4f})")
print(f"  N = {len(df_robust1):,}")

# Robustness 2: Any employment as outcome
print("\n--- Robustness 2: Any Employment as Outcome ---")
model_r2 = smf.wls('employed ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college',
                    data=df, weights=df['PERWT'])
results_r2 = model_r2.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"daca_x_post: {results_r2.params['daca_x_post']:.4f} (SE: {results_r2.bse['daca_x_post']:.4f})")

# Robustness 3: Males only
print("\n--- Robustness 3: Males Only ---")
df_male = df[df['female'] == 0].copy()
model_r3 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + married + educ_hs + educ_some_college + educ_college',
                    data=df_male, weights=df_male['PERWT'])
results_r3 = model_r3.fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
print(f"daca_x_post: {results_r3.params['daca_x_post']:.4f} (SE: {results_r3.bse['daca_x_post']:.4f})")
print(f"  N = {len(df_male):,}")

# Robustness 4: Females only
print("\n--- Robustness 4: Females Only ---")
df_female = df[df['female'] == 1].copy()
model_r4 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + married + educ_hs + educ_some_college + educ_college',
                    data=df_female, weights=df_female['PERWT'])
results_r4 = model_r4.fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})
print(f"daca_x_post: {results_r4.params['daca_x_post']:.4f} (SE: {results_r4.bse['daca_x_post']:.4f})")
print(f"  N = {len(df_female):,}")

# Robustness 5: Unweighted regression
print("\n--- Robustness 5: Unweighted Regression ---")
model_r5 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college',
                    data=df)
results_r5 = model_r5.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"daca_x_post: {results_r5.params['daca_x_post']:.4f} (SE: {results_r5.bse['daca_x_post']:.4f})")

# =============================================================================
# STEP 9: EVENT STUDY / DYNAMIC EFFECTS
# =============================================================================
print("\n" + "="*80)
print("[STEP 9] EVENT STUDY - DYNAMIC EFFECTS")
print("="*80)

df['daca_x_2006'] = df['daca_eligible'] * (df['YEAR'] == 2006).astype(int)
df['daca_x_2007'] = df['daca_eligible'] * (df['YEAR'] == 2007).astype(int)
df['daca_x_2008'] = df['daca_eligible'] * (df['YEAR'] == 2008).astype(int)
df['daca_x_2009'] = df['daca_eligible'] * (df['YEAR'] == 2009).astype(int)
df['daca_x_2010'] = df['daca_eligible'] * (df['YEAR'] == 2010).astype(int)
df['daca_x_2013'] = df['daca_eligible'] * (df['YEAR'] == 2013).astype(int)
df['daca_x_2014'] = df['daca_eligible'] * (df['YEAR'] == 2014).astype(int)
df['daca_x_2015'] = df['daca_eligible'] * (df['YEAR'] == 2015).astype(int)
df['daca_x_2016'] = df['daca_eligible'] * (df['YEAR'] == 2016).astype(int)

# Create year dummies for this model
year_dummies_ev = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
df_event = pd.concat([df, year_dummies_ev], axis=1)
year_cols_ev = [col for col in df_event.columns if col.startswith('year_')]

event_formula = 'fulltime ~ daca_eligible + daca_x_2006 + daca_x_2007 + daca_x_2008 + daca_x_2009 + daca_x_2010 + daca_x_2013 + daca_x_2014 + daca_x_2015 + daca_x_2016 + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college + ' + ' + '.join(year_cols_ev)

model_event = smf.wls(event_formula, data=df_event, weights=df_event['PERWT'])
results_event = model_event.fit(cov_type='cluster', cov_kwds={'groups': df_event['STATEFIP']})

print("\nYear-specific treatment effects (relative to 2011):")
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_coefs = []
event_ses = []
for yr in event_years:
    coef = results_event.params[f'daca_x_{yr}']
    se = results_event.bse[f'daca_x_{yr}']
    pval = results_event.pvalues[f'daca_x_{yr}']
    event_coefs.append(coef)
    event_ses.append(se)
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    print(f"  {yr}: {coef:.4f} (SE: {se:.4f}) {sig}")

event_years_plot = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs_plot = event_coefs[:5] + [0] + event_coefs[5:]
event_ses_plot = event_ses[:5] + [0] + event_ses[5:]

# =============================================================================
# STEP 10: CREATE VISUALIZATIONS
# =============================================================================
print("\n" + "="*80)
print("[STEP 10] CREATING VISUALIZATIONS")
print("="*80)

# Figure 1: Event Study Plot
plt.figure(figsize=(10, 6))
plt.errorbar(event_years_plot, event_coefs_plot, yerr=[1.96*se for se in event_ses_plot],
             fmt='o-', capsize=5, capthick=2, color='navy', markersize=8)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Effect on Full-Time Employment', fontsize=12)
plt.title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('event_study_17.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: event_study_17.png")

# Figure 2: Trends by Treatment Status
trends_data = df.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
trends_data.columns = ['Ineligible', 'Eligible']

plt.figure(figsize=(10, 6))
plt.plot(trends_data.index, trends_data['Eligible'], 'o-', color='blue',
         linewidth=2, markersize=8, label='DACA Eligible')
plt.plot(trends_data.index, trends_data['Ineligible'], 's--', color='red',
         linewidth=2, markersize=8, label='DACA Ineligible')
plt.axvline(x=2012, color='gray', linestyle='--', alpha=0.7, label='DACA (2012)')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trends_17.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: trends_17.png")

# Figure 3: Age Distribution
plt.figure(figsize=(10, 6))
eligible_ages = df[df['daca_eligible'] == 1]['AGE']
ineligible_ages = df[df['daca_eligible'] == 0]['AGE']
plt.hist(eligible_ages, bins=range(18, 65, 2), alpha=0.6, label='DACA Eligible', color='blue', density=True)
plt.hist(ineligible_ages, bins=range(18, 65, 2), alpha=0.6, label='DACA Ineligible', color='red', density=True)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Age Distribution by DACA Eligibility Status', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('age_dist_17.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: age_dist_17.png")

# =============================================================================
# STEP 11: PLACEBO TEST
# =============================================================================
print("\n" + "="*80)
print("[STEP 11] PLACEBO TEST - Fake Treatment in 2009")
print("="*80)

df_pre = df[df['post'] == 0].copy()
df_pre['fake_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['daca_x_fake_post'] = df_pre['daca_eligible'] * df_pre['fake_post']

model_placebo = smf.wls('fulltime ~ daca_eligible + fake_post + daca_x_fake_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college',
                         data=df_pre, weights=df_pre['PERWT'])
results_placebo = model_placebo.fit(cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})
print(f"Placebo DiD (fake treatment 2009): {results_placebo.params['daca_x_fake_post']:.4f}")
print(f"  SE: {results_placebo.bse['daca_x_fake_post']:.4f}")
print(f"  p-value: {results_placebo.pvalues['daca_x_fake_post']:.4f}")

# =============================================================================
# STEP 12: SUMMARY AND FINAL RESULTS
# =============================================================================
print("\n" + "="*80)
print("[STEP 12] FINAL RESULTS SUMMARY")
print("="*80)

print("\n*** PREFERRED SPECIFICATION: Model 3 (DiD with Demographics + Education) ***")
print(f"\nMain Effect (daca_x_post):")
print(f"  Coefficient: {results3.params['daca_x_post']:.4f}")
print(f"  Standard Error: {results3.bse['daca_x_post']:.4f}")
print(f"  95% CI: [{results3.conf_int().loc['daca_x_post', 0]:.4f}, {results3.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"  t-statistic: {results3.tvalues['daca_x_post']:.3f}")
print(f"  p-value: {results3.pvalues['daca_x_post']:.4f}")
print(f"\nSample Size: {int(results3.nobs):,}")
print(f"R-squared: {results3.rsquared:.4f}")

results_dict = {
    'Model 1 (Basic)': results1,
    'Model 2 (Demographics)': results2,
    'Model 3 (Full Controls)': results3,
    'Model 4 (State-Year FE)': results4
}

print("\n--- Coefficient Comparison Table ---")
print(f"{'Model':<30} {'daca_x_post':<12} {'SE':<10} {'N':<15}")
print("-"*70)
for name, res in results_dict.items():
    coef = res.params['daca_x_post']
    se = res.bse['daca_x_post']
    n = int(res.nobs)
    print(f"{name:<30} {coef:<12.4f} {se:<10.4f} {n:<15,}")

# Save key results
with open('results_summary_17.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - KEY RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write("PREFERRED ESTIMATE (Model 3):\n")
    f.write(f"  Effect Size: {results3.params['daca_x_post']:.4f}\n")
    f.write(f"  Standard Error: {results3.bse['daca_x_post']:.4f}\n")
    f.write(f"  95% CI: [{results3.conf_int().loc['daca_x_post', 0]:.4f}, {results3.conf_int().loc['daca_x_post', 1]:.4f}]\n")
    f.write(f"  Sample Size: {int(results3.nobs):,}\n")
    f.write(f"  R-squared: {results3.rsquared:.4f}\n\n")

    f.write("SIMPLE DiD CALCULATION:\n")
    f.write(f"  Pre-DACA Eligible Mean: {pre_treat_mean:.4f}\n")
    f.write(f"  Pre-DACA Ineligible Mean: {pre_control_mean:.4f}\n")
    f.write(f"  Post-DACA Eligible Mean: {post_treat_mean:.4f}\n")
    f.write(f"  Post-DACA Ineligible Mean: {post_control_mean:.4f}\n")
    f.write(f"  Simple DiD: {simple_did:.4f}\n\n")

    f.write("INTERPRETATION:\n")
    eff_pct = results3.params['daca_x_post']*100
    f.write(f"  DACA eligibility is associated with a {eff_pct:.2f} percentage point\n")
    f.write("  change in the probability of full-time employment.\n\n")

    f.write("ROBUSTNESS SUMMARY:\n")
    f.write(f"  Alternative Control (31-45): {results_r1.params['daca_x_post']:.4f}\n")
    f.write(f"  Any Employment Outcome: {results_r2.params['daca_x_post']:.4f}\n")
    f.write(f"  Males Only: {results_r3.params['daca_x_post']:.4f}\n")
    f.write(f"  Females Only: {results_r4.params['daca_x_post']:.4f}\n")
    f.write(f"  Unweighted: {results_r5.params['daca_x_post']:.4f}\n")
    f.write(f"  Placebo (2009): {results_placebo.params['daca_x_fake_post']:.4f}\n\n")

    f.write("EVENT STUDY COEFFICIENTS:\n")
    for yr, coef, se in zip(event_years, event_coefs, event_ses):
        f.write(f"  {yr}: {coef:.4f} (SE: {se:.4f})\n")

print("\nResults saved to results_summary_17.txt")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Store results for use in LaTeX
results_for_latex = {
    'model1': results1,
    'model2': results2,
    'model3': results3,
    'model4': results4,
    'robust1': results_r1,
    'robust2': results_r2,
    'robust3': results_r3,
    'robust4': results_r4,
    'robust5': results_r5,
    'placebo': results_placebo,
    'event': results_event,
    'simple_did': simple_did,
    'pre_treat_mean': pre_treat_mean,
    'pre_control_mean': pre_control_mean,
    'post_treat_mean': post_treat_mean,
    'post_control_mean': post_control_mean,
    'n_eligible': df['daca_eligible'].sum(),
    'n_ineligible': (1-df['daca_eligible']).sum(),
    'n_total': len(df)
}
