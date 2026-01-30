"""
DACA Impact on Full-Time Employment - Replication Analysis
==========================================================
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the US,
what was the causal impact of DACA eligibility on full-time employment (35+ hours/week)?

DACA implemented June 15, 2012. Examining effects 2013-2016.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set paths
DATA_PATH = 'data/data.csv'
OUTPUT_PATH = '.'

print("=" * 70)
print("DACA REPLICATION ANALYSIS - SESSION 27")
print("=" * 70)

# =============================================================================
# STEP 1: Load and Filter Data (chunked processing for large file)
# =============================================================================
print("\n[STEP 1] Loading and filtering data...")

# Define columns we need
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'MARST', 'EMPSTAT', 'UHRSWORK', 'LABFORCE']

# Process in chunks and filter to relevant population
# We want: Hispanic-Mexican (HISPAN==1), Born in Mexico (BPL==200), Non-citizen (CITIZEN==3)
chunk_size = 500000
filtered_chunks = []

for i, chunk in enumerate(pd.read_csv(DATA_PATH, usecols=cols_needed, chunksize=chunk_size)):
    # Filter to Mexican-born Hispanic non-citizens
    mask = (
        (chunk['HISPAN'] == 1) &  # Mexican Hispanic
        (chunk['BPL'] == 200) &   # Born in Mexico
        (chunk['CITIZEN'] == 3)   # Not a citizen
    )
    filtered = chunk[mask].copy()
    if len(filtered) > 0:
        filtered_chunks.append(filtered)

    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

# Combine filtered data
df = pd.concat(filtered_chunks, ignore_index=True)
print(f"\nFiltered dataset: {len(df):,} observations")
print(f"Years covered: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: Construct DACA Eligibility Variable
# =============================================================================
print("\n[STEP 2] Constructing DACA eligibility variable...")

"""
DACA Eligibility Criteria:
1. Arrived in US before 16th birthday: (YRIMMIG - BIRTHYR) < 16
2. Not yet 31 as of June 15, 2012: Born after June 15, 1981
   - BIRTHYR > 1981, OR
   - BIRTHYR == 1981 and BIRTHQTR >= 3 (July onwards)
3. Lived continuously in US since June 15, 2007: YRIMMIG <= 2007
4. Present in US on June 15, 2012 (assume all in sample)
5. Not a citizen (already filtered)

We cannot distinguish documented vs undocumented among non-citizens.
Per instructions: assume non-citizens without naturalization are undocumented.
"""

# Calculate age at arrival
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Flag: arrived before 16th birthday
df['arrived_before_16'] = (df['age_at_arrival'] < 16) & (df['age_at_arrival'] >= 0)

# Flag: born after June 15, 1981 (not yet 31 on June 15, 2012)
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Born after June 15, 1981 means: BIRTHYR > 1981, OR BIRTHYR==1981 and BIRTHQTR >= 3
df['born_after_june1981'] = (df['BIRTHYR'] > 1981) | ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))

# Flag: in US since at least June 2007 (at least 5 years by 2012)
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007

# Flag: has valid immigration year
df['has_yrimmig'] = df['YRIMMIG'] > 0

# DACA eligible: all criteria met
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['born_after_june1981'] &
    df['in_us_since_2007'] &
    df['has_yrimmig']
).astype(int)

print(f"DACA eligible observations: {df['daca_eligible'].sum():,} ({df['daca_eligible'].mean()*100:.1f}%)")

# =============================================================================
# STEP 3: Construct Outcome and Treatment Variables
# =============================================================================
print("\n[STEP 3] Constructing outcome and treatment variables...")

# Outcome: Full-time employment (35+ hours per week)
# Only consider those in labor force (LABFORCE==2) or working
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Post-DACA period (2013-2016; exclude 2012 as implementation year)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Treatment indicator (interaction)
df['treat_post'] = df['daca_eligible'] * df['post']

# Restrict sample to working-age population (18-50)
# This ensures we're looking at labor market outcomes for relevant ages
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 50)].copy()
print(f"Working-age sample (18-50): {len(df):,} observations")

# Further restrict to pre-period (2006-2011) and post-period (2013-2016)
# Exclude 2012 as it's the implementation year
df = df[df['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df):,} observations")

# =============================================================================
# STEP 4: Create Control Variables
# =============================================================================
print("\n[STEP 4] Creating control variables...")

# Age categories
df['age_cat'] = pd.cut(df['AGE'], bins=[17, 25, 35, 45, 51], labels=['18-25', '26-35', '36-45', '46-50'])

# Education categories (simplified)
df['educ_cat'] = 'Less than HS'
df.loc[df['EDUC'] == 6, 'educ_cat'] = 'High School'
df.loc[(df['EDUC'] >= 7) & (df['EDUC'] <= 9), 'educ_cat'] = 'Some College'
df.loc[df['EDUC'] >= 10, 'educ_cat'] = 'College+'

# Sex
df['female'] = (df['SEX'] == 2).astype(int)

# Marital status (married vs not)
df['married'] = (df['MARST'] <= 2).astype(int)

# Years in US
df['yrs_in_us'] = df['YEAR'] - df['YRIMMIG']
df.loc[df['yrs_in_us'] < 0, 'yrs_in_us'] = 0

# =============================================================================
# STEP 5: Descriptive Statistics
# =============================================================================
print("\n[STEP 5] Computing descriptive statistics...")

# Summary by treatment and period
print("\n" + "="*70)
print("SAMPLE COMPOSITION")
print("="*70)

summary_stats = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'PERWT': 'sum'
}).round(4)
print(summary_stats)

# Full-time employment rates by group and year
ft_by_year = df.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: pd.Series({
        'ft_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n_obs': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).unstack()
print("\n" + "="*70)
print("FULL-TIME EMPLOYMENT RATES BY YEAR AND DACA ELIGIBILITY")
print("="*70)
print(ft_by_year)

# Save for plotting
ft_by_year.to_csv(f'{OUTPUT_PATH}/fulltime_by_year.csv')

# =============================================================================
# STEP 6: Difference-in-Differences Estimation
# =============================================================================
print("\n[STEP 6] Running Difference-in-Differences regression...")

# Basic DiD specification
print("\n" + "="*70)
print("MODEL 1: BASIC DIFFERENCE-IN-DIFFERENCES")
print("="*70)

# Using weighted least squares
model1 = smf.wls('fulltime ~ daca_eligible + post + treat_post',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model1.summary())

# Save key coefficient
did_coef = model1.params['treat_post']
did_se = model1.bse['treat_post']
did_pval = model1.pvalues['treat_post']
did_ci_low = model1.conf_int().loc['treat_post', 0]
did_ci_high = model1.conf_int().loc['treat_post', 1]

print(f"\n*** MAIN DiD ESTIMATE ***")
print(f"Coefficient: {did_coef:.4f}")
print(f"Std. Error:  {did_se:.4f}")
print(f"95% CI:      [{did_ci_low:.4f}, {did_ci_high:.4f}]")
print(f"P-value:     {did_pval:.4f}")

# Model 2: With demographic controls
print("\n" + "="*70)
print("MODEL 2: DiD WITH DEMOGRAPHIC CONTROLS")
print("="*70)

model2 = smf.wls('fulltime ~ daca_eligible + post + treat_post + female + married + C(age_cat) + C(educ_cat)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: With state fixed effects
print("\n" + "="*70)
print("MODEL 3: DiD WITH STATE FIXED EFFECTS")
print("="*70)

model3 = smf.wls('fulltime ~ daca_eligible + post + treat_post + female + married + C(age_cat) + C(educ_cat) + C(STATEFIP)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Print only key coefficients
print("Key Coefficients (state FE coefficients suppressed):")
for var in ['Intercept', 'daca_eligible', 'post', 'treat_post', 'female', 'married']:
    if var in model3.params.index:
        print(f"  {var}: {model3.params[var]:.4f} (SE: {model3.bse[var]:.4f})")

# Model 4: With year fixed effects instead of post dummy
print("\n" + "="*70)
print("MODEL 4: DiD WITH YEAR FIXED EFFECTS")
print("="*70)

df['daca_post'] = df['daca_eligible'] * df['post']  # Same as treat_post
model4 = smf.wls('fulltime ~ daca_eligible + C(YEAR) + daca_post + female + married + C(age_cat) + C(educ_cat) + C(STATEFIP)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("Key Coefficients:")
for var in ['Intercept', 'daca_eligible', 'daca_post', 'female', 'married']:
    if var in model4.params.index:
        print(f"  {var}: {model4.params[var]:.4f} (SE: {model4.bse[var]:.4f})")

# =============================================================================
# STEP 7: Event Study / Pre-Trends Analysis
# =============================================================================
print("\n[STEP 7] Running event study analysis...")

# Create year interactions with DACA eligibility (omit 2011 as reference)
years = sorted(df['YEAR'].unique())
for y in years:
    df[f'year_{y}'] = (df['YEAR'] == y).astype(int)
    df[f'daca_year_{y}'] = df['daca_eligible'] * df[f'year_{y}']

# Drop 2011 as reference year
year_vars = [f'daca_year_{y}' for y in years if y != 2011]
formula = 'fulltime ~ daca_eligible + ' + ' + '.join([f'year_{y}' for y in years if y != 2011]) + ' + ' + ' + '.join(year_vars)
formula += ' + female + married + C(age_cat) + C(educ_cat) + C(STATEFIP)'

model_event = smf.wls(formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Extract event study coefficients
event_coefs = []
for y in years:
    if y == 2011:
        event_coefs.append({'year': y, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
    else:
        var = f'daca_year_{y}'
        if var in model_event.params.index:
            coef = model_event.params[var]
            se = model_event.bse[var]
            ci = model_event.conf_int().loc[var]
            event_coefs.append({'year': y, 'coef': coef, 'se': se, 'ci_low': ci[0], 'ci_high': ci[1]})

event_df = pd.DataFrame(event_coefs)
print("\nEvent Study Coefficients (reference: 2011):")
print(event_df.to_string(index=False))
event_df.to_csv(f'{OUTPUT_PATH}/event_study_coefs.csv', index=False)

# =============================================================================
# STEP 8: Robustness Checks
# =============================================================================
print("\n[STEP 8] Running robustness checks...")

# Robustness 1: Alternative age restriction (16-45)
print("\n--- Robustness 1: Age 16-45 ---")
df_full = pd.concat(filtered_chunks, ignore_index=True)
df_full['age_at_arrival'] = df_full['YRIMMIG'] - df_full['BIRTHYR']
df_full['arrived_before_16'] = (df_full['age_at_arrival'] < 16) & (df_full['age_at_arrival'] >= 0)
df_full['born_after_june1981'] = (df_full['BIRTHYR'] > 1981) | ((df_full['BIRTHYR'] == 1981) & (df_full['BIRTHQTR'] >= 3))
df_full['in_us_since_2007'] = df_full['YRIMMIG'] <= 2007
df_full['has_yrimmig'] = df_full['YRIMMIG'] > 0
df_full['daca_eligible'] = (df_full['arrived_before_16'] & df_full['born_after_june1981'] & df_full['in_us_since_2007'] & df_full['has_yrimmig']).astype(int)
df_full['fulltime'] = (df_full['UHRSWORK'] >= 35).astype(int)
df_full['post'] = (df_full['YEAR'] >= 2013).astype(int)
df_full['treat_post'] = df_full['daca_eligible'] * df_full['post']

df_rob1 = df_full[(df_full['AGE'] >= 16) & (df_full['AGE'] <= 45) & (df_full['YEAR'] != 2012)].copy()
model_rob1 = smf.wls('fulltime ~ daca_eligible + post + treat_post',
                      data=df_rob1, weights=df_rob1['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (age 16-45): {model_rob1.params['treat_post']:.4f} (SE: {model_rob1.bse['treat_post']:.4f})")

# Robustness 2: Employed (any hours) as outcome
print("\n--- Robustness 2: Employed (any hours > 0) as outcome ---")
df['employed'] = (df['UHRSWORK'] > 0).astype(int)
model_rob2 = smf.wls('employed ~ daca_eligible + post + treat_post',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (employed): {model_rob2.params['treat_post']:.4f} (SE: {model_rob2.bse['treat_post']:.4f})")

# Robustness 3: In labor force as outcome
print("\n--- Robustness 3: In labor force as outcome ---")
df['in_lf'] = (df['LABFORCE'] == 2).astype(int)
model_rob3 = smf.wls('in_lf ~ daca_eligible + post + treat_post',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (labor force): {model_rob3.params['treat_post']:.4f} (SE: {model_rob3.bse['treat_post']:.4f})")

# Robustness 4: Men only
print("\n--- Robustness 4: Men only ---")
df_men = df[df['female'] == 0].copy()
model_rob4 = smf.wls('fulltime ~ daca_eligible + post + treat_post',
                      data=df_men, weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (men): {model_rob4.params['treat_post']:.4f} (SE: {model_rob4.bse['treat_post']:.4f})")

# Robustness 5: Women only
print("\n--- Robustness 5: Women only ---")
df_women = df[df['female'] == 1].copy()
model_rob5 = smf.wls('fulltime ~ daca_eligible + post + treat_post',
                      data=df_women, weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (women): {model_rob5.params['treat_post']:.4f} (SE: {model_rob5.bse['treat_post']:.4f})")

# =============================================================================
# STEP 9: Create Visualizations
# =============================================================================
print("\n[STEP 9] Creating visualizations...")

# Figure 1: Trends in full-time employment
fig, ax = plt.subplots(figsize=(10, 6))
years_plot = sorted(df['YEAR'].unique())

# Calculate weighted means by year and group
trends = df.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

ax.plot(trends.index, trends[0], 'b-o', label='Non-DACA Eligible', linewidth=2, markersize=8)
ax.plot(trends.index, trends[1], 'r-s', label='DACA Eligible', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle='--', label='DACA Implementation (June 2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/figure1_trends.png', dpi=150)
plt.close()

# Figure 2: Event study plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(event_df['year'], event_df['coef'],
            yerr=[event_df['coef'] - event_df['ci_low'], event_df['ci_high'] - event_df['coef']],
            fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8, color='darkblue')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation', linewidth=2)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Effect on Full-Time Employment\n(relative to 2011)', fontsize=12)
ax.set_title('Event Study: DACA Effect on Full-Time Employment', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/figure2_event_study.png', dpi=150)
plt.close()

print("Figures saved.")

# =============================================================================
# STEP 10: Save Results Summary
# =============================================================================
print("\n[STEP 10] Saving results summary...")

results_summary = {
    'Main DiD Estimate': {
        'coefficient': did_coef,
        'std_error': did_se,
        'ci_low': did_ci_low,
        'ci_high': did_ci_high,
        'p_value': did_pval,
        'sample_size': len(df)
    },
    'Model 2 (with controls)': {
        'coefficient': model2.params['treat_post'],
        'std_error': model2.bse['treat_post'],
        'p_value': model2.pvalues['treat_post']
    },
    'Model 3 (with state FE)': {
        'coefficient': model3.params['treat_post'],
        'std_error': model3.bse['treat_post'],
        'p_value': model3.pvalues['treat_post']
    },
    'Model 4 (with year FE)': {
        'coefficient': model4.params['daca_post'],
        'std_error': model4.bse['daca_post'],
        'p_value': model4.pvalues['daca_post']
    }
}

import json
with open(f'{OUTPUT_PATH}/results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

# Create formatted results table
results_table = pd.DataFrame({
    'Model': ['Basic DiD', 'With Controls', 'With State FE', 'With Year FE'],
    'Coefficient': [model1.params['treat_post'], model2.params['treat_post'],
                   model3.params['treat_post'], model4.params['daca_post']],
    'Std. Error': [model1.bse['treat_post'], model2.bse['treat_post'],
                  model3.bse['treat_post'], model4.bse['daca_post']],
    'P-value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
               model3.pvalues['treat_post'], model4.pvalues['daca_post']]
})
results_table.to_csv(f'{OUTPUT_PATH}/results_table.csv', index=False)
print(results_table.to_string(index=False))

# =============================================================================
# FINAL OUTPUT
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nPreferred Estimate (Basic DiD):")
print(f"  Effect Size: {did_coef:.4f} ({did_coef*100:.2f} percentage points)")
print(f"  Std. Error:  {did_se:.4f}")
print(f"  95% CI:      [{did_ci_low:.4f}, {did_ci_high:.4f}]")
print(f"  Sample Size: {len(df):,}")
print(f"\nInterpretation: DACA eligibility is associated with a {did_coef*100:.2f} percentage point")
print(f"{'increase' if did_coef > 0 else 'decrease'} in full-time employment (p = {did_pval:.4f}).")

# Descriptive stats for report
print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS FOR REPORT")
print("="*70)

desc_stats = df.groupby('daca_eligible').agg({
    'fulltime': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)
print(desc_stats)

# Pre/post comparison
print("\nPre-Post Full-Time Employment Rates:")
prepost = df.groupby(['daca_eligible', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
prepost.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
prepost.index = ['Non-Eligible', 'DACA Eligible']
print(prepost.round(4))

# Calculate simple DD
dd_simple = (prepost.loc['DACA Eligible', 'Post-DACA (2013-2016)'] - prepost.loc['DACA Eligible', 'Pre-DACA (2006-2011)']) - \
            (prepost.loc['Non-Eligible', 'Post-DACA (2013-2016)'] - prepost.loc['Non-Eligible', 'Pre-DACA (2006-2011)'])
print(f"\nSimple DD calculation: {dd_simple:.4f}")

print("\n" + "="*70)
print("OUTPUT FILES CREATED:")
print("="*70)
print("  - fulltime_by_year.csv")
print("  - event_study_coefs.csv")
print("  - results_summary.json")
print("  - results_table.csv")
print("  - figure1_trends.png")
print("  - figure2_event_study.png")
