"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals.

Identification Strategy: Difference-in-Differences
- Treatment Group: Ages 26-30 as of June 15, 2012 (born 1982-1986)
- Control Group: Ages 31-35 as of June 15, 2012 (born 1977-1981)
- Pre-period: 2006-2011
- Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 80)
print("DACA REPLICATION ANALYSIS")
print("=" * 80)

# -----------------------------------------------------------------------------
# Step 1: Load Data in Chunks with Initial Filtering
# -----------------------------------------------------------------------------
print("\n[Step 1] Loading data with initial filtering...")

# Only keep columns we need
cols_needed = ['YEAR', 'STATEFIP', 'PERNUM', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK']

# Process in chunks and filter
chunk_size = 1000000
chunks = []
n_original = 0

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size):
    n_original += len(chunk)

    # Apply initial filters to reduce memory
    # 1. Hispanic-Mexican (HISPAN = 1)
    # 2. Born in Mexico (BPL = 200)
    # 3. Non-citizen (CITIZEN = 3)
    # 4. Birth years 1977-1986 for our target cohorts
    # 5. Exclude 2012

    filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3) &
        (chunk['BIRTHYR'] >= 1977) &
        (chunk['BIRTHYR'] <= 1986) &
        (chunk['YEAR'] != 2012)
    ]

    if len(filtered) > 0:
        chunks.append(filtered)

    print(f"  Processed {n_original:,} rows...")

df_sample = pd.concat(chunks, ignore_index=True)
print(f"\nTotal observations in raw data: {n_original:,}")
print(f"After initial filtering: {len(df_sample):,}")

# -----------------------------------------------------------------------------
# Step 2: Apply Remaining Sample Selection Criteria
# -----------------------------------------------------------------------------
print("\n[Step 2] Applying remaining sample selection criteria...")

n_initial = len(df_sample)

# Restriction: Immigrated before age 16
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[(df_sample['YRIMMIG'] > 0) & (df_sample['age_at_immig'] < 16)]
n_after_age_immig = len(df_sample)
print(f"After arrived before age 16 restriction: {n_after_age_immig:,} (dropped {n_initial - n_after_age_immig:,})")

# Restriction: In US by 2007 (continuous residence requirement)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]
n_final = len(df_sample)
print(f"After in US by 2007 restriction: {n_final:,} (dropped {n_after_age_immig - n_final:,})")

print(f"\nFinal analytic sample: {n_final:,} observations")

# Store sample sizes for flow diagram
sample_sizes = {
    'original': n_original,
    'after_initial_filter': n_initial,
    'after_age_immig': n_after_age_immig,
    'final': n_final
}

# -----------------------------------------------------------------------------
# Step 3: Create Analysis Variables
# -----------------------------------------------------------------------------
print("\n[Step 3] Creating analysis variables...")

# Treatment indicator: Born 1982-1986 (ages 26-30 as of June 2012)
df_sample['treat'] = ((df_sample['BIRTHYR'] >= 1982) & (df_sample['BIRTHYR'] <= 1986)).astype(int)

# Post indicator: Years 2013-2016
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Interaction term
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

# Outcome: Full-time employment (UHRSWORK >= 35)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Additional covariates
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'] <= 2).astype(int)  # Married spouse present or absent

# Education categories
df_sample['educ_lesshs'] = (df_sample['EDUC'] <= 5).astype(int)  # Less than high school
df_sample['educ_hs'] = (df_sample['EDUC'] == 6).astype(int)  # High school
df_sample['educ_somecol'] = ((df_sample['EDUC'] >= 7) & (df_sample['EDUC'] <= 9)).astype(int)  # Some college
df_sample['educ_college'] = (df_sample['EDUC'] >= 10).astype(int)  # College or more

# Age at survey time
df_sample['age'] = df_sample['AGE']
df_sample['age_sq'] = df_sample['age'] ** 2

print(f"Treatment group (ages 26-30): {df_sample['treat'].sum():,} observations")
print(f"Control group (ages 31-35): {(1-df_sample['treat']).sum():,} observations")
print(f"Pre-period observations: {(1-df_sample['post']).sum():,}")
print(f"Post-period observations: {df_sample['post'].sum():,}")

# -----------------------------------------------------------------------------
# Step 4: Summary Statistics
# -----------------------------------------------------------------------------
print("\n[Step 4] Summary Statistics")
print("-" * 60)

# Overall sample statistics
print("\nOverall Sample Characteristics:")
print(f"  Mean full-time employment rate: {df_sample['fulltime'].mean():.4f}")
print(f"  Mean age: {df_sample['age'].mean():.2f}")
print(f"  Proportion female: {df_sample['female'].mean():.4f}")
print(f"  Proportion married: {df_sample['married'].mean():.4f}")
print(f"  Proportion less than HS: {df_sample['educ_lesshs'].mean():.4f}")
print(f"  Proportion HS: {df_sample['educ_hs'].mean():.4f}")
print(f"  Proportion some college: {df_sample['educ_somecol'].mean():.4f}")
print(f"  Proportion college+: {df_sample['educ_college'].mean():.4f}")

# Group means by treatment status
print("\nCharacteristics by Treatment Status:")
for treat_val, treat_name in [(1, "Treatment (26-30)"), (0, "Control (31-35)")]:
    subset = df_sample[df_sample['treat'] == treat_val]
    print(f"\n  {treat_name}:")
    print(f"    N: {len(subset):,}")
    print(f"    Mean full-time: {subset['fulltime'].mean():.4f}")
    print(f"    Mean age: {subset['age'].mean():.2f}")
    print(f"    Prop female: {subset['female'].mean():.4f}")
    print(f"    Prop married: {subset['married'].mean():.4f}")

# 2x2 DiD table
print("\n\nDifference-in-Differences Table (Full-time Employment Rate):")
print("-" * 60)
did_table = df_sample.groupby(['treat', 'post']).agg(
    mean_fulltime=('fulltime', 'mean'),
    n=('fulltime', 'count'),
    weighted_mean=('fulltime', lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT']))
).round(4)
print(did_table)

# Calculate raw DiD
pre_treat = df_sample[(df_sample['treat']==1) & (df_sample['post']==0)]['fulltime'].mean()
post_treat = df_sample[(df_sample['treat']==1) & (df_sample['post']==1)]['fulltime'].mean()
pre_control = df_sample[(df_sample['treat']==0) & (df_sample['post']==0)]['fulltime'].mean()
post_control = df_sample[(df_sample['treat']==0) & (df_sample['post']==1)]['fulltime'].mean()

diff_treat = post_treat - pre_treat
diff_control = post_control - pre_control
did_estimate_raw = diff_treat - diff_control

print(f"\nRaw DiD Calculation:")
print(f"  Treatment change: {post_treat:.4f} - {pre_treat:.4f} = {diff_treat:.4f}")
print(f"  Control change: {post_control:.4f} - {pre_control:.4f} = {diff_control:.4f}")
print(f"  DiD Estimate: {diff_treat:.4f} - {diff_control:.4f} = {did_estimate_raw:.4f}")

# -----------------------------------------------------------------------------
# Step 5: Regression Analysis
# -----------------------------------------------------------------------------
print("\n\n" + "=" * 80)
print("[Step 5] Regression Analysis")
print("=" * 80)

# Model 1: Basic DiD (no weights)
print("\n--- Model 1: Basic DiD (OLS, no weights) ---")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_sample).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with person weights
print("\n--- Model 2: DiD with Person Weights (WLS) ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post', data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with covariates and weights
print("\n--- Model 3: DiD with Covariates and Weights ---")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with Year Fixed Effects ---")
df_sample['year_factor'] = pd.Categorical(df_sample['YEAR'])
model4 = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# Model 5: DiD with state fixed effects
print("\n--- Model 5: DiD with State Fixed Effects ---")
model5 = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')
# Just print key coefficients
print("\nKey Coefficients from Model 5:")
print(f"  treat_post coefficient: {model5.params['treat_post']:.6f}")
print(f"  treat_post std error: {model5.bse['treat_post']:.6f}")
print(f"  treat_post p-value: {model5.pvalues['treat_post']:.6f}")
print(f"  95% CI: [{model5.conf_int().loc['treat_post', 0]:.6f}, {model5.conf_int().loc['treat_post', 1]:.6f}]")

# Model 6: Clustered standard errors by state
print("\n--- Model 6: DiD with State Clusters ---")
model6 = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_sample, weights=df_sample['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print("\nKey Coefficients from Model 6 (clustered SE by state):")
print(f"  treat_post coefficient: {model6.params['treat_post']:.6f}")
print(f"  treat_post std error: {model6.bse['treat_post']:.6f}")
print(f"  treat_post p-value: {model6.pvalues['treat_post']:.6f}")
print(f"  95% CI: [{model6.conf_int().loc['treat_post', 0]:.6f}, {model6.conf_int().loc['treat_post', 1]:.6f}]")

# -----------------------------------------------------------------------------
# Step 6: Robustness Checks
# -----------------------------------------------------------------------------
print("\n\n" + "=" * 80)
print("[Step 6] Robustness Checks")
print("=" * 80)

# Robustness 1: Alternative age groups
print("\n--- Robustness 1: Narrower Age Bands (27-29 vs 32-34) ---")
df_narrow = df_sample[(df_sample['BIRTHYR'] >= 1978) & (df_sample['BIRTHYR'] <= 1985)]
df_narrow = df_narrow.copy()
df_narrow['treat_narrow'] = ((df_narrow['BIRTHYR'] >= 1983) & (df_narrow['BIRTHYR'] <= 1985)).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treat_narrow'] * df_narrow['post']

model_robust1 = smf.wls('fulltime ~ treat_narrow + C(YEAR) + treat_post_narrow + female + married + educ_hs + educ_somecol + educ_college',
                         data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"  treat_post coefficient: {model_robust1.params['treat_post_narrow']:.6f}")
print(f"  treat_post std error: {model_robust1.bse['treat_post_narrow']:.6f}")
print(f"  95% CI: [{model_robust1.conf_int().loc['treat_post_narrow', 0]:.6f}, {model_robust1.conf_int().loc['treat_post_narrow', 1]:.6f}]")
print(f"  N: {len(df_narrow):,}")

# Robustness 2: By gender
print("\n--- Robustness 2: Heterogeneity by Gender ---")
for sex_val, sex_name in [(0, "Male"), (1, "Female")]:
    df_sex = df_sample[df_sample['female'] == sex_val]
    model_sex = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + married + educ_hs + educ_somecol + educ_college',
                         data=df_sex, weights=df_sex['PERWT']).fit(cov_type='HC1')
    print(f"  {sex_name}: coef={model_sex.params['treat_post']:.4f}, SE={model_sex.bse['treat_post']:.4f}, N={len(df_sex):,}")

# Robustness 3: Placebo test (pre-treatment trends)
print("\n--- Robustness 3: Pre-treatment Parallel Trends Check ---")
df_pre = df_sample[df_sample['YEAR'] <= 2011].copy()
df_pre['year_treat'] = df_pre['treat'] * df_pre['YEAR']
model_pretrends = smf.wls('fulltime ~ treat + C(YEAR) + treat:C(YEAR)',
                           data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print("Interaction coefficients (treat x year) in pre-period:")
for param in model_pretrends.params.index:
    if 'treat:' in param:
        print(f"  {param}: {model_pretrends.params[param]:.4f} (p={model_pretrends.pvalues[param]:.4f})")

# Robustness 4: Event study
print("\n--- Robustness 4: Event Study Coefficients ---")
df_sample['year_x_treat'] = df_sample.apply(lambda x: f"year{x['YEAR']}_treat" if x['treat']==1 else 'base', axis=1)

# Create year dummies interacted with treatment
for yr in df_sample['YEAR'].unique():
    df_sample[f'treat_year_{yr}'] = ((df_sample['treat'] == 1) & (df_sample['YEAR'] == yr)).astype(int)

# Use 2011 as reference year
year_vars = [f'treat_year_{yr}' for yr in sorted(df_sample['YEAR'].unique()) if yr != 2011]
formula = 'fulltime ~ treat + C(YEAR) + ' + ' + '.join(year_vars) + ' + female + married + educ_hs + educ_somecol + educ_college'
model_event = smf.wls(formula, data=df_sample, weights=df_sample['PERWT']).fit(cov_type='HC1')

print("Event study coefficients (reference year: 2011):")
event_results = []
for yr in sorted(df_sample['YEAR'].unique()):
    if yr == 2011:
        event_results.append({'year': yr, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
    else:
        var = f'treat_year_{yr}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        ci = model_event.conf_int().loc[var]
        event_results.append({'year': yr, 'coef': coef, 'se': se, 'ci_low': ci[0], 'ci_high': ci[1]})
        print(f"  Year {yr}: coef={coef:.4f}, SE={se:.4f}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

event_df = pd.DataFrame(event_results)

# -----------------------------------------------------------------------------
# Step 7: Create Output Tables for LaTeX
# -----------------------------------------------------------------------------
print("\n\n" + "=" * 80)
print("[Step 7] Creating Output for LaTeX Report")
print("=" * 80)

# Summary statistics table
summary_stats = {
    'Variable': ['Full-time Employment', 'Age', 'Female', 'Married',
                 'Less than HS', 'High School', 'Some College', 'College+'],
    'Treatment Mean': [
        df_sample[df_sample['treat']==1]['fulltime'].mean(),
        df_sample[df_sample['treat']==1]['age'].mean(),
        df_sample[df_sample['treat']==1]['female'].mean(),
        df_sample[df_sample['treat']==1]['married'].mean(),
        df_sample[df_sample['treat']==1]['educ_lesshs'].mean(),
        df_sample[df_sample['treat']==1]['educ_hs'].mean(),
        df_sample[df_sample['treat']==1]['educ_somecol'].mean(),
        df_sample[df_sample['treat']==1]['educ_college'].mean(),
    ],
    'Control Mean': [
        df_sample[df_sample['treat']==0]['fulltime'].mean(),
        df_sample[df_sample['treat']==0]['age'].mean(),
        df_sample[df_sample['treat']==0]['female'].mean(),
        df_sample[df_sample['treat']==0]['married'].mean(),
        df_sample[df_sample['treat']==0]['educ_lesshs'].mean(),
        df_sample[df_sample['treat']==0]['educ_hs'].mean(),
        df_sample[df_sample['treat']==0]['educ_somecol'].mean(),
        df_sample[df_sample['treat']==0]['educ_college'].mean(),
    ]
}
summary_df = pd.DataFrame(summary_stats)
summary_df['Difference'] = summary_df['Treatment Mean'] - summary_df['Control Mean']
print("\nSummary Statistics Table:")
print(summary_df.round(4).to_string(index=False))

# Main results table
print("\n\nMain Regression Results:")
results_table = pd.DataFrame({
    'Model': ['(1) Basic', '(2) Weighted', '(3) + Covariates', '(4) + Year FE', '(5) + State FE', '(6) Clustered SE'],
    'DiD Estimate': [
        model1.params['treat_post'],
        model2.params['treat_post'],
        model3.params['treat_post'],
        model4.params['treat_post'],
        model5.params['treat_post'],
        model6.params['treat_post']
    ],
    'Std. Error': [
        model1.bse['treat_post'],
        model2.bse['treat_post'],
        model3.bse['treat_post'],
        model4.bse['treat_post'],
        model5.bse['treat_post'],
        model6.bse['treat_post']
    ],
    'p-value': [
        model1.pvalues['treat_post'],
        model2.pvalues['treat_post'],
        model3.pvalues['treat_post'],
        model4.pvalues['treat_post'],
        model5.pvalues['treat_post'],
        model6.pvalues['treat_post']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model5.nobs),
        int(model6.nobs)
    ]
})
print(results_table.round(4).to_string(index=False))

# Save key results
print("\n\n" + "=" * 80)
print("PREFERRED ESTIMATE (Model 6 - with clustered standard errors)")
print("=" * 80)
preferred_coef = model6.params['treat_post']
preferred_se = model6.bse['treat_post']
preferred_ci = model6.conf_int().loc['treat_post']
preferred_n = int(model6.nobs)

print(f"\nEffect Size: {preferred_coef:.6f}")
print(f"Standard Error: {preferred_se:.6f}")
print(f"95% Confidence Interval: [{preferred_ci[0]:.6f}, {preferred_ci[1]:.6f}]")
print(f"Sample Size: {preferred_n:,}")
print(f"p-value: {model6.pvalues['treat_post']:.6f}")

# Save results to CSV for LaTeX
summary_df.to_csv('summary_statistics.csv', index=False)
results_table.to_csv('regression_results.csv', index=False)
event_df.to_csv('event_study.csv', index=False)

# DiD table for LaTeX
did_means = pd.DataFrame({
    'Group': ['Treatment (26-30)', 'Treatment (26-30)', 'Control (31-35)', 'Control (31-35)'],
    'Period': ['Pre (2006-2011)', 'Post (2013-2016)', 'Pre (2006-2011)', 'Post (2013-2016)'],
    'Mean FT Employment': [pre_treat, post_treat, pre_control, post_control],
    'N': [
        len(df_sample[(df_sample['treat']==1) & (df_sample['post']==0)]),
        len(df_sample[(df_sample['treat']==1) & (df_sample['post']==1)]),
        len(df_sample[(df_sample['treat']==0) & (df_sample['post']==0)]),
        len(df_sample[(df_sample['treat']==0) & (df_sample['post']==1)])
    ]
})
did_means.to_csv('did_means.csv', index=False)

# Sample flow table (with correct counts)
sample_flow = pd.DataFrame({
    'Step': ['Original ACS data (2006-2016, excl 2012)',
             'Hispanic-Mexican, Born Mexico, Non-citizen, Birth year 1977-1986',
             'Arrived before age 16',
             'In US by 2007 (Final Sample)'],
    'N': [n_original, n_initial, n_after_age_immig, n_final]
})
sample_flow.to_csv('sample_flow.csv', index=False)

# Year-by-year means for plotting
yearly_means = df_sample.groupby(['YEAR', 'treat']).agg(
    mean_fulltime=('fulltime', 'mean'),
    weighted_mean=('fulltime', lambda x: np.average(x, weights=df_sample.loc[x.index, 'PERWT'])),
    n=('fulltime', 'count')
).reset_index()
yearly_means.to_csv('yearly_means.csv', index=False)

# Gender-specific results
gender_results = []
for sex_val, sex_name in [(0, "Male"), (1, "Female")]:
    df_sex = df_sample[df_sample['female'] == sex_val]
    model_sex = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + married + educ_hs + educ_somecol + educ_college',
                         data=df_sex, weights=df_sex['PERWT']).fit(cov_type='HC1')
    gender_results.append({
        'Gender': sex_name,
        'Coefficient': model_sex.params['treat_post'],
        'SE': model_sex.bse['treat_post'],
        'p_value': model_sex.pvalues['treat_post'],
        'N': len(df_sex)
    })
gender_df = pd.DataFrame(gender_results)
gender_df.to_csv('gender_results.csv', index=False)

print("\n\nAnalysis complete. Output files created:")
print("  - summary_statistics.csv")
print("  - regression_results.csv")
print("  - event_study.csv")
print("  - did_means.csv")
print("  - sample_flow.csv")
print("  - yearly_means.csv")
print("  - gender_results.csv")
