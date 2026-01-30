#!/usr/bin/env python3
"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the United States.

Uses Difference-in-Differences identification strategy.
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

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n[Step 1] Loading data...")

# Specify dtypes to reduce memory usage
dtypes = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int8',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'YRSUSA1': 'float32',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8',
    'MARST': 'int8',
}

# Load data in chunks for memory efficiency
chunks = []
print("Loading data in chunks...")
for chunk in pd.read_csv('data/data.csv', chunksize=500000, dtype=dtypes,
                          usecols=['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE',
                                   'BIRTHQTR', 'BIRTHYR', 'HISPAN', 'BPL', 'BPLD',
                                   'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT',
                                   'UHRSWORK', 'MARST']):
    # Filter to Hispanic-Mexican born in Mexico
    # HISPAN == 1 (Mexican) and BPL == 200 (Mexico)
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk)
    print(f"  Processed chunk, {len(chunk)} Hispanic-Mexican Mexican-born obs retained")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal observations after filtering to Hispanic-Mexican Mexican-born: {len(df):,}")

# =============================================================================
# STEP 2: Create Key Variables
# =============================================================================
print("\n[Step 2] Creating key variables...")

# Exclude 2012 (DACA implemented mid-year June 15, 2012)
df = df[df['YEAR'] != 2012].copy()
print(f"Observations after excluding 2012: {len(df):,}")

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Full-time employment outcome (working 35+ hours per week)
# Need to be employed to be working full-time
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)

# Calculate age at immigration
# YRIMMIG = 0 means N/A (native born or missing), so we need to handle this
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# =============================================================================
# STEP 3: Define DACA Eligibility
# =============================================================================
print("\n[Step 3] Defining DACA eligibility...")

"""
DACA Eligibility Requirements:
1. Arrived in US before 16th birthday: age_at_immig < 16
2. Under 31 as of June 15, 2012: birth year >= 1982 (born after June 15, 1981)
3. Continuous residence since June 15, 2007: immigrated by 2007 (YRIMMIG <= 2007)
4. Present in US on June 15, 2012 without lawful status: non-citizen (CITIZEN == 3)
5. Not a citizen: CITIZEN == 3

Notes:
- We cannot observe undocumented status directly; assume non-citizens without
  naturalization are potentially undocumented
- We use conservative thresholds given data limitations
"""

# Non-citizen (proxy for undocumented)
df['noncitizen'] = (df['CITIZEN'] == 3).astype(int)

# Filter to non-citizens only for the main analysis
df_noncit = df[df['noncitizen'] == 1].copy()
print(f"Observations of non-citizen Hispanic-Mexican Mexican-born: {len(df_noncit):,}")

# DACA eligibility criteria
# 1. Arrived before age 16
df_noncit['arrived_before_16'] = (df_noncit['age_at_immig'] < 16).astype(int)

# 2. Under 31 as of June 15, 2012 (born in 1982 or later)
df_noncit['under31_june2012'] = (df_noncit['BIRTHYR'] >= 1982).astype(int)

# 3. In US since at least 2007 (immigrated by 2007)
df_noncit['in_us_since_2007'] = (df_noncit['YRIMMIG'] <= 2007).astype(int)

# 4. Valid immigration year (not 0 or missing)
df_noncit['valid_immig'] = (df_noncit['YRIMMIG'] > 0).astype(int)

# DACA eligible = all criteria met
df_noncit['daca_eligible'] = (
    (df_noncit['arrived_before_16'] == 1) &
    (df_noncit['under31_june2012'] == 1) &
    (df_noncit['in_us_since_2007'] == 1) &
    (df_noncit['valid_immig'] == 1)
).astype(int)

print(f"\nDACA eligibility breakdown:")
print(f"  Arrived before age 16: {df_noncit['arrived_before_16'].sum():,}")
print(f"  Under 31 as of June 2012: {df_noncit['under31_june2012'].sum():,}")
print(f"  In US since 2007: {df_noncit['in_us_since_2007'].sum():,}")
print(f"  Valid immigration year: {df_noncit['valid_immig'].sum():,}")
print(f"  DACA Eligible: {df_noncit['daca_eligible'].sum():,}")
print(f"  Not DACA Eligible: {(df_noncit['daca_eligible'] == 0).sum():,}")

# =============================================================================
# STEP 4: Create Control Group
# =============================================================================
print("\n[Step 4] Defining control group...")

"""
Control Group Strategy:
We compare DACA-eligible individuals to non-eligible Hispanic-Mexican Mexican-born
non-citizens. The key variation is the age/arrival timing requirements.

Control group: Those who arrived at age 16+ OR were 31+ as of June 2012 OR
arrived after 2007, but are otherwise similar (Hispanic-Mexican, Mexican-born, non-citizen).
"""

# For cleaner identification, focus on working-age population (16-64)
df_noncit = df_noncit[(df_noncit['AGE'] >= 16) & (df_noncit['AGE'] <= 64)].copy()
print(f"Working age (16-64) non-citizen sample: {len(df_noncit):,}")

# Create treatment variable
df_noncit['treated'] = df_noncit['daca_eligible']

# DiD interaction
df_noncit['did'] = df_noncit['treated'] * df_noncit['post']

print(f"\nSample by treatment and period:")
print(df_noncit.groupby(['treated', 'post']).size().unstack())

# =============================================================================
# STEP 5: Summary Statistics
# =============================================================================
print("\n[Step 5] Summary Statistics...")

def weighted_mean(x, w):
    """Calculate weighted mean."""
    return np.average(x, weights=w) if len(x) > 0 else np.nan

def weighted_std(x, w):
    """Calculate weighted standard deviation."""
    avg = np.average(x, weights=w)
    variance = np.average((x - avg)**2, weights=w)
    return np.sqrt(variance)

# Summary stats by treatment group
print("\n" + "=" * 60)
print("TABLE 1: Summary Statistics by Treatment Status")
print("=" * 60)

for treat in [0, 1]:
    group = df_noncit[df_noncit['treated'] == treat]
    group_pre = group[group['post'] == 0]
    group_post = group[group['post'] == 1]

    label = "DACA Eligible" if treat == 1 else "Not DACA Eligible"
    print(f"\n{label} (N = {len(group):,})")
    print("-" * 40)

    for period, data in [("Pre-DACA", group_pre), ("Post-DACA", group_post)]:
        if len(data) > 0:
            ft_mean = weighted_mean(data['fulltime'], data['PERWT'])
            age_mean = weighted_mean(data['AGE'], data['PERWT'])
            male_mean = weighted_mean((data['SEX'] == 1).astype(int), data['PERWT'])
            print(f"  {period}:")
            print(f"    Full-time employed: {ft_mean:.3f}")
            print(f"    Mean age: {age_mean:.1f}")
            print(f"    Male share: {male_mean:.3f}")
            print(f"    N: {len(data):,}")

# =============================================================================
# STEP 6: Difference-in-Differences Estimation
# =============================================================================
print("\n[Step 6] Difference-in-Differences Estimation...")

# Model 1: Basic DiD
print("\n" + "=" * 60)
print("MODEL 1: Basic Difference-in-Differences")
print("=" * 60)

model1 = smf.wls('fulltime ~ treated + post + did',
                  data=df_noncit,
                  weights=df_noncit['PERWT']).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n" + "=" * 60)
print("MODEL 2: DiD with Demographic Controls")
print("=" * 60)

# Create control variables
df_noncit['male'] = (df_noncit['SEX'] == 1).astype(int)
df_noncit['age_sq'] = df_noncit['AGE'] ** 2
df_noncit['married'] = (df_noncit['MARST'] <= 2).astype(int)

# Education categories
df_noncit['educ_hs'] = (df_noncit['EDUC'] >= 6).astype(int)  # HS or more
df_noncit['educ_col'] = (df_noncit['EDUC'] >= 10).astype(int)  # College or more

model2 = smf.wls('fulltime ~ treated + post + did + AGE + age_sq + male + married + educ_hs + educ_col',
                  data=df_noncit,
                  weights=df_noncit['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with year fixed effects
print("\n" + "=" * 60)
print("MODEL 3: DiD with Year Fixed Effects")
print("=" * 60)

# Create year dummies
df_noncit['year'] = df_noncit['YEAR'].astype(str)

model3 = smf.wls('fulltime ~ treated + did + C(year) + AGE + age_sq + male + married + educ_hs + educ_col',
                  data=df_noncit,
                  weights=df_noncit['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with state fixed effects
print("\n" + "=" * 60)
print("MODEL 4: DiD with Year and State Fixed Effects")
print("=" * 60)

df_noncit['state'] = df_noncit['STATEFIP'].astype(str)

model4 = smf.wls('fulltime ~ treated + did + C(year) + C(state) + AGE + age_sq + male + married + educ_hs + educ_col',
                  data=df_noncit,
                  weights=df_noncit['PERWT']).fit(cov_type='HC1')
print("\nDiD coefficient (Model 4 - Preferred):")
print(f"  Estimate: {model4.params['did']:.4f}")
print(f"  Std. Error: {model4.bse['did']:.4f}")
print(f"  t-stat: {model4.tvalues['did']:.4f}")
print(f"  p-value: {model4.pvalues['did']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['did', 0]:.4f}, {model4.conf_int().loc['did', 1]:.4f}]")

# =============================================================================
# STEP 7: Robustness Checks
# =============================================================================
print("\n[Step 7] Robustness Checks...")

# Robustness 1: Alternative age range (18-45)
print("\n" + "=" * 60)
print("ROBUSTNESS 1: Restricted Age Range (18-45)")
print("=" * 60)

df_robust1 = df_noncit[(df_noncit['AGE'] >= 18) & (df_noncit['AGE'] <= 45)].copy()
model_r1 = smf.wls('fulltime ~ treated + did + C(year) + C(state) + AGE + age_sq + male + married + educ_hs + educ_col',
                    data=df_robust1,
                    weights=df_robust1['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r1.params['did']:.4f} (SE: {model_r1.bse['did']:.4f})")
print(f"N = {len(df_robust1):,}")

# Robustness 2: Men only
print("\n" + "=" * 60)
print("ROBUSTNESS 2: Men Only")
print("=" * 60)

df_men = df_noncit[df_noncit['male'] == 1].copy()
model_r2 = smf.wls('fulltime ~ treated + did + C(year) + C(state) + AGE + age_sq + married + educ_hs + educ_col',
                    data=df_men,
                    weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r2.params['did']:.4f} (SE: {model_r2.bse['did']:.4f})")
print(f"N = {len(df_men):,}")

# Robustness 3: Women only
print("\n" + "=" * 60)
print("ROBUSTNESS 3: Women Only")
print("=" * 60)

df_women = df_noncit[df_noncit['male'] == 0].copy()
model_r3 = smf.wls('fulltime ~ treated + did + C(year) + C(state) + AGE + age_sq + married + educ_hs + educ_col',
                    data=df_women,
                    weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r3.params['did']:.4f} (SE: {model_r3.bse['did']:.4f})")
print(f"N = {len(df_women):,}")

# =============================================================================
# STEP 8: Event Study / Pre-trends
# =============================================================================
print("\n[Step 8] Event Study Analysis...")

print("\n" + "=" * 60)
print("EVENT STUDY: Year-by-Year Effects")
print("=" * 60)

# Create year interactions with treatment
years = sorted(df_noncit['YEAR'].unique())
ref_year = 2011  # Reference year (last pre-treatment year)

for yr in years:
    if yr != ref_year:
        df_noncit[f'treat_x_{yr}'] = (df_noncit['treated'] * (df_noncit['YEAR'] == yr)).astype(int)

interaction_terms = ' + '.join([f'treat_x_{yr}' for yr in years if yr != ref_year])
event_formula = f'fulltime ~ treated + C(year) + C(state) + AGE + age_sq + male + married + educ_hs + educ_col + {interaction_terms}'

model_event = smf.wls(event_formula,
                       data=df_noncit,
                       weights=df_noncit['PERWT']).fit(cov_type='HC1')

print("\nYear-by-Treatment Interaction Coefficients:")
print("-" * 50)
for yr in sorted(years):
    if yr != ref_year:
        coef = model_event.params[f'treat_x_{yr}']
        se = model_event.bse[f'treat_x_{yr}']
        pval = model_event.pvalues[f'treat_x_{yr}']
        print(f"  {yr}: {coef:7.4f} (SE: {se:.4f}) p = {pval:.4f}")
    else:
        print(f"  {yr}: Reference year")

# =============================================================================
# STEP 9: Save Results for Report
# =============================================================================
print("\n[Step 9] Saving results...")

results = {
    'preferred_estimate': model4.params['did'],
    'preferred_se': model4.bse['did'],
    'preferred_ci_low': model4.conf_int().loc['did', 0],
    'preferred_ci_high': model4.conf_int().loc['did', 1],
    'preferred_pval': model4.pvalues['did'],
    'sample_size': len(df_noncit),
    'n_treated': df_noncit['treated'].sum(),
    'n_control': (df_noncit['treated'] == 0).sum(),
}

# Save to file
with open('results_summary.txt', 'w') as f:
    f.write("DACA Replication Analysis - Summary Results\n")
    f.write("=" * 60 + "\n\n")
    f.write("PREFERRED SPECIFICATION (Model 4: Year + State FE)\n")
    f.write("-" * 60 + "\n")
    f.write(f"DiD Coefficient (Effect on Full-Time Employment): {results['preferred_estimate']:.4f}\n")
    f.write(f"Standard Error: {results['preferred_se']:.4f}\n")
    f.write(f"95% Confidence Interval: [{results['preferred_ci_low']:.4f}, {results['preferred_ci_high']:.4f}]\n")
    f.write(f"P-value: {results['preferred_pval']:.4f}\n")
    f.write(f"\nSample Size: {results['sample_size']:,}\n")
    f.write(f"Treatment Group (DACA Eligible): {results['n_treated']:,}\n")
    f.write(f"Control Group (Not DACA Eligible): {results['n_control']:,}\n")

    # Model comparison
    f.write("\n" + "=" * 60 + "\n")
    f.write("MODEL COMPARISON\n")
    f.write("-" * 60 + "\n")
    f.write(f"Model 1 (Basic DiD):           {model1.params['did']:.4f} (SE: {model1.bse['did']:.4f})\n")
    f.write(f"Model 2 (+ Demographics):      {model2.params['did']:.4f} (SE: {model2.bse['did']:.4f})\n")
    f.write(f"Model 3 (+ Year FE):           {model3.params['did']:.4f} (SE: {model3.bse['did']:.4f})\n")
    f.write(f"Model 4 (+ State FE):          {model4.params['did']:.4f} (SE: {model4.bse['did']:.4f})\n")

    # Robustness
    f.write("\n" + "=" * 60 + "\n")
    f.write("ROBUSTNESS CHECKS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Age 18-45:                     {model_r1.params['did']:.4f} (SE: {model_r1.bse['did']:.4f})\n")
    f.write(f"Men Only:                      {model_r2.params['did']:.4f} (SE: {model_r2.bse['did']:.4f})\n")
    f.write(f"Women Only:                    {model_r3.params['did']:.4f} (SE: {model_r3.bse['did']:.4f})\n")

print("\nResults saved to results_summary.txt")

# =============================================================================
# STEP 10: Generate LaTeX Tables
# =============================================================================
print("\n[Step 10] Generating LaTeX tables...")

# Create summary statistics table
def create_summary_table(df, weights_col='PERWT'):
    """Create summary statistics table."""
    pre_treat = df[(df['post'] == 0) & (df['treated'] == 1)]
    post_treat = df[(df['post'] == 1) & (df['treated'] == 1)]
    pre_control = df[(df['post'] == 0) & (df['treated'] == 0)]
    post_control = df[(df['post'] == 1) & (df['treated'] == 0)]

    def calc_stats(data, var):
        return weighted_mean(data[var], data[weights_col])

    stats_dict = {}
    for var in ['fulltime', 'AGE', 'male', 'married', 'educ_hs']:
        stats_dict[var] = {
            'Pre-Treat': calc_stats(pre_treat, var),
            'Post-Treat': calc_stats(post_treat, var),
            'Pre-Control': calc_stats(pre_control, var),
            'Post-Control': calc_stats(post_control, var),
        }
    return pd.DataFrame(stats_dict).T

summary_df = create_summary_table(df_noncit)
print("\nSummary Statistics Table:")
print(summary_df.round(3))

# Save detailed results
summary_df.to_csv('summary_statistics.csv')

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nPreferred Estimate: {results['preferred_estimate']:.4f}")
print(f"Standard Error: {results['preferred_se']:.4f}")
print(f"Sample Size: {results['sample_size']:,}")
