"""
DACA Replication Study - Analysis Script
=========================================
Research Question: What was the causal impact of DACA eligibility on
full-time employment among Hispanic-Mexican, Mexican-born individuals?

Identification Strategy: Difference-in-Differences using arrival age cutoff
- Treatment: Arrived before age 16 (DACA eligible)
- Control: Arrived at age 16+ (DACA ineligible)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (excluding 2012 implementation year)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Set paths
DATA_PATH = "data/data.csv"
OUTPUT_PATH = "."

print("=" * 60)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 60)

# =============================================================================
# STEP 1: Load and Filter Data
# =============================================================================
print("\n[1] Loading data (this may take a while due to file size)...")

# Define columns we need
cols_needed = [
    'YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
    'MARST', 'RACE', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EMPSTAT', 'UHRSWORK', 'LABFORCE'
]

# Read data in chunks to filter
chunks = []
chunksize = 500000

for i, chunk in enumerate(pd.read_csv(DATA_PATH, usecols=cols_needed, chunksize=chunksize)):
    # Filter to Mexican-born, Hispanic-Mexican, non-citizens
    filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Hispanic-Mexican
        (chunk['BPL'] == 200) &    # Born in Mexico
        (chunk['CITIZEN'] == 3) &  # Not a citizen
        (chunk['YEAR'] != 2012)    # Exclude 2012 (implementation year)
    ].copy()

    if len(filtered) > 0:
        chunks.append(filtered)

    if (i + 1) % 20 == 0:
        print(f"  Processed {(i + 1) * chunksize:,} rows...")

print("  Combining filtered chunks...")
df = pd.concat(chunks, ignore_index=True)
print(f"  Initial filtered sample: {len(df):,} observations")

# =============================================================================
# STEP 2: Create Analysis Variables
# =============================================================================
print("\n[2] Creating analysis variables...")

# Calculate age at arrival
# YRIMMIG = 0 means N/A (born in US), which shouldn't apply to our sample
df = df[df['YRIMMIG'] > 0].copy()
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA eligibility criteria:
# 1. Arrived before 16th birthday
# 2. Born after June 15, 1981 (not yet 31 as of June 15, 2012) -> BIRTHYR >= 1982
# 3. Arrived by 2007 (continuous presence since June 15, 2007)
# 4. Not a citizen (already filtered)

# Treatment: Eligible for DACA (arrived before age 16, born >= 1982, arrived <= 2007)
df['eligible'] = (
    (df['age_at_arrival'] < 16) &
    (df['BIRTHYR'] >= 1982) &
    (df['YRIMMIG'] <= 2007)
).astype(int)

# Control: Similar but arrived at age 16+ (ineligible due to arrival age)
# We want to compare to similar immigrants who just missed the age cutoff
df['arrived_young'] = (df['age_at_arrival'] < 16).astype(int)

# Post-DACA period indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Outcome: Full-time employment (usually work 35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# In labor force
df['in_labor_force'] = (df['LABFORCE'] == 2).astype(int)

# Create demographic control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # HS or above
df['educ_somecoll'] = (df['EDUC'] >= 7).astype(int)  # Some college or above
df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # Bachelor's or above

# Age squared
df['age_sq'] = df['AGE'] ** 2

# =============================================================================
# STEP 3: Define Analysis Sample
# =============================================================================
print("\n[3] Defining analysis sample...")

# Restrict to working age (18-45)
# and those with valid immigration info
analysis_df = df[
    (df['AGE'] >= 18) &
    (df['AGE'] <= 45) &
    (df['age_at_arrival'] >= 0) &
    (df['age_at_arrival'] <= 40) &
    (df['YRIMMIG'] <= 2011)  # Must have arrived before end of pre-period
].copy()

print(f"  Working-age sample (18-45): {len(analysis_df):,} observations")

# For DiD, we want to compare those who arrived young (<16) vs. older (16+)
# Among those who otherwise meet DACA criteria (born >= 1982, arrived <= 2007)
did_sample = analysis_df[
    (analysis_df['BIRTHYR'] >= 1982) &
    (analysis_df['YRIMMIG'] <= 2007) &
    (analysis_df['age_at_arrival'] <= 25)  # Reasonable upper bound for comparison
].copy()

print(f"  DiD sample (born >= 1982, arrived <= 2007): {len(did_sample):,} observations")

# =============================================================================
# STEP 4: Summary Statistics
# =============================================================================
print("\n[4] Generating summary statistics...")

# Pre-period summary by treatment status
pre_df = did_sample[did_sample['post'] == 0]
post_df = did_sample[did_sample['post'] == 1]

def weighted_stats(data, weight_col='PERWT'):
    """Calculate weighted statistics"""
    stats_dict = {}
    for col in ['fulltime', 'employed', 'in_labor_force', 'AGE', 'female',
                'married', 'educ_hs', 'educ_college']:
        if col in data.columns:
            weights = data[weight_col]
            values = data[col]
            w_mean = np.average(values, weights=weights)
            stats_dict[col] = w_mean
    stats_dict['n'] = len(data)
    stats_dict['weighted_n'] = data[weight_col].sum()
    return stats_dict

# Statistics by treatment and period
groups = {
    'Pre-DACA, Eligible (arrived <16)': pre_df[pre_df['arrived_young'] == 1],
    'Pre-DACA, Ineligible (arrived 16+)': pre_df[pre_df['arrived_young'] == 0],
    'Post-DACA, Eligible (arrived <16)': post_df[post_df['arrived_young'] == 1],
    'Post-DACA, Ineligible (arrived 16+)': post_df[post_df['arrived_young'] == 0]
}

summary_stats = {}
for name, group in groups.items():
    summary_stats[name] = weighted_stats(group)

summary_df = pd.DataFrame(summary_stats).T
print("\nSummary Statistics (Weighted Means):")
print(summary_df.to_string())

# =============================================================================
# STEP 5: Difference-in-Differences Estimation
# =============================================================================
print("\n[5] Running Difference-in-Differences regressions...")

# Create interaction term
did_sample['treat_post'] = did_sample['arrived_young'] * did_sample['post']

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls(
    'fulltime ~ arrived_young + post + treat_post',
    data=did_sample,
    weights=did_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': did_sample['STATEFIP']})
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographics ---")
model2 = smf.wls(
    'fulltime ~ arrived_young + post + treat_post + AGE + age_sq + female + married + educ_hs + educ_college',
    data=did_sample,
    weights=did_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': did_sample['STATEFIP']})
print(model2.summary())

# Model 3: DiD with year and state fixed effects
print("\n--- Model 3: DiD with Year and State Fixed Effects ---")
did_sample['year_fe'] = did_sample['YEAR'].astype(str)
did_sample['state_fe'] = did_sample['STATEFIP'].astype(str)

model3 = smf.wls(
    'fulltime ~ arrived_young + treat_post + AGE + age_sq + female + married + educ_hs + educ_college + C(year_fe) + C(state_fe)',
    data=did_sample,
    weights=did_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': did_sample['STATEFIP']})

# Extract key coefficient
treat_post_coef = model3.params['treat_post']
treat_post_se = model3.bse['treat_post']
treat_post_pval = model3.pvalues['treat_post']
treat_post_ci = model3.conf_int().loc['treat_post']

print(f"\nPreferred Estimate (Model 3 - Full Controls):")
print(f"  Treatment Effect (treat_post): {treat_post_coef:.4f}")
print(f"  Standard Error: {treat_post_se:.4f}")
print(f"  P-value: {treat_post_pval:.4f}")
print(f"  95% CI: [{treat_post_ci[0]:.4f}, {treat_post_ci[1]:.4f}]")

# =============================================================================
# STEP 6: Robustness Checks
# =============================================================================
print("\n[6] Running robustness checks...")

# Robustness 1: Restrict to narrower age-at-arrival window (12-19)
print("\n--- Robustness 1: Narrow age-at-arrival window (12-19) ---")
robust1_sample = did_sample[
    (did_sample['age_at_arrival'] >= 12) &
    (did_sample['age_at_arrival'] <= 19)
].copy()

if len(robust1_sample) > 100:
    robust1 = smf.wls(
        'fulltime ~ arrived_young + treat_post + AGE + age_sq + female + married + educ_hs + educ_college + C(year_fe) + C(state_fe)',
        data=robust1_sample,
        weights=robust1_sample['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': robust1_sample['STATEFIP']})
    print(f"  N = {len(robust1_sample):,}")
    print(f"  Treatment Effect: {robust1.params['treat_post']:.4f} (SE: {robust1.bse['treat_post']:.4f})")
else:
    print("  Insufficient observations for this robustness check")

# Robustness 2: Employment as outcome (instead of full-time)
print("\n--- Robustness 2: Employment as outcome ---")
robust2 = smf.wls(
    'employed ~ arrived_young + treat_post + AGE + age_sq + female + married + educ_hs + educ_college + C(year_fe) + C(state_fe)',
    data=did_sample,
    weights=did_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': did_sample['STATEFIP']})
print(f"  Treatment Effect on Employment: {robust2.params['treat_post']:.4f} (SE: {robust2.bse['treat_post']:.4f})")

# Robustness 3: Labor force participation as outcome
print("\n--- Robustness 3: Labor Force Participation as outcome ---")
robust3 = smf.wls(
    'in_labor_force ~ arrived_young + treat_post + AGE + age_sq + female + married + educ_hs + educ_college + C(year_fe) + C(state_fe)',
    data=did_sample,
    weights=did_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': did_sample['STATEFIP']})
print(f"  Treatment Effect on LFP: {robust3.params['treat_post']:.4f} (SE: {robust3.bse['treat_post']:.4f})")

# Robustness 4: Men only
print("\n--- Robustness 4: Men only ---")
men_sample = did_sample[did_sample['female'] == 0].copy()
robust4 = smf.wls(
    'fulltime ~ arrived_young + treat_post + AGE + age_sq + married + educ_hs + educ_college + C(year_fe) + C(state_fe)',
    data=men_sample,
    weights=men_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': men_sample['STATEFIP']})
print(f"  N = {len(men_sample):,}")
print(f"  Treatment Effect (Men): {robust4.params['treat_post']:.4f} (SE: {robust4.bse['treat_post']:.4f})")

# Robustness 5: Women only
print("\n--- Robustness 5: Women only ---")
women_sample = did_sample[did_sample['female'] == 1].copy()
robust5 = smf.wls(
    'fulltime ~ arrived_young + treat_post + AGE + age_sq + married + educ_hs + educ_college + C(year_fe) + C(state_fe)',
    data=women_sample,
    weights=women_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': women_sample['STATEFIP']})
print(f"  N = {len(women_sample):,}")
print(f"  Treatment Effect (Women): {robust5.params['treat_post']:.4f} (SE: {robust5.bse['treat_post']:.4f})")

# =============================================================================
# STEP 7: Event Study / Parallel Trends Check
# =============================================================================
print("\n[7] Event study for parallel trends...")

# Create year dummies interacted with treatment
did_sample['year_2006'] = (did_sample['YEAR'] == 2006).astype(int)
did_sample['year_2007'] = (did_sample['YEAR'] == 2007).astype(int)
did_sample['year_2008'] = (did_sample['YEAR'] == 2008).astype(int)
did_sample['year_2009'] = (did_sample['YEAR'] == 2009).astype(int)
did_sample['year_2010'] = (did_sample['YEAR'] == 2010).astype(int)
did_sample['year_2011'] = (did_sample['YEAR'] == 2011).astype(int)
did_sample['year_2013'] = (did_sample['YEAR'] == 2013).astype(int)
did_sample['year_2014'] = (did_sample['YEAR'] == 2014).astype(int)
did_sample['year_2015'] = (did_sample['YEAR'] == 2015).astype(int)
did_sample['year_2016'] = (did_sample['YEAR'] == 2016).astype(int)

# Interactions (2011 as reference year)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    did_sample[f'treat_x_{yr}'] = did_sample['arrived_young'] * did_sample[f'year_{yr}']

event_formula = 'fulltime ~ arrived_young + ' + \
    ' + '.join([f'treat_x_{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]) + \
    ' + AGE + age_sq + female + married + educ_hs + educ_college + C(year_fe) + C(state_fe)'

event_model = smf.wls(
    event_formula,
    data=did_sample,
    weights=did_sample['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': did_sample['STATEFIP']})

print("\nEvent Study Coefficients (relative to 2011):")
event_study_results = []
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = event_model.params[f'treat_x_{yr}']
    se = event_model.bse[f'treat_x_{yr}']
    pval = event_model.pvalues[f'treat_x_{yr}']
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    print(f"  {yr}: {coef:.4f} (SE: {se:.4f}, p={pval:.3f})")
    event_study_results.append({
        'year': yr,
        'coef': coef,
        'se': se,
        'ci_low': ci_low,
        'ci_high': ci_high
    })

# =============================================================================
# STEP 8: Save Results for Report
# =============================================================================
print("\n[8] Saving results...")

results = {
    'sample_size': len(did_sample),
    'n_treated_pre': len(pre_df[pre_df['arrived_young'] == 1]),
    'n_control_pre': len(pre_df[pre_df['arrived_young'] == 0]),
    'n_treated_post': len(post_df[post_df['arrived_young'] == 1]),
    'n_control_post': len(post_df[post_df['arrived_young'] == 0]),
    'preferred_estimate': {
        'coefficient': float(treat_post_coef),
        'standard_error': float(treat_post_se),
        'p_value': float(treat_post_pval),
        'ci_lower': float(treat_post_ci[0]),
        'ci_upper': float(treat_post_ci[1])
    },
    'model1_basic': {
        'coefficient': float(model1.params['treat_post']),
        'standard_error': float(model1.bse['treat_post']),
        'r_squared': float(model1.rsquared)
    },
    'model2_demographics': {
        'coefficient': float(model2.params['treat_post']),
        'standard_error': float(model2.bse['treat_post']),
        'r_squared': float(model2.rsquared)
    },
    'model3_full': {
        'coefficient': float(model3.params['treat_post']),
        'standard_error': float(model3.bse['treat_post']),
        'r_squared': float(model3.rsquared)
    },
    'robustness': {
        'employment_outcome': {
            'coefficient': float(robust2.params['treat_post']),
            'standard_error': float(robust2.bse['treat_post'])
        },
        'lfp_outcome': {
            'coefficient': float(robust3.params['treat_post']),
            'standard_error': float(robust3.bse['treat_post'])
        },
        'men_only': {
            'coefficient': float(robust4.params['treat_post']),
            'standard_error': float(robust4.bse['treat_post']),
            'n': len(men_sample)
        },
        'women_only': {
            'coefficient': float(robust5.params['treat_post']),
            'standard_error': float(robust5.bse['treat_post']),
            'n': len(women_sample)
        }
    },
    'event_study': event_study_results,
    'summary_stats': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else int(vv)
                          for kk, vv in v.items()}
                     for k, v in summary_stats.items()}
}

with open('results_39.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to results_39.json")

# =============================================================================
# STEP 9: Create Figures
# =============================================================================
print("\n[9] Creating figures...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Event Study Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    years = [r['year'] for r in event_study_results]
    coefs = [r['coef'] for r in event_study_results]
    ci_lows = [r['ci_low'] for r in event_study_results]
    ci_highs = [r['ci_high'] for r in event_study_results]

    # Add 2011 as reference
    years_plot = years[:5] + [2011] + years[5:]
    coefs_plot = coefs[:5] + [0] + coefs[5:]
    ci_lows_plot = ci_lows[:5] + [0] + ci_lows[5:]
    ci_highs_plot = ci_highs[:5] + [0] + ci_highs[5:]

    ax.errorbar(years_plot, coefs_plot,
                yerr=[np.array(coefs_plot) - np.array(ci_lows_plot),
                      np.array(ci_highs_plot) - np.array(coefs_plot)],
                fmt='o', capsize=5, capthick=2, markersize=8, color='navy')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Effect on Full-Time Employment', fontsize=12)
    ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Reference Year: 2011)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved figure1_event_study.png")

    # Figure 2: Trends in Full-Time Employment by Treatment Group
    fig, ax = plt.subplots(figsize=(10, 6))

    yearly_means = did_sample.groupby(['YEAR', 'arrived_young']).apply(
        lambda x: np.average(x['fulltime'], weights=x['PERWT'])
    ).unstack()

    ax.plot(yearly_means.index, yearly_means[1], 'o-', label='Eligible (arrived <16)',
            color='navy', linewidth=2, markersize=8)
    ax.plot(yearly_means.index, yearly_means[0], 's--', label='Ineligible (arrived 16+)',
            color='darkred', linewidth=2, markersize=8)
    ax.axvline(x=2012, color='gray', linestyle=':', alpha=0.7, label='DACA Implementation')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
    ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved figure2_trends.png")

except ImportError:
    print("  matplotlib not available - skipping figure generation")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nSample Size: {len(did_sample):,}")
print(f"\nPreferred Estimate (DiD with full controls):")
print(f"  Effect of DACA Eligibility on Full-Time Employment: {treat_post_coef:.4f}")
print(f"  Standard Error (clustered by state): {treat_post_se:.4f}")
print(f"  95% Confidence Interval: [{treat_post_ci[0]:.4f}, {treat_post_ci[1]:.4f}]")
print(f"  P-value: {treat_post_pval:.4f}")

if treat_post_pval < 0.05:
    direction = "increased" if treat_post_coef > 0 else "decreased"
    print(f"\n  Interpretation: DACA eligibility {direction} the probability of")
    print(f"  full-time employment by {abs(treat_post_coef)*100:.1f} percentage points.")
else:
    print(f"\n  Interpretation: No statistically significant effect detected at the 5% level.")

print("\nOutput files:")
print("  - results_39.json")
print("  - figure1_event_study.png")
print("  - figure2_trends.png")
