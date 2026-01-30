"""
DACA Replication Study: Effect of DACA Eligibility on Full-Time Employment
Among Hispanic-Mexican Mexican-born individuals in the United States

This script implements a difference-in-differences analysis to estimate the causal
effect of DACA eligibility on full-time employment probability.
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
print("DACA REPLICATION STUDY - INDEPENDENT ANALYSIS")
print("Effect of DACA Eligibility on Full-Time Employment")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")
data_path = "data/data.csv"
df = pd.read_csv(data_path)
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. SAMPLE SELECTION
# =============================================================================
print("\n[2] Sample Selection...")

# Step 2a: Restrict to Hispanic-Mexican ethnicity (HISPAN = 1)
# Per data dict: HISPAN 1 = Mexican
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After restricting to Hispanic-Mexican (HISPAN=1): {len(df_sample):,}")

# Step 2b: Restrict to born in Mexico (BPL = 200)
# Per data dict: BPL 200 = Mexico
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After restricting to Mexico-born (BPL=200): {len(df_sample):,}")

# Step 2c: Restrict to non-citizens (CITIZEN = 3)
# Per data dict: CITIZEN 3 = Not a citizen
# The instructions say: "Assume that anyone who is not a citizen and who has not
# received immigration papers is undocumented for DACA purposes"
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After restricting to non-citizens (CITIZEN=3): {len(df_sample):,}")

# =============================================================================
# 3. DEFINE DACA ELIGIBILITY CRITERIA
# =============================================================================
print("\n[3] Defining DACA Eligibility...")

# DACA eligibility requirements (as of June 15, 2012):
# 1. Arrived in US before 16th birthday
# 2. Not yet 31 years old on June 15, 2012 (born after June 15, 1981)
# 3. Lived continuously in US since June 15, 2007 (immigrated by 2007)
# 4. Were present in US on June 15, 2012 and not a citizen (already filtered)

# Calculate age at immigration
# Age at immigration = YRIMMIG - BIRTHYR
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Define arrived before 16th birthday
df_sample['arrived_before_16'] = df_sample['age_at_immig'] < 16

# Define not yet 31 on June 15, 2012
# Born after June 15, 1981 means BIRTHYR >= 1982
# (conservative: if born in 1981, could be 31 by June 15, 2012)
# More precisely: must be under 31 on June 15, 2012
# If BIRTHYR = 1981, person turns 31 sometime in 1981+31=2012
# We'll use BIRTHYR > 1981 as conservative criterion (born 1982 or later definitely eligible)
# Or BIRTHYR >= 1981 (could be eligible if born after June 15, 1981)
# Given we don't have exact birth dates, use BIRTHYR >= 1981 as more inclusive
df_sample['under_31_in_2012'] = df_sample['BIRTHYR'] >= 1981

# Define continuously present since 2007
# YRIMMIG <= 2007 means they arrived by 2007
df_sample['present_since_2007'] = df_sample['YRIMMIG'] <= 2007

# DACA eligible = all three conditions met
df_sample['daca_eligible'] = (
    df_sample['arrived_before_16'] &
    df_sample['under_31_in_2012'] &
    df_sample['present_since_2007']
)

print(f"\nDACA eligibility breakdown:")
print(f"  Arrived before 16: {df_sample['arrived_before_16'].sum():,} ({100*df_sample['arrived_before_16'].mean():.1f}%)")
print(f"  Under 31 in 2012: {df_sample['under_31_in_2012'].sum():,} ({100*df_sample['under_31_in_2012'].mean():.1f}%)")
print(f"  Present since 2007: {df_sample['present_since_2007'].sum():,} ({100*df_sample['present_since_2007'].mean():.1f}%)")
print(f"  DACA eligible: {df_sample['daca_eligible'].sum():,} ({100*df_sample['daca_eligible'].mean():.1f}%)")

# =============================================================================
# 4. DEFINE OUTCOME VARIABLE
# =============================================================================
print("\n[4] Defining Outcome Variable (Full-Time Employment)...")

# Full-time employment: usually working 35+ hours per week
# UHRSWORK = usual hours worked per week
df_sample['fulltime_employed'] = (df_sample['UHRSWORK'] >= 35).astype(int)

print(f"Full-time employment rate (overall): {100*df_sample['fulltime_employed'].mean():.1f}%")

# =============================================================================
# 5. DEFINE TREATMENT PERIOD
# =============================================================================
print("\n[5] Defining Treatment Period...")

# DACA announced June 15, 2012. Applications started August 15, 2012.
# Per instructions: examine effects in years 2013-2016
# Pre-period: 2006-2011
# Post-period: 2013-2016
# 2012 is ambiguous (DACA announced mid-year), so we exclude it

df_sample['post_daca'] = (df_sample['YEAR'] >= 2013).astype(int)
df_sample['pre_period'] = (df_sample['YEAR'] <= 2011).astype(int)

# Create year indicator for 2012 (excluded from main analysis)
df_sample['year_2012'] = (df_sample['YEAR'] == 2012).astype(int)

print(f"Pre-DACA period (2006-2011): {df_sample['pre_period'].sum():,}")
print(f"Year 2012 (excluded): {df_sample['year_2012'].sum():,}")
print(f"Post-DACA period (2013-2016): {df_sample['post_daca'].sum():,}")

# =============================================================================
# 6. RESTRICT TO WORKING-AGE POPULATION
# =============================================================================
print("\n[6] Restricting to Working-Age Population...")

# Restrict to ages 16-64 (standard working age)
df_analysis = df_sample[(df_sample['AGE'] >= 16) & (df_sample['AGE'] <= 64)].copy()
print(f"After restricting to ages 16-64: {len(df_analysis):,}")

# Exclude 2012 for main analysis (ambiguous treatment year)
df_analysis_main = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis_main):,}")

# =============================================================================
# 7. CREATE ANALYSIS VARIABLES
# =============================================================================
print("\n[7] Creating Analysis Variables...")

# Interaction term (DiD estimator)
df_analysis_main['daca_x_post'] = df_analysis_main['daca_eligible'].astype(int) * df_analysis_main['post_daca']

# Create numeric versions
df_analysis_main['treatment'] = df_analysis_main['daca_eligible'].astype(int)

# Create state fixed effects
df_analysis_main['state'] = df_analysis_main['STATEFIP']

# Create age squared for controls
df_analysis_main['age_sq'] = df_analysis_main['AGE'] ** 2

# Create education categories
df_analysis_main['educ_cat'] = pd.cut(
    df_analysis_main['EDUC'],
    bins=[-1, 2, 6, 10, 11],
    labels=['less_than_hs', 'hs', 'some_college', 'college_plus']
)

# Female indicator
df_analysis_main['female'] = (df_analysis_main['SEX'] == 2).astype(int)

# Married indicator
df_analysis_main['married'] = (df_analysis_main['MARST'] == 1).astype(int)

print(f"Final analysis sample: {len(df_analysis_main):,}")

# =============================================================================
# 8. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[8] Descriptive Statistics...")
print("-" * 60)

# Summary by treatment and period
summary_stats = df_analysis_main.groupby(['treatment', 'post_daca']).agg({
    'fulltime_employed': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'PERWT': 'sum'
}).round(3)

print("\nSummary by Treatment and Period:")
print(summary_stats)

# Pre-trends check
print("\nPre-period trends (2006-2011):")
pre_trends = df_analysis_main[df_analysis_main['post_daca'] == 0].groupby(['YEAR', 'treatment']).agg({
    'fulltime_employed': 'mean',
    'PERWT': 'sum'
}).round(3)
print(pre_trends)

# =============================================================================
# 9. MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "=" * 80)
print("[9] MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls(
    'fulltime_employed ~ treatment + post_daca + daca_x_post',
    data=df_analysis_main,
    weights=df_analysis_main['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_analysis_main['STATEFIP']})
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls(
    'fulltime_employed ~ treatment + post_daca + daca_x_post + AGE + age_sq + female + married',
    data=df_analysis_main,
    weights=df_analysis_main['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_analysis_main['STATEFIP']})
print(model2.summary())

# Model 3: DiD with demographic controls + year fixed effects
print("\n--- Model 3: DiD with Controls + Year FE ---")
model3 = smf.wls(
    'fulltime_employed ~ treatment + daca_x_post + AGE + age_sq + female + married + C(YEAR)',
    data=df_analysis_main,
    weights=df_analysis_main['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_analysis_main['STATEFIP']})
print(model3.summary())

# Model 4: Full model with state and year fixed effects
print("\n--- Model 4: Full Model with State and Year FE (PREFERRED) ---")
model4 = smf.wls(
    'fulltime_employed ~ treatment + daca_x_post + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
    data=df_analysis_main,
    weights=df_analysis_main['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_analysis_main['STATEFIP']})
print(model4.summary())

# =============================================================================
# 10. EVENT STUDY / DYNAMIC EFFECTS
# =============================================================================
print("\n" + "=" * 80)
print("[10] EVENT STUDY ANALYSIS")
print("=" * 80)

# Create year interactions with treatment (relative to 2011)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]  # 2011 is reference
for year in years:
    df_analysis_main[f'treat_x_{year}'] = df_analysis_main['treatment'] * (df_analysis_main['YEAR'] == year).astype(int)

# Event study regression
event_study_formula = 'fulltime_employed ~ treatment + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)'
for year in years:
    event_study_formula += f' + treat_x_{year}'

model_event = smf.wls(
    event_study_formula,
    data=df_analysis_main,
    weights=df_analysis_main['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_analysis_main['STATEFIP']})

print("\nEvent Study Coefficients:")
event_coefs = {}
for year in years:
    coef_name = f'treat_x_{year}'
    coef = model_event.params.get(coef_name, np.nan)
    se = model_event.bse.get(coef_name, np.nan)
    event_coefs[year] = {'coef': coef, 'se': se}
    print(f"  Year {year}: coef = {coef:.4f}, se = {se:.4f}")

# =============================================================================
# 11. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("[11] ROBUSTNESS CHECKS")
print("=" * 80)

# Robustness 1: Alternative age range (18-45)
print("\n--- Robustness 1: Ages 18-45 ---")
df_robust1 = df_analysis_main[(df_analysis_main['AGE'] >= 18) & (df_analysis_main['AGE'] <= 45)].copy()
model_robust1 = smf.wls(
    'fulltime_employed ~ treatment + daca_x_post + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
    data=df_robust1,
    weights=df_robust1['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_robust1['STATEFIP']})
print(f"DiD estimate (ages 18-45): {model_robust1.params['daca_x_post']:.4f} (se: {model_robust1.bse['daca_x_post']:.4f})")
print(f"Sample size: {len(df_robust1):,}")

# Robustness 2: Males only
print("\n--- Robustness 2: Males Only ---")
df_robust2 = df_analysis_main[df_analysis_main['female'] == 0].copy()
model_robust2 = smf.wls(
    'fulltime_employed ~ treatment + daca_x_post + AGE + age_sq + married + C(YEAR) + C(STATEFIP)',
    data=df_robust2,
    weights=df_robust2['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_robust2['STATEFIP']})
print(f"DiD estimate (males): {model_robust2.params['daca_x_post']:.4f} (se: {model_robust2.bse['daca_x_post']:.4f})")
print(f"Sample size: {len(df_robust2):,}")

# Robustness 3: Females only
print("\n--- Robustness 3: Females Only ---")
df_robust3 = df_analysis_main[df_analysis_main['female'] == 1].copy()
model_robust3 = smf.wls(
    'fulltime_employed ~ treatment + daca_x_post + AGE + age_sq + married + C(YEAR) + C(STATEFIP)',
    data=df_robust3,
    weights=df_robust3['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_robust3['STATEFIP']})
print(f"DiD estimate (females): {model_robust3.params['daca_x_post']:.4f} (se: {model_robust3.bse['daca_x_post']:.4f})")
print(f"Sample size: {len(df_robust3):,}")

# Robustness 4: Alternative DACA eligibility definition (more restrictive)
# Must have arrived by 2006 (not 2007)
print("\n--- Robustness 4: Stricter Arrival Requirement (by 2006) ---")
df_robust4 = df_analysis_main.copy()
df_robust4['daca_strict'] = (
    (df_robust4['age_at_immig'] < 16) &
    (df_robust4['BIRTHYR'] >= 1981) &
    (df_robust4['YRIMMIG'] <= 2006)
).astype(int)
df_robust4['daca_strict_x_post'] = df_robust4['daca_strict'] * df_robust4['post_daca']
model_robust4 = smf.wls(
    'fulltime_employed ~ daca_strict + daca_strict_x_post + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
    data=df_robust4,
    weights=df_robust4['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_robust4['STATEFIP']})
print(f"DiD estimate (strict): {model_robust4.params['daca_strict_x_post']:.4f} (se: {model_robust4.bse['daca_strict_x_post']:.4f})")

# Robustness 5: Include 2012 with partial treatment
print("\n--- Robustness 5: Include 2012 ---")
df_robust5 = df_analysis[(df_analysis['AGE'] >= 16) & (df_analysis['AGE'] <= 64)].copy()
df_robust5['treatment'] = df_robust5['daca_eligible'].astype(int)
df_robust5['post_daca'] = (df_robust5['YEAR'] >= 2013).astype(int)
df_robust5['daca_x_post'] = df_robust5['treatment'] * df_robust5['post_daca']
df_robust5['age_sq'] = df_robust5['AGE'] ** 2
df_robust5['female'] = (df_robust5['SEX'] == 2).astype(int)
df_robust5['married'] = (df_robust5['MARST'] == 1).astype(int)
model_robust5 = smf.wls(
    'fulltime_employed ~ treatment + daca_x_post + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
    data=df_robust5,
    weights=df_robust5['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_robust5['STATEFIP']})
print(f"DiD estimate (with 2012): {model_robust5.params['daca_x_post']:.4f} (se: {model_robust5.bse['daca_x_post']:.4f})")
print(f"Sample size: {len(df_robust5):,}")

# =============================================================================
# 12. PLACEBO TEST
# =============================================================================
print("\n" + "=" * 80)
print("[12] PLACEBO TEST")
print("=" * 80)

# Use only pre-period data and create fake treatment in 2009
df_placebo = df_analysis_main[df_analysis_main['YEAR'] <= 2011].copy()
df_placebo['fake_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['fake_did'] = df_placebo['treatment'] * df_placebo['fake_post']

model_placebo = smf.wls(
    'fulltime_employed ~ treatment + fake_post + fake_did + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
    data=df_placebo,
    weights=df_placebo['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_placebo['STATEFIP']})
print(f"Placebo DiD estimate (fake treatment 2009): {model_placebo.params['fake_did']:.4f} (se: {model_placebo.bse['fake_did']:.4f})")
print(f"P-value: {model_placebo.pvalues['fake_did']:.4f}")

# =============================================================================
# 13. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("[13] SUMMARY OF RESULTS")
print("=" * 80)

results_summary = {
    'Model 1 (Basic DiD)': {
        'coef': model1.params['daca_x_post'],
        'se': model1.bse['daca_x_post'],
        'n': int(model1.nobs)
    },
    'Model 2 (w/ Demographics)': {
        'coef': model2.params['daca_x_post'],
        'se': model2.bse['daca_x_post'],
        'n': int(model2.nobs)
    },
    'Model 3 (w/ Year FE)': {
        'coef': model3.params['daca_x_post'],
        'se': model3.bse['daca_x_post'],
        'n': int(model3.nobs)
    },
    'Model 4 (Full - PREFERRED)': {
        'coef': model4.params['daca_x_post'],
        'se': model4.bse['daca_x_post'],
        'n': int(model4.nobs)
    }
}

print("\nMain Results Summary:")
print("-" * 70)
print(f"{'Model':<30} {'Coefficient':>12} {'Std. Error':>12} {'N':>12}")
print("-" * 70)
for model_name, res in results_summary.items():
    ci_low = res['coef'] - 1.96 * res['se']
    ci_high = res['coef'] + 1.96 * res['se']
    print(f"{model_name:<30} {res['coef']:>12.4f} {res['se']:>12.4f} {res['n']:>12,}")
print("-" * 70)

print("\nPreferred Estimate (Model 4):")
preferred_coef = model4.params['daca_x_post']
preferred_se = model4.bse['daca_x_post']
preferred_ci = (preferred_coef - 1.96 * preferred_se, preferred_coef + 1.96 * preferred_se)
preferred_pval = model4.pvalues['daca_x_post']
print(f"  Effect size: {preferred_coef:.4f}")
print(f"  Standard error: {preferred_se:.4f}")
print(f"  95% CI: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"  P-value: {preferred_pval:.4f}")
print(f"  Sample size: {int(model4.nobs):,}")

# =============================================================================
# 14. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n[14] Saving results...")

# Save results to file
results_file = open('analysis_results.txt', 'w')
results_file.write("DACA REPLICATION STUDY - RESULTS\n")
results_file.write("=" * 80 + "\n\n")

results_file.write("PREFERRED ESTIMATE (Model 4: Full model with state and year FE)\n")
results_file.write("-" * 60 + "\n")
results_file.write(f"Effect size: {preferred_coef:.4f}\n")
results_file.write(f"Standard error: {preferred_se:.4f}\n")
results_file.write(f"95% CI: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]\n")
results_file.write(f"P-value: {preferred_pval:.4f}\n")
results_file.write(f"Sample size: {int(model4.nobs):,}\n\n")

results_file.write("Interpretation:\n")
if preferred_coef > 0:
    results_file.write(f"DACA eligibility is associated with a {preferred_coef*100:.2f} percentage point\n")
    results_file.write(f"increase in the probability of full-time employment.\n")
else:
    results_file.write(f"DACA eligibility is associated with a {abs(preferred_coef)*100:.2f} percentage point\n")
    results_file.write(f"decrease in the probability of full-time employment.\n")

if preferred_pval < 0.05:
    results_file.write("This effect is statistically significant at the 5% level.\n")
elif preferred_pval < 0.10:
    results_file.write("This effect is marginally significant at the 10% level.\n")
else:
    results_file.write("This effect is not statistically significant.\n")

results_file.close()

# Save key statistics for LaTeX tables
import json

stats_for_latex = {
    'preferred': {
        'coef': float(preferred_coef),
        'se': float(preferred_se),
        'ci_low': float(preferred_ci[0]),
        'ci_high': float(preferred_ci[1]),
        'pval': float(preferred_pval),
        'n': int(model4.nobs)
    },
    'models': {},
    'robustness': {},
    'event_study': event_coefs,
    'placebo': {
        'coef': float(model_placebo.params['fake_did']),
        'se': float(model_placebo.bse['fake_did']),
        'pval': float(model_placebo.pvalues['fake_did'])
    },
    'sample_stats': {
        'total_obs': len(df),
        'hispanic_mexican': len(df[df['HISPAN'] == 1]),
        'mexico_born': len(df[(df['HISPAN'] == 1) & (df['BPL'] == 200)]),
        'non_citizen': len(df_sample),
        'working_age': len(df_analysis),
        'final_sample': len(df_analysis_main),
        'n_daca_eligible': int(df_analysis_main['daca_eligible'].sum()),
        'n_control': int((~df_analysis_main['daca_eligible']).sum())
    }
}

for name, res in results_summary.items():
    stats_for_latex['models'][name] = {
        'coef': float(res['coef']),
        'se': float(res['se']),
        'n': res['n']
    }

stats_for_latex['robustness'] = {
    'ages_18_45': {
        'coef': float(model_robust1.params['daca_x_post']),
        'se': float(model_robust1.bse['daca_x_post']),
        'n': len(df_robust1)
    },
    'males_only': {
        'coef': float(model_robust2.params['daca_x_post']),
        'se': float(model_robust2.bse['daca_x_post']),
        'n': len(df_robust2)
    },
    'females_only': {
        'coef': float(model_robust3.params['daca_x_post']),
        'se': float(model_robust3.bse['daca_x_post']),
        'n': len(df_robust3)
    },
    'strict_eligibility': {
        'coef': float(model_robust4.params['daca_strict_x_post']),
        'se': float(model_robust4.bse['daca_strict_x_post'])
    },
    'with_2012': {
        'coef': float(model_robust5.params['daca_x_post']),
        'se': float(model_robust5.bse['daca_x_post']),
        'n': len(df_robust5)
    }
}

# Convert numpy types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    return obj

stats_for_latex = convert_numpy(stats_for_latex)

with open('stats_for_latex.json', 'w') as f:
    json.dump(stats_for_latex, f, indent=2)

# Save descriptive statistics
desc_stats = df_analysis_main.groupby(['treatment', 'post_daca']).agg({
    'fulltime_employed': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')

# Pre-trends table
pre_trends_by_year = df_analysis_main[df_analysis_main['post_daca'] == 0].groupby(['YEAR', 'treatment'])['fulltime_employed'].mean().unstack()
pre_trends_by_year.to_csv('pre_trends.csv')

# Post-trends table
post_trends_by_year = df_analysis_main[df_analysis_main['post_daca'] == 1].groupby(['YEAR', 'treatment'])['fulltime_employed'].mean().unstack()
post_trends_by_year.to_csv('post_trends.csv')

print("\nResults saved to:")
print("  - analysis_results.txt")
print("  - stats_for_latex.json")
print("  - descriptive_stats.csv")
print("  - pre_trends.csv")
print("  - post_trends.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
