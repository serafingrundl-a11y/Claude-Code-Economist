"""
DACA Replication Study - Analysis Script v2
Replication 50

Alternative specification addressing pre-trends concern.
Using arrival-age based identification in addition to birth-year cutoff.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 80)
print("DACA REPLICATION STUDY - Analysis Script v2")
print("Alternative Identification Strategy")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations loaded: {len(df):,}")

# ============================================================================
# 2. FILTER TO TARGET POPULATION
# ============================================================================
print("\n2. Filtering to target population...")

# Hispanic-Mexican, born in Mexico, non-citizen
df_filtered = df[
    (df['HISPAN'] == 1) &
    (df['BPL'] == 200) &
    (df['CITIZEN'] == 3)
].copy()
print(f"   Mexican-born Hispanic non-citizens: {len(df_filtered):,}")

# Valid immigration year
df_filtered = df_filtered[df_filtered['YRIMMIG'] > 0].copy()
print(f"   With valid immigration year: {len(df_filtered):,}")

# ============================================================================
# 3. CONSTRUCT VARIABLES
# ============================================================================
print("\n3. Constructing variables...")

# Age at immigration
df_filtered['age_at_immigration'] = df_filtered['YRIMMIG'] - df_filtered['BIRTHYR']

# DACA criteria
df_filtered['arrived_before_16'] = (df_filtered['age_at_immigration'] < 16).astype(int)
df_filtered['in_us_since_2007'] = (df_filtered['YRIMMIG'] <= 2007).astype(int)

# Age as of June 15, 2012
df_filtered['age_june_2012'] = 2012 - df_filtered['BIRTHYR']

# Under 31 as of June 15, 2012
df_filtered['under_31_june_2012'] = (
    (df_filtered['BIRTHYR'] >= 1982) |
    ((df_filtered['BIRTHYR'] == 1981) & (df_filtered['BIRTHQTR'].isin([3, 4])))
).astype(int)

# ============================================================================
# 4. ALTERNATIVE IDENTIFICATION: ARRIVAL AGE CUTOFF
# ============================================================================
print("\n4. Alternative Identification Strategy")
print("=" * 80)
print("""
Strategy: Compare those who arrived just before age 16 vs just after age 16.
Both groups are similar in terms of:
- Being Mexican-born Hispanic non-citizens
- Having been in US for similar duration
- Similar background characteristics

But only those arriving before age 16 are DACA-eligible.

Treatment: Arrived at age 12-15 (just under the cutoff)
Control: Arrived at age 16-19 (just over the cutoff)

Restrict to those who:
- Were under 31 as of June 2012 (meet age requirement)
- Were in US since at least 2007 (meet residence requirement)
""")

# Both groups must meet age and residence requirements
df_eligible_age = df_filtered[
    (df_filtered['under_31_june_2012'] == 1) &
    (df_filtered['in_us_since_2007'] == 1)
].copy()

# Treatment: arrived at age 12-15 (DACA eligible by arrival age)
# Control: arrived at age 16-19 (not DACA eligible)
df_eligible_age['treat_arrival'] = (
    (df_eligible_age['age_at_immigration'] >= 12) &
    (df_eligible_age['age_at_immigration'] <= 15)
).astype(int)

df_eligible_age['control_arrival'] = (
    (df_eligible_age['age_at_immigration'] >= 16) &
    (df_eligible_age['age_at_immigration'] <= 19)
).astype(int)

# Analysis sample
df_arrival = df_eligible_age[
    (df_eligible_age['treat_arrival'] == 1) |
    (df_eligible_age['control_arrival'] == 1)
].copy()

# Restrict to working age
df_arrival = df_arrival[(df_arrival['AGE'] >= 18) & (df_arrival['AGE'] <= 55)].copy()

# Exclude 2012
df_arrival = df_arrival[df_arrival['YEAR'] != 2012].copy()

print(f"\nSample sizes:")
print(f"   Treatment (arrived age 12-15): {df_arrival['treat_arrival'].sum():,}")
print(f"   Control (arrived age 16-19): {df_arrival['control_arrival'].sum():,}")
print(f"   Total: {len(df_arrival):,}")

# Outcome and treatment variables
df_arrival['fulltime'] = (df_arrival['UHRSWORK'] >= 35).astype(int)
df_arrival['employed'] = (df_arrival['EMPSTAT'] == 1).astype(int)
df_arrival['post'] = (df_arrival['YEAR'] >= 2013).astype(int)
df_arrival['treat'] = df_arrival['treat_arrival']
df_arrival['treat_post'] = df_arrival['treat'] * df_arrival['post']

# Controls
df_arrival['female'] = (df_arrival['SEX'] == 2).astype(int)
df_arrival['married'] = (df_arrival['MARST'] == 1).astype(int)
df_arrival['educ_hs'] = (df_arrival['EDUC'] >= 6).astype(int)
df_arrival['year_factor'] = df_arrival['YEAR'].astype(str)

# ============================================================================
# 5. SUMMARY STATISTICS - ARRIVAL AGE DESIGN
# ============================================================================
print("\n5. Summary Statistics - Arrival Age Design")
print("=" * 80)

summary = df_arrival.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'age_at_immigration': 'mean',
    'PERWT': 'sum'
}).round(3)
print(summary)

# ============================================================================
# 6. DIFFERENCE-IN-DIFFERENCES - ARRIVAL AGE DESIGN
# ============================================================================
print("\n6. DiD Analysis - Arrival Age Design")
print("=" * 80)

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_arrival,
                  weights=df_arrival['PERWT']).fit(cov_type='HC1')
print(f"   treat_post: {model1.params['treat_post']:.4f} (SE: {model1.bse['treat_post']:.4f}, p: {model1.pvalues['treat_post']:.4f})")

# Model 2: With controls
print("\n--- Model 2: DiD with controls ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post + AGE + I(AGE**2) + female + married + educ_hs',
                  data=df_arrival,
                  weights=df_arrival['PERWT']).fit(cov_type='HC1')
print(f"   treat_post: {model2.params['treat_post']:.4f} (SE: {model2.bse['treat_post']:.4f}, p: {model2.pvalues['treat_post']:.4f})")

# Model 3: With year FE
print("\n--- Model 3: DiD with year FE ---")
model3 = smf.wls('fulltime ~ treat + treat_post + C(year_factor) + AGE + I(AGE**2) + female + married + educ_hs',
                  data=df_arrival,
                  weights=df_arrival['PERWT']).fit(cov_type='HC1')
print(f"   treat_post: {model3.params['treat_post']:.4f} (SE: {model3.bse['treat_post']:.4f}, p: {model3.pvalues['treat_post']:.4f})")

# Full model output for Model 3
print("\nFull Model 3 Output:")
print(model3.summary())

# ============================================================================
# 7. EVENT STUDY - ARRIVAL AGE DESIGN
# ============================================================================
print("\n7. Event Study - Arrival Age Design")
print("=" * 80)

years = sorted(df_arrival['YEAR'].unique())
base_year = 2011

for year in years:
    if year != base_year:
        df_arrival[f'treat_year_{year}'] = (df_arrival['treat'] * (df_arrival['YEAR'] == year)).astype(int)

year_vars = ' + '.join([f'treat_year_{y}' for y in years if y != base_year])
formula = f'fulltime ~ treat + C(year_factor) + {year_vars} + AGE + I(AGE**2) + female + married + educ_hs'

model_event = smf.wls(formula, data=df_arrival, weights=df_arrival['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Treatment × Year):")
print("(Base year: 2011)")
for year in sorted(years):
    if year != base_year:
        coef = model_event.params[f'treat_year_{year}']
        se = model_event.bse[f'treat_year_{year}']
        pval = model_event.pvalues[f'treat_year_{year}']
        sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"   {year}: {coef:7.4f} (SE: {se:.4f}) {sig}")

# ============================================================================
# 8. ROBUSTNESS - NARROWER BANDWIDTH
# ============================================================================
print("\n8. Robustness: Narrower Arrival Age Bandwidth")
print("=" * 80)

# Even tighter comparison: arrived age 14-15 vs 16-17
df_narrow = df_eligible_age[
    ((df_eligible_age['age_at_immigration'] >= 14) & (df_eligible_age['age_at_immigration'] <= 15)) |
    ((df_eligible_age['age_at_immigration'] >= 16) & (df_eligible_age['age_at_immigration'] <= 17))
].copy()

df_narrow = df_narrow[(df_narrow['AGE'] >= 18) & (df_narrow['AGE'] <= 55)].copy()
df_narrow = df_narrow[df_narrow['YEAR'] != 2012].copy()

df_narrow['treat'] = (df_narrow['age_at_immigration'] <= 15).astype(int)
df_narrow['fulltime'] = (df_narrow['UHRSWORK'] >= 35).astype(int)
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['treat_post'] = df_narrow['treat'] * df_narrow['post']
df_narrow['female'] = (df_narrow['SEX'] == 2).astype(int)
df_narrow['married'] = (df_narrow['MARST'] == 1).astype(int)
df_narrow['educ_hs'] = (df_narrow['EDUC'] >= 6).astype(int)
df_narrow['year_factor'] = df_narrow['YEAR'].astype(str)

print(f"\nSample sizes (narrow bandwidth):")
print(f"   Treatment (arrived age 14-15): {df_narrow['treat'].sum():,}")
print(f"   Control (arrived age 16-17): {(1-df_narrow['treat']).sum():,}")
print(f"   Total: {len(df_narrow):,}")

model_narrow = smf.wls('fulltime ~ treat + treat_post + C(year_factor) + AGE + I(AGE**2) + female + married + educ_hs',
                        data=df_narrow,
                        weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"\n   treat_post: {model_narrow.params['treat_post']:.4f} (SE: {model_narrow.bse['treat_post']:.4f}, p: {model_narrow.pvalues['treat_post']:.4f})")

# ============================================================================
# 9. GENDER SUBGROUP ANALYSIS
# ============================================================================
print("\n9. Gender Subgroup Analysis")
print("=" * 80)

# Men
df_men = df_arrival[df_arrival['female'] == 0].copy()
model_men = smf.wls('fulltime ~ treat + treat_post + C(year_factor) + AGE + I(AGE**2) + married + educ_hs',
                     data=df_men,
                     weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"\nMen (N={len(df_men):,}):")
print(f"   treat_post: {model_men.params['treat_post']:.4f} (SE: {model_men.bse['treat_post']:.4f}, p: {model_men.pvalues['treat_post']:.4f})")

# Women
df_women = df_arrival[df_arrival['female'] == 1].copy()
model_women = smf.wls('fulltime ~ treat + treat_post + C(year_factor) + AGE + I(AGE**2) + married + educ_hs',
                       data=df_women,
                       weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"\nWomen (N={len(df_women):,}):")
print(f"   treat_post: {model_women.params['treat_post']:.4f} (SE: {model_women.bse['treat_post']:.4f}, p: {model_women.pvalues['treat_post']:.4f})")

# ============================================================================
# 10. EMPLOYMENT (ANY) OUTCOME
# ============================================================================
print("\n10. Alternative Outcome: Any Employment")
print("=" * 80)

model_emp = smf.wls('employed ~ treat + treat_post + C(year_factor) + AGE + I(AGE**2) + female + married + educ_hs',
                     data=df_arrival,
                     weights=df_arrival['PERWT']).fit(cov_type='HC1')
print(f"   treat_post: {model_emp.params['treat_post']:.4f} (SE: {model_emp.bse['treat_post']:.4f}, p: {model_emp.pvalues['treat_post']:.4f})")

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================
print("\n11. Saving Results")
print("=" * 80)

# Save arrival age design results
results_arrival = {
    'Model': ['Basic DiD', 'DiD + Controls', 'DiD + Year FE', 'Narrow Bandwidth',
              'Men Only', 'Women Only', 'Employment Outcome'],
    'DiD_Estimate': [
        model1.params['treat_post'],
        model2.params['treat_post'],
        model3.params['treat_post'],
        model_narrow.params['treat_post'],
        model_men.params['treat_post'],
        model_women.params['treat_post'],
        model_emp.params['treat_post']
    ],
    'SE': [
        model1.bse['treat_post'],
        model2.bse['treat_post'],
        model3.bse['treat_post'],
        model_narrow.bse['treat_post'],
        model_men.bse['treat_post'],
        model_women.bse['treat_post'],
        model_emp.bse['treat_post']
    ],
    'p_value': [
        model1.pvalues['treat_post'],
        model2.pvalues['treat_post'],
        model3.pvalues['treat_post'],
        model_narrow.pvalues['treat_post'],
        model_men.pvalues['treat_post'],
        model_women.pvalues['treat_post'],
        model_emp.pvalues['treat_post']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model_narrow.nobs),
        len(df_men),
        len(df_women),
        int(model_emp.nobs)
    ]
}

results_df = pd.DataFrame(results_arrival)
results_df['CI_lower'] = results_df['DiD_Estimate'] - 1.96 * results_df['SE']
results_df['CI_upper'] = results_df['DiD_Estimate'] + 1.96 * results_df['SE']
results_df.to_csv('regression_results_arrival_age.csv', index=False)
print("   Results saved to regression_results_arrival_age.csv")

# Event study for arrival age design
event_results = []
for year in sorted(years):
    if year != base_year:
        event_results.append({
            'Year': year,
            'Coefficient': model_event.params[f'treat_year_{year}'],
            'SE': model_event.bse[f'treat_year_{year}'],
            'p_value': model_event.pvalues[f'treat_year_{year}']
        })
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_arrival_age.csv', index=False)
print("   Event study results saved to event_study_arrival_age.csv")

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY - ARRIVAL AGE DESIGN")
print("=" * 80)

print("\nPreferred Estimate (Model 3: DiD with Year FE):")
print(f"   Effect of DACA eligibility on full-time employment: {model3.params['treat_post']:.4f}")
print(f"   Standard Error: {model3.bse['treat_post']:.4f}")
ci_low = model3.params['treat_post'] - 1.96*model3.bse['treat_post']
ci_high = model3.params['treat_post'] + 1.96*model3.bse['treat_post']
print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"   p-value: {model3.pvalues['treat_post']:.4f}")
print(f"   Sample Size: {int(model3.nobs):,}")

print("\n" + "=" * 80)
print("Analysis Complete")
print("=" * 80)
