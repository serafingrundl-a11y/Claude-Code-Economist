"""
DACA Replication Study - Analysis Script
Replication 50

Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born non-citizens in the United States.
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
print("DACA REPLICATION STUDY - Analysis Script")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")

# Read the CSV file
df = pd.read_csv('data/data.csv')
print(f"   Total observations loaded: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# 2. FILTER TO TARGET POPULATION
# ============================================================================
print("\n2. Filtering to target population...")

# Filter to Hispanic-Mexican (HISPAN = 1)
df_filtered = df[df['HISPAN'] == 1].copy()
print(f"   After Hispanic-Mexican filter: {len(df_filtered):,}")

# Filter to born in Mexico (BPL = 200)
df_filtered = df_filtered[df_filtered['BPL'] == 200].copy()
print(f"   After Mexico birthplace filter: {len(df_filtered):,}")

# Filter to non-citizens (CITIZEN = 3)
df_filtered = df_filtered[df_filtered['CITIZEN'] == 3].copy()
print(f"   After non-citizen filter: {len(df_filtered):,}")

# ============================================================================
# 3. CONSTRUCT KEY VARIABLES
# ============================================================================
print("\n3. Constructing key variables...")

# Age at immigration (approximate)
# YRIMMIG is year of immigration, BIRTHYR is birth year
df_filtered['age_at_immigration'] = df_filtered['YRIMMIG'] - df_filtered['BIRTHYR']
print(f"   Created age_at_immigration variable")

# For cases where YRIMMIG is 0 or missing, we cannot determine eligibility precisely
# Keep only those with valid immigration year
df_filtered = df_filtered[df_filtered['YRIMMIG'] > 0].copy()
print(f"   After removing missing immigration year: {len(df_filtered):,}")

# Arrived before age 16 (DACA requirement)
df_filtered['arrived_before_16'] = (df_filtered['age_at_immigration'] < 16).astype(int)
print(f"   Arrived before age 16: {df_filtered['arrived_before_16'].sum():,} ({df_filtered['arrived_before_16'].mean()*100:.1f}%)")

# In US since at least 2007 (continuous residence requirement)
df_filtered['in_us_since_2007'] = (df_filtered['YRIMMIG'] <= 2007).astype(int)
print(f"   In US since 2007: {df_filtered['in_us_since_2007'].sum():,} ({df_filtered['in_us_since_2007'].mean()*100:.1f}%)")

# Age as of June 15, 2012
# Use BIRTHYR and assume mid-year (approximate)
df_filtered['age_june_2012'] = 2012 - df_filtered['BIRTHYR']

# Under 31 as of June 15, 2012 (born after June 15, 1981)
# Being conservative: born 1982 or later means definitely under 31
# Born in 1981 could be under or over depending on birth month
# Use BIRTHQTR to be more precise
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Those born in 1981 Q3 or Q4 (after June 15) would be under 31
df_filtered['under_31_june_2012'] = (
    (df_filtered['BIRTHYR'] >= 1982) |
    ((df_filtered['BIRTHYR'] == 1981) & (df_filtered['BIRTHQTR'].isin([3, 4])))
).astype(int)
print(f"   Under 31 as of June 2012: {df_filtered['under_31_june_2012'].sum():,} ({df_filtered['under_31_june_2012'].mean()*100:.1f}%)")

# ============================================================================
# 4. DEFINE DACA ELIGIBILITY
# ============================================================================
print("\n4. Defining DACA eligibility...")

# DACA eligible: meets all criteria
# - Non-citizen (already filtered)
# - Born in Mexico (already filtered)
# - Hispanic-Mexican (already filtered)
# - Arrived before age 16
# - Under 31 as of June 2012
# - In US since at least 2007
df_filtered['daca_eligible'] = (
    (df_filtered['arrived_before_16'] == 1) &
    (df_filtered['under_31_june_2012'] == 1) &
    (df_filtered['in_us_since_2007'] == 1)
).astype(int)

print(f"   DACA eligible: {df_filtered['daca_eligible'].sum():,} ({df_filtered['daca_eligible'].mean()*100:.1f}%)")

# ============================================================================
# 5. DEFINE CONTROL GROUP
# ============================================================================
print("\n5. Defining control group...")

# Control group: Similar individuals who are NOT DACA-eligible due to age
# They arrived before 16 and have been in US since 2007, but were over 31 in 2012
# Age range: 31-40 as of June 2012 (born 1972-1981 Q1-Q2)

df_filtered['control_group'] = (
    (df_filtered['arrived_before_16'] == 1) &
    (df_filtered['in_us_since_2007'] == 1) &
    (df_filtered['under_31_june_2012'] == 0) &
    (df_filtered['age_june_2012'] <= 40)  # Born 1972 or later
).astype(int)

print(f"   Control group (31-40 in 2012): {df_filtered['control_group'].sum():,} ({df_filtered['control_group'].mean()*100:.1f}%)")

# Create analysis sample: DACA eligible + control group
df_analysis = df_filtered[(df_filtered['daca_eligible'] == 1) | (df_filtered['control_group'] == 1)].copy()
print(f"   Analysis sample size: {len(df_analysis):,}")

# ============================================================================
# 6. CONSTRUCT OUTCOME VARIABLE
# ============================================================================
print("\n6. Constructing outcome variable...")

# Full-time employment: usually working 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A)
# First need to be in labor force age (16-64 typically)

# Restrict to working age (18-55 to have comparable ages across years)
df_analysis = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 55)].copy()
print(f"   After age restriction (18-55): {len(df_analysis):,}")

# Full-time employment indicator
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)
print(f"   Full-time employed: {df_analysis['fulltime'].sum():,} ({df_analysis['fulltime'].mean()*100:.1f}%)")

# Employment indicator (any employment)
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)
print(f"   Employed: {df_analysis['employed'].sum():,} ({df_analysis['employed'].mean()*100:.1f}%)")

# ============================================================================
# 7. DEFINE TREATMENT PERIOD
# ============================================================================
print("\n7. Defining treatment period...")

# DACA announced June 15, 2012
# Pre-period: 2006-2011
# Post-period: 2013-2016 (excluding 2012 as transition year)

df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
df_analysis['treat'] = df_analysis['daca_eligible']  # Treatment group indicator

print(f"   Pre-period (2006-2011) observations: {len(df_analysis[df_analysis['post']==0]):,}")
print(f"   Post-period (2013-2016) observations: {len(df_analysis[df_analysis['post']==1]):,}")

# Exclude 2012 (transition year)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_analysis):,}")

# ============================================================================
# 8. SUMMARY STATISTICS
# ============================================================================
print("\n8. Summary Statistics")
print("=" * 80)

# Summary by treatment group and period
summary_stats = df_analysis.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean', 'std'],
    'AGE': ['mean', 'std'],
    'PERWT': 'sum'
}).round(3)

print("\nSummary by Treatment Group and Period:")
print(summary_stats)

# Save summary statistics
summary_df = df_analysis.groupby(['treat', 'post']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'AGE': 'mean',
    'PERWT': ['sum', 'count']
}).reset_index()
summary_df.columns = ['Treatment', 'Post', 'FullTime_Rate', 'Employment_Rate', 'Mean_Age', 'Sum_Weight', 'N_Obs']
summary_df.to_csv('summary_statistics.csv', index=False)
print("\n   Summary statistics saved to summary_statistics.csv")

# ============================================================================
# 9. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n9. Difference-in-Differences Analysis")
print("=" * 80)

# Create interaction term
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

# Basic DiD without controls
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model1.summary())

# DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
# Create control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # High school or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # College or more

model2 = smf.wls('fulltime ~ treat + post + treat_post + AGE + I(AGE**2) + female + married + educ_hs',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# DiD with year fixed effects
print("\n--- Model 3: DiD with year fixed effects ---")
df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)
model3 = smf.wls('fulltime ~ treat + treat_post + C(year_factor) + AGE + I(AGE**2) + female + married + educ_hs',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# DiD with state fixed effects
print("\n--- Model 4: DiD with state and year fixed effects ---")
df_analysis['state_factor'] = df_analysis['STATEFIP'].astype(str)
model4 = smf.wls('fulltime ~ treat + treat_post + C(year_factor) + C(state_factor) + AGE + I(AGE**2) + female + married + educ_hs',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

# Print just the key coefficients for model 4
print("\nModel 4 Key Coefficients:")
print(f"   treat_post (DiD estimate): {model4.params['treat_post']:.4f} (SE: {model4.bse['treat_post']:.4f})")
print(f"   treat: {model4.params['treat']:.4f} (SE: {model4.bse['treat']:.4f})")

# ============================================================================
# 10. ROBUSTNESS CHECKS
# ============================================================================
print("\n10. Robustness Checks")
print("=" * 80)

# Alternative outcome: Employment (any)
print("\n--- Robustness 1: Employment (any) as outcome ---")
model_emp = smf.wls('employed ~ treat + treat_post + C(year_factor) + AGE + I(AGE**2) + female + married + educ_hs',
                     data=df_analysis,
                     weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"   treat_post (DiD estimate): {model_emp.params['treat_post']:.4f} (SE: {model_emp.bse['treat_post']:.4f})")

# Alternative control group: Narrower age band (31-35)
print("\n--- Robustness 2: Narrower control group (age 31-35 in 2012) ---")
df_narrow = df_analysis[
    (df_analysis['daca_eligible'] == 1) |
    ((df_analysis['control_group'] == 1) & (df_analysis['age_june_2012'] <= 35))
].copy()
df_narrow['treat_post'] = df_narrow['treat'] * df_narrow['post']

if len(df_narrow) > 100:
    model_narrow = smf.wls('fulltime ~ treat + treat_post + C(year_factor) + AGE + I(AGE**2) + female + married + educ_hs',
                            data=df_narrow,
                            weights=df_narrow['PERWT']).fit(cov_type='HC1')
    print(f"   treat_post (DiD estimate): {model_narrow.params['treat_post']:.4f} (SE: {model_narrow.bse['treat_post']:.4f})")
    print(f"   Sample size: {len(df_narrow):,}")

# Men only
print("\n--- Robustness 3: Men only ---")
df_men = df_analysis[df_analysis['female'] == 0].copy()
model_men = smf.wls('fulltime ~ treat + treat_post + C(year_factor) + AGE + I(AGE**2) + married + educ_hs',
                     data=df_men,
                     weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"   treat_post (DiD estimate): {model_men.params['treat_post']:.4f} (SE: {model_men.bse['treat_post']:.4f})")
print(f"   Sample size: {len(df_men):,}")

# Women only
print("\n--- Robustness 4: Women only ---")
df_women = df_analysis[df_analysis['female'] == 1].copy()
model_women = smf.wls('fulltime ~ treat + treat_post + C(year_factor) + AGE + I(AGE**2) + married + educ_hs',
                       data=df_women,
                       weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"   treat_post (DiD estimate): {model_women.params['treat_post']:.4f} (SE: {model_women.bse['treat_post']:.4f})")
print(f"   Sample size: {len(df_women):,}")

# ============================================================================
# 11. EVENT STUDY / PARALLEL TRENDS
# ============================================================================
print("\n11. Event Study Analysis (Parallel Trends Check)")
print("=" * 80)

# Create year interactions
years = sorted(df_analysis['YEAR'].unique())
base_year = 2011  # Last pre-treatment year as reference

for year in years:
    if year != base_year:
        df_analysis[f'treat_year_{year}'] = (df_analysis['treat'] * (df_analysis['YEAR'] == year)).astype(int)

# Event study regression
year_vars = ' + '.join([f'treat_year_{y}' for y in years if y != base_year])
formula = f'fulltime ~ treat + C(year_factor) + {year_vars} + AGE + I(AGE**2) + female + married + educ_hs'

model_event = smf.wls(formula, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

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
# 12. SAVE RESULTS
# ============================================================================
print("\n12. Saving Results")
print("=" * 80)

# Collect main results
results = {
    'Model': ['Basic DiD', 'DiD + Demographics', 'DiD + Year FE', 'DiD + Year & State FE',
              'Employment Outcome', 'Narrow Control', 'Men Only', 'Women Only'],
    'DiD_Estimate': [
        model1.params['treat_post'],
        model2.params['treat_post'],
        model3.params['treat_post'],
        model4.params['treat_post'],
        model_emp.params['treat_post'],
        model_narrow.params['treat_post'] if len(df_narrow) > 100 else np.nan,
        model_men.params['treat_post'],
        model_women.params['treat_post']
    ],
    'SE': [
        model1.bse['treat_post'],
        model2.bse['treat_post'],
        model3.bse['treat_post'],
        model4.bse['treat_post'],
        model_emp.bse['treat_post'],
        model_narrow.bse['treat_post'] if len(df_narrow) > 100 else np.nan,
        model_men.bse['treat_post'],
        model_women.bse['treat_post']
    ],
    'p_value': [
        model1.pvalues['treat_post'],
        model2.pvalues['treat_post'],
        model3.pvalues['treat_post'],
        model4.pvalues['treat_post'],
        model_emp.pvalues['treat_post'],
        model_narrow.pvalues['treat_post'] if len(df_narrow) > 100 else np.nan,
        model_men.pvalues['treat_post'],
        model_women.pvalues['treat_post']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model_emp.nobs),
        len(df_narrow) if len(df_narrow) > 100 else np.nan,
        len(df_men),
        len(df_women)
    ]
}

results_df = pd.DataFrame(results)
results_df['CI_lower'] = results_df['DiD_Estimate'] - 1.96 * results_df['SE']
results_df['CI_upper'] = results_df['DiD_Estimate'] + 1.96 * results_df['SE']
results_df.to_csv('regression_results.csv', index=False)
print("   Regression results saved to regression_results.csv")

# Event study results
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
event_df.to_csv('event_study_results.csv', index=False)
print("   Event study results saved to event_study_results.csv")

# ============================================================================
# 13. PRINT FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\nPreferred Estimate (Model 3: DiD with Year FE and Demographics):")
print(f"   Effect of DACA eligibility on full-time employment: {model3.params['treat_post']:.4f}")
print(f"   Standard Error: {model3.bse['treat_post']:.4f}")
print(f"   95% CI: [{model3.params['treat_post'] - 1.96*model3.bse['treat_post']:.4f}, {model3.params['treat_post'] + 1.96*model3.bse['treat_post']:.4f}]")
print(f"   p-value: {model3.pvalues['treat_post']:.4f}")
print(f"   Sample Size: {int(model3.nobs):,}")

print("\nInterpretation:")
effect_pct = model3.params['treat_post'] * 100
print(f"   DACA eligibility is associated with a {effect_pct:.2f} percentage point")
if model3.params['treat_post'] > 0:
    print("   increase in the probability of full-time employment.")
else:
    print("   decrease in the probability of full-time employment.")

# Statistical significance
if model3.pvalues['treat_post'] < 0.01:
    print("   This effect is statistically significant at the 1% level.")
elif model3.pvalues['treat_post'] < 0.05:
    print("   This effect is statistically significant at the 5% level.")
elif model3.pvalues['treat_post'] < 0.1:
    print("   This effect is statistically significant at the 10% level.")
else:
    print("   This effect is not statistically significant at conventional levels.")

print("\n" + "=" * 80)
print("Analysis Complete")
print("=" * 80)
