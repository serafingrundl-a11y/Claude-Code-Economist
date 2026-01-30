"""
DACA Replication Study Analysis
Research Question: Impact of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the US.

Author: Replication 08
Date: 2025-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD DATA (in chunks to handle large file)
# =============================================================================
print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

print("\n1. Loading data in chunks...")

# Define columns we need to reduce memory
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
           'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
           'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK']

# Read data in chunks and filter immediately
chunks = []
chunk_size = 1000000  # 1 million rows at a time

for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunk_size):
    # Filter to Hispanic-Mexican (HISPAN == 1) and born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"   Processed chunk, kept {len(filtered):,} rows")

df = pd.concat(chunks, ignore_index=True)
print(f"\n   Total observations after filtering: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. ADDITIONAL SAMPLE RESTRICTIONS
# =============================================================================
print("\n2. Applying additional sample restrictions...")

print(f"   Hispanic-Mexican, Mexico-born: {len(df):,}")

# Exclude year 2012 (DACA implemented mid-year on June 15, 2012)
df = df[df['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df):,}")

# Keep non-citizens only (for potential DACA eligibility)
# CITIZEN == 3 means "Not a citizen"
df_noncitizen = df[df['CITIZEN'] == 3].copy()
print(f"   Non-citizens (potential DACA-eligible pool): {len(df_noncitizen):,}")

# =============================================================================
# 3. CONSTRUCT DACA ELIGIBILITY
# =============================================================================
print("\n3. Constructing DACA eligibility indicator...")

# DACA Eligibility Criteria:
# 1. Arrived before 16th birthday: age at immigration < 16
# 2. Under 31 as of June 15, 2012: born after June 15, 1981
#    - Conservative: BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
# 3. Continuous presence since June 15, 2007: YRIMMIG <= 2007
# 4. Not a citizen (already filtered above)

# Calculate age at immigration
df_noncitizen['age_at_immig'] = df_noncitizen['YRIMMIG'] - df_noncitizen['BIRTHYR']

# Criterion 1: Arrived before age 16
df_noncitizen['arrived_before_16'] = (df_noncitizen['age_at_immig'] < 16) & (df_noncitizen['age_at_immig'] >= 0)

# Criterion 2: Under 31 as of June 15, 2012
# Born on or after June 16, 1981
# Using BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# To be under 31 on June 15, 2012: born after June 15, 1981
# Conservative approach: BIRTHYR >= 1982 or (BIRTHYR == 1981 and BIRTHQTR >= 3)
df_noncitizen['under_31_2012'] = (df_noncitizen['BIRTHYR'] >= 1982) | \
                                  ((df_noncitizen['BIRTHYR'] == 1981) & (df_noncitizen['BIRTHQTR'] >= 3))

# Criterion 3: Continuous presence since June 15, 2007
# YRIMMIG <= 2007
df_noncitizen['continuous_presence'] = df_noncitizen['YRIMMIG'] <= 2007

# Criterion 4: Already filtered to non-citizens

# Combined DACA eligibility
df_noncitizen['daca_eligible'] = (df_noncitizen['arrived_before_16'] &
                                   df_noncitizen['under_31_2012'] &
                                   df_noncitizen['continuous_presence']).astype(int)

print(f"   DACA-eligible: {df_noncitizen['daca_eligible'].sum():,}")
print(f"   Not DACA-eligible: {(df_noncitizen['daca_eligible'] == 0).sum():,}")

# =============================================================================
# 4. CONSTRUCT OUTCOME AND TREATMENT VARIABLES
# =============================================================================
print("\n4. Constructing outcome and treatment variables...")

# Outcome: Full-time employment (UHRSWORK >= 35)
df_noncitizen['full_time'] = (df_noncitizen['UHRSWORK'] >= 35).astype(int)

# Post-DACA indicator
df_noncitizen['post'] = (df_noncitizen['YEAR'] >= 2013).astype(int)

# Interaction term
df_noncitizen['daca_x_post'] = df_noncitizen['daca_eligible'] * df_noncitizen['post']

print(f"   Full-time employed: {df_noncitizen['full_time'].sum():,}")
print(f"   Post-DACA period observations: {df_noncitizen['post'].sum():,}")

# =============================================================================
# 5. RESTRICT TO WORKING-AGE POPULATION
# =============================================================================
print("\n5. Restricting to working-age population...")

# Restrict to ages 18-64 (working age)
df_analysis = df_noncitizen[(df_noncitizen['AGE'] >= 18) & (df_noncitizen['AGE'] <= 64)].copy()
print(f"   Working-age (18-64) sample: {len(df_analysis):,}")

# Additional restriction: those with valid YRIMMIG (not 0 or missing)
df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0].copy()
print(f"   With valid immigration year: {len(df_analysis):,}")

# =============================================================================
# 6. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n6. Descriptive Statistics")
print("-" * 60)

# Summary by eligibility and period
summary = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'full_time': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(4)
print("\nMean full-time employment by group:")
print(summary)

# Pre-post means by group
print("\n" + "=" * 60)
print("FULL-TIME EMPLOYMENT RATES BY GROUP AND PERIOD")
print("=" * 60)

for elig in [0, 1]:
    elig_label = "DACA-Eligible" if elig == 1 else "Non-Eligible"
    print(f"\n{elig_label}:")
    for period in [0, 1]:
        period_label = "Pre-DACA (2006-2011)" if period == 0 else "Post-DACA (2013-2016)"
        subset = df_analysis[(df_analysis['daca_eligible'] == elig) & (df_analysis['post'] == period)]
        mean_ft = subset['full_time'].mean()
        n = len(subset)
        weighted_mean = np.average(subset['full_time'], weights=subset['PERWT'])
        print(f"   {period_label}: {mean_ft:.4f} (unweighted), {weighted_mean:.4f} (weighted), N={n:,}")

# =============================================================================
# 7. MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "=" * 60)
print("7. MAIN DiD REGRESSION RESULTS")
print("=" * 60)

# Model 1: Basic DiD (unweighted)
print("\nModel 1: Basic DiD (unweighted)")
model1 = smf.ols('full_time ~ daca_eligible + post + daca_x_post', data=df_analysis).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']}
)
print(model1.summary2().tables[1])

# Model 2: Basic DiD (weighted)
print("\nModel 2: Basic DiD (weighted)")
model2 = smf.wls('full_time ~ daca_eligible + post + daca_x_post',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']}
)
print(model2.summary2().tables[1])

# Model 3: DiD with demographic controls (unweighted)
print("\nModel 3: DiD with demographic controls (unweighted)")
# Create age squared and education categories
df_analysis['age_sq'] = df_analysis['AGE'] ** 2
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

# Education categories based on EDUC
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # HS or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # 4+ years college

model3 = smf.ols('full_time ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_hs',
                  data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(model3.summary2().tables[1])

# Model 4: DiD with demographic controls (weighted)
print("\nModel 4: DiD with demographic controls and state FE (weighted)")
# Add state fixed effects
df_analysis['state_fe'] = df_analysis['STATEFIP'].astype('category')

model4 = smf.wls('full_time ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_hs + C(STATEFIP)',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']}
)
# Print only key coefficients
print("\nKey coefficients from Model 4:")
print(f"   daca_eligible:    {model4.params['daca_eligible']:.4f} (SE: {model4.bse['daca_eligible']:.4f})")
print(f"   post:             {model4.params['post']:.4f} (SE: {model4.bse['post']:.4f})")
print(f"   daca_x_post:      {model4.params['daca_x_post']:.4f} (SE: {model4.bse['daca_x_post']:.4f})")

# =============================================================================
# 8. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 60)
print("8. ROBUSTNESS CHECKS")
print("=" * 60)

# Robustness 1: Different age restrictions (16-35 to focus on younger workers)
print("\nRobustness 1: Ages 16-35 only")
df_young = df_noncitizen[(df_noncitizen['AGE'] >= 16) & (df_noncitizen['AGE'] <= 35)].copy()
df_young = df_young[df_young['YRIMMIG'] > 0].copy()
df_young['age_sq'] = df_young['AGE'] ** 2
df_young['female'] = (df_young['SEX'] == 2).astype(int)
df_young['married'] = (df_young['MARST'] == 1).astype(int)
df_young['educ_hs'] = (df_young['EDUC'] >= 6).astype(int)

if len(df_young) > 100:
    model_young = smf.wls('full_time ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_hs',
                          data=df_young, weights=df_young['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': df_young['STATEFIP']}
    )
    print(f"   Sample size: {len(df_young):,}")
    print(f"   daca_x_post coefficient: {model_young.params['daca_x_post']:.4f}")
    print(f"   Standard error: {model_young.bse['daca_x_post']:.4f}")
    print(f"   p-value: {model_young.pvalues['daca_x_post']:.4f}")

# Robustness 2: Men only
print("\nRobustness 2: Men only")
df_men = df_analysis[df_analysis['SEX'] == 1].copy()
if len(df_men) > 100:
    model_men = smf.wls('full_time ~ daca_eligible + post + daca_x_post + AGE + age_sq + married + educ_hs',
                        data=df_men, weights=df_men['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': df_men['STATEFIP']}
    )
    print(f"   Sample size: {len(df_men):,}")
    print(f"   daca_x_post coefficient: {model_men.params['daca_x_post']:.4f}")
    print(f"   Standard error: {model_men.bse['daca_x_post']:.4f}")
    print(f"   p-value: {model_men.pvalues['daca_x_post']:.4f}")

# Robustness 3: Women only
print("\nRobustness 3: Women only")
df_women = df_analysis[df_analysis['SEX'] == 2].copy()
if len(df_women) > 100:
    model_women = smf.wls('full_time ~ daca_eligible + post + daca_x_post + AGE + age_sq + married + educ_hs',
                          data=df_women, weights=df_women['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': df_women['STATEFIP']}
    )
    print(f"   Sample size: {len(df_women):,}")
    print(f"   daca_x_post coefficient: {model_women.params['daca_x_post']:.4f}")
    print(f"   Standard error: {model_women.bse['daca_x_post']:.4f}")
    print(f"   p-value: {model_women.pvalues['daca_x_post']:.4f}")

# Robustness 4: Employment (any) instead of full-time
print("\nRobustness 4: Any employment (EMPSTAT == 1)")
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)
model_emp = smf.wls('employed ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_hs',
                    data=df_analysis, weights=df_analysis['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']}
)
print(f"   Sample size: {len(df_analysis):,}")
print(f"   daca_x_post coefficient: {model_emp.params['daca_x_post']:.4f}")
print(f"   Standard error: {model_emp.bse['daca_x_post']:.4f}")
print(f"   p-value: {model_emp.pvalues['daca_x_post']:.4f}")

# =============================================================================
# 9. EVENT STUDY (Year-by-Year Effects)
# =============================================================================
print("\n" + "=" * 60)
print("9. EVENT STUDY ANALYSIS")
print("=" * 60)

# Create year dummies
df_analysis['year_factor'] = pd.Categorical(df_analysis['YEAR'])
years = sorted(df_analysis['YEAR'].unique())

# Interaction of DACA eligible with each year (reference: 2011)
for year in years:
    if year != 2011:  # 2011 as reference year
        df_analysis[f'daca_x_{year}'] = ((df_analysis['daca_eligible'] == 1) &
                                          (df_analysis['YEAR'] == year)).astype(int)

# Build formula
year_interactions = ' + '.join([f'daca_x_{year}' for year in years if year != 2011])
formula_es = f'full_time ~ daca_eligible + C(YEAR) + {year_interactions} + AGE + age_sq + female + married + educ_hs'

model_es = smf.wls(formula_es, data=df_analysis, weights=df_analysis['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']}
)

print("\nYear-by-Year DACA Eligibility Effects (reference: 2011):")
print("-" * 50)
for year in years:
    if year != 2011:
        var_name = f'daca_x_{year}'
        coef = model_es.params[var_name]
        se = model_es.bse[var_name]
        pval = model_es.pvalues[var_name]
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"   {year}: {coef:7.4f} (SE: {se:.4f}) {sig}")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("10. SUMMARY OF KEY RESULTS")
print("=" * 60)

# Preferred specification: Model 4
print("\nPREFERRED ESTIMATE (Model 4: Weighted DiD with controls and state FE)")
print("-" * 60)
print(f"Effect of DACA eligibility on full-time employment (DiD):")
print(f"   Coefficient:     {model4.params['daca_x_post']:.4f}")
print(f"   Standard Error:  {model4.bse['daca_x_post']:.4f}")
print(f"   95% CI:          [{model4.conf_int().loc['daca_x_post', 0]:.4f}, {model4.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"   p-value:         {model4.pvalues['daca_x_post']:.4f}")
print(f"   Sample Size:     {int(model4.nobs):,}")

# Save summary statistics to file
summary_stats = pd.DataFrame({
    'DACA_Eligible': [0, 0, 1, 1],
    'Post_Period': [0, 1, 0, 1],
    'Mean_FullTime': [
        df_analysis[(df_analysis['daca_eligible']==0) & (df_analysis['post']==0)]['full_time'].mean(),
        df_analysis[(df_analysis['daca_eligible']==0) & (df_analysis['post']==1)]['full_time'].mean(),
        df_analysis[(df_analysis['daca_eligible']==1) & (df_analysis['post']==0)]['full_time'].mean(),
        df_analysis[(df_analysis['daca_eligible']==1) & (df_analysis['post']==1)]['full_time'].mean()
    ],
    'N': [
        len(df_analysis[(df_analysis['daca_eligible']==0) & (df_analysis['post']==0)]),
        len(df_analysis[(df_analysis['daca_eligible']==0) & (df_analysis['post']==1)]),
        len(df_analysis[(df_analysis['daca_eligible']==1) & (df_analysis['post']==0)]),
        len(df_analysis[(df_analysis['daca_eligible']==1) & (df_analysis['post']==1)])
    ]
})
summary_stats.to_csv('summary_stats.csv', index=False)
print("\nSummary statistics saved to 'summary_stats.csv'")

# Calculate simple DiD by hand for verification
pre_control = df_analysis[(df_analysis['daca_eligible']==0) & (df_analysis['post']==0)]['full_time'].mean()
post_control = df_analysis[(df_analysis['daca_eligible']==0) & (df_analysis['post']==1)]['full_time'].mean()
pre_treat = df_analysis[(df_analysis['daca_eligible']==1) & (df_analysis['post']==0)]['full_time'].mean()
post_treat = df_analysis[(df_analysis['daca_eligible']==1) & (df_analysis['post']==1)]['full_time'].mean()

did_manual = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\nManual DiD calculation:")
print(f"   Control group change: {post_control:.4f} - {pre_control:.4f} = {post_control - pre_control:.4f}")
print(f"   Treatment group change: {post_treat:.4f} - {pre_treat:.4f} = {post_treat - pre_treat:.4f}")
print(f"   DiD estimate: {did_manual:.4f}")

# Save regression results
results_df = pd.DataFrame({
    'Model': ['Basic DiD', 'Weighted DiD', 'With Controls', 'With Controls + State FE'],
    'DiD_Coefficient': [model1.params['daca_x_post'], model2.params['daca_x_post'],
                        model3.params['daca_x_post'], model4.params['daca_x_post']],
    'Std_Error': [model1.bse['daca_x_post'], model2.bse['daca_x_post'],
                  model3.bse['daca_x_post'], model4.bse['daca_x_post']],
    'p_value': [model1.pvalues['daca_x_post'], model2.pvalues['daca_x_post'],
                model3.pvalues['daca_x_post'], model4.pvalues['daca_x_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs)]
})
results_df.to_csv('regression_results.csv', index=False)
print("Regression results saved to 'regression_results.csv'")

# Export event study coefficients
event_study_results = []
for year in years:
    if year != 2011:
        var_name = f'daca_x_{year}'
        event_study_results.append({
            'Year': year,
            'Coefficient': model_es.params[var_name],
            'Std_Error': model_es.bse[var_name],
            'CI_Lower': model_es.conf_int().loc[var_name, 0],
            'CI_Upper': model_es.conf_int().loc[var_name, 1]
        })
es_df = pd.DataFrame(event_study_results)
es_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to 'event_study_results.csv'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
