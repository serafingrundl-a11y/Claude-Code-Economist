"""
DACA Replication Analysis
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on full-time employment (>=35 hrs/week)?

Analysis Period: 2006-2016, with treatment effects examined 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Loading data...")
df = pd.read_csv('data/data.csv')
print(f"Total observations: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# 2. SAMPLE RESTRICTION: Hispanic-Mexican born in Mexico
# ============================================================================
print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# HISPAN = 1 indicates Mexican Hispanic origin
# BPL = 200 indicates born in Mexico
# BPLD = 20000 for detailed Mexico code

# Check HISPAN values
print(f"\nHISPAN value counts (top 5):")
print(df['HISPAN'].value_counts().head())

# Check BPL values for Mexico
print(f"\nBPL value counts for Mexico (200):")
print(f"  BPL=200: {(df['BPL']==200).sum():,}")

# Restrict to Hispanic-Mexican born in Mexico
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
print(f"\nAfter restricting to Hispanic-Mexican born in Mexico: {len(df_mex):,}")

# ============================================================================
# 3. DEFINE DACA ELIGIBILITY
# ============================================================================
print("\n" + "="*60)
print("DACA ELIGIBILITY CRITERIA")
print("="*60)

"""
DACA eligibility criteria (as of June 15, 2012):
1. Arrived in US before their 16th birthday
2. Under 31 years old as of June 15, 2012 (born after June 15, 1981)
3. Continuously present in US since June 15, 2007 (at least 5 years)
4. Present in US on June 15, 2012 (we assume all in ACS are present)
5. Not a citizen (CITIZEN = 3: "Not a citizen")

Note: We cannot observe documentation status. We assume non-citizens who
haven't received papers (CITIZEN=3) are potentially undocumented.
"""

# First, let's examine the relevant variables
print("\nCITIZEN value counts:")
print(df_mex['CITIZEN'].value_counts().sort_index())

print("\nYRIMMIG summary (year of immigration):")
print(df_mex['YRIMMIG'].describe())

# Calculate age at immigration
# Age at immigration = YEAR - YRIMMIG approximately
# But more precisely: person's age at survey - years in US

# Create arrival age variable
# YRSUSA1 is years in US (specific number)
# If not available, use YRIMMIG

# Calculate age at arrival using YRIMMIG and BIRTHYR
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# For DACA eligibility:
# Criterion 1: Arrived before 16th birthday
df_mex['arrived_before_16'] = df_mex['age_at_arrival'] < 16

# Criterion 2: Under 31 as of June 15, 2012 (born after June 15, 1981)
# Being conservative: born in 1982 or later definitely qualifies
# Born in 1981 might qualify depending on birth quarter
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Born June 15, 1981 or later: Q3 1981 definitely qualifies (July onward)
# Q2 1981 is borderline (Apr-Jun includes June 15)
# Conservative approach: treat born in 1982+ as definitely eligible
# For 1981, Q3 and Q4 are definitely eligible

df_mex['under_31_in_2012'] = (
    (df_mex['BIRTHYR'] >= 1982) |
    ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))
)

# Alternative: any 1981 birth year might qualify
df_mex['under_31_in_2012_broad'] = df_mex['BIRTHYR'] >= 1981

# Criterion 3: Present in US since June 15, 2007 (continuous residence)
# Immigration year should be 2007 or earlier
# YRIMMIG = 0 means N/A (likely native born, but we're already filtering to Mexico-born)
df_mex['present_since_2007'] = (df_mex['YRIMMIG'] > 0) & (df_mex['YRIMMIG'] <= 2007)

# Criterion 4: Not a citizen
# CITIZEN = 3 means "Not a citizen"
# CITIZEN = 2 means "Naturalized citizen" (not eligible)
# CITIZEN = 1 means "Born abroad of American parents" (not eligible)
df_mex['not_citizen'] = df_mex['CITIZEN'] == 3

# Combine criteria for DACA eligibility
df_mex['daca_eligible'] = (
    df_mex['arrived_before_16'] &
    df_mex['under_31_in_2012'] &
    df_mex['present_since_2007'] &
    df_mex['not_citizen']
)

# Also create a broader eligibility measure
df_mex['daca_eligible_broad'] = (
    df_mex['arrived_before_16'] &
    df_mex['under_31_in_2012_broad'] &
    df_mex['present_since_2007'] &
    df_mex['not_citizen']
)

print("\nDACA eligibility breakdown:")
print(f"  Arrived before 16: {df_mex['arrived_before_16'].sum():,} ({df_mex['arrived_before_16'].mean()*100:.1f}%)")
print(f"  Under 31 in 2012 (strict): {df_mex['under_31_in_2012'].sum():,} ({df_mex['under_31_in_2012'].mean()*100:.1f}%)")
print(f"  Present since 2007: {df_mex['present_since_2007'].sum():,} ({df_mex['present_since_2007'].mean()*100:.1f}%)")
print(f"  Not a citizen: {df_mex['not_citizen'].sum():,} ({df_mex['not_citizen'].mean()*100:.1f}%)")
print(f"  DACA eligible (all criteria): {df_mex['daca_eligible'].sum():,} ({df_mex['daca_eligible'].mean()*100:.1f}%)")

# ============================================================================
# 4. DEFINE OUTCOME: FULL-TIME EMPLOYMENT
# ============================================================================
print("\n" + "="*60)
print("OUTCOME VARIABLE: FULL-TIME EMPLOYMENT")
print("="*60)

# Full-time employment: usually working 35+ hours per week
# UHRSWORK = 0 means N/A or not working
# EMPSTAT = 1 means employed

print("\nEMPSTAT value counts:")
print(df_mex['EMPSTAT'].value_counts().sort_index())

print("\nUHRSWORK summary (among employed, EMPSTAT=1):")
print(df_mex[df_mex['EMPSTAT']==1]['UHRSWORK'].describe())

# Create full-time employment indicator
# Option 1: Among all working-age individuals
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

# Option 2: Among those in labor force
df_mex['employed'] = (df_mex['EMPSTAT'] == 1).astype(int)
df_mex['fulltime_if_employed'] = np.where(
    df_mex['EMPSTAT'] == 1,
    (df_mex['UHRSWORK'] >= 35).astype(int),
    np.nan
)

print(f"\nFull-time employment rate (all): {df_mex['fulltime'].mean()*100:.1f}%")
print(f"Employment rate: {df_mex['employed'].mean()*100:.1f}%")
print(f"Full-time rate among employed: {df_mex[df_mex['EMPSTAT']==1]['fulltime'].mean()*100:.1f}%")

# ============================================================================
# 5. RESTRICT SAMPLE FOR ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("ANALYTICAL SAMPLE")
print("="*60)

# Restrict to working-age population (typically 16-64)
# DACA was for those under 31 as of 2012, so by 2016 they'd be under 35
# Control group: similar age non-citizens who arrived after age 16 or too old for DACA

# Focus on ages where we have both eligible and ineligible individuals
# Age range: 18-40 for meaningful comparison

df_analysis = df_mex[(df_mex['AGE'] >= 18) & (df_mex['AGE'] <= 40)].copy()
print(f"After restricting to ages 18-40: {len(df_analysis):,}")

# Further restrict to non-citizens for cleaner comparison
# (comparing DACA-eligible non-citizens to non-eligible non-citizens)
df_noncit = df_analysis[df_analysis['not_citizen']].copy()
print(f"After restricting to non-citizens: {len(df_noncit):,}")

# ============================================================================
# 6. DEFINE TREATMENT AND POST PERIOD
# ============================================================================
print("\n" + "="*60)
print("DIFFERENCE-IN-DIFFERENCES SETUP")
print("="*60)

# Treatment: DACA eligible (arrived before 16, under 31 in 2012, present since 2007)
# Among non-citizens, the control group is those who don't meet all criteria

# Post period: 2013-2016 (DACA implemented in 2012, but ACS 2012 has mix of pre/post)
# Pre period: 2006-2011

df_noncit['post'] = (df_noncit['YEAR'] >= 2013).astype(int)
df_noncit['treat'] = df_noncit['daca_eligible'].astype(int)
df_noncit['treat_post'] = df_noncit['treat'] * df_noncit['post']

print("\nTreatment by Year:")
print(pd.crosstab(df_noncit['YEAR'], df_noncit['treat'], margins=True))

print("\nTreatment by Post period:")
print(pd.crosstab(df_noncit['post'], df_noncit['treat'], margins=True))

# ============================================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

# Pre-treatment characteristics
pre_data = df_noncit[df_noncit['YEAR'] < 2012]

print("\nPre-treatment means by treatment status:")
desc_vars = ['AGE', 'fulltime', 'employed', 'EDUC', 'SEX']
for var in desc_vars:
    if var in pre_data.columns:
        treat_mean = pre_data[pre_data['treat']==1][var].mean()
        control_mean = pre_data[pre_data['treat']==0][var].mean()
        print(f"  {var}: Treated={treat_mean:.3f}, Control={control_mean:.3f}, Diff={treat_mean-control_mean:.3f}")

# Sample sizes by year and treatment
print("\nSample sizes by year and treatment:")
print(df_noncit.groupby(['YEAR', 'treat']).size().unstack(fill_value=0))

# ============================================================================
# 8. MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("MAIN REGRESSION ANALYSIS")
print("="*60)

# Model 1: Basic DiD
print("\n--- Model 1: Basic Difference-in-Differences ---")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_noncit).fit(
    cov_type='cluster', cov_kwds={'groups': df_noncit['STATEFIP']}
)
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
df_noncit['female'] = (df_noncit['SEX'] == 2).astype(int)
df_noncit['married'] = (df_noncit['MARST'] == 1).astype(int)

model2 = smf.ols('fulltime ~ treat + post + treat_post + AGE + I(AGE**2) + female + married + C(EDUC)',
                 data=df_noncit).fit(cov_type='cluster', cov_kwds={'groups': df_noncit['STATEFIP']})
print("\nCoefficient on treat_post (DiD estimate):")
print(f"  Estimate: {model2.params['treat_post']:.4f}")
print(f"  Std Error: {model2.bse['treat_post']:.4f}")
print(f"  t-stat: {model2.tvalues['treat_post']:.3f}")
print(f"  p-value: {model2.pvalues['treat_post']:.4f}")
print(f"  95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")

# Model 3: DiD with year and state fixed effects
print("\n--- Model 3: DiD with Year and State Fixed Effects ---")
model3 = smf.ols('fulltime ~ treat + treat_post + C(YEAR) + C(STATEFIP) + AGE + I(AGE**2) + female + married',
                 data=df_noncit).fit(cov_type='cluster', cov_kwds={'groups': df_noncit['STATEFIP']})
print("\nCoefficient on treat_post (DiD estimate):")
print(f"  Estimate: {model3.params['treat_post']:.4f}")
print(f"  Std Error: {model3.bse['treat_post']:.4f}")
print(f"  t-stat: {model3.tvalues['treat_post']:.3f}")
print(f"  p-value: {model3.pvalues['treat_post']:.4f}")
print(f"  95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")

# ============================================================================
# 9. WEIGHTED ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("WEIGHTED ANALYSIS (using PERWT)")
print("="*60)

# Using WLS with person weights
import statsmodels.api as sm

# Model 4: Weighted DiD with controls
print("\n--- Model 4: Weighted DiD with Controls ---")
model4 = smf.wls('fulltime ~ treat + post + treat_post + AGE + I(AGE**2) + female + married + C(EDUC)',
                 data=df_noncit, weights=df_noncit['PERWT']).fit(
                     cov_type='cluster', cov_kwds={'groups': df_noncit['STATEFIP']})
print("\nCoefficient on treat_post (DiD estimate):")
print(f"  Estimate: {model4.params['treat_post']:.4f}")
print(f"  Std Error: {model4.bse['treat_post']:.4f}")
print(f"  t-stat: {model4.tvalues['treat_post']:.3f}")
print(f"  p-value: {model4.pvalues['treat_post']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

# ============================================================================
# 10. ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*60)
print("ROBUSTNESS CHECKS")
print("="*60)

# Robustness 1: Employment as outcome (instead of full-time)
print("\n--- Robustness 1: Employment as Outcome ---")
rob1 = smf.ols('employed ~ treat + post + treat_post + AGE + I(AGE**2) + female + married + C(EDUC)',
               data=df_noncit).fit(cov_type='cluster', cov_kwds={'groups': df_noncit['STATEFIP']})
print(f"  DiD estimate: {rob1.params['treat_post']:.4f} (SE: {rob1.bse['treat_post']:.4f})")

# Robustness 2: Different age ranges
print("\n--- Robustness 2: Age 18-35 only ---")
df_young = df_noncit[(df_noncit['AGE'] >= 18) & (df_noncit['AGE'] <= 35)]
rob2 = smf.ols('fulltime ~ treat + post + treat_post + AGE + I(AGE**2) + female + married + C(EDUC)',
               data=df_young).fit(cov_type='cluster', cov_kwds={'groups': df_young['STATEFIP']})
print(f"  DiD estimate: {rob2.params['treat_post']:.4f} (SE: {rob2.bse['treat_post']:.4f})")

# Robustness 3: Males only
print("\n--- Robustness 3: Males Only ---")
df_male = df_noncit[df_noncit['SEX'] == 1]
rob3 = smf.ols('fulltime ~ treat + post + treat_post + AGE + I(AGE**2) + married + C(EDUC)',
               data=df_male).fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
print(f"  DiD estimate: {rob3.params['treat_post']:.4f} (SE: {rob3.bse['treat_post']:.4f})")

# Robustness 4: Females only
print("\n--- Robustness 4: Females Only ---")
df_female = df_noncit[df_noncit['SEX'] == 2]
rob4 = smf.ols('fulltime ~ treat + post + treat_post + AGE + I(AGE**2) + married + C(EDUC)',
               data=df_female).fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})
print(f"  DiD estimate: {rob4.params['treat_post']:.4f} (SE: {rob4.bse['treat_post']:.4f})")

# Robustness 5: Placebo test - Use 2009 as fake treatment
print("\n--- Robustness 5: Placebo Test (2009 as fake treatment year) ---")
df_pre = df_noncit[df_noncit['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['treat_placebo'] = df_pre['treat'] * df_pre['placebo_post']
rob5 = smf.ols('fulltime ~ treat + placebo_post + treat_placebo + AGE + I(AGE**2) + female + married + C(EDUC)',
               data=df_pre).fit(cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})
print(f"  Placebo DiD estimate: {rob5.params['treat_placebo']:.4f} (SE: {rob5.bse['treat_placebo']:.4f})")

# ============================================================================
# 11. EVENT STUDY
# ============================================================================
print("\n" + "="*60)
print("EVENT STUDY ANALYSIS")
print("="*60)

# Create year dummies interacted with treatment
df_noncit['year_2006'] = (df_noncit['YEAR'] == 2006).astype(int)
df_noncit['year_2007'] = (df_noncit['YEAR'] == 2007).astype(int)
df_noncit['year_2008'] = (df_noncit['YEAR'] == 2008).astype(int)
df_noncit['year_2009'] = (df_noncit['YEAR'] == 2009).astype(int)
df_noncit['year_2010'] = (df_noncit['YEAR'] == 2010).astype(int)
# 2011 is reference year
df_noncit['year_2012'] = (df_noncit['YEAR'] == 2012).astype(int)
df_noncit['year_2013'] = (df_noncit['YEAR'] == 2013).astype(int)
df_noncit['year_2014'] = (df_noncit['YEAR'] == 2014).astype(int)
df_noncit['year_2015'] = (df_noncit['YEAR'] == 2015).astype(int)
df_noncit['year_2016'] = (df_noncit['YEAR'] == 2016).astype(int)

# Interactions
for yr in [2006, 2007, 2008, 2009, 2010, 2012, 2013, 2014, 2015, 2016]:
    df_noncit[f'treat_x_{yr}'] = df_noncit['treat'] * df_noncit[f'year_{yr}']

event_formula = 'fulltime ~ treat + ' + ' + '.join([f'year_{yr}' for yr in [2006,2007,2008,2009,2010,2012,2013,2014,2015,2016]]) + ' + ' + \
                ' + '.join([f'treat_x_{yr}' for yr in [2006,2007,2008,2009,2010,2012,2013,2014,2015,2016]]) + \
                ' + AGE + I(AGE**2) + female + married'

event_model = smf.ols(event_formula, data=df_noncit).fit(
    cov_type='cluster', cov_kwds={'groups': df_noncit['STATEFIP']})

print("\nEvent Study Coefficients (Treat x Year):")
print("Year  Coef      SE        t-stat   p-value")
print("-" * 50)
for yr in [2006, 2007, 2008, 2009, 2010, 2012, 2013, 2014, 2015, 2016]:
    coef = event_model.params[f'treat_x_{yr}']
    se = event_model.bse[f'treat_x_{yr}']
    t = event_model.tvalues[f'treat_x_{yr}']
    p = event_model.pvalues[f'treat_x_{yr}']
    print(f"{yr}  {coef:8.4f}  {se:8.4f}  {t:7.3f}  {p:7.4f}")

# ============================================================================
# 12. SUMMARY STATISTICS TABLE
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS TABLE")
print("="*60)

# Create summary stats for report
def summary_stats(data, varlist, weights=None):
    stats_list = []
    for var in varlist:
        if weights is not None:
            mean = np.average(data[var].dropna(), weights=data.loc[data[var].notna(), weights])
        else:
            mean = data[var].mean()
        stats_list.append({
            'Variable': var,
            'N': data[var].notna().sum(),
            'Mean': mean,
            'Std': data[var].std(),
            'Min': data[var].min(),
            'Max': data[var].max()
        })
    return pd.DataFrame(stats_list)

summary_vars = ['fulltime', 'employed', 'AGE', 'female', 'married', 'EDUC', 'treat', 'post']
print("\nFull Sample:")
print(summary_stats(df_noncit, summary_vars).to_string(index=False))

print("\nTreatment Group (DACA eligible):")
print(summary_stats(df_noncit[df_noncit['treat']==1], summary_vars).to_string(index=False))

print("\nControl Group (DACA ineligible):")
print(summary_stats(df_noncit[df_noncit['treat']==0], summary_vars).to_string(index=False))

# ============================================================================
# 13. SAVE RESULTS FOR REPORT
# ============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save key results to files for LaTeX report
results = {
    'main_estimate': model3.params['treat_post'],
    'main_se': model3.bse['treat_post'],
    'main_ci_low': model3.conf_int().loc['treat_post', 0],
    'main_ci_high': model3.conf_int().loc['treat_post', 1],
    'main_pvalue': model3.pvalues['treat_post'],
    'n_total': len(df_noncit),
    'n_treated': df_noncit['treat'].sum(),
    'n_control': (df_noncit['treat']==0).sum(),
    'weighted_estimate': model4.params['treat_post'],
    'weighted_se': model4.bse['treat_post'],
}

# Print preferred estimate
print("\n" + "="*60)
print("PREFERRED ESTIMATE (Model 3: DiD with Year and State FE)")
print("="*60)
print(f"Effect of DACA eligibility on full-time employment:")
print(f"  Coefficient: {results['main_estimate']:.4f}")
print(f"  Standard Error: {results['main_se']:.4f}")
print(f"  95% CI: [{results['main_ci_low']:.4f}, {results['main_ci_high']:.4f}]")
print(f"  p-value: {results['main_pvalue']:.4f}")
print(f"  Sample size: {results['n_total']:,}")
print(f"    Treated: {results['n_treated']:,}")
print(f"    Control: {results['n_control']:,}")

# ============================================================================
# 14. ADDITIONAL ANALYSIS: BY YEAR EFFECTS
# ============================================================================
print("\n" + "="*60)
print("EFFECTS BY POST-TREATMENT YEAR")
print("="*60)

# Separate effects for 2013, 2014, 2015, 2016
df_noncit['treat_2013'] = df_noncit['treat'] * (df_noncit['YEAR'] == 2013).astype(int)
df_noncit['treat_2014'] = df_noncit['treat'] * (df_noncit['YEAR'] == 2014).astype(int)
df_noncit['treat_2015'] = df_noncit['treat'] * (df_noncit['YEAR'] == 2015).astype(int)
df_noncit['treat_2016'] = df_noncit['treat'] * (df_noncit['YEAR'] == 2016).astype(int)

year_effects = smf.ols('fulltime ~ treat + C(YEAR) + treat_2013 + treat_2014 + treat_2015 + treat_2016 + AGE + I(AGE**2) + female + married',
                       data=df_noncit).fit(cov_type='cluster', cov_kwds={'groups': df_noncit['STATEFIP']})

print("\nYear-specific treatment effects:")
for yr in [2013, 2014, 2015, 2016]:
    var = f'treat_{yr}'
    print(f"  {yr}: {year_effects.params[var]:.4f} (SE: {year_effects.bse[var]:.4f})")

# ============================================================================
# 15. MEAN OUTCOMES BY GROUP AND PERIOD
# ============================================================================
print("\n" + "="*60)
print("MEAN OUTCOMES BY GROUP AND PERIOD")
print("="*60)

means = df_noncit.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'PERWT': 'sum'
}).round(4)
print(means)

# Calculate simple DiD
pre_treat = df_noncit[(df_noncit['treat']==1) & (df_noncit['post']==0)]['fulltime'].mean()
post_treat = df_noncit[(df_noncit['treat']==1) & (df_noncit['post']==1)]['fulltime'].mean()
pre_control = df_noncit[(df_noncit['treat']==0) & (df_noncit['post']==0)]['fulltime'].mean()
post_control = df_noncit[(df_noncit['treat']==0) & (df_noncit['post']==1)]['fulltime'].mean()

simple_did = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\nSimple DiD calculation:")
print(f"  Pre-treatment mean (treated): {pre_treat:.4f}")
print(f"  Post-treatment mean (treated): {post_treat:.4f}")
print(f"  Change for treated: {post_treat - pre_treat:.4f}")
print(f"  Pre-treatment mean (control): {pre_control:.4f}")
print(f"  Post-treatment mean (control): {post_control:.4f}")
print(f"  Change for control: {post_control - pre_control:.4f}")
print(f"  Simple DiD: {simple_did:.4f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

# Save summary stats and results to CSV for LaTeX tables
summary_all = df_noncit.groupby(['treat', 'post']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'EDUC': 'mean'
}).round(4)
summary_all.to_csv('summary_stats.csv')

# Save event study coefficients
event_coefs = []
for yr in [2006, 2007, 2008, 2009, 2010, 2012, 2013, 2014, 2015, 2016]:
    event_coefs.append({
        'year': yr,
        'coefficient': event_model.params[f'treat_x_{yr}'],
        'se': event_model.bse[f'treat_x_{yr}'],
        'ci_low': event_model.conf_int().loc[f'treat_x_{yr}', 0],
        'ci_high': event_model.conf_int().loc[f'treat_x_{yr}', 1]
    })
pd.DataFrame(event_coefs).to_csv('event_study_coefs.csv', index=False)

print("Results saved to summary_stats.csv and event_study_coefs.csv")
