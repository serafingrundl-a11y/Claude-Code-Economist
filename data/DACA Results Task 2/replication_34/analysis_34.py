"""
DACA Replication Analysis - Participant 34
============================================
Examining the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican individuals born in Mexico.

Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
Control: Ages 31-35 at DACA implementation
Outcome: Full-time employment (35+ hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

print("="*70)
print("DACA REPLICATION ANALYSIS - PARTICIPANT 34")
print("="*70)

# =============================================================================
# 1. LOAD DATA IN CHUNKS (due to large file size)
# =============================================================================
print("\n[1] Loading data in chunks...")
data_path = "data/data.csv"

# Define columns we need
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
               'MARST', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
               'EDUC', 'EMPSTAT', 'LABFORCE', 'UHRSWORK']

# Read and filter in chunks
chunks = []
chunk_size = 1000000

print("    Reading and filtering data...")
for i, chunk in enumerate(pd.read_csv(data_path, usecols=cols_needed, chunksize=chunk_size)):
    # Filter: Hispanic-Mexican (HISPAN=1), Born in Mexico (BPL=200), Non-citizen (CITIZEN=3)
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3)].copy()
    if len(filtered) > 0:
        chunks.append(filtered)
    if (i+1) % 10 == 0:
        print(f"    Processed {(i+1)*chunk_size:,} rows...")
    del chunk
    gc.collect()

df_mex = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"    Total Hispanic-Mexican non-citizens from Mexico: {len(df_mex):,}")
print(f"    Years available: {sorted(df_mex['YEAR'].unique())}")

# =============================================================================
# 2. APPLY ADDITIONAL SAMPLE RESTRICTIONS
# =============================================================================
print("\n[2] Applying additional sample restrictions...")

# Step 2a: Calculate age at immigration
# Arrived before age 16 is a DACA requirement
df_mex['age_at_immig'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# Filter: valid immigration year and arrived before age 16
df_mex = df_mex[(df_mex['YRIMMIG'] > 0) & (df_mex['age_at_immig'] < 16)].copy()
print(f"    After arrived before age 16 filter: {len(df_mex):,}")

# Step 2b: Present in US since June 2007
# They need to have arrived by 2007
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007].copy()
print(f"    After in US since 2007 filter: {len(df_mex):,}")

# =============================================================================
# 3. DEFINE TREATMENT AND CONTROL GROUPS
# =============================================================================
print("\n[3] Defining treatment and control groups...")

# DACA implemented June 15, 2012
# Treatment: Ages 26-30 on that date (born 1982-1986)
# Control: Ages 31-35 on that date (born 1977-1981)

df_mex['treated'] = ((df_mex['BIRTHYR'] >= 1982) & (df_mex['BIRTHYR'] <= 1986)).astype(int)
df_mex['control'] = ((df_mex['BIRTHYR'] >= 1977) & (df_mex['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control groups
df_analysis = df_mex[(df_mex['treated'] == 1) | (df_mex['control'] == 1)].copy()
del df_mex
gc.collect()

print(f"    Treatment group (born 1982-1986): {len(df_analysis[df_analysis['treated']==1]):,}")
print(f"    Control group (born 1977-1981): {len(df_analysis[df_analysis['control']==1]):,}")

# =============================================================================
# 4. DEFINE TIME PERIODS
# =============================================================================
print("\n[4] Defining time periods...")

# Exclude 2012 due to mid-year DACA implementation
# Pre-period: 2006-2011
# Post-period: 2013-2016

df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
df_analysis['pre'] = (df_analysis['YEAR'] <= 2011).astype(int)

# Exclude 2012
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"    After excluding 2012: {len(df_analysis):,}")
print(f"    Pre-period observations (2006-2011): {len(df_analysis[df_analysis['pre']==1]):,}")
print(f"    Post-period observations (2013-2016): {len(df_analysis[df_analysis['post']==1]):,}")

# =============================================================================
# 5. DEFINE OUTCOME VARIABLE
# =============================================================================
print("\n[5] Defining outcome variable...")

# Full-time employment: Usually work 35+ hours per week
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Also create employment indicator
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)

print(f"    Overall full-time rate: {df_analysis['fulltime'].mean()*100:.1f}%")
print(f"    Overall employment rate: {df_analysis['employed'].mean()*100:.1f}%")

# =============================================================================
# 6. CREATE INTERACTION TERM
# =============================================================================
print("\n[6] Creating DiD interaction term...")

df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

# =============================================================================
# 7. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[7] Descriptive Statistics")
print("-"*70)

# Create summary by group and period
def weighted_mean(group, var, weight='PERWT'):
    return np.average(group[var], weights=group[weight])

def weighted_std(group, var, weight='PERWT'):
    avg = np.average(group[var], weights=group[weight])
    variance = np.average((group[var] - avg)**2, weights=group[weight])
    return np.sqrt(variance)

print("\nTable 1: Sample Sizes by Group and Period")
print("-"*50)
for period_name, period_val in [('Pre (2006-2011)', 0), ('Post (2013-2016)', 1)]:
    for group_name, group_val in [('Treatment (26-30)', 1), ('Control (31-35)', 0)]:
        n = len(df_analysis[(df_analysis['post']==period_val) & (df_analysis['treated']==group_val)])
        print(f"    {period_name}, {group_name}: {n:,}")

print("\nTable 2: Full-Time Employment Rates (Weighted)")
print("-"*50)
results_table = []
for period_name, period_val in [('Pre', 0), ('Post', 1)]:
    for group_name, group_val in [('Treatment', 1), ('Control', 0)]:
        subset = df_analysis[(df_analysis['post']==period_val) & (df_analysis['treated']==group_val)]
        ft_rate = weighted_mean(subset, 'fulltime') * 100
        n = len(subset)
        results_table.append({
            'Period': period_name,
            'Group': group_name,
            'FT_Rate': ft_rate,
            'N': n
        })
        print(f"    {period_name}, {group_name}: {ft_rate:.2f}% (n={n:,})")

# Calculate simple DiD
pre_treat = results_table[0]['FT_Rate']  # Pre, Treatment
pre_control = results_table[1]['FT_Rate']  # Pre, Control
post_treat = results_table[2]['FT_Rate']  # Post, Treatment
post_control = results_table[3]['FT_Rate']  # Post, Control

did_simple = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n    Simple DiD estimate: {did_simple:.2f} percentage points")

# =============================================================================
# 8. MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n[8] Main Regression Analysis: Difference-in-Differences")
print("-"*70)

# Model 1: Basic DiD without controls
print("\nModel 1: Basic DiD (no controls)")
model1 = smf.wls('fulltime ~ treated + post + treated_post',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit(cov_type='cluster',
                                                    cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"    DiD Coefficient (treated_post): {model1.params['treated_post']:.4f}")
print(f"    Standard Error (clustered): {model1.bse['treated_post']:.4f}")
print(f"    t-statistic: {model1.tvalues['treated_post']:.3f}")
print(f"    p-value: {model1.pvalues['treated_post']:.4f}")
print(f"    95% CI: [{model1.conf_int().loc['treated_post', 0]:.4f}, {model1.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    N: {int(model1.nobs):,}")
print(f"    R-squared: {model1.rsquared:.4f}")

# Model 2: DiD with year fixed effects
print("\nModel 2: DiD with Year Fixed Effects")
df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)
model2 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit(cov_type='cluster',
                                                    cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"    DiD Coefficient (treated_post): {model2.params['treated_post']:.4f}")
print(f"    Standard Error (clustered): {model2.bse['treated_post']:.4f}")
print(f"    t-statistic: {model2.tvalues['treated_post']:.3f}")
print(f"    p-value: {model2.pvalues['treated_post']:.4f}")
print(f"    95% CI: [{model2.conf_int().loc['treated_post', 0]:.4f}, {model2.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with demographic controls
print("\nModel 3: DiD with Demographic Controls")
# Add demographic controls
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)

# Education levels (simplified)
df_analysis['hs_or_more'] = (df_analysis['EDUC'] >= 6).astype(int)

model3 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + hs_or_more',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit(cov_type='cluster',
                                                    cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"    DiD Coefficient (treated_post): {model3.params['treated_post']:.4f}")
print(f"    Standard Error (clustered): {model3.bse['treated_post']:.4f}")
print(f"    t-statistic: {model3.tvalues['treated_post']:.3f}")
print(f"    p-value: {model3.pvalues['treated_post']:.4f}")
print(f"    95% CI: [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    R-squared: {model3.rsquared:.4f}")

# Model 4: Full model with state FE
print("\nModel 4: DiD with Year and State Fixed Effects + Controls")
model4 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + hs_or_more',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit(cov_type='cluster',
                                                    cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"    DiD Coefficient (treated_post): {model4.params['treated_post']:.4f}")
print(f"    Standard Error (clustered): {model4.bse['treated_post']:.4f}")
print(f"    t-statistic: {model4.tvalues['treated_post']:.3f}")
print(f"    p-value: {model4.pvalues['treated_post']:.4f}")
print(f"    95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    R-squared: {model4.rsquared:.4f}")

# =============================================================================
# 9. ROBUSTNESS CHECK: Event Study / Parallel Trends
# =============================================================================
print("\n[9] Parallel Trends Check (Event Study)")
print("-"*70)

# Create year-specific treatment effects (relative to 2011)
df_analysis['treat_2006'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['treat_2007'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['treat_2008'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['treat_2009'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['treat_2010'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2010).astype(int)
# 2011 is reference year (omitted)
df_analysis['treat_2013'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['treat_2014'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['treat_2015'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['treat_2016'] = df_analysis['treated'] * (df_analysis['YEAR'] == 2016).astype(int)

event_study = smf.wls(
    'fulltime ~ treated + C(YEAR) + treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2013 + treat_2014 + treat_2015 + treat_2016',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

print("\nEvent Study Coefficients (Reference: 2011):")
print("-"*50)
event_years = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
               'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
for year_var in event_years:
    coef = event_study.params[year_var]
    se = event_study.bse[year_var]
    pval = event_study.pvalues[year_var]
    year = year_var.split('_')[1]
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"    {year}: {coef:+.4f} ({se:.4f}) {sig}")

# =============================================================================
# 10. ROBUSTNESS: Alternative Employment Outcomes
# =============================================================================
print("\n[10] Alternative Outcomes")
print("-"*70)

# Employment (any hours)
print("\nAlternative: Employment Rate (any hours)")
model_emp = smf.wls('employed ~ treated + C(YEAR) + treated_post + female + married + hs_or_more',
                    data=df_analysis,
                    weights=df_analysis['PERWT']).fit(cov_type='cluster',
                                                       cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"    DiD Coefficient: {model_emp.params['treated_post']:.4f}")
print(f"    Standard Error: {model_emp.bse['treated_post']:.4f}")
print(f"    p-value: {model_emp.pvalues['treated_post']:.4f}")

# Labor force participation
df_analysis['in_lf'] = (df_analysis['LABFORCE'] == 2).astype(int)
print("\nAlternative: Labor Force Participation")
model_lf = smf.wls('in_lf ~ treated + C(YEAR) + treated_post + female + married + hs_or_more',
                   data=df_analysis,
                   weights=df_analysis['PERWT']).fit(cov_type='cluster',
                                                      cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"    DiD Coefficient: {model_lf.params['treated_post']:.4f}")
print(f"    Standard Error: {model_lf.bse['treated_post']:.4f}")
print(f"    p-value: {model_lf.pvalues['treated_post']:.4f}")

# =============================================================================
# 11. HETEROGENEITY ANALYSIS
# =============================================================================
print("\n[11] Heterogeneity Analysis")
print("-"*70)

# By gender
print("\nBy Gender:")
for gender, label in [(1, 'Male'), (2, 'Female')]:
    df_gender = df_analysis[df_analysis['SEX'] == gender]
    model_gender = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                           data=df_gender,
                           weights=df_gender['PERWT']).fit(cov_type='cluster',
                                                            cov_kwds={'groups': df_gender['STATEFIP']})
    print(f"    {label}: DiD = {model_gender.params['treated_post']:.4f} (SE={model_gender.bse['treated_post']:.4f}), N={int(model_gender.nobs):,}")

# By education
print("\nBy Education:")
for educ_level, label in [(0, 'Less than HS'), (1, 'HS or more')]:
    df_educ = df_analysis[df_analysis['hs_or_more'] == educ_level]
    if len(df_educ) > 100:
        model_educ = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                             data=df_educ,
                             weights=df_educ['PERWT']).fit(cov_type='cluster',
                                                            cov_kwds={'groups': df_educ['STATEFIP']})
        print(f"    {label}: DiD = {model_educ.params['treated_post']:.4f} (SE={model_educ.bse['treated_post']:.4f}), N={int(model_educ.nobs):,}")

# =============================================================================
# 12. SUMMARY STATISTICS TABLE
# =============================================================================
print("\n[12] Detailed Summary Statistics")
print("-"*70)

# Descriptive statistics for analysis sample
print("\nSample Characteristics (Full Analysis Sample):")
print(f"    Total observations: {len(df_analysis):,}")
print(f"    Unique years: {sorted(df_analysis['YEAR'].unique())}")

# Weighted means
print("\nWeighted Means by Treatment Status:")
for treat_val, label in [(1, 'Treatment'), (0, 'Control')]:
    subset = df_analysis[df_analysis['treated'] == treat_val]
    print(f"\n  {label} Group:")
    print(f"    Full-time rate: {weighted_mean(subset, 'fulltime')*100:.1f}%")
    print(f"    Employment rate: {weighted_mean(subset, 'employed')*100:.1f}%")
    print(f"    Female: {weighted_mean(subset, 'female')*100:.1f}%")
    print(f"    Married: {weighted_mean(subset, 'married')*100:.1f}%")
    print(f"    HS or more: {weighted_mean(subset, 'hs_or_more')*100:.1f}%")
    print(f"    Avg age (unweighted): {subset['AGE'].mean():.1f}")
    print(f"    Avg hours worked: {weighted_mean(subset, 'UHRSWORK'):.1f}")

# =============================================================================
# 13. SAVE KEY RESULTS FOR REPORT
# =============================================================================
print("\n" + "="*70)
print("SUMMARY OF KEY RESULTS")
print("="*70)

# Preferred specification (Model 3 with controls but without state FE for power)
preferred = model3

print(f"\nPreferred Estimate (Model 3):")
print(f"    Effect size: {preferred.params['treated_post']:.4f}")
print(f"    Standard error: {preferred.bse['treated_post']:.4f}")
print(f"    95% CI: [{preferred.conf_int().loc['treated_post', 0]:.4f}, {preferred.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {preferred.pvalues['treated_post']:.4f}")
print(f"    Sample size: {int(preferred.nobs):,}")

print(f"\nInterpretation:")
effect_pct = preferred.params['treated_post'] * 100
print(f"    DACA eligibility is associated with a {effect_pct:+.2f} percentage point")
print(f"    change in the probability of full-time employment.")

# Calculate effect relative to baseline
baseline_ft = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==0)]['fulltime'].mean()
relative_effect = (preferred.params['treated_post'] / baseline_ft) * 100
print(f"\n    Relative to pre-treatment mean ({baseline_ft*100:.1f}%),")
print(f"    this represents a {relative_effect:+.1f}% change.")

# =============================================================================
# 14. EXPORT RESULTS FOR LATEX
# =============================================================================
print("\n[14] Exporting results for LaTeX...")

# Save key numbers to a file for easy import into LaTeX
results_dict = {
    'n_total': len(df_analysis),
    'n_treatment': len(df_analysis[df_analysis['treated']==1]),
    'n_control': len(df_analysis[df_analysis['control']==1]),
    'n_pre': len(df_analysis[df_analysis['post']==0]),
    'n_post': len(df_analysis[df_analysis['post']==1]),

    # Model 1
    'model1_coef': model1.params['treated_post'],
    'model1_se': model1.bse['treated_post'],
    'model1_pval': model1.pvalues['treated_post'],
    'model1_ci_low': model1.conf_int().loc['treated_post', 0],
    'model1_ci_high': model1.conf_int().loc['treated_post', 1],
    'model1_n': int(model1.nobs),
    'model1_r2': model1.rsquared,

    # Model 2
    'model2_coef': model2.params['treated_post'],
    'model2_se': model2.bse['treated_post'],
    'model2_pval': model2.pvalues['treated_post'],
    'model2_r2': model2.rsquared,

    # Model 3 (preferred)
    'model3_coef': model3.params['treated_post'],
    'model3_se': model3.bse['treated_post'],
    'model3_pval': model3.pvalues['treated_post'],
    'model3_ci_low': model3.conf_int().loc['treated_post', 0],
    'model3_ci_high': model3.conf_int().loc['treated_post', 1],
    'model3_r2': model3.rsquared,
    'model3_female': model3.params['female'],
    'model3_female_se': model3.bse['female'],
    'model3_married': model3.params['married'],
    'model3_married_se': model3.bse['married'],
    'model3_hs': model3.params['hs_or_more'],
    'model3_hs_se': model3.bse['hs_or_more'],

    # Model 4
    'model4_coef': model4.params['treated_post'],
    'model4_se': model4.bse['treated_post'],
    'model4_pval': model4.pvalues['treated_post'],
    'model4_r2': model4.rsquared,

    # Employment outcome
    'emp_coef': model_emp.params['treated_post'],
    'emp_se': model_emp.bse['treated_post'],
    'emp_pval': model_emp.pvalues['treated_post'],

    # LFP outcome
    'lfp_coef': model_lf.params['treated_post'],
    'lfp_se': model_lf.bse['treated_post'],
    'lfp_pval': model_lf.pvalues['treated_post'],

    # Descriptive
    'ft_pre_treat': pre_treat,
    'ft_pre_control': pre_control,
    'ft_post_treat': post_treat,
    'ft_post_control': post_control,
    'did_simple': did_simple,
    'baseline_ft': baseline_ft * 100,
    'relative_effect': relative_effect,
}

# Save as Python file for import
with open('results_34.py', 'w') as f:
    f.write("# Auto-generated results from analysis_34.py\n")
    f.write("results = {\n")
    for key, value in results_dict.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            f.write(f"    '{key}': {value},\n")
        else:
            f.write(f"    '{key}': '{value}',\n")
    f.write("}\n")

print("    Results saved to results_34.py")

# =============================================================================
# 15. CREATE EVENT STUDY DATA FOR PLOTTING
# =============================================================================
event_study_data = []
reference_year = 2011

# Add reference year (coefficient = 0)
event_study_data.append({
    'year': reference_year,
    'coef': 0,
    'se': 0,
    'ci_low': 0,
    'ci_high': 0
})

for year_var in event_years:
    year = int(year_var.split('_')[1])
    coef = event_study.params[year_var]
    se = event_study.bse[year_var]
    ci = event_study.conf_int().loc[year_var]
    event_study_data.append({
        'year': year,
        'coef': coef,
        'se': se,
        'ci_low': ci[0],
        'ci_high': ci[1]
    })

event_df = pd.DataFrame(event_study_data).sort_values('year')
event_df.to_csv('event_study_34.csv', index=False)
print("    Event study data saved to event_study_34.csv")

# =============================================================================
# 16. CREATE YEARLY MEANS FOR PLOTTING
# =============================================================================
yearly_means = []
for year in sorted(df_analysis['YEAR'].unique()):
    for treat_val in [0, 1]:
        subset = df_analysis[(df_analysis['YEAR']==year) & (df_analysis['treated']==treat_val)]
        if len(subset) > 0:
            ft_rate = weighted_mean(subset, 'fulltime')
            yearly_means.append({
                'year': year,
                'treated': treat_val,
                'fulltime_rate': ft_rate,
                'n': len(subset)
            })

yearly_df = pd.DataFrame(yearly_means)
yearly_df.to_csv('yearly_means_34.csv', index=False)
print("    Yearly means saved to yearly_means_34.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
