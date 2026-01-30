"""
DACA Replication Study - Full-Time Employment Analysis
=======================================================
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for DACA on the
probability of full-time employment (35+ hours/week)?

Author: Independent Replication
Date: 2025
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================
DATA_PATH = "data/data.csv"
OUTPUT_DIR = "."

# DACA Implementation Date: June 15, 2012
# Key dates:
# - Must have arrived before 16th birthday
# - Must have been under 31 as of June 15, 2012 (born after June 15, 1981)
# - Must have lived in US since June 15, 2007 (immigrated by 2007)
# - Must be present in US on June 15, 2012 without lawful status

# Columns we need for analysis
KEEP_COLS = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
             'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
             'EMPSTAT', 'UHRSWORK']

# =============================================================================
# DATA LOADING AND INITIAL PROCESSING
# =============================================================================
print("=" * 70)
print("DACA REPLICATION STUDY - FULL-TIME EMPLOYMENT ANALYSIS")
print("=" * 70)
print("\n1. LOADING DATA (chunked processing)...")

# Read in chunks and filter as we go to reduce memory usage
chunks = []
chunksize = 500000

for i, chunk in enumerate(pd.read_csv(DATA_PATH, usecols=KEEP_COLS, chunksize=chunksize)):
    # Filter to Hispanic-Mexican (HISPAN=1), Mexico-born (BPL=200), non-citizen (CITIZEN=3)
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3)].copy()
    if len(filtered) > 0:
        chunks.append(filtered)
    if (i + 1) % 10 == 0:
        print(f"   Processed {(i+1) * chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"\n   Total observations after initial filter: {len(df):,}")
print(f"   Years covered: {df['YEAR'].min()} - {df['YEAR'].max()}")

# =============================================================================
# SAMPLE SELECTION
# =============================================================================
print("\n2. SAMPLE SELECTION...")

# Working-age population (16-64 for labor market analysis)
df_sample = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
print(f"   After filtering working age 16-64: {len(df_sample):,}")

# Valid immigration year (need to determine DACA eligibility)
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
print(f"   After filtering valid immigration year: {len(df_sample):,}")

# =============================================================================
# DACA ELIGIBILITY CRITERIA
# =============================================================================
print("\n3. CONSTRUCTING DACA ELIGIBILITY...")

# Calculate age at immigration
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Calculate age as of June 15, 2012
# Using midpoint of birth year for simplicity since we don't have exact birth month
df_sample['age_june2012'] = 2012 - df_sample['BIRTHYR']

# DACA Eligibility Criteria (applied as of June 15, 2012):
# 1. Arrived before 16th birthday: age_at_immig < 16
# 2. Under 31 as of June 15, 2012: born after June 15, 1981 -> BIRTHYR >= 1982 conservatively
#    (or age_june2012 < 31)
# 3. Continuous presence since June 15, 2007: YRIMMIG <= 2007
# 4. Present without lawful status: CITIZEN = 3 (already filtered)

# Note: We cannot perfectly observe condition 3 (continuous presence) but proxy with
# immigration year <= 2007

df_sample['arrived_before_16'] = (df_sample['age_at_immig'] < 16).astype(int)
df_sample['under_31_in_2012'] = (df_sample['age_june2012'] < 31).astype(int)
df_sample['in_us_since_2007'] = (df_sample['YRIMMIG'] <= 2007).astype(int)

# DACA eligible: meets all three criteria (citizenship already filtered)
df_sample['daca_eligible'] = ((df_sample['arrived_before_16'] == 1) &
                               (df_sample['under_31_in_2012'] == 1) &
                               (df_sample['in_us_since_2007'] == 1)).astype(int)

print(f"   DACA eligible observations: {df_sample['daca_eligible'].sum():,} "
      f"({100*df_sample['daca_eligible'].mean():.1f}%)")

# =============================================================================
# OUTCOME VARIABLE: FULL-TIME EMPLOYMENT
# =============================================================================
print("\n4. CONSTRUCTING OUTCOME VARIABLE...")

# Full-time employment: usually works 35+ hours per week
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create employment indicator
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

print(f"   Full-time employment rate: {100*df_sample['fulltime'].mean():.1f}%")
print(f"   Employment rate: {100*df_sample['employed'].mean():.1f}%")

# =============================================================================
# TIME PERIOD INDICATORS
# =============================================================================
print("\n5. CREATING TIME PERIOD INDICATORS...")

# Pre-DACA: 2006-2011
# Post-DACA: 2013-2016 (examining effects 2013-2016 as specified)
# Exclude 2012 as transitional year (DACA implemented June 15, 2012)

df_sample['post_daca'] = (df_sample['YEAR'] >= 2013).astype(int)
df_sample['pre_daca'] = (df_sample['YEAR'] <= 2011).astype(int)

# Create sample excluding 2012 (ambiguous treatment year)
df_analysis = df_sample[df_sample['YEAR'] != 2012].copy()

print(f"   Observations excluding 2012: {len(df_analysis):,}")
print(f"   Pre-DACA (2006-2011): {len(df_analysis[df_analysis['post_daca']==0]):,}")
print(f"   Post-DACA (2013-2016): {len(df_analysis[df_analysis['post_daca']==1]):,}")

# =============================================================================
# DIFFERENCE-IN-DIFFERENCES SETUP
# =============================================================================
print("\n6. DIFFERENCE-IN-DIFFERENCES ANALYSIS...")

# DiD interaction term
df_analysis['did'] = df_analysis['daca_eligible'] * df_analysis['post_daca']

# Control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)

# Education categories
df_analysis['educ_lesshs'] = (df_analysis['EDUC'] < 6).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] == 6).astype(int)
df_analysis['educ_somecol'] = ((df_analysis['EDUC'] >= 7) & (df_analysis['EDUC'] <= 9)).astype(int)
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)

# Age squared for flexibility
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Years since immigration
df_analysis['years_in_us'] = df_analysis['YEAR'] - df_analysis['YRIMMIG']

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================
print("\n7. DESCRIPTIVE STATISTICS...")

# Summary by treatment group and period
summary_stats = df_analysis.groupby(['daca_eligible', 'post_daca']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\n   Summary Statistics by Group:")
print(summary_stats.to_string())

# Calculate raw DiD
ft_elig_pre = df_analysis[(df_analysis['daca_eligible']==1) & (df_analysis['post_daca']==0)]['fulltime'].mean()
ft_elig_post = df_analysis[(df_analysis['daca_eligible']==1) & (df_analysis['post_daca']==1)]['fulltime'].mean()
ft_inelig_pre = df_analysis[(df_analysis['daca_eligible']==0) & (df_analysis['post_daca']==0)]['fulltime'].mean()
ft_inelig_post = df_analysis[(df_analysis['daca_eligible']==0) & (df_analysis['post_daca']==1)]['fulltime'].mean()

raw_did = (ft_elig_post - ft_elig_pre) - (ft_inelig_post - ft_inelig_pre)

print(f"\n   Raw DiD Calculation:")
print(f"   Eligible Pre:  {ft_elig_pre:.4f}")
print(f"   Eligible Post: {ft_elig_post:.4f}")
print(f"   Ineligible Pre:  {ft_inelig_pre:.4f}")
print(f"   Ineligible Post: {ft_inelig_post:.4f}")
print(f"   Raw DiD Estimate: {raw_did:.4f}")

# =============================================================================
# REGRESSION ANALYSIS
# =============================================================================
print("\n8. REGRESSION ANALYSIS...")

# Model 1: Basic DiD (no controls)
print("\n   Model 1: Basic DiD")
model1 = smf.ols('fulltime ~ daca_eligible + post_daca + did', data=df_analysis).fit()
print(f"   DiD coefficient: {model1.params['did']:.4f} (SE: {model1.bse['did']:.4f})")
print(f"   p-value: {model1.pvalues['did']:.4f}")

# Model 2: DiD with demographic controls
print("\n   Model 2: DiD with demographic controls")
model2 = smf.ols('fulltime ~ daca_eligible + post_daca + did + AGE + age_sq + female + married',
                  data=df_analysis).fit()
print(f"   DiD coefficient: {model2.params['did']:.4f} (SE: {model2.bse['did']:.4f})")
print(f"   p-value: {model2.pvalues['did']:.4f}")

# Model 3: DiD with demographic + education controls
print("\n   Model 3: DiD with demographic + education controls")
model3 = smf.ols('fulltime ~ daca_eligible + post_daca + did + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college',
                  data=df_analysis).fit()
print(f"   DiD coefficient: {model3.params['did']:.4f} (SE: {model3.bse['did']:.4f})")
print(f"   p-value: {model3.pvalues['did']:.4f}")

# Model 4: DiD with demographic + education + years in US
print("\n   Model 4: DiD with full controls")
model4 = smf.ols('fulltime ~ daca_eligible + post_daca + did + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + years_in_us',
                  data=df_analysis).fit()
print(f"   DiD coefficient: {model4.params['did']:.4f} (SE: {model4.bse['did']:.4f})")
print(f"   p-value: {model4.pvalues['did']:.4f}")

# Model 5: Add year fixed effects
print("\n   Model 5: DiD with year fixed effects")
model5 = smf.ols('fulltime ~ daca_eligible + post_daca + did + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + years_in_us + C(YEAR)',
                  data=df_analysis).fit()
print(f"   DiD coefficient: {model5.params['did']:.4f} (SE: {model5.bse['did']:.4f})")
print(f"   p-value: {model5.pvalues['did']:.4f}")

# Model 6: Add state fixed effects
print("\n   Model 6: DiD with year + state fixed effects (preferred)")
model6 = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + years_in_us + C(YEAR) + C(STATEFIP)',
                  data=df_analysis).fit()
print(f"   DiD coefficient: {model6.params['did']:.4f} (SE: {model6.bse['did']:.4f})")
print(f"   p-value: {model6.pvalues['did']:.4f}")

# =============================================================================
# WEIGHTED ANALYSIS
# =============================================================================
print("\n9. WEIGHTED ANALYSIS (using person weights)...")

# Model with weights
model_weighted = smf.wls('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + years_in_us + C(YEAR) + C(STATEFIP)',
                          data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"   Weighted DiD coefficient: {model_weighted.params['did']:.4f} (SE: {model_weighted.bse['did']:.4f})")
print(f"   p-value: {model_weighted.pvalues['did']:.4f}")

# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================
print("\n10. ROBUSTNESS CHECKS...")

# Check 1: Alternative age range (18-30 for treated, comparison within similar ages)
print("\n   Robustness 1: Restricted age range (18-35)")
df_age_restrict = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 35)].copy()
model_age = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + C(YEAR) + C(STATEFIP)',
                     data=df_age_restrict).fit()
print(f"   DiD coefficient: {model_age.params['did']:.4f} (SE: {model_age.bse['did']:.4f})")
print(f"   N = {len(df_age_restrict):,}")

# Check 2: Employment as outcome (broader measure)
print("\n   Robustness 2: Employment (any) as outcome")
model_emp = smf.ols('employed ~ daca_eligible + did + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + years_in_us + C(YEAR) + C(STATEFIP)',
                     data=df_analysis).fit()
print(f"   DiD coefficient: {model_emp.params['did']:.4f} (SE: {model_emp.bse['did']:.4f})")

# Check 3: Placebo test - pre-period only (2006-2011), false treatment at 2009
print("\n   Robustness 3: Placebo test (false treatment in 2009)")
df_placebo = df_analysis[df_analysis['post_daca'] == 0].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['did_placebo'] = df_placebo['daca_eligible'] * df_placebo['post_placebo']
model_placebo = smf.ols('fulltime ~ daca_eligible + post_placebo + did_placebo + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + C(YEAR) + C(STATEFIP)',
                         data=df_placebo).fit()
print(f"   Placebo DiD coefficient: {model_placebo.params['did_placebo']:.4f} (SE: {model_placebo.bse['did_placebo']:.4f})")
print(f"   p-value: {model_placebo.pvalues['did_placebo']:.4f}")

# Check 4: Event study - year-by-year effects
print("\n   Robustness 4: Event Study - Year-specific effects")
df_analysis['year_2006'] = (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['year_2007'] = (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['year_2008'] = (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['year_2009'] = (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['year_2010'] = (df_analysis['YEAR'] == 2010).astype(int)
# 2011 is reference year
df_analysis['year_2013'] = (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['year_2014'] = (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['year_2015'] = (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['year_2016'] = (df_analysis['YEAR'] == 2016).astype(int)

# Interactions with eligibility
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'elig_x_{yr}'] = df_analysis['daca_eligible'] * df_analysis[f'year_{yr}']

model_event = smf.ols('fulltime ~ daca_eligible + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + elig_x_2006 + elig_x_2007 + elig_x_2008 + elig_x_2009 + elig_x_2010 + elig_x_2013 + elig_x_2014 + elig_x_2015 + elig_x_2016 + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_college + C(STATEFIP)',
                       data=df_analysis).fit()

print("   Year-specific treatment effects (relative to 2011):")
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'elig_x_{yr}']
    se = model_event.bse[f'elig_x_{yr}']
    print(f"   {yr}: {coef:.4f} (SE: {se:.4f})")

# =============================================================================
# HETEROGENEITY ANALYSIS
# =============================================================================
print("\n11. HETEROGENEITY ANALYSIS...")

# By gender
print("\n   By Gender:")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex]
    mod = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + married + educ_hs + educ_somecol + educ_college + C(YEAR) + C(STATEFIP)',
                   data=df_sub).fit()
    print(f"   {label}: DiD = {mod.params['did']:.4f} (SE: {mod.bse['did']:.4f}), N = {len(df_sub):,}")

# By education level
print("\n   By Education:")
for educ_level, label in [(0, 'Less than HS'), (1, 'HS or more')]:
    if educ_level == 0:
        df_sub = df_analysis[df_analysis['educ_lesshs'] == 1]
    else:
        df_sub = df_analysis[df_analysis['educ_lesshs'] == 0]
    mod = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                   data=df_sub).fit()
    print(f"   {label}: DiD = {mod.params['did']:.4f} (SE: {mod.bse['did']:.4f}), N = {len(df_sub):,}")

# By age group
print("\n   By Age Group:")
for age_min, age_max, label in [(16, 24, '16-24'), (25, 35, '25-35')]:
    df_sub = df_analysis[(df_analysis['AGE'] >= age_min) & (df_analysis['AGE'] <= age_max)]
    if len(df_sub) > 100:
        mod = smf.ols('fulltime ~ daca_eligible + did + AGE + female + married + educ_hs + educ_somecol + educ_college + C(YEAR) + C(STATEFIP)',
                       data=df_sub).fit()
        print(f"   {label}: DiD = {mod.params['did']:.4f} (SE: {mod.bse['did']:.4f}), N = {len(df_sub):,}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\nSample Size: {len(df_analysis):,}")
print(f"DACA Eligible: {df_analysis['daca_eligible'].sum():,}")
print(f"Pre-period observations: {len(df_analysis[df_analysis['post_daca']==0]):,}")
print(f"Post-period observations: {len(df_analysis[df_analysis['post_daca']==1]):,}")

print("\nPREFERRED ESTIMATE (Model 6: Year + State FE):")
print(f"   DiD Effect: {model6.params['did']:.4f}")
print(f"   Standard Error: {model6.bse['did']:.4f}")
print(f"   95% CI: [{model6.conf_int().loc['did', 0]:.4f}, {model6.conf_int().loc['did', 1]:.4f}]")
print(f"   p-value: {model6.pvalues['did']:.4f}")

# Interpretation
effect_pct = model6.params['did'] * 100
print(f"\nInterpretation: DACA eligibility is associated with a {effect_pct:.1f} percentage point")
print(f"change in the probability of full-time employment (35+ hours/week).")

# =============================================================================
# SAVE RESULTS FOR LATEX
# =============================================================================
print("\n12. SAVING RESULTS...")

# Create results dictionary for LaTeX tables
results = {
    'sample_size': len(df_analysis),
    'n_eligible': int(df_analysis['daca_eligible'].sum()),
    'n_ineligible': len(df_analysis) - int(df_analysis['daca_eligible'].sum()),
    'n_pre': len(df_analysis[df_analysis['post_daca']==0]),
    'n_post': len(df_analysis[df_analysis['post_daca']==1]),
    'raw_did': raw_did,
    'ft_elig_pre': ft_elig_pre,
    'ft_elig_post': ft_elig_post,
    'ft_inelig_pre': ft_inelig_pre,
    'ft_inelig_post': ft_inelig_post,
    'model1_coef': model1.params['did'],
    'model1_se': model1.bse['did'],
    'model1_pval': model1.pvalues['did'],
    'model2_coef': model2.params['did'],
    'model2_se': model2.bse['did'],
    'model2_pval': model2.pvalues['did'],
    'model3_coef': model3.params['did'],
    'model3_se': model3.bse['did'],
    'model3_pval': model3.pvalues['did'],
    'model4_coef': model4.params['did'],
    'model4_se': model4.bse['did'],
    'model4_pval': model4.pvalues['did'],
    'model5_coef': model5.params['did'],
    'model5_se': model5.bse['did'],
    'model5_pval': model5.pvalues['did'],
    'model6_coef': model6.params['did'],
    'model6_se': model6.bse['did'],
    'model6_pval': model6.pvalues['did'],
    'model6_ci_low': model6.conf_int().loc['did', 0],
    'model6_ci_high': model6.conf_int().loc['did', 1],
    'model6_r2': model6.rsquared,
    'weighted_coef': model_weighted.params['did'],
    'weighted_se': model_weighted.bse['did'],
    'placebo_coef': model_placebo.params['did_placebo'],
    'placebo_se': model_placebo.bse['did_placebo'],
    'placebo_pval': model_placebo.pvalues['did_placebo'],
    'model_age_coef': model_age.params['did'],
    'model_age_se': model_age.bse['did'],
    'model_age_n': len(df_age_restrict),
    'model_emp_coef': model_emp.params['did'],
    'model_emp_se': model_emp.bse['did'],
}

# Save results to file
with open('results_summary.txt', 'w') as f:
    for key, value in results.items():
        f.write(f"{key}: {value}\n")

# Save full model summary
with open('model6_summary.txt', 'w') as f:
    f.write(model6.summary().as_text())

# Event study coefficients
event_coefs = {}
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    event_coefs[yr] = {
        'coef': model_event.params[f'elig_x_{yr}'],
        'se': model_event.bse[f'elig_x_{yr}'],
        'pval': model_event.pvalues[f'elig_x_{yr}']
    }

with open('event_study_coefs.txt', 'w') as f:
    for yr, vals in event_coefs.items():
        f.write(f"{yr}: {vals['coef']:.6f}, {vals['se']:.6f}, {vals['pval']:.6f}\n")

# Descriptive stats table
desc_stats = df_analysis.groupby(['daca_eligible', 'post_daca']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_lesshs': 'mean',
    'educ_hs': 'mean',
    'educ_somecol': 'mean',
    'educ_college': 'mean',
    'years_in_us': 'mean',
    'PERWT': 'sum'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')

# Additional statistics
additional_stats = {
    'mean_age_eligible': df_analysis[df_analysis['daca_eligible']==1]['AGE'].mean(),
    'mean_age_ineligible': df_analysis[df_analysis['daca_eligible']==0]['AGE'].mean(),
    'female_pct_eligible': df_analysis[df_analysis['daca_eligible']==1]['female'].mean(),
    'female_pct_ineligible': df_analysis[df_analysis['daca_eligible']==0]['female'].mean(),
    'married_pct_eligible': df_analysis[df_analysis['daca_eligible']==1]['married'].mean(),
    'married_pct_ineligible': df_analysis[df_analysis['daca_eligible']==0]['married'].mean(),
    'lesshs_pct_eligible': df_analysis[df_analysis['daca_eligible']==1]['educ_lesshs'].mean(),
    'lesshs_pct_ineligible': df_analysis[df_analysis['daca_eligible']==0]['educ_lesshs'].mean(),
}

with open('additional_stats.txt', 'w') as f:
    for key, value in additional_stats.items():
        f.write(f"{key}: {value:.4f}\n")

# Heterogeneity results
het_results = {}
for sex, label in [(1, 'male'), (2, 'female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex]
    mod = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + married + educ_hs + educ_somecol + educ_college + C(YEAR) + C(STATEFIP)',
                   data=df_sub).fit()
    het_results[f'{label}_coef'] = mod.params['did']
    het_results[f'{label}_se'] = mod.bse['did']
    het_results[f'{label}_n'] = len(df_sub)

with open('heterogeneity_results.txt', 'w') as f:
    for key, value in het_results.items():
        f.write(f"{key}: {value}\n")

# Years by group
years_summary = df_analysis.groupby('YEAR').agg({
    'daca_eligible': ['sum', 'mean'],
    'fulltime': 'mean',
    'employed': 'mean'
}).round(4)
years_summary.to_csv('years_summary.csv')

print("\nResults saved to output files.")
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
