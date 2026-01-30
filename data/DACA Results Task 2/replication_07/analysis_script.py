"""
DACA Employment Replication Analysis
=====================================
This script performs a difference-in-differences analysis to estimate the causal
effect of DACA eligibility on full-time employment among Hispanic-Mexican
Mexican-born individuals.

Research Question: Among ethnically Hispanic-Mexican Mexican-born people in the US,
what was the causal impact of DACA eligibility on full-time employment (35+ hrs/week)?

Treatment: Ages 26-30 as of June 15, 2012
Control: Ages 31-35 as of June 15, 2012
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
print("="*70)
print("DACA EMPLOYMENT REPLICATION ANALYSIS")
print("="*70)
print("\n[1] Loading data...")

# Read the CSV file
df = pd.read_csv('data/data.csv')
print(f"    Total observations loaded: {len(df):,}")
print(f"    Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# 2. SAMPLE SELECTION
# ============================================================================
print("\n[2] Applying sample selection criteria...")

# 2a. Hispanic-Mexican ethnicity
# HISPAN=1 is Mexican, HISPAND includes 100-107 for Mexican variants
df_sample = df[df['HISPAN'] == 1].copy()
print(f"    After Hispanic-Mexican filter: {len(df_sample):,}")

# 2b. Born in Mexico
# BPL=200 is Mexico, BPLD=20000 is Mexico
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"    After born in Mexico filter: {len(df_sample):,}")

# 2c. Not a citizen (CITIZEN=3 means "Not a citizen")
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"    After non-citizen filter: {len(df_sample):,}")

# ============================================================================
# 3. AGE CALCULATION AND GROUP ASSIGNMENT
# ============================================================================
print("\n[3] Calculating age as of June 15, 2012...")

# Age as of June 15, 2012:
# If birth quarter is 1 or 2 (Jan-June), they've had birthday by June 15
# If birth quarter is 3 or 4 (July-Dec), they haven't had birthday yet
# Age = 2012 - BIRTHYR if birthday passed, else 2012 - BIRTHYR - 1

df_sample['age_2012'] = 2012 - df_sample['BIRTHYR']
# Adjust for those born July-Dec (BIRTHQTR 3 or 4) - they hadn't had birthday yet
df_sample.loc[df_sample['BIRTHQTR'].isin([3, 4]), 'age_2012'] -= 1

print(f"    Age range in sample: {df_sample['age_2012'].min()} to {df_sample['age_2012'].max()}")

# Define treatment and control groups
# Treatment: Ages 26-30 as of June 15, 2012
# Control: Ages 31-35 as of June 15, 2012

df_sample['treat'] = ((df_sample['age_2012'] >= 26) & (df_sample['age_2012'] <= 30)).astype(int)
df_sample['control'] = ((df_sample['age_2012'] >= 31) & (df_sample['age_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df_sample[(df_sample['treat'] == 1) | (df_sample['control'] == 1)].copy()
print(f"    Treatment group (ages 26-30): {df_analysis['treat'].sum():,}")
print(f"    Control group (ages 31-35): {(df_analysis['control'] == 1).sum():,}")

# ============================================================================
# 4. ADDITIONAL ELIGIBILITY CRITERIA
# ============================================================================
print("\n[4] Applying DACA eligibility criteria...")

# Arrived in US before age 16
# age_at_arrival = YRIMMIG - BIRTHYR
# We need YRIMMIG > 0 (not N/A) and they must have arrived before age 16
df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0].copy()
df_analysis['age_at_arrival'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']
df_analysis = df_analysis[df_analysis['age_at_arrival'] < 16].copy()
print(f"    After arrival before age 16 filter: {len(df_analysis):,}")

# Continuous residence since June 15, 2007 (proxy: arrived by 2007)
df_analysis = df_analysis[df_analysis['YRIMMIG'] <= 2007].copy()
print(f"    After continuous residence (YRIMMIG<=2007) filter: {len(df_analysis):,}")

# ============================================================================
# 5. DEFINE TIME PERIODS
# ============================================================================
print("\n[5] Defining time periods...")

# Exclude 2012 due to timing ambiguity (DACA implemented June 15, 2012)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"    After excluding 2012: {len(df_analysis):,}")

# Post = 2013-2016, Pre = 2006-2011
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
print(f"    Pre-period observations (2006-2011): {(df_analysis['post']==0).sum():,}")
print(f"    Post-period observations (2013-2016): {(df_analysis['post']==1).sum():,}")

# ============================================================================
# 6. CREATE OUTCOME VARIABLE
# ============================================================================
print("\n[6] Creating outcome variable...")

# Full-time employment = usually works 35+ hours per week
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)
print(f"    Full-time employment rate: {df_analysis['fulltime'].mean()*100:.1f}%")

# ============================================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[7] Descriptive statistics by group and period...")

# Summary by treatment and period
summary = df_analysis.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'PERWT': 'sum',
    'AGE': 'mean',
    'SEX': 'mean',
    'UHRSWORK': 'mean'
}).round(3)

print("\n    Group Means (unweighted):")
print("    " + "="*60)
for treat_val in [0, 1]:
    group_name = "Treatment (26-30)" if treat_val == 1 else "Control (31-35)"
    print(f"\n    {group_name}:")
    for post_val in [0, 1]:
        period = "Post (2013-16)" if post_val == 1 else "Pre (2006-11)"
        subset = df_analysis[(df_analysis['treat']==treat_val) & (df_analysis['post']==post_val)]
        print(f"      {period}: FT rate = {subset['fulltime'].mean()*100:.2f}%, N = {len(subset):,}")

# ============================================================================
# 8. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("[8] DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*70)

# Create interaction term
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

# 8a. Simple DiD without covariates (OLS)
print("\n--- Model 1: Simple DiD (no covariates) ---")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit()
print(f"    DiD Estimate (treat_post): {model1.params['treat_post']:.4f}")
print(f"    Std Error: {model1.bse['treat_post']:.4f}")
print(f"    t-statistic: {model1.tvalues['treat_post']:.3f}")
print(f"    p-value: {model1.pvalues['treat_post']:.4f}")
print(f"    95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")
print(f"    N: {int(model1.nobs):,}")
print(f"    R-squared: {model1.rsquared:.4f}")

# 8b. DiD with demographic covariates
print("\n--- Model 2: DiD with demographics (sex, education, marital status) ---")

# Create education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # HS or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # College or more

# Create marital status indicator
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Female indicator
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

model2 = smf.ols('fulltime ~ treat + post + treat_post + female + educ_hs + married',
                 data=df_analysis).fit()
print(f"    DiD Estimate (treat_post): {model2.params['treat_post']:.4f}")
print(f"    Std Error: {model2.bse['treat_post']:.4f}")
print(f"    t-statistic: {model2.tvalues['treat_post']:.3f}")
print(f"    p-value: {model2.pvalues['treat_post']:.4f}")
print(f"    95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"    N: {int(model2.nobs):,}")
print(f"    R-squared: {model2.rsquared:.4f}")

# 8c. DiD with year fixed effects
print("\n--- Model 3: DiD with year fixed effects ---")
model3 = smf.ols('fulltime ~ treat + C(YEAR) + treat_post + female + educ_hs + married',
                 data=df_analysis).fit()
print(f"    DiD Estimate (treat_post): {model3.params['treat_post']:.4f}")
print(f"    Std Error: {model3.bse['treat_post']:.4f}")
print(f"    t-statistic: {model3.tvalues['treat_post']:.3f}")
print(f"    p-value: {model3.pvalues['treat_post']:.4f}")
print(f"    95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"    N: {int(model3.nobs):,}")
print(f"    R-squared: {model3.rsquared:.4f}")

# 8d. DiD with robust standard errors (HC1)
print("\n--- Model 4: DiD with robust standard errors (HC1) ---")
model4 = smf.ols('fulltime ~ treat + post + treat_post + female + educ_hs + married',
                 data=df_analysis).fit(cov_type='HC1')
print(f"    DiD Estimate (treat_post): {model4.params['treat_post']:.4f}")
print(f"    Robust Std Error: {model4.bse['treat_post']:.4f}")
print(f"    t-statistic: {model4.tvalues['treat_post']:.3f}")
print(f"    p-value: {model4.pvalues['treat_post']:.4f}")
print(f"    95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

# ============================================================================
# 9. WEIGHTED ANALYSIS
# ============================================================================
print("\n--- Model 5: Weighted DiD (using PERWT) ---")
import statsmodels.api as sm

X = df_analysis[['treat', 'post', 'treat_post', 'female', 'educ_hs', 'married']]
X = sm.add_constant(X)
y = df_analysis['fulltime']
weights = df_analysis['PERWT']

model5 = sm.WLS(y, X, weights=weights).fit()
print(f"    DiD Estimate (treat_post): {model5.params['treat_post']:.4f}")
print(f"    Std Error: {model5.bse['treat_post']:.4f}")
print(f"    t-statistic: {model5.tvalues['treat_post']:.3f}")
print(f"    p-value: {model5.pvalues['treat_post']:.4f}")
print(f"    95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")

# ============================================================================
# 10. PARALLEL TRENDS CHECK
# ============================================================================
print("\n" + "="*70)
print("[9] PARALLEL TRENDS ANALYSIS")
print("="*70)

# Calculate group means by year
trends = df_analysis.groupby(['YEAR', 'treat']).agg({
    'fulltime': 'mean',
    'PERWT': 'sum'
}).reset_index()

print("\n    Full-time employment rates by year and group:")
print("    " + "-"*50)
print("    Year    Treatment    Control    Difference")
print("    " + "-"*50)

pre_diffs = []
for year in sorted(df_analysis['YEAR'].unique()):
    treat_rate = trends[(trends['YEAR']==year) & (trends['treat']==1)]['fulltime'].values[0]
    control_rate = trends[(trends['YEAR']==year) & (trends['treat']==0)]['fulltime'].values[0]
    diff = treat_rate - control_rate
    period = "POST" if year >= 2013 else "PRE"
    print(f"    {year}    {treat_rate*100:6.2f}%     {control_rate*100:6.2f}%    {diff*100:+6.2f}pp  [{period}]")
    if year < 2012:
        pre_diffs.append(diff)

# Test for parallel trends in pre-period
print("\n    Pre-period difference statistics:")
print(f"    Mean pre-period difference: {np.mean(pre_diffs)*100:.2f}pp")
print(f"    Std dev of pre-period difference: {np.std(pre_diffs)*100:.2f}pp")

# Event study style regression
print("\n    Event Study Coefficients (treatment × year interactions):")
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

# Interactions
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'treat_year_{year}'] = df_analysis['treat'] * df_analysis[f'year_{year}']

event_formula = ('fulltime ~ treat + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + '
                 'year_2013 + year_2014 + year_2015 + year_2016 + '
                 'treat_year_2006 + treat_year_2007 + treat_year_2008 + treat_year_2009 + treat_year_2010 + '
                 'treat_year_2013 + treat_year_2014 + treat_year_2015 + treat_year_2016 + '
                 'female + educ_hs + married')

event_model = smf.ols(event_formula, data=df_analysis).fit()

print("\n    Year  Coef (vs 2011)  Std Err    95% CI")
print("    " + "-"*55)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_year_{year}'
    coef = event_model.params[var]
    se = event_model.bse[var]
    ci = event_model.conf_int().loc[var]
    sig = "*" if event_model.pvalues[var] < 0.05 else ""
    pre_post = "PRE" if year < 2012 else "POST"
    print(f"    {year}  {coef:+.4f}        {se:.4f}     [{ci[0]:+.4f}, {ci[1]:+.4f}] {sig} {pre_post}")

# ============================================================================
# 11. ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*70)
print("[10] ROBUSTNESS CHECKS")
print("="*70)

# 10a. Alternative age windows
print("\n--- Robustness 1: Narrower age window (27-29 vs 32-34) ---")
df_narrow = df_analysis[(df_analysis['age_2012'].isin([27,28,29])) |
                         (df_analysis['age_2012'].isin([32,33,34]))].copy()
df_narrow['treat_narrow'] = df_narrow['age_2012'].isin([27,28,29]).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treat_narrow'] * df_narrow['post']

model_narrow = smf.ols('fulltime ~ treat_narrow + post + treat_post_narrow + female + educ_hs + married',
                       data=df_narrow).fit()
print(f"    DiD Estimate: {model_narrow.params['treat_post_narrow']:.4f}")
print(f"    Std Error: {model_narrow.bse['treat_post_narrow']:.4f}")
print(f"    95% CI: [{model_narrow.conf_int().loc['treat_post_narrow', 0]:.4f}, {model_narrow.conf_int().loc['treat_post_narrow', 1]:.4f}]")
print(f"    N: {int(model_narrow.nobs):,}")

# 10b. By gender
print("\n--- Robustness 2: Heterogeneity by Gender ---")
df_male = df_analysis[df_analysis['female'] == 0]
df_female = df_analysis[df_analysis['female'] == 1]

model_male = smf.ols('fulltime ~ treat + post + treat_post + educ_hs + married', data=df_male).fit()
model_female = smf.ols('fulltime ~ treat + post + treat_post + educ_hs + married', data=df_female).fit()

print(f"    Males:   DiD = {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f}), N = {int(model_male.nobs):,}")
print(f"    Females: DiD = {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f}), N = {int(model_female.nobs):,}")

# 10c. Placebo test - use 2009 as fake treatment year
print("\n--- Robustness 3: Placebo Test (fake treatment in 2009) ---")
df_placebo = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treat_post_placebo'] = df_placebo['treat'] * df_placebo['post_placebo']

model_placebo = smf.ols('fulltime ~ treat + post_placebo + treat_post_placebo + female + educ_hs + married',
                        data=df_placebo).fit()
print(f"    Placebo DiD Estimate: {model_placebo.params['treat_post_placebo']:.4f}")
print(f"    Std Error: {model_placebo.bse['treat_post_placebo']:.4f}")
print(f"    p-value: {model_placebo.pvalues['treat_post_placebo']:.4f}")
print(f"    (Expecting: not statistically significant)")

# 10d. Alternative outcome - any employment
print("\n--- Robustness 4: Alternative Outcome (Any Employment) ---")
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)
model_employed = smf.ols('fulltime ~ treat + post + treat_post + female + educ_hs + married',
                         data=df_analysis).fit()
model_any_emp = smf.ols('employed ~ treat + post + treat_post + female + educ_hs + married',
                        data=df_analysis).fit()
print(f"    Full-time employment DiD: {model_employed.params['treat_post']:.4f}")
print(f"    Any employment DiD: {model_any_emp.params['treat_post']:.4f}")

# ============================================================================
# 12. SUMMARY OUTPUT FOR REPORT
# ============================================================================
print("\n" + "="*70)
print("SUMMARY OF MAIN RESULTS")
print("="*70)

print(f"""
PREFERRED SPECIFICATION: Model 2 (DiD with demographic controls)

Main Finding:
-------------
The difference-in-differences estimate of the effect of DACA eligibility on
full-time employment is {model2.params['treat_post']:.4f} (SE = {model2.bse['treat_post']:.4f}).

This represents a {model2.params['treat_post']*100:.2f} percentage point {'increase' if model2.params['treat_post'] > 0 else 'decrease'}
in the probability of full-time employment for DACA-eligible individuals
(ages 26-30 as of June 2012) relative to the comparison group (ages 31-35).

Statistical Significance:
-------------------------
t-statistic: {model2.tvalues['treat_post']:.3f}
p-value: {model2.pvalues['treat_post']:.4f}
95% Confidence Interval: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]

Sample Information:
-------------------
Total sample size: {int(model2.nobs):,}
Treatment group (ages 26-30): {df_analysis['treat'].sum():,}
Control group (ages 31-35): {(df_analysis['treat']==0).sum():,}
Pre-period years: 2006-2011
Post-period years: 2013-2016
""")

# Save results for LaTeX tables
results_dict = {
    'model1_coef': model1.params['treat_post'],
    'model1_se': model1.bse['treat_post'],
    'model1_pval': model1.pvalues['treat_post'],
    'model1_ci_low': model1.conf_int().loc['treat_post', 0],
    'model1_ci_high': model1.conf_int().loc['treat_post', 1],
    'model1_n': int(model1.nobs),
    'model1_r2': model1.rsquared,
    'model2_coef': model2.params['treat_post'],
    'model2_se': model2.bse['treat_post'],
    'model2_pval': model2.pvalues['treat_post'],
    'model2_ci_low': model2.conf_int().loc['treat_post', 0],
    'model2_ci_high': model2.conf_int().loc['treat_post', 1],
    'model2_n': int(model2.nobs),
    'model2_r2': model2.rsquared,
    'model3_coef': model3.params['treat_post'],
    'model3_se': model3.bse['treat_post'],
    'model4_coef': model4.params['treat_post'],
    'model4_se': model4.bse['treat_post'],
    'model5_coef': model5.params['treat_post'],
    'model5_se': model5.bse['treat_post'],
    'narrow_coef': model_narrow.params['treat_post_narrow'],
    'narrow_se': model_narrow.bse['treat_post_narrow'],
    'male_coef': model_male.params['treat_post'],
    'male_se': model_male.bse['treat_post'],
    'female_coef': model_female.params['treat_post'],
    'female_se': model_female.bse['treat_post'],
    'placebo_coef': model_placebo.params['treat_post_placebo'],
    'placebo_se': model_placebo.bse['treat_post_placebo'],
    'placebo_pval': model_placebo.pvalues['treat_post_placebo'],
}

# Save full model summaries
print("\n" + "="*70)
print("FULL MODEL 2 SUMMARY (Preferred Specification)")
print("="*70)
print(model2.summary())

# Create tables for export
print("\n\nCreating summary tables for export...")

# Summary statistics table
summary_stats = df_analysis.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'educ_hs': 'mean',
    'married': 'mean',
    'AGE': 'mean',
    'UHRSWORK': 'mean'
}).round(3)

print("\nSummary Statistics:")
print(summary_stats)

# Save key statistics to file
with open('analysis_results.txt', 'w') as f:
    f.write("DACA EMPLOYMENT REPLICATION - KEY RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"PREFERRED ESTIMATE (Model 2 with demographics):\n")
    f.write(f"  Coefficient: {model2.params['treat_post']:.6f}\n")
    f.write(f"  Standard Error: {model2.bse['treat_post']:.6f}\n")
    f.write(f"  t-statistic: {model2.tvalues['treat_post']:.4f}\n")
    f.write(f"  p-value: {model2.pvalues['treat_post']:.6f}\n")
    f.write(f"  95% CI: [{model2.conf_int().loc['treat_post', 0]:.6f}, {model2.conf_int().loc['treat_post', 1]:.6f}]\n")
    f.write(f"  Sample size: {int(model2.nobs)}\n")
    f.write(f"  R-squared: {model2.rsquared:.4f}\n\n")

    f.write("COVARIATE COEFFICIENTS:\n")
    for var in ['treat', 'post', 'treat_post', 'female', 'educ_hs', 'married']:
        f.write(f"  {var}: {model2.params[var]:.6f} (SE: {model2.bse[var]:.6f})\n")

    f.write("\n\nSAMPLE COMPOSITION:\n")
    f.write(f"  Treatment group (ages 26-30): {df_analysis['treat'].sum()}\n")
    f.write(f"  Control group (ages 31-35): {(df_analysis['treat']==0).sum()}\n")
    f.write(f"  Pre-period: {(df_analysis['post']==0).sum()}\n")
    f.write(f"  Post-period: {(df_analysis['post']==1).sum()}\n")

    f.write("\n\nYEAR-BY-GROUP MEANS:\n")
    for year in sorted(df_analysis['YEAR'].unique()):
        for treat in [0, 1]:
            subset = df_analysis[(df_analysis['YEAR']==year) & (df_analysis['treat']==treat)]
            grp = "Treat" if treat else "Control"
            f.write(f"  {year} {grp}: FT rate = {subset['fulltime'].mean():.4f}, N = {len(subset)}\n")

print("\nAnalysis complete! Results saved to analysis_results.txt")
print("="*70)
