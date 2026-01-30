"""
DACA Replication Study - Analysis Script
Replication 70

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals

Identification: Difference-in-differences
- Treatment: Ages 26-30 at June 15, 2012
- Control: Ages 31-35 at June 15, 2012
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

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

# Specify dtypes to avoid mixed type warnings
dtype_spec = {
    'YEAR': 'int32',
    'SAMPLE': 'int32',
    'SERIAL': 'int64',
    'HHWT': 'float64',
    'CLUSTER': 'int64',
    'REGION': 'int32',
    'STATEFIP': 'int32',
    'PUMA': 'int32',
    'METRO': 'int32',
    'GQ': 'int32',
    'FOODSTMP': 'int32',
    'PERNUM': 'int32',
    'PERWT': 'float64',
    'FAMSIZE': 'int32',
    'NCHILD': 'int32',
    'RELATE': 'int32',
    'RELATED': 'int32',
    'SEX': 'int32',
    'AGE': 'int32',
    'BIRTHQTR': 'int32',
    'MARST': 'int32',
    'BIRTHYR': 'int32',
    'RACE': 'int32',
    'RACED': 'int32',
    'HISPAN': 'int32',
    'HISPAND': 'int32',
    'BPL': 'int32',
    'BPLD': 'int32',
    'CITIZEN': 'int32',
    'YRIMMIG': 'int32',
    'YRSUSA1': 'int32',
    'YRSUSA2': 'int32',
    'EDUC': 'int32',
    'EDUCD': 'int32',
    'EMPSTAT': 'int32',
    'EMPSTATD': 'int32',
    'LABFORCE': 'int32',
    'CLASSWKR': 'int32',
    'CLASSWKRD': 'int32',
    'OCC': 'int32',
    'IND': 'int32',
    'WKSWORK2': 'int32',
    'UHRSWORK': 'int32',
    'INCTOT': 'int64',
    'FTOTINC': 'int64',
    'INCWAGE': 'int64',
    'POVERTY': 'int32'
}

df = pd.read_csv('data/data.csv', dtype=dtype_spec, low_memory=False)
print(f"   Total observations loaded: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. SAMPLE SELECTION
# =============================================================================
print("\n[2] Applying sample selection criteria...")

# Keep only relevant years (2006-2016, excluding 2012)
df = df[df['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])]
print(f"   After year filter (2006-2011, 2013-2016): {len(df):,}")

# Filter 1: Hispanic-Mexican ethnicity (HISPAN == 1)
df = df[df['HISPAN'] == 1]
print(f"   After Hispanic-Mexican filter (HISPAN==1): {len(df):,}")

# Filter 2: Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200]
print(f"   After born in Mexico filter (BPL==200): {len(df):,}")

# Filter 3: Not a citizen (CITIZEN == 3) - proxy for undocumented
df = df[df['CITIZEN'] == 3]
print(f"   After non-citizen filter (CITIZEN==3): {len(df):,}")

# Filter 4: Arrived in US by 2007 (continuous presence since June 15, 2007)
# YRIMMIG == 0 means N/A, so we need YRIMMIG > 0 and YRIMMIG <= 2007
df = df[(df['YRIMMIG'] > 0) & (df['YRIMMIG'] <= 2007)]
print(f"   After continuous presence filter (YRIMMIG<=2007): {len(df):,}")

# Calculate age at time of DACA implementation (June 15, 2012)
# Using mid-year approximation: age = 2012 - BIRTHYR
df['age_at_daca'] = 2012 - df['BIRTHYR']

# Filter 5: Arrived before 16th birthday
# Year of arrival - birth year < 16
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_arrival'] < 16]
print(f"   After arrival before age 16 filter: {len(df):,}")

# Filter 6: Age groups for treatment/control
# Treatment: ages 26-30 at June 15, 2012 (born 1982-1986)
# Control: ages 31-35 at June 15, 2012 (born 1977-1981)
df = df[(df['age_at_daca'] >= 26) & (df['age_at_daca'] <= 35)]
print(f"   After age range filter (26-35 at DACA): {len(df):,}")

# =============================================================================
# 3. CREATE ANALYSIS VARIABLES
# =============================================================================
print("\n[3] Creating analysis variables...")

# Treatment group indicator (1 if ages 26-30, 0 if ages 31-35)
df['treated'] = (df['age_at_daca'] <= 30).astype(int)

# Post-treatment indicator (1 if 2013-2016, 0 if 2006-2011)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term (DID estimator)
df['treated_post'] = df['treated'] * df['post']

# Outcome: Full-time employment (UHRSWORK >= 35)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)
df['has_children'] = (df['NCHILD'] > 0).astype(int)

# Education categories
df['educ_lesshs'] = (df['EDUC'] < 6).astype(int)  # Less than high school
df['educ_hs'] = (df['EDUC'] == 6).astype(int)      # High school
df['educ_somecoll'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)  # Some college
df['educ_coll'] = (df['EDUC'] >= 10).astype(int)   # College+

# Age in survey year (for controls)
df['age'] = df['AGE']
df['age_sq'] = df['age'] ** 2

print(f"   Treatment group (ages 26-30): {df['treated'].sum():,}")
print(f"   Control group (ages 31-35): {(1-df['treated']).sum():,}")
print(f"   Pre-period observations: {(1-df['post']).sum():,}")
print(f"   Post-period observations: {df['post'].sum():,}")

# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[4] Descriptive Statistics")
print("="*80)

# Summary by treatment/period
summary_table = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'age': 'mean',
    'educ_lesshs': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary by Treatment Status and Period:")
print(summary_table)

# Weighted means
print("\n" + "-"*80)
print("Weighted Full-Time Employment Rates:")
print("-"*80)

for treat in [0, 1]:
    for period in [0, 1]:
        subset = df[(df['treated'] == treat) & (df['post'] == period)]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        n = len(subset)
        weighted_n = subset['PERWT'].sum()
        treat_label = "Treatment (26-30)" if treat == 1 else "Control (31-35)"
        period_label = "Post (2013-16)" if period == 1 else "Pre (2006-11)"
        print(f"   {treat_label}, {period_label}: {weighted_mean:.4f} (n={n:,}, weighted n={weighted_n:,.0f})")

# =============================================================================
# 5. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n[5] Difference-in-Differences Analysis")
print("="*80)

# 5.1 Simple 2x2 DiD (unweighted)
print("\n5.1 Simple 2x2 DiD Calculation:")
print("-"*80)

# Calculate means for each cell
mean_treat_post = df[(df['treated']==1) & (df['post']==1)]['fulltime'].mean()
mean_treat_pre = df[(df['treated']==1) & (df['post']==0)]['fulltime'].mean()
mean_ctrl_post = df[(df['treated']==0) & (df['post']==1)]['fulltime'].mean()
mean_ctrl_pre = df[(df['treated']==0) & (df['post']==0)]['fulltime'].mean()

did_simple = (mean_treat_post - mean_treat_pre) - (mean_ctrl_post - mean_ctrl_pre)

print(f"   Treatment, Post:  {mean_treat_post:.4f}")
print(f"   Treatment, Pre:   {mean_treat_pre:.4f}")
print(f"   Control, Post:    {mean_ctrl_post:.4f}")
print(f"   Control, Pre:     {mean_ctrl_pre:.4f}")
print(f"\n   DiD Estimate (simple): {did_simple:.4f}")
print(f"   Treatment effect: {(mean_treat_post - mean_treat_pre):.4f}")
print(f"   Control trend:    {(mean_ctrl_post - mean_ctrl_pre):.4f}")

# 5.2 Basic DiD Regression (no controls, no weights)
print("\n5.2 Basic DiD Regression (no controls):")
print("-"*80)

model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit(cov_type='HC1')
print(model1.summary2().tables[1])

# 5.3 DiD with demographic controls
print("\n5.3 DiD with Demographic Controls:")
print("-"*80)

model2 = smf.ols('fulltime ~ treated + post + treated_post + female + married + has_children + age + age_sq + educ_hs + educ_somecoll + educ_coll',
                 data=df).fit(cov_type='HC1')
print(model2.summary2().tables[1])

# 5.4 Weighted DiD regression (preferred specification)
print("\n5.4 Weighted DiD Regression with Controls (PREFERRED):")
print("-"*80)

import statsmodels.api as sm

# Using WLS for weighted regression
X = df[['treated', 'post', 'treated_post', 'female', 'married', 'has_children',
        'age', 'age_sq', 'educ_hs', 'educ_somecoll', 'educ_coll']]
X = sm.add_constant(X)
y = df['fulltime']
weights = df['PERWT']

model3 = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')
print(model3.summary2().tables[1])

# Store preferred estimate
preferred_estimate = model3.params['treated_post']
preferred_se = model3.bse['treated_post']
preferred_ci_low = model3.conf_int().loc['treated_post', 0]
preferred_ci_high = model3.conf_int().loc['treated_post', 1]
preferred_n = len(df)

print("\n" + "="*80)
print("PREFERRED ESTIMATE SUMMARY")
print("="*80)
print(f"   Effect Size (DiD coefficient): {preferred_estimate:.4f}")
print(f"   Standard Error: {preferred_se:.4f}")
print(f"   95% CI: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"   Sample Size: {preferred_n:,}")
print(f"   t-statistic: {model3.tvalues['treated_post']:.4f}")
print(f"   p-value: {model3.pvalues['treated_post']:.4f}")

# =============================================================================
# 6. ROBUSTNESS CHECKS
# =============================================================================
print("\n[6] Robustness Checks")
print("="*80)

# 6.1 Year-by-year effects (event study)
print("\n6.1 Event Study - Year-by-Year Effects:")
print("-"*80)

# Create year dummies interacted with treatment
df['year_2006'] = (df['YEAR'] == 2006).astype(int)
df['year_2007'] = (df['YEAR'] == 2007).astype(int)
df['year_2008'] = (df['YEAR'] == 2008).astype(int)
df['year_2009'] = (df['YEAR'] == 2009).astype(int)
df['year_2010'] = (df['YEAR'] == 2010).astype(int)
df['year_2011'] = (df['YEAR'] == 2011).astype(int)
df['year_2013'] = (df['YEAR'] == 2013).astype(int)
df['year_2014'] = (df['YEAR'] == 2014).astype(int)
df['year_2015'] = (df['YEAR'] == 2015).astype(int)
df['year_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions with treatment (2011 is reference year)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'treat_x_{year}'] = df['treated'] * df[f'year_{year}']

event_formula = 'fulltime ~ treated + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + female + married + has_children + age + age_sq + educ_hs + educ_somecoll + educ_coll'

model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("Event study coefficients (treatment x year interactions):")
event_vars = ['treat_x_2006', 'treat_x_2007', 'treat_x_2008', 'treat_x_2009', 'treat_x_2010',
              'treat_x_2013', 'treat_x_2014', 'treat_x_2015', 'treat_x_2016']
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    print(f"   {var}: {coef:+.4f} (SE: {se:.4f}, p={pval:.3f})")

# 6.2 Separate analysis by gender
print("\n6.2 Analysis by Gender:")
print("-"*80)

for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex]
    X_sub = df_sub[['treated', 'post', 'treated_post', 'married', 'has_children',
                    'age', 'age_sq', 'educ_hs', 'educ_somecoll', 'educ_coll']]
    X_sub = sm.add_constant(X_sub)
    y_sub = df_sub['fulltime']
    weights_sub = df_sub['PERWT']

    model_sub = sm.WLS(y_sub, X_sub, weights=weights_sub).fit(cov_type='HC1')
    print(f"   {label}: DiD = {model_sub.params['treated_post']:.4f} (SE: {model_sub.bse['treated_post']:.4f}), n={len(df_sub):,}")

# 6.3 Placebo test - using pre-treatment years only
print("\n6.3 Placebo Test (Pre-Treatment Only, 2009 as fake treatment):")
print("-"*80)

df_pre = df[df['YEAR'] <= 2011].copy()
df_pre['post_placebo'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['treated_post_placebo'] = df_pre['treated'] * df_pre['post_placebo']

X_placebo = df_pre[['treated', 'post_placebo', 'treated_post_placebo', 'female', 'married',
                    'has_children', 'age', 'age_sq', 'educ_hs', 'educ_somecoll', 'educ_coll']]
X_placebo = sm.add_constant(X_placebo)
y_placebo = df_pre['fulltime']
weights_placebo = df_pre['PERWT']

model_placebo = sm.WLS(y_placebo, X_placebo, weights=weights_placebo).fit(cov_type='HC1')
print(f"   Placebo DiD: {model_placebo.params['treated_post_placebo']:.4f} (SE: {model_placebo.bse['treated_post_placebo']:.4f})")
print(f"   p-value: {model_placebo.pvalues['treated_post_placebo']:.4f}")

# =============================================================================
# 7. ADDITIONAL OUTCOMES
# =============================================================================
print("\n[7] Additional Outcomes")
print("="*80)

# Employment (any)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Labor force participation
df['in_labor_force'] = (df['LABFORCE'] == 2).astype(int)

for outcome_var, outcome_label in [('employed', 'Employment (any)'),
                                    ('in_labor_force', 'Labor Force Participation')]:
    X_out = df[['treated', 'post', 'treated_post', 'female', 'married', 'has_children',
                'age', 'age_sq', 'educ_hs', 'educ_somecoll', 'educ_coll']]
    X_out = sm.add_constant(X_out)
    y_out = df[outcome_var]
    weights_out = df['PERWT']

    model_out = sm.WLS(y_out, X_out, weights=weights_out).fit(cov_type='HC1')
    print(f"   {outcome_label}: DiD = {model_out.params['treated_post']:.4f} (SE: {model_out.bse['treated_post']:.4f})")

# =============================================================================
# 8. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n[8] Saving Results")
print("="*80)

# Save summary statistics
summary_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'in_labor_force': 'mean',
    'female': 'mean',
    'married': 'mean',
    'age': 'mean',
    'educ_lesshs': 'mean',
    'educ_hs': 'mean',
    'educ_somecoll': 'mean',
    'educ_coll': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)

summary_stats.to_csv('summary_stats.csv')
print("   Saved summary_stats.csv")

# Create event study results for plotting
event_results = pd.DataFrame({
    'year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'coefficient': [model_event.params.get(f'treat_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]],
    'se': [model_event.bse.get(f'treat_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]]
})
event_results.loc[event_results['year'] == 2011, ['coefficient', 'se']] = 0, 0  # Reference year
event_results.to_csv('event_study_results.csv', index=False)
print("   Saved event_study_results.csv")

# Final summary
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"\nPreferred Specification: Weighted DiD with Demographic Controls")
print(f"\nSample Selection:")
print(f"   - Hispanic-Mexican (HISPAN=1)")
print(f"   - Born in Mexico (BPL=200)")
print(f"   - Non-citizen (CITIZEN=3)")
print(f"   - Arrived by 2007 (YRIMMIG<=2007)")
print(f"   - Arrived before age 16")
print(f"   - Treatment: Ages 26-30 at DACA (born 1982-1986)")
print(f"   - Control: Ages 31-35 at DACA (born 1977-1981)")
print(f"\nMain Result:")
print(f"   Effect on Full-Time Employment: {preferred_estimate:.4f}")
print(f"   Standard Error: {preferred_se:.4f}")
print(f"   95% Confidence Interval: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"   Sample Size: {preferred_n:,}")
print(f"\nInterpretation:")
if preferred_estimate > 0:
    print(f"   DACA eligibility is associated with a {preferred_estimate*100:.2f} percentage point")
    print(f"   increase in full-time employment probability.")
else:
    print(f"   DACA eligibility is associated with a {abs(preferred_estimate)*100:.2f} percentage point")
    print(f"   decrease in full-time employment probability.")

if model3.pvalues['treated_post'] < 0.05:
    print(f"   This effect is statistically significant at the 5% level.")
else:
    print(f"   This effect is NOT statistically significant at the 5% level (p={model3.pvalues['treated_post']:.4f}).")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
