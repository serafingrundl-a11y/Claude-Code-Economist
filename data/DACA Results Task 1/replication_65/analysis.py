"""
DACA Replication Study - Analysis Script
Examining the effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born non-citizens
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n1. LOADING DATA...")
print("-" * 40)

# Read the data
df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. SAMPLE RESTRICTIONS
# =============================================================================
print("\n2. APPLYING SAMPLE RESTRICTIONS...")
print("-" * 40)

# Track sample sizes
sample_tracking = []
sample_tracking.append(("Full ACS sample", len(df)))

# Restrict to Hispanic-Mexican ethnicity (HISPAN == 1)
df = df[df['HISPAN'] == 1].copy()
sample_tracking.append(("Hispanic-Mexican ethnicity (HISPAN==1)", len(df)))

# Restrict to born in Mexico (BPL == 200)
df = df[df['BPL'] == 200].copy()
sample_tracking.append(("Born in Mexico (BPL==200)", len(df)))

# Restrict to non-citizens (CITIZEN == 3)
df = df[df['CITIZEN'] == 3].copy()
sample_tracking.append(("Non-citizens (CITIZEN==3)", len(df)))

# Exclude 2012 (transition year)
df = df[df['YEAR'] != 2012].copy()
sample_tracking.append(("Exclude 2012 transition year", len(df)))

# Print sample tracking
print("\nSample restriction tracking:")
for desc, n in sample_tracking:
    print(f"  {desc}: {n:,}")

# =============================================================================
# 3. CREATE VARIABLES
# =============================================================================
print("\n3. CREATING ANALYSIS VARIABLES...")
print("-" * 40)

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"Post-DACA observations: {df['post'].sum():,} ({100*df['post'].mean():.1f}%)")

# Age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA Eligibility criteria:
# 1. Arrived before age 16
# 2. Under 31 as of June 15, 2012 (born after June 15, 1981)
# 3. In US since at least June 2007 (YRIMMIG <= 2007)

# Criterion 1: Arrived before age 16
df['arrived_before_16'] = (df['age_at_immig'] < 16).astype(int)

# Criterion 2: Under 31 as of June 15, 2012
# Born after June 15, 1981
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born in 1981 Q3 or Q4, eligible. If born 1982+, eligible.
df['under_31_june2012'] = ((df['BIRTHYR'] > 1981) |
                           ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))).astype(int)

# Criterion 3: In US since June 2007
df['in_us_since_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# DACA eligible: meets all three criteria
df['daca_eligible'] = ((df['arrived_before_16'] == 1) &
                       (df['under_31_june2012'] == 1) &
                       (df['in_us_since_2007'] == 1)).astype(int)

print(f"\nDACA eligibility components:")
print(f"  Arrived before age 16: {df['arrived_before_16'].sum():,} ({100*df['arrived_before_16'].mean():.1f}%)")
print(f"  Under 31 as of June 2012: {df['under_31_june2012'].sum():,} ({100*df['under_31_june2012'].mean():.1f}%)")
print(f"  In US since 2007: {df['in_us_since_2007'].sum():,} ({100*df['in_us_since_2007'].mean():.1f}%)")
print(f"  DACA eligible (all criteria): {df['daca_eligible'].sum():,} ({100*df['daca_eligible'].mean():.1f}%)")

# Outcome variable: Full-time employment (usually works 35+ hours/week)
# UHRSWORK == 0 means N/A (not working)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
print(f"\nFull-time employment (UHRSWORK>=35): {df['fulltime'].sum():,} ({100*df['fulltime'].mean():.1f}%)")

# Employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
print(f"Employed (EMPSTAT==1): {df['employed'].sum():,} ({100*df['employed'].mean():.1f}%)")

# In labor force
df['in_laborforce'] = (df['LABFORCE'] == 2).astype(int)
print(f"In labor force (LABFORCE==2): {df['in_laborforce'].sum():,} ({100*df['in_laborforce'].mean():.1f}%)")

# DiD interaction term
df['daca_x_post'] = df['daca_eligible'] * df['post']

# Control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_lesshs'] = (df['EDUC'] <= 5).astype(int)  # Less than high school
df['educ_hs'] = (df['EDUC'] == 6).astype(int)      # High school
df['educ_somecoll'] = (df['EDUC'].isin([7, 8, 9])).astype(int)  # Some college
df['educ_ba_plus'] = (df['EDUC'] >= 10).astype(int)  # BA or more

# Age groups for analysis
df['age_18_24'] = ((df['AGE'] >= 18) & (df['AGE'] <= 24)).astype(int)
df['age_25_30'] = ((df['AGE'] >= 25) & (df['AGE'] <= 30)).astype(int)
df['age_31_40'] = ((df['AGE'] >= 31) & (df['AGE'] <= 40)).astype(int)
df['age_41_plus'] = (df['AGE'] >= 41).astype(int)

# =============================================================================
# 4. RESTRICT TO WORKING-AGE POPULATION
# =============================================================================
print("\n4. RESTRICTING TO WORKING-AGE POPULATION...")
print("-" * 40)

# Restrict to ages 16-64 (working age)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
sample_tracking.append(("Working age 16-64", len(df)))
print(f"Sample after age restriction (16-64): {len(df):,}")

# =============================================================================
# 5. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n5. DESCRIPTIVE STATISTICS")
print("-" * 40)

# Summary by DACA eligibility
print("\nMeans by DACA eligibility status:")
desc_vars = ['AGE', 'female', 'married', 'educ_lesshs', 'educ_hs',
             'educ_somecoll', 'educ_ba_plus', 'fulltime', 'employed', 'in_laborforce']
desc_stats = df.groupby('daca_eligible')[desc_vars].mean()
print(desc_stats.T.round(3))

# Summary by period and treatment
print("\nFull-time employment by treatment and period:")
ft_by_group = df.groupby(['daca_eligible', 'post'])['fulltime'].agg(['mean', 'count'])
print(ft_by_group.round(3))

# Calculate raw DiD
pre_treat = df[(df['daca_eligible']==1) & (df['post']==0)]['fulltime'].mean()
post_treat = df[(df['daca_eligible']==1) & (df['post']==1)]['fulltime'].mean()
pre_ctrl = df[(df['daca_eligible']==0) & (df['post']==0)]['fulltime'].mean()
post_ctrl = df[(df['daca_eligible']==0) & (df['post']==1)]['fulltime'].mean()

raw_did = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)
print(f"\nRaw DiD estimate: {raw_did:.4f}")
print(f"  Treatment pre: {pre_treat:.4f}, post: {post_treat:.4f}, change: {post_treat-pre_treat:.4f}")
print(f"  Control pre: {pre_ctrl:.4f}, post: {post_ctrl:.4f}, change: {post_ctrl-pre_ctrl:.4f}")

# =============================================================================
# 6. MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n6. MAIN REGRESSION ANALYSIS")
print("-" * 40)

# Model 1: Basic DiD
print("\nModel 1: Basic DiD (no controls)")
model1 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\nModel 2: DiD with demographic controls")
model2 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + I(AGE**2) + female + married',
                 data=df).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with demographic and education controls
print("\nModel 3: DiD with demographic and education controls")
model3 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + I(AGE**2) + female + married + educ_hs + educ_somecoll + educ_ba_plus',
                 data=df).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: DiD with state fixed effects
print("\nModel 4: DiD with state fixed effects")
model4 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + I(AGE**2) + female + married + educ_hs + educ_somecoll + educ_ba_plus + C(STATEFIP)',
                 data=df).fit(cov_type='HC1')
# Extract key coefficients
key_vars = ['daca_eligible', 'post', 'daca_x_post', 'AGE', 'I(AGE ** 2)', 'female', 'married', 'educ_hs', 'educ_somecoll', 'educ_ba_plus']
print("\nKey coefficients (state FE included but not shown):")
for var in key_vars:
    if var in model4.params.index:
        coef = model4.params[var]
        se = model4.bse[var]
        pval = model4.pvalues[var]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {var:20s}: {coef:8.4f} ({se:.4f}){stars}")

# Model 5: DiD with year fixed effects
print("\nModel 5: DiD with year and state fixed effects")
model5 = smf.ols('fulltime ~ daca_eligible + daca_x_post + AGE + I(AGE**2) + female + married + educ_hs + educ_somecoll + educ_ba_plus + C(STATEFIP) + C(YEAR)',
                 data=df).fit(cov_type='HC1')
print("\nKey coefficients (year and state FE included but not shown):")
key_vars2 = ['daca_eligible', 'daca_x_post', 'AGE', 'I(AGE ** 2)', 'female', 'married', 'educ_hs', 'educ_somecoll', 'educ_ba_plus']
for var in key_vars2:
    if var in model5.params.index:
        coef = model5.params[var]
        se = model5.bse[var]
        pval = model5.pvalues[var]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {var:20s}: {coef:8.4f} ({se:.4f}){stars}")

# =============================================================================
# 7. PREFERRED SPECIFICATION - WEIGHTED
# =============================================================================
print("\n7. PREFERRED SPECIFICATION (WEIGHTED)")
print("-" * 40)

# Using person weights
print("\nModel 6: Preferred specification with person weights")
model6 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + I(AGE**2) + female + married + educ_hs + educ_somecoll + educ_ba_plus + C(STATEFIP) + C(YEAR)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("\nKey coefficients (weighted, year and state FE included):")
for var in key_vars2:
    if var in model6.params.index:
        coef = model6.params[var]
        se = model6.bse[var]
        pval = model6.pvalues[var]
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {var:20s}: {coef:8.4f} ({se:.4f}){stars}  [{ci_low:.4f}, {ci_high:.4f}]")

# =============================================================================
# 8. ROBUSTNESS CHECKS
# =============================================================================
print("\n8. ROBUSTNESS CHECKS")
print("-" * 40)

# 8.1 Alternative outcome: Employment (any hours)
print("\n8.1 Alternative outcome: Employment (EMPSTAT==1)")
model_emp = smf.wls('employed ~ daca_eligible + daca_x_post + AGE + I(AGE**2) + female + married + educ_hs + educ_somecoll + educ_ba_plus + C(STATEFIP) + C(YEAR)',
                    data=df, weights=df['PERWT']).fit(cov_type='HC1')
coef = model_emp.params['daca_x_post']
se = model_emp.bse['daca_x_post']
pval = model_emp.pvalues['daca_x_post']
print(f"  DACA x Post effect on employment: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# 8.2 Labor force participation
print("\n8.2 Alternative outcome: Labor force participation")
model_lfp = smf.wls('in_laborforce ~ daca_eligible + daca_x_post + AGE + I(AGE**2) + female + married + educ_hs + educ_somecoll + educ_ba_plus + C(STATEFIP) + C(YEAR)',
                    data=df, weights=df['PERWT']).fit(cov_type='HC1')
coef = model_lfp.params['daca_x_post']
se = model_lfp.bse['daca_x_post']
pval = model_lfp.pvalues['daca_x_post']
print(f"  DACA x Post effect on LFP: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# 8.3 Subgroup analysis by sex
print("\n8.3 Subgroup analysis by sex")
df_male = df[df['female'] == 0].copy()
df_female = df[df['female'] == 1].copy()

model_male = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + I(AGE**2) + married + educ_hs + educ_somecoll + educ_ba_plus + C(STATEFIP) + C(YEAR)',
                     data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + I(AGE**2) + married + educ_hs + educ_somecoll + educ_ba_plus + C(STATEFIP) + C(YEAR)',
                       data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"  Males - DACA x Post: {model_male.params['daca_x_post']:.4f} (SE: {model_male.bse['daca_x_post']:.4f})")
print(f"  Females - DACA x Post: {model_female.params['daca_x_post']:.4f} (SE: {model_female.bse['daca_x_post']:.4f})")

# 8.4 Event study specification
print("\n8.4 Event study (year-specific effects)")
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

# Interactions (2011 as reference)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'daca_x_{yr}'] = df['daca_eligible'] * df[f'year_{yr}']

event_formula = 'fulltime ~ daca_eligible + daca_x_2006 + daca_x_2007 + daca_x_2008 + daca_x_2009 + daca_x_2010 + daca_x_2013 + daca_x_2014 + daca_x_2015 + daca_x_2016 + AGE + I(AGE**2) + female + married + educ_hs + educ_somecoll + educ_ba_plus + C(STATEFIP) + C(YEAR)'
model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent study coefficients (relative to 2011):")
event_vars = ['daca_x_2006', 'daca_x_2007', 'daca_x_2008', 'daca_x_2009', 'daca_x_2010',
              'daca_x_2013', 'daca_x_2014', 'daca_x_2015', 'daca_x_2016']
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"  {var}: {coef:8.4f} ({se:.4f}){stars}")

# =============================================================================
# 9. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("9. SUMMARY OF MAIN RESULTS")
print("=" * 80)

# Preferred specification summary
preferred_coef = model6.params['daca_x_post']
preferred_se = model6.bse['daca_x_post']
preferred_pval = model6.pvalues['daca_x_post']
preferred_ci_low = preferred_coef - 1.96 * preferred_se
preferred_ci_high = preferred_coef + 1.96 * preferred_se

print(f"\nPreferred estimate (Model 6: weighted with year and state FE):")
print(f"  Effect of DACA eligibility on full-time employment: {preferred_coef:.4f}")
print(f"  Standard error: {preferred_se:.4f}")
print(f"  95% CI: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"  P-value: {preferred_pval:.4f}")
print(f"  Sample size: {len(df):,}")

# Save results to file for report
results_dict = {
    'preferred_coef': float(preferred_coef),
    'preferred_se': float(preferred_se),
    'preferred_pval': float(preferred_pval),
    'preferred_ci_low': float(preferred_ci_low),
    'preferred_ci_high': float(preferred_ci_high),
    'sample_size': int(len(df)),
    'n_treated': int(df['daca_eligible'].sum()),
    'n_control': int(len(df) - df['daca_eligible'].sum()),
    'raw_did': float(raw_did),
    'model1_coef': float(model1.params['daca_x_post']),
    'model1_se': float(model1.bse['daca_x_post']),
    'model2_coef': float(model2.params['daca_x_post']),
    'model2_se': float(model2.bse['daca_x_post']),
    'model3_coef': float(model3.params['daca_x_post']),
    'model3_se': float(model3.bse['daca_x_post']),
    'model4_coef': float(model4.params['daca_x_post']),
    'model4_se': float(model4.bse['daca_x_post']),
    'model5_coef': float(model5.params['daca_x_post']),
    'model5_se': float(model5.bse['daca_x_post']),
    'emp_coef': float(model_emp.params['daca_x_post']),
    'emp_se': float(model_emp.bse['daca_x_post']),
    'lfp_coef': float(model_lfp.params['daca_x_post']),
    'lfp_se': float(model_lfp.bse['daca_x_post']),
    'male_coef': float(model_male.params['daca_x_post']),
    'male_se': float(model_male.bse['daca_x_post']),
    'female_coef': float(model_female.params['daca_x_post']),
    'female_se': float(model_female.bse['daca_x_post'])
}

# Save results
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print("\nResults saved to analysis_results.json")

# =============================================================================
# 10. CREATE TABLES FOR REPORT
# =============================================================================
print("\n10. CREATING SUMMARY TABLES")
print("-" * 40)

# Table 1: Sample characteristics
print("\nTable 1: Sample Characteristics by DACA Eligibility")
print("-" * 60)
table1 = df.groupby('daca_eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_lesshs': 'mean',
    'educ_hs': 'mean',
    'educ_somecoll': 'mean',
    'educ_ba_plus': 'mean',
    'fulltime': 'mean',
    'employed': 'mean',
    'in_laborforce': 'mean',
    'PERWT': 'count'
}).round(3)
table1.columns = ['Age', 'Female', 'Married', 'Less HS', 'HS', 'Some Coll', 'BA+', 'Full-time', 'Employed', 'In LF', 'N']
table1.index = ['Non-eligible', 'DACA-eligible']
print(table1.T)

# Table 2: DiD means
print("\nTable 2: Full-time Employment by Group and Period")
print("-" * 60)
table2 = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count']
}).round(4)
print(table2)

# Table 3: Regression results summary
print("\nTable 3: Regression Results - Effect of DACA on Full-time Employment")
print("-" * 60)
print(f"{'Model':<35} {'Coef':<10} {'SE':<10} {'N':<12}")
print("-" * 60)
print(f"{'(1) Basic DiD':<35} {model1.params['daca_x_post']:<10.4f} {model1.bse['daca_x_post']:<10.4f} {model1.nobs:<12.0f}")
print(f"{'(2) + Demographics':<35} {model2.params['daca_x_post']:<10.4f} {model2.bse['daca_x_post']:<10.4f} {model2.nobs:<12.0f}")
print(f"{'(3) + Education':<35} {model3.params['daca_x_post']:<10.4f} {model3.bse['daca_x_post']:<10.4f} {model3.nobs:<12.0f}")
print(f"{'(4) + State FE':<35} {model4.params['daca_x_post']:<10.4f} {model4.bse['daca_x_post']:<10.4f} {model4.nobs:<12.0f}")
print(f"{'(5) + Year FE':<35} {model5.params['daca_x_post']:<10.4f} {model5.bse['daca_x_post']:<10.4f} {model5.nobs:<12.0f}")
print(f"{'(6) Weighted (Preferred)':<35} {model6.params['daca_x_post']:<10.4f} {model6.bse['daca_x_post']:<10.4f} {len(df):<12.0f}")

# Save descriptive stats for report
desc_stats_df = table1.T
desc_stats_df.to_csv('descriptive_stats.csv')

# Full-time means by year and treatment
ft_by_year = df.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
ft_by_year.columns = ['Non-eligible', 'DACA-eligible']
ft_by_year.to_csv('fulltime_by_year.csv')
print("\nFull-time employment by year:")
print(ft_by_year.round(4))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
