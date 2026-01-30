"""
DACA Replication Study - Difference-in-Differences Analysis
Effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DACA REPLICATION STUDY - DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*70)

# =============================================================================
# 1. LOAD AND FILTER DATA
# =============================================================================
print("\n[1] Loading data...")

# Read data in chunks due to large file size
chunks = []
chunksize = 500000

for chunk in pd.read_csv('data/data.csv', chunksize=chunksize, low_memory=False):
    # Filter to relevant population immediately to reduce memory
    # Hispanic-Mexican (HISPAN = 1), Born in Mexico (BPL = 200), Not a citizen (CITIZEN = 3)
    filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3)
    ]
    if len(filtered) > 0:
        chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"  Filtered sample size: {len(df):,} observations")
print(f"  Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. DEFINE TREATMENT AND CONTROL GROUPS
# =============================================================================
print("\n[2] Defining treatment and control groups...")

# DACA was implemented June 15, 2012
# Treatment: Ages 26-30 as of June 15, 2012 (born June 16, 1981 to June 15, 1986)
# Control: Ages 31-35 as of June 15, 2012 (born June 16, 1976 to June 15, 1981)

# Using BIRTHYR and BIRTHQTR to determine precise birth timing
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

# For treatment group (26-30 on June 15, 2012):
# - Must be born after June 15, 1981 (age < 31)
# - Must be born on or before June 15, 1986 (age >= 26)

# For control group (31-35 on June 15, 2012):
# - Must be born after June 15, 1976 (age < 36)
# - Must be born on or before June 15, 1981 (age >= 31)

# Create a function to determine if someone falls in treatment/control group
def assign_group(row):
    birthyr = row['BIRTHYR']
    birthqtr = row['BIRTHQTR']

    # Handle missing birth quarter - assume middle of year
    if pd.isna(birthqtr) or birthqtr == 0 or birthqtr == 9:
        birthqtr = 2  # April-June

    # Treatment group: Born between June 16, 1981 and June 15, 1986
    # This means ages 26-30 as of June 15, 2012

    # Born in 1982-1985: definitely in treatment range
    if 1982 <= birthyr <= 1985:
        return 'treatment'
    # Born in 1986: only if Q1 or Q2 (before June 16)
    elif birthyr == 1986 and birthqtr <= 2:
        return 'treatment'
    # Born in 1981: only if Q3 or Q4 (after June 15)
    elif birthyr == 1981 and birthqtr >= 3:
        return 'treatment'

    # Control group: Born between June 16, 1976 and June 15, 1981
    # This means ages 31-35 as of June 15, 2012

    # Born in 1977-1980: definitely in control range
    elif 1977 <= birthyr <= 1980:
        return 'control'
    # Born in 1981: only if Q1 or Q2 (before June 16)
    elif birthyr == 1981 and birthqtr <= 2:
        return 'control'
    # Born in 1976: only if Q3 or Q4 (after June 15)
    elif birthyr == 1976 and birthqtr >= 3:
        return 'control'

    return 'other'

df['group'] = df.apply(assign_group, axis=1)

# Keep only treatment and control groups
df = df[df['group'].isin(['treatment', 'control'])].copy()
print(f"  Sample after group assignment: {len(df):,} observations")
print(f"  Treatment group: {(df['group']=='treatment').sum():,}")
print(f"  Control group: {(df['group']=='control').sum():,}")

# =============================================================================
# 3. DEFINE TIME PERIODS
# =============================================================================
print("\n[3] Defining time periods...")

# Exclude 2012 (implementation year - can't distinguish pre/post)
df = df[df['YEAR'] != 2012].copy()

# Pre-period: 2006-2011
# Post-period: 2013-2016
df['post'] = (df['YEAR'] >= 2013).astype(int)

print(f"  Sample after excluding 2012: {len(df):,} observations")
print(f"  Pre-period (2006-2011): {(df['post']==0).sum():,}")
print(f"  Post-period (2013-2016): {(df['post']==1).sum():,}")

# =============================================================================
# 4. CREATE OUTCOME VARIABLE
# =============================================================================
print("\n[4] Creating outcome variable (full-time employment)...")

# Full-time employment: Usually working 35+ hours per week
# UHRSWORK = 0 indicates not working or N/A
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

print(f"  Full-time employment rate: {df['fulltime'].mean()*100:.2f}%")

# =============================================================================
# 5. CREATE ADDITIONAL VARIABLES FOR ANALYSIS
# =============================================================================
print("\n[5] Creating additional analysis variables...")

# Treatment indicator
df['treat'] = (df['group'] == 'treatment').astype(int)

# Interaction term for DiD
df['treat_post'] = df['treat'] * df['post']

# Calculate age as of survey year
df['age_survey'] = df['YEAR'] - df['BIRTHYR']

# Create demographic controls
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories (EDUC general version)
# 0-5 = less than high school, 6 = high school, 7+ = some college or more
df['educ_lths'] = (df['EDUC'] < 6).astype(int)
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_somecol'] = (df['EDUC'] > 6).astype(int)

# Years in US
df['yrsusa'] = df['YRSUSA1'].fillna(0)

# =============================================================================
# 6. SUMMARY STATISTICS
# =============================================================================
print("\n[6] Summary Statistics")
print("="*70)

# By group and period
summary = df.groupby(['group', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'age_survey': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_lths': 'mean',
    'educ_hs': 'mean',
    'educ_somecol': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary by Group and Period:")
print(summary.to_string())

# Calculate raw DiD
treat_pre = df[(df['treat']==1) & (df['post']==0)]['fulltime'].mean()
treat_post = df[(df['treat']==1) & (df['post']==1)]['fulltime'].mean()
control_pre = df[(df['treat']==0) & (df['post']==0)]['fulltime'].mean()
control_post = df[(df['treat']==0) & (df['post']==1)]['fulltime'].mean()

raw_did = (treat_post - treat_pre) - (control_post - control_pre)

print("\n" + "="*70)
print("RAW DIFFERENCE-IN-DIFFERENCES:")
print("="*70)
print(f"  Treatment Pre:  {treat_pre:.4f}")
print(f"  Treatment Post: {treat_post:.4f}")
print(f"  Treatment Diff: {treat_post - treat_pre:.4f}")
print(f"  Control Pre:    {control_pre:.4f}")
print(f"  Control Post:   {control_post:.4f}")
print(f"  Control Diff:   {control_post - control_pre:.4f}")
print(f"  RAW DiD EFFECT: {raw_did:.4f} ({raw_did*100:.2f} percentage points)")

# =============================================================================
# 7. REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("REGRESSION ANALYSIS")
print("="*70)

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (Unweighted) ---")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: Basic DiD (weighted by person weight)
print("\n--- Model 2: Basic DiD (Weighted) ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with demographic controls (weighted)
print("\n--- Model 3: DiD with Demographic Controls (Weighted) ---")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + educ_somecol',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects (weighted)
print("\n--- Model 4: DiD with Year Fixed Effects (Weighted) ---")
df['year_factor'] = df['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ treat + treat_post + female + married + educ_hs + educ_somecol + C(YEAR)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
# Just print treat_post coefficient
print(f"  treat_post coefficient: {model4.params['treat_post']:.6f}")
print(f"  treat_post std error:   {model4.bse['treat_post']:.6f}")
print(f"  treat_post t-stat:      {model4.tvalues['treat_post']:.4f}")
print(f"  treat_post p-value:     {model4.pvalues['treat_post']:.6f}")

# Model 5: DiD with state fixed effects (weighted)
print("\n--- Model 5: DiD with State Fixed Effects (Weighted) ---")
model5 = smf.wls('fulltime ~ treat + treat_post + female + married + educ_hs + educ_somecol + C(YEAR) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  treat_post coefficient: {model5.params['treat_post']:.6f}")
print(f"  treat_post std error:   {model5.bse['treat_post']:.6f}")
print(f"  treat_post t-stat:      {model5.tvalues['treat_post']:.4f}")
print(f"  treat_post p-value:     {model5.pvalues['treat_post']:.6f}")

# =============================================================================
# 8. PREFERRED SPECIFICATION SUMMARY
# =============================================================================
print("\n" + "="*70)
print("PREFERRED SPECIFICATION (Model 5: DiD with Controls + Year + State FE)")
print("="*70)

coef = model5.params['treat_post']
se = model5.bse['treat_post']
ci_low = coef - 1.96 * se
ci_high = coef + 1.96 * se
n = len(df)

print(f"\n  Effect Size: {coef:.6f} ({coef*100:.2f} percentage points)")
print(f"  Standard Error: {se:.6f}")
print(f"  95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
print(f"  Sample Size: {n:,}")
print(f"  T-statistic: {model5.tvalues['treat_post']:.4f}")
print(f"  P-value: {model5.pvalues['treat_post']:.6f}")

# =============================================================================
# 9. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*70)
print("ROBUSTNESS CHECKS")
print("="*70)

# Check 1: Employment (any hours) instead of full-time
print("\n--- Robustness Check 1: Any Employment (UHRSWORK > 0) ---")
df['employed'] = (df['UHRSWORK'] > 0).astype(int)
model_rob1 = smf.wls('employed ~ treat + treat_post + female + married + educ_hs + educ_somecol + C(YEAR) + C(STATEFIP)',
                     data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  treat_post coefficient: {model_rob1.params['treat_post']:.6f}")
print(f"  treat_post std error:   {model_rob1.bse['treat_post']:.6f}")
print(f"  treat_post p-value:     {model_rob1.pvalues['treat_post']:.6f}")

# Check 2: Using EMPSTAT for employment
print("\n--- Robustness Check 2: Employment Status (EMPSTAT == 1) ---")
df['employed_empstat'] = (df['EMPSTAT'] == 1).astype(int)
model_rob2 = smf.wls('employed_empstat ~ treat + treat_post + female + married + educ_hs + educ_somecol + C(YEAR) + C(STATEFIP)',
                     data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  treat_post coefficient: {model_rob2.params['treat_post']:.6f}")
print(f"  treat_post std error:   {model_rob2.bse['treat_post']:.6f}")
print(f"  treat_post p-value:     {model_rob2.pvalues['treat_post']:.6f}")

# Check 3: By gender
print("\n--- Robustness Check 3: Effect by Gender ---")
df_male = df[df['female'] == 0]
df_female = df[df['female'] == 1]

model_male = smf.wls('fulltime ~ treat + treat_post + married + educ_hs + educ_somecol + C(YEAR) + C(STATEFIP)',
                     data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ treat + treat_post + married + educ_hs + educ_somecol + C(YEAR) + C(STATEFIP)',
                       data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"  Males - treat_post: {model_male.params['treat_post']:.6f} (SE: {model_male.bse['treat_post']:.6f})")
print(f"  Females - treat_post: {model_female.params['treat_post']:.6f} (SE: {model_female.bse['treat_post']:.6f})")

# =============================================================================
# 10. PARALLEL TRENDS CHECK
# =============================================================================
print("\n" + "="*70)
print("PARALLEL TRENDS CHECK (Pre-treatment years)")
print("="*70)

# Calculate yearly means
yearly_means = df.groupby(['YEAR', 'treat']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
yearly_means.columns = ['Control', 'Treatment']
print("\nYearly Full-Time Employment Rates:")
print(yearly_means.round(4).to_string())

# Event study: Interaction of treatment with each year
print("\n--- Event Study Coefficients ---")
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

# Interactions (omit 2011 as reference)
df['treat_2006'] = df['treat'] * df['year_2006']
df['treat_2007'] = df['treat'] * df['year_2007']
df['treat_2008'] = df['treat'] * df['year_2008']
df['treat_2009'] = df['treat'] * df['year_2009']
df['treat_2010'] = df['treat'] * df['year_2010']
df['treat_2013'] = df['treat'] * df['year_2013']
df['treat_2014'] = df['treat'] * df['year_2014']
df['treat_2015'] = df['treat'] * df['year_2015']
df['treat_2016'] = df['treat'] * df['year_2016']

model_event = smf.wls('''fulltime ~ treat + C(YEAR) +
                        treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 +
                        treat_2013 + treat_2014 + treat_2015 + treat_2016 +
                        female + married + educ_hs + educ_somecol + C(STATEFIP)''',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (reference year: 2011):")
event_vars = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
              'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
for var in event_vars:
    print(f"  {var}: {model_event.params[var]:.6f} (SE: {model_event.bse[var]:.6f}, p: {model_event.pvalues[var]:.4f})")

# =============================================================================
# 11. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results = {
    'preferred_effect': float(coef),
    'preferred_se': float(se),
    'preferred_ci_low': float(ci_low),
    'preferred_ci_high': float(ci_high),
    'sample_size': int(n),
    'treat_n': int((df['treat']==1).sum()),
    'control_n': int((df['treat']==0).sum()),
    'raw_did': float(raw_did),
    'treat_pre': float(treat_pre),
    'treat_post_rate': float(treat_post),
    'control_pre': float(control_pre),
    'control_post': float(control_post),
    'model1_coef': float(model1.params['treat_post']),
    'model1_se': float(model1.bse['treat_post']),
    'model2_coef': float(model2.params['treat_post']),
    'model2_se': float(model2.bse['treat_post']),
    'model3_coef': float(model3.params['treat_post']),
    'model3_se': float(model3.bse['treat_post']),
    'model4_coef': float(model4.params['treat_post']),
    'model4_se': float(model4.bse['treat_post']),
    'model5_coef': float(model5.params['treat_post']),
    'model5_se': float(model5.bse['treat_post']),
}

# Save results to file
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to results.json")

# Save model summaries
with open('model_summaries.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("MODEL 1: BASIC DID (UNWEIGHTED)\n")
    f.write("="*70 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("="*70 + "\n")
    f.write("MODEL 2: BASIC DID (WEIGHTED)\n")
    f.write("="*70 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("="*70 + "\n")
    f.write("MODEL 3: DID WITH DEMOGRAPHIC CONTROLS\n")
    f.write("="*70 + "\n")
    f.write(str(model3.summary()) + "\n\n")

    f.write("="*70 + "\n")
    f.write("MODEL 5 (PREFERRED): DID WITH YEAR AND STATE FE\n")
    f.write("="*70 + "\n")
    f.write(str(model5.summary()) + "\n\n")

print("Model summaries saved to model_summaries.txt")

# Save yearly means for plotting
yearly_means.to_csv('yearly_means.csv')
print("Yearly means saved to yearly_means.csv")

# Save event study coefficients
event_study_results = pd.DataFrame({
    'year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'coef': [model_event.params.get(f'treat_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] +
            [0] + [model_event.params.get(f'treat_{y}', 0) for y in [2013, 2014, 2015, 2016]],
    'se': [model_event.bse.get(f'treat_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] +
          [0] + [model_event.bse.get(f'treat_{y}', 0) for y in [2013, 2014, 2015, 2016]]
})
event_study_results.to_csv('event_study.csv', index=False)
print("Event study results saved to event_study.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
