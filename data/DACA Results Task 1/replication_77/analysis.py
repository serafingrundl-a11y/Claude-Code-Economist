"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment (35+ hours/week)
among Hispanic-Mexican, Mexican-born individuals in the US.

DACA Eligibility Criteria:
1. Arrived in US before 16th birthday
2. Had not turned 31 by June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (arrived by 2007)
4. Present in US on June 15, 2012 (approximated by being in 2012+ ACS)
5. Not a citizen or legal resident

Key dates:
- DACA implemented: June 15, 2012
- Post-period: 2013-2016
- Pre-period: 2006-2011
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DACA REPLICATION ANALYSIS")
print("=" * 60)

# Define the columns we need
usecols = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR', 'MARST',
           'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
           'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK', 'STATEFIP']

print("\nStep 1: Loading and filtering data...")
print("Reading data in chunks and filtering to relevant population...")

# Process in chunks and filter early
chunks = []
chunk_size = 500000

for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunk_size, low_memory=False):
    # Filter to Hispanic-Mexican (HISPAN == 1) born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    print(f"  Processed chunk, filtered rows: {len(filtered):,}")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal Hispanic-Mexican Mexican-born sample: {len(df):,}")

# Save initial sample info
initial_sample_size = len(df)

print("\nStep 2: Examining year distribution...")
print(df['YEAR'].value_counts().sort_index())

print("\nStep 3: Examining citizenship status...")
print(df['CITIZEN'].value_counts())

print("\nStep 4: Creating analysis variables...")

# Define full-time employment (35+ hours per week, among those employed)
# EMPSTAT: 1 = Employed, 2 = Unemployed, 3 = Not in labor force
# UHRSWORK: usual hours worked per week

df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Post-DACA period: 2013-2016
# Pre-DACA period: 2006-2011
# 2012 is ambiguous (DACA announced June 15, 2012) - will exclude from main analysis
df['post'] = (df['YEAR'] >= 2013).astype(int)

# DACA Eligibility Criteria:
# 1. Arrived before 16th birthday: YRIMMIG - BIRTHYR < 16 or age at arrival < 16
# 2. Not yet 31 on June 15, 2012: born after June 15, 1981
#    - If BIRTHQTR 1 or 2 (Jan-June), need BIRTHYR >= 1982 to be sure under 31
#    - If BIRTHQTR 3 or 4 (July-Dec), BIRTHYR >= 1981 is sufficient
# 3. In US since June 15, 2007: YRIMMIG <= 2007
# 4. Not a citizen: CITIZEN == 3 (not a citizen) and not naturalized (CITIZEN != 2)

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Arrived before 16th birthday
df['arrived_before_16'] = (df['age_at_immig'] < 16) & (df['YRIMMIG'] > 0)

# Not yet 31 on June 15, 2012
# Conservative: born 1982 or later ensures under 31 as of June 15, 2012
# More precise: account for birth quarter
df['under_31_in_2012'] = (
    ((df['BIRTHYR'] >= 1982)) |  # Born 1982+ is definitely under 31
    ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))  # Born July-Dec 1981 would be under 31 on June 15
)

# In US since June 15, 2007 (arrived 2007 or earlier)
df['in_us_since_2007'] = (df['YRIMMIG'] <= 2007) & (df['YRIMMIG'] > 0)

# Not a citizen (undocumented proxy)
# CITIZEN: 0=N/A, 1=Born abroad of American parents, 2=Naturalized, 3=Not a citizen
df['non_citizen'] = (df['CITIZEN'] == 3)

# DACA eligible: meets all criteria
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['under_31_in_2012'] &
    df['in_us_since_2007'] &
    df['non_citizen']
)

print("\nDACA eligibility breakdown:")
print(f"  Arrived before age 16: {df['arrived_before_16'].sum():,}")
print(f"  Under 31 in 2012: {df['under_31_in_2012'].sum():,}")
print(f"  In US since 2007: {df['in_us_since_2007'].sum():,}")
print(f"  Non-citizen: {df['non_citizen'].sum():,}")
print(f"  DACA eligible (all criteria): {df['daca_eligible'].sum():,}")

# Create interaction term for difference-in-differences
df['daca_x_post'] = df['daca_eligible'].astype(int) * df['post']

print("\nStep 5: Sample restrictions for analysis...")

# Restrict to working-age population (18-64)
df_analysis = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)].copy()
print(f"  After age restriction (18-64): {len(df_analysis):,}")

# Exclude 2012 (treatment year is ambiguous)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012]
print(f"  After excluding 2012: {len(df_analysis):,}")

# For employment analysis, we can use full sample (employed + not employed)
# Full-time is conditional on employment

print(f"\nFinal analysis sample: {len(df_analysis):,}")

print("\nStep 6: Summary statistics...")

# Summary by eligibility and period
summary = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean', 'std'],
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary by DACA eligibility and period:")
print(summary)

# Calculate difference-in-differences manually
eligible_pre = df_analysis[(df_analysis['daca_eligible']) & (df_analysis['post'] == 0)]['fulltime'].mean()
eligible_post = df_analysis[(df_analysis['daca_eligible']) & (df_analysis['post'] == 1)]['fulltime'].mean()
ineligible_pre = df_analysis[(~df_analysis['daca_eligible']) & (df_analysis['post'] == 0)]['fulltime'].mean()
ineligible_post = df_analysis[(~df_analysis['daca_eligible']) & (df_analysis['post'] == 1)]['fulltime'].mean()

print("\n" + "=" * 60)
print("DIFFERENCE-IN-DIFFERENCES (Unconditional Full-Time Employment)")
print("=" * 60)
print(f"\nDACA Eligible:")
print(f"  Pre (2006-2011):  {eligible_pre:.4f}")
print(f"  Post (2013-2016): {eligible_post:.4f}")
print(f"  Difference:       {eligible_post - eligible_pre:.4f}")

print(f"\nDACA Ineligible:")
print(f"  Pre (2006-2011):  {ineligible_pre:.4f}")
print(f"  Post (2013-2016): {ineligible_post:.4f}")
print(f"  Difference:       {ineligible_post - ineligible_pre:.4f}")

did_effect = (eligible_post - eligible_pre) - (ineligible_post - ineligible_pre)
print(f"\nDifference-in-Differences: {did_effect:.4f}")

print("\n" + "=" * 60)
print("Step 7: Regression Analysis")
print("=" * 60)

# Create control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Education categories
df_analysis['educ_cat'] = pd.cut(df_analysis['EDUC'],
                                  bins=[-1, 3, 6, 10, 11],
                                  labels=['less_hs', 'hs', 'some_college', 'college_plus'])

# Model 1: Basic DiD
print("\nModel 1: Basic Difference-in-Differences")
model1 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n" + "=" * 60)
print("Model 2: DiD with Demographic Controls")
model2 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with controls and year fixed effects
print("\n" + "=" * 60)
print("Model 3: DiD with Controls and Year Fixed Effects")
df_analysis['year_cat'] = df_analysis['YEAR'].astype(str)
model3 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + C(year_cat)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# Model 4: Add state fixed effects
print("\n" + "=" * 60)
print("Model 4: DiD with Controls and State + Year Fixed Effects")
df_analysis['state_cat'] = df_analysis['STATEFIP'].astype(str)
model4 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + C(year_cat) + C(state_cat)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

# Print only key coefficients
print("\nKey Coefficients:")
print(f"  DACA Eligible:         {model4.params['daca_eligible[T.True]']:.6f} (SE: {model4.bse['daca_eligible[T.True]']:.6f})")
print(f"  DACA x Post (DiD):     {model4.params['daca_x_post']:.6f} (SE: {model4.bse['daca_x_post']:.6f})")

# Calculate t-stat and p-value for main coefficient
did_coef = model4.params['daca_x_post']
did_se = model4.bse['daca_x_post']
did_tstat = model4.tvalues['daca_x_post']
did_pval = model4.pvalues['daca_x_post']
did_ci = model4.conf_int().loc['daca_x_post']

print(f"\n  t-statistic: {did_tstat:.4f}")
print(f"  p-value: {did_pval:.4f}")
print(f"  95% CI: [{did_ci[0]:.6f}, {did_ci[1]:.6f}]")

print("\n" + "=" * 60)
print("Step 8: Alternative Specifications")
print("=" * 60)

# Analysis conditional on being employed
print("\nAlternative 1: Full-time (35+ hrs) conditional on being employed")
df_employed = df_analysis[df_analysis['employed'] == 1].copy()
print(f"Sample size (employed only): {len(df_employed):,}")

model_emp = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + C(year_cat)',
                     data=df_employed,
                     weights=df_employed['PERWT']).fit(cov_type='HC1')

print(f"\n  DACA x Post (DiD): {model_emp.params['daca_x_post']:.6f}")
print(f"  SE: {model_emp.bse['daca_x_post']:.6f}")
print(f"  95% CI: [{model_emp.conf_int().loc['daca_x_post'][0]:.6f}, {model_emp.conf_int().loc['daca_x_post'][1]:.6f}]")

# Analysis for employment (extensive margin)
print("\nAlternative 2: Employment (any work)")
model_extensive = smf.wls('employed ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + C(year_cat)',
                           data=df_analysis,
                           weights=df_analysis['PERWT']).fit(cov_type='HC1')

print(f"\n  DACA x Post (DiD): {model_extensive.params['daca_x_post']:.6f}")
print(f"  SE: {model_extensive.bse['daca_x_post']:.6f}")
print(f"  95% CI: [{model_extensive.conf_int().loc['daca_x_post'][0]:.6f}, {model_extensive.conf_int().loc['daca_x_post'][1]:.6f}]")

print("\n" + "=" * 60)
print("Step 9: Sample Statistics for Report")
print("=" * 60)

# Get sample sizes by group
print("\nSample sizes by group:")
for (elig, post_period), group in df_analysis.groupby(['daca_eligible', 'post']):
    status = "Eligible" if elig else "Ineligible"
    period = "Post" if post_period else "Pre"
    print(f"  {status}, {period}: {len(group):,} (weighted: {group['PERWT'].sum():,.0f})")

# Demographics by eligibility
print("\nDemographics by DACA eligibility status:")
for elig, group in df_analysis.groupby('daca_eligible'):
    status = "DACA Eligible" if elig else "DACA Ineligible"
    print(f"\n{status}:")
    print(f"  N: {len(group):,}")
    print(f"  Mean age: {group['AGE'].mean():.1f}")
    print(f"  % Female: {group['female'].mean()*100:.1f}%")
    print(f"  % Married: {group['married'].mean()*100:.1f}%")
    print(f"  % Employed: {group['employed'].mean()*100:.1f}%")
    print(f"  % Full-time: {group['fulltime'].mean()*100:.1f}%")

# Year-by-year statistics
print("\nYear-by-year full-time employment rates:")
yearly = df_analysis.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
yearly.columns = ['Ineligible', 'Eligible']
print(yearly.round(4))

print("\n" + "=" * 60)
print("PREFERRED ESTIMATE (Model 4 with State and Year FE)")
print("=" * 60)
print(f"\nEffect of DACA eligibility on full-time employment:")
print(f"  Coefficient: {did_coef:.6f}")
print(f"  Standard Error: {did_se:.6f}")
print(f"  95% CI: [{did_ci[0]:.6f}, {did_ci[1]:.6f}]")
print(f"  t-statistic: {did_tstat:.4f}")
print(f"  p-value: {did_pval:.4f}")
print(f"  N: {int(model4.nobs):,}")

# Save results to file
results = {
    'effect': did_coef,
    'se': did_se,
    'ci_lower': did_ci[0],
    'ci_upper': did_ci[1],
    'tstat': did_tstat,
    'pval': did_pval,
    'n': int(model4.nobs),
    'n_eligible_pre': len(df_analysis[(df_analysis['daca_eligible']) & (df_analysis['post'] == 0)]),
    'n_eligible_post': len(df_analysis[(df_analysis['daca_eligible']) & (df_analysis['post'] == 1)]),
    'n_ineligible_pre': len(df_analysis[(~df_analysis['daca_eligible']) & (df_analysis['post'] == 0)]),
    'n_ineligible_post': len(df_analysis[(~df_analysis['daca_eligible']) & (df_analysis['post'] == 1)]),
    'raw_did': did_effect,
    'eligible_pre_mean': eligible_pre,
    'eligible_post_mean': eligible_post,
    'ineligible_pre_mean': ineligible_pre,
    'ineligible_post_mean': ineligible_post,
}

# Save models for the report
import pickle
with open('analysis_results.pkl', 'wb') as f:
    pickle.dump({
        'model1': model1,
        'model2': model2,
        'model3': model3,
        'model4': model4,
        'model_emp': model_emp,
        'model_extensive': model_extensive,
        'results': results,
        'yearly_rates': yearly,
        'summary': summary
    }, f)

print("\nResults saved to analysis_results.pkl")
print("\nAnalysis complete!")
