"""
DACA Replication Study - Analysis Script
Replication 21

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the US,
what was the causal impact of DACA eligibility on full-time employment (35+ hours/week)?

Author: Independent Replication
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

print("\n1. Loading data...")
data_path = "data/data.csv"

# Load data in chunks to handle large file
chunks = []
for chunk in pd.read_csv(data_path, chunksize=1000000):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

print(f"   Total observations loaded: {len(df):,}")
print(f"   Years in data: {df['YEAR'].min()} - {df['YEAR'].max()}")

# =============================================================================
# 2. Sample Selection
# =============================================================================
print("\n2. Applying sample restrictions...")

# Step 2a: Hispanic-Mexican ethnicity (HISPAN == 1)
df_sample = df[df['HISPAN'] == 1].copy()
print(f"   After Hispanic-Mexican restriction: {len(df_sample):,}")

# Step 2b: Born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"   After Mexico birthplace restriction: {len(df_sample):,}")

# Step 2c: Non-citizen (CITIZEN == 3)
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"   After non-citizen restriction: {len(df_sample):,}")

# Step 2d: Working age (16-45)
df_sample = df_sample[(df_sample['AGE'] >= 16) & (df_sample['AGE'] <= 45)].copy()
print(f"   After age 16-45 restriction: {len(df_sample):,}")

# Step 2e: Exclude 2012 (partial treatment year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_sample):,}")

# =============================================================================
# 3. Create Variables
# =============================================================================
print("\n3. Creating analysis variables...")

# 3a. Full-time employment indicator (outcome)
# Full-time = usually works 35+ hours per week
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)
print(f"   Full-time employment rate: {df_sample['fulltime'].mean():.3f}")

# 3b. Post-DACA indicator
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
print(f"   Observations in post period (2013-2016): {df_sample['post'].sum():,}")

# 3c. DACA eligibility indicator
# Criteria:
#   1. Arrived before 16th birthday: age at arrival < 16
#   2. Born after June 15, 1981 (not yet 31 as of June 15, 2012)
#   3. Arrived by 2007 (lived continuously since June 15, 2007)
#   4. Non-citizen (already filtered)

# Calculate age at arrival
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# DACA eligibility
# Note: Using BIRTHQTR to refine the birthdate cutoff
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Born after June 15, 1981 means BIRTHYR > 1981, OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
df_sample['born_after_cutoff'] = (
    (df_sample['BIRTHYR'] > 1981) |
    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))
)

df_sample['daca_eligible'] = (
    (df_sample['age_at_arrival'] < 16) &  # Arrived before 16th birthday
    (df_sample['born_after_cutoff']) &     # Born after June 15, 1981
    (df_sample['YRIMMIG'] <= 2007) &       # Arrived by 2007
    (df_sample['YRIMMIG'] > 0)             # Valid immigration year
).astype(int)

print(f"   DACA eligible: {df_sample['daca_eligible'].sum():,} ({df_sample['daca_eligible'].mean():.3f})")

# 3d. Interaction term (DiD coefficient)
df_sample['daca_post'] = df_sample['daca_eligible'] * df_sample['post']

# 3e. Create education categories
# EDUC: 0=N/A, 1=None/preschool, 2=Grade 1-4, 3=Grade 5-8, 4=Grade 9, 5=Grade 10,
#       6=Grade 11, 7=Grade 12/HS diploma, 8=1 yr college, 9=2 yrs college,
#       10=3 yrs college, 11=4+ yrs college
df_sample['educ_cat'] = pd.cut(df_sample['EDUC'],
                                bins=[-1, 3, 6, 7, 10, 11],
                                labels=['less_than_hs', 'some_hs', 'hs_grad', 'some_college', 'college_plus'])

# 3f. Create female indicator
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)

# =============================================================================
# 4. Descriptive Statistics
# =============================================================================
print("\n4. Descriptive Statistics")
print("-" * 70)

# Summary by treatment group and period
print("\n   Table 1: Sample means by DACA eligibility and period")
print("-" * 70)

summary_stats = df_sample.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'AGE': 'mean',
    'female': 'mean',
    'UHRSWORK': 'mean'
}).round(3)

print(summary_stats)

# Full-time employment by group
print("\n   Full-time employment rates:")
for eligible in [0, 1]:
    for post in [0, 1]:
        subset = df_sample[(df_sample['daca_eligible'] == eligible) & (df_sample['post'] == post)]
        rate = subset['fulltime'].mean()
        n = len(subset)
        elig_label = "Eligible" if eligible else "Ineligible"
        period_label = "Post" if post else "Pre"
        print(f"   {elig_label}, {period_label}: {rate:.4f} (n={n:,})")

# Simple DiD calculation
pre_treat = df_sample[(df_sample['daca_eligible'] == 1) & (df_sample['post'] == 0)]['fulltime'].mean()
post_treat = df_sample[(df_sample['daca_eligible'] == 1) & (df_sample['post'] == 1)]['fulltime'].mean()
pre_control = df_sample[(df_sample['daca_eligible'] == 0) & (df_sample['post'] == 0)]['fulltime'].mean()
post_control = df_sample[(df_sample['daca_eligible'] == 0) & (df_sample['post'] == 1)]['fulltime'].mean()

simple_did = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n   Simple DiD estimate: {simple_did:.4f}")
print(f"   Treatment group change: {post_treat - pre_treat:.4f}")
print(f"   Control group change: {post_control - pre_control:.4f}")

# =============================================================================
# 5. Main DiD Regression Analysis
# =============================================================================
print("\n5. Difference-in-Differences Regression Analysis")
print("-" * 70)

# 5a. Basic DiD (no controls)
print("\n   Model 1: Basic DiD (no controls)")
model1 = smf.ols('fulltime ~ daca_eligible + post + daca_post', data=df_sample).fit()
print(f"   DiD coefficient: {model1.params['daca_post']:.4f}")
print(f"   Standard error: {model1.bse['daca_post']:.4f}")
print(f"   t-statistic: {model1.tvalues['daca_post']:.4f}")
print(f"   p-value: {model1.pvalues['daca_post']:.4f}")
print(f"   N: {int(model1.nobs):,}")

# 5b. DiD with demographic controls
print("\n   Model 2: DiD with demographic controls")
model2 = smf.ols('fulltime ~ daca_eligible + post + daca_post + AGE + I(AGE**2) + female',
                 data=df_sample).fit()
print(f"   DiD coefficient: {model2.params['daca_post']:.4f}")
print(f"   Standard error: {model2.bse['daca_post']:.4f}")
print(f"   t-statistic: {model2.tvalues['daca_post']:.4f}")
print(f"   p-value: {model2.pvalues['daca_post']:.4f}")
print(f"   N: {int(model2.nobs):,}")

# 5c. DiD with demographic controls and year fixed effects
print("\n   Model 3: DiD with demographics + year FE")
df_sample['year_factor'] = df_sample['YEAR'].astype(str)
model3 = smf.ols('fulltime ~ daca_eligible + daca_post + AGE + I(AGE**2) + female + C(year_factor)',
                 data=df_sample).fit()
print(f"   DiD coefficient: {model3.params['daca_post']:.4f}")
print(f"   Standard error: {model3.bse['daca_post']:.4f}")
print(f"   t-statistic: {model3.tvalues['daca_post']:.4f}")
print(f"   p-value: {model3.pvalues['daca_post']:.4f}")
print(f"   N: {int(model3.nobs):,}")

# 5d. Full model with state fixed effects
print("\n   Model 4: DiD with demographics + year FE + state FE")
df_sample['state_factor'] = df_sample['STATEFIP'].astype(str)
model4 = smf.ols('fulltime ~ daca_eligible + daca_post + AGE + I(AGE**2) + female + C(year_factor) + C(state_factor)',
                 data=df_sample).fit()
print(f"   DiD coefficient: {model4.params['daca_post']:.4f}")
print(f"   Standard error: {model4.bse['daca_post']:.4f}")
print(f"   t-statistic: {model4.tvalues['daca_post']:.4f}")
print(f"   p-value: {model4.pvalues['daca_post']:.4f}")
print(f"   N: {int(model4.nobs):,}")
print(f"   R-squared: {model4.rsquared:.4f}")

# 5e. Clustered standard errors at state level (Preferred specification)
print("\n   Model 5: PREFERRED - DiD with controls + FE + clustered SE")
model5 = smf.ols('fulltime ~ daca_eligible + daca_post + AGE + I(AGE**2) + female + C(year_factor) + C(state_factor)',
                 data=df_sample).fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"   DiD coefficient: {model5.params['daca_post']:.4f}")
print(f"   Clustered SE: {model5.bse['daca_post']:.4f}")
print(f"   t-statistic: {model5.tvalues['daca_post']:.4f}")
print(f"   p-value: {model5.pvalues['daca_post']:.4f}")

# Confidence interval
ci_low = model5.params['daca_post'] - 1.96 * model5.bse['daca_post']
ci_high = model5.params['daca_post'] + 1.96 * model5.bse['daca_post']
print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"   N: {int(model5.nobs):,}")
print(f"   R-squared: {model5.rsquared:.4f}")

# =============================================================================
# 6. Robustness Checks
# =============================================================================
print("\n6. Robustness Checks")
print("-" * 70)

# 6a. Restrict to employed individuals only
print("\n   6a. Conditional on employment (EMPSTAT == 1)")
df_employed = df_sample[df_sample['EMPSTAT'] == 1].copy()
model_employed = smf.ols('fulltime ~ daca_eligible + daca_post + AGE + I(AGE**2) + female + C(year_factor) + C(state_factor)',
                         data=df_employed).fit(cov_type='cluster', cov_kwds={'groups': df_employed['STATEFIP']})
print(f"   DiD coefficient: {model_employed.params['daca_post']:.4f}")
print(f"   Clustered SE: {model_employed.bse['daca_post']:.4f}")
print(f"   p-value: {model_employed.pvalues['daca_post']:.4f}")
print(f"   N: {int(model_employed.nobs):,}")

# 6b. Alternative age range (18-35)
print("\n   6b. Restricted age range (18-35)")
df_age_restrict = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 35)].copy()
model_age = smf.ols('fulltime ~ daca_eligible + daca_post + AGE + I(AGE**2) + female + C(year_factor) + C(state_factor)',
                    data=df_age_restrict).fit(cov_type='cluster', cov_kwds={'groups': df_age_restrict['STATEFIP']})
print(f"   DiD coefficient: {model_age.params['daca_post']:.4f}")
print(f"   Clustered SE: {model_age.bse['daca_post']:.4f}")
print(f"   p-value: {model_age.pvalues['daca_post']:.4f}")
print(f"   N: {int(model_age.nobs):,}")

# 6c. Male only
print("\n   6c. Male only sample")
df_male = df_sample[df_sample['female'] == 0].copy()
model_male = smf.ols('fulltime ~ daca_eligible + daca_post + AGE + I(AGE**2) + C(year_factor) + C(state_factor)',
                     data=df_male).fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
print(f"   DiD coefficient: {model_male.params['daca_post']:.4f}")
print(f"   Clustered SE: {model_male.bse['daca_post']:.4f}")
print(f"   p-value: {model_male.pvalues['daca_post']:.4f}")
print(f"   N: {int(model_male.nobs):,}")

# 6d. Female only
print("\n   6d. Female only sample")
df_female = df_sample[df_sample['female'] == 1].copy()
model_female = smf.ols('fulltime ~ daca_eligible + daca_post + AGE + I(AGE**2) + C(year_factor) + C(state_factor)',
                       data=df_female).fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})
print(f"   DiD coefficient: {model_female.params['daca_post']:.4f}")
print(f"   Clustered SE: {model_female.bse['daca_post']:.4f}")
print(f"   p-value: {model_female.pvalues['daca_post']:.4f}")
print(f"   N: {int(model_female.nobs):,}")

# 6e. Placebo test: Pre-period only (using 2010 as fake treatment)
print("\n   6e. Placebo test: Pre-2010 vs 2010-2011")
df_pre = df_sample[df_sample['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['placebo_did'] = df_pre['daca_eligible'] * df_pre['placebo_post']
model_placebo = smf.ols('fulltime ~ daca_eligible + placebo_post + placebo_did + AGE + I(AGE**2) + female + C(state_factor)',
                        data=df_pre).fit(cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})
print(f"   Placebo DiD coefficient: {model_placebo.params['placebo_did']:.4f}")
print(f"   Clustered SE: {model_placebo.bse['placebo_did']:.4f}")
print(f"   p-value: {model_placebo.pvalues['placebo_did']:.4f}")
print(f"   N: {int(model_placebo.nobs):,}")

# =============================================================================
# 7. Event Study / Dynamic Effects
# =============================================================================
print("\n7. Event Study (Year-by-Year Effects)")
print("-" * 70)

# Create year interactions with DACA eligibility
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]  # excluding 2012
for year in years:
    df_sample[f'daca_x_{year}'] = (df_sample['daca_eligible'] * (df_sample['YEAR'] == year)).astype(int)

# Reference year: 2011 (last pre-treatment year)
event_formula = 'fulltime ~ daca_eligible + ' + ' + '.join([f'daca_x_{y}' for y in years if y != 2011]) + \
                ' + AGE + I(AGE**2) + female + C(year_factor) + C(state_factor)'
model_event = smf.ols(event_formula, data=df_sample).fit(cov_type='cluster',
                                                          cov_kwds={'groups': df_sample['STATEFIP']})

print("   Year-specific effects (relative to 2011):")
for year in years:
    if year != 2011:
        coef = model_event.params[f'daca_x_{year}']
        se = model_event.bse[f'daca_x_{year}']
        pval = model_event.pvalues[f'daca_x_{year}']
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"   {year}: {coef:7.4f} ({se:.4f}) {sig}")

# =============================================================================
# 8. Summary Output
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF RESULTS")
print("=" * 70)

print(f"""
PREFERRED ESTIMATE (Model 5):
-----------------------------
Effect of DACA eligibility on full-time employment: {model5.params['daca_post']:.4f}
Standard Error (clustered at state level): {model5.bse['daca_post']:.4f}
95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]
Sample Size: {int(model5.nobs):,}
P-value: {model5.pvalues['daca_post']:.4f}

INTERPRETATION:
DACA eligibility is associated with a {abs(model5.params['daca_post'])*100:.2f} percentage point
{'increase' if model5.params['daca_post'] > 0 else 'decrease'} in the probability of full-time employment
among Mexican-born Hispanic non-citizens.
{'This effect is statistically significant at the 5% level.' if model5.pvalues['daca_post'] < 0.05 else 'This effect is not statistically significant at the 5% level.'}

BASELINE RATE:
Pre-treatment full-time employment rate (eligible group): {pre_treat:.4f}
""")

# =============================================================================
# 9. Save Results for LaTeX Report
# =============================================================================
print("\n9. Saving results for report...")

# Create results dictionary
results = {
    'preferred_coef': model5.params['daca_post'],
    'preferred_se': model5.bse['daca_post'],
    'preferred_ci_low': ci_low,
    'preferred_ci_high': ci_high,
    'preferred_pval': model5.pvalues['daca_post'],
    'preferred_n': int(model5.nobs),
    'preferred_rsq': model5.rsquared,
    'simple_did': simple_did,
    'pre_treat': pre_treat,
    'post_treat': post_treat,
    'pre_control': pre_control,
    'post_control': post_control,
    'n_eligible_pre': len(df_sample[(df_sample['daca_eligible'] == 1) & (df_sample['post'] == 0)]),
    'n_eligible_post': len(df_sample[(df_sample['daca_eligible'] == 1) & (df_sample['post'] == 1)]),
    'n_ineligible_pre': len(df_sample[(df_sample['daca_eligible'] == 0) & (df_sample['post'] == 0)]),
    'n_ineligible_post': len(df_sample[(df_sample['daca_eligible'] == 0) & (df_sample['post'] == 1)]),
    'model1_coef': model1.params['daca_post'],
    'model1_se': model1.bse['daca_post'],
    'model2_coef': model2.params['daca_post'],
    'model2_se': model2.bse['daca_post'],
    'model3_coef': model3.params['daca_post'],
    'model3_se': model3.bse['daca_post'],
    'model4_coef': model4.params['daca_post'],
    'model4_se': model4.bse['daca_post'],
    'robust_employed_coef': model_employed.params['daca_post'],
    'robust_employed_se': model_employed.bse['daca_post'],
    'robust_age_coef': model_age.params['daca_post'],
    'robust_age_se': model_age.bse['daca_post'],
    'robust_male_coef': model_male.params['daca_post'],
    'robust_male_se': model_male.bse['daca_post'],
    'robust_female_coef': model_female.params['daca_post'],
    'robust_female_se': model_female.bse['daca_post'],
    'placebo_coef': model_placebo.params['placebo_did'],
    'placebo_se': model_placebo.bse['placebo_did'],
}

# Save to pickle for LaTeX
import pickle
with open('results_21.pkl', 'wb') as f:
    pickle.dump(results, f)

print("   Results saved to results_21.pkl")

# Also save descriptive stats
desc_stats = df_sample.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': ['mean', 'std'],
    'female': 'mean',
    'UHRSWORK': ['mean', 'std']
}).round(4)

desc_stats.to_csv('descriptive_stats_21.csv')
print("   Descriptive statistics saved to descriptive_stats_21.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
