"""
DACA Replication Study: Effect on Full-Time Employment
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on full-time employment (35+ hrs/week)?

Identification Strategy: Difference-in-Differences
- Treatment: DACA-eligible individuals (based on age at arrival, age as of 2012, and non-citizen status)
- Control: Similar Mexican-born Hispanic non-citizens who are NOT DACA-eligible
- Pre-period: 2006-2011
- Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DACA REPLICATION STUDY: EFFECT ON FULL-TIME EMPLOYMENT")
print("=" * 80)

# Load data
print("\n1. LOADING DATA...")
data_path = "data/data.csv"

# Read in chunks due to large file size
chunks = []
chunksize = 1000000

# First, filter to only Mexican-born, Hispanic-Mexican individuals to reduce memory usage
print("   Reading data in chunks and filtering to target population...")

for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize)):
    # Filter to Hispanic-Mexican (HISPAN == 1) and born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    if (i + 1) % 10 == 0:
        print(f"   Processed {(i+1) * chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"   Total observations after filtering to Mexican-born Hispanic-Mexican: {len(df):,}")

# Save memory
del chunks

print("\n2. VARIABLE CONSTRUCTION...")

# ============================================================================
# Define DACA Eligibility
# ============================================================================
# DACA requirements (as of June 15, 2012):
# 1. Arrived in US before 16th birthday (age at arrival < 16)
# 2. Under 31 years old as of June 15, 2012 (born after June 15, 1981)
# 3. Continuous residence since June 15, 2007 (arrived by 2007)
# 4. Not a citizen (undocumented - CITIZEN == 3 means "Not a citizen")

# Calculate age at immigration
# YRIMMIG gives year of immigration, BIRTHYR gives birth year
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']

# Age as of June 15, 2012
# Since we don't have exact birth month, we use birth year
# Someone born in 1981 would be 30 or 31 on June 15, 2012
# To be conservative, those born in 1982 or later definitely meet the age requirement
df['age_in_2012'] = 2012 - df['BIRTHYR']

# Criterion 1: Arrived before 16th birthday
df['arrived_before_16'] = (df['age_at_immigration'] < 16) & (df['age_at_immigration'] >= 0)

# Criterion 2: Under 31 as of June 15, 2012
# Born after June 15, 1981 means age < 31 on June 15, 2012
# Since we only have birth year, use BIRTHYR >= 1982 to be conservative
# or BIRTHYR == 1981 with some uncertainty
df['under_31_in_2012'] = df['BIRTHYR'] >= 1982

# Criterion 3: Continuous residence since June 15, 2007
# Must have arrived by 2007
df['arrived_by_2007'] = df['YRIMMIG'] <= 2007

# Criterion 4: Not a citizen (undocumented)
# CITIZEN == 3 means "Not a citizen"
# We cannot distinguish documented from undocumented, so we assume non-citizens are potentially undocumented
df['not_citizen'] = df['CITIZEN'] == 3

# Additional requirement: Must be at least 15 years old to apply (working age)
# We'll focus on working-age population (16-40 to have good overlap between treated and control)
df['working_age'] = (df['AGE'] >= 16) & (df['AGE'] <= 40)

# DACA eligible: meets all criteria
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['under_31_in_2012'] &
    df['arrived_by_2007'] &
    df['not_citizen']
)

print(f"   DACA eligible observations: {df['daca_eligible'].sum():,}")
print(f"   Non-DACA eligible observations: {(~df['daca_eligible']).sum():,}")

# ============================================================================
# Define Outcome: Full-Time Employment
# ============================================================================
# Full-time employment = usually working 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A)
# EMPSTAT: 1 = Employed, 2 = Unemployed, 3 = Not in labor force

df['employed'] = df['EMPSTAT'] == 1
df['fulltime'] = (df['UHRSWORK'] >= 35) & (df['UHRSWORK'] < 99)

# Full-time employed: employed AND working 35+ hours
df['fulltime_employed'] = df['employed'] & df['fulltime']

print(f"   Full-time employed observations: {df['fulltime_employed'].sum():,}")

# ============================================================================
# Define Treatment Period
# ============================================================================
# Pre-DACA: 2006-2011
# Post-DACA: 2013-2016 (2012 excluded as transition year)
df['post_daca'] = df['YEAR'] >= 2013

# Create year indicators
df['year'] = df['YEAR']

print(f"   Pre-DACA (2006-2011) observations: {(df['YEAR'] <= 2011).sum():,}")
print(f"   Post-DACA (2013-2016) observations: {(df['YEAR'] >= 2013).sum():,}")
print(f"   Transition year (2012) observations: {(df['YEAR'] == 2012).sum():,}")

# ============================================================================
# Sample Restrictions
# ============================================================================
print("\n3. SAMPLE RESTRICTIONS...")

# Restrict to working-age population
sample = df[df['working_age']].copy()
print(f"   After restricting to working age (16-40): {len(sample):,}")

# Exclude 2012 (transition year - DACA announced mid-year)
sample = sample[sample['YEAR'] != 2012]
print(f"   After excluding 2012: {len(sample):,}")

# Restrict to non-citizens only (our comparison is within non-citizens)
sample = sample[sample['not_citizen']].copy()
print(f"   After restricting to non-citizens: {len(sample):,}")

# Create clean treatment indicator
sample['treated'] = sample['daca_eligible'].astype(int)
sample['post'] = sample['post_daca'].astype(int)
sample['treated_post'] = sample['treated'] * sample['post']

# Outcome as integer
sample['y'] = sample['fulltime_employed'].astype(int)

print(f"\n   Final analysis sample: {len(sample):,}")
print(f"   Treatment group (DACA-eligible): {sample['treated'].sum():,}")
print(f"   Control group (not DACA-eligible): {(1 - sample['treated']).sum():,}")

# ============================================================================
# Descriptive Statistics
# ============================================================================
print("\n4. DESCRIPTIVE STATISTICS...")

# Summary by treatment status and period
summary = sample.groupby(['treated', 'post']).agg({
    'y': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'SEX': lambda x: (x == 1).mean(),  # Male proportion
    'EDUC': 'mean'
}).round(4)

print("\n   Summary Statistics by Treatment Status and Period:")
print(summary)

# Pre-period means
pre_treat = sample[(sample['treated'] == 1) & (sample['post'] == 0)]['y'].mean()
pre_control = sample[(sample['treated'] == 0) & (sample['post'] == 0)]['y'].mean()
post_treat = sample[(sample['treated'] == 1) & (sample['post'] == 1)]['y'].mean()
post_control = sample[(sample['treated'] == 0) & (sample['post'] == 1)]['y'].mean()

print(f"\n   Full-time employment rates:")
print(f"   Pre-period, DACA-eligible: {pre_treat:.4f}")
print(f"   Pre-period, Control: {pre_control:.4f}")
print(f"   Post-period, DACA-eligible: {post_treat:.4f}")
print(f"   Post-period, Control: {post_control:.4f}")

# Simple DiD estimate
did_simple = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n   Simple DiD estimate: {did_simple:.4f}")

# ============================================================================
# Main Regression Analysis
# ============================================================================
print("\n5. MAIN REGRESSION ANALYSIS...")

# Model 1: Basic DiD
print("\n   Model 1: Basic Difference-in-Differences")
model1 = smf.ols('y ~ treated + post + treated_post', data=sample).fit(cov_type='HC1')
print(f"   DiD coefficient (treated_post): {model1.params['treated_post']:.6f}")
print(f"   Standard error: {model1.bse['treated_post']:.6f}")
print(f"   95% CI: [{model1.conf_int().loc['treated_post', 0]:.6f}, {model1.conf_int().loc['treated_post', 1]:.6f}]")
print(f"   p-value: {model1.pvalues['treated_post']:.6f}")
print(f"   N: {int(model1.nobs):,}")
print(f"   R-squared: {model1.rsquared:.4f}")

# Model 2: With demographic controls
print("\n   Model 2: DiD with Demographic Controls")
sample['male'] = (sample['SEX'] == 1).astype(int)
sample['age_sq'] = sample['AGE'] ** 2
sample['married'] = (sample['MARST'] == 1).astype(int)

model2 = smf.ols('y ~ treated + post + treated_post + AGE + age_sq + male + married + C(EDUC)',
                 data=sample).fit(cov_type='HC1')
print(f"   DiD coefficient (treated_post): {model2.params['treated_post']:.6f}")
print(f"   Standard error: {model2.bse['treated_post']:.6f}")
print(f"   95% CI: [{model2.conf_int().loc['treated_post', 0]:.6f}, {model2.conf_int().loc['treated_post', 1]:.6f}]")
print(f"   p-value: {model2.pvalues['treated_post']:.6f}")
print(f"   N: {int(model2.nobs):,}")
print(f"   R-squared: {model2.rsquared:.4f}")

# Model 3: With state and year fixed effects
print("\n   Model 3: DiD with State and Year Fixed Effects")
model3 = smf.ols('y ~ treated + treated_post + AGE + age_sq + male + married + C(EDUC) + C(STATEFIP) + C(YEAR)',
                 data=sample).fit(cov_type='HC1')
print(f"   DiD coefficient (treated_post): {model3.params['treated_post']:.6f}")
print(f"   Standard error: {model3.bse['treated_post']:.6f}")
print(f"   95% CI: [{model3.conf_int().loc['treated_post', 0]:.6f}, {model3.conf_int().loc['treated_post', 1]:.6f}]")
print(f"   p-value: {model3.pvalues['treated_post']:.6f}")
print(f"   N: {int(model3.nobs):,}")
print(f"   R-squared: {model3.rsquared:.4f}")

# ============================================================================
# Weighted Analysis (using person weights)
# ============================================================================
print("\n6. WEIGHTED ANALYSIS...")

# Model 4: Weighted DiD with full controls
print("\n   Model 4: Weighted DiD with Full Controls")
import statsmodels.api as sm

# Prepare data for weighted regression
X_cols = ['treated', 'treated_post', 'AGE', 'age_sq', 'male', 'married']

# Add education dummies
for educ_val in sample['EDUC'].unique():
    if educ_val != sample['EDUC'].min():  # Use minimum as reference
        sample[f'educ_{educ_val}'] = (sample['EDUC'] == educ_val).astype(int)
        X_cols.append(f'educ_{educ_val}')

# Add state dummies
for state in sample['STATEFIP'].unique()[1:]:  # Skip first as reference
    sample[f'state_{state}'] = (sample['STATEFIP'] == state).astype(int)
    X_cols.append(f'state_{state}')

# Add year dummies
for year in sample['YEAR'].unique()[1:]:  # Skip first as reference
    sample[f'year_{year}'] = (sample['YEAR'] == year).astype(int)
    X_cols.append(f'year_{year}')

# Weighted least squares
X = sample[X_cols].copy()
X = sm.add_constant(X)
y = sample['y']
weights = sample['PERWT']

# Drop any rows with missing weights
valid_idx = ~(weights.isna() | y.isna() | X.isna().any(axis=1))
X_clean = X[valid_idx]
y_clean = y[valid_idx]
weights_clean = weights[valid_idx]

model4 = sm.WLS(y_clean, X_clean, weights=weights_clean).fit(cov_type='HC1')
print(f"   DiD coefficient (treated_post): {model4.params['treated_post']:.6f}")
print(f"   Standard error: {model4.bse['treated_post']:.6f}")
print(f"   95% CI: [{model4.conf_int().loc['treated_post', 0]:.6f}, {model4.conf_int().loc['treated_post', 1]:.6f}]")
print(f"   p-value: {model4.pvalues['treated_post']:.6f}")
print(f"   N: {int(model4.nobs):,}")

# ============================================================================
# Robustness Checks
# ============================================================================
print("\n7. ROBUSTNESS CHECKS...")

# 7.1 Placebo test: Use 2010 as fake treatment year
print("\n   7.1 Placebo Test (fake treatment in 2010):")
sample_placebo = sample[sample['YEAR'] <= 2011].copy()
sample_placebo['post_placebo'] = (sample_placebo['YEAR'] >= 2010).astype(int)
sample_placebo['treated_post_placebo'] = sample_placebo['treated'] * sample_placebo['post_placebo']

model_placebo = smf.ols('y ~ treated + post_placebo + treated_post_placebo + AGE + age_sq + male + married + C(EDUC)',
                        data=sample_placebo).fit(cov_type='HC1')
print(f"   Placebo DiD coefficient: {model_placebo.params['treated_post_placebo']:.6f}")
print(f"   Standard error: {model_placebo.bse['treated_post_placebo']:.6f}")
print(f"   p-value: {model_placebo.pvalues['treated_post_placebo']:.6f}")

# 7.2 Alternative age restrictions
print("\n   7.2 Sensitivity to Age Restrictions (18-35):")
sample_alt = df[(df['AGE'] >= 18) & (df['AGE'] <= 35) & (df['YEAR'] != 2012) & df['not_citizen']].copy()
sample_alt['treated'] = sample_alt['daca_eligible'].astype(int)
sample_alt['post'] = sample_alt['post_daca'].astype(int)
sample_alt['treated_post'] = sample_alt['treated'] * sample_alt['post']
sample_alt['y'] = sample_alt['fulltime_employed'].astype(int)
sample_alt['male'] = (sample_alt['SEX'] == 1).astype(int)
sample_alt['age_sq'] = sample_alt['AGE'] ** 2
sample_alt['married'] = (sample_alt['MARST'] == 1).astype(int)

model_alt = smf.ols('y ~ treated + post + treated_post + AGE + age_sq + male + married + C(EDUC)',
                    data=sample_alt).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_alt.params['treated_post']:.6f}")
print(f"   Standard error: {model_alt.bse['treated_post']:.6f}")
print(f"   p-value: {model_alt.pvalues['treated_post']:.6f}")
print(f"   N: {int(model_alt.nobs):,}")

# 7.3 By gender
print("\n   7.3 Heterogeneity by Gender:")
for gender, label in [(1, 'Male'), (2, 'Female')]:
    sample_gender = sample[sample['SEX'] == gender].copy()
    model_gender = smf.ols('y ~ treated + post + treated_post + AGE + age_sq + married + C(EDUC)',
                           data=sample_gender).fit(cov_type='HC1')
    print(f"   {label}: DiD = {model_gender.params['treated_post']:.6f} (SE = {model_gender.bse['treated_post']:.6f})")

# 7.4 Event study
print("\n   7.4 Event Study (Year-by-Year Effects):")
sample['year_2006'] = (sample['YEAR'] == 2006).astype(int)
sample['year_2007'] = (sample['YEAR'] == 2007).astype(int)
sample['year_2008'] = (sample['YEAR'] == 2008).astype(int)
sample['year_2009'] = (sample['YEAR'] == 2009).astype(int)
sample['year_2010'] = (sample['YEAR'] == 2010).astype(int)
sample['year_2011'] = (sample['YEAR'] == 2011).astype(int)
sample['year_2013'] = (sample['YEAR'] == 2013).astype(int)
sample['year_2014'] = (sample['YEAR'] == 2014).astype(int)
sample['year_2015'] = (sample['YEAR'] == 2015).astype(int)
sample['year_2016'] = (sample['YEAR'] == 2016).astype(int)

# Interactions (2011 as reference year)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    sample[f'treated_year_{year}'] = sample['treated'] * sample[f'year_{year}']

event_formula = 'y ~ treated + ' + ' + '.join([f'year_{y}' for y in [2006,2007,2008,2009,2010,2013,2014,2015,2016]]) + ' + ' + \
                ' + '.join([f'treated_year_{y}' for y in [2006,2007,2008,2009,2010,2013,2014,2015,2016]]) + \
                ' + AGE + age_sq + male + married + C(EDUC)'

model_event = smf.ols(event_formula, data=sample).fit(cov_type='HC1')

print("   Year-by-treatment interactions (reference: 2011):")
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treated_year_{year}']
    se = model_event.bse[f'treated_year_{year}']
    pval = model_event.pvalues[f'treated_year_{year}']
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"   {year}: {coef:.4f} ({se:.4f}) {sig}")

# ============================================================================
# Save Results for Report
# ============================================================================
print("\n8. SAVING RESULTS...")

results = {
    'preferred_estimate': model3.params['treated_post'],
    'preferred_se': model3.bse['treated_post'],
    'preferred_ci_lower': model3.conf_int().loc['treated_post', 0],
    'preferred_ci_upper': model3.conf_int().loc['treated_post', 1],
    'preferred_pvalue': model3.pvalues['treated_post'],
    'preferred_n': int(model3.nobs),
    'model1_coef': model1.params['treated_post'],
    'model1_se': model1.bse['treated_post'],
    'model2_coef': model2.params['treated_post'],
    'model2_se': model2.bse['treated_post'],
    'model3_coef': model3.params['treated_post'],
    'model3_se': model3.bse['treated_post'],
    'model4_coef': model4.params['treated_post'],
    'model4_se': model4.bse['treated_post'],
    'pre_treat_mean': pre_treat,
    'pre_control_mean': pre_control,
    'post_treat_mean': post_treat,
    'post_control_mean': post_control,
    'simple_did': did_simple,
    'n_treatment': int(sample['treated'].sum()),
    'n_control': int((1-sample['treated']).sum()),
    'n_total': len(sample)
}

# Save results to file
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("   Results saved to results.json")

# ============================================================================
# Create summary tables for report
# ============================================================================

# Table 1: Summary Statistics
print("\n" + "=" * 80)
print("TABLE 1: SUMMARY STATISTICS")
print("=" * 80)

summary_stats = sample.groupby('treated').agg({
    'y': 'mean',
    'AGE': 'mean',
    'male': 'mean',
    'married': 'mean',
    'EDUC': 'mean',
    'PERWT': 'sum'
}).round(4)
summary_stats.columns = ['Full-time Employment', 'Age', 'Male', 'Married', 'Education', 'Population (weighted)']
summary_stats.index = ['Control', 'DACA-Eligible']
print(summary_stats.to_string())

# Table 2: Main Results
print("\n" + "=" * 80)
print("TABLE 2: MAIN DIFFERENCE-IN-DIFFERENCES RESULTS")
print("=" * 80)
print(f"{'Model':<40} {'Coefficient':<15} {'SE':<15} {'p-value':<10}")
print("-" * 80)
print(f"{'Basic DiD':<40} {model1.params['treated_post']:.6f}       {model1.bse['treated_post']:.6f}       {model1.pvalues['treated_post']:.4f}")
print(f"{'+ Demographics':<40} {model2.params['treated_post']:.6f}       {model2.bse['treated_post']:.6f}       {model2.pvalues['treated_post']:.4f}")
print(f"{'+ State & Year FE':<40} {model3.params['treated_post']:.6f}       {model3.bse['treated_post']:.6f}       {model3.pvalues['treated_post']:.4f}")
print(f"{'Weighted with Full Controls':<40} {model4.params['treated_post']:.6f}       {model4.bse['treated_post']:.6f}       {model4.pvalues['treated_post']:.4f}")

# Table 3: Event Study
print("\n" + "=" * 80)
print("TABLE 3: EVENT STUDY ESTIMATES")
print("=" * 80)
print(f"{'Year':<10} {'Coefficient':<15} {'SE':<15} {'p-value':<10}")
print("-" * 50)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treated_year_{year}']
    se = model_event.bse[f'treated_year_{year}']
    pval = model_event.pvalues[f'treated_year_{year}']
    print(f"{year:<10} {coef:.6f}       {se:.6f}       {pval:.4f}")

print("\nReference year: 2011 (year before DACA)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"""
PREFERRED ESTIMATE SUMMARY:
==========================
Effect of DACA eligibility on full-time employment: {results['preferred_estimate']:.4f}
Standard Error: {results['preferred_se']:.4f}
95% Confidence Interval: [{results['preferred_ci_lower']:.4f}, {results['preferred_ci_upper']:.4f}]
p-value: {results['preferred_pvalue']:.4f}
Sample Size: {results['preferred_n']:,}

Interpretation: DACA eligibility is associated with a {results['preferred_estimate']*100:.2f} percentage point
change in the probability of full-time employment among Mexican-born Hispanic non-citizens.
""")
