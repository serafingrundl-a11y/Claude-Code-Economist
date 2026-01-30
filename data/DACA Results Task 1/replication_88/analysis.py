"""
DACA Replication Analysis
Research Question: What was the causal impact of DACA eligibility on full-time employment
among ethnically Hispanic-Mexican Mexican-born people in the US?

Analysis uses difference-in-differences approach comparing DACA-eligible vs ineligible
Mexican-born Hispanic individuals before and after DACA implementation (June 2012).

Memory-efficient version using chunked processing.
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
print("DACA REPLICATION ANALYSIS")
print("="*80)

# -----------------------------------------------------------------
# STEP 1: LOAD DATA IN CHUNKS, FILTERING AS WE GO
# -----------------------------------------------------------------
print("\n1. LOADING DATA (in chunks, filtering for relevant sample)...")

# Define columns to keep (reduce memory)
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
           'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT', 'UHRSWORK']

# Read in chunks and filter
chunk_size = 1000000
chunks = []
total_rows = 0

for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunk_size):
    total_rows += len(chunk)
    # Filter for our population of interest:
    # Hispanic-Mexican (HISPAN==1), Mexico-born (BPL==200), non-citizen (CITIZEN==3)
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3)].copy()
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"   Processed {total_rows:,} rows, kept {sum(len(c) for c in chunks):,} so far...")

df = pd.concat(chunks, ignore_index=True)
del chunks  # Free memory

print(f"\n   Total observations in raw data: ~{total_rows:,}")
print(f"   Observations after filtering for Hispanic-Mexican, Mexico-born, non-citizen: {len(df):,}")

# -----------------------------------------------------------------
# STEP 2: FURTHER SAMPLE RESTRICTIONS
# -----------------------------------------------------------------
print("\n2. FURTHER SAMPLE RESTRICTIONS...")

# Keep working-age population (16-64) for employment analysis
print(f"   - Filtering for working age (16-64)...")
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
print(f"     After filter: {len(df):,} observations")

# -----------------------------------------------------------------
# STEP 3: DEFINE DACA ELIGIBILITY
# -----------------------------------------------------------------
print("\n3. DEFINING DACA ELIGIBILITY...")
"""
DACA eligibility requirements (as of June 15, 2012):
1. Arrived in US before 16th birthday
2. Under 31 years old as of June 15, 2012 (born after June 15, 1981)
3. Continuously lived in US since June 15, 2007 (at least 5 years)
4. Present in US on June 15, 2012 (cannot verify directly, assume present if in ACS)
5. Not a citizen (already filtered)

For identification strategy, we need to identify:
- Who would have been eligible if present in 2012
- Key constraint: Under 31 on June 15, 2012 (birth year >= 1982, or 1981 with birthqtr > 2)
- Arrived before age 16
"""

def is_daca_eligible(row):
    """
    Determine if individual meets DACA eligibility criteria.
    We calculate eligibility as of June 15, 2012.
    """
    # Must have immigration year data
    if row['YRIMMIG'] == 0 or pd.isna(row['YRIMMIG']):
        return np.nan

    # 1. Arrived before 16th birthday
    age_at_immigration = row['YRIMMIG'] - row['BIRTHYR']
    if age_at_immigration >= 16:
        return 0

    # 2. Under 31 as of June 15, 2012 (born after June 15, 1981)
    # For simplicity, use birth year >= 1982, or 1981 with birth quarter 3 or 4
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']

    if birth_year > 1981:
        under_31 = True
    elif birth_year == 1981 and birth_qtr in [3, 4]:
        # Born July-Dec 1981: would be 30 years old in June 2012
        under_31 = True
    else:
        under_31 = False

    if not under_31:
        return 0

    # 3. Continuously in US since June 15, 2007 (5 years)
    # Immigration year must be <= 2007
    if row['YRIMMIG'] > 2007:
        return 0

    # All criteria met
    return 1

print("   Applying DACA eligibility criteria...")
df['daca_eligible'] = df.apply(is_daca_eligible, axis=1)

# Drop observations with missing eligibility
df_analysis = df[df['daca_eligible'].notna()].copy()
print(f"   Observations with determinable eligibility: {len(df_analysis):,}")

eligible_count = df_analysis['daca_eligible'].sum()
ineligible_count = len(df_analysis) - eligible_count
print(f"   DACA Eligible: {int(eligible_count):,}")
print(f"   DACA Ineligible: {int(ineligible_count):,}")

# Free memory
del df

# -----------------------------------------------------------------
# STEP 4: DEFINE OUTCOME VARIABLE
# -----------------------------------------------------------------
print("\n4. DEFINING OUTCOME VARIABLE...")
"""
Full-time employment: Usually working 35+ hours per week
UHRSWORK = usual hours worked per week
"""

# Create full-time employment indicator
# EMPSTAT == 1 means employed
# UHRSWORK >= 35 means full-time
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)
df_analysis['fulltime'] = ((df_analysis['EMPSTAT'] == 1) & (df_analysis['UHRSWORK'] >= 35)).astype(int)

print(f"   Overall employment rate: {df_analysis['employed'].mean()*100:.1f}%")
print(f"   Full-time employment rate: {df_analysis['fulltime'].mean()*100:.1f}%")

# -----------------------------------------------------------------
# STEP 5: DEFINE TREATMENT PERIOD
# -----------------------------------------------------------------
print("\n5. DEFINING TREATMENT PERIOD...")
"""
DACA was implemented June 15, 2012.
- Pre-period: 2006-2011 (before DACA)
- Transition: 2012 (partial implementation, exclude from main analysis)
- Post-period: 2013-2016 (after DACA fully in effect)
"""

# Create post indicator (excluding 2012 which is ambiguous)
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# For robustness, also create version that includes 2012 as post
df_analysis['post_incl_2012'] = (df_analysis['YEAR'] >= 2012).astype(int)

print("   Pre-period (2006-2011):")
pre_data = df_analysis[df_analysis['YEAR'] < 2012]
print(f"     Observations: {len(pre_data):,}")
print(f"     Years: {sorted(pre_data['YEAR'].unique())}")

print("   Post-period (2013-2016):")
post_data = df_analysis[(df_analysis['YEAR'] >= 2013) & (df_analysis['YEAR'] <= 2016)]
print(f"     Observations: {len(post_data):,}")
print(f"     Years: {sorted(post_data['YEAR'].unique())}")

# Exclude 2012 for main analysis
df_main = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"\n   Main analysis sample (excl. 2012): {len(df_main):,}")

# -----------------------------------------------------------------
# STEP 6: DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------
print("\n6. DESCRIPTIVE STATISTICS...")

# Create control variables
df_main['female'] = (df_main['SEX'] == 2).astype(int)
df_main['age_sq'] = df_main['AGE'] ** 2
df_main['married'] = (df_main['MARST'].isin([1, 2])).astype(int)
df_main['educ_hs'] = (df_main['EDUC'] >= 6).astype(int)  # High school or more

# Summary by eligibility and period
print("\n   Full-time Employment Rates by Group and Period:")
print("   " + "-"*60)

summary_stats = df_main.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean'],
    'AGE': ['mean'],
    'female': ['mean'],
    'PERWT': ['sum']
}).round(4)

print(summary_stats)

# Calculate simple DiD
pre_elig = df_main[(df_main['daca_eligible']==1) & (df_main['post']==0)]['fulltime'].mean()
post_elig = df_main[(df_main['daca_eligible']==1) & (df_main['post']==1)]['fulltime'].mean()
pre_inelig = df_main[(df_main['daca_eligible']==0) & (df_main['post']==0)]['fulltime'].mean()
post_inelig = df_main[(df_main['daca_eligible']==0) & (df_main['post']==1)]['fulltime'].mean()

print(f"\n   Simple Difference-in-Differences Calculation:")
print(f"   " + "-"*60)
print(f"   DACA Eligible:   Pre={pre_elig:.4f}  Post={post_elig:.4f}  Diff={post_elig-pre_elig:.4f}")
print(f"   DACA Ineligible: Pre={pre_inelig:.4f}  Post={post_inelig:.4f}  Diff={post_inelig-pre_inelig:.4f}")
print(f"   DiD Estimate: {(post_elig-pre_elig)-(post_inelig-pre_inelig):.4f}")

# -----------------------------------------------------------------
# STEP 7: REGRESSION ANALYSIS
# -----------------------------------------------------------------
print("\n7. REGRESSION ANALYSIS...")

# Create interaction term
df_main['eligible_x_post'] = df_main['daca_eligible'] * df_main['post']

# Model 1: Basic DiD (no controls)
print("\n   Model 1: Basic Difference-in-Differences")
print("   " + "-"*60)

model1 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post', data=df_main).fit()
print(f"   DiD Coefficient (eligible_x_post): {model1.params['eligible_x_post']:.4f}")
print(f"   Standard Error: {model1.bse['eligible_x_post']:.4f}")
print(f"   t-statistic: {model1.tvalues['eligible_x_post']:.4f}")
print(f"   p-value: {model1.pvalues['eligible_x_post']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['eligible_x_post', 0]:.4f}, {model1.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"   N: {int(model1.nobs):,}")
print(f"   R-squared: {model1.rsquared:.4f}")

# Model 2: DiD with demographic controls
print("\n   Model 2: DiD with Demographic Controls")
print("   " + "-"*60)

model2 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + AGE + age_sq + female + married + educ_hs',
                  data=df_main).fit()
print(f"   DiD Coefficient (eligible_x_post): {model2.params['eligible_x_post']:.4f}")
print(f"   Standard Error: {model2.bse['eligible_x_post']:.4f}")
print(f"   t-statistic: {model2.tvalues['eligible_x_post']:.4f}")
print(f"   p-value: {model2.pvalues['eligible_x_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['eligible_x_post', 0]:.4f}, {model2.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"   N: {int(model2.nobs):,}")
print(f"   R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with year fixed effects
print("\n   Model 3: DiD with Year Fixed Effects")
print("   " + "-"*60)

model3 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + AGE + age_sq + female + married + educ_hs',
                  data=df_main).fit()
print(f"   DiD Coefficient (eligible_x_post): {model3.params['eligible_x_post']:.4f}")
print(f"   Standard Error: {model3.bse['eligible_x_post']:.4f}")
print(f"   t-statistic: {model3.tvalues['eligible_x_post']:.4f}")
print(f"   p-value: {model3.pvalues['eligible_x_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['eligible_x_post', 0]:.4f}, {model3.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"   N: {int(model3.nobs):,}")
print(f"   R-squared: {model3.rsquared:.4f}")

# Model 4: DiD with state fixed effects
print("\n   Model 4: DiD with Year and State Fixed Effects")
print("   " + "-"*60)

model4 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs',
                  data=df_main).fit()
print(f"   DiD Coefficient (eligible_x_post): {model4.params['eligible_x_post']:.4f}")
print(f"   Standard Error: {model4.bse['eligible_x_post']:.4f}")
print(f"   t-statistic: {model4.tvalues['eligible_x_post']:.4f}")
print(f"   p-value: {model4.pvalues['eligible_x_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['eligible_x_post', 0]:.4f}, {model4.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"   N: {int(model4.nobs):,}")
print(f"   R-squared: {model4.rsquared:.4f}")

# -----------------------------------------------------------------
# STEP 8: WEIGHTED ANALYSIS
# -----------------------------------------------------------------
print("\n8. WEIGHTED REGRESSION ANALYSIS (using PERWT)...")
print("   " + "-"*60)

# Model 5: Preferred specification with weights using smf.wls
model5 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs',
                  data=df_main, weights=df_main['PERWT']).fit()

# Find the eligible_x_post coefficient
elig_coef = model5.params['eligible_x_post']
elig_se = model5.bse['eligible_x_post']
elig_t = model5.tvalues['eligible_x_post']
elig_p = model5.pvalues['eligible_x_post']
ci_low = elig_coef - 1.96 * elig_se
ci_high = elig_coef + 1.96 * elig_se

print(f"   DiD Coefficient (eligible_x_post): {elig_coef:.4f}")
print(f"   Standard Error: {elig_se:.4f}")
print(f"   t-statistic: {elig_t:.4f}")
print(f"   p-value: {elig_p:.4f}")
print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"   N (unweighted): {int(model5.nobs):,}")
print(f"   R-squared: {model5.rsquared:.4f}")

# -----------------------------------------------------------------
# STEP 9: ROBUSTNESS CHECKS
# -----------------------------------------------------------------
print("\n9. ROBUSTNESS CHECKS...")

# Create controls for df_analysis (includes 2012)
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['age_sq'] = df_analysis['AGE'] ** 2
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)
df_analysis['eligible_x_post'] = df_analysis['daca_eligible'] * df_analysis['post_incl_2012']

# Robustness 1: Include 2012 as post-period
print("\n   Robustness 1: Including 2012 as post-period")
print("   " + "-"*60)

model_r1 = smf.ols('fulltime ~ daca_eligible + post_incl_2012 + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs',
                    data=df_analysis).fit()
print(f"   DiD Coefficient: {model_r1.params['eligible_x_post']:.4f}")
print(f"   SE: {model_r1.bse['eligible_x_post']:.4f}, p-value: {model_r1.pvalues['eligible_x_post']:.4f}")

# Robustness 2: Different age restrictions (18-40)
print("\n   Robustness 2: Restricted age range (18-40)")
print("   " + "-"*60)
df_robust2 = df_main[(df_main['AGE'] >= 18) & (df_main['AGE'] <= 40)].copy()
model_r2 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs',
                    data=df_robust2).fit()
print(f"   DiD Coefficient: {model_r2.params['eligible_x_post']:.4f}")
print(f"   SE: {model_r2.bse['eligible_x_post']:.4f}, p-value: {model_r2.pvalues['eligible_x_post']:.4f}")
print(f"   N: {int(model_r2.nobs):,}")

# Robustness 3: Outcome = any employment
print("\n   Robustness 3: Outcome = Any Employment (not just full-time)")
print("   " + "-"*60)
model_r3 = smf.ols('employed ~ daca_eligible + post + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs',
                    data=df_main).fit()
print(f"   DiD Coefficient: {model_r3.params['eligible_x_post']:.4f}")
print(f"   SE: {model_r3.bse['eligible_x_post']:.4f}, p-value: {model_r3.pvalues['eligible_x_post']:.4f}")

# Robustness 4: By gender
print("\n   Robustness 4: By Gender")
print("   " + "-"*60)
df_male = df_main[df_main['female'] == 0]
df_female = df_main[df_main['female'] == 1]

model_male = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + married + educ_hs',
                      data=df_male).fit()
model_female = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + married + educ_hs',
                        data=df_female).fit()
print(f"   Males:   DiD = {model_male.params['eligible_x_post']:.4f} (SE: {model_male.bse['eligible_x_post']:.4f})")
print(f"   Females: DiD = {model_female.params['eligible_x_post']:.4f} (SE: {model_female.bse['eligible_x_post']:.4f})")

# -----------------------------------------------------------------
# STEP 10: EVENT STUDY / PRE-TRENDS CHECK
# -----------------------------------------------------------------
print("\n10. EVENT STUDY / PRE-TRENDS ANALYSIS...")
print("   " + "-"*60)

# Create year-specific treatment effects
years = sorted(df_main['YEAR'].unique())

# Create interaction dummies for each year with eligibility (base year: 2011)
for year in years:
    if year != 2011:  # Use 2011 as reference year
        df_main[f'elig_x_{year}'] = ((df_main['YEAR'] == year) & (df_main['daca_eligible'] == 1)).astype(int)

event_cols = [f'elig_x_{y}' for y in years if y != 2011]
formula = 'fulltime ~ daca_eligible + ' + ' + '.join(event_cols) + ' + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs'
model_event = smf.ols(formula, data=df_main).fit()

print("   Year-specific effects relative to 2011 (pre-trend test):")
for year in years:
    if year != 2011:
        coef = model_event.params[f'elig_x_{year}']
        se = model_event.bse[f'elig_x_{year}']
        pval = model_event.pvalues[f'elig_x_{year}']
        sig = '*' if pval < 0.05 else ''
        print(f"   {year}: {coef:8.4f} ({se:.4f}) {sig}")

# -----------------------------------------------------------------
# STEP 11: SAVE RESULTS FOR REPORT
# -----------------------------------------------------------------
print("\n11. SAVING RESULTS...")

# Create results dictionary
results = {
    'sample_size': len(df_main),
    'eligible_n': int(df_main['daca_eligible'].sum()),
    'ineligible_n': int(len(df_main) - df_main['daca_eligible'].sum()),
    'model1_coef': model1.params['eligible_x_post'],
    'model1_se': model1.bse['eligible_x_post'],
    'model1_pval': model1.pvalues['eligible_x_post'],
    'model2_coef': model2.params['eligible_x_post'],
    'model2_se': model2.bse['eligible_x_post'],
    'model2_pval': model2.pvalues['eligible_x_post'],
    'model3_coef': model3.params['eligible_x_post'],
    'model3_se': model3.bse['eligible_x_post'],
    'model3_pval': model3.pvalues['eligible_x_post'],
    'model4_coef': model4.params['eligible_x_post'],
    'model4_se': model4.bse['eligible_x_post'],
    'model4_pval': model4.pvalues['eligible_x_post'],
    'model5_coef': elig_coef,
    'model5_se': elig_se,
    'model5_pval': elig_p,
    'pre_elig': pre_elig,
    'post_elig': post_elig,
    'pre_inelig': pre_inelig,
    'post_inelig': post_inelig,
    'robust_r1_coef': model_r1.params['eligible_x_post'],
    'robust_r1_se': model_r1.bse['eligible_x_post'],
    'robust_r2_coef': model_r2.params['eligible_x_post'],
    'robust_r2_se': model_r2.bse['eligible_x_post'],
    'robust_r3_coef': model_r3.params['eligible_x_post'],
    'robust_r3_se': model_r3.bse['eligible_x_post'],
    'male_coef': model_male.params['eligible_x_post'],
    'male_se': model_male.bse['eligible_x_post'],
    'female_coef': model_female.params['eligible_x_post'],
    'female_se': model_female.bse['eligible_x_post'],
}

# Save to CSV for easy loading
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)

# -----------------------------------------------------------------
# STEP 12: CREATE SUMMARY TABLES FOR LATEX
# -----------------------------------------------------------------
print("\n12. CREATING SUMMARY TABLES...")

# Table 1: Sample descriptive statistics
desc_stats = df_main.groupby('daca_eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'PERWT': ['sum', 'count']
}).round(3)
desc_stats.columns = ['Mean Age', 'Female Share', 'Married Share', 'HS+ Share',
                       'Employment Rate', 'FT Employment Rate', 'Weighted N', 'Unweighted N']
desc_stats.to_csv('table1_descriptive.csv')
print("   Saved: table1_descriptive.csv")

# Table 2: DiD Summary by group and period
did_table = df_main.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'PERWT': 'sum'
}).round(4)
did_table.to_csv('table2_did_summary.csv')
print("   Saved: table2_did_summary.csv")

# Table 3: Regression results
reg_results = pd.DataFrame({
    'Model': ['Basic DiD', 'With Demographics', 'Year FE', 'Year + State FE', 'Weighted'],
    'Coefficient': [model1.params['eligible_x_post'], model2.params['eligible_x_post'],
                   model3.params['eligible_x_post'], model4.params['eligible_x_post'], elig_coef],
    'Std Error': [model1.bse['eligible_x_post'], model2.bse['eligible_x_post'],
                  model3.bse['eligible_x_post'], model4.bse['eligible_x_post'], elig_se],
    'P-value': [model1.pvalues['eligible_x_post'], model2.pvalues['eligible_x_post'],
                model3.pvalues['eligible_x_post'], model4.pvalues['eligible_x_post'], elig_p],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)],
    'R-squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared, model5.rsquared]
})
reg_results.to_csv('table3_regression.csv', index=False)
print("   Saved: table3_regression.csv")

# Event study coefficients
event_results = []
for year in years:
    if year != 2011:
        event_results.append({
            'Year': year,
            'Coefficient': model_event.params[f'elig_x_{year}'],
            'Std Error': model_event.bse[f'elig_x_{year}'],
            'P-value': model_event.pvalues[f'elig_x_{year}']
        })
    else:
        event_results.append({
            'Year': 2011,
            'Coefficient': 0,
            'Std Error': 0,
            'P-value': 1
        })
event_df = pd.DataFrame(event_results).sort_values('Year')
event_df.to_csv('table4_event_study.csv', index=False)
print("   Saved: table4_event_study.csv")

# Full regression output for Model 4
with open('model4_full_output.txt', 'w') as f:
    f.write(model4.summary().as_text())
print("   Saved: model4_full_output.txt")

# Additional descriptive statistics table
desc_by_period = df_main.groupby(['daca_eligible', 'post']).agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'UHRSWORK': 'mean',
    'PERWT': 'sum'
}).round(3)
desc_by_period.to_csv('table5_desc_by_period.csv')
print("   Saved: table5_desc_by_period.csv")

# -----------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"\nPreferred Estimate (Model 4: Year + State FE, with controls):")
print(f"   Effect of DACA eligibility on full-time employment: {model4.params['eligible_x_post']:.4f}")
print(f"   Standard Error: {model4.bse['eligible_x_post']:.4f}")
print(f"   95% Confidence Interval: [{model4.conf_int().loc['eligible_x_post', 0]:.4f}, {model4.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"   P-value: {model4.pvalues['eligible_x_post']:.4f}")
print(f"   Sample Size: {int(model4.nobs):,}")
print(f"\nInterpretation:")
effect_pct = model4.params['eligible_x_post'] * 100
print(f"   DACA eligibility is associated with a {effect_pct:.2f} percentage point")
if model4.params['eligible_x_post'] > 0:
    print(f"   increase in the probability of full-time employment.")
else:
    print(f"   decrease in the probability of full-time employment.")

if model4.pvalues['eligible_x_post'] < 0.05:
    print(f"   This effect is statistically significant at the 5% level.")
else:
    print(f"   This effect is NOT statistically significant at the 5% level.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
