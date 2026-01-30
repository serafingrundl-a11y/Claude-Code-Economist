"""
DACA Replication Study - Difference-in-Differences Analysis
Effect of DACA eligibility on full-time employment among Mexican-born Hispanic individuals

Research Design:
- Treatment group: Ages 26-30 as of June 15, 2012 (birth years 1982-1986)
- Control group: Ages 31-35 as of June 15, 2012 (birth years 1977-1981)
- Pre-treatment: 2006-2011 (excluding 2012 due to mid-year implementation)
- Post-treatment: 2013-2016
- Outcome: Full-time employment (UHRSWORK >= 35)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("=" * 80)
print("DACA REPLICATION STUDY - DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)
print()

# =============================================================================
# STEP 1: Load data in chunks with filtering
# =============================================================================
print("Loading and filtering data in chunks...")

# Columns we need
usecols = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
           'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
           'EDUC', 'EMPSTAT', 'UHRSWORK', 'STATEFIP']

# Read in chunks and filter
chunks = []
chunk_size = 500000
total_raw = 0

for chunk in pd.read_csv("data/data.csv", usecols=usecols, chunksize=chunk_size, low_memory=False):
    total_raw += len(chunk)

    # Apply basic filters immediately to reduce memory
    # Filter 1: Hispanic-Mexican (HISPAN == 1)
    filtered = chunk[chunk['HISPAN'] == 1].copy()

    # Filter 2: Born in Mexico (BPL == 200)
    filtered = filtered[filtered['BPL'] == 200]

    # Filter 3: Non-citizen (CITIZEN == 3)
    filtered = filtered[filtered['CITIZEN'] == 3]

    # Filter 4: Valid immigration year
    filtered = filtered[filtered['YRIMMIG'] > 0]

    if len(filtered) > 0:
        chunks.append(filtered)

    print(f"  Processed {total_raw:,} rows...", end='\r')

print(f"\nTotal observations in raw data: {total_raw:,}")

# Combine filtered chunks
sample = pd.concat(chunks, ignore_index=True)
print(f"After initial filters (Hispanic-Mexican, Mexico-born, Non-citizen): {len(sample):,}")

# =============================================================================
# STEP 2: Apply remaining DACA eligibility criteria
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: APPLY DACA ELIGIBILITY CRITERIA")
print("=" * 80)

# Filter: Arrived before age 16
sample['age_at_immigration'] = sample['YRIMMIG'] - sample['BIRTHYR']
sample = sample[sample['age_at_immigration'] < 16].copy()
print(f"After arrived before age 16 filter: {len(sample):,}")

# Filter: Continuous residence since 2007 (YRIMMIG <= 2007)
sample = sample[sample['YRIMMIG'] <= 2007].copy()
print(f"After residence since 2007 filter (YRIMMIG<=2007): {len(sample):,}")

# =============================================================================
# STEP 3: Define Treatment and Control Groups Based on Age
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: DEFINE TREATMENT AND CONTROL GROUPS")
print("=" * 80)

# Calculate age as of June 15, 2012
sample['age_2012'] = 2012 - sample['BIRTHYR']

# Treatment group: ages 26-30 in 2012 (DACA eligible)
# Control group: ages 31-35 in 2012 (too old for DACA, otherwise eligible)
sample['treat'] = np.where((sample['age_2012'] >= 26) & (sample['age_2012'] <= 30), 1,
                           np.where((sample['age_2012'] >= 31) & (sample['age_2012'] <= 35), 0, np.nan))

# Keep only treatment and control groups
sample = sample[sample['treat'].notna()].copy()
print(f"After age restriction (26-35 in 2012): {len(sample):,}")

# =============================================================================
# STEP 4: Define Time Periods
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: DEFINE TIME PERIODS")
print("=" * 80)

# Exclude 2012 (policy implemented mid-year)
sample = sample[sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(sample):,}")

# Create post-treatment indicator
sample['post'] = np.where(sample['YEAR'] >= 2013, 1, 0)

print(f"\nPre-treatment years (2006-2011): {sorted(sample[sample['post']==0]['YEAR'].unique())}")
print(f"Post-treatment years (2013-2016): {sorted(sample[sample['post']==1]['YEAR'].unique())}")

# =============================================================================
# STEP 5: Define Outcome Variable
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: DEFINE OUTCOME VARIABLE - FULL-TIME EMPLOYMENT")
print("=" * 80)

# Full-time employment: Usually working 35+ hours per week
sample['fulltime'] = np.where(sample['UHRSWORK'] >= 35, 1, 0)

print(f"Full-time employment rate in sample: {sample['fulltime'].mean():.3f}")
print(f"Mean usual hours worked (among those working): {sample[sample['UHRSWORK']>0]['UHRSWORK'].mean():.1f}")

# =============================================================================
# STEP 6: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: DESCRIPTIVE STATISTICS")
print("=" * 80)

# Create interaction term
sample['treat_post'] = sample['treat'] * sample['post']

# Summary statistics by group
print("\n--- Sample Sizes ---")
print(f"Treatment group (ages 26-30 in 2012): {int((sample['treat']==1).sum()):,}")
print(f"Control group (ages 31-35 in 2012): {int((sample['treat']==0).sum()):,}")
print(f"Pre-period observations: {int((sample['post']==0).sum()):,}")
print(f"Post-period observations: {int((sample['post']==1).sum()):,}")

# 2x2 Table of means
print("\n--- Full-Time Employment Rates (2x2 Table) ---")
table_22 = sample.groupby(['treat', 'post'])['fulltime'].agg(['mean', 'count']).round(4)
print(table_22)

# Calculate simple difference-in-differences
pre_treat = sample[(sample['treat']==1) & (sample['post']==0)]['fulltime'].mean()
post_treat = sample[(sample['treat']==1) & (sample['post']==1)]['fulltime'].mean()
pre_control = sample[(sample['treat']==0) & (sample['post']==0)]['fulltime'].mean()
post_control = sample[(sample['treat']==0) & (sample['post']==1)]['fulltime'].mean()

diff_treat = post_treat - pre_treat
diff_control = post_control - pre_control
did_simple = diff_treat - diff_control

print("\n--- Simple Difference-in-Differences Calculation ---")
print(f"Treatment group: Pre = {pre_treat:.4f}, Post = {post_treat:.4f}, Diff = {diff_treat:.4f}")
print(f"Control group: Pre = {pre_control:.4f}, Post = {post_control:.4f}, Diff = {diff_control:.4f}")
print(f"DiD estimate = {diff_treat:.4f} - ({diff_control:.4f}) = {did_simple:.4f}")

# Descriptive stats by treatment group
print("\n--- Demographic Characteristics by Group ---")
for treat_val, treat_name in [(1, "Treatment (26-30)"), (0, "Control (31-35)")]:
    subset = sample[sample['treat'] == treat_val]
    print(f"\n{treat_name}:")
    print(f"  N = {len(subset):,}")
    print(f"  Female (%) = {(subset['SEX']==2).mean()*100:.1f}")
    print(f"  Married (%) = {(subset['MARST']==1).mean()*100:.1f}")
    print(f"  Mean age in survey = {subset['AGE'].mean():.1f}")
    print(f"  Mean years in US = {subset['YRSUSA1'].mean():.1f}")
    print(f"  Full-time employed (%) = {subset['fulltime'].mean()*100:.1f}")

# Year-by-year trends
print("\n--- Full-Time Employment by Year and Group ---")
yearly = sample.groupby(['YEAR', 'treat'])['fulltime'].mean().unstack()
yearly.columns = ['Control (31-35)', 'Treatment (26-30)']
print(yearly.round(4))

# =============================================================================
# STEP 7: MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: REGRESSION ANALYSIS")
print("=" * 80)

# Create control variables
sample['female'] = (sample['SEX'] == 2).astype(int)
sample['married'] = (sample['MARST'] == 1).astype(int)
sample['highschool_plus'] = (sample['EDUC'] >= 6).astype(int)
sample['college'] = (sample['EDUC'] >= 10).astype(int)
sample['age'] = sample['AGE']
sample['age_sq'] = sample['AGE'] ** 2

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic Difference-in-Differences ---")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=sample).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with year fixed effects
print("\n--- Model 2: DiD with Year Fixed Effects ---")
model2 = smf.ols('fulltime ~ treat + C(YEAR) + treat_post', data=sample).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with demographic controls
print("\n--- Model 3: DiD with Demographic Controls ---")
model3 = smf.ols('fulltime ~ treat + C(YEAR) + treat_post + female + married + highschool_plus + college + age + age_sq',
                 data=sample).fit(cov_type='HC1')
print(model3.summary())

# Model 4: Full model with state fixed effects
print("\n--- Model 4: Full Model with State Fixed Effects ---")
model4 = smf.ols('fulltime ~ treat + C(YEAR) + treat_post + female + married + highschool_plus + college + age + age_sq + C(STATEFIP)',
                 data=sample).fit(cov_type='HC1')

# Print only key coefficients
print("\nKey Coefficients from Full Model (Model 4):")
print(f"Intercept: {model4.params['Intercept']:.4f} (SE: {model4.bse['Intercept']:.4f})")
print(f"treat: {model4.params['treat']:.4f} (SE: {model4.bse['treat']:.4f})")
print(f"treat_post (DiD): {model4.params['treat_post']:.4f} (SE: {model4.bse['treat_post']:.4f})")
print(f"female: {model4.params['female']:.4f} (SE: {model4.bse['female']:.4f})")
print(f"married: {model4.params['married']:.4f} (SE: {model4.bse['married']:.4f})")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {int(model4.nobs):,}")

# =============================================================================
# STEP 8: WEIGHTED ANALYSIS (using survey weights)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: WEIGHTED REGRESSION ANALYSIS")
print("=" * 80)

# Model 5: Weighted DiD with controls
print("\n--- Model 5: Weighted DiD with Full Controls ---")

# Prepare data for weighted regression - ensure numeric columns only
X_vars = sample[['treat', 'treat_post', 'female', 'married', 'highschool_plus', 'college', 'age', 'age_sq']].astype(float)

# Add year dummies
year_dummies = pd.get_dummies(sample['YEAR'], prefix='year', drop_first=True).astype(float)
# Add state dummies
state_dummies = pd.get_dummies(sample['STATEFIP'], prefix='state', drop_first=True).astype(float)

X = pd.concat([X_vars, year_dummies, state_dummies], axis=1)
X = sm.add_constant(X)
y = sample['fulltime'].astype(float)
weights = sample['PERWT'].astype(float)

model5 = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')

print("\nKey Coefficients from Weighted Model (Model 5):")
print(f"treat: {model5.params['treat']:.4f} (SE: {model5.bse['treat']:.4f})")
print(f"treat_post (DiD): {model5.params['treat_post']:.4f} (SE: {model5.bse['treat_post']:.4f})")
print(f"female: {model5.params['female']:.4f} (SE: {model5.bse['female']:.4f})")
print(f"married: {model5.params['married']:.4f} (SE: {model5.bse['married']:.4f})")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {int(model5.nobs):,}")

# Calculate 95% CI for preferred estimate
did_coef = model5.params['treat_post']
did_se = model5.bse['treat_post']
ci_low = did_coef - 1.96 * did_se
ci_high = did_coef + 1.96 * did_se

print(f"\n*** PREFERRED ESTIMATE (Weighted DiD with controls): ***")
print(f"Effect of DACA eligibility on full-time employment: {did_coef:.4f}")
print(f"Standard Error: {did_se:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"p-value: {model5.pvalues['treat_post']:.4f}")

# =============================================================================
# STEP 9: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 9: ROBUSTNESS CHECKS")
print("=" * 80)

# Robustness 1: Placebo test with pre-treatment data only (using 2009 as fake treatment)
print("\n--- Robustness Check 1: Placebo Test (2009 as fake treatment) ---")
sample_placebo = sample[sample['post'] == 0].copy()  # Pre-treatment only
sample_placebo['fake_post'] = np.where(sample_placebo['YEAR'] >= 2009, 1, 0)
sample_placebo['fake_treat_post'] = sample_placebo['treat'] * sample_placebo['fake_post']

model_placebo = smf.ols('fulltime ~ treat + fake_post + fake_treat_post', data=sample_placebo).fit(cov_type='HC1')
print(f"Placebo DiD Coefficient: {model_placebo.params['fake_treat_post']:.4f} (SE: {model_placebo.bse['fake_treat_post']:.4f})")
print(f"p-value: {model_placebo.pvalues['fake_treat_post']:.4f}")

# Robustness 2: Employment (any) instead of full-time
print("\n--- Robustness Check 2: Any Employment as Outcome ---")
sample['employed'] = np.where(sample['EMPSTAT'] == 1, 1, 0)
model_emp = smf.ols('employed ~ treat + post + treat_post', data=sample).fit(cov_type='HC1')
print(f"DiD Coefficient (any employment): {model_emp.params['treat_post']:.4f} (SE: {model_emp.bse['treat_post']:.4f})")
print(f"p-value: {model_emp.pvalues['treat_post']:.4f}")

# Robustness 3: By gender
print("\n--- Robustness Check 3: By Gender ---")
for sex_val, sex_name in [(1, "Male"), (2, "Female")]:
    sample_sex = sample[sample['SEX'] == sex_val]
    model_sex = smf.ols('fulltime ~ treat + post + treat_post', data=sample_sex).fit(cov_type='HC1')
    print(f"{sex_name}: DiD = {model_sex.params['treat_post']:.4f} (SE: {model_sex.bse['treat_post']:.4f}), p = {model_sex.pvalues['treat_post']:.4f}, N = {int(model_sex.nobs):,}")

# Robustness 4: Different control group (32-36 instead of 31-35)
print("\n--- Robustness Check 4: Alternative Control Group (32-36) ---")
sample_alt = sample.copy()
sample_alt['treat_alt'] = np.where((sample_alt['age_2012'] >= 26) & (sample_alt['age_2012'] <= 30), 1,
                                    np.where((sample_alt['age_2012'] >= 32) & (sample_alt['age_2012'] <= 36), 0, np.nan))
sample_alt = sample_alt[sample_alt['treat_alt'].notna()].copy()
sample_alt['treat_post_alt'] = sample_alt['treat_alt'] * sample_alt['post']
model_alt = smf.ols('fulltime ~ treat_alt + post + treat_post_alt', data=sample_alt).fit(cov_type='HC1')
print(f"DiD Coefficient (alt control): {model_alt.params['treat_post_alt']:.4f} (SE: {model_alt.bse['treat_post_alt']:.4f})")
print(f"p-value: {model_alt.pvalues['treat_post_alt']:.4f}")
print(f"N: {int(model_alt.nobs):,}")

# =============================================================================
# STEP 10: EVENT STUDY / DYNAMIC EFFECTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 10: EVENT STUDY ANALYSIS")
print("=" * 80)

# Create year-specific treatment effects (relative to 2011)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    sample[f'year_{yr}'] = np.where(sample['YEAR'] == yr, 1, 0)
    sample[f'treat_x_{yr}'] = sample['treat'] * sample[f'year_{yr}']

event_formula = 'fulltime ~ treat + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016'
model_event = smf.ols(event_formula, data=sample).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 60)
print(f"{'Year':<8} {'Coef':>10} {'SE':>10} {'p-value':>10} {'Sig':>5}")
print("-" * 60)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treat_x_{yr}']
    se = model_event.bse[f'treat_x_{yr}']
    pval = model_event.pvalues[f'treat_x_{yr}']
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
    marker = " <-- DACA" if yr == 2013 else ""
    print(f"{yr:<8} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {sig:>5}{marker}")

# =============================================================================
# STEP 11: SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 11: SUMMARY OF RESULTS")
print("=" * 80)

print("""
RESEARCH QUESTION:
Among ethnically Hispanic-Mexican, Mexican-born people living in the US, what was
the causal impact of DACA eligibility on full-time employment (35+ hours/week)?

IDENTIFICATION STRATEGY:
- Difference-in-differences comparing ages 26-30 (treated) vs 31-35 (control)
- Pre-period: 2006-2011, Post-period: 2013-2016
- Both groups meet all DACA criteria except age cutoff

SAMPLE SELECTION:
- Hispanic-Mexican ethnicity (HISPAN == 1)
- Born in Mexico (BPL == 200)
- Non-citizen (CITIZEN == 3, proxy for undocumented)
- Arrived before age 16 (DACA requirement)
- Continuous residence since 2007 (YRIMMIG <= 2007)

KEY FINDINGS:
""")

print(f"Sample size: {len(sample):,}")
print(f"Treatment group (ages 26-30): {int((sample['treat']==1).sum()):,}")
print(f"Control group (ages 31-35): {int((sample['treat']==0).sum()):,}")
print()
print("Full-time employment rates:")
print(f"  Treatment pre:  {pre_treat:.4f} ({pre_treat*100:.1f}%)")
print(f"  Treatment post: {post_treat:.4f} ({post_treat*100:.1f}%)")
print(f"  Control pre:    {pre_control:.4f} ({pre_control*100:.1f}%)")
print(f"  Control post:   {post_control:.4f} ({post_control*100:.1f}%)")
print()
print("PREFERRED ESTIMATE (Model 5 - Weighted with controls):")
print(f"  DiD coefficient: {did_coef:.4f} ({did_coef*100:.2f} percentage points)")
print(f"  Standard error:  {did_se:.4f}")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  t-statistic: {did_coef/did_se:.3f}")
print(f"  p-value: {model5.pvalues['treat_post']:.4f}")

interpretation = "positive" if did_coef > 0 else "negative"
significant = "statistically significant" if model5.pvalues['treat_post'] < 0.05 else "not statistically significant"

print(f"""
INTERPRETATION:
The estimated effect of DACA eligibility on full-time employment is {interpretation}
({did_coef:.4f}, or {abs(did_coef)*100:.2f} percentage points). This estimate is {significant}
at the 5% level (p = {model5.pvalues['treat_post']:.4f}).

LIMITATIONS:
1. We cannot directly observe undocumented status; non-citizenship is a proxy
2. ACS is cross-sectional, not panel data (different individuals each wave)
3. Potential spillover effects to control group (e.g., increased competition)
4. Self-selection into DACA application not captured
""")

# =============================================================================
# SAVE RESULTS FOR REPORT
# =============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save yearly means for plotting
yearly_means = sample.groupby(['YEAR', 'treat'])['fulltime'].mean().unstack()
yearly_means.columns = ['control', 'treatment']
yearly_means.to_csv('yearly_means.csv')

# Save model summaries
with open('model_results.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DACA REPLICATION STUDY - REGRESSION RESULTS\n")
    f.write("=" * 80 + "\n\n")

    f.write("MODEL 1: Basic DiD\n")
    f.write("-" * 40 + "\n")
    f.write(model1.summary().as_text())

    f.write("\n\n" + "=" * 80 + "\n")
    f.write("MODEL 2: DiD with Year FE\n")
    f.write("-" * 40 + "\n")
    f.write(model2.summary().as_text())

    f.write("\n\n" + "=" * 80 + "\n")
    f.write("MODEL 3: DiD with Controls\n")
    f.write("-" * 40 + "\n")
    f.write(model3.summary().as_text())

    f.write("\n\n" + "=" * 80 + "\n")
    f.write("MODEL 5: Weighted DiD with Controls (PREFERRED)\n")
    f.write("-" * 40 + "\n")
    f.write(f"DiD Coefficient: {did_coef:.4f}\n")
    f.write(f"Standard Error: {did_se:.4f}\n")
    f.write(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n")
    f.write(f"p-value: {model5.pvalues['treat_post']:.4f}\n")
    f.write(f"N: {int(model5.nobs):,}\n")
    f.write(f"R-squared: {model5.rsquared:.4f}\n")

# Save summary statistics table
summary_stats = pd.DataFrame({
    'Statistic': ['N', 'N Treatment', 'N Control',
                  'Pre-Treatment Rate (Treat)', 'Post-Treatment Rate (Treat)',
                  'Pre-Treatment Rate (Control)', 'Post-Treatment Rate (Control)',
                  'Simple DiD', 'Regression DiD (Basic)', 'Regression DiD (Weighted+Controls)',
                  'Standard Error', '95% CI Lower', '95% CI Upper', 'p-value'],
    'Value': [len(sample), int((sample['treat']==1).sum()), int((sample['treat']==0).sum()),
              pre_treat, post_treat, pre_control, post_control,
              did_simple, model1.params['treat_post'], did_coef,
              did_se, ci_low, ci_high, model5.pvalues['treat_post']]
})
summary_stats.to_csv('summary_statistics.csv', index=False)

print("Results saved to:")
print("  - yearly_means.csv")
print("  - model_results.txt")
print("  - summary_statistics.csv")
print("\nAnalysis complete!")
