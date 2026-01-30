"""
DACA Impact on Full-Time Employment Analysis
Replication Study

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in
the United States, what was the causal impact of eligibility for the Deferred Action
for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible
person is employed full-time (outcome), defined as usually working 35 hours per week or more?

Analysis Period: 2006-2016 (pre-treatment: 2006-2011, post-treatment: 2013-2016)
Note: 2012 is excluded as the policy was implemented mid-year (June 15, 2012)

DACA Eligibility Criteria:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

For identification, we compare:
- Treatment group: Hispanic-Mexican, born in Mexico, non-citizen, arrived before age 16,
  born after June 15, 1981 (age < 31 on June 15, 2012)
- Control group: Hispanic-Mexican, born in Mexico, non-citizen, arrived at age 16 or older
  OR born before June 15, 1981 (age >= 31 on June 15, 2012)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 1\replication_37")

print("="*80)
print("DACA IMPACT ON FULL-TIME EMPLOYMENT - REPLICATION ANALYSIS")
print("="*80)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv("data/data.csv")
print(f"Total observations in raw data: {len(df):,}")
print(f"Years covered: {df['YEAR'].min()} to {df['YEAR'].max()}")

# Display column names
print("\nVariables available:")
print(df.columns.tolist())

# Basic data exploration
print("\n[2] Data exploration...")
print(f"\nYear distribution:")
print(df['YEAR'].value_counts().sort_index())

# Key variables for analysis
# HISPAN: Hispanic origin (1 = Mexican)
# BPL: Birthplace (200 = Mexico)
# CITIZEN: Citizenship status (3 = Not a citizen)
# YRIMMIG: Year of immigration
# BIRTHYR: Birth year
# BIRTHQTR: Birth quarter
# UHRSWORK: Usual hours worked per week
# AGE: Age

print("\n[3] Filtering to target population: Hispanic-Mexican, born in Mexico...")

# Filter to Hispanic-Mexican ethnicity and born in Mexico
# HISPAN == 1 (Mexican) and BPL == 200 (Mexico)
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
print(f"Observations after filtering to Hispanic-Mexican born in Mexico: {len(df_mex):,}")

print("\nCitizenship status distribution in this population:")
print(df_mex['CITIZEN'].value_counts())

# Filter to non-citizens only (CITIZEN == 3)
# Per instructions: "Assume that anyone who is not a citizen and who has
# not received immigration papers is undocumented for DACA purposes"
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"\nObservations after filtering to non-citizens: {len(df_mex):,}")

print("\n[4] Creating DACA eligibility indicators...")

# Calculate age at arrival in the US
# We need year of immigration and birth year
# Age at immigration = YRIMMIG - BIRTHYR (approximately)

# First, let's look at YRIMMIG distribution
print("\nYear of immigration distribution (sample):")
print(df_mex['YRIMMIG'].value_counts().head(20))

# Check for missing/invalid YRIMMIG values
print(f"\nYRIMMIG == 0 (N/A): {(df_mex['YRIMMIG'] == 0).sum():,}")

# Filter to those with valid immigration year
df_mex = df_mex[df_mex['YRIMMIG'] > 0].copy()
print(f"Observations after filtering to valid immigration year: {len(df_mex):,}")

# Calculate age at immigration
df_mex['age_at_immig'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
print("\nAge at immigration distribution:")
print(df_mex['age_at_immig'].describe())

# DACA eligibility criterion 1: Arrived before 16th birthday
df_mex['arrived_before_16'] = (df_mex['age_at_immig'] < 16).astype(int)
print(f"\nArrived before age 16: {df_mex['arrived_before_16'].sum():,} ({df_mex['arrived_before_16'].mean()*100:.1f}%)")

# DACA eligibility criterion 2: Born after June 15, 1981 (not yet 31 as of June 15, 2012)
# Being conservative: born in 1982 or later to be safe
# More precisely: those born after June 15, 1981
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Born after June 15, 1981 means:
# - BIRTHYR >= 1982, OR
# - BIRTHYR == 1981 and BIRTHQTR >= 3 (July or later)
df_mex['born_after_june_1981'] = (
    (df_mex['BIRTHYR'] >= 1982) |
    ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))
).astype(int)
print(f"Born after June 15, 1981: {df_mex['born_after_june_1981'].sum():,} ({df_mex['born_after_june_1981'].mean()*100:.1f}%)")

# DACA eligibility criterion 3: Lived continuously in US since June 15, 2007
# We approximate this by: arrived by 2007 or earlier
df_mex['in_us_by_2007'] = (df_mex['YRIMMIG'] <= 2007).astype(int)
print(f"In US by 2007: {df_mex['in_us_by_2007'].sum():,} ({df_mex['in_us_by_2007'].mean()*100:.1f}%)")

# Combined DACA eligibility indicator
# Eligible if: arrived before 16 AND born after June 1981 AND in US by 2007
df_mex['daca_eligible'] = (
    (df_mex['arrived_before_16'] == 1) &
    (df_mex['born_after_june_1981'] == 1) &
    (df_mex['in_us_by_2007'] == 1)
).astype(int)
print(f"\nDACA eligible: {df_mex['daca_eligible'].sum():,} ({df_mex['daca_eligible'].mean()*100:.1f}%)")

print("\n[5] Creating outcome variable: Full-time employment (35+ hours/week)...")

# UHRSWORK: Usual hours worked per week
# Full-time = 35 or more hours per week
print("\nUsual hours worked distribution:")
print(df_mex['UHRSWORK'].describe())

# Full-time employment indicator
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)
print(f"\nFull-time employed: {df_mex['fulltime'].sum():,} ({df_mex['fulltime'].mean()*100:.1f}%)")

print("\n[6] Creating treatment period indicator...")

# DACA was implemented June 15, 2012
# Treatment period: 2013-2016 (excluding 2012 as transition year)
# Control period: 2006-2011
df_mex['post_daca'] = (df_mex['YEAR'] >= 2013).astype(int)
print(f"\nPost-DACA (2013-2016): {df_mex['post_daca'].sum():,} ({df_mex['post_daca'].mean()*100:.1f}%)")

# Exclude 2012 (transition year)
df_analysis = df_mex[df_mex['YEAR'] != 2012].copy()
print(f"\nObservations after excluding 2012: {len(df_analysis):,}")

print("\n[7] Restricting to working-age population (18-64)...")
df_analysis = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 64)].copy()
print(f"Observations after restricting to ages 18-64: {len(df_analysis):,}")

print("\n[8] Summary statistics by treatment group and period...")

# Create summary table
summary = df_analysis.groupby(['daca_eligible', 'post_daca']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'UHRSWORK': 'mean',
    'PERWT': 'sum'
}).round(4)
print("\nSummary by DACA eligibility and period:")
print(summary)

# Calculate DiD estimate manually first
print("\n[9] Difference-in-Differences estimate (unweighted)...")

# Mean full-time employment by group and period
means = df_analysis.groupby(['daca_eligible', 'post_daca'])['fulltime'].mean()
print("\nMean full-time employment rates:")
print(means)

# DiD calculation
# Eligible: Post - Pre
eligible_diff = means[(1, 1)] - means[(1, 0)]
# Ineligible: Post - Pre
ineligible_diff = means[(0, 1)] - means[(0, 0)]
# DiD = Eligible_diff - Ineligible_diff
did_estimate = eligible_diff - ineligible_diff

print(f"\nEligible group change (post - pre): {eligible_diff:.4f}")
print(f"Ineligible group change (post - pre): {ineligible_diff:.4f}")
print(f"Difference-in-Differences estimate: {did_estimate:.4f}")

print("\n[10] Regression analysis...")

# Create interaction term
df_analysis['eligible_x_post'] = df_analysis['daca_eligible'] * df_analysis['post_daca']

# Simple DiD regression without controls
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.ols('fulltime ~ daca_eligible + post_daca + eligible_x_post', data=df_analysis).fit()
print(model1.summary())

# DiD regression with controls
print("\n--- Model 2: DiD with controls (age, sex, education, marital status) ---")
# Add control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Education categories
# EDUC: 0=N/A, 1=No school, 2=Grade 1-4, 3=Grade 5-8, 4=Grade 9, 5=Grade 10,
# 6=Grade 11, 7=Grade 12, 8=1 year college, 9=2 years college, 10=3 years college, 11=4+ years college
df_analysis['less_than_hs'] = (df_analysis['EDUC'] <= 6).astype(int)
df_analysis['hs_diploma'] = (df_analysis['EDUC'] == 7).astype(int)
df_analysis['some_college'] = (df_analysis['EDUC'].isin([8, 9, 10])).astype(int)
df_analysis['college_plus'] = (df_analysis['EDUC'] >= 11).astype(int)

model2 = smf.ols('fulltime ~ daca_eligible + post_daca + eligible_x_post + AGE + age_sq + female + married + hs_diploma + some_college + college_plus',
                 data=df_analysis).fit()
print(model2.summary())

# Model 3: With year fixed effects
print("\n--- Model 3: DiD with year fixed effects ---")
model3 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + hs_diploma + some_college + college_plus + C(YEAR)',
                 data=df_analysis).fit()
print(model3.summary())

# Model 4: With state fixed effects
print("\n--- Model 4: DiD with year and state fixed effects ---")
model4 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + hs_diploma + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                 data=df_analysis).fit()
print(model4.summary())

print("\n[11] Weighted analysis using person weights (PERWT)...")

# Weighted DiD regression
print("\n--- Model 5: Weighted DiD with controls and fixed effects ---")
model5 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + hs_diploma + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                 data=df_analysis, weights=df_analysis['PERWT']).fit()
print(model5.summary())

# Robust standard errors
print("\n--- Model 6: Weighted DiD with robust standard errors ---")
model6 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + hs_diploma + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                 data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model6.summary())

print("\n[12] Extracting key results...")

# Extract coefficient and standard error for eligible_x_post from preferred model
coef = model6.params['eligible_x_post']
se = model6.bse['eligible_x_post']
ci_low = model6.conf_int().loc['eligible_x_post', 0]
ci_high = model6.conf_int().loc['eligible_x_post', 1]
pval = model6.pvalues['eligible_x_post']

print(f"\n" + "="*80)
print("PREFERRED ESTIMATE (Model 6: Weighted DiD with controls, FEs, robust SEs)")
print("="*80)
print(f"DiD Coefficient (eligible_x_post): {coef:.6f}")
print(f"Standard Error: {se:.6f}")
print(f"95% Confidence Interval: [{ci_low:.6f}, {ci_high:.6f}]")
print(f"P-value: {pval:.6f}")
print(f"Sample size: {len(df_analysis):,}")
print("="*80)

print("\n[13] Creating visualizations...")

# Figure 1: Trends in full-time employment by eligibility group
fig1, ax1 = plt.subplots(figsize=(10, 6))
yearly_means = df_analysis.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
yearly_means.columns = ['Ineligible', 'Eligible']
yearly_means.plot(ax=ax1, marker='o', linewidth=2)
ax1.axvline(x=2012.5, color='red', linestyle='--', label='DACA Implementation')
ax1.set_xlabel('Year')
ax1.set_ylabel('Full-Time Employment Rate')
ax1.set_title('Full-Time Employment Trends by DACA Eligibility Status')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=150)
plt.close()
print("Saved: figure1_trends.png")

# Figure 2: Weighted trends
fig2, ax2 = plt.subplots(figsize=(10, 6))
weighted_means = df_analysis.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
weighted_means.columns = ['Ineligible', 'Eligible']
weighted_means.plot(ax=ax2, marker='o', linewidth=2)
ax2.axvline(x=2012.5, color='red', linestyle='--', label='DACA Implementation')
ax2.set_xlabel('Year')
ax2.set_ylabel('Full-Time Employment Rate (Weighted)')
ax2.set_title('Weighted Full-Time Employment Trends by DACA Eligibility Status')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_weighted_trends.png', dpi=150)
plt.close()
print("Saved: figure2_weighted_trends.png")

# Figure 3: Age distribution by eligibility
fig3, ax3 = plt.subplots(figsize=(10, 6))
df_analysis[df_analysis['daca_eligible']==1]['AGE'].hist(ax=ax3, alpha=0.5, label='Eligible', bins=30)
df_analysis[df_analysis['daca_eligible']==0]['AGE'].hist(ax=ax3, alpha=0.5, label='Ineligible', bins=30)
ax3.set_xlabel('Age')
ax3.set_ylabel('Frequency')
ax3.set_title('Age Distribution by DACA Eligibility Status')
ax3.legend()
plt.tight_layout()
plt.savefig('figure3_age_distribution.png', dpi=150)
plt.close()
print("Saved: figure3_age_distribution.png")

print("\n[14] Robustness checks...")

# Robustness 1: Alternative age restriction (16-55)
print("\n--- Robustness 1: Ages 16-55 ---")
df_robust1 = df_mex[(df_mex['YEAR'] != 2012) & (df_mex['AGE'] >= 16) & (df_mex['AGE'] <= 55)].copy()
df_robust1['eligible_x_post'] = df_robust1['daca_eligible'] * df_robust1['post_daca']
df_robust1['female'] = (df_robust1['SEX'] == 2).astype(int)
df_robust1['married'] = (df_robust1['MARST'] == 1).astype(int)
df_robust1['age_sq'] = df_robust1['AGE'] ** 2
df_robust1['hs_diploma'] = (df_robust1['EDUC'] == 7).astype(int)
df_robust1['some_college'] = (df_robust1['EDUC'].isin([8, 9, 10])).astype(int)
df_robust1['college_plus'] = (df_robust1['EDUC'] >= 11).astype(int)

model_r1 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + hs_diploma + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                   data=df_robust1, weights=df_robust1['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (ages 16-55): {model_r1.params['eligible_x_post']:.6f} (SE: {model_r1.bse['eligible_x_post']:.6f})")

# Robustness 2: Including 2012
print("\n--- Robustness 2: Including 2012 ---")
df_robust2 = df_mex[(df_mex['AGE'] >= 18) & (df_mex['AGE'] <= 64)].copy()
df_robust2['post_daca'] = (df_robust2['YEAR'] >= 2012).astype(int)  # Including 2012 as post
df_robust2['eligible_x_post'] = df_robust2['daca_eligible'] * df_robust2['post_daca']
df_robust2['female'] = (df_robust2['SEX'] == 2).astype(int)
df_robust2['married'] = (df_robust2['MARST'] == 1).astype(int)
df_robust2['age_sq'] = df_robust2['AGE'] ** 2
df_robust2['hs_diploma'] = (df_robust2['EDUC'] == 7).astype(int)
df_robust2['some_college'] = (df_robust2['EDUC'].isin([8, 9, 10])).astype(int)
df_robust2['college_plus'] = (df_robust2['EDUC'] >= 11).astype(int)

model_r2 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + hs_diploma + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                   data=df_robust2, weights=df_robust2['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (including 2012): {model_r2.params['eligible_x_post']:.6f} (SE: {model_r2.bse['eligible_x_post']:.6f})")

# Robustness 3: Men only
print("\n--- Robustness 3: Men only ---")
df_robust3 = df_analysis[df_analysis['SEX'] == 1].copy()
model_r3 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + married + hs_diploma + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                   data=df_robust3, weights=df_robust3['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (men only): {model_r3.params['eligible_x_post']:.6f} (SE: {model_r3.bse['eligible_x_post']:.6f})")

# Robustness 4: Women only
print("\n--- Robustness 4: Women only ---")
df_robust4 = df_analysis[df_analysis['SEX'] == 2].copy()
model_r4 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + married + hs_diploma + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                   data=df_robust4, weights=df_robust4['PERWT']).fit(cov_type='HC1')
print(f"DiD estimate (women only): {model_r4.params['eligible_x_post']:.6f} (SE: {model_r4.bse['eligible_x_post']:.6f})")

print("\n[15] Placebo test: Pre-treatment period (2006-2011, fake treatment at 2009)...")
df_placebo = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_placebo['fake_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['eligible_x_fake_post'] = df_placebo['daca_eligible'] * df_placebo['fake_post']

model_placebo = smf.wls('fulltime ~ daca_eligible + eligible_x_fake_post + AGE + age_sq + female + married + hs_diploma + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                        data=df_placebo, weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD estimate (fake 2009 treatment): {model_placebo.params['eligible_x_fake_post']:.6f} (SE: {model_placebo.bse['eligible_x_fake_post']:.6f}, p={model_placebo.pvalues['eligible_x_fake_post']:.4f})")

print("\n[16] Summary of all results...")
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"\nSample: Hispanic-Mexican, born in Mexico, non-citizens, ages 18-64")
print(f"Treatment: DACA eligibility (arrived before age 16, born after June 1981, in US by 2007)")
print(f"Outcome: Full-time employment (35+ hours/week)")
print(f"Pre-period: 2006-2011")
print(f"Post-period: 2013-2016")
print(f"\nTotal sample size: {len(df_analysis):,}")
print(f"DACA-eligible observations: {df_analysis['daca_eligible'].sum():,}")
print(f"DACA-ineligible observations: {(df_analysis['daca_eligible']==0).sum():,}")

print("\n" + "-"*80)
print("Model Estimates:")
print("-"*80)
print(f"{'Model':<50} {'Coefficient':>12} {'SE':>12} {'p-value':>12}")
print("-"*80)
print(f"{'Model 1: Basic DiD':<50} {model1.params['eligible_x_post']:>12.4f} {model1.bse['eligible_x_post']:>12.4f} {model1.pvalues['eligible_x_post']:>12.4f}")
print(f"{'Model 2: DiD + controls':<50} {model2.params['eligible_x_post']:>12.4f} {model2.bse['eligible_x_post']:>12.4f} {model2.pvalues['eligible_x_post']:>12.4f}")
print(f"{'Model 3: DiD + controls + year FE':<50} {model3.params['eligible_x_post']:>12.4f} {model3.bse['eligible_x_post']:>12.4f} {model3.pvalues['eligible_x_post']:>12.4f}")
print(f"{'Model 4: DiD + controls + year + state FE':<50} {model4.params['eligible_x_post']:>12.4f} {model4.bse['eligible_x_post']:>12.4f} {model4.pvalues['eligible_x_post']:>12.4f}")
print(f"{'Model 5: Weighted + controls + FE':<50} {model5.params['eligible_x_post']:>12.4f} {model5.bse['eligible_x_post']:>12.4f} {model5.pvalues['eligible_x_post']:>12.4f}")
print(f"{'Model 6: Weighted + controls + FE + robust SE *':<50} {model6.params['eligible_x_post']:>12.4f} {model6.bse['eligible_x_post']:>12.4f} {model6.pvalues['eligible_x_post']:>12.4f}")
print("-"*80)
print("* Preferred specification")

print("\n" + "-"*80)
print("Robustness Checks:")
print("-"*80)
print(f"{'Check':<50} {'Coefficient':>12} {'SE':>12} {'p-value':>12}")
print("-"*80)
print(f"{'Ages 16-55':<50} {model_r1.params['eligible_x_post']:>12.4f} {model_r1.bse['eligible_x_post']:>12.4f} {model_r1.pvalues['eligible_x_post']:>12.4f}")
print(f"{'Including 2012':<50} {model_r2.params['eligible_x_post']:>12.4f} {model_r2.bse['eligible_x_post']:>12.4f} {model_r2.pvalues['eligible_x_post']:>12.4f}")
print(f"{'Men only':<50} {model_r3.params['eligible_x_post']:>12.4f} {model_r3.bse['eligible_x_post']:>12.4f} {model_r3.pvalues['eligible_x_post']:>12.4f}")
print(f"{'Women only':<50} {model_r4.params['eligible_x_post']:>12.4f} {model_r4.bse['eligible_x_post']:>12.4f} {model_r4.pvalues['eligible_x_post']:>12.4f}")
print(f"{'Placebo (fake 2009 treatment)':<50} {model_placebo.params['eligible_x_fake_post']:>12.4f} {model_placebo.bse['eligible_x_fake_post']:>12.4f} {model_placebo.pvalues['eligible_x_fake_post']:>12.4f}")
print("-"*80)

# Save key results to file
results_dict = {
    'preferred_coefficient': coef,
    'preferred_se': se,
    'preferred_ci_low': ci_low,
    'preferred_ci_high': ci_high,
    'preferred_pvalue': pval,
    'sample_size': len(df_analysis),
    'n_eligible': int(df_analysis['daca_eligible'].sum()),
    'n_ineligible': int((df_analysis['daca_eligible']==0).sum())
}

# Create summary statistics table
summary_stats = df_analysis.groupby('daca_eligible').agg({
    'AGE': ['mean', 'std'],
    'female': 'mean',
    'married': 'mean',
    'fulltime': 'mean',
    'UHRSWORK': ['mean', 'std'],
    'hs_diploma': 'mean',
    'some_college': 'mean',
    'college_plus': 'mean'
}).round(4)
summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
print("\n" + "-"*80)
print("Summary Statistics by DACA Eligibility:")
print("-"*80)
print(summary_stats.T)

# Save summary statistics
summary_stats.to_csv('summary_statistics.csv')
print("\nSaved: summary_statistics.csv")

# Save yearly trends for plotting
yearly_trends = df_analysis.groupby(['YEAR', 'daca_eligible']).agg({
    'fulltime': 'mean',
    'PERWT': 'sum'
}).reset_index()
yearly_trends.to_csv('yearly_trends.csv', index=False)
print("Saved: yearly_trends.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
