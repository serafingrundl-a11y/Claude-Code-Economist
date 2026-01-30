"""
DACA Impact on Full-Time Employment: Independent Replication Analysis
======================================================================

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for DACA on the
probability that the eligible person is employed full-time (35+ hours/week)?

Author: Replication 60
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

print("="*80)
print("DACA IMPACT ON FULL-TIME EMPLOYMENT: REPLICATION ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: Load the data
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load main ACS data
print("Loading ACS data from data.csv...")
df = pd.read_csv('data/data.csv')

print(f"Total observations loaded: {len(df):,}")
print(f"Variables: {df.columns.tolist()}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# STEP 2: Initial Data Exploration
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATA EXPLORATION")
print("="*80)

print("\nSample sizes by year:")
print(df.groupby('YEAR').size())

print("\nHispanic origin distribution (HISPAN):")
print(df['HISPAN'].value_counts().sort_index())

print("\nBirthplace distribution for foreign-born (top 10):")
print(df[df['BPL'] >= 100]['BPL'].value_counts().head(10))

print("\nCitizenship status distribution (CITIZEN):")
print(df['CITIZEN'].value_counts().sort_index())

# ============================================================================
# STEP 3: Construct Analysis Sample
# ============================================================================
print("\n" + "="*80)
print("STEP 3: CONSTRUCTING ANALYSIS SAMPLE")
print("="*80)

# Start with full sample
print(f"Starting sample: {len(df):,}")

# Restriction 1: Hispanic-Mexican ethnicity (HISPAN == 1)
sample = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican restriction: {len(sample):,}")

# Restriction 2: Born in Mexico (BPL == 200)
sample = sample[sample['BPL'] == 200].copy()
print(f"After Mexico birthplace restriction: {len(sample):,}")

# Restriction 3: Non-citizen (CITIZEN == 3)
# Per instructions: "anyone who is not a citizen and who has not received
# immigration papers is undocumented for DACA purposes"
sample = sample[sample['CITIZEN'] == 3].copy()
print(f"After non-citizen restriction: {len(sample):,}")

# Restriction 4: Working age (16-45) to focus on relevant labor market participants
sample = sample[(sample['AGE'] >= 16) & (sample['AGE'] <= 45)].copy()
print(f"After working age (16-45) restriction: {len(sample):,}")

# Restriction 5: Exclude 2012 (DACA implemented mid-year)
sample = sample[sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(sample):,}")

# Restriction 6: Valid year of immigration
sample = sample[sample['YRIMMIG'] > 0].copy()
print(f"After valid YRIMMIG restriction: {len(sample):,}")

print(f"\nFinal analysis sample: {len(sample):,}")

# ============================================================================
# STEP 4: Define DACA Eligibility
# ============================================================================
print("\n" + "="*80)
print("STEP 4: DEFINING DACA ELIGIBILITY")
print("="*80)

"""
DACA Eligibility Criteria:
1. Arrived in US before 16th birthday: (YRIMMIG - BIRTHYR) < 16
2. Under 31 on June 15, 2012: Born after June 15, 1981
   - BIRTHYR >= 1982, OR
   - BIRTHYR == 1981 and BIRTHQTR >= 3 (July-Dec)
3. In US continuously since June 15, 2007: YRIMMIG <= 2007
4. Not a citizen: Already restricted
5. At least 15 years old at time of application (August 2012):
   Born before August 1997, so BIRTHYR <= 1996

Note: We cannot directly observe continuous presence or unlawful entry,
so we use year of immigration as proxy.
"""

# Calculate age at arrival
sample['age_at_arrival'] = sample['YRIMMIG'] - sample['BIRTHYR']

# Criterion 1: Arrived before 16th birthday
sample['arrived_before_16'] = (sample['age_at_arrival'] < 16).astype(int)

# Criterion 2: Under 31 on June 15, 2012 (born after June 15, 1981)
# Born 1982 or later, OR born 1981 in Q3 or Q4 (after June)
sample['under_31_in_2012'] = (
    (sample['BIRTHYR'] >= 1982) |
    ((sample['BIRTHYR'] == 1981) & (sample['BIRTHQTR'] >= 3))
).astype(int)

# Criterion 3: In US since June 2007 (arrived 2007 or earlier)
sample['in_us_since_2007'] = (sample['YRIMMIG'] <= 2007).astype(int)

# Criterion 4: At least 15 at time of application (born 1997 or earlier)
sample['at_least_15_in_2012'] = (sample['BIRTHYR'] <= 1997).astype(int)

# DACA eligible = meets all criteria
sample['daca_eligible'] = (
    (sample['arrived_before_16'] == 1) &
    (sample['under_31_in_2012'] == 1) &
    (sample['in_us_since_2007'] == 1) &
    (sample['at_least_15_in_2012'] == 1)
).astype(int)

print("Eligibility criteria breakdown:")
print(f"  Arrived before 16: {sample['arrived_before_16'].sum():,} ({sample['arrived_before_16'].mean()*100:.1f}%)")
print(f"  Under 31 in 2012: {sample['under_31_in_2012'].sum():,} ({sample['under_31_in_2012'].mean()*100:.1f}%)")
print(f"  In US since 2007: {sample['in_us_since_2007'].sum():,} ({sample['in_us_since_2007'].mean()*100:.1f}%)")
print(f"  At least 15 in 2012: {sample['at_least_15_in_2012'].sum():,} ({sample['at_least_15_in_2012'].mean()*100:.1f}%)")
print(f"  DACA eligible (all criteria): {sample['daca_eligible'].sum():,} ({sample['daca_eligible'].mean()*100:.1f}%)")

# ============================================================================
# STEP 5: Define Outcome Variable and Treatment Period
# ============================================================================
print("\n" + "="*80)
print("STEP 5: DEFINING OUTCOME AND TREATMENT")
print("="*80)

# Full-time employment: Usually working 35+ hours per week
sample['fulltime'] = (sample['UHRSWORK'] >= 35).astype(int)

# Post-DACA indicator (2013-2016)
sample['post'] = (sample['YEAR'] >= 2013).astype(int)

# Interaction term for DiD
sample['eligible_x_post'] = sample['daca_eligible'] * sample['post']

print(f"Full-time employment rate: {sample['fulltime'].mean()*100:.1f}%")
print(f"\nSample distribution by treatment status:")
print(sample.groupby(['daca_eligible', 'post']).size().unstack(fill_value=0))

# ============================================================================
# STEP 6: Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("STEP 6: SUMMARY STATISTICS")
print("="*80)

# Overall sample characteristics
print("\nSample by year:")
year_stats = sample.groupby('YEAR').agg({
    'fulltime': ['count', 'mean'],
    'daca_eligible': 'mean',
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(3)
print(year_stats)

# By eligibility status
print("\nCharacteristics by DACA eligibility status:")
for eligible in [0, 1]:
    subset = sample[sample['daca_eligible'] == eligible]
    status = "DACA Eligible" if eligible == 1 else "Not Eligible"
    print(f"\n{status} (N = {len(subset):,}):")
    print(f"  Mean age: {subset['AGE'].mean():.1f}")
    print(f"  Female (%): {(subset['SEX'] == 2).mean()*100:.1f}")
    print(f"  Mean age at arrival: {subset['age_at_arrival'].mean():.1f}")
    print(f"  Mean year of immigration: {subset['YRIMMIG'].mean():.0f}")
    print(f"  Full-time employment (%): {subset['fulltime'].mean()*100:.1f}")

# Pre-post comparison
print("\nFull-time employment by eligibility and period:")
ft_table = sample.groupby(['daca_eligible', 'post'])['fulltime'].mean().unstack()
ft_table.index = ['Not Eligible', 'DACA Eligible']
ft_table.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
print(ft_table.round(4))

# Calculate raw DiD
pre_eligible = sample[(sample['daca_eligible']==1) & (sample['post']==0)]['fulltime'].mean()
post_eligible = sample[(sample['daca_eligible']==1) & (sample['post']==1)]['fulltime'].mean()
pre_ineligible = sample[(sample['daca_eligible']==0) & (sample['post']==0)]['fulltime'].mean()
post_ineligible = sample[(sample['daca_eligible']==0) & (sample['post']==1)]['fulltime'].mean()

raw_did = (post_eligible - pre_eligible) - (post_ineligible - pre_ineligible)
print(f"\nRaw DiD estimate: {raw_did:.4f} ({raw_did*100:.2f} percentage points)")

# ============================================================================
# STEP 7: Main Difference-in-Differences Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 7: DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post', data=sample).fit()
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
sample['female'] = (sample['SEX'] == 2).astype(int)
sample['married'] = (sample['MARST'] <= 2).astype(int)

# Education categories
sample['educ_hs'] = (sample['EDUC'] >= 6).astype(int)  # High school or more
sample['educ_college'] = (sample['EDUC'] >= 10).astype(int)  # College or more

model2 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + AGE + I(AGE**2) + female + married + educ_hs',
                  data=sample).fit()
print(model2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
sample['year_factor'] = pd.Categorical(sample['YEAR'])
model3 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + AGE + I(AGE**2) + female + married + educ_hs',
                  data=sample).fit()
# Print only key coefficients
print("Key coefficients:")
for var in ['daca_eligible', 'eligible_x_post']:
    coef = model3.params[var]
    se = model3.bse[var]
    pval = model3.pvalues[var]
    print(f"  {var}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# Model 4: DiD with year and state fixed effects
print("\n--- Model 4: DiD with Year and State Fixed Effects ---")
model4 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + I(AGE**2) + female + married + educ_hs',
                  data=sample).fit()
print("Key coefficients:")
for var in ['daca_eligible', 'eligible_x_post']:
    coef = model4.params[var]
    se = model4.bse[var]
    pval = model4.pvalues[var]
    print(f"  {var}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# ============================================================================
# STEP 8: Weighted Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 8: WEIGHTED ANALYSIS (Using Person Weights)")
print("="*80)

# Model 5: Weighted DiD with full controls
print("\n--- Model 5: Weighted DiD with Full Controls ---")
model5 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + I(AGE**2) + female + married + educ_hs',
                  data=sample, weights=sample['PERWT']).fit()
print("Key coefficients:")
for var in ['daca_eligible', 'eligible_x_post']:
    coef = model5.params[var]
    se = model5.bse[var]
    pval = model5.pvalues[var]
    print(f"  {var}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# ============================================================================
# STEP 9: Robustness Checks
# ============================================================================
print("\n" + "="*80)
print("STEP 9: ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Alternative age range (18-35)
print("\n--- Robustness 1: Restricted Age Range (18-35) ---")
sample_r1 = sample[(sample['AGE'] >= 18) & (sample['AGE'] <= 35)].copy()
model_r1 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + I(AGE**2) + female + married + educ_hs',
                    data=sample_r1).fit()
print(f"Sample size: {len(sample_r1):,}")
print(f"DiD coefficient: {model_r1.params['eligible_x_post']:.4f} (SE: {model_r1.bse['eligible_x_post']:.4f})")

# Robustness 2: Alternative control group (arrived at 16-18 instead of 16+)
print("\n--- Robustness 2: Alternative Control Group ---")
# Create tighter control: those who arrived between 16-25
sample_r2 = sample[(sample['arrived_before_16'] == 1) |
                   ((sample['age_at_arrival'] >= 16) & (sample['age_at_arrival'] <= 25))].copy()
model_r2 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + I(AGE**2) + female + married + educ_hs',
                    data=sample_r2).fit()
print(f"Sample size: {len(sample_r2):,}")
print(f"DiD coefficient: {model_r2.params['eligible_x_post']:.4f} (SE: {model_r2.bse['eligible_x_post']:.4f})")

# Robustness 3: Employed (any hours) as outcome
print("\n--- Robustness 3: Any Employment as Outcome ---")
sample['employed'] = (sample['UHRSWORK'] > 0).astype(int)
model_r3 = smf.ols('employed ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + I(AGE**2) + female + married + educ_hs',
                    data=sample).fit()
print(f"DiD coefficient: {model_r3.params['eligible_x_post']:.4f} (SE: {model_r3.bse['eligible_x_post']:.4f})")

# Robustness 4: Linear Probability Model with Clustered SE (by state)
print("\n--- Robustness 4: Clustered Standard Errors (by State) ---")
model_r4 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + AGE + I(AGE**2) + female + married + educ_hs',
                    data=sample).fit(cov_type='cluster', cov_kwds={'groups': sample['STATEFIP']})
print(f"DiD coefficient: {model_r4.params['eligible_x_post']:.4f} (SE: {model_r4.bse['eligible_x_post']:.4f})")

# ============================================================================
# STEP 10: Event Study / Placebo Test
# ============================================================================
print("\n" + "="*80)
print("STEP 10: EVENT STUDY ANALYSIS")
print("="*80)

# Create year-specific treatment effects
sample['eligible_2006'] = sample['daca_eligible'] * (sample['YEAR'] == 2006)
sample['eligible_2007'] = sample['daca_eligible'] * (sample['YEAR'] == 2007)
sample['eligible_2008'] = sample['daca_eligible'] * (sample['YEAR'] == 2008)
sample['eligible_2009'] = sample['daca_eligible'] * (sample['YEAR'] == 2009)
sample['eligible_2010'] = sample['daca_eligible'] * (sample['YEAR'] == 2010)
sample['eligible_2011'] = sample['daca_eligible'] * (sample['YEAR'] == 2011)  # Reference
sample['eligible_2013'] = sample['daca_eligible'] * (sample['YEAR'] == 2013)
sample['eligible_2014'] = sample['daca_eligible'] * (sample['YEAR'] == 2014)
sample['eligible_2015'] = sample['daca_eligible'] * (sample['YEAR'] == 2015)
sample['eligible_2016'] = sample['daca_eligible'] * (sample['YEAR'] == 2016)

model_event = smf.ols('''fulltime ~ daca_eligible +
                         eligible_2006 + eligible_2007 + eligible_2008 + eligible_2009 + eligible_2010 +
                         eligible_2013 + eligible_2014 + eligible_2015 + eligible_2016 +
                         C(YEAR) + AGE + I(AGE**2) + female + married + educ_hs''',
                       data=sample).fit()

print("Event Study Coefficients (relative to 2011):")
print(f"  2006: {model_event.params.get('eligible_2006', 'N/A'):.4f} (SE: {model_event.bse.get('eligible_2006', 'N/A'):.4f})")
print(f"  2007: {model_event.params.get('eligible_2007', 'N/A'):.4f} (SE: {model_event.bse.get('eligible_2007', 'N/A'):.4f})")
print(f"  2008: {model_event.params.get('eligible_2008', 'N/A'):.4f} (SE: {model_event.bse.get('eligible_2008', 'N/A'):.4f})")
print(f"  2009: {model_event.params.get('eligible_2009', 'N/A'):.4f} (SE: {model_event.bse.get('eligible_2009', 'N/A'):.4f})")
print(f"  2010: {model_event.params.get('eligible_2010', 'N/A'):.4f} (SE: {model_event.bse.get('eligible_2010', 'N/A'):.4f})")
print(f"  2011: REFERENCE")
print(f"  2013: {model_event.params.get('eligible_2013', 'N/A'):.4f} (SE: {model_event.bse.get('eligible_2013', 'N/A'):.4f})")
print(f"  2014: {model_event.params.get('eligible_2014', 'N/A'):.4f} (SE: {model_event.bse.get('eligible_2014', 'N/A'):.4f})")
print(f"  2015: {model_event.params.get('eligible_2015', 'N/A'):.4f} (SE: {model_event.bse.get('eligible_2015', 'N/A'):.4f})")
print(f"  2016: {model_event.params.get('eligible_2016', 'N/A'):.4f} (SE: {model_event.bse.get('eligible_2016', 'N/A'):.4f})")

# ============================================================================
# STEP 11: Heterogeneity Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 11: HETEROGENEITY ANALYSIS")
print("="*80)

# By gender
print("\n--- By Gender ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = sample[sample['SEX'] == sex]
    model_het = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + AGE + I(AGE**2) + married + educ_hs',
                         data=subset).fit()
    print(f"{label}: DiD = {model_het.params['eligible_x_post']:.4f} (SE: {model_het.bse['eligible_x_post']:.4f}), N = {len(subset):,}")

# By education
print("\n--- By Education ---")
for educ_val, label in [(0, 'Less than HS'), (1, 'HS or more')]:
    subset = sample[sample['educ_hs'] == educ_val]
    if len(subset) > 100:
        model_het = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + AGE + I(AGE**2) + female + married',
                             data=subset).fit()
        print(f"{label}: DiD = {model_het.params['eligible_x_post']:.4f} (SE: {model_het.bse['eligible_x_post']:.4f}), N = {len(subset):,}")

# By age at arrival
print("\n--- By Age at Arrival ---")
sample['young_arrival'] = (sample['age_at_arrival'] <= 10).astype(int)
for arr_val, label in [(1, 'Arrived age <= 10'), (0, 'Arrived age > 10')]:
    subset = sample[(sample['young_arrival'] == arr_val) & (sample['daca_eligible'] == 1) | (sample['daca_eligible'] == 0)]
    model_het = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + AGE + I(AGE**2) + female + married + educ_hs',
                         data=subset).fit()
    print(f"{label}: DiD = {model_het.params['eligible_x_post']:.4f} (SE: {model_het.bse['eligible_x_post']:.4f}), N = {len(subset):,}")

# ============================================================================
# STEP 12: Final Results Summary
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print("\n" + "="*60)
print("PREFERRED SPECIFICATION: Model 4 (Year + State FE)")
print("="*60)
print(f"\nDiD Coefficient (eligible_x_post): {model4.params['eligible_x_post']:.4f}")
print(f"Standard Error: {model4.bse['eligible_x_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['eligible_x_post', 0]:.4f}, {model4.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"P-value: {model4.pvalues['eligible_x_post']:.4f}")
print(f"Sample size: {int(model4.nobs):,}")

print("\n" + "-"*60)
print("Interpretation:")
effect_pct = model4.params['eligible_x_post'] * 100
print(f"DACA eligibility is associated with a {effect_pct:.2f} percentage point")
if model4.params['eligible_x_post'] > 0:
    print("INCREASE in the probability of full-time employment.")
else:
    print("DECREASE in the probability of full-time employment.")

if model4.pvalues['eligible_x_post'] < 0.05:
    print("This effect is STATISTICALLY SIGNIFICANT at the 5% level.")
else:
    print("This effect is NOT statistically significant at the 5% level.")

# Save key results for the report
results_dict = {
    'model1_coef': model1.params['eligible_x_post'],
    'model1_se': model1.bse['eligible_x_post'],
    'model2_coef': model2.params['eligible_x_post'],
    'model2_se': model2.bse['eligible_x_post'],
    'model3_coef': model3.params['eligible_x_post'],
    'model3_se': model3.bse['eligible_x_post'],
    'model4_coef': model4.params['eligible_x_post'],
    'model4_se': model4.bse['eligible_x_post'],
    'model4_ci_low': model4.conf_int().loc['eligible_x_post', 0],
    'model4_ci_high': model4.conf_int().loc['eligible_x_post', 1],
    'model4_pval': model4.pvalues['eligible_x_post'],
    'model4_nobs': int(model4.nobs),
    'model5_coef': model5.params['eligible_x_post'],
    'model5_se': model5.bse['eligible_x_post'],
    'raw_did': raw_did,
    'n_eligible': sample['daca_eligible'].sum(),
    'n_ineligible': (sample['daca_eligible'] == 0).sum(),
    'ft_rate_overall': sample['fulltime'].mean(),
}

# Export results
pd.Series(results_dict).to_csv('results_summary.csv')
print("\nResults saved to results_summary.csv")

# ============================================================================
# STEP 13: Export Tables for LaTeX
# ============================================================================
print("\n" + "="*80)
print("STEP 13: EXPORTING TABLES FOR LATEX")
print("="*80)

# Table 1: Sample characteristics
table1_data = []
for eligible in [0, 1]:
    subset = sample[sample['daca_eligible'] == eligible]
    row = {
        'Group': 'DACA Eligible' if eligible == 1 else 'Not Eligible',
        'N': len(subset),
        'Age': subset['AGE'].mean(),
        'Female (%)': (subset['SEX'] == 2).mean() * 100,
        'Married (%)': subset['married'].mean() * 100,
        'HS+ (%)': subset['educ_hs'].mean() * 100,
        'Full-time (%)': subset['fulltime'].mean() * 100,
        'Age at Arrival': subset['age_at_arrival'].mean(),
        'Year of Immigration': subset['YRIMMIG'].mean(),
    }
    table1_data.append(row)

table1 = pd.DataFrame(table1_data)
table1.to_csv('table1_summary_stats.csv', index=False)
print("Table 1 saved to table1_summary_stats.csv")

# Table 2: DiD Results
table2_data = {
    'Model': ['(1) Basic', '(2) Demographics', '(3) Year FE', '(4) Year+State FE', '(5) Weighted'],
    'DiD Coefficient': [
        f"{model1.params['eligible_x_post']:.4f}",
        f"{model2.params['eligible_x_post']:.4f}",
        f"{model3.params['eligible_x_post']:.4f}",
        f"{model4.params['eligible_x_post']:.4f}",
        f"{model5.params['eligible_x_post']:.4f}",
    ],
    'Std Error': [
        f"({model1.bse['eligible_x_post']:.4f})",
        f"({model2.bse['eligible_x_post']:.4f})",
        f"({model3.bse['eligible_x_post']:.4f})",
        f"({model4.bse['eligible_x_post']:.4f})",
        f"({model5.bse['eligible_x_post']:.4f})",
    ],
    'Demographics': ['No', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Year FE': ['No', 'No', 'Yes', 'Yes', 'Yes'],
    'State FE': ['No', 'No', 'No', 'Yes', 'Yes'],
    'Weights': ['No', 'No', 'No', 'No', 'Yes'],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)],
}
table2 = pd.DataFrame(table2_data)
table2.to_csv('table2_did_results.csv', index=False)
print("Table 2 saved to table2_did_results.csv")

# Table 3: Robustness checks
table3_data = {
    'Specification': [
        'Main result (Model 4)',
        'Age 18-35 only',
        'Tighter control group',
        'Any employment outcome',
        'Clustered SE by state'
    ],
    'Coefficient': [
        f"{model4.params['eligible_x_post']:.4f}",
        f"{model_r1.params['eligible_x_post']:.4f}",
        f"{model_r2.params['eligible_x_post']:.4f}",
        f"{model_r3.params['eligible_x_post']:.4f}",
        f"{model_r4.params['eligible_x_post']:.4f}",
    ],
    'Std Error': [
        f"({model4.bse['eligible_x_post']:.4f})",
        f"({model_r1.bse['eligible_x_post']:.4f})",
        f"({model_r2.bse['eligible_x_post']:.4f})",
        f"({model_r3.bse['eligible_x_post']:.4f})",
        f"({model_r4.bse['eligible_x_post']:.4f})",
    ],
    'N': [
        int(model4.nobs),
        len(sample_r1),
        len(sample_r2),
        int(model_r3.nobs),
        int(model_r4.nobs),
    ],
}
table3 = pd.DataFrame(table3_data)
table3.to_csv('table3_robustness.csv', index=False)
print("Table 3 saved to table3_robustness.csv")

# Table 4: Event study results
event_years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs = []
event_ses = []
for year in event_years:
    if year == 2011:
        event_coefs.append(0.0)
        event_ses.append('(ref)')
    else:
        var = f'eligible_{year}'
        event_coefs.append(model_event.params.get(var, 0))
        event_ses.append(f"({model_event.bse.get(var, 0):.4f})")

table4 = pd.DataFrame({
    'Year': event_years,
    'Coefficient': event_coefs,
    'Std Error': event_ses
})
table4.to_csv('table4_event_study.csv', index=False)
print("Table 4 saved to table4_event_study.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
