"""
DACA Replication Analysis
Independent replication of the causal effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals.

Research Design: Difference-in-Differences
- Treatment group: Ages 26-30 as of June 15, 2012 (born 1982-1986)
- Control group: Ages 31-35 as of June 15, 2012 (born 1977-1981)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded due to policy implementation mid-year)
- Outcome: Full-time employment (>=35 hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = r"C:\Users\seraf\DACA Results Task 2\replication_08\data\data.csv"
OUTPUT_DIR = r"C:\Users\seraf\DACA Results Task 2\replication_08"

print("=" * 60)
print("DACA REPLICATION ANALYSIS")
print("=" * 60)

# Load data - only columns we need to reduce memory usage
cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR', 'MARST',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK', 'STATEFIP']

print("\nLoading data...")
df = pd.read_csv(DATA_PATH, usecols=cols_needed, low_memory=False)
print(f"Total observations loaded: {len(df):,}")

# ============================================================================
# STEP 1: DEFINE DACA ELIGIBILITY CRITERIA
# ============================================================================
print("\n" + "=" * 60)
print("STEP 1: IDENTIFYING DACA-ELIGIBLE POPULATION")
print("=" * 60)

# Filter to Hispanic-Mexican ethnicity (HISPAN == 1 for Mexican)
# According to data dictionary: HISPAN 1 = Mexican
df_hisp = df[df['HISPAN'] == 1].copy()
print(f"Hispanic-Mexican (HISPAN=1): {len(df_hisp):,}")

# Filter to born in Mexico (BPL == 200 for Mexico)
df_mex = df_hisp[df_hisp['BPL'] == 200].copy()
print(f"Born in Mexico (BPL=200): {len(df_mex):,}")

# Filter to non-citizens (CITIZEN == 3 for "Not a citizen")
# Assume non-citizens who haven't received papers are undocumented
df_noncit = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"Non-citizens (CITIZEN=3): {len(df_noncit):,}")

# DACA eligibility requires arrival before age 16
# Calculate approximate age at immigration
# YRIMMIG gives year of immigration
# We need to check if they arrived before turning 16

# For those with valid immigration year
df_noncit['age_at_immig'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']

# Keep those who immigrated before age 16
# Note: YRIMMIG == 0 means N/A, so we need valid immigration years
df_arrived_young = df_noncit[(df_noncit['YRIMMIG'] > 0) &
                             (df_noncit['age_at_immig'] < 16)].copy()
print(f"Arrived before age 16: {len(df_arrived_young):,}")

# DACA requires continuous presence since June 15, 2007
# We use YRIMMIG <= 2007 as a proxy for this requirement
df_continuous = df_arrived_young[df_arrived_young['YRIMMIG'] <= 2007].copy()
print(f"In US since 2007 or earlier (YRIMMIG<=2007): {len(df_continuous):,}")

# ============================================================================
# STEP 2: DEFINE TREATMENT AND CONTROL GROUPS BASED ON AGE
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: DEFINING TREATMENT AND CONTROL GROUPS")
print("=" * 60)

# DACA implemented June 15, 2012
# Treatment: Ages 26-30 as of June 15, 2012 -> Born between June 16, 1981 and June 15, 1986
# Control: Ages 31-35 as of June 15, 2012 -> Born between June 16, 1976 and June 15, 1981
# Since we don't have exact birth dates, use birth year:
# Treatment: BIRTHYR 1982-1986 (definitely under 31 on June 15, 2012)
# Control: BIRTHYR 1977-1981 (definitely 31+ on June 15, 2012)

# The cutoff is having 31st birthday AFTER June 15, 2012
# For safety and clear identification:
# Treatment (eligible): Born 1982-1986 (ages 26-30 on June 15, 2012)
# Control (ineligible due to age): Born 1977-1981 (ages 31-35 on June 15, 2012)

# Create treatment indicator based on birth year
df_analysis = df_continuous.copy()

# Treatment group: Born 1982-1986
# Control group: Born 1977-1981
df_analysis['treat'] = np.where(
    (df_analysis['BIRTHYR'] >= 1982) & (df_analysis['BIRTHYR'] <= 1986), 1,
    np.where((df_analysis['BIRTHYR'] >= 1977) & (df_analysis['BIRTHYR'] <= 1981), 0, np.nan)
)

# Keep only observations in treatment or control groups
df_analysis = df_analysis[df_analysis['treat'].notna()].copy()
print(f"Observations in treatment/control age range: {len(df_analysis):,}")

# ============================================================================
# STEP 3: DEFINE PRE AND POST PERIODS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: DEFINING PRE AND POST PERIODS")
print("=" * 60)

# Pre-period: 2006-2011 (before DACA)
# Post-period: 2013-2016 (after DACA implementation)
# Exclude 2012 due to mid-year implementation

df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = np.where(df_analysis['YEAR'] >= 2013, 1, 0)

print(f"Observations excluding 2012: {len(df_analysis):,}")
print(f"Pre-period (2006-2011): {len(df_analysis[df_analysis['post']==0]):,}")
print(f"Post-period (2013-2016): {len(df_analysis[df_analysis['post']==1]):,}")

# ============================================================================
# STEP 4: DEFINE OUTCOME VARIABLE
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: DEFINING OUTCOME VARIABLE")
print("=" * 60)

# Full-time employment: Usually working 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A for non-workers)
df_analysis['fulltime'] = np.where(df_analysis['UHRSWORK'] >= 35, 1, 0)

print(f"Full-time employed (35+ hrs/week): {df_analysis['fulltime'].sum():,}")
print(f"Full-time employment rate: {df_analysis['fulltime'].mean()*100:.2f}%")

# Also create employed indicator (EMPSTAT == 1)
df_analysis['employed'] = np.where(df_analysis['EMPSTAT'] == 1, 1, 0)
print(f"Employed: {df_analysis['employed'].sum():,}")
print(f"Employment rate: {df_analysis['employed'].mean()*100:.2f}%")

# ============================================================================
# STEP 5: CREATE COVARIATES
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: PREPARING COVARIATES")
print("=" * 60)

# Sex (1=Male, 2=Female)
df_analysis['female'] = np.where(df_analysis['SEX'] == 2, 1, 0)

# Marital status (MARST 1-2 = married)
df_analysis['married'] = np.where(df_analysis['MARST'].isin([1, 2]), 1, 0)

# Education categories based on EDUC
# 0-5: Less than HS, 6: HS, 7-9: Some college, 10-11: College+
df_analysis['educ_lths'] = np.where(df_analysis['EDUC'] < 6, 1, 0)
df_analysis['educ_hs'] = np.where(df_analysis['EDUC'] == 6, 1, 0)
df_analysis['educ_somecol'] = np.where((df_analysis['EDUC'] >= 7) & (df_analysis['EDUC'] <= 9), 1, 0)
df_analysis['educ_college'] = np.where(df_analysis['EDUC'] >= 10, 1, 0)

# Age at survey (for controlling within-group age effects)
df_analysis['age'] = df_analysis['AGE']

# Birth year (for cohort effects)
df_analysis['birthyr'] = df_analysis['BIRTHYR']

# State fixed effects
df_analysis['state'] = df_analysis['STATEFIP']

# Year
df_analysis['year'] = df_analysis['YEAR']

# Interaction term for DID
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

print(f"Female: {df_analysis['female'].mean()*100:.1f}%")
print(f"Married: {df_analysis['married'].mean()*100:.1f}%")
print(f"Less than HS: {df_analysis['educ_lths'].mean()*100:.1f}%")
print(f"HS: {df_analysis['educ_hs'].mean()*100:.1f}%")
print(f"Some college: {df_analysis['educ_somecol'].mean()*100:.1f}%")
print(f"College+: {df_analysis['educ_college'].mean()*100:.1f}%")

# ============================================================================
# STEP 6: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: DESCRIPTIVE STATISTICS")
print("=" * 60)

# Summary by treatment status and period
summary_stats = df_analysis.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'female': 'mean',
    'married': 'mean',
    'age': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary Statistics by Treatment Status and Period:")
print(summary_stats)

# Calculate weighted means
def weighted_mean(df, value_col, weight_col):
    return np.average(df[value_col], weights=df[weight_col])

print("\n" + "-" * 40)
print("WEIGHTED FULL-TIME EMPLOYMENT RATES:")
print("-" * 40)

for treat_val in [0, 1]:
    treat_label = "Treatment (26-30)" if treat_val == 1 else "Control (31-35)"
    for post_val in [0, 1]:
        period_label = "Post (2013-2016)" if post_val == 1 else "Pre (2006-2011)"
        subset = df_analysis[(df_analysis['treat'] == treat_val) & (df_analysis['post'] == post_val)]
        wt_mean = weighted_mean(subset, 'fulltime', 'PERWT')
        n_obs = len(subset)
        print(f"{treat_label}, {period_label}: {wt_mean*100:.2f}% (n={n_obs:,})")

# ============================================================================
# STEP 7: DIFFERENCE-IN-DIFFERENCES ESTIMATION
# ============================================================================
print("\n" + "=" * 60)
print("STEP 7: DIFFERENCE-IN-DIFFERENCES ESTIMATION")
print("=" * 60)

# Model 1: Basic DID without covariates
print("\n--- MODEL 1: Basic DID ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DID with demographic controls
print("\n--- MODEL 2: DID with Demographic Controls ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + age',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DID with demographic controls and education
print("\n--- MODEL 3: DID with Demographics and Education ---")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + age + educ_hs + educ_somecol + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: DID with year fixed effects
print("\n--- MODEL 4: DID with Year Fixed Effects ---")
df_analysis['year_factor'] = df_analysis['year'].astype('category')
model4 = smf.wls('fulltime ~ treat + treat_post + female + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

# Extract key coefficients
print("\nKey Coefficient (treat_post):")
print(f"Coefficient: {model4.params['treat_post']:.4f}")
print(f"Std Error: {model4.bse['treat_post']:.4f}")
print(f"t-stat: {model4.tvalues['treat_post']:.4f}")
print(f"p-value: {model4.pvalues['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

# Model 5: DID with state fixed effects
print("\n--- MODEL 5: DID with Year and State Fixed Effects ---")
model5 = smf.wls('fulltime ~ treat + treat_post + female + married + age + educ_hs + educ_somecol + educ_college + C(year) + C(state)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nKey Coefficient (treat_post):")
print(f"Coefficient: {model5.params['treat_post']:.4f}")
print(f"Std Error: {model5.bse['treat_post']:.4f}")
print(f"t-stat: {model5.tvalues['treat_post']:.4f}")
print(f"p-value: {model5.pvalues['treat_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")

# ============================================================================
# STEP 8: ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 8: ROBUSTNESS CHECKS")
print("=" * 60)

# Robustness 1: Alternative outcome - any employment
print("\n--- Robustness Check 1: Any Employment as Outcome ---")
model_emp = smf.wls('employed ~ treat + treat_post + female + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                     data=df_analysis,
                     weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"Effect on any employment: {model_emp.params['treat_post']:.4f} (SE: {model_emp.bse['treat_post']:.4f})")

# Robustness 2: Subgroup analysis by gender
print("\n--- Robustness Check 2: By Gender ---")
for sex_val, sex_label in [(0, 'Male'), (1, 'Female')]:
    subset = df_analysis[df_analysis['female'] == sex_val]
    model_sex = smf.wls('fulltime ~ treat + treat_post + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                        data=subset,
                        weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"{sex_label}: {model_sex.params['treat_post']:.4f} (SE: {model_sex.bse['treat_post']:.4f})")

# Robustness 3: Narrower age bands (27-29 vs 32-34)
print("\n--- Robustness Check 3: Narrower Age Bands (28-30 vs 32-34) ---")
df_narrow = df_analysis[
    ((df_analysis['BIRTHYR'] >= 1982) & (df_analysis['BIRTHYR'] <= 1984)) |
    ((df_analysis['BIRTHYR'] >= 1978) & (df_analysis['BIRTHYR'] <= 1980))
].copy()
df_narrow['treat_narrow'] = np.where(df_narrow['BIRTHYR'] >= 1982, 1, 0)
df_narrow['treat_post_narrow'] = df_narrow['treat_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treat_narrow + treat_post_narrow + female + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                        data=df_narrow,
                        weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"Narrower bands effect: {model_narrow.params['treat_post_narrow']:.4f} (SE: {model_narrow.bse['treat_post_narrow']:.4f})")

# Robustness 4: Placebo test using pre-period only (fake treatment at 2009)
print("\n--- Robustness Check 4: Placebo Test (Pre-period only, fake policy 2009) ---")
df_placebo = df_analysis[df_analysis['post'] == 0].copy()
df_placebo['post_placebo'] = np.where(df_placebo['year'] >= 2009, 1, 0)
df_placebo['treat_post_placebo'] = df_placebo['treat'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treat + treat_post_placebo + female + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                         data=df_placebo,
                         weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"Placebo effect: {model_placebo.params['treat_post_placebo']:.4f} (SE: {model_placebo.bse['treat_post_placebo']:.4f})")
print(f"p-value: {model_placebo.pvalues['treat_post_placebo']:.4f}")

# ============================================================================
# STEP 9: EVENT STUDY / DYNAMIC EFFECTS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 9: EVENT STUDY / DYNAMIC EFFECTS")
print("=" * 60)

# Create year-specific treatment effects (event study)
# Reference year: 2011 (last pre-treatment year)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]  # excluding 2011 as reference, 2012 excluded

for yr in years:
    df_analysis[f'treat_year_{yr}'] = np.where((df_analysis['treat'] == 1) & (df_analysis['year'] == yr), 1, 0)

event_formula = 'fulltime ~ treat + ' + ' + '.join([f'treat_year_{yr}' for yr in years]) + ' + female + married + age + educ_hs + educ_somecol + educ_college + C(year)'
model_event = smf.wls(event_formula, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
for yr in years:
    coef = model_event.params[f'treat_year_{yr}']
    se = model_event.bse[f'treat_year_{yr}']
    pval = model_event.pvalues[f'treat_year_{yr}']
    print(f"Year {yr}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 10: FINAL RESULTS SUMMARY")
print("=" * 60)

# Store all results for the report
results = {
    'n_total': len(df_analysis),
    'n_treat': len(df_analysis[df_analysis['treat'] == 1]),
    'n_control': len(df_analysis[df_analysis['treat'] == 0]),
    'n_pre': len(df_analysis[df_analysis['post'] == 0]),
    'n_post': len(df_analysis[df_analysis['post'] == 1]),

    # Model 1 (basic)
    'model1_coef': model1.params['treat_post'],
    'model1_se': model1.bse['treat_post'],
    'model1_pval': model1.pvalues['treat_post'],

    # Model 4 (with year FE - preferred)
    'model4_coef': model4.params['treat_post'],
    'model4_se': model4.bse['treat_post'],
    'model4_pval': model4.pvalues['treat_post'],
    'model4_ci_low': model4.conf_int().loc['treat_post', 0],
    'model4_ci_high': model4.conf_int().loc['treat_post', 1],

    # Model 5 (with state FE)
    'model5_coef': model5.params['treat_post'],
    'model5_se': model5.bse['treat_post'],
    'model5_pval': model5.pvalues['treat_post'],
}

print(f"\n{'='*60}")
print("PREFERRED ESTIMATE (Model 4: Year FE, Demographics, Education)")
print(f"{'='*60}")
print(f"Effect Size: {results['model4_coef']*100:.2f} percentage points")
print(f"Standard Error: {results['model4_se']*100:.2f} percentage points")
print(f"95% CI: [{results['model4_ci_low']*100:.2f}, {results['model4_ci_high']*100:.2f}]")
print(f"p-value: {results['model4_pval']:.4f}")
print(f"Sample Size: {results['n_total']:,}")

# Save results to file
import json
with open(f"{OUTPUT_DIR}/analysis_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/analysis_results.json")

# ============================================================================
# ADDITIONAL OUTPUTS FOR REPORT
# ============================================================================

# Table 1: Sample composition
print("\n" + "=" * 60)
print("TABLE 1: SAMPLE COMPOSITION")
print("=" * 60)

table1_data = []
for treat_val in [0, 1]:
    treat_label = "Control (31-35)" if treat_val == 0 else "Treatment (26-30)"
    subset = df_analysis[df_analysis['treat'] == treat_val]
    row = {
        'Group': treat_label,
        'N': len(subset),
        'Full-time (%)': subset['fulltime'].mean() * 100,
        'Employed (%)': subset['employed'].mean() * 100,
        'Female (%)': subset['female'].mean() * 100,
        'Married (%)': subset['married'].mean() * 100,
        'Mean Age': subset['age'].mean(),
        'Less than HS (%)': subset['educ_lths'].mean() * 100,
        'HS (%)': subset['educ_hs'].mean() * 100,
        'Some College (%)': subset['educ_somecol'].mean() * 100,
        'College+ (%)': subset['educ_college'].mean() * 100
    }
    table1_data.append(row)

table1 = pd.DataFrame(table1_data)
print(table1.to_string(index=False))

# Table 2: Full-time employment by group and period
print("\n" + "=" * 60)
print("TABLE 2: FULL-TIME EMPLOYMENT RATES BY GROUP AND PERIOD")
print("=" * 60)

table2_data = []
for treat_val in [0, 1]:
    treat_label = "Control (31-35)" if treat_val == 0 else "Treatment (26-30)"
    for post_val in [0, 1]:
        period_label = "Pre (2006-2011)" if post_val == 0 else "Post (2013-2016)"
        subset = df_analysis[(df_analysis['treat'] == treat_val) & (df_analysis['post'] == post_val)]
        wt_mean = weighted_mean(subset, 'fulltime', 'PERWT')
        row = {
            'Group': treat_label,
            'Period': period_label,
            'N': len(subset),
            'Full-time Rate (weighted)': wt_mean * 100
        }
        table2_data.append(row)

table2 = pd.DataFrame(table2_data)
print(table2.to_string(index=False))

# Calculate simple DID manually
pre_treat = df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==0)]['fulltime'].mean()
post_treat = df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==1)]['fulltime'].mean()
pre_control = df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==0)]['fulltime'].mean()
post_control = df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==1)]['fulltime'].mean()

print(f"\nSimple DID Calculation (unweighted):")
print(f"Treatment group change: {(post_treat - pre_treat)*100:.2f} pp")
print(f"Control group change: {(post_control - pre_control)*100:.2f} pp")
print(f"DID estimate: {((post_treat - pre_treat) - (post_control - pre_control))*100:.2f} pp")

# Save model summaries
with open(f"{OUTPUT_DIR}/model_summaries.txt", 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DACA REPLICATION - MODEL SUMMARIES\n")
    f.write("=" * 80 + "\n\n")

    f.write("MODEL 1: Basic DID\n")
    f.write("-" * 40 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("MODEL 2: DID with Demographics\n")
    f.write("-" * 40 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("MODEL 3: DID with Demographics and Education\n")
    f.write("-" * 40 + "\n")
    f.write(str(model3.summary()) + "\n\n")

    f.write("MODEL 4: DID with Year Fixed Effects (PREFERRED)\n")
    f.write("-" * 40 + "\n")
    f.write(str(model4.summary()) + "\n\n")

    f.write("MODEL 5: DID with Year and State Fixed Effects\n")
    f.write("-" * 40 + "\n")
    f.write(str(model5.summary()) + "\n\n")

print(f"\nModel summaries saved to {OUTPUT_DIR}/model_summaries.txt")

# Create data for figures
print("\n" + "=" * 60)
print("CREATING DATA FOR FIGURES")
print("=" * 60)

# Figure 1 data: Trends over time
fig1_data = df_analysis.groupby(['year', 'treat']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'employed_rate': np.average(x['employed'], weights=x['PERWT']),
        'n': len(x)
    })
).reset_index()
fig1_data.to_csv(f"{OUTPUT_DIR}/figure1_data.csv", index=False)
print("Figure 1 data saved.")

# Figure 2 data: Event study coefficients
event_study_data = []
for yr in years:
    event_study_data.append({
        'year': yr,
        'coef': model_event.params[f'treat_year_{yr}'],
        'se': model_event.bse[f'treat_year_{yr}'],
        'ci_low': model_event.conf_int().loc[f'treat_year_{yr}', 0],
        'ci_high': model_event.conf_int().loc[f'treat_year_{yr}', 1]
    })
# Add reference year 2011
event_study_data.append({'year': 2011, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
event_study_df = pd.DataFrame(event_study_data).sort_values('year')
event_study_df.to_csv(f"{OUTPUT_DIR}/figure2_event_study.csv", index=False)
print("Event study data saved.")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
