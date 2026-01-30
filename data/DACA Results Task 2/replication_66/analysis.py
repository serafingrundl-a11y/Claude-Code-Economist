"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals

Treatment: Ages 26-30 as of June 15, 2012 (DACA eligible)
Control: Ages 31-35 as of June 15, 2012 (too old for DACA)
Pre-period: 2006-2011
Post-period: 2013-2016 (excluding 2012 due to implementation timing)
Outcome: Full-time employment (working 35+ hours per week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# LOAD DATA
# ==========================================
print("Loading data...")
# Read only necessary columns to save memory
cols_needed = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX', 'HISPAN',
               'BPL', 'CITIZEN', 'YRIMMIG', 'UHRSWORK', 'EMPSTAT', 'EDUC',
               'MARST', 'STATEFIP']

df = pd.read_csv('data/data.csv', usecols=cols_needed, low_memory=False)
print(f"Total observations loaded: {len(df):,}")

# ==========================================
# SAMPLE RESTRICTIONS
# ==========================================
print("\n" + "="*50)
print("APPLYING SAMPLE RESTRICTIONS")
print("="*50)

# Step 1: Restrict to Hispanic-Mexican ethnicity
# HISPAN = 1 indicates Mexican
df_mex = df[df['HISPAN'] == 1].copy()
print(f"After restricting to Hispanic-Mexican (HISPAN=1): {len(df_mex):,}")

# Step 2: Restrict to born in Mexico
# BPL = 200 indicates Mexico
df_mex = df_mex[df_mex['BPL'] == 200].copy()
print(f"After restricting to born in Mexico (BPL=200): {len(df_mex):,}")

# Step 3: Restrict to non-citizens (proxy for undocumented)
# CITIZEN = 3 means not a citizen
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"After restricting to non-citizens (CITIZEN=3): {len(df_mex):,}")

# Step 4: Exclude 2012 (DACA implementation year - cannot distinguish pre/post)
df_mex = df_mex[df_mex['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_mex):,}")

# ==========================================
# CALCULATE AGE AS OF JUNE 15, 2012
# ==========================================
print("\n" + "="*50)
print("CALCULATING AGE AS OF JUNE 15, 2012")
print("="*50)

# DACA was implemented June 15, 2012
# Age as of June 15, 2012 depends on birth year and quarter
# BIRTHQTR: 1 = Jan-Mar, 2 = Apr-Jun, 3 = Jul-Sep, 4 = Oct-Dec

# If born in Q1 (Jan-Mar) or Q2 (Apr-Jun before June 15), already had birthday by June 15
# If born in Q3 or Q4, hadn't had birthday yet by June 15

# Calculate age as of June 15, 2012
# For Q1-Q2 births: age = 2012 - BIRTHYR
# For Q3-Q4 births: age = 2012 - BIRTHYR - 1

df_mex['age_june2012'] = 2012 - df_mex['BIRTHYR']
# Adjust for those who hadn't had birthday yet (Q3, Q4)
df_mex.loc[df_mex['BIRTHQTR'].isin([3, 4]), 'age_june2012'] -= 1

print(f"Age as of June 2012 distribution:")
print(df_mex['age_june2012'].describe())

# ==========================================
# DEFINE TREATMENT AND CONTROL GROUPS
# ==========================================
print("\n" + "="*50)
print("DEFINING TREATMENT AND CONTROL GROUPS")
print("="*50)

# Treatment group: ages 26-30 as of June 15, 2012 (DACA eligible by age)
# Control group: ages 31-35 as of June 15, 2012 (too old for DACA)

df_mex['treated'] = ((df_mex['age_june2012'] >= 26) & (df_mex['age_june2012'] <= 30)).astype(int)
df_mex['control'] = ((df_mex['age_june2012'] >= 31) & (df_mex['age_june2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df_mex[(df_mex['treated'] == 1) | (df_mex['control'] == 1)].copy()
print(f"After restricting to treatment (26-30) and control (31-35): {len(df_analysis):,}")

print(f"\nTreatment group (ages 26-30): {df_analysis['treated'].sum():,}")
print(f"Control group (ages 31-35): {(df_analysis['control'] == 1).sum():,}")

# ==========================================
# CHECK DACA ELIGIBILITY CRITERIA
# ==========================================
print("\n" + "="*50)
print("CHECKING DACA ELIGIBILITY CRITERIA")
print("="*50)

# DACA eligibility requires:
# 1. Arrived before age 16 - need YRIMMIG and BIRTHYR
# 2. In US since June 15, 2007 - need YRIMMIG <= 2007
# 3. Under 31 as of June 15, 2012 - already restricted by age groups
# 4. Not a citizen - already restricted

# Calculate age at immigration
df_analysis['age_at_immigration'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']

# DACA requires arrival before age 16
# Also requires arrival by June 15, 2007 (using YRIMMIG <= 2007 as proxy)
df_analysis['daca_eligible_arrival'] = (
    (df_analysis['age_at_immigration'] < 16) &
    (df_analysis['YRIMMIG'] <= 2007) &
    (df_analysis['YRIMMIG'] > 0)  # Exclude missing values
)

print(f"Observations meeting arrival criteria (arrived <16 and by 2007): {df_analysis['daca_eligible_arrival'].sum():,}")

# Final sample: those meeting arrival criteria
df_final = df_analysis[df_analysis['daca_eligible_arrival']].copy()
print(f"\nFinal sample size: {len(df_final):,}")

# ==========================================
# CREATE OUTCOME VARIABLE
# ==========================================
print("\n" + "="*50)
print("CREATING OUTCOME VARIABLE")
print("="*50)

# Full-time employment: UHRSWORK >= 35
df_final['fulltime'] = (df_final['UHRSWORK'] >= 35).astype(int)

# Also create employed variable for additional analysis
# EMPSTAT = 1 means employed
df_final['employed'] = (df_final['EMPSTAT'] == 1).astype(int)

print(f"Full-time employment rate: {df_final['fulltime'].mean()*100:.2f}%")
print(f"Employment rate: {df_final['employed'].mean()*100:.2f}%")

# ==========================================
# CREATE PERIOD VARIABLE
# ==========================================
print("\n" + "="*50)
print("CREATING PERIOD VARIABLE")
print("="*50)

# Post = 1 for 2013-2016, 0 for 2006-2011
df_final['post'] = (df_final['YEAR'] >= 2013).astype(int)

print(f"Pre-period (2006-2011) observations: {(df_final['post'] == 0).sum():,}")
print(f"Post-period (2013-2016) observations: {(df_final['post'] == 1).sum():,}")

# ==========================================
# SUMMARY STATISTICS
# ==========================================
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

# Summary by group and period
summary = df_final.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary Statistics by Group and Period:")
print(summary)

# Calculate weighted means
print("\n" + "="*50)
print("WEIGHTED SUMMARY STATISTICS")
print("="*50)

for treat in [0, 1]:
    for post in [0, 1]:
        subset = df_final[(df_final['treated'] == treat) & (df_final['post'] == post)]
        group_name = "Treatment" if treat == 1 else "Control"
        period_name = "Post" if post == 1 else "Pre"
        weighted_ft = np.average(subset['fulltime'], weights=subset['PERWT'])
        weighted_emp = np.average(subset['employed'], weights=subset['PERWT'])
        n_obs = len(subset)
        pop = subset['PERWT'].sum()
        print(f"{group_name} ({period_name}): FT={weighted_ft*100:.2f}%, Emp={weighted_emp*100:.2f}%, N={n_obs:,}, Pop={pop:,.0f}")

# ==========================================
# DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ==========================================
print("\n" + "="*50)
print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*50)

# Create interaction term
df_final['treat_post'] = df_final['treated'] * df_final['post']

# Model 1: Basic DiD without covariates
print("\n--- Model 1: Basic DiD (unweighted) ---")
model1 = smf.ols('fulltime ~ treated + post + treat_post', data=df_final).fit(cov_type='HC1')
print(model1.summary())

# Model 2: Basic DiD with person weights
print("\n--- Model 2: Basic DiD (weighted) ---")
model2 = smf.wls('fulltime ~ treated + post + treat_post',
                  data=df_final,
                  weights=df_final['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with covariates (weighted)
print("\n--- Model 3: DiD with covariates (weighted) ---")

# Create covariate dummies
df_final['male'] = (df_final['SEX'] == 1).astype(int)
df_final['married'] = (df_final['MARST'].isin([1, 2])).astype(int)

# Education categories
df_final['educ_hs'] = (df_final['EDUC'] >= 6).astype(int)  # High school or more
df_final['educ_college'] = (df_final['EDUC'] >= 10).astype(int)  # Some college or more

model3 = smf.wls('fulltime ~ treated + post + treat_post + male + married + educ_hs',
                  data=df_final,
                  weights=df_final['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with year fixed effects (weighted)
print("\n--- Model 4: DiD with year fixed effects (weighted) ---")
df_final['year_factor'] = pd.Categorical(df_final['YEAR'])
model4 = smf.wls('fulltime ~ treated + treat_post + C(YEAR) + male + married + educ_hs',
                  data=df_final,
                  weights=df_final['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# Model 5: DiD with state fixed effects (weighted)
print("\n--- Model 5: DiD with state and year fixed effects (weighted) ---")
model5 = smf.wls('fulltime ~ treated + treat_post + C(YEAR) + C(STATEFIP) + male + married + educ_hs',
                  data=df_final,
                  weights=df_final['PERWT']).fit(cov_type='HC1')

# Print key coefficients
print("\nKey Coefficients (Model 5):")
print(f"treat_post coefficient: {model5.params['treat_post']:.4f}")
print(f"treat_post std error: {model5.bse['treat_post']:.4f}")
print(f"treat_post 95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"treat_post p-value: {model5.pvalues['treat_post']:.4f}")

# ==========================================
# CALCULATE SIMPLE DiD ESTIMATE
# ==========================================
print("\n" + "="*50)
print("SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION")
print("="*50)

# Calculate weighted means for 2x2 table
def weighted_mean(df, var, weight):
    return np.average(df[var], weights=df[weight])

treat_pre = df_final[(df_final['treated'] == 1) & (df_final['post'] == 0)]
treat_post = df_final[(df_final['treated'] == 1) & (df_final['post'] == 1)]
control_pre = df_final[(df_final['treated'] == 0) & (df_final['post'] == 0)]
control_post = df_final[(df_final['treated'] == 0) & (df_final['post'] == 1)]

ft_treat_pre = weighted_mean(treat_pre, 'fulltime', 'PERWT')
ft_treat_post = weighted_mean(treat_post, 'fulltime', 'PERWT')
ft_control_pre = weighted_mean(control_pre, 'fulltime', 'PERWT')
ft_control_post = weighted_mean(control_post, 'fulltime', 'PERWT')

diff_treat = ft_treat_post - ft_treat_pre
diff_control = ft_control_post - ft_control_pre
did_estimate = diff_treat - diff_control

print("\n2x2 Table of Full-Time Employment Rates (Weighted):")
print(f"                    Pre-DACA    Post-DACA    Difference")
print(f"Treatment (26-30)   {ft_treat_pre*100:.2f}%       {ft_treat_post*100:.2f}%        {diff_treat*100:+.2f}pp")
print(f"Control (31-35)     {ft_control_pre*100:.2f}%       {ft_control_post*100:.2f}%        {diff_control*100:+.2f}pp")
print(f"")
print(f"DiD Estimate: {did_estimate*100:.2f} percentage points")

# ==========================================
# EVENT STUDY ANALYSIS
# ==========================================
print("\n" + "="*50)
print("EVENT STUDY ANALYSIS")
print("="*50)

# Create year dummies interacted with treatment
df_final['treat_2006'] = df_final['treated'] * (df_final['YEAR'] == 2006).astype(int)
df_final['treat_2007'] = df_final['treated'] * (df_final['YEAR'] == 2007).astype(int)
df_final['treat_2008'] = df_final['treated'] * (df_final['YEAR'] == 2008).astype(int)
df_final['treat_2009'] = df_final['treated'] * (df_final['YEAR'] == 2009).astype(int)
df_final['treat_2010'] = df_final['treated'] * (df_final['YEAR'] == 2010).astype(int)
df_final['treat_2011'] = df_final['treated'] * (df_final['YEAR'] == 2011).astype(int)  # Reference year
df_final['treat_2013'] = df_final['treated'] * (df_final['YEAR'] == 2013).astype(int)
df_final['treat_2014'] = df_final['treated'] * (df_final['YEAR'] == 2014).astype(int)
df_final['treat_2015'] = df_final['treated'] * (df_final['YEAR'] == 2015).astype(int)
df_final['treat_2016'] = df_final['treated'] * (df_final['YEAR'] == 2016).astype(int)

# Event study regression (2011 as reference)
event_formula = 'fulltime ~ treated + treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2013 + treat_2014 + treat_2015 + treat_2016 + C(YEAR) + male + married + educ_hs'
event_model = smf.wls(event_formula, data=df_final, weights=df_final['PERWT']).fit(cov_type='HC1')

# Extract coefficients for plot
event_years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs = [
    event_model.params.get('treat_2006', 0),
    event_model.params.get('treat_2007', 0),
    event_model.params.get('treat_2008', 0),
    event_model.params.get('treat_2009', 0),
    event_model.params.get('treat_2010', 0),
    0,  # 2011 is reference
    event_model.params.get('treat_2013', 0),
    event_model.params.get('treat_2014', 0),
    event_model.params.get('treat_2015', 0),
    event_model.params.get('treat_2016', 0)
]
event_ses = [
    event_model.bse.get('treat_2006', 0),
    event_model.bse.get('treat_2007', 0),
    event_model.bse.get('treat_2008', 0),
    event_model.bse.get('treat_2009', 0),
    event_model.bse.get('treat_2010', 0),
    0,  # 2011 is reference
    event_model.bse.get('treat_2013', 0),
    event_model.bse.get('treat_2014', 0),
    event_model.bse.get('treat_2015', 0),
    event_model.bse.get('treat_2016', 0)
]

print("\nEvent Study Coefficients:")
for y, c, s in zip(event_years, event_coefs, event_ses):
    ci_low = c - 1.96*s
    ci_high = c + 1.96*s
    print(f"Year {y}: {c:.4f} (SE: {s:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}])")

# ==========================================
# CREATE FIGURES
# ==========================================
print("\n" + "="*50)
print("CREATING FIGURES")
print("="*50)

# Figure 1: Event Study Plot
plt.figure(figsize=(10, 6))
plt.errorbar(event_years, event_coefs, yerr=[1.96*s for s in event_ses],
             fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Effect on Full-Time Employment (Treatment - Control)', fontsize=12)
plt.title('Event Study: Effect of DACA on Full-Time Employment', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: event_study.png")

# Figure 2: Trends in Full-Time Employment by Group
yearly_means = df_final.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

plt.figure(figsize=(10, 6))
plt.plot(yearly_means.index, yearly_means[0]*100, 'o-', label='Control (Ages 31-35)', linewidth=2, markersize=8)
plt.plot(yearly_means.index, yearly_means[1]*100, 's-', label='Treatment (Ages 26-30)', linewidth=2, markersize=8)
plt.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate (%)', fontsize=12)
plt.title('Full-Time Employment Trends by Treatment Status', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: trends.png")

# Figure 3: Distribution of Age at Immigration
plt.figure(figsize=(10, 6))
age_imm_treat = df_final[df_final['treated'] == 1]['age_at_immigration']
age_imm_control = df_final[df_final['treated'] == 0]['age_at_immigration']
plt.hist(age_imm_treat, bins=range(0, 20), alpha=0.5, label='Treatment (Ages 26-30)', density=True)
plt.hist(age_imm_control, bins=range(0, 20), alpha=0.5, label='Control (Ages 31-35)', density=True)
plt.xlabel('Age at Immigration', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Age at Immigration by Treatment Status', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('age_immigration.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: age_immigration.png")

# ==========================================
# ROBUSTNESS CHECKS
# ==========================================
print("\n" + "="*50)
print("ROBUSTNESS CHECKS")
print("="*50)

# Robustness 1: Employment (any work) as outcome
print("\n--- Robustness Check 1: Employment (any) as outcome ---")
model_emp = smf.wls('employed ~ treated + post + treat_post + male + married + educ_hs',
                     data=df_final,
                     weights=df_final['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient on employment: {model_emp.params['treat_post']:.4f} (SE: {model_emp.bse['treat_post']:.4f})")

# Robustness 2: Narrower age bands (27-29 vs 32-34)
print("\n--- Robustness Check 2: Narrower age bands (27-29 vs 32-34) ---")
df_narrow = df_final[
    ((df_final['age_june2012'] >= 27) & (df_final['age_june2012'] <= 29)) |
    ((df_final['age_june2012'] >= 32) & (df_final['age_june2012'] <= 34))
].copy()
df_narrow['treated_narrow'] = ((df_narrow['age_june2012'] >= 27) & (df_narrow['age_june2012'] <= 29)).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treated_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treated_narrow + post + treat_post_narrow + male + married + educ_hs',
                        data=df_narrow,
                        weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient (narrow bands): {model_narrow.params['treat_post_narrow']:.4f} (SE: {model_narrow.bse['treat_post_narrow']:.4f})")

# Robustness 3: Only post-2010 pre-period
print("\n--- Robustness Check 3: 2010-2011 vs 2013-2014 (closer to cutoff) ---")
df_close = df_final[df_final['YEAR'].isin([2010, 2011, 2013, 2014])].copy()
df_close['post_close'] = (df_close['YEAR'] >= 2013).astype(int)
df_close['treat_post_close'] = df_close['treated'] * df_close['post_close']

model_close = smf.wls('fulltime ~ treated + post_close + treat_post_close + male + married + educ_hs',
                       data=df_close,
                       weights=df_close['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient (close years): {model_close.params['treat_post_close']:.4f} (SE: {model_close.bse['treat_post_close']:.4f})")

# Robustness 4: By gender
print("\n--- Robustness Check 4: By Gender ---")
df_male = df_final[df_final['male'] == 1]
df_female = df_final[df_final['male'] == 0]

model_male = smf.wls('fulltime ~ treated + post + treat_post',
                      data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ treated + post + treat_post',
                        data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient (males): {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f})")
print(f"DiD coefficient (females): {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f})")

# ==========================================
# SAVE KEY RESULTS
# ==========================================
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)

print("\n*** PREFERRED ESTIMATE (Model 4 with year FE and covariates) ***")
print(f"Effect of DACA eligibility on full-time employment:")
print(f"  Coefficient: {model4.params['treat_post']:.4f}")
print(f"  Standard Error: {model4.bse['treat_post']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  p-value: {model4.pvalues['treat_post']:.4f}")
print(f"  Sample Size: {len(df_final):,}")

# Save results to file
with open('results_summary.txt', 'w') as f:
    f.write("DACA Replication Study - Results Summary\n")
    f.write("="*50 + "\n\n")
    f.write("Research Question: Effect of DACA eligibility on full-time employment\n")
    f.write("Treatment: Hispanic-Mexican Mexican-born non-citizens aged 26-30 as of June 2012\n")
    f.write("Control: Hispanic-Mexican Mexican-born non-citizens aged 31-35 as of June 2012\n")
    f.write("Additional eligibility: Arrived before age 16, in US by 2007\n\n")

    f.write("Sample Information:\n")
    f.write(f"  Total observations: {len(df_final):,}\n")
    f.write(f"  Treatment group: {(df_final['treated'] == 1).sum():,}\n")
    f.write(f"  Control group: {(df_final['treated'] == 0).sum():,}\n")
    f.write(f"  Pre-period (2006-2011): {(df_final['post'] == 0).sum():,}\n")
    f.write(f"  Post-period (2013-2016): {(df_final['post'] == 1).sum():,}\n\n")

    f.write("2x2 Table (Weighted Full-Time Employment Rates):\n")
    f.write(f"                    Pre-DACA    Post-DACA    Difference\n")
    f.write(f"Treatment (26-30)   {ft_treat_pre*100:.2f}%       {ft_treat_post*100:.2f}%        {diff_treat*100:+.2f}pp\n")
    f.write(f"Control (31-35)     {ft_control_pre*100:.2f}%       {ft_control_post*100:.2f}%        {diff_control*100:+.2f}pp\n")
    f.write(f"DiD Estimate: {did_estimate*100:.2f} percentage points\n\n")

    f.write("Preferred Estimate (Weighted OLS with year FE and covariates):\n")
    f.write(f"  Coefficient: {model4.params['treat_post']:.4f}\n")
    f.write(f"  Standard Error: {model4.bse['treat_post']:.4f}\n")
    f.write(f"  95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]\n")
    f.write(f"  p-value: {model4.pvalues['treat_post']:.4f}\n")

print("\nSaved: results_summary.txt")

# Save detailed statistics for report
stats_dict = {
    'n_total': len(df_final),
    'n_treat': (df_final['treated'] == 1).sum(),
    'n_control': (df_final['treated'] == 0).sum(),
    'n_pre': (df_final['post'] == 0).sum(),
    'n_post': (df_final['post'] == 1).sum(),
    'ft_treat_pre': ft_treat_pre,
    'ft_treat_post': ft_treat_post,
    'ft_control_pre': ft_control_pre,
    'ft_control_post': ft_control_post,
    'did_simple': did_estimate,
    'did_coef': model4.params['treat_post'],
    'did_se': model4.bse['treat_post'],
    'did_ci_low': model4.conf_int().loc['treat_post', 0],
    'did_ci_high': model4.conf_int().loc['treat_post', 1],
    'did_pval': model4.pvalues['treat_post']
}

import json
with open('stats_for_report.json', 'w') as f:
    json.dump(stats_dict, f, indent=2)
print("Saved: stats_for_report.json")

# Export regression tables
with open('regression_tables.txt', 'w') as f:
    f.write("Model 1: Basic DiD (unweighted)\n")
    f.write(model1.summary().as_text())
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("Model 2: Basic DiD (weighted)\n")
    f.write(model2.summary().as_text())
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("Model 3: DiD with covariates (weighted)\n")
    f.write(model3.summary().as_text())
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("Model 4: DiD with year FE (weighted) - PREFERRED\n")
    f.write(model4.summary().as_text())
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("Model 5: DiD with state and year FE (weighted)\n")
    f.write(model5.summary().as_text())
print("Saved: regression_tables.txt")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
