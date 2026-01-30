"""
DACA Effect on Full-Time Employment: Replication Analysis
Independent replication using difference-in-differences estimation

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the
United States, what was the causal impact of eligibility for the Deferred Action for
Childhood Arrivals (DACA) program on the probability that the eligible person is
employed full-time (defined as usually working 35 hours per week or more)?

Treatment Group: People aged 26-30 at the time DACA went into effect (June 2012)
Control Group: People aged 31-35 at the time DACA went into effect (who otherwise would
               have been eligible if not for their age)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("DACA REPLICATION ANALYSIS - Full-Time Employment Effects")
print("="*80)

# Load the data
print("\nLoading data...")
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations loaded: {len(df):,}")

# Check for the labelled version to understand coding better
df_labels = pd.read_csv('data/prepared_data_labelled_version.csv')

# Display basic info
print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)

# Check key variables
print("\nKey variables for analysis:")
print(f"  - FT (Full-time employment): {df['FT'].value_counts().to_dict()}")
print(f"  - AFTER (Post-DACA): {df['AFTER'].value_counts().to_dict()}")
print(f"  - ELIGIBLE (Treatment group): {df['ELIGIBLE'].value_counts().to_dict()}")

# Check years
print(f"\n  Years in data: {sorted(df['YEAR'].unique())}")

# Check AGE_IN_JUNE_2012 distribution
print(f"\n  AGE_IN_JUNE_2012 range: {df['AGE_IN_JUNE_2012'].min():.1f} - {df['AGE_IN_JUNE_2012'].max():.1f}")

# Check treatment/control group definition
print("\nVerifying treatment and control groups by AGE_IN_JUNE_2012:")
for elig in [0, 1]:
    age_range = df[df['ELIGIBLE'] == elig]['AGE_IN_JUNE_2012'].agg(['min', 'max', 'mean'])
    print(f"  ELIGIBLE={elig}: Age range {age_range['min']:.1f}-{age_range['max']:.1f}, Mean={age_range['mean']:.2f}")

# Sample sizes by group and time
print("\nSample sizes by treatment status and time period:")
cross = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
cross.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
cross.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)', 'Total']
print(cross)

# Full-time employment rates
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS: FULL-TIME EMPLOYMENT RATES")
print("="*80)

# Calculate mean FT by group and period
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
print("\nFull-time employment rates:")
for (elig, after), row in ft_means.iterrows():
    group = "Treatment (26-30)" if elig == 1 else "Control (31-35)"
    period = "Post-DACA" if after == 1 else "Pre-DACA"
    print(f"  {group}, {period}: {row['mean']*100:.2f}% (n={row['count']:,})")

# Calculate raw DiD estimate
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

did_raw = (post_treat - pre_treat) - (post_control - pre_control)

print(f"\nRaw Difference-in-Differences Calculation:")
print(f"  Treatment group change: {post_treat:.4f} - {pre_treat:.4f} = {post_treat - pre_treat:.4f}")
print(f"  Control group change:   {post_control:.4f} - {pre_control:.4f} = {post_control - pre_control:.4f}")
print(f"  DiD estimate:          {did_raw:.4f} ({did_raw*100:.2f} percentage points)")

# Store results for report
results = {
    'raw_did': did_raw,
    'pre_treat': pre_treat,
    'post_treat': post_treat,
    'pre_control': pre_control,
    'post_control': post_control,
    'n_total': len(df),
    'n_treatment': len(df[df['ELIGIBLE']==1]),
    'n_control': len(df[df['ELIGIBLE']==0])
}

print("\n" + "="*80)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD (no covariates)
print("\n--- Model 1: Basic DiD without covariates ---")
df['TREAT_POST'] = df['ELIGIBLE'] * df['AFTER']
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + TREAT_POST', data=df).fit(cov_type='HC1')
print(model1.summary())

results['model1_coef'] = model1.params['TREAT_POST']
results['model1_se'] = model1.bse['TREAT_POST']
results['model1_pval'] = model1.pvalues['TREAT_POST']
results['model1_ci_low'] = model1.conf_int().loc['TREAT_POST', 0]
results['model1_ci_high'] = model1.conf_int().loc['TREAT_POST', 1]
results['model1_n'] = int(model1.nobs)
results['model1_r2'] = model1.rsquared

# Model 2: DiD with year fixed effects
print("\n--- Model 2: DiD with year fixed effects ---")
df['YEAR_cat'] = df['YEAR'].astype('category')
model2 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + TREAT_POST', data=df).fit(cov_type='HC1')
print(f"DiD Coefficient: {model2.params['TREAT_POST']:.4f}")
print(f"Standard Error:  {model2.bse['TREAT_POST']:.4f}")
print(f"P-value:         {model2.pvalues['TREAT_POST']:.4f}")
print(f"95% CI:          [{model2.conf_int().loc['TREAT_POST', 0]:.4f}, {model2.conf_int().loc['TREAT_POST', 1]:.4f}]")
print(f"R-squared:       {model2.rsquared:.4f}")

results['model2_coef'] = model2.params['TREAT_POST']
results['model2_se'] = model2.bse['TREAT_POST']
results['model2_pval'] = model2.pvalues['TREAT_POST']
results['model2_ci_low'] = model2.conf_int().loc['TREAT_POST', 0]
results['model2_ci_high'] = model2.conf_int().loc['TREAT_POST', 1]
results['model2_r2'] = model2.rsquared

# Model 3: DiD with year and state fixed effects
print("\n--- Model 3: DiD with year and state fixed effects ---")
model3 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + TREAT_POST', data=df).fit(cov_type='HC1')
print(f"DiD Coefficient: {model3.params['TREAT_POST']:.4f}")
print(f"Standard Error:  {model3.bse['TREAT_POST']:.4f}")
print(f"P-value:         {model3.pvalues['TREAT_POST']:.4f}")
print(f"95% CI:          [{model3.conf_int().loc['TREAT_POST', 0]:.4f}, {model3.conf_int().loc['TREAT_POST', 1]:.4f}]")
print(f"R-squared:       {model3.rsquared:.4f}")

results['model3_coef'] = model3.params['TREAT_POST']
results['model3_se'] = model3.bse['TREAT_POST']
results['model3_pval'] = model3.pvalues['TREAT_POST']
results['model3_ci_low'] = model3.conf_int().loc['TREAT_POST', 0]
results['model3_ci_high'] = model3.conf_int().loc['TREAT_POST', 1]
results['model3_r2'] = model3.rsquared

# Model 4: Full model with covariates
print("\n--- Model 4: Full model with demographic covariates ---")

# Create binary indicators for certain variables
# SEX: 1=Male, 2=Female in IPUMS
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# MARST: 1=Married spouse present, others not
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Education categories
df['EDUC_LT_HS'] = (df['EDUC'] < 6).astype(int)  # Less than HS
df['EDUC_HS'] = (df['EDUC'] == 6).astype(int)     # HS degree
df['EDUC_SOME_COLLEGE'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)
df['EDUC_BA_PLUS'] = (df['EDUC'] >= 10).astype(int)

# Number of children
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Build the model formula
formula = 'FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + TREAT_POST + FEMALE + MARRIED + EDUC_HS + EDUC_SOME_COLLEGE + EDUC_BA_PLUS + NCHILD + AGE'

model4 = smf.ols(formula, data=df).fit(cov_type='HC1')
print(f"DiD Coefficient: {model4.params['TREAT_POST']:.4f}")
print(f"Standard Error:  {model4.bse['TREAT_POST']:.4f}")
print(f"P-value:         {model4.pvalues['TREAT_POST']:.4f}")
print(f"95% CI:          [{model4.conf_int().loc['TREAT_POST', 0]:.4f}, {model4.conf_int().loc['TREAT_POST', 1]:.4f}]")
print(f"R-squared:       {model4.rsquared:.4f}")

# Print covariate coefficients
print("\nCovariate coefficients:")
for var in ['FEMALE', 'MARRIED', 'EDUC_HS', 'EDUC_SOME_COLLEGE', 'EDUC_BA_PLUS', 'NCHILD', 'AGE']:
    print(f"  {var}: {model4.params[var]:.4f} (SE: {model4.bse[var]:.4f})")

results['model4_coef'] = model4.params['TREAT_POST']
results['model4_se'] = model4.bse['TREAT_POST']
results['model4_pval'] = model4.pvalues['TREAT_POST']
results['model4_ci_low'] = model4.conf_int().loc['TREAT_POST', 0]
results['model4_ci_high'] = model4.conf_int().loc['TREAT_POST', 1]
results['model4_r2'] = model4.rsquared

# Store covariate effects
results['coef_female'] = model4.params['FEMALE']
results['coef_married'] = model4.params['MARRIED']
results['coef_educ_hs'] = model4.params['EDUC_HS']
results['coef_educ_some_college'] = model4.params['EDUC_SOME_COLLEGE']
results['coef_educ_ba_plus'] = model4.params['EDUC_BA_PLUS']
results['coef_nchild'] = model4.params['NCHILD']
results['coef_age'] = model4.params['AGE']

# Model 5: With state-level policy variables
print("\n--- Model 5: Including state-level policy controls ---")
policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID', 'EVERIFY', 'SECURECOMMUNITIES']
formula5 = formula + ' + ' + ' + '.join(policy_vars)

model5 = smf.ols(formula5, data=df).fit(cov_type='HC1')
print(f"DiD Coefficient: {model5.params['TREAT_POST']:.4f}")
print(f"Standard Error:  {model5.bse['TREAT_POST']:.4f}")
print(f"P-value:         {model5.pvalues['TREAT_POST']:.4f}")
print(f"95% CI:          [{model5.conf_int().loc['TREAT_POST', 0]:.4f}, {model5.conf_int().loc['TREAT_POST', 1]:.4f}]")
print(f"R-squared:       {model5.rsquared:.4f}")

results['model5_coef'] = model5.params['TREAT_POST']
results['model5_se'] = model5.bse['TREAT_POST']
results['model5_pval'] = model5.pvalues['TREAT_POST']
results['model5_ci_low'] = model5.conf_int().loc['TREAT_POST', 0]
results['model5_ci_high'] = model5.conf_int().loc['TREAT_POST', 1]
results['model5_r2'] = model5.rsquared

print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Probit/Logit model
print("\n--- Robustness 1: Logit model ---")
logit_model = smf.logit('FT ~ ELIGIBLE + C(YEAR) + TREAT_POST', data=df).fit(disp=0)
# Calculate marginal effects
margeff = logit_model.get_margeff(at='overall')
# Find the index for TREAT_POST in the marginal effects summary
marg_summary = margeff.summary_frame()
print(f"Logit Marginal Effect of TREAT_POST: {marg_summary.loc['TREAT_POST', 'dy/dx']:.4f}")

results['logit_margeff'] = marg_summary.loc['TREAT_POST', 'dy/dx']

# Robustness 2: Weighted regression using PERWT
print("\n--- Robustness 2: Weighted regression with PERWT ---")
model_weighted = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + TREAT_POST', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Weighted DiD Coefficient: {model_weighted.params['TREAT_POST']:.4f}")
print(f"Standard Error:           {model_weighted.bse['TREAT_POST']:.4f}")
print(f"95% CI:                   [{model_weighted.conf_int().loc['TREAT_POST', 0]:.4f}, {model_weighted.conf_int().loc['TREAT_POST', 1]:.4f}]")

results['weighted_coef'] = model_weighted.params['TREAT_POST']
results['weighted_se'] = model_weighted.bse['TREAT_POST']

# Robustness 3: By gender
print("\n--- Robustness 3: Effects by gender ---")
df_male = df[df['SEX'] == 1]
df_female = df[df['SEX'] == 2]

model_male = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + TREAT_POST', data=df_male).fit(cov_type='HC1')
model_female = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + TREAT_POST', data=df_female).fit(cov_type='HC1')

print(f"Male DiD Coefficient:   {model_male.params['TREAT_POST']:.4f} (SE: {model_male.bse['TREAT_POST']:.4f})")
print(f"Female DiD Coefficient: {model_female.params['TREAT_POST']:.4f} (SE: {model_female.bse['TREAT_POST']:.4f})")

results['male_coef'] = model_male.params['TREAT_POST']
results['male_se'] = model_male.bse['TREAT_POST']
results['female_coef'] = model_female.params['TREAT_POST']
results['female_se'] = model_female.bse['TREAT_POST']
results['n_male'] = len(df_male)
results['n_female'] = len(df_female)

# Robustness 4: Placebo test - using 2010 as fake treatment year
print("\n--- Robustness 4: Placebo test (fake treatment in 2010) ---")
df_pre = df[df['AFTER'] == 0].copy()
df_pre['PLACEBO_POST'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['PLACEBO_TREAT_POST'] = df_pre['ELIGIBLE'] * df_pre['PLACEBO_POST']

model_placebo = smf.ols('FT ~ ELIGIBLE + PLACEBO_POST + PLACEBO_TREAT_POST', data=df_pre).fit(cov_type='HC1')
print(f"Placebo DiD Coefficient: {model_placebo.params['PLACEBO_TREAT_POST']:.4f}")
print(f"Standard Error:          {model_placebo.bse['PLACEBO_TREAT_POST']:.4f}")
print(f"P-value:                 {model_placebo.pvalues['PLACEBO_TREAT_POST']:.4f}")

results['placebo_coef'] = model_placebo.params['PLACEBO_TREAT_POST']
results['placebo_se'] = model_placebo.bse['PLACEBO_TREAT_POST']
results['placebo_pval'] = model_placebo.pvalues['PLACEBO_TREAT_POST']

# Event study / dynamic effects
print("\n--- Event Study: Year-by-year treatment effects ---")
df['TREAT_2008'] = ((df['ELIGIBLE'] == 1) & (df['YEAR'] == 2008)).astype(int)
df['TREAT_2009'] = ((df['ELIGIBLE'] == 1) & (df['YEAR'] == 2009)).astype(int)
df['TREAT_2010'] = ((df['ELIGIBLE'] == 1) & (df['YEAR'] == 2010)).astype(int)
# 2011 is the reference year
df['TREAT_2013'] = ((df['ELIGIBLE'] == 1) & (df['YEAR'] == 2013)).astype(int)
df['TREAT_2014'] = ((df['ELIGIBLE'] == 1) & (df['YEAR'] == 2014)).astype(int)
df['TREAT_2015'] = ((df['ELIGIBLE'] == 1) & (df['YEAR'] == 2015)).astype(int)
df['TREAT_2016'] = ((df['ELIGIBLE'] == 1) & (df['YEAR'] == 2016)).astype(int)

event_formula = 'FT ~ ELIGIBLE + C(YEAR) + TREAT_2008 + TREAT_2009 + TREAT_2010 + TREAT_2013 + TREAT_2014 + TREAT_2015 + TREAT_2016'
model_event = smf.ols(event_formula, data=df).fit(cov_type='HC1')

print("Year-by-year treatment effects (relative to 2011):")
event_years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_results = {}
for year in event_years:
    var = f'TREAT_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    print(f"  {year}: {coef:.4f} (SE: {se:.4f})")
    event_results[year] = {'coef': coef, 'se': se}

results['event_study'] = event_results

print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

# Figure 1: Parallel trends and DiD visualization
print("\nCreating Figure 1: Trends in full-time employment by group...")

yearly_means = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))
years = yearly_means.index.values
ax.plot(years, yearly_means[1], 'b-o', linewidth=2, markersize=8, label='Treatment (26-30)')
ax.plot(years, yearly_means[0], 'r-s', linewidth=2, markersize=8, label='Control (31-35)')
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')
ax.fill_between([2012, 2016.5], [0, 0], [1, 1], alpha=0.1, color='green')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends: Treatment vs Control Groups', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(2007.5, 2016.5)
ax.set_ylim(0.3, 0.7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure1_trends.png")

# Figure 2: Event study plot
print("Creating Figure 2: Event study plot...")

fig, ax = plt.subplots(figsize=(10, 6))
event_years_plot = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = []
ses = []
for year in event_years_plot:
    if year == 2011:
        coefs.append(0)
        ses.append(0)
    else:
        coefs.append(event_results[year]['coef'])
        ses.append(event_results[year]['se'])

ax.errorbar(event_years_plot, coefs, yerr=[1.96*se for se in ses],
            fmt='o', color='blue', markersize=8, capsize=5, capthick=2, linewidth=2)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
ax.fill_between([2012, 2016.5], [-0.2, -0.2], [0.2, 0.2], alpha=0.1, color='green')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('DiD Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Year-by-Year Treatment Effects', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(2007.5, 2016.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure2_event_study.png")

# Figure 3: Model comparison
print("Creating Figure 3: Model comparison...")

fig, ax = plt.subplots(figsize=(10, 6))
models = ['Basic\nDiD', 'Year\nFE', 'Year+State\nFE', 'Full\nCovariates', 'Policy\nControls', 'Weighted']
coefs_models = [results['model1_coef'], results['model2_coef'], results['model3_coef'],
                results['model4_coef'], results['model5_coef'], results['weighted_coef']]
ses_models = [results['model1_se'], results['model2_se'], results['model3_se'],
              results['model4_se'], results['model5_se'], results['weighted_se']]

x_pos = np.arange(len(models))
ax.bar(x_pos, coefs_models, yerr=[1.96*se for se in ses_models],
       capsize=5, color='steelblue', edgecolor='black', alpha=0.8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('DiD Coefficient', fontsize=12)
ax.set_title('DACA Effect on Full-Time Employment Across Model Specifications', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figure3_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure3_model_comparison.png")

# Figure 4: Effects by gender
print("Creating Figure 4: Effects by gender...")

fig, ax = plt.subplots(figsize=(8, 6))
categories = ['Male', 'Female', 'Overall']
coefs_gender = [results['male_coef'], results['female_coef'], results['model3_coef']]
ses_gender = [results['male_se'], results['female_se'], results['model3_se']]

x_pos = np.arange(len(categories))
colors = ['lightblue', 'lightcoral', 'lightgray']
ax.bar(x_pos, coefs_gender, yerr=[1.96*se for se in ses_gender],
       capsize=5, color=colors, edgecolor='black', alpha=0.8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel('DiD Coefficient', fontsize=12)
ax.set_title('DACA Effect on Full-Time Employment by Gender', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figure4_gender.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure4_gender.png")

print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print("\n" + "-"*60)
print("PREFERRED SPECIFICATION: Model 3 (Year + State Fixed Effects)")
print("-"*60)
print(f"  DiD Coefficient:  {results['model3_coef']:.4f}")
print(f"  Standard Error:   {results['model3_se']:.4f}")
print(f"  95% CI:           [{results['model3_ci_low']:.4f}, {results['model3_ci_high']:.4f}]")
print(f"  P-value:          {results['model3_pval']:.4f}")
print(f"  Sample size:      {results['n_total']:,}")
print("-"*60)

# Interpretation
if results['model3_pval'] < 0.05:
    significance = "statistically significant at the 5% level"
else:
    significance = "not statistically significant at the 5% level"

print(f"\nInterpretation: DACA eligibility is associated with a ")
print(f"{results['model3_coef']*100:.2f} percentage point change in the probability of ")
print(f"full-time employment. This effect is {significance}.")

# Save all results
results['ft_pre_treat'] = pre_treat
results['ft_post_treat'] = post_treat
results['ft_pre_control'] = pre_control
results['ft_post_control'] = post_control

print("\n" + "="*80)
print("ADDITIONAL DESCRIPTIVE STATISTICS FOR REPORT")
print("="*80)

# Balance table
print("\nCovariate balance (pre-treatment period):")
df_pre_balance = df[df['AFTER'] == 0]
balance_vars = ['AGE', 'FEMALE', 'MARRIED', 'NCHILD', 'EDUC']

print("\n{:<20} {:>15} {:>15} {:>15}".format('Variable', 'Treatment', 'Control', 'Difference'))
print("-"*65)
for var in balance_vars:
    treat_mean = df_pre_balance[df_pre_balance['ELIGIBLE']==1][var].mean()
    control_mean = df_pre_balance[df_pre_balance['ELIGIBLE']==0][var].mean()
    diff = treat_mean - control_mean
    print("{:<20} {:>15.3f} {:>15.3f} {:>15.3f}".format(var, treat_mean, control_mean, diff))

# Store balance statistics
results['balance_age_treat'] = df_pre_balance[df_pre_balance['ELIGIBLE']==1]['AGE'].mean()
results['balance_age_control'] = df_pre_balance[df_pre_balance['ELIGIBLE']==0]['AGE'].mean()
results['balance_female_treat'] = df_pre_balance[df_pre_balance['ELIGIBLE']==1]['FEMALE'].mean()
results['balance_female_control'] = df_pre_balance[df_pre_balance['ELIGIBLE']==0]['FEMALE'].mean()
results['balance_married_treat'] = df_pre_balance[df_pre_balance['ELIGIBLE']==1]['MARRIED'].mean()
results['balance_married_control'] = df_pre_balance[df_pre_balance['ELIGIBLE']==0]['MARRIED'].mean()

# State distribution
print("\nTop 5 states by sample size:")
state_counts = df.groupby('STATEFIP').size().sort_values(ascending=False).head(10)
print(state_counts)

# Save results to file for LaTeX generation
import json
with open('analysis_results.json', 'w') as f:
    # Convert numpy types to Python types
    results_clean = {}
    for k, v in results.items():
        if isinstance(v, (np.integer, np.floating)):
            results_clean[k] = float(v)
        elif isinstance(v, dict):
            results_clean[k] = {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                                for kk, vv in v.items()}
        else:
            results_clean[k] = v
    json.dump(results_clean, f, indent=2)
print("\nResults saved to analysis_results.json")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
