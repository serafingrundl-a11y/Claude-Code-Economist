"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the United States.

Treatment: ELIGIBLE (ages 26-30 at June 15, 2012)
Control: Ages 31-35 at June 15, 2012
Outcome: FT (full-time employment, 35+ hours/week)
Method: Difference-in-Differences
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
np.random.seed(83)

# Load data
print("=" * 60)
print("DACA REPLICATION ANALYSIS")
print("=" * 60)
print("\nLoading data...")
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Data shape: {df.shape}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# SECTION 1: DATA DESCRIPTION
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 1: DATA DESCRIPTION")
print("=" * 60)

print(f"\nTotal observations: {len(df)}")
print(f"\nObservations by period:")
print(f"  Pre-DACA (2008-2011): {len(df[df['AFTER']==0])}")
print(f"  Post-DACA (2013-2016): {len(df[df['AFTER']==1])}")

print(f"\nObservations by treatment group:")
print(f"  Eligible (ages 26-30): {len(df[df['ELIGIBLE']==1])}")
print(f"  Control (ages 31-35): {len(df[df['ELIGIBLE']==0])}")

# Create 2x2 table
print("\nSample distribution (2x2):")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (31-35)', 'Eligible (26-30)', 'Total']
crosstab.columns = ['Pre-DACA', 'Post-DACA', 'Total']
print(crosstab)

# Save sample sizes for later
n_eligible_before = len(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)])
n_eligible_after = len(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)])
n_control_before = len(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)])
n_control_after = len(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)])

# ============================================================================
# SECTION 2: SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 2: SUMMARY STATISTICS")
print("=" * 60)

# Key variables summary
print("\nOutcome Variable:")
print(f"  FT (Full-time employment): mean = {df['FT'].mean():.4f}, n = {len(df)}")

# Demographics
print("\nDemographic Variables:")
print(f"  Female (SEX=2): {(df['SEX']==2).mean():.4f}")
print(f"  Mean age in survey: {df['AGE'].mean():.2f}")
print(f"  Mean age at June 2012: {df['AGE_IN_JUNE_2012'].mean():.2f}")

# Education
print("\nEducation (EDUC_RECODE):")
print(df['EDUC_RECODE'].value_counts(normalize=True).round(4))

# Marital status
print("\nMarital Status (MARST):")
marst_labels = {1: 'Married, spouse present', 2: 'Married, spouse absent',
                3: 'Separated', 4: 'Divorced', 5: 'Widowed', 6: 'Never married'}
print(df['MARST'].value_counts(normalize=True).round(4))

# ============================================================================
# SECTION 3: BALANCE CHECK - PRE-TREATMENT CHARACTERISTICS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 3: BALANCE CHECK - PRE-TREATMENT CHARACTERISTICS")
print("=" * 60)

# Focus on pre-period
df_pre = df[df['AFTER'] == 0].copy()

# Compare treatment and control groups
print("\nPre-period comparison (eligible vs control):")
print(f"{'Variable':<25} {'Eligible':>12} {'Control':>12} {'Diff':>12} {'p-value':>10}")
print("-" * 71)

balance_vars = ['FT', 'SEX', 'FAMSIZE', 'NCHILD', 'MARST']
balance_results = []

for var in balance_vars:
    eligible_mean = df_pre[df_pre['ELIGIBLE']==1][var].mean()
    control_mean = df_pre[df_pre['ELIGIBLE']==0][var].mean()
    diff = eligible_mean - control_mean
    # t-test
    t_stat, p_val = stats.ttest_ind(
        df_pre[df_pre['ELIGIBLE']==1][var].dropna(),
        df_pre[df_pre['ELIGIBLE']==0][var].dropna()
    )
    print(f"{var:<25} {eligible_mean:>12.4f} {control_mean:>12.4f} {diff:>12.4f} {p_val:>10.4f}")
    balance_results.append({
        'Variable': var,
        'Eligible': eligible_mean,
        'Control': control_mean,
        'Difference': diff,
        'p_value': p_val
    })

balance_df = pd.DataFrame(balance_results)
balance_df.to_csv('balance_check.csv', index=False)

# ============================================================================
# SECTION 4: MAIN RESULTS - DIFFERENCE-IN-DIFFERENCES
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 4: MAIN RESULTS - DIFFERENCE-IN-DIFFERENCES")
print("=" * 60)

# 4a. Simple 2x2 DiD calculation
print("\n4a. Simple 2x2 DiD Calculation:")
eligible_before = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
eligible_after = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
control_before = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
control_after = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"\nMean Full-Time Employment Rates:")
print(f"{'Group':<20} {'Pre-DACA':>12} {'Post-DACA':>12} {'Change':>12}")
print("-" * 56)
print(f"{'Eligible (26-30)':<20} {eligible_before:>12.4f} {eligible_after:>12.4f} {eligible_after - eligible_before:>12.4f}")
print(f"{'Control (31-35)':<20} {control_before:>12.4f} {control_after:>12.4f} {control_after - control_before:>12.4f}")
print("-" * 56)
did_simple = (eligible_after - eligible_before) - (control_after - control_before)
print(f"{'DiD Estimate':<20} {'':<12} {'':<12} {did_simple:>12.4f}")

# 4b. Regression-based DiD (no controls)
print("\n4b. Regression-based DiD (Model 1 - No Controls):")
# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model1.summary())

# Extract key statistics for Model 1
did_coef1 = model1.params['ELIGIBLE_AFTER']
did_se1 = model1.bse['ELIGIBLE_AFTER']
did_ci1 = model1.conf_int().loc['ELIGIBLE_AFTER']
did_pval1 = model1.pvalues['ELIGIBLE_AFTER']
n1 = int(model1.nobs)

print(f"\nDiD Coefficient (ELIGIBLE x AFTER): {did_coef1:.4f}")
print(f"Robust Standard Error: {did_se1:.4f}")
print(f"95% CI: [{did_ci1[0]:.4f}, {did_ci1[1]:.4f}]")
print(f"p-value: {did_pval1:.4f}")
print(f"N: {n1}")

# 4c. DiD with demographic controls
print("\n4c. Regression-based DiD (Model 2 - With Demographic Controls):")
# Create dummy variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)
# Education dummies (baseline: Less than High School)
df['HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['BA_PLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)
# Marital status dummy
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + HS + SOMECOLL + TWOYEAR + BA_PLUS',
                 data=df).fit(cov_type='HC1')
print(model2.summary())

did_coef2 = model2.params['ELIGIBLE_AFTER']
did_se2 = model2.bse['ELIGIBLE_AFTER']
did_ci2 = model2.conf_int().loc['ELIGIBLE_AFTER']
did_pval2 = model2.pvalues['ELIGIBLE_AFTER']
n2 = int(model2.nobs)

print(f"\nDiD Coefficient (ELIGIBLE x AFTER): {did_coef2:.4f}")
print(f"Robust Standard Error: {did_se2:.4f}")
print(f"95% CI: [{did_ci2[0]:.4f}, {did_ci2[1]:.4f}]")
print(f"p-value: {did_pval2:.4f}")
print(f"N: {n2}")

# 4d. DiD with state and year fixed effects
print("\n4d. Regression-based DiD (Model 3 - With State and Year Fixed Effects):")

# Add state and year fixed effects using C() for categorical
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + HS + SOMECOLL + TWOYEAR + BA_PLUS + C(STATEFIP) + C(YEAR)',
                 data=df).fit(cov_type='HC1')

did_coef3 = model3.params['ELIGIBLE_AFTER']
did_se3 = model3.bse['ELIGIBLE_AFTER']
did_ci3 = model3.conf_int().loc['ELIGIBLE_AFTER']
did_pval3 = model3.pvalues['ELIGIBLE_AFTER']
n3 = int(model3.nobs)

print(f"\nDiD Coefficient (ELIGIBLE x AFTER): {did_coef3:.4f}")
print(f"Robust Standard Error: {did_se3:.4f}")
print(f"95% CI: [{did_ci3[0]:.4f}, {did_ci3[1]:.4f}]")
print(f"p-value: {did_pval3:.4f}")
print(f"N: {n3}")

# Also print other key coefficients
print("\nOther key coefficients (Model 3):")
for var in ['ELIGIBLE', 'FEMALE', 'MARRIED', 'HS', 'BA_PLUS']:
    if var in model3.params:
        print(f"  {var}: {model3.params[var]:.4f} (SE: {model3.bse[var]:.4f})")

# ============================================================================
# SECTION 5: ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 5: ROBUSTNESS CHECKS")
print("=" * 60)

# 5a. Using survey weights (PERWT)
print("\n5a. Weighted Regression (using PERWT):")
import statsmodels.api as sm

# Prepare data for weighted regression
X_weighted = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED', 'FAMSIZE', 'NCHILD', 'HS', 'SOMECOLL', 'TWOYEAR', 'BA_PLUS']].copy()
X_weighted = sm.add_constant(X_weighted)
y_weighted = df['FT']
weights = df['PERWT']

model_weighted = sm.WLS(y_weighted, X_weighted, weights=weights).fit(cov_type='HC1')
print(f"DiD Coefficient (weighted): {model_weighted.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE (weighted): {model_weighted.bse['ELIGIBLE_AFTER']:.4f}")

# 5b. State-level clustering for standard errors
print("\n5b. Clustered Standard Errors (by State):")
model_clustered = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + HS + SOMECOLL + TWOYEAR + BA_PLUS',
                          data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model_clustered.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {model_clustered.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_clustered.conf_int().loc['ELIGIBLE_AFTER'][0]:.4f}, {model_clustered.conf_int().loc['ELIGIBLE_AFTER'][1]:.4f}]")

# 5c. Subgroup analysis by sex
print("\n5c. Subgroup Analysis by Sex:")
# Males
df_male = df[df['SEX'] == 1]
model_male = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MARRIED + FAMSIZE + NCHILD + HS + SOMECOLL + TWOYEAR + BA_PLUS',
                     data=df_male).fit(cov_type='HC1')
print(f"Males: DiD = {model_male.params['ELIGIBLE_AFTER']:.4f} (SE: {model_male.bse['ELIGIBLE_AFTER']:.4f}), N = {int(model_male.nobs)}")

# Females
df_female = df[df['SEX'] == 2]
model_female = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MARRIED + FAMSIZE + NCHILD + HS + SOMECOLL + TWOYEAR + BA_PLUS',
                       data=df_female).fit(cov_type='HC1')
print(f"Females: DiD = {model_female.params['ELIGIBLE_AFTER']:.4f} (SE: {model_female.bse['ELIGIBLE_AFTER']:.4f}), N = {int(model_female.nobs)}")

# ============================================================================
# SECTION 6: EVENT STUDY / PARALLEL TRENDS CHECK
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 6: EVENT STUDY / PARALLEL TRENDS")
print("=" * 60)

# Create year dummies interacted with ELIGIBLE
# Use 2011 as reference year (last pre-treatment year)
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
# 2011 is reference
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Create interactions
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Event study regression
model_event = smf.ols('''FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 +
                         ELIGIBLE_YEAR_2008 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 +
                         ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016 +
                         FEMALE + MARRIED + FAMSIZE + NCHILD + HS + SOMECOLL + TWOYEAR + BA_PLUS''',
                      data=df).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
event_years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_results = []
for year in event_years:
    var = f'ELIGIBLE_YEAR_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci = model_event.conf_int().loc[var]
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}), 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    event_results.append({
        'year': year,
        'coef': coef,
        'se': se,
        'ci_low': ci[0],
        'ci_high': ci[1]
    })

# Add 2011 (reference year with 0)
event_results.append({'year': 2011, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
event_df = pd.DataFrame(event_results).sort_values('year')
event_df.to_csv('event_study_results.csv', index=False)

# Create event study plot
plt.figure(figsize=(10, 6))
event_df_plot = event_df.sort_values('year')
plt.errorbar(event_df_plot['year'], event_df_plot['coef'],
             yerr=[event_df_plot['coef'] - event_df_plot['ci_low'],
                   event_df_plot['ci_high'] - event_df_plot['coef']],
             fmt='o-', capsize=5, capthick=2, markersize=8, color='navy')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Coefficient (relative to 2011)', fontsize=12)
plt.title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('event_study_plot.png', dpi=150)
plt.close()
print("\nEvent study plot saved as 'event_study_plot.png'")

# ============================================================================
# SECTION 7: TRENDS BY GROUP
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 7: TRENDS BY GROUP")
print("=" * 60)

# Calculate mean FT by year and group
trends = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
trends.columns = ['Control (31-35)', 'Eligible (26-30)']
print("\nFull-Time Employment Rate by Year and Group:")
print(trends.round(4))

# Create trends plot
plt.figure(figsize=(10, 6))
plt.plot(trends.index, trends['Eligible (26-30)'], 'o-', label='Eligible (ages 26-30)',
         color='navy', markersize=8, linewidth=2)
plt.plot(trends.index, trends['Control (31-35)'], 's-', label='Control (ages 31-35)',
         color='darkred', markersize=8, linewidth=2)
plt.axvline(x=2012, color='gray', linestyle='--', alpha=0.7, label='DACA Implementation (2012)')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.ylim(0.55, 0.75)
plt.tight_layout()
plt.savefig('trends_plot.png', dpi=150)
plt.close()
print("\nTrends plot saved as 'trends_plot.png'")

# Save trends data
trends.to_csv('trends_data.csv')

# ============================================================================
# SECTION 8: SUMMARY OF RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 8: SUMMARY OF RESULTS")
print("=" * 60)

print("\n" + "=" * 80)
print("SUMMARY TABLE: DiD ESTIMATES OF DACA EFFECT ON FULL-TIME EMPLOYMENT")
print("=" * 80)
print(f"{'Model':<45} {'Coef':>10} {'SE':>10} {'95% CI':>22} {'N':>8}")
print("-" * 95)
print(f"{'(1) Basic DiD':<45} {did_coef1:>10.4f} {did_se1:>10.4f} {'[{:.4f}, {:.4f}]'.format(did_ci1[0], did_ci1[1]):>22} {n1:>8}")
print(f"{'(2) + Demographics':<45} {did_coef2:>10.4f} {did_se2:>10.4f} {'[{:.4f}, {:.4f}]'.format(did_ci2[0], did_ci2[1]):>22} {n2:>8}")
print(f"{'(3) + State & Year FE':<45} {did_coef3:>10.4f} {did_se3:>10.4f} {'[{:.4f}, {:.4f}]'.format(did_ci3[0], did_ci3[1]):>22} {n3:>8}")
print(f"{'(4) Clustered SE (by state)':<45} {model_clustered.params['ELIGIBLE_AFTER']:>10.4f} {model_clustered.bse['ELIGIBLE_AFTER']:>10.4f} {'[{:.4f}, {:.4f}]'.format(model_clustered.conf_int().loc['ELIGIBLE_AFTER'][0], model_clustered.conf_int().loc['ELIGIBLE_AFTER'][1]):>22} {n2:>8}")
print(f"{'(5) Weighted (PERWT)':<45} {model_weighted.params['ELIGIBLE_AFTER']:>10.4f} {model_weighted.bse['ELIGIBLE_AFTER']:>10.4f} {'':<22} {n2:>8}")
print("-" * 95)
print(f"{'Males only':<45} {model_male.params['ELIGIBLE_AFTER']:>10.4f} {model_male.bse['ELIGIBLE_AFTER']:>10.4f} {'':<22} {int(model_male.nobs):>8}")
print(f"{'Females only':<45} {model_female.params['ELIGIBLE_AFTER']:>10.4f} {model_female.bse['ELIGIBLE_AFTER']:>10.4f} {'':<22} {int(model_female.nobs):>8}")
print("=" * 95)

# ============================================================================
# SAVE ALL RESULTS TO CSV FILES
# ============================================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Main results table
main_results = pd.DataFrame({
    'Model': ['(1) Basic DiD', '(2) + Demographics', '(3) + State & Year FE',
              '(4) Clustered SE', '(5) Weighted', 'Males only', 'Females only'],
    'Coefficient': [did_coef1, did_coef2, did_coef3,
                   model_clustered.params['ELIGIBLE_AFTER'],
                   model_weighted.params['ELIGIBLE_AFTER'],
                   model_male.params['ELIGIBLE_AFTER'],
                   model_female.params['ELIGIBLE_AFTER']],
    'SE': [did_se1, did_se2, did_se3,
           model_clustered.bse['ELIGIBLE_AFTER'],
           model_weighted.bse['ELIGIBLE_AFTER'],
           model_male.bse['ELIGIBLE_AFTER'],
           model_female.bse['ELIGIBLE_AFTER']],
    'N': [n1, n2, n3, n2, n2, int(model_male.nobs), int(model_female.nobs)]
})
main_results.to_csv('main_results.csv', index=False)
print("Main results saved to 'main_results.csv'")

# Cell means
cell_means = pd.DataFrame({
    'Group': ['Eligible (26-30)', 'Eligible (26-30)', 'Control (31-35)', 'Control (31-35)'],
    'Period': ['Pre-DACA', 'Post-DACA', 'Pre-DACA', 'Post-DACA'],
    'FT_Rate': [eligible_before, eligible_after, control_before, control_after],
    'N': [n_eligible_before, n_eligible_after, n_control_before, n_control_after]
})
cell_means.to_csv('cell_means.csv', index=False)
print("Cell means saved to 'cell_means.csv'")

# ============================================================================
# PREFERRED ESTIMATE
# ============================================================================
print("\n" + "=" * 60)
print("PREFERRED ESTIMATE")
print("=" * 60)
print(f"\nPreferred Model: (2) DiD with Demographic Controls")
print(f"Effect Size: {did_coef2:.4f}")
print(f"Standard Error: {did_se2:.4f}")
print(f"95% Confidence Interval: [{did_ci2[0]:.4f}, {did_ci2[1]:.4f}]")
print(f"Sample Size: {n2}")
print(f"p-value: {did_pval2:.4f}")

# Interpretation
print(f"\nInterpretation:")
print(f"DACA eligibility is associated with a {did_coef2*100:.1f} percentage point")
print(f"increase in the probability of full-time employment (p = {did_pval2:.3f}).")
if did_pval2 < 0.05:
    print("This effect is statistically significant at the 5% level.")
elif did_pval2 < 0.10:
    print("This effect is statistically significant at the 10% level.")
else:
    print("This effect is not statistically significant at conventional levels.")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
