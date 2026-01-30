"""
DACA Replication Analysis
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the
United States, what was the causal impact of eligibility for DACA (treatment) on the
probability that the eligible person is employed full-time (outcome)?

Treatment: ELIGIBLE individuals ages 26-30 at time of policy (June 2012)
Control: Individuals ages 31-35 at time of policy who would have been eligible if not for age
Design: Difference-in-Differences comparing pre-DACA (2008-2011) to post-DACA (2013-2016)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

# Use the numeric version for analysis
data = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"\nDataset loaded: {len(data)} observations")

# Also load labelled version for reference
data_labels = pd.read_csv('data/prepared_data_labelled_version.csv')

# =============================================================================
# 1. Data Exploration
# =============================================================================
print("\n" + "="*80)
print("1. DATA EXPLORATION")
print("="*80)

print("\n--- Variable Summary ---")
print(f"Years in data: {sorted(data['YEAR'].unique())}")
print(f"ELIGIBLE values: {data['ELIGIBLE'].unique()}")
print(f"AFTER values: {data['AFTER'].unique()}")
print(f"FT values: {data['FT'].unique()}")

print("\n--- Sample Sizes by Treatment/Control and Period ---")
crosstab = pd.crosstab(data['ELIGIBLE'], data['AFTER'], margins=True)
crosstab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
crosstab.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)', 'Total']
print(crosstab)

print("\n--- Sample by Year ---")
year_counts = data.groupby('YEAR').size()
print(year_counts)

print("\n--- Full-Time Employment Rate by Group and Period ---")
ft_means = data.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean()
print("\nFT Rate by ELIGIBLE and AFTER:")
print(ft_means)

# Calculate simple DiD
ft_treat_pre = data[(data['ELIGIBLE']==1) & (data['AFTER']==0)]['FT'].mean()
ft_treat_post = data[(data['ELIGIBLE']==1) & (data['AFTER']==1)]['FT'].mean()
ft_ctrl_pre = data[(data['ELIGIBLE']==0) & (data['AFTER']==0)]['FT'].mean()
ft_ctrl_post = data[(data['ELIGIBLE']==0) & (data['AFTER']==1)]['FT'].mean()

print("\n--- Simple DiD Calculation ---")
print(f"Treatment (ELIGIBLE=1) Pre:  {ft_treat_pre:.4f}")
print(f"Treatment (ELIGIBLE=1) Post: {ft_treat_post:.4f}")
print(f"Treatment Change:            {ft_treat_post - ft_treat_pre:.4f}")
print(f"\nControl (ELIGIBLE=0) Pre:    {ft_ctrl_pre:.4f}")
print(f"Control (ELIGIBLE=0) Post:   {ft_ctrl_post:.4f}")
print(f"Control Change:              {ft_ctrl_post - ft_ctrl_pre:.4f}")
print(f"\nDiD Estimate: {(ft_treat_post - ft_treat_pre) - (ft_ctrl_post - ft_ctrl_pre):.4f}")

# =============================================================================
# 2. Demographic Summary Statistics
# =============================================================================
print("\n" + "="*80)
print("2. DEMOGRAPHIC SUMMARY STATISTICS")
print("="*80)

# Key variables
print("\n--- Sex Distribution ---")
print(data_labels['SEX'].value_counts(normalize=True))

print("\n--- Age Distribution ---")
print(data['AGE'].describe())

print("\n--- Education Distribution ---")
print(data_labels['EDUC_RECODE'].value_counts(normalize=True))

print("\n--- Marital Status Distribution ---")
print(data_labels['MARST'].value_counts(normalize=True))

print("\n--- Employment Status ---")
print(data_labels['EMPSTAT'].value_counts(normalize=True))

# =============================================================================
# 3. MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("3. MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

# Create interaction term
data['ELIGIBLE_AFTER'] = data['ELIGIBLE'] * data['AFTER']

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data).fit()
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")

# Prepare control variables
# SEX: 1=Male, 2=Female in IPUMS coding
data['FEMALE'] = (data['SEX'] == 2).astype(int)

# Age and age squared
data['AGE_SQ'] = data['AGE'] ** 2

# Marital status dummies (MARST: 1=married spouse present, 2=married absent, 3=separated, 4=divorced, 5=widowed, 6=never married)
data['MARRIED'] = (data['MARST'].isin([1, 2])).astype(int)

# Education dummies based on EDUC variable
# Create education dummies
data['ED_LTHS'] = (data['EDUC'] < 6).astype(int)  # Less than high school
data['ED_HS'] = (data['EDUC'] == 6).astype(int)   # High school
data['ED_SOMECOL'] = ((data['EDUC'] > 6) & (data['EDUC'] < 10)).astype(int)  # Some college
data['ED_BA_PLUS'] = (data['EDUC'] >= 10).astype(int)  # BA or higher

# Number of children
data['HAS_CHILDREN'] = (data['NCHILD'] > 0).astype(int)

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + AGE_SQ + MARRIED + ED_HS + ED_SOMECOL + ED_BA_PLUS + NCHILD', data=data).fit()
print(model2.summary())

# Model 3: DiD with state fixed effects
print("\n--- Model 3: DiD with Year and State Fixed Effects ---")
model3 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP)', data=data).fit(cov_type='HC1')
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model3.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared: {model3.rsquared:.4f}")
print(f"N: {model3.nobs:.0f}")

# Model 4: Full model with all controls
print("\n--- Model 4: Full Model with Controls, Year and State FE ---")
model4 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + AGE + AGE_SQ + MARRIED + ED_HS + ED_SOMECOL + ED_BA_PLUS + NCHILD + C(YEAR) + C(STATEFIP)', data=data).fit(cov_type='HC1')
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {model4.nobs:.0f}")

# =============================================================================
# 4. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("4. ROBUSTNESS CHECKS")
print("="*80)

# 4.1 Weighted regression using person weights
print("\n--- 4.1 Weighted Regression (using PERWT) ---")
model_weighted = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + AGE_SQ + MARRIED + ED_HS + ED_SOMECOL + ED_BA_PLUS + NCHILD',
                         data=data, weights=data['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model_weighted.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model_weighted.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_weighted.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_weighted.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# 4.2 By sex subgroups
print("\n--- 4.2 Subgroup Analysis by Sex ---")
data_male = data[data['FEMALE'] == 0]
data_female = data[data['FEMALE'] == 1]

model_male = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data_male).fit(cov_type='HC1')
model_female = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data_female).fit(cov_type='HC1')

print(f"Male DiD: {model_male.params['ELIGIBLE_AFTER']:.4f} (SE: {model_male.bse['ELIGIBLE_AFTER']:.4f})")
print(f"Female DiD: {model_female.params['ELIGIBLE_AFTER']:.4f} (SE: {model_female.bse['ELIGIBLE_AFTER']:.4f})")

# 4.3 By education
print("\n--- 4.3 Subgroup Analysis by Education ---")
data_no_hs = data[data['ED_LTHS'] == 1]
data_hs_plus = data[data['ED_LTHS'] == 0]

model_no_hs = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data_no_hs).fit(cov_type='HC1')
model_hs_plus = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data_hs_plus).fit(cov_type='HC1')

print(f"Less than HS DiD: {model_no_hs.params['ELIGIBLE_AFTER']:.4f} (SE: {model_no_hs.bse['ELIGIBLE_AFTER']:.4f})")
print(f"HS or more DiD: {model_hs_plus.params['ELIGIBLE_AFTER']:.4f} (SE: {model_hs_plus.bse['ELIGIBLE_AFTER']:.4f})")

# 4.4 Event study / Pre-trends test
print("\n--- 4.4 Event Study / Pre-Trends Analysis ---")
# Create year dummies interacted with eligible
for year in data['YEAR'].unique():
    data[f'YEAR_{year}'] = (data['YEAR'] == year).astype(int)
    data[f'ELIGIBLE_YEAR_{year}'] = data['ELIGIBLE'] * data[f'YEAR_{year}']

# Use 2011 as reference year (last pre-treatment year)
year_vars = [f'ELIGIBLE_YEAR_{y}' for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]]
formula_event = 'FT ~ ELIGIBLE + ' + ' + '.join(year_vars) + ' + C(YEAR)'
model_event = smf.ols(formula_event, data=data).fit(cov_type='HC1')

print("Event Study Coefficients (ELIGIBLE x Year interactions):")
print("(Reference year: 2011)")
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'ELIGIBLE_YEAR_{year}'
    print(f"  {year}: {model_event.params[var]:.4f} (SE: {model_event.bse[var]:.4f}, p: {model_event.pvalues[var]:.4f})")

# =============================================================================
# 5. ADDITIONAL ANALYSES
# =============================================================================
print("\n" + "="*80)
print("5. ADDITIONAL ANALYSES")
print("="*80)

# 5.1 State policy interactions
print("\n--- 5.1 State Policy Analysis ---")
# Driver's licenses for undocumented
if 'DRIVERSLICENSES' in data.columns:
    data['DL_ELIGIBLE_AFTER'] = data['DRIVERSLICENSES'] * data['ELIGIBLE_AFTER']
    model_dl = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + DRIVERSLICENSES + DL_ELIGIBLE_AFTER', data=data).fit(cov_type='HC1')
    print(f"Main DiD effect: {model_dl.params['ELIGIBLE_AFTER']:.4f}")
    print(f"Driver's License x DiD: {model_dl.params['DL_ELIGIBLE_AFTER']:.4f}")

# 5.2 Parallel trends visualization data
print("\n--- 5.2 Parallel Trends Data ---")
trends = data.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
trends.columns = ['Control (31-35)', 'Treatment (26-30)']
print(trends)

# =============================================================================
# 6. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "="*80)
print("6. SUMMARY OF MAIN RESULTS")
print("="*80)

print("\n--- Main Specification Results ---")
print(f"Sample Size: {len(data)}")
print(f"Treatment Group (ELIGIBLE=1): {len(data[data['ELIGIBLE']==1])}")
print(f"Control Group (ELIGIBLE=0): {len(data[data['ELIGIBLE']==0])}")
print(f"\nPre-DACA period: 2008-2011")
print(f"Post-DACA period: 2013-2016")
print(f"\nDependent Variable: Full-time employment (FT)")
print(f"Baseline FT rate (Treatment, Pre): {ft_treat_pre:.4f}")
print(f"Baseline FT rate (Control, Pre): {ft_ctrl_pre:.4f}")

print("\n--- DiD Estimates Across Specifications ---")
print(f"Model 1 (Basic):              {model1.params['ELIGIBLE_AFTER']:.4f} (SE: {model1.bse['ELIGIBLE_AFTER']:.4f})")
print(f"Model 2 (Demographic):        {model2.params['ELIGIBLE_AFTER']:.4f} (SE: {model2.bse['ELIGIBLE_AFTER']:.4f})")
print(f"Model 3 (Year + State FE):    {model3.params['ELIGIBLE_AFTER']:.4f} (SE: {model3.bse['ELIGIBLE_AFTER']:.4f})")
print(f"Model 4 (Full):               {model4.params['ELIGIBLE_AFTER']:.4f} (SE: {model4.bse['ELIGIBLE_AFTER']:.4f})")
print(f"Model Weighted:               {model_weighted.params['ELIGIBLE_AFTER']:.4f} (SE: {model_weighted.bse['ELIGIBLE_AFTER']:.4f})")

# Save key results to CSV
results_summary = {
    'Model': ['Basic DiD', 'With Demographics', 'With Year+State FE', 'Full Model', 'Weighted'],
    'Coefficient': [model1.params['ELIGIBLE_AFTER'], model2.params['ELIGIBLE_AFTER'],
                    model3.params['ELIGIBLE_AFTER'], model4.params['ELIGIBLE_AFTER'],
                    model_weighted.params['ELIGIBLE_AFTER']],
    'SE': [model1.bse['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'],
           model3.bse['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'],
           model_weighted.bse['ELIGIBLE_AFTER']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model_weighted.nobs)]
}
results_df = pd.DataFrame(results_summary)
results_df.to_csv('results_summary.csv', index=False)
print("\nResults saved to results_summary.csv")

# Save event study coefficients
event_study_results = []
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    if year == 2011:
        event_study_results.append({'Year': year, 'Coefficient': 0, 'SE': 0, 'p_value': 1})
    else:
        var = f'ELIGIBLE_YEAR_{year}'
        event_study_results.append({
            'Year': year,
            'Coefficient': model_event.params[var],
            'SE': model_event.bse[var],
            'p_value': model_event.pvalues[var]
        })
event_df = pd.DataFrame(event_study_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save trends data
trends.to_csv('parallel_trends_data.csv')
print("Parallel trends data saved to parallel_trends_data.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
