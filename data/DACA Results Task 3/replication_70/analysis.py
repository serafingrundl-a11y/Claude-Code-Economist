"""
DACA Replication Analysis - Difference-in-Differences Estimation
================================================================
Research Question: What was the causal impact of DACA eligibility on
full-time employment among Hispanic-Mexican Mexican-born individuals?

Treatment Group: Ages 26-30 at time of policy (ELIGIBLE=1)
Control Group: Ages 31-35 at time of policy (ELIGIBLE=0)
Pre-period: 2008-2011 (AFTER=0)
Post-period: 2013-2016 (AFTER=1)
Outcome: FT (full-time employment)
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
print("DACA REPLICATION ANALYSIS - DIFFERENCE-IN-DIFFERENCES ESTIMATION")
print("="*80)

# Load data
print("\n1. LOADING DATA")
print("-"*40)
data_path = "data/prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)
print(f"Loaded {len(df):,} observations")

# Examine key variables
print("\n2. DATA EXPLORATION")
print("-"*40)

print("\nYears in data:")
print(df['YEAR'].value_counts().sort_index())

print("\nELIGIBLE (Treatment indicator):")
print(df['ELIGIBLE'].value_counts())

print("\nAFTER (Post-treatment indicator):")
print(df['AFTER'].value_counts())

print("\nFT (Full-time employment outcome):")
print(df['FT'].value_counts())

print("\nAge distribution:")
print(df['AGE'].describe())

print("\nAge at June 2012 distribution:")
print(df['AGE_IN_JUNE_2012'].describe())

# Check SEX variable coding (1=Male, 2=Female per IPUMS)
print("\nSEX variable:")
print(df['SEX'].value_counts())

# Create analysis sample verification
print("\n3. SAMPLE VERIFICATION")
print("-"*40)

# Verify treatment and control groups by age
print("\nAge in June 2012 by ELIGIBLE status:")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe())

# Create 2x2 table for DiD
print("\n4. DIFFERENCE-IN-DIFFERENCES SETUP")
print("-"*40)

# Calculate means for 2x2 table
did_table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'std'])
print("\nMean Full-Time Employment by Group and Period:")
print(did_table)

# Manual DiD calculation
ft_treat_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
ft_treat_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
ft_control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
ft_control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()

print(f"\nTreatment Group (ELIGIBLE=1):")
print(f"  Pre-DACA (2008-2011): {ft_treat_pre:.4f}")
print(f"  Post-DACA (2013-2016): {ft_treat_post:.4f}")
print(f"  Change: {ft_treat_post - ft_treat_pre:.4f}")

print(f"\nControl Group (ELIGIBLE=0):")
print(f"  Pre-DACA (2008-2011): {ft_control_pre:.4f}")
print(f"  Post-DACA (2013-2016): {ft_control_post:.4f}")
print(f"  Change: {ft_control_post - ft_control_pre:.4f}")

did_estimate_manual = (ft_treat_post - ft_treat_pre) - (ft_control_post - ft_control_pre)
print(f"\nDifference-in-Differences Estimate (manual): {did_estimate_manual:.4f}")

# 5. REGRESSION ANALYSIS
print("\n5. REGRESSION ANALYSIS")
print("-"*40)

# Create interaction term
df['ELIGIBLE_x_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls)
print("\nModel 1: Basic Difference-in-Differences (No Controls)")
print("-"*60)
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER', data=df).fit(cov_type='HC1')
print(model1.summary())

# Save Model 1 results
model1_coef = model1.params['ELIGIBLE_x_AFTER']
model1_se = model1.bse['ELIGIBLE_x_AFTER']
model1_ci = model1.conf_int().loc['ELIGIBLE_x_AFTER']
model1_n = int(model1.nobs)

# Model 2: DiD with year fixed effects
print("\nModel 2: DiD with Year Fixed Effects")
print("-"*60)
model2 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_x_AFTER', data=df).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with state fixed effects
print("\nModel 3: DiD with State Fixed Effects")
print("-"*60)
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + C(STATEFIP)', data=df).fit(cov_type='HC1')
print(f"DiD coefficient: {model3.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"SE (robust): {model3.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"t-stat: {model3.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"p-value: {model3.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"R-squared: {model3.rsquared:.4f}")
print(f"N: {int(model3.nobs)}")

# Model 4: DiD with year and state fixed effects
print("\nModel 4: DiD with Year and State Fixed Effects")
print("-"*60)
model4 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_x_AFTER', data=df).fit(cov_type='HC1')
print(f"DiD coefficient: {model4.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"SE (robust): {model4.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"t-stat: {model4.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"p-value: {model4.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {int(model4.nobs)}")

# Model 5: DiD with individual-level covariates
print("\nModel 5: DiD with Individual Covariates (SEX, EDUC_RECODE, MARST)")
print("-"*60)

# Create factor variables for controls
df['MALE'] = (df['SEX'] == 1).astype(int)

model5 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_x_AFTER + MALE + C(EDUC_RECODE) + C(MARST)',
                 data=df).fit(cov_type='HC1')
print(f"DiD coefficient: {model5.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"SE (robust): {model5.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"t-stat: {model5.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"p-value: {model5.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {int(model5.nobs)}")

# Model 6: Full model with additional covariates
print("\nModel 6: Full Model (Year + State FE + Individual Covariates + Family Size)")
print("-"*60)

model6 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_x_AFTER + MALE + C(EDUC_RECODE) + C(MARST) + FAMSIZE + NCHILD',
                 data=df).fit(cov_type='HC1')
print(f"DiD coefficient: {model6.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"SE (robust): {model6.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"t-stat: {model6.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"p-value: {model6.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"R-squared: {model6.rsquared:.4f}")
print(f"N: {int(model6.nobs)}")

# Model 7: Weighted regression using PERWT
print("\nModel 7: Weighted DiD using PERWT (Survey Weights)")
print("-"*60)
model7 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_x_AFTER + MALE + C(EDUC_RECODE) + C(MARST)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model7.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"SE (robust): {model7.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"t-stat: {model7.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"p-value: {model7.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"R-squared: {model7.rsquared:.4f}")
print(f"N: {int(model7.nobs)}")

# 6. PREFERRED SPECIFICATION
print("\n" + "="*80)
print("PREFERRED SPECIFICATION: Model 5 (Year + State FE + Demographic Controls)")
print("="*80)

preferred_model = model5
print(f"\nDiD Effect (ELIGIBLE x AFTER): {preferred_model.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Robust Standard Error: {preferred_model.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% Confidence Interval: [{preferred_model.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {preferred_model.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"t-statistic: {preferred_model.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"p-value: {preferred_model.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"Sample Size: {int(preferred_model.nobs)}")
print(f"R-squared: {preferred_model.rsquared:.4f}")

# 7. ROBUSTNESS CHECKS
print("\n6. ROBUSTNESS CHECKS")
print("-"*40)

# Check by gender subgroups (not for main estimate, just for exploration)
print("\nHeterogeneity by Gender (exploratory):")
male_df = df[df['SEX']==1]
female_df = df[df['SEX']==2]

model_male = smf.ols('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_x_AFTER', data=male_df).fit(cov_type='HC1')
model_female = smf.ols('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_x_AFTER', data=female_df).fit(cov_type='HC1')

print(f"Males - DiD: {model_male.params['ELIGIBLE_x_AFTER']:.4f} (SE: {model_male.bse['ELIGIBLE_x_AFTER']:.4f}), N={int(model_male.nobs)}")
print(f"Females - DiD: {model_female.params['ELIGIBLE_x_AFTER']:.4f} (SE: {model_female.bse['ELIGIBLE_x_AFTER']:.4f}), N={int(model_female.nobs)}")

# Pre-trends check (year-by-year)
print("\nPre-Trends Analysis (Event Study):")
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Interaction with eligible (2011 as reference)
df['ELIGx2008'] = df['ELIGIBLE'] * df['YEAR_2008']
df['ELIGx2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIGx2010'] = df['ELIGIBLE'] * df['YEAR_2010']
df['ELIGx2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIGx2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIGx2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIGx2016'] = df['ELIGIBLE'] * df['YEAR_2016']

event_study = smf.ols('FT ~ ELIGIBLE + C(YEAR) + ELIGx2008 + ELIGx2009 + ELIGx2010 + ELIGx2013 + ELIGx2014 + ELIGx2015 + ELIGx2016',
                      data=df).fit(cov_type='HC1')

print("\nEvent Study Coefficients (ref: 2011):")
for year_var in ['ELIGx2008', 'ELIGx2009', 'ELIGx2010', 'ELIGx2013', 'ELIGx2014', 'ELIGx2015', 'ELIGx2016']:
    coef = event_study.params[year_var]
    se = event_study.bse[year_var]
    pval = event_study.pvalues[year_var]
    print(f"  {year_var}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# 8. SUMMARY STATISTICS
print("\n7. SUMMARY STATISTICS")
print("-"*40)

print("\nSample Size by Group and Period:")
sample_table = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
sample_table.index = ['Control (Age 31-35)', 'Treatment (Age 26-30)']
sample_table.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)']
print(sample_table)
print(f"\nTotal Sample Size: {len(df):,}")

print("\nDescriptive Statistics by Treatment Status:")
desc_vars = ['FT', 'AGE', 'MALE', 'FAMSIZE', 'NCHILD']
desc_stats = df.groupby('ELIGIBLE')[desc_vars].mean()
desc_stats.index = ['Control (Age 31-35)', 'Treatment (Age 26-30)']
print(desc_stats.round(4))

# 9. EXPORT RESULTS FOR LATEX
print("\n8. EXPORTING RESULTS")
print("-"*40)

results_dict = {
    'Model': ['Basic DiD', 'Year FE', 'State FE', 'Year+State FE', 'With Controls', 'Full Model', 'Weighted'],
    'Coefficient': [
        model1.params['ELIGIBLE_x_AFTER'],
        model2.params['ELIGIBLE_x_AFTER'],
        model3.params['ELIGIBLE_x_AFTER'],
        model4.params['ELIGIBLE_x_AFTER'],
        model5.params['ELIGIBLE_x_AFTER'],
        model6.params['ELIGIBLE_x_AFTER'],
        model7.params['ELIGIBLE_x_AFTER']
    ],
    'Std_Error': [
        model1.bse['ELIGIBLE_x_AFTER'],
        model2.bse['ELIGIBLE_x_AFTER'],
        model3.bse['ELIGIBLE_x_AFTER'],
        model4.bse['ELIGIBLE_x_AFTER'],
        model5.bse['ELIGIBLE_x_AFTER'],
        model6.bse['ELIGIBLE_x_AFTER'],
        model7.bse['ELIGIBLE_x_AFTER']
    ],
    'p_value': [
        model1.pvalues['ELIGIBLE_x_AFTER'],
        model2.pvalues['ELIGIBLE_x_AFTER'],
        model3.pvalues['ELIGIBLE_x_AFTER'],
        model4.pvalues['ELIGIBLE_x_AFTER'],
        model5.pvalues['ELIGIBLE_x_AFTER'],
        model6.pvalues['ELIGIBLE_x_AFTER'],
        model7.pvalues['ELIGIBLE_x_AFTER']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model5.nobs),
        int(model6.nobs),
        int(model7.nobs)
    ],
    'R_squared': [
        model1.rsquared,
        model2.rsquared,
        model3.rsquared,
        model4.rsquared,
        model5.rsquared,
        model6.rsquared,
        model7.rsquared
    ]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('regression_results.csv', index=False)
print("Results saved to regression_results.csv")

# Export summary stats
summary_stats = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'MALE': 'mean',
    'FAMSIZE': 'mean'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')
print("Summary statistics saved to summary_statistics.csv")

# Export event study coefficients
event_coefs = pd.DataFrame({
    'Year': ['2008', '2009', '2010', '2011 (ref)', '2013', '2014', '2015', '2016'],
    'Coefficient': [
        event_study.params['ELIGx2008'],
        event_study.params['ELIGx2009'],
        event_study.params['ELIGx2010'],
        0,
        event_study.params['ELIGx2013'],
        event_study.params['ELIGx2014'],
        event_study.params['ELIGx2015'],
        event_study.params['ELIGx2016']
    ],
    'Std_Error': [
        event_study.bse['ELIGx2008'],
        event_study.bse['ELIGx2009'],
        event_study.bse['ELIGx2010'],
        0,
        event_study.bse['ELIGx2013'],
        event_study.bse['ELIGx2014'],
        event_study.bse['ELIGx2015'],
        event_study.bse['ELIGx2016']
    ]
})
event_coefs.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Final Summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"""
PREFERRED ESTIMATE (Model 5: Year + State FE + Demographic Controls):
  - Effect of DACA on Full-Time Employment: {preferred_model.params['ELIGIBLE_x_AFTER']:.4f}
  - Interpretation: DACA eligibility {'increased' if preferred_model.params['ELIGIBLE_x_AFTER'] > 0 else 'decreased'}
    the probability of full-time employment by {abs(preferred_model.params['ELIGIBLE_x_AFTER']*100):.2f} percentage points
  - Standard Error: {preferred_model.bse['ELIGIBLE_x_AFTER']:.4f}
  - 95% CI: [{preferred_model.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {preferred_model.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]
  - p-value: {preferred_model.pvalues['ELIGIBLE_x_AFTER']:.4f}
  - Statistical Significance: {'Yes' if preferred_model.pvalues['ELIGIBLE_x_AFTER'] < 0.05 else 'No'} (at 5% level)
  - Sample Size: {int(preferred_model.nobs):,}
""")

print("Analysis complete.")
