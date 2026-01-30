"""
DACA Replication Analysis - Replication 55
============================================
Research Question: Effect of DACA eligibility on full-time employment
Method: Difference-in-Differences
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
print("DACA REPLICATION ANALYSIS - STUDY 55")
print("="*80)

# ============================================================================
# STEP 1: Load and Explore Data
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("="*80)

# Load the data
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_55\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"\nDataset dimensions: {df.shape[0]:,} observations, {df.shape[1]} variables")

# Check key variables
print("\n--- Key Variable Summary ---")
print(f"\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

print(f"\nELIGIBLE distribution (1=Treatment ages 26-30, 0=Control ages 31-35):")
print(df['ELIGIBLE'].value_counts())

print(f"\nAFTER distribution (1=Post-DACA 2013-2016, 0=Pre-DACA 2008-2011):")
print(df['AFTER'].value_counts())

print(f"\nFT (Full-time employment) distribution:")
print(df['FT'].value_counts())

print(f"\nFT rate overall: {df['FT'].mean()*100:.2f}%")

# Check AGE_IN_JUNE_2012 to confirm eligible/control groups
print("\n--- Age at Policy Time by ELIGIBLE Status ---")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe())

# ============================================================================
# STEP 2: Descriptive Statistics - 2x2 Table
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DESCRIPTIVE STATISTICS")
print("="*80)

# Unweighted means
print("\n--- Unweighted Full-Time Employment Rates ---")
means_table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'std'])
print(means_table)

# Calculate simple DiD manually
ft_treat_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
ft_treat_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
ft_ctrl_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
ft_ctrl_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()

print(f"\n--- Simple DiD Calculation (Unweighted) ---")
print(f"Treatment group (Eligible, ages 26-30):")
print(f"  Pre-DACA (2008-2011):  {ft_treat_pre*100:.2f}%")
print(f"  Post-DACA (2013-2016): {ft_treat_post*100:.2f}%")
print(f"  Change:                {(ft_treat_post-ft_treat_pre)*100:.2f} pp")

print(f"\nControl group (Non-eligible, ages 31-35):")
print(f"  Pre-DACA (2008-2011):  {ft_ctrl_pre*100:.2f}%")
print(f"  Post-DACA (2013-2016): {ft_ctrl_post*100:.2f}%")
print(f"  Change:                {(ft_ctrl_post-ft_ctrl_pre)*100:.2f} pp")

did_simple = (ft_treat_post - ft_treat_pre) - (ft_ctrl_post - ft_ctrl_pre)
print(f"\nDifference-in-Differences: {did_simple*100:.3f} percentage points")

# Weighted means
print("\n--- Weighted Full-Time Employment Rates ---")
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

def weighted_stats(group):
    wm = np.average(group['FT'], weights=group['PERWT'])
    n = len(group)
    return pd.Series({'weighted_mean': wm, 'n': n})

weighted_means = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_stats)
print(weighted_means)

# Weighted DiD
wft_treat_post = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'],
                            weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['PERWT'])
wft_treat_pre = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'],
                           weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['PERWT'])
wft_ctrl_post = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'],
                           weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['PERWT'])
wft_ctrl_pre = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'],
                          weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['PERWT'])

print(f"\n--- Simple DiD Calculation (Weighted) ---")
print(f"Treatment group (Eligible, ages 26-30):")
print(f"  Pre-DACA (2008-2011):  {wft_treat_pre*100:.2f}%")
print(f"  Post-DACA (2013-2016): {wft_treat_post*100:.2f}%")
print(f"  Change:                {(wft_treat_post-wft_treat_pre)*100:.2f} pp")

print(f"\nControl group (Non-eligible, ages 31-35):")
print(f"  Pre-DACA (2008-2011):  {wft_ctrl_pre*100:.2f}%")
print(f"  Post-DACA (2013-2016): {wft_ctrl_post*100:.2f}%")
print(f"  Change:                {(wft_ctrl_post-wft_ctrl_pre)*100:.2f} pp")

did_weighted = (wft_treat_post - wft_treat_pre) - (wft_ctrl_post - wft_ctrl_pre)
print(f"\nDifference-in-Differences (Weighted): {did_weighted*100:.3f} percentage points")

# ============================================================================
# STEP 3: Covariate Balance Check
# ============================================================================
print("\n" + "="*80)
print("STEP 3: COVARIATE BALANCE CHECK")
print("="*80)

# Check balance between treatment and control groups pre-treatment
pre_data = df[df['AFTER'] == 0].copy()

covariates = ['SEX', 'FAMSIZE', 'NCHILD', 'MARST']
print("\n--- Pre-Treatment Covariate Balance ---")
for cov in covariates:
    if cov in pre_data.columns:
        treat_mean = pre_data[pre_data['ELIGIBLE']==1][cov].mean()
        ctrl_mean = pre_data[pre_data['ELIGIBLE']==0][cov].mean()
        print(f"{cov}: Treatment={treat_mean:.3f}, Control={ctrl_mean:.3f}, Diff={treat_mean-ctrl_mean:.3f}")

# Check education distribution
if 'EDUC_RECODE' in df.columns:
    print("\n--- Education Distribution by Treatment Status (Pre-period) ---")
    print(pd.crosstab(pre_data['ELIGIBLE'], pre_data['EDUC_RECODE'], normalize='index'))

# ============================================================================
# STEP 4: REGRESSION ANALYSIS - Difference-in-Differences
# ============================================================================
print("\n" + "="*80)
print("STEP 4: REGRESSION ANALYSIS")
print("="*80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (OLS, No Controls) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])

# Model 2: Basic DiD with weights
print("\n--- Model 2: Basic DiD (WLS, Person Weights) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with demographic controls
print("\n--- Model 3: DiD with Demographic Controls ---")
# Prepare categorical variables
df['SEX_cat'] = pd.Categorical(df['SEX'])
df['MARST_cat'] = pd.Categorical(df['MARST'])

# Check available education variable
if 'EDUC_RECODE' in df.columns:
    df['EDUC_cat'] = pd.Categorical(df['EDUC_RECODE'])
    model3_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(SEX) + C(MARST) + FAMSIZE + NCHILD + C(EDUC_RECODE)'
else:
    model3_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(SEX) + C(MARST) + FAMSIZE + NCHILD + EDUC'

model3 = smf.wls(model3_formula, data=df, weights=df['PERWT']).fit()
print(f"DiD Estimate (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.5f}")
print(f"Std Error: {model3.bse['ELIGIBLE_AFTER']:.5f}")
print(f"t-stat: {model3.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with State Fixed Effects ---")
model4_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)'
model4 = smf.wls(model4_formula, data=df, weights=df['PERWT']).fit()
print(f"DiD Estimate (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.5f}")
print(f"Std Error: {model4.bse['ELIGIBLE_AFTER']:.5f}")
print(f"t-stat: {model4.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")

# Model 5: DiD with state and year fixed effects + controls
print("\n--- Model 5: Full Model (State + Year FE + Controls) ---")
model5_formula = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(STATEFIP) + C(YEAR) + C(SEX) + C(MARST) + FAMSIZE + NCHILD'
model5 = smf.wls(model5_formula, data=df, weights=df['PERWT']).fit()
print(f"DiD Estimate (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.5f}")
print(f"Std Error: {model5.bse['ELIGIBLE_AFTER']:.5f}")
print(f"t-stat: {model5.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")

# ============================================================================
# STEP 5: Clustered Standard Errors
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CLUSTERED STANDARD ERRORS")
print("="*80)

# Model with state-clustered standard errors (preferred specification)
print("\n--- Preferred Model: DiD with State-Clustered SE ---")
model_cluster = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(SEX) + C(MARST) + FAMSIZE + NCHILD',
                        data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Estimate (ELIGIBLE_AFTER): {model_cluster.params['ELIGIBLE_AFTER']:.5f}")
print(f"Clustered Std Error: {model_cluster.bse['ELIGIBLE_AFTER']:.5f}")
print(f"t-stat: {model_cluster.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"p-value: {model_cluster.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model_cluster.conf_int().loc['ELIGIBLE_AFTER', 0]:.5f}, {model_cluster.conf_int().loc['ELIGIBLE_AFTER', 1]:.5f}]")

# ============================================================================
# STEP 6: Parallel Trends Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 6: PARALLEL TRENDS ANALYSIS")
print("="*80)

# Year-by-year treatment effects (event study)
print("\n--- Pre-Treatment Trends by Year ---")
pre_period = df[df['AFTER'] == 0].copy()

for year in sorted(pre_period['YEAR'].unique()):
    year_data = pre_period[pre_period['YEAR'] == year]
    treat_ft = np.average(year_data[year_data['ELIGIBLE']==1]['FT'],
                          weights=year_data[year_data['ELIGIBLE']==1]['PERWT'])
    ctrl_ft = np.average(year_data[year_data['ELIGIBLE']==0]['FT'],
                         weights=year_data[year_data['ELIGIBLE']==0]['PERWT'])
    print(f"{year}: Treatment FT={treat_ft*100:.2f}%, Control FT={ctrl_ft*100:.2f}%, Gap={((treat_ft-ctrl_ft)*100):.2f} pp")

print("\n--- Post-Treatment Trends by Year ---")
post_period = df[df['AFTER'] == 1].copy()

for year in sorted(post_period['YEAR'].unique()):
    year_data = post_period[post_period['YEAR'] == year]
    treat_ft = np.average(year_data[year_data['ELIGIBLE']==1]['FT'],
                          weights=year_data[year_data['ELIGIBLE']==1]['PERWT'])
    ctrl_ft = np.average(year_data[year_data['ELIGIBLE']==0]['FT'],
                         weights=year_data[year_data['ELIGIBLE']==0]['PERWT'])
    print(f"{year}: Treatment FT={treat_ft*100:.2f}%, Control FT={ctrl_ft*100:.2f}%, Gap={((treat_ft-ctrl_ft)*100):.2f} pp")

# Event study regression
print("\n--- Event Study Regression ---")
# Create year dummies interacted with treatment
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Use 2011 as reference year
df['ELIG_2008'] = df['ELIGIBLE'] * df['YEAR_2008']
df['ELIG_2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIG_2010'] = df['ELIGIBLE'] * df['YEAR_2010']
df['ELIG_2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIG_2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIG_2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIG_2016'] = df['ELIGIBLE'] * df['YEAR_2016']

event_formula = 'FT ~ ELIGIBLE + C(YEAR) + ELIG_2008 + ELIG_2009 + ELIG_2010 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016'
event_model = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("Year-specific treatment effects (reference: 2011):")
for var in ['ELIG_2008', 'ELIG_2009', 'ELIG_2010', 'ELIG_2013', 'ELIG_2014', 'ELIG_2015', 'ELIG_2016']:
    coef = event_model.params[var]
    se = event_model.bse[var]
    pval = event_model.pvalues[var]
    ci = event_model.conf_int().loc[var]
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"  {var}: {coef:.4f} ({se:.4f}){stars} [95% CI: {ci[0]:.4f}, {ci[1]:.4f}]")

# ============================================================================
# STEP 7: Placebo Test
# ============================================================================
print("\n" + "="*80)
print("STEP 7: PLACEBO TEST")
print("="*80)

# Use only pre-treatment data and create fake treatment at 2010
pre_only = df[df['AFTER'] == 0].copy()
pre_only['FAKE_AFTER'] = (pre_only['YEAR'] >= 2010).astype(int)
pre_only['FAKE_INTERACTION'] = pre_only['ELIGIBLE'] * pre_only['FAKE_AFTER']

placebo_model = smf.wls('FT ~ ELIGIBLE + FAKE_AFTER + FAKE_INTERACTION',
                        data=pre_only, weights=pre_only['PERWT']).fit(
                            cov_type='cluster', cov_kwds={'groups': pre_only['STATEFIP']})
print(f"\nPlacebo DiD (fake treatment at 2010):")
print(f"Coefficient: {placebo_model.params['FAKE_INTERACTION']:.5f}")
print(f"Std Error: {placebo_model.bse['FAKE_INTERACTION']:.5f}")
print(f"p-value: {placebo_model.pvalues['FAKE_INTERACTION']:.4f}")
print("(Should not be significant if parallel trends hold)")

# ============================================================================
# STEP 8: Subgroup Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 8: SUBGROUP ANALYSIS")
print("="*80)

# By sex
print("\n--- DiD by Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    sub_df = df[df['SEX'] == sex_val]
    sub_model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=sub_df, weights=sub_df['PERWT']).fit(
                            cov_type='cluster', cov_kwds={'groups': sub_df['STATEFIP']})
    print(f"{sex_name}: DiD = {sub_model.params['ELIGIBLE_AFTER']:.4f} (SE: {sub_model.bse['ELIGIBLE_AFTER']:.4f}), p = {sub_model.pvalues['ELIGIBLE_AFTER']:.4f}")

# By education
print("\n--- DiD by Education Level ---")
if 'EDUC_RECODE' in df.columns:
    for educ in df['EDUC_RECODE'].unique():
        if pd.notna(educ):
            sub_df = df[df['EDUC_RECODE'] == educ]
            if len(sub_df) > 100:  # Only if sufficient sample
                try:
                    sub_model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                                        data=sub_df, weights=sub_df['PERWT']).fit()
                    print(f"{educ}: DiD = {sub_model.params['ELIGIBLE_AFTER']:.4f} (SE: {sub_model.bse['ELIGIBLE_AFTER']:.4f}), n = {len(sub_df)}")
                except:
                    print(f"{educ}: Could not estimate (insufficient variation)")

# ============================================================================
# STEP 9: Summary Table
# ============================================================================
print("\n" + "="*80)
print("STEP 9: SUMMARY OF RESULTS")
print("="*80)

print("\n" + "-"*80)
print("REGRESSION RESULTS SUMMARY")
print("-"*80)
print(f"{'Model':<40} {'Estimate':>12} {'SE':>10} {'p-value':>10}")
print("-"*80)

# Re-run all models for clean summary
models_summary = []

# Model 1: Basic OLS
m1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
models_summary.append(('Basic OLS (no controls)', m1.params['ELIGIBLE_AFTER'], m1.bse['ELIGIBLE_AFTER'], m1.pvalues['ELIGIBLE_AFTER'], m1.nobs))

# Model 2: Basic WLS
m2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
models_summary.append(('Basic WLS (person weights)', m2.params['ELIGIBLE_AFTER'], m2.bse['ELIGIBLE_AFTER'], m2.pvalues['ELIGIBLE_AFTER'], m2.nobs))

# Model 3: WLS with controls
m3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(SEX) + C(MARST) + FAMSIZE + NCHILD',
             data=df, weights=df['PERWT']).fit()
models_summary.append(('WLS + Demographics', m3.params['ELIGIBLE_AFTER'], m3.bse['ELIGIBLE_AFTER'], m3.pvalues['ELIGIBLE_AFTER'], m3.nobs))

# Model 4: WLS with state FE
m4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)', data=df, weights=df['PERWT']).fit()
models_summary.append(('WLS + State FE', m4.params['ELIGIBLE_AFTER'], m4.bse['ELIGIBLE_AFTER'], m4.pvalues['ELIGIBLE_AFTER'], m4.nobs))

# Model 5: Full model with clustered SE (PREFERRED)
m5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(SEX) + C(MARST) + FAMSIZE + NCHILD + C(STATEFIP)',
             data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
models_summary.append(('Full + Clustered SE (PREFERRED)', m5.params['ELIGIBLE_AFTER'], m5.bse['ELIGIBLE_AFTER'], m5.pvalues['ELIGIBLE_AFTER'], m5.nobs))

for name, est, se, pval, n in models_summary:
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"{name:<40} {est:>11.5f}{stars} {se:>10.5f} {pval:>10.4f}")

print("-"*80)
print(f"Sample size: {int(models_summary[0][4]):,}")
print("*** p<0.01, ** p<0.05, * p<0.1")

# ============================================================================
# PREFERRED ESTIMATE
# ============================================================================
print("\n" + "="*80)
print("PREFERRED ESTIMATE")
print("="*80)
preferred_est = m5.params['ELIGIBLE_AFTER']
preferred_se = m5.bse['ELIGIBLE_AFTER']
preferred_ci = m5.conf_int().loc['ELIGIBLE_AFTER']
preferred_pval = m5.pvalues['ELIGIBLE_AFTER']

print(f"\nDifference-in-Differences Estimate of DACA Effect on Full-Time Employment:")
print(f"  Effect Size:     {preferred_est:.5f} ({preferred_est*100:.3f} percentage points)")
print(f"  Standard Error:  {preferred_se:.5f} (clustered by state)")
print(f"  95% CI:          [{preferred_ci[0]:.5f}, {preferred_ci[1]:.5f}]")
print(f"  p-value:         {preferred_pval:.4f}")
print(f"  Sample Size:     {int(m5.nobs):,}")

if preferred_pval < 0.05:
    print(f"\n  INTERPRETATION: DACA eligibility is associated with a statistically significant")
    if preferred_est > 0:
        print(f"  INCREASE of {abs(preferred_est*100):.2f} percentage points in full-time employment.")
    else:
        print(f"  DECREASE of {abs(preferred_est*100):.2f} percentage points in full-time employment.")
else:
    print(f"\n  INTERPRETATION: The effect is not statistically significant at the 5% level.")
    print(f"  We cannot reject the null hypothesis of no effect.")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save key results to a file for the LaTeX report
results_dict = {
    'n_obs': int(m5.nobs),
    'n_treatment': int(df['ELIGIBLE'].sum()),
    'n_control': int((df['ELIGIBLE']==0).sum()),
    'n_pre': int((df['AFTER']==0).sum()),
    'n_post': int((df['AFTER']==1).sum()),
    'ft_treat_pre': wft_treat_pre,
    'ft_treat_post': wft_treat_post,
    'ft_ctrl_pre': wft_ctrl_pre,
    'ft_ctrl_post': wft_ctrl_post,
    'did_simple': did_weighted,
    'preferred_estimate': preferred_est,
    'preferred_se': preferred_se,
    'preferred_ci_low': preferred_ci[0],
    'preferred_ci_high': preferred_ci[1],
    'preferred_pval': preferred_pval,
    'models': models_summary
}

# Save for LaTeX report
import json
with open(r'C:\Users\seraf\DACA Results Task 3\replication_55\results.json', 'w') as f:
    # Convert to serializable format
    results_serializable = {k: v if not isinstance(v, (np.floating, np.integer)) else float(v)
                           for k, v in results_dict.items() if k != 'models'}
    results_serializable['models'] = [(n, float(e), float(s), float(p), int(obs))
                                      for n, e, s, p, obs in models_summary]
    json.dump(results_serializable, f, indent=2)

print("Results saved to results.json")

# ============================================================================
# ADDITIONAL DATA FOR REPORT
# ============================================================================
print("\n" + "="*80)
print("ADDITIONAL STATISTICS FOR REPORT")
print("="*80)

# Demographics summary
print("\n--- Sample Demographics ---")
print(f"Total observations: {len(df):,}")
print(f"Treatment group (ELIGIBLE=1): {df['ELIGIBLE'].sum():,} ({df['ELIGIBLE'].mean()*100:.1f}%)")
print(f"Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum():,} ({(df['ELIGIBLE']==0).mean()*100:.1f}%)")
print(f"Pre-period (AFTER=0): {(df['AFTER']==0).sum():,}")
print(f"Post-period (AFTER=1): {(df['AFTER']==1).sum():,}")

print(f"\n--- Sex Distribution ---")
sex_dist = df.groupby('SEX').size()
print(f"Male (1): {sex_dist.get(1, 0):,} ({sex_dist.get(1, 0)/len(df)*100:.1f}%)")
print(f"Female (2): {sex_dist.get(2, 0):,} ({sex_dist.get(2, 0)/len(df)*100:.1f}%)")

print(f"\n--- Years in Sample ---")
print(df['YEAR'].value_counts().sort_index())

print(f"\n--- State Distribution (Top 10) ---")
state_counts = df.groupby('STATEFIP').size().sort_values(ascending=False).head(10)
print(state_counts)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
