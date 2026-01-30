"""
DACA Full-Time Employment Replication Analysis
Replication 76

This script performs a difference-in-differences analysis to estimate the causal
effect of DACA eligibility on full-time employment among Hispanic-Mexican
Mexican-born individuals in the United States.
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
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print("="*80)
print("DACA REPLICATION ANALYSIS - REPLICATION 76")
print("="*80)

# Load data
print("\n1. LOADING DATA")
print("-"*40)
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Total observations: {len(df):,}")
print(f"Total variables: {len(df.columns)}")

# Basic data summary
print("\n2. DATA SUMMARY")
print("-"*40)

print("\nYear distribution:")
year_counts = df['YEAR'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"  {year}: {count:,}")

print("\nELIGIBLE (Treatment) distribution:")
print(f"  Treatment (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum():,}")
print(f"  Control (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum():,}")

print("\nAFTER (Time Period) distribution:")
print(f"  Pre-DACA (AFTER=0): {(df['AFTER']==0).sum():,}")
print(f"  Post-DACA (AFTER=1): {(df['AFTER']==1).sum():,}")

print("\nFT (Full-Time Employment) distribution:")
print(f"  Full-time (FT=1): {(df['FT']==1).sum():,}")
print(f"  Not full-time (FT=0): {(df['FT']==0).sum():,}")

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Descriptive statistics by group
print("\n3. DESCRIPTIVE STATISTICS BY GROUP")
print("-"*40)

groups = df.groupby(['ELIGIBLE', 'AFTER'])

# Unweighted means
print("\n3a. Unweighted FT Rates:")
unweighted = groups['FT'].agg(['mean', 'std', 'count'])
unweighted.columns = ['Mean', 'Std', 'N']
print(unweighted.round(4))

# Weighted means
print("\n3b. Weighted FT Rates (using PERWT):")
def weighted_mean(group, col, weight):
    return np.average(group[col], weights=group[weight])

def weighted_std(group, col, weight):
    average = np.average(group[col], weights=group[weight])
    variance = np.average((group[col] - average)**2, weights=group[weight])
    return np.sqrt(variance)

weighted_stats = []
for (eligible, after), group in groups:
    w_mean = weighted_mean(group, 'FT', 'PERWT')
    w_std = weighted_std(group, 'FT', 'PERWT')
    n = len(group)
    total_weight = group['PERWT'].sum()
    weighted_stats.append({
        'ELIGIBLE': eligible,
        'AFTER': after,
        'Weighted_Mean': w_mean,
        'Weighted_Std': w_std,
        'N': n,
        'Total_Weight': total_weight
    })

weighted_df = pd.DataFrame(weighted_stats)
print(weighted_df.round(4))

# Calculate DiD manually
print("\n4. DIFFERENCE-IN-DIFFERENCES CALCULATION (MANUAL)")
print("-"*40)

# Get weighted means for each cell
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]

wm_pre_control = np.average(pre_control['FT'], weights=pre_control['PERWT'])
wm_post_control = np.average(post_control['FT'], weights=post_control['PERWT'])
wm_pre_treat = np.average(pre_treat['FT'], weights=pre_treat['PERWT'])
wm_post_treat = np.average(post_treat['FT'], weights=post_treat['PERWT'])

print(f"Control group (ELIGIBLE=0):")
print(f"  Pre-DACA:  {wm_pre_control:.4f}")
print(f"  Post-DACA: {wm_post_control:.4f}")
print(f"  Change:    {wm_post_control - wm_pre_control:.4f}")

print(f"\nTreatment group (ELIGIBLE=1):")
print(f"  Pre-DACA:  {wm_pre_treat:.4f}")
print(f"  Post-DACA: {wm_post_treat:.4f}")
print(f"  Change:    {wm_post_treat - wm_pre_treat:.4f}")

did_manual = (wm_post_treat - wm_pre_treat) - (wm_post_control - wm_pre_control)
print(f"\nDifference-in-Differences estimate: {did_manual:.4f}")
print(f"  ({did_manual*100:.2f} percentage points)")

# Regression Analysis
print("\n5. REGRESSION ANALYSIS")
print("-"*40)

# Model 1: Basic DiD without controls (unweighted)
print("\n5a. Model 1: Basic DiD (Unweighted)")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(f"DiD Coefficient: {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error:  {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic:     {model1.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI:          [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared:       {model1.rsquared:.4f}")
print(f"N:               {int(model1.nobs):,}")

# Model 2: Basic DiD with weights
print("\n5b. Model 2: Basic DiD (Weighted)")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error:  {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic:     {model2.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model2.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI:          [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared:       {model2.rsquared:.4f}")
print(f"N:               {int(model2.nobs):,}")

# Model 3: DiD with robust standard errors clustered by state
print("\n5c. Model 3: Basic DiD with Clustered SE (by State)")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE:    {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic:     {model3.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI:          [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Prepare variables for extended models
# Create binary variables for sex (male=1, female=0)
df['MALE'] = (df['SEX'] == 1).astype(int)

# Create age variable (centered at mean)
df['AGE_CENTERED'] = df['AGE'] - df['AGE'].mean()

# Create education dummies
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_2YR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Married indicator
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)

# Has children indicator
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Model 4: DiD with individual controls
print("\n5d. Model 4: DiD with Individual Covariates")
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + AGE_CENTERED + EDUC_HS + EDUC_SOMECOL + EDUC_2YR + EDUC_BA + MARRIED + HAS_CHILDREN',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE:    {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic:     {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI:          [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared:       {model4.rsquared:.4f}")

# Model 5: DiD with state fixed effects
print("\n5e. Model 5: DiD with State Fixed Effects")
model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE:    {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic:     {model5.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI:          [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 6: DiD with year fixed effects
print("\n5f. Model 6: DiD with Year Fixed Effects")
# Note: AFTER is collinear with year FEs, so we drop it
model6 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR)',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE:    {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic:     {model6.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI:          [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 7: Full model with all controls and fixed effects
print("\n5g. Model 7: Full Model (Covariates + State + Year FEs)")
model7 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + AGE_CENTERED + EDUC_HS + EDUC_SOMECOL + EDUC_2YR + EDUC_BA + MARRIED + HAS_CHILDREN + C(STATEFIP) + C(YEAR)',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE:    {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic:     {model7.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:         {model7.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI:          [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared:       {model7.rsquared:.4f}")

# Print full model 7 summary for key variables
print("\nFull Model 7 Coefficients (selected):")
key_vars = ['Intercept', 'ELIGIBLE', 'ELIGIBLE_AFTER', 'MALE', 'AGE_CENTERED',
            'EDUC_HS', 'EDUC_SOMECOL', 'EDUC_2YR', 'EDUC_BA', 'MARRIED', 'HAS_CHILDREN']
for var in key_vars:
    if var in model7.params.index:
        print(f"  {var:20s}: {model7.params[var]:8.4f} (SE: {model7.bse[var]:.4f}, p: {model7.pvalues[var]:.4f})")

# Pre-trend analysis
print("\n6. PARALLEL TRENDS ANALYSIS")
print("-"*40)

# Year-by-year FT rates
print("\n6a. Year-by-Year FT Rates by Group (Weighted):")
yearly_rates = []
for year in sorted(df['YEAR'].unique()):
    for eligible in [0, 1]:
        subset = df[(df['YEAR']==year) & (df['ELIGIBLE']==eligible)]
        if len(subset) > 0:
            wm = np.average(subset['FT'], weights=subset['PERWT'])
            yearly_rates.append({'Year': year, 'ELIGIBLE': eligible, 'FT_Rate': wm, 'N': len(subset)})

yearly_df = pd.DataFrame(yearly_rates)
yearly_pivot = yearly_df.pivot(index='Year', columns='ELIGIBLE', values='FT_Rate')
yearly_pivot.columns = ['Control', 'Treatment']
yearly_pivot['Difference'] = yearly_pivot['Treatment'] - yearly_pivot['Control']
print(yearly_pivot.round(4))

# Event study
print("\n6b. Event Study (Year-Specific Treatment Effects):")
# Create year dummies and interactions with ELIGIBLE
for year in df['YEAR'].unique():
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Run event study regression (base year: 2011)
year_vars = ' + '.join([f'YEAR_{y}' for y in sorted(df['YEAR'].unique()) if y != 2011])
interact_vars = ' + '.join([f'ELIGIBLE_YEAR_{y}' for y in sorted(df['YEAR'].unique()) if y != 2011])
formula_event = f'FT ~ ELIGIBLE + {year_vars} + {interact_vars}'

model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("\nEvent Study Coefficients (relative to 2011):")
print(f"{'Year':<8} {'Coefficient':>12} {'SE':>10} {'95% CI':>24}")
print("-" * 54)
for year in sorted(df['YEAR'].unique()):
    if year != 2011:
        var = f'ELIGIBLE_YEAR_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        ci_low, ci_high = model_event.conf_int().loc[var]
        period = "Pre" if year < 2012 else "Post"
        print(f"{year} ({period})  {coef:12.4f} {se:10.4f}   [{ci_low:8.4f}, {ci_high:8.4f}]")

# Heterogeneity Analysis
print("\n7. HETEROGENEITY ANALYSIS")
print("-"*40)

# By sex
print("\n7a. By Sex:")
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex_val]
    model_sex = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=subset, weights=subset['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': subset['STATEFIP']})
    print(f"  {sex_label}: DiD = {model_sex.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sex.bse['ELIGIBLE_AFTER']:.4f}, p: {model_sex.pvalues['ELIGIBLE_AFTER']:.4f}), N = {len(subset):,}")

# By education
print("\n7b. By Education:")
for educ in df['EDUC_RECODE'].unique():
    subset = df[df['EDUC_RECODE'] == educ]
    if len(subset) > 100:  # Only if sufficient sample
        model_educ = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
        print(f"  {educ}: DiD = {model_educ.params['ELIGIBLE_AFTER']:.4f} (SE: {model_educ.bse['ELIGIBLE_AFTER']:.4f}, p: {model_educ.pvalues['ELIGIBLE_AFTER']:.4f}), N = {len(subset):,}")

# Summary table
print("\n8. SUMMARY OF MAIN RESULTS")
print("-"*40)
print("\nModel Comparison Table:")
print(f"{'Model':<50} {'DiD Coef':>10} {'SE':>10} {'p-value':>10}")
print("-" * 80)
print(f"{'1. Basic (Unweighted)':<50} {model1.params['ELIGIBLE_AFTER']:>10.4f} {model1.bse['ELIGIBLE_AFTER']:>10.4f} {model1.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'2. Basic (Weighted)':<50} {model2.params['ELIGIBLE_AFTER']:>10.4f} {model2.bse['ELIGIBLE_AFTER']:>10.4f} {model2.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'3. Weighted + Clustered SE':<50} {model3.params['ELIGIBLE_AFTER']:>10.4f} {model3.bse['ELIGIBLE_AFTER']:>10.4f} {model3.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'4. + Individual Covariates':<50} {model4.params['ELIGIBLE_AFTER']:>10.4f} {model4.bse['ELIGIBLE_AFTER']:>10.4f} {model4.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'5. + State FEs':<50} {model5.params['ELIGIBLE_AFTER']:>10.4f} {model5.bse['ELIGIBLE_AFTER']:>10.4f} {model5.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'6. + Year FEs':<50} {model6.params['ELIGIBLE_AFTER']:>10.4f} {model6.bse['ELIGIBLE_AFTER']:>10.4f} {model6.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'7. Full Model (Covariates + State + Year FEs)':<50} {model7.params['ELIGIBLE_AFTER']:>10.4f} {model7.bse['ELIGIBLE_AFTER']:>10.4f} {model7.pvalues['ELIGIBLE_AFTER']:>10.4f}")

# Preferred estimate
print("\n" + "="*80)
print("PREFERRED ESTIMATE (Model 7: Full Model)")
print("="*80)
print(f"Effect Size:      {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"                  ({model7.params['ELIGIBLE_AFTER']*100:.2f} percentage points)")
print(f"Standard Error:   {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI:           [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"t-statistic:      {model7.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value:          {model7.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample Size:      {int(model7.nobs):,}")
print("="*80)

# Save results to file
results = {
    'preferred_effect': model7.params['ELIGIBLE_AFTER'],
    'preferred_se': model7.bse['ELIGIBLE_AFTER'],
    'preferred_ci_low': model7.conf_int().loc['ELIGIBLE_AFTER', 0],
    'preferred_ci_high': model7.conf_int().loc['ELIGIBLE_AFTER', 1],
    'preferred_pvalue': model7.pvalues['ELIGIBLE_AFTER'],
    'sample_size': int(model7.nobs)
}

print("\nAnalysis complete.")
