"""
DACA Replication Study - Analysis Script
Replication 73

Research Question: Impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US.

Treatment: DACA eligibility (ages 26-30 at June 15, 2012)
Control: Ages 31-35 at June 15, 2012
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
np.random.seed(73)

print("="*70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. LOADING DATA...")
print("-"*50)

# Load the numeric version for analysis
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_73\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"Number of observations: {len(df):,}")
print(f"Number of variables: {len(df.columns)}")

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================
print("\n2. DATA EXPLORATION...")
print("-"*50)

# Check key variables
print("\nKey variables summary:")
print(f"  ELIGIBLE: {df['ELIGIBLE'].value_counts().to_dict()}")
print(f"  AFTER: {df['AFTER'].value_counts().to_dict()}")
print(f"  FT: {df['FT'].value_counts().to_dict()}")

# Year distribution
print("\nYear distribution:")
print(df['YEAR'].value_counts().sort_index())

# Verify 2012 is excluded
assert 2012 not in df['YEAR'].values, "Error: 2012 should be excluded"
print("\nConfirmed: 2012 is excluded from the data")

# Check PERWT
print(f"\nPERWT (person weights) summary:")
print(f"  Min: {df['PERWT'].min():.2f}")
print(f"  Max: {df['PERWT'].max():.2f}")
print(f"  Mean: {df['PERWT'].mean():.2f}")

# ============================================================================
# 3. CREATE ANALYSIS VARIABLES
# ============================================================================
print("\n3. CREATING ANALYSIS VARIABLES...")
print("-"*50)

# DiD interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

print("Created ELIGIBLE_AFTER interaction term")

# Check sample sizes by group
print("\nSample sizes by treatment/period:")
print(pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True))

# ============================================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n4. DESCRIPTIVE STATISTICS...")
print("-"*50)

# Full-time employment rates by group and period (weighted)
def weighted_mean(data, weight_col, value_col):
    """Calculate weighted mean"""
    return np.average(data[value_col], weights=data[weight_col])

def weighted_std(data, weight_col, value_col):
    """Calculate weighted standard deviation"""
    mean = weighted_mean(data, weight_col, value_col)
    variance = np.average((data[value_col] - mean)**2, weights=data[weight_col])
    return np.sqrt(variance)

# Calculate FT rates for each group
groups = df.groupby(['ELIGIBLE', 'AFTER'])
ft_rates = {}
for (eligible, after), group in groups:
    ft_rate = weighted_mean(group, 'PERWT', 'FT')
    n = len(group)
    ft_rates[(eligible, after)] = {'rate': ft_rate, 'n': n}

print("\nFull-Time Employment Rates (Weighted):")
print("-"*50)
print(f"{'Group':<25} {'Pre-DACA':<15} {'Post-DACA':<15} {'Diff':<15}")
print("-"*50)

# Control group (ELIGIBLE=0)
ctrl_pre = ft_rates[(0, 0)]['rate']
ctrl_post = ft_rates[(0, 1)]['rate']
ctrl_diff = ctrl_post - ctrl_pre

# Treatment group (ELIGIBLE=1)
treat_pre = ft_rates[(1, 0)]['rate']
treat_post = ft_rates[(1, 1)]['rate']
treat_diff = treat_post - treat_pre

print(f"{'Control (31-35)':<25} {ctrl_pre:.4f}         {ctrl_post:.4f}          {ctrl_diff:+.4f}")
print(f"{'Treatment (26-30)':<25} {treat_pre:.4f}         {treat_post:.4f}          {treat_diff:+.4f}")
print("-"*50)

# DiD estimate (simple calculation)
did_simple = treat_diff - ctrl_diff
print(f"{'DiD Estimate':<25} {'':<15} {'':<15} {did_simple:+.4f}")

print(f"\nSimple DiD effect: {did_simple:.4f} ({did_simple*100:.2f} percentage points)")

# ============================================================================
# 5. MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# ============================================================================
print("\n5. MAIN DiD REGRESSION...")
print("-"*50)

# Model 1: Basic DiD (no covariates)
print("\nModel 1: Basic DiD (no covariates)")
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

print(model1.summary().tables[1])
print(f"\nDiD Coefficient (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")

# ============================================================================
# 6. DiD WITH COVARIATES
# ============================================================================
print("\n6. DiD WITH COVARIATES...")
print("-"*50)

# Check what covariates are available
print("\nAvailable columns in dataset:")
print(df.columns.tolist())

# Model 2: DiD with demographic controls
# Create SEX binary (1=Male, 2=Female in IPUMS)
df['MALE'] = (df['SEX'] == 1).astype(int)

# Check for education recode variable
if 'EDUC_RECODE' in df.columns:
    educ_var = 'EDUC_RECODE'
else:
    educ_var = 'EDUC'

# Marital status - married indicator
df['MARRIED'] = (df['MARST'] == 1).astype(int)  # 1 = Married, spouse present

# Check for state variable
if 'STATEFIP' in df.columns:
    has_state = True
else:
    has_state = False

print("\nModel 2: DiD with demographic controls")
formula2 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + AGE + C(YEAR)'
model2 = smf.wls(formula2, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print(f"\nDiD Coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with more controls (if available)
print("\nModel 3: DiD with additional controls")

# Build formula with available variables
covariates = ['MALE', 'MARRIED', 'AGE', 'C(YEAR)']

# Add education if available
if 'EDUC' in df.columns:
    covariates.append('C(EDUC)')

# Add number of children if available
if 'NCHILD' in df.columns:
    covariates.append('NCHILD')

# Add state fixed effects if available
if 'STATEFIP' in df.columns:
    covariates.append('C(STATEFIP)')

formula3 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + ' + ' + '.join(covariates)
model3 = smf.wls(formula3, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print(f"\nDiD Coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model3.rsquared:.4f}")

# ============================================================================
# 7. PARALLEL TRENDS CHECK
# ============================================================================
print("\n7. PARALLEL TRENDS ANALYSIS...")
print("-"*50)

# Create year-by-treatment interactions for event study
pre_years = [2008, 2009, 2010, 2011]
post_years = [2013, 2014, 2015, 2016]

# Use 2011 as reference year (last pre-treatment year)
for year in pre_years + post_years:
    if year != 2011:  # Reference year
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
        df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Event study regression
event_formula = 'FT ~ ELIGIBLE'
for year in pre_years + post_years:
    if year != 2011:
        event_formula += f' + YEAR_{year} + ELIGIBLE_YEAR_{year}'

event_model = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Treatment x Year interactions):")
print("Reference Year: 2011")
print("-"*50)
event_coefs = {}
for year in pre_years + post_years:
    if year != 2011:
        coef = event_model.params[f'ELIGIBLE_YEAR_{year}']
        se = event_model.bse[f'ELIGIBLE_YEAR_{year}']
        pval = event_model.pvalues[f'ELIGIBLE_YEAR_{year}']
        event_coefs[year] = {'coef': coef, 'se': se, 'pval': pval}
        print(f"  Year {year}: {coef:+.4f} (SE: {se:.4f}, p={pval:.4f})")

# ============================================================================
# 8. ROBUSTNESS CHECKS
# ============================================================================
print("\n8. ROBUSTNESS CHECKS...")
print("-"*50)

# Check 1: Unweighted regression
print("\nRobustness Check 1: Unweighted DiD")
model_unweighted = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(f"  DiD Coefficient: {model_unweighted.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error: {model_unweighted.bse['ELIGIBLE_AFTER']:.4f}")

# Check 2: By gender
print("\nRobustness Check 2: By Gender")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex_val]
    model_sex = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=df_sex, weights=df_sex['PERWT']).fit(cov_type='HC1')
    print(f"  {sex_name}: DiD = {model_sex.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sex.bse['ELIGIBLE_AFTER']:.4f})")

# Check 3: Clustered standard errors by state
print("\nRobustness Check 3: Clustered SEs by State")
if 'STATEFIP' in df.columns:
    model_clustered = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                              data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                                 cov_kwds={'groups': df['STATEFIP']})
    print(f"  DiD Coefficient: {model_clustered.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  Clustered SE: {model_clustered.bse['ELIGIBLE_AFTER']:.4f}")
else:
    print("  State variable not available for clustering")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n9. SAVING RESULTS...")
print("-"*50)

# Create results summary
results = {
    'Model': ['Basic DiD', 'With Controls', 'Full Model'],
    'DiD_Estimate': [
        model1.params['ELIGIBLE_AFTER'],
        model2.params['ELIGIBLE_AFTER'],
        model3.params['ELIGIBLE_AFTER']
    ],
    'Std_Error': [
        model1.bse['ELIGIBLE_AFTER'],
        model2.bse['ELIGIBLE_AFTER'],
        model3.bse['ELIGIBLE_AFTER']
    ],
    'CI_Lower': [
        model1.conf_int().loc['ELIGIBLE_AFTER', 0],
        model2.conf_int().loc['ELIGIBLE_AFTER', 0],
        model3.conf_int().loc['ELIGIBLE_AFTER', 0]
    ],
    'CI_Upper': [
        model1.conf_int().loc['ELIGIBLE_AFTER', 1],
        model2.conf_int().loc['ELIGIBLE_AFTER', 1],
        model3.conf_int().loc['ELIGIBLE_AFTER', 1]
    ],
    'P_Value': [
        model1.pvalues['ELIGIBLE_AFTER'],
        model2.pvalues['ELIGIBLE_AFTER'],
        model3.pvalues['ELIGIBLE_AFTER']
    ],
    'R_Squared': [
        model1.rsquared,
        model2.rsquared,
        model3.rsquared
    ],
    'N': [model1.nobs, model2.nobs, model3.nobs]
}

results_df = pd.DataFrame(results)
results_df.to_csv(r'C:\Users\seraf\DACA Results Task 3\replication_73\regression_results.csv', index=False)
print("Saved regression results to regression_results.csv")

# Save event study coefficients
event_df = pd.DataFrame([
    {'Year': year, 'Coefficient': data['coef'], 'SE': data['se'], 'P_Value': data['pval']}
    for year, data in event_coefs.items()
])
event_df = event_df.sort_values('Year')
event_df.to_csv(r'C:\Users\seraf\DACA Results Task 3\replication_73\event_study_results.csv', index=False)
print("Saved event study results to event_study_results.csv")

# ============================================================================
# 10. CREATE VISUALIZATIONS
# ============================================================================
print("\n10. CREATING VISUALIZATIONS...")
print("-"*50)

# Figure 1: Full-time employment rates over time
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Calculate yearly FT rates by group
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()

yearly_rates.columns = ['Control (31-35)', 'Treatment (26-30)']
yearly_rates.plot(ax=ax1, marker='o', linewidth=2, markersize=8)

ax1.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Rates by DACA Eligibility Status', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_73\figure1_ft_rates.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 1: Full-time employment rates over time")

# Figure 2: Event study plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

years = sorted(event_coefs.keys())
coefs = [event_coefs[y]['coef'] for y in years]
ses = [event_coefs[y]['se'] for y in years]

# Add reference year (2011) with coefficient 0
years_plot = years[:3] + [2011] + years[3:]
coefs_plot = coefs[:3] + [0] + coefs[3:]
ses_plot = ses[:3] + [0] + ses[3:]

ax2.errorbar(years_plot, coefs_plot, yerr=[1.96*s for s in ses_plot],
             fmt='o-', capsize=4, capthick=2, linewidth=2, markersize=8, color='darkblue')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax2.fill_between([2007.5, 2012], [-0.1, -0.1], [0.1, 0.1], alpha=0.1, color='gray', label='Pre-treatment')
ax2.fill_between([2012, 2016.5], [-0.1, -0.1], [0.1, 0.1], alpha=0.1, color='blue', label='Post-treatment')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_73\figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 2: Event study plot")

# Figure 3: DiD visualization
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Pre/Post means
x_pre = 0.5
x_post = 1.5

# Control group
ax3.plot([x_pre, x_post], [ctrl_pre, ctrl_post], 'b-o', linewidth=2, markersize=10, label='Control (31-35)')
# Counterfactual for treatment
counterfactual = treat_pre + ctrl_diff
ax3.plot([x_pre, x_post], [treat_pre, counterfactual], 'g--o', linewidth=2, markersize=10,
         alpha=0.5, label='Treatment (counterfactual)')
# Actual treatment
ax3.plot([x_pre, x_post], [treat_pre, treat_post], 'r-o', linewidth=2, markersize=10, label='Treatment (26-30)')

# DiD arrow
ax3.annotate('', xy=(x_post + 0.1, treat_post), xytext=(x_post + 0.1, counterfactual),
             arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax3.text(x_post + 0.15, (treat_post + counterfactual)/2, f'DiD = {did_simple:.3f}',
         fontsize=12, color='purple', fontweight='bold')

ax3.set_xlim(0, 2.5)
ax3.set_xticks([x_pre, x_post])
ax3.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax3.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences Estimate of DACA Effect', fontsize=14)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_73\figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 3: DiD visualization")

# ============================================================================
# 11. SUMMARY STATISTICS TABLE
# ============================================================================
print("\n11. SUMMARY STATISTICS...")
print("-"*50)

# Create summary stats
summary_vars = ['FT', 'AGE', 'MALE', 'MARRIED', 'NCHILD'] if 'NCHILD' in df.columns else ['FT', 'AGE', 'MALE', 'MARRIED']

summary_stats = []
for var in summary_vars:
    if var in df.columns:
        # Overall
        overall_mean = weighted_mean(df, 'PERWT', var)
        overall_std = weighted_std(df, 'PERWT', var)

        # Treatment
        treat_df = df[df['ELIGIBLE'] == 1]
        treat_mean = weighted_mean(treat_df, 'PERWT', var)
        treat_std = weighted_std(treat_df, 'PERWT', var)

        # Control
        ctrl_df = df[df['ELIGIBLE'] == 0]
        ctrl_mean = weighted_mean(ctrl_df, 'PERWT', var)
        ctrl_std = weighted_std(ctrl_df, 'PERWT', var)

        summary_stats.append({
            'Variable': var,
            'Overall_Mean': overall_mean,
            'Overall_SD': overall_std,
            'Treatment_Mean': treat_mean,
            'Treatment_SD': treat_std,
            'Control_Mean': ctrl_mean,
            'Control_SD': ctrl_std
        })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(r'C:\Users\seraf\DACA Results Task 3\replication_73\summary_statistics.csv', index=False)
print("Saved summary statistics to summary_statistics.csv")

print("\nSummary Statistics Table:")
print(summary_df.to_string(index=False))

# ============================================================================
# 12. FINAL OUTPUT
# ============================================================================
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print(f"""
PREFERRED ESTIMATE (Model 3 - Full Model with Controls):
---------------------------------------------------------
Effect Size (DiD): {model3.params['ELIGIBLE_AFTER']:.4f}
  This represents a {model3.params['ELIGIBLE_AFTER']*100:.2f} percentage point change
  in full-time employment probability due to DACA eligibility.

Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}

95% Confidence Interval: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]

P-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}

Sample Size: {int(model3.nobs):,}

Interpretation:
The difference-in-differences estimate suggests that DACA eligibility
{'increased' if model3.params['ELIGIBLE_AFTER'] > 0 else 'decreased'} the probability of full-time employment
by approximately {abs(model3.params['ELIGIBLE_AFTER']*100):.1f} percentage points for eligible
individuals aged 26-30 compared to the comparison group (aged 31-35).
""")

# Save model summaries to text files for report
with open(r'C:\Users\seraf\DACA Results Task 3\replication_73\model1_summary.txt', 'w') as f:
    f.write(str(model1.summary()))

with open(r'C:\Users\seraf\DACA Results Task 3\replication_73\model3_summary.txt', 'w') as f:
    f.write(str(model3.summary()))

print("\nAnalysis complete. All results saved.")
print("="*70)
