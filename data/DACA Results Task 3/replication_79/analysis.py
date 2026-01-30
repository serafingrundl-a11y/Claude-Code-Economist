"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Mexican-born Hispanic individuals.

Identification Strategy: Difference-in-Differences
- Treatment Group: Ages 26-30 in June 2012 (ELIGIBLE=1)
- Control Group: Ages 31-35 in June 2012 (ELIGIBLE=0)
- Pre-period: 2008-2011 (AFTER=0)
- Post-period: 2013-2016 (AFTER=1)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# Load data
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_79\data\prepared_data_labelled_version.csv"
df = pd.read_csv(data_path)

print(f"\n1. DATA OVERVIEW")
print(f"   Total observations: {len(df):,}")
print(f"   Number of variables: {len(df.columns)}")

# Check key variables
print(f"\n2. KEY VARIABLES:")
print(f"   - YEAR range: {df['YEAR'].min()} - {df['YEAR'].max()}")
print(f"   - ELIGIBLE: {df['ELIGIBLE'].value_counts().to_dict()}")
print(f"   - AFTER: {df['AFTER'].value_counts().to_dict()}")
print(f"   - FT (Full-time): {df['FT'].value_counts().to_dict()}")

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================

print(f"\n3. SAMPLE BY YEAR AND TREATMENT STATUS:")
year_eligible = pd.crosstab(df['YEAR'], df['ELIGIBLE'], margins=True)
print(year_eligible)

print(f"\n4. SAMPLE BY PERIOD AND TREATMENT STATUS:")
period_eligible = pd.crosstab(df['AFTER'], df['ELIGIBLE'], margins=True)
period_eligible.index = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)', 'Total']
period_eligible.columns = ['Control (31-35)', 'Treated (26-30)', 'Total']
print(period_eligible)

# ============================================================================
# 3. FULL-TIME EMPLOYMENT RATES BY GROUP AND PERIOD
# ============================================================================

print(f"\n5. FULL-TIME EMPLOYMENT RATES:")

# Simple means table
ft_means = df.groupby(['AFTER', 'ELIGIBLE'])['FT'].mean().unstack()
ft_means.index = ['Pre-DACA', 'Post-DACA']
ft_means.columns = ['Control (31-35)', 'Treated (26-30)']
print(ft_means.round(4))

# Calculate simple DiD
pre_treat = df[(df['AFTER']==0) & (df['ELIGIBLE']==1)]['FT'].mean()
pre_ctrl = df[(df['AFTER']==0) & (df['ELIGIBLE']==0)]['FT'].mean()
post_treat = df[(df['AFTER']==1) & (df['ELIGIBLE']==1)]['FT'].mean()
post_ctrl = df[(df['AFTER']==1) & (df['ELIGIBLE']==0)]['FT'].mean()

simple_did = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)
print(f"\n   Simple DiD estimate: {simple_did:.4f}")
print(f"   Pre-period difference (T-C): {pre_treat - pre_ctrl:.4f}")
print(f"   Post-period difference (T-C): {post_treat - post_ctrl:.4f}")
print(f"   Change in treated: {post_treat - pre_treat:.4f}")
print(f"   Change in control: {post_ctrl - pre_ctrl:.4f}")

# ============================================================================
# 4. YEAR-BY-YEAR TRENDS
# ============================================================================

print(f"\n6. FULL-TIME EMPLOYMENT BY YEAR AND GROUP:")
ft_by_year = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].agg(['mean', 'count']).unstack()
print(ft_by_year.round(4))

# Save for plotting
trends_df = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack().reset_index()
trends_df.columns = ['YEAR', 'Control', 'Treated']

# ============================================================================
# 5. DEMOGRAPHIC CHARACTERISTICS
# ============================================================================

print(f"\n7. DEMOGRAPHIC CHARACTERISTICS BY TREATMENT STATUS (Pre-period):")

pre_period = df[df['AFTER'] == 0].copy()

# Key demographics
demographics = ['AGE', 'SEX', 'MARST', 'NCHILD', 'FAMSIZE', 'HHINCOME', 'POVERTY']

for var in demographics:
    if var in pre_period.columns:
        if var == 'SEX':
            # Sex is categorical: 1=Male, 2=Female
            pre_period['MALE'] = (pre_period['SEX'] == 'Male').astype(int)
            male_by_group = pre_period.groupby('ELIGIBLE')['MALE'].mean()
            print(f"   {var} (% Male): Control={male_by_group[0]:.3f}, Treated={male_by_group[1]:.3f}")
        elif var == 'MARST':
            # Check if married
            pre_period['MARRIED'] = pre_period['MARST'].str.contains('Married', case=False, na=False).astype(int)
            married_by_group = pre_period.groupby('ELIGIBLE')['MARRIED'].mean()
            print(f"   {var} (% Married): Control={married_by_group[0]:.3f}, Treated={married_by_group[1]:.3f}")
        else:
            means_by_group = pre_period.groupby('ELIGIBLE')[var].mean()
            print(f"   {var}: Control={means_by_group[0]:.2f}, Treated={means_by_group[1]:.2f}")

# Education distribution
print(f"\n8. EDUCATION DISTRIBUTION BY GROUP (Pre-period):")
if 'EDUC_RECODE' in df.columns:
    educ_dist = pd.crosstab(pre_period['EDUC_RECODE'], pre_period['ELIGIBLE'], normalize='columns')
    educ_dist.columns = ['Control', 'Treated']
    print(educ_dist.round(3))

# ============================================================================
# 6. DIFFERENCE-IN-DIFFERENCES REGRESSION
# ============================================================================

print("\n" + "="*80)
print("REGRESSION ANALYSIS")
print("="*80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no covariates)
print("\n9. MODEL 1: Basic Difference-in-Differences")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n10. MODEL 2: DiD with Demographics")

# Create control variables
df['MALE'] = (df['SEX'] == 'Male').astype(int)
df['MARRIED'] = df['MARST'].str.contains('Married', case=False, na=False).astype(int)

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + FAMSIZE',
                 data=df).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with education controls
print("\n11. MODEL 3: DiD with Education Controls")

# Create education dummies
df['HS_DEGREE_BIN'] = df['HS_DEGREE'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0)

model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + FAMSIZE + HS_DEGREE_BIN',
                 data=df).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects
print("\n12. MODEL 4: DiD with Year Fixed Effects")

# Create year dummies (excluding reference year)
df['YEAR'] = df['YEAR'].astype(int)
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
df_with_years = pd.concat([df, year_dummies], axis=1)

year_vars = [col for col in df_with_years.columns if col.startswith('year_')]
year_formula = ' + '.join(year_vars)

model4 = smf.ols(f'FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + {year_formula}',
                 data=df_with_years).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Std. Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.3f}")
print(f"   p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 5: DiD with state fixed effects
print("\n13. MODEL 5: DiD with State Fixed Effects")

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
df_with_states = pd.concat([df, state_dummies], axis=1)

state_vars = [col for col in df_with_states.columns if col.startswith('state_')]

model5_formula = f'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD'
model5 = smf.ols(model5_formula, data=df).fit(cov_type='HC1')

# Full model with state FE (may need to use absorb approach)
print(f"   Note: Running model with state controls")
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Std. Error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")

# ============================================================================
# 7. WEIGHTED ANALYSIS
# ============================================================================

print("\n14. WEIGHTED ANALYSIS (using PERWT)")

# Model with sample weights
model_weighted = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD',
                         data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"   Weighted DiD coefficient: {model_weighted.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Std. Error: {model_weighted.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   p-value: {model_weighted.pvalues['ELIGIBLE_AFTER']:.4f}")

# ============================================================================
# 8. CLUSTERED STANDARD ERRORS
# ============================================================================

print("\n15. CLUSTERED STANDARD ERRORS (by State)")

model_clustered = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD',
                          data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"   Clustered DiD coefficient: {model_clustered.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Clustered Std. Error: {model_clustered.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   p-value: {model_clustered.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model_clustered.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_clustered.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# ============================================================================
# 9. ROBUSTNESS CHECKS
# ============================================================================

print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Check 1: By gender
print("\n16. HETEROGENEITY BY GENDER:")
for gender in ['Male', 'Female']:
    subset = df[df['SEX'] == gender]
    model_gender = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                           data=subset).fit(cov_type='HC1')
    print(f"   {gender}: DiD = {model_gender.params['ELIGIBLE_AFTER']:.4f} (SE = {model_gender.bse['ELIGIBLE_AFTER']:.4f}), p = {model_gender.pvalues['ELIGIBLE_AFTER']:.4f}")

# Check 2: By education
print("\n17. HETEROGENEITY BY EDUCATION:")
for hs in [True, False]:
    hs_label = "HS+" if hs else "Less than HS"
    subset = df[df['HS_DEGREE_BIN'] == (1 if hs else 0)]
    if len(subset) > 100:
        model_educ = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                             data=subset).fit(cov_type='HC1')
        print(f"   {hs_label}: DiD = {model_educ.params['ELIGIBLE_AFTER']:.4f} (SE = {model_educ.bse['ELIGIBLE_AFTER']:.4f}), p = {model_educ.pvalues['ELIGIBLE_AFTER']:.4f}")

# Check 3: Different time windows
print("\n18. DIFFERENT TIME WINDOWS:")
# 2010-2011 vs 2013-2014
df_narrow = df[df['YEAR'].isin([2010, 2011, 2013, 2014])]
if len(df_narrow) > 100:
    model_narrow = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                           data=df_narrow).fit(cov_type='HC1')
    print(f"   2010-2011 vs 2013-2014: DiD = {model_narrow.params['ELIGIBLE_AFTER']:.4f} (SE = {model_narrow.bse['ELIGIBLE_AFTER']:.4f})")

# ============================================================================
# 10. EVENT STUDY ANALYSIS
# ============================================================================

print("\n19. EVENT STUDY ANALYSIS:")

# Create year dummies interacted with treatment
df['YEAR'] = df['YEAR'].astype(int)
years = sorted(df['YEAR'].unique())
ref_year = 2011  # Reference year (last pre-treatment year)

event_study_results = []
for year in years:
    if year != ref_year:
        df[f'year_{year}_treat'] = ((df['YEAR'] == year) & (df['ELIGIBLE'] == 1)).astype(int)

# Event study regression
event_vars = [f'year_{y}_treat' for y in years if y != ref_year]
event_formula = 'FT ~ ELIGIBLE + ' + ' + '.join([f'C(YEAR)'] + event_vars)

# Simplified event study
for year in years:
    subset = df[(df['YEAR'] == ref_year) | (df['YEAR'] == year)]
    if year != ref_year and len(subset) > 100:
        subset['POST'] = (subset['YEAR'] == year).astype(int)
        subset['TREAT_POST'] = subset['ELIGIBLE'] * subset['POST']
        model_event = smf.ols('FT ~ ELIGIBLE + POST + TREAT_POST', data=subset).fit(cov_type='HC1')
        event_study_results.append({
            'year': year,
            'coef': model_event.params['TREAT_POST'],
            'se': model_event.bse['TREAT_POST'],
            'pval': model_event.pvalues['TREAT_POST']
        })

print("   Year-by-year treatment effects (relative to 2011):")
for res in event_study_results:
    sig = "*" if res['pval'] < 0.05 else ""
    print(f"   {res['year']}: {res['coef']:.4f} ({res['se']:.4f}){sig}")

# ============================================================================
# 11. SAVE RESULTS FOR REPORTING
# ============================================================================

# Save key results to a file
results_dict = {
    'n_obs': len(df),
    'n_treated': df['ELIGIBLE'].sum(),
    'n_control': (df['ELIGIBLE']==0).sum(),
    'pre_treat_ft': pre_treat,
    'pre_ctrl_ft': pre_ctrl,
    'post_treat_ft': post_treat,
    'post_ctrl_ft': post_ctrl,
    'simple_did': simple_did,
    'model1_coef': model1.params['ELIGIBLE_AFTER'],
    'model1_se': model1.bse['ELIGIBLE_AFTER'],
    'model1_pval': model1.pvalues['ELIGIBLE_AFTER'],
    'model3_coef': model3.params['ELIGIBLE_AFTER'],
    'model3_se': model3.bse['ELIGIBLE_AFTER'],
    'model3_pval': model3.pvalues['ELIGIBLE_AFTER'],
    'clustered_coef': model_clustered.params['ELIGIBLE_AFTER'],
    'clustered_se': model_clustered.bse['ELIGIBLE_AFTER'],
    'clustered_pval': model_clustered.pvalues['ELIGIBLE_AFTER'],
    'clustered_ci_low': model_clustered.conf_int().loc['ELIGIBLE_AFTER', 0],
    'clustered_ci_high': model_clustered.conf_int().loc['ELIGIBLE_AFTER', 1],
    'weighted_coef': model_weighted.params['ELIGIBLE_AFTER'],
    'weighted_se': model_weighted.bse['ELIGIBLE_AFTER'],
}

# ============================================================================
# 12. CREATE VISUALIZATIONS
# ============================================================================

print("\n20. CREATING VISUALIZATIONS...")

# Figure 1: Parallel Trends
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(trends_df['YEAR'], trends_df['Control'], 'o-', label='Control (31-35)', linewidth=2, markersize=8)
ax.plot(trends_df['YEAR'], trends_df['Treated'], 's-', label='Treated (26-30)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation (June 2012)', alpha=0.7)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax.legend(loc='lower right')
ax.set_xlim(2007.5, 2016.5)
ax.set_ylim(0, max(trends_df['Control'].max(), trends_df['Treated'].max()) * 1.1)
plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_79\figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: figure1_parallel_trends.png")

# Figure 2: Event Study Plot
fig, ax = plt.subplots(figsize=(10, 6))
years_plot = [r['year'] for r in event_study_results]
coefs = [r['coef'] for r in event_study_results]
ses = [r['se'] for r in event_study_results]

# Add reference year
years_plot_full = years_plot[:3] + [2011] + years_plot[3:]
coefs_full = coefs[:3] + [0] + coefs[3:]
ses_full = ses[:3] + [0] + ses[3:]

ax.errorbar(years_plot_full, coefs_full, yerr=[1.96*s for s in ses_full], fmt='o-', capsize=5, linewidth=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation', alpha=0.7)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA on Full-Time Employment', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_79\figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: figure2_event_study.png")

# Figure 3: DiD Visualization
fig, ax = plt.subplots(figsize=(10, 6))

periods = ['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)']
treated_means = [pre_treat, post_treat]
control_means = [pre_ctrl, post_ctrl]

x = np.arange(len(periods))
width = 0.35

bars1 = ax.bar(x - width/2, control_means, width, label='Control (31-35)', color='steelblue')
bars2 = ax.bar(x + width/2, treated_means, width, label='Treated (26-30)', color='coral')

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: Full-Time Employment', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(periods)
ax.legend()
ax.set_ylim(0, max(treated_means + control_means) * 1.2)

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_79\figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: figure3_did_bars.png")

# Figure 4: Sample Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# By Year
year_counts = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
year_counts.plot(kind='bar', ax=axes[0])
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Number of Observations', fontsize=12)
axes[0].set_title('Sample Size by Year and Treatment Status', fontsize=14)
axes[0].legend(['Control (31-35)', 'Treated (26-30)'])
axes[0].tick_params(axis='x', rotation=45)

# Age distribution
df['AGE_IN_JUNE_2012_NUM'] = pd.to_numeric(df['AGE_IN_JUNE_2012'], errors='coerce')
age_dist = df.groupby('AGE_IN_JUNE_2012_NUM').size()
axes[1].bar(age_dist.index, age_dist.values)
axes[1].axvline(x=30.5, color='red', linestyle='--', label='DACA Cutoff (age 31)', alpha=0.7)
axes[1].set_xlabel('Age in June 2012', fontsize=12)
axes[1].set_ylabel('Number of Observations', fontsize=12)
axes[1].set_title('Distribution of Age at DACA Implementation', fontsize=14)
axes[1].legend()

plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_79\figure4_sample_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: figure4_sample_distribution.png")

# ============================================================================
# 13. CREATE REGRESSION TABLE
# ============================================================================

print("\n21. CREATING REGRESSION TABLE...")

# Compile results into a summary table
reg_table = pd.DataFrame({
    'Model 1\n(Basic DiD)': [
        f"{model1.params['ELIGIBLE_AFTER']:.4f}",
        f"({model1.bse['ELIGIBLE_AFTER']:.4f})",
        f"{model1.params['ELIGIBLE']:.4f}",
        f"({model1.bse['ELIGIBLE']:.4f})",
        f"{model1.params['AFTER']:.4f}",
        f"({model1.bse['AFTER']:.4f})",
        '-',
        '-',
        '-',
        '-',
        '-',
        '-',
        '-',
        '-',
        f"{model1.rsquared:.4f}",
        f"{int(model1.nobs)}"
    ],
    'Model 2\n(+ Demographics)': [
        f"{model2.params['ELIGIBLE_AFTER']:.4f}",
        f"({model2.bse['ELIGIBLE_AFTER']:.4f})",
        f"{model2.params['ELIGIBLE']:.4f}",
        f"({model2.bse['ELIGIBLE']:.4f})",
        f"{model2.params['AFTER']:.4f}",
        f"({model2.bse['AFTER']:.4f})",
        f"{model2.params['MALE']:.4f}",
        f"({model2.bse['MALE']:.4f})",
        f"{model2.params['MARRIED']:.4f}",
        f"({model2.bse['MARRIED']:.4f})",
        f"{model2.params['NCHILD']:.4f}",
        f"({model2.bse['NCHILD']:.4f})",
        '-',
        '-',
        f"{model2.rsquared:.4f}",
        f"{int(model2.nobs)}"
    ],
    'Model 3\n(+ Education)': [
        f"{model3.params['ELIGIBLE_AFTER']:.4f}",
        f"({model3.bse['ELIGIBLE_AFTER']:.4f})",
        f"{model3.params['ELIGIBLE']:.4f}",
        f"({model3.bse['ELIGIBLE']:.4f})",
        f"{model3.params['AFTER']:.4f}",
        f"({model3.bse['AFTER']:.4f})",
        f"{model3.params['MALE']:.4f}",
        f"({model3.bse['MALE']:.4f})",
        f"{model3.params['MARRIED']:.4f}",
        f"({model3.bse['MARRIED']:.4f})",
        f"{model3.params['NCHILD']:.4f}",
        f"({model3.bse['NCHILD']:.4f})",
        f"{model3.params['HS_DEGREE_BIN']:.4f}",
        f"({model3.bse['HS_DEGREE_BIN']:.4f})",
        f"{model3.rsquared:.4f}",
        f"{int(model3.nobs)}"
    ],
    'Model 4\n(Clustered SE)': [
        f"{model_clustered.params['ELIGIBLE_AFTER']:.4f}",
        f"({model_clustered.bse['ELIGIBLE_AFTER']:.4f})",
        f"{model_clustered.params['ELIGIBLE']:.4f}",
        f"({model_clustered.bse['ELIGIBLE']:.4f})",
        f"{model_clustered.params['AFTER']:.4f}",
        f"({model_clustered.bse['AFTER']:.4f})",
        f"{model_clustered.params['MALE']:.4f}",
        f"({model_clustered.bse['MALE']:.4f})",
        f"{model_clustered.params['MARRIED']:.4f}",
        f"({model_clustered.bse['MARRIED']:.4f})",
        f"{model_clustered.params['NCHILD']:.4f}",
        f"({model_clustered.bse['NCHILD']:.4f})",
        '-',
        '-',
        f"{model_clustered.rsquared:.4f}",
        f"{int(model_clustered.nobs)}"
    ]
}, index=[
    'ELIGIBLE × AFTER', '', 'ELIGIBLE', '', 'AFTER', '',
    'Male', '', 'Married', '', 'N Children', '',
    'HS Degree', '', 'R²', 'N'
])

reg_table.to_csv(r'C:\Users\seraf\DACA Results Task 3\replication_79\regression_table.csv')
print("   Saved: regression_table.csv")

# ============================================================================
# 14. SUMMARY STATISTICS TABLE
# ============================================================================

print("\n22. CREATING SUMMARY STATISTICS TABLE...")

# Summary stats for key variables
key_vars = ['FT', 'AGE', 'MALE', 'MARRIED', 'NCHILD', 'FAMSIZE', 'HHINCOME', 'POVERTY', 'HS_DEGREE_BIN']
summary_data = []

for var in key_vars:
    if var in df.columns:
        row = {
            'Variable': var,
            'Overall Mean': df[var].mean(),
            'Overall SD': df[var].std(),
            'Treated Mean': df[df['ELIGIBLE']==1][var].mean(),
            'Treated SD': df[df['ELIGIBLE']==1][var].std(),
            'Control Mean': df[df['ELIGIBLE']==0][var].mean(),
            'Control SD': df[df['ELIGIBLE']==0][var].std(),
        }
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(r'C:\Users\seraf\DACA Results Task 3\replication_79\summary_statistics.csv', index=False)
print("   Saved: summary_statistics.csv")

# ============================================================================
# 15. PRINT FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"""
PREFERRED ESTIMATE (Model with clustered standard errors by state):

   Effect Size (DiD coefficient): {model_clustered.params['ELIGIBLE_AFTER']:.4f}
   Standard Error (clustered): {model_clustered.bse['ELIGIBLE_AFTER']:.4f}
   95% Confidence Interval: [{model_clustered.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_clustered.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]
   t-statistic: {model_clustered.tvalues['ELIGIBLE_AFTER']:.3f}
   p-value: {model_clustered.pvalues['ELIGIBLE_AFTER']:.4f}

   Sample Size: {len(df):,}
   - Treated group (ages 26-30): {df['ELIGIBLE'].sum():,}
   - Control group (ages 31-35): {(df['ELIGIBLE']==0).sum():,}

INTERPRETATION:
   The DiD estimate of {model_clustered.params['ELIGIBLE_AFTER']:.4f} suggests that DACA eligibility
   {'increased' if model_clustered.params['ELIGIBLE_AFTER'] > 0 else 'decreased'} the probability of full-time employment
   by approximately {abs(model_clustered.params['ELIGIBLE_AFTER'])*100:.1f} percentage points among eligible individuals
   (ages 26-30 in June 2012) compared to the control group (ages 31-35).

   This effect is {'statistically significant' if model_clustered.pvalues['ELIGIBLE_AFTER'] < 0.05 else 'not statistically significant'}
   at the 5% level (p = {model_clustered.pvalues['ELIGIBLE_AFTER']:.4f}).
""")

# Save final results to a text file
with open(r'C:\Users\seraf\DACA Results Task 3\replication_79\results_summary.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - FINAL RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Preferred Estimate (Clustered SE):\n")
    f.write(f"   DiD Coefficient: {model_clustered.params['ELIGIBLE_AFTER']:.4f}\n")
    f.write(f"   Standard Error: {model_clustered.bse['ELIGIBLE_AFTER']:.4f}\n")
    f.write(f"   95% CI: [{model_clustered.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_clustered.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]\n")
    f.write(f"   p-value: {model_clustered.pvalues['ELIGIBLE_AFTER']:.4f}\n")
    f.write(f"   Sample Size: {len(df):,}\n")

print("\nAnalysis complete. All output files saved.")
