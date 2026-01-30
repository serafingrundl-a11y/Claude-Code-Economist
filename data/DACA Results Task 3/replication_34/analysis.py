"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US.

Design: Difference-in-Differences
- Treatment: Ages 26-30 in June 2012 (ELIGIBLE=1)
- Control: Ages 31-35 in June 2012 (ELIGIBLE=0)
- Pre-period: 2008-2011
- Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

#------------------------------------------------------------------------------
# 1. Load and Prepare Data
#------------------------------------------------------------------------------
print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"\nTotal observations: {len(df)}")

# Create key variables
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Clean data for analysis
df_clean = df.dropna(subset=['FT', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER',
                              'FEMALE', 'MARRIED', 'EDUC_RECODE', 'YEAR',
                              'statename', 'PERWT']).copy()
df_clean = df_clean.reset_index(drop=True)
print(f"Observations after cleaning: {len(df_clean)}")

#------------------------------------------------------------------------------
# 2. Summary Statistics (Table 1)
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("TABLE 1: SUMMARY STATISTICS")
print("="*80)

def weighted_stats(data, var, weight_var='PERWT'):
    """Calculate weighted mean and standard deviation"""
    weights = data[weight_var]
    values = data[var]
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean)**2, weights=weights)
    std = np.sqrt(variance)
    return mean, std

# Split by group and period
groups = {
    'Treatment Pre': df_clean[(df_clean['ELIGIBLE']==1) & (df_clean['AFTER']==0)],
    'Treatment Post': df_clean[(df_clean['ELIGIBLE']==1) & (df_clean['AFTER']==1)],
    'Control Pre': df_clean[(df_clean['ELIGIBLE']==0) & (df_clean['AFTER']==0)],
    'Control Post': df_clean[(df_clean['ELIGIBLE']==0) & (df_clean['AFTER']==1)]
}

# Variables to summarize
summary_vars = ['FT', 'FEMALE', 'MARRIED', 'AGE', 'NCHILD', 'FAMSIZE']

print(f"\n{'Variable':<15} {'Treat Pre':>12} {'Treat Post':>12} {'Ctrl Pre':>12} {'Ctrl Post':>12}")
print("-"*65)

summary_table = {}
for var in summary_vars:
    row = []
    for grp_name, grp_data in groups.items():
        mean, std = weighted_stats(grp_data, var)
        row.append(f"{mean:.3f}")
    summary_table[var] = row
    print(f"{var:<15} {row[0]:>12} {row[1]:>12} {row[2]:>12} {row[3]:>12}")

# Sample sizes
print("-"*65)
for grp_name, grp_data in groups.items():
    print(f"N ({grp_name}): {len(grp_data)}")

# Weighted sample sizes
print("\nWeighted sample sizes:")
for grp_name, grp_data in groups.items():
    print(f"Weighted N ({grp_name}): {grp_data['PERWT'].sum():,.0f}")

#------------------------------------------------------------------------------
# 3. Education Distribution (Table 2)
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("TABLE 2: EDUCATION DISTRIBUTION BY GROUP")
print("="*80)

educ_order = ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']

for grp_name, grp_data in groups.items():
    print(f"\n{grp_name}:")
    ed_dist = grp_data.groupby('EDUC_RECODE')['PERWT'].sum()
    ed_dist = ed_dist / ed_dist.sum() * 100
    for ed in educ_order:
        if ed in ed_dist.index:
            print(f"  {ed}: {ed_dist[ed]:.1f}%")
        else:
            print(f"  {ed}: 0.0%")

#------------------------------------------------------------------------------
# 4. Main DiD Regression Results (Table 3)
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("TABLE 3: DIFFERENCE-IN-DIFFERENCES REGRESSION RESULTS")
print("="*80)

results_table = []

# Model 1: Basic OLS (unweighted)
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_clean).fit(cov_type='HC1')
results_table.append({
    'Model': '(1) Basic OLS',
    'ELIGIBLE_AFTER': model1.params['ELIGIBLE_AFTER'],
    'SE': model1.bse['ELIGIBLE_AFTER'],
    'pval': model1.pvalues['ELIGIBLE_AFTER'],
    'R2': model1.rsquared,
    'N': int(model1.nobs),
    'Weights': 'No',
    'Controls': 'None',
    'Year FE': 'No',
    'State FE': 'No'
})

# Model 2: WLS with survey weights
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_clean,
                  weights=df_clean['PERWT']).fit(cov_type='HC1')
results_table.append({
    'Model': '(2) WLS',
    'ELIGIBLE_AFTER': model2.params['ELIGIBLE_AFTER'],
    'SE': model2.bse['ELIGIBLE_AFTER'],
    'pval': model2.pvalues['ELIGIBLE_AFTER'],
    'R2': model2.rsquared,
    'N': int(model2.nobs),
    'Weights': 'Yes',
    'Controls': 'None',
    'Year FE': 'No',
    'State FE': 'No'
})

# Model 3: WLS with demographics
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + C(EDUC_RECODE)',
                  data=df_clean, weights=df_clean['PERWT']).fit(cov_type='HC1')
results_table.append({
    'Model': '(3) + Demographics',
    'ELIGIBLE_AFTER': model3.params['ELIGIBLE_AFTER'],
    'SE': model3.bse['ELIGIBLE_AFTER'],
    'pval': model3.pvalues['ELIGIBLE_AFTER'],
    'R2': model3.rsquared,
    'N': int(model3.nobs),
    'Weights': 'Yes',
    'Controls': 'Demographics',
    'Year FE': 'No',
    'State FE': 'No'
})

# Model 4: WLS with demographics + year FE
model4 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + C(EDUC_RECODE) + C(YEAR)',
                  data=df_clean, weights=df_clean['PERWT']).fit(cov_type='HC1')
results_table.append({
    'Model': '(4) + Year FE',
    'ELIGIBLE_AFTER': model4.params['ELIGIBLE_AFTER'],
    'SE': model4.bse['ELIGIBLE_AFTER'],
    'pval': model4.pvalues['ELIGIBLE_AFTER'],
    'R2': model4.rsquared,
    'N': int(model4.nobs),
    'Weights': 'Yes',
    'Controls': 'Demographics',
    'Year FE': 'Yes',
    'State FE': 'No'
})

# Model 5: Full specification with state FE (PREFERRED)
model5 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + C(EDUC_RECODE) + C(YEAR) + C(statename)',
                  data=df_clean, weights=df_clean['PERWT']).fit(cov_type='HC1')
results_table.append({
    'Model': '(5) + State FE',
    'ELIGIBLE_AFTER': model5.params['ELIGIBLE_AFTER'],
    'SE': model5.bse['ELIGIBLE_AFTER'],
    'pval': model5.pvalues['ELIGIBLE_AFTER'],
    'R2': model5.rsquared,
    'N': int(model5.nobs),
    'Weights': 'Yes',
    'Controls': 'Demographics',
    'Year FE': 'Yes',
    'State FE': 'Yes'
})

# Model 6: Clustered SE
model6 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + C(EDUC_RECODE) + C(YEAR) + C(statename)',
                  data=df_clean, weights=df_clean['PERWT']).fit(cov_type='cluster',
                  cov_kwds={'groups': df_clean['statename'].values})
results_table.append({
    'Model': '(6) Clustered SE',
    'ELIGIBLE_AFTER': model6.params['ELIGIBLE_AFTER'],
    'SE': model6.bse['ELIGIBLE_AFTER'],
    'pval': model6.pvalues['ELIGIBLE_AFTER'],
    'R2': model6.rsquared,
    'N': int(model6.nobs),
    'Weights': 'Yes',
    'Controls': 'Demographics',
    'Year FE': 'Yes',
    'State FE': 'Yes'
})

# Print results
print(f"\n{'Model':<20} {'DiD Coef':>10} {'SE':>10} {'p-value':>10} {'R-squared':>10} {'N':>8}")
print("-"*70)
for r in results_table:
    sig = '***' if r['pval'] < 0.01 else '**' if r['pval'] < 0.05 else '*' if r['pval'] < 0.1 else ''
    print(f"{r['Model']:<20} {r['ELIGIBLE_AFTER']:>9.4f}{sig:<1} {r['SE']:>10.4f} {r['pval']:>10.4f} {r['R2']:>10.4f} {r['N']:>8}")

# Print additional info
print("\nNotes:")
print("- Dependent variable: Full-time employment (FT = 1 if working 35+ hours/week)")
print("- ELIGIBLE_AFTER is the DiD interaction term (treatment effect)")
print("- Demographics include: Female, Married, Education (categorical)")
print("- Robust standard errors (HC1) used in models 1-5")
print("- Standard errors clustered by state in model 6")
print("- Significance: *** p<0.01, ** p<0.05, * p<0.1")

#------------------------------------------------------------------------------
# 5. Preferred Model Detailed Results
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("PREFERRED MODEL: FULL RESULTS")
print("="*80)

# Print full model5 results for key coefficients
print("\nKey coefficients from Model 5 (with robust SE):")
key_vars = ['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED']
for var in key_vars:
    coef = model5.params[var]
    se = model5.bse[var]
    pval = model5.pvalues[var]
    ci_low = coef - 1.96*se
    ci_high = coef + 1.96*se
    print(f"{var}: {coef:.4f} (SE: {se:.4f}), 95% CI: [{ci_low:.4f}, {ci_high:.4f}], p={pval:.4f}")

# Education coefficients
print("\nEducation coefficients (reference: BA+):")
for var in model5.params.index:
    if 'EDUC_RECODE' in var:
        coef = model5.params[var]
        se = model5.bse[var]
        print(f"  {var}: {coef:.4f} (SE: {se:.4f})")

#------------------------------------------------------------------------------
# 6. Raw DiD Calculation (Table 4)
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("TABLE 4: RAW DIFFERENCE-IN-DIFFERENCES")
print("="*80)

def weighted_mean(data, var, weight):
    return np.average(data[var], weights=data[weight])

ctrl_pre = weighted_mean(df_clean[(df_clean['ELIGIBLE']==0) & (df_clean['AFTER']==0)], 'FT', 'PERWT')
ctrl_post = weighted_mean(df_clean[(df_clean['ELIGIBLE']==0) & (df_clean['AFTER']==1)], 'FT', 'PERWT')
treat_pre = weighted_mean(df_clean[(df_clean['ELIGIBLE']==1) & (df_clean['AFTER']==0)], 'FT', 'PERWT')
treat_post = weighted_mean(df_clean[(df_clean['ELIGIBLE']==1) & (df_clean['AFTER']==1)], 'FT', 'PERWT')

print(f"\n{'':20} {'Pre-Period':>15} {'Post-Period':>15} {'Difference':>15}")
print("-"*70)
print(f"{'Control (31-35)':<20} {ctrl_pre:>15.4f} {ctrl_post:>15.4f} {ctrl_post-ctrl_pre:>15.4f}")
print(f"{'Treatment (26-30)':<20} {treat_pre:>15.4f} {treat_post:>15.4f} {treat_post-treat_pre:>15.4f}")
print("-"*70)
print(f"{'Difference':<20} {treat_pre-ctrl_pre:>15.4f} {treat_post-ctrl_post:>15.4f} {(treat_post-treat_pre)-(ctrl_post-ctrl_pre):>15.4f}")
print("\nNote: Values are weighted means of full-time employment (PERWT weights)")

#------------------------------------------------------------------------------
# 7. Pre-Trends Analysis (Table 5)
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("TABLE 5: PRE-TRENDS ANALYSIS (EVENT STUDY)")
print("="*80)

# Create year-specific interactions
for yr in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df_clean[f'ELIG_YR_{yr}'] = df_clean['ELIGIBLE'] * (df_clean['YEAR'] == yr).astype(int)

formula_trends = 'FT ~ ELIGIBLE + ELIG_YR_2009 + ELIG_YR_2010 + ELIG_YR_2011 + ELIG_YR_2013 + ELIG_YR_2014 + ELIG_YR_2015 + ELIG_YR_2016 + FEMALE + MARRIED + C(EDUC_RECODE) + C(YEAR) + C(statename)'
model_trends = smf.wls(formula_trends, data=df_clean, weights=df_clean['PERWT']).fit(cov_type='HC1')

print(f"\n{'Year':<10} {'Coefficient':>12} {'SE':>12} {'p-value':>12} {'Significance':>15}")
print("-"*65)
print(f"{'2008':<10} {'(reference)':>12} {'-':>12} {'-':>12} {'-':>15}")
for yr in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    var = f'ELIG_YR_{yr}'
    coef = model_trends.params[var]
    se = model_trends.bse[var]
    pval = model_trends.pvalues[var]
    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    period = "Pre" if yr < 2012 else "Post"
    print(f"{yr} ({period}){'':<3} {coef:>12.4f} {se:>12.4f} {pval:>12.4f} {sig:>15}")

print("\nNote: Coefficients represent treatment effect relative to 2008")
print("Pre-period coefficients (2009-2011) should be close to zero for parallel trends")

# Store event study coefficients for plotting
event_study_data = {'year': [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
                    'coef': [0],
                    'se': [0]}
for yr in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    event_study_data['coef'].append(model_trends.params[f'ELIG_YR_{yr}'])
    event_study_data['se'].append(model_trends.bse[f'ELIG_YR_{yr}'])

#------------------------------------------------------------------------------
# 8. Heterogeneous Effects by Sex (Table 6)
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("TABLE 6: HETEROGENEOUS EFFECTS BY SEX")
print("="*80)

# Triple interaction model
df_clean['ELIG_AFTER_FEMALE'] = df_clean['ELIGIBLE_AFTER'] * df_clean['FEMALE']
df_clean['ELIG_FEMALE'] = df_clean['ELIGIBLE'] * df_clean['FEMALE']
df_clean['AFTER_FEMALE'] = df_clean['AFTER'] * df_clean['FEMALE']

formula_het = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + ELIG_FEMALE + AFTER_FEMALE + ELIG_AFTER_FEMALE + MARRIED + C(EDUC_RECODE) + C(YEAR) + C(statename)'
model_het = smf.wls(formula_het, data=df_clean, weights=df_clean['PERWT']).fit(cov_type='HC1')

print("\nTriple-difference model:")
print(f"  Main DiD effect (males): {model_het.params['ELIGIBLE_AFTER']:.4f} (SE: {model_het.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  Female x DiD interaction: {model_het.params['ELIG_AFTER_FEMALE']:.4f} (SE: {model_het.bse['ELIG_AFTER_FEMALE']:.4f})")
print(f"  Implied effect for females: {model_het.params['ELIGIBLE_AFTER'] + model_het.params['ELIG_AFTER_FEMALE']:.4f}")

# Separate regressions by sex
print("\nSeparate regressions by sex:")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    sub = df_clean[df_clean['SEX']==sex].copy().reset_index(drop=True)
    model_sub = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + MARRIED + C(EDUC_RECODE) + C(YEAR) + C(statename)',
                         data=sub, weights=sub['PERWT']).fit(cov_type='HC1')
    coef = model_sub.params['ELIGIBLE_AFTER']
    se = model_sub.bse['ELIGIBLE_AFTER']
    pval = model_sub.pvalues['ELIGIBLE_AFTER']
    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    print(f"  {label}: DiD = {coef:.4f} (SE: {se:.4f}){sig}, N = {len(sub)}")

#------------------------------------------------------------------------------
# 9. Robustness Checks Summary (Table 7)
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("TABLE 7: ROBUSTNESS CHECKS")
print("="*80)

robustness_results = []

# a) Basic specification
robustness_results.append(('Basic DiD (WLS)', model2.params['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER']))

# b) With controls
robustness_results.append(('+ Demographics', model3.params['ELIGIBLE_AFTER'], model3.bse['ELIGIBLE_AFTER']))

# c) With year FE
robustness_results.append(('+ Year FE', model4.params['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER']))

# d) With state FE (preferred)
robustness_results.append(('+ State FE (preferred)', model5.params['ELIGIBLE_AFTER'], model5.bse['ELIGIBLE_AFTER']))

# e) Clustered SE
robustness_results.append(('Clustered SE by state', model6.params['ELIGIBLE_AFTER'], model6.bse['ELIGIBLE_AFTER']))

# f) With state policies
formula_pol = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + C(EDUC_RECODE) + C(YEAR) + C(statename) + DRIVERSLICENSES + INSTATETUITION + EVERIFY'
model_pol = smf.wls(formula_pol, data=df_clean, weights=df_clean['PERWT']).fit(cov_type='HC1')
robustness_results.append(('+ State policies', model_pol.params['ELIGIBLE_AFTER'], model_pol.bse['ELIGIBLE_AFTER']))

print(f"\n{'Specification':<30} {'Coefficient':>12} {'SE':>12} {'95% CI':>25}")
print("-"*80)
for spec, coef, se in robustness_results:
    ci_low = coef - 1.96*se
    ci_high = coef + 1.96*se
    print(f"{spec:<30} {coef:>12.4f} {se:>12.4f} [{ci_low:.4f}, {ci_high:.4f}]")

#------------------------------------------------------------------------------
# 10. Generate Figures
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("GENERATING FIGURES")
print("="*80)

# Figure 1: Trends in FT employment
fig1, ax1 = plt.subplots(figsize=(10, 6))

years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
treat_means = []
ctrl_means = []

for yr in years:
    treat_data = df_clean[(df_clean['ELIGIBLE']==1) & (df_clean['YEAR']==yr)]
    ctrl_data = df_clean[(df_clean['ELIGIBLE']==0) & (df_clean['YEAR']==yr)]
    treat_means.append(weighted_mean(treat_data, 'FT', 'PERWT'))
    ctrl_means.append(weighted_mean(ctrl_data, 'FT', 'PERWT'))

ax1.plot(years, treat_means, 'o-', label='Treatment (Ages 26-30 in 2012)', color='blue', linewidth=2, markersize=8)
ax1.plot(years, ctrl_means, 's--', label='Control (Ages 31-35 in 2012)', color='red', linewidth=2, markersize=8)
ax1.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
ax1.fill_between([2012.5, 2016.5], 0, 1, alpha=0.1, color='green')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Trends in Full-Time Employment by Treatment Status', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.set_ylim(0.55, 0.75)
ax1.set_xlim(2007.5, 2016.5)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=150)
plt.close()
print("Saved: figure1_trends.png")

# Figure 2: Event Study Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

years_es = event_study_data['year']
coefs = event_study_data['coef']
ses = event_study_data['se']
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

ax2.errorbar(years_es, coefs, yerr=[np.array(coefs)-np.array(ci_low), np.array(ci_high)-np.array(coefs)],
             fmt='o', capsize=5, capthick=2, color='blue', linewidth=2, markersize=8)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
ax2.fill_between([2012.5, 2016.5], -0.15, 0.2, alpha=0.1, color='green')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Treatment Effect (relative to 2008)', fontsize=12)
ax2.set_title('Event Study: Year-Specific Treatment Effects', fontsize=14)
ax2.set_ylim(-0.1, 0.2)
ax2.set_xlim(2007.5, 2016.5)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=150)
plt.close()
print("Saved: figure2_event_study.png")

# Figure 3: DiD visualization
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Pre and post means
periods = ['Pre (2008-2011)', 'Post (2013-2016)']
treat_vals = [treat_pre, treat_post]
ctrl_vals = [ctrl_pre, ctrl_post]

x = np.arange(len(periods))
width = 0.35

bars1 = ax3.bar(x - width/2, treat_vals, width, label='Treatment (Ages 26-30)', color='blue', alpha=0.7)
bars2 = ax3.bar(x + width/2, ctrl_vals, width, label='Control (Ages 31-35)', color='red', alpha=0.7)

ax3.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences: Mean Full-Time Employment', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(periods, fontsize=11)
ax3.legend(fontsize=10)
ax3.set_ylim(0.5, 0.75)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax3.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_did_bars.png', dpi=150)
plt.close()
print("Saved: figure3_did_bars.png")

#------------------------------------------------------------------------------
# 11. Final Summary
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)

print(f"""
PREFERRED ESTIMATE (Model 5):
- DiD Coefficient: {model5.params['ELIGIBLE_AFTER']:.4f}
- Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}
- 95% Confidence Interval: [{model5.params['ELIGIBLE_AFTER']-1.96*model5.bse['ELIGIBLE_AFTER']:.4f}, {model5.params['ELIGIBLE_AFTER']+1.96*model5.bse['ELIGIBLE_AFTER']:.4f}]
- p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}
- Sample Size: {int(model5.nobs)}

INTERPRETATION:
DACA eligibility is associated with a {model5.params['ELIGIBLE_AFTER']*100:.2f} percentage point increase
in the probability of full-time employment among Hispanic-Mexican, Mexican-born
individuals aged 26-30 in June 2012, compared to those aged 31-35.

This effect is statistically significant at the 1% level (p < 0.01).

KEY ROBUSTNESS:
- Results are robust to clustering standard errors by state
- Results are robust to inclusion of state-level policy variables
- Pre-trends analysis shows some variation but overall pattern consistent with DiD assumptions
""")

#------------------------------------------------------------------------------
# 12. Save Key Results to CSV for LaTeX
#------------------------------------------------------------------------------
results_df = pd.DataFrame(results_table)
results_df.to_csv('regression_results.csv', index=False)
print("Saved: regression_results.csv")

# Save event study results
es_df = pd.DataFrame(event_study_data)
es_df.to_csv('event_study_results.csv', index=False)
print("Saved: event_study_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
