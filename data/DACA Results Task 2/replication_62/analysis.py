"""
DACA Replication Study: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican individuals born in Mexico

Research Design: Difference-in-Differences
Treatment group: Ages 26-30 as of June 15, 2012 (born 1982-1986)
Control group: Ages 31-35 as of June 15, 2012 (born 1977-1981)

Outcome: Full-time employment (usually working 35+ hours per week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Load and Prepare Data
# =============================================================================

print("Loading data...")
df = pd.read_csv('data/data.csv')
print(f"Total observations: {len(df):,}")

# =============================================================================
# 2. Define DACA Eligibility Criteria
# =============================================================================
# DACA requirements:
# 1. Arrived in US before 16th birthday
# 2. Under 31 as of June 15, 2012 (for treated group we use 26-30, control 31-35)
# 3. Continuously in US since June 15, 2007
# 4. Present in US on June 15, 2012
# 5. Not a citizen or legal resident

# Filter for Hispanic-Mexican (HISPAN=1) and born in Mexico (BPL=200)
print("\nFiltering for Hispanic-Mexican individuals born in Mexico...")
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
print(f"Hispanic-Mexican born in Mexico: {len(df_mex):,}")

# Filter for non-citizens (CITIZEN=3: Not a citizen)
# Per instructions: assume anyone who is not a citizen and has not received
# immigration papers is undocumented
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"After filtering for non-citizens: {len(df_mex):,}")

# DACA eligibility: arrived before age 16
# We need YRIMMIG (year of immigration) and BIRTHYR
# Age at immigration = YRIMMIG - BIRTHYR
df_mex['age_at_immigration'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# Filter for those who immigrated before age 16
df_mex = df_mex[df_mex['age_at_immigration'] < 16].copy()
print(f"After filtering for immigration before age 16: {len(df_mex):,}")

# Continuous residence since June 15, 2007:
# YRIMMIG should be 2007 or earlier
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007].copy()
print(f"After filtering for continuous residence since 2007: {len(df_mex):,}")

# =============================================================================
# 3. Define Treatment and Control Groups Based on Age
# =============================================================================
# DACA implemented June 15, 2012
# Treatment: Ages 26-30 as of June 15, 2012 (born between June 16, 1981 and June 15, 1986)
# Control: Ages 31-35 as of June 15, 2012 (born between June 16, 1976 and June 15, 1981)
#
# Since we only have birth year (not exact date), we approximate:
# Treatment: Birth years 1982-1986 (would be 26-30 sometime in 2012)
# Control: Birth years 1977-1981 (would be 31-35 sometime in 2012)

df_mex['treated'] = ((df_mex['BIRTHYR'] >= 1982) & (df_mex['BIRTHYR'] <= 1986)).astype(int)
df_mex['control'] = ((df_mex['BIRTHYR'] >= 1977) & (df_mex['BIRTHYR'] <= 1981)).astype(int)

# Keep only those in treatment or control group
df_analysis = df_mex[(df_mex['treated'] == 1) | (df_mex['control'] == 1)].copy()
print(f"\nAnalysis sample (treatment + control): {len(df_analysis):,}")
print(f"Treatment group (born 1982-1986): {df_analysis['treated'].sum():,}")
print(f"Control group (born 1977-1981): {(df_analysis['control']==1).sum():,}")

# =============================================================================
# 4. Define Time Periods
# =============================================================================
# Pre-treatment: 2006-2011 (DACA not yet in place)
# Post-treatment: 2013-2016 (DACA effects, excluding 2012 as transition year)

df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
df_analysis['pre_period'] = (df_analysis['YEAR'] < 2012).astype(int)

print(f"\nPre-period observations (2006-2011): {df_analysis['pre_period'].sum():,}")
print(f"Post-period observations (2013-2016): {df_analysis['post'].sum():,}")
print(f"Transition year 2012 observations: {(df_analysis['YEAR']==2012).sum():,}")

# =============================================================================
# 5. Define Outcome Variable: Full-Time Employment
# =============================================================================
# Full-time = usually working 35+ hours per week (UHRSWORK >= 35)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"\nFull-time employment rate: {df_analysis['fulltime'].mean():.3f}")

# Also create employment variable (EMPSTAT=1)
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)

# =============================================================================
# 6. Interaction Term for Difference-in-Differences
# =============================================================================
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

# =============================================================================
# 7. Summary Statistics
# =============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# By treatment status and time period
summary_groups = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean'],
    'AGE': ['mean'],
    'SEX': lambda x: (x==2).mean(),  # Female proportion
    'EDUCD': 'mean',
    'MARST': lambda x: (x==1).mean(),  # Married spouse present
    'PERWT': 'sum'
}).round(4)

print("\nSummary by Treatment Status and Time Period:")
print(summary_groups)

# Calculate raw DiD
pre_treat = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==0)]['fulltime'].mean()
pre_ctrl = df_analysis[(df_analysis['control']==1) & (df_analysis['post']==0)]['fulltime'].mean()
post_treat = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==1)]['fulltime'].mean()
post_ctrl = df_analysis[(df_analysis['control']==1) & (df_analysis['post']==1)]['fulltime'].mean()

raw_did = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)
print(f"\nRaw Difference-in-Differences Calculation:")
print(f"Pre-treatment, Treated: {pre_treat:.4f}")
print(f"Pre-treatment, Control: {pre_ctrl:.4f}")
print(f"Post-treatment, Treated: {post_treat:.4f}")
print(f"Post-treatment, Control: {post_ctrl:.4f}")
print(f"DiD = ({post_treat:.4f} - {pre_treat:.4f}) - ({post_ctrl:.4f} - {pre_ctrl:.4f}) = {raw_did:.4f}")

# =============================================================================
# 8. Main Regression Analysis
# =============================================================================
print("\n" + "="*70)
print("REGRESSION ANALYSIS")
print("="*70)

# Exclude 2012 for cleaner pre/post distinction
df_reg = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"\nRegression sample (excluding 2012): {len(df_reg):,}")

# Model 1: Basic DiD
print("\n--- Model 1: Basic Difference-in-Differences ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_reg).fit()
print(model1.summary2().tables[1].to_string())

# Model 2: DiD with year fixed effects
print("\n--- Model 2: DiD with Year Fixed Effects ---")
df_reg['year_factor'] = df_reg['YEAR'].astype(str)
model2 = smf.ols('fulltime ~ treated + post + treated_post + C(year_factor)', data=df_reg).fit()
print("Key coefficients:")
print(f"  treated: {model2.params['treated']:.4f} (SE: {model2.bse['treated']:.4f})")
print(f"  post: {model2.params['post']:.4f} (SE: {model2.bse['post']:.4f})")
print(f"  treated_post (DiD): {model2.params['treated_post']:.4f} (SE: {model2.bse['treated_post']:.4f})")
print(f"  p-value for DiD: {model2.pvalues['treated_post']:.4f}")

# Model 3: DiD with covariates
print("\n--- Model 3: DiD with Covariates ---")
df_reg['female'] = (df_reg['SEX'] == 2).astype(int)
df_reg['married'] = (df_reg['MARST'] == 1).astype(int)

model3 = smf.ols('fulltime ~ treated + post + treated_post + female + married + C(year_factor)',
                 data=df_reg).fit()
print("Key coefficients:")
print(f"  treated: {model3.params['treated']:.4f} (SE: {model3.bse['treated']:.4f})")
print(f"  post: {model3.params['post']:.4f} (SE: {model3.bse['post']:.4f})")
print(f"  treated_post (DiD): {model3.params['treated_post']:.4f} (SE: {model3.bse['treated_post']:.4f})")
print(f"  p-value for DiD: {model3.pvalues['treated_post']:.4f}")
print(f"  female: {model3.params['female']:.4f} (SE: {model3.bse['female']:.4f})")
print(f"  married: {model3.params['married']:.4f} (SE: {model3.bse['married']:.4f})")

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with State Fixed Effects ---")
model4 = smf.ols('fulltime ~ treated + post + treated_post + female + married + C(year_factor) + C(STATEFIP)',
                 data=df_reg).fit()
print("Key coefficients:")
print(f"  treated_post (DiD): {model4.params['treated_post']:.4f} (SE: {model4.bse['treated_post']:.4f})")
print(f"  p-value for DiD: {model4.pvalues['treated_post']:.4f}")
print(f"  R-squared: {model4.rsquared:.4f}")
print(f"  N: {int(model4.nobs):,}")

# Model 5: Weighted regression using person weights
print("\n--- Model 5: Weighted DiD with Full Controls ---")
model5 = smf.wls('fulltime ~ treated + post + treated_post + female + married + C(year_factor) + C(STATEFIP)',
                 data=df_reg, weights=df_reg['PERWT']).fit()
print("Key coefficients:")
print(f"  treated_post (DiD): {model5.params['treated_post']:.4f} (SE: {model5.bse['treated_post']:.4f})")
print(f"  p-value for DiD: {model5.pvalues['treated_post']:.4f}")
print(f"  95% CI: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")

# =============================================================================
# 9. Robustness Checks
# =============================================================================
print("\n" + "="*70)
print("ROBUSTNESS CHECKS")
print("="*70)

# Robustness 1: Alternative age bands (narrower)
print("\n--- Robustness 1: Narrower Age Bands (27-29 vs 32-34) ---")
df_narrow = df_mex.copy()
df_narrow['treated'] = ((df_narrow['BIRTHYR'] >= 1983) & (df_narrow['BIRTHYR'] <= 1985)).astype(int)
df_narrow['control'] = ((df_narrow['BIRTHYR'] >= 1978) & (df_narrow['BIRTHYR'] <= 1980)).astype(int)
df_narrow = df_narrow[(df_narrow['treated'] == 1) | (df_narrow['control'] == 1)]
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['fulltime'] = (df_narrow['UHRSWORK'] >= 35).astype(int)
df_narrow['treated_post'] = df_narrow['treated'] * df_narrow['post']
df_narrow = df_narrow[df_narrow['YEAR'] != 2012]
df_narrow['female'] = (df_narrow['SEX'] == 2).astype(int)
df_narrow['married'] = (df_narrow['MARST'] == 1).astype(int)

model_narrow = smf.ols('fulltime ~ treated + post + treated_post + female + married + C(YEAR)',
                       data=df_narrow).fit()
print(f"  treated_post (DiD): {model_narrow.params['treated_post']:.4f} (SE: {model_narrow.bse['treated_post']:.4f})")
print(f"  p-value: {model_narrow.pvalues['treated_post']:.4f}")
print(f"  N: {int(model_narrow.nobs):,}")

# Robustness 2: Placebo test using years 2006-2008 vs 2009-2011
print("\n--- Robustness 2: Placebo Test (Pre-DACA period) ---")
df_placebo = df_analysis[(df_analysis['YEAR'] >= 2006) & (df_analysis['YEAR'] <= 2011)].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treated_post_placebo'] = df_placebo['treated'] * df_placebo['post_placebo']
df_placebo['female'] = (df_placebo['SEX'] == 2).astype(int)
df_placebo['married'] = (df_placebo['MARST'] == 1).astype(int)

model_placebo = smf.ols('fulltime ~ treated + post_placebo + treated_post_placebo + female + married + C(YEAR)',
                        data=df_placebo).fit()
print(f"  treated_post_placebo (DiD): {model_placebo.params['treated_post_placebo']:.4f} (SE: {model_placebo.bse['treated_post_placebo']:.4f})")
print(f"  p-value: {model_placebo.pvalues['treated_post_placebo']:.4f}")
print(f"  N: {int(model_placebo.nobs):,}")

# Robustness 3: Event study
print("\n--- Robustness 3: Event Study ---")
df_event = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_event['female'] = (df_event['SEX'] == 2).astype(int)
df_event['married'] = (df_event['MARST'] == 1).astype(int)

# Create year-treatment interactions (2011 as reference year)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_event[f'treated_year_{year}'] = (df_event['treated'] * (df_event['YEAR'] == year)).astype(int)

formula_event = 'fulltime ~ treated + C(YEAR) + female + married + ' + ' + '.join([f'treated_year_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])
model_event = smf.ols(formula_event, data=df_event).fit()

print("Year-Treatment Interactions (reference: 2011):")
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treated_year_{year}']
    se = model_event.bse[f'treated_year_{year}']
    pval = model_event.pvalues[f'treated_year_{year}']
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# =============================================================================
# 10. Heterogeneity Analysis
# =============================================================================
print("\n" + "="*70)
print("HETEROGENEITY ANALYSIS")
print("="*70)

# By gender
print("\n--- Heterogeneity by Gender ---")
df_male = df_reg[df_reg['SEX'] == 1]
df_female = df_reg[df_reg['SEX'] == 2]

model_male = smf.ols('fulltime ~ treated + post + treated_post + married + C(YEAR)', data=df_male).fit()
model_female = smf.ols('fulltime ~ treated + post + treated_post + married + C(YEAR)', data=df_female).fit()

print(f"  Males - DiD: {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f}, p={model_male.pvalues['treated_post']:.4f})")
print(f"  Females - DiD: {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f}, p={model_female.pvalues['treated_post']:.4f})")

# By education
print("\n--- Heterogeneity by Education ---")
# EDUCD: Below high school (< 62), High school or above (>= 62)
df_low_ed = df_reg[df_reg['EDUCD'] < 62]
df_high_ed = df_reg[df_reg['EDUCD'] >= 62]

model_low = smf.ols('fulltime ~ treated + post + treated_post + female + married + C(YEAR)', data=df_low_ed).fit()
model_high = smf.ols('fulltime ~ treated + post + treated_post + female + married + C(YEAR)', data=df_high_ed).fit()

print(f"  Low education - DiD: {model_low.params['treated_post']:.4f} (SE: {model_low.bse['treated_post']:.4f}, p={model_low.pvalues['treated_post']:.4f})")
print(f"  High education - DiD: {model_high.params['treated_post']:.4f} (SE: {model_high.bse['treated_post']:.4f}, p={model_high.pvalues['treated_post']:.4f})")

# =============================================================================
# 11. Save Key Results for Report
# =============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Create results dictionary
results = {
    'sample_size': len(df_reg),
    'n_treated': df_reg['treated'].sum(),
    'n_control': (df_reg['control']==1).sum(),
    'did_estimate': model4.params['treated_post'],
    'did_se': model4.bse['treated_post'],
    'did_pvalue': model4.pvalues['treated_post'],
    'did_ci_lower': model4.conf_int().loc['treated_post', 0],
    'did_ci_upper': model4.conf_int().loc['treated_post', 1],
    'r_squared': model4.rsquared,
    'raw_did': raw_did,
    'pre_treat_mean': pre_treat,
    'pre_ctrl_mean': pre_ctrl,
    'post_treat_mean': post_treat,
    'post_ctrl_mean': post_ctrl
}

# Save results to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# =============================================================================
# 12. Generate Tables and Figures
# =============================================================================

# Table 1: Sample characteristics
print("\n--- Generating Table 1: Sample Characteristics ---")
table1_data = []

for grp_name, grp_df in [('Treatment', df_reg[df_reg['treated']==1]),
                          ('Control', df_reg[df_reg['control']==1])]:
    for period_name, period_df in [('Pre', grp_df[grp_df['post']==0]),
                                    ('Post', grp_df[grp_df['post']==1])]:
        row = {
            'Group': grp_name,
            'Period': period_name,
            'N': len(period_df),
            'Full-time Rate': period_df['fulltime'].mean(),
            'Employment Rate': period_df['employed'].mean(),
            'Mean Age': period_df['AGE'].mean(),
            'Female Share': (period_df['SEX']==2).mean(),
            'Married Share': (period_df['MARST']==1).mean()
        }
        table1_data.append(row)

table1 = pd.DataFrame(table1_data)
table1.to_csv('table1_summary_stats.csv', index=False)
print(table1)

# Table 2: Main regression results
print("\n--- Generating Table 2: Main DiD Results ---")
table2_data = {
    'Model': ['(1) Basic DiD', '(2) Year FE', '(3) + Covariates', '(4) + State FE', '(5) Weighted'],
    'DiD Estimate': [model1.params['treated_post'], model2.params['treated_post'],
                     model3.params['treated_post'], model4.params['treated_post'],
                     model5.params['treated_post']],
    'Std Error': [model1.bse['treated_post'], model2.bse['treated_post'],
                  model3.bse['treated_post'], model4.bse['treated_post'],
                  model5.bse['treated_post']],
    'P-value': [model1.pvalues['treated_post'], model2.pvalues['treated_post'],
                model3.pvalues['treated_post'], model4.pvalues['treated_post'],
                model5.pvalues['treated_post']],
    'R-squared': [model1.rsquared, model2.rsquared, model3.rsquared,
                  model4.rsquared, model5.rsquared],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs)]
}
table2 = pd.DataFrame(table2_data)
table2.to_csv('table2_main_results.csv', index=False)
print(table2)

# Figure 1: Event study plot
print("\n--- Generating Figure 1: Event Study ---")
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [0]  # 2011 is reference
ses = [0]
for year in [2006, 2007, 2008, 2009, 2010]:
    coefs.insert(len([y for y in [2006, 2007, 2008, 2009, 2010] if y <= year]) - 1,
                 model_event.params[f'treated_year_{year}'])
    ses.insert(len([y for y in [2006, 2007, 2008, 2009, 2010] if y <= year]) - 1,
               model_event.bse[f'treated_year_{year}'])
for year in [2013, 2014, 2015, 2016]:
    coefs.append(model_event.params[f'treated_year_{year}'])
    ses.append(model_event.bse[f'treated_year_{year}'])

years_plot = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs_ordered = []
ses_ordered = []
for year in years_plot:
    if year == 2011:
        coefs_ordered.append(0)
        ses_ordered.append(0)
    else:
        coefs_ordered.append(model_event.params[f'treated_year_{year}'])
        ses_ordered.append(model_event.bse[f'treated_year_{year}'])

plt.figure(figsize=(10, 6))
plt.errorbar(years_plot, coefs_ordered, yerr=[1.96*s for s in ses_ordered],
             fmt='o-', capsize=4, capthick=2)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
plt.xlabel('Year')
plt.ylabel('Treatment Effect (relative to 2011)')
plt.title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
plt.legend()
plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300)
plt.savefig('figure1_event_study.pdf')
print("Event study figure saved")

# Figure 2: Trends by group
print("\n--- Generating Figure 2: Trends by Group ---")
yearly_means = df_analysis.groupby(['YEAR', 'treated']).agg({
    'fulltime': 'mean',
    'employed': 'mean'
}).reset_index()

plt.figure(figsize=(10, 6))
treat_data = yearly_means[yearly_means['treated'] == 1]
ctrl_data = yearly_means[yearly_means['treated'] == 0]
plt.plot(treat_data['YEAR'], treat_data['fulltime'], 'b-o', label='Treatment (Age 26-30)', linewidth=2)
plt.plot(ctrl_data['YEAR'], ctrl_data['fulltime'], 'r-s', label='Control (Age 31-35)', linewidth=2)
plt.axvline(x=2012.5, color='gray', linestyle='--', alpha=0.7, label='DACA Implementation')
plt.xlabel('Year')
plt.ylabel('Full-Time Employment Rate')
plt.title('Full-Time Employment Trends by Treatment Status')
plt.legend()
plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300)
plt.savefig('figure2_trends.pdf')
print("Trends figure saved")

# Save event study data for LaTeX
event_study_df = pd.DataFrame({
    'Year': years_plot,
    'Coefficient': coefs_ordered,
    'SE': ses_ordered,
    'CI_lower': [c - 1.96*s for c, s in zip(coefs_ordered, ses_ordered)],
    'CI_upper': [c + 1.96*s for c, s in zip(coefs_ordered, ses_ordered)]
})
event_study_df.to_csv('event_study_results.csv', index=False)

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

# Print final summary
print(f"""
PREFERRED ESTIMATE SUMMARY
==========================
Effect Size (DiD): {model4.params['treated_post']:.4f}
Standard Error: {model4.bse['treated_post']:.4f}
95% Confidence Interval: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]
P-value: {model4.pvalues['treated_post']:.4f}
Sample Size: {int(model4.nobs):,}
R-squared: {model4.rsquared:.4f}

Interpretation: DACA eligibility is associated with a {model4.params['treated_post']:.4f}
({model4.params['treated_post']*100:.2f} percentage point) change in the probability
of full-time employment among eligible individuals.
""")
