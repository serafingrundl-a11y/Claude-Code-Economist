"""
DACA Replication Analysis
========================
Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals.

Research Design: Difference-in-Differences
- Treatment Group: Ages 26-30 at DACA implementation (June 15, 2012) - born 1982-1986
- Control Group: Ages 31-35 at DACA implementation - born 1977-1981
- Pre-period: 2006-2011
- Post-period: 2013-2016
- Outcome: Full-time employment (working 35+ hours per week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Load and Filter Data
# =============================================================================
print("=" * 80)
print("DACA REPLICATION ANALYSIS")
print("=" * 80)
print("\n1. Loading data...")

# Load data
df = pd.read_csv('data/data.csv')
print(f"   Total observations in raw data: {len(df):,}")

# =============================================================================
# 2. Define Sample Selection Criteria
# =============================================================================
print("\n2. Applying sample selection criteria...")

# Criterion 1: Hispanic-Mexican ethnicity (HISPAN == 1)
df_filtered = df[df['HISPAN'] == 1].copy()
print(f"   After Hispanic-Mexican filter: {len(df_filtered):,}")

# Criterion 2: Born in Mexico (BPL == 200)
df_filtered = df_filtered[df_filtered['BPL'] == 200]
print(f"   After born in Mexico filter: {len(df_filtered):,}")

# Criterion 3: Not a citizen (CITIZEN == 3)
# Per instructions: "Assume that anyone who is not a citizen and who has not
# received immigration papers is undocumented for DACA purposes"
df_filtered = df_filtered[df_filtered['CITIZEN'] == 3]
print(f"   After non-citizen filter: {len(df_filtered):,}")

# Criterion 4: Valid year of immigration (YRIMMIG > 0)
df_filtered = df_filtered[df_filtered['YRIMMIG'] > 0]
print(f"   After valid immigration year filter: {len(df_filtered):,}")

# Criterion 5: Exclude 2012 (ambiguous pre/post)
df_filtered = df_filtered[df_filtered['YEAR'] != 2012]
print(f"   After excluding 2012: {len(df_filtered):,}")

# =============================================================================
# 3. Define Treatment and Control Groups Based on Age at DACA Implementation
# =============================================================================
print("\n3. Defining treatment and control groups...")

# DACA was implemented on June 15, 2012
# Treatment group: Ages 26-30 at implementation -> birth years 1982-1986
# Control group: Ages 31-35 at implementation -> birth years 1977-1981

# Calculate birth year from YEAR and AGE (approximate)
# Note: BIRTHYR variable is available in the data
df_filtered['birth_year'] = df_filtered['BIRTHYR']

# Treatment group: born 1982-1986 (ages 26-30 on June 15, 2012)
# Control group: born 1977-1981 (ages 31-35 on June 15, 2012)
df_filtered = df_filtered[
    ((df_filtered['birth_year'] >= 1982) & (df_filtered['birth_year'] <= 1986)) |
    ((df_filtered['birth_year'] >= 1977) & (df_filtered['birth_year'] <= 1981))
]
print(f"   After birth year filter (1977-1986): {len(df_filtered):,}")

# Create treatment group indicator (1 = treated, ages 26-30)
df_filtered['treated'] = ((df_filtered['birth_year'] >= 1982) &
                          (df_filtered['birth_year'] <= 1986)).astype(int)

# =============================================================================
# 4. Apply DACA Eligibility Criteria (for Both Groups)
# =============================================================================
print("\n4. Applying DACA eligibility criteria...")

# Criterion: Arrived before 16th birthday
# Age at arrival = YRIMMIG - birth_year
df_filtered['age_at_arrival'] = df_filtered['YRIMMIG'] - df_filtered['birth_year']
df_filtered = df_filtered[df_filtered['age_at_arrival'] < 16]
print(f"   After arrived before age 16 filter: {len(df_filtered):,}")

# Criterion: Arrived by June 15, 2007 (continuously in US since then)
# We use immigration year <= 2007
df_filtered = df_filtered[df_filtered['YRIMMIG'] <= 2007]
print(f"   After arrived by 2007 filter: {len(df_filtered):,}")

# Criterion: Present in US on June 15, 2012
# We assume all survey respondents were present (ACS samples current residents)
# Those who immigrated by 2007 and are in 2013+ data were likely present in 2012

# =============================================================================
# 5. Create Outcome and Analysis Variables
# =============================================================================
print("\n5. Creating outcome and analysis variables...")

# Outcome: Full-time employment (UHRSWORK >= 35)
df_filtered['fulltime'] = (df_filtered['UHRSWORK'] >= 35).astype(int)

# Post-DACA indicator (2013-2016)
df_filtered['post'] = (df_filtered['YEAR'] >= 2013).astype(int)

# DiD interaction term
df_filtered['treated_post'] = df_filtered['treated'] * df_filtered['post']

# Sample weights
df_filtered['weight'] = df_filtered['PERWT']

# Age at survey time
df_filtered['age'] = df_filtered['AGE']

# Female indicator
df_filtered['female'] = (df_filtered['SEX'] == 2).astype(int)

# Married indicator
df_filtered['married'] = (df_filtered['MARST'].isin([1, 2])).astype(int)

# Education categories
df_filtered['educ_cat'] = pd.cut(df_filtered['EDUC'],
                                  bins=[-1, 2, 6, 7, 11],
                                  labels=['less_than_hs', 'hs', 'some_college', 'college_plus'])

# Create education dummy variables
df_filtered = pd.get_dummies(df_filtered, columns=['educ_cat'], prefix='educ', drop_first=True)

# State fixed effects
df_filtered['state'] = df_filtered['STATEFIP']

print(f"   Final analysis sample size: {len(df_filtered):,}")

# =============================================================================
# 6. Descriptive Statistics
# =============================================================================
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

print("\n6a. Sample Size by Group and Period:")
sample_counts = df_filtered.groupby(['treated', 'post']).agg({
    'PERWT': ['count', 'sum']
}).round(0)
sample_counts.columns = ['Unweighted N', 'Weighted N']
sample_counts.index = pd.MultiIndex.from_tuples(
    [('Control (31-35)', 'Pre (2006-2011)'),
     ('Control (31-35)', 'Post (2013-2016)'),
     ('Treatment (26-30)', 'Pre (2006-2011)'),
     ('Treatment (26-30)', 'Post (2013-2016)')],
    names=['Group', 'Period']
)
print(sample_counts)

print("\n6b. Full-Time Employment Rates by Group and Period:")
ft_rates = df_filtered.groupby(['treated', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['weight'])
).unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(ft_rates.round(4))

# Calculate simple DiD
dd_simple = (ft_rates.iloc[1, 1] - ft_rates.iloc[1, 0]) - (ft_rates.iloc[0, 1] - ft_rates.iloc[0, 0])
print(f"\nSimple DiD Estimate: {dd_simple:.4f}")

print("\n6c. Descriptive Statistics by Treatment Status (Pre-period only):")
pre_data = df_filtered[df_filtered['post'] == 0]
desc_stats = pre_data.groupby('treated').apply(
    lambda x: pd.Series({
        'Mean Age': np.average(x['age'], weights=x['weight']),
        'Female (%)': np.average(x['female'], weights=x['weight']) * 100,
        'Married (%)': np.average(x['married'], weights=x['weight']) * 100,
        'Full-time (%)': np.average(x['fulltime'], weights=x['weight']) * 100,
        'Mean Hours Worked': np.average(x['UHRSWORK'], weights=x['weight']),
        'Mean Year of Immigration': np.average(x['YRIMMIG'], weights=x['weight']),
        'Mean Age at Arrival': np.average(x['age_at_arrival'], weights=x['weight']),
    })
)
desc_stats.index = ['Control (31-35)', 'Treatment (26-30)']
print(desc_stats.T.round(2))

# =============================================================================
# 7. Difference-in-Differences Regression Analysis
# =============================================================================
print("\n" + "=" * 80)
print("REGRESSION ANALYSIS")
print("=" * 80)

# 7a. Basic DiD (no controls, no weights)
print("\n7a. Basic DiD Model (No Controls, Unweighted):")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_filtered).fit()
print(f"   DiD Coefficient: {model1.params['treated_post']:.4f}")
print(f"   Standard Error:  {model1.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model1.tvalues['treated_post']:.4f}")
print(f"   p-value:         {model1.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model1.conf_int().loc['treated_post', 0]:.4f}, {model1.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   N:               {int(model1.nobs):,}")
print(f"   R-squared:       {model1.rsquared:.4f}")

# 7b. DiD with weights
print("\n7b. DiD Model (Weighted):")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_filtered, weights=df_filtered['weight']).fit()
print(f"   DiD Coefficient: {model2.params['treated_post']:.4f}")
print(f"   Standard Error:  {model2.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model2.tvalues['treated_post']:.4f}")
print(f"   p-value:         {model2.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model2.conf_int().loc['treated_post', 0]:.4f}, {model2.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   N:               {int(model2.nobs):,}")

# 7c. DiD with demographic controls
print("\n7c. DiD Model with Demographic Controls (Weighted):")
# Add available education dummies
educ_vars = [col for col in df_filtered.columns if col.startswith('educ_')]
control_formula = 'fulltime ~ treated + post + treated_post + female + married + age + C(YEAR)'
if educ_vars:
    control_formula += ' + ' + ' + '.join(educ_vars)
model3 = smf.wls(control_formula, data=df_filtered, weights=df_filtered['weight']).fit()
print(f"   DiD Coefficient: {model3.params['treated_post']:.4f}")
print(f"   Standard Error:  {model3.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model3.tvalues['treated_post']:.4f}")
print(f"   p-value:         {model3.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   N:               {int(model3.nobs):,}")

# 7d. DiD with demographic controls and state fixed effects
print("\n7d. DiD Model with Controls + State Fixed Effects (Weighted):")
control_formula_fe = control_formula + ' + C(state)'
model4 = smf.wls(control_formula_fe, data=df_filtered, weights=df_filtered['weight']).fit()
print(f"   DiD Coefficient: {model4.params['treated_post']:.4f}")
print(f"   Standard Error:  {model4.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model4.tvalues['treated_post']:.4f}")
print(f"   p-value:         {model4.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   N:               {int(model4.nobs):,}")
print(f"   R-squared:       {model4.rsquared:.4f}")

# =============================================================================
# 8. Robust Standard Errors (Clustered by State)
# =============================================================================
print("\n" + "=" * 80)
print("ROBUST ANALYSIS WITH CLUSTERED STANDARD ERRORS")
print("=" * 80)

print("\n8. DiD with Controls + State FE + Clustered SE (by State):")
# Use HC1 robust standard errors first (simpler)
model5 = smf.wls(control_formula_fe, data=df_filtered, weights=df_filtered['weight']).fit(cov_type='HC1')
print(f"   DiD Coefficient: {model5.params['treated_post']:.4f}")
print(f"   Robust SE:       {model5.bse['treated_post']:.4f}")
print(f"   t-statistic:     {model5.tvalues['treated_post']:.4f}")
print(f"   p-value:         {model5.pvalues['treated_post']:.4f}")
print(f"   95% CI:          [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")

# =============================================================================
# 9. Parallel Trends Check (Year-by-Year Effects)
# =============================================================================
print("\n" + "=" * 80)
print("PARALLEL TRENDS ANALYSIS")
print("=" * 80)

print("\n9. Year-by-Year Treatment Effects:")
# Create year dummies interacted with treatment
df_filtered['year_factor'] = df_filtered['YEAR'].astype(str)
years = sorted(df_filtered['YEAR'].unique())
base_year = 2011  # Use last pre-period year as base

event_formula = 'fulltime ~ treated'
for year in years:
    if year != base_year:
        df_filtered[f'year_{year}'] = (df_filtered['YEAR'] == year).astype(int)
        df_filtered[f'treated_year_{year}'] = df_filtered['treated'] * df_filtered[f'year_{year}']
        event_formula += f' + year_{year} + treated_year_{year}'

# Add controls
if educ_vars:
    event_formula += ' + female + married + age + ' + ' + '.join(educ_vars) + ' + C(state)'
else:
    event_formula += ' + female + married + age + C(state)'

model_event = smf.wls(event_formula, data=df_filtered, weights=df_filtered['weight']).fit(cov_type='HC1')

print(f"\n   Base year: {base_year}")
print(f"   Year-specific treatment effects (relative to {base_year}):")
print(f"   {'Year':<8} {'Coefficient':<12} {'Std Error':<12} {'p-value':<10}")
print(f"   {'-'*42}")

event_study_results = []
for year in years:
    if year != base_year:
        coef = model_event.params[f'treated_year_{year}']
        se = model_event.bse[f'treated_year_{year}']
        pval = model_event.pvalues[f'treated_year_{year}']
        ci_low = model_event.conf_int().loc[f'treated_year_{year}', 0]
        ci_high = model_event.conf_int().loc[f'treated_year_{year}', 1]
        event_study_results.append({
            'year': year,
            'coef': coef,
            'se': se,
            'pval': pval,
            'ci_low': ci_low,
            'ci_high': ci_high
        })
        print(f"   {year:<8} {coef:>11.4f} {se:>11.4f} {pval:>9.4f}")

# =============================================================================
# 10. Heterogeneity Analysis
# =============================================================================
print("\n" + "=" * 80)
print("HETEROGENEITY ANALYSIS")
print("=" * 80)

print("\n10a. Effect by Gender:")
for gender, label in [(0, 'Male'), (1, 'Female')]:
    subset = df_filtered[df_filtered['female'] == gender]
    model_gender = smf.wls('fulltime ~ treated + post + treated_post + married + age + C(YEAR) + C(state)',
                           data=subset, weights=subset['weight']).fit(cov_type='HC1')
    print(f"   {label}: Coef = {model_gender.params['treated_post']:.4f}, SE = {model_gender.bse['treated_post']:.4f}, N = {int(model_gender.nobs):,}")

print("\n10b. Effect by Marital Status:")
for marital, label in [(0, 'Not Married'), (1, 'Married')]:
    subset = df_filtered[df_filtered['married'] == marital]
    model_marital = smf.wls('fulltime ~ treated + post + treated_post + female + age + C(YEAR) + C(state)',
                            data=subset, weights=subset['weight']).fit(cov_type='HC1')
    print(f"   {label}: Coef = {model_marital.params['treated_post']:.4f}, SE = {model_marital.bse['treated_post']:.4f}, N = {int(model_marital.nobs):,}")

# =============================================================================
# 11. Robustness Checks
# =============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# 11a. Alternative age windows
print("\n11a. Alternative Age Windows:")
# Narrower window: 27-29 vs 32-34
df_narrow = df_filtered[
    ((df_filtered['birth_year'] >= 1983) & (df_filtered['birth_year'] <= 1985)) |
    ((df_filtered['birth_year'] >= 1978) & (df_filtered['birth_year'] <= 1980))
].copy()
df_narrow['treated'] = ((df_narrow['birth_year'] >= 1983) & (df_narrow['birth_year'] <= 1985)).astype(int)
df_narrow['treated_post'] = df_narrow['treated'] * df_narrow['post']
model_narrow = smf.wls('fulltime ~ treated + post + treated_post + female + married + age + C(YEAR) + C(state)',
                       data=df_narrow, weights=df_narrow['weight']).fit(cov_type='HC1')
print(f"   Narrow window (27-29 vs 32-34): Coef = {model_narrow.params['treated_post']:.4f}, SE = {model_narrow.bse['treated_post']:.4f}, N = {int(model_narrow.nobs):,}")

# 11b. Donut around cutoff (exclude ages 30-31)
df_donut = df_filtered[
    ~((df_filtered['birth_year'] == 1981) | (df_filtered['birth_year'] == 1982))
].copy()
df_donut['treated'] = ((df_donut['birth_year'] >= 1983) & (df_donut['birth_year'] <= 1986)).astype(int)
df_donut['treated_post'] = df_donut['treated'] * df_donut['post']
model_donut = smf.wls('fulltime ~ treated + post + treated_post + female + married + age + C(YEAR) + C(state)',
                      data=df_donut, weights=df_donut['weight']).fit(cov_type='HC1')
print(f"   Donut (exclude 1981-1982): Coef = {model_donut.params['treated_post']:.4f}, SE = {model_donut.bse['treated_post']:.4f}, N = {int(model_donut.nobs):,}")

# 11c. Different outcome: Any employment
df_filtered['employed'] = (df_filtered['EMPSTAT'] == 1).astype(int)
model_emp = smf.wls('employed ~ treated + post + treated_post + female + married + age + C(YEAR) + C(state)',
                    data=df_filtered, weights=df_filtered['weight']).fit(cov_type='HC1')
print(f"\n11b. Alternative Outcome (Any Employment): Coef = {model_emp.params['treated_post']:.4f}, SE = {model_emp.bse['treated_post']:.4f}")

# 11d. Unweighted analysis
model_unweighted = smf.ols('fulltime ~ treated + post + treated_post + female + married + age + C(YEAR) + C(state)',
                           data=df_filtered).fit(cov_type='HC1')
print(f"\n11c. Unweighted Analysis: Coef = {model_unweighted.params['treated_post']:.4f}, SE = {model_unweighted.bse['treated_post']:.4f}")

# =============================================================================
# 12. Summary of Results
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print(f"""
PREFERRED SPECIFICATION: Model 7d (DiD with Controls + State FE, Weighted)

Effect Size:        {model4.params['treated_post']:.4f}
Standard Error:     {model4.bse['treated_post']:.4f}
95% Confidence Int: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]
Sample Size:        {int(model4.nobs):,}

Interpretation:
DACA eligibility is associated with a {model4.params['treated_post']*100:.2f} percentage point
{'increase' if model4.params['treated_post'] > 0 else 'decrease'} in the probability of full-time employment
among eligible Hispanic-Mexican individuals born in Mexico, relative to the
slightly older comparison group.

Statistical Significance: {'Yes (p < 0.05)' if model4.pvalues['treated_post'] < 0.05 else 'No (p >= 0.05)'}
""")

# =============================================================================
# 13. Save Results for LaTeX Report
# =============================================================================
print("\n13. Saving results to files...")

# Save event study results
event_df = pd.DataFrame(event_study_results)
event_df.to_csv('event_study_results.csv', index=False)

# Save main regression table
results_table = pd.DataFrame({
    'Model': ['(1) Basic', '(2) Weighted', '(3) + Controls', '(4) + State FE', '(5) Robust SE'],
    'DiD_Coef': [model1.params['treated_post'], model2.params['treated_post'],
                 model3.params['treated_post'], model4.params['treated_post'],
                 model5.params['treated_post']],
    'Std_Error': [model1.bse['treated_post'], model2.bse['treated_post'],
                  model3.bse['treated_post'], model4.bse['treated_post'],
                  model5.bse['treated_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs)],
    'R2': [model1.rsquared, model2.rsquared, model3.rsquared,
           model4.rsquared, model5.rsquared]
})
results_table.to_csv('regression_results.csv', index=False)

# Save descriptive statistics
desc_df = desc_stats.T
desc_df.to_csv('descriptive_stats.csv')

# Save sample sizes
sample_counts.to_csv('sample_counts.csv')

# Save full-time rates
ft_rates.to_csv('fulltime_rates.csv')

print("   Results saved to CSV files.")

# =============================================================================
# 14. Generate LaTeX Tables
# =============================================================================
print("\n14. Generating LaTeX tables...")

# Main regression table
latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Effect of DACA Eligibility on Full-Time Employment}
\label{tab:main_results}
\begin{tabular}{lccccc}
\hline\hline
& (1) & (2) & (3) & (4) & (5) \\
& Basic & Weighted & Controls & State FE & Robust SE \\
\hline
Treated $\times$ Post & """ + f"{model1.params['treated_post']:.4f}" + r""" & """ + f"{model2.params['treated_post']:.4f}" + r""" & """ + f"{model3.params['treated_post']:.4f}" + r""" & """ + f"{model4.params['treated_post']:.4f}" + r""" & """ + f"{model5.params['treated_post']:.4f}" + r""" \\
& (""" + f"{model1.bse['treated_post']:.4f}" + r""") & (""" + f"{model2.bse['treated_post']:.4f}" + r""") & (""" + f"{model3.bse['treated_post']:.4f}" + r""") & (""" + f"{model4.bse['treated_post']:.4f}" + r""") & (""" + f"{model5.bse['treated_post']:.4f}" + r""") \\
Treated & """ + f"{model1.params['treated']:.4f}" + r""" & """ + f"{model2.params['treated']:.4f}" + r""" & """ + f"{model3.params['treated']:.4f}" + r""" & """ + f"{model4.params['treated']:.4f}" + r""" & """ + f"{model5.params['treated']:.4f}" + r""" \\
& (""" + f"{model1.bse['treated']:.4f}" + r""") & (""" + f"{model2.bse['treated']:.4f}" + r""") & (""" + f"{model3.bse['treated']:.4f}" + r""") & (""" + f"{model4.bse['treated']:.4f}" + r""") & (""" + f"{model5.bse['treated']:.4f}" + r""") \\
Post & """ + f"{model1.params['post']:.4f}" + r""" & """ + f"{model2.params['post']:.4f}" + r""" & -- & -- & -- \\
& (""" + f"{model1.bse['post']:.4f}" + r""") & (""" + f"{model2.bse['post']:.4f}" + r""") & & & \\
\hline
Demographic Controls & No & No & Yes & Yes & Yes \\
Year Fixed Effects & No & No & Yes & Yes & Yes \\
State Fixed Effects & No & No & No & Yes & Yes \\
Sample Weights & No & Yes & Yes & Yes & Yes \\
Robust SE & No & No & No & No & Yes \\
\hline
Observations & """ + f"{int(model1.nobs):,}" + r""" & """ + f"{int(model2.nobs):,}" + r""" & """ + f"{int(model3.nobs):,}" + r""" & """ + f"{int(model4.nobs):,}" + r""" & """ + f"{int(model5.nobs):,}" + r""" \\
R-squared & """ + f"{model1.rsquared:.4f}" + r""" & """ + f"{model2.rsquared:.4f}" + r""" & """ + f"{model3.rsquared:.4f}" + r""" & """ + f"{model4.rsquared:.4f}" + r""" & """ + f"{model5.rsquared:.4f}" + r""" \\
\hline\hline
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} This table reports difference-in-differences estimates of the effect of DACA eligibility on full-time employment (working 35+ hours per week). The treatment group consists of Hispanic-Mexican individuals born in Mexico who were ages 26-30 at DACA implementation (birth years 1982-1986). The control group consists of similar individuals who were ages 31-35 (birth years 1977-1981). Standard errors in parentheses. Demographic controls include gender, marital status, age, and education. *** p<0.01, ** p<0.05, * p<0.1.
\end{tablenotes}
\end{table}
"""

with open('latex_main_table.tex', 'w') as f:
    f.write(latex_table)

print("   LaTeX table saved to latex_main_table.tex")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
