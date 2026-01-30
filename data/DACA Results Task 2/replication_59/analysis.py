#!/usr/bin/env python3
"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals.

Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
Control: Ages 31-35 at DACA implementation
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 2\replication_59")

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

# Read data in chunks due to large file size
chunks = []
chunk_size = 500000

dtypes = {
    'YEAR': 'int16',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'UHRSWORK': 'int8',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int8',
    'EDUC': 'int8',
    'MARST': 'int8',
    'STATEFIP': 'int8',
    'EMPSTAT': 'int8'
}

usecols = ['YEAR', 'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
           'UHRSWORK', 'PERWT', 'SEX', 'AGE', 'EDUC', 'MARST', 'STATEFIP', 'EMPSTAT']

for chunk in pd.read_csv('data/data.csv', chunksize=chunk_size, usecols=usecols,
                         dtype={k: v for k, v in dtypes.items() if k in usecols}):
    # Initial filter: Mexican Hispanic, born in Mexico
    chunk_filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk_filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"  - Loaded {len(df):,} Mexican-born Hispanic-Mexican individuals")

# =============================================================================
# 2. SAMPLE CONSTRUCTION
# =============================================================================
print("\n[2] Constructing sample...")

# DACA implementation date reference: June 15, 2012
# Treatment group: Ages 26-30 as of June 15, 2012 -> born 1982-1986
# Control group: Ages 31-35 as of June 15, 2012 -> born 1977-1981

# Age calculation: person's age on June 15, 2012
# Born in 1982 -> turns 30 in 2012 (if birthday before June 15, age 30; else age 29)
# For simplicity, we use birth year ranges as per instructions

# Filter: Non-citizens (proxy for undocumented)
df = df[df['CITIZEN'] == 3]
print(f"  - After citizenship filter (non-citizen): {len(df):,}")

# Filter: Arrived before age 16
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[(df['age_at_arrival'] > 0) & (df['age_at_arrival'] < 16)]
print(f"  - After arrival age filter (<16): {len(df):,}")

# Filter: Arrived by 2007 (continuous presence since June 15, 2007)
df = df[df['YRIMMIG'] <= 2007]
print(f"  - After continuous presence filter (arrived <=2007): {len(df):,}")

# Define treatment and control groups based on birth year
# Treatment: born 1982-1986 (ages 26-30 on June 15, 2012)
# Control: born 1977-1981 (ages 31-35 on June 15, 2012)
df['treatment'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)
df['control_group'] = ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control groups
df = df[(df['treatment'] == 1) | (df['control_group'] == 1)]
print(f"  - After age group filter (born 1977-1986): {len(df):,}")

# Define pre/post periods
# Pre: 2006-2011 (before DACA)
# Post: 2013-2016 (after DACA - using 2013-2016 as specified)
# Exclude 2012 as DACA was implemented mid-year
df = df[df['YEAR'] != 2012]
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"  - After excluding 2012: {len(df):,}")

# Define outcome: Full-time employment (usually works 35+ hours/week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Create DiD interaction term
df['treat_post'] = df['treatment'] * df['post']

# Store sample sizes
sample_info = {
    'total_sample': len(df),
    'treatment_pre': len(df[(df['treatment'] == 1) & (df['post'] == 0)]),
    'treatment_post': len(df[(df['treatment'] == 1) & (df['post'] == 1)]),
    'control_pre': len(df[(df['treatment'] == 0) & (df['post'] == 0)]),
    'control_post': len(df[(df['treatment'] == 0) & (df['post'] == 1)])
}

print(f"\n  Sample breakdown:")
print(f"    Treatment (ages 26-30), Pre-DACA:  {sample_info['treatment_pre']:,}")
print(f"    Treatment (ages 26-30), Post-DACA: {sample_info['treatment_post']:,}")
print(f"    Control (ages 31-35), Pre-DACA:    {sample_info['control_pre']:,}")
print(f"    Control (ages 31-35), Post-DACA:   {sample_info['control_post']:,}")
print(f"    Total sample:                      {sample_info['total_sample']:,}")

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[3] Computing descriptive statistics...")

# Pre-period means by group
pre_data = df[df['post'] == 0]
post_data = df[df['post'] == 1]

# Weighted means
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

def weighted_std(data, value_col, weight_col):
    avg = weighted_mean(data, value_col, weight_col)
    variance = np.average((data[value_col] - avg)**2, weights=data[weight_col])
    return np.sqrt(variance)

# Summary statistics by group and period
summary_stats = {}
for period, period_name in [(0, 'pre'), (1, 'post')]:
    for treat, group_name in [(1, 'treatment'), (0, 'control')]:
        subset = df[(df['post'] == period) & (df['treatment'] == treat)]
        key = f"{group_name}_{period_name}"
        working_subset = subset[subset['UHRSWORK'] > 0]
        summary_stats[key] = {
            'fulltime_rate': weighted_mean(subset, 'fulltime', 'PERWT'),
            'fulltime_rate_se': weighted_std(subset, 'fulltime', 'PERWT') / np.sqrt(len(subset)),
            'n': len(subset),
            'mean_age': weighted_mean(subset, 'AGE', 'PERWT'),
            'female_share': weighted_mean(subset, 'SEX', 'PERWT') - 1,  # SEX: 1=male, 2=female
            'mean_uhrs': np.average(working_subset['UHRSWORK'], weights=working_subset['PERWT']) if len(working_subset) > 0 else 0
        }

print("\n  Full-time employment rates:")
print(f"                    Pre-DACA    Post-DACA   Change")
print(f"    Treatment:      {summary_stats['treatment_pre']['fulltime_rate']:.3f}       {summary_stats['treatment_post']['fulltime_rate']:.3f}        {summary_stats['treatment_post']['fulltime_rate'] - summary_stats['treatment_pre']['fulltime_rate']:+.3f}")
print(f"    Control:        {summary_stats['control_pre']['fulltime_rate']:.3f}       {summary_stats['control_post']['fulltime_rate']:.3f}        {summary_stats['control_post']['fulltime_rate'] - summary_stats['control_pre']['fulltime_rate']:+.3f}")

# DiD estimate (simple)
did_simple = (summary_stats['treatment_post']['fulltime_rate'] - summary_stats['treatment_pre']['fulltime_rate']) - \
             (summary_stats['control_post']['fulltime_rate'] - summary_stats['control_pre']['fulltime_rate'])
print(f"\n  Simple DiD estimate: {did_simple:+.4f}")

# =============================================================================
# 4. MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n[4] Running regression analyses...")

# Model 1: Basic DiD (unweighted)
print("\n  Model 1: Basic DiD (unweighted)")
model1 = smf.ols('fulltime ~ treatment + post + treat_post', data=df).fit()
print(f"    DiD coefficient (treat_post): {model1.params['treat_post']:.4f}")
print(f"    Standard error: {model1.bse['treat_post']:.4f}")
print(f"    t-statistic: {model1.tvalues['treat_post']:.4f}")
print(f"    p-value: {model1.pvalues['treat_post']:.4f}")

# Model 2: Basic DiD (weighted)
print("\n  Model 2: Basic DiD (weighted)")
model2 = smf.wls('fulltime ~ treatment + post + treat_post', data=df, weights=df['PERWT']).fit()
print(f"    DiD coefficient (treat_post): {model2.params['treat_post']:.4f}")
print(f"    Standard error: {model2.bse['treat_post']:.4f}")
print(f"    t-statistic: {model2.tvalues['treat_post']:.4f}")
print(f"    p-value: {model2.pvalues['treat_post']:.4f}")

# Model 3: DiD with covariates (weighted)
print("\n  Model 3: DiD with covariates (weighted)")
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] == 1).astype(int)

model3 = smf.wls('fulltime ~ treatment + post + treat_post + female + married + C(EDUC)',
                 data=df, weights=df['PERWT']).fit()
print(f"    DiD coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"    Standard error: {model3.bse['treat_post']:.4f}")
print(f"    t-statistic: {model3.tvalues['treat_post']:.4f}")
print(f"    p-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: DiD with covariates and state fixed effects (weighted)
print("\n  Model 4: DiD with covariates + state FE (weighted)")
model4 = smf.wls('fulltime ~ treatment + post + treat_post + female + married + C(EDUC) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit()
print(f"    DiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"    Standard error: {model4.bse['treat_post']:.4f}")
print(f"    t-statistic: {model4.tvalues['treat_post']:.4f}")
print(f"    p-value: {model4.pvalues['treat_post']:.4f}")

# Model 5: DiD with covariates, state FE, and year FE (weighted) - PREFERRED
print("\n  Model 5: DiD with covariates + state FE + year FE (weighted) - PREFERRED")
model5 = smf.wls('fulltime ~ treatment + treat_post + female + married + C(EDUC) + C(STATEFIP) + C(YEAR)',
                 data=df, weights=df['PERWT']).fit()
print(f"    DiD coefficient (treat_post): {model5.params['treat_post']:.4f}")
print(f"    Standard error: {model5.bse['treat_post']:.4f}")
print(f"    t-statistic: {model5.tvalues['treat_post']:.4f}")
print(f"    p-value: {model5.pvalues['treat_post']:.4f}")

# Calculate robust standard errors for preferred model
print("\n  Model 5 with robust (HC1) standard errors:")
model5_robust = smf.wls('fulltime ~ treatment + treat_post + female + married + C(EDUC) + C(STATEFIP) + C(YEAR)',
                        data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"    DiD coefficient (treat_post): {model5_robust.params['treat_post']:.4f}")
print(f"    Robust SE: {model5_robust.bse['treat_post']:.4f}")
print(f"    t-statistic: {model5_robust.tvalues['treat_post']:.4f}")
print(f"    p-value: {model5_robust.pvalues['treat_post']:.4f}")

# =============================================================================
# 5. ROBUSTNESS CHECKS
# =============================================================================
print("\n[5] Robustness checks...")

# 5a. Alternative age bandwidth (24-30 vs 31-37)
print("\n  5a. Alternative age bandwidth (born 1975-1988)")
df_alt = pd.read_csv('data/data.csv', chunksize=chunk_size, usecols=usecols,
                     dtype={k: v for k, v in dtypes.items() if k in usecols})
chunks_alt = []
for chunk in df_alt:
    chunk_filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks_alt.append(chunk_filtered)
df_alt = pd.concat(chunks_alt, ignore_index=True)

df_alt = df_alt[df_alt['CITIZEN'] == 3]
df_alt['age_at_arrival'] = df_alt['YRIMMIG'] - df_alt['BIRTHYR']
df_alt = df_alt[(df_alt['age_at_arrival'] > 0) & (df_alt['age_at_arrival'] < 16)]
df_alt = df_alt[df_alt['YRIMMIG'] <= 2007]
df_alt['treatment'] = ((df_alt['BIRTHYR'] >= 1980) & (df_alt['BIRTHYR'] <= 1986)).astype(int)
df_alt['control_group'] = ((df_alt['BIRTHYR'] >= 1975) & (df_alt['BIRTHYR'] <= 1979)).astype(int)
df_alt = df_alt[(df_alt['treatment'] == 1) | (df_alt['control_group'] == 1)]
df_alt = df_alt[df_alt['YEAR'] != 2012]
df_alt['post'] = (df_alt['YEAR'] >= 2013).astype(int)
df_alt['fulltime'] = (df_alt['UHRSWORK'] >= 35).astype(int)
df_alt['treat_post'] = df_alt['treatment'] * df_alt['post']
df_alt['female'] = (df_alt['SEX'] == 2).astype(int)
df_alt['married'] = (df_alt['MARST'] == 1).astype(int)

model_alt = smf.wls('fulltime ~ treatment + treat_post + female + married + C(EDUC) + C(STATEFIP) + C(YEAR)',
                    data=df_alt, weights=df_alt['PERWT']).fit(cov_type='HC1')
print(f"    DiD coefficient: {model_alt.params['treat_post']:.4f} (SE: {model_alt.bse['treat_post']:.4f})")
print(f"    Sample size: {len(df_alt):,}")

# 5b. By gender
print("\n  5b. By gender")
df_male = df[df['SEX'] == 1]
df_female = df[df['SEX'] == 2]

model_male = smf.wls('fulltime ~ treatment + treat_post + married + C(EDUC) + C(STATEFIP) + C(YEAR)',
                     data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ treatment + treat_post + married + C(EDUC) + C(STATEFIP) + C(YEAR)',
                       data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"    Male:   {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f}), n={len(df_male):,}")
print(f"    Female: {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f}), n={len(df_female):,}")

# 5c. Placebo test: use 2010 as fake treatment year
print("\n  5c. Placebo test (fake treatment in 2010)")
df_placebo = df[df['YEAR'] <= 2011].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2010).astype(int)
df_placebo['treat_post_placebo'] = df_placebo['treatment'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treatment + treat_post_placebo + female + married + C(EDUC) + C(STATEFIP) + C(YEAR)',
                        data=df_placebo, weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"    Placebo DiD coefficient: {model_placebo.params['treat_post_placebo']:.4f} (SE: {model_placebo.bse['treat_post_placebo']:.4f})")
print(f"    p-value: {model_placebo.pvalues['treat_post_placebo']:.4f}")

# 5d. Event study - year-specific effects
print("\n  5d. Event study coefficients")
df['year_treat'] = df['treatment'] * df['YEAR']
for year in sorted(df['YEAR'].unique()):
    df[f'treat_{year}'] = (df['treatment'] * (df['YEAR'] == year)).astype(int)

# Reference year: 2011 (last pre-treatment year)
years_for_es = [y for y in sorted(df['YEAR'].unique()) if y != 2011]
es_formula = 'fulltime ~ treatment + ' + ' + '.join([f'treat_{y}' for y in years_for_es]) + ' + female + married + C(EDUC) + C(STATEFIP) + C(YEAR)'
model_es = smf.wls(es_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("    Year   Coefficient   SE        95% CI")
for year in years_for_es:
    coef = model_es.params[f'treat_{year}']
    se = model_es.bse[f'treat_{year}']
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    marker = "*" if year >= 2013 else ""
    print(f"    {year}   {coef:+.4f}       {se:.4f}    [{ci_low:+.4f}, {ci_high:+.4f}] {marker}")

# =============================================================================
# 6. SAVE RESULTS
# =============================================================================
print("\n[6] Saving results...")

# Preferred estimate
preferred_estimate = model5_robust.params['treat_post']
preferred_se = model5_robust.bse['treat_post']
preferred_ci = (preferred_estimate - 1.96 * preferred_se, preferred_estimate + 1.96 * preferred_se)
preferred_pval = model5_robust.pvalues['treat_post']
preferred_n = len(df)

results = {
    'preferred_estimate': float(preferred_estimate),
    'standard_error': float(preferred_se),
    'confidence_interval_95': [float(preferred_ci[0]), float(preferred_ci[1])],
    'p_value': float(preferred_pval),
    'sample_size': int(preferred_n),
    'model_description': 'WLS DiD with covariates (sex, marital status, education), state FE, and year FE, robust SEs',
    'summary_stats': summary_stats,
    'sample_info': sample_info
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save full model summary for preferred model
with open('model_summary.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("PREFERRED MODEL SUMMARY\n")
    f.write("Model 5: WLS DiD with covariates, state FE, year FE, robust SEs\n")
    f.write("=" * 70 + "\n\n")
    f.write(model5_robust.summary().as_text())

# Save event study results
es_results = []
for year in years_for_es:
    es_results.append({
        'year': int(year),
        'coefficient': float(model_es.params[f'treat_{year}']),
        'se': float(model_es.bse[f'treat_{year}']),
        'post_daca': bool(year >= 2013)
    })

with open('event_study_results.json', 'w') as f:
    json.dump(es_results, f, indent=2)

# =============================================================================
# 7. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"""
Research Question: Effect of DACA eligibility on full-time employment
                   among Hispanic-Mexican, Mexican-born non-citizens

Identification Strategy: Difference-in-Differences
  - Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)
  - Control:   Born 1977-1981 (ages 31-35 on June 15, 2012)
  - Pre-period:  2006-2011
  - Post-period: 2013-2016

PREFERRED ESTIMATE (Model 5 with robust SEs):
  Effect:          {preferred_estimate:+.4f} ({preferred_estimate*100:+.2f} percentage points)
  Standard Error:  {preferred_se:.4f}
  95% CI:          [{preferred_ci[0]:+.4f}, {preferred_ci[1]:+.4f}]
  p-value:         {preferred_pval:.4f}
  Sample Size:     {preferred_n:,}

INTERPRETATION:
  DACA eligibility is associated with a {abs(preferred_estimate*100):.2f} percentage point
  {"increase" if preferred_estimate > 0 else "decrease"} in full-time employment among eligible
  Hispanic-Mexican, Mexican-born non-citizens, {"statistically significant" if preferred_pval < 0.05 else "not statistically significant"} at the 5% level.
""")

print("Results saved to: results.json, model_summary.txt, event_study_results.json")
print("=" * 70)

# =============================================================================
# 8. GENERATE TABLES FOR LATEX REPORT
# =============================================================================
print("\n[8] Generating LaTeX tables...")

# Table 1: Sample Characteristics
latex_table1 = r"""
\begin{table}[htbp]
\centering
\caption{Sample Characteristics by Treatment Status and Period}
\label{tab:summary}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Pre-DACA (2006-2011)} & \multicolumn{2}{c}{Post-DACA (2013-2016)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & Treatment & Control & Treatment & Control \\
 & (Ages 26-30) & (Ages 31-35) & (Ages 26-30) & (Ages 31-35) \\
\midrule
"""
latex_table1 += f"Full-time employment rate & {summary_stats['treatment_pre']['fulltime_rate']:.3f} & {summary_stats['control_pre']['fulltime_rate']:.3f} & {summary_stats['treatment_post']['fulltime_rate']:.3f} & {summary_stats['control_post']['fulltime_rate']:.3f} \\\\\n"
latex_table1 += f"Mean age & {summary_stats['treatment_pre']['mean_age']:.1f} & {summary_stats['control_pre']['mean_age']:.1f} & {summary_stats['treatment_post']['mean_age']:.1f} & {summary_stats['control_post']['mean_age']:.1f} \\\\\n"
latex_table1 += f"Female share & {summary_stats['treatment_pre']['female_share']:.3f} & {summary_stats['control_pre']['female_share']:.3f} & {summary_stats['treatment_post']['female_share']:.3f} & {summary_stats['control_post']['female_share']:.3f} \\\\\n"
latex_table1 += f"Mean hours (if working) & {summary_stats['treatment_pre']['mean_uhrs']:.1f} & {summary_stats['control_pre']['mean_uhrs']:.1f} & {summary_stats['treatment_post']['mean_uhrs']:.1f} & {summary_stats['control_post']['mean_uhrs']:.1f} \\\\\n"
latex_table1 += f"N & {summary_stats['treatment_pre']['n']:,} & {summary_stats['control_pre']['n']:,} & {summary_stats['treatment_post']['n']:,} & {summary_stats['control_post']['n']:,} \\\\\n"
latex_table1 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Treatment group consists of individuals born 1982-1986 (ages 26-30 as of June 15, 2012). Control group consists of individuals born 1977-1981 (ages 31-35 as of June 15, 2012). Sample restricted to Mexican-born, Hispanic-Mexican non-citizens who arrived in the US before age 16 and by 2007. Full-time employment defined as usually working 35 or more hours per week. Statistics are weighted using ACS person weights.
\end{tablenotes}
\end{table}
"""

# Table 2: Main Regression Results
latex_table2 = r"""
\begin{table}[htbp]
\centering
\caption{Difference-in-Differences Estimates: Effect of DACA on Full-Time Employment}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
 & (1) & (2) & (3) & (4) & (5) \\
\midrule
"""

models_for_table = [
    ('DiD (Treatment $\times$ Post)',
     [model1.params['treat_post'], model2.params['treat_post'], model3.params['treat_post'],
      model4.params['treat_post'], model5_robust.params['treat_post']],
     [model1.bse['treat_post'], model2.bse['treat_post'], model3.bse['treat_post'],
      model4.bse['treat_post'], model5_robust.bse['treat_post']])
]

for label, coefs, ses in models_for_table:
    latex_table2 += f"{label} & {coefs[0]:.4f} & {coefs[1]:.4f} & {coefs[2]:.4f} & {coefs[3]:.4f} & {coefs[4]:.4f} \\\\\n"
    latex_table2 += f" & ({ses[0]:.4f}) & ({ses[1]:.4f}) & ({ses[2]:.4f}) & ({ses[3]:.4f}) & ({ses[4]:.4f}) \\\\\n"

latex_table2 += r"""
\\
Weighted & No & Yes & Yes & Yes & Yes \\
Covariates & No & No & Yes & Yes & Yes \\
State FE & No & No & No & Yes & Yes \\
Year FE & No & No & No & No & Yes \\
Robust SE & No & No & No & No & Yes \\
"""
latex_table2 += f"N & {len(df):,} & {len(df):,} & {len(df):,} & {len(df):,} & {len(df):,} \\\\\n"
latex_table2 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Dependent variable is an indicator for full-time employment (working 35+ hours per week). Treatment group: born 1982-1986 (ages 26-30 at DACA implementation). Control group: born 1977-1981 (ages 31-35 at DACA implementation). Covariates include sex, marital status, and education level. Standard errors in parentheses. Column (5) uses heteroskedasticity-robust standard errors. *** p$<$0.01, ** p$<$0.05, * p$<$0.1.
\end{tablenotes}
\end{table}
"""

# Save tables
with open('latex_tables.tex', 'w') as f:
    f.write("% LaTeX tables for DACA replication study\n\n")
    f.write(latex_table1)
    f.write("\n\n")
    f.write(latex_table2)

print("  LaTeX tables saved to latex_tables.tex")

# Generate data for figures
print("\n[9] Generating figure data...")

# Figure 1: Full-time employment rates by year and group
yearly_rates = df.groupby(['YEAR', 'treatment']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x)
    })
).reset_index()

yearly_rates_pivot = yearly_rates.pivot(index='YEAR', columns='treatment', values='fulltime_rate')
yearly_rates_pivot.columns = ['Control', 'Treatment']
yearly_rates_pivot.to_csv('yearly_rates.csv')

print("  Yearly rates saved to yearly_rates.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
