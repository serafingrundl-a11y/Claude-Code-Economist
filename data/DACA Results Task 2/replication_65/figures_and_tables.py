#!/usr/bin/env python3
"""
DACA Replication - Generate Figures and Tables for LaTeX Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

DATA_PATH = 'data/data.csv'

print("Loading and processing data for figures...")

# Load data with same filtering as main analysis
cols_needed = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX', 'MARST',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'EMPSTAT', 'LABFORCE', 'UHRSWORK',
               'STATEFIP', 'METRO', 'FAMSIZE', 'NCHILD']

chunks = []
for chunk in pd.read_csv(DATA_PATH, usecols=cols_needed, chunksize=1000000):
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

# Apply filters
df = df[df['CITIZEN'] == 3]
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_immig'] < 16]
df = df[df['YRIMMIG'] <= 2007]

# Define groups
df['treatment'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)
df['control'] = ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)).astype(int)
df = df[(df['treatment'] == 1) | (df['control'] == 1)]
df['treated'] = df['treatment']

# Time periods
df = df[df['YEAR'] != 2012]
df['post'] = (df['YEAR'] >= 2013).astype(int)
df['treated_post'] = df['treated'] * df['post']

# Outcome
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Demographics
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)
df['educ_less_hs'] = (df['EDUCD'] < 62).astype(int)
df['educ_hs'] = ((df['EDUCD'] >= 62) & (df['EDUCD'] < 65)).astype(int)
df['educ_some_college'] = ((df['EDUCD'] >= 65) & (df['EDUCD'] < 101)).astype(int)
df['educ_college'] = (df['EDUCD'] >= 101).astype(int)

print(f"Sample size: {len(df):,}")

# ============================================================================
# FIGURE 1: Parallel Trends - Full-time Employment by Year and Group
# ============================================================================
print("\nGenerating Figure 1: Parallel Trends...")

# Calculate weighted means by year and group
trends_data = []
for year in sorted(df['YEAR'].unique()):
    for treat in [0, 1]:
        subset = df[(df['YEAR'] == year) & (df['treated'] == treat)]
        if len(subset) > 0:
            # Weighted mean
            weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
            n = len(subset)
            # Weighted standard error (approximate)
            weighted_var = np.average((subset['fulltime'] - weighted_mean)**2, weights=subset['PERWT'])
            se = np.sqrt(weighted_var / n)
            trends_data.append({
                'year': year,
                'treated': treat,
                'fulltime_rate': weighted_mean,
                'se': se,
                'n': n
            })

trends_df = pd.DataFrame(trends_data)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot treatment group
treat_data = trends_df[trends_df['treated'] == 1].sort_values('year')
ax.errorbar(treat_data['year'], treat_data['fulltime_rate'],
            yerr=1.96*treat_data['se'], fmt='o-', capsize=3,
            label='Treatment (Ages 26-30 in 2012)', color='blue', linewidth=2, markersize=8)

# Plot control group
ctrl_data = trends_df[trends_df['treated'] == 0].sort_values('year')
ax.errorbar(ctrl_data['year'], ctrl_data['fulltime_rate'],
            yerr=1.96*ctrl_data['se'], fmt='s--', capsize=3,
            label='Control (Ages 31-35 in 2012)', color='red', linewidth=2, markersize=8)

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')

ax.set_xlabel('Year')
ax.set_ylabel('Full-time Employment Rate')
ax.set_title('Full-time Employment Trends by Treatment Status')
ax.legend(loc='lower right')
ax.set_xticks(sorted(df['YEAR'].unique()))
ax.set_ylim(0.45, 0.70)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure1_parallel_trends.png/pdf")

# ============================================================================
# FIGURE 2: Event Study
# ============================================================================
print("\nGenerating Figure 2: Event Study...")

# Create relative time
df['rel_year'] = df['YEAR'] - 2012

# Run regressions for each relative year
event_study_results = []
ref_year = -1  # 2011 as reference

for ry in sorted(df['rel_year'].unique()):
    if ry != ref_year:
        # Create dummy for this relative year
        df['treat_ry'] = (df['treated'] * (df['rel_year'] == ry)).astype(int)
        df['treat_ref'] = (df['treated'] * (df['rel_year'] == ref_year)).astype(int)

        # Simple regression comparing this year to reference
        subset = df[df['rel_year'].isin([ry, ref_year])]
        if len(subset) > 100:
            model = smf.wls('fulltime ~ treated + treat_ry + C(YEAR)',
                           data=subset, weights=subset['PERWT']).fit()
            if 'treat_ry' in model.params:
                event_study_results.append({
                    'rel_year': ry,
                    'coef': model.params['treat_ry'],
                    'se': model.bse['treat_ry'],
                    'ci_lower': model.conf_int().loc['treat_ry', 0],
                    'ci_upper': model.conf_int().loc['treat_ry', 1]
                })

# Add reference year (coefficient = 0)
event_study_results.append({
    'rel_year': ref_year,
    'coef': 0,
    'se': 0,
    'ci_lower': 0,
    'ci_upper': 0
})

es_df = pd.DataFrame(event_study_results).sort_values('rel_year')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
ax.errorbar(es_df['rel_year'], es_df['coef'],
            yerr=[es_df['coef'] - es_df['ci_lower'], es_df['ci_upper'] - es_df['coef']],
            fmt='o', capsize=5, capthick=2, color='blue', markersize=10, linewidth=2)

# Add reference lines
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=0, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')

# Shade pre and post periods
ax.axvspan(-7, 0, alpha=0.1, color='gray', label='Pre-DACA')
ax.axvspan(0, 5, alpha=0.1, color='blue', label='Post-DACA')

ax.set_xlabel('Years Relative to DACA Implementation (2012)')
ax.set_ylabel('Estimated Treatment Effect')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-time Employment')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Custom x-axis labels
ax.set_xticks(sorted(es_df['rel_year']))
ax.set_xticklabels([f'{int(y)} ({2012+int(y)})' for y in sorted(es_df['rel_year'])], rotation=45)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure2_event_study.png/pdf")

# ============================================================================
# FIGURE 3: DiD Illustration
# ============================================================================
print("\nGenerating Figure 3: DiD Illustration...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate weighted means for 2x2 DiD
did_means = {}
for treat in [0, 1]:
    for post in [0, 1]:
        subset = df[(df['treated'] == treat) & (df['post'] == post)]
        did_means[(treat, post)] = np.average(subset['fulltime'], weights=subset['PERWT'])

# Pre-period means
ax.scatter([0, 0], [did_means[(0, 0)], did_means[(1, 0)]], s=150, zorder=5)
ax.scatter([1, 1], [did_means[(0, 1)], did_means[(1, 1)]], s=150, zorder=5)

# Lines for actual trends
ax.plot([0, 1], [did_means[(0, 0)], did_means[(0, 1)]], 's--',
        color='red', linewidth=2, markersize=10, label='Control (Ages 31-35)')
ax.plot([0, 1], [did_means[(1, 0)], did_means[(1, 1)]], 'o-',
        color='blue', linewidth=2, markersize=10, label='Treatment (Ages 26-30)')

# Counterfactual for treatment group
counterfactual = did_means[(1, 0)] + (did_means[(0, 1)] - did_means[(0, 0)])
ax.plot([0, 1], [did_means[(1, 0)], counterfactual], 'o:',
        color='blue', linewidth=2, alpha=0.5, label='Treatment Counterfactual')

# DiD effect arrow
ax.annotate('', xy=(1.05, did_means[(1, 1)]), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (did_means[(1, 1)] + counterfactual)/2,
        f'DiD Effect\n= {did_means[(1, 1)] - counterfactual:.3f}',
        fontsize=11, color='green', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-time Employment Rate')
ax.set_title('Difference-in-Differences Design')
ax.legend(loc='lower right')
ax.set_xlim(-0.2, 1.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_did_illustration.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_illustration.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure3_did_illustration.png/pdf")

# ============================================================================
# FIGURE 4: Heterogeneity Analysis
# ============================================================================
print("\nGenerating Figure 4: Heterogeneity Analysis...")

# Run models for different subgroups
subgroup_results = []

# By sex
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex_val]
    model = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                    data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
    subgroup_results.append({
        'group': sex_name,
        'coef': model.params['treated_post'],
        'se': model.bse['treated_post'],
        'ci_lower': model.conf_int().loc['treated_post', 0],
        'ci_upper': model.conf_int().loc['treated_post', 1]
    })

# By education
for ed_name, ed_filter in [('Less than HS', df['educ_less_hs'] == 1),
                            ('HS or More', df['educ_less_hs'] == 0)]:
    subset = df[ed_filter]
    if len(subset) > 100:
        model = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                        data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
        subgroup_results.append({
            'group': ed_name,
            'coef': model.params['treated_post'],
            'se': model.bse['treated_post'],
            'ci_lower': model.conf_int().loc['treated_post', 0],
            'ci_upper': model.conf_int().loc['treated_post', 1]
        })

# By marital status
for mar_name, mar_filter in [('Married', df['married'] == 1),
                              ('Not Married', df['married'] == 0)]:
    subset = df[mar_filter]
    if len(subset) > 100:
        model = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                        data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
        subgroup_results.append({
            'group': mar_name,
            'coef': model.params['treated_post'],
            'se': model.bse['treated_post'],
            'ci_lower': model.conf_int().loc['treated_post', 0],
            'ci_upper': model.conf_int().loc['treated_post', 1]
        })

# Overall
model_all = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                    data=df, weights=df['PERWT']).fit(cov_type='HC1')
subgroup_results.append({
    'group': 'Overall',
    'coef': model_all.params['treated_post'],
    'se': model_all.bse['treated_post'],
    'ci_lower': model_all.conf_int().loc['treated_post', 0],
    'ci_upper': model_all.conf_int().loc['treated_post', 1]
})

sg_df = pd.DataFrame(subgroup_results)

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = range(len(sg_df))
ax.errorbar(sg_df['coef'], y_pos,
            xerr=[sg_df['coef'] - sg_df['ci_lower'], sg_df['ci_upper'] - sg_df['coef']],
            fmt='o', capsize=5, capthick=2, color='blue', markersize=10)

ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(sg_df['group'])
ax.set_xlabel('DiD Estimate (Effect on Full-time Employment)')
ax.set_title('Heterogeneity in Treatment Effects')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('figure4_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_heterogeneity.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure4_heterogeneity.png/pdf")

# Save event study results
es_df.to_csv('event_study_results.csv', index=False)
print("\n  Saved: event_study_results.csv")

# Save heterogeneity results
sg_df.to_csv('heterogeneity_results.csv', index=False)
print("  Saved: heterogeneity_results.csv")

# ============================================================================
# Generate LaTeX Tables
# ============================================================================
print("\nGenerating LaTeX tables...")

# Table 1: Sample Characteristics
print("\n--- Table 1: Sample Characteristics ---")

chars_data = []
for treat in [0, 1]:
    for post in [0, 1]:
        subset = df[(df['treated'] == treat) & (df['post'] == post)]
        w = subset['PERWT']
        chars_data.append({
            'Group': 'Treatment' if treat == 1 else 'Control',
            'Period': 'Post' if post == 1 else 'Pre',
            'N': len(subset),
            'Full-time Rate': np.average(subset['fulltime'], weights=w),
            'Employment Rate': np.average(subset['employed'], weights=w),
            'Female (%)': np.average(subset['female'], weights=w) * 100,
            'Married (%)': np.average(subset['married'], weights=w) * 100,
            'Less than HS (%)': np.average(subset['educ_less_hs'], weights=w) * 100,
            'College+ (%)': np.average(subset['educ_college'], weights=w) * 100,
            'Mean Age': np.average(subset['AGE'], weights=w)
        })

chars_df = pd.DataFrame(chars_data)
print(chars_df.to_string(index=False))

# LaTeX version
latex_table1 = r"""
\begin{table}[htbp]
\centering
\caption{Sample Characteristics by Treatment Status and Period}
\label{tab:sample_chars}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Treatment (Ages 26-30)} & \multicolumn{2}{c}{Control (Ages 31-35)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
& Pre-DACA & Post-DACA & Pre-DACA & Post-DACA \\
\midrule
"""

# Get the values
t_pre = chars_df[(chars_df['Group']=='Treatment') & (chars_df['Period']=='Pre')].iloc[0]
t_post = chars_df[(chars_df['Group']=='Treatment') & (chars_df['Period']=='Post')].iloc[0]
c_pre = chars_df[(chars_df['Group']=='Control') & (chars_df['Period']=='Pre')].iloc[0]
c_post = chars_df[(chars_df['Group']=='Control') & (chars_df['Period']=='Post')].iloc[0]

latex_table1 += f"N & {t_pre['N']:,} & {t_post['N']:,} & {c_pre['N']:,} & {c_post['N']:,} \\\\\n"
latex_table1 += f"Full-time Employment Rate & {t_pre['Full-time Rate']:.3f} & {t_post['Full-time Rate']:.3f} & {c_pre['Full-time Rate']:.3f} & {c_post['Full-time Rate']:.3f} \\\\\n"
latex_table1 += f"Employment Rate & {t_pre['Employment Rate']:.3f} & {t_post['Employment Rate']:.3f} & {c_pre['Employment Rate']:.3f} & {c_post['Employment Rate']:.3f} \\\\\n"
latex_table1 += f"Female (\\%) & {t_pre['Female (%)']:.1f} & {t_post['Female (%)']:.1f} & {c_pre['Female (%)']:.1f} & {c_post['Female (%)']:.1f} \\\\\n"
latex_table1 += f"Married (\\%) & {t_pre['Married (%)']:.1f} & {t_post['Married (%)']:.1f} & {c_pre['Married (%)']:.1f} & {c_post['Married (%)']:.1f} \\\\\n"
latex_table1 += f"Less than HS (\\%) & {t_pre['Less than HS (%)']:.1f} & {t_post['Less than HS (%)']:.1f} & {c_pre['Less than HS (%)']:.1f} & {c_post['Less than HS (%)']:.1f} \\\\\n"
latex_table1 += f"College+ (\\%) & {t_pre['College+ (%)']:.1f} & {t_post['College+ (%)']:.1f} & {c_pre['College+ (%)']:.1f} & {c_post['College+ (%)']:.1f} \\\\\n"
latex_table1 += f"Mean Age & {t_pre['Mean Age']:.1f} & {t_post['Mean Age']:.1f} & {c_pre['Mean Age']:.1f} & {c_post['Mean Age']:.1f} \\\\\n"

latex_table1 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Treatment group consists of individuals aged 26-30 on June 15, 2012 (born 1982-1986). Control group consists of individuals aged 31-35 on June 15, 2012 (born 1977-1981). Pre-DACA period includes years 2006-2011; Post-DACA period includes years 2013-2016. Sample restricted to Hispanic-Mexican individuals born in Mexico who are non-citizens, arrived in the US before age 16, and have been in the US since at least 2007. Statistics are weighted using ACS person weights.
\end{tablenotes}
\end{table}
"""

with open('table1_sample_chars.tex', 'w') as f:
    f.write(latex_table1)
print("  Saved: table1_sample_chars.tex")

# Table 2: Main Results
print("\n--- Table 2: Main Regression Results ---")

results_df = pd.read_csv('results_summary.csv')

latex_table2 = r"""
\begin{table}[htbp]
\centering
\caption{Difference-in-Differences Estimates: Effect of DACA Eligibility on Full-time Employment}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
Specification & Estimate & Std. Error & 95\% CI & p-value & N \\
\midrule
"""

for _, row in results_df.iterrows():
    spec_name = row['Specification'].split('. ')[1] if '. ' in row['Specification'] else row['Specification']
    ci_str = f"[{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]"
    pval_str = f"{row['P_Value']:.4f}" if row['P_Value'] >= 0.0001 else "$<$0.0001"
    latex_table2 += f"{spec_name} & {row['Estimate']:.4f} & {row['Std_Error']:.4f} & {ci_str} & {pval_str} & {int(row['N']):,} \\\\\n"

latex_table2 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Dependent variable is an indicator for full-time employment (usually working 35+ hours per week). All weighted specifications use ACS person weights. Year FE includes year fixed effects. Controls include gender, marital status, education level, and number of children. State FE includes state fixed effects. Robust SE uses heteroskedasticity-robust (HC1) standard errors. The preferred specification is Model 6.
\end{tablenotes}
\end{table}
"""

with open('table2_main_results.tex', 'w') as f:
    f.write(latex_table2)
print("  Saved: table2_main_results.tex")

# Table 3: Heterogeneity
print("\n--- Table 3: Heterogeneity Analysis ---")

latex_table3 = r"""
\begin{table}[htbp]
\centering
\caption{Heterogeneity in Treatment Effects}
\label{tab:heterogeneity}
\begin{tabular}{lcccc}
\toprule
Subgroup & Estimate & Std. Error & 95\% CI & p-value \\
\midrule
"""

for _, row in sg_df.iterrows():
    ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
    pval = 2 * (1 - stats.norm.cdf(abs(row['coef'] / row['se']))) if row['se'] > 0 else 0
    pval_str = f"{pval:.4f}" if pval >= 0.0001 else "$<$0.0001"
    latex_table3 += f"{row['group']} & {row['coef']:.4f} & {row['se']:.4f} & {ci_str} & {pval_str} \\\\\n"

latex_table3 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Each row reports the DiD estimate from a separate regression on the indicated subgroup. All regressions include year fixed effects and are weighted using ACS person weights. Standard errors are heteroskedasticity-robust (HC1).
\end{tablenotes}
\end{table}
"""

with open('table3_heterogeneity.tex', 'w') as f:
    f.write(latex_table3)
print("  Saved: table3_heterogeneity.tex")

print("\nAll figures and tables generated successfully!")
