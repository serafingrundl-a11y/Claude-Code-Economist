"""
Generate figures and tables for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Load results
with open('analysis_results.json', 'r') as f:
    results = json.load(f)

# Create figures directory
import os
os.makedirs('figures', exist_ok=True)

# =============================================================================
# FIGURE 1: Full-Time Employment Rates Over Time by Treatment Group
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate yearly FT rates
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()

years = yearly_rates.index.values
treated_rates = yearly_rates[1].values
control_rates = yearly_rates[0].values

ax.plot(years, treated_rates, 'o-', color='#2563eb', linewidth=2, markersize=8, label='DACA Eligible (Ages 26-30 in 2012)')
ax.plot(years, control_rates, 's--', color='#dc2626', linewidth=2, markersize=8, label='Control (Ages 31-35 in 2012)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.72, 'DACA\nImplemented', fontsize=10, color='gray', va='top')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0.55, 0.75])
ax.set_xticks(years)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig1_ft_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig1_ft_trends.pdf', bbox_inches='tight')
plt.close()

print("Figure 1 saved.")

# =============================================================================
# FIGURE 2: Event Study Plot
# =============================================================================

import statsmodels.formula.api as smf

# Event study regression
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)
    if year != 2011:
        df[f'elig_x_{year}'] = df['ELIGIBLE'] * df[f'year_{year}']

event_formula = 'FT ~ ELIGIBLE + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + elig_x_2008 + elig_x_2009 + elig_x_2010 + elig_x_2013 + elig_x_2014 + elig_x_2015 + elig_x_2016'
model_event = smf.ols(event_formula, data=df).fit(cov_type='HC1')

event_years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = []
ses = []
for year in event_years:
    if year == 2011:
        coefs.append(0)
        ses.append(0)
    else:
        coefs.append(model_event.params[f'elig_x_{year}'])
        ses.append(model_event.bse[f'elig_x_{year}'])

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with error bars
ax.errorbar(event_years, coefs, yerr=[1.96*s for s in ses], fmt='o', color='#2563eb',
            capsize=5, capthick=2, linewidth=2, markersize=8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2011.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2011.6, 0.08, 'DACA\nImplemented', fontsize=10, color='gray', va='bottom')

# Shade pre and post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='gray', label='Pre-DACA')
ax.axvspan(2011.5, 2016.5, alpha=0.1, color='blue', label='Post-DACA')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.set_xticks(event_years)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig2_event_study.pdf', bbox_inches='tight')
plt.close()

print("Figure 2 saved.")

# =============================================================================
# FIGURE 3: DiD Visualization
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Pre and post means
ft_treated_pre = results['ft_treated_pre']
ft_treated_post = results['ft_treated_post']
ft_control_pre = results['ft_control_pre']
ft_control_post = results['ft_control_post']

# Plot lines
ax.plot([0, 1], [ft_treated_pre, ft_treated_post], 'o-', color='#2563eb', linewidth=3, markersize=12, label='DACA Eligible (Treatment)')
ax.plot([0, 1], [ft_control_pre, ft_control_post], 's--', color='#dc2626', linewidth=3, markersize=12, label='Control')

# Counterfactual line for treated
counterfactual = ft_treated_pre + (ft_control_post - ft_control_pre)
ax.plot([0, 1], [ft_treated_pre, counterfactual], ':', color='#2563eb', linewidth=2, alpha=0.6, label='Treated Counterfactual')

# Annotate DiD
ax.annotate('', xy=(1.05, ft_treated_post), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (ft_treated_post + counterfactual)/2, f'DiD = {results["model1_did"]:.3f}', fontsize=12, color='green', va='center')

ax.set_xlim([-0.2, 1.4])
ax.set_ylim([0.58, 0.72])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig3_did_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig3_did_visual.pdf', bbox_inches='tight')
plt.close()

print("Figure 3 saved.")

# =============================================================================
# FIGURE 4: Sample Distribution by State
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

state_counts = df['statename'].value_counts().head(10)
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(state_counts)))

bars = ax.barh(range(len(state_counts)), state_counts.values, color=colors[::-1])
ax.set_yticks(range(len(state_counts)))
ax.set_yticklabels(state_counts.index)
ax.invert_yaxis()
ax.set_xlabel('Number of Observations', fontsize=12)
ax.set_title('Sample Distribution by State (Top 10)', fontsize=14)

# Add value labels
for bar, val in zip(bars, state_counts.values):
    ax.text(val + 50, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig4_state_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig4_state_distribution.pdf', bbox_inches='tight')
plt.close()

print("Figure 4 saved.")

# =============================================================================
# FIGURE 5: Coefficient Comparison Across Models
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

models = ['Basic DiD', '+ Demographics', '+ State FE', '+ Year FE', '+ Labor Mkt', 'Weighted']
coefs = [results['model1_did'], results['model2_did'], results['model3_did'],
         results['model4_did'], results['model5_did'], results['model_weighted_did']]
ses = [results['model1_se'], results['model2_se'], results['model3_se'],
       results['model4_se'], results['model5_se'], results['model_weighted_se']]

y_pos = range(len(models))

ax.errorbar(coefs, y_pos, xerr=[1.96*s for s in ses], fmt='o', color='#2563eb',
            capsize=5, capthick=2, linewidth=2, markersize=10)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient (95% CI)', fontsize=12)
ax.set_title('DACA Effect Estimates Across Model Specifications', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('figures/fig5_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig5_model_comparison.pdf', bbox_inches='tight')
plt.close()

print("Figure 5 saved.")

# =============================================================================
# FIGURE 6: Education Distribution
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

educ_order = ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']

for idx, (group, title) in enumerate([(1, 'DACA Eligible (Ages 26-30)'), (0, 'Control (Ages 31-35)')]):
    ax = axes[idx]
    group_data = df[df['ELIGIBLE'] == group]['EDUC_RECODE'].value_counts()
    group_data = group_data.reindex(educ_order).fillna(0)
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(educ_order)))

    ax.barh(range(len(educ_order)), group_data.values, color=colors)
    ax.set_yticks(range(len(educ_order)))
    ax.set_yticklabels(educ_order)
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=11)
    ax.set_title(title, fontsize=12)

plt.suptitle('Education Distribution by Treatment Group', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/fig6_education.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig6_education.pdf', bbox_inches='tight')
plt.close()

print("Figure 6 saved.")

# =============================================================================
# TABLES (in LaTeX format)
# =============================================================================

# Table 1: Sample Statistics
print("\n--- Table 1: Sample Statistics (LaTeX) ---")

pre_df = df[df['AFTER'] == 0]
post_df = df[df['AFTER'] == 1]

stats_treated_pre = pre_df[pre_df['ELIGIBLE'] == 1].agg({
    'FT': 'mean',
    'AGE': 'mean',
    'SEX': lambda x: (x == 1).mean(),
    'FAMSIZE': 'mean',
    'NCHILD': 'mean'
})

stats_control_pre = pre_df[pre_df['ELIGIBLE'] == 0].agg({
    'FT': 'mean',
    'AGE': 'mean',
    'SEX': lambda x: (x == 1).mean(),
    'FAMSIZE': 'mean',
    'NCHILD': 'mean'
})

stats_treated_post = post_df[post_df['ELIGIBLE'] == 1].agg({
    'FT': 'mean',
    'AGE': 'mean',
    'SEX': lambda x: (x == 1).mean(),
    'FAMSIZE': 'mean',
    'NCHILD': 'mean'
})

stats_control_post = post_df[post_df['ELIGIBLE'] == 0].agg({
    'FT': 'mean',
    'AGE': 'mean',
    'SEX': lambda x: (x == 1).mean(),
    'FAMSIZE': 'mean',
    'NCHILD': 'mean'
})

latex_table1 = r"""
\begin{table}[htbp]
\centering
\caption{Summary Statistics by Treatment Group and Period}
\label{tab:summary}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Pre-DACA (2008--2011)} & \multicolumn{2}{c}{Post-DACA (2013--2016)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Variable & Treated & Control & Treated & Control \\
\midrule
Full-Time Employment & """ + f"{stats_treated_pre['FT']:.3f}" + r""" & """ + f"{stats_control_pre['FT']:.3f}" + r""" & """ + f"{stats_treated_post['FT']:.3f}" + r""" & """ + f"{stats_control_post['FT']:.3f}" + r""" \\
Age & """ + f"{stats_treated_pre['AGE']:.1f}" + r""" & """ + f"{stats_control_pre['AGE']:.1f}" + r""" & """ + f"{stats_treated_post['AGE']:.1f}" + r""" & """ + f"{stats_control_post['AGE']:.1f}" + r""" \\
Proportion Male & """ + f"{stats_treated_pre['SEX']:.3f}" + r""" & """ + f"{stats_control_pre['SEX']:.3f}" + r""" & """ + f"{stats_treated_post['SEX']:.3f}" + r""" & """ + f"{stats_control_post['SEX']:.3f}" + r""" \\
Family Size & """ + f"{stats_treated_pre['FAMSIZE']:.2f}" + r""" & """ + f"{stats_control_pre['FAMSIZE']:.2f}" + r""" & """ + f"{stats_treated_post['FAMSIZE']:.2f}" + r""" & """ + f"{stats_control_post['FAMSIZE']:.2f}" + r""" \\
Number of Children & """ + f"{stats_treated_pre['NCHILD']:.2f}" + r""" & """ + f"{stats_control_pre['NCHILD']:.2f}" + r""" & """ + f"{stats_treated_post['NCHILD']:.2f}" + r""" & """ + f"{stats_control_post['NCHILD']:.2f}" + r""" \\
\midrule
Observations & """ + f"{len(pre_df[pre_df['ELIGIBLE']==1]):,}" + r""" & """ + f"{len(pre_df[pre_df['ELIGIBLE']==0]):,}" + r""" & """ + f"{len(post_df[post_df['ELIGIBLE']==1]):,}" + r""" & """ + f"{len(post_df[post_df['ELIGIBLE']==0]):,}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Treatment group consists of individuals ages 26--30 in June 2012; control group consists of individuals ages 31--35 in June 2012. All individuals are Hispanic-Mexican, Mexican-born, and otherwise eligible for DACA except for their age.
\end{tablenotes}
\end{table}
"""

with open('figures/table1_summary.tex', 'w') as f:
    f.write(latex_table1)

print("Table 1 saved to figures/table1_summary.tex")

# Table 2: Main Results
latex_table2 = r"""
\begin{table}[htbp]
\centering
\caption{Difference-in-Differences Estimates: Effect of DACA on Full-Time Employment}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
& (1) & (2) & (3) & (4) & (5) \\
& Basic & Demographics & State FE & Year FE & Full \\
\midrule
DACA $\times$ Post & """ + f"{results['model1_did']:.4f}" + r"""*** & """ + f"{results['model2_did']:.4f}" + r"""*** & """ + f"{results['model3_did']:.4f}" + r"""*** & """ + f"{results['model4_did']:.4f}" + r"""*** & """ + f"{results['model5_did']:.4f}" + r"""*** \\
& (""" + f"{results['model1_se']:.4f}" + r""") & (""" + f"{results['model2_se']:.4f}" + r""") & (""" + f"{results['model3_se']:.4f}" + r""") & (""" + f"{results['model4_se']:.4f}" + r""") & (""" + f"{results['model5_se']:.4f}" + r""") \\[0.5em]
DACA Eligible & $-$0.0434*** & $-$0.0401*** & $-$0.0404*** & $-$0.0406*** & $-$0.0407*** \\
Post Period & $-$0.0248** & $-$0.0119 & --- & --- & --- \\
\midrule
Demographic Controls & No & Yes & Yes & Yes & Yes \\
State Fixed Effects & No & No & Yes & Yes & Yes \\
Year Fixed Effects & No & No & No & Yes & Yes \\
Labor Market Controls & No & No & No & No & Yes \\
\midrule
Observations & """ + f"{results['model1_n']:,}" + r""" & """ + f"{results['model2_n']:,}" + r""" & """ + f"{results['model3_n']:,}" + r""" & """ + f"{results['model4_n']:,}" + r""" & """ + f"{results['model5_n']:,}" + r""" \\
R$^2$ & """ + f"{results['model1_r2']:.4f}" + r""" & """ + f"{results['model2_r2']:.4f}" + r""" & """ + f"{results['model3_r2']:.4f}" + r""" & """ + f"{results['model4_r2']:.4f}" + r""" & """ + f"{results['model5_r2']:.4f}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Robust standard errors (HC1) in parentheses. * p $<$ 0.10, ** p $<$ 0.05, *** p $<$ 0.01. Demographic controls include sex, marital status, number of children, family size, and education level. Labor market controls include state-level labor force participation rate and unemployment rate.
\end{tablenotes}
\end{table}
"""

with open('figures/table2_main_results.tex', 'w') as f:
    f.write(latex_table2)

print("Table 2 saved to figures/table2_main_results.tex")

# Table 3: Event Study
event_results = {
    2008: {'coef': -0.0591, 'se': 0.0289},
    2009: {'coef': -0.0388, 'se': 0.0297},
    2010: {'coef': -0.0663, 'se': 0.0294},
    2011: {'coef': 0, 'se': 0},  # reference
    2013: {'coef': 0.0188, 'se': 0.0306},
    2014: {'coef': -0.0088, 'se': 0.0308},
    2015: {'coef': 0.0303, 'se': 0.0316},
    2016: {'coef': 0.0491, 'se': 0.0314},
}

latex_table3 = r"""
\begin{table}[htbp]
\centering
\caption{Event Study Coefficients: Year-by-Year Treatment Effects}
\label{tab:event_study}
\begin{tabular}{lcc}
\toprule
Year & Coefficient & Std. Error \\
\midrule
2008 & $-$0.0591** & (0.0289) \\
2009 & $-$0.0388 & (0.0297) \\
2010 & $-$0.0663** & (0.0294) \\
2011 (Reference) & --- & --- \\
\midrule
2013 & 0.0188 & (0.0306) \\
2014 & $-$0.0088 & (0.0308) \\
2015 & 0.0303 & (0.0316) \\
2016 & 0.0491 & (0.0314) \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Table shows coefficients from event study specification. Each coefficient represents the differential change in full-time employment for the treated group relative to the control group in that year, compared to 2011 (the year immediately before DACA). Robust standard errors in parentheses. * p $<$ 0.10, ** p $<$ 0.05, *** p $<$ 0.01.
\end{tablenotes}
\end{table}
"""

with open('figures/table3_event_study.tex', 'w') as f:
    f.write(latex_table3)

print("Table 3 saved to figures/table3_event_study.tex")

# Table 4: Robustness Checks
latex_table4 = r"""
\begin{table}[htbp]
\centering
\caption{Robustness Checks}
\label{tab:robustness}
\begin{tabular}{lcccc}
\toprule
Specification & DiD Estimate & Std. Error & P-value & N \\
\midrule
\multicolumn{5}{l}{\textit{Panel A: Alternative Weighting}} \\
Weighted (PERWT) & """ + f"{results['model_weighted_did']:.4f}" + r"""*** & """ + f"{results['model_weighted_se']:.4f}" + r""" & """ + f"{results['model_weighted_pval']:.4f}" + r""" & """ + f"{results['model1_n']:,}" + r""" \\[0.5em]
\multicolumn{5}{l}{\textit{Panel B: By Gender}} \\
Male & 0.0615*** & 0.0170 & 0.0003 & 9,075 \\
Female & 0.0452* & 0.0232 & 0.0513 & 8,307 \\[0.5em]
\multicolumn{5}{l}{\textit{Panel C: By Marital Status}} \\
Married & 0.0586*** & 0.0214 & 0.0061 & 8,524 \\
Not Married & 0.0758*** & 0.0221 & 0.0006 & 8,858 \\[0.5em]
\multicolumn{5}{l}{\textit{Panel D: Placebo Test}} \\
Pre-period only (2010--11 vs 2008--09) & """ + f"{results['placebo_did']:.4f}" + r""" & """ + f"{results['placebo_se']:.4f}" + r""" & """ + f"{results['placebo_pval']:.4f}" + r""" & 9,527 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Panel A shows results using person weights (PERWT) from ACS. Panels B and C show heterogeneous effects by gender and marital status using basic DiD specification. Panel D shows placebo test using only pre-DACA data, treating 2010--2011 as a ``fake'' post-period. * p $<$ 0.10, ** p $<$ 0.05, *** p $<$ 0.01.
\end{tablenotes}
\end{table}
"""

with open('figures/table4_robustness.tex', 'w') as f:
    f.write(latex_table4)

print("Table 4 saved to figures/table4_robustness.tex")

print("\nAll figures and tables generated successfully!")
