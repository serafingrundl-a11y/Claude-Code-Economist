"""
Generate figures and tables for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/prepared_data_labelled_version.csv')
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['MALE'] = (df['SEX'] == 'Male').astype(int)

# Figure 1: Trends in FT employment by group
print("Creating Figure 1: Trends in FT Employment...")

years = sorted(df['YEAR'].unique())
ft_treated = []
ft_control = []

for year in years:
    subset_t = df[(df['YEAR']==year) & (df['ELIGIBLE']==1)]
    subset_c = df[(df['YEAR']==year) & (df['ELIGIBLE']==0)]
    ft_treated.append(np.average(subset_t['FT'], weights=subset_t['PERWT']))
    ft_control.append(np.average(subset_c['FT'], weights=subset_c['PERWT']))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years, ft_treated, 'b-o', label='Treated (Ages 26-30 in 2012)', linewidth=2, markersize=8)
ax.plot(years, ft_control, 'r-s', label='Control (Ages 31-35 in 2012)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle='--', alpha=0.7, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Group', fontsize=14)
ax.legend(loc='lower right')
ax.set_xticks(years)
ax.set_ylim(0.55, 0.75)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure1_trends.png/pdf")

# Figure 2: Event Study Plot
print("Creating Figure 2: Event Study...")

# Run event study regression
ref_year = 2011
for year in years:
    if year != ref_year:
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
        df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

event_vars = [f'ELIGIBLE_YEAR_{year}' for year in years if year != ref_year]
formula_event = 'FT ~ ELIGIBLE + ' + ' + '.join([f'YEAR_{year}' for year in years if year != ref_year]) + ' + ' + ' + '.join(event_vars)
model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Extract coefficients
event_coefs = [0]  # Reference year = 0
event_ses = [0]
event_years = [2011]

for year in years:
    if year != ref_year:
        event_years.append(year)
        event_coefs.append(model_event.params[f'ELIGIBLE_YEAR_{year}'])
        event_ses.append(model_event.bse[f'ELIGIBLE_YEAR_{year}'])

# Sort by year
sorted_data = sorted(zip(event_years, event_coefs, event_ses))
event_years, event_coefs, event_ses = zip(*sorted_data)
event_years = list(event_years)
event_coefs = np.array(event_coefs)
event_ses = np.array(event_ses)

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(event_years, event_coefs, yerr=1.96*event_ses, fmt='o', capsize=4,
            color='navy', linewidth=2, markersize=8, label='Point Estimate & 95% CI')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax.fill_between([years[0]-0.5, 2012], [-0.2, -0.2], [0.2, 0.2], alpha=0.1, color='gray')
ax.fill_between([2012, years[-1]+0.5], [-0.2, -0.2], [0.2, 0.2], alpha=0.1, color='blue')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Difference in FT Employment\n(Treated - Control, relative to 2011)', fontsize=12)
ax.set_title('Event Study: Difference-in-Differences over Time', fontsize=14)
ax.set_xticks(event_years)
ax.set_ylim(-0.15, 0.15)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure2_eventstudy.png/pdf")

# Figure 3: Sample Size by Year and Group
print("Creating Figure 3: Sample Size Distribution...")

sample_counts = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
sample_counts.columns = ['Control', 'Treated']

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(years))
width = 0.35
bars1 = ax.bar(x - width/2, [sample_counts.loc[y, 'Treated'] for y in years], width, label='Treated', color='steelblue')
bars2 = ax.bar(x + width/2, [sample_counts.loc[y, 'Control'] for y in years], width, label='Control', color='coral')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Observations', fontsize=12)
ax.set_title('Sample Size by Year and Treatment Group', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()
ax.axvline(x=3.5, color='gray', linestyle='--', alpha=0.7)  # Between 2011 and 2013
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figure3_samplesize.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_samplesize.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure3_samplesize.png/pdf")

# Create LaTeX tables
print("\nCreating LaTeX tables...")

# Table 1: Summary Statistics
print("Creating Table 1: Summary Statistics...")
summ_data = []
for elig, elig_name in [(1, 'Treatment'), (0, 'Control')]:
    for after, after_name in [(0, 'Pre-DACA'), (1, 'Post-DACA')]:
        subset = df[(df['ELIGIBLE']==elig) & (df['AFTER']==after)]
        w_ft = np.average(subset['FT'], weights=subset['PERWT'])
        w_male = np.average(subset['MALE'], weights=subset['PERWT'])
        w_age = np.average(subset['AGE'], weights=subset['PERWT'])
        w_nchild = np.average(subset['NCHILD'], weights=subset['PERWT'])
        n = len(subset)
        summ_data.append({
            'Group': elig_name,
            'Period': after_name,
            'FT Rate': f'{w_ft:.3f}',
            'Male Share': f'{w_male:.3f}',
            'Mean Age': f'{w_age:.1f}',
            'Mean Children': f'{w_nchild:.2f}',
            'N': f'{n:,}'
        })

summ_df = pd.DataFrame(summ_data)
print(summ_df)

# Save as LaTeX
with open('table1_summary.tex', 'w') as f:
    f.write(r"""\begin{table}[htbp]
\centering
\caption{Summary Statistics by Treatment Group and Time Period}
\label{tab:summary}
\begin{tabular}{llccccc}
\hline\hline
Group & Period & FT Rate & Male Share & Mean Age & Mean Children & N \\
\hline
""")
    for row in summ_data:
        f.write(f"{row['Group']} & {row['Period']} & {row['FT Rate']} & {row['Male Share']} & {row['Mean Age']} & {row['Mean Children']} & {row['N']} \\\\\n")
    f.write(r"""\hline\hline
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Statistics are weighted using ACS person weights (PERWT). Treatment group consists of individuals aged 26-30 in June 2012; control group consists of individuals aged 31-35 in June 2012. Pre-DACA period includes years 2008-2011; post-DACA period includes years 2013-2016.
\end{tablenotes}
\end{table}
""")
print("  Saved: table1_summary.tex")

# Table 2: Main Regression Results
print("Creating Table 2: Main Regression Results...")

# Run all models
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')

formula3 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + C(MARST) + NCHILD'
model3 = smf.wls(formula3, data=df, weights=df['PERWT']).fit(cov_type='HC1')

formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + C(MARST) + NCHILD + C(EDUC_RECODE)'
model4 = smf.wls(formula4, data=df, weights=df['PERWT']).fit(cov_type='HC1')

formula5 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + MALE + C(MARST) + NCHILD + C(EDUC_RECODE)'
model5 = smf.wls(formula5, data=df, weights=df['PERWT']).fit(cov_type='HC1')

formula6 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP) + MALE + C(MARST) + NCHILD + C(EDUC_RECODE)'
model6 = smf.wls(formula6, data=df, weights=df['PERWT']).fit(cov_type='HC1')

def format_coef(coef, se, pval):
    stars = ''
    if pval < 0.01:
        stars = '***'
    elif pval < 0.05:
        stars = '**'
    elif pval < 0.1:
        stars = '*'
    return f'{coef:.4f}{stars}', f'({se:.4f})'

with open('table2_regression.tex', 'w') as f:
    f.write(r"""\begin{table}[htbp]
\centering
\caption{Difference-in-Differences Estimates of DACA Effect on Full-Time Employment}
\label{tab:regression}
\begin{tabular}{lcccccc}
\hline\hline
 & (1) & (2) & (3) & (4) & (5) & (6) \\
\hline
""")

    # DiD coefficient
    models = [model1, model2, model3, model4, model5, model6]
    coefs = []
    ses = []
    for m in models:
        c, s = format_coef(m.params['ELIGIBLE_AFTER'], m.bse['ELIGIBLE_AFTER'], m.pvalues['ELIGIBLE_AFTER'])
        coefs.append(c)
        ses.append(s)

    f.write(f"ELIGIBLE $\\times$ AFTER & {' & '.join(coefs)} \\\\\n")
    f.write(f" & {' & '.join(ses)} \\\\\n")

    # ELIGIBLE coefficient
    coefs = []
    ses = []
    for m in models:
        c, s = format_coef(m.params['ELIGIBLE'], m.bse['ELIGIBLE'], m.pvalues['ELIGIBLE'])
        coefs.append(c)
        ses.append(s)

    f.write(f"ELIGIBLE & {' & '.join(coefs)} \\\\\n")
    f.write(f" & {' & '.join(ses)} \\\\\n")

    f.write(r"""\hline
Weighted & No & Yes & Yes & Yes & Yes & Yes \\
Robust SE & No & Yes & Yes & Yes & Yes & Yes \\
Demographics & No & No & Yes & Yes & Yes & Yes \\
Education & No & No & No & Yes & Yes & Yes \\
Year FE & No & No & No & No & Yes & Yes \\
State FE & No & No & No & No & No & Yes \\
""")

    f.write(f"N & {int(model1.nobs):,} & {int(model2.nobs):,} & {int(model3.nobs):,} & {int(model4.nobs):,} & {int(model5.nobs):,} & {int(model6.nobs):,} \\\\\n")
    f.write(f"$R^2$ & {model1.rsquared:.3f} & {model2.rsquared:.3f} & {model3.rsquared:.3f} & {model4.rsquared:.3f} & {model5.rsquared:.3f} & {model6.rsquared:.3f} \\\\\n")

    f.write(r"""\hline\hline
\end{tabular}
\begin{tablenotes}
\small
\item Notes: * p$<$0.10, ** p$<$0.05, *** p$<$0.01. Robust standard errors in parentheses. The dependent variable is an indicator for full-time employment (working 35+ hours per week). ELIGIBLE indicates the treatment group (ages 26-30 in June 2012). AFTER indicates the post-DACA period (2013-2016). Demographics include sex, marital status, and number of children. Education controls include categories for less than high school, high school degree, some college, two-year degree, and BA+.
\end{tablenotes}
\end{table}
""")
print("  Saved: table2_regression.tex")

# Table 3: Heterogeneity Analysis
print("Creating Table 3: Heterogeneity Analysis...")

formula_base = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER'
df_male = df[df['SEX']=='Male']
df_female = df[df['SEX']=='Female']

model_male = smf.wls(formula_base, data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls(formula_base, data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')

df_hs = df[df['EDUC_RECODE']=='High School Degree']
df_somecol = df[df['EDUC_RECODE'].isin(['Some College', 'Two-Year Degree'])]
df_ba = df[df['EDUC_RECODE']=='BA+']

model_hs = smf.wls(formula_base, data=df_hs, weights=df_hs['PERWT']).fit(cov_type='HC1')
model_somecol = smf.wls(formula_base, data=df_somecol, weights=df_somecol['PERWT']).fit(cov_type='HC1')
model_ba = smf.wls(formula_base, data=df_ba, weights=df_ba['PERWT']).fit(cov_type='HC1')

with open('table3_heterogeneity.tex', 'w') as f:
    f.write(r"""\begin{table}[htbp]
\centering
\caption{Heterogeneity in DACA Effects by Subgroup}
\label{tab:heterogeneity}
\begin{tabular}{lccc}
\hline\hline
Subgroup & DiD Estimate & Std. Error & N \\
\hline
\multicolumn{4}{l}{\textit{By Sex}} \\
Males & """)
    f.write(f"{model_male.params['ELIGIBLE_AFTER']:.4f} & {model_male.bse['ELIGIBLE_AFTER']:.4f} & {int(model_male.nobs):,} \\\\\n")
    f.write(f"Females & {model_female.params['ELIGIBLE_AFTER']:.4f} & {model_female.bse['ELIGIBLE_AFTER']:.4f} & {int(model_female.nobs):,} \\\\\n")
    f.write(r"""\hline
\multicolumn{4}{l}{\textit{By Education}} \\
High School & """)
    f.write(f"{model_hs.params['ELIGIBLE_AFTER']:.4f} & {model_hs.bse['ELIGIBLE_AFTER']:.4f} & {int(model_hs.nobs):,} \\\\\n")
    f.write(f"Some College/2-Year & {model_somecol.params['ELIGIBLE_AFTER']:.4f} & {model_somecol.bse['ELIGIBLE_AFTER']:.4f} & {int(model_somecol.nobs):,} \\\\\n")
    f.write(f"BA+ & {model_ba.params['ELIGIBLE_AFTER']:.4f} & {model_ba.bse['ELIGIBLE_AFTER']:.4f} & {int(model_ba.nobs):,} \\\\\n")
    f.write(r"""\hline\hline
\end{tabular}
\begin{tablenotes}
\small
\item Notes: All estimates use weighted least squares with person weights and robust standard errors. Each row presents a separate regression estimated on the indicated subgroup.
\end{tablenotes}
\end{table}
""")
print("  Saved: table3_heterogeneity.tex")

# Table 4: DiD Components
print("Creating Table 4: DiD Components...")

def weighted_mean(data, col, weight):
    return np.average(data[col], weights=data[weight])

cells = {}
for elig in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE']==elig) & (df['AFTER']==after)]
        cells[(elig, after)] = weighted_mean(subset, 'FT', 'PERWT')

with open('table4_did_components.tex', 'w') as f:
    f.write(r"""\begin{table}[htbp]
\centering
\caption{Difference-in-Differences Decomposition}
\label{tab:did_components}
\begin{tabular}{lccc}
\hline\hline
 & Pre-DACA & Post-DACA & Difference \\
\hline
""")
    f.write(f"Treatment (ELIGIBLE=1) & {cells[(1,0)]:.4f} & {cells[(1,1)]:.4f} & {cells[(1,1)]-cells[(1,0)]:.4f} \\\\\n")
    f.write(f"Control (ELIGIBLE=0) & {cells[(0,0)]:.4f} & {cells[(0,1)]:.4f} & {cells[(0,1)]-cells[(0,0)]:.4f} \\\\\n")
    f.write(r"""\hline
""")
    f.write(f"Difference & {cells[(1,0)]-cells[(0,0)]:.4f} & {cells[(1,1)]-cells[(0,1)]:.4f} & \\textbf{{{(cells[(1,1)]-cells[(1,0)])-(cells[(0,1)]-cells[(0,0)]):.4f}}} \\\\\n")
    f.write(r"""\hline\hline
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Cell entries are weighted mean full-time employment rates using person weights (PERWT). The difference-in-differences estimate (bottom right, bold) equals 0.0748.
\end{tablenotes}
\end{table}
""")
print("  Saved: table4_did_components.tex")

print("\nAll figures and tables created successfully!")
