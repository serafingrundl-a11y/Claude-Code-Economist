"""
Generate LaTeX tables for DACA replication report
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

output_path = r'C:\Users\seraf\DACA Results Task 3\replication_78'

# Load data
df = pd.read_csv(f'{output_path}/data/prepared_data_numeric_version.csv', low_memory=False)
df_lab = pd.read_csv(f'{output_path}/data/prepared_data_labelled_version.csv', low_memory=False)

# Prepare variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['IN_LF'] = (df['LABFORCE'] == 2).astype(int)
df['EMPLOYED'] = (df['EMPSTAT'] == 1).astype(int)

def weighted_mean(data, weights):
    return np.average(data, weights=weights)

def weighted_std(data, weights):
    avg = np.average(data, weights=weights)
    variance = np.average((data - avg)**2, weights=weights)
    return np.sqrt(variance)

# ============================================================================
# TABLE 1: Sample Characteristics
# ============================================================================
print("="*60)
print("TABLE 1: Sample Characteristics by Treatment and Period")
print("="*60)

latex_table1 = r"""
\begin{table}[htbp]
\centering
\caption{Descriptive Statistics by Treatment Status and Time Period}
\label{tab:descriptive}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Pre-DACA (2008-2011)} & \multicolumn{2}{c}{Post-DACA (2013-2016)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Variable & Treatment & Control & Treatment & Control \\
\midrule
"""

# Calculate statistics for each group
groups = {
    'treat_pre': df[(df['ELIGIBLE']==1) & (df['AFTER']==0)],
    'control_pre': df[(df['ELIGIBLE']==0) & (df['AFTER']==0)],
    'treat_post': df[(df['ELIGIBLE']==1) & (df['AFTER']==1)],
    'control_post': df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]
}

variables = [
    ('N (unweighted)', None),
    ('Age', 'AGE'),
    ('Female (\%)', 'FEMALE'),
    ('Married (\%)', 'MARRIED'),
    ('Family size', 'FAMSIZE'),
    ('Number of children', 'NCHILD'),
    ('Full-time employed (\%)', 'FT'),
    ('In labor force (\%)', 'IN_LF'),
    ('Any employment (\%)', 'EMPLOYED'),
    ('Usual hours worked', 'UHRSWORK')
]

for var_name, var_code in variables:
    if var_code is None:
        # Sample size row
        vals = [len(groups['treat_pre']), len(groups['control_pre']),
                len(groups['treat_post']), len(groups['control_post'])]
        latex_table1 += f"{var_name} & {vals[0]:,} & {vals[1]:,} & {vals[2]:,} & {vals[3]:,} \\\\\n"
    elif '%' in var_name:
        vals = [weighted_mean(groups[g][var_code], groups[g]['PERWT'])*100
                for g in ['treat_pre', 'control_pre', 'treat_post', 'control_post']]
        latex_table1 += f"{var_name} & {vals[0]:.2f} & {vals[1]:.2f} & {vals[2]:.2f} & {vals[3]:.2f} \\\\\n"
    else:
        vals = [weighted_mean(groups[g][var_code], groups[g]['PERWT'])
                for g in ['treat_pre', 'control_pre', 'treat_post', 'control_post']]
        latex_table1 += f"{var_name} & {vals[0]:.2f} & {vals[1]:.2f} & {vals[2]:.2f} & {vals[3]:.2f} \\\\\n"

latex_table1 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Treatment group consists of individuals aged 26-30 in June 2012; control group consists of individuals aged 31-35 in June 2012. All statistics are weighted using ACS person weights (PERWT). Pre-DACA period: 2008-2011; Post-DACA period: 2013-2016.
\end{tablenotes}
\end{table}
"""

print(latex_table1)

# ============================================================================
# TABLE 2: DiD Results
# ============================================================================
print("\n" + "="*60)
print("TABLE 2: Difference-in-Differences 2x2 Table")
print("="*60)

# Calculate weighted means
did = {}
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE']==eligible) & (df['AFTER']==after)]
        did[(eligible, after)] = weighted_mean(subset['FT'], subset['PERWT'])

latex_table2 = r"""
\begin{table}[htbp]
\centering
\caption{Difference-in-Differences: Full-Time Employment Rates}
\label{tab:did}
\begin{tabular}{lccc}
\toprule
 & Pre-DACA & Post-DACA & Difference \\
\midrule
"""
# Control
diff_c = did[(0,1)] - did[(0,0)]
latex_table2 += f"Control (Ages 31-35) & {did[(0,0)]*100:.2f}\\% & {did[(0,1)]*100:.2f}\\% & {diff_c*100:.2f} pp \\\\\n"

# Treatment
diff_t = did[(1,1)] - did[(1,0)]
latex_table2 += f"Treatment (Ages 26-30) & {did[(1,0)]*100:.2f}\\% & {did[(1,1)]*100:.2f}\\% & {diff_t*100:.2f} pp \\\\\n"

latex_table2 += r"\midrule" + "\n"
latex_table2 += f"Diff-in-Diff & & & {(diff_t - diff_c)*100:.2f} pp \\\\\n"

latex_table2 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Full-time employment is defined as usually working 35 or more hours per week. Rates are weighted using ACS person weights. pp = percentage points.
\end{tablenotes}
\end{table}
"""

print(latex_table2)

# ============================================================================
# TABLE 3: Main Regression Results
# ============================================================================
print("\n" + "="*60)
print("TABLE 3: Main Regression Results")
print("="*60)

# Run models
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
model4 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + C(YEAR)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
model5 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + C(YEAR) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Clustered SE
model6 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD',
                 data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

def stars(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.1: return "*"
    return ""

latex_table3 = r"""
\begin{table}[htbp]
\centering
\caption{Effect of DACA Eligibility on Full-Time Employment}
\label{tab:regression}
\begin{tabular}{lcccccc}
\toprule
 & (1) & (2) & (3) & (4) & (5) & (6) \\
\midrule
"""

# DiD coefficient
row = "ELIGIBLE $\\times$ AFTER"
for m in [model1, model2, model3, model4, model5, model6]:
    coef = m.params['ELIGIBLE_AFTER']
    se = m.bse['ELIGIBLE_AFTER']
    p = m.pvalues['ELIGIBLE_AFTER']
    row += f" & {coef:.4f}{stars(p)}"
latex_table3 += row + " \\\\\n"

# SE row
row = ""
for m in [model1, model2, model3, model4, model5, model6]:
    se = m.bse['ELIGIBLE_AFTER']
    row += f" & ({se:.4f})"
latex_table3 += row + " \\\\\n"

# ELIGIBLE main effect
row = "ELIGIBLE"
for m in [model1, model2, model3, model4, model5, model6]:
    coef = m.params['ELIGIBLE']
    se = m.bse['ELIGIBLE']
    p = m.pvalues['ELIGIBLE']
    row += f" & {coef:.4f}{stars(p)}"
latex_table3 += row + " \\\\\n"

row = ""
for m in [model1, model2, model3, model4, model5, model6]:
    se = m.bse['ELIGIBLE']
    row += f" & ({se:.4f})"
latex_table3 += row + " \\\\\n"

# AFTER effect (where applicable)
row = "AFTER"
for i, m in enumerate([model1, model2, model3, model4, model5, model6]):
    if 'AFTER' in m.params:
        coef = m.params['AFTER']
        p = m.pvalues['AFTER']
        row += f" & {coef:.4f}{stars(p)}"
    else:
        row += " & --"
latex_table3 += row + " \\\\\n"

# Model specifications
latex_table3 += r"\midrule" + "\n"
latex_table3 += "Weighted & No & Yes & Yes & Yes & Yes & Yes \\\\\n"
latex_table3 += "Demographics & No & No & Yes & Yes & Yes & Yes \\\\\n"
latex_table3 += "Year FE & No & No & No & Yes & Yes & No \\\\\n"
latex_table3 += "State FE & No & No & No & No & Yes & No \\\\\n"
latex_table3 += "Clustered SE & No & No & No & No & No & Yes \\\\\n"

# R-squared and N
row = "R$^2$"
for m in [model1, model2, model3, model4, model5, model6]:
    row += f" & {m.rsquared:.3f}"
latex_table3 += row + " \\\\\n"

row = "N"
for m in [model1, model2, model3, model4, model5, model6]:
    row += f" & {int(m.nobs):,}"
latex_table3 += row + " \\\\\n"

latex_table3 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Dependent variable is full-time employment (1 = works 35+ hours/week, 0 otherwise). Standard errors in parentheses. Column (1) uses unweighted OLS. Columns (2)-(5) use weighted least squares with heteroskedasticity-robust standard errors. Column (6) uses clustered standard errors by state. Demographics include female, married, family size, and number of children. *** p$<$0.01, ** p$<$0.05, * p$<$0.1.
\end{tablenotes}
\end{table}
"""

print(latex_table3)

# ============================================================================
# TABLE 4: Heterogeneity Analysis
# ============================================================================
print("\n" + "="*60)
print("TABLE 4: Heterogeneity Analysis")
print("="*60)

latex_table4 = r"""
\begin{table}[htbp]
\centering
\caption{Heterogeneous Effects by Subgroup}
\label{tab:heterogeneity}
\begin{tabular}{lcccc}
\toprule
Subgroup & Coefficient & SE & p-value & N \\
\midrule
"""

# Full sample
model_full = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
latex_table4 += f"Full Sample & {model_full.params['ELIGIBLE_AFTER']:.4f}{stars(model_full.pvalues['ELIGIBLE_AFTER'])} & ({model_full.bse['ELIGIBLE_AFTER']:.4f}) & {model_full.pvalues['ELIGIBLE_AFTER']:.4f} & {int(model_full.nobs):,} \\\\\n"

latex_table4 += r"\midrule" + "\n"
latex_table4 += r"\multicolumn{5}{l}{\textit{By Gender}} \\\\" + "\n"

# By gender
for gender, label in [(0, 'Male'), (1, 'Female')]:
    df_g = df[df['FEMALE'] == gender]
    m = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_g, weights=df_g['PERWT']).fit(cov_type='HC1')
    latex_table4 += f"\\quad {label} & {m.params['ELIGIBLE_AFTER']:.4f}{stars(m.pvalues['ELIGIBLE_AFTER'])} & ({m.bse['ELIGIBLE_AFTER']:.4f}) & {m.pvalues['ELIGIBLE_AFTER']:.4f} & {int(m.nobs):,} \\\\\n"

latex_table4 += r"\midrule" + "\n"
latex_table4 += r"\multicolumn{5}{l}{\textit{By Education Level}} \\\\" + "\n"

# By education
for educ in ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']:
    df_e = df[df['EDUC_RECODE'] == educ]
    if len(df_e) > 100:
        try:
            m = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_e, weights=df_e['PERWT']).fit(cov_type='HC1')
            latex_table4 += f"\\quad {educ} & {m.params['ELIGIBLE_AFTER']:.4f}{stars(m.pvalues['ELIGIBLE_AFTER'])} & ({m.bse['ELIGIBLE_AFTER']:.4f}) & {m.pvalues['ELIGIBLE_AFTER']:.4f} & {int(m.nobs):,} \\\\\n"
        except:
            pass

latex_table4 += r"\midrule" + "\n"
latex_table4 += r"\multicolumn{5}{l}{\textit{By Marital Status}} \\\\" + "\n"

# By marital status
for married, label in [(1, 'Married'), (0, 'Not Married')]:
    df_m = df[df['MARRIED'] == married]
    m = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_m, weights=df_m['PERWT']).fit(cov_type='HC1')
    latex_table4 += f"\\quad {label} & {m.params['ELIGIBLE_AFTER']:.4f}{stars(m.pvalues['ELIGIBLE_AFTER'])} & ({m.bse['ELIGIBLE_AFTER']:.4f}) & {m.pvalues['ELIGIBLE_AFTER']:.4f} & {int(m.nobs):,} \\\\\n"

latex_table4 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Each row reports the DiD coefficient (ELIGIBLE $\times$ AFTER) from a separate weighted regression. Standard errors are heteroskedasticity-robust. *** p$<$0.01, ** p$<$0.05, * p$<$0.1.
\end{tablenotes}
\end{table}
"""

print(latex_table4)

# ============================================================================
# TABLE 5: Event Study Coefficients
# ============================================================================
print("\n" + "="*60)
print("TABLE 5: Event Study Results")
print("="*60)

# Create year interactions
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'ELIG_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

event_formula = 'FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIG_2008 + ELIG_2009 + ELIG_2010 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016'
model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

latex_table5 = r"""
\begin{table}[htbp]
\centering
\caption{Event Study: Year-Specific Treatment Effects}
\label{tab:eventstudy}
\begin{tabular}{lcccc}
\toprule
Year & Coefficient & SE & 95\% CI & p-value \\
\midrule
\multicolumn{5}{l}{\textit{Pre-DACA Period}} \\
"""

for year in [2008, 2009, 2010]:
    var = f'ELIG_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci = model_event.conf_int().loc[var]
    p = model_event.pvalues[var]
    latex_table5 += f"{year} & {coef:.4f}{stars(p)} & ({se:.4f}) & [{ci[0]:.4f}, {ci[1]:.4f}] & {p:.4f} \\\\\n"

latex_table5 += "2011 (ref) & 0.0000 & -- & -- & -- \\\\\n"
latex_table5 += r"\midrule" + "\n"
latex_table5 += r"\multicolumn{5}{l}{\textit{Post-DACA Period}} \\\\" + "\n"

for year in [2013, 2014, 2015, 2016]:
    var = f'ELIG_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci = model_event.conf_int().loc[var]
    p = model_event.pvalues[var]
    latex_table5 += f"{year} & {coef:.4f}{stars(p)} & ({se:.4f}) & [{ci[0]:.4f}, {ci[1]:.4f}] & {p:.4f} \\\\\n"

latex_table5 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Coefficients represent the interaction of ELIGIBLE with year indicators (ELIGIBLE $\times$ Year). Reference year is 2011. Standard errors are heteroskedasticity-robust. *** p$<$0.01, ** p$<$0.05, * p$<$0.1.
\end{tablenotes}
\end{table}
"""

print(latex_table5)

# ============================================================================
# TABLE 6: Alternative Outcomes
# ============================================================================
print("\n" + "="*60)
print("TABLE 6: Effects on Alternative Outcomes")
print("="*60)

latex_table6 = r"""
\begin{table}[htbp]
\centering
\caption{Effects on Alternative Labor Market Outcomes}
\label{tab:outcomes}
\begin{tabular}{lcccc}
\toprule
Outcome & Coefficient & SE & p-value & Mean (Pre, Treat) \\
\midrule
"""

outcomes = [
    ('Full-Time Employment', 'FT'),
    ('Any Employment', 'EMPLOYED'),
    ('Labor Force Participation', 'IN_LF'),
    ('Usual Hours Worked', 'UHRSWORK')
]

for label, var in outcomes:
    m = smf.wls(f'{var} ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
    pre_treat_mean = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)][var],
                                   df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['PERWT'])
    latex_table6 += f"{label} & {m.params['ELIGIBLE_AFTER']:.4f}{stars(m.pvalues['ELIGIBLE_AFTER'])} & ({m.bse['ELIGIBLE_AFTER']:.4f}) & {m.pvalues['ELIGIBLE_AFTER']:.4f} & {pre_treat_mean:.3f} \\\\\n"

latex_table6 += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Each row reports results from a separate weighted DiD regression. The outcome variable is listed in the first column. The last column shows the weighted mean for the treatment group in the pre-DACA period. Standard errors are heteroskedasticity-robust. *** p$<$0.01, ** p$<$0.05, * p$<$0.1.
\end{tablenotes}
\end{table}
"""

print(latex_table6)

# Save all tables to a file
with open(f'{output_path}/latex_tables.tex', 'w') as f:
    f.write("% LaTeX tables for DACA Replication Report\n\n")
    f.write(latex_table1)
    f.write("\n\\clearpage\n\n")
    f.write(latex_table2)
    f.write("\n\\clearpage\n\n")
    f.write(latex_table3)
    f.write("\n\\clearpage\n\n")
    f.write(latex_table4)
    f.write("\n\\clearpage\n\n")
    f.write(latex_table5)
    f.write("\n\\clearpage\n\n")
    f.write(latex_table6)

print("\n" + "="*60)
print("All tables saved to latex_tables.tex")
print("="*60)
