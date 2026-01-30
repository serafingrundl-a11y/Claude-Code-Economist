"""
DACA Replication Study - Extended Analysis and Visualizations
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_77\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

# Create variables
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)
df['HS_DEGREE_DUM'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['SOME_COLLEGE'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['TWO_YEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['BA_PLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Create dummies
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True)
df = pd.concat([df, year_dummies], axis=1)
year_cols = [c for c in df.columns if c.startswith('YEAR_')]

state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)
state_cols = [c for c in df.columns if c.startswith('STATE_')]

# ============================================================================
# FIGURE 1: Full-time Employment Trends by Treatment Status
# ============================================================================
print("Creating Figure 1: Employment trends...")

ft_by_year = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()

plt.figure(figsize=(10, 6))
plt.plot(ft_by_year.index, ft_by_year[1], 'b-o', linewidth=2, markersize=8, label='Treatment (ages 26-30)')
plt.plot(ft_by_year.index, ft_by_year[0], 'r--s', linewidth=2, markersize=8, label='Control (ages 31-35)')
plt.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.ylim(0.55, 0.75)
plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_77\figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure1_trends.png")

# ============================================================================
# FIGURE 2: Event Study Plot
# ============================================================================
print("Creating Figure 2: Event study...")

# Create year-specific interactions
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

df['ELIG_2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIG_2010'] = df['ELIGIBLE'] * df['YEAR_2010']
df['ELIG_2011'] = df['ELIGIBLE'] * df['YEAR_2011']
df['ELIG_2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIG_2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIG_2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIG_2016'] = df['ELIGIBLE'] * df['YEAR_2016']

event_formula = f'FT ~ ELIGIBLE + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIG_2009 + ELIG_2010 + ELIG_2011 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016 + {" + ".join(state_cols)} + FEMALE + MARRIED + NCHILD + HS_DEGREE_DUM + SOME_COLLEGE + TWO_YEAR + BA_PLUS'

model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [0]  # 2008 is reference
ses = [0]

for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    var = f'ELIG_{year}'
    coefs.append(model_event.params[var])
    ses.append(model_event.bse[var])

coefs = np.array(coefs)
ses = np.array(ses)

plt.figure(figsize=(10, 6))
plt.errorbar(years, coefs, yerr=1.96*ses, fmt='o-', capsize=5, capthick=2,
             linewidth=2, markersize=8, color='blue')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Coefficient (relative to 2008)', fontsize=12)
plt.title('Event Study: Year-Specific Treatment Effects', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_77\figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure2_eventstudy.png")

# ============================================================================
# TABLE: Summary Statistics
# ============================================================================
print("Creating summary statistics table...")

# Summary by group
summary_stats = []

for elig in [1, 0]:
    for after in [0, 1]:
        mask = (df['ELIGIBLE'] == elig) & (df['AFTER'] == after)
        subset = df[mask]

        group = 'Treatment' if elig == 1 else 'Control'
        period = 'Post' if after == 1 else 'Pre'

        stats_dict = {
            'Group': group,
            'Period': period,
            'N (unweighted)': len(subset),
            'N (weighted)': subset['PERWT'].sum(),
            'FT Rate': np.average(subset['FT'], weights=subset['PERWT']),
            'Mean Age': np.average(subset['AGE'], weights=subset['PERWT']),
            'Female %': np.average(subset['FEMALE'], weights=subset['PERWT']) * 100,
            'Married %': np.average(subset['MARRIED'], weights=subset['PERWT']) * 100,
            'Mean Children': np.average(subset['NCHILD'], weights=subset['PERWT']),
            'HS Degree %': np.average(subset['HS_DEGREE_DUM'], weights=subset['PERWT']) * 100,
            'Some College %': np.average(subset['SOME_COLLEGE'], weights=subset['PERWT']) * 100,
            'BA+ %': np.average(subset['BA_PLUS'], weights=subset['PERWT']) * 100
        }
        summary_stats.append(stats_dict)

summary_df = pd.DataFrame(summary_stats)
print("\nSummary Statistics by Group and Period:")
print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv(r'C:\Users\seraf\DACA Results Task 3\replication_77\summary_statistics.csv', index=False)
print("  Saved summary_statistics.csv")

# ============================================================================
# SUBGROUP ANALYSIS: By Gender
# ============================================================================
print("\nSubgroup Analysis by Gender...")

formula_full = f'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + {" + ".join(year_cols)} + {" + ".join(state_cols)} + MARRIED + NCHILD + HS_DEGREE_DUM + SOME_COLLEGE + TWO_YEAR + BA_PLUS'

# Males only
df_male = df[df['FEMALE'] == 0].copy()
model_male = smf.wls(formula_full, data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
print(f"Males: DiD = {model_male.params['ELIGIBLE_X_AFTER']:.4f} (SE: {model_male.bse['ELIGIBLE_X_AFTER']:.4f}), N = {int(model_male.nobs):,}")

# Females only
df_female = df[df['FEMALE'] == 1].copy()
model_female = smf.wls(formula_full, data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"Females: DiD = {model_female.params['ELIGIBLE_X_AFTER']:.4f} (SE: {model_female.bse['ELIGIBLE_X_AFTER']:.4f}), N = {int(model_female.nobs):,}")

# ============================================================================
# ROBUSTNESS: Linear Probability Model vs Logit
# ============================================================================
print("\nComparing LPM to Logit (marginal effects)...")

# LPM (already done - Model 5)
formula5 = f'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + {" + ".join(year_cols)} + {" + ".join(state_cols)} + FEMALE + MARRIED + NCHILD + HS_DEGREE_DUM + SOME_COLLEGE + TWO_YEAR + BA_PLUS'
model_lpm = smf.wls(formula5, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print(f"LPM: DiD = {model_lpm.params['ELIGIBLE_X_AFTER']:.4f} (SE: {model_lpm.bse['ELIGIBLE_X_AFTER']:.4f})")

# Logit model
try:
    formula_logit = f'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + {" + ".join(year_cols)} + FEMALE + MARRIED + NCHILD + HS_DEGREE_DUM + SOME_COLLEGE + TWO_YEAR + BA_PLUS'
    model_logit = smf.logit(formula_logit, data=df).fit(disp=0)
    # Get marginal effect at mean
    coef = model_logit.params['ELIGIBLE_X_AFTER']
    pred = model_logit.predict(df).mean()
    me = coef * pred * (1 - pred)
    print(f"Logit (marginal effect): DiD ≈ {me:.4f}")
except Exception as e:
    print(f"Logit estimation issue: {e}")

# ============================================================================
# PLACEBO TEST: Pre-treatment period only (2008-2009 vs 2010-2011)
# ============================================================================
print("\nPlacebo Test (using only pre-treatment data)...")

df_pre = df[df['AFTER'] == 0].copy()
df_pre['PLACEBO_AFTER'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['PLACEBO_INTERACTION'] = df_pre['ELIGIBLE'] * df_pre['PLACEBO_AFTER']

# Recreate dummies for pre-period
year_dummies_pre = pd.get_dummies(df_pre['YEAR'], prefix='YEAR', drop_first=True)
year_cols_pre = list(year_dummies_pre.columns)
df_pre = pd.concat([df_pre.reset_index(drop=True), year_dummies_pre.reset_index(drop=True)], axis=1)

formula_placebo = f'FT ~ ELIGIBLE + PLACEBO_INTERACTION + {" + ".join(year_cols_pre)} + {" + ".join([c for c in state_cols if c in df_pre.columns])} + FEMALE + MARRIED + NCHILD + HS_DEGREE_DUM + SOME_COLLEGE + TWO_YEAR + BA_PLUS'

try:
    model_placebo = smf.wls(formula_placebo, data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
    print(f"Placebo DiD (2010-2011 vs 2008-2009): {model_placebo.params['PLACEBO_INTERACTION']:.4f} (SE: {model_placebo.bse['PLACEBO_INTERACTION']:.4f}), p = {model_placebo.pvalues['PLACEBO_INTERACTION']:.4f}")
except Exception as e:
    print(f"Placebo test issue: {e}")

# ============================================================================
# BALANCE TABLE: Pre-treatment characteristics
# ============================================================================
print("\nBalance Table (Pre-treatment characteristics)...")

df_pre_only = df[df['AFTER'] == 0].copy()

balance_vars = ['AGE', 'FEMALE', 'MARRIED', 'NCHILD', 'HS_DEGREE_DUM', 'SOME_COLLEGE', 'BA_PLUS', 'FT']

balance_results = []
for var in balance_vars:
    treat_mean = np.average(df_pre_only[df_pre_only['ELIGIBLE']==1][var],
                           weights=df_pre_only[df_pre_only['ELIGIBLE']==1]['PERWT'])
    control_mean = np.average(df_pre_only[df_pre_only['ELIGIBLE']==0][var],
                             weights=df_pre_only[df_pre_only['ELIGIBLE']==0]['PERWT'])
    diff = treat_mean - control_mean

    # Simple t-test for difference
    from scipy import stats
    treat_vals = df_pre_only[df_pre_only['ELIGIBLE']==1][var]
    control_vals = df_pre_only[df_pre_only['ELIGIBLE']==0][var]
    t_stat, p_val = stats.ttest_ind(treat_vals, control_vals)

    balance_results.append({
        'Variable': var,
        'Treatment': treat_mean,
        'Control': control_mean,
        'Difference': diff,
        'p-value': p_val
    })

balance_df = pd.DataFrame(balance_results)
print(balance_df.to_string(index=False))
balance_df.to_csv(r'C:\Users\seraf\DACA Results Task 3\replication_77\balance_table.csv', index=False)
print("  Saved balance_table.csv")

# ============================================================================
# Save All Results to a Single File
# ============================================================================
print("\nSaving all results...")

with open(r'C:\Users\seraf\DACA Results Task 3\replication_77\analysis_results.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - COMPLETE RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("1. MAIN RESULTS (Model 5 with robust SEs)\n")
    f.write("-"*50 + "\n")
    f.write(f"DiD Estimate: {model_lpm.params['ELIGIBLE_X_AFTER']:.4f}\n")
    f.write(f"Robust Standard Error: {model_lpm.bse['ELIGIBLE_X_AFTER']:.4f}\n")
    ci = model_lpm.conf_int().loc['ELIGIBLE_X_AFTER']
    f.write(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")
    f.write(f"p-value: {model_lpm.pvalues['ELIGIBLE_X_AFTER']:.4f}\n")
    f.write(f"Sample Size: {int(model_lpm.nobs):,}\n\n")

    f.write("2. SUBGROUP RESULTS\n")
    f.write("-"*50 + "\n")
    f.write(f"Males: DiD = {model_male.params['ELIGIBLE_X_AFTER']:.4f} (SE: {model_male.bse['ELIGIBLE_X_AFTER']:.4f})\n")
    f.write(f"Females: DiD = {model_female.params['ELIGIBLE_X_AFTER']:.4f} (SE: {model_female.bse['ELIGIBLE_X_AFTER']:.4f})\n\n")

    f.write("3. SUMMARY STATISTICS\n")
    f.write("-"*50 + "\n")
    f.write(summary_df.to_string(index=False) + "\n\n")

    f.write("4. BALANCE TABLE (Pre-treatment)\n")
    f.write("-"*50 + "\n")
    f.write(balance_df.to_string(index=False) + "\n")

print("  Saved analysis_results.txt")
print("\nExtended analysis complete!")
