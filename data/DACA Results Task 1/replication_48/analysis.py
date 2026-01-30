"""
DACA Replication Study - Analysis Script
=========================================
Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States (2013-2016)?

Author: Independent Replication
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 80)
print("DACA REPLICATION STUDY - DATA ANALYSIS")
print("=" * 80)

# =============================================================================
# STEP 1: Load and filter data
# =============================================================================
print("\n[1] Loading data...")

# Define columns we need to reduce memory
cols_needed = [
    'YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'UHRSWORK', 'LABFORCE',
    'MARST', 'STATEFIP', 'METRO'
]

# Load data in chunks for memory efficiency
chunks = []
chunk_size = 500000

for chunk in pd.read_csv(
    r"C:\Users\seraf\DACA Results Task 1\replication_48\data\data.csv",
    usecols=cols_needed,
    chunksize=chunk_size,
    low_memory=False
):
    # Filter early to reduce memory: Hispanic-Mexican (HISPAN=1) born in Mexico (BPL=200)
    # Keep non-citizens (CITIZEN=3) for analysis
    filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Mexican Hispanic origin
        (chunk['BPL'] == 200)      # Born in Mexico
    ]
    chunks.append(filtered)
    print(f"  Processed chunk, kept {len(filtered):,} rows")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal observations after initial filter: {len(df):,}")

# =============================================================================
# STEP 2: Create analysis variables
# =============================================================================
print("\n[2] Creating analysis variables...")

# Full-time employment outcome (35+ hours per week)
# UHRSWORK = 0 means not working or N/A
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# In labor force indicator
df['in_labor_force'] = (df['LABFORCE'] == 2).astype(int)

# Post-DACA period (2013-2016)
# Exclude 2012 because DACA was implemented mid-year (June 15, 2012)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Create age at immigration (approximate)
# YRIMMIG = 0 means native born or N/A
df['age_at_immig'] = np.where(
    df['YRIMMIG'] > 0,
    df['YRIMMIG'] - df['BIRTHYR'],
    np.nan
)

# =============================================================================
# STEP 3: Define DACA eligibility
# =============================================================================
print("\n[3] Defining DACA eligibility criteria...")

"""
DACA Eligibility Criteria (June 15, 2012):
1. Arrived in US before age 16
2. Under age 31 as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (immigrated by 2007)
4. Present in US on June 15, 2012 with no lawful status (non-citizen)

Since we only have year of birth (not month), we use conservative estimates:
- Born in 1982 or later ensures under 31 on June 15, 2012
- Immigration by 2007 ensures continuous presence since June 15, 2007
"""

# Non-citizen status (required for undocumented status proxy)
df['non_citizen'] = (df['CITIZEN'] == 3).astype(int)

# Immigrated before age 16
df['arrived_young'] = (df['age_at_immig'] < 16).astype(int)

# Under 31 as of June 15, 2012 (born 1982 or later to be conservative)
df['young_enough'] = (df['BIRTHYR'] >= 1982).astype(int)

# Continuous presence: immigrated by 2007
df['continuous_presence'] = ((df['YRIMMIG'] > 0) & (df['YRIMMIG'] <= 2007)).astype(int)

# DACA eligible: meets ALL criteria
df['daca_eligible'] = (
    (df['non_citizen'] == 1) &
    (df['arrived_young'] == 1) &
    (df['young_enough'] == 1) &
    (df['continuous_presence'] == 1)
).astype(int)

print(f"  Non-citizens: {df['non_citizen'].sum():,}")
print(f"  Arrived young (<16): {df['arrived_young'].sum():,}")
print(f"  Young enough (born>=1982): {df['young_enough'].sum():,}")
print(f"  Continuous presence (immig<=2007): {df['continuous_presence'].sum():,}")
print(f"  DACA eligible: {df['daca_eligible'].sum():,}")

# =============================================================================
# STEP 4: Create control group
# =============================================================================
print("\n[4] Creating control group...")

"""
Control group: Mexican-born non-citizens who are NOT DACA eligible
We'll use those who:
- Are non-citizens
- Failed age criteria (too old - born before 1982)
- But otherwise similar: arrived as children or young adults

This creates a plausible counterfactual group.
"""

# Control: non-citizen, not DACA eligible, but arrived in US
df['control'] = (
    (df['non_citizen'] == 1) &
    (df['daca_eligible'] == 0) &
    (df['YRIMMIG'] > 0)  # Has valid immigration year
).astype(int)

# Analysis sample: either DACA eligible or control
df['in_sample'] = ((df['daca_eligible'] == 1) | (df['control'] == 1)).astype(int)

print(f"  Control group: {df['control'].sum():,}")
print(f"  Analysis sample: {df['in_sample'].sum():,}")

# =============================================================================
# STEP 5: Additional restrictions and covariates
# =============================================================================
print("\n[5] Applying additional sample restrictions...")

# Keep working-age adults (18-64) for employment analysis
df['working_age'] = ((df['AGE'] >= 18) & (df['AGE'] <= 64)).astype(int)

# Exclude 2012 from analysis (treatment timing unclear)
df['analysis_year'] = (df['YEAR'] != 2012).astype(int)

# Create final analysis sample
analysis_df = df[
    (df['in_sample'] == 1) &
    (df['working_age'] == 1) &
    (df['analysis_year'] == 1)
].copy()

print(f"  Final analysis sample: {len(analysis_df):,}")

# Create demographic covariates
analysis_df['female'] = (analysis_df['SEX'] == 2).astype(int)
analysis_df['married'] = (analysis_df['MARST'].isin([1, 2])).astype(int)

# Education categories
analysis_df['educ_hs'] = (analysis_df['EDUC'] >= 6).astype(int)  # HS or more
analysis_df['educ_college'] = (analysis_df['EDUC'] >= 7).astype(int)  # Some college+

# Age squared for non-linear age effects
analysis_df['age_sq'] = analysis_df['AGE'] ** 2

# Years in US
analysis_df['years_in_us'] = analysis_df['YEAR'] - analysis_df['YRIMMIG']
analysis_df.loc[analysis_df['years_in_us'] < 0, 'years_in_us'] = np.nan

# =============================================================================
# STEP 6: Summary statistics
# =============================================================================
print("\n[6] Generating summary statistics...")

# Pre-period (2006-2011) vs Post-period (2013-2016)
pre_df = analysis_df[analysis_df['post'] == 0]
post_df = analysis_df[analysis_df['post'] == 1]

# By treatment status
pre_treat = pre_df[pre_df['daca_eligible'] == 1]
pre_control = pre_df[pre_df['daca_eligible'] == 0]
post_treat = post_df[post_df['daca_eligible'] == 1]
post_control = post_df[post_df['daca_eligible'] == 0]

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

def weighted_mean(x, w):
    """Calculate weighted mean"""
    valid = ~(x.isna() | w.isna())
    return np.average(x[valid], weights=w[valid])

def weighted_std(x, w):
    """Calculate weighted standard deviation"""
    valid = ~(x.isna() | w.isna())
    mean = np.average(x[valid], weights=w[valid])
    variance = np.average((x[valid] - mean)**2, weights=w[valid])
    return np.sqrt(variance)

summary_vars = ['fulltime', 'employed', 'in_labor_force', 'AGE', 'female',
                'married', 'educ_hs', 'years_in_us']

print("\nTable 1: Pre-period characteristics (2006-2011)")
print("-" * 60)
print(f"{'Variable':<20} {'DACA Eligible':>15} {'Control':>15}")
print("-" * 60)
for var in summary_vars:
    treat_mean = weighted_mean(pre_treat[var], pre_treat['PERWT'])
    ctrl_mean = weighted_mean(pre_control[var], pre_control['PERWT'])
    print(f"{var:<20} {treat_mean:>15.3f} {ctrl_mean:>15.3f}")

print(f"\n{'N (unweighted)':<20} {len(pre_treat):>15,} {len(pre_control):>15,}")

print("\n\nTable 2: Sample sizes by year and treatment status")
print("-" * 60)
sample_counts = analysis_df.groupby(['YEAR', 'daca_eligible']).size().unstack(fill_value=0)
sample_counts.columns = ['Control', 'DACA Eligible']
print(sample_counts)

# =============================================================================
# STEP 7: Difference-in-Differences Analysis
# =============================================================================
print("\n" + "=" * 80)
print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# Create interaction term
analysis_df['treat_post'] = analysis_df['daca_eligible'] * analysis_df['post']

# Simple DiD (no controls)
print("\n[7a] Simple DiD (no controls)...")
formula_simple = 'fulltime ~ daca_eligible + post + treat_post'
model_simple = smf.wls(formula_simple, data=analysis_df, weights=analysis_df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']}
)
print(model_simple.summary().tables[1])

# DiD with demographic controls
print("\n[7b] DiD with demographic controls...")
formula_demo = 'fulltime ~ daca_eligible + post + treat_post + AGE + age_sq + female + married + educ_hs'
model_demo = smf.wls(formula_demo, data=analysis_df, weights=analysis_df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']}
)
print(model_demo.summary().tables[1])

# DiD with year and state fixed effects
print("\n[7c] DiD with year fixed effects...")
analysis_df['year_factor'] = pd.Categorical(analysis_df['YEAR'])
formula_fe = 'fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + C(YEAR)'
model_fe = smf.wls(formula_fe, data=analysis_df, weights=analysis_df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']}
)

# Extract key coefficient
treat_post_idx = [i for i, name in enumerate(model_fe.params.index) if 'treat_post' in name][0]
coef = model_fe.params.iloc[treat_post_idx]
se = model_fe.bse.iloc[treat_post_idx]
ci_low = coef - 1.96 * se
ci_high = coef + 1.96 * se
pval = model_fe.pvalues.iloc[treat_post_idx]

print(f"\nPreferred Specification Results:")
print(f"  DiD Coefficient (treat_post): {coef:.4f}")
print(f"  Standard Error: {se:.4f}")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  P-value: {pval:.4f}")

# =============================================================================
# STEP 8: Robustness Checks
# =============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# 8a. Alternative age bandwidth
print("\n[8a] Alternative age bandwidth (ages 20-40)...")
robust_df_age = analysis_df[(analysis_df['AGE'] >= 20) & (analysis_df['AGE'] <= 40)]
model_robust_age = smf.wls(formula_fe, data=robust_df_age, weights=robust_df_age['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': robust_df_age['STATEFIP']}
)
idx = [i for i, name in enumerate(model_robust_age.params.index) if 'treat_post' in name][0]
print(f"  DiD Coefficient: {model_robust_age.params.iloc[idx]:.4f} (SE: {model_robust_age.bse.iloc[idx]:.4f})")

# 8b. Men only
print("\n[8b] Men only...")
robust_df_men = analysis_df[analysis_df['female'] == 0]
model_robust_men = smf.wls(formula_fe, data=robust_df_men, weights=robust_df_men['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': robust_df_men['STATEFIP']}
)
idx = [i for i, name in enumerate(model_robust_men.params.index) if 'treat_post' in name][0]
print(f"  DiD Coefficient: {model_robust_men.params.iloc[idx]:.4f} (SE: {model_robust_men.bse.iloc[idx]:.4f})")

# 8c. Women only
print("\n[8c] Women only...")
robust_df_women = analysis_df[analysis_df['female'] == 1]
model_robust_women = smf.wls(formula_fe, data=robust_df_women, weights=robust_df_women['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': robust_df_women['STATEFIP']}
)
idx = [i for i, name in enumerate(model_robust_women.params.index) if 'treat_post' in name][0]
print(f"  DiD Coefficient: {model_robust_women.params.iloc[idx]:.4f} (SE: {model_robust_women.bse.iloc[idx]:.4f})")

# 8d. Employment (any) as outcome
print("\n[8d] Employment (any hours) as outcome...")
formula_emp = 'employed ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + C(YEAR)'
model_emp = smf.wls(formula_emp, data=analysis_df, weights=analysis_df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']}
)
idx = [i for i, name in enumerate(model_emp.params.index) if 'treat_post' in name][0]
print(f"  DiD Coefficient: {model_emp.params.iloc[idx]:.4f} (SE: {model_emp.bse.iloc[idx]:.4f})")

# 8e. Labor force participation as outcome
print("\n[8e] Labor force participation as outcome...")
formula_lf = 'in_labor_force ~ daca_eligible + treat_post + AGE + age_sq + female + married + educ_hs + C(YEAR)'
model_lf = smf.wls(formula_lf, data=analysis_df, weights=analysis_df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']}
)
idx = [i for i, name in enumerate(model_lf.params.index) if 'treat_post' in name][0]
print(f"  DiD Coefficient: {model_lf.params.iloc[idx]:.4f} (SE: {model_lf.bse.iloc[idx]:.4f})")

# =============================================================================
# STEP 9: Event Study / Pre-trend Analysis
# =============================================================================
print("\n" + "=" * 80)
print("EVENT STUDY / PRE-TREND ANALYSIS")
print("=" * 80)

# Create year dummies interacted with treatment
for year in analysis_df['YEAR'].unique():
    analysis_df[f'treat_year_{year}'] = (analysis_df['daca_eligible'] * (analysis_df['YEAR'] == year)).astype(int)

# Use 2011 as reference year
year_vars = [f'treat_year_{y}' for y in sorted(analysis_df['YEAR'].unique()) if y != 2011]
formula_event = f'fulltime ~ daca_eligible + {" + ".join(year_vars)} + AGE + age_sq + female + married + educ_hs + C(YEAR)'

model_event = smf.wls(formula_event, data=analysis_df, weights=analysis_df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']}
)

print("\nEvent Study Coefficients (Reference: 2011)")
print("-" * 50)
for year in sorted(analysis_df['YEAR'].unique()):
    if year == 2011:
        print(f"  {year}: 0.0000 (reference)")
    else:
        var_name = f'treat_year_{year}'
        idx = [i for i, name in enumerate(model_event.params.index) if var_name in name]
        if idx:
            coef = model_event.params.iloc[idx[0]]
            se = model_event.bse.iloc[idx[0]]
            print(f"  {year}: {coef:.4f} (SE: {se:.4f})")

# =============================================================================
# STEP 10: Save key results for report
# =============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Create results dictionary
results = {
    'preferred_estimate': {
        'coefficient': coef,
        'std_error': se,
        'ci_lower': ci_low,
        'ci_upper': ci_high,
        'p_value': pval,
        'n_obs': len(analysis_df),
        'n_treated': analysis_df['daca_eligible'].sum(),
        'n_control': (analysis_df['daca_eligible'] == 0).sum()
    }
}

# Save to file
import json
with open(r"C:\Users\seraf\DACA Results Task 1\replication_48\results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results.json")

# =============================================================================
# STEP 11: Generate tables for LaTeX
# =============================================================================
print("\n[11] Generating LaTeX tables...")

# Table 1: Summary Statistics
print("\n% LaTeX Table 1: Summary Statistics")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(r"\caption{Summary Statistics: Pre-Period (2006-2011)}")
print(r"\label{tab:summary}")
print(r"\begin{tabular}{lcc}")
print(r"\hline\hline")
print(r"Variable & DACA Eligible & Control \\")
print(r"\hline")
for var in summary_vars:
    treat_mean = weighted_mean(pre_treat[var], pre_treat['PERWT'])
    ctrl_mean = weighted_mean(pre_control[var], pre_control['PERWT'])
    treat_sd = weighted_std(pre_treat[var], pre_treat['PERWT'])
    ctrl_sd = weighted_std(pre_control[var], pre_control['PERWT'])
    var_label = var.replace('_', ' ').title()
    print(f"{var_label} & {treat_mean:.3f} & {ctrl_mean:.3f} \\\\")
    print(f" & ({treat_sd:.3f}) & ({ctrl_sd:.3f}) \\\\")
print(r"\hline")
print(f"N (unweighted) & {len(pre_treat):,} & {len(pre_control):,} \\\\")
print(r"\hline\hline")
print(r"\end{tabular}")
print(r"\begin{tablenotes}")
print(r"\small")
print(r"\item Notes: Standard deviations in parentheses. Sample weights applied.")
print(r"\end{tablenotes}")
print(r"\end{table}")

# Table 2: Main Results
print("\n% LaTeX Table 2: Main Difference-in-Differences Results")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(r"\caption{Effect of DACA Eligibility on Full-Time Employment}")
print(r"\label{tab:main}")
print(r"\begin{tabular}{lccc}")
print(r"\hline\hline")
print(r" & (1) & (2) & (3) \\")
print(r" & Simple DiD & With Controls & Year FE \\")
print(r"\hline")

# Get coefficients from each model
models = [model_simple, model_demo, model_fe]
for i, model in enumerate(models, 1):
    idx = [j for j, name in enumerate(model.params.index) if 'treat_post' in name][0]
    coef_val = model.params.iloc[idx]
    se_val = model.bse.iloc[idx]
    pval_val = model.pvalues.iloc[idx]
    stars = ''
    if pval_val < 0.01:
        stars = '***'
    elif pval_val < 0.05:
        stars = '**'
    elif pval_val < 0.1:
        stars = '*'

print(r"DACA $\times$ Post & ", end='')
coefs_str = []
for model in models:
    idx = [j for j, name in enumerate(model.params.index) if 'treat_post' in name][0]
    coef_val = model.params.iloc[idx]
    se_val = model.bse.iloc[idx]
    pval_val = model.pvalues.iloc[idx]
    stars = ''
    if pval_val < 0.01:
        stars = '***'
    elif pval_val < 0.05:
        stars = '**'
    elif pval_val < 0.1:
        stars = '*'
    coefs_str.append(f"{coef_val:.4f}{stars}")
print(" & ".join(coefs_str) + r" \\")

print(r" & ", end='')
ses_str = []
for model in models:
    idx = [j for j, name in enumerate(model.params.index) if 'treat_post' in name][0]
    se_val = model.bse.iloc[idx]
    ses_str.append(f"({se_val:.4f})")
print(" & ".join(ses_str) + r" \\")

print(r"\hline")
print(r"Demographic Controls & No & Yes & Yes \\")
print(r"Year Fixed Effects & No & No & Yes \\")
print(f"Observations & {len(analysis_df):,} & {len(analysis_df):,} & {len(analysis_df):,} \\\\")
print(r"\hline\hline")
print(r"\end{tabular}")
print(r"\begin{tablenotes}")
print(r"\small")
print(r"\item Notes: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$. Standard errors clustered at state level in parentheses.")
print(r"\item Demographic controls include age, age squared, female, married, and high school education.")
print(r"\end{tablenotes}")
print(r"\end{table}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Final summary
print(f"""
FINAL RESULTS SUMMARY
=====================
Research Question: Effect of DACA eligibility on full-time employment

Preferred Estimate (Model with Year FE):
  - DiD Coefficient: {coef:.4f}
  - Standard Error: {se:.4f}
  - 95% CI: [{ci_low:.4f}, {ci_high:.4f}]
  - P-value: {pval:.4f}

Sample:
  - Total observations: {len(analysis_df):,}
  - DACA eligible: {analysis_df['daca_eligible'].sum():,}
  - Control group: {(analysis_df['daca_eligible'] == 0).sum():,}

Interpretation:
DACA eligibility is associated with a {coef*100:.2f} percentage point
{"increase" if coef > 0 else "decrease"} in the probability of full-time employment.
""")
