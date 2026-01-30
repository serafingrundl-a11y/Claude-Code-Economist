#!/usr/bin/env python3
"""
DACA Replication Analysis - Task 46
Analyzing the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the United States.

Author: Anonymous (for blind review)
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# For saving outputs
import os

print("="*70)
print("DACA REPLICATION ANALYSIS - TASK 46")
print("="*70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

# Read the data (this may take a while given file size)
data_path = "data/data.csv"
print(f"    Reading from: {data_path}")

# Read in chunks to handle large file
chunks = []
chunk_size = 500000
for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
    # Pre-filter to reduce memory: Mexican-born Hispanics only
    chunk_filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Hispanic-Mexican
        (chunk['BPL'] == 200)     # Born in Mexico
    ]
    chunks.append(chunk_filtered)
    print(f"    Processed chunk, kept {len(chunk_filtered)} Mexican-born Hispanics")

df = pd.concat(chunks, ignore_index=True)
print(f"\n    Total Mexican-born Hispanic observations: {len(df):,}")

# =============================================================================
# 2. DATA CLEANING AND VARIABLE CREATION
# =============================================================================
print("\n[2] Data cleaning and variable creation...")

# Keep only working-age individuals (16-64)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
print(f"    After age restriction (16-64): {len(df):,}")

# Keep only non-citizens (CITIZEN = 3)
# This is our population of interest for DACA eligibility
df = df[df['CITIZEN'] == 3].copy()
print(f"    After restricting to non-citizens: {len(df):,}")

# Exclude 2012 (implementation year - ambiguous pre/post)
df = df[df['YEAR'] != 2012].copy()
print(f"    After excluding 2012: {len(df):,}")

# Create post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"    Post-DACA period observations: {df['post'].sum():,}")
print(f"    Pre-DACA period observations: {(df['post']==0).sum():,}")

# =============================================================================
# 3. DEFINE DACA ELIGIBILITY
# =============================================================================
print("\n[3] Defining DACA eligibility...")

# Eligibility criteria:
# 1. Arrived before 16th birthday
# 2. Born after June 15, 1981 (not yet 31 on June 15, 2012)
# 3. Arrived by 2007 (continuous residence since June 15, 2007)
# 4. Not a citizen (already filtered)

# Calculate age at immigration
# Age at immigration = Year of immigration - Birth year
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Criterion 1: Arrived before 16th birthday
df['arrived_young'] = (df['age_at_immig'] < 16).astype(int)

# Criterion 2: Born after June 15, 1981
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Born after June 15, 1981 means:
# - BIRTHYR > 1981, OR
# - BIRTHYR == 1981 AND BIRTHQTR >= 3 (born July or later)
df['born_after_cutoff'] = (
    (df['BIRTHYR'] > 1981) |
    ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))
).astype(int)

# Criterion 3: Arrived by 2007
df['arrived_by_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# Valid immigration year (not 0 or missing)
df['valid_yrimmig'] = (df['YRIMMIG'] > 0).astype(int)

# DACA eligible if all criteria met
df['eligible'] = (
    (df['arrived_young'] == 1) &
    (df['born_after_cutoff'] == 1) &
    (df['arrived_by_2007'] == 1) &
    (df['valid_yrimmig'] == 1)
).astype(int)

print(f"    DACA eligible: {df['eligible'].sum():,} ({100*df['eligible'].mean():.1f}%)")
print(f"    Non-eligible: {(df['eligible']==0).sum():,} ({100*(1-df['eligible'].mean()):.1f}%)")

# Check eligibility by period
print("\n    Eligibility by period:")
print(df.groupby(['post', 'eligible']).size().unstack(fill_value=0))

# =============================================================================
# 4. CREATE OUTCOME VARIABLE
# =============================================================================
print("\n[4] Creating outcome variable...")

# Full-time employment: Usually work 35+ hours per week
# UHRSWORK = 0 means N/A or not working
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator (in case we want to examine extensive margin)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

print(f"    Full-time employment rate: {100*df['fulltime'].mean():.1f}%")
print(f"    Employment rate: {100*df['employed'].mean():.1f}%")

# =============================================================================
# 5. CREATE CONTROL VARIABLES
# =============================================================================
print("\n[5] Creating control variables...")

# Age squared
df['age_sq'] = df['AGE'] ** 2

# Female indicator
df['female'] = (df['SEX'] == 2).astype(int)

# Married indicator
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
# EDUC: 0=N/A, 1=None, 2=Grade 1-4, 3=Grade 5-8, 4=Grade 9,
#       5=Grade 10, 6=Grade 11, 7=HS diploma, 8=1yr college,
#       9=2yr college, 10=4yr college, 11=5+ yr college
df['educ_hs'] = (df['EDUC'] >= 7).astype(int)  # High school or more
df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # College degree or more

print("    Control variables created: age_sq, female, married, educ_hs, educ_college")

# =============================================================================
# 6. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[6] Generating descriptive statistics...")

# Summary statistics by eligibility and period
summary_stats = df.groupby(['eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary Statistics by Eligibility and Period:")
print(summary_stats)

# Calculate means for each group-period
means = df.groupby(['eligible', 'post'])['fulltime'].mean()
print("\n    Full-time employment rates:")
print(f"    Eligible, Pre-DACA:  {means.get((1, 0), 'N/A'):.4f}")
print(f"    Eligible, Post-DACA: {means.get((1, 1), 'N/A'):.4f}")
print(f"    Non-eligible, Pre-DACA:  {means.get((0, 0), 'N/A'):.4f}")
print(f"    Non-eligible, Post-DACA: {means.get((0, 1), 'N/A'):.4f}")

# Simple DID estimate
if all(k in means.index for k in [(1, 0), (1, 1), (0, 0), (0, 1)]):
    did_simple = (means[(1, 1)] - means[(1, 0)]) - (means[(0, 1)] - means[(0, 0)])
    print(f"\n    Simple DID estimate: {did_simple:.4f}")

# =============================================================================
# 7. DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n[7] Running Difference-in-Differences regressions...")

# Create interaction term
df['eligible_post'] = df['eligible'] * df['post']

# Model 1: Basic DID (no controls)
print("\n--- Model 1: Basic DID ---")
model1 = smf.ols('fulltime ~ eligible + post + eligible_post', data=df).fit(
    cov_type='HC1'  # Robust standard errors
)
print(model1.summary())

# Model 2: DID with demographic controls
print("\n--- Model 2: DID with demographic controls ---")
model2 = smf.ols('fulltime ~ eligible + post + eligible_post + AGE + age_sq + female + married + educ_hs',
                 data=df).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DID with year fixed effects
print("\n--- Model 3: DID with year fixed effects ---")
df['year_factor'] = pd.Categorical(df['YEAR'])
model3 = smf.ols('fulltime ~ eligible + eligible_post + AGE + age_sq + female + married + educ_hs + C(YEAR)',
                 data=df).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DID with state and year fixed effects
print("\n--- Model 4: DID with state and year fixed effects ---")
model4 = smf.ols('fulltime ~ eligible + eligible_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                 data=df).fit(cov_type='HC1')
print(model4.summary())

# =============================================================================
# 8. WEIGHTED REGRESSION (PREFERRED SPECIFICATION)
# =============================================================================
print("\n[8] Running weighted regression (preferred specification)...")

# Model 5: Weighted DID with controls and fixed effects
print("\n--- Model 5: Weighted DID (PREFERRED) ---")
model5 = smf.wls('fulltime ~ eligible + eligible_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary())

# =============================================================================
# 9. ROBUSTNESS CHECKS
# =============================================================================
print("\n[9] Robustness checks...")

# Check 1: Alternative control group (arrived age 16-20 instead of 16+)
print("\n--- Robustness Check 1: Restricted control group (arrived age 16-25) ---")
df_robust1 = df[(df['eligible'] == 1) | ((df['age_at_immig'] >= 16) & (df['age_at_immig'] <= 25))].copy()
if len(df_robust1) > 0:
    model_r1 = smf.wls('fulltime ~ eligible + eligible_post + AGE + age_sq + female + married + educ_hs + C(YEAR)',
                       data=df_robust1, weights=df_robust1['PERWT']).fit(cov_type='HC1')
    print(f"    DID coefficient: {model_r1.params['eligible_post']:.4f}")
    print(f"    Standard error: {model_r1.bse['eligible_post']:.4f}")
    print(f"    P-value: {model_r1.pvalues['eligible_post']:.4f}")
    print(f"    Sample size: {len(df_robust1):,}")

# Check 2: Extensive margin (employed vs not)
print("\n--- Robustness Check 2: Employment (extensive margin) ---")
model_r2 = smf.wls('employed ~ eligible + eligible_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                   data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"    DID coefficient: {model_r2.params['eligible_post']:.4f}")
print(f"    Standard error: {model_r2.bse['eligible_post']:.4f}")
print(f"    P-value: {model_r2.pvalues['eligible_post']:.4f}")

# Check 3: Men only
print("\n--- Robustness Check 3: Men only ---")
df_men = df[df['female'] == 0]
model_r3 = smf.wls('fulltime ~ eligible + eligible_post + AGE + age_sq + married + educ_hs + C(YEAR) + C(STATEFIP)',
                   data=df_men, weights=df_men['PERWT']).fit(cov_type='HC1')
print(f"    DID coefficient: {model_r3.params['eligible_post']:.4f}")
print(f"    Standard error: {model_r3.bse['eligible_post']:.4f}")
print(f"    P-value: {model_r3.pvalues['eligible_post']:.4f}")
print(f"    Sample size: {len(df_men):,}")

# Check 4: Women only
print("\n--- Robustness Check 4: Women only ---")
df_women = df[df['female'] == 1]
model_r4 = smf.wls('fulltime ~ eligible + eligible_post + AGE + age_sq + married + educ_hs + C(YEAR) + C(STATEFIP)',
                   data=df_women, weights=df_women['PERWT']).fit(cov_type='HC1')
print(f"    DID coefficient: {model_r4.params['eligible_post']:.4f}")
print(f"    Standard error: {model_r4.bse['eligible_post']:.4f}")
print(f"    P-value: {model_r4.pvalues['eligible_post']:.4f}")
print(f"    Sample size: {len(df_women):,}")

# =============================================================================
# 10. EVENT STUDY / PARALLEL TRENDS
# =============================================================================
print("\n[10] Event study / parallel trends analysis...")

# Create year-specific treatment effects
years = sorted(df['YEAR'].unique())
print(f"    Years in sample: {years}")

# Create year dummies interacted with eligibility
for year in years:
    df[f'eligible_year_{year}'] = (df['eligible'] == 1) & (df['YEAR'] == year)
    df[f'eligible_year_{year}'] = df[f'eligible_year_{year}'].astype(int)

# Reference year: 2011 (last pre-treatment year)
year_vars = [f'eligible_year_{y}' for y in years if y != 2011]
formula_event = 'fulltime ~ eligible + ' + ' + '.join(year_vars) + ' + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)'

model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
for year in years:
    if year != 2011:
        var = f'eligible_year_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        pval = model_event.pvalues[var]
        print(f"    {year}: coef = {coef:7.4f}, SE = {se:.4f}, p = {pval:.4f}")

# =============================================================================
# 11. SAVE RESULTS
# =============================================================================
print("\n[11] Saving results...")

# Extract key results for the report
results = {
    'preferred_model': 'Model 5 (Weighted DID with state and year FE)',
    'did_coefficient': model5.params['eligible_post'],
    'did_se': model5.bse['eligible_post'],
    'did_pvalue': model5.pvalues['eligible_post'],
    'did_ci_lower': model5.conf_int().loc['eligible_post', 0],
    'did_ci_upper': model5.conf_int().loc['eligible_post', 1],
    'n_obs': int(model5.nobs),
    'r_squared': model5.rsquared,
    'n_eligible_pre': len(df[(df['eligible']==1) & (df['post']==0)]),
    'n_eligible_post': len(df[(df['eligible']==1) & (df['post']==1)]),
    'n_control_pre': len(df[(df['eligible']==0) & (df['post']==0)]),
    'n_control_post': len(df[(df['eligible']==0) & (df['post']==1)]),
}

# Save results to file
with open('analysis_results.txt', 'w') as f:
    f.write("DACA REPLICATION ANALYSIS RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Preferred Specification: {results['preferred_model']}\n\n")
    f.write("MAIN RESULT:\n")
    f.write(f"  DID Coefficient: {results['did_coefficient']:.4f}\n")
    f.write(f"  Standard Error: {results['did_se']:.4f}\n")
    f.write(f"  P-value: {results['did_pvalue']:.4f}\n")
    f.write(f"  95% CI: [{results['did_ci_lower']:.4f}, {results['did_ci_upper']:.4f}]\n\n")
    f.write("SAMPLE SIZES:\n")
    f.write(f"  Total observations: {results['n_obs']:,}\n")
    f.write(f"  Eligible, pre-DACA: {results['n_eligible_pre']:,}\n")
    f.write(f"  Eligible, post-DACA: {results['n_eligible_post']:,}\n")
    f.write(f"  Control, pre-DACA: {results['n_control_pre']:,}\n")
    f.write(f"  Control, post-DACA: {results['n_control_post']:,}\n\n")
    f.write(f"R-squared: {results['r_squared']:.4f}\n")

print(f"    Results saved to analysis_results.txt")

# =============================================================================
# 12. CREATE TABLES FOR LATEX REPORT
# =============================================================================
print("\n[12] Creating tables for LaTeX report...")

# Table 1: Descriptive Statistics
desc_stats = df.groupby('eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'educ_college': 'mean',
    'fulltime': 'mean',
    'employed': 'mean',
    'PERWT': 'sum'
}).round(4)
desc_stats['N'] = df.groupby('eligible').size()

print("\nDescriptive Statistics by Eligibility Status:")
print(desc_stats)

# Save as LaTeX table
with open('table1_descriptives.tex', 'w') as f:
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Descriptive Statistics by DACA Eligibility Status}\n")
    f.write("\\label{tab:descriptives}\n")
    f.write("\\begin{tabular}{lcc}\n")
    f.write("\\hline\\hline\n")
    f.write(" & Non-Eligible & DACA-Eligible \\\\\n")
    f.write("\\hline\n")
    f.write(f"Age & {desc_stats.loc[0, 'AGE']:.2f} & {desc_stats.loc[1, 'AGE']:.2f} \\\\\n")
    f.write(f"Female (\\%) & {100*desc_stats.loc[0, 'female']:.1f} & {100*desc_stats.loc[1, 'female']:.1f} \\\\\n")
    f.write(f"Married (\\%) & {100*desc_stats.loc[0, 'married']:.1f} & {100*desc_stats.loc[1, 'married']:.1f} \\\\\n")
    f.write(f"High School+ (\\%) & {100*desc_stats.loc[0, 'educ_hs']:.1f} & {100*desc_stats.loc[1, 'educ_hs']:.1f} \\\\\n")
    f.write(f"College+ (\\%) & {100*desc_stats.loc[0, 'educ_college']:.1f} & {100*desc_stats.loc[1, 'educ_college']:.1f} \\\\\n")
    f.write(f"Full-time Employed (\\%) & {100*desc_stats.loc[0, 'fulltime']:.1f} & {100*desc_stats.loc[1, 'fulltime']:.1f} \\\\\n")
    f.write(f"Employed (\\%) & {100*desc_stats.loc[0, 'employed']:.1f} & {100*desc_stats.loc[1, 'employed']:.1f} \\\\\n")
    f.write("\\hline\n")
    f.write(f"N & {desc_stats.loc[0, 'N']:,} & {desc_stats.loc[1, 'N']:,} \\\\\n")
    f.write("\\hline\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}\n")
    f.write("\\small\n")
    f.write("\\item Note: Sample includes Mexican-born, Hispanic-Mexican, non-citizen individuals aged 16-64 from ACS 2006-2011 and 2013-2016. Full-time employment defined as usually working 35+ hours per week.\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{table}\n")

print("    Table 1 saved to table1_descriptives.tex")

# Table 2: Main DID Results
with open('table2_main_results.tex', 'w') as f:
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Effect of DACA Eligibility on Full-Time Employment}\n")
    f.write("\\label{tab:main_results}\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\\hline\n")
    f.write(" & (1) & (2) & (3) & (4) \\\\\n")
    f.write(" & Basic & Controls & Year FE & State \\& Year FE \\\\\n")
    f.write("\\hline\n")
    f.write(f"Eligible $\\times$ Post & {model1.params['eligible_post']:.4f} & {model2.params['eligible_post']:.4f} & {model3.params['eligible_post']:.4f} & {model5.params['eligible_post']:.4f} \\\\\n")
    f.write(f" & ({model1.bse['eligible_post']:.4f}) & ({model2.bse['eligible_post']:.4f}) & ({model3.bse['eligible_post']:.4f}) & ({model5.bse['eligible_post']:.4f}) \\\\\n")
    f.write(f"Eligible & {model1.params['eligible']:.4f} & {model2.params['eligible']:.4f} & {model3.params['eligible']:.4f} & {model5.params['eligible']:.4f} \\\\\n")
    f.write(f" & ({model1.bse['eligible']:.4f}) & ({model2.bse['eligible']:.4f}) & ({model3.bse['eligible']:.4f}) & ({model5.bse['eligible']:.4f}) \\\\\n")
    f.write("\\hline\n")
    f.write("Controls & No & Yes & Yes & Yes \\\\\n")
    f.write("Year FE & No & No & Yes & Yes \\\\\n")
    f.write("State FE & No & No & No & Yes \\\\\n")
    f.write("Weighted & No & No & No & Yes \\\\\n")
    f.write("\\hline\n")
    f.write(f"N & {int(model1.nobs):,} & {int(model2.nobs):,} & {int(model3.nobs):,} & {int(model5.nobs):,} \\\\\n")
    f.write(f"R-squared & {model1.rsquared:.4f} & {model2.rsquared:.4f} & {model3.rsquared:.4f} & {model5.rsquared:.4f} \\\\\n")
    f.write("\\hline\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}\n")
    f.write("\\small\n")
    f.write("\\item Note: Robust standard errors in parentheses. The dependent variable is a binary indicator for full-time employment (35+ hours/week). Controls include age, age squared, female, married, and high school education indicators. Column (4) uses person weights from ACS.\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{table}\n")

print("    Table 2 saved to table2_main_results.tex")

# Table 3: Event Study Results
event_results = []
for year in years:
    if year != 2011:
        var = f'eligible_year_{year}'
        event_results.append({
            'Year': year,
            'Coefficient': model_event.params[var],
            'SE': model_event.bse[var],
            'P-value': model_event.pvalues[var]
        })

event_df = pd.DataFrame(event_results)

with open('table3_event_study.tex', 'w') as f:
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Event Study: Year-Specific Treatment Effects}\n")
    f.write("\\label{tab:event_study}\n")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\hline\\hline\n")
    f.write("Year & Coefficient & Std. Error & P-value \\\\\n")
    f.write("\\hline\n")
    for _, row in event_df.iterrows():
        f.write(f"{int(row['Year'])} & {row['Coefficient']:.4f} & {row['SE']:.4f} & {row['P-value']:.4f} \\\\\n")
    f.write("\\hline\n")
    f.write("Reference: 2011 & --- & --- & --- \\\\\n")
    f.write("\\hline\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}\n")
    f.write("\\small\n")
    f.write("\\item Note: Coefficients represent the difference in full-time employment between eligible and non-eligible groups in each year, relative to 2011 (the last pre-DACA year). Weighted regression with state and year fixed effects, demographic controls, and robust standard errors.\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{table}\n")

print("    Table 3 saved to table3_event_study.tex")

# Table 4: Robustness Checks
with open('table4_robustness.tex', 'w') as f:
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Robustness Checks}\n")
    f.write("\\label{tab:robustness}\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\\hline\n")
    f.write(" & Restricted & Employment & Men & Women \\\\\n")
    f.write(" & Control & (Extensive) & Only & Only \\\\\n")
    f.write("\\hline\n")
    f.write(f"Eligible $\\times$ Post & {model_r1.params['eligible_post']:.4f} & {model_r2.params['eligible_post']:.4f} & {model_r3.params['eligible_post']:.4f} & {model_r4.params['eligible_post']:.4f} \\\\\n")
    f.write(f" & ({model_r1.bse['eligible_post']:.4f}) & ({model_r2.bse['eligible_post']:.4f}) & ({model_r3.bse['eligible_post']:.4f}) & ({model_r4.bse['eligible_post']:.4f}) \\\\\n")
    f.write("\\hline\n")
    f.write(f"N & {int(model_r1.nobs):,} & {int(model_r2.nobs):,} & {int(model_r3.nobs):,} & {int(model_r4.nobs):,} \\\\\n")
    f.write("\\hline\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}\n")
    f.write("\\small\n")
    f.write("\\item Note: Robust standard errors in parentheses. All specifications include demographic controls and year fixed effects, weighted by person weights. ``Restricted Control'' limits the control group to those who arrived at ages 16-25. ``Employment'' uses any employment as the outcome. ``Men Only'' and ``Women Only'' estimate effects separately by gender.\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{table}\n")

print("    Table 4 saved to table4_robustness.tex")

# =============================================================================
# 13. ADDITIONAL STATISTICS FOR REPORT
# =============================================================================
print("\n[13] Computing additional statistics for report...")

# Pre-trends test: joint significance of pre-period coefficients
pre_years = [y for y in years if y < 2011]
pre_vars = [f'eligible_year_{y}' for y in pre_years]
print(f"\n    Pre-DACA years for trends test: {pre_years}")

# Compute means by year and eligibility for trend visualization
yearly_means = df.groupby(['YEAR', 'eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
yearly_means.columns = ['Non-Eligible', 'Eligible']
print("\n    Full-time employment rates by year:")
print(yearly_means.round(4))

# Save yearly means
yearly_means.to_csv('yearly_means.csv')
print("    Yearly means saved to yearly_means.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nPREFERRED ESTIMATE:")
print(f"  Effect of DACA eligibility on full-time employment: {results['did_coefficient']:.4f}")
print(f"  Standard error: {results['did_se']:.4f}")
print(f"  95% CI: [{results['did_ci_lower']:.4f}, {results['did_ci_upper']:.4f}]")
print(f"  P-value: {results['did_pvalue']:.4f}")
print(f"\n  Interpretation: DACA eligibility is associated with a")
print(f"  {abs(results['did_coefficient'])*100:.2f} percentage point {'increase' if results['did_coefficient'] > 0 else 'decrease'}")
print(f"  in the probability of full-time employment.")
print(f"\n  Sample size: {results['n_obs']:,}")
print("="*70)
