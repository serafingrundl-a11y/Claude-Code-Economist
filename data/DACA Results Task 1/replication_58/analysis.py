"""
DACA Replication Study - Analysis Script
Research Question: Impact of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the US.

Author: Replication 58
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*70)

# =============================================================================
# 1. DATA LOADING AND INITIAL FILTERING
# =============================================================================
print("\n1. Loading and filtering data...")
print("-" * 50)

# Define data types to reduce memory usage
dtypes = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'EMPSTAT': 'int8',
    'LABFORCE': 'int8',
    'UHRSWORK': 'int8',
    'MARST': 'int8',
    'CLUSTER': 'int64',
    'STRATA': 'int64'
}

# Columns needed for analysis
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'EMPSTAT', 'LABFORCE', 'UHRSWORK', 'MARST',
               'CLUSTER', 'STRATA']

# Read data in chunks to manage memory
print("Reading data file (this may take a while due to 6GB file size)...")
chunk_size = 500000
chunks = []

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, dtype=dtypes,
                         chunksize=chunk_size, low_memory=False):
    # Initial filter: Hispanic-Mexican (HISPAN == 1) and born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"Total observations (Hispanic-Mexican, Mexico-born): {len(df):,}")

# =============================================================================
# 2. EXCLUDE 2012 (IMPLEMENTATION YEAR)
# =============================================================================
print("\n2. Excluding 2012 (implementation year)...")
print("-" * 50)
df = df[df['YEAR'] != 2012]
print(f"Observations after excluding 2012: {len(df):,}")

# =============================================================================
# 3. CREATE ANALYSIS VARIABLES
# =============================================================================
print("\n3. Creating analysis variables...")
print("-" * 50)

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"Post-DACA observations (2013-2016): {df['post'].sum():,}")
print(f"Pre-DACA observations (2006-2011): {(df['post'] == 0).sum():,}")

# Calculate approximate age at arrival
# YRIMMIG is year of immigration, if available
# For those with YRIMMIG > 0, we can compute approximate arrival age
df['age_at_arrival'] = np.where(
    df['YRIMMIG'] > 0,
    df['AGE'] - (df['YEAR'] - df['YRIMMIG']),
    np.nan
)

# DACA eligibility criteria:
# 1. Arrived before age 16
# 2. Born in 1982 or later (under 31 as of June 15, 2012)
# 3. In US since at least 2007 (YRIMMIG <= 2007)
# 4. Non-citizen (CITIZEN == 3)

# Create eligibility components
df['arrived_before_16'] = (df['age_at_arrival'] < 16).astype(int)
df['born_1982_plus'] = (df['BIRTHYR'] >= 1982).astype(int)
df['in_us_since_2007'] = (df['YRIMMIG'] <= 2007).astype(int)
df['non_citizen'] = (df['CITIZEN'] == 3).astype(int)

# Full DACA eligibility (meets all criteria)
df['daca_eligible'] = (
    (df['arrived_before_16'] == 1) &
    (df['born_1982_plus'] == 1) &
    (df['in_us_since_2007'] == 1) &
    (df['non_citizen'] == 1)
).astype(int)

print(f"\nDACA eligibility components:")
print(f"  Arrived before age 16: {df['arrived_before_16'].sum():,}")
print(f"  Born 1982 or later: {df['born_1982_plus'].sum():,}")
print(f"  In US since 2007: {df['in_us_since_2007'].sum():,}")
print(f"  Non-citizen: {df['non_citizen'].sum():,}")
print(f"  Fully DACA eligible: {df['daca_eligible'].sum():,}")

# Full-time employment outcome
# UHRSWORK >= 35 hours and employed (EMPSTAT == 1)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['employed'] == 1)).astype(int)
print(f"\nEmployed: {df['employed'].sum():,}")
print(f"Full-time employed (35+ hours): {df['fulltime'].sum():,}")

# =============================================================================
# 4. RESTRICT TO WORKING-AGE SAMPLE
# =============================================================================
print("\n4. Restricting to working-age sample (18-45)...")
print("-" * 50)

# Focus on working-age population
# Use 18-45 to capture both eligible young adults and slightly older comparison group
df_analysis = df[(df['AGE'] >= 18) & (df['AGE'] <= 45)].copy()
print(f"Working-age sample (18-45): {len(df_analysis):,}")
print(f"DACA eligible in working-age sample: {df_analysis['daca_eligible'].sum():,}")

# =============================================================================
# 5. CREATE CONTROL GROUP
# =============================================================================
print("\n5. Defining treatment and control groups...")
print("-" * 50)

# Control group strategy: Mexican-born, Hispanic-Mexican individuals who are
# similar but NOT DACA-eligible. Key distinction: arrived at age 16+ OR
# are citizens (naturalized).

# For a cleaner comparison, focus on non-citizens and use age-at-arrival cutoff
df_noncit = df_analysis[df_analysis['non_citizen'] == 1].copy()

# Create treatment indicator based on arrival age and birth year
# Treatment: Arrived < 16 AND born >= 1982 AND in US since 2007
# Control: Same population but arrived >= 16 (missed the "arrived as child" requirement)

# Alternative approach: Compare those who meet age/arrival criteria vs those who don't
# but are in similar age cohort

# Let's use a cleaner design:
# Treatment: Full DACA eligible
# Control: Similar age cohort (born 1982+) but arrived at 16 or older

df_noncit['treatment'] = df_noncit['daca_eligible']

# For control, include those who:
# - Are non-citizens
# - Were born 1982+ (same age cohort)
# - BUT arrived at age 16 or older
df_noncit['control_pool'] = (
    (df_noncit['non_citizen'] == 1) &
    (df_noncit['born_1982_plus'] == 1) &
    (df_noncit['arrived_before_16'] == 0)  # Key distinction: arrived at 16+
).astype(int)

# Analysis sample: Treatment OR Control pool
df_did = df_noncit[(df_noncit['treatment'] == 1) | (df_noncit['control_pool'] == 1)].copy()

print(f"Treatment group (DACA eligible): {df_did['treatment'].sum():,}")
print(f"Control group (arrived at 16+): {df_did['control_pool'].sum():,}")
print(f"Total DiD sample: {len(df_did):,}")

# =============================================================================
# 6. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*70)
print("6. DESCRIPTIVE STATISTICS")
print("="*70)

# Create female indicator
df_did['female'] = (df_did['SEX'] == 2).astype(int)

# Create education categories
df_did['less_than_hs'] = (df_did['EDUC'] < 6).astype(int)
df_did['high_school'] = ((df_did['EDUC'] >= 6) & (df_did['EDUC'] <= 7)).astype(int)
df_did['some_college'] = (df_did['EDUC'] == 8).astype(int)
df_did['college_plus'] = (df_did['EDUC'] >= 10).astype(int)

# Create married indicator
df_did['married'] = (df_did['MARST'].isin([1, 2])).astype(int)

# Summary stats by treatment status
def weighted_stats(df, var, weight='PERWT'):
    """Calculate weighted mean and std"""
    mean = np.average(df[var], weights=df[weight])
    variance = np.average((df[var] - mean)**2, weights=df[weight])
    return mean, np.sqrt(variance)

print("\nTable 1: Sample Characteristics by Treatment Status")
print("-" * 70)
print(f"{'Variable':<25} {'Treatment':<20} {'Control':<20}")
print(f"{'':25} {'Mean (SD)':<20} {'Mean (SD)':<20}")
print("-" * 70)

vars_to_summarize = ['AGE', 'female', 'married', 'less_than_hs', 'high_school',
                     'some_college', 'college_plus', 'fulltime', 'employed']
var_labels = ['Age', 'Female', 'Married', 'Less than HS', 'High School',
              'Some College', 'College+', 'Full-time Employed', 'Employed']

treat_df = df_did[df_did['treatment'] == 1]
control_df = df_did[df_did['treatment'] == 0]

summary_stats = []
for var, label in zip(vars_to_summarize, var_labels):
    t_mean, t_sd = weighted_stats(treat_df, var)
    c_mean, c_sd = weighted_stats(control_df, var)
    print(f"{label:<25} {t_mean:>6.3f} ({t_sd:.3f}){'':<4} {c_mean:>6.3f} ({c_sd:.3f})")
    summary_stats.append({
        'Variable': label,
        'Treatment Mean': t_mean,
        'Treatment SD': t_sd,
        'Control Mean': c_mean,
        'Control SD': c_sd
    })

print("-" * 70)
print(f"{'N':<25} {len(treat_df):>10,}{'':<10} {len(control_df):>10,}")

# Save summary stats
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_statistics.csv', index=False)

# =============================================================================
# 7. PRE-TREATMENT TRENDS CHECK
# =============================================================================
print("\n" + "="*70)
print("7. PRE-TREATMENT TRENDS")
print("="*70)

# Calculate mean full-time employment by year and treatment status
trends = df_did.groupby(['YEAR', 'treatment']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x)
    })
).reset_index()

print("\nFull-time Employment Rate by Year and Treatment Status:")
print("-" * 60)
print(f"{'Year':<8} {'Treatment':<15} {'Control':<15} {'Difference':<15}")
print("-" * 60)

trend_data = []
for year in sorted(df_did['YEAR'].unique()):
    treat_rate = trends[(trends['YEAR'] == year) & (trends['treatment'] == 1)]['fulltime_rate'].values
    control_rate = trends[(trends['YEAR'] == year) & (trends['treatment'] == 0)]['fulltime_rate'].values
    if len(treat_rate) > 0 and len(control_rate) > 0:
        diff = treat_rate[0] - control_rate[0]
        print(f"{year:<8} {treat_rate[0]*100:>12.1f}%{'':<3} {control_rate[0]*100:>12.1f}%{'':<3} {diff*100:>12.1f}pp")
        trend_data.append({
            'Year': year,
            'Treatment': treat_rate[0],
            'Control': control_rate[0],
            'Difference': diff
        })

trends_df = pd.DataFrame(trend_data)
trends_df.to_csv('trends_data.csv', index=False)

# =============================================================================
# 8. DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "="*70)
print("8. DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*70)

# Main DiD specification
# Y = β0 + β1*treatment + β2*post + β3*(treatment*post) + ε
# β3 is the DiD estimator

df_did['treat_post'] = df_did['treatment'] * df_did['post']

print("\n8.1 Basic DiD Specification (No Controls)")
print("-" * 50)

# Basic model without controls
model1 = smf.wls('fulltime ~ treatment + post + treat_post',
                  data=df_did, weights=df_did['PERWT'])
results1 = model1.fit(cov_type='cluster', cov_kwds={'groups': df_did['STATEFIP']})
print(results1.summary().tables[1])

print("\n8.2 DiD with Demographic Controls")
print("-" * 50)

# Model with controls
model2 = smf.wls('fulltime ~ treatment + post + treat_post + AGE + female + married + high_school + some_college + college_plus',
                  data=df_did, weights=df_did['PERWT'])
results2 = model2.fit(cov_type='cluster', cov_kwds={'groups': df_did['STATEFIP']})
print(results2.summary().tables[1])

print("\n8.3 DiD with Year and State Fixed Effects")
print("-" * 50)

# Create year dummies
df_did['year_str'] = df_did['YEAR'].astype(str)
df_did['state_str'] = df_did['STATEFIP'].astype(str)

# Model with fixed effects
model3 = smf.wls('fulltime ~ treatment + treat_post + C(YEAR) + C(STATEFIP) + AGE + female + married + high_school + some_college + college_plus',
                  data=df_did, weights=df_did['PERWT'])
results3 = model3.fit(cov_type='cluster', cov_kwds={'groups': df_did['STATEFIP']})

# Extract key coefficient
treat_post_coef = results3.params['treat_post']
treat_post_se = results3.bse['treat_post']
treat_post_pval = results3.pvalues['treat_post']
ci_low = treat_post_coef - 1.96 * treat_post_se
ci_high = treat_post_coef + 1.96 * treat_post_se

print(f"\nDiD Estimate (Treatment x Post):")
print(f"  Coefficient: {treat_post_coef:.4f}")
print(f"  Std. Error:  {treat_post_se:.4f}")
print(f"  95% CI:      [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  p-value:     {treat_post_pval:.4f}")

# =============================================================================
# 9. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*70)
print("9. ROBUSTNESS CHECKS")
print("="*70)

# 9.1 Separate by gender
print("\n9.1 Heterogeneity by Gender")
print("-" * 50)

for gender, gender_name in [(0, 'Male'), (1, 'Female')]:
    df_gender = df_did[df_did['female'] == gender]
    model_g = smf.wls('fulltime ~ treatment + treat_post + C(YEAR) + C(STATEFIP) + AGE + married + high_school + some_college + college_plus',
                      data=df_gender, weights=df_gender['PERWT'])
    results_g = model_g.fit(cov_type='cluster', cov_kwds={'groups': df_gender['STATEFIP']})
    coef = results_g.params['treat_post']
    se = results_g.bse['treat_post']
    print(f"{gender_name}: DiD estimate = {coef:.4f} (SE: {se:.4f}), N = {len(df_gender):,}")

# 9.2 Alternative control group: Use age-based comparison
print("\n9.2 Alternative Specification: Broader Age Range (18-50)")
print("-" * 50)

df_broad = df[(df['AGE'] >= 18) & (df['AGE'] <= 50) & (df['non_citizen'] == 1)].copy()
df_broad['post'] = (df_broad['YEAR'] >= 2013).astype(int)
df_broad['treatment'] = df_broad['daca_eligible']
df_broad['treat_post'] = df_broad['treatment'] * df_broad['post']
df_broad['female'] = (df_broad['SEX'] == 2).astype(int)
df_broad['married'] = (df_broad['MARST'].isin([1, 2])).astype(int)
df_broad['high_school'] = ((df_broad['EDUC'] >= 6) & (df_broad['EDUC'] <= 7)).astype(int)
df_broad['some_college'] = (df_broad['EDUC'] == 8).astype(int)
df_broad['college_plus'] = (df_broad['EDUC'] >= 10).astype(int)

model_broad = smf.wls('fulltime ~ treatment + treat_post + C(YEAR) + C(STATEFIP) + AGE + female + married + high_school + some_college + college_plus',
                      data=df_broad, weights=df_broad['PERWT'])
results_broad = model_broad.fit(cov_type='cluster', cov_kwds={'groups': df_broad['STATEFIP']})
print(f"DiD estimate = {results_broad.params['treat_post']:.4f} (SE: {results_broad.bse['treat_post']:.4f})")

# 9.3 Employment (any) instead of full-time
print("\n9.3 Alternative Outcome: Any Employment")
print("-" * 50)

model_emp = smf.wls('employed ~ treatment + treat_post + C(YEAR) + C(STATEFIP) + AGE + female + married + high_school + some_college + college_plus',
                    data=df_did, weights=df_did['PERWT'])
results_emp = model_emp.fit(cov_type='cluster', cov_kwds={'groups': df_did['STATEFIP']})
print(f"DiD estimate = {results_emp.params['treat_post']:.4f} (SE: {results_emp.bse['treat_post']:.4f})")

# =============================================================================
# 10. EVENT STUDY ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("10. EVENT STUDY ANALYSIS")
print("="*70)

# Create year-specific treatment effects (relative to 2011)
df_did['year_2006'] = ((df_did['YEAR'] == 2006) & (df_did['treatment'] == 1)).astype(int)
df_did['year_2007'] = ((df_did['YEAR'] == 2007) & (df_did['treatment'] == 1)).astype(int)
df_did['year_2008'] = ((df_did['YEAR'] == 2008) & (df_did['treatment'] == 1)).astype(int)
df_did['year_2009'] = ((df_did['YEAR'] == 2009) & (df_did['treatment'] == 1)).astype(int)
df_did['year_2010'] = ((df_did['YEAR'] == 2010) & (df_did['treatment'] == 1)).astype(int)
# 2011 is reference
df_did['year_2013'] = ((df_did['YEAR'] == 2013) & (df_did['treatment'] == 1)).astype(int)
df_did['year_2014'] = ((df_did['YEAR'] == 2014) & (df_did['treatment'] == 1)).astype(int)
df_did['year_2015'] = ((df_did['YEAR'] == 2015) & (df_did['treatment'] == 1)).astype(int)
df_did['year_2016'] = ((df_did['YEAR'] == 2016) & (df_did['treatment'] == 1)).astype(int)

model_event = smf.wls('fulltime ~ treatment + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + C(YEAR) + C(STATEFIP) + AGE + female + married + high_school + some_college + college_plus',
                       data=df_did, weights=df_did['PERWT'])
results_event = model_event.fit(cov_type='cluster', cov_kwds={'groups': df_did['STATEFIP']})

print("\nEvent Study Coefficients (Reference: 2011):")
print("-" * 50)
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_coefs = []
for year in event_years:
    if year == 2011:
        continue
    var_name = f'year_{year}'
    coef = results_event.params[var_name]
    se = results_event.bse[var_name]
    print(f"  {year}: {coef:>7.4f} (SE: {se:.4f})")
    event_coefs.append({'year': year, 'coef': coef, 'se': se})

event_df = pd.DataFrame(event_coefs)
event_df.to_csv('event_study_coefs.csv', index=False)

# =============================================================================
# 11. CREATE FIGURES
# =============================================================================
print("\n" + "="*70)
print("11. CREATING FIGURES")
print("="*70)

# Figure 1: Trends in full-time employment
plt.figure(figsize=(10, 6))
years = trends_df['Year'].values
plt.plot(years, trends_df['Treatment']*100, 'b-o', label='Treatment (DACA Eligible)', linewidth=2, markersize=8)
plt.plot(years, trends_df['Control']*100, 'r--s', label='Control (Arrived 16+)', linewidth=2, markersize=8)
plt.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-time Employment Rate (%)', fontsize=12)
plt.title('Full-time Employment Trends: Treatment vs. Control', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(years)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
print("Saved: figure1_trends.png/pdf")

# Figure 2: Event study plot
plt.figure(figsize=(10, 6))
event_years_plot = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs_plot = []
event_ses_plot = []
for year in event_years_plot:
    if year == 2011:
        event_coefs_plot.append(0)
        event_ses_plot.append(0)
    else:
        var_name = f'year_{year}'
        event_coefs_plot.append(results_event.params[var_name])
        event_ses_plot.append(results_event.bse[var_name])

event_coefs_plot = np.array(event_coefs_plot)
event_ses_plot = np.array(event_ses_plot)

plt.errorbar(event_years_plot, event_coefs_plot, yerr=1.96*event_ses_plot,
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8, color='blue')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.axvline(x=2012, color='gray', linestyle=':', linewidth=2)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Treatment Effect (Full-time Employment)', fontsize=12)
plt.title('Event Study: Effect of DACA Eligibility on Full-time Employment', fontsize=14)
plt.xticks(event_years_plot)
plt.grid(True, alpha=0.3)
# Add annotation
plt.annotate('DACA\nImplementation', xy=(2012, 0), xytext=(2012.5, 0.05),
             fontsize=10, ha='left')
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
print("Saved: figure2_event_study.png/pdf")

# Figure 3: Difference plot
plt.figure(figsize=(10, 6))
plt.bar(trends_df['Year'], trends_df['Difference']*100, color=['blue' if y < 2012 else 'green' for y in trends_df['Year']])
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.axvline(x=2011.5, color='red', linestyle='--', linewidth=2)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Difference in Full-time Employment (pp)', fontsize=12)
plt.title('Difference in Full-time Employment: Treatment - Control', fontsize=14)
plt.xticks(trends_df['Year'])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figure3_difference.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_difference.pdf', bbox_inches='tight')
print("Saved: figure3_difference.png/pdf")

# =============================================================================
# 12. SAVE REGRESSION RESULTS
# =============================================================================
print("\n" + "="*70)
print("12. SAVING RESULTS")
print("="*70)

# Save main results
results_summary = {
    'Specification': ['Basic DiD', 'With Demographics', 'With FE + Demographics'],
    'DiD_Estimate': [results1.params['treat_post'], results2.params['treat_post'], results3.params['treat_post']],
    'Std_Error': [results1.bse['treat_post'], results2.bse['treat_post'], results3.bse['treat_post']],
    'CI_Low': [results1.params['treat_post'] - 1.96*results1.bse['treat_post'],
               results2.params['treat_post'] - 1.96*results2.bse['treat_post'],
               results3.params['treat_post'] - 1.96*results3.bse['treat_post']],
    'CI_High': [results1.params['treat_post'] + 1.96*results1.bse['treat_post'],
                results2.params['treat_post'] + 1.96*results2.bse['treat_post'],
                results3.params['treat_post'] + 1.96*results3.bse['treat_post']],
    'P_Value': [results1.pvalues['treat_post'], results2.pvalues['treat_post'], results3.pvalues['treat_post']],
    'N': [int(results1.nobs), int(results2.nobs), int(results3.nobs)]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('regression_results.csv', index=False)
print("Saved: regression_results.csv")

# Save full model summary
with open('model_summary.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - FULL MODEL RESULTS\n")
    f.write("="*70 + "\n\n")

    f.write("PREFERRED SPECIFICATION: DiD with Year and State Fixed Effects\n")
    f.write("-"*70 + "\n")
    f.write(str(results3.summary()))
    f.write("\n\n")

    f.write("SAMPLE INFORMATION\n")
    f.write("-"*70 + "\n")
    f.write(f"Total observations: {len(df_did):,}\n")
    f.write(f"Treatment group (DACA eligible): {df_did['treatment'].sum():,}\n")
    f.write(f"Control group (arrived at 16+): {(df_did['treatment']==0).sum():,}\n")
    f.write(f"Pre-DACA (2006-2011): {(df_did['post']==0).sum():,}\n")
    f.write(f"Post-DACA (2013-2016): {(df_did['post']==1).sum():,}\n")

print("Saved: model_summary.txt")

# =============================================================================
# 13. FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)

print(f"""
PREFERRED ESTIMATE (DiD with Year and State Fixed Effects):

Effect of DACA Eligibility on Full-time Employment:

  Point Estimate:  {treat_post_coef:.4f} ({treat_post_coef*100:.2f} percentage points)
  Standard Error:  {treat_post_se:.4f}
  95% CI:          [{ci_low:.4f}, {ci_high:.4f}]
  p-value:         {treat_post_pval:.4f}

Sample Size:       {int(results3.nobs):,}
Treatment Group:   {df_did['treatment'].sum():,}
Control Group:     {(df_did['treatment']==0).sum():,}

INTERPRETATION:
DACA eligibility is associated with a {abs(treat_post_coef*100):.2f} percentage point
{'increase' if treat_post_coef > 0 else 'decrease'} in the probability of full-time
employment (35+ hours per week) among Hispanic-Mexican, Mexican-born non-citizens,
comparing those who meet DACA eligibility criteria (arrived before age 16, born 1982
or later, in US since 2007) to those who arrived at age 16 or older.

This effect is {'statistically significant' if treat_post_pval < 0.05 else 'not statistically significant'} at the 5% level.
""")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nOutput files created:")
print("  - summary_statistics.csv")
print("  - trends_data.csv")
print("  - event_study_coefs.csv")
print("  - regression_results.csv")
print("  - model_summary.txt")
print("  - figure1_trends.png/pdf")
print("  - figure2_event_study.png/pdf")
print("  - figure3_difference.png/pdf")
