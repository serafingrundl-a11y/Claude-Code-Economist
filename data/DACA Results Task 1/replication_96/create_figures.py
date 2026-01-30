"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# =============================================================================
# 1. EVENT STUDY PLOT
# =============================================================================

print("Creating event study plot...")

event_study = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
years = event_study['Year']
coefs = event_study['Coef']
ci_low = event_study['CI_low']
ci_high = event_study['CI_high']

# Plot confidence intervals
ax.fill_between(years, ci_low, ci_high, alpha=0.3, color='steelblue')
ax.plot(years, coefs, 'o-', color='steelblue', linewidth=2, markersize=8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation (June 2012)')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xticks(years)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
plt.close()

print("Saved figure1_event_study.png and .pdf")

# =============================================================================
# 2. PARALLEL TRENDS PLOT
# =============================================================================

print("Creating parallel trends plot...")

# Load data
df = pd.read_csv('data/filtered_sample.csv')
df = df[df['CITIZEN'] == 3]  # Non-citizens only
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)]  # Working age

# Create eligibility indicator
def calc_age_at_daca(row):
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']
    if birth_qtr in [1, 2]:
        return 2012 - birth_year
    else:
        return 2012 - birth_year - 1

df['AGE_AT_DACA'] = df.apply(calc_age_at_daca, axis=1)
df['ARRIVAL_AGE'] = df['YRIMMIG'] - df['BIRTHYR']

def is_daca_eligible(row):
    age_at_daca = row['AGE_AT_DACA']
    arrival_age = row['ARRIVAL_AGE']
    yrimmig = row['YRIMMIG']
    age_under_31 = age_at_daca < 31
    arrived_before_16 = arrival_age < 16
    in_us_since_2007 = yrimmig <= 2007
    return int(age_under_31 and arrived_before_16 and in_us_since_2007)

df['DACA_ELIGIBLE'] = df.apply(is_daca_eligible, axis=1)
df['FULLTIME_EMPLOYED'] = ((df['EMPSTAT'] == 1) & (df['UHRSWORK'] >= 35)).astype(int)

# Calculate yearly employment rates by eligibility
yearly_rates = df.groupby(['YEAR', 'DACA_ELIGIBLE'])['FULLTIME_EMPLOYED'].mean().unstack()
yearly_rates.columns = ['Not Eligible', 'DACA Eligible']

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(yearly_rates.index, yearly_rates['Not Eligible'], 'o-', color='gray',
        linewidth=2, markersize=8, label='Not DACA Eligible')
ax.plot(yearly_rates.index, yearly_rates['DACA Eligible'], 's-', color='steelblue',
        linewidth=2, markersize=8, label='DACA Eligible')
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status')
ax.set_xticks(range(2006, 2017))
ax.legend(loc='lower right')
ax.set_ylim(0.35, 0.60)

plt.tight_layout()
plt.savefig('figure2_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_parallel_trends.pdf', bbox_inches='tight')
plt.close()

print("Saved figure2_parallel_trends.png and .pdf")

# =============================================================================
# 3. COEFFICIENT COMPARISON PLOT
# =============================================================================

print("Creating coefficient comparison plot...")

results = pd.read_csv('results_summary.csv')

fig, ax = plt.subplots(figsize=(10, 6))

models = results['Model']
coefs = results['Coefficient']
ses = results['SE']

# Calculate confidence intervals
ci_low = coefs - 1.96 * ses
ci_high = coefs + 1.96 * ses

y_pos = np.arange(len(models))

ax.barh(y_pos, coefs, xerr=1.96*ses, color='steelblue', alpha=0.7,
        capsize=5, error_kw={'linewidth': 1.5})
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment Probability)')
ax.set_title('Effect of DACA Eligibility on Full-Time Employment: Model Comparison')

# Add coefficient labels
for i, (coef, se) in enumerate(zip(coefs, ses)):
    ax.text(coef + 0.01, i, f'{coef:.3f} ({se:.3f})', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure3_coefficients.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_coefficients.pdf', bbox_inches='tight')
plt.close()

print("Saved figure3_coefficients.png and .pdf")

# =============================================================================
# 4. SAMPLE COMPOSITION PLOT
# =============================================================================

print("Creating sample composition plot...")

# Year distribution
year_counts = df.groupby('YEAR').size()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Sample size by year
axes[0].bar(year_counts.index, year_counts.values, color='steelblue', alpha=0.7)
axes[0].axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Sample Size')
axes[0].set_title('(A) Sample Size by Year')
axes[0].set_xticks(range(2006, 2017))
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()

# Panel B: Age distribution by eligibility
eligible = df[df['DACA_ELIGIBLE'] == 1]['AGE']
not_eligible = df[df['DACA_ELIGIBLE'] == 0]['AGE']

axes[1].hist(not_eligible, bins=range(18, 66), alpha=0.5, color='gray',
             label='Not Eligible', density=True)
axes[1].hist(eligible, bins=range(18, 66), alpha=0.5, color='steelblue',
             label='DACA Eligible', density=True)
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Density')
axes[1].set_title('(B) Age Distribution by Eligibility Status')
axes[1].legend()

plt.tight_layout()
plt.savefig('figure4_sample_composition.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_sample_composition.pdf', bbox_inches='tight')
plt.close()

print("Saved figure4_sample_composition.png and .pdf")

# =============================================================================
# 5. ROBUSTNESS CHECKS FOREST PLOT
# =============================================================================

print("Creating robustness forest plot...")

# Robustness check results
robustness = {
    'Specification': [
        'Main (Year FE)',
        'Year + State FE',
        'Ages 25-55',
        'Men Only',
        'Women Only',
        'Include 2012',
        'Weighted'
    ],
    'Coefficient': [0.0160, 0.0153, 0.0097, 0.0115, 0.0185, 0.0131, 0.0247],
    'SE': [0.0038, 0.0038, 0.0063, 0.0051, 0.0055, 0.0036, 0.0046]
}
rob_df = pd.DataFrame(robustness)

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(rob_df))
coefs = rob_df['Coefficient']
ses = rob_df['SE']

# Plot with error bars
ax.errorbar(coefs, y_pos, xerr=1.96*ses, fmt='o', color='steelblue',
            capsize=5, markersize=8, linewidth=2, capthick=1.5)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=coefs[0], color='steelblue', linestyle='--', linewidth=1, alpha=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(rob_df['Specification'])
ax.set_xlabel('DiD Coefficient (95% CI)')
ax.set_title('Robustness Checks: Effect of DACA Eligibility on Full-Time Employment')

# Highlight main estimate
ax.scatter([coefs[0]], [0], color='red', s=100, zorder=5, marker='D')

plt.tight_layout()
plt.savefig('figure5_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_robustness.pdf', bbox_inches='tight')
plt.close()

print("Saved figure5_robustness.png and .pdf")

print("\nAll figures created successfully!")
