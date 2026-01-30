"""
Generate figures and additional analyses for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Load data
print("Loading data...")
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Create derived variables
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] <= 2).astype(int)
df['YEAR_factor'] = df['YEAR'].astype(str)
df['STATE_factor'] = df['STATEFIP'].astype(str)

# Helper function for weighted mean
def weighted_mean(group, var='FT'):
    return np.average(group[var], weights=group['PERWT'])

# =============================================================================
# FIGURE 1: Parallel Trends - Year-by-Year Employment Rates
# =============================================================================
print("Generating Figure 1: Parallel trends...")

yearly_rates = df.groupby(['YEAR', 'ELIGIBLE']).apply(lambda x: weighted_mean(x, 'FT')).unstack()
yearly_rates.columns = ['Control (31-35)', 'Treatment (26-30)']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(yearly_rates.index, yearly_rates['Control (31-35)'], 'o-',
        label='Control (ages 31-35 in 2012)', color='blue', linewidth=2, markersize=8)
ax.plot(yearly_rates.index, yearly_rates['Treatment (26-30)'], 's-',
        label='Treatment (ages 26-30 in 2012)', color='red', linewidth=2, markersize=8)

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation (June 2012)')

# Shade pre and post periods
ax.axvspan(2008, 2011.5, alpha=0.1, color='blue', label='Pre-period')
ax.axvspan(2012.5, 2016, alpha=0.1, color='green', label='Post-period')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate (Weighted)', fontsize=12)
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim(0.5, 0.85)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure1_parallel_trends.png/pdf")

# =============================================================================
# FIGURE 2: Difference-in-Differences Visualization
# =============================================================================
print("Generating Figure 2: DiD visualization...")

# Calculate means by group and period
did_data = df.groupby(['ELIGIBLE', 'AFTER']).apply(lambda x: weighted_mean(x, 'FT')).unstack()
did_data.index = ['Control', 'Treatment']
did_data.columns = ['Pre', 'Post']

fig, ax = plt.subplots(figsize=(8, 6))

# Plot lines
x_pos = [0, 1]
ax.plot(x_pos, [did_data.loc['Control', 'Pre'], did_data.loc['Control', 'Post']],
        'o-', label='Control (31-35)', color='blue', linewidth=2, markersize=10)
ax.plot(x_pos, [did_data.loc['Treatment', 'Pre'], did_data.loc['Treatment', 'Post']],
        's-', label='Treatment (26-30)', color='red', linewidth=2, markersize=10)

# Plot counterfactual
counterfactual = did_data.loc['Treatment', 'Pre'] + (did_data.loc['Control', 'Post'] - did_data.loc['Control', 'Pre'])
ax.plot([0, 1], [did_data.loc['Treatment', 'Pre'], counterfactual],
        '--', label='Treatment Counterfactual', color='red', linewidth=1.5, alpha=0.6)

# Annotate DiD
did_effect = did_data.loc['Treatment', 'Post'] - counterfactual
ax.annotate('', xy=(1.05, did_data.loc['Treatment', 'Post']),
            xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text(1.1, (did_data.loc['Treatment', 'Post'] + counterfactual)/2,
        f'DiD = {did_effect:.3f}', fontsize=11, va='center')

ax.set_xticks(x_pos)
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.55, 0.75)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_did_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_did_visual.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure2_did_visual.png/pdf")

# =============================================================================
# FIGURE 3: Event Study
# =============================================================================
print("Generating Figure 3: Event study...")

# Create year-specific interactions with treatment
# Use 2011 as reference year
df_event = df.copy()
years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]

for year in years:
    df_event[f'YEAR_{year}'] = (df_event['YEAR'] == year).astype(int)
    df_event[f'ELIGIBLE_YEAR_{year}'] = df_event['ELIGIBLE'] * df_event[f'YEAR_{year}']

# Run event study regression (excluding 2011 as reference)
event_vars = [f'ELIGIBLE_YEAR_{y}' for y in years if y != 2011]
formula_event = 'FT ~ ELIGIBLE + C(YEAR_factor) + C(STATE_factor) + ' + ' + '.join(event_vars)
model_event = smf.wls(formula_event, data=df_event, weights=df_event['PERWT']).fit()

# Extract coefficients
event_coefs = []
event_ses = []
for year in years:
    if year == 2011:
        event_coefs.append(0)  # Reference year
        event_ses.append(0)
    else:
        var = f'ELIGIBLE_YEAR_{year}'
        event_coefs.append(model_event.params[var])
        event_ses.append(model_event.bse[var])

# Plot event study
fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(years, event_coefs, yerr=[1.96*se for se in event_ses],
            fmt='o', capsize=5, capthick=2, color='blue', markersize=8, linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')

# Shade pre and post
ax.axvspan(2008, 2011.5, alpha=0.1, color='blue')
ax.axvspan(2012.5, 2016, alpha=0.1, color='green')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Treatment × Year)', fontsize=12)
ax.set_title('Event Study: Treatment Effect by Year\n(Reference: 2011)', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks(years)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure3_event_study.png/pdf")

# =============================================================================
# FIGURE 4: Heterogeneity by Sex
# =============================================================================
print("Generating Figure 4: Heterogeneity by sex...")

# Run models by sex
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (sex, label) in enumerate([(1, 'Male'), (2, 'Female')]):
    df_sex = df[df['SEX'] == sex].copy()

    yearly = df_sex.groupby(['YEAR', 'ELIGIBLE']).apply(lambda x: weighted_mean(x, 'FT')).unstack()
    yearly.columns = ['Control', 'Treatment']

    axes[idx].plot(yearly.index, yearly['Control'], 'o-', label='Control (31-35)', color='blue', linewidth=2, markersize=6)
    axes[idx].plot(yearly.index, yearly['Treatment'], 's-', label='Treatment (26-30)', color='red', linewidth=2, markersize=6)
    axes[idx].axvline(x=2012, color='gray', linestyle='--', linewidth=2)

    # Calculate DiD for this subgroup
    model_sex = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATE_factor)',
                        data=df_sex, weights=df_sex['PERWT']).fit()
    did_sex = model_sex.params['ELIGIBLE_AFTER']
    se_sex = model_sex.bse['ELIGIBLE_AFTER']

    axes[idx].set_xlabel('Year', fontsize=11)
    axes[idx].set_ylabel('Full-Time Employment Rate', fontsize=11)
    axes[idx].set_title(f'{label}\nDiD = {did_sex:.3f} (SE = {se_sex:.3f})', fontsize=12)
    axes[idx].legend(loc='lower right', fontsize=9)
    axes[idx].set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
    axes[idx].set_ylim(0.3, 0.95)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure4_by_sex.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_by_sex.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure4_by_sex.png/pdf")

# =============================================================================
# FIGURE 5: Heterogeneity by Education
# =============================================================================
print("Generating Figure 5: Heterogeneity by education...")

# Create HS degree indicator - handle NaN values
df['HS_DEGREE_int'] = df['HS_DEGREE'].fillna(0).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (hs, label) in enumerate([(0, 'No HS Degree'), (1, 'HS Degree or Higher')]):
    df_edu = df[df['HS_DEGREE_int'] == hs].copy()

    if len(df_edu) < 100:
        continue

    yearly = df_edu.groupby(['YEAR', 'ELIGIBLE']).apply(lambda x: weighted_mean(x, 'FT')).unstack()
    yearly.columns = ['Control', 'Treatment']

    axes[idx].plot(yearly.index, yearly['Control'], 'o-', label='Control (31-35)', color='blue', linewidth=2, markersize=6)
    axes[idx].plot(yearly.index, yearly['Treatment'], 's-', label='Treatment (26-30)', color='red', linewidth=2, markersize=6)
    axes[idx].axvline(x=2012, color='gray', linestyle='--', linewidth=2)

    # Calculate DiD for this subgroup
    model_edu = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATE_factor)',
                        data=df_edu, weights=df_edu['PERWT']).fit()
    did_edu = model_edu.params['ELIGIBLE_AFTER']
    se_edu = model_edu.bse['ELIGIBLE_AFTER']

    axes[idx].set_xlabel('Year', fontsize=11)
    axes[idx].set_ylabel('Full-Time Employment Rate', fontsize=11)
    axes[idx].set_title(f'{label}\nDiD = {did_edu:.3f} (SE = {se_edu:.3f})', fontsize=12)
    axes[idx].legend(loc='lower right', fontsize=9)
    axes[idx].set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
    axes[idx].set_ylim(0.4, 0.85)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure5_by_education.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_by_education.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure5_by_education.png/pdf")

# =============================================================================
# Generate Summary Statistics Table
# =============================================================================
print("\nGenerating summary statistics...")

# Define variables for summary
summary_vars = ['FT', 'FEMALE', 'MARRIED', 'AGE', 'NCHILD', 'HS_DEGREE_int',
                'DRIVERSLICENSES', 'INSTATETUITION', 'EVERIFY']

# Summary by treatment group
summary_rows = []
for var in summary_vars:
    if var in df.columns:
        row = {'Variable': var}
        for group in [0, 1]:
            for period in [0, 1]:
                subset = df[(df['ELIGIBLE'] == group) & (df['AFTER'] == period)]
                mean_val = np.average(subset[var], weights=subset['PERWT'])
                row[f'Elig{group}_After{period}'] = mean_val
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('summary_statistics.csv', index=False)
print("  Saved: summary_statistics.csv")

# =============================================================================
# Placebo Test - Using Pre-Period Only
# =============================================================================
print("\nRunning placebo test...")

# Use 2008-2009 as "pre" and 2010-2011 as "post" (before actual treatment)
df_placebo = df[df['AFTER'] == 0].copy()
df_placebo['PLACEBO_AFTER'] = (df_placebo['YEAR'] >= 2010).astype(int)
df_placebo['PLACEBO_INTERACTION'] = df_placebo['ELIGIBLE'] * df_placebo['PLACEBO_AFTER']

model_placebo = smf.wls('FT ~ ELIGIBLE + PLACEBO_AFTER + PLACEBO_INTERACTION + C(STATE_factor)',
                        data=df_placebo, weights=df_placebo['PERWT']).fit()

print(f"  Placebo DiD (pre-treatment period): {model_placebo.params['PLACEBO_INTERACTION']:.4f}")
print(f"  Placebo SE: {model_placebo.bse['PLACEBO_INTERACTION']:.4f}")
print(f"  Placebo p-value: {model_placebo.pvalues['PLACEBO_INTERACTION']:.4f}")

# Save placebo results
with open('placebo_results.txt', 'w') as f:
    f.write("Placebo Test Results\n")
    f.write("="*50 + "\n")
    f.write(f"Coefficient: {model_placebo.params['PLACEBO_INTERACTION']:.4f}\n")
    f.write(f"Standard Error: {model_placebo.bse['PLACEBO_INTERACTION']:.4f}\n")
    f.write(f"t-statistic: {model_placebo.tvalues['PLACEBO_INTERACTION']:.4f}\n")
    f.write(f"p-value: {model_placebo.pvalues['PLACEBO_INTERACTION']:.4f}\n")
    ci = model_placebo.conf_int().loc['PLACEBO_INTERACTION']
    f.write(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")

print("  Saved: placebo_results.txt")

# =============================================================================
# Additional subgroup analyses
# =============================================================================
print("\nRunning subgroup analyses...")

subgroup_results = []

# By sex
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex]
    model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATE_factor) + C(YEAR_factor)',
                    data=df_sub, weights=df_sub['PERWT']).fit()
    subgroup_results.append({
        'Subgroup': label,
        'N': len(df_sub),
        'Estimate': model.params['ELIGIBLE_AFTER'],
        'SE': model.bse['ELIGIBLE_AFTER'],
        'pvalue': model.pvalues['ELIGIBLE_AFTER']
    })

# By education
for hs, label in [(0, 'No HS Degree'), (1, 'HS Degree+')]:
    df_sub = df[df['HS_DEGREE_int'] == hs]
    if len(df_sub) > 100:
        model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATE_factor) + C(YEAR_factor)',
                        data=df_sub, weights=df_sub['PERWT']).fit()
        subgroup_results.append({
            'Subgroup': label,
            'N': len(df_sub),
            'Estimate': model.params['ELIGIBLE_AFTER'],
            'SE': model.bse['ELIGIBLE_AFTER'],
            'pvalue': model.pvalues['ELIGIBLE_AFTER']
        })

# By marital status
for mar, label in [(1, 'Married'), (0, 'Not Married')]:
    df_sub = df[df['MARRIED'] == mar]
    model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATE_factor) + C(YEAR_factor)',
                    data=df_sub, weights=df_sub['PERWT']).fit()
    subgroup_results.append({
        'Subgroup': label,
        'N': len(df_sub),
        'Estimate': model.params['ELIGIBLE_AFTER'],
        'SE': model.bse['ELIGIBLE_AFTER'],
        'pvalue': model.pvalues['ELIGIBLE_AFTER']
    })

subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv('subgroup_results.csv', index=False)
print("  Saved: subgroup_results.csv")

print(subgroup_df.to_string())

print("\nAll figures and additional analyses complete!")
