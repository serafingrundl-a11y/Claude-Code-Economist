"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# Load data
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_50\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)
output_dir = r"C:\Users\seraf\DACA Results Task 3\replication_50"

# Calculate yearly means
yearly_data = df.groupby(['YEAR', 'ELIGIBLE']).agg({
    'FT': ['mean', 'std', 'count']
}).reset_index()
yearly_data.columns = ['Year', 'Eligible', 'FT_Mean', 'FT_Std', 'N']
yearly_data['SE'] = yearly_data['FT_Std'] / np.sqrt(yearly_data['N'])

# Figure 1: Parallel Trends Plot
fig, ax = plt.subplots(figsize=(12, 7))

# Separate by eligibility
control = yearly_data[yearly_data['Eligible'] == 0].copy()
treated = yearly_data[yearly_data['Eligible'] == 1].copy()

# Plot with confidence intervals
ax.errorbar(control['Year'], control['FT_Mean'], yerr=1.96*control['SE'],
            fmt='o-', color='#2171b5', linewidth=2, markersize=8, capsize=5,
            label='Control (ages 31-35)')
ax.errorbar(treated['Year'], treated['FT_Mean'], yerr=1.96*treated['SE'],
            fmt='s-', color='#cb181d', linewidth=2, markersize=8, capsize=5,
            label='Treated (ages 26-30)')

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.74, 'DACA\nImplemented', fontsize=11, color='gray', va='top')

# Shade pre and post periods
ax.axvspan(2007.5, 2012, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2012, 2016.5, alpha=0.1, color='red', label='Post-DACA')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status')
ax.legend(loc='lower left')
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_xlim(2007.5, 2016.5)
ax.set_ylim(0.55, 0.78)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure1_parallel_trends.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1: Parallel trends saved")

# Figure 2: Event Study Plot
fig, ax = plt.subplots(figsize=(12, 7))

# Event study coefficients (from analysis output)
# Relative to 2011 (the year before DACA)
event_years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs = [-0.0591, -0.0388, -0.0663, 0, 0.0188, -0.0088, 0.0303, 0.0491]
event_ses = [0.0289, 0.0297, 0.0294, 0, 0.0306, 0.0308, 0.0316, 0.0314]

# Calculate CIs
ci_low = [c - 1.96*s for c, s in zip(event_coefs, event_ses)]
ci_high = [c + 1.96*s for c, s in zip(event_coefs, event_ses)]

# Plot
ax.errorbar(event_years, event_coefs, yerr=[np.array(event_coefs) - np.array(ci_low),
                                             np.array(ci_high) - np.array(event_coefs)],
            fmt='o', color='#2171b5', markersize=10, capsize=6, capthick=2, linewidth=2)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at DACA implementation (between 2011 and 2013)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.10, 'DACA\nImplemented', fontsize=11, color='gray', va='top')

# Shade pre and post periods
ax.axvspan(2007.5, 2012, alpha=0.1, color='blue')
ax.axvspan(2012, 2016.5, alpha=0.1, color='red')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_xlim(2007.5, 2016.5)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure2_event_study.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2: Event study saved")

# Figure 3: DiD Visualization (2x2)
fig, ax = plt.subplots(figsize=(10, 7))

# Calculate group means
treated_before = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]['FT'].mean()
treated_after = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]['FT'].mean()
control_before = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]['FT'].mean()
control_after = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]['FT'].mean()

# Plot
periods = ['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)']
x = np.array([0, 1])
width = 0.35

bars1 = ax.bar(x - width/2, [control_before, control_after], width, label='Control (31-35)',
               color='#2171b5', alpha=0.8)
bars2 = ax.bar(x + width/2, [treated_before, treated_after], width, label='Treated (26-30)',
               color='#cb181d', alpha=0.8)

# Add value labels on bars
for bar, val in zip(bars1, [control_before, control_after]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=11)
for bar, val in zip(bars2, [treated_before, treated_after]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: DACA Effect on Full-Time Employment')
ax.set_xticks(x)
ax.set_xticklabels(periods)
ax.legend()
ax.set_ylim(0, 0.8)

# Add DiD annotation
did_estimate = (treated_after - treated_before) - (control_after - control_before)
ax.text(0.5, 0.15, f'DiD Estimate = {did_estimate:.4f}\n(p < 0.001)',
        transform=ax.transAxes, fontsize=14, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure3_did_bars.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3: DiD bar chart saved")

# Figure 4: Sample Size by Year
fig, ax = plt.subplots(figsize=(10, 6))

sample_by_year = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
sample_by_year.columns = ['Control (31-35)', 'Treated (26-30)']

sample_by_year.plot(kind='bar', ax=ax, color=['#2171b5', '#cb181d'], alpha=0.8)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Observations')
ax.set_title('Sample Size by Year and Treatment Status')
ax.legend(title='Group')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure4_sample_size.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure4_sample_size.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4: Sample size saved")

# Figure 5: Robustness Results
fig, ax = plt.subplots(figsize=(12, 8))

# Model estimates and SEs
models = ['Basic DiD', 'Robust SE', 'Demo Controls', 'Year FE', 'State+Year FE', 'Weighted']
estimates = [0.0643, 0.0643, 0.0581, 0.0566, 0.0568, 0.0674]
ses = [0.0153, 0.0153, 0.0142, 0.0142, 0.0142, 0.0168]

# Calculate CIs
ci_low = [e - 1.96*s for e, s in zip(estimates, ses)]
ci_high = [e + 1.96*s for e, s in zip(estimates, ses)]

y_pos = np.arange(len(models))

# Plot
ax.errorbar(estimates, y_pos, xerr=[np.array(estimates) - np.array(ci_low),
                                     np.array(ci_high) - np.array(estimates)],
            fmt='o', color='#2171b5', markersize=12, capsize=6, capthick=2, linewidth=2)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=0.0566, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Preferred estimate')

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('Difference-in-Differences Estimate')
ax.set_title('Robustness of DACA Effect Estimates Across Specifications')
ax.legend(loc='upper right')

# Add value labels
for i, (est, se) in enumerate(zip(estimates, ses)):
    ax.text(est + 0.02, i, f'{est:.4f}\n(SE: {se:.4f})', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure5_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure5_robustness.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5: Robustness saved")

# Figure 6: By Gender
fig, ax = plt.subplots(figsize=(8, 6))

genders = ['Male', 'Female', 'Overall']
estimates_gender = [0.0615, 0.0452, 0.0566]
ses_gender = [0.0170, 0.0232, 0.0142]

ci_low_g = [e - 1.96*s for e, s in zip(estimates_gender, ses_gender)]
ci_high_g = [e + 1.96*s for e, s in zip(estimates_gender, ses_gender)]

y_pos = np.arange(len(genders))

ax.errorbar(estimates_gender, y_pos, xerr=[np.array(estimates_gender) - np.array(ci_low_g),
                                            np.array(ci_high_g) - np.array(estimates_gender)],
            fmt='o', color='#2171b5', markersize=12, capsize=6, capthick=2, linewidth=2)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(genders)
ax.set_xlabel('Difference-in-Differences Estimate')
ax.set_title('DACA Effect on Full-Time Employment by Gender')
ax.set_xlim(-0.02, 0.12)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure6_by_gender.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure6_by_gender.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 6: By gender saved")

print("\nAll figures created successfully!")
