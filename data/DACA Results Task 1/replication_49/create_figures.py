"""
DACA Replication Study: Generate Figures for Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

print("Creating figures for DACA replication report...")

# Load data
data_path = "data/data.csv"
chunks = []
chunksize = 1000000
for chunk in pd.read_csv(data_path, chunksize=chunksize, low_memory=False):
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

# Apply filters
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)]
df = df[df['CITIZEN'] == 3]
df = df[df['YEAR'] != 2012]

# Create variables
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df['daca_eligible'] = (
    (df['age_at_immigration'] < 16) &
    (df['age_at_immigration'] >= 0) &
    (df['BIRTHYR'] >= 1982) &
    (df['YRIMMIG'] <= 2007) &
    (df['YRIMMIG'] > 0)
).astype(int)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Figure 1: Trends in Full-Time Employment by Eligibility
print("Creating Figure 1: Employment trends...")
trends = df.groupby(['YEAR', 'daca_eligible']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'PERWT': 'sum'
}).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
eligible = trends[trends['daca_eligible'] == 1]
control = trends[trends['daca_eligible'] == 0]

ax.plot(eligible['YEAR'], eligible['fulltime'], 'o-', color='blue',
        linewidth=2, markersize=8, label='DACA Eligible')
ax.plot(control['YEAR'], control['fulltime'], 's--', color='red',
        linewidth=2, markersize=8, label='Control Group')

ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(0.35, 0.70)
ax.grid(True, alpha=0.3)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure1_trends.png")

# Figure 2: Event Study Coefficients
print("Creating Figure 2: Event study...")
# Event study results from main analysis (manually entered from output)
event_years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs = [-0.0248, -0.0198, -0.0052, 0.0027, 0.0062, 0.0, 0.0109, 0.0254, 0.0450, 0.0446]
event_ses = [0.0083, 0.0061, 0.0094, 0.0076, 0.0106, 0.0, 0.0092, 0.0139, 0.0094, 0.0096]

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(event_years, event_coefs, yerr=[1.96*se for se in event_ses],
            fmt='o', color='blue', capsize=4, capthick=2, linewidth=2, markersize=8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(event_years)

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure2_eventstudy.png")

# Figure 3: Age Distribution by Eligibility
print("Creating Figure 3: Age distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

eligible_ages = df[df['daca_eligible'] == 1]['AGE']
control_ages = df[df['daca_eligible'] == 0]['AGE']

ax1.hist(eligible_ages, bins=range(16, 66, 2), color='blue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Age', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Age Distribution: DACA Eligible', fontsize=14)

ax2.hist(control_ages, bins=range(16, 66, 2), color='red', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Age', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Age Distribution: Control Group', fontsize=14)

plt.tight_layout()
plt.savefig('figure3_agedist.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure3_agedist.png")

# Figure 4: Employment rates by gender and eligibility
print("Creating Figure 4: Employment by gender...")
df['female'] = (df['SEX'] == 2).astype(int)
df['post'] = (df['YEAR'] >= 2013).astype(int)

gender_trends = df.groupby(['daca_eligible', 'post', 'female']).agg({
    'fulltime': 'mean'
}).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

width = 0.2
x = np.array([0, 1, 3, 4])
labels = ['Male Pre', 'Male Post', 'Female Pre', 'Female Post']

eligible_vals = []
control_vals = []
for post in [0, 1]:
    for female in [0, 1]:
        elig_val = gender_trends[(gender_trends['daca_eligible']==1) &
                                  (gender_trends['post']==post) &
                                  (gender_trends['female']==female)]['fulltime'].values[0]
        ctrl_val = gender_trends[(gender_trends['daca_eligible']==0) &
                                  (gender_trends['post']==post) &
                                  (gender_trends['female']==female)]['fulltime'].values[0]
        eligible_vals.append(elig_val)
        control_vals.append(ctrl_val)

# Reorder to Male Pre, Male Post, Female Pre, Female Post
eligible_ordered = [eligible_vals[0], eligible_vals[2], eligible_vals[1], eligible_vals[3]]
control_ordered = [control_vals[0], control_vals[2], control_vals[1], control_vals[3]]

ax.bar(x - width/2, eligible_ordered, width, label='DACA Eligible', color='blue', alpha=0.8)
ax.bar(x + width/2, control_ordered, width, label='Control Group', color='red', alpha=0.8)

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment by Gender and DACA Status', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 0.85)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure4_gender.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure4_gender.png")

# Figure 5: Simple DiD visualization
print("Creating Figure 5: DiD visualization...")
fig, ax = plt.subplots(figsize=(10, 6))

# Data points
ft_elig_pre = 0.4263
ft_elig_post = 0.4948
ft_ctrl_pre = 0.6034
ft_ctrl_post = 0.5787

# Plot eligible group
ax.plot([0, 1], [ft_elig_pre, ft_elig_post], 'o-', color='blue',
        linewidth=2, markersize=10, label='DACA Eligible')

# Plot control group
ax.plot([0, 1], [ft_ctrl_pre, ft_ctrl_post], 's-', color='red',
        linewidth=2, markersize=10, label='Control Group')

# Counterfactual
counterfactual = ft_elig_pre + (ft_ctrl_post - ft_ctrl_pre)
ax.plot([0, 1], [ft_elig_pre, counterfactual], ':', color='blue',
        linewidth=2, label='Eligible (Counterfactual)')

# DiD effect arrow
ax.annotate('', xy=(1.05, ft_elig_post), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (ft_elig_post + counterfactual)/2, f'DiD = {ft_elig_post - counterfactual:.3f}',
        fontsize=11, color='green', va='center')

ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)'])
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0.35, 0.70)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure5_did.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure5_did.png")

print("\nAll figures created successfully!")
