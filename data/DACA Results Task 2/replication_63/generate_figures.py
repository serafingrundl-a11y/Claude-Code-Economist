"""
Generate figures and additional tables for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data (same as main analysis)
print("Loading data...")
df = pd.read_csv('data/data.csv')

# Apply filters for DACA eligibility
df_mex = df[df['HISPAN'] == 1].copy()
df_mex = df_mex[df_mex['BPL'] == 200].copy()
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
df_mex['age_at_daca'] = 2012 - df_mex['BIRTHYR']
df_analysis = df_mex[(df_mex['age_at_daca'] >= 26) & (df_mex['age_at_daca'] <= 35)].copy()
df_analysis['age_at_arrival'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']
df_analysis = df_analysis[df_analysis['age_at_arrival'] < 16].copy()
df_analysis = df_analysis[df_analysis['YRIMMIG'] <= 2007].copy()
df_analysis['treated'] = (df_analysis['age_at_daca'] <= 30).astype(int)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

# Add covariates
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = df_analysis['MARST'].isin([1, 2]).astype(int)
df_analysis['educ_less_hs'] = (df_analysis['EDUC'] < 6).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] == 6).astype(int)
df_analysis['educ_some_college'] = df_analysis['EDUC'].isin([7, 8, 9]).astype(int)
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)
df_analysis['age_survey'] = df_analysis['AGE']
df_analysis['age_sq'] = df_analysis['age_survey'] ** 2

print(f"Sample size: {len(df_analysis)}")

# Figure 1: Full-time employment trends over time
print("\nGenerating Figure 1: Employment trends...")
fig1, ax1 = plt.subplots(figsize=(10, 6))

years = sorted(df_analysis['YEAR'].unique())

# Calculate weighted means for each year and group
treat_means = []
ctrl_means = []
for year in years:
    year_data = df_analysis[df_analysis['YEAR'] == year]
    treat_data = year_data[year_data['treated']==1]
    ctrl_data = year_data[year_data['treated']==0]
    if len(treat_data) > 0:
        treat_means.append(np.average(treat_data['fulltime'], weights=treat_data['PERWT']))
    else:
        treat_means.append(np.nan)
    if len(ctrl_data) > 0:
        ctrl_means.append(np.average(ctrl_data['fulltime'], weights=ctrl_data['PERWT']))
    else:
        ctrl_means.append(np.nan)

ax1.plot(years, treat_means, 'b-o', label='Treatment (Ages 26-30 at DACA)', linewidth=2, markersize=8)
ax1.plot(years, ctrl_means, 'r-s', label='Control (Ages 31-35 at DACA)', linewidth=2, markersize=8)
ax1.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation (2012)')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Rates by Treatment Status', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.set_ylim([0.50, 0.75])
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()
print("Saved figure1_trends.png/pdf")

# Figure 2: Event study plot
print("\nGenerating Figure 2: Event study...")

# Create year interactions
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'treated_{year}'] = df_analysis['treated'] * (df_analysis['YEAR'] == year).astype(int)

model_event = smf.wls('''fulltime ~ treated + C(YEAR) +
                         treated_2006 + treated_2007 + treated_2008 + treated_2009 + treated_2010 +
                         treated_2013 + treated_2014 + treated_2015 + treated_2016 +
                         female + married + age_survey + age_sq''',
                      data=df_analysis,
                      weights=df_analysis['PERWT']).fit(cov_type='HC1')

event_years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs = []
event_cis_low = []
event_cis_high = []

for year in event_years:
    if year == 2011:
        event_coefs.append(0)
        event_cis_low.append(0)
        event_cis_high.append(0)
    else:
        var = f'treated_{year}'
        event_coefs.append(model_event.params[var])
        ci = model_event.conf_int().loc[var]
        event_cis_low.append(ci[0])
        event_cis_high.append(ci[1])

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.errorbar(event_years, event_coefs,
             yerr=[np.array(event_coefs) - np.array(event_cis_low),
                   np.array(event_cis_high) - np.array(event_coefs)],
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8, color='navy')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=2011.5, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Difference in Full-Time Employment Rate\n(Relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Treatment Effect Over Time', fontsize=14)
ax2.set_xticks(event_years)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()
print("Saved figure2_eventstudy.png/pdf")

# Figure 3: DiD visualization
print("\nGenerating Figure 3: DiD illustration...")

fig3, ax3 = plt.subplots(figsize=(10, 6))

# Calculate means
pre_treat = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==0)]
post_treat = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==1)]
pre_ctrl = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==0)]
post_ctrl = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==1)]

pre_treat_mean = np.average(pre_treat['fulltime'], weights=pre_treat['PERWT'])
post_treat_mean = np.average(post_treat['fulltime'], weights=post_treat['PERWT'])
pre_ctrl_mean = np.average(pre_ctrl['fulltime'], weights=pre_ctrl['PERWT'])
post_ctrl_mean = np.average(post_ctrl['fulltime'], weights=post_ctrl['PERWT'])

# Plot actual trends
ax3.plot([0, 1], [pre_treat_mean, post_treat_mean], 'b-o', label='Treatment Group (Actual)',
         linewidth=2, markersize=10)
ax3.plot([0, 1], [pre_ctrl_mean, post_ctrl_mean], 'r-s', label='Control Group',
         linewidth=2, markersize=10)

# Plot counterfactual (treatment group without treatment)
counterfactual = pre_treat_mean + (post_ctrl_mean - pre_ctrl_mean)
ax3.plot([0, 1], [pre_treat_mean, counterfactual], 'b--',
         label='Treatment Group (Counterfactual)', linewidth=2, alpha=0.7)

# Draw arrow showing DiD effect
ax3.annotate('', xy=(1, post_treat_mean), xytext=(1, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.05, (post_treat_mean + counterfactual)/2,
         f'DiD Effect\n= {post_treat_mean - counterfactual:.3f}', fontsize=11, color='green')

ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)'], fontsize=11)
ax3.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences Visualization', fontsize=14)
ax3.legend(loc='lower left', fontsize=10)
ax3.set_ylim([0.55, 0.70])
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
plt.close()
print("Saved figure3_did.png/pdf")

# Figure 4: Distribution of hours worked
print("\nGenerating Figure 4: Hours worked distribution...")

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

# Pre-period
pre_data = df_analysis[df_analysis['post'] == 0]
hours_bins = range(0, 100, 5)

ax4a = axes4[0]
treat_hours = pre_data[pre_data['treated']==1]['UHRSWORK']
ctrl_hours = pre_data[pre_data['treated']==0]['UHRSWORK']
ax4a.hist(treat_hours[treat_hours > 0], bins=hours_bins, alpha=0.5, label='Treatment', density=True)
ax4a.hist(ctrl_hours[ctrl_hours > 0], bins=hours_bins, alpha=0.5, label='Control', density=True)
ax4a.axvline(x=35, color='red', linestyle='--', linewidth=2, label='Full-Time Threshold (35 hrs)')
ax4a.set_xlabel('Usual Hours Worked per Week', fontsize=11)
ax4a.set_ylabel('Density', fontsize=11)
ax4a.set_title('Pre-DACA Period (2006-2011)', fontsize=12)
ax4a.legend(fontsize=9)

# Post-period
post_data = df_analysis[df_analysis['post'] == 1]
ax4b = axes4[1]
treat_hours = post_data[post_data['treated']==1]['UHRSWORK']
ctrl_hours = post_data[post_data['treated']==0]['UHRSWORK']
ax4b.hist(treat_hours[treat_hours > 0], bins=hours_bins, alpha=0.5, label='Treatment', density=True)
ax4b.hist(ctrl_hours[ctrl_hours > 0], bins=hours_bins, alpha=0.5, label='Control', density=True)
ax4b.axvline(x=35, color='red', linestyle='--', linewidth=2, label='Full-Time Threshold (35 hrs)')
ax4b.set_xlabel('Usual Hours Worked per Week', fontsize=11)
ax4b.set_ylabel('Density', fontsize=11)
ax4b.set_title('Post-DACA Period (2013-2016)', fontsize=12)
ax4b.legend(fontsize=9)

plt.suptitle('Distribution of Hours Worked by Treatment Status', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figure4_hours.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_hours.pdf', bbox_inches='tight')
plt.close()
print("Saved figure4_hours.png/pdf")

# Figure 5: Heterogeneity by gender
print("\nGenerating Figure 5: Heterogeneity by gender...")

fig5, ax5 = plt.subplots(figsize=(10, 6))

# Calculate means by gender and group
categories = ['Male Treatment', 'Male Control', 'Female Treatment', 'Female Control']
pre_means = []
post_means = []

for sex in [1, 2]:
    for treat in [1, 0]:
        pre_sub = df_analysis[(df_analysis['SEX']==sex) & (df_analysis['treated']==treat) & (df_analysis['post']==0)]
        post_sub = df_analysis[(df_analysis['SEX']==sex) & (df_analysis['treated']==treat) & (df_analysis['post']==1)]
        pre_means.append(np.average(pre_sub['fulltime'], weights=pre_sub['PERWT']))
        post_means.append(np.average(post_sub['fulltime'], weights=post_sub['PERWT']))

x = np.arange(len(categories))
width = 0.35

bars1 = ax5.bar(x - width/2, pre_means, width, label='Pre-DACA', color='steelblue', alpha=0.8)
bars2 = ax5.bar(x + width/2, post_means, width, label='Post-DACA', color='coral', alpha=0.8)

ax5.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax5.set_title('Full-Time Employment by Gender and Treatment Status', fontsize=14)
ax5.set_xticks(x)
ax5.set_xticklabels(categories, fontsize=10)
ax5.legend(fontsize=10)
ax5.set_ylim([0, 0.9])
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure5_gender.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_gender.pdf', bbox_inches='tight')
plt.close()
print("Saved figure5_gender.png/pdf")

# Figure 6: Age distribution of sample
print("\nGenerating Figure 6: Age distribution...")

fig6, ax6 = plt.subplots(figsize=(10, 6))

pre_data = df_analysis[df_analysis['post'] == 0]

ax6.hist(pre_data[pre_data['treated']==1]['AGE'], bins=range(18, 45), alpha=0.5,
         label='Treatment Group', density=True, edgecolor='black')
ax6.hist(pre_data[pre_data['treated']==0]['AGE'], bins=range(18, 45), alpha=0.5,
         label='Control Group', density=True, edgecolor='black')

ax6.axvline(x=26, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax6.axvline(x=30, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax6.axvline(x=31, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax6.axvline(x=35, color='red', linestyle='--', linewidth=2, alpha=0.7)

ax6.set_xlabel('Age at Survey (Pre-DACA Period)', fontsize=12)
ax6.set_ylabel('Density', fontsize=12)
ax6.set_title('Age Distribution of Treatment and Control Groups', fontsize=14)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure6_age.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_age.pdf', bbox_inches='tight')
plt.close()
print("Saved figure6_age.png/pdf")

print("\nAll figures generated successfully!")

# Print summary statistics for tables
print("\n" + "="*80)
print("TABLE DATA FOR LATEX")
print("="*80)

# Table 1: Sample characteristics
print("\nTable 1: Sample Characteristics")
print("-"*60)
pre_data = df_analysis[df_analysis['post'] == 0]
vars_list = [
    ('Age at survey', 'AGE'),
    ('Female', 'female'),
    ('Married', 'married'),
    ('Less than HS', 'educ_less_hs'),
    ('High school', 'educ_hs'),
    ('Some college', 'educ_some_college'),
    ('College+', 'educ_college'),
    ('Full-time employed', 'fulltime'),
    ('Hours worked (if >0)', 'UHRSWORK')
]

print(f"{'Variable':<25} {'Treatment':<15} {'Control':<15} {'Overall':<15}")
for name, var in vars_list:
    t_mean = pre_data[pre_data['treated']==1][var].mean()
    c_mean = pre_data[pre_data['treated']==0][var].mean()
    o_mean = pre_data[var].mean()
    print(f"{name:<25} {t_mean:<15.3f} {c_mean:<15.3f} {o_mean:<15.3f}")

print(f"\n{'N':<25} {len(pre_data[pre_data['treated']==1]):<15} {len(pre_data[pre_data['treated']==0]):<15} {len(pre_data):<15}")
