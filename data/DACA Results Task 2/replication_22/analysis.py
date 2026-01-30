"""
DACA Replication Study - Difference-in-Differences Analysis
Research Question: Effect of DACA eligibility on full-time employment
Treatment: Ages 26-30 at June 15, 2012
Control: Ages 31-35 at June 15, 2012

Memory-efficient version with chunked processing
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*60)
print("DACA REPLICATION STUDY")
print("="*60)

# ============================================
# STEP 1: Load Data with filtering during read
# ============================================

print("\nStep 1: Loading and filtering data...")

# Read data in chunks and filter immediately
chunks = []
chunksize = 100000

# Define columns we need
usecols = ['YEAR', 'PERWT', 'BIRTHYR', 'BIRTHQTR', 'BPL', 'HISPAN', 'CITIZEN',
           'YRIMMIG', 'UHRSWORK', 'EMPSTAT', 'SEX', 'AGE', 'EDUC', 'MARST',
           'NCHILD', 'STATEFIP']

for chunk in pd.read_csv("data/data.csv", usecols=usecols, chunksize=chunksize):
    # Filter for Mexican-born Hispanic non-citizens immediately
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3)].copy()
    if len(filtered) > 0:
        chunks.append(filtered)

data = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"Loaded filtered data: {len(data)} observations")
print(f"Years in data: {sorted(data['YEAR'].unique())}")

# ============================================
# STEP 2: Apply Additional Sample Restrictions
# ============================================

print("\n" + "="*60)
print("Step 2: Applying additional restrictions...")
print("="*60)

# Arrived before age 16 (YRIMMIG - BIRTHYR < 16)
data = data[data['YRIMMIG'] > 0].copy()
data['age_at_arrival'] = data['YRIMMIG'] - data['BIRTHYR']
data = data[data['age_at_arrival'] < 16].copy()
print(f"After arrived before age 16: {len(data)}")

# Arrived by June 15, 2007 (continuous US residence)
data = data[data['YRIMMIG'] <= 2007].copy()
print(f"After arrived by 2007: {len(data)}")

# ============================================
# STEP 3: Calculate Age at DACA Implementation
# ============================================

print("\n" + "="*60)
print("Step 3: Calculating age at DACA implementation...")
print("="*60)

def calc_age_at_daca(row):
    birth_month_map = {1: 2, 2: 5, 3: 8, 4: 11}
    birth_month = birth_month_map.get(row['BIRTHQTR'], 6)
    age = 2012 - row['BIRTHYR']
    if birth_month > 6:
        age -= 1
    return age

data['age_at_daca'] = data.apply(calc_age_at_daca, axis=1)

print("\nAge at DACA distribution (ages 20-40):")
age_counts = data[(data['age_at_daca'] >= 20) & (data['age_at_daca'] <= 40)]['age_at_daca'].value_counts().sort_index()
print(age_counts)

# ============================================
# STEP 4: Define Treatment and Control Groups
# ============================================

print("\n" + "="*60)
print("Step 4: Defining treatment and control groups...")
print("="*60)

# Treatment: Ages 26-30 at June 15, 2012
# Control: Ages 31-35 at June 15, 2012
data_a = data[(data['age_at_daca'] >= 26) & (data['age_at_daca'] <= 35)].copy()
del data
gc.collect()

data_a['treated'] = ((data_a['age_at_daca'] >= 26) & (data_a['age_at_daca'] <= 30)).astype(int)
data_a['treatment_group'] = np.where(data_a['treated'] == 1,
                                      "Treatment (26-30)", "Control (31-35)")

print("\nSample by treatment group:")
print(data_a['treatment_group'].value_counts())

# ============================================
# STEP 5: Define Time Periods
# ============================================

print("\n" + "="*60)
print("Step 5: Defining time periods...")
print("="*60)

# Exclude 2012 (transition year)
data_a = data_a[data_a['YEAR'] != 2012].copy()
data_a['post'] = (data_a['YEAR'] >= 2013).astype(int)
data_a['period'] = np.where(data_a['post'] == 1,
                            "Post-DACA (2013-2016)", "Pre-DACA (2006-2011)")

print("\nSample by year:")
print(data_a['YEAR'].value_counts().sort_index())

print("\nSample by period:")
print(data_a['period'].value_counts())

# ============================================
# STEP 6: Define Outcome Variable
# ============================================

print("\n" + "="*60)
print("Step 6: Creating outcome variable...")
print("="*60)

data_a['employed'] = (data_a['EMPSTAT'] == 1).astype(int)
data_a['fulltime'] = ((data_a['UHRSWORK'] >= 35) & (data_a['employed'] == 1)).astype(int)

print(f"\nEmployed: {data_a['employed'].sum()} / {len(data_a)} ({data_a['employed'].mean()*100:.1f}%)")
print(f"Full-time: {data_a['fulltime'].sum()} / {len(data_a)} ({data_a['fulltime'].mean()*100:.1f}%)")

# ============================================
# STEP 7: Create Control Variables
# ============================================

print("\n" + "="*60)
print("Step 7: Creating control variables...")
print("="*60)

data_a['female'] = (data_a['SEX'] == 2).astype(int)
data_a['age'] = data_a['AGE']
data_a['age_sq'] = data_a['AGE'] ** 2
data_a['edu_less_hs'] = (data_a['EDUC'] < 6).astype(int)
data_a['edu_hs'] = (data_a['EDUC'] == 6).astype(int)
data_a['edu_some_college'] = ((data_a['EDUC'] >= 7) & (data_a['EDUC'] <= 9)).astype(int)
data_a['edu_college_plus'] = (data_a['EDUC'] >= 10).astype(int)
data_a['married'] = ((data_a['MARST'] == 1) | (data_a['MARST'] == 2)).astype(int)
data_a['nchild'] = data_a['NCHILD']
data_a['years_in_us'] = data_a['YEAR'] - data_a['YRIMMIG']

# DiD interaction term
data_a['treated_post'] = data_a['treated'] * data_a['post']

# Final sample size
print(f"\nFinal analysis sample size: {len(data_a)}")

# ============================================
# STEP 8: Summary Statistics
# ============================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

def weighted_mean(df, col, weight_col='PERWT'):
    return np.average(df[col], weights=df[weight_col])

summary_stats = []
for group in data_a['treatment_group'].unique():
    for period in data_a['period'].unique():
        subset = data_a[(data_a['treatment_group'] == group) & (data_a['period'] == period)]
        summary_stats.append({
            'Group': group,
            'Period': period,
            'N': len(subset),
            'N_weighted': int(subset['PERWT'].sum()),
            'Fulltime_Rate': round(weighted_mean(subset, 'fulltime'), 4),
            'Employed_Rate': round(weighted_mean(subset, 'employed'), 4),
            'Mean_Age': round(weighted_mean(subset, 'age'), 2),
            'Female_Pct': round(weighted_mean(subset, 'female'), 4),
            'Married_Pct': round(weighted_mean(subset, 'married'), 4)
        })

summary_df = pd.DataFrame(summary_stats)
print("\n", summary_df.to_string(index=False))
summary_df.to_csv('summary_statistics.csv', index=False)

# Simple DiD calculation
pre_treat = summary_df[(summary_df['Group'] == 'Treatment (26-30)') &
                        (summary_df['Period'] == 'Pre-DACA (2006-2011)')]['Fulltime_Rate'].values[0]
post_treat = summary_df[(summary_df['Group'] == 'Treatment (26-30)') &
                         (summary_df['Period'] == 'Post-DACA (2013-2016)')]['Fulltime_Rate'].values[0]
pre_control = summary_df[(summary_df['Group'] == 'Control (31-35)') &
                          (summary_df['Period'] == 'Pre-DACA (2006-2011)')]['Fulltime_Rate'].values[0]
post_control = summary_df[(summary_df['Group'] == 'Control (31-35)') &
                           (summary_df['Period'] == 'Post-DACA (2013-2016)')]['Fulltime_Rate'].values[0]

simple_did = (post_treat - pre_treat) - (post_control - pre_control)

print("\n" + "-"*40)
print("Simple DiD Calculation:")
print(f"Treatment Pre:     {pre_treat:.4f}")
print(f"Treatment Post:    {post_treat:.4f}")
print(f"Treatment Change:  {post_treat - pre_treat:.4f}")
print(f"Control Pre:       {pre_control:.4f}")
print(f"Control Post:      {post_control:.4f}")
print(f"Control Change:    {post_control - pre_control:.4f}")
print(f"DiD Estimate:      {simple_did:.4f}")
print("-"*40)

# ============================================
# STEP 9: Regression Analysis
# ============================================

print("\n" + "="*60)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("="*60)

# Model 1: Basic DiD
print("\nModel 1: Basic DiD (no controls)")
model1 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=data_a, weights=data_a['PERWT']).fit(cov_type='HC1')
print(f"treated_post: {model1.params['treated_post']:.6f} (SE: {model1.bse['treated_post']:.6f}, p: {model1.pvalues['treated_post']:.4f})")

# Model 2: DiD with demographic controls
print("\nModel 2: DiD with demographic controls")
model2 = smf.wls('fulltime ~ treated + post + treated_post + female + age + age_sq + married + nchild + years_in_us',
                  data=data_a, weights=data_a['PERWT']).fit(cov_type='HC1')
print(f"treated_post: {model2.params['treated_post']:.6f} (SE: {model2.bse['treated_post']:.6f}, p: {model2.pvalues['treated_post']:.4f})")

# Model 3: DiD with education controls
print("\nModel 3: DiD with education controls")
model3 = smf.wls('fulltime ~ treated + post + treated_post + female + age + age_sq + married + nchild + years_in_us + edu_hs + edu_some_college + edu_college_plus',
                  data=data_a, weights=data_a['PERWT']).fit(cov_type='HC1')
print(f"treated_post: {model3.params['treated_post']:.6f} (SE: {model3.bse['treated_post']:.6f}, p: {model3.pvalues['treated_post']:.4f})")

# Model 4: DiD with state fixed effects
print("\nModel 4: DiD with state fixed effects")
model4 = smf.wls('fulltime ~ treated + post + treated_post + female + age + age_sq + married + nchild + years_in_us + edu_hs + edu_some_college + edu_college_plus + C(STATEFIP)',
                  data=data_a, weights=data_a['PERWT']).fit(cov_type='HC1')
print(f"treated_post: {model4.params['treated_post']:.6f} (SE: {model4.bse['treated_post']:.6f}, p: {model4.pvalues['treated_post']:.4f})")

# Model 5: DiD with state and year fixed effects (PREFERRED)
print("\nModel 5: DiD with state and year fixed effects (PREFERRED)")
model5 = smf.wls('fulltime ~ treated + treated_post + female + age + age_sq + married + nchild + years_in_us + edu_hs + edu_some_college + edu_college_plus + C(STATEFIP) + C(YEAR)',
                  data=data_a, weights=data_a['PERWT']).fit(cov_type='HC1')
print(f"treated_post: {model5.params['treated_post']:.6f} (SE: {model5.bse['treated_post']:.6f}, p: {model5.pvalues['treated_post']:.4f})")

# Extract key results
did_coef = model5.params['treated_post']
did_se = model5.bse['treated_post']
did_t = model5.tvalues['treated_post']
did_pvalue = model5.pvalues['treated_post']
did_ci = model5.conf_int().loc['treated_post']

# ============================================
# STEP 10: Robustness Checks
# ============================================

print("\n" + "="*60)
print("ROBUSTNESS CHECKS")
print("="*60)

# Placebo test
print("\nPlacebo Test: Fake treatment in 2009 (pre-period only)")
data_placebo = data_a[data_a['YEAR'] <= 2011].copy()
data_placebo['post_placebo'] = (data_placebo['YEAR'] >= 2009).astype(int)
data_placebo['treated_post_placebo'] = data_placebo['treated'] * data_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treated + post_placebo + treated_post_placebo + female + age + age_sq + married + nchild + years_in_us + C(STATEFIP)',
                         data=data_placebo, weights=data_placebo['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD: {model_placebo.params['treated_post_placebo']:.6f} (SE: {model_placebo.bse['treated_post_placebo']:.6f}, p: {model_placebo.pvalues['treated_post_placebo']:.4f})")

# Gender heterogeneity
print("\nHeterogeneous Effects by Gender:")
data_male = data_a[data_a['female'] == 0].copy()
model_male = smf.wls('fulltime ~ treated + treated_post + age + age_sq + married + nchild + years_in_us + edu_hs + edu_some_college + edu_college_plus + C(STATEFIP) + C(YEAR)',
                      data=data_male, weights=data_male['PERWT']).fit(cov_type='HC1')
print(f"Males:   {model_male.params['treated_post']:.6f} (SE: {model_male.bse['treated_post']:.6f}, p: {model_male.pvalues['treated_post']:.4f})")

data_female = data_a[data_a['female'] == 1].copy()
model_female = smf.wls('fulltime ~ treated + treated_post + age + age_sq + married + nchild + years_in_us + edu_hs + edu_some_college + edu_college_plus + C(STATEFIP) + C(YEAR)',
                        data=data_female, weights=data_female['PERWT']).fit(cov_type='HC1')
print(f"Females: {model_female.params['treated_post']:.6f} (SE: {model_female.bse['treated_post']:.6f}, p: {model_female.pvalues['treated_post']:.4f})")

# ============================================
# STEP 11: Event Study
# ============================================

print("\n" + "="*60)
print("EVENT STUDY ANALYSIS")
print("="*60)

years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in years:
    data_a[f'treat_{year}'] = (data_a['treated'] * (data_a['YEAR'] == year)).astype(int)

formula = 'fulltime ~ treated + ' + ' + '.join([f'treat_{y}' for y in years]) + ' + female + age + age_sq + married + nchild + years_in_us + C(STATEFIP) + C(YEAR)'
model_event = smf.wls(formula, data=data_a, weights=data_a['PERWT']).fit(cov_type='HC1')

event_coefs = []
for year in years:
    event_coefs.append({
        'year': year,
        'coef': model_event.params[f'treat_{year}'],
        'se': model_event.bse[f'treat_{year}'],
        'ci_lower': model_event.conf_int().loc[f'treat_{year}'][0],
        'ci_upper': model_event.conf_int().loc[f'treat_{year}'][1]
    })
event_coefs.append({'year': 2011, 'coef': 0, 'se': 0, 'ci_lower': 0, 'ci_upper': 0})
event_coefs_df = pd.DataFrame(event_coefs).sort_values('year')
print("\nEvent Study Coefficients (Reference: 2011):")
print(event_coefs_df.to_string(index=False))
event_coefs_df.to_csv('event_study_coefficients.csv', index=False)

# ============================================
# STEP 12: Create Figures
# ============================================

print("\n" + "="*60)
print("CREATING FIGURES")
print("="*60)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Trends
    trends_data = data_a.groupby(['YEAR', 'treatment_group']).apply(
        lambda x: np.average(x['fulltime'], weights=x['PERWT'])
    ).reset_index(name='fulltime_rate')

    fig, ax = plt.subplots(figsize=(10, 6))
    for group in trends_data['treatment_group'].unique():
        group_data = trends_data[trends_data['treatment_group'] == group].sort_values('YEAR')
        ax.plot(group_data['YEAR'], group_data['fulltime_rate'], marker='o', label=group, linewidth=2)

    ax.axvline(x=2012, color='gray', linestyle='--', alpha=0.7)
    ax.text(2012.1, ax.get_ylim()[1]*0.95, 'DACA\nImplementation', fontsize=9)
    ax.set_xlabel('Year')
    ax.set_ylabel('Full-Time Employment Rate')
    ax.set_title('Full-Time Employment Rates Over Time\nTreatment (Ages 26-30) vs Control (Ages 31-35) at DACA Implementation')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(2006, 2017))
    plt.tight_layout()
    plt.savefig('figure1_trends.png', dpi=300)
    plt.close()
    print("Figure 1 saved: figure1_trends.png")

    # Figure 2: Event Study
    fig, ax = plt.subplots(figsize=(10, 6))
    event_plot = event_coefs_df.copy()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=2012, color='red', linestyle='--', alpha=0.5)
    ax.errorbar(event_plot['year'], event_plot['coef'],
                yerr=[event_plot['coef'] - event_plot['ci_lower'],
                      event_plot['ci_upper'] - event_plot['coef']],
                fmt='o', capsize=4, color='blue', markersize=8)
    ax.text(2012.1, ax.get_ylim()[1]*0.9, 'DACA', fontsize=10, color='red')
    ax.set_xlabel('Year')
    ax.set_ylabel('Treatment Effect on Full-Time Employment')
    ax.set_title('Event Study: Dynamic Treatment Effects\nReference Year: 2011 (year before DACA)')
    ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure2_event_study.png', dpi=300)
    plt.close()
    print("Figure 2 saved: figure2_event_study.png")

except Exception as e:
    print(f"Could not create figures: {e}")

# ============================================
# STEP 13: Save Results
# ============================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_summary = pd.DataFrame({
    'Model': ['Basic DiD', 'Demographic Controls', 'Education Controls',
              'State FE', 'State + Year FE (Preferred)'],
    'DiD_Coefficient': [
        model1.params['treated_post'],
        model2.params['treated_post'],
        model3.params['treated_post'],
        model4.params['treated_post'],
        model5.params['treated_post']
    ],
    'Std_Error': [
        model1.bse['treated_post'],
        model2.bse['treated_post'],
        model3.bse['treated_post'],
        model4.bse['treated_post'],
        model5.bse['treated_post']
    ],
    'p_value': [
        model1.pvalues['treated_post'],
        model2.pvalues['treated_post'],
        model3.pvalues['treated_post'],
        model4.pvalues['treated_post'],
        model5.pvalues['treated_post']
    ],
    'Sample_Size': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model5.nobs)
    ]
})

results_summary['CI_Lower'] = results_summary['DiD_Coefficient'] - 1.96 * results_summary['Std_Error']
results_summary['CI_Upper'] = results_summary['DiD_Coefficient'] + 1.96 * results_summary['Std_Error']

print("\nRegression Results Summary:")
print(results_summary.to_string(index=False))
results_summary.to_csv('regression_results.csv', index=False)

# ============================================
# FINAL RESULTS
# ============================================

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print(f"\nPreferred Estimate (Model 5 with State and Year Fixed Effects):")
print(f"DiD Coefficient (Effect of DACA on Full-Time Employment): {did_coef:.6f}")
direction = "increased" if did_coef > 0 else "decreased"
print(f"Interpretation: DACA eligibility {direction} full-time employment by {abs(did_coef * 100):.2f} percentage points")
print(f"Standard Error: {did_se:.6f}")
print(f"95% Confidence Interval: [{did_ci[0]:.6f}, {did_ci[1]:.6f}]")
print(f"Sample Size: {int(model5.nobs)}")
sig = "significant at 5% level" if did_pvalue < 0.05 else "not significant at 5% level"
print(f"Statistical Significance: p = {did_pvalue:.4f} ({sig})")

# Save final results
final_results = {
    'preferred_estimate': did_coef,
    'standard_error': did_se,
    'ci_lower': did_ci[0],
    'ci_upper': did_ci[1],
    'sample_size': int(model5.nobs),
    'p_value': did_pvalue,
    'simple_did': simple_did
}
pd.DataFrame([final_results]).to_csv('final_results.csv', index=False)

print("\n" + "="*60)
print("Analysis complete. Results saved to files.")
print("="*60)
