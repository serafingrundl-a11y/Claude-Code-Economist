"""
DACA Replication Analysis - Optimized for Large Dataset
Independent replication of the causal effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals.

Research Design: Difference-in-Differences
- Treatment group: Ages 26-30 as of June 15, 2012 (born 1982-1986)
- Control group: Ages 31-35 as of June 15, 2012 (born 1977-1981)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded due to policy implementation mid-year)
- Outcome: Full-time employment (>=35 hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = r"C:\Users\seraf\DACA Results Task 2\replication_08\data\data.csv"
OUTPUT_DIR = r"C:\Users\seraf\DACA Results Task 2\replication_08"

print("=" * 60)
print("DACA REPLICATION ANALYSIS - OPTIMIZED")
print("=" * 60)

# Load data in chunks, filtering as we go
cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR', 'MARST',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EMPSTAT', 'UHRSWORK', 'STATEFIP']

print("\nLoading and filtering data in chunks...")

# Define dtypes for efficiency
dtypes = {
    'YEAR': 'int16',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'MARST': 'int8',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8',
    'STATEFIP': 'int8'
}

# Process in chunks
chunk_size = 500000
filtered_chunks = []

for i, chunk in enumerate(pd.read_csv(DATA_PATH, usecols=cols_needed, dtype=dtypes, chunksize=chunk_size)):
    print(f"Processing chunk {i+1}...")

    # Apply filters progressively:
    # 1. Hispanic-Mexican (HISPAN == 1)
    chunk = chunk[chunk['HISPAN'] == 1]

    # 2. Born in Mexico (BPL == 200)
    chunk = chunk[chunk['BPL'] == 200]

    # 3. Non-citizen (CITIZEN == 3)
    chunk = chunk[chunk['CITIZEN'] == 3]

    # 4. Valid immigration year and arrived before 16
    chunk = chunk[chunk['YRIMMIG'] > 0]
    chunk['age_at_immig'] = chunk['YRIMMIG'] - chunk['BIRTHYR']
    chunk = chunk[chunk['age_at_immig'] < 16]

    # 5. In US since 2007 or earlier
    chunk = chunk[chunk['YRIMMIG'] <= 2007]

    # 6. Birth year in our age range (1977-1986)
    chunk = chunk[(chunk['BIRTHYR'] >= 1977) & (chunk['BIRTHYR'] <= 1986)]

    # 7. Exclude 2012
    chunk = chunk[chunk['YEAR'] != 2012]

    if len(chunk) > 0:
        filtered_chunks.append(chunk)

    gc.collect()

print("Combining filtered data...")
df_analysis = pd.concat(filtered_chunks, ignore_index=True)
del filtered_chunks
gc.collect()

print(f"Total filtered observations: {len(df_analysis):,}")

# ============================================================================
# CREATE VARIABLES
# ============================================================================
print("\n" + "=" * 60)
print("CREATING ANALYSIS VARIABLES")
print("=" * 60)

# Treatment indicator (1=treatment, 0=control)
df_analysis['treat'] = np.where(df_analysis['BIRTHYR'] >= 1982, 1, 0).astype('int8')

# Post indicator (1=post-DACA, 0=pre-DACA)
df_analysis['post'] = np.where(df_analysis['YEAR'] >= 2013, 1, 0).astype('int8')

# Interaction term
df_analysis['treat_post'] = (df_analysis['treat'] * df_analysis['post']).astype('int8')

# Outcome: Full-time employment (35+ hours/week)
df_analysis['fulltime'] = np.where(df_analysis['UHRSWORK'] >= 35, 1, 0).astype('int8')

# Alternative outcome: Any employment
df_analysis['employed'] = np.where(df_analysis['EMPSTAT'] == 1, 1, 0).astype('int8')

# Covariates
df_analysis['female'] = np.where(df_analysis['SEX'] == 2, 1, 0).astype('int8')
df_analysis['married'] = np.where(df_analysis['MARST'].isin([1, 2]), 1, 0).astype('int8')
df_analysis['educ_lths'] = np.where(df_analysis['EDUC'] < 6, 1, 0).astype('int8')
df_analysis['educ_hs'] = np.where(df_analysis['EDUC'] == 6, 1, 0).astype('int8')
df_analysis['educ_somecol'] = np.where((df_analysis['EDUC'] >= 7) & (df_analysis['EDUC'] <= 9), 1, 0).astype('int8')
df_analysis['educ_college'] = np.where(df_analysis['EDUC'] >= 10, 1, 0).astype('int8')
df_analysis['age'] = df_analysis['AGE'].astype('int16')
df_analysis['year'] = df_analysis['YEAR'].astype('int16')
df_analysis['state'] = df_analysis['STATEFIP'].astype('int8')

print(f"Treatment group (26-30): {(df_analysis['treat']==1).sum():,}")
print(f"Control group (31-35): {(df_analysis['treat']==0).sum():,}")
print(f"Pre-period (2006-2011): {(df_analysis['post']==0).sum():,}")
print(f"Post-period (2013-2016): {(df_analysis['post']==1).sum():,}")

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

def weighted_mean(df, value_col, weight_col):
    return np.average(df[value_col], weights=df[weight_col])

def weighted_std(df, value_col, weight_col):
    avg = weighted_mean(df, value_col, weight_col)
    variance = np.average((df[value_col] - avg)**2, weights=df[weight_col])
    return np.sqrt(variance)

print("\n--- Sample Characteristics by Treatment Status ---")
for treat_val in [0, 1]:
    treat_label = "Treatment (26-30)" if treat_val == 1 else "Control (31-35)"
    subset = df_analysis[df_analysis['treat'] == treat_val]
    print(f"\n{treat_label} (N={len(subset):,}):")
    print(f"  Full-time rate: {weighted_mean(subset, 'fulltime', 'PERWT')*100:.2f}%")
    print(f"  Employment rate: {weighted_mean(subset, 'employed', 'PERWT')*100:.2f}%")
    print(f"  Female: {weighted_mean(subset, 'female', 'PERWT')*100:.1f}%")
    print(f"  Married: {weighted_mean(subset, 'married', 'PERWT')*100:.1f}%")
    print(f"  Mean Age: {weighted_mean(subset, 'age', 'PERWT'):.1f}")
    print(f"  Less than HS: {weighted_mean(subset, 'educ_lths', 'PERWT')*100:.1f}%")
    print(f"  HS: {weighted_mean(subset, 'educ_hs', 'PERWT')*100:.1f}%")
    print(f"  Some College: {weighted_mean(subset, 'educ_somecol', 'PERWT')*100:.1f}%")
    print(f"  College+: {weighted_mean(subset, 'educ_college', 'PERWT')*100:.1f}%")

print("\n--- Full-time Employment by Group and Period ---")
for treat_val in [0, 1]:
    treat_label = "Treatment (26-30)" if treat_val == 1 else "Control (31-35)"
    for post_val in [0, 1]:
        period_label = "Post (2013-2016)" if post_val == 1 else "Pre (2006-2011)"
        subset = df_analysis[(df_analysis['treat'] == treat_val) & (df_analysis['post'] == post_val)]
        wt_mean = weighted_mean(subset, 'fulltime', 'PERWT')
        print(f"{treat_label}, {period_label}: {wt_mean*100:.2f}% (n={len(subset):,})")

# Simple DID calculation
pre_treat = weighted_mean(df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==0)], 'fulltime', 'PERWT')
post_treat = weighted_mean(df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==1)], 'fulltime', 'PERWT')
pre_control = weighted_mean(df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==0)], 'fulltime', 'PERWT')
post_control = weighted_mean(df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==1)], 'fulltime', 'PERWT')

print(f"\nSimple DID Calculation (weighted):")
print(f"Treatment group change: {(post_treat - pre_treat)*100:.2f} pp ({pre_treat*100:.2f}% -> {post_treat*100:.2f}%)")
print(f"Control group change: {(post_control - pre_control)*100:.2f} pp ({pre_control*100:.2f}% -> {post_control*100:.2f}%)")
simple_did = (post_treat - pre_treat) - (post_control - pre_control)
print(f"DID estimate: {simple_did*100:.2f} pp")

# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("=" * 60)

# Model 1: Basic DID
print("\n--- MODEL 1: Basic DID (no controls) ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treat_post: {model1.params['treat_post']:.4f} (SE: {model1.bse['treat_post']:.4f}, p={model1.pvalues['treat_post']:.4f})")
print(f"R-squared: {model1.rsquared:.4f}")

# Model 2: DID with demographic controls
print("\n--- MODEL 2: DID with Demographics ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + age',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treat_post: {model2.params['treat_post']:.4f} (SE: {model2.bse['treat_post']:.4f}, p={model2.pvalues['treat_post']:.4f})")
print(f"R-squared: {model2.rsquared:.4f}")

# Model 3: DID with demographics and education
print("\n--- MODEL 3: DID with Demographics and Education ---")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + age + educ_hs + educ_somecol + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treat_post: {model3.params['treat_post']:.4f} (SE: {model3.bse['treat_post']:.4f}, p={model3.pvalues['treat_post']:.4f})")
print(f"R-squared: {model3.rsquared:.4f}")

# Model 4: DID with year fixed effects (PREFERRED)
print("\n--- MODEL 4: DID with Year Fixed Effects (PREFERRED) ---")
model4 = smf.wls('fulltime ~ treat + treat_post + female + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treat_post: {model4.params['treat_post']:.4f} (SE: {model4.bse['treat_post']:.4f}, p={model4.pvalues['treat_post']:.4f})")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"R-squared: {model4.rsquared:.4f}")

# Model 5: DID with year and state fixed effects
print("\n--- MODEL 5: DID with Year and State Fixed Effects ---")
model5 = smf.wls('fulltime ~ treat + treat_post + female + married + age + educ_hs + educ_somecol + educ_college + C(year) + C(state)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treat_post: {model5.params['treat_post']:.4f} (SE: {model5.bse['treat_post']:.4f}, p={model5.pvalues['treat_post']:.4f})")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"R-squared: {model5.rsquared:.4f}")

# ============================================================================
# ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 60)
print("ROBUSTNESS CHECKS")
print("=" * 60)

# Robustness 1: Any employment as outcome
print("\n--- Robustness 1: Any Employment ---")
model_emp = smf.wls('employed ~ treat + treat_post + female + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                     data=df_analysis,
                     weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"treat_post: {model_emp.params['treat_post']:.4f} (SE: {model_emp.bse['treat_post']:.4f}, p={model_emp.pvalues['treat_post']:.4f})")

# Robustness 2: By gender
print("\n--- Robustness 2: By Gender ---")
for sex_val, sex_label in [(0, 'Male'), (1, 'Female')]:
    subset = df_analysis[df_analysis['female'] == sex_val]
    model_sex = smf.wls('fulltime ~ treat + treat_post + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                        data=subset,
                        weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"{sex_label}: {model_sex.params['treat_post']:.4f} (SE: {model_sex.bse['treat_post']:.4f}, p={model_sex.pvalues['treat_post']:.4f})")

# Robustness 3: Narrower age bands
print("\n--- Robustness 3: Narrower Age Bands (1982-1984 vs 1978-1980) ---")
df_narrow = df_analysis[
    ((df_analysis['BIRTHYR'] >= 1982) & (df_analysis['BIRTHYR'] <= 1984)) |
    ((df_analysis['BIRTHYR'] >= 1978) & (df_analysis['BIRTHYR'] <= 1980))
].copy()
df_narrow['treat_narrow'] = np.where(df_narrow['BIRTHYR'] >= 1982, 1, 0)
df_narrow['treat_post_narrow'] = df_narrow['treat_narrow'] * df_narrow['post']
model_narrow = smf.wls('fulltime ~ treat_narrow + treat_post_narrow + female + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                        data=df_narrow,
                        weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"treat_post: {model_narrow.params['treat_post_narrow']:.4f} (SE: {model_narrow.bse['treat_post_narrow']:.4f}, p={model_narrow.pvalues['treat_post_narrow']:.4f})")

# Robustness 4: Placebo test
print("\n--- Robustness 4: Placebo Test (Pre-period, fake policy 2009) ---")
df_placebo = df_analysis[df_analysis['post'] == 0].copy()
df_placebo['post_placebo'] = np.where(df_placebo['year'] >= 2009, 1, 0)
df_placebo['treat_post_placebo'] = df_placebo['treat'] * df_placebo['post_placebo']
model_placebo = smf.wls('fulltime ~ treat + treat_post_placebo + female + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                         data=df_placebo,
                         weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"Placebo effect: {model_placebo.params['treat_post_placebo']:.4f} (SE: {model_placebo.bse['treat_post_placebo']:.4f}, p={model_placebo.pvalues['treat_post_placebo']:.4f})")

# ============================================================================
# EVENT STUDY
# ============================================================================
print("\n" + "=" * 60)
print("EVENT STUDY")
print("=" * 60)

years_event = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for yr in years_event:
    df_analysis[f'treat_year_{yr}'] = np.where((df_analysis['treat'] == 1) & (df_analysis['year'] == yr), 1, 0)

event_formula = 'fulltime ~ treat + ' + ' + '.join([f'treat_year_{yr}' for yr in years_event]) + ' + female + married + age + educ_hs + educ_somecol + educ_college + C(year)'
model_event = smf.wls(event_formula, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 55)
print(f"{'Year':>6} {'Coef':>10} {'SE':>10} {'95% CI Low':>12} {'95% CI High':>12}")
print("-" * 55)
for yr in sorted(years_event + [2011]):
    if yr == 2011:
        print(f"{yr:>6} {0.0:>10.4f} {'(ref)':>10} {'-':>12} {'-':>12}")
    else:
        coef = model_event.params[f'treat_year_{yr}']
        se = model_event.bse[f'treat_year_{yr}']
        ci_low = model_event.conf_int().loc[f'treat_year_{yr}', 0]
        ci_high = model_event.conf_int().loc[f'treat_year_{yr}', 1]
        print(f"{yr:>6} {coef:>10.4f} {se:>10.4f} {ci_low:>12.4f} {ci_high:>12.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save main results
results = {
    'n_total': int(len(df_analysis)),
    'n_treat': int((df_analysis['treat']==1).sum()),
    'n_control': int((df_analysis['treat']==0).sum()),
    'n_pre': int((df_analysis['post']==0).sum()),
    'n_post': int((df_analysis['post']==1).sum()),

    'simple_did': float(simple_did),

    'model1_coef': float(model1.params['treat_post']),
    'model1_se': float(model1.bse['treat_post']),
    'model1_pval': float(model1.pvalues['treat_post']),

    'model4_coef': float(model4.params['treat_post']),
    'model4_se': float(model4.bse['treat_post']),
    'model4_pval': float(model4.pvalues['treat_post']),
    'model4_ci_low': float(model4.conf_int().loc['treat_post', 0]),
    'model4_ci_high': float(model4.conf_int().loc['treat_post', 1]),

    'model5_coef': float(model5.params['treat_post']),
    'model5_se': float(model5.bse['treat_post']),
    'model5_pval': float(model5.pvalues['treat_post']),
    'model5_ci_low': float(model5.conf_int().loc['treat_post', 0]),
    'model5_ci_high': float(model5.conf_int().loc['treat_post', 1]),

    'employment_effect': float(model_emp.params['treat_post']),
    'male_effect': float(smf.wls('fulltime ~ treat + treat_post + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                                  data=df_analysis[df_analysis['female']==0],
                                  weights=df_analysis[df_analysis['female']==0]['PERWT']).fit(cov_type='HC1').params['treat_post']),
    'female_effect': float(smf.wls('fulltime ~ treat + treat_post + married + age + educ_hs + educ_somecol + educ_college + C(year)',
                                    data=df_analysis[df_analysis['female']==1],
                                    weights=df_analysis[df_analysis['female']==1]['PERWT']).fit(cov_type='HC1').params['treat_post']),
    'narrow_effect': float(model_narrow.params['treat_post_narrow']),
    'placebo_effect': float(model_placebo.params['treat_post_placebo']),
    'placebo_pval': float(model_placebo.pvalues['treat_post_placebo']),

    'pre_treat_rate': float(pre_treat),
    'post_treat_rate': float(post_treat),
    'pre_control_rate': float(pre_control),
    'post_control_rate': float(post_control),
}

import json
with open(f"{OUTPUT_DIR}/analysis_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to analysis_results.json")

# Save figure data
fig1_data = df_analysis.groupby(['year', 'treat']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'employed_rate': np.average(x['employed'], weights=x['PERWT']),
        'n': len(x)
    })
).reset_index()
fig1_data.to_csv(f"{OUTPUT_DIR}/figure1_data.csv", index=False)
print("Figure 1 data saved.")

# Event study data
event_study_data = []
for yr in years_event:
    event_study_data.append({
        'year': yr,
        'coef': float(model_event.params[f'treat_year_{yr}']),
        'se': float(model_event.bse[f'treat_year_{yr}']),
        'ci_low': float(model_event.conf_int().loc[f'treat_year_{yr}', 0]),
        'ci_high': float(model_event.conf_int().loc[f'treat_year_{yr}', 1])
    })
event_study_data.append({'year': 2011, 'coef': 0.0, 'se': 0.0, 'ci_low': 0.0, 'ci_high': 0.0})
event_study_df = pd.DataFrame(event_study_data).sort_values('year')
event_study_df.to_csv(f"{OUTPUT_DIR}/figure2_event_study.csv", index=False)
print("Event study data saved.")

# Save descriptive statistics
desc_stats = []
for treat_val in [0, 1]:
    for post_val in [0, 1]:
        subset = df_analysis[(df_analysis['treat'] == treat_val) & (df_analysis['post'] == post_val)]
        desc_stats.append({
            'treat': treat_val,
            'post': post_val,
            'n': len(subset),
            'fulltime_rate': weighted_mean(subset, 'fulltime', 'PERWT'),
            'employed_rate': weighted_mean(subset, 'employed', 'PERWT'),
            'female_pct': weighted_mean(subset, 'female', 'PERWT'),
            'married_pct': weighted_mean(subset, 'married', 'PERWT'),
            'mean_age': weighted_mean(subset, 'age', 'PERWT'),
            'educ_lths_pct': weighted_mean(subset, 'educ_lths', 'PERWT'),
            'educ_hs_pct': weighted_mean(subset, 'educ_hs', 'PERWT'),
            'educ_somecol_pct': weighted_mean(subset, 'educ_somecol', 'PERWT'),
            'educ_college_pct': weighted_mean(subset, 'educ_college', 'PERWT'),
        })
desc_df = pd.DataFrame(desc_stats)
desc_df.to_csv(f"{OUTPUT_DIR}/descriptive_stats.csv", index=False)
print("Descriptive statistics saved.")

# Save model summaries
with open(f"{OUTPUT_DIR}/model_summaries.txt", 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DACA REPLICATION - MODEL SUMMARIES\n")
    f.write("=" * 80 + "\n\n")

    f.write("MODEL 1: Basic DID\n")
    f.write("-" * 40 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("MODEL 2: DID with Demographics\n")
    f.write("-" * 40 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("MODEL 3: DID with Demographics and Education\n")
    f.write("-" * 40 + "\n")
    f.write(str(model3.summary()) + "\n\n")

    f.write("MODEL 4: DID with Year Fixed Effects (PREFERRED)\n")
    f.write("-" * 40 + "\n")
    f.write(str(model4.summary()) + "\n\n")

    f.write("MODEL 5: DID with Year and State Fixed Effects\n")
    f.write("-" * 40 + "\n")
    f.write(str(model5.summary()) + "\n\n")

    f.write("EVENT STUDY MODEL\n")
    f.write("-" * 40 + "\n")
    f.write(str(model_event.summary()) + "\n\n")
print("Model summaries saved.")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"\nSample Size: {results['n_total']:,}")
print(f"  Treatment group: {results['n_treat']:,}")
print(f"  Control group: {results['n_control']:,}")

print(f"\nPREFERRED ESTIMATE (Model 4: Year FE + Controls)")
print(f"  Effect Size: {results['model4_coef']*100:.2f} percentage points")
print(f"  Standard Error: {results['model4_se']*100:.2f} pp")
print(f"  95% CI: [{results['model4_ci_low']*100:.2f}, {results['model4_ci_high']*100:.2f}]")
print(f"  p-value: {results['model4_pval']:.4f}")

print(f"\nInterpretation:")
if results['model4_coef'] > 0:
    print(f"  DACA eligibility is associated with a {abs(results['model4_coef']*100):.2f} percentage point")
    print(f"  INCREASE in full-time employment for the treatment group relative to the control group.")
else:
    print(f"  DACA eligibility is associated with a {abs(results['model4_coef']*100):.2f} percentage point")
    print(f"  DECREASE in full-time employment for the treatment group relative to the control group.")

if results['model4_pval'] < 0.05:
    print(f"  This effect is statistically significant at the 5% level.")
else:
    print(f"  This effect is NOT statistically significant at the 5% level.")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
