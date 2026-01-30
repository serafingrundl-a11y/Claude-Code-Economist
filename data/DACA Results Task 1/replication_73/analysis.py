"""
DACA Replication Study: Effect on Full-Time Employment
Session 73

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for DACA on
full-time employment (>=35 hours/week)?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Set output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_filter_data(filepath):
    """
    Load ACS data and filter to target population:
    - Hispanic-Mexican ethnicity (HISPAN == 1)
    - Born in Mexico (BPL == 200)
    """
    print("Loading data...")

    # Define columns to use
    usecols = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN',
               'YRIMMIG', 'YRSUSA1', 'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK']

    # Load data in chunks due to large file size
    chunks = []
    chunksize = 1000000

    for chunk in pd.read_csv(filepath, usecols=usecols, chunksize=chunksize):
        # Filter to Hispanic-Mexican born in Mexico
        filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
        chunks.append(filtered)
        print(f"  Processed chunk, kept {len(filtered)} rows")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Total rows after filtering to Mexican-born Hispanic-Mexican: {len(df)}")

    return df


def create_daca_eligibility(df):
    """
    Determine DACA eligibility based on program criteria:
    1. Arrived in US before age 16
    2. Born on or after June 15, 1981 (age < 31 as of June 15, 2012)
    3. Immigrated by 2007 (continuous presence since June 15, 2007)
    4. Not a citizen (proxy for undocumented status)

    We use age-at-arrival which can be approximated from YRIMMIG and BIRTHYR
    """
    print("\nCreating DACA eligibility indicator...")

    # Calculate age at immigration
    # YRIMMIG = 0 means N/A (likely born in US or unknown)
    df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']

    # Create eligibility criteria
    # Criterion 1: Arrived before age 16
    df['arrived_before_16'] = (df['age_at_immigration'] < 16) & (df['age_at_immigration'] >= 0)

    # Criterion 2: Born on or after June 15, 1981 (under 31 as of June 15, 2012)
    # Being conservative: born in 1981 or later
    df['age_eligible'] = df['BIRTHYR'] >= 1981

    # Criterion 3: Continuous presence since June 15, 2007
    # YRIMMIG <= 2007
    df['presence_eligible'] = df['YRIMMIG'] <= 2007

    # Criterion 4: Non-citizen (proxy for undocumented)
    # CITIZEN = 3 means "Not a citizen"
    df['non_citizen'] = df['CITIZEN'] == 3

    # Combined DACA eligibility
    df['daca_eligible'] = (df['arrived_before_16'] &
                           df['age_eligible'] &
                           df['presence_eligible'] &
                           df['non_citizen'])

    # Valid immigration year (exclude N/A values)
    df['valid_yrimmig'] = df['YRIMMIG'] > 0

    print(f"  Arrived before 16: {df['arrived_before_16'].sum():,}")
    print(f"  Age eligible (born >= 1981): {df['age_eligible'].sum():,}")
    print(f"  Presence eligible (YRIMMIG <= 2007): {df['presence_eligible'].sum():,}")
    print(f"  Non-citizen: {df['non_citizen'].sum():,}")
    print(f"  DACA eligible: {df['daca_eligible'].sum():,}")

    return df


def create_analysis_variables(df):
    """
    Create outcome and control variables for analysis
    """
    print("\nCreating analysis variables...")

    # Outcome: Full-time employment (35+ hours/week)
    df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

    # Employed indicator
    df['employed'] = (df['EMPSTAT'] == 1).astype(int)

    # Post-DACA indicator (2013-2016)
    # Exclude 2012 as transition year
    df['post'] = (df['YEAR'] >= 2013).astype(int)

    # Interaction term for DiD
    df['eligible_post'] = df['daca_eligible'].astype(int) * df['post']

    # Control variables
    df['age_sq'] = df['AGE'] ** 2
    df['female'] = (df['SEX'] == 2).astype(int)
    df['married'] = (df['MARST'].isin([1, 2])).astype(int)

    # Education categories
    df['less_than_hs'] = (df['EDUC'] < 6).astype(int)
    df['hs_graduate'] = (df['EDUC'] == 6).astype(int)
    df['some_college'] = (df['EDUC'].isin([7, 8, 9])).astype(int)
    df['college_plus'] = (df['EDUC'] >= 10).astype(int)

    # Convert eligibility to int
    df['eligible'] = df['daca_eligible'].astype(int)

    return df


def restrict_sample(df):
    """
    Apply sample restrictions:
    1. Working-age population (16-64)
    2. Valid immigration year
    3. Exclude 2012 (transition year)
    """
    print("\nApplying sample restrictions...")
    print(f"  Initial sample size: {len(df):,}")

    # Working age (16-64)
    df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)]
    print(f"  After working age restriction (16-64): {len(df):,}")

    # Valid immigration year
    df = df[df['valid_yrimmig']]
    print(f"  After valid immigration year: {len(df):,}")

    # Exclude 2012
    df = df[df['YEAR'] != 2012]
    print(f"  After excluding 2012: {len(df):,}")

    return df


def compute_descriptive_stats(df):
    """
    Compute descriptive statistics by eligibility status and time period
    """
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)

    # Overall statistics
    print("\nOverall Sample Characteristics:")
    print(f"  Total observations: {len(df):,}")
    print(f"  DACA eligible: {df['eligible'].sum():,} ({100*df['eligible'].mean():.1f}%)")
    print(f"  Pre-period (2006-2011): {(df['post']==0).sum():,}")
    print(f"  Post-period (2013-2016): {(df['post']==1).sum():,}")

    # Statistics by group and period
    stats_dict = {}

    for eligible in [0, 1]:
        for post in [0, 1]:
            mask = (df['eligible'] == eligible) & (df['post'] == post)
            subset = df[mask]

            group_name = f"{'Eligible' if eligible else 'Ineligible'}_{'Post' if post else 'Pre'}"

            # Weighted means
            wt = subset['PERWT']

            stats_dict[group_name] = {
                'N': len(subset),
                'N_weighted': wt.sum(),
                'fulltime_rate': np.average(subset['fulltime'], weights=wt) if len(subset) > 0 else np.nan,
                'employed_rate': np.average(subset['employed'], weights=wt) if len(subset) > 0 else np.nan,
                'mean_age': np.average(subset['AGE'], weights=wt) if len(subset) > 0 else np.nan,
                'female_share': np.average(subset['female'], weights=wt) if len(subset) > 0 else np.nan,
                'married_share': np.average(subset['married'], weights=wt) if len(subset) > 0 else np.nan,
                'hs_or_more': np.average(subset['EDUC'] >= 6, weights=wt) if len(subset) > 0 else np.nan,
            }

    # Print statistics table
    print("\n" + "-"*80)
    print(f"{'Variable':<25} {'Inelig-Pre':>12} {'Inelig-Post':>12} {'Elig-Pre':>12} {'Elig-Post':>12}")
    print("-"*80)

    for var in ['N', 'fulltime_rate', 'employed_rate', 'mean_age', 'female_share', 'married_share', 'hs_or_more']:
        vals = [stats_dict[f'{e}_{p}'][var] for e, p in [('Ineligible', 'Pre'), ('Ineligible', 'Post'),
                                                          ('Eligible', 'Pre'), ('Eligible', 'Post')]]
        if var == 'N':
            print(f"{var:<25} {vals[0]:>12,} {vals[1]:>12,} {vals[2]:>12,} {vals[3]:>12,}")
        else:
            print(f"{var:<25} {vals[0]:>12.3f} {vals[1]:>12.3f} {vals[2]:>12.3f} {vals[3]:>12.3f}")
    print("-"*80)

    # Simple DiD calculation
    did_fulltime = ((stats_dict['Eligible_Post']['fulltime_rate'] - stats_dict['Eligible_Pre']['fulltime_rate']) -
                    (stats_dict['Ineligible_Post']['fulltime_rate'] - stats_dict['Ineligible_Pre']['fulltime_rate']))

    print(f"\nSimple DiD Estimate (Full-time Employment): {did_fulltime:.4f}")
    print(f"  Change for Eligible: {stats_dict['Eligible_Post']['fulltime_rate'] - stats_dict['Eligible_Pre']['fulltime_rate']:.4f}")
    print(f"  Change for Ineligible: {stats_dict['Ineligible_Post']['fulltime_rate'] - stats_dict['Ineligible_Pre']['fulltime_rate']:.4f}")

    return stats_dict, did_fulltime


def run_did_regression(df, add_controls=False, add_state_fe=False, add_year_fe=False):
    """
    Run difference-in-differences regression
    """
    # Base formula
    formula = "fulltime ~ eligible + post + eligible_post"

    # Add controls
    if add_controls:
        formula += " + AGE + age_sq + female + married + less_than_hs + some_college + college_plus"

    # Prepare data - drop missing values
    model_vars = ['fulltime', 'eligible', 'post', 'eligible_post', 'PERWT',
                  'AGE', 'age_sq', 'female', 'married', 'less_than_hs', 'some_college',
                  'college_plus', 'YEAR', 'STATEFIP']
    model_df = df[model_vars].dropna()

    # Add fixed effects as dummies if requested
    if add_year_fe:
        year_dummies = pd.get_dummies(model_df['YEAR'], prefix='year', drop_first=True)
        model_df = pd.concat([model_df, year_dummies], axis=1)
        formula += " + " + " + ".join(year_dummies.columns)

    if add_state_fe:
        state_dummies = pd.get_dummies(model_df['STATEFIP'], prefix='state', drop_first=True)
        model_df = pd.concat([model_df, state_dummies], axis=1)
        formula += " + " + " + ".join(state_dummies.columns)

    # Run weighted OLS
    model = smf.wls(formula, data=model_df, weights=model_df['PERWT'])
    results = model.fit(cov_type='HC1')  # Heteroskedasticity-robust standard errors

    return results, len(model_df)


def run_main_analysis(df):
    """
    Run all regression specifications
    """
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS")
    print("="*60)

    results_list = []

    # Specification 1: Basic DiD
    print("\nSpecification 1: Basic DiD (no controls)")
    res1, n1 = run_did_regression(df, add_controls=False, add_state_fe=False, add_year_fe=False)
    print(f"  DiD coefficient (eligible_post): {res1.params['eligible_post']:.4f}")
    print(f"  Standard error: {res1.bse['eligible_post']:.4f}")
    print(f"  t-statistic: {res1.tvalues['eligible_post']:.3f}")
    print(f"  p-value: {res1.pvalues['eligible_post']:.4f}")
    print(f"  N: {n1:,}")
    results_list.append(('Basic DiD', res1, n1))

    # Specification 2: DiD with demographic controls
    print("\nSpecification 2: DiD with demographic controls")
    res2, n2 = run_did_regression(df, add_controls=True, add_state_fe=False, add_year_fe=False)
    print(f"  DiD coefficient (eligible_post): {res2.params['eligible_post']:.4f}")
    print(f"  Standard error: {res2.bse['eligible_post']:.4f}")
    print(f"  t-statistic: {res2.tvalues['eligible_post']:.3f}")
    print(f"  p-value: {res2.pvalues['eligible_post']:.4f}")
    print(f"  N: {n2:,}")
    results_list.append(('With Controls', res2, n2))

    # Specification 3: DiD with controls + year FE
    print("\nSpecification 3: DiD with controls + year FE")
    res3, n3 = run_did_regression(df, add_controls=True, add_state_fe=False, add_year_fe=True)
    print(f"  DiD coefficient (eligible_post): {res3.params['eligible_post']:.4f}")
    print(f"  Standard error: {res3.bse['eligible_post']:.4f}")
    print(f"  t-statistic: {res3.tvalues['eligible_post']:.3f}")
    print(f"  p-value: {res3.pvalues['eligible_post']:.4f}")
    print(f"  N: {n3:,}")
    results_list.append(('Year FE', res3, n3))

    # Specification 4: DiD with controls + year FE + state FE (PREFERRED)
    print("\nSpecification 4: DiD with controls + year FE + state FE (PREFERRED)")
    res4, n4 = run_did_regression(df, add_controls=True, add_state_fe=True, add_year_fe=True)
    print(f"  DiD coefficient (eligible_post): {res4.params['eligible_post']:.4f}")
    print(f"  Standard error: {res4.bse['eligible_post']:.4f}")
    print(f"  95% CI: [{res4.conf_int().loc['eligible_post', 0]:.4f}, {res4.conf_int().loc['eligible_post', 1]:.4f}]")
    print(f"  t-statistic: {res4.tvalues['eligible_post']:.3f}")
    print(f"  p-value: {res4.pvalues['eligible_post']:.4f}")
    print(f"  N: {n4:,}")
    print(f"  R-squared: {res4.rsquared:.4f}")
    results_list.append(('Full Model (Preferred)', res4, n4))

    return results_list


def run_robustness_checks(df):
    """
    Run robustness checks
    """
    print("\n" + "="*60)
    print("ROBUSTNESS CHECKS")
    print("="*60)

    robustness_results = {}

    # 1. Alternative control group: Include naturalized citizens
    print("\n1. Alternative sample: Include naturalized citizens in control")
    df_alt1 = df.copy()
    df_alt1.loc[df_alt1['CITIZEN'] == 2, 'eligible'] = 0  # naturalized are ineligible
    df_alt1['eligible_post'] = df_alt1['eligible'] * df_alt1['post']
    res_alt1, n_alt1 = run_did_regression(df_alt1, add_controls=True, add_state_fe=True, add_year_fe=True)
    print(f"  DiD coefficient: {res_alt1.params['eligible_post']:.4f} (SE: {res_alt1.bse['eligible_post']:.4f})")
    robustness_results['alt_control'] = (res_alt1.params['eligible_post'], res_alt1.bse['eligible_post'], n_alt1)

    # 2. Restrict to ages 18-35 (most likely affected)
    print("\n2. Restricted age range: 18-35")
    df_alt2 = df[(df['AGE'] >= 18) & (df['AGE'] <= 35)].copy()
    res_alt2, n_alt2 = run_did_regression(df_alt2, add_controls=True, add_state_fe=True, add_year_fe=True)
    print(f"  DiD coefficient: {res_alt2.params['eligible_post']:.4f} (SE: {res_alt2.bse['eligible_post']:.4f})")
    print(f"  N: {n_alt2:,}")
    robustness_results['age_18_35'] = (res_alt2.params['eligible_post'], res_alt2.bse['eligible_post'], n_alt2)

    # 3. Males only
    print("\n3. Males only")
    df_male = df[df['female'] == 0].copy()
    res_male, n_male = run_did_regression(df_male, add_controls=True, add_state_fe=True, add_year_fe=True)
    print(f"  DiD coefficient: {res_male.params['eligible_post']:.4f} (SE: {res_male.bse['eligible_post']:.4f})")
    print(f"  N: {n_male:,}")
    robustness_results['males'] = (res_male.params['eligible_post'], res_male.bse['eligible_post'], n_male)

    # 4. Females only
    print("\n4. Females only")
    df_female = df[df['female'] == 1].copy()
    res_female, n_female = run_did_regression(df_female, add_controls=True, add_state_fe=True, add_year_fe=True)
    print(f"  DiD coefficient: {res_female.params['eligible_post']:.4f} (SE: {res_female.bse['eligible_post']:.4f})")
    print(f"  N: {n_female:,}")
    robustness_results['females'] = (res_female.params['eligible_post'], res_female.bse['eligible_post'], n_female)

    # 5. Employment (any hours > 0) as outcome
    print("\n5. Alternative outcome: Any employment (UHRSWORK > 0)")
    df_emp = df.copy()
    df_emp['any_work'] = (df_emp['UHRSWORK'] > 0).astype(int)
    formula = "any_work ~ eligible + post + eligible_post + AGE + age_sq + female + married + less_than_hs + some_college + college_plus"
    model_vars = ['any_work', 'eligible', 'post', 'eligible_post', 'PERWT',
                  'AGE', 'age_sq', 'female', 'married', 'less_than_hs', 'some_college',
                  'college_plus', 'YEAR', 'STATEFIP']
    model_df = df_emp[model_vars].dropna()
    year_dummies = pd.get_dummies(model_df['YEAR'], prefix='year', drop_first=True)
    state_dummies = pd.get_dummies(model_df['STATEFIP'], prefix='state', drop_first=True)
    model_df = pd.concat([model_df, year_dummies, state_dummies], axis=1)
    formula += " + " + " + ".join(year_dummies.columns) + " + " + " + ".join(state_dummies.columns)
    model = smf.wls(formula, data=model_df, weights=model_df['PERWT'])
    res_emp = model.fit(cov_type='HC1')
    print(f"  DiD coefficient: {res_emp.params['eligible_post']:.4f} (SE: {res_emp.bse['eligible_post']:.4f})")
    print(f"  N: {len(model_df):,}")
    robustness_results['any_employment'] = (res_emp.params['eligible_post'], res_emp.bse['eligible_post'], len(model_df))

    return robustness_results


def compute_event_study(df):
    """
    Compute year-by-year effects for event study
    """
    print("\n" + "="*60)
    print("EVENT STUDY ANALYSIS")
    print("="*60)

    # Create year interactions
    years = sorted(df['YEAR'].unique())
    base_year = 2011  # year before treatment

    # Create interaction terms
    for year in years:
        df[f'eligible_year_{year}'] = (df['eligible'] * (df['YEAR'] == year)).astype(int)

    # Build formula
    year_vars = [f'eligible_year_{y}' for y in years if y != base_year]
    formula = "fulltime ~ eligible + " + " + ".join([f'eligible_year_{y}' for y in years if y != base_year])
    formula += " + AGE + age_sq + female + married + less_than_hs + some_college + college_plus"

    # Add year and state FE
    model_vars = ['fulltime', 'eligible', 'PERWT', 'AGE', 'age_sq', 'female', 'married',
                  'less_than_hs', 'some_college', 'college_plus', 'YEAR', 'STATEFIP'] + year_vars
    model_df = df[model_vars].dropna()

    year_dummies = pd.get_dummies(model_df['YEAR'], prefix='year', drop_first=True)
    state_dummies = pd.get_dummies(model_df['STATEFIP'], prefix='state', drop_first=True)
    model_df = pd.concat([model_df, year_dummies, state_dummies], axis=1)
    formula += " + " + " + ".join(year_dummies.columns) + " + " + " + ".join(state_dummies.columns)

    model = smf.wls(formula, data=model_df, weights=model_df['PERWT'])
    results = model.fit(cov_type='HC1')

    # Extract coefficients
    event_study_results = {}
    for year in years:
        if year == base_year:
            event_study_results[year] = (0, 0)  # reference year
        else:
            var = f'eligible_year_{year}'
            event_study_results[year] = (results.params[var], results.bse[var])

    print(f"\nEvent Study Coefficients (base year = {base_year}):")
    print(f"{'Year':<8} {'Coefficient':>12} {'Std. Error':>12} {'95% CI':>25}")
    print("-"*60)
    for year in sorted(event_study_results.keys()):
        coef, se = event_study_results[year]
        ci_low, ci_high = coef - 1.96*se, coef + 1.96*se
        print(f"{year:<8} {coef:>12.4f} {se:>12.4f} [{ci_low:>10.4f}, {ci_high:>10.4f}]")

    return event_study_results


def save_results_for_latex(stats_dict, results_list, robustness_results, event_study_results, simple_did):
    """
    Save results to JSON for LaTeX report
    """
    # Extract main results
    preferred_result = results_list[-1][1]  # Full model is last
    preferred_n = results_list[-1][2]

    output = {
        'simple_did': simple_did,
        'descriptive_stats': {},
        'main_results': [],
        'robustness': {},
        'event_study': {},
        'preferred_estimate': {
            'coefficient': float(preferred_result.params['eligible_post']),
            'se': float(preferred_result.bse['eligible_post']),
            'ci_low': float(preferred_result.conf_int().loc['eligible_post', 0]),
            'ci_high': float(preferred_result.conf_int().loc['eligible_post', 1]),
            'pvalue': float(preferred_result.pvalues['eligible_post']),
            'n': int(preferred_n),
            'r2': float(preferred_result.rsquared)
        }
    }

    # Descriptive stats
    for key, val in stats_dict.items():
        output['descriptive_stats'][key] = {k: float(v) if not pd.isna(v) else None for k, v in val.items()}

    # Main results
    for name, res, n in results_list:
        output['main_results'].append({
            'name': name,
            'coefficient': float(res.params['eligible_post']),
            'se': float(res.bse['eligible_post']),
            'pvalue': float(res.pvalues['eligible_post']),
            'n': int(n),
            'r2': float(res.rsquared)
        })

    # Robustness
    for key, (coef, se, n) in robustness_results.items():
        output['robustness'][key] = {'coefficient': float(coef), 'se': float(se), 'n': int(n)}

    # Event study
    for year, (coef, se) in event_study_results.items():
        output['event_study'][int(year)] = {'coefficient': float(coef), 'se': float(se)}

    # Save
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results.json")

    return output


def main():
    """
    Main analysis pipeline
    """
    print("="*60)
    print("DACA REPLICATION STUDY - SESSION 73")
    print("Effect on Full-Time Employment")
    print("="*60)

    # Load and filter data
    data_path = os.path.join(OUTPUT_DIR, 'data', 'data.csv')
    df = load_and_filter_data(data_path)

    # Create eligibility indicator
    df = create_daca_eligibility(df)

    # Create analysis variables
    df = create_analysis_variables(df)

    # Apply sample restrictions
    df = restrict_sample(df)

    # Descriptive statistics
    stats_dict, simple_did = compute_descriptive_stats(df)

    # Main regression analysis
    results_list = run_main_analysis(df)

    # Robustness checks
    robustness_results = run_robustness_checks(df)

    # Event study
    event_study_results = compute_event_study(df)

    # Save results
    output = save_results_for_latex(stats_dict, results_list, robustness_results, event_study_results, simple_did)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF PREFERRED ESTIMATE")
    print("="*60)
    print(f"\nPreferred specification: DiD with demographic controls, year FE, and state FE")
    print(f"\nEffect of DACA eligibility on full-time employment:")
    print(f"  Coefficient: {output['preferred_estimate']['coefficient']:.4f}")
    print(f"  Standard Error: {output['preferred_estimate']['se']:.4f}")
    print(f"  95% Confidence Interval: [{output['preferred_estimate']['ci_low']:.4f}, {output['preferred_estimate']['ci_high']:.4f}]")
    print(f"  p-value: {output['preferred_estimate']['pvalue']:.4f}")
    print(f"  Sample Size: {output['preferred_estimate']['n']:,}")
    print(f"  R-squared: {output['preferred_estimate']['r2']:.4f}")

    # Interpretation
    coef = output['preferred_estimate']['coefficient']
    print(f"\nInterpretation: DACA eligibility is associated with a {abs(coef)*100:.2f} percentage point")
    if coef > 0:
        print("INCREASE in the probability of full-time employment.")
    else:
        print("DECREASE in the probability of full-time employment.")

    return df, output


if __name__ == "__main__":
    df, output = main()
