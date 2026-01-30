"""
DACA Replication Study: Effect on Full-Time Employment
Difference-in-Differences Analysis

This script analyzes the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born non-citizens in the United States.

Treatment Group: Ages 26-30 as of June 15, 2012 (born 1982-1986)
Control Group: Ages 31-35 as of June 15, 2012 (born 1977-1981)

Outcome: Full-time employment (usually working 35+ hours per week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'data/data.csv'
OUTPUT_DIR = '.'

# DACA implementation date reference: June 15, 2012
# Pre-period: 2006-2011 (up to 2011 to exclude ambiguous 2012)
# Post-period: 2013-2016 (as specified in instructions)

def load_and_filter_data():
    """
    Load the ACS data and filter to the analytical sample:
    - Hispanic-Mexican ethnicity (HISPAN == 1)
    - Born in Mexico (BPL == 200)
    - Non-citizen without papers (CITIZEN == 3, meaning not a citizen)
    - Ages relevant to treatment/control groups across all years
    """
    print("Loading data...")

    # Read in chunks due to large file size
    chunks = []
    chunk_size = 500000

    # Define columns we need
    cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
                   'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'MARST',
                   'EMPSTAT', 'UHRSWORK', 'STATEFIP', 'METRO']

    for chunk in pd.read_csv(DATA_PATH, usecols=cols_needed, chunksize=chunk_size,
                             dtype={'YEAR': 'int32', 'PERWT': 'float64',
                                   'SEX': 'int8', 'AGE': 'int16',
                                   'BIRTHQTR': 'int8', 'BIRTHYR': 'int16',
                                   'HISPAN': 'int8', 'BPL': 'int16',
                                   'CITIZEN': 'int8', 'YRIMMIG': 'int16',
                                   'EDUC': 'int8', 'MARST': 'int8',
                                   'EMPSTAT': 'int8', 'UHRSWORK': 'int8',
                                   'STATEFIP': 'int8', 'METRO': 'int8'}):
        # Filter to Hispanic-Mexican, born in Mexico, non-citizen
        mask = (
            (chunk['HISPAN'] == 1) &  # Mexican Hispanic
            (chunk['BPL'] == 200) &    # Born in Mexico
            (chunk['CITIZEN'] == 3)    # Not a citizen (undocumented proxy)
        )
        filtered = chunk[mask].copy()

        if len(filtered) > 0:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    print(f"After filtering to Hispanic-Mexican, Mexican-born, non-citizens: {len(df):,} observations")

    return df


def define_treatment_control(df):
    """
    Define treatment and control groups based on age at DACA implementation.

    DACA was implemented June 15, 2012. To be eligible, one needed to be under 31
    as of that date. Per instructions:
    - Treatment: Ages 26-30 on June 15, 2012 (just eligible by age)
    - Control: Ages 31-35 on June 15, 2012 (just ineligible due to age)

    We approximate using birth year:
    - Treatment: Born 1982-1986 (would be ~26-30 in mid-2012)
    - Control: Born 1977-1981 (would be ~31-35 in mid-2012)
    """
    # Define birth year cohorts
    # Treatment: born 1982-1986 (ages 26-30 as of June 2012)
    # Control: born 1977-1981 (ages 31-35 as of June 2012)

    df['treat'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)
    df['control'] = ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)).astype(int)

    # Keep only treatment and control cohorts
    df = df[(df['treat'] == 1) | (df['control'] == 1)].copy()

    print(f"After filtering to treatment/control birth cohorts: {len(df):,} observations")

    return df


def define_periods(df):
    """
    Define pre and post periods.
    - Exclude 2012 as it's ambiguous (DACA implemented mid-year)
    - Pre: 2006-2011
    - Post: 2013-2016
    """
    # Filter to analysis years
    df = df[(df['YEAR'] != 2012)].copy()

    # Define post period indicator
    df['post'] = (df['YEAR'] >= 2013).astype(int)

    print(f"After excluding 2012: {len(df):,} observations")
    print(f"Pre-period years: {sorted(df[df['post']==0]['YEAR'].unique())}")
    print(f"Post-period years: {sorted(df[df['post']==1]['YEAR'].unique())}")

    return df


def create_outcome_variable(df):
    """
    Create full-time employment outcome.
    Full-time = usually working 35+ hours per week (UHRSWORK >= 35)

    UHRSWORK values:
    - 0: N/A (not employed or not in labor force)
    - 1-98: Actual hours
    - 99: 99 or more hours
    """
    # Full-time employment: UHRSWORK >= 35
    # Note: Those not employed have UHRSWORK = 0
    df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

    # Also create an employed indicator
    df['employed'] = (df['EMPSTAT'] == 1).astype(int)  # 1 = Employed

    return df


def create_covariates(df):
    """
    Create additional covariates for regression analysis.
    """
    # Female indicator
    df['female'] = (df['SEX'] == 2).astype(int)

    # Married indicator (1 = married spouse present)
    df['married'] = (df['MARST'] == 1).astype(int)

    # Education categories
    # EDUC codes: 0-1 = less than HS, 2-5 = some HS, 6 = HS grad, 7-9 = some college, 10-11 = college+
    df['educ_lesshs'] = (df['EDUC'] <= 5).astype(int)
    df['educ_hs'] = (df['EDUC'] == 6).astype(int)
    df['educ_somecoll'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)
    df['educ_college'] = (df['EDUC'] >= 10).astype(int)

    # Age (continuous)
    df['age'] = df['AGE']
    df['age_sq'] = df['AGE'] ** 2

    # Metropolitan status
    df['metro'] = (df['METRO'] >= 2).astype(int)  # 2,3,4 = in metro area

    # Years in US
    df['yrs_in_us'] = df['YEAR'] - df['YRIMMIG']
    df.loc[df['YRIMMIG'] == 0, 'yrs_in_us'] = np.nan  # 0 means N/A

    return df


def descriptive_statistics(df):
    """
    Generate descriptive statistics by treatment/control and pre/post.
    """
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)

    # Sample sizes
    print("\nSample sizes by group and period:")
    print(df.groupby(['treat', 'post']).size().unstack())

    # Weighted sample sizes
    print("\nWeighted sample sizes:")
    print(df.groupby(['treat', 'post'])['PERWT'].sum().unstack().round(0))

    # Mean outcomes
    print("\nUnweighted mean full-time employment by group and period:")
    means = df.groupby(['treat', 'post'])['fulltime'].mean().unstack()
    print(means.round(4))

    # Calculate simple difference-in-differences
    if means.shape == (2, 2):
        dd_simple = (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])
        print(f"\nSimple DiD estimate (unweighted): {dd_simple:.4f}")

    # Weighted means
    print("\nWeighted mean full-time employment by group and period:")
    def weighted_mean(group):
        return np.average(group['fulltime'], weights=group['PERWT'])

    weighted_means = df.groupby(['treat', 'post']).apply(weighted_mean).unstack()
    print(weighted_means.round(4))

    if weighted_means.shape == (2, 2):
        dd_weighted = (weighted_means.loc[1, 1] - weighted_means.loc[1, 0]) - (weighted_means.loc[0, 1] - weighted_means.loc[0, 0])
        print(f"\nSimple DiD estimate (weighted): {dd_weighted:.4f}")

    # Covariates summary
    print("\nCovariate means by treatment status (pre-period):")
    pre_df = df[df['post'] == 0]
    covars = ['female', 'married', 'age', 'educ_lesshs', 'educ_hs', 'educ_somecoll',
              'educ_college', 'metro', 'employed']

    for var in covars:
        if var in pre_df.columns:
            treat_mean = pre_df[pre_df['treat']==1][var].mean()
            control_mean = pre_df[pre_df['treat']==0][var].mean()
            print(f"  {var:15s}: Treatment={treat_mean:.4f}, Control={control_mean:.4f}, Diff={treat_mean-control_mean:.4f}")

    return means, weighted_means


def run_did_regression(df, with_covariates=False, with_state_fe=False, with_year_fe=False):
    """
    Run difference-in-differences regression.

    Basic model: Y = beta0 + beta1*treat + beta2*post + beta3*(treat*post) + epsilon

    The coefficient of interest is beta3, the interaction term.
    """
    # Create interaction term
    df['treat_post'] = df['treat'] * df['post']

    # Build formula
    if with_covariates and with_year_fe and with_state_fe:
        formula = 'fulltime ~ treat + treat_post + female + married + age + age_sq + educ_hs + educ_somecoll + educ_college + metro + C(YEAR) + C(STATEFIP)'
    elif with_covariates and with_year_fe:
        formula = 'fulltime ~ treat + treat_post + female + married + age + age_sq + educ_hs + educ_somecoll + educ_college + metro + C(YEAR)'
    elif with_covariates:
        formula = 'fulltime ~ treat + post + treat_post + female + married + age + age_sq + educ_hs + educ_somecoll + educ_college + metro'
    elif with_year_fe:
        formula = 'fulltime ~ treat + treat_post + C(YEAR)'
    else:
        formula = 'fulltime ~ treat + post + treat_post'

    # Run weighted OLS
    model = smf.wls(formula, data=df, weights=df['PERWT'])
    results = model.fit(cov_type='HC1')  # Robust standard errors

    return results


def run_analysis_by_year(df):
    """
    Run year-by-year analysis to examine trends.
    """
    print("\n" + "="*80)
    print("YEAR-BY-YEAR FULL-TIME EMPLOYMENT RATES")
    print("="*80)

    results = []
    for year in sorted(df['YEAR'].unique()):
        year_df = df[df['YEAR'] == year]

        treat_rate = np.average(year_df[year_df['treat']==1]['fulltime'],
                                weights=year_df[year_df['treat']==1]['PERWT'])
        control_rate = np.average(year_df[year_df['treat']==0]['fulltime'],
                                  weights=year_df[year_df['treat']==0]['PERWT'])

        treat_n = len(year_df[year_df['treat']==1])
        control_n = len(year_df[year_df['treat']==0])

        results.append({
            'year': year,
            'treat_rate': treat_rate,
            'control_rate': control_rate,
            'diff': treat_rate - control_rate,
            'treat_n': treat_n,
            'control_n': control_n
        })

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    return results_df


def parallel_trends_test(df):
    """
    Test for parallel trends in pre-period.
    """
    print("\n" + "="*80)
    print("PARALLEL TRENDS TEST (PRE-PERIOD)")
    print("="*80)

    pre_df = df[df['post'] == 0].copy()

    # Create year trend
    pre_df['year_trend'] = pre_df['YEAR'] - 2006
    pre_df['treat_trend'] = pre_df['treat'] * pre_df['year_trend']

    # Run regression with interaction of treatment and time trend
    formula = 'fulltime ~ treat + year_trend + treat_trend'
    model = smf.wls(formula, data=pre_df, weights=pre_df['PERWT'])
    results = model.fit(cov_type='HC1')

    print("\nRegression testing differential pre-trends:")
    print(results.summary())

    treat_trend_coef = results.params['treat_trend']
    treat_trend_se = results.bse['treat_trend']
    treat_trend_p = results.pvalues['treat_trend']

    print(f"\nDifferential trend coefficient: {treat_trend_coef:.6f}")
    print(f"Standard error: {treat_trend_se:.6f}")
    print(f"P-value: {treat_trend_p:.6f}")

    if treat_trend_p > 0.05:
        print("Result: No statistically significant differential pre-trend (good for parallel trends assumption)")
    else:
        print("Warning: Statistically significant differential pre-trend detected")

    return results


def event_study(df):
    """
    Run event study to visualize treatment effects over time.
    """
    print("\n" + "="*80)
    print("EVENT STUDY ANALYSIS")
    print("="*80)

    df = df.copy()

    # Create year dummies (2011 as reference year, last pre-treatment year)
    years = sorted(df['YEAR'].unique())
    ref_year = 2011

    # Create interaction terms for each year (except reference)
    for year in years:
        if year != ref_year:
            df[f'treat_year_{year}'] = (df['treat'] == 1) & (df['YEAR'] == year)
            df[f'treat_year_{year}'] = df[f'treat_year_{year}'].astype(int)

    # Build formula
    year_vars = [f'treat_year_{y}' for y in years if y != ref_year]
    formula = 'fulltime ~ treat + C(YEAR) + ' + ' + '.join(year_vars)

    model = smf.wls(formula, data=df, weights=df['PERWT'])
    results = model.fit(cov_type='HC1')

    # Extract event study coefficients
    event_study_results = []
    for year in years:
        if year == ref_year:
            coef = 0
            se = 0
        else:
            var_name = f'treat_year_{year}'
            coef = results.params[var_name]
            se = results.bse[var_name]

        event_study_results.append({
            'year': year,
            'coef': coef,
            'se': se,
            'ci_lower': coef - 1.96 * se,
            'ci_upper': coef + 1.96 * se
        })

    event_df = pd.DataFrame(event_study_results)
    print("\nEvent study coefficients (reference year = 2011):")
    print(event_df.to_string(index=False))

    return event_df, results


def robustness_checks(df):
    """
    Run various robustness checks.
    """
    print("\n" + "="*80)
    print("ROBUSTNESS CHECKS")
    print("="*80)

    results_dict = {}

    # 1. Basic DiD
    print("\n1. Basic DiD (no covariates):")
    res1 = run_did_regression(df, with_covariates=False, with_state_fe=False, with_year_fe=False)
    print(f"   DiD coefficient: {res1.params['treat_post']:.4f} (SE: {res1.bse['treat_post']:.4f})")
    print(f"   P-value: {res1.pvalues['treat_post']:.4f}")
    results_dict['basic'] = res1

    # 2. With year fixed effects
    print("\n2. With year fixed effects:")
    res2 = run_did_regression(df, with_covariates=False, with_state_fe=False, with_year_fe=True)
    print(f"   DiD coefficient: {res2.params['treat_post']:.4f} (SE: {res2.bse['treat_post']:.4f})")
    print(f"   P-value: {res2.pvalues['treat_post']:.4f}")
    results_dict['year_fe'] = res2

    # 3. With covariates and year fixed effects
    print("\n3. With covariates and year fixed effects:")
    res3 = run_did_regression(df, with_covariates=True, with_state_fe=False, with_year_fe=True)
    print(f"   DiD coefficient: {res3.params['treat_post']:.4f} (SE: {res3.bse['treat_post']:.4f})")
    print(f"   P-value: {res3.pvalues['treat_post']:.4f}")
    results_dict['covariates_year_fe'] = res3

    # 4. With covariates, year FE, and state FE
    print("\n4. With covariates, year FE, and state FE:")
    res4 = run_did_regression(df, with_covariates=True, with_state_fe=True, with_year_fe=True)
    print(f"   DiD coefficient: {res4.params['treat_post']:.4f} (SE: {res4.bse['treat_post']:.4f})")
    print(f"   P-value: {res4.pvalues['treat_post']:.4f}")
    results_dict['full_model'] = res4

    # 5. By gender
    print("\n5. By gender:")
    for gender, name in [(1, 'Male'), (2, 'Female')]:
        sub_df = df[df['SEX'] == gender].copy()
        res = run_did_regression(sub_df, with_covariates=True, with_state_fe=False, with_year_fe=True)
        print(f"   {name}: DiD = {res.params['treat_post']:.4f} (SE: {res.bse['treat_post']:.4f}), p = {res.pvalues['treat_post']:.4f}")
        results_dict[f'gender_{name.lower()}'] = res

    # 6. Alternative treatment group (narrower age band)
    print("\n6. Narrower age bands (born 1983-1985 vs 1978-1980):")
    narrow_df = df[((df['BIRTHYR'] >= 1983) & (df['BIRTHYR'] <= 1985)) |
                   ((df['BIRTHYR'] >= 1978) & (df['BIRTHYR'] <= 1980))].copy()
    narrow_df['treat'] = ((narrow_df['BIRTHYR'] >= 1983) & (narrow_df['BIRTHYR'] <= 1985)).astype(int)
    narrow_df['treat_post'] = narrow_df['treat'] * narrow_df['post']
    res_narrow = run_did_regression(narrow_df, with_covariates=True, with_state_fe=False, with_year_fe=True)
    print(f"   DiD coefficient: {res_narrow.params['treat_post']:.4f} (SE: {res_narrow.bse['treat_post']:.4f})")
    print(f"   P-value: {res_narrow.pvalues['treat_post']:.4f}")
    results_dict['narrow_bands'] = res_narrow

    return results_dict


def save_results(df, means, weighted_means, year_results, event_df, regression_results):
    """
    Save key results to files.
    """
    # Save year-by-year results
    year_results.to_csv(f'{OUTPUT_DIR}/year_by_year_results.csv', index=False)

    # Save event study results
    event_df.to_csv(f'{OUTPUT_DIR}/event_study_results.csv', index=False)

    # Create summary table
    summary = []
    for name, res in regression_results.items():
        if 'treat_post' in res.params:
            summary.append({
                'model': name,
                'coefficient': res.params['treat_post'],
                'std_error': res.bse['treat_post'],
                'p_value': res.pvalues['treat_post'],
                'ci_lower': res.conf_int().loc['treat_post', 0],
                'ci_upper': res.conf_int().loc['treat_post', 1],
                'n_obs': int(res.nobs)
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{OUTPUT_DIR}/regression_summary.csv', index=False)

    print("\nResults saved to CSV files.")

    return summary_df


def main():
    """
    Main analysis pipeline.
    """
    print("="*80)
    print("DACA REPLICATION STUDY: EFFECT ON FULL-TIME EMPLOYMENT")
    print("Difference-in-Differences Analysis")
    print("="*80)

    # Load and prepare data
    df = load_and_filter_data()
    df = define_treatment_control(df)
    df = define_periods(df)
    df = create_outcome_variable(df)
    df = create_covariates(df)

    # Check for DACA eligibility additional criteria
    # DACA requires: arrived before 16th birthday, in US since June 2007
    # We cannot perfectly identify all criteria, but we can proxy
    print("\n" + "="*80)
    print("APPLYING ADDITIONAL DACA-LIKE FILTERS")
    print("="*80)

    # Filter: must have immigrated by a reasonable age (proxy for arrived before 16)
    # Birth year + 16 should be >= YRIMMIG for arrival before 16
    # But YRIMMIG = 0 means N/A, so we filter those out

    original_n = len(df)
    df = df[df['YRIMMIG'] > 0].copy()  # Valid immigration year
    print(f"After filtering to valid YRIMMIG: {len(df):,} (dropped {original_n - len(df):,})")

    # Check if arrived before age 16
    df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
    original_n = len(df)
    df = df[df['age_at_arrival'] < 16].copy()
    print(f"After filtering to arrived before age 16: {len(df):,} (dropped {original_n - len(df):,})")

    # Filter: must have been in US since at least 2007 (5 years before DACA)
    # This means YRIMMIG <= 2007
    original_n = len(df)
    df = df[df['YRIMMIG'] <= 2007].copy()
    print(f"After filtering to arrived by 2007: {len(df):,} (dropped {original_n - len(df):,})")

    print(f"\nFinal analytical sample: {len(df):,} person-year observations")
    print(f"Unique years: {sorted(df['YEAR'].unique())}")

    # Run analyses
    means, weighted_means = descriptive_statistics(df)
    year_results = run_analysis_by_year(df)
    parallel_results = parallel_trends_test(df)
    event_df, event_results = event_study(df)
    regression_results = robustness_checks(df)

    # Save results
    summary_df = save_results(df, means, weighted_means, year_results, event_df, regression_results)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("\nRegression Results Summary:")
    print(summary_df.to_string(index=False))

    # Preferred estimate
    preferred = regression_results['covariates_year_fe']
    print("\n" + "="*80)
    print("PREFERRED ESTIMATE")
    print("="*80)
    print(f"Model: DiD with covariates and year fixed effects")
    print(f"Effect size: {preferred.params['treat_post']:.4f}")
    print(f"Standard error: {preferred.bse['treat_post']:.4f}")
    print(f"95% CI: [{preferred.conf_int().loc['treat_post', 0]:.4f}, {preferred.conf_int().loc['treat_post', 1]:.4f}]")
    print(f"P-value: {preferred.pvalues['treat_post']:.4f}")
    print(f"Sample size: {int(preferred.nobs):,}")

    # Print full preferred model results
    print("\n" + "="*80)
    print("FULL PREFERRED MODEL OUTPUT")
    print("="*80)
    print(preferred.summary())

    return df, regression_results, summary_df


if __name__ == "__main__":
    df, results, summary = main()
