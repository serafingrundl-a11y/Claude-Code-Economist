"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States.
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
CHUNK_SIZE = 500000

def load_and_filter_data():
    """Load data in chunks and filter to relevant sample."""
    print("Loading and filtering data...")

    # Columns we need
    cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR',
                   'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'UHRSWORK',
                   'SEX', 'EDUC', 'MARST', 'EMPSTAT']

    chunks = []
    total_rows = 0
    filtered_rows = 0

    for chunk in pd.read_csv(DATA_PATH, usecols=cols_needed, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

        # Filter to target population:
        # 1. Hispanic-Mexican (HISPAN == 1)
        # 2. Born in Mexico (BPL == 200)
        # 3. Non-citizen (CITIZEN == 3)
        # 4. Working age (16-45)
        # 5. Exclude 2012 (implementation year)

        mask = (
            (chunk['HISPAN'] == 1) &  # Mexican Hispanic
            (chunk['BPL'] == 200) &    # Born in Mexico
            (chunk['CITIZEN'] == 3) &  # Not a citizen
            (chunk['AGE'] >= 16) &     # Working age
            (chunk['AGE'] <= 45) &     # Upper bound for relevant ages
            (chunk['YEAR'] != 2012)    # Exclude implementation year
        )

        filtered_chunk = chunk[mask].copy()
        filtered_rows += len(filtered_chunk)
        chunks.append(filtered_chunk)

        print(f"  Processed {total_rows:,} rows, kept {filtered_rows:,}")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal sample: {len(df):,} observations")
    return df


def create_variables(df):
    """Create treatment indicators and outcome variable."""
    print("\nCreating analysis variables...")

    # Full-time employment outcome (35+ hours per week)
    df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

    # Employed indicator
    df['employed'] = (df['EMPSTAT'] == 1).astype(int)

    # Post-DACA indicator (2013-2016)
    df['post'] = (df['YEAR'] >= 2013).astype(int)

    # Age in 2012 (for eligibility determination)
    # Use BIRTHYR to calculate age on June 15, 2012
    df['age_2012'] = 2012 - df['BIRTHYR']

    # Adjust for birth quarter (June 15 is in Q2)
    # If born in Q3 or Q4, they haven't had their birthday by June 15
    df.loc[df['BIRTHQTR'] >= 3, 'age_2012'] = df.loc[df['BIRTHQTR'] >= 3, 'age_2012'] - 1

    # Age at arrival in US
    df['age_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

    # DACA eligibility criteria:
    # 1. Under 31 on June 15, 2012 (age_2012 < 31)
    # 2. Arrived before 16th birthday (age_arrival < 16)
    # 3. Present in US since June 2007 (YRIMMIG <= 2007)
    # 4. Not a citizen (already filtered)

    # Treatment: DACA-eligible
    df['daca_eligible'] = (
        (df['age_2012'] < 31) &           # Under 31 on June 15, 2012
        (df['age_arrival'] < 16) &         # Arrived before 16
        (df['YRIMMIG'] <= 2007) &          # In US since at least 2007
        (df['YRIMMIG'] > 0)                # Valid immigration year
    ).astype(int)

    # Control group: Similar but too old (31-35 on June 15, 2012)
    # who also arrived before 16 and by 2007
    df['control_group'] = (
        (df['age_2012'] >= 31) &
        (df['age_2012'] <= 35) &
        (df['age_arrival'] < 16) &
        (df['YRIMMIG'] <= 2007) &
        (df['YRIMMIG'] > 0)
    ).astype(int)

    # DID interaction term
    df['treat_post'] = df['daca_eligible'] * df['post']

    # Create analysis sample (treatment or control)
    df['in_analysis'] = ((df['daca_eligible'] == 1) | (df['control_group'] == 1)).astype(int)

    # Demographic controls
    df['female'] = (df['SEX'] == 2).astype(int)
    df['married'] = (df['MARST'] <= 2).astype(int)
    df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # High school or more
    df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # Some college or more

    return df


def summary_statistics(df):
    """Generate summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Full sample stats
    print(f"\nFull filtered sample: {len(df):,} observations")
    print(f"  Years: {df['YEAR'].min()} - {df['YEAR'].max()}")
    print(f"  Age range: {df['AGE'].min()} - {df['AGE'].max()}")

    # Analysis sample
    analysis_df = df[df['in_analysis'] == 1].copy()
    print(f"\nAnalysis sample: {len(analysis_df):,} observations")

    treat_df = analysis_df[analysis_df['daca_eligible'] == 1]
    control_df = analysis_df[analysis_df['control_group'] == 1]

    print(f"  Treatment (DACA-eligible): {len(treat_df):,}")
    print(f"  Control (too old): {len(control_df):,}")

    # Pre/post breakdown
    pre_treat = treat_df[treat_df['post'] == 0]
    post_treat = treat_df[treat_df['post'] == 1]
    pre_control = control_df[control_df['post'] == 0]
    post_control = control_df[control_df['post'] == 1]

    print(f"\n  Treatment - Pre-DACA: {len(pre_treat):,}")
    print(f"  Treatment - Post-DACA: {len(post_treat):,}")
    print(f"  Control - Pre-DACA: {len(pre_control):,}")
    print(f"  Control - Post-DACA: {len(post_control):,}")

    # Full-time employment rates
    print("\nFull-time Employment Rates:")

    def weighted_mean(data, col, weight_col):
        return np.average(data[col], weights=data[weight_col])

    ft_pre_treat = weighted_mean(pre_treat, 'fulltime', 'PERWT')
    ft_post_treat = weighted_mean(post_treat, 'fulltime', 'PERWT')
    ft_pre_control = weighted_mean(pre_control, 'fulltime', 'PERWT')
    ft_post_control = weighted_mean(post_control, 'fulltime', 'PERWT')

    print(f"  Treatment - Pre:  {ft_pre_treat:.3f}")
    print(f"  Treatment - Post: {ft_post_treat:.3f}")
    print(f"  Control - Pre:    {ft_pre_control:.3f}")
    print(f"  Control - Post:   {ft_post_control:.3f}")

    # Simple DID estimate
    did_simple = (ft_post_treat - ft_pre_treat) - (ft_post_control - ft_pre_control)
    print(f"\nSimple DID estimate: {did_simple:.4f}")

    # Demographics
    print("\nDemographics (Analysis Sample):")
    print(f"  Mean age: {analysis_df['AGE'].mean():.1f}")
    print(f"  Female: {weighted_mean(analysis_df, 'female', 'PERWT'):.1%}")
    print(f"  Married: {weighted_mean(analysis_df, 'married', 'PERWT'):.1%}")
    print(f"  High school+: {weighted_mean(analysis_df, 'educ_hs', 'PERWT'):.1%}")

    return {
        'n_total': len(analysis_df),
        'n_treat': len(treat_df),
        'n_control': len(control_df),
        'ft_pre_treat': ft_pre_treat,
        'ft_post_treat': ft_post_treat,
        'ft_pre_control': ft_pre_control,
        'ft_post_control': ft_post_control,
        'did_simple': did_simple
    }


def run_did_regression(df, outcome='fulltime'):
    """Run difference-in-differences regression."""
    print("\n" + "="*60)
    print(f"DID REGRESSION - Outcome: {outcome}")
    print("="*60)

    analysis_df = df[df['in_analysis'] == 1].copy()

    # Model 1: Basic DID
    print("\nModel 1: Basic DID (no controls)")
    model1 = smf.wls(
        f'{outcome} ~ daca_eligible + post + treat_post',
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})
    print(model1.summary().tables[1])

    # Model 2: With demographic controls
    print("\nModel 2: With demographic controls")
    model2 = smf.wls(
        f'{outcome} ~ daca_eligible + post + treat_post + female + married + educ_hs + AGE + I(AGE**2)',
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})
    print(model2.summary().tables[1])

    # Model 3: With state fixed effects
    print("\nModel 3: With state fixed effects")
    analysis_df['state_fe'] = pd.Categorical(analysis_df['STATEFIP'])
    model3 = smf.wls(
        f'{outcome} ~ daca_eligible + post + treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP)',
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})

    # Extract key coefficient
    coef = model3.params['treat_post']
    se = model3.bse['treat_post']
    pval = model3.pvalues['treat_post']
    ci_low, ci_high = model3.conf_int().loc['treat_post']

    print(f"\nKey Result (Model 3 - treat_post):")
    print(f"  Coefficient: {coef:.4f}")
    print(f"  Std. Error:  {se:.4f}")
    print(f"  t-statistic: {coef/se:.3f}")
    print(f"  p-value:     {pval:.4f}")
    print(f"  95% CI:      [{ci_low:.4f}, {ci_high:.4f}]")

    # Model 4: With year fixed effects
    print("\nModel 4: With state and year fixed effects (preferred)")
    model4 = smf.wls(
        f'{outcome} ~ daca_eligible + treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})

    coef4 = model4.params['treat_post']
    se4 = model4.bse['treat_post']
    pval4 = model4.pvalues['treat_post']
    ci_low4, ci_high4 = model4.conf_int().loc['treat_post']

    print(f"\nKey Result (Model 4 - treat_post - PREFERRED):")
    print(f"  Coefficient: {coef4:.4f}")
    print(f"  Std. Error:  {se4:.4f}")
    print(f"  t-statistic: {coef4/se4:.3f}")
    print(f"  p-value:     {pval4:.4f}")
    print(f"  95% CI:      [{ci_low4:.4f}, {ci_high4:.4f}]")
    print(f"  N:           {int(model4.nobs):,}")

    return {
        'model1': model1,
        'model2': model2,
        'model3': model3,
        'model4': model4,
        'preferred_coef': coef4,
        'preferred_se': se4,
        'preferred_pval': pval4,
        'preferred_ci': (ci_low4, ci_high4),
        'n_obs': int(model4.nobs)
    }


def run_robustness_checks(df):
    """Run additional robustness checks."""
    print("\n" + "="*60)
    print("ROBUSTNESS CHECKS")
    print("="*60)

    analysis_df = df[df['in_analysis'] == 1].copy()

    results = {}

    # 1. Alternative age bandwidth for control group
    print("\n1. Alternative control group (ages 31-40 in 2012)")
    df_alt = df.copy()
    df_alt['control_alt'] = (
        (df_alt['age_2012'] >= 31) &
        (df_alt['age_2012'] <= 40) &
        (df_alt['age_arrival'] < 16) &
        (df_alt['YRIMMIG'] <= 2007) &
        (df_alt['YRIMMIG'] > 0)
    ).astype(int)
    df_alt['in_analysis_alt'] = ((df_alt['daca_eligible'] == 1) | (df_alt['control_alt'] == 1)).astype(int)
    analysis_alt = df_alt[df_alt['in_analysis_alt'] == 1].copy()

    model_alt = smf.wls(
        'fulltime ~ daca_eligible + treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=analysis_alt,
        weights=analysis_alt['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_alt['STATEFIP']})

    print(f"  Coefficient: {model_alt.params['treat_post']:.4f}")
    print(f"  Std. Error:  {model_alt.bse['treat_post']:.4f}")
    print(f"  N:           {int(model_alt.nobs):,}")
    results['alt_control'] = model_alt

    # 2. Males only
    print("\n2. Males only")
    males = analysis_df[analysis_df['female'] == 0]
    model_males = smf.wls(
        'fulltime ~ daca_eligible + treat_post + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=males,
        weights=males['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': males['STATEFIP']})

    print(f"  Coefficient: {model_males.params['treat_post']:.4f}")
    print(f"  Std. Error:  {model_males.bse['treat_post']:.4f}")
    print(f"  N:           {int(model_males.nobs):,}")
    results['males'] = model_males

    # 3. Females only
    print("\n3. Females only")
    females = analysis_df[analysis_df['female'] == 1]
    model_females = smf.wls(
        'fulltime ~ daca_eligible + treat_post + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=females,
        weights=females['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': females['STATEFIP']})

    print(f"  Coefficient: {model_females.params['treat_post']:.4f}")
    print(f"  Std. Error:  {model_females.bse['treat_post']:.4f}")
    print(f"  N:           {int(model_females.nobs):,}")
    results['females'] = model_females

    # 4. Employment (extensive margin)
    print("\n4. Employment outcome (extensive margin)")
    model_emp = smf.wls(
        'employed ~ daca_eligible + treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})

    print(f"  Coefficient: {model_emp.params['treat_post']:.4f}")
    print(f"  Std. Error:  {model_emp.bse['treat_post']:.4f}")
    print(f"  N:           {int(model_emp.nobs):,}")
    results['employment'] = model_emp

    # 5. Placebo test: pre-trends (2006-2008 vs 2009-2011)
    print("\n5. Placebo test (pre-period only: 2009-2011 as 'post')")
    pre_df = analysis_df[analysis_df['YEAR'] <= 2011].copy()
    pre_df['placebo_post'] = (pre_df['YEAR'] >= 2009).astype(int)
    pre_df['placebo_treat_post'] = pre_df['daca_eligible'] * pre_df['placebo_post']

    model_placebo = smf.wls(
        'fulltime ~ daca_eligible + placebo_post + placebo_treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=pre_df,
        weights=pre_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': pre_df['STATEFIP']})

    print(f"  Placebo Coefficient: {model_placebo.params['placebo_treat_post']:.4f}")
    print(f"  Std. Error:          {model_placebo.bse['placebo_treat_post']:.4f}")
    print(f"  p-value:             {model_placebo.pvalues['placebo_treat_post']:.4f}")
    results['placebo'] = model_placebo

    return results


def event_study(df):
    """Run event study analysis."""
    print("\n" + "="*60)
    print("EVENT STUDY ANALYSIS")
    print("="*60)

    analysis_df = df[df['in_analysis'] == 1].copy()

    # Create year dummies interacted with treatment
    years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
    for year in years:
        analysis_df[f'year_{year}'] = (analysis_df['YEAR'] == year).astype(int)
        analysis_df[f'treat_year_{year}'] = analysis_df['daca_eligible'] * analysis_df[f'year_{year}']

    # Reference year is 2011 (last pre-treatment year)
    treat_year_vars = ' + '.join([f'treat_year_{y}' for y in years if y != 2011])
    year_vars = ' + '.join([f'year_{y}' for y in years if y != 2011])

    formula = f'fulltime ~ daca_eligible + {year_vars} + {treat_year_vars} + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP)'

    model_es = smf.wls(
        formula,
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})

    print("\nEvent Study Coefficients (relative to 2011):")
    print("-" * 45)

    es_results = []
    for year in years:
        if year != 2011:
            coef = model_es.params[f'treat_year_{year}']
            se = model_es.bse[f'treat_year_{year}']
            ci_low, ci_high = model_es.conf_int().loc[f'treat_year_{year}']
            print(f"  {year}: {coef:7.4f} (SE: {se:.4f}) [{ci_low:.4f}, {ci_high:.4f}]")
            es_results.append({'year': year, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})
        else:
            print(f"  {year}: 0.0000 (reference)")
            es_results.append({'year': year, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})

    return pd.DataFrame(es_results)


def main():
    """Main analysis function."""
    print("="*60)
    print("DACA REPLICATION STUDY - ANALYSIS")
    print("="*60)

    # Load and prepare data
    df = load_and_filter_data()
    df = create_variables(df)

    # Summary statistics
    summary_stats = summary_statistics(df)

    # Main DID regressions
    did_results = run_did_regression(df, outcome='fulltime')

    # Robustness checks
    robustness = run_robustness_checks(df)

    # Event study
    es_results = event_study(df)

    # Save results
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    print(f"\nPreferred Estimate (Model 4 - State and Year FE):")
    print(f"  Effect of DACA eligibility on full-time employment:")
    print(f"  Coefficient: {did_results['preferred_coef']:.4f}")
    print(f"  Standard Error: {did_results['preferred_se']:.4f}")
    print(f"  95% CI: [{did_results['preferred_ci'][0]:.4f}, {did_results['preferred_ci'][1]:.4f}]")
    print(f"  p-value: {did_results['preferred_pval']:.4f}")
    print(f"  Sample Size: {did_results['n_obs']:,}")

    # Save key results for report
    results_dict = {
        'summary': summary_stats,
        'did': did_results,
        'robustness': robustness,
        'event_study': es_results
    }

    return results_dict


if __name__ == "__main__":
    results = main()
