"""
DACA Replication Study - Analysis Script (Version 2)
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States.

Refined approach using narrow age bandwidth around the eligibility cutoff.
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
        # 4. Working age (18-45)
        # 5. Exclude 2012 (implementation year)

        mask = (
            (chunk['HISPAN'] == 1) &  # Mexican Hispanic
            (chunk['BPL'] == 200) &    # Born in Mexico
            (chunk['CITIZEN'] == 3) &  # Not a citizen
            (chunk['AGE'] >= 18) &     # Working age (adults)
            (chunk['AGE'] <= 45) &     # Upper bound
            (chunk['YEAR'] != 2012)    # Exclude implementation year
        )

        filtered_chunk = chunk[mask].copy()
        filtered_rows += len(filtered_chunk)
        chunks.append(filtered_chunk)

        if total_rows % 5000000 == 0:
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

    # Age on June 15, 2012
    df['age_2012'] = 2012 - df['BIRTHYR']
    # Adjust for birth quarter (if born Q3/Q4, haven't had birthday by June 15)
    df.loc[df['BIRTHQTR'] >= 3, 'age_2012'] = df.loc[df['BIRTHQTR'] >= 3, 'age_2012'] - 1

    # Age at arrival in US
    df['age_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

    # DACA eligibility criteria:
    # 1. Under 31 on June 15, 2012
    # 2. Arrived before 16th birthday
    # 3. Present in US since June 2007 (proxy: arrived by 2007)
    # 4. Not a citizen (already filtered)

    # Core eligibility based on arrival (arrived young, present since 2007)
    df['arrived_eligible'] = (
        (df['age_arrival'] < 16) &
        (df['YRIMMIG'] <= 2007) &
        (df['YRIMMIG'] > 0)
    ).astype(int)

    # Treatment: DACA-eligible = under 31 AND arrived eligible
    df['daca_eligible'] = (
        (df['age_2012'] < 31) &
        (df['arrived_eligible'] == 1)
    ).astype(int)

    # For a cleaner DID, use those who would be eligible based on arrival
    # but are just above the age cutoff as control
    # Control: ages 31-35 on June 15, 2012, who also arrived young and by 2007
    df['control_group'] = (
        (df['age_2012'] >= 31) &
        (df['age_2012'] <= 35) &
        (df['arrived_eligible'] == 1)
    ).astype(int)

    # DID treatment indicator
    df['treat_post'] = df['daca_eligible'] * df['post']

    # Analysis sample: those who meet arrival requirements
    df['in_analysis'] = df['arrived_eligible']

    # Demographic controls
    df['female'] = (df['SEX'] == 2).astype(int)
    df['married'] = (df['MARST'] <= 2).astype(int)
    df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # HS or more
    df['educ_college'] = (df['EDUC'] >= 10).astype(int)

    return df


def summary_statistics(df):
    """Generate summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    analysis_df = df[df['in_analysis'] == 1].copy()

    print(f"\nAnalysis sample (arrived eligible): {len(analysis_df):,} observations")
    print(f"  Years: {analysis_df['YEAR'].min()} - {analysis_df['YEAR'].max()}")

    treat_df = analysis_df[analysis_df['daca_eligible'] == 1]
    control_df = analysis_df[analysis_df['daca_eligible'] == 0]

    print(f"\nDACA eligible (age < 31 in 2012): {len(treat_df):,}")
    print(f"Not DACA eligible (age >= 31 in 2012): {len(control_df):,}")

    # Narrow control (31-35)
    narrow_control = analysis_df[analysis_df['control_group'] == 1]
    print(f"Narrow control (age 31-35 in 2012): {len(narrow_control):,}")

    # Pre/post breakdown
    def get_counts_rates(data, label):
        pre = data[data['post'] == 0]
        post_data = data[data['post'] == 1]

        def wmean(d, col):
            if len(d) == 0:
                return np.nan
            return np.average(d[col], weights=d['PERWT'])

        ft_pre = wmean(pre, 'fulltime')
        ft_post = wmean(post_data, 'fulltime')

        return {
            'n_pre': len(pre),
            'n_post': len(post_data),
            'ft_pre': ft_pre,
            'ft_post': ft_post,
            'label': label
        }

    treat_stats = get_counts_rates(treat_df, 'Treatment')
    control_stats = get_counts_rates(narrow_control, 'Control (31-35)')

    print(f"\n{'Group':<20} {'N (Pre)':<12} {'N (Post)':<12} {'FT% (Pre)':<12} {'FT% (Post)':<12}")
    print("-" * 68)
    for s in [treat_stats, control_stats]:
        print(f"{s['label']:<20} {s['n_pre']:<12,} {s['n_post']:<12,} {s['ft_pre']:<12.3f} {s['ft_post']:<12.3f}")

    # Simple DID
    did = (treat_stats['ft_post'] - treat_stats['ft_pre']) - (control_stats['ft_post'] - control_stats['ft_pre'])
    print(f"\nSimple DID estimate: {did:.4f}")

    # Demographics comparison
    print("\nDemographics by Treatment Status:")

    def demo_stats(data):
        return {
            'age': data['AGE'].mean(),
            'female': np.average(data['female'], weights=data['PERWT']),
            'married': np.average(data['married'], weights=data['PERWT']),
            'hs': np.average(data['educ_hs'], weights=data['PERWT'])
        }

    treat_demo = demo_stats(treat_df)
    control_demo = demo_stats(narrow_control)

    print(f"{'Characteristic':<15} {'Treatment':<15} {'Control (31-35)':<15}")
    print("-" * 45)
    print(f"{'Mean Age':<15} {treat_demo['age']:<15.1f} {control_demo['age']:<15.1f}")
    print(f"{'% Female':<15} {treat_demo['female']*100:<15.1f} {control_demo['female']*100:<15.1f}")
    print(f"{'% Married':<15} {treat_demo['married']*100:<15.1f} {control_demo['married']*100:<15.1f}")
    print(f"{'% HS+':<15} {treat_demo['hs']*100:<15.1f} {control_demo['hs']*100:<15.1f}")

    return {
        'n_analysis': len(analysis_df),
        'n_treat': len(treat_df),
        'n_control': len(narrow_control),
        'treat_stats': treat_stats,
        'control_stats': control_stats,
        'did_simple': did
    }


def run_did_regression(df):
    """Run difference-in-differences regression using narrow bandwidth."""
    print("\n" + "="*60)
    print("MAIN DID REGRESSION ANALYSIS")
    print("="*60)

    # Use sample with narrow age bandwidth: treatment (age 26-30 in 2012) vs control (31-35 in 2012)
    analysis_df = df[(df['in_analysis'] == 1) &
                     (((df['age_2012'] >= 26) & (df['age_2012'] <= 30)) |  # Treatment near cutoff
                      ((df['age_2012'] >= 31) & (df['age_2012'] <= 35)))   # Control
                    ].copy()

    # Re-define treatment for this narrower sample
    analysis_df['treat'] = (analysis_df['age_2012'] < 31).astype(int)
    analysis_df['treat_post'] = analysis_df['treat'] * analysis_df['post']

    print(f"\nNarrow bandwidth sample (ages 26-35 in 2012): {len(analysis_df):,}")
    print(f"  Treatment (26-30): {(analysis_df['treat']==1).sum():,}")
    print(f"  Control (31-35): {(analysis_df['treat']==0).sum():,}")

    results = {}

    # Model 1: Basic DID
    print("\nModel 1: Basic DID")
    model1 = smf.wls(
        'fulltime ~ treat + post + treat_post',
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})
    print(model1.summary().tables[1])
    results['model1'] = model1

    # Model 2: With controls
    print("\nModel 2: With demographic controls")
    model2 = smf.wls(
        'fulltime ~ treat + post + treat_post + female + married + educ_hs + AGE + I(AGE**2)',
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})
    print(model2.summary().tables[1])
    results['model2'] = model2

    # Model 3: State FE
    print("\nModel 3: With state fixed effects")
    model3 = smf.wls(
        'fulltime ~ treat + post + treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP)',
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})

    coef3 = model3.params['treat_post']
    se3 = model3.bse['treat_post']
    print(f"  treat_post coefficient: {coef3:.4f} (SE: {se3:.4f})")
    results['model3'] = model3

    # Model 4: State + Year FE (preferred)
    print("\nModel 4: State + Year FE (PREFERRED)")
    model4 = smf.wls(
        'fulltime ~ treat + treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})

    coef4 = model4.params['treat_post']
    se4 = model4.bse['treat_post']
    pval4 = model4.pvalues['treat_post']
    ci4 = model4.conf_int().loc['treat_post']

    print(f"\n*** PREFERRED ESTIMATE ***")
    print(f"  Coefficient: {coef4:.4f}")
    print(f"  Std. Error:  {se4:.4f}")
    print(f"  t-statistic: {coef4/se4:.3f}")
    print(f"  p-value:     {pval4:.4f}")
    print(f"  95% CI:      [{ci4[0]:.4f}, {ci4[1]:.4f}]")
    print(f"  N:           {int(model4.nobs):,}")

    results['model4'] = model4
    results['preferred'] = {
        'coef': coef4,
        'se': se4,
        'pval': pval4,
        'ci': (ci4[0], ci4[1]),
        'n': int(model4.nobs)
    }

    return results, analysis_df


def robustness_checks(df, analysis_df):
    """Robustness checks."""
    print("\n" + "="*60)
    print("ROBUSTNESS CHECKS")
    print("="*60)

    results = {}

    # 1. Wider bandwidth (ages 22-38)
    print("\n1. Wider bandwidth (ages 22-38 in 2012)")
    wide_df = df[(df['in_analysis'] == 1) &
                 (df['age_2012'] >= 22) & (df['age_2012'] <= 38)].copy()
    wide_df['treat'] = (wide_df['age_2012'] < 31).astype(int)
    wide_df['treat_post'] = wide_df['treat'] * wide_df['post']

    model_wide = smf.wls(
        'fulltime ~ treat + treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=wide_df,
        weights=wide_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': wide_df['STATEFIP']})
    print(f"  Coefficient: {model_wide.params['treat_post']:.4f} (SE: {model_wide.bse['treat_post']:.4f}), N={int(model_wide.nobs):,}")
    results['wide_bandwidth'] = model_wide

    # 2. Narrower bandwidth (ages 28-33)
    print("\n2. Narrower bandwidth (ages 28-33 in 2012)")
    narrow_df = df[(df['in_analysis'] == 1) &
                   (df['age_2012'] >= 28) & (df['age_2012'] <= 33)].copy()
    narrow_df['treat'] = (narrow_df['age_2012'] < 31).astype(int)
    narrow_df['treat_post'] = narrow_df['treat'] * narrow_df['post']

    model_narrow = smf.wls(
        'fulltime ~ treat + treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=narrow_df,
        weights=narrow_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': narrow_df['STATEFIP']})
    print(f"  Coefficient: {model_narrow.params['treat_post']:.4f} (SE: {model_narrow.bse['treat_post']:.4f}), N={int(model_narrow.nobs):,}")
    results['narrow_bandwidth'] = model_narrow

    # 3. Males only
    print("\n3. Males only")
    males_df = analysis_df[analysis_df['female'] == 0].copy()
    model_males = smf.wls(
        'fulltime ~ treat + treat_post + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=males_df,
        weights=males_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': males_df['STATEFIP']})
    print(f"  Coefficient: {model_males.params['treat_post']:.4f} (SE: {model_males.bse['treat_post']:.4f}), N={int(model_males.nobs):,}")
    results['males'] = model_males

    # 4. Females only
    print("\n4. Females only")
    females_df = analysis_df[analysis_df['female'] == 1].copy()
    model_females = smf.wls(
        'fulltime ~ treat + treat_post + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=females_df,
        weights=females_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': females_df['STATEFIP']})
    print(f"  Coefficient: {model_females.params['treat_post']:.4f} (SE: {model_females.bse['treat_post']:.4f}), N={int(model_females.nobs):,}")
    results['females'] = model_females

    # 5. Employment (extensive margin)
    print("\n5. Employment (extensive margin)")
    model_emp = smf.wls(
        'employed ~ treat + treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})
    print(f"  Coefficient: {model_emp.params['treat_post']:.4f} (SE: {model_emp.bse['treat_post']:.4f}), N={int(model_emp.nobs):,}")
    results['employment'] = model_emp

    # 6. Placebo test
    print("\n6. Placebo test (pre-period: 2009-2011 as 'post')")
    pre_df = analysis_df[analysis_df['YEAR'] <= 2011].copy()
    pre_df['placebo_post'] = (pre_df['YEAR'] >= 2009).astype(int)
    pre_df['placebo_treat_post'] = pre_df['treat'] * pre_df['placebo_post']

    model_placebo = smf.wls(
        'fulltime ~ treat + placebo_post + placebo_treat_post + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP) + C(YEAR)',
        data=pre_df,
        weights=pre_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': pre_df['STATEFIP']})
    print(f"  Placebo coefficient: {model_placebo.params['placebo_treat_post']:.4f} (SE: {model_placebo.bse['placebo_treat_post']:.4f})")
    print(f"  p-value: {model_placebo.pvalues['placebo_treat_post']:.4f}")
    results['placebo'] = model_placebo

    return results


def event_study(analysis_df):
    """Event study analysis."""
    print("\n" + "="*60)
    print("EVENT STUDY")
    print("="*60)

    years = sorted(analysis_df['YEAR'].unique())

    # Create year interactions
    for year in years:
        analysis_df[f'year_{year}'] = (analysis_df['YEAR'] == year).astype(int)
        analysis_df[f'treat_year_{year}'] = analysis_df['treat'] * analysis_df[f'year_{year}']

    # Reference: 2011
    ref_year = 2011
    treat_vars = [f'treat_year_{y}' for y in years if y != ref_year]
    year_vars = [f'year_{y}' for y in years if y != ref_year]

    formula = f'fulltime ~ treat + {" + ".join(year_vars)} + {" + ".join(treat_vars)} + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP)'

    model_es = smf.wls(
        formula,
        data=analysis_df,
        weights=analysis_df['PERWT']
    ).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['STATEFIP']})

    print("\nEvent Study Coefficients (ref = 2011):")
    print("-" * 50)

    es_data = []
    for year in years:
        if year == ref_year:
            print(f"  {year}: 0.0000 (reference)")
            es_data.append({'year': year, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
        else:
            coef = model_es.params[f'treat_year_{year}']
            se = model_es.bse[f'treat_year_{year}']
            ci = model_es.conf_int().loc[f'treat_year_{year}']
            print(f"  {year}: {coef:7.4f} (SE: {se:.4f}) [{ci[0]:.4f}, {ci[1]:.4f}]")
            es_data.append({'year': year, 'coef': coef, 'se': se, 'ci_low': ci[0], 'ci_high': ci[1]})

    return pd.DataFrame(es_data)


def year_by_year_analysis(df):
    """Analyze treatment effects year by year."""
    print("\n" + "="*60)
    print("YEAR-BY-YEAR TREATMENT EFFECTS")
    print("="*60)

    # Use the narrow bandwidth sample
    analysis_df = df[(df['in_analysis'] == 1) &
                     (((df['age_2012'] >= 26) & (df['age_2012'] <= 30)) |
                      ((df['age_2012'] >= 31) & (df['age_2012'] <= 35)))].copy()
    analysis_df['treat'] = (analysis_df['age_2012'] < 31).astype(int)

    post_years = [2013, 2014, 2015, 2016]
    results = []

    for year in post_years:
        # Compare each post year to 2011
        year_df = analysis_df[(analysis_df['YEAR'] == 2011) | (analysis_df['YEAR'] == year)].copy()
        year_df['post_year'] = (year_df['YEAR'] == year).astype(int)
        year_df['treat_post_year'] = year_df['treat'] * year_df['post_year']

        model = smf.wls(
            'fulltime ~ treat + post_year + treat_post_year + female + married + educ_hs + AGE + I(AGE**2) + C(STATEFIP)',
            data=year_df,
            weights=year_df['PERWT']
        ).fit(cov_type='cluster', cov_kwds={'groups': year_df['STATEFIP']})

        coef = model.params['treat_post_year']
        se = model.bse['treat_post_year']
        ci = model.conf_int().loc['treat_post_year']

        print(f"  {year} vs 2011: {coef:.4f} (SE: {se:.4f}) [{ci[0]:.4f}, {ci[1]:.4f}]")
        results.append({'year': year, 'coef': coef, 'se': se, 'ci_low': ci[0], 'ci_high': ci[1]})

    return pd.DataFrame(results)


def main():
    """Main analysis function."""
    print("="*60)
    print("DACA REPLICATION STUDY - ANALYSIS (Version 2)")
    print("="*60)

    # Load and prepare data
    df = load_and_filter_data()
    df = create_variables(df)

    # Summary statistics
    summary = summary_statistics(df)

    # Main DID regression with narrow bandwidth
    did_results, analysis_df = run_did_regression(df)

    # Robustness checks
    robustness = robustness_checks(df, analysis_df)

    # Event study
    es_results = event_study(analysis_df)

    # Year-by-year
    yearly = year_by_year_analysis(df)

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    pref = did_results['preferred']
    print(f"\nPREFERRED ESTIMATE (Narrow Bandwidth, State+Year FE):")
    print(f"  Effect of DACA eligibility on full-time employment")
    print(f"  Sample: Ages 26-35 in 2012 who arrived before age 16 and by 2007")
    print(f"  Coefficient: {pref['coef']:.4f}")
    print(f"  Std. Error:  {pref['se']:.4f}")
    print(f"  95% CI:      [{pref['ci'][0]:.4f}, {pref['ci'][1]:.4f}]")
    print(f"  p-value:     {pref['pval']:.4f}")
    print(f"  N:           {pref['n']:,}")

    return {
        'summary': summary,
        'did_results': did_results,
        'robustness': robustness,
        'event_study': es_results,
        'yearly': yearly,
        'df': df,
        'analysis_df': analysis_df
    }


if __name__ == "__main__":
    results = main()
