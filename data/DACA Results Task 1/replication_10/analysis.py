"""
DACA Replication Analysis
=========================
This script analyzes the causal effect of DACA eligibility on full-time employment
among ethnically Hispanic-Mexican, Mexican-born individuals in the US.

Research Design: Difference-in-Differences
- Treatment group: DACA-eligible individuals
- Control group: DACA-ineligible Mexican-born Hispanic individuals
- Pre-period: 2006-2011 (before DACA implementation in June 2012)
- Post-period: 2013-2016 (after DACA implementation)
- 2012 is excluded due to ambiguity (DACA implemented mid-year)

DACA Eligibility Criteria (as of June 15, 2012):
1. Arrived in US before 16th birthday
2. Had not yet turned 31 as of June 15, 2012 (born on or after June 16, 1981)
3. Lived continuously in US since June 15, 2007 (immigrated by 2007)
4. Were present in US on June 15, 2012 (proxy: in ACS sample)
5. Did not have lawful status (not a citizen, not naturalized)

Outcome: Full-time employment (working 35+ hours per week usually)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import warnings
import os

warnings.filterwarnings('ignore')

# Set paths
DATA_DIR = r"C:\Users\seraf\DACA Results Task 1\replication_10\data"
OUTPUT_DIR = r"C:\Users\seraf\DACA Results Task 1\replication_10"

def process_data():
    """
    Process the ACS data in chunks to handle memory constraints.
    Filter to Hispanic-Mexican, Mexican-born individuals.
    """
    print("Processing data in chunks...")

    # Define columns to use - minimizing memory
    usecols = [
        'YEAR', 'PERWT', 'STATEFIP', 'AGE', 'BIRTHYR', 'BIRTHQTR',
        'SEX', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
        'EMPSTAT', 'UHRSWORK', 'EDUC', 'MARST'
    ]

    # Define dtypes to minimize memory
    dtypes = {
        'YEAR': 'int16',
        'PERWT': 'float32',
        'STATEFIP': 'int8',
        'AGE': 'int8',
        'BIRTHYR': 'int16',
        'BIRTHQTR': 'int8',
        'SEX': 'int8',
        'HISPAN': 'int8',
        'BPL': 'int16',
        'CITIZEN': 'int8',
        'YRIMMIG': 'int16',
        'EMPSTAT': 'int8',
        'UHRSWORK': 'int8',
        'EDUC': 'int8',
        'MARST': 'int8'
    }

    chunks = []
    chunk_num = 0

    for chunk in pd.read_csv(
        os.path.join(DATA_DIR, 'data.csv'),
        usecols=usecols,
        dtype=dtypes,
        chunksize=2000000
    ):
        chunk_num += 1
        print(f"  Processing chunk {chunk_num}...")

        # Filter to Hispanic-Mexican ethnicity (HISPAN == 1 is Mexican)
        chunk = chunk[chunk['HISPAN'] == 1]

        # Filter to Mexican-born (BPL == 200 is Mexico)
        chunk = chunk[chunk['BPL'] == 200]

        # Exclude 2012 (ambiguous treatment year)
        chunk = chunk[chunk['YEAR'] != 2012]

        chunks.append(chunk)

    print("Combining chunks...")
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total filtered observations: {len(df):,}")

    return df


def create_variables(df):
    """
    Create analysis variables including DACA eligibility and outcome.
    """
    print("\nCreating analysis variables...")

    # ===== OUTCOME VARIABLE =====
    # Full-time employment: working 35+ hours per week usually
    # EMPSTAT == 1 means employed
    # UHRSWORK >= 35 means full-time
    df['fulltime'] = ((df['EMPSTAT'] == 1) & (df['UHRSWORK'] >= 35)).astype(int)

    # Also create employed variable for comparison
    df['employed'] = (df['EMPSTAT'] == 1).astype(int)

    # ===== POST-DACA INDICATOR =====
    # DACA was implemented June 15, 2012
    # Post period: 2013-2016
    df['post'] = (df['YEAR'] >= 2013).astype(int)

    # ===== DACA ELIGIBILITY =====
    # Criterion 1: Arrived before 16th birthday
    # age_at_arrival = year of immigration - birth year
    # We need YRIMMIG > 0 (valid immigration year)
    df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
    df['arrived_before_16'] = (df['age_at_arrival'] < 16) & (df['YRIMMIG'] > 0)

    # Criterion 2: Born on or after June 16, 1981 (not yet 31 as of June 15, 2012)
    # Being conservative: use born in 1982 or later to be clearly under 31
    # For those born in 1981, they could be under or over 31 depending on birth month
    # Using BIRTHQTR: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
    # Born June 16 or later 1981 would be: Q3 or Q4 of 1981, or 1982+
    df['under_31_2012'] = (
        (df['BIRTHYR'] >= 1982) |
        ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))
    )

    # Criterion 3: Immigrated by June 15, 2007 (continuous presence since then)
    # Conservative: immigrated 2007 or earlier
    df['immigrated_by_2007'] = (df['YRIMMIG'] <= 2007) & (df['YRIMMIG'] > 0)

    # Criterion 4: Present in US on June 15, 2012
    # Proxy: being in ACS sample (we assume this for everyone in the data)

    # Criterion 5: Not a citizen (undocumented proxy)
    # CITIZEN == 3 means "Not a citizen"
    # We exclude naturalized citizens and those born abroad to American parents
    df['noncitizen'] = (df['CITIZEN'] == 3)

    # ===== DACA ELIGIBLE =====
    # Must meet all criteria
    df['daca_eligible'] = (
        df['arrived_before_16'] &
        df['under_31_2012'] &
        df['immigrated_by_2007'] &
        df['noncitizen']
    ).astype(int)

    # ===== CONTROL VARIABLES =====
    df['female'] = (df['SEX'] == 2).astype(int)
    df['married'] = (df['MARST'].isin([1, 2])).astype(int)

    # Education categories
    df['less_than_hs'] = (df['EDUC'] <= 5).astype(int)  # Less than high school
    df['hs_graduate'] = (df['EDUC'] == 6).astype(int)   # High school graduate
    df['some_college'] = (df['EDUC'].isin([7, 8, 9])).astype(int)  # Some college
    df['college_plus'] = (df['EDUC'] >= 10).astype(int)  # Bachelor's or higher

    # Age squared for non-linear age effects
    df['age_sq'] = df['AGE'] ** 2

    # ===== INTERACTION TERM (DID) =====
    df['daca_x_post'] = df['daca_eligible'] * df['post']

    print(f"DACA eligible: {df['daca_eligible'].sum():,} ({100*df['daca_eligible'].mean():.1f}%)")
    print(f"Post-DACA period: {df['post'].sum():,} ({100*df['post'].mean():.1f}%)")
    print(f"Full-time employed: {df['fulltime'].sum():,} ({100*df['fulltime'].mean():.1f}%)")

    return df


def create_analysis_sample(df):
    """
    Create the final analysis sample with appropriate restrictions.
    """
    print("\nCreating analysis sample...")

    # Working-age sample (16-64)
    # Also need to be in the age range where DACA eligibility makes sense
    # In post period, DACA eligible must be under 31 + years since 2012
    # Use a consistent age restriction across years

    df_analysis = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
    print(f"After age restriction (16-64): {len(df_analysis):,}")

    # Exclude those with invalid immigration year for DACA calculations
    df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0].copy()
    print(f"After excluding missing immigration year: {len(df_analysis):,}")

    # Focus on non-citizens only (more precise comparison group)
    df_analysis = df_analysis[df_analysis['noncitizen']].copy()
    print(f"After restricting to non-citizens: {len(df_analysis):,}")

    return df_analysis


def run_did_analysis(df):
    """
    Run the difference-in-differences analysis.
    """
    print("\n" + "="*60)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("="*60)

    results = {}

    # ===== SUMMARY STATISTICS =====
    print("\n--- Summary Statistics ---")

    # By treatment status and period
    summary = df.groupby(['daca_eligible', 'post']).agg({
        'fulltime': ['mean', 'count'],
        'employed': 'mean',
        'AGE': 'mean',
        'female': 'mean',
        'PERWT': 'sum'
    }).round(3)
    print(summary)
    results['summary'] = summary

    # Weighted means
    print("\n--- Weighted Full-Time Employment Rates ---")
    for eligible in [0, 1]:
        for post in [0, 1]:
            mask = (df['daca_eligible'] == eligible) & (df['post'] == post)
            subset = df[mask]
            wtd_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
            n = len(subset)
            period = "Post" if post else "Pre"
            group = "DACA Eligible" if eligible else "Ineligible"
            print(f"  {group}, {period}-DACA: {wtd_mean:.4f} (N={n:,})")
    results['weighted_means'] = True

    # ===== BASIC DID REGRESSION =====
    print("\n--- Model 1: Basic DID ---")

    X1 = df[['daca_eligible', 'post', 'daca_x_post']].copy()
    X1 = sm.add_constant(X1)
    y = df['fulltime']
    weights = df['PERWT']

    model1 = sm.WLS(y, X1, weights=weights).fit(cov_type='HC1')
    print(model1.summary())
    results['model1'] = model1

    # ===== DID WITH DEMOGRAPHICS =====
    print("\n--- Model 2: DID with Demographics ---")

    X2 = df[['daca_eligible', 'post', 'daca_x_post',
             'AGE', 'age_sq', 'female', 'married',
             'hs_graduate', 'some_college', 'college_plus']].copy()
    X2 = sm.add_constant(X2)

    model2 = sm.WLS(y, X2, weights=weights).fit(cov_type='HC1')
    print(model2.summary())
    results['model2'] = model2

    # ===== DID WITH YEAR AND STATE FIXED EFFECTS =====
    print("\n--- Model 3: DID with Year and State Fixed Effects ---")

    # Create year dummies (omit one year as reference)
    year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True, dtype=float)

    # Create state dummies (omit one state as reference)
    state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True, dtype=float)

    X3 = pd.concat([
        df[['daca_eligible', 'post', 'daca_x_post',
            'AGE', 'age_sq', 'female', 'married',
            'hs_graduate', 'some_college', 'college_plus']].astype(float),
        year_dummies,
        state_dummies
    ], axis=1)
    X3 = sm.add_constant(X3)

    model3 = sm.WLS(y.astype(float), X3.astype(float), weights=weights.astype(float)).fit(cov_type='HC1')

    # Print only key coefficients
    print("\nKey Coefficients:")
    key_vars = ['const', 'daca_eligible', 'post', 'daca_x_post']
    for var in key_vars:
        if var in model3.params.index:
            coef = model3.params[var]
            se = model3.bse[var]
            pval = model3.pvalues[var]
            print(f"  {var}: {coef:.5f} (SE: {se:.5f}, p: {pval:.4f})")
    print(f"\nN = {model3.nobs:,.0f}")
    print(f"R-squared = {model3.rsquared:.4f}")
    results['model3'] = model3

    # ===== PREFERRED SPECIFICATION: Model 3 =====
    print("\n" + "="*60)
    print("PREFERRED ESTIMATE (Model 3)")
    print("="*60)

    coef = model3.params['daca_x_post']
    se = model3.bse['daca_x_post']
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se

    print(f"\nEffect of DACA eligibility on full-time employment:")
    print(f"  Coefficient: {coef:.5f}")
    print(f"  Std. Error:  {se:.5f}")
    print(f"  95% CI:      [{ci_low:.5f}, {ci_high:.5f}]")
    print(f"  p-value:     {model3.pvalues['daca_x_post']:.4f}")
    print(f"  Sample Size: {int(model3.nobs):,}")

    results['preferred'] = {
        'coef': coef,
        'se': se,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'pvalue': model3.pvalues['daca_x_post'],
        'n': int(model3.nobs)
    }

    return results


def run_robustness_checks(df, base_results):
    """
    Run robustness checks and alternative specifications.
    """
    print("\n" + "="*60)
    print("ROBUSTNESS CHECKS")
    print("="*60)

    robustness = {}
    y = df['fulltime']
    weights = df['PERWT']

    # ===== 1. Alternative age restriction (18-35) =====
    print("\n--- Robustness 1: Age 18-35 only ---")
    df_young = df[(df['AGE'] >= 18) & (df['AGE'] <= 35)].copy()

    year_dummies = pd.get_dummies(df_young['YEAR'], prefix='year', drop_first=True, dtype=float)
    state_dummies = pd.get_dummies(df_young['STATEFIP'], prefix='state', drop_first=True, dtype=float)

    X = pd.concat([
        df_young[['daca_eligible', 'post', 'daca_x_post',
                  'AGE', 'age_sq', 'female', 'married',
                  'hs_graduate', 'some_college', 'college_plus']].astype(float),
        year_dummies,
        state_dummies
    ], axis=1)
    X = sm.add_constant(X)

    model_young = sm.WLS(df_young['fulltime'].astype(float), X.astype(float), weights=df_young['PERWT'].astype(float)).fit(cov_type='HC1')
    print(f"  DID coefficient: {model_young.params['daca_x_post']:.5f}")
    print(f"  SE: {model_young.bse['daca_x_post']:.5f}")
    print(f"  N = {model_young.nobs:,.0f}")
    robustness['age_18_35'] = model_young

    # ===== 2. Employed (any) as outcome =====
    print("\n--- Robustness 2: Any Employment as Outcome ---")
    y_emp = df['employed'].astype(float)

    year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True, dtype=float)
    state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True, dtype=float)

    X = pd.concat([
        df[['daca_eligible', 'post', 'daca_x_post',
            'AGE', 'age_sq', 'female', 'married',
            'hs_graduate', 'some_college', 'college_plus']].astype(float),
        year_dummies,
        state_dummies
    ], axis=1)
    X = sm.add_constant(X)

    model_emp = sm.WLS(y_emp, X.astype(float), weights=weights.astype(float)).fit(cov_type='HC1')
    print(f"  DID coefficient: {model_emp.params['daca_x_post']:.5f}")
    print(f"  SE: {model_emp.bse['daca_x_post']:.5f}")
    print(f"  N = {model_emp.nobs:,.0f}")
    robustness['any_employment'] = model_emp

    # ===== 3. Male subsample =====
    print("\n--- Robustness 3: Males Only ---")
    df_male = df[df['female'] == 0].copy()

    year_dummies = pd.get_dummies(df_male['YEAR'], prefix='year', drop_first=True, dtype=float)
    state_dummies = pd.get_dummies(df_male['STATEFIP'], prefix='state', drop_first=True, dtype=float)

    X = pd.concat([
        df_male[['daca_eligible', 'post', 'daca_x_post',
                 'AGE', 'age_sq', 'married',
                 'hs_graduate', 'some_college', 'college_plus']].astype(float),
        year_dummies,
        state_dummies
    ], axis=1)
    X = sm.add_constant(X)

    model_male = sm.WLS(df_male['fulltime'].astype(float), X.astype(float), weights=df_male['PERWT'].astype(float)).fit(cov_type='HC1')
    print(f"  DID coefficient: {model_male.params['daca_x_post']:.5f}")
    print(f"  SE: {model_male.bse['daca_x_post']:.5f}")
    print(f"  N = {model_male.nobs:,.0f}")
    robustness['males'] = model_male

    # ===== 4. Female subsample =====
    print("\n--- Robustness 4: Females Only ---")
    df_female = df[df['female'] == 1].copy()

    year_dummies = pd.get_dummies(df_female['YEAR'], prefix='year', drop_first=True, dtype=float)
    state_dummies = pd.get_dummies(df_female['STATEFIP'], prefix='state', drop_first=True, dtype=float)

    X = pd.concat([
        df_female[['daca_eligible', 'post', 'daca_x_post',
                   'AGE', 'age_sq', 'married',
                   'hs_graduate', 'some_college', 'college_plus']].astype(float),
        year_dummies,
        state_dummies
    ], axis=1)
    X = sm.add_constant(X)

    model_female = sm.WLS(df_female['fulltime'].astype(float), X.astype(float), weights=df_female['PERWT'].astype(float)).fit(cov_type='HC1')
    print(f"  DID coefficient: {model_female.params['daca_x_post']:.5f}")
    print(f"  SE: {model_female.bse['daca_x_post']:.5f}")
    print(f"  N = {model_female.nobs:,.0f}")
    robustness['females'] = model_female

    # ===== 5. Without survey weights =====
    print("\n--- Robustness 5: Unweighted ---")

    year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True, dtype=float)
    state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True, dtype=float)

    X = pd.concat([
        df[['daca_eligible', 'post', 'daca_x_post',
            'AGE', 'age_sq', 'female', 'married',
            'hs_graduate', 'some_college', 'college_plus']].astype(float),
        year_dummies,
        state_dummies
    ], axis=1)
    X = sm.add_constant(X)

    model_unwtd = sm.OLS(y.astype(float), X.astype(float)).fit(cov_type='HC1')
    print(f"  DID coefficient: {model_unwtd.params['daca_x_post']:.5f}")
    print(f"  SE: {model_unwtd.bse['daca_x_post']:.5f}")
    print(f"  N = {model_unwtd.nobs:,.0f}")
    robustness['unweighted'] = model_unwtd

    return robustness


def compute_event_study(df):
    """
    Compute event study estimates to check parallel trends.
    """
    print("\n" + "="*60)
    print("EVENT STUDY / PARALLEL TRENDS CHECK")
    print("="*60)

    # Create year-specific treatment effects
    # Reference year: 2011 (last pre-treatment year)
    years = sorted(df['YEAR'].unique())
    ref_year = 2011

    y = df['fulltime'].astype(float)
    weights = df['PERWT'].astype(float)

    # Create interactions for each year
    df_es = df.copy()
    for year in years:
        if year != ref_year:
            df_es[f'daca_x_{year}'] = (df_es['daca_eligible'] * (df_es['YEAR'] == year)).astype(float)

    # Year dummies
    year_dummies = pd.get_dummies(df_es['YEAR'], prefix='year', drop_first=True, dtype=float)
    state_dummies = pd.get_dummies(df_es['STATEFIP'], prefix='state', drop_first=True, dtype=float)

    # Interaction columns
    interaction_cols = [col for col in df_es.columns if col.startswith('daca_x_') and col != 'daca_x_post']

    X = pd.concat([
        df_es[['daca_eligible'] + interaction_cols +
              ['AGE', 'age_sq', 'female', 'married',
               'hs_graduate', 'some_college', 'college_plus']].astype(float),
        year_dummies,
        state_dummies
    ], axis=1)
    X = sm.add_constant(X)

    model_es = sm.WLS(y, X.astype(float), weights=weights).fit(cov_type='HC1')

    print("\nEvent Study Coefficients (relative to 2011):")
    print("-" * 50)
    event_study_results = {}
    for year in years:
        if year != ref_year:
            col = f'daca_x_{year}'
            if col in model_es.params.index:
                coef = model_es.params[col]
                se = model_es.bse[col]
                event_study_results[year] = {'coef': coef, 'se': se}
                sig = '*' if model_es.pvalues[col] < 0.1 else ''
                sig = '**' if model_es.pvalues[col] < 0.05 else sig
                sig = '***' if model_es.pvalues[col] < 0.01 else sig
                print(f"  {year}: {coef:8.5f} ({se:.5f}) {sig}")
        else:
            event_study_results[year] = {'coef': 0, 'se': 0}
            print(f"  {year}: Reference year")

    return event_study_results, model_es


def save_results(df, results, robustness, event_study):
    """
    Save results to files.
    """
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save summary statistics
    summary_stats = df.groupby(['daca_eligible', 'post']).agg({
        'fulltime': ['mean', 'std', 'count'],
        'employed': 'mean',
        'AGE': 'mean',
        'female': 'mean',
        'married': 'mean',
        'PERWT': 'sum'
    }).round(4)
    summary_stats.to_csv(os.path.join(OUTPUT_DIR, 'summary_statistics.csv'))
    print("Saved summary_statistics.csv")

    # Save event study coefficients
    es_df = pd.DataFrame(event_study).T
    es_df.index.name = 'year'
    es_df.to_csv(os.path.join(OUTPUT_DIR, 'event_study_coefficients.csv'))
    print("Saved event_study_coefficients.csv")

    # Save main results
    main_results = {
        'Model': ['Basic DID', 'With Demographics', 'With Year/State FE'],
        'Coefficient': [
            results['model1'].params['daca_x_post'],
            results['model2'].params['daca_x_post'],
            results['model3'].params['daca_x_post']
        ],
        'Std_Error': [
            results['model1'].bse['daca_x_post'],
            results['model2'].bse['daca_x_post'],
            results['model3'].bse['daca_x_post']
        ],
        'P_Value': [
            results['model1'].pvalues['daca_x_post'],
            results['model2'].pvalues['daca_x_post'],
            results['model3'].pvalues['daca_x_post']
        ],
        'N': [
            int(results['model1'].nobs),
            int(results['model2'].nobs),
            int(results['model3'].nobs)
        ],
        'R_Squared': [
            results['model1'].rsquared,
            results['model2'].rsquared,
            results['model3'].rsquared
        ]
    }
    main_df = pd.DataFrame(main_results)
    main_df.to_csv(os.path.join(OUTPUT_DIR, 'main_results.csv'), index=False)
    print("Saved main_results.csv")

    # Save robustness results
    rob_results = {
        'Specification': ['Age 18-35', 'Any Employment', 'Males Only', 'Females Only', 'Unweighted'],
        'Coefficient': [
            robustness['age_18_35'].params['daca_x_post'],
            robustness['any_employment'].params['daca_x_post'],
            robustness['males'].params['daca_x_post'],
            robustness['females'].params['daca_x_post'],
            robustness['unweighted'].params['daca_x_post']
        ],
        'Std_Error': [
            robustness['age_18_35'].bse['daca_x_post'],
            robustness['any_employment'].bse['daca_x_post'],
            robustness['males'].bse['daca_x_post'],
            robustness['females'].bse['daca_x_post'],
            robustness['unweighted'].bse['daca_x_post']
        ],
        'N': [
            int(robustness['age_18_35'].nobs),
            int(robustness['any_employment'].nobs),
            int(robustness['males'].nobs),
            int(robustness['females'].nobs),
            int(robustness['unweighted'].nobs)
        ]
    }
    rob_df = pd.DataFrame(rob_results)
    rob_df.to_csv(os.path.join(OUTPUT_DIR, 'robustness_results.csv'), index=False)
    print("Saved robustness_results.csv")

    # Save preferred estimate details
    pref = results['preferred']
    with open(os.path.join(OUTPUT_DIR, 'preferred_estimate.txt'), 'w') as f:
        f.write("PREFERRED ESTIMATE\n")
        f.write("="*50 + "\n")
        f.write(f"Coefficient: {pref['coef']:.6f}\n")
        f.write(f"Standard Error: {pref['se']:.6f}\n")
        f.write(f"95% CI: [{pref['ci_low']:.6f}, {pref['ci_high']:.6f}]\n")
        f.write(f"P-value: {pref['pvalue']:.6f}\n")
        f.write(f"Sample Size: {pref['n']:,}\n")
    print("Saved preferred_estimate.txt")


def main():
    """
    Main execution function.
    """
    print("="*60)
    print("DACA REPLICATION ANALYSIS")
    print("Effect on Full-Time Employment")
    print("="*60)

    # Process data
    df = process_data()

    # Create variables
    df = create_variables(df)

    # Create analysis sample
    df_analysis = create_analysis_sample(df)

    # Run main DID analysis
    results = run_did_analysis(df_analysis)

    # Run robustness checks
    robustness = run_robustness_checks(df_analysis, results)

    # Event study for parallel trends
    event_study, es_model = compute_event_study(df_analysis)

    # Save results
    save_results(df_analysis, results, robustness, event_study)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    return df_analysis, results, robustness, event_study


if __name__ == "__main__":
    df, results, robustness, event_study = main()
