"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("=" * 60)
print("DACA REPLICATION STUDY - DATA EXPLORATION")
print("=" * 60)

# Load numeric version of data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Number of observations: {len(df):,}")
print(f"Number of variables: {df.shape[1]}")

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Basic info about key variables
print("\n" + "=" * 60)
print("KEY VARIABLES SUMMARY")
print("=" * 60)

# Key variables for analysis
key_vars = ['YEAR', 'FT', 'ELIGIBLE', 'AFTER', 'PERWT', 'AGE', 'SEX', 'EDUC', 'MARST']
for var in key_vars:
    if var in df.columns:
        print(f"\n{var}:")
        print(df[var].value_counts().sort_index().head(20))

# Check the years in dataset
print("\n" + "=" * 60)
print("YEAR DISTRIBUTION")
print("=" * 60)
print(df['YEAR'].value_counts().sort_index())

# Check ELIGIBLE distribution
print("\n" + "=" * 60)
print("ELIGIBLE (Treatment) DISTRIBUTION")
print("=" * 60)
print(df['ELIGIBLE'].value_counts())
print(f"\nTreatment group (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum():,}")
print(f"Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum():,}")

# Check AFTER distribution
print("\n" + "=" * 60)
print("AFTER (Post-treatment) DISTRIBUTION")
print("=" * 60)
print(df['AFTER'].value_counts())

# Check FT (outcome) distribution
print("\n" + "=" * 60)
print("FT (Full-time employment) DISTRIBUTION")
print("=" * 60)
print(df['FT'].value_counts())
print(f"\nFull-time rate overall: {df['FT'].mean():.4f}")

# Cross-tabulation: ELIGIBLE x AFTER
print("\n" + "=" * 60)
print("CROSS-TABULATION: ELIGIBLE x AFTER")
print("=" * 60)
print(pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True))

# Mean FT by group
print("\n" + "=" * 60)
print("MEAN FT BY ELIGIBLE x AFTER (Unweighted)")
print("=" * 60)
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'std'])
print(ft_means)

# Calculate simple DiD estimate
print("\n" + "=" * 60)
print("SIMPLE DIFFERENCE-IN-DIFFERENCES (Unweighted)")
print("=" * 60)

# Mean FT for each cell
ft_treat_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
ft_treat_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
ft_control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
ft_control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"Treatment Pre:  {ft_treat_pre:.4f}")
print(f"Treatment Post: {ft_treat_post:.4f}")
print(f"Control Pre:    {ft_control_pre:.4f}")
print(f"Control Post:   {ft_control_post:.4f}")

# DiD estimate
did_simple = (ft_treat_post - ft_treat_pre) - (ft_control_post - ft_control_pre)
print(f"\nDiD Estimate: {did_simple:.4f}")
print(f"  = ({ft_treat_post:.4f} - {ft_treat_pre:.4f}) - ({ft_control_post:.4f} - {ft_control_pre:.4f})")
print(f"  = {ft_treat_post - ft_treat_pre:.4f} - {ft_control_post - ft_control_pre:.4f}")

# Demographics check
print("\n" + "=" * 60)
print("DEMOGRAPHICS BY GROUP")
print("=" * 60)

# Age distribution by ELIGIBLE
print("\nAge distribution by ELIGIBLE status:")
print(df.groupby('ELIGIBLE')['AGE'].describe())

# Sex distribution
print("\nSex distribution (1=Male, 2=Female):")
print(pd.crosstab(df['ELIGIBLE'], df['SEX'], normalize='index'))

# Check sample weights
print("\n" + "=" * 60)
print("SURVEY WEIGHTS (PERWT)")
print("=" * 60)
print(df['PERWT'].describe())
