# Code recovered from run_log_21.md
# Task 3, Replication 21
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Shape:', df.shape)
# Output: (17382, 105)

# --- Code block 2 ---
import statsmodels.api as sm
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
X1 = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
y = df['FT']
model1 = sm.OLS(y, X1).fit(cov_type='HC1')

# --- Code block 3 ---
model2 = sm.WLS(y, X1, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 4 ---
covariates = ['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE_c', 'AGE_c_sq',
              'MARRIED', 'HAS_CHILDREN', 'ed_hs', 'ed_somecoll', 'ed_ba']
X3 = sm.add_constant(df[covariates])
model3 = sm.WLS(y, X3, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 5 ---
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
# ... added state dummies to regression

# --- Code block 6 ---
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
# ... included both state and year FE, dropped AFTER (absorbed by year FE)

# --- Code block 7 ---
# Plotted FT rates by year for both groups
# Pre-period trends appear roughly parallel
# Post-period shows divergence favoring eligible group

# --- Code block 8 ---
# Created year-specific treatment effects relative to 2011
# Pre-period coefficients: 2008 (-0.068), 2009 (-0.050), 2010 (-0.082)
# Post-period coefficients: 2013 (0.016), 2014 (0.000), 2015 (0.001), 2016 (0.074)

# --- Code block 9 ---
# F-statistic = 1.96, p-value = 0.118
# Fails to reject null of parallel pre-trends

# --- Code block 10 ---
# Male: DiD = 0.072 (SE = 0.020, p < 0.001), N = 9,075
# Female: DiD = 0.053 (SE = 0.028, p = 0.061), N = 8,307

# --- Code block 11 ---
# High School: DiD = 0.061 (p = 0.005)
# Some College: DiD = 0.067 (p = 0.124)
# Two-Year Degree: DiD = 0.182 (p = 0.018)
# BA+: DiD = 0.162 (p = 0.023)

# --- Code block 12 ---
plt.plot(years, ft_elig, 'o-', label='Eligible')
plt.plot(years, ft_comp, 's--', label='Comparison')
plt.savefig('figures/parallel_trends.pdf')

# --- Code block 13 ---
plt.errorbar(years_plot, coefs, yerr=errors, fmt='o', capsize=5)
plt.savefig('figures/event_study.pdf')

# --- Code block 14 ---
# Created 2x2 visualization with counterfactual
plt.savefig('figures/did_visualization.pdf')

# --- Code block 15 ---
# Bar charts showing FT rates by group and period
plt.savefig('figures/ft_distribution.pdf')

# --- Code block 16 ---
# Coefficient plot across all 5 specifications
plt.savefig('figures/robustness.pdf')

