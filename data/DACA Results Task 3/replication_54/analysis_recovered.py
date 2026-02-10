# Code recovered from run_log_54.md
# Task 3, Replication 54
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# --- Code block 2 ---
import statsmodels.api as sm

df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
y = df['FT']
model1 = sm.OLS(y, X).fit(cov_type='HC1')

# --- Code block 3 ---
model2 = sm.WLS(y, X, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 4 ---
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_2YR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

controls = ['EDUC_HS', 'EDUC_SOMECOL', 'EDUC_2YR', 'EDUC_BA',
            'FEMALE', 'MARRIED', 'NCHILD', 'FAMSIZE']
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER'] + controls])
model3 = sm.OLS(y, X).fit(cov_type='HC1')

# --- Code block 5 ---
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True, dtype=int)
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True, dtype=int)

X_df = pd.concat([df[['ELIGIBLE', 'ELIGIBLE_AFTER'] + controls],
                  state_dummies, year_dummies], axis=1)
X = sm.add_constant(X_df.astype(float))
model4 = sm.OLS(y, X).fit(cov_type='HC1')

# --- Code block 6 ---
model5 = sm.WLS(y, X, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 7 ---
model_cluster = sm.OLS(y, X_basic).fit(cov_type='cluster',
                                        cov_kwds={'groups': df['STATEFIP']})

# --- Code block 8 ---
probit_model = sm.Probit(y, X).fit()
marginal_effects = probit_model.get_margeff()

# --- Code block 9 ---
logit_model = sm.Logit(y, X).fit()

