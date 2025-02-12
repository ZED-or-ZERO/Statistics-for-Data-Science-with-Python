import piplite
await piplite.install(['numpy'],['pandas'])
await piplite.install(['seaborn'])

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats 
import statsmodels.api as sm
from statsmodels.formula.api import ols

from js import fetch
import io

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
resp = await fetch(URL)
boston_url = io.BytesIO((await resp.arrayBuffer()).to_py())

boston_df=pd.read_csv(boston_url)

print(boston_df.head())
print(boston_df.info())
print(boston_df.isnull().sum())

# MEDV 
plt.figure(figsize=(8, 6))
sns.histplot(boston_df['MEDV'], bins=30, kde=True)
plt.title('Гистограмма медианной стоимости домов (MEDV)')
plt.xlabel('MEDV (тыс. $)')
plt.ylabel('Частота')
plt.show()

# Charles River (CHAS)
plt.figure(figsize=(6, 4))
sns.histplot(boston_df['CHAS'], bins=2, discrete=True)
plt.title('Гистограмма CHAS (Принадлежность к реке Charles)')
plt.xlabel('CHAS (0 - нет, 1 - да)')
plt.ylabel('Частота')
plt.xticks([0, 1])
plt.show()

# Создадим группы по возрасту домов
#MEDV vs AGE
boston_df['AGE_group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, 100], labels=['≤35', '35-70', '≥70'])

plt.figure(figsize=(10, 6))
sns.histplot(data=boston_df, x='MEDV', hue='AGE_group', bins=30, kde=True, legend=True)
plt.title('Гистограмма стоимости домов (MEDV) по возрастным группам')
plt.xlabel('MEDV (тыс. $)')
plt.ylabel('Частота')
plt.legend(title='Возраст домов', labels=['≤35 лет', '35-70 лет', '≥70 лет'])
plt.show()

# NOX vs INDUS
plt.figure(figsize=(8, 6))
sns.scatterplot(data=boston_df, x='INDUS', y='NOX')
plt.title('Взаимосвязь между NOX и INDUS')
plt.xlabel('Доля неторговых земель (INDUS)')
plt.ylabel('Концентрация оксидов азота (NOX)')
plt.show()

# PTRATIO
plt.figure(figsize=(8, 6))
sns.histplot(boston_df['PTRATIO'], bins=20, kde=True)
plt.title('Гистограмма соотношения учеников и учителей')
plt.xlabel('PTRATIO')
plt.ylabel('Частота')
plt.show()

# Т-test (CHAS vs MEDV)
chas_1 = boston_df[boston_df['CHAS'] == 1]['MEDV']
chas_0 = boston_df[boston_df['CHAS'] == 0]['MEDV']

t_stat, p_value = stats.ttest_ind(chas_1, chas_0, equal_var=False)
print(f'T-статистика: {t_stat:.4f}, p-значение: {p_value:.4f}')

# ANOVA (MEDV vs AGE)
anova_model = ols('MEDV ~ C(AGE_group)', data=boston_df).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

# NOX vs INDUS
corr_coef, corr_p_value = stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])
print(f'Корреляция: {corr_coef:.4f}, p-значение: {corr_p_value:.4f}')

# DIS vs MEDV 
reg_model = ols('MEDV ~ DIS', data=boston_df).fit()
print(reg_model.summary())

