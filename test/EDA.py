import pandas as pd
import datetime
starttime = datetime.datetime.now()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# This data cleaning part gets ideas from https://www.kaggle.com/apryor6/detailed-cleaning-visualization-python
# HAN YIKAI takes charge of feature 1-9
# official file
file = './input/train_ver2.csv'

# sns.set(rc = {'figure.figsize':(20,18)})
limit_rows = 7000000
df = pd.read_csv(file,
                 dtype={'sexo': str,

                        'ind_nuevo': str,
                        'ult_fec_cli_1t': str,
                        'indext': str},
                 nrows=limit_rows,
                 low_memory=False)

# resample the data for saving memory and shuffle the dataset
is_sample = True
if is_sample is True:
    # df = df.sample(n=100000)
    df = df.sample(frac=0.5)
# for 'age' :
df['age'] = pd.to_numeric(df['age'], errors='coerce')

plt.figure(dpi=300, figsize=(24, 8))
sns.displot(df['age'],
            bins=np.arange(0, 121, 2),
            kde=False)
plt.ylabel('Customers', fontsize=12)
plt.xlabel('Age', fontsize=12)
plt.savefig('Age-1.png')

# find mean of age in different stage
low_mean_age = df.loc[(df.age >= 18) & (df.age <= 30), 'age'].mean(skipna=True)
high_mean_age = df.loc[(df.age >= 30) & (df.age <= 100), 'age'].mean(skipna=True)
total_mean_age = df['age'].mean()
# fill age values
df.loc[df.age > 100, 'age'] = high_mean_age
df.loc[df.age < 18, 'age'] = low_mean_age
df['age'].fillna(total_mean_age, inplace=True)
plt.figure(dpi=300, figsize=(24, 8))
sns.displot(df['age'],
            bins=np.arange(0, 101, 2),
            kde=False)
plt.ylabel('Customers', fontsize=12)
plt.xlabel('Age', fontsize=12)
plt.savefig('Age-2.png')

# for nomprov
# fix special value
df.loc[df.nomprov == "CORU\xc3\x91A, A", "nomprov"] = "CORUNA, A"

# for 'ind_nuevo' : because the max of months_active is 6, so we fill ind_nuevo by 1
months_active = df.loc[df['ind_nuevo'].isnull(), :].groupby('ncodpers', sort=False).size()
# the result is 6 which means all missing values of customers are new.
months_active.max()
# 1 means new customer
df.loc[df['ind_nuevo'].isnull(), 'ind_nuevo'] = 1

# for 'fecha_alta' : sort dates and use index to find median_date
dates = df.loc[:, 'fecha_alta'].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
dates = df.loc[:, 'fecha_alta'].sort_values().reset_index()
df.loc[df.fecha_alta.isnull(), 'fecha_alta'] = dates.loc[median_date, 'fecha_alta']
df['fecha_alta'].describe()

# for 'pais_residencia', 'sexo', 'ind_empleado' and 'indresi' :
df.loc[df['pais_residencia'].isnull(), 'pais_residencia'] = 'NA'
df.loc[df['sexo'].isnull(), 'sexo'] = 'NA'
df.loc[df['ind_empleado'].isnull(), 'ind_empleado'] = 'NA'
df.loc[df['indresi'].isnull(), 'indresi'] = 'NA'

# for antiguedad
df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')
df.loc[df['antiguedad'].isnull(), 'antiguedad'] = df['antiguedad'].min()
# treat for -999999 outliers
df.loc[df['antiguedad'] < 0, 'antiguedad'] = 0

# LIU JIAYU takes charge of feature 15-19
# for 'indext' : most of the customers have their birth country different than the bank country
# for 'conyuemp' : huge missing values, drop it.
# for 'tipodom' : not seem to be useful, the value of it almost be '1', drop it.
df.drop(['conyuemp', 'tipodom'], axis=1, inplace=True)
# for 'indfall' : deceased index, set missing value to be 'N' means not deceased.
df.loc[df.indfall.isnull(), 'indfall'] = 'N'
# drop all missing values of else features 'indext' and 'canal_entrada'.
# draw two figures for 'canal_entrada'
plt.figure(figsize=(20, 14))
df['canal_entrada'].value_counts().plot(x=None, y=None, kind='pie')
plt.savefig('canal_entrada-1.png')

# extract the necessary columns
canal_label = df.loc[:, ['canal_entrada']].join(df.iloc[:, 24:])
# in five major channels
canal_subset = ['KHE', 'KAT', 'KFC', 'KFA', 'KHK']
canal_label = canal_label.loc[canal_label['canal_entrada'].isin(canal_subset)]
canal_label = canal_label.groupby('canal_entrada').agg('sum')
canal_label.T.plot(kind='barh', stacked=True, fontsize=14, figsize=[16, 12])
plt.title('Major join channels associated with products')
plt.xlabel('Total number of customers')
plt.ylabel('Products')
plt.legend()
plt.savefig('canal_entrada-1.png')

# WU SHANSHAN takes charge of feature 10-14
# for 'indrel' : fill in the missing value with common status 1
unique_ids = pd.Series(df['ncodpers'].unique())
pd.Series([i for i in df.indrel]).value_counts()
df.loc[df.indrel.isnull(), 'indrel'] = 1
# for 'tiprel_1mes' : fill in the missing value with common status 'A'
df.loc[df.tiprel_1mes.isnull(), 'tiprel_1mes'] = 'A'
df.tiprel_1mes = df.tiprel_1mes.astype("category")
# for 'indrel_1mes' : fill in the missing value with common status 'P'
map_dict = {1.0: "1",
            "1.0": "1",
            "1": "1",
            "3.0": "3",
            "P": "P",
            3.0: "3",
            2.0: "2",
            "3": "3",
            "2.0": "2",
            "4.0": "4",
            "4": "4",
            "2": "2"}
df.indrel_1mes.fillna('P', inplace=True)
df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x, x))
df.indrel_1mes = df.indrel_1mes.astype('category')

# LU YUCHENG takes charge of feature 20-24
# for 'cod_prov' : assign a new category(the value is 0)
df.loc[df['cod_prov'].isnull(), 'cod_prov'] = 0
# for 'renta' : replace missing values with the median of gross income by province
#               if all values in one province are missing values, replace them with the median of all incomes
total_median = df.loc[df['renta'].notnull(), 'renta'].median()
df.loc[df['renta'].isnull(), 'renta'] = 0
renta_by_province = df.groupby('cod_prov').agg({'renta': lambda x: x.median(skipna=True)}).reset_index()
renta_by_province = renta_by_province.replace(0, total_median)
for i in range(len(renta_by_province)):
    province_data = df[(df['cod_prov'] == i) & (df['renta'] == 0)]
    df.loc[(df['cod_prov'] == i) & (df['renta'] == 0), 'renta'] = renta_by_province.loc[i, 'renta']
# for 'segmento' : map string value with int value and assign a new category(the value is 0)
mapping = {'01 - TOP': 1, '02 - PARTICULARES': 2, '03 - UNIVERSITARIO': 3}
df['segmento'] = df['segmento'].map(mapping)
df.loc[df['segmento'].isnull(), 'segmento'] = 0
# df.drop(['nomprov'], axis=1, inplace=True)

# Drop the columns with majority of missing values
df.drop(["ult_fec_cli_1t"], axis=1, inplace=True)
df.dropna(inplace=True)
print(df.isnull().any())
print(df.info())

is_sample =False
# resample for memory
if is_sample is True:
    # df = df.sample(n=50000)
    df = df.sample(frac=0.5)

df.to_csv('./tmp.csv')
#do something other
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
