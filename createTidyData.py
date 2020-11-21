# Data wrangling
import pandas as pd 

# Dates
import datetime

# Directories
import os 

# Array math
import numpy as np 
import math

# Listing all the directories in the data folder 
dataFolders = os.listdir('data')
dataFolders = [x for x in dataFolders if not x.endswith('.csv')]

# Converting the folder names to datetime. That way when we sort we can be sure that the 
# newest date will be the last coordinate
dataFolders = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in dataFolders]

# Sorting 
dataFolders.sort()

# Extracting the newest date 
newestDate = dataFolders[-1].strftime('%Y-%m-%d')

# Reading the newest files
dMunicipality = pd.read_csv(f'data/{newestDate}/municipality_data.csv')
dPatient = pd.read_csv(f'data/{newestDate}/patient_data.csv')
dUS = pd.read_csv(f'data/{newestDate}/US_data.csv')

# Fixing some names in the sex column
dPatient['sex'] = ['Moteris' if x == 'mot.' else x for x in dPatient['sex']]

# Converting the day column to date format 
dMunicipality['day'] = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in dMunicipality['day']]
dPatient['day'] = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in dPatient['day']]

# Creating logical columns for certain boolean columns 
for col in ['imported', 'foreigner', 'hospitalized']:
    dPatient[f'is_{col}'] = [1 if x == 'Taip' else 0 for x in dPatient[col]]

# Creating a column indicating a case 
dPatient['is_covid'] = 1

# Infering information from the status column
dPatient['status'] = dPatient['status'].fillna('no_info')

dPatient['is_cured'] = [1 if x == 'Pasveiko' else 0 for x in dPatient['status']]
dPatient['is_death'] = [1 if x == 'Mirė' else 0 for x in dPatient['status']]
dPatient['is_other'] = [1 if x == 'Kita' else 0 for x in dPatient['status']]
dPatient['is_treated'] = [1 if x == 'Gydomas' else 0 for x in dPatient['status']]
dPatient['is_nonsick'] = [1 if x == 'Nesirgo' else 0 for x in dPatient['status']]
dPatient['is_noinfo'] = [1 if x == 'no_info' else 0 for x in dPatient['status']] 

# Creating a concat column for gender and age 
dPatient['sex_age'] = dPatient['age'] + dPatient['sex']

# Saving the column names that we will be aggregating 
agg_cols = [ 
    'is_cured', 
    'is_death', 
    'is_other', 
    'is_treated', 
    'is_nonsick', 
    'is_noinfo', 
    'is_imported',
    'is_foreigner',
    'is_hospitalized',
    'is_covid'
    ]

# Function to aggregate to the daily level by a given column
def aggToDay(
    col, 
    data, 
    agg_cols=[
    'is_cured', 
    'is_death', 
    'is_other', 
    'is_treated', 
    'is_nonsick', 
    'is_noinfo', 
    'is_imported',
    'is_foreigner',
    'is_hospitalized'
    ]):
    """
    Creates a tidy matrix where each row is a day
    """
    d = dPatient.groupby(['day', col], as_index=False)[agg_cols].sum()
    d = d.pivot_table(index='day', columns=col)
    d.columns = ['-'.join(x).strip() for x in d.columns.values]
    d.fillna(0, inplace=True)

    return d 

# High level daily stats 
d = dPatient.groupby('day', as_index=False)[agg_cols].sum()

# Getting covid cases by age group 
#dByAge = aggToDay('age', dPatient, agg_cols=['is_covid'])

# Aggregating to daily data 
#dAge = aggToDay('sex_age', dPatient, agg_cols=agg_cols)

# Calculating daily covid cases 
#dTotal = dPatient.groupby('day', as_index=False)['is_hospitalized'].sum()

# Merging all the data together 
#d = pd.merge(dTotal, dByAge, how='left', on='day')

# Merging additional info 
#d = pd.merge(dTotal, dInfo, how='left', on='day')

# Creating the column for weekday number 
weekday = pd.Series([str(x.weekday() + 1) for x in d['day']])
wkd = pd.get_dummies(weekday)
wkd.columns = [f'weekday_{x}' for x in wkd.columns.values]
d = pd.concat([d, wkd], axis=1)

# Getting total daily test data 
#tests = dMunicipality.groupby('day', as_index=False)['tests_total'].sum()
#d = pd.merge(d, tests, how='left', on='day')
#d.fillna(0, inplace=True)

# Creating a dummy variable for quarantine 
d['is_quarantine'] = [1 if (((x >= datetime.date(2020, 3, 16)) & (x <= datetime.date(2020, 6, 16))) | (x >= datetime.date(2020, 10, 27))) else 0 for x in d['day']]

# Wrangling US cases and hospitalizations 
dUS['day'] = [datetime.datetime.strptime(str(x), '%Y%m%d').date() for x in dUS['date']]
dUS = dUS.drop('date', axis=1)
dUS = dUS.rename(columns={'positiveIncrease': 'is_covid', 'hospitalizedCurrently': 'is_hospitalized'})

#dUShospital = dUS[['day', 'state', 'is_hospitalized']].pivot_table(index='day', columns='state')
#dUShospital.columns = ['_'.join(x).strip() for x in dUShospital.columns.values]

#dUSCovid = dUS[['day', 'state', 'is_covid']].pivot_table(index='day', columns='state')
#dUSCovid.columns = ['_'.join(x).strip() for x in dUSCovid.columns.values]

#dUSfinal = pd.merge(dUShospital, dUSCovid, how='left', on='day')
#dUSfinal.fillna(0, inplace=True)

dUSfinal = dUS.groupby('day', as_index=False)[['is_hospitalized', 'is_covid']].sum()

# Saving the tidy data to the data folder 
d.to_csv('data/tidy_data_LT.csv', index=False)
dUSfinal.to_csv('data/tidy_data_US.csv', index=False)