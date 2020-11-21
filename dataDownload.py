# Reading data
import pandas as pd 

# Directory manipulation
import os 

# Date wrangling
import datetime 

# Memory handling
import gc 

# Track timing
import time 

# Request making for different APIS
import requests

# Saving the current data when the script was initiated 
curDate = datetime.datetime.now().date()

# Reading US data 
start = time.time()
USdata = requests.get('https://api.covidtracking.com/v1/states/daily.json')
USdata = pd.DataFrame(USdata.json())[['date', 'state', 'positiveIncrease', 'hospitalizedCurrently']]
print(f"Read US data in {round(time.time() - start, 2)} seconds")
print(f"Rows read: {USdata.shape[0]}")

# Reading the high level municipality data 
start = time.time()
dMunicipality = pd.read_csv('https://raw.githubusercontent.com/mpiktas/covid19lt/master/data/lt-covid19-tests.csv') 
print(f"Read municipality data in {round(time.time() - start, 2)} seconds")
print(f"Rows read: {dMunicipality.shape[0]}")

# Reading the micro level data for each patient
start = time.time()
dPatient = pd.read_csv('https://raw.githubusercontent.com/mpiktas/covid19lt/master/data/lt-covid19-individual.csv')
print(f"Read patient data in {round(time.time() - start, 2)} seconds")
print(f"Rows read: {dPatient.shape[0]}")

# Creating the folder to store data 
dataFolder = 'data'
if not os.path.exists(dataFolder):
    os.mkdir(dataFolder)

# Creating the current date folder to store the downloaded data frames
curDateFolder = f'{dataFolder}/{curDate}'
if not os.path.exists(curDateFolder):
    os.mkdir(curDateFolder)

# Saving the results to two separate raw data files
dMunicipality.to_csv(f"{curDateFolder}/municipality_data.csv", index=False)
dPatient.to_csv(f"{curDateFolder}/patient_data.csv", index=False)
USdata.to_csv(f"{curDateFolder}/US_data.csv", index=False)

print(f"Data saved in {curDateFolder}")

# Removing the objects from memory
del dMunicipality, dPatient, USdata 
gc.collect()