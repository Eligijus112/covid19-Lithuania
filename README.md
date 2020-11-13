# covid19-Lithuania

Analysis of COVID situation in Lithuania using local data

# Activating the virtual environment

For the creation of virtual environments (VE) anaconda is used. You can download anaconda via here: https://www.anaconda.com/distribution/

```
# Creating the base for the environment
conda create python=3.7 --name covid

# Activating it 
conda activate covid

# Populating the environment with packages
pip install -r requirements.txt
```

# Building the docker container 

The environment creation is automated through docker. 

```
sudo docker-compose build 
sudo docker-compose up 
```

This will create a jupyter instance that can be accesed via port 8080 on your local machine. 

# Scripts

## dataDownload.py 

Script that downloads the newest data from the links 

https://raw.githubusercontent.com/mpiktas/covid19lt/master/data/lt-covid19-tests.csv

and 

https://raw.githubusercontent.com/mpiktas/covid19lt/master/data/lt-covid19-individual.csv

Usage: 

```
python dataDownload.py
```

