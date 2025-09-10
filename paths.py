import os
from os import mkdir
from os.path import isdir, dirname, abspath
from datetime import date

# Get the directory where this script is located
script_dir = dirname(abspath(__file__))
bro = script_dir + "/"
docs = "/home/tank/Documents/"
data = bro + "Data/"
logs = bro + "Logs/"

def return_folder(folder):
    if not isdir(folder):
        # Create parent directories if they don't exist
        os.makedirs(folder, exist_ok=True)
    return folder

def today():
    return return_folder(data + date.today().strftime("%Y%m%d") + "/")

def oceanoptics():
    return return_folder(today() + "oceanoptics/")