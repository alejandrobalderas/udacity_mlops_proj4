import os
import subprocess

import json

from ingestion import get_csv_files
from scoring import score_model
from training import train_model
from app import app
import signal

with open('config.json', 'r') as f:
    config = json.load(f)

path_input_folder = config["input_folder_path"]
path_ingested_files = os.path.join(
    config["prod_deployment_path"], "ingestedfiles.txt")
path_last_score = os.path.join(
    config["prod_deployment_path"], "latestscore.txt")
path_model = os.path.join(
    config["prod_deployment_path"], "trainedmodel.pkl")
path_finaldata = os.path.join(config["output_folder_path"], "finaldata.csv")

# Check and read new data
# first, read ingestedfiles.txt
with open(path_ingested_files, "r") as my_file:
    ingested_files = my_file.read()

ingested_files = {f for f in ingested_files.split("\n") if ".csv" in f}

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
input_files = set(get_csv_files(config["input_folder_path"]))

# Check for the presence of non-ingested data in the /sourcedata/ folder

# The fullprocess.py script should call the deployment.py script only under certain conditions: when there is new data ingested, AND when there is model drift.
# if input_files - ingested_files:
if True:
    print(f"Starting Rest API")
    api = subprocess.Popen(["python3", "app.py"])
    subprocess.run(["python3", "ingestion.py"])
    print(f"New data found")

    # Checking for model drift
    # check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    with open(path_last_score, "r") as my_file:
        last_score = float(my_file.read())
    current_score = score_model(path_finaldata, path_model)

    # Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the process here

    # Check for whether the most recent model performs better than the previously deployed model
    # if current_score < last_score:
    if True:
        # Drift has occured
        # Re-deployment
        # if you found evidence for model drift, re-run the deployment.py script
        subprocess.run(["python3", "training.py"])
        subprocess.run(["python3", "scoring.py"])
        subprocess.run(["python3", "deployment.py"])
        ##################Diagnostics and reporting
        # run diagnostics.py and reporting.py for the re-deployed model
        # Run REST API
        subprocess.run(["python3", "apicalls.py"])
        subprocess.run(["python3", "reporting.py"])

    # os.killpg(os.getpgid(rest_api.pid), signal.SIGTERM)
    api.terminate()
