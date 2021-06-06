import os
import subprocess
import json
from src.ingestion import get_csv_files
from src.scoring import score_model

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

with open(path_ingested_files, "r") as my_file:
    ingested_files = my_file.read()
ingested_files = {f for f in ingested_files.split("\n") if ".csv" in f}
input_files = set(get_csv_files(config["input_folder_path"]))

# The fullprocess.py script should call the deployment.py script only under certain conditions: when there is new data ingested, AND when there is model drift.
if input_files - ingested_files:
    print(f"Found new input files")
    print(f"Starting Rest API")
    rest_api = subprocess.Popen(["python3", "src/app.py"])
    subprocess.run(["python3", "src/ingestion.py"])

    # check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    with open(path_last_score, "r") as my_file:
        last_score = float(my_file.read())
    current_score = score_model(path_finaldata, path_model)

    # Check for whether the most recent model performs better than the previously deployed model
    # Checking for model drift
    if current_score < last_score:
        print(f"Model drift ocurred. Retraining model")
        # Drift has occured
        subprocess.run(["python3", "src/training.py"])
        subprocess.run(["python3", "src/scoring.py"])
        subprocess.run(["python3", "src/deployment.py"])
        subprocess.run(["python3", "src/apicalls.py"])
        subprocess.run(["python3", "src/reporting.py"])

    rest_api.terminate()
