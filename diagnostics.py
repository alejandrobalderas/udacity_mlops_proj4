
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle

import subprocess

from typing import List

from pandas.core.frame import DataFrame

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv")
test_data_path = os.path.join(config['test_data_path'])
prod_path = config["prod_deployment_path"]

mdl_path = os.path.join(prod_path, "trainedmodel.pkl")

# Function to get model predictions


def model_predictions(data: pd.DataFrame) -> List:
    # Students will create a function for making predictions based on the deployed model and a dataset.
    df = data.copy()
    with open(mdl_path, "rb") as f:
        mdl = pickle.load(f)
    # read the deployed model and a test dataset, calculate predictions
    if "corporation" in data.columns:
        df.drop("corporation", axis=1, inplace=True)
    if "exited" in data.columns:
        df.drop("exited", axis=1, inplace=True)

    return list(mdl.predict(df))

# Function to get summary statistics


def dataframe_summary() -> pd.DataFrame:
    # calculate summary statistics here
    # return #return value should be a list containing all summary statistics

    # Summary statistics (means, medians, and modes) should be checked for each numeric column.

    # 1 Read data in the "output_folder_path" in the config.json
    df = pd.read_csv(dataset_csv_path)
    # 2 Calculate stats for each column
    metrics = ["mean", "50%", "std"]
    ss = df.describe().loc[metrics]
    ss.rename(index={"50%": "median"}, inplace=True)
    # 3 Output list with all summary statistics
    return ss


def check_missing_data() -> pd.Series:
    # Data integrity should be checked by measuring the percentage of NA values in each of the numeric dataset’s columns.
    df = pd.read_csv(dataset_csv_path)
    return df.isnull().sum() * 100 / len(df)


# Function to get timings
def measure_module_time(script: str) -> float:
    starttime = timeit.default_timer()
    subprocess.run(["python3", script])
    timing = timeit.default_timer() - starttime
    return timing


def execution_time():
    # calculate timing of training.py and ingestion.py
    # return #return a list of 2 timing values in seconds

    # Timing should be checked for both data ingestion and training in seconds.
    return [measure_module_time(script) for script in ["ingestion.py", "training.py"]]


# Function to check dependencies

def package_current_version(name: str):
    version = str(subprocess.check_output(
        ["pip", "show", name])).split("\\n")[1]
    return version.split()[1]


def outdated_packages_list() -> pd.DataFrame:
    # All modules in requirements.txt need to have their latest versions and currently installed versions checked.
    df = pd.DataFrame(columns=["name", "current_version", "newest_version"])
    installed = str(subprocess.check_output(["pip", "list"])).split("\\n")[2:]
    for idx, pkg in enumerate(installed):
        try:
            pkg_name, pkg_version = pkg.split()
        except ValueError as e:
            # Last value of the output is not a package
            continue
        pkg_newest_version = package_current_version(pkg_name)
        df.loc[idx] = (pkg_name, pkg_version, pkg_newest_version)
    return df


if __name__ == '__main__':
    print(f"Getting diagnostics")
    data = pd.read_csv(dataset_csv_path)
    pred = model_predictions(data)
    print(pred)
    dataframe_summary()
    exec_times = execution_time()
    print(exec_times)
    p = outdated_packages_list()
    print(p)
