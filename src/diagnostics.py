
import pandas as pd
import timeit
import numpy as np
import os
import json
import pickle
import subprocess
from typing import List


with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv")
test_data_path = os.path.join(config['test_data_path'])
prod_path = config["prod_deployment_path"]
mdl_path = os.path.join(prod_path, "trainedmodel.pkl")


def model_predictions(data: pd.DataFrame) -> List:
    # Students will create a function for making predictions based on the deployed model and a dataset.
    df = data.copy()
    with open(mdl_path, "rb") as f:
        mdl = pickle.load(f)
    if "corporation" in data.columns:
        df.drop("corporation", axis=1, inplace=True)
    if "exited" in data.columns:
        df.drop("exited", axis=1, inplace=True)
    return list(mdl.predict(df))

# Function to get summary statistics


def dataframe_summary() -> pd.DataFrame:
    df = pd.read_csv(dataset_csv_path)
    ss = df.agg(["mean", "std", "median"])
    return ss


def check_missing_data() -> pd.Series:
    # Data integrity should be checked by measuring the percentage of NA values in each of the numeric dataset’s columns.
    df = pd.read_csv(dataset_csv_path)
    return df.isnull().sum() * 100 / len(df)


def measure_module_time(script: str, num_of_loops: int = 5) -> float:
    """
    Return the mean amount of time to execute a script
    """
    times = []
    for _ in range(num_of_loops):
        starttime = timeit.default_timer()
        subprocess.run(["python3", f"src/{script}"])
        timing = timeit.default_timer() - starttime
        times.append(timing)
    return np.mean(timing)


def execution_time():
    # Timing should be checked for both data ingestion and training in seconds.
    return [measure_module_time(script) for script in ["ingestion.py", "training.py"]]


def outdated_packages_list() -> pd.DataFrame:
    # All modules in requirements.txt need to have their latest versions and currently installed versions checked.
    df = pd.DataFrame(columns=["name", "current_version", "newest_version"])
    pkg_list = str(subprocess.check_output(["pip", "list", "--outdated"]))
    for idx, row in enumerate(pkg_list.split("\\n")[2:]):
        try:
            pkg, req_version, cur_version, _ = row.split()
            df.loc[idx] = (pkg, req_version, cur_version)
        except ValueError as e:
            # End of list
            continue
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
