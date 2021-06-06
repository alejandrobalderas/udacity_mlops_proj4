import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import List


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def get_csv_files(dir: str) -> List[str]:
    return [f for f in os.listdir(os.path.join(os.getcwd(), dir)) if f.endswith(".csv")]


# Function for data ingestion
def save_ingested_files(files: List[str], output_file: str):
    with open(os.path.join(output_folder_path, output_file), "w+") as my_file:
        d = datetime.now()
        thetimenow = str(d.year) + '/'+str(d.month) + '/'+str(d.day)
        my_file.write(f"Date: {thetimenow}\n")
        for file in files:
            my_file.write(f"{file}\n")


def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file

    df = pd.DataFrame(columns=["corporation", "lastmonth_activity",
                      "lastyear_activity", "number_of_employees", "exited"])

    # Every file contained in the data folder needs to be read into Python.
    csv_files = get_csv_files(input_folder_path)
    for csv_file in csv_files:
        tmp_df = pd.read_csv(os.path.join(
            os.getcwd(), input_folder_path, csv_file))
        df = df.append(tmp_df)

    # Store the ingestion record in a file called “ingestedfiles.txt”.
    ingested_file = "ingestedfiles.txt"

    save_ingested_files(csv_files, ingested_file)

    # All files should be compiled into a pandas data frame and written to a csv file called “finaldata.csv”. De-dupe the compiled data frame before saving.
    df = df.drop_duplicates()
    df.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
