from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

from shutil import copy, copyfile


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
mdl_output_path = config["output_model_path"]


# function for deployment
def copy_for_deployment():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    files_to_copy = ["trainedmodel.pkl",
                     "latestscore.txt", "ingestedfiles.txt"]
    src_files = [os.path.join(mdl_output_path, cur_file)
                 for cur_file in files_to_copy[:-1]]
    src_files += [os.path.join(dataset_csv_path, files_to_copy[-1])]

    dst_files = [os.path.join(prod_deployment_path, f) for f in files_to_copy]

    # The deployment.py script should copy the trained model, the F1 score, and the ingested file record to a production deployment directory.
    for src, dst in zip(src_files, dst_files):
        print(f"copying: {src} to: {dst}")
        copyfile(src, dst)


if __name__ == "__main__":
    copy_for_deployment()
