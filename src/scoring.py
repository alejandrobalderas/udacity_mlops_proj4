from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], "testdata.csv")
model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl")
metric_output_path = os.path.join(
    config["output_model_path"], "latestscore.txt")


# Function for model scoring
def score_model(path_data: str = test_data_path, path_model: str = model_path) -> float:
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    with open(path_model, "rb") as f:
        mdl = pickle.load(f)
    data = pd.read_csv(path_data)
    X_test = data.copy().drop("corporation", axis=1)
    y_test = X_test.pop("exited").ravel()

    pred = mdl.predict(X_test)

    # Students should write a scoring script in the scoring.py starter file. Scoring should be performed using the F1 score.
    score = metrics.f1_score(y_test, pred)
    return score


def save_result(score: float):
    with open(metric_output_path, "w+") as metric_file:
        metric_file.write(str(score))


if __name__ == "__main__":
    print(f"Scoring model")
    score = score_model()
    # The scoring.py script should write the F1 score to a .txt file called latestscore.txt.
    print(f"Saving score")
    save_result(score)
