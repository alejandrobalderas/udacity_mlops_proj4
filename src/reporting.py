import pickle
from pandas.core.algorithms import mode
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data = os.path.join(config["test_data_path"], "testdata.csv")
cm_path = os.path.join(config["output_model_path"], "confusionmatrix.png")

# Function for reporting


def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace

    # Students will create a function in reporting.py that generates a confusion matrix that shows the accuracy of the model on test data (found in /testdata/).

    data = pd.read_csv(os.path.join(test_data))
    y_true = data.pop("exited")

    y_pred = model_predictions(data)
    cm = metrics.confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm)
    sns.heatmap(df_cm, cmap="YlGnBu")
    plt.savefig(cm_path)


if __name__ == '__main__':
    print(f"Building Report")
    score_model()
