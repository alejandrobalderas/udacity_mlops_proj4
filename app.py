from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis
# import predict_exited_from_saved_model
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list, check_missing_data
from scoring import score_model
import json
import os


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # An endpoint for model predictions needs to return predictions from the deployed model (found in the directory specified in the prod_deployment path in config.json) for an input dataset (passed to the endpoint as an input)
    # call the prediction function you created in Step 3
    data_path = request.args.get("filename")
    df = pd.read_csv(data_path)
    # add return value for prediction outputs
    return str(model_predictions(df))

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    # check the score of the deployed model
    # An endpoint for scoring needs to provide model scores based on test datasets and models (found in /testdata/).
    return str(score_model())  # add return value (a single F1 score number)

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    # return a list of all calculated summary statistics

    # An endpoint for summary statistics needs to provide summary statistics for the ingested data (found in the directory specified by the output_folder_path in config.json)
    return dataframe_summary().to_json()

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    # An endpoint for diagnostics needs to provide diagnostics for the ingested data (found in the directory specified by the output_folder_path in config.json). The diagnostics should include timing, dependency checks, and missing data checks.
    # check timing and percent NA values
    timing = execution_time()
    na_vals = check_missing_data()
    packages = outdated_packages_list()
    res = {
        "timing": timing,
        "na_vals": list(na_vals),
        "packages": packages.to_json()
    }
    return res  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
