from flask import Flask, session, jsonify, request, make_response
import pandas as pd
import numpy as np
from src.diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list, check_missing_data
from src.scoring import score_model
import json
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = os.environ.get("secret_key")

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # Accepts two arguments and constructs a data frame from them
    data_path = request.args.get("filename", default=None)
    data = request.args.get("data", default=None)

    # creates a df depending on which parameter was passed in the request
    df = pd.read_csv(data_path) if data_path else pd.DataFrame(data)
    if not data_path and not data:
        return make_response(jsonify("No Data passed"), 400)
    res = str(model_predictions(df))
    return make_response(jsonify(res), 200)


@ app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    # An endpoint for scoring needs to provide model scores based on test datasets and models (found in /testdata/).
    # add return value (a single F1 score number)
    return make_response(jsonify(score_model()), 200)


@ app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # An endpoint for summary statistics needs to provide summary statistics for the ingested data (found in the directory specified by the output_folder_path in config.json)
    return make_response(jsonify(dataframe_summary().to_json()), 200)

# Diagnostics Endpoint


@ app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    # An endpoint for diagnostics needs to provide diagnostics for the ingested data (found in the directory specified by the output_folder_path in config.json). The diagnostics should include timing, dependency checks, and missing data checks.
    timing = execution_time()
    na_vals = check_missing_data()
    packages = outdated_packages_list()
    res = {
        "timing": timing,
        "na_vals": list(na_vals),
        "packages": packages.to_json()
    }
    # add return value for all diagnostics
    return make_response(jsonify(res), 200)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
