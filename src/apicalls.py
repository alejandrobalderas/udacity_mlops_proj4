from flask.globals import request
import requests
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://localhost:8000"


with open('config.json', 'r') as f:
    config = json.load(f)


output_file = os.path.join(config["output_model_path"], "apireturns.txt")
# Call each API endpoint and store the responses
# put an API call here

# In apicalls.py, call API’s to get the model predictions, accuracy score, summary statistics, and diagnostics that are returned by the API endpoints
print(f"Testing api calls")
response1 = requests.post(
    f"{URL}/prediction?filename=testdata/testdata.csv")
response2 = requests.get(f"{URL}/scoring")
response3 = requests.get(f"{URL}/summarystats")
response4 = requests.get(f"{URL}/diagnostics")

# #combine all API responses
# responses = #combine reponses here

# The apicalls.py script needs to combine these API outputs and write the combined outputs to the workspace, to a file called apireturns.txt.
with open(output_file, "w+") as output:
    output.write(f"Predictions: {response1.text} \n")
    output.write(f"Scoring: {response2.text}\n")
    output.write(f"Summary Statistics: {response3.text}\n")
    output.write(f"Diagnostics: {response4.text}")
# write the responses to your workspace
