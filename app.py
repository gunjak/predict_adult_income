from flask import Flask, request
import sys

import pip
from income_prediction.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from  income_prediction.logger import logging
from  income_prediction.exception import IncomeException
import os, sys
import json
from  income_prediction.config.configuration import Configuration
from  income_prediction.constant import *
from  income_prediction.pipeline.pipeline import Pipeline
from  income_prediction.entity.income_predictor import IncomePredictor, IncomeData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "income_prediction"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


from income_prediction.logger import get_log_dataframe

INCOME_DATA_KEY = "income_data"
PREDICTION_VALUE_KEY = "wages"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'income_prediction'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("income_prediction", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
        try:
            return render_template('index.html')
        except Exception as e:
            return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    experiment_df = pipeline.get_experiments_status()
    context = {
            "experiment": experiment_df.to_html(classes='table table-striped col-12')
        }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        INCOME_DATA_KEY: None,
        PREDICTION_VALUE_KEY: None
    }

    if request.method == 'POST':
        age= int(request.form['age'])
        workclass = request.form['workclass']
        fnlwgt = int(request.form['fnlwgt'])
        education_num = int(request.form['education_num'])
        marital_status = request.form['marital_status']
        occupation = request.form['occupation']
        relationship = request.form['relationship']
        race = request.form['race']
        sex = request.form['sex']
        capital_gain = int(request.form['capital_gain'])
        capital_loss= int(request.form['capital_loss'])
        hours_per_week= int(request.form['hours_per_week'])
        native_country= request.form['native_country']

        income_data = IncomeData(age=age,
                                  workclass=workclass,
                                  fnlwgt=fnlwgt,
                                  education_num=education_num,
                                  marital_status=marital_status,
                                  occupation=occupation,
                                  relationship=relationship,
                                  race=race,
                                  sex=sex,
                                  capital_gain=capital_gain,
                                  capital_loss=capital_loss,
                                  hours_per_week=hours_per_week,
                                  native_country=native_country
                                   )
        income_df = income_data.get_income_input_data_frame()
        income_predictor = IncomePredictor(model_dir=MODEL_DIR)
        prediction_value = income_predictor.predict(X=income_df)
        context = {
            INCOME_DATA_KEY: income_data.get_income_data_as_dict(),
            PREDICTION_VALUE_KEY: prediction_value,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
      app.run()