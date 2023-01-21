#importing the required libraries
import numpy as np 
import os
import warnings
import pandas as pd 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from get_data import read_params
import joblib
import argparse
import json

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config['split_data']['test_path']
    train_data_path = config['split_data']['train_path']
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    n_estimators = config['estimators']['n_estimators']
    criterion = config['estimators']['criterion']
    max_depth= config['estimators']['max_depth']
    min_samples_split= config['estimators']['min_samples_split']
    min_samples_leaf= config['estimators']['min_samples_leaf']
    min_weight_fraction_leaf= config['estimators']['min_weight_fraction_leaf']
    max_features= config['estimators']['max_features']

    target = [config['base']['target_col']]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    #building model

    model=RandomForestRegressor(
        n_estimators= n_estimators,
        criterion= criterion,
        max_depth= max_depth,
        min_samples_split= min_samples_split,
        min_samples_leaf= min_samples_leaf,
        min_weight_fraction_leaf= min_weight_fraction_leaf,
        max_features= max_features)
    model.fit(train_x,train_y.values.ravel())

    predicted_test = model.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_test)

    metrics_file = config['reports']['metrics']
    params_file = config['reports']['params']

    with open(metrics_file,"w") as f:
        metrics ={"rmse": rmse,'mae':mae,'r2':r2}
        json.dump(metrics,f,indent=4)


    with open(params_file,"w") as k:
        params ={"n_estimators": n_estimators,"criterion" : criterion,'max_depth': max_depth,'min_samples_split': min_samples_split,'min_samples_leaf': min_samples_leaf,'min_weight_fraction_leaf': min_weight_fraction_leaf,'max_features': max_features}
        json.dump(params,k,indent=4)


        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.joblib")

        joblib.dump(model, model_path)


if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path = parsed_args.config)