import os 
from get_data import read_parms, get_data
import argparse

def load_and_save(config_path):
    config = read_parms(config_path)
    df = get_data(config_path)
    new_cols = [col.replace(" ","_") for col in df.columns]
    raw_data_path = config['load_data']['raw_data_csv']
    df.to_csv(raw_data_path , sep = ',',index = False , header = new_cols)
    


if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="parms.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path = parsed_args.config)