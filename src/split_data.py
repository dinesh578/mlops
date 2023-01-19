#split the raw data
# save the data in data/processed folder

import os
import argparse
import pandas as pd
from  sklearn.model_selection import train_test_split
from get_data import read_parms

def split_and_save_data(config_path):
    config = read_parms(config_path)
    train_data = config['split_data']['train_path']
    test_data =config['split_data']['test_path']
    split_ratio = config['split_data']['test_size']
    raw_data = config['load_data']['raw_data_csv']
    random_state = config["base"]["random_state"]

    df =pd.read_csv(raw_data,sep=",",index_col=0)
    train,test = train_test_split(df,test_size=split_ratio,random_state=random_state)

    train.to_csv(train_data,sep=",",encoding="utf-8",index=False)
    test.to_csv(test_data,sep=",",encoding="utf-8",index=False)

if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="parms.yaml")
    parsed_args = args.parse_args()
    split_and_save_data(config_path = parsed_args.config)
    
