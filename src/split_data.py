#split the raw data
# save the data in data/processed folder

import os
import argparse
import pandas as pd
from  sklearn.model_selection import train_test_split
from get_data import read_parms
#import category_encoders as ce

def encoded_data(config_path):
    config = read_parms(config_path)
    encoded_data = config['load_data']['raw_data_csv']
    df1 =pd.read_csv(encoded_data,sep=",")
    print(df1.shape)
    #droping the unnamed column
    df1=df1.iloc[:, 1:]
    # Treating the missing values with mean
    df1['depth'].fillna(df1['depth'].mean(), inplace=True)
    #checking and removing the duplicates
    print(f'before removing duplicates{df1.shape}')
    df1.drop_duplicates(inplace=True)
    print(f'after removing duplicates{df1.shape}')
    df1=df1[~((df1['x']==0)|(df1['y']==0)|(df1['z']==0))]
    df1['y'].values[df1['y'].values>50]=df1['y'].mode()[0]
    df1['z'].values[df1['z'].values>30]=df1['z'].mode()[0]
    
    #fit and transform train data encoding
    scale_mapper = {'Fair': 1, 'Good': 2,'Very Good': 3,'Premium':4,'Ideal': 5}
    df1["cut"] = df1["cut"].replace(scale_mapper)
    print(df1.columns)

    #Creating the dummy variables 
    colors_dummies=pd.get_dummies(df1['color'], drop_first=True)
    clarity_dummies=pd.get_dummies(df1['clarity'], drop_first=True)
    dummies=pd.concat([colors_dummies, clarity_dummies], axis=1)
    #combining data and dummies
    df1=pd.concat([df1, dummies], axis=1)
    #Droping the original variables
    df1.drop(['color', 'clarity'], axis=1, inplace=True)
    #droping the x,y,z variables
    df1.drop(['x', 'y', 'z'], axis=1, inplace=True)
    df1.to_csv(encoded_data,sep=",",encoding="utf-8",index=False)



def split_and_save_data(config_path):
    config = read_parms(config_path)
    train_data = config['split_data']['train_path']
    test_data =config['split_data']['test_path']
    split_ratio = config['split_data']['test_size']
    raw_data = config['load_data']['raw_data_csv']
    random_state = config["base"]["random_state"]

    df =pd.read_csv(raw_data,sep=",")
    train,test = train_test_split(df,test_size=split_ratio,random_state=random_state)

    train.to_csv(train_data,sep=",",encoding="utf-8",index=False)
    test.to_csv(test_data,sep=",",encoding="utf-8",index=False)

if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="parms.yaml")
    parsed_args = args.parse_args()
    encoded_data(config_path = parsed_args.config)
    split_and_save_data(config_path = parsed_args.config)
    
