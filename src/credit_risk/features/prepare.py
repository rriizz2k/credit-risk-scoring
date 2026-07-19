import numpy as np
import pandas as pd

def split_features_target(df):
    df['class'] = df['class'].map({"good":0, "bad":1})

    x = df.drop(columns=['class'])
    y = df['class']


    return x, y

def add_engineered_features(df):
    df['credit_amount_log'] = np.log1p(df['credit_amount'])
    #credit_amount_per_month
    df['credit_amount_per_month'] = df['credit_amount']/df['duration']


    return df