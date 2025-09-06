import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def drop_high_missing(df, threshold=0.8):
    missing = (df.isnull().mean())
    cols = missing[missing > threshold].index.tolist()
    return df.drop(columns=cols)

def fill_numeric_with_mean(df, cols):
    for c in cols:
        df[c] = df[c].fillna(df[c].mean())
    return df

if __name__ == '__main__':
    print('Preprocessing utilities')
