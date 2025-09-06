import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def encode_ordinal(df, mapping):
    for col, mapper in mapping.items():
        df[col] = df[col].map(mapper)
    return df

def select_features_by_mutual_info(X, y, percentile=20):
    from sklearn.feature_selection import SelectPercentile, mutual_info_regression
    selector = SelectPercentile(mutual_info_regression, percentile=percentile)
    selector.fit(X, y)
    return X.columns[selector.get_support()].tolist()

if __name__ == '__main__':
    print('Feature engineering utilities')
