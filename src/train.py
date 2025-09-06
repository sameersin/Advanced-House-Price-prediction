import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model(train_csv, features, target='saleprice', model_path='artifacts/model.pkl'):
    df = pd.read_csv(train_csv)
    X = df[features]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    r2 = r2_score(y_val, preds)
    joblib.dump(model, model_path)
    print(f'RMSE: {rmse:.4f}, R2: {r2:.4f}')
    return model

if __name__ == '__main__':
    # Example usage: update `features` with the selected feature names
    features = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF']
    train_model('data/train.csv', features)
