# Advanced House Price Prediction with Feature Engineering & ML

This project predicts house sale prices using the **Kaggle House Prices dataset**.  
It demonstrates a complete ML workflow: data cleaning, feature engineering, feature selection, model training, and evaluation.

---

## 🚀 Features
- **Data Preprocessing**
  - Handle missing values, outliers, and high-null columns.
  - Encode categorical variables.
  - Normalize & clean numerical features.
- **Feature Engineering**
  - Correlation analysis for redundant feature removal.
  - Mutual Information & percentile-based feature selection.
- **Modeling**
  - Regression models to predict `SalePrice`.
  - Comparison of model performance.
- **Evaluation**
  - Feature importance analysis and visualization.
  - Metrics: RMSE, R².

---

## 📂 Project Structure
```
HousePricePrediction/
│── data/
│   ├── train.csv
│   ├── test.csv
│── notebooks/
│   └── CS_559_Project_final_code.ipynb
│── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│── README.md
│── requirements.txt
```

---

## 🛠️ Setup & Usage

1. **Clone the repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/HousePricePrediction.git
   cd HousePricePrediction
   ```

2. **Create virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run notebook:**
   ```bash
   jupyter notebook notebooks/CS_559_Project_final_code.ipynb
   ```

4. **Or train from script:**
   ```bash
   python src/train.py
   ```

---

## 📊 Results
- Top predictive features include `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`.
- Regression models achieved strong accuracy in predicting house sale prices.

---

## 📌 Future Work
- Add hyperparameter tuning (Optuna/RandomSearch).
- Try ensemble models (XGBoost, LightGBM, CatBoost).
- Deploy with FastAPI for live predictions.

---

## 👨‍💻 Authors
- Team Project — CS 559  
