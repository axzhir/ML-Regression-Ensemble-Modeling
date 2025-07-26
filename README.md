# Machine Learning Project: Regression & Ensemble Modeling with Airbnb Data 🏡📊

## 📚 Overview  
This project focuses on solving a regression problem using multiple regression models — including individual regressors and ensemble methods — applied to the Airbnb NYC listings dataset. You will train, evaluate, and compare different models to identify the best performer.

---

## 🎯 Objectives  
By the end of this project, you will:

- Build and prepare your dataset for regression modeling  
- Create labeled examples and split data into training and testing sets  
- Train and evaluate two individual regression models  
- Train and evaluate three ensemble regression models  
- Visualize and compare model performances to select the best approach  

---

## 🧠 Problem Statement  
Predict a continuous outcome from Airbnb listings — such as listing price or availability — using various regression algorithms and ensembles to improve accuracy.

---

## 🛠️ Project Steps  

### 1. Data Preparation  
- Load and clean the Airbnb listings dataset  
- Define the regression label and features  
- Split the dataset into training and test subsets  

### 2. Model Training & Evaluation  
- Train two individual regressors (e.g., Linear Regression, Decision Tree Regressor)  
- Train three ensemble regressors (e.g., Random Forest, Gradient Boosting, AdaBoost)  
- Evaluate models using regression metrics like RMSE, MAE, and R²  

### 3. Visualization  
- Plot predictions vs. actual values  
- Compare error metrics across models in visual form  

---

## 💻 Sample Code Snippet  
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv('data/airbnbData_Prepared.csv')

# Define features and target
X = df.drop(columns=['price'])
y = df['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42)
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")
