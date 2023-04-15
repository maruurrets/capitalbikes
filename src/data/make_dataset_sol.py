import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the Xset
df = pd.read_csv("../../data/raw/daily-bike-share.csv")

# Define the features and the target
X = df.drop(['rentals', 'dteday'],axis=1)
y = df['rentals']

# Feature engineer
## Convert columns to categorical type
cat_cols = ["season","yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
for col in cat_cols:
    X[col] = X[col].astype('category')

## Create dummies for categorical features - encode
X = pd.get_dummies(X, columns=["season","yr", "mnth", "holiday", "weekday", "workingday", "weathersit"])

def compare_algorithms(X, y):

    # Split the X into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a dictionary to store the evaluatio scores for each algorithm
    scores = {}

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    scores["Linear Regression"] = mse

    # Random Forest Regression
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    scores["Random Forest"] = mse

    # Gradient Boosting Regression
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    scores["Gradient Boosting"] = mse

    # XGBoost Regression
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    scores["XGBoost"] = mse

    return scores

scores = compare_algorithms(X, y)
print(scores)

# To find the best model 
best_model = min(scores, key=scores.get)
print("Best model: ",best_model)


   