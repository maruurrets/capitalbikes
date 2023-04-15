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

# Extract the column names
df.columns

# Define the features and the target
X = df.drop(['rentals', 'dteday'],axis=1)
y = df['rentals']

# Feature engineer
## Convert columns to categorical type
cat_cols = ["season","yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
for col in cat_cols:
    X[col] = X[col].astype('category')

## Create dummies for categorical features - encode
dummies_cols = ["season","yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
for col in dummies_cols:
    dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
    X = pd.concat([X, dummies], axis=1)

X.info()

def compare_models():

    # Split the X into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regression": RandomForestRegressor(),
        "Gradient Boosting Regression": GradientBoostingRegressor(),
        "Support Vector Regression": SVR(),
        "Neural Network Regression": MLPRegressor(hidden_layer_sizes=(100, 50))
    }

    # Train and evaluate the models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R2": r2}

    # Print the results
    for name, result in results.items():
        print("{} - MSE: {:.2f}, R2: {:.2f}".format(name, result["MSE"], result["R2"]))

    # Return the model with the best MSE
    best_model_name = min(results, key=lambda x: results[x]["MSE"])
    print(best_model_name)
    #return models[best_model_name]

best_model = compare_models()
    