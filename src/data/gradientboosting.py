import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV

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

# Split the X into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = GradientBoostingRegressor()

# Define the hyperparameters to tune
param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

# Define the evaluation metrics
scoring = ["neg_mean_squared_error", "r2"]

# Perform the grid search with cross-validation
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring=scoring, refit='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
#print("Grid search results: ", grid_search.cv_results_)

# Print the results
#print("Best hyperparameters: ", grid_search.best_estimator_.get_params())
print("Best parameters: ", grid_search.best_params_)
#print("Best MSE score: ", -grid_search.best_score_)
#print("Best R^2 score: ", grid_search.best_score_)
print("Best MSE score: ", -grid_search.cv_results_['mean_test_neg_mean_squared_error'][grid_search.best_index_])
print("Best R^2 score: ", grid_search.cv_results_['mean_test_r2'][grid_search.best_index_])