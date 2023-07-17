
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

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

# Use the best parameters from the grid search
params = {'learning_rate': 0.1, 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}

# Create the gradient boosting model with the best parameters
model = GradientBoostingRegressor(**params)

# Fit the model to the training data
model.fit(X_train, y_train)

# Export and save the model 
joblib.dump(model, "../../models/trained_model.pkl")

# Make predictions for the test data
y_pred = model.predict(X_test)

# Calculate the R-squared and mean squared error scores
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print the scores
print("R-squared score: {:.3f}".format(r2))
print("MSE score: {:.3f}".format(mse))

# Create a scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)
plt.plot([0, 800], [0, 800], '--k')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()

# Create a scatter plot of residuals vs predicted values
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.plot([0, 800], [0, 0], '--k')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.show()

# Create a histogram of residuals
plt.hist(residuals, bins=50)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Create a plot of feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

