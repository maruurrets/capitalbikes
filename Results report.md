# Report - Results
Implemented multiple regression models to predict the number of bikes that will be rented out on a given day.

### LINEAR REGRESSION MODEL
I encoded categorical features, but didnt normalize numerical features because this made R¨2 score be much reduced.
The R¨2 score is 0.7076.

After trying a linear regression model I tried some other regression models to see which one fitted the data better:

- RANDOM FOREST REGRESSOR

- GRADIENT BOOSTING REGRESSOR

- XGBOOST REGRESSION

- SUPPORT VECTOR REGRESSION

- RANDOM FOREST REGRESSOR

Gradient Boosting Regressor was the one that gave the best results in terms of MSE and R-squared with resultd of MSE: 60783.48, R2: 0.84
After performing a grid search with cross-validation to find the best hyperparameters, we achieved an R-squared value of 0.856 and a mean squared error of 67905.947.

Best parameters:  {'learning_rate': 0.1, 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}


