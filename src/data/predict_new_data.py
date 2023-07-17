import joblib
import pandas as pd

# Use the trained model to make predictions on new data
# Load the trained model from file
trained_model = joblib.load("../../models/trained_model.pkl")

# Load new data to make predictions on
new_data = pd.read_csv("C:/Users/Maru/Documents/REPASO/capitalbikes/data/raw/new_data.csv")

new_data.drop(['dteday', 'rentals'], axis=1, inplace=True)

# Create dummies for categorical features - encode
cat_cols = ["season","yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
for col in cat_cols:
    new_data[col] = new_data[col].astype('category')
    dummies = pd.get_dummies(new_data[col], prefix=col, drop_first=False)
    new_data = pd.concat([new_data, dummies], axis=1)
    #new_data.drop(col, axis=1, inplace=True)

# Make predictions on the new data
predictions = trained_model.predict(new_data)

# Print the predictions
rounded_predictions = [round(prediction) for prediction in predictions]
print(rounded_predictions)