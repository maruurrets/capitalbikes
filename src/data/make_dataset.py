import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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

# Split the X into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training X
model.fit(X_train, y_train)

# Evaluate the model on the test X
test_score = model.score(X_test, y_test)
print("Test R¨2:", test_score)