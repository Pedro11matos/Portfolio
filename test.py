import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE

listings = pd.read_csv('listings_processed.csv')

# Separate the target variable from the independent variables
X = listings.drop('price', axis=1)
y = listings['price']

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)

reg = LinearRegression(n_jobs=-1)

score = 0

for i in range(1, len(X.columns)):
    rfe = RFE(reg, n_features_to_select=i)

    # Fit the model
    rfe.fit(x_train, y_train)

    # Test the model
    y_pred = rfe.predict(x_test)
    temp_score = r2_score(y_test, y_pred)

    if temp_score > score:
        score = temp_score
        # Extract the selected features
        selected_features = x_train.columns[rfe.support_]

# Evaluate model performance
X_selected = X[selected_features]

# Train your model on X_selected and y
reg.fit(X_selected, y)

y_pred = reg.predict(x_test)

print(r2_score(y_pred, y_test))
print(X_selected.columns)
