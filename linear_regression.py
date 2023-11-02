import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

# Path to my csv file
path = "/project/arcc-students/klofthus/p_ML/warmup-FidelisPyro/winequality-red.csv"

# Read csv file into a pandas dataframe
df = pd.read_csv(path, delimiter = ";")

# Calculate the correlation matrix
corr_matrix = df.corr()

# Select the most important features relative to the target
important_features = corr_matrix['quality'].sort_values(ascending = False)[1:6].index.tolist()

# Trying combinations of features with PolynomialFeatures
poly = PolynomialFeatures(degree = 2, include_bias = False)

# Create a new dataframe without the quality column
#X = df.drop('quality', axis = 1)
x = df[important_features]
y = df['quality']
x_poly = poly.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2, random_state = 42)
print(f"X_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Create a linear regression model
model = LinearRegression()

# Testing 5-fold cross validation
scores = cross_val_score(model, x_train, y_train, cv = 10)

print(f"Cross Validation Scores: {scores}")
print(f"Mean Cross Validation Score: {np.mean(scores)}")

# Fit the model to the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)

# Calculate the mean squared error and r2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

plt.plot(y_test, y_pred)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Predicted vs Actual Quality")
plt.show()
