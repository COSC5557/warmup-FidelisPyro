import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Path to my csv file
path = "/mnt/c/Users/kylel/Programming/School/PracticalML/wine+quality/winequality-red.csv"

# Read csv file into a pandas dataframe
df = pd.read_csv(path, delimiter = ";")





# Calculate the correlation matrix
corr_matrix = df.corr()

# Select the most important features relative to the target
important_features = corr_matrix['quality'].sort_values(ascending = False)[1:6].index.tolist()

# Create a new dataframe without the quality column
#X = df.drop('quality', axis = 1)
X = df[important_features]
y = df['quality']
#y = df['quality_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Create a dictionary of hyperparameters to search
"""
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
"""

# Create a random forest classification model
model = RandomForestClassifier(random_state = 42)

"""
# Search for the best hyperparameters
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
"""

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

#print(f"Best parameters: {grid_search.best_params_}")

#best_model = grid_search.best_estimator_
#y_pred = best_model.predict(X_test)


# Calculate the mean squared error and r2 score
#mse = mean_squared_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)

#print(f"Mean Squared Error: {mse}")
#print(f"R2 Score: {r2}")

print(f"Accuracy: {accuracy}")
print(f"Confusion matrix:\n{confusion}")

sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Confusion Matrix")
plt.savefig('Confusion_matrix.png')

