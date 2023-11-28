import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


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
x = df[important_features]
y = df['quality']
#y = df['quality_category']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(f"X_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Create a random forest classification model
model = RandomForestClassifier(random_state = 42)

# Testing 10-fold cross validation
scores = cross_val_score(model, x_train, y_train, cv = 5)

# Fit the model to the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)

# Calculate the accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion matrix:\n{confusion}")

x_quality = x_test.iloc[:, -1]
x_range = np.linspace(x_quality.min(), x_quality.max(), len(y_test))

plt.figure(figsize = (10, 10))
plt.scatter(y_test, y_pred, c = 'red')
p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.axis('equal')
plt.title("Classification Predicted vs Actual Quality")
plt.savefig("Classification_graph.png")
plt.close()

sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Classification Confusion Matrix")
plt.savefig('Classification_con_matrix.png')

