from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# load the california housing dataset
data = fetch_california_housing()
# initialize the X array and the vector y
X, y = data.data, data.target

# Part 3.1:
# split the dataset into training and testing sets (the split is 70% for training and 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create a Linear Regression object from the class we created
model = LinearRegression()

# fit the model using the method from the linear regression class
model.fit(X_train, y_train)

# evaluate the model on the training set using the method from the linear regression class
y_test_pred, mse_test = model.evaluate(X_test, y_test)

# calculate and print the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse_test)
print("\nMy Linear Regression - Root Mean Squared Error (RMSE):", rmse)

# Part 3.2:Στο
num_repetitions = 20
rmse_values = []

# we will repeat the same procedure 20 times and we will keep the rmse values in an array
for _ in range(num_repetitions):
    # split the dataset into training and testing sets (the split is 70% for training and 30% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model.fit(X_train, y_train)

    y_test_pred, mse_test = model.evaluate(X_test, y_test)

    rmse = np.sqrt(mse_test)
    rmse_values.append(rmse)

# calculate and print the mean and standard deviation of RMSE values
mean_rmse = np.mean(rmse_values)
std_rmse = np.std(rmse_values)
print("My Linear Regression - Mean RMSE:", mean_rmse)
print("My Linear Regression - Standard Deviation of RMSE:", std_rmse)

# Part 3.3:

# split the dataset into training and testing sets (the split is 70% for training and 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create Linear Regression object from the class given in by scikit-learn
sklearn_model = SklearnLinearRegression()

# fit the model using the method from the linear regression class
sklearn_model.fit(X_train, y_train)

# evaluate the model on the training set using the method from the linear regression class
y_test_pred_sklearn = sklearn_model.predict(X_test)

# calculate and print the Root Mean Squared Error (RMSE)
rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_test_pred_sklearn))
print("\nScikit-learn Linear Regression - Root Mean Squared Error (RMSE):", rmse_sklearn)

# initialize variables to store RMSE values
rmse_values_sklearn = []

# we will make again 20 repetitions this time using the model from the scikit-learn
for _ in range(num_repetitions):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    sklearn_model.fit(X_train, y_train)

    y_test_pred_sklearn = sklearn_model.predict(X_test)

    # calculate the RMSE
    rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_test_pred_sklearn))

    rmse_values_sklearn.append(rmse_sklearn)

# calculate the mean and standard deviation of RMSE values
mean_rmse_sklearn = np.mean(rmse_values_sklearn)
std_rmse_sklearn = np.std(rmse_values_sklearn)

# print the results
print("Scikit-learn Linear Regression - Mean RMSE:", mean_rmse_sklearn)
print("Scikit-learn Linear Regression - Standard Deviation of RMSE:", std_rmse_sklearn)

