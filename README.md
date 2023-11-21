# Machine Learning and Deep Learning Notes


# Data Processing

- Importing Libraries: Essential Python libraries including NumPy, Matplotlib, and Pandas are imported for handling arrays, plotting graphs, and manipulating datasets respectively.

- Importing the Dataset: The process of importing a dataset using Pandas is demonstrated, focusing on separating it into independent features (`X`) and the dependent target variable (`y`).

- Splitting the Dataset: The dataset is split into Training and Test sets using scikit-learn's `train_test_split` function, an essential step for model evaluation.

- Handling Missing Data: Techniques for handling missing data are covered, highlighting the importance of proper data imputation.

- Encoding Categorical Data: The notebooks discuss encoding categorical variables to convert them into a machine-readable format, using methods like OneHotEncoder and LabelEncoder.

- Feature Scaling: Feature scaling methods, especially standardization, are emphasized to ensure all features contribute equally to the model's performance.

- Print Transformed Data: Finally, the transformed feature sets are printed to verify the effectiveness of the preprocessing steps.
---

# Machine Learning Regression Models

Regression models (both linear and non-linear) are used for predicting a real value, like salary for example. If your independent variable is time, then you are forecasting future values, otherwise your model is predicting present but unknown values. Regression technique vary from Linear Regression to SVR and Random Forests Regression.

## Table of Contents
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression

---

### Simple Linear Regression

Simple Linear Regression uses a single feature to predict a response, assuming a linear relationship between input and output.

#### Implementation
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Replace X and y with your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```
### Multiple Linear Regression

Multiple Linear Regression uses several explanatory variables to predict the outcome of a response variable, establishing a linear relationship between them.

#### Implementation
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Replace X and y with your dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

## Polynomial Regression

Polynomial Regression is suitable for modeling the non-linear relationship between the dependent and independent variables. It extends linear regression by introducing polynomial terms into the regression equation.

#### Implementation
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Example with degree 2 polynomial
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```



## Support Vector Regression (SVR)

SVR uses the same principles as SVM for classification but is applied to regression problems. It tries to fit the error within a certain threshold and can be effective in high-dimensional spaces.

#### Implementation
```python
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

model = SVR(kernel='rbf') # Other kernels can be used
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```



## Decision Tree Regression

Decision Tree Regression uses a decision tree to model the decisions made and to make predictions. It is useful for non-linear relationships that are hard to model with other techniques.

#### Implementation
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

model = DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

```

## Random Forest Regression

Random Forest Regression is an ensemble learning method. It builds multiple decision trees and merges them together to get a more accurate and stable prediction.

#### Implementation
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

model = RandomForestRegressor(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```