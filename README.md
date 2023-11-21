# Machine Learning and Deep Learning Notes



# Machine Learning Regression Models
---
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

#### Overview
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

### Multiple Linear Regression

#### Overview
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

### Overview
Polynomial Regression is suitable for modeling the non-linear relationship between the dependent and independent variables. It extends linear regression by introducing polynomial terms into the regression equation.

### Implementation in Python
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




## Support Vector Regression (SVR)

### Overview
SVR uses the same principles as SVM for classification but is applied to regression problems. It tries to fit the error within a certain threshold and can be effective in high-dimensional spaces.

### Implementation in Python
```python
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

model = SVR(kernel='rbf') # Other kernels can be used
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)




## Decision Tree Regression

### Overview
Decision Tree Regression uses a decision tree to model the decisions made and to make predictions. It is useful for non-linear relationships that are hard to model with other techniques.

### Implementation in Python
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

model = DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)



## Random Forest Regression

### Overview
Random Forest Regression is an ensemble learning method. It builds multiple decision trees and merges them together to get a more accurate and stable prediction.

### Implementation in Python
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

model = RandomForestRegressor(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
