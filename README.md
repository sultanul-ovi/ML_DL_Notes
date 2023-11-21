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

