# Machine Learning and Deep Learning Notes


# Data Processing

- **Importing Libraries:** Essential Python libraries including NumPy, Matplotlib, and Pandas are imported for handling arrays, plotting graphs, and manipulating datasets respectively.

- **Importing the Dataset:** The process of importing a dataset using Pandas is demonstrated, focusing on separating it into independent features (`X`) and the dependent target variable (`y`).

- **Splitting the Dataset:** The dataset is split into Training and Test sets using scikit-learn's `train_test_split` function, an essential step for model evaluation.

- **Handling Missing Data:** Techniques for handling missing data are covered, highlighting the importance of proper data imputation.

- **Encoding Categorical Data: **The notebooks discuss encoding categorical variables to convert them into a machine-readable format, using methods like OneHotEncoder and LabelEncoder.

- **Feature Scaling:** Feature scaling methods, especially standardization, are emphasized to ensure all features contribute equally to the model's performance.

- **Print Transformed Data:** Finally, the transformed feature sets are printed to verify the effectiveness of the preprocessing steps.
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
**How It Works:** Simple Linear Regression models the relationship between a single independent variable and a dependent variable by fitting a linear equation. The equation is `y = β0 + β1*x`, where `y` is the dependent variable, `x` is the independent variable, and `β0` and `β1` are the model coefficients. The aim is to find the line that best fits the data, typically by minimizing the sum of squared differences between the observed and predicted values.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Replace X and y with your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```
- **When to Use**: Best for predicting an outcome with a single independent variable. Ideal for understanding the relationship between two continuous variables.
- **When Not to Use**: Not suitable for complex relationships or datasets with multiple features influencing the outcome.
  


### Multiple Linear Regression
**How It Works:** Multiple Linear Regression extends Simple Linear Regression to multiple independent variables. The model fits a linear equation to the data, represented as `y = β0 + β1*x1 + β2*x2 + ... + βn*xn`.  Here, `y` is the dependent variable, `x1, x2, ..., xn` are the independent variables, and `β0, β1, ..., βn` are the coefficients. The model seeks a hyperplane that best fits the data in a multi-dimensional feature space.
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Replace X and y with your dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```
- **When to Use**: Effective when multiple variables affect the dependent variable. Useful in cases like predicting house prices based on various features.
- **When Not to Use**: Ineffective for non-linear relationships. Not recommended when independent variables are highly correlated (multicollinearity).



## Polynomial Regression

**How It Works:** Polynomial Regression is used for modeling non-linear relationships between the independent and dependent variables. It transforms the original features into polynomial features of a given degree and then applies linear regression. The model is represented as `y = β0 + β1*x + β2*x^2 + ... + βn*x^n`. This approach allows the model to fit a wide range of curvatures in the data, making it versatile for non-linear datasets.

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
- **When to Use**: Suitable for modeling non-linear relationships. Useful in cases where the relationship between variables is curvilinear.
- **When Not to Use**: Avoid for simple linear relationships. Can lead to overfitting if the polynomial degree is set too high.





## Support Vector Regression (SVR)

**How It Works:** SVR applies the principles of Support Vector Machines (SVM) to regression problems. It focuses on fitting as many instances as possible within a certain threshold (epsilon) from the actual value, while also trying to minimize model complexity. The core idea is to find a function (linear or non-linear) that has at most an ε deviation from the actual values for all the training data, and at the same time is as flat as possible.
```python
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

model = SVR(kernel='rbf') # Other kernels can be used
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
- **When to Use**: Effective in high-dimensional spaces and for datasets with non-linear relationships. Robust against outliers.
- **When Not to Use**: Not ideal for very large datasets as it can become computationally intensive. Performance can significantly depend on the correct kernel choice.

## Decision Tree Regression
**How It Works:** Decision Tree Regression uses a decision tree to model the decision-making process. It splits the data into subsets based on different values of the features. These splits form a tree structure with nodes and leaves. Each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents the predicted value. The path from the root to the leaf represents classification rules.
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

model = DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
- **When to Use**: Good for complex datasets with non-linear relationships. Easy to interpret and understand.
- **When Not to Use**: Prone to overfitting, especially with noisy data. Not suitable for extrapolation beyond the range of the training data.



## Random Forest Regression

**How It Works:** Random Forest Regression is an ensemble method that builds multiple decision trees during training and outputs the average of the predictions of the individual trees. It introduces randomness into the model while building the trees, either by sampling the data points (bagging) or by sampling the features. This randomness helps to make the model more robust than a single decision tree and less likely to overfit on the training data.
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

model = RandomForestRegressor(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
- **When to Use**: Excellent for dealing with overfitting in decision trees. Works well with a large number of features and complex, non-linear relationships.
- **When Not to Use**: Not the best choice for very high dimensional, sparse data. Can be computationally expensive and time-consuming for training and predictions.





