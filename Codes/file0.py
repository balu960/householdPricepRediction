# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # change the working directory to parent directory
    os.chdir("..")

# %%
data = pd.read_csv(
    "g:\projects_by_balu\python\householdPricepRediction\Data\sample_data_household_prices.csv"
)
# %%
# data has columns: ['ID', 'OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF',
#    'FullBath', 'HalfBath', 'GarageCars', 'GarageArea', 'SalePrice']

# %%
# get the type of each column
print(data.dtypes)

# %%
# get missing values in each column
print(data.isnull().sum())
# %%
# get number of unique values in each column
print(data.nunique())

# %%
# normalize the data using sklearn
data0 = data.copy()
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
data0 = pd.DataFrame(scaler.fit_transform(data0), columns=data0.columns)

# remove the ID column
data0 = data0.drop(columns=["ID"])

target = data0.SalePrice
data0 = data0.drop(columns=["SalePrice"])
# %%
# get the correlation matrix
correlation_matrix = data.corr()
print(correlation_matrix)
# %%
# get the correlation of each column with the target
correlation_with_target = correlation_matrix.SalePrice
print(correlation_with_target)
# %%

from scipy.stats import pearsonr

results = []

# Iterate over each column in the DataFrame except 'SalePrice'
for column in data.columns:
    if column != "SalePrice":
        # Calculate correlation coefficient and p-value
        correlation_coefficient, p_value = pearsonr(data[column], data["SalePrice"])

        results.append((column, correlation_coefficient, p_value))

results_df = pd.DataFrame(
    results, columns=["Variable", "Correlation Coefficient", "P-value"]
)

print(results_df)

# %%
# split the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data0, target, test_size=0.2, random_state=23
)
# %%
# fit a linear regression model as a baseline
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

# evaluate the model
print(model.score(X_test, y_test))

predictions = model.predict(X_test)
df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})

# %%
# trying all possible models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

models = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),
    DecisionTreeRegressor(),
]
model_preds = {}
for model in models:
    # fit the model
    model.fit(X_train, y_train)
    # evaluate the model
    scores = cross_val_score(model, X_train, y_train, cv=5)

    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(
        f"{model} : R^2 = {r2:.2f}, MAE = {mae:.2f}, MSE = {mse:.2f}, RMSE = {rmse:.2f}"
    )
    model_preds[model] = model.predict(X_test)

# create a dataframe of the predictions
df = pd.DataFrame(model_preds)
df["Actual"] = y_test.values


# %%
