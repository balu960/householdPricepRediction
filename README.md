# householdPricePrediction

## Usage

Code is in Codes\file0.py

# Missing values

we checked for missing data but it was clean.
If there was any missing data we could have tried interpolations, droppings or fillings.

# column explanations and encodings

ID is just a key.

'OverallQual', 'YearBuilt', 'FullBath', 'HalfBath', 'GarageCars' are categorical variables but they are ordinal categorical variables. So we can do Ordinal Encoding to encode the data. which is ame as their number

for all variables we used MinMaxScaler to normalize.

# correlation and P value checking.

by using statistical methods we checked each variable's dependency on target but found out data seems to be random but further analysis is needed.

# By sklearn we split the data

# Creating ML models

we analyzed data by building below models

    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    RandomForestRegressor,
    GradientBoostingRegressor,
    SVR,
    DecisionTreeRegressor

# model evaluation

we checked model scoring with r2_score, mean_absolute_error mean_squared_error
all the models produced the negative r^2 results with test data meaning that they are performing worser than random. This proves there is randomness in the data. To improve the models we want new fields which can have better correlation with the target variable.
