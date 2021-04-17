import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max.columns', None)

df = pd.read_csv('bmw.csv')
print(df.head())

# Check if there is missing value
print(df.isnull().sum())

print(df.describe())

#correlation-plot
print(sns.heatmap(df.corr()))

#compare model sales
print(sns.countplot(y = df['model']))


# one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df)

# Perform Standardization
stdzn = StandardScaler()
df_stdzd = stdzn.fit_transform(df_encoded.drop(columns=['price']))

df_stdzd = pd.DataFrame(df_stdzd)
print(df_stdzd.shape)

X_train, X_test, y_train, y_test = train_test_split(
                                                    df_stdzd
                                                    ,df['price'])
print(X_train.shape)
print(X_test.shape)


selector = SelectKBest(f_regression, k = 23)
X_train_transformed = selector.fit_transform(X_train, y_train)
X_test_transformed = selector.transform(X_test)


def regression_model(model):

    regressor = model
    regressor.fit(X_train_transformed, y_train)
    score = regressor.score(X_test_transformed, y_test)
    return regressor, score


model_performance = pd.DataFrame(columns = ["Features", "Model", "Score"])

models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), RandomForestRegressor()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Linear","Model": model, "Score": score}, ignore_index=True)

print(model_performance)