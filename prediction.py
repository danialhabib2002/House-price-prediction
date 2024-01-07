from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
import xgboost as xgb
import seaborn as sns


data = pd.read_csv('data.csv')
data = data.drop(['city','date','statezip','street','yr_built','yr_renovated','condition','sqft_lot','country'],axis=1)

# print the first 5 rows of the data
print(data.head())

data = pd.get_dummies(data)

print(data.isnull().sum())  

# target_col = 'price'
# # Calculate the correlation matrix
# correlation_matrix = data.corr()

# # Extract the correlation of each attribute with the target variable
# correlation_with_target = correlation_matrix[target_col]

# # Display the correlation values
# for column in data.columns:
#     correlation = correlation_matrix.loc[column, target_col]
#     print(f"Correlation between {column} and {target_col}: {correlation}")

X = data.drop('price',axis=1)
y = data['price']


print(X.head())

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

model = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=850,learning_rate=0.009,max_depth=4,subsample=1,colsample_bytree=0.8,reg_alpha=0.5,reg_lambda=1)



model.fit(X_train,y_train)
train_data_pred = model.predict(X_train)
print(metrics.r2_score(y_train,train_data_pred))
print(metrics.d2_absolute_error_score(y_train,train_data_pred))

plt.scatter(y_train,train_data_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.show()

test_data_pred = model.predict(X_test)
print(metrics.r2_score(y_test,test_data_pred))
print(metrics.d2_absolute_error_score(y_test,test_data_pred))



















