import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print("imported")

housing=pd.read_csv('house_train.csv', index_col=0)
housing['Age']=housing['YrSold'] - housing['YearBuilt']

#Getting Neighboorhoods.value with more than 30 observations

counts=housing['Neighborhood'].value_counts()
more_than_30=list(counts[counts>30].index)
housing=housing.loc[housing['Neighborhood'].isin(more_than_30)]
#print(housing)
features=['CentralAir', 'LotArea','OverallQual','OverallCond','1stFlrSF',
          '2ndFlrSF','BedroomAbvGr','Age']
target='SalePrice'

#Transforming Neighborhood and aircentral to the one-hot encoding format
#neigh
dummies_nb=pd.get_dummies(housing['Neighborhood'], drop_first=True)
housing=pd.concat([housing,dummies_nb], axis=1)
#central air
housing['CentralAir']=housing['CentralAir'].map({'N':0,'Y':1}).astype(int)

features +=list(dummies_nb.columns)

X=housing[features].values
y=housing[target].values
n=housing.shape[0]

#what is the simplest possible model? just predict the average!

y_mean=np.mean(y)
print(y_mean)

RMSE_null_model=np.sqrt(np.sum((y-y_mean)**2)/n)
#print(RMSE_null_model)

#Building Linear regression MOdel
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
housing['predictions']=regressor.predict(X)
y_pred=housing['predictions'].values
RMSE_regressor=np.sqrt(np.sum((y-y_pred)**2)/n)
#print(RMSE_regressor)

housing.plot.scatter(x='SalePrice', y='predictions')
plt.plot(x='SalePrice', y='predictions')
plt.show()
print("worked")


new_house=np.array([[1,12000,6,6,1200,500,3,5,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]])
prediction=regressor.predict(new_house)
print("For a house with the following characteristics:\n")
for feature, feature_value in zip(features, new_house[0]):
    if feature_value > 0:
        print("{}:{}".format(feature, feature_value))
print("\nThe predicted value for the house is: {:,}".format(round(prediction[0])))

