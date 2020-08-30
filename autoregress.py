
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA as pca
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest,chi2
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels 
import operator
import datetime
from mpl_toolkits.mplot3d import Axes3D


def replace_outliers(dataframe):
    #dcols=[x for x in dataframe.columns if x not in ["dates","stationid"]]
    #print(dcols)
    for col in [x for x in dataframe.columns if x !="CreationTime"]:
        #print(dataframe.col)
        #plt.boxplot(dataframe[col])
        q1,q3=np.percentile(dataframe[col],[25,75])
        med=dataframe[col].median()
        #print(med)
        iqr=q3-q1
        #print(q3+1.5*iqr)
        #print(q1-1.5*iqr)
        #print(dataframe.loc[dataframe[col]<q1-1.5*iqr,col])
        dataframe.loc[dataframe[col]>q3+1.5*iqr,col]=np.nan
        dataframe.loc[dataframe[col]<q1-1.5*iqr,col]=np.nan
        #print(dataframe.loc[dataframe[col]<q1-1.5*iqr,col])
        #print(dataframe)
        dataframe[col]=dataframe[col].fillna(med)
        #print(dataframe.loc[412,"humidity"])
        plt.boxplot(dataframe[col])
        print(col)
        plt.show()
    return dataframe



def load_dataset(path_to_file):
    data=pd.read_csv(path_to_file)
    return data
path_to_file="group2"
data=load_dataset(path_to_file+".csv")
replace_outliers(data)
Y=data["ActiveCount"]
persist=pd.DataFrame({"t-1":data["ActiveCount"].shift(1),"t":data["ActiveCount"]})
persist.drop(0,inplace=True)
plt.scatter(persist["t"],persist["t-1"])
plt.xlabel("t")
plt.ylabel("t-1")
plt.savefig("correlforpersist.png")
plt.show()
print("rmse for persistence model",mse(persist["t-1"],persist["t"])**0.5)
print()
X=data.drop(["ActiveCount"],axis=1)
X=X.drop(["CreationTime"],axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,shuffle=False)
plot_acf(Y_train,lags=30)
plt.savefig("Autocorrelationplot.png")
plt.show()
armodel=statsmodels.tsa.ar_model.AR(Y_train)

y_pred=armodel.fit().predict(start=len(X_train), end=len(X_train)+len(X_test)-1, dynamic=False)

print("rmse for autoregression",(mse(y_pred,Y_test))**0.5)
print("optimum lag",armodel.fit().k_ar)

scaler=StandardScaler()
X_standard=scaler.fit_transform(X)
Y_reshaped=(Y.values).reshape(-1,1)
Y_scaled=scaler.fit_transform(Y_reshaped)


X_train,X_test,Y_train,Y_test=train_test_split(X_standard,Y_scaled,test_size=0.3,shuffle=False)

armodel=statsmodels.tsa.ar_model.AR(Y_train)

y_pred=armodel.fit().predict(start=len(X_train), end=len(X_train)+len(X_test)-1, dynamic=False)

print("rmse for autoregression after standardisation",(mse(y_pred,Y_test))**0.5)
print("optimum lag",armodel.fit().k_ar)


scaler=MinMaxScaler()
X_standard=scaler.fit_transform(X)
Y_reshaped=(Y.values).reshape(-1,1)
Y_scaled=scaler.fit_transform(Y_reshaped)


X_train,X_test,Y_train,Y_test=train_test_split(X_standard,Y_scaled,test_size=0.3,shuffle=False)

armodel=statsmodels.tsa.ar_model.AR(Y_train)

y_pred=armodel.fit().predict(start=len(X_train), end=len(X_train)+len(X_test)-1, dynamic=False)

print("rmse for autoregression after minmax normalisation",(mse(y_pred,Y_test))**0.5)
print("optimum lag",armodel.fit().k_ar)

