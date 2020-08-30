
# coding: utf-8

# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.model_selection import train_test_split as tits
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, normalize
from sklearn.decomposition import PCA

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r

data = pd.read_csv("group2.csv")
data2 = data.copy()
Y = np.array(data["ActiveCount"])
data.drop(columns=["ActiveCount","CreationTime"],inplace=True)
X = np.array(data)

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
        plt.boxplot(dataframe[col])
        print(col)
        plt.show()
    return dataframe
X_stan = pd.read_csv("standardised.csv")
X_norm = pd.read_csv("normalised.csv")

X_stan = np.array(X_stan.drop(columns = ["ActiveCount", "CreationTime"]))
X_norm = np.array(X_norm.drop(columns = ["ActiveCount", "CreationTime"]))

l2Score = {"norm":[],"stan":[],"orig":[]}

for i in range(1,13):
    Xt = PCA(n_components=i).fit_transform(X)
    pf = PolynomialFeatures(degree=2)             ###You can change degree = 1, 3 , 4, 5 and so on.(degree = 2 is the best model)
    polyXt = pf.fit_transform(Xt)
    x_train, x_test, y_train, y_test = tits(Xt,Y,test_size=0.3,random_state=42)
    y_test = np.array(y_test).reshape(-1,1)
    y_test.reshape(-1,1)
    model = LinearRegression().fit(x_train,y_train)
    y_pred = model.predict(x_test).reshape(-1,1)
    score = r(y_test,y_pred)
    rmse = mse(y_pred,y_test)**0.5
    l2Score["orig"].append([score,rmse])

    Xt = PCA(n_components=i).fit_transform(X_norm)
    pf = PolynomialFeatures(degree=2)
    polyXt = pf.fit_transform(Xt)
    x_train, x_test, y_train, y_test = tits(Xt,Y,test_size=0.3,random_state=42)
    y_test = np.array(y_test).reshape(-1,1)
    model = LinearRegression().fit(x_train,y_train)
    y_pred = model.predict(x_test).reshape(-1,1)
    np.array(y_test).reshape(-1, 1)
    score = r(y_test,y_pred)
    rmse = mse(y_pred,y_test)**0.5
    l2Score["norm"].append([score,rmse])

    Xt = PCA(n_components=i).fit_transform(X_stan)
    pf = PolynomialFeatures(degree=2)
    polyXt = pf.fit_transform(Xt)
    x_train, x_test, y_train, y_test = tits(Xt,Y,test_size=0.3,random_state=42)
    y_test = np.array(y_test).reshape(-1,1)
    model = LinearRegression().fit(x_train,y_train)
    np.array(y_test).reshape(-1, 1)
    y_pred = model.predict(x_test).reshape(-1,1)
    score = r(y_test,y_pred)
    rmse = mse(y_pred,y_test)**0.5
    l3 = [score,rmse]
    l2Score["stan"].append([score,rmse])


# In[48]:


orig = l2Score["orig"]
norm = l2Score["norm"]
stan = l2Score["stan"]


# In[19]:


def callrmse(lis):
    rmse = []
    for i in range(len(lis)):
        rmse.append(lis[i][1])
    return rmse


# In[42]:


def callrscore(lis):
    rscore = []
    for i in range(len(lis)):
        rscore.append(lis[i][0])
    return rscore


# In[49]:


N = 12
ind = np.arange(N) 
ax = plt.subplot(111)
ax.bar(ind, callrmse(orig), width=0.2, color='c', align='center')
ax.bar(ind+0.2,callrmse(norm), width=0.2, color='r', align='center')
ax.bar(ind+0.4,callrmse(stan), width=0.2, color='b', align='center')
ax.set_xticklabels( ["1","2","3","4","5","6","7","8","9","10","11","12"] )
ax.set_xlabel("Number of components in PCA")
ax.set_ylabel("RMSE")
plt.show()


# In[51]:


N = 12
ind = np.arange(N) 
ax = plt.subplot(111)
ax.bar(ind, callrscore(orig), width=0.2, color='c', align='center')
ax.bar(ind+0.2,callrscore(norm), width=0.2, color='r', align='center')
ax.bar(ind+0.4,callrscore(stan), width=0.2, color='b', align='center')
ax.set_xticklabels( ["1","2","3","4","5","6","7","8","9","10","11","12"] )
ax.set_xlabel("Number of components in PCA")
ax.set_ylabel("R2 score")
plt.show()

