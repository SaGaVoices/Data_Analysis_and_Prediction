
# coding: utf-8

# In[84]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[85]:


def split(df):
    y = df["ActiveCount"]
    features = []
    for col in df.columns:
        if col=="ActiveCount":
            continue
        features.append(col)
    X = df[features]
    return train_test_split(X,y,test_size=0.3,random_state = 42,shuffle="False")


def replace_outliers(dataframe):
    index =  len(dataframe.index)-1
    for col in dataframe.columns:
        if col == "CreationTime":
            continue;
        i=0
        count_out = 0
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)
        iqr = q3-q1
        low = q1 - 1.5*iqr
        high = q3 + 1.5*iqr
        med = dataframe[col].median()
            #print("for col = ",col,low,high)
        while i<=index:
            if dataframe[col].iloc[i]<=low or dataframe[col].iloc[i]>=high:
            #print(dataframe[col].iloc[i])
                dataframe.at[i,col] = med
                count_out+=1
            i += 1 
    return df
        #print("total outliers for ",col," is ->",count_out)


# In[86]:


df = pd.read_csv("group2.csv")
df = replace_outliers(df)
X_train, X_test, X_label_train, X_label_test = split(df)


# In[87]:


def corr(df1,df2,k):
    cols = df1.columns
    main_attr = df2.values
    d = dict()
    import scipy.stats as st
    for i in cols:
        try:
            coeff = st.pearsonr(df1[i].values,main_attr)
            #print(i,coeff[0])
            d[i] = coeff[0]
        except:
            EOFError
    d_s =sorted(d.items(),key = lambda kv:(kv[1],kv[0]) )
    return d_s[-k][0]


# In[88]:


def simple_linear_regression(df1,df2,df3,df4,k):
    indep = corr(df1,df2,k)
    independent_attr = df1[indep].values.reshape(len(df1),1)
    dependent_attr = df2.values.reshape(len(df2),1)
    regr = LinearRegression()
    regr.fit(independent_attr,dependent_attr)
    predicted_values = regr.predict(independent_attr)
    error = 0
    for i in range(len(independent_attr)):
        error += ((predicted_values[i][0]-dependent_attr[i][0])**2)
    print(indep)
    rmse = error**.5
    print("For train data, RMSE =",rmse)

    test_data = df3[indep].values.reshape(len(X_test),1)
    test_predict = regr.predict(test_data)
    
    error=0
    for i in range(len(X_test)):
        error += ((test_predict[i][0]-df4.values[i])**2)
    
    rmse = error**0.5
    print("For test data, RMSE =",rmse)

    plt.scatter(test_data,df4.values,color="cyan",alpha=1,s=1)
    plt.plot(test_data,test_predict,"r:")
    plt.show()


# In[83]:


#simple_linear_regression(X_train,X_label_train,X_test,X_label_test,1)


# In[39]:


#len(X_train.columns)


# In[89]:


for i in range(1,13):
    print(i)
    simple_linear_regression(X_train,X_label_train,X_test,X_label_test,i)


# In[90]:


def multiple_linear_regression(df1,df2,df3,df4):
    from sklearn.metrics import mean_squared_error as mse
    independent_attr = df1.values
    dependent_attr = df2.values.reshape(len(df2),1)
    regr = LinearRegression()
    regr.fit(independent_attr,dependent_attr)
    predicted_values = regr.predict(independent_attr)
    error = 0
    #for i in range(len(independent_attr)):
     #   error += ((predicted_values[i][0]-dependent_attr[i][0])**2)
    #print(indep)
    #rmse = error**.5
    print("For train data, RMSE =")
    print(np.sqrt(mse(df2,predicted_values)))
    test_data = df3.values
    test_predict = regr.predict(test_data)
    
    error=0
    #for i in range(len(X_test)):
     #   error += ((test_predict[i][0]-df4.values[i])**2)
    
    #rmse = error**0.5
    print("For test data, RMSE =")
    print(np.sqrt(mse(df4,test_predict)))
    plt.scatter(range(len(df4)),df4.values,color="cyan",alpha=1,s=1)
    plt.plot(range(len(df4)),test_predict,"r:")
    plt.show()


# In[91]:


multiple_linear_regression(X_train.iloc[:,1:],X_label_train,X_test.iloc[:,1:],X_label_test)


# In[92]:


from sklearn.metrics import mean_squared_error as mse
A = X_train["InTotalPPS"].values.reshape(len(X_train),1)
B = X_train["TempAvg"].values.reshape(len(X_train),1)
D = X_train["CPUUtil"].values.reshape(len(X_train),1)
C = np.concatenate((A,B,D),axis=1)
independent_attribute = C
dependent_attribute = X_label_train.values.reshape(len(X_train),1)

regr = LinearRegression()
regr.fit(independent_attribute,dependent_attribute)
predicted_values = regr.predict(independent_attribute)


error = 0
for i in range(len(independent_attribute)):
    error += ((predicted_values[i][0]-dependent_attribute[i][0])**2)

rmse = (error/len(independent_attribute))**.5
print("For train data, RMSE =",rmse)
print(np.sqrt(mse(X_label_train,predicted_values)))
At = X_test["InTotalPPS"].values.reshape(len(X_test),1)
Bt = X_test["TempAvg"].values.reshape(len(X_test),1)
Dt = X_test["CPUUtil"].values.reshape(len(X_test),1)
test_data = np.concatenate((At,Bt,Dt),axis=1)
test_predict = regr.predict(test_data)

error=0
for i in range(len(X_test)):
    error += ((test_predict[i][0]-X_label_test.values[i])**2)
    
rmse = (error/len(X_test))**0.5
print("For test data, RMSE =",rmse)
print(np.sqrt(mse(X_label_test,test_predict)))
plt.scatter(range(len(X_label_test)),X_label_test.values,color="cyan",alpha=1,s=1)
plt.plot(range(len(X_label_test)),test_predict,"r:")
plt.show()


# In[93]:


from sklearn.metrics import mean_squared_error as mse
A = X_train["InTotalPPS"].values.reshape(len(X_train),1)
B = X_train["TempAvg"].values.reshape(len(X_train),1)
#D = X_train["CPUUtil"].values.reshape(len(X_train),1)
C = np.concatenate((A,B),axis=1)
independent_attribute = C
dependent_attribute = X_label_train.values.reshape(len(X_train),1)

regr = LinearRegression()
regr.fit(independent_attribute,dependent_attribute)
predicted_values = regr.predict(independent_attribute)


error = 0
for i in range(len(independent_attribute)):
    error += ((predicted_values[i][0]-dependent_attribute[i][0])**2)

rmse = (error/len(independent_attribute))**.5
print("For train data, RMSE =",rmse)
print(np.sqrt(mse(X_label_train,predicted_values)))
At = X_test["InTotalPPS"].values.reshape(len(X_test),1)
Bt = X_test["TempAvg"].values.reshape(len(X_test),1)
#Dt = X_test["CPUUtil"].values.reshape(len(X_test),1)
test_data = np.concatenate((At,Bt),axis=1)
test_predict = regr.predict(test_data)

error=0
for i in range(len(X_test)):
    error += ((test_predict[i][0]-X_label_test.values[i])**2)
    
rmse = (error/len(X_test))**0.5
print("For test data, RMSE =",rmse)
print(np.sqrt(mse(X_label_test,test_predict)))
plt.scatter(range(len(X_label_test)),X_label_test.values,color="cyan",alpha=1,s=1)
plt.plot(range(len(X_label_test)),test_predict,"r:")
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
z_data = dependent_attribute
z_data = z_data.reshape(1,len(z_data))[0]
z_new = predicted_values.reshape(1,len(z_data))[0]
x_data = A
x_data = x_data.reshape(1,len(z_data))[0]
y_data = B
y_data = y_data.reshape(1,len(z_data))[0]
ax.set_xlabel("InTotalPPS")
ax.set_ylabel("TempAvg")
ax.set_zlabel('ActiveCount')
ax.scatter3D(x_data, y_data, z_data, c="r",s=1)
ax.scatter3D(x_data, y_data , z_new, c="cyan",s=1)
ax.plot_trisurf(x_data, y_data, z_new, alpha = 0.5)
plt.show()


# In[94]:


from sklearn.metrics import mean_squared_error as mse
A = X_train["InTotalPPS"].values.reshape(len(X_train),1)
#B = X_train["TempAvg"].values.reshape(len(X_train),1)
D = X_train["CPUUtil"].values.reshape(len(X_train),1)
C = np.concatenate((A,D),axis=1)
independent_attribute = C
dependent_attribute = X_label_train.values.reshape(len(X_train),1)

regr = LinearRegression()
regr.fit(independent_attribute,dependent_attribute)
predicted_values = regr.predict(independent_attribute)


error = 0
for i in range(len(independent_attribute)):
    error += ((predicted_values[i][0]-dependent_attribute[i][0])**2)

rmse = (error/len(independent_attribute))**.5
print("For train data, RMSE =",rmse)
print(np.sqrt(mse(X_label_train,predicted_values)))
At = X_test["InTotalPPS"].values.reshape(len(X_test),1)
#Bt = X_test["TempAvg"].values.reshape(len(X_test),1)
Dt = X_test["CPUUtil"].values.reshape(len(X_test),1)
test_data = np.concatenate((At,Dt),axis=1)
test_predict = regr.predict(test_data)

error=0
for i in range(len(X_test)):
    error += ((test_predict[i][0]-X_label_test.values[i])**2)
    
rmse = (error/len(X_test))**0.5
print("For test data, RMSE =",rmse)
print(np.sqrt(mse(X_label_test,test_predict)))
plt.scatter(range(len(X_label_test)),X_label_test.values,color="cyan",alpha=1,s=1)
plt.plot(range(len(X_label_test)),test_predict,"r:")
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
z_data = dependent_attribute
z_data = z_data.reshape(1,len(z_data))[0]
z_new = predicted_values.reshape(1,len(z_data))[0]
x_data = A
x_data = x_data.reshape(1,len(z_data))[0]
y_data = D
y_data = y_data.reshape(1,len(z_data))[0]
ax.set_xlabel('InTotalPPS')
ax.set_ylabel('CPUUtil')
ax.set_zlabel('ActiveCount')
ax.scatter3D(x_data, y_data, z_data, c="r",s=1)
ax.scatter3D(x_data, y_data , z_new, c="cyan",s=1)
ax.plot_trisurf(x_data, y_data, z_new, alpha = 0.5)
plt.show()


# In[95]:


from sklearn.metrics import mean_squared_error as mse
#A = X_train["InTotalPPS"].values.reshape(len(X_train),1)
B = X_train["TempAvg"].values.reshape(len(X_train),1)
D = X_train["CPUUtil"].values.reshape(len(X_train),1)
C = np.concatenate((B,D),axis=1)
independent_attribute = C
dependent_attribute = X_label_train.values.reshape(len(X_train),1)

regr = LinearRegression()
regr.fit(independent_attribute,dependent_attribute)
predicted_values = regr.predict(independent_attribute)


error = 0
for i in range(len(independent_attribute)):
    error += ((predicted_values[i][0]-dependent_attribute[i][0])**2)

rmse = (error/len(independent_attribute))**.5
print("For train data, RMSE =",rmse)
print(np.sqrt(mse(X_label_train,predicted_values)))
#At = X_test["InTotalPPS"].values.reshape(len(X_test),1)
Bt = X_test["TempAvg"].values.reshape(len(X_test),1)
Dt = X_test["CPUUtil"].values.reshape(len(X_test),1)
test_data = np.concatenate((Bt,Dt),axis=1)
test_predict = regr.predict(test_data)

error=0
for i in range(len(X_test)):
    error += ((test_predict[i][0]-X_label_test.values[i])**2)
    
rmse = (error/len(X_test))**0.5
print("For test data, RMSE =",rmse)
print(np.sqrt(mse(X_label_test,test_predict)))
plt.scatter(range(len(X_label_test)),X_label_test.values,color="cyan",alpha=1,s=1)
plt.plot(range(len(X_label_test)),test_predict,"r:")
plt.show()



import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
z_data = dependent_attribute
z_data = z_data.reshape(1,len(z_data))[0]
z_new = predicted_values.reshape(1,len(z_data))[0]
x_data = B
x_data = x_data.reshape(1,len(z_data))[0]
y_data = D
y_data = y_data.reshape(1,len(z_data))[0]
ax.set_xlabel('TempAvg')
ax.set_ylabel('CPUUtil')
ax.set_zlabel('ActiveCount')
ax.scatter3D(x_data, y_data, z_data, c="r",s=1)
ax.scatter3D(x_data, y_data , z_new, c="cyan",s=1)
ax.plot_trisurf(x_data, y_data, z_new, alpha = 0.5)
plt.show()


# In[104]:


from sklearn.metrics import mean_squared_error as mse
A = X_train["InTotalPPS"].values.reshape(len(X_train),1)
B = X_train["TempAvg"].values.reshape(len(X_train),1)
#D = X_train["CPUUtil"].values.reshape(len(X_train),1)
C = np.concatenate((A,B),axis=1)
independent_attribute = C
polyreg = PolynomialFeatures(degree=2)       ##checked on different values of degree, 2 is best 
new_independent = polyreg.fit_transform(independent_attribute)
dependent_attribute = X_label_train.values.reshape(len(X_train),1)

regr = LinearRegression()
regr.fit(new_independent,dependent_attribute)
predicted_values = regr.predict(new_independent)


error = 0
for i in range(len(independent_attribute)):
    error += ((predicted_values[i][0]-dependent_attribute[i][0])**2)

rmse = (error/len(independent_attribute))**.5
print("For train data, RMSE =",rmse)
print(np.sqrt(mse(X_label_train,predicted_values)))
At = X_test["InTotalPPS"].values.reshape(len(X_test),1)
Bt = X_test["TempAvg"].values.reshape(len(X_test),1)
#Dt = X_test["CPUUtil"].values.reshape(len(X_test),1)
test_data = np.concatenate((At,Bt),axis=1)
new_test = polyreg.fit_transform(test_data)
test_predict = regr.predict(new_test)

error=0
for i in range(len(X_test)):
    error += ((test_predict[i][0]-X_label_test.values[i])**2)
    
rmse = (error/len(X_test))**0.5
print("For test data, RMSE =",rmse)
print(np.sqrt(mse(X_label_test,test_predict)))
plt.scatter(range(len(X_label_test)),X_label_test.values,color="cyan",alpha=1,s=1)
plt.plot(range(len(X_label_test)),test_predict,"r:")
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
z_data = dependent_attribute
z_data = z_data.reshape(1,len(z_data))[0]
z_new = predicted_values.reshape(1,len(z_data))[0]
x_data = A
x_data = x_data.reshape(1,len(z_data))[0]
y_data = B
y_data = y_data.reshape(1,len(z_data))[0]
ax.set_xlabel("InTotalPPS")
ax.set_ylabel("TempAvg")
ax.set_zlabel('ActiveCount')
ax.scatter3D(x_data, y_data, z_data, c="r",s=0.5)
ax.scatter3D(x_data, y_data , z_new, c="cyan",s=1)
ax.plot_trisurf(x_data, y_data, z_new, alpha = 0.5)
plt.show()


# In[105]:


from sklearn.metrics import mean_squared_error as mse
A = X_train["InTotalPPS"].values.reshape(len(X_train),1)
#B = X_train["TempAvg"].values.reshape(len(X_train),1)
D = X_train["CPUUtil"].values.reshape(len(X_train),1)
C = np.concatenate((A,D),axis=1)
independent_attribute = C
polyreg = PolynomialFeatures(degree=2)       ##checked on different values of degree, 2 is best 
new_independent = polyreg.fit_transform(independent_attribute)
dependent_attribute = X_label_train.values.reshape(len(X_train),1)

regr = LinearRegression()
regr.fit(new_independent,dependent_attribute)
predicted_values = regr.predict(new_independent)


error = 0
for i in range(len(independent_attribute)):
    error += ((predicted_values[i][0]-dependent_attribute[i][0])**2)

rmse = (error/len(independent_attribute))**.5
print("For train data, RMSE =",rmse)
print(np.sqrt(mse(X_label_train,predicted_values)))
At = X_test["InTotalPPS"].values.reshape(len(X_test),1)
#Bt = X_test["TempAvg"].values.reshape(len(X_test),1)
Dt = X_test["CPUUtil"].values.reshape(len(X_test),1)
test_data = np.concatenate((At,Dt),axis=1)
new_test = polyreg.fit_transform(test_data)
test_predict = regr.predict(new_test)


error=0
for i in range(len(X_test)):
    error += ((test_predict[i][0]-X_label_test.values[i])**2)
    
rmse = (error/len(X_test))**0.5
print("For test data, RMSE =",rmse)
print(np.sqrt(mse(X_label_test,test_predict)))
plt.scatter(range(len(X_label_test)),X_label_test.values,color="cyan",alpha=1,s=1)
plt.plot(range(len(X_label_test)),test_predict,"r:")
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
z_data = dependent_attribute
z_data = z_data.reshape(1,len(z_data))[0]
z_new = predicted_values.reshape(1,len(z_data))[0]
x_data = A
x_data = x_data.reshape(1,len(z_data))[0]
y_data = D
y_data = y_data.reshape(1,len(z_data))[0]
ax.set_xlabel('InTotalPPS')
ax.set_ylabel('CPUUtil')
ax.set_zlabel('ActiveCount')
ax.scatter3D(x_data, y_data, z_data, c="r",s=1)
ax.scatter3D(x_data, y_data , z_new, c="cyan",s=1)
ax.plot_trisurf(x_data, y_data, z_new, alpha = 0.5)
plt.show()


# In[106]:


from sklearn.metrics import mean_squared_error as mse
#A = X_train["InTotalPPS"].values.reshape(len(X_train),1)
B = X_train["TempAvg"].values.reshape(len(X_train),1)
D = X_train["CPUUtil"].values.reshape(len(X_train),1)
C = np.concatenate((B,D),axis=1)
independent_attribute = C
polyreg = PolynomialFeatures(degree=2)       ##checked on different values of degree, 2 is best 
new_independent = polyreg.fit_transform(independent_attribute)
dependent_attribute = X_label_train.values.reshape(len(X_train),1)

regr = LinearRegression()
regr.fit(new_independent,dependent_attribute)
predicted_values = regr.predict(new_independent)


error = 0
for i in range(len(independent_attribute)):
    error += ((predicted_values[i][0]-dependent_attribute[i][0])**2)

rmse = (error/len(independent_attribute))**.5
print("For train data, RMSE =",rmse)
print(np.sqrt(mse(X_label_train,predicted_values)))
#At = X_test["InTotalPPS"].values.reshape(len(X_test),1)
Bt = X_test["TempAvg"].values.reshape(len(X_test),1)
Dt = X_test["CPUUtil"].values.reshape(len(X_test),1)
test_data = np.concatenate((Bt,Dt),axis=1)
new_test = polyreg.fit_transform(test_data)
test_predict = regr.predict(new_test)


error=0
for i in range(len(X_test)):
    error += ((test_predict[i][0]-X_label_test.values[i])**2)
    
rmse = (error/len(X_test))**0.5
print("For test data, RMSE =",rmse)
print(np.sqrt(mse(X_label_test,test_predict)))
plt.scatter(range(len(X_label_test)),X_label_test.values,color="cyan",alpha=1,s=1)
plt.plot(range(len(X_label_test)),test_predict,"r:")
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
z_data = dependent_attribute
z_data = z_data.reshape(1,len(z_data))[0]
z_new = predicted_values.reshape(1,len(z_data))[0]
x_data = A
x_data = x_data.reshape(1,len(z_data))[0]
y_data = D
y_data = y_data.reshape(1,len(z_data))[0]
ax.set_xlabel('InTotalPPS')
ax.set_ylabel('CPUUtil')
ax.set_zlabel('ActiveCount')
ax.scatter3D(x_data, y_data, z_data, c="r",s=1)
ax.scatter3D(x_data, y_data , z_new, c="cyan",s=1)
ax.plot_trisurf(x_data, y_data, z_new, alpha = 0.5)
plt.show()

