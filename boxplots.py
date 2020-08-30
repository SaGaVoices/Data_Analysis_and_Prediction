#descriptive analysis of bng devices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("group2.csv")

def mean(df):#for calculating the mean of each column
    for col in df.columns:
        if col=="CreationTime":
            continue
        print("mean for ",col," is->",round(df[col].mean(),4))
    
#print(df[df["OutBandwidth"]==0])

def median(df):# for median of each column
    for col in df.columns:
        if col=="CreationTime":
            continue
        print("median for ",col," is->",round(df[col].median(),4))
        
def mode(df):# for mode of each column
     for col in df.columns:
        if col=="CreationTime":
            continue
        print("mode for ",col," is->",round(df[col].mode().mean(),4))

def corrcoef(df): #for correlation coefficient between each attribute
    for col1 in df.columns:
        if col1=="CreationTime":
            continue
        for col2 in df.columns:
            if col2 == "CreationTime":
                continue
            if col1!= col2:
                corr = np.corrcoef(df[col1],df[col2])
                print("correlation b/w ",col1," and ",col2," is->",corr[0][1])
                
                
mean(df)
median(df)
mode(df)
corrcoef(df)

print(df.describe())
#%%

correlation = df.corr(method="pearson")
plt.rcParams['figure.figsize'] = (15,15)
sb.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,cmap="RdBu_r",linewidth=0.5,annot=True)




#%%
fig=plt.figure() 
def box_plot(df):
    for col in df.columns:
        if col=="CreationTime":
            continue
        df.boxplot(column=col)
        plt.show()

box_plot(df)

#%%

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
     print("total outliers for ",col," is ->",count_out)
     
     
replace_outliers(df)
#df.to_csv("clean_group2.csv")























    