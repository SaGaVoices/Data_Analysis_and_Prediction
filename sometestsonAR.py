
# coding: utf-8

# In[2]:


import numpy as np
from math import sin
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

t = np.linspace(0,100,num=1000)

def f(A,w):
    global t
    y = []
    for i in range(len(t)):
        y.append(A*sin(w*t[i]))
    return np.array(y)

y = f(20,0.1) + f(20,3)

plot_acf(y, lags=500)
plt.show()

plt.plot(t,y)
plt.show()

