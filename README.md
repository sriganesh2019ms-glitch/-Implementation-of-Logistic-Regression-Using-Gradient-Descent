# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the necessary python packages

Read the dataset.

Define X and Y array.

Define a function for costFunction,cost and gradient.

Define a function to plot the decision boundary and predict the Regression value

## Program and output:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SRIGANESH S
RegisterNumber:  212225220102
*/
```



```
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data=pd.read_csv('drive/MyDrive/ML/Placement_Data.csv') 
data.head()
```
<img width="1386" height="244" alt="image" src="https://github.com/user-attachments/assets/092b1bd2-046d-4ac6-a992-d19ef80e8925" />
```
data1=data.copy()
data1.head()
data1=data.drop(['sl_no','salary'],axis=1)
data1
```

<img width="1264" height="480" alt="image" src="https://github.com/user-attachments/assets/005b559f-7dae-4e7f-80c1-c22ea027e717" />

```
data=data.drop('sl_no',axis=1) 
data=data.drop('salary',axis=1) 

data["gender"]=data["gender"].astype('category') 
data["ssc_b"]=data["ssc_b"].astype('category') 
data["hsc_b"]=data["hsc_b"].astype('category') 
data["degree_t"]=data["degree_t"].astype('category') 
data["workex"]=data["workex"].astype('category') 
data["specialisation"]=data["specialisation"].astype('category') 
data["status"]=data["status"].astype('category') 
data["hsc_s"]=data["hsc_s"].astype('category') 
data.dtypes
```
<img width="300" height="612" alt="image" src="https://github.com/user-attachments/assets/a6677820-2176-4494-a2ea-0cd09c722d1a" />

```
data["gender"]=data["gender"].cat.codes 
data["ssc_b"]=data["ssc_b"].cat.codes 
data["hsc_b"]=data["hsc_b"].cat. codes
data["degree_t"]=data["degree_t"].cat.codes 
data["workex"]=data["workex"].cat.codes 
data["specialisation"]=data["specialisation"].cat.codes 
data["status"]=data["status"].cat.codes 
data["hsc_s"]=data["hsc_s"].cat.codes 
data
```
<img width="1146" height="483" alt="image" src="https://github.com/user-attachments/assets/ead63ade-7511-4e0b-9120-b9c2658b6f80" />

```
x=data.iloc[:,:-1].values 
y=data.iloc[:,-1].values

theta = np.random.randn(x.shape[1]) 
Y=y 

def sigmoid(z): 
   return 1/(1+np.exp(-z))
def loss(theta,X,y): 
   h=sigmoid(X.dot(theta))
   return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations): 
  m=len(y)
  for i in range(num_iterations): 
    h=sigmoid(X.dot(theta)) 
    gradient = X.T.dot(h-y)/m 
    theta-=alpha * gradient 
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta)) 
  y_pred=np.where(h>=0.5,1,0) 
  return y_pred 

y_pred = predict(theta,x) 
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
```

<img width="826" height="343" alt="image" src="https://github.com/user-attachments/assets/b97e3409-de58-495e-80b3-0a61308650c3" />

```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print("Predicted Result:",y_prednew)
```
<img width="286" height="45" alt="image" src="https://github.com/user-attachments/assets/3d09ec0b-89a6-4259-b092-c815bff395ea" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

