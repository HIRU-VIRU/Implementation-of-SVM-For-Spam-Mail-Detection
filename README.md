# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.
 

## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HIRUTHIK SUDHAKAR
RegisterNumber:  212223240054
*/

import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()

data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:
### result
![image](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/5c4d783d-54fc-41ab-84c2-7b33b8869706)
### data.head()
![image](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/1e43ac15-6a8d-4bce-aab8-545c0484089b)
### data.info()
![image](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/bad3c268-13b8-40d4-8c21-259a927e0118)
### data.isnull().sum()
![image](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/01badbf3-b9c0-46e4-b73b-5daa0b65dee0)
### y_pred
![image](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/61eaae9f-d103-481e-b63c-1f26f13f08b1)
### accuracy
![image](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/7691b77f-02f3-4979-b4b1-dd318a216339)








## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
