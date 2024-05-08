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
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HIRUTHIK SUDHAKAR
RegisterNumber:  212223240054
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```


## Output:
### data
![328855338-4a522776-209c-4329-932c-be8f8102c5ba](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/5d1f74f3-8c4b-4fa4-b393-1b43b1c65733)

### data.shape()
![328855480-f8f741f4-3206-4526-92fd-c890f6ecb1e5](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/db7053c4-c364-44e3-a600-60906ff539e9)

### x.shape()
![328855694-2d7d8faa-ef77-405b-aedb-3009855bfeb9](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/90ac9c48-2d53-4143-91ee-9c5876c2e1e0)

### y.shape()  
![328855981-d3439f11-7e22-4ade-b5c6-917b3352cb8d](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/f653ba87-2754-4838-9126-6a1c045e7d88)


### x_train
![328856257-67edd510-0d60-49f7-bde7-cd13ba895357](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/5f30d732-9234-40b3-b80f-a36db89825fb)


### x_train.shape()
![328856601-e9f7eb9a-89b8-4d67-b58c-29e8bd8668e0](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/41dcdbf5-b54d-4c65-b0b9-177f2bb5ad59)

### y_pred
![328856854-d98bdfde-aa7b-46a0-814a-020100201f28](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/c990414a-5280-4964-8eeb-5d0ffaead7a9)


### acc (accuracy)
![328857052-da3dd64d-b341-4b5b-824e-dd621396b816](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/d1d69362-7f5b-4249-a95b-e9f5e894a917)

### con (confusion matrix)
![328857298-43ae6fb1-8477-4118-abea-e8b2891123aa](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/98ae0d93-d355-43f3-b59a-52bb58e13d5e)

### cl (classification report)
![328857416-c1a9e002-dc90-4f21-bb0d-daf799640c92](https://github.com/HIRU-VIRU/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145972122/3f61b52d-1468-48d1-bbb5-291896323919)









## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
