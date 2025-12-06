# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Input: Load and read the spam dataset containing messages and their labels.

2.Processing: Clean the text, remove stopwords, and convert it into TF-IDF feature vectors.

3.Decision: Train the SVM classifier and use it to predict whether a message is spam or not.

4.Output: Display results such as accuracy, confusion matrix, and message prediction.



## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SOMESHWAR KUMAR  
RegisterNumber: 212224240157
*/
```
```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result
```
```
import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
```
```
data.info()
```
```
data.isnull().sum()
```
```
x=data["v1"].values
y=data["v2"].values
```
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
```
```
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```
```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

<img width="708" height="150" alt="11" src="https://github.com/user-attachments/assets/3951d484-a991-409d-bf3b-a2bcba98c2b5" />

-------------------------------------------------------

<img width="726" height="237" alt="22" src="https://github.com/user-attachments/assets/e843a51d-bc4c-4195-ba53-a5ac49dd90c9" />

------------------------------------------------

<img width="414" height="274" alt="33" src="https://github.com/user-attachments/assets/276e2b95-b221-4038-8c1b-c7ef9ed989e2" />

-------------------------------------------------------
<img width="220" height="169" alt="44" src="https://github.com/user-attachments/assets/2706ff5b-c484-4c51-8178-5f1747d924c3" />

---------------------------------------------------------

<img width="653" height="184" alt="55" src="https://github.com/user-attachments/assets/9cc92fbd-bf83-4a2b-b179-6fac37678b35" />

----------------------------------------------------------

<img width="422" height="131" alt="66" src="https://github.com/user-attachments/assets/46944b5c-86cf-446b-96be-ec2f90a51017" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
