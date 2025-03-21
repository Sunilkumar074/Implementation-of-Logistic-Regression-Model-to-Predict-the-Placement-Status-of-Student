# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results. 


## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SUNIL KUMAR P.B.
RegisterNumber:  212223040213
*/
```
```python
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### TOP 5 ELEMENTS
![image](https://github.com/user-attachments/assets/a2c84458-df28-42e3-b016-b714cef78246)

![image](https://github.com/user-attachments/assets/4eb20563-a914-4da5-8776-fbed3c2f74a3)

![image](https://github.com/user-attachments/assets/dc2c9a17-2cf0-44d7-8a93-fe4c505befcd)

### Data Duplicate:
![image](https://github.com/user-attachments/assets/7dfd8a27-4a26-4611-bde7-5a1d31baa08a)

<br>
<br>
<br>
<br>
<br>
<br>
<br>


### Print Data:
![image](https://github.com/user-attachments/assets/62412b0c-af17-487a-b4fd-2f4ad263803a)

### Data-Status:
![image](https://github.com/user-attachments/assets/c04864cf-3ab6-4e0b-9f07-fad50df6e3e5)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>



### y_prediction array:
![image](https://github.com/user-attachments/assets/5b6b39cf-6406-4bc2-8395-a637f24a01c1)



### Confusion array:
![image](https://github.com/user-attachments/assets/e42ff2e9-ab06-4d94-b44e-1241d357b8df)


### Accuracy Value:
![image](https://github.com/user-attachments/assets/9e462298-c4c9-4fd4-b8f1-a00ca7a6212e)


### Classification Report:
![image](https://github.com/user-attachments/assets/c01a4d5e-44fa-4f64-b20d-f229b989e137)

### Prediction of LR:
![image](https://github.com/user-attachments/assets/a466fc62-6ab9-473d-be3d-584bbaa3ac2e)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
