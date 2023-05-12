# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Prepare your data

    -Collect and clean data on employee salaries and features
    -Split data into training and testing sets

2.Define your model

    -Use a Decision Tree Regressor to recursively partition data based on input features
    -Determine maximum depth of tree and other hyperparameters

3.Train your model

    -Fit model to training data
    -Calculate mean salary value for each subset

4.Evaluate your model

    -Use model to make predictions on testing data
    -Calculate metrics such as MAE and MSE to evaluate performance

5.Tune hyperparameters

    -Experiment with different hyperparameters to improve performance

6.Deploy your model

    Use model to make predictions on new data in real-world application.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Aldrin lijo J E
RegisterNumber:  212222240007
*/
```
```py
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

### Initial dataset:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118544279/3d8c24b6-1d6d-4663-a483-9ee261590edf)

### Data Info:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118544279/f3a9b311-4589-42dd-baf8-01a9d708150e)

### Optimization of null values:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118544279/8de5891d-ab45-41bf-8bdd-6dcd5790d97c)

### Converting string literals to numericl values using label encoder:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118544279/34abf0ee-ee27-4166-81de-fefc3eba40b6)

### Assigning x and y values:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118544279/d9dd1d98-5ce7-47e0-aa74-093f3ab8b3c6)

### Mean Squared Error:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118544279/84a8e661-fe8d-41c0-bd7c-0c8deb643ba0)

### R2 (variance):

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118544279/f89b418b-4bbd-4ea9-b633-1546b239fc30)

### Prediction:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118544279/126caa8e-76e5-45d1-9258-ce4f51e8eac2)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
