#KNN- K Nearest Neighborx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/muhammadtaqiriaz/Desktop/Machine Learning/titanic2.csv")
print(data.head())
print(data.info())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data["Sex"] = encoder.fit_transform(data["Sex"])
data["Embarked"] = encoder.fit_transform(data["Embarked"])
print(data.head())

data = data.drop("Cabin",axis = 1)
data = data.drop("PassengerId",axis = 1)
data = data.drop("Name",axis = 1)
data = data.drop("Ticket",axis = 1)
data = data.drop("SibSp",axis = 1)
data = data.drop("Parch",axis = 1)
data = data.drop("Fare",axis = 1)
data = data.drop("Embarked",axis = 1)
print(data.head())

data["Age"] = data["Age"].fillna(data["Age"].median(skipna = True),inplace = True)

x = data[["Pclass","Sex","Age"]]
y = data["Survived"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 6)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
accuracy = accuracy_score(y_test,y_predict)
accuracy = round(accuracy*100,2)
print("Accuracy = ",accuracy,"%")

import seaborn as sns
matrix = confusion_matrix(y_test,y_predict)
sns.heatmap(matrix,annot = True,fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()