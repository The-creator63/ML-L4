#KNN- K Nearest Neighborx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/muhammadtaqiriaz/Desktop/Machine Learning/L4/data.csv")
print(data.head())
print(data.info())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data["species"] = encoder.fit_transform(data["species"])
print(data.head())

x = data[["sepal_length","sepal_width","petal_length","petal_width"]]
y = data["species"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)

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