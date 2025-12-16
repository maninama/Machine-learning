import pandas as pd 

dataset = pd.read_csv("datasets/multiple_regression_dataset.csv")
print(dataset.head())
print(dataset.isnull().sum())

x = dataset.drop(columns="Salary")
y=  dataset["Salary"]

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.20, random_state=42)

from sklearn.neighbors import KNeighborsRegressor

kn = KNeighborsRegressor(n_neighbors=5)
kn.fit(x_train,y_train)
print(kn.score(x_test,y_test)*100)