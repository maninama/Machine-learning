import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np

dataset = pd.read_csv("datasets/Housing.csv")

print(dataset.head())

# $

x = dataset.iloc[:,:-1]
y = dataset["price"]

sc = StandardScaler()
sc.fit(x)
x = pd.DataFrame(sc.transform(x),columns=x.columns)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)
plt.bar(x.columns,lr.coef_)
plt.title("LinearRegression")
plt.xlabel("Columns")
plt.ylabel("Coef")
# plt.show()

print("LinearRegression")
print(mean_squared_error(y_test,lr.predict(x_test)))
print(mean_absolute_error(y_test,lr.predict(x_test)))
print(np.sqrt(mean_squared_error(y_test,lr.predict(x_test))))

#Lasso
la = Lasso(alpha=10)
la.fit(x_train,y_train)
print(la.score(x_test,y_test)*100)
plt.bar(x.columns,la.coef_)
plt.title("Lasso")
plt.xlabel("Columns")
plt.ylabel("Coef")
# plt.show()

print("Lasso")
print(mean_squared_error(y_test,la.predict(x_test)))
print(mean_absolute_error(y_test,la.predict(x_test)))
print(np.sqrt(mean_squared_error(y_test,la.predict(x_test))))

#Ridge
ri = Ridge(alpha=10)
ri.fit(x_train,y_train)
print(ri.score(x_test,y_test)*100)
plt.bar(x.columns,ri.coef_)
plt.title("Ridge")
plt.xlabel("Columns")
plt.ylabel("Coef")
# plt.show()
print("Ridge")
print(mean_squared_error(y_test,ri.predict(x_test)))
print(mean_absolute_error(y_test,ri.predict(x_test)))
print(np.sqrt(mean_squared_error(y_test,ri.predict(x_test))))


df = pd.DataFrame({"col_name":x.columns,"LinearRegression":lr.coef_,"Lasso":la.coef_,"Ridge":ri.coef_})
print(df)