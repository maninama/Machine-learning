import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("datasets/placement.csv")
# print(dataset.head())
# print(dataset.isnull().sum())

# sns.scatterplot(x="cgpa",y="package",data=dataset)
# plt.show()

x = dataset[["cgpa"]]
y = dataset["package"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

print(lr.coef_)
print(lr.intercept_)
print(lr.predict([[6.89]]))

y_prd = lr.predict(x)

plt.figure(figsize=(5,4))
sns.scatterplot(x="cgpa",y="package",data=dataset)
plt.plot(dataset["cgpa"],y_prd,c="red")
plt.legend(["org data","predict line"])
plt.show()