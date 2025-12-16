import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("datasets/multiple_regression_dataset.csv")
print(dataset.head())
print(dataset.isnull().sum())

sns.pairplot(data=dataset)
plt.show()

x = dataset.iloc[:,:-1]
y = dataset["Experience"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

from sklearn.tree import DecisionTreeRegressor,plot_tree

dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
print(dtr.score(x_train,y_train)*100)

plot_tree(dtr)
plt.show()