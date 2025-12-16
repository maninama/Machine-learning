import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv("datasets/multiple_regression_dataset.csv")
print(dataset.head())
print(dataset.shape)
# print(dataset.isnull().sum())

# sns.pairplot(data=dataset)
# plt.show()

# sns.heatmap(data=dataset.corr(),annot=True)
# plt.show()

x = dataset.iloc[:,:-1]
y = dataset["Salary"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)

print(lr.coef_)
print(lr.intercept_)
print(lr.predict(x_test))