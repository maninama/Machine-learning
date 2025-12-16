import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("datasets/placement_classification.csv")
print(dataset.head())

# sns.scatterplot(x="cgpa",y="resume_score",data=dataset,hue="placed")
# plt.legend(loc=1)
# plt.show()

x = dataset.iloc[:,:-1]
y = dataset["placed"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)

print(lr.predict([[8.14,6.52 ]]))

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=lr)
# plt.show()


print(lr.coef_)
print(lr.intercept_)