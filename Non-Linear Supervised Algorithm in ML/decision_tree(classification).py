import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

dataset = pd.read_csv("datasets/Social_Network_Ads.csv")
print(dataset.head())

# print(dataset.isnull().sum())
sns.scatterplot(x="Age",y="EstimatedSalary",data=dataset,hue="Purchased")
# plt.show()
x = dataset.iloc[:,:-1]
y = dataset["Purchased"]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x)
x = pd.DataFrame(sc.transform(x),columns=x.columns)
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
for i in range(1,20):
    dt = DecisionTreeClassifier(max_depth=i)
    dt.fit(x_train,y_train)
    print(dt.score(x_train,y_train)*100,dt.score(x_test,y_test)*100,i)



dt = DecisionTreeClassifier(max_depth=3)
dt.fit(x_train,y_train)
print(dt.score(x_test,y_test)*100)
print(dt.predict([[35,20000]]))

plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=dt)
plt.show()

from sklearn.tree import plot_tree
plt.figure(figsize=(21,21))
plot_tree(dt)
plt.show()


