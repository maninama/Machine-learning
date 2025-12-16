import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from mlxtend.plotting import plot_decision_regions

dataset = pd.read_csv("datasets/Social_Network_Ads.csv")
print(dataset.head())

sns.scatterplot(x="Age",y="EstimatedSalary",data=dataset,hue="Purchased")
# plt.show()

x = dataset.iloc[:,:-1]
y = dataset["Purchased"]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x)
x = pd.DataFrame(sc.transform(x),columns=x.columns)
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 7)
knn.fit(x_train,y_train)

for i in range(1,30):
    knn1 = KNeighborsClassifier(n_neighbors=i)
    knn1.fit(x_train,y_train)
    print(i,knn1.score(x_train,y_train)*100,knn1.score(x_test,y_test)*100)

print(knn.score(x_test,y_test)*100)
print(knn.score(x_train,y_train)*100)


print(knn.predict([[1.083596,-0.990844]]))

plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=knn)
plt.show()