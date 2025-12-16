import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from mlxtend.plotting import plot_decision_regions

dataset = pd.read_csv("datasets/placement_classification.csv")
print(dataset.head())

sns.scatterplot(x="cgpa",y="resume_score",data=dataset,hue="placed")
# plt.show()

x = dataset.iloc[:,:-1]
y = dataset["placed"]

from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=15)
#rbf , sigmoid , precomputed
from sklearn.svm import SVC

sv = SVC(kernel="precomputed")
sv.fit(x_train,y_train)
print(sv.score(x_test,y_test)*100)
print(sv.score(x_train,y_train)*100)

plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=sv)
plt.show()