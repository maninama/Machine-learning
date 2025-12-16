import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
dataset = pd.read_csv("datasets/placement_classification.csv")

print(dataset.head())
# sns.kdeplot(data=dataset['resume_score'])
# sns.scatterplot(x="cgpa",y="resume_score",data=dataset,hue="placed")
# plt.show()

x = dataset.iloc[:,:-1]
y = dataset['placed']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

gnb = GaussianNB()
gnb.fit(x_train,y_train) 
print(gnb.score(x_test,y_test)*100 , gnb.score(x_train,y_train)*100)
# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=gnb)
# plt.show()
print("gnb prediction is",gnb.predict([[6.17,5.17]]))

mnb = MultinomialNB()
mnb.fit(x_train,y_train)
print(mnb.score(x_test,y_test)*100 , mnb.score(x_train,y_train)*100)
# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=mnb)
# plt.show()
print("mnb prediction is",mnb.predict([[6.17,5.17]]))

bnb = BernoulliNB()
bnb.fit(x_train,y_train)
print(bnb.score(x_test,y_test)*100 , bnb.score(x_train,y_train)*100)
# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=bnb)
# plt.show()
print("bnb prediction is",bnb.predict([[6.17,5.17]]))