import pandas as pd
dataset = pd.read_csv("datasets/Social_Network_Ads.csv")
print(dataset.head())

print(dataset["Purchased"].value_counts())

x = dataset.iloc[:,:-1]
y= dataset["Purchased"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)

print(lr.predict([[47,30000]]))