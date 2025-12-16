# balancing dataset using imblearn

import pandas as pd
dataset = pd.read_csv("datasets/Social_Network_Ads.csv")
print(dataset.head())

print(dataset["Purchased"].value_counts())

# using imblearn
x = dataset.iloc[:,:-1]
y= dataset["Purchased"]

from imblearn.under_sampling import RandomUnderSampler

ru = RandomUnderSampler()
ru_x , ru_y = ru.fit_resample(x,y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(ru_x,ru_y,test_size=0.20,random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train,y_train)
# print(lr.score(x_test,y_test)*100)

print(lr.predict([[27,137000]]))

print(ru_y.value_counts())