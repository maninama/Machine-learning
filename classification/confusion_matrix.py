import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("datasets/placement_classification.csv")
print(dataset.head())

x = dataset.iloc[:,:-1]
y = dataset["placed"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)


from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

cf = confusion_matrix(y_test,lr.predict(x_test))
sns.heatmap(cf,annot=True)
plt.show()

ps = precision_score(y_test,lr.predict(x_test))
print(ps*100)

print(recall_score(y_test,lr.predict(x_test))*100)

print(f1_score(y_test,lr.predict(x_test))*100)