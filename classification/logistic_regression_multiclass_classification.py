import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#OVR likhna h notebook me
dataset = pd.read_csv("datasets/iris.csv")
print(dataset.head())

# print(dataset['Species'].unique())
# sns.pairplot(data=dataset,hue='Species')
# plt.show()

x = dataset.iloc[:,:-1]
y = dataset['Species']

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

# from sklearn.linear_model import LogisticRegression
# # OVR method
# lr = LogisticRegression(multi_class='ovr')
# lr.fit(x_train,y_train)
# print(lr.score(x_test,y_test)*100)

# #multinomial
# lr1 = LogisticRegression(multi_class='multinomial')
# lr1.fit(x_train,y_train)
# print(lr.score(x_test,y_test)*100)