# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


# dataset = pd.read_csv("datasets/precise_sample_ml_data.csv")
# print(dataset.head())

# sns.scatterplot(x="data1",y="data2",data=dataset,hue="output")
# # plt.show()

# x = dataset.iloc[:,:-1]
# y = dataset["output"]

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression()
# lr.fit(x_train,y_train)
# print(lr.score(x_test,y_test)*100)

# from mlxtend.plotting import plot_decision_regions

# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=lr)
# plt.show()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

# correct predict graph

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# CSV file load karo
df = pd.read_csv("datasets/precise_sample_ml_data.csv")  # Update path if needed

# Features & labels split
X = df[['data1', 'data2']]
y = df['output']

# Train-test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model train karo
model = LogisticRegression()
model.fit(X_train, y_train)

# Decision boundary plot karo
plt.figure(figsize=(8, 6))
plot_decision_regions(X=X.values, y=y.values, clf=model, legend=2)

plt.xlabel("data1")
plt.ylabel("data2")
plt.title("Decision Boundary using Logistic Regression")
plt.show()
