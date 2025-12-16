import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('loan.csv')
print(dataset.isnull().sum())
dataset["ApplicantIncome"].fillna(dataset["ApplicantIncome"].mean(),inplace=True)
# sns.distplot(dataset["ApplicantIncome"])
# plt.show()
from sklearn.preprocessing import StandardScaler
ss =  StandardScaler()
ss.fit(dataset[["ApplicantIncome"]])
dataset["ApplicantIncome_ss"] = pd.DataFrame(ss.transform(dataset[["ApplicantIncome"]]),columns=["x"])
print(dataset.head(3))