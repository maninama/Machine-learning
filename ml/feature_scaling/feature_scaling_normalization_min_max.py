import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("loan.csv")
print(dataset.isnull().sum())
# sns.distplot(dataset['CoapplicantIncome'])
# plt.show()
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
ms.fit(dataset[["CoapplicantIncome"]])
dataset["CoapplicantIncome_min"] = ms.transform(dataset[["CoapplicantIncome"]])
print(dataset.head())

plt.figure(figsize=(10,5))  # Ek hi figure me dono plots aayenge

# Pehla subplot (Before)
plt.subplot(1,2,1)
plt.title("Before")
sns.distplot(dataset["CoapplicantIncome"])

# Dusra subplot (After)
plt.subplot(1,2,2)  # 1 row, 2 columns, 2nd position
plt.title("After")
sns.distplot(dataset["CoapplicantIncome_min"])

plt.show()  # Ek hi frame me dono plots display honge