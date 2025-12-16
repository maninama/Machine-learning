import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

dataset = pd.read_csv("loan.csv")
print(dataset.describe())

# sns.boxplot(x = "ApplicantIncome",data=dataset)
# sns.boxplot(x = "CoapplicantIncome",data=dataset)
# plt.show()

sns.distplot(dataset["ApplicantIncome"])
plt.show()

## Outliers remove krne ki do technique h first IQR and second z score