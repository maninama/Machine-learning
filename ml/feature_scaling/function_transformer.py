import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("loan.csv")
sns.distplot(dataset["CoapplicantIncome"])
plt.show()