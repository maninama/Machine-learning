import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("loan.csv")
# print(dataset.head())



# Remove outliers for direct method
# min_range = dataset["CoapplicantIncome"].mean() - (3*dataset["CoapplicantIncome"].std())
# max_range = dataset["CoapplicantIncome"].mean() + (3*dataset["CoapplicantIncome"].std())

# new_dataset =  dataset[dataset["CoapplicantIncome"]<=max_range]

# sns.boxplot(x="CoapplicantIncome",data=new_dataset)
# plt.show()

# Remove outliers for Z score method
z_score = (dataset["CoapplicantIncome"] - dataset["CoapplicantIncome"].mean())/dataset["CoapplicantIncome"].std()
dataset["z_score"] = z_score
# print(dataset)
print(dataset[dataset["z_score"]<3].shape)