import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("datasets/polynomial.csv")
print(dataset.head())

plt.scatter(dataset["Level"],dataset["Salary"])
plt.show()