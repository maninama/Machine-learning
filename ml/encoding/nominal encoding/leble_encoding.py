import pandas as pd

df = pd.DataFrame({"name":["wscube","cow","cat","dog","black"]})
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["en_name"]=le.fit_transform(df["name"])
# print(df)

dataset = pd.read_csv("loan.csv")

la = LabelEncoder()
la.fit(dataset["Property_Area"])
dataset["Property_Area"] = la.transform(dataset["Property_Area"])
print(dataset["Property_Area"].unique())
