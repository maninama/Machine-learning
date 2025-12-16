import pandas as pd

# ye he cycle line encoding isme autometic encoding hoti hai
df = pd.DataFrame({"size":["s","m","l","xl","s","m","l","s","s","l","xl","m"]})
print(df.head(3))
ord_data = [["s","m","l","xl"]] # isme two dimensional deta use hota hai
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=ord_data)
oe.fit(df[["size"]])
df[["size_en"]] = oe.transform(df[["size"]])

# map function isme hum manualy values deke ise encode kr skte hai
ord_data1 = {"s":0,"m":1,"l":2,"xl":3}
df["size_en_map"] = df["size"].map(ord_data1)
print(df)

dataset = pd.read_csv("loan.csv")
# print(dataset.head())
# dataset["Property_Area"].fillna(dataset["Property_Area"].mode()[0],inplace=True)
dataset.fillna({'Property_Area': dataset['Property_Area'].mode()[0]},inplace=True)
# print(dataset["Property_Area"].unique())
en_data_ord = [["Rural","Semiurban","Urban"]]
oen = OrdinalEncoder(categories=en_data_ord)
dataset["Property_Area"]=oen.fit_transform(dataset[["Property_Area"]])
print(dataset.head())