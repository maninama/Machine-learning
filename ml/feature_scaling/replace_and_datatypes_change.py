import pandas as pd
dataset = pd.read_csv("loan.csv")
print(dataset["Dependents"].value_counts()) #isse hume ye pta chlta h ki hmare pass kon kon sa data h or konsa data hai
dataset["Dependents"].fillna(dataset["Dependents"].mode()[0],inplace=True) # isse mode fill hota hai
dataset["Dependents"].replace("3+","3",inplace=True) # isse value replace hoti hai
dataset["Dependents"] = dataset["Dependents"].astype("int64") # isse datatype change hota hai
print(dataset["Dependents"].value_counts())
print(dataset.info())