import pandas as pd

# data = {"name":["a","b","c","d","a","c"],"eng":[8,7,5,8,8,5],"hindi":[2,3,4,5,2,6]}
# df = pd.DataFrame(data)
# df["duplicated"] = df.duplicated()  #isse duplicated name ka column create ho jaega or sari duplicate values show hogi
# df.drop_duplicates(inplace=True)
# print(df)


# ab dataset per work krenge

dataset = pd.read_csv("loan.csv")
dataset["duplicated"] = dataset.duplicated()
print(dataset.head(50))







# isme el keep="first" krke ek function hota hai jo orignal value ko rhne deta hai or duplicate value ko remove kr deta hai