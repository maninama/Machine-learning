import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\py_machine_learning\datasets\loan.csv")

# Find the missing values 
print(dataset.isnull().sum())
# print(dataset.isnull().sum()/dataset.shape[0]*100) # har value per null value find out
# print(dataset.isnull().sum().sum()) # kitni jgh null value h use count krke bta dega
# print(dataset.isnull().sum().sum()/(dataset.shape[0]*dataset.shape[1])*100) # over all null value find out
# print(dataset.notnull().sum()) # fill value ko per value k hisab se count krke bta dega
# print( dataset.notnull().sum().sum()) # overall fill value ko count krke bta dega
# print(dataset.head())
# print(dataset)

# create a graph view
# sns.heatmap(dataset.isnull())
# plt.show()


# Handling missing values
# dataset.drop(columns=['Credit_History'],inplace=True) # Delete column for dataset
# dataset.dropna(inplace=True) # Ye null vali row ko delete kr deta hai
# print(dataset.isnull().sum())
# sns.heatmap(dataset.isnull())
# plt.show()


# Handling missing data (Categorical Data)
# print(dataset.head(5).fillna(10)) # ye h menual data fill krne ke liye
# print(dataset.head().fillna(method="bfill")) # ye h Backword filling krne ke liye
# print(dataset.head().fillna(method="ffill",axis=1))  # ye h Front filling krne ke liye
# print(dataset["Gender"].fillna(dataset["Gender"].mode()[0],inplace=True))
# print(dataset.head())
# print(dataset.head(5))


# Handling missing values (Scikit-learn)
# from sklearn.impute import SimpleImputer
# print(dataset.select_dtypes(include="float64").columns)
# si = SimpleImputer(strategy="mean")
# ar = (si.fit_transform(dataset[['CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
#        'Credit_History']])) # inse humm missing value me mean fill kr skate hai
# new_dataset=pd.DataFrame(ar,columns=dataset.select_dtypes(include="float64").columns)
# print(new_dataset.isnull().sum())
# print(new_dataset)