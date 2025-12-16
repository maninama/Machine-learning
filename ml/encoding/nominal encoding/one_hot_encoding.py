import numpy as np
import pandas as pd

# dataset = pd.read_csv('loan.csv')
# # print(dataset.head())

# dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)
# dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)

# en_data = dataset[["Gender","Married"]]
# # print(en_data)
# pd.get_dummies(en_data)
# # print(dataset.isnull().sum())
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(drop="first")
# ar = ohe.fit_transform(en_data).toarray()
# pd.DataFrame(ar,columns=["Gender_Female","Gender_Male","Married_No","Married_Yes"])
# print(ar)

#One-Hot encoding


from sklearn.preprocessing import OneHotEncoder

# Load dataset
dataset = pd.read_csv('loan.csv')

# Handling missing values
dataset.fillna({'Gender': dataset['Gender'].mode()[0], 
                'Married': dataset['Married'].mode()[0]}, inplace=True)

# Selecting categorical columns
en_data = dataset[["Gender", "Married"]]

# One-Hot Encoding using sklearn
ohe = OneHotEncoder(drop="first", sparse_output=False)  # Ensure dense output
ar = ohe.fit_transform(en_data)

# Convert to DataFrame with correct column names
columns = ohe.get_feature_names_out(en_data.columns)  # Auto-generate correct column names
df_encoded = pd.DataFrame(ar, columns=columns)

# Print the encoded array
print(df_encoded.head())
