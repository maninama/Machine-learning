import pandas as pd 
from mlxtend.feature_selection import SequentialFeatureSelector

import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector

dataset = pd.read_csv("diabetes.csv")

x = dataset.iloc[:, :-1]
y = dataset["Outcome"]

print("Number of features:", x.shape[1])  # 👈 Check how many features you have

lr = LogisticRegression(max_iter=1000)

fs = SequentialFeatureSelector(lr, k_features="best", forward=True)  # ✅ Fix here
fs.fit(x, y)

print("Selected features:", fs.k_feature_names_)

print("features K score:", fs.k_score_)