import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector

# Load dataset
dataset = pd.read_csv("diabetes.csv")

# Features & Target
x = dataset.iloc[:, :-1]
y = dataset["Outcome"]

print("Number of features:", x.shape[1])

# Model
lr = LogisticRegression(max_iter=1000)

# 🔽 Backward Elimination (forward=False)
fs_backward = SequentialFeatureSelector(
    lr,
    k_features="best",
    forward=False,
)

fs_backward.fit(x, y)

# Results
print("Selected features:", fs_backward.k_feature_names_)
print("features K score:", fs_backward.k_score_)
