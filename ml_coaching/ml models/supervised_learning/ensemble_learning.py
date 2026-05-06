import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


x, y = make_classification(n_samples=1000,n_features=20,random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state =42)

base_learners = []

num_base_learners = 10

for i in range(num_base_learners):
    boostrap_indices = np.random.choice(len(x_train),replace=True)

    x_bootstrap = x_train[boostrap_indices]
    y_bootstrap = y_train[boostrap_indices]

    base_learners = RandomForestClassifier(n_estimators=10,random_state=42) 
    base_learners.fit(x_bootstrap,y_bootstrap)

    base_learners.append(base_learners)

    

