# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 2: Create a dummy dataset
# Features: Area (sqft), Bedrooms, Age (in years)
data = {
    'Area': [1500, 1800, 2400, 3000, 3500],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Age': [10, 15, 20, 18, 25],
    'Price': [300000, 400000, 500000, 600000, 650000]
}

df = pd.DataFrame(data)

# Step 3: Split into features and target
X = df[['Area', 'Bedrooms', 'Age']]   # Independent Variables
y = df['Price']                       # Target Variable

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict the price for a new house
# Example: Area=2800 sqft, Bedrooms=4, Age=15 years
new_house = np.array([[2800, 4, 15]])
predicted_price = model.predict(new_house)

# Step 7: Output
print(f"Predicted Price of the House: ₹{predicted_price[0]:,.2f}")
