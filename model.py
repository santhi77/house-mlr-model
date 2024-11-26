import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv(r"H:\sir daily notes tasks-nov\18th- mlr\MLR\House_data.csv")  # Replace with your local path to the dataset

# Independent and dependent variables
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'sqft_lot']]
y = data['price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a pickle file
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'house_price_model.pkl'")

