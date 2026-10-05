import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("autos_dataset.csv")

# Replace missing values
df = df.replace("?", pd.NA)

# Convert cylinder values into numbers
cylinder_mapping = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "eight": 8,
    "twelve": 12
}

df["num-of-cylinders"] = df["num-of-cylinders"].map(cylinder_mapping)

# Convert numeric columns
numeric_columns = [
    "price",
    "horsepower",
    "peak-rpm"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Fill missing numeric values with median
df = df.fillna(df.median(numeric_only=True))

# Select features
features = [
    "symboling",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "num-of-cylinders",
    "engine-size",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg"
]

X = df[features]
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

train_r2 = r2_score(y_train, train_prediction)
test_r2 = r2_score(y_test, test_prediction)

rmse = mean_squared_error(y_test, test_prediction) ** 0.5

print("Model training completed!")
print("Training R2 Score:", train_r2)
print("Testing R2 Score:", test_r2)
print("RMSE:", rmse)

# Save model and feature names
model_data = {
    "model": model,
    "features": features
}

with open("car_price_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("car_price_model.pkl file created successfully!")