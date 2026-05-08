import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load data
data = pd.read_csv("data.csv")

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Train-test split (important for real ML)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

rf = RandomForestRegressor()
rf.fit(X_train_scaled, y_train)

# SAVE MODELS
pickle.dump(lr, open("lr_model.pkl", "wb"))
pickle.dump(rf, open("rf_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Models + scaler saved successfully!")