import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------- LOAD DATA ----------------
data = pd.read_csv("data.csv")

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- MODELS ----------------
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# ---------------- TRAIN ----------------
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

lr_score = r2_score(y_test, lr_pred)
rf_score = r2_score(y_test, rf_pred)

print("Linear Regression R2:", lr_score)
print("Random Forest R2:", rf_score)

# ---------------- SAVE MODELS ----------------
pickle.dump(lr, open("lr_model.pkl", "wb"))
pickle.dump(rf, open("rf_model.pkl", "wb"))

# Save scores also (important for UI)
pickle.dump({"lr": lr_score, "rf": rf_score}, open("scores.pkl", "wb"))

print("Both models saved successfully!")