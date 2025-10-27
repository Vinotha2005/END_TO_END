# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1️⃣ Load data
df = pd.read_csv("yield.csv")

# 2️⃣ Handle missing values & encode categoricals
df = df.fillna(0)
df = df.apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x)

# 3️⃣ Separate features and target
target_col = "Value"
X = df.drop(columns=[target_col])
y = df[target_col]

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Scale numeric data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Fit model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7️⃣ Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"✅ Model Trained Successfully!")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# 8️⃣ Save model and scaler for app.py
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
