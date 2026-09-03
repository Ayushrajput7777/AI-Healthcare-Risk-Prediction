import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Data load
df = pd.read_csv("data/heart.csv")

# 2. String ko Number me badlo (ye line naya hai)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("num", axis=1)
y = df["num"]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Save - models folder me
pickle.dump(model, open("models/heart_model.pkl", "wb"))

print("✅ Model ban gaya! Accuracy:", model.score(X_test, y_test))
print("Columns:", list(X.columns))