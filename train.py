import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).parent
# dataset dhoondo
csv_path = None
for p in [BASE/"data"/"heart.csv", BASE/"heart.csv", BASE/"dataset"/"heart.csv", BASE/"data"/"heart_disease.csv"]:
    if p.exists():
        csv_path = p
        break

if csv_path is None:
    # agar csv nahi mila to online wala dataset use karo
    print("Local CSV nahi mila, sample data se train kar raha hu...")
    from sklearn.datasets import load_breast_cancer
    print("Please apna heart.csv data folder me daalo")
    # dummy train to create model file
    import numpy as np
    df = pd.DataFrame(np.random.rand(100, 26))
    df['target'] = np.random.randint(0,2,100)
    X = df.drop('target', axis=1)
    y = df['target']
else:
    print(f"Dataset mil gaya: {csv_path}")
    df = pd.read_csv(csv_path)
    # target column ko 'target' ya 'num' ya last column maano
    if 'target' not in df.columns and 'num' in df.columns:
        df['target'] = (df['num'] > 0).astype(int)
    if 'target' not in df.columns:
        df['target'] = df.iloc[:, -1]
    
    X = df.drop(['target','num'], axis=1, errors='ignore')
    X = pd.get_dummies(X)
    X.columns = X.columns.str.replace('reversible', 'reversable')
    y = df['target']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# save
Path("models").mkdir(exist_ok=True)
with open("models/heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Model ban gaya! Columns: {list(X.columns)[:5]}... Total: {len(X.columns)}")
print("models/heart_model.pkl saved!")