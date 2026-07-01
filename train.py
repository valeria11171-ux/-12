
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

# Генерация данных
np.random.seed(42)
n_samples = 10000

data = {
    'store_id': np.random.choice(['store_A', 'store_B', 'store_C', 'store_D', 'store_E'], n_samples),
    'items_count': np.random.randint(1, 15, n_samples),
    'order_price': np.random.uniform(100, 5000, n_samples),
    'delivery_distance': np.random.uniform(0.5, 15, n_samples),
    'planned_prep_time': np.random.randint(10, 60, n_samples),
    'hour': np.random.randint(8, 23, n_samples),
    'is_weekend': np.random.choice([0, 1], n_samples)
}
df = pd.DataFrame(data)

# Целевая переменная
df['on_time'] = ((df['delivery_distance'] < 5) & (df['planned_prep_time'] < 40) & 
                 (df['items_count'] < 10) & (df['hour'] < 20)).astype(int)
noise = np.random.choice(df.index, size=int(0.1*len(df)), replace=False)
df.loc[noise, 'on_time'] = 1 - df.loc[noise, 'on_time']

# Предобработка
le = LabelEncoder()
df['store_id_encoded'] = le.fit_transform(df['store_id'])

features = ['store_id_encoded', 'items_count', 'order_price', 
            'delivery_distance', 'planned_prep_time', 'hour', 'is_weekend']
X = df[features]
y = df['on_time']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Обучение
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")

# Сохранение
os.makedirs('models', exist_ok=True)
pickle.dump(model, open('models/model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(le, open('models/label_encoder.pkl', 'wb'))
pickle.dump(features, open('models/features.pkl', 'wb'))

print("Модель сохранена в папку models/")
