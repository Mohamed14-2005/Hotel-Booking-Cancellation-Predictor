import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC

# Load data
print('Loading data...')
df = pd.read_csv('hotel_bookings.csv')

# Clean data
df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)
df['country'] = df['country'].fillna('Unknown')
df = df[(df['adults'] + df['children'] + df['babies']) > 0]
df = df[df['adr'] >= 0]
df = df[df['lead_time'] >= 0]
df.drop_duplicates(inplace=True)

# Split target and features
Y = df['is_canceled']
X = df.drop('is_canceled', axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Identify feature types
num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

print('Numeric features:', num_features)
print('Categorical features:', cat_features)

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

model = Pipeline([
    ('preprocess', preprocess),
    ('svd', TruncatedSVD(n_components=10, random_state=42)),
    ('svm', SVC(C=1, gamma='scale', kernel='rbf', probability=True))
])

print('Training model...')
model.fit(X_train, y_train)

joblib.dump(model, 'hotel_model.pkl')
print('Model saved to hotel_model.pkl')

# Optional evaluation
from sklearn.metrics import accuracy_score

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f'Test accuracy: {acc:.4f}')
