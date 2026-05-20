import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("hotel_bookings.csv")

print("Dataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nStatistical Summary:")
print(df.describe())



df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)


df['country'] = df['country'].fillna('Unknown')

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())


df = df[(df['adults'] + df['children'] + df['babies']) > 0]

df = df[df['adr'] >= 0]
df = df[df['lead_time'] >= 0]

df.drop_duplicates(inplace=True)

print("\nShape After Cleaning:")
print(df.shape)



sns.set_style("whitegrid")


plt.figure(figsize=(6,4))
sns.countplot(x='is_canceled', data=df)
plt.title("Booking Cancellation Distribution")
plt.xlabel("Canceled")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='hotel', data=df)
plt.title("Hotel Type Distribution")
plt.xlabel("Hotel Type")
plt.ylabel("Count")
plt.show()



plt.figure(figsize=(8,5))
sns.histplot(df['lead_time'], bins=50, kde=True)
plt.title("Lead Time Distribution")
plt.xlabel("Lead Time")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x=df['adr'])
plt.title("ADR Distribution")
plt.xlabel("Average Daily Rate")
plt.show()


plt.figure(figsize=(7,5))
sns.barplot(x='hotel', y='is_canceled', data=df)
plt.title("Cancellation Rate by Hotel Type")
plt.xlabel("Hotel")
plt.ylabel("Cancellation Rate")
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(x='market_segment', data=df,
              order=df['market_segment'].value_counts().index)
plt.title("Market Segment Distribution")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12,5))
sns.countplot(
    x='arrival_date_month',
    data=df,
    order=[
        'January','February','March','April',
        'May','June','July','August',
        'September','October','November','December'
    ]
)
plt.title("Bookings by Month")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(7,5))
sns.barplot(x='total_of_special_requests',
            y='is_canceled',
            data=df)

plt.title("Special Requests vs Cancellation")
plt.xlabel("Total Special Requests")
plt.ylabel("Cancellation Rate")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(x='customer_type', data=df)
plt.title("Customer Type Distribution")
plt.xticks(rotation=15)
plt.show()



plt.figure(figsize=(8,5))
sns.histplot(df['previous_cancellations'], bins=30)
plt.title("Previous Cancellations Distribution")
plt.xlabel("Previous Cancellations")
plt.ylabel("Frequency")
plt.show()

y = df['is_canceled']
X = df.drop('is_canceled', axis=1)

num_cols = X.select_dtypes(include=np.number).columns

plt.figure(figsize=(14, 10))
corr = X[num_cols].corr()
sns.heatmap(
    corr,
    annot=True,       
    fmt=".2f",         
    cmap="coolwarm",   
    linewidths=0.5,    
    square=True        
)

plt.title("Feature Correlation Matrix")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=X[num_cols])
plt.xticks(rotation=90)
plt.title("Boxplot for All Numerical Features")
plt.show()

def detect_outliers_iqr(data, cols):
    """Detect outliers using IQR method"""
    outliers_count = 0
    outlier_cols = []
    
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        col_outliers = ((data[col] < lower) | (data[col] > upper)).sum()
        if col_outliers > 0:
            outliers_count += col_outliers
            outlier_cols.append((col, col_outliers))
    
    return outliers_count, outlier_cols

# Check for outliers
outliers_count, outlier_cols = detect_outliers_iqr(X, num_cols)

print(f"Total outliers found: {outliers_count}")
if outlier_cols:
    print("\nOutliers per column:")
    for col, count in outlier_cols:
        print(f"  {col}: {count} outliers")
else:
    print("No outliers detected!")

# ==========================
# IQR function - Apply ONLY if outliers exist
# ==========================

if outliers_count > 0:
    print("\n✓ Applying outlier removal method...")
    
    def remove_outliers_iqr(data, cols):
        for col in cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            data[col] = np.clip(data[col], lower, upper)

        return data

    # Apply outlier removal
    X = remove_outliers_iqr(X, num_cols)
    print(f"After removing outliers: {X.shape}")
else:
    print("\n✗ No outliers to remove - keeping data as is")
print("\nFinal Missing Values:\n")
print(X.isnull().sum())

print("\nFinal Dataset Shape:", X.shape)

target_col = y.name

df_balanced = pd.concat([X, y], axis=1)

class_0 = df_balanced[df_balanced[target_col] == 0]
class_1 = df_balanced[df_balanced[target_col] == 1]

# Under-sample majority
class_0_under = class_0.sample(len(class_1), random_state=42)

# Over-sample minority
class_1_over = class_1.sample(len(class_0_under), replace=True, random_state=42)

# Combine
df_balanced = pd.concat([class_0_under, class_1_over])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Split again
X = df_balanced.drop(columns=[target_col])
y = df_balanced[target_col]

print("After balancing:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


num_features = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_features = X_train.select_dtypes(include=['object']).columns

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)
svd = TruncatedSVD(n_components=10, random_state=42)
svm = SVC()

pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('svd', svd),
    ('svm', svm)
])
param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.01],
    "svm__kernel": ["rbf"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)


grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

y_pred = best_model.predict(X_test)


print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))