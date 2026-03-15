from dotenv import load_dotenv
from huggingface_hub import login
import os
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = load_dataset("electricsheepafrica/nigerian_retail_and_ecommerce_supply_chain_logistics_data")
df = dataset['train'].to_pandas()

# Convert dates
df['ship_date'] = pd.to_datetime(df['ship_date'])
df['expected_delivery_date'] = pd.to_datetime(df['expected_delivery_date'])
df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])

# Feature engineering
df['days_late'] = (df['actual_delivery_date'] - df['expected_delivery_date']).dt.days
df['shipping_duration_expected'] = (df['expected_delivery_date'] - df['ship_date']).dt.days

# Target
df['target'] = (df['days_late'] > 2).astype(int)

# Label encoding
le = LabelEncoder()
df['logistics_company_enc'] = le.fit_transform(df['logistics_company'])
df['origin_city_enc'] = le.fit_transform(df['origin_city'])
df['destination_city_enc'] = le.fit_transform(df['destination_city'])

# Features
X = df[['quantity', 'shipping_cost_ngn', 'logistics_company_enc',
        'origin_city_enc', 'destination_city_enc', 'shipping_duration_expected']]
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data loaded successfully!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=50, random_state=42, class_weight='balanced')
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

train_pred = dt_model.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
print(f"Testing Accuracy:  {accuracy_score(y_test, y_pred):.4f}")

# for depth in [3, 5, 7, 10, 150, 2000]:
#     dt_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
#     dt_model.fit(X_train, y_train)
#     train_acc = accuracy_score(y_train, dt_model.predict(X_train))
#     test_acc = accuracy_score(y_test, dt_model.predict(X_test))
#     print(f"depth={depth:2d} | train={train_acc:.4f} | test={test_acc:.4f} | gap={train_acc-test_acc:.4f}")

print("\nDecision Tree Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save confusion matrix chart
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Decision Tree - Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['On Time', 'Delayed'])
plt.yticks([0, 1], ['On Time', 'Delayed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("decision_tree_confusion_matrix.png")
print("\nChart saved!")

# Feature importance — unique to decision trees
print("\nFeature Importances:")
feature_names = ['quantity', 'shipping_cost_ngn', 'logistics_company_enc',
                 'origin_city_enc', 'destination_city_enc', 'shipping_duration_expected']
for name, importance in zip(feature_names, dt_model.feature_importances_):
    print(f"  {name}: {importance:.4f}")

# Random Forest

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,        # 100 trees
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1                # use all CPU cores
)
rf_model.fit(X_train, y_train)

# Evaluate
rf_pred = rf_model.predict(X_test)
rf_train_pred = rf_model.predict(X_train)

print(f"\nRandom Forest Results:")
print(f"Training Accuracy: {accuracy_score(y_train, rf_train_pred):.4f}")
print(f"Testing Accuracy:  {accuracy_score(y_test, rf_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Feature importances
print("\nFeature Importances:")
for name, importance in zip(feature_names, rf_model.feature_importances_):
    print(f"  {name}: {importance:.4f}")