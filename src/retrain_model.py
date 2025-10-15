import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the training data
print("Loading training data...")
df = pd.read_csv('imdb_cleaned_train.csv')
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Prepare features
print("Preparing features...")
features = ['runtimeMinutes', 'averageRating', 'numVotes', 'budget', 'gross']
X = df[features].copy()

# Handle missing values
X = X.fillna(X.median())

# Target variable
y = df['rating_status']

# Check target classes
print(f"Target classes: {y.unique()}")
print(f"Target distribution:\n{y.value_counts()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# Test the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model
print("Saving model...")
with open('Random_Forest_Tuned_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully as 'Random_Forest_Tuned_model.pkl'")

# Test loading the model
print("Testing model loading...")
try:
    with open('Random_Forest_Tuned_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    print("Model loaded successfully!")
    
    # Test prediction
    sample_data = X_test.iloc[0:1]
    prediction = loaded_model.predict(sample_data)
    probabilities = loaded_model.predict_proba(sample_data)
    print(f"Sample prediction: {prediction[0]}")
    print(f"Prediction probabilities: {probabilities[0]}")
    
except Exception as e:
    print(f"Error loading model: {e}")
