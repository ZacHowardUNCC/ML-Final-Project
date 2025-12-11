import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
import joblib
import time


def main():
    # Extract features and labels from npz file
    data = np.load("data/preprocessed_data.npz")
    X = data["X"]
    y = data["y"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train models
    print("Training SVM model...")
    start_time = time.time()
    svm_model = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_time = time.time() - start_time
    print(f"SVM training completed in {svm_time:.2f} seconds.")

    print("Training Random Forest model...")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start_time
    print(f"Random Forest training completed in {rf_time:.2f} seconds.")

    # Evaluate models
    for name, model in [("SVM", svm_model), ("Random Forest", rf_model)]:
        y_pred = model.predict(X_test)
        print(f"--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred))
        print(f"--- {name} Confusion Matrix ---")
        print(confusion_matrix(y_test, y_pred))

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(svm_model, "models/svm_model.joblib")
    joblib.dump(rf_model, "models/rf_model.joblib")
    print("Models have been saved to the models/ directory.")

if __name__ == "__main__":
    main()