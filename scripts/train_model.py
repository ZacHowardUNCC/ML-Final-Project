import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import joblib


def main():
    # Extract features and labels from npz file
    data = np.load("data/preprocessed_data.npz")
    X = data["X"]
    y = data["y"]

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(kernel='linear', probability=True)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Evaluate models
    for name, model in [("SVM", svm_model), ("Random Forest", rf_model)]:
        y_pred = svm_model.predict(X_test)
        print(f"--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred))
        print(f"--- {name} Confusion Matrix ---")
        print(confusion_matrix(y_test, y_pred))

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(svm_model, "models/svm_model.joblib")
    joblib.dump(rf_model, "models/rf_model.joblib")
    print("Models have been saved to the models/ directory.")

if __name__ == "__main__":
    main()